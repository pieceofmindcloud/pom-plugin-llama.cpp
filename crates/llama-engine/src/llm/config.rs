use super::*;

pub(super) fn model_metadata_string(model: *const sys::llama_model, key: &str) -> Option<String> {
    let key = CString::new(key).ok()?;
    let mut buf = vec![0u8; 32768];
    let n = unsafe {
        sys::llama_model_meta_val_str(model, key.as_ptr(), buf.as_mut_ptr() as *mut i8, buf.len())
    };
    if n <= 0 {
        return None;
    }
    buf.truncate(n as usize);
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// O lado dominante dos buffers `context` + `compute` de um contexto MTP.
/// Buffers host pequenos de staging não transformam um contexto cujo KV e
/// grafo estão na VRAM em um recuo para RAM; quando a maioria fica no host, o
/// operador precisa ver RAM no painel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpAllocation {
    Vram,
    Ram,
}

impl MtpAllocation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vram => "vram",
            Self::Ram => "ram",
        }
    }
}

pub fn mtp_allocation_from_context_memory(value: &serde_json::Value) -> Option<MtpAllocation> {
    let mut vram_bytes = 0u64;
    let mut ram_bytes = 0u64;
    for buffer in value.get("buffers")?.as_array()? {
        let bytes = buffer
            .get("context")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            .saturating_add(
                buffer
                    .get("compute")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
            );
        if bytes == 0 {
            continue;
        }
        let is_host = buffer
            .get("host")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or_else(|| {
                buffer
                    .get("device")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|device| device.eq_ignore_ascii_case("CPU"))
            });
        if is_host {
            ram_bytes = ram_bytes.saturating_add(bytes);
        } else {
            vram_bytes = vram_bytes.saturating_add(bytes);
        }
    }
    match (vram_bytes, ram_bytes) {
        (0, 0) => None,
        (vram, ram) if ram > vram => Some(MtpAllocation::Ram),
        _ => Some(MtpAllocation::Vram),
    }
}

pub fn context_allocation(ctx: *const sys::llama_context) -> Option<MtpAllocation> {
    let value = unsafe { json_from_wrapper(|out| sys::llama_rs_context_memory_json(ctx, out)) }?;
    mtp_allocation_from_context_memory(&value)
}

// ── Configuration ─────────────────────────────────────────────────────────────

pub const MTP_NO_NEXTN_REASON: &str = "o modelo carregado não traz o bloco nextn";

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Projector multimodal opcional, carregado como o `--mmproj` do llama.cpp.
    pub mmproj_path: Option<String>,
    /// GGUF externo de rascunho MTP ("shared MTP module"), usado quando o
    /// modelo alvo não embute pesos `nextn` próprios. Mesmo padrão de
    /// descoberta do `mmproj_path` (arquivo irmão na pasta de modelos) - ver
    /// `crate::mtp` e `catalog::find_mtp_draft_sibling`. `None` preserva o
    /// comportamento atual: MTP pedido sem `nextn` embutido e sem este campo
    /// segue sem MTP.
    pub mtp_draft_path: Option<String>,
    /// Capacidade total do cache KV. O runtime já multiplica a janela pública
    /// por sessão por `max_sessions` antes de montar esta configuração.
    pub n_ctx: u32,
    pub offload_kqv: bool,
    pub cache_type_k: KvCacheType,
    pub cache_type_v: KvCacheType,
    pub fit_memory: bool,
    /// VRAM que fica livre após a projeção do backend, por dispositivo.
    /// Processos atuais já ficam fora do snapshot de memória livre do backend;
    /// este campo é somente a reserva de segurança configurável.
    pub fit_memory_margin_mb: u32,
    pub n_batch: u32,
    /// The physical micro-batch (llama.cpp's `n_ubatch`) while the memory fit
    /// is on: the context decodes at most this many tokens at once, and its
    /// compute buffers are sized for it. Never more than `n_batch`. With the
    /// fit off the micro-batch is the batch, as before.
    pub n_ubatch: u32,
    pub n_gpu_layers: i32,
    pub n_threads: i32,
    pub max_sessions: u32,
    /// True for non-last peers that need to hand the residual to the next peer.
    pub need_embeddings: bool,
    /// Ligar a predição multi-token pelo bloco `nextn` do modelo, quando ele
    /// tiver um. Distinto da especulação por lookup (`find_draft`) - ver
    /// `crate::mtp`. Um modelo sem `nextn` não é erro de configuração: o
    /// contexto informa essa ausência para a telemetria e segue sem MTP.
    pub mtp: bool,
    /// Permite recuar o bloco `nextn` para pesos e cache na RAM se a tentativa
    /// com a VRAM falhar. Só é lido quando `mtp` está ligado.
    pub mtp_cpu_fallback_enabled: bool,
    /// Where the deployment wants the MTP block. See `crate::MtpPlacement`.
    pub mtp_placement: crate::MtpPlacement,
    /// Keep every routed MoE expert in host memory, computed by the CPU
    /// (llama.cpp's `--cpu-moe`). The memory fit is skipped: it refuses
    /// overrides set before it.
    pub moe_experts_in_ram: bool,
    /// GPU cache, in MiB, for the MoE experts kept in host memory
    /// (`llama_context_params::moe_cache_size`); `0` disables it.
    pub moe_cache_mib: u32,
    /// Quantos snapshots de estado recorrente manter por sequência
    /// (`llama_context_params::n_rs_seq`), para permitir desfazer um
    /// rascunho especulativo recusado num modelo híbrido/recorrente
    /// (Qwen3.5+: `llm_arch_supports_rs_rollback` no llama.cpp). `0` é o
    /// padrão do llama.cpp - sem isto, `llama_memory_recurrent::seq_rm`
    /// nunca aceita um rollback parcial, e vale a pena pedir um valor maior
    /// que a maior largura de lote especulativo configurada (`speculative_k`,
    /// `mtp_draft_tokens + 1`); arquitetura sem suporte simplesmente zera o
    /// pedido de volta (`llama-context.cpp`), então pedir aqui não machuca
    /// modelo comum. Ver `infer_server::should_draft` e
    /// `gateway::rollback_sessions`, que são o resto do mecanismo.
    pub rs_seq_window: u32,
    /// Load routed MoE expert metadata without allocating expert weights in llama.cpp.
    pub moe_external_executor: bool,
    /// Per-layer ownership passed to the generic llama.cpp hook. Zero keeps
    /// the native experts, one delegates the whole layer to the device cache.
    pub moe_external_executor_layers: Vec<u8>,
}

pub fn apply_cache_types(
    cparams: &mut sys::llama_context_params,
    cache_type_k: KvCacheType,
    cache_type_v: KvCacheType,
) {
    cparams.type_k = cache_type_to_ggml(cache_type_k);
    cparams.type_v = cache_type_to_ggml(cache_type_v);
}

pub(super) fn cache_type_to_ggml(cache_type: KvCacheType) -> sys::ggml_type {
    match cache_type {
        KvCacheType::F32 => sys::GGML_TYPE_F32,
        KvCacheType::F16 => sys::GGML_TYPE_F16,
        KvCacheType::Bf16 => sys::GGML_TYPE_BF16,
        KvCacheType::Q8_0 => sys::GGML_TYPE_Q8_0,
        KvCacheType::Q4_0 => sys::GGML_TYPE_Q4_0,
        KvCacheType::Q4_1 => sys::GGML_TYPE_Q4_1,
        KvCacheType::Iq4Nl => sys::GGML_TYPE_IQ4_NL,
        KvCacheType::Q5_0 => sys::GGML_TYPE_Q5_0,
        KvCacheType::Q5_1 => sys::GGML_TYPE_Q5_1,
    }
}

// ── Engine ────────────────────────────────────────────────────────────────────

/// Os pesos, carregados uma vez e compartilhados por todos os contextos.
///
/// Separar isto do contexto é o que torna o paralelismo viável sem duplicar
/// memória: o que pesa são os pesos (gigabytes), e o llama.cpp os trata como
/// somente-leitura depois da carga - vários `llama_context` podem apontar para
/// o mesmo `llama_model`. O contexto carrega o cache KV; o `n_ctx` interno já
/// é a capacidade total necessária para todas as sequências.
pub struct LoadedModel {
    pub(super) model: *mut sys::llama_model,
    pub(super) vocab: *const sys::llama_vocab,
    pub n_embd: usize,
    pub n_vocab: usize,
    /// Whether this model instance has the optional `nextn` tensors loaded.
    /// `llama_model_n_layer_nextn` only reports metadata and remains positive
    /// when the caller deliberately loaded the target with `load_mtp=false`.
    pub(super) mtp_weights_loaded: bool,
}

// SAFETY: depois da carga o `llama_model` é somente-leitura; é o próprio
// llama.cpp que permite vários contextos sobre um modelo.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

/// Primeiro palpite para o buffer do template de chat. Não é limite: quando o
/// llama.cpp informa que precisa de mais, `apply_chat_template` realoca com o
/// tamanho exato e renderiza de novo.
pub(super) const CHAT_TEMPLATE_INITIAL_BUF: usize = 65536;

/// Quantas sequências de KV o contexto precisa reservar para `max_sessions`
/// sessões simultâneas.
///
/// **Duas por sessão**: uma guarda o prompt e ninguém escreve nela; a outra
/// recebe a geração do turno e é apagada inteira no fim (ver
/// `LlmEngine::kv_copy_seq` e `infer_server::InferState::open_generation`). O
/// par é fixo - a sequência de geração da sessão que ocupa `seq` é
/// `seq + max_sessions` -, então nunca falta slot de geração para uma sessão
/// que já tem slot de prompt.
///
/// Vem sempre acompanhado de `kv_unified = true`: sem ele o llama.cpp fatia
/// `n_ctx` por sequência (`n_ctx_seq = n_ctx / n_seq_max`) e dobrar as
/// sequências cortaria a janela de prompt pela metade. Unificado, as duas
/// sequências dividem as mesmas células, a janela continua sendo `n_ctx`
/// inteiro e a cópia prompt→geração não move tensor nenhum.
///
/// **O que isto custa, declarado.** Duas famílias de cache não compartilham
/// células e passam a reservar de verdade duas fatias por sessão:
///
/// - o estado recorrente (`llama_memory_recurrent`, dimensionado por
///   `n_seq_max`), porque estado de prompt e estado de geração são estados
///   diferentes - não há como um modelo Mamba/GDN reter o prompt sem guardar o
///   estado dele em separado;
/// - o cache de estados com múltiplos fluxos, que hoje ignora o pedido de
///   unificação (`unified_raw = false`) e mantém um stream por sequência.
///
/// Em ambos os casos o `fit_memory` (`llama_rs_fit_params`, que recebe estes
/// mesmos `cparams`) dimensiona o contexto já contando as duas sequências, e
/// `/health` publica `n_seq_max` para o operador conferir. A alternativa seria
/// oferecer o cache só a alguns modelos, que é exatamente o que a decisão de
/// produto recusou.
pub fn kv_seq_budget(max_sessions: u32) -> u32 {
    max_sessions.max(1).saturating_mul(2)
}

pub(super) const DEFAULT_GPU_LAYERS: i32 = 99;
pub(super) const FIT_MEMORY_MIN_MARGIN_BYTES: usize = 4 * 1024 * 1024;
pub(super) const FIT_MIN_CONTEXT: u32 = 4096;
/// The CUDA graph and kernel workspaces scale with the microbatch, while the
/// public `n_batch` can stay large enough to accept a prompt chunk. A smaller
/// ubatch is the safe automatic fallback before llama.cpp reserves those
/// workspaces on a nearly-full GPU.
pub(super) const FIT_MAX_UBATCH: u32 = crate::DEFAULT_N_UBATCH;
/// Exceção aprovada para o contexto principal que usa o executor externo de
/// MoE. O microlote menor devolve VRAM ao cache residente; não é a política
/// do contexto denso nem do MTP, que continuam em `FIT_MAX_UBATCH`.
pub(super) const MOE_EXTERNAL_MAX_UBATCH: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FitDeviceMemory {
    pub(super) total_bytes: usize,
    pub(super) free_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FitMemorySnapshot {
    pub(super) total_bytes: usize,
    pub(super) free_bytes: usize,
    pub(super) margin_bytes: usize,
}

pub(super) fn fit_memory_configured_margin_bytes(configured_margin_mb: u32) -> usize {
    usize::try_from(u64::from(configured_margin_mb) * 1024 * 1024)
        .unwrap_or(usize::MAX)
        .max(FIT_MEMORY_MIN_MARGIN_BYTES)
}

/// Returns the amount that `common_fit_params` should leave unused on one
/// device. The backend's `free_bytes` is already the memory available to POM:
/// CUDA obtains it from `cudaMemGetInfo`, so allocations owned by other
/// processes are already excluded. The margin therefore covers only allocator
/// and late-workspace drift, plus a process that starts after this snapshot.
///
/// When the card has less free memory than the requested reserve, using the
/// whole free amount would make every positive allocation fail the fit. A
/// smaller reserve is still useful until the card cannot even provide the
/// minimum safety gap; at that point zero is the only honest target.
pub(super) fn fit_memory_margin_bytes(memory: FitDeviceMemory, configured_margin_mb: u32) -> usize {
    let free_bytes = memory.free_bytes.min(memory.total_bytes);
    if free_bytes < FIT_MEMORY_MIN_MARGIN_BYTES {
        return 0;
    }
    free_bytes.min(fit_memory_configured_margin_bytes(configured_margin_mb))
}

pub(super) fn fit_memory_margins(
    memories: &[FitDeviceMemory],
    n_devices: usize,
    configured_margin_mb: u32,
) -> Vec<usize> {
    let fallback = fit_memory_configured_margin_bytes(configured_margin_mb);
    (0..n_devices)
        .map(|index| {
            memories
                .get(index)
                .map(|memory| fit_memory_margin_bytes(*memory, configured_margin_mb))
                .unwrap_or(fallback)
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MtpFitMode {
    Disabled,
    Vram,
    Ram,
}

pub(super) fn mtp_fit_mode_for_model(
    requested: MtpFitMode,
    has_nextn: bool,
    architecture: Option<&str>,
) -> (MtpFitMode, Option<String>) {
    if requested == MtpFitMode::Disabled || !has_nextn {
        return (mtp_mode_for_nextn_availability(requested, has_nextn), None);
    }

    if let Some(reason) = crate::mtp::mtp_graph_unavailable_reason(architecture) {
        return (MtpFitMode::Disabled, Some(reason));
    }

    (requested, None)
}

pub(super) fn mtp_mode_for_nextn_availability(
    requested: MtpFitMode,
    has_nextn: bool,
) -> MtpFitMode {
    if !has_nextn {
        MtpFitMode::Disabled
    } else {
        requested
    }
}

/// Decides where MTP goes once the joint fit answered.
///
/// By default the joint fit may only enable MTP in VRAM if it keeps the full
/// offload the main model got on its own: `common_fit_params` counts the main
/// layers and the `nextn` layers together, and accepting a reduced result
/// would let the MTP suffix push main layers to the CPU. MTP then goes to host
/// memory.
///
/// With `mtp_first` (the deployment placed MTP in VRAM), MTP owns
/// its share of VRAM: any joint fit that succeeded is accepted, and the main
/// model takes what is left, giving up layers to host memory if it has to.
pub(super) fn choose_mtp_fit_mode(
    requested: MtpFitMode,
    primary_gpu_layers: i32,
    mtp_gpu_layers: Option<i32>,
    mtp_first: bool,
) -> MtpFitMode {
    match requested {
        MtpFitMode::Disabled => MtpFitMode::Disabled,
        MtpFitMode::Vram if primary_gpu_layers == -1 && mtp_gpu_layers == Some(-1) => {
            MtpFitMode::Vram
        }
        MtpFitMode::Vram if mtp_first && mtp_gpu_layers.is_some() => MtpFitMode::Vram,
        MtpFitMode::Vram | MtpFitMode::Ram => MtpFitMode::Ram,
    }
}

/// The MTP mode when no GPU fit ran (experts kept in RAM, explicit GPU layers,
/// or a fit that found nothing). Without a plan to protect, MTP goes to host
/// memory - unless the deployment placed it in VRAM, which is then honored: the
/// load still falls back to RAM if the VRAM context really fails.
pub(super) fn mtp_mode_without_gpu_fit(requested: MtpFitMode, placement: crate::MtpPlacement) -> MtpFitMode {
    match requested {
        MtpFitMode::Disabled => MtpFitMode::Disabled,
        MtpFitMode::Vram if placement == crate::MtpPlacement::Vram => MtpFitMode::Vram,
        MtpFitMode::Vram | MtpFitMode::Ram => MtpFitMode::Ram,
    }
}

/// What the compute buffers of an external MTP draft context are allowed for
/// in the fit. The draft runs one block over a `n_ubatch` batch; 510 MiB was
/// measured for a Qwen NextN block with `n_ubatch = 512`.
const EXTERNAL_MTP_DRAFT_COMPUTE_BYTES: usize = 512 << 20;

/// Device memory an external MTP draft needs besides its KV: its weights,
/// taken as the size of the file that is really loaded (an upper bound: the
/// token embedding in it stays in host memory), plus its compute buffers.
/// A shared draft is loaded as its completed copy, which carries the target's
/// embedding and output head and is larger than the draft itself (1.8 GB
/// became 2.7 GB), so that copy is what is measured. `None` when there is no
/// external draft or its file cannot be read.
pub(super) fn external_mtp_draft_bytes(
    draft_path: Option<&str>,
    target_path: &str,
) -> Option<usize> {
    let draft = std::path::Path::new(draft_path?);
    let loaded = crate::mtp_draft::ready_draft_path(draft, std::path::Path::new(target_path))
        .unwrap_or_else(|_| draft.to_path_buf());
    let size = std::fs::metadata(loaded).ok()?.len();
    usize::try_from(size)
        .ok()?
        .checked_add(EXTERNAL_MTP_DRAFT_COMPUTE_BYTES)
}

/// Adds the external draft's weights and compute to the device its block is
/// loaded on: the one that holds the last NextN layer's KV, which is the last
/// device with a cost, or the last device when the KV costs nothing.
pub(super) fn with_external_mtp_draft(
    mut costs: Vec<usize>,
    draft_bytes: Option<usize>,
) -> Vec<usize> {
    let Some(draft_bytes) = draft_bytes else {
        return costs;
    };
    let device = costs
        .iter()
        .rposition(|&cost| cost > 0)
        .or(costs.len().checked_sub(1));
    if let Some(device) = device {
        costs[device] = costs[device].saturating_add(draft_bytes);
    }
    costs
}

pub(super) fn fit_memory_margins_with_mtp(
    margins: &[usize],
    mtp_cost: &[usize],
    mode: MtpFitMode,
) -> Vec<usize> {
    margins
        .iter()
        .enumerate()
        .map(|(index, &margin)| {
            if mode == MtpFitMode::Vram {
                margin.saturating_add(mtp_cost.get(index).copied().unwrap_or(0))
            } else {
                margin
            }
        })
        .collect()
}

/// The split and KV budget used for the accepted MTP fit. `tensor_split` is
/// always explicit here, including when llama.cpp's fit left it all zero to
/// request the native free-memory split.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct MtpVramEstimate {
    pub costs: Vec<usize>,
    pub tensor_split: Vec<f32>,
}

/// Estimates the MTP context's KV cache without loading the model or building
/// its graph. The fitted `n_ctx` is already the total context capacity, and
/// `kv_unified=true` allocates one `n_ctx`-cell buffer regardless of
/// `n_seq_max`; only the sequence IDs, not the cell count, use
/// `kv_seq_budget`.
///
/// llama.cpp places each KV layer on `model.dev_layer(il)`. The NextN layers
/// are a suffix of the model, but a multi-layer NextN suffix can cross device
/// boundaries. Use the joint fit's `tensor_split` to mirror its layer mapping
/// rather than charging every NextN layer to the final device. When the fit
/// returns an all-zero split, it made no placement changes and the model
/// loader will choose its native split from free memory. Resolve that split
/// from the same memory snapshot and pass it explicitly to the real load, so
/// the reservation and placement agree instead of rejecting every multi-GPU
/// fit that needed no adjustment.
pub(super) fn estimate_mtp_vram_costs(
    info: &crate::gguf::GgufInfo,
    cfg: &EngineConfig,
    n_ctx: u32,
    tensor_split: &[f32],
    device_memories: &[FitDeviceMemory],
    n_devices: usize,
) -> Option<MtpVramEstimate> {
    let resolved_split = resolve_mtp_tensor_split(tensor_split, device_memories, n_devices)?;
    let mut costs = vec![0usize; n_devices];
    if !cfg.offload_kqv || n_devices == 0 {
        return Some(MtpVramEstimate {
            costs,
            tensor_split: resolved_split,
        });
    }

    let block_count = info.block_count?;
    let nextn_layers = info.nextn_predict_layers.unwrap_or(1).max(1);
    let first_nextn_layer = block_count.checked_sub(nextn_layers)?;
    let last_layer = block_count.checked_sub(1)?;
    let geometry = info
        .kv_geometry_for_range(first_nextn_layer, last_layer)
        .ok()?;

    for (offset, layer_geometry) in geometry.iter().enumerate() {
        let layer = first_nextn_layer.checked_add(u32::try_from(offset).ok()?)?;
        let device = llama_layer_device_from_split(layer, block_count, &resolved_split, n_devices)?;
        let bytes = crate::kv_budget::kv_bytes_for_layers(
            u64::from(n_ctx),
            1,
            std::slice::from_ref(layer_geometry),
            cfg.cache_type_k,
            cfg.cache_type_v,
        )
        .ok()?;
        let bytes = usize::try_from(bytes).ok()?;
        costs[device] = costs[device].checked_add(bytes)?;
    }
    Some(MtpVramEstimate {
        costs,
        tensor_split: resolved_split,
    })
}

/// Resolve an all-zero split exactly as llama.cpp's model loader does: use
/// free-memory weights in device order. The returned vector retains the
/// compile-time capacity because llama.cpp copies `llama_max_devices()` values.
fn resolve_mtp_tensor_split(
    tensor_split: &[f32],
    device_memories: &[FitDeviceMemory],
    n_devices: usize,
) -> Option<Vec<f32>> {
    if n_devices == 0 {
        return Some(tensor_split.to_vec());
    }
    let splits = tensor_split.get(..n_devices)?;
    if splits
        .iter()
        .any(|split| !split.is_finite() || *split < 0.0)
    {
        return None;
    }
    if splits.iter().any(|split| *split > 0.0) {
        let split_sum = splits.iter().copied().sum::<f32>();
        return (split_sum.is_finite() && split_sum > 0.0).then(|| tensor_split.to_vec());
    }

    let mut resolved = tensor_split.to_vec();
    if n_devices == 1 {
        resolved[0] = 1.0;
        return Some(resolved);
    }
    let memories = device_memories.get(..n_devices)?;
    let mut split_sum = 0.0f32;
    for (index, memory) in memories.iter().enumerate() {
        if memory.total_bytes == 0 || memory.free_bytes > memory.total_bytes {
            return None;
        }
        let free_bytes = memory.free_bytes as f32;
        if !free_bytes.is_finite() {
            return None;
        }
        resolved[index] = free_bytes;
        split_sum += free_bytes;
    }
    (split_sum.is_finite() && split_sum > 0.0).then_some(resolved)
}

/// Mirrors llama.cpp's layer-device selection for `LLAMA_SPLIT_MODE_LAYER`:
/// cumulative `tensor_split` fractions are compared with `il/(n_layer_all+1)`.
fn llama_layer_device_from_split(
    layer: u32,
    block_count: u32,
    tensor_split: &[f32],
    n_devices: usize,
) -> Option<usize> {
    if n_devices == 1 {
        return Some(0);
    }
    let splits = tensor_split.get(..n_devices)?;
    if splits
        .iter()
        .any(|split| !split.is_finite() || *split < 0.0)
    {
        return None;
    }
    let split_sum = splits.iter().copied().sum::<f32>();
    if !split_sum.is_finite() || split_sum <= 0.0 {
        return None;
    }

    let layer_slots = block_count.checked_add(1)?;
    let layer_fraction = layer as f32 / layer_slots as f32;
    let mut cumulative = 0.0f32;
    for (device, split) in splits.iter().enumerate() {
        cumulative += *split;
        if cumulative / split_sum > layer_fraction {
            return Some(device);
        }
    }
    Some(n_devices - 1)
}

/// Quantas GPUs físicas o fit deve considerar.
///
/// `max_devices` é `llama_max_devices()`, a **capacidade de compilação** dos
/// vetores que o wrapper C preenche (16), nunca o número de placas presentes.
/// Confundir os dois fazia o aviso de snapshot parcial disparar em toda máquina
/// com menos de 16 GPUs, e - o defeito real - fazia a reserva de KV do contexto
/// MTP cair no índice 15, um dispositivo inexistente, em vez da última placa
/// verdadeira. O fit então acreditava ter VRAM livre que já estava prometida.
///
/// A contagem elegível vem de `llama_rs_get_devices`; quando ela é vazia, vale
/// a contagem do snapshot de memória. Sem nenhuma das duas resta um slot para
/// o host, porque o fit sempre planeja ao menos um destino.
pub(super) fn fit_device_count(
    available_count: usize,
    memory_count: usize,
    max_devices: usize,
) -> usize {
    let counted = if available_count > 0 {
        available_count
    } else {
        memory_count
    };
    counted.clamp(1, max_devices.max(1))
}

pub(super) fn fit_memory_snapshots(
    memories: &[FitDeviceMemory],
    margins: &[usize],
) -> Vec<FitMemorySnapshot> {
    margins
        .iter()
        .enumerate()
        .map(|(index, &margin_bytes)| {
            let memory = memories.get(index).copied().unwrap_or(FitDeviceMemory {
                total_bytes: 0,
                free_bytes: 0,
            });
            FitMemorySnapshot {
                total_bytes: memory.total_bytes,
                free_bytes: memory.free_bytes,
                margin_bytes,
            }
        })
        .collect()
}

#[cfg(test)]
mod fit_memory_margin_tests {
    use super::*;

    const MIB: usize = 1024 * 1024;

    fn memory(total_mib: usize, free_mib: usize) -> FitDeviceMemory {
        FitDeviceMemory {
            total_bytes: total_mib * MIB,
            free_bytes: free_mib * MIB,
        }
    }

    #[test]
    fn without_other_processes_reserves_the_small_default() {
        assert_eq!(
            fit_memory_margin_bytes(memory(16 * 1024, 16 * 1024), 256),
            256 * MIB
        );
    }

    #[test]
    fn existing_processes_are_not_discounted_twice() {
        assert_eq!(
            fit_memory_margin_bytes(memory(16 * 1024, 4 * 1024), 256),
            256 * MIB
        );
    }

    #[test]
    fn free_memory_below_the_request_shrinks_the_reserve() {
        assert_eq!(
            fit_memory_margin_bytes(memory(16 * 1024, 64), 256),
            64 * MIB
        );
    }

    #[test]
    fn memory_too_small_even_for_the_minimum_does_not_underflow() {
        assert_eq!(fit_memory_margin_bytes(memory(16 * 1024, 2), 256), 0);
    }

    #[test]
    fn margins_are_computed_independently_for_each_device() {
        let margins = fit_memory_margins(
            &[
                memory(16 * 1024, 16 * 1024),
                memory(24 * 1024, 64),
                memory(8 * 1024, 2),
            ],
            3,
            256,
        );

        assert_eq!(margins, vec![256 * MIB, 64 * MIB, 0]);
    }

    #[test]
    fn fit_snapshot_preserves_memory_and_budget_per_device() {
        let memories = vec![memory(16 * 1024, 15 * 1024), memory(24 * 1024, 20 * 1024)];
        let margins = fit_memory_margins(&memories, memories.len(), 256);

        assert_eq!(
            fit_memory_snapshots(&memories, &margins),
            vec![
                FitMemorySnapshot {
                    total_bytes: 16 * 1024 * MIB,
                    free_bytes: 15 * 1024 * MIB,
                    margin_bytes: 256 * MIB,
                },
                FitMemorySnapshot {
                    total_bytes: 24 * 1024 * MIB,
                    free_bytes: 20 * 1024 * MIB,
                    margin_bytes: 256 * MIB,
                },
            ]
        );
    }

    #[test]
    fn fit_uses_the_eligible_device_count_and_not_the_generic_limit() {
        assert_eq!(fit_device_count(2, 0, 16), 2);
        assert_eq!(fit_device_count(0, 2, 16), 2);
        assert_eq!(fit_device_count(0, 0, 16), 1);
    }

    #[test]
    fn a_file_without_nextn_asks_for_no_mtp_load_or_context() {
        assert_eq!(
            mtp_mode_for_nextn_availability(MtpFitMode::Vram, false),
            MtpFitMode::Disabled
        );
        assert_eq!(
            mtp_mode_for_nextn_availability(MtpFitMode::Ram, false),
            MtpFitMode::Disabled
        );
        assert_eq!(
            mtp_mode_for_nextn_availability(MtpFitMode::Vram, true),
            MtpFitMode::Vram
        );
    }

    #[test]
    fn an_architecture_with_nextn_and_no_graph_turns_off_the_mtp_fit_with_a_reason() {
        let (mode, reason) = mtp_fit_mode_for_model(MtpFitMode::Vram, true, Some("graniteswitch"));

        assert_eq!(mode, MtpFitMode::Disabled);
        assert_eq!(
            reason.as_deref(),
            Some("a arquitetura graniteswitch não possui grafo MTP suportado pelo llama.cpp")
        );
    }

    #[test]
    fn an_architecture_with_an_mtp_graph_keeps_the_vram_attempt() {
        assert_eq!(
            mtp_fit_mode_for_model(MtpFitMode::Vram, true, Some("qwen35moe")),
            (MtpFitMode::Vram, None)
        );
    }

    fn cfg_for_mtp_vram_estimate(n_ctx: u32, max_sessions: u32) -> EngineConfig {
        EngineConfig {
            n_ctx,
            max_sessions,
            offload_kqv: true,
            cache_type_k: crate::KvCacheType::F16,
            cache_type_v: crate::KvCacheType::F16,
            fit_memory: true,
            fit_memory_margin_mb: 256,
            n_batch: 128,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: DEFAULT_GPU_LAYERS,
            n_threads: 0,
            need_embeddings: false,
            mtp: true,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            moe_external_executor: false,
            moe_external_executor_layers: Vec::new(),
            mmproj_path: None,
            mtp_draft_path: None,
        }
    }

    #[test]
    fn mtp_reservation_counts_the_unified_context_cells_once() {
        let info = crate::gguf::GgufInfo {
            block_count: Some(41),
            nextn_predict_layers: Some(1),
            attention_head_count_kv: Some(vec![2]),
            attention_key_length: Some(vec![64]),
            attention_value_length: Some(vec![64]),
            ..Default::default()
        };
        let cfg = cfg_for_mtp_vram_estimate(4096, 3);
        let fitted_n_ctx = 2048;
        // `n_ctx` is the total context, while max_sessions only increases the
        // sequence-ID budget. llama.cpp allocates one unified n_ctx-cell KV.
        let tensor_split = [39.0, 3.0];

        assert_eq!(kv_seq_budget(cfg.max_sessions), 6);
        assert_eq!(
            estimate_mtp_vram_costs(&info, &cfg, fitted_n_ctx, &tensor_split, &[], 2),
            Some(MtpVramEstimate {
                costs: vec![0, 2048 * 2 * 64 * 2 * 2],
                tensor_split: vec![39.0, 3.0],
            })
        );
    }

    #[test]
    fn multi_layer_nextn_reservation_follows_llama_cpp_layer_split() {
        let info = crate::gguf::GgufInfo {
            block_count: Some(10),
            nextn_predict_layers: Some(4),
            attention_head_count_kv: Some(vec![1]),
            attention_key_length: Some(vec![64]),
            attention_value_length: Some(vec![64]),
            ..Default::default()
        };
        let cfg = cfg_for_mtp_vram_estimate(1024, 2);
        // With 11 layer slots (including the output head), llama.cpp's split
        // puts nextn layers 6-7 on device 0 and layers 8-9 on device 1.
        let tensor_split = [7.5, 3.5];
        let bytes_per_layer = 1024 * 2 * 64 * 2;

        assert_eq!(
            estimate_mtp_vram_costs(&info, &cfg, cfg.n_ctx, &tensor_split, &[], 2),
            Some(MtpVramEstimate {
                costs: vec![2 * bytes_per_layer, 2 * bytes_per_layer],
                tensor_split: vec![7.5, 3.5],
            })
        );
    }

    #[test]
    fn an_external_mtp_draft_is_reserved_on_the_device_of_its_block() {
        // The KV of the last NextN layer is on device 1: the draft goes there.
        assert_eq!(
            with_external_mtp_draft(vec![0, 160], Some(3000)),
            vec![0, 3160]
        );
        assert_eq!(
            with_external_mtp_draft(vec![160, 0], Some(3000)),
            vec![3160, 0]
        );
        // No KV cost (host KV): the last device, where the final layers are.
        assert_eq!(
            with_external_mtp_draft(vec![0, 0], Some(3000)),
            vec![0, 3000]
        );
        // A draft inside the model itself reserves nothing extra.
        assert_eq!(with_external_mtp_draft(vec![0, 160], None), vec![0, 160]);
        assert_eq!(
            with_external_mtp_draft(Vec::new(), Some(3000)),
            Vec::<usize>::new()
        );

        assert_eq!(external_mtp_draft_bytes(None, "/models/target.gguf"), None);
        assert_eq!(
            external_mtp_draft_bytes(Some("/nonexistent/draft.gguf"), "/models/target.gguf"),
            None
        );
        // A file that is not a readable draft is measured as it is.
        let draft =
            std::env::temp_dir().join(format!("pom-draft-size-{}.gguf", std::process::id()));
        std::fs::write(&draft, vec![0u8; 4096]).unwrap();
        let bytes = external_mtp_draft_bytes(draft.to_str(), "/models/target.gguf");
        std::fs::remove_file(&draft).unwrap();
        assert_eq!(bytes, Some(4096 + EXTERNAL_MTP_DRAFT_COMPUTE_BYTES));
    }

    #[test]
    fn multi_device_mtp_fit_requires_a_split_or_a_valid_free_memory_snapshot() {
        let info = crate::gguf::GgufInfo {
            block_count: Some(41),
            nextn_predict_layers: Some(1),
            attention_head_count_kv: Some(vec![2]),
            attention_key_length: Some(vec![64]),
            attention_value_length: Some(vec![64]),
            ..Default::default()
        };
        let cfg = cfg_for_mtp_vram_estimate(4096, 1);

        assert_eq!(
            estimate_mtp_vram_costs(&info, &cfg, cfg.n_ctx, &[0.0, 0.0], &[], 2),
            None
        );
    }

    #[test]
    fn all_zero_joint_fit_uses_native_free_memory_split_and_keeps_capacity() {
        let info = crate::gguf::GgufInfo {
            block_count: Some(41),
            nextn_predict_layers: Some(1),
            attention_head_count_kv: Some(vec![2]),
            attention_key_length: Some(vec![64]),
            attention_value_length: Some(vec![64]),
            ..Default::default()
        };
        let cfg = cfg_for_mtp_vram_estimate(4096, 1);
        let tensor_split = vec![0.0; 16];
        let memories = [
            FitDeviceMemory {
                total_bytes: 64,
                free_bytes: 39,
            },
            FitDeviceMemory {
                total_bytes: 16,
                free_bytes: 3,
            },
        ];
        let expected_cost = 4096 * 2 * 64 * 2 * 2;

        let estimate = estimate_mtp_vram_costs(&info, &cfg, cfg.n_ctx, &tensor_split, &memories, 2)
            .expect("all-zero fit split should resolve from free-memory weights");
        let mut expected_split = vec![0.0; 16];
        expected_split[0] = 39.0;
        expected_split[1] = 3.0;
        assert_eq!(estimate.costs, vec![0, expected_cost]);
        assert_eq!(estimate.tensor_split, expected_split);
    }

    #[test]
    fn mtp_reservation_uses_the_physical_device_count() {
        let info = crate::gguf::GgufInfo {
            block_count: Some(41),
            nextn_predict_layers: Some(1),
            attention_head_count_kv: Some(vec![2]),
            attention_key_length: Some(vec![64]),
            attention_value_length: Some(vec![64]),
            ..Default::default()
        };
        let cfg = cfg_for_mtp_vram_estimate(4096, 1);
        let physical = fit_device_count(2, 2, 16);
        let tensor_split = [39.0, 3.0];
        let estimate =
            estimate_mtp_vram_costs(&info, &cfg, cfg.n_ctx, &tensor_split, &[], physical)
                .expect("the MTP context cost should be estimable");

        assert_eq!(estimate.costs.len(), 2);
        assert!(
            estimate.costs[1] > 0,
            "the final nextn layer maps to the final device"
        );
    }

    #[test]
    fn an_mtp_request_reserves_the_second_context_cost_on_each_device() {
        let margins = vec![256 * MIB, 256 * MIB];
        let mtp_cost = vec![727 * MIB, 311 * MIB];

        assert_eq!(
            fit_memory_margins_with_mtp(&margins, &mtp_cost, MtpFitMode::Vram),
            vec![983 * MIB, 567 * MIB]
        );
    }

    #[test]
    fn a_tight_mtp_cannot_reduce_the_main_layers() {
        // Reproduz a inversão de 27fda8b: o fit principal cabe inteiro, mas
        // o fit conjunto só cabe se remover camadas. Nesse caso o MTP cede.
        assert_eq!(
            choose_mtp_fit_mode(MtpFitMode::Vram, -1, Some(50), false),
            MtpFitMode::Ram
        );
    }

    #[test]
    fn mtp_in_vram_is_only_accepted_when_the_joint_fit_also_fits() {
        assert_eq!(
            choose_mtp_fit_mode(MtpFitMode::Vram, -1, Some(-1), false),
            MtpFitMode::Vram
        );
        assert_eq!(
            choose_mtp_fit_mode(MtpFitMode::Vram, -1, None, false),
            MtpFitMode::Ram
        );
    }

    #[test]
    fn a_reduced_main_model_does_not_entitle_mtp_to_vram() {
        assert_eq!(
            choose_mtp_fit_mode(MtpFitMode::Vram, 50, Some(-1), false),
            MtpFitMode::Ram
        );
    }

    #[test]
    fn mtp_placed_in_vram_keeps_its_vram_and_the_model_takes_the_rest() {
        // The joint fit had to take main layers off the GPU: accepted.
        assert_eq!(
            choose_mtp_fit_mode(MtpFitMode::Vram, -1, Some(40), true),
            MtpFitMode::Vram
        );
        // Even when the main model alone was already partly in host memory.
        assert_eq!(
            choose_mtp_fit_mode(MtpFitMode::Vram, 30, Some(26), true),
            MtpFitMode::Vram
        );
        // No joint fit at all: there is nothing to accept.
        assert_eq!(
            choose_mtp_fit_mode(MtpFitMode::Vram, -1, None, true),
            MtpFitMode::Ram
        );
        assert_eq!(
            choose_mtp_fit_mode(MtpFitMode::Disabled, -1, Some(-1), true),
            MtpFitMode::Disabled
        );
    }

    #[test]
    fn mtp_off_keeps_the_main_fit_distribution() {
        assert_eq!(
            choose_mtp_fit_mode(MtpFitMode::Disabled, 50, Some(50), false),
            MtpFitMode::Disabled
        );
    }

    #[test]
    fn unchecked_mtp_reserves_no_budget() {
        let margins = vec![256 * MIB, 256 * MIB];
        let mtp_cost = vec![727 * MIB, 311 * MIB];

        assert_eq!(
            fit_memory_margins_with_mtp(&margins, &mtp_cost, MtpFitMode::Disabled),
            margins
        );
    }

    #[test]
    fn unchecked_mtp_does_not_trigger_the_architecture_preflight() {
        assert_eq!(
            mtp_fit_mode_for_model(MtpFitMode::Disabled, true, Some("graniteswitch")),
            (MtpFitMode::Disabled, None)
        );
    }

    /// Um `.gguf` mínimo, só o bastante para `has_nextn_tensors`/`read_info`
    /// lerem os nomes dos tensores: sem KVs, um tensor de uma dimensão por
    /// nome pedido.
    fn minimal_gguf_with_tensor_names(names: &[&str]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
        b.extend_from_slice(&2u32.to_le_bytes()); // version
        b.extend_from_slice(&(names.len() as u64).to_le_bytes()); // n_tensors
        b.extend_from_slice(&0u64.to_le_bytes()); // n_kv
        for name in names {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&1u32.to_le_bytes()); // n_dimensions
            b.extend_from_slice(&1u64.to_le_bytes()); // dim[0]
            b.extend_from_slice(&0u32.to_le_bytes()); // dtype
            b.extend_from_slice(&0u64.to_le_bytes()); // offset
        }
        b
    }

    /// The cache stays only where the fit sent routed experts to host memory.
    #[test]
    fn a_plan_keeps_experts_in_host_only_with_an_exps_override() {
        let mut plan = empty_fit_plan(&cfg_for_fit_plan());
        assert!(!plan_keeps_experts_in_host(&plan));
        plan.tensor_buft_overrides = vec![
            sys::llama_model_tensor_buft_override { pattern: c"blk\\.3\\.ffn_(up|down|gate_up|gate)_(ch|)exps".as_ptr(), buft: ptr::null_mut() },
            sys::llama_model_tensor_buft_override { pattern: ptr::null(), buft: ptr::null_mut() },
        ];
        assert!(plan_keeps_experts_in_host(&plan));
        plan.tensor_buft_overrides = vec![
            sys::llama_model_tensor_buft_override { pattern: c"blk\\.3\\.attn_q".as_ptr(), buft: ptr::null_mut() },
        ];
        assert!(!plan_keeps_experts_in_host(&plan));
    }

    /// Without a GPU fit (experts kept in RAM), MTP placed in VRAM stays there.
    #[test]
    fn mtp_placed_in_vram_is_honored_without_a_gpu_fit() {
        use crate::MtpPlacement;
        assert_eq!(mtp_mode_without_gpu_fit(MtpFitMode::Vram, MtpPlacement::Vram), MtpFitMode::Vram);
        assert_eq!(mtp_mode_without_gpu_fit(MtpFitMode::Vram, MtpPlacement::Auto), MtpFitMode::Ram);
        assert_eq!(mtp_mode_without_gpu_fit(MtpFitMode::Ram, MtpPlacement::Ram), MtpFitMode::Ram);
        assert_eq!(mtp_mode_without_gpu_fit(MtpFitMode::Disabled, MtpPlacement::Vram), MtpFitMode::Disabled);
    }

    /// A cache that does not fit is halved until it does, then turned off.
    #[test]
    fn the_moe_cache_halves_then_turns_off() {
        let mut steps = vec![8000];
        while *steps.last().unwrap() > 0 {
            steps.push(moe_cache_retry_mib(*steps.last().unwrap()));
        }
        assert_eq!(steps, vec![8000, 4000, 2000, 1000, 0]);
    }

    fn cfg_for_fit_plan() -> EngineConfig {
        EngineConfig {
            n_ctx: 4096,
            max_sessions: 1,
            offload_kqv: true,
            cache_type_k: crate::KvCacheType::F16,
            cache_type_v: crate::KvCacheType::F16,
            // Toma o retorno antecipado de `fit_plan` (sem `fit_can_adjust_layers`),
            // que já passou pela decisão de disponibilidade do MTP mas não
            // precisa sondar dispositivo real nenhum - o teste fica sobre a
            // decisão, não sobre o fit de memória.
            fit_memory: false,
            fit_memory_margin_mb: 256,
            n_batch: 128,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: DEFAULT_GPU_LAYERS,
            n_threads: 0,
            need_embeddings: false,
            mtp: true,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            moe_external_executor: false,
            moe_external_executor_layers: Vec::new(),
            mmproj_path: None,
            mtp_draft_path: None,
        }
    }

    /// Reproduz o defeito ao vivo na ksia/salem: implantar com MTP ligado e
    /// um rascunho externo já correto no disco (ver
    /// `models::catalog::find_mtp_draft_sibling`) ainda relatava "MTP
    /// indisponível: o arquivo não traz tensores nextn", porque `fit_plan`
    /// só olhava `has_nextn_tensors` do arquivo do modelo **principal** -
    /// nunca soube que `EngineConfig.mtp_draft_path` existe, mesmo esse campo
    /// já tendo sido introduzido para exatamente este caso.
    #[test]
    fn fit_plan_recognizes_an_external_mtp_draft_when_the_target_embeds_no_nextn() {
        let dir = std::env::temp_dir().join(format!(
            "pom-fit-plan-external-draft-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Arquitetura deliberadamente fora da lista gerada de arquiteturas
        // MTP (como `qwen4exp`): o alvo em si nunca vai construir o grafo
        // `DECODER_MTP`, e essa checagem não deve nem rodar quando quem tem
        // `nextn` é o rascunho, não o alvo.
        let target_path = dir.join("target-sem-nextn.gguf");
        std::fs::write(
            &target_path,
            minimal_gguf_with_tensor_names(&["token_embd.weight"]),
        )
        .unwrap();
        let draft_path = dir.join("mtp-target-sem-nextn-shared.gguf");
        std::fs::write(
            &draft_path,
            minimal_gguf_with_tensor_names(&["blk.0.nextn.eh_proj.weight"]),
        )
        .unwrap();

        let mut cfg = cfg_for_fit_plan();
        cfg.mtp_draft_path = Some(draft_path.to_string_lossy().into_owned());

        let plan = fit_plan(target_path.to_str().unwrap(), &cfg, MtpFitMode::Vram)
            .expect("fit_plan não deveria falhar com um GGUF válido");

        assert_ne!(
            plan.mtp_mode,
            MtpFitMode::Disabled,
            "um rascunho MTP externo válido deve manter o MTP tentável mesmo \
             sem nextn embutido no arquivo principal"
        );
        assert_eq!(
            plan.mtp_unavailable_reason, None,
            "não deveria haver razão de indisponibilidade quando o rascunho supre o nextn"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Sem `nextn` embutido e sem rascunho configurado, o comportamento
    /// permanece exatamente o de antes: MTP indisponível.
    #[test]
    fn fit_plan_stays_unavailable_without_nextn_and_without_a_draft() {
        let dir = std::env::temp_dir().join(format!(
            "pom-fit-plan-no-draft-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let target_path = dir.join("target-sem-nextn.gguf");
        std::fs::write(
            &target_path,
            minimal_gguf_with_tensor_names(&["token_embd.weight"]),
        )
        .unwrap();

        let cfg = cfg_for_fit_plan();
        let plan = fit_plan(target_path.to_str().unwrap(), &cfg, MtpFitMode::Vram)
            .expect("fit_plan não deveria falhar com um GGUF válido");

        assert_eq!(plan.mtp_mode, MtpFitMode::Disabled);

        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Parâmetros do modelo já ajustados pelo `common_fit_params`.
///
/// Os vetores ficam vivos até `llama_model_load_from_file` consumir os
/// ponteiros guardados em `llama_model_params`. Isso é necessário no caso de
/// mais de uma GPU local, em que o fit também escreve `tensor_split` e
/// `tensor_buft_overrides`.
pub(super) struct FitPlan {
    pub(super) config: EngineConfig,
    pub(super) tensor_split: Vec<f32>,
    pub(super) tensor_buft_overrides: Vec<sys::llama_model_tensor_buft_override>,
    /// A NULL-terminated device list, kept alive for the model load. It is
    /// empty when speed could not be ranked confidently, preserving llama's
    /// default device selection.
    pub(super) devices: Vec<sys::ggml_backend_dev_t>,
    /// Whether the target model should load its optional `nextn` weights in
    /// VRAM. `Ram` keeps those weights out of the target so the CPU companion
    /// can own them without changing the primary placement.
    pub(super) mtp_mode: MtpFitMode,
    /// Razão para manter o MTP desligado quando a metadata anuncia `nextn`,
    /// mas o fork não possui um grafo seguro para a arquitetura.
    pub(super) mtp_unavailable_reason: Option<String>,
    pub(super) applied: bool,
}

pub(super) fn fit_can_adjust_layers(cfg: &EngineConfig) -> bool {
    cfg.fit_memory && cfg.n_gpu_layers == DEFAULT_GPU_LAYERS && !cfg.moe_experts_in_ram
}

/// llama.cpp's `LLM_FFN_EXPS_REGEX`: the routed expert tensors `--cpu-moe`
/// sends to the CPU buffer.
pub(super) const MOE_EXPERTS_PATTERN: &std::ffi::CStr = c"\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

/// The next MoE expert cache size to try after a context failed to fit: half,
/// and off once half is under 512 MiB (fewer slots than that rarely hold the
/// experts of one token).
pub(super) fn moe_cache_retry_mib(mib: u32) -> u32 {
    let half = mib / 2;
    if half < 512 { 0 } else { half }
}

/// The cache size llama.cpp takes in bytes, from the deployment's MiB. It is
/// per GPU: with several GPUs each one caches the experts of its own layers.
pub(super) fn moe_cache_bytes(cfg: &EngineConfig) -> usize {
    cfg.moe_cache_mib as usize * 1024 * 1024
}

pub(super) fn empty_fit_plan(cfg: &EngineConfig) -> FitPlan {
    let mtp_mode = if cfg.mtp {
        MtpFitMode::Vram
    } else {
        MtpFitMode::Disabled
    };
    empty_fit_plan_with_mtp_mode(cfg, mtp_mode)
}

pub(super) fn empty_fit_plan_with_mtp_mode(cfg: &EngineConfig, mtp_mode: MtpFitMode) -> FitPlan {
    empty_fit_plan_with_mtp_mode_and_reason(cfg, mtp_mode, None)
}

pub(super) fn empty_fit_plan_with_mtp_mode_and_reason(
    cfg: &EngineConfig,
    mtp_mode: MtpFitMode,
    mtp_unavailable_reason: Option<String>,
) -> FitPlan {
    FitPlan {
        config: cfg.clone(),
        tensor_split: Vec::new(),
        tensor_buft_overrides: Vec::new(),
        devices: Vec::new(),
        mtp_mode,
        mtp_unavailable_reason,
        applied: false,
    }
}

/// O POM exporta o residual pelo callback, mas o llama.cpp também pode tentar
/// copiar o resultado de embeddings. Essa cópia usa `n_embd_out` como se fosse
/// a largura real do tensor; quando as duas larguras divergem, o pedido pode
/// ultrapassar `ggml_nbytes(tensor)` antes de o callback ser consumido.
pub(super) fn generic_embeddings_compatible(
    need_embeddings: bool,
    n_embd: usize,
    n_embd_out: usize,
) -> bool {
    need_embeddings && n_embd == n_embd_out
}

/// Whether this processor has the x86 instruction set of that name.
fn cpu_has(instruction_set: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        match instruction_set {
            "sse3" => std::arch::is_x86_feature_detected!("sse3"),
            "ssse3" => std::arch::is_x86_feature_detected!("ssse3"),
            "avx" => std::arch::is_x86_feature_detected!("avx"),
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            "bmi2" => std::arch::is_x86_feature_detected!("bmi2"),
            "f16c" => std::arch::is_x86_feature_detected!("f16c"),
            "fma" => std::arch::is_x86_feature_detected!("fma"),
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            "avx512vnni" => std::arch::is_x86_feature_detected!("avx512vnni"),
            _ => true,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = instruction_set;
        true
    }
}

/// The instruction sets the engine was built with that the processor lacks.
pub(super) fn missing_instruction_sets(
    built_with: &[&'static str],
    cpu_has: impl Fn(&str) -> bool,
) -> Vec<&'static str> {
    built_with
        .iter()
        .copied()
        .filter(|instruction_set| !cpu_has(instruction_set))
        .collect()
}

/// Registers the llama.cpp backends; idempotent, called before every load.
///
/// Refuses first on a processor that lacks an instruction set the CPU backend
/// was compiled to use. ggml does not check: its first vector instruction
/// would kill the whole node with SIGILL, with nothing but a native crash
/// report in the log. A refusal keeps the node up and says what is missing.
pub(super) fn init_backend() -> Result<()> {
    let missing = missing_instruction_sets(&sys::ggml_cpu_built_with(), cpu_has);
    if !missing.is_empty() {
        bail!(
            "this processor cannot run this build: the inference engine was compiled to use {} \
             and the processor does not have {}. Use a build whose CPU backend is chosen at run \
             time (the Docker image built in `release` mode), or, on a virtual machine, expose \
             the host processor to the guest",
            sys::ggml_cpu_built_with().join(", "),
            missing.join(", "),
        );
    }
    unsafe { sys::llama_backend_init() };
    crate::sysmon::mark_backend_initialized();
    Ok(())
}

pub(super) fn backend_registry_name(device: sys::ggml_backend_dev_t) -> Option<String> {
    if device.is_null() {
        return None;
    }

    let registry = unsafe { sys::ggml_backend_dev_backend_reg(device) };
    if registry.is_null() {
        return None;
    }

    let name = unsafe { sys::ggml_backend_reg_name(registry) };
    if name.is_null() {
        return None;
    }

    unsafe { CStr::from_ptr(name).to_str().ok().map(str::to_owned) }
}

pub fn effective_n_ubatch(cfg: &EngineConfig) -> u32 {
    effective_n_ubatch_with_limit(cfg, cfg.moe_external_executor)
}

/// Retorna o microlote físico do contexto MTP.
///
/// O MTP compartilha a configuração de sessão com o alvo MoE, mas o captain
/// fixou 512 para o buffer de `nextn`. Por isso esta política não herda a
/// exceção do contexto principal externo.
pub fn effective_mtp_n_ubatch(cfg: &EngineConfig) -> u32 {
    effective_n_ubatch_with_limit(cfg, false)
}

fn effective_n_ubatch_with_limit(cfg: &EngineConfig, external_moe: bool) -> u32 {
    let n_batch = cfg.n_batch.max(1);
    if cfg.fit_memory {
        let chosen = cfg.n_ubatch.max(1);
        let limit = if external_moe {
            MOE_EXTERNAL_MAX_UBATCH.min(chosen)
        } else {
            chosen
        };
        n_batch.min(limit)
    } else {
        n_batch
    }
}

/// Quantos threads de CPU o llama.cpp usa num contexto.
///
/// Sem valor configurado (`n_threads <= 0`), o POM usa todos os threads que o
/// processo tem disponíveis na máquina onde roda: `available_parallelism`
/// respeita a afinidade e a cota de CPU do container. O padrão do llama.cpp é
/// um número fixo e pequeno, que deixa a CPU quase ociosa justamente quando
/// parte das camadas não coube na VRAM e roda nela.
pub fn effective_n_threads(requested: i32) -> i32 {
    if requested > 0 {
        return requested;
    }
    std::thread::available_parallelism()
        .map(|threads| i32::try_from(threads.get()).unwrap_or(i32::MAX))
        .unwrap_or(1)
}

/// Aplica `effective_n_threads` à geração e ao lote do contexto.
pub fn apply_thread_params(cparams: &mut sys::llama_context_params, cfg: &EngineConfig) {
    let threads = effective_n_threads(cfg.n_threads);
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
}

/// Aplica a política de lote público e microlote do contexto principal.
pub fn apply_context_batch_params(cparams: &mut sys::llama_context_params, cfg: &EngineConfig) {
    cparams.n_batch = cfg.n_batch;
    cparams.n_ubatch = effective_n_ubatch(cfg);
}

/// Aplica os parâmetros do contexto MTP sem herdar a exceção de microbatch do
/// contexto principal MoE.
pub fn apply_mtp_context_batch_params(cparams: &mut sys::llama_context_params, cfg: &EngineConfig) {
    cparams.n_batch = cfg.n_batch;
    cparams.n_ubatch = effective_mtp_n_ubatch(cfg);
}

/// The memory fit, with the MoE expert cache only where it pays.
///
/// The cache size is one value for the whole deployment, but it only helps a
/// node whose experts do not all fit in VRAM: elsewhere it pushes expert layers
/// out of the GPU just to serve some of them back, and takes the VRAM other
/// contexts (the MTP draft) needed. So the fit runs first without the cache,
/// and the cache is kept only when that plan leaves routed experts in host
/// memory.
pub(super) fn fit_plan(
    model_path: &str,
    cfg: &EngineConfig,
    mtp_mode: MtpFitMode,
) -> Result<FitPlan> {
    if cfg.moe_cache_mib == 0 || !fit_can_adjust_layers(cfg) {
        return fit_plan_with_cache(model_path, cfg, mtp_mode);
    }
    let without_cache = EngineConfig {
        moe_cache_mib: 0,
        ..cfg.clone()
    };
    let plan = fit_plan_with_cache(model_path, &without_cache, mtp_mode)?;
    if plan.applied && !plan_keeps_experts_in_host(&plan) {
        tracing::info!(
            moe_cache_mib = cfg.moe_cache_mib,
            "cache de experts do MoE desligado neste nó: todos os experts cabem na VRAM"
        );
        return Ok(plan);
    }
    fit_plan_with_cache(model_path, cfg, mtp_mode)
}

/// Whether a fitted plan sends routed MoE experts to host memory: the fit
/// does it with `tensor_buft_overrides` on the `*_exps` tensors.
pub(super) fn plan_keeps_experts_in_host(plan: &FitPlan) -> bool {
    plan.tensor_buft_overrides.iter().any(|entry| {
        !entry.pattern.is_null()
            && unsafe { std::ffi::CStr::from_ptr(entry.pattern) }
                .to_str()
                .is_ok_and(|pattern| pattern.contains("exps"))
    })
}

fn fit_plan_with_cache(
    model_path: &str,
    cfg: &EngineConfig,
    mtp_mode: MtpFitMode,
) -> Result<FitPlan> {
    let (mtp_mode, mtp_unavailable_reason) = if mtp_mode == MtpFitMode::Disabled {
        (MtpFitMode::Disabled, None)
    } else {
        let path = std::path::Path::new(model_path);
        let has_nextn = crate::gguf::has_nextn_tensors(path).unwrap_or(false);
        // Um GGUF de rascunho MTP separado ("shared MTP module", ex.
        // qwen4exp/Qwen3.8-Flash-Next) supre os pesos `nextn` que o arquivo
        // principal não traz - ver `crate::mtp` e
        // `models::catalog::find_mtp_draft_sibling` no lado do backend. Sem
        // isto, `fit_plan` desligava o MTP para qualquer alvo assim mesmo
        // com um rascunho válido já detectado no disco, porque só olhava o
        // arquivo principal (defeito confirmado ao vivo na ksia/salem,
        // 2026-09-18).
        let has_external_draft = !has_nextn
            && cfg
                .mtp_draft_path
                .as_deref()
                .map(std::path::Path::new)
                .is_some_and(|draft_path| {
                    crate::gguf::has_nextn_tensors(draft_path).unwrap_or(false)
                });

        if has_nextn {
            // Embutido: o grafo do próprio alvo precisa saber construir a
            // cabeça `nextn` (`LLM_GRAPH_TYPE_DECODER_MTP`), então a
            // arquitetura dele tem que estar na lista gerada.
            let architecture = crate::gguf::read_info(path)
                .ok()
                .and_then(|info| info.architecture);
            let (mode, reason) = mtp_fit_mode_for_model(mtp_mode, true, architecture.as_deref());
            if let Some(reason) = reason.as_deref() {
                warn!(model = %model_path, reason, "MTP indisponível para a arquitetura do modelo");
            }
            (mode, reason)
        } else if has_external_draft {
            // Rascunho externo: quem constrói a cabeça `nextn` é o arquivo
            // de rascunho, com a própria arquitetura dele - checado depois,
            // em `MtpDrafter::attach`, contra o rascunho, não contra o alvo.
            // O alvo só precisa exportar `h_nextn` do próprio grafo, o que
            // não é um fato estático por arquitetura que `fit_plan` possa
            // conferir aqui; um alvo que não exportar falha de forma
            // graciosa no primeiro `observe` (ver `crate::mtp`), nunca
            // aborta.
            (mtp_mode, None)
        } else {
            tracing::info!(
                model = %model_path,
                "MTP indisponível: o arquivo não traz tensores nextn nem há um rascunho externo válido"
            );
            (MtpFitMode::Disabled, None)
        }
    };

    if !fit_can_adjust_layers(cfg) {
        if cfg.fit_memory && cfg.n_gpu_layers != DEFAULT_GPU_LAYERS {
            warn!(
                "llama.cpp fit ignorado porque n_gpu_layers foi definido explicitamente como {}",
                cfg.n_gpu_layers
            );
        }
        return Ok(empty_fit_plan_with_mtp_mode_and_reason(
            cfg,
            mtp_mode_without_gpu_fit(mtp_mode, cfg.mtp_placement),
            mtp_unavailable_reason,
        ));
    }

    let path = CString::new(model_path)
        .with_context(|| format!("invalid model path for llama.cpp fit: {model_path}"))?;
    // `common_fit_params` queries the backend devices and their free memory.
    // The call is idempotent and mirrors `LoadedModel::load`, which also keeps
    // the initialization close to every possible model-loading entry point.
    init_backend()?;
    let n_devices = unsafe { sys::llama_max_devices() }.max(1);
    let n_overrides = unsafe { sys::llama_max_tensor_buft_overrides() }.max(1);

    // Take the memory snapshot before the speed probes allocate their small
    // benchmark graph. This preserves the fit snapshot timing guarantee and
    // makes the measured order orthogonal to the memory budget.
    let mut available_devices = vec![ptr::null_mut(); n_devices];
    let available_count = unsafe {
        sys::llama_rs_get_devices(available_devices.as_mut_ptr(), available_devices.len())
    };
    available_devices.truncate(available_count.min(n_devices));
    let mut device_memories = vec![
        sys::llama_rs_device_memory {
            free_bytes: 0,
            total_bytes: 0,
        };
        n_devices
    ];
    let memory_count = if available_devices.is_empty() {
        // Keep the old wrapper path for an empty device registry so a missing
        // snapshot continues to take the existing safe fit fallback.
        let count = unsafe {
            sys::llama_rs_get_device_memory(device_memories.as_mut_ptr(), device_memories.len())
        };
        device_memories.truncate(count.min(n_devices));
        count
    } else {
        for (memory, &device) in device_memories.iter_mut().zip(&available_devices) {
            let mut free_bytes = 0;
            let mut total_bytes = 0;
            unsafe { sys::ggml_backend_dev_memory(device, &mut free_bytes, &mut total_bytes) };
            memory.free_bytes = free_bytes;
            memory.total_bytes = total_bytes;
        }
        device_memories.truncate(available_devices.len());
        available_devices.len()
    };

    let memory_snapshot_complete = device_memories
        .iter()
        .all(|memory| memory.total_bytes > 0 && memory.free_bytes <= memory.total_bytes);
    let speed_scores = if available_devices.len() > 1 && memory_snapshot_complete {
        available_devices
            .iter()
            .map(|&device| {
                let score = unsafe { sys::llama_rs_benchmark_device(device) };
                if score.is_finite() && score > 0.0 {
                    Some(f64::from(score))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let backend_names = available_devices
        .iter()
        .map(|&device| backend_registry_name(device))
        .collect::<Vec<_>>();
    let backend_name_refs = backend_names
        .iter()
        .map(Option::as_deref)
        .collect::<Vec<_>>();
    let total_memories = device_memories
        .iter()
        .map(|memory| memory.total_bytes as u64)
        .collect::<Vec<_>>();
    let ordered_devices = device_placement::largest_memory_first_order_for_backends(
        &total_memories,
        &speed_scores,
        &backend_name_refs,
    )
    // A partial memory snapshot names fewer devices than there are: no order.
    .filter(|order| order.len() == available_devices.len())
    .filter(|order| {
        order
            .iter()
            .enumerate()
            .any(|(position, &original)| position != original)
    })
    .map(|order| {
        let mut devices = order
            .iter()
            .map(|&index| available_devices[index])
            .collect::<Vec<_>>();
        let reordered_memories = order
            .iter()
            .map(|&index| device_memories[index])
            .collect::<Vec<_>>();
        device_memories = reordered_memories;
        devices.push(ptr::null_mut());
        devices
    })
    .unwrap_or_default();
    if available_devices.len() > 1 {
        tracing::info!(
            device_backend_names = ?backend_names,
            device_speed_scores = ?speed_scores,
            device_total_bytes = ?total_memories,
            memory_order_applied = !ordered_devices.is_empty(),
            "llama.cpp fit ordenou os dispositivos pela memória (velocidade só desempata)"
        );
    }
    let fit_memories = device_memories
        .iter()
        .map(|memory| FitDeviceMemory {
            total_bytes: memory.total_bytes,
            free_bytes: memory.free_bytes,
        })
        .collect::<Vec<_>>();
    // Os vetores entregues ao C ficam com a capacidade de compilação, que o
    // wrapper preenche até o número de placas que ele mesmo enxerga; as
    // decisões físicas usam a contagem real. Ver `fit_device_count`.
    let physical_devices = fit_device_count(available_devices.len(), memory_count, n_devices);
    let mut margins = fit_memory_margins(&fit_memories, n_devices, cfg.fit_memory_margin_mb);
    if memory_count != physical_devices {
        warn!(
            reported_devices = memory_count,
            expected_devices = physical_devices,
            "llama.cpp fit recebeu um snapshot parcial de memória"
        );
    }
    let snapshots = fit_memory_snapshots(&fit_memories, &margins);
    let process_gpu_memory_mb = crate::sysmon::current_process_gpu_memory_mb();
    tracing::info!(
        process_id = std::process::id(),
        process_gpu_memory_mb = process_gpu_memory_mb.unwrap_or(0.0),
        process_gpu_memory_known = process_gpu_memory_mb.is_some(),
        "llama.cpp fit memória do processo no ponto da sondagem"
    );
    for (device, snapshot) in snapshots.iter().enumerate() {
        tracing::info!(
            device,
            total_bytes = snapshot.total_bytes,
            free_bytes = snapshot.free_bytes,
            margin_bytes = snapshot.margin_bytes,
            budget_bytes = snapshot.free_bytes.saturating_sub(snapshot.margin_bytes),
            "llama.cpp fit memória e orçamento por dispositivo"
        );
    }
    let make_model_params = |load_mtp: bool| {
        let mut params = unsafe { sys::llama_model_default_params() };
        params.n_gpu_layers = -1;
        if !ordered_devices.is_empty() {
            // `ordered_devices` includes the NULL sentinel required by
            // llama.cpp. Keep it in FitPlan so the same order reaches the real
            // model load.
            params.devices = ordered_devices.as_ptr() as *mut sys::ggml_backend_dev_t;
        }
        params.load_mtp = load_mtp;
        params.moe_external_executor = cfg.moe_external_executor;
        if !cfg.moe_external_executor_layers.is_empty() {
            params.moe_external_executor_layers = cfg.moe_external_executor_layers.as_ptr();
            params.moe_external_executor_layer_count = cfg.moe_external_executor_layers.len();
        }
        params
    };
    let make_context_params = |mtp: bool| {
        let mut params = unsafe { sys::llama_context_default_params() };
        params.n_ctx = cfg.n_ctx;
        if mtp {
            apply_mtp_context_batch_params(&mut params, cfg);
        } else {
            apply_context_batch_params(&mut params, cfg);
        }
        params.n_seq_max = kv_seq_budget(cfg.max_sessions);
        params.kv_unified = true;
        params.offload_kqv = cfg.offload_kqv;
        apply_cache_types(&mut params, cfg.cache_type_k, cfg.cache_type_v);
        params.embeddings = cfg.need_embeddings;
        params.n_rs_seq = cfg.rs_seq_window;
        // The fit has to count the expert cache, or it fills the VRAM the
        // cache is allocated from at context creation. `mtp` here is the main
        // context sized for MTP's batch (the joint fit), not the draft's own
        // context, so it carries the cache too.
        params.moe_cache_size = moe_cache_bytes(cfg);
        params
    };

    // First fit only the work that every token must execute. The MTP pass below
    // is allowed to reserve its own context only after this plan fits intact.
    let mut primary_tensor_split = vec![0.0_f32; n_devices];
    let mut primary_tensor_buft_overrides = vec![
        sys::llama_model_tensor_buft_override {
            pattern: ptr::null(),
            buft: ptr::null_mut(),
        };
        n_overrides
    ];
    let mut primary_mparams = make_model_params(false);
    let mut primary_cparams = make_context_params(false);
    let primary_status = unsafe {
        sys::llama_rs_fit_params(
            path.as_ptr(),
            &mut primary_mparams,
            &mut primary_cparams,
            primary_tensor_split.as_mut_ptr(),
            primary_tensor_buft_overrides.as_mut_ptr(),
            margins.as_mut_ptr(),
            FIT_MIN_CONTEXT,
            sys::GGML_LOG_LEVEL_WARN,
        )
    };

    if primary_status != 0 {
        warn!(
            "llama.cpp fit não encontrou uma configuração automática (status {primary_status}); usando os parâmetros solicitados sem MTP na GPU"
        );
        return Ok(empty_fit_plan_with_mtp_mode_and_reason(
            cfg,
            mtp_mode_without_gpu_fit(mtp_mode, cfg.mtp_placement),
            mtp_unavailable_reason,
        ));
    }

    let primary_gpu_layers = primary_mparams.n_gpu_layers;
    // The deployment placed MTP in VRAM: it gets its share first and the main
    // model fits in the rest.
    let mtp_first = cfg.mtp_placement == crate::MtpPlacement::Vram;
    let mut selected_mtp_mode = choose_mtp_fit_mode(mtp_mode, primary_gpu_layers, None, mtp_first);
    let mut selected_n_gpu_layers = primary_mparams.n_gpu_layers;
    let mut selected_n_ctx = primary_cparams.n_ctx;
    let mut selected_tensor_split = primary_tensor_split;
    let mut selected_tensor_buft_overrides = primary_tensor_buft_overrides;

    if mtp_mode == MtpFitMode::Vram && (primary_gpu_layers == -1 || mtp_first) {
        // First discover the native layer split, then charge each NextN layer's
        // KV to the device llama.cpp assigned to it. Refit until the split and
        // the reservation agree; otherwise use the CPU companion rather than
        // accepting a fit with an unreserved device.
        const MAX_MTP_FIT_ATTEMPTS: usize = 4;
        let mtp_info = crate::gguf::read_info(std::path::Path::new(model_path)).ok();
        let mut reserved_mtp_cost = vec![0usize; physical_devices];
        let mut exhausted_attempts = true;
        let external_draft_bytes =
            external_mtp_draft_bytes(cfg.mtp_draft_path.as_deref(), model_path);

        for attempt in 0..MAX_MTP_FIT_ATTEMPTS {
            let mut joint_margins =
                fit_memory_margins_with_mtp(&margins, &reserved_mtp_cost, MtpFitMode::Vram);
            let mut mtp_mparams = make_model_params(true);
            let mut mtp_cparams = make_context_params(true);
            let mut mtp_tensor_split = vec![0.0_f32; n_devices];
            let mut mtp_tensor_buft_overrides = vec![
                sys::llama_model_tensor_buft_override {
                    pattern: ptr::null(),
                    buft: ptr::null_mut(),
                };
                n_overrides
            ];
            let mtp_status = unsafe {
                sys::llama_rs_fit_params(
                    path.as_ptr(),
                    &mut mtp_mparams,
                    &mut mtp_cparams,
                    mtp_tensor_split.as_mut_ptr(),
                    mtp_tensor_buft_overrides.as_mut_ptr(),
                    joint_margins.as_mut_ptr(),
                    FIT_MIN_CONTEXT,
                    sys::GGML_LOG_LEVEL_WARN,
                )
            };
            let joint_gpu_layers = (mtp_status == 0).then_some(mtp_mparams.n_gpu_layers);
            let joint_mode =
                choose_mtp_fit_mode(mtp_mode, primary_gpu_layers, joint_gpu_layers, mtp_first);
            if joint_mode != MtpFitMode::Vram {
                selected_mtp_mode = joint_mode;
                tracing::info!(
                    primary_gpu_layers,
                    joint_gpu_layers = ?joint_gpu_layers,
                    "llama.cpp fit recusou a reserva MTP para preservar as camadas principais"
                );
                exhausted_attempts = false;
                break;
            }

            let Some(mtp_info) = mtp_info.as_ref() else {
                warn!("não foi possível ler a geometria GGUF do KV MTP; usando o recuo para RAM");
                selected_mtp_mode = MtpFitMode::Ram;
                exhausted_attempts = false;
                break;
            };
            let Some(estimate) = estimate_mtp_vram_costs(
                mtp_info,
                cfg,
                mtp_cparams.n_ctx,
                &mtp_tensor_split,
                &fit_memories,
                physical_devices,
            ) else {
                warn!("não foi possível estimar o KV MTP pela divisão nativa; usando o recuo para RAM");
                selected_mtp_mode = MtpFitMode::Ram;
                exhausted_attempts = false;
                break;
            };
            // An external draft is a second model on the device of the last
            // NextN layer. Reserving only its KV let the fit fill that card and
            // the draft then failed to allocate its weights and compute.
            let actual_mtp_cost = with_external_mtp_draft(estimate.costs, external_draft_bytes);
            let effective_tensor_split = estimate.tensor_split;
            tracing::info!(
                attempt,
                mtp_vram_cost_estimate = ?actual_mtp_cost,
                mtp_tensor_split = ?effective_tensor_split.get(..physical_devices),
                external_draft_bytes = external_draft_bytes.unwrap_or(0),
                "llama.cpp fit calculou a reserva MTP por dispositivo pela divisão efetiva de camadas"
            );

            if actual_mtp_cost == reserved_mtp_cost {
                selected_mtp_mode = MtpFitMode::Vram;
                selected_n_gpu_layers = mtp_mparams.n_gpu_layers;
                selected_n_ctx = mtp_cparams.n_ctx;
                selected_tensor_split = effective_tensor_split;
                selected_tensor_buft_overrides = mtp_tensor_buft_overrides;
                exhausted_attempts = false;
                break;
            }

            tracing::debug!(
                attempt,
                reserved_mtp_cost = ?reserved_mtp_cost,
                actual_mtp_cost = ?actual_mtp_cost,
                "llama.cpp fit repetirá para alinhar o orçamento MTP à divisão de camadas"
            );
            reserved_mtp_cost = actual_mtp_cost;
        }

        if exhausted_attempts {
            tracing::info!(
                "MTP usará o acompanhante em RAM porque a reserva por dispositivo não convergiu"
            );
        }
    } else if mtp_mode == MtpFitMode::Vram {
        tracing::info!(
            primary_gpu_layers,
            "llama.cpp fit manteve o MTP fora da VRAM porque o encaixe principal já precisou reduzir camadas"
        );
    }

    let mut effective = cfg.clone();
    effective.n_gpu_layers = selected_n_gpu_layers;
    effective.n_ctx = selected_n_ctx;

    tracing::info!(
        requested_gpu_layers = cfg.n_gpu_layers,
        effective_gpu_layers = effective.n_gpu_layers,
        requested_context = cfg.n_ctx,
        effective_context = effective.n_ctx,
        mtp_fit_mode = ?selected_mtp_mode,
        "llama.cpp fit ajustou os parâmetros de memória"
    );

    Ok(FitPlan {
        config: effective,
        tensor_split: selected_tensor_split,
        tensor_buft_overrides: selected_tensor_buft_overrides,
        devices: ordered_devices,
        mtp_mode: selected_mtp_mode,
        mtp_unavailable_reason,
        applied: true,
    })
}
