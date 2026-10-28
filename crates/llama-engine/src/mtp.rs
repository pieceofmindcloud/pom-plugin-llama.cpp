//! Predição multi-token (MTP) pelo bloco `nextn` do próprio modelo.
//!
//! # O que isto é, e o que **não** é
//!
//! O nó já tinha uma "decodificação especulativa": `infer_server::find_draft`,
//! que procura no texto já gerado uma repetição do próprio final e chuta que a
//! continuação se repete. É *prompt-lookup decoding* - uma busca em array que
//! nunca toca os pesos - e por isso só acelera saída repetitiva. Num prompt
//! comum ela não acha nada, o rascunho sai vazio e a geração custa exatamente o
//! que custava.
//!
//! MTP é outro mecanismo. Modelos como o Qwen3.5/3.8 (`qwen35`, `qwen35moe`)
//! trazem **dentro do GGUF** um bloco extra treinado, o `nextn`, que recebe o
//! estado oculto do tronco na posição `p` junto com o embedding do token da
//! posição `p+1` e prevê o token de `p+2`. É uma cabeça de rascunho de
//! verdade, treinada com o modelo, e é ela que o LM Studio liga quando oferece
//! "MTP". Os dois são independentes e ligam por chaves separadas
//! (`speculative_decoding_enabled` e `mtp_enabled`): reaproveitar um
//! interruptor faria "ligar especulação" significar duas coisas diferentes.
//!
//! # O que muda na resposta: nada
//!
//! Rascunho é aposta sobre **o que entra no lote**, nunca sobre o que sai. A
//! verificação em `infer_server::step_blocking` amostra cada posição de novo,
//! pelo mesmo `sample_token_profiled` de sempre, e só aceita o prefixo que
//! bateu. Um rascunho ruim custa aceitação (velocidade); nunca correção. É por
//! isso que dá para rodar com uma única cabeça `nextn` mesmo num modelo que
//! traga várias treinadas: o pior caso é um lote desperdiçado.
//!
//! # Onde o bloco `nextn` mora, e por que isso limita o cluster
//!
//! O bloco vive na **última** camada do modelo (`blk.<n-1>.nextn.*`), então
//! num modelo fatiado ele está na fatia final - a cauda do anel. Só a cauda
//! consegue rascunhar, e o rascunho tem que chegar à cabeça, que é quem monta
//! o lote. Hoje, portanto, o MTP só entra em ação quando **uma máquina serve o
//! modelo inteiro** (cabeça e cauda são o mesmo processo, e o rascunho não
//! atravessa rede nenhuma). Num pipeline de N nós o motor da cabeça não tem o
//! bloco, `MtpDrafter::attach` devolve `None` e o nó simplesmente não anuncia
//! a capacidade. Ver `docs/subsystems/inference.md` para a conta do que
//! custaria estender ao caso fatiado.
//!
//! # GGUF de rascunho separado ("shared MTP module")
//!
//! Nem todo modelo com MTP embute o bloco `nextn` no próprio arquivo. A
//! Unsloth distribui o rascunho do Qwen3.8-Flash-Next (arquitetura
//! `qwen4exp`) num GGUF à parte (`mtp-<modelo>-shared-Q8_0.gguf`) que
//! reaproveita `token_embd`/`output` do modelo principal em vez de duplicá-los
//! - o mesmo motivo de espaço que o llama.cpp já resolve para EAGLE3/DFLASH
//! via `ctx_other` (`llama-ext.h`), embora o formato aqui seja mais simples: o
//! arquivo de rascunho de fato inclui sua própria cópia pequena de
//! `token_embd`/`output` (o "mtp_only" que `qwen35.cpp`/`qwen3next.cpp`/
//! `deepseek4.cpp`/`qwen4exp.cpp` já sabem carregar - `blk.0.<primeiro tensor
//! de tronco da arquitetura>` ausente marca o arquivo como só-nextn), então
//! **não precisa** de `ctx_other`: carrega como um
//! `LoadedModel` inteiramente independente
//! (`LoadedModel::load_mtp_draft`) e vira o `companion` de
//! `mtp_model_for_attach`, exatamente como o irmão CPU-only do recuo para RAM
//! - os dois nunca coexistem.
//!
//! `EngineConfig::mtp_draft_path` é onde esse caminho chega. A descoberta
//! fica junto ao GGUF e segue a convenção própria `mtp-` (ver
//! `models::catalog::find_mtp_draft_sibling`). O `mmproj` usa identidade GGUF
//! e compatibilidade dimensional, e é
//! automática - **não há um interruptor novo na UI**: o operador já liga MTP
//! pelo campo que existe hoje, e o rascunho externo entra quando o alvo não
//! tem `nextn` embutido e o arquivo existe ao lado do GGUF. A descoberta
//! acontece na própria máquina que carrega o modelo (`run_apply`,
//! `reconfigure_local`), então só a máquina que serve a fatia final - onde o
//! MTP roda de fato - precisa ter o arquivo no disco; nem o orquestrador do
//! deploy nem as demais máquinas do pipeline precisam dele.
//!
//! `LoadedModel::load_mtp_draft` sozinho sempre tenta VRAM
//! (`MtpFitMode::Vram`, `DEFAULT_GPU_LAYERS`) sem medir memória livre - ao
//! contrário do alvo, cujo `fit_plan` mede a VRAM restante e reduz camadas
//! antes de carregar. `LlmEngine::load_with_cache` fecha essa lacuna em duas
//! camadas, cada uma confirmada ao vivo na ksia num teste separado
//! (2026-09-18):
//!
//! 1. **Carga dos pesos.** Quando o `fit_plan` do alvo já decidiu
//!    `MtpFitMode::Ram` (VRAM saturada), pedir GPU para os pesos do
//!    rascunho só repete o mesmo `cudaMalloc` que falhou para o alvo -
//!    visto ao vivo como `mtp_fit_mode=Ram` no log do fit seguido de
//!    `cudaMalloc failed: out of memory` dentro do próprio
//!    `llama_model_load_from_file` do rascunho. `load_with_cache` agora
//!    pula direto para `LoadedModel::load_mtp_draft_cpu`
//!    (`n_gpu_layers=0`, mesmo padrão de `load_mtp_cpu`) quando o fit já
//!    disse Ram, com um recuo reativo para a mesma função se
//!    `load_mtp_draft` falhar mesmo partindo de `Vram`.
//! 2. **Criação do contexto.** Mesmo com os pesos na CPU (pela camada 1 ou
//!    organicamente, porque `llama.cpp` não achou espaço tensor por
//!    tensor), a primeira tentativa de anexar sempre criava o contexto com
//!    `cfg.offload_kqv` do alvo - com `op_offload=true` o agendador ainda
//!    reserva buffer de compute na GPU mesmo para pesos já em CPU, e essa
//!    reserva de pior caso (ubatch cheio) podia estourar a VRAM que sobrou
//!    (`cudaMalloc failed: out of memory` num teste anterior, mesma
//!    investigação). Corrigido: o contexto já nasce com
//!    `offload_kqv=false` quando os pesos entraram pela RAM, e
//!    `load_with_cache` ainda repete com `offload_kqv=false` se os pesos
//!    couberam na VRAM mas o contexto não coube.
//!
//! As duas camadas só disparam com `mtp_cpu_fallback_enabled` ligado; nunca
//! recarregam peso além do necessário - a camada 2 nunca recarrega peso
//! nenhum, só recria o contexto.
//!
//! `llm::config::fit_plan` também precisa saber disso, separado de
//! `load_with_cache`: é ele quem decide `MtpFitMode` (e portanto se MTP
//! sequer é tentado) antes da carga de verdade, e antes de 2026-09-18 só
//! olhava `has_nextn_tensors` do arquivo **principal** - reportando
//! "MTP indisponível" (defeito confirmado ao vivo na ksia/salem) mesmo com um
//! rascunho externo correto já no disco, porque nunca conferia
//! `EngineConfig::mtp_draft_path`.
//!
//! **Pré-requisito que o rascunho externo não supre sozinho**: o par
//! `(h_p, x_{p+1})` que a cabeça `nextn` consome exige que o **alvo** exporte
//! `h_nextn` do próprio grafo (`pom_llama_set_embeddings_nextn`), e isso é
//! código por arquitetura no fork (`build_arch_graph`/`res->t_h_nextn`) - o
//! rascunho não pode fornecer o que o alvo nunca calculou. `qwen35.cpp` já
//! fazia isso; `qwen4exp.cpp` (Qwen3.8-Flash-Next) ganhou o mesmo em
//! `pieceofmindcloud/llama.cpp` (`pom`), gated por
//! `cparams.embeddings_nextn && !cparams.embeddings_nextn_masked` para não
//! regredir o corte antecipado de linhas que a arquitetura já fazia na
//! última camada (`inp_out_ids`) - sem MTP pedido esse corte continua
//! idêntico ao de antes, sem custo extra de FFN/hyper-connection por posição.
//! `h_nextn` é sempre o estado **largo** pré-mixer (`n_embd*hc` para
//! arquiteturas de hyper-connection como `qwen4exp`/`deepseek4`, `n_embd`
//! nas demais) - exportar a versão já colapsada pelo mixer final é um erro
//! silencioso, não um crash: a cabeça de rascunho ainda decodifica, só que
//! sobre menos informação, derrubando a taxa de aceitação sem avisar.
//!
//! A lista gerada em `build/mtp_arch.rs` (`MTP_GRAPH_ARCHITECTURES`)
//! certifica quem consegue **construir o grafo `DECODER_MTP`** (a capacidade
//! do lado do rascunho, checada em `MtpDrafter::attach` contra a arquitetura
//! do rascunho) - não há checagem estática do lado Rust para quem exporta
//! `h_nextn` do lado do alvo. As duas coincidem sempre que o rascunho é
//! `qwen4exp` (embutido ou externo): a mesma `qwen4exp.cpp` exporta o estado
//! largo do alvo (`graph::graph`) e constrói a cabeça de rascunho
//! (`graph::graph_mtp`, despachada por `params.gtype ==
//! LLM_GRAPH_TYPE_DECODER_MTP`) - só o `ctx_type` do contexto muda qual das
//! duas roda. Um alvo cujo grafo não exporta `h_nextn` simplesmente falha no
//! primeiro `observe` (`llama_get_embeddings_nextn_ith` devolve `NULL`) e
//! desliga o MTP daquela sessão com aviso - a mesma degradação que já existe
//! para um peer velho que não ecoa a composição do lote no MTP fatiado (ver
//! `docs/subsystems/inference.md`, 5.2). Ver a sonda
//! `an_external_mtp_draft_attaches_to_a_target_without_embedded_nextn` nos
//! testes deste módulo para o roteiro de validação com os arquivos reais, e
//! `crates/llama-engine/src/llm/model.rs::mtp_draft_tests` para a
//! reprodução sintética que provou a causa raiz e a correção.
//!
//! **Fatiamento multi-nó do rascunho externo**: fora de escopo por ora. O
//! mesmo limite da seção acima vale aqui - MTP só roda hoje quando uma máquina
//! serve o modelo inteiro -, e como o rascunho depende do estado oculto final
//! do alvo, ele teria que morar na mesma máquina que serve a última fatia,
//! nunca na cabeça nem numa terceira máquina. Não implementado.
//!
//! # De onde veio o algoritmo
//!
//! De `llama.cpp/common/speculative.cpp`, `common_speculative_impl_draft_mtp`
//! (o driver que o `llama-server` usa). Aqui está portado o caminho de uma
//! sequência, uma cabeça e memória não compartilhada - qwen35/qwen35moe. Os
//! outros dois modos do upstream (`chain_heads` do step35, `is_mem_shared` do
//! gemma4) não estão: um modelo com várias cabeças roda com a primeira, o que
//! é um rascunho mais fraco e nunca uma resposta errada.

use anyhow::{bail, Result};
use std::ptr;

use llama_cpp_sys_2 as sys;

// ── As entradas NextN que o bindgen não emite ────────────────────────────────
//
// Ver `llama_mtp_shim.cpp`: elas moram em `llama-ext.h`, fora do `extern "C"`
// de `llama.h`, e o `wrapper.h` da crate `-sys` não inclui esse cabeçalho.
extern "C" {
    fn pom_llama_set_embeddings_nextn(ctx: *mut sys::llama_context, value: bool, masked: bool);
    fn pom_llama_get_embeddings_nextn_ith(
        ctx: *mut sys::llama_context,
        i: std::os::raw::c_int,
    ) -> *mut f32;
}

/// A cabeça de rascunho `nextn`, com o contexto MTP e o estado que atravessa
/// as rodadas.
///
/// O contexto de rascunho é criado sobre os pesos do alvo no caminho VRAM
/// (`llama_init_from_model` com `ctx_type = LLAMA_CONTEXT_TYPE_MTP`). No
/// recuo para RAM, ele recebe um modelo CPU-only irmão para impedir que o
/// scheduler devolva o grafo ao dispositivo do alvo. Nesse caso há dois
/// objetos de modelo, mas o GGUF pode compartilhar as páginas mmap do arquivo;
/// o custo adicional relevante continua sendo o cache KV de **uma** camada.
pub struct MtpDrafter {
    /// O contexto que roda o grafo `DECODER_MTP`. Dono: liberado no `Drop`.
    ctx: *mut sys::llama_context,
    /// Modelo que sustenta o contexto MTP. No recuo para RAM é um modelo
    /// CPU-only separado, para que o output head compartilhado não arraste o
    /// grafo de volta para a GPU do modelo alvo.
    _mtp_model: Option<std::sync::Arc<crate::llm::LoadedModel>>,
    hidden_width: usize,
    n_vocab: usize,

    /// O `h` que a próxima chamada precisa parear com o primeiro token dela.
    ///
    /// MTP na posição `p+1` consome o par `(h_p, x_{p+1})`, e `h_p` nasce no
    /// decode **anterior**. Sem guardar essa linha, todo lote começaria com um
    /// `h` errado (ou zerado) na primeira posição.
    pending_h: Vec<f32>,

    /// As linhas de `h` do último lote do alvo, uma por posição decodificada.
    /// É daqui que `accept` escolhe o novo `pending_h` depois de saber quantos
    /// tokens do rascunho passaram na verificação.
    verify_h: Vec<f32>,
    verify_rows: usize,
    allocation: crate::llm::MtpAllocation,
    /// O modelo que sustenta este contexto foi carregado com os experts em
    /// modo metadata-only. Nesse caso o contexto MTP também precisa do hook
    /// externo; sem ele `llama_decode` recusa o grafo antes de executar a
    /// cabeça nextn.
    external_moe_executor: bool,
}

// SAFETY: o `llama_context` é usado sob o mesmo `Mutex` que serializa o
// contexto alvo em `infer_server::SharedState` - nunca por duas threads ao
// mesmo tempo. É a mesma promessa que `LlmEngine` já faz.
unsafe impl Send for MtpDrafter {}

impl Drop for MtpDrafter {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                // The callback userdata belongs to the parent engine. Clear
                // the native hook before freeing this context so a destructor
                // or an in-flight backend teardown cannot call a dead Arc.
                sys::llama_set_moe_ffn_callback(self.ctx, None, ptr::null_mut());
                sys::llama_free(self.ctx);
                self.ctx = ptr::null_mut();
            }
        }
    }
}

/// Quantas cabeças `nextn` este modelo traz. Zero significa "sem MTP" - é o
/// caso de todo modelo sem o bloco e de **toda fatia não-final** de um modelo
/// que o tenha (ver `splitter::split`, que zera `nextn_predict_layers` fora da
/// fatia final justamente porque o bloco não viaja com ela).
pub fn nextn_layers(model: *mut sys::llama_model) -> i32 {
    unsafe { sys::llama_model_n_layer_nextn(model as *const _) }
}

// O build lê os arquivos do fork que está sendo compilado e gera esta lista em
// `OUT_DIR`. O arquivo de origem, o dispatch de `llama-model.cpp` e a tabela de
// nomes de `llama-arch.cpp` são a fonte da verdade - não há uma tabela manual
// de arquiteturas no POM.
include!(concat!(env!("OUT_DIR"), "/mtp_graph_architectures.rs"));

const MTP_GRAPH_ARCHITECTURES: &[&str] = GENERATED_MTP_GRAPH_ARCHITECTURES;

/// Retorna a razão operacional para não criar um grafo MTP, ou `None` quando
/// o fork conhece o grafo da arquitetura. Falhar fechado é importante: um
/// `GGML_ASSERT` no construtor do grafo aborta o processo e não pode ser
/// convertido em um `Result` depois do fato.
pub fn mtp_graph_unavailable_reason(architecture: Option<&str>) -> Option<String> {
    let architecture = architecture
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if architecture.is_some_and(|value| {
        MTP_GRAPH_ARCHITECTURES
            .iter()
            .any(|supported| supported.eq_ignore_ascii_case(value))
    }) {
        return None;
    }

    Some(match architecture {
        Some(value) => {
            format!("a arquitetura {value} não possui grafo MTP suportado pelo llama.cpp")
        }
        None => "a arquitetura do modelo não foi informada; grafo MTP não habilitado".to_string(),
    })
}

#[cfg(test)]
fn context_params(cfg: &crate::llm::EngineConfig) -> sys::llama_context_params {
    context_params_with_offload(cfg, cfg.offload_kqv)
}

pub fn context_params_with_offload(
    cfg: &crate::llm::EngineConfig,
    offload_kqv: bool,
) -> sys::llama_context_params {
    let mut cparams = unsafe { sys::llama_context_default_params() };
    cparams.n_ctx = cfg.n_ctx;
    crate::llm::apply_mtp_context_batch_params(&mut cparams, cfg);
    cparams.n_seq_max = crate::llm::kv_seq_budget(cfg.max_sessions);
    cparams.kv_unified = true;
    cparams.ctx_type = sys::LLAMA_CONTEXT_TYPE_MTP;
    cparams.offload_kqv = offload_kqv;
    // `op_offload` is independent of the KV flag. With it left at the
    // llama.cpp default, a CPU-resident MTP weight can still move a large
    // operation back to the GPU. The RAM fallback needs host execution too.
    cparams.op_offload = offload_kqv;
    crate::llm::apply_cache_types(&mut cparams, cfg.cache_type_k, cfg.cache_type_v);
    // Sem instantâneos de estado recorrente: o bloco `nextn` é uma camada
    // de atenção comum, e o upstream zera este campo pelo mesmo motivo.
    cparams.n_rs_seq = 0;
    crate::llm::apply_thread_params(&mut cparams, cfg);
    cparams
}

impl MtpDrafter {
    /// Monta o contexto de rascunho sobre um modelo já carregado.
    ///
    /// `Ok(None)` - e não erro - quando o modelo não tem bloco `nextn`: é a
    /// resposta certa para a fatia do meio de um pipeline e para qualquer
    /// modelo sem MTP, e quem chama trata como "este nó não anuncia MTP".
    ///
    /// `model_mtp` pode ser um modelo CPU-only separado no recuo para RAM.
    /// `ctx_tgt` é o contexto alvo, que passa a emitir o estado oculto do
    /// tronco (`llama_set_embeddings_nextn`, sem máscara: para **toda**
    /// posição do lote, não só as com logits). Sem isso não há `h` para
    /// alimentar a cabeça.
    pub fn attach(
        model_mtp: &crate::llm::LoadedModel,
        ctx_tgt: *mut sys::llama_context,
        cfg: &crate::llm::EngineConfig,
        hidden_width: usize,
        n_vocab: usize,
        mtp_model: Option<std::sync::Arc<crate::llm::LoadedModel>>,
        offload_kqv: bool,
        external_moe_executor: bool,
    ) -> Result<Option<Self>> {
        // `llama_model_n_layer_nextn` is metadata only: it stays positive on
        // a target loaded with `load_mtp=false`, even though its nextn tensor
        // pointers are null. Check the load capability before llama.cpp can
        // reserve the MTP graph, because that graph aborts on the null tensor.
        if !model_mtp.has_nextn_weights() || nextn_layers(model_mtp.model_ptr()) <= 0 {
            return Ok(None);
        }
        if mtp_graph_unavailable_reason(model_mtp.model_architecture().as_deref()).is_some() {
            return Ok(None);
        }

        // A janela e o lote acompanham o contexto alvo: o rascunho anda
        // posição a posição com ele, e um cache menor faria o `seq_rm` da
        // rodada seguinte apagar o que ainda vale. O microlote de computação
        // segue a política compartilhada do contexto principal, para que a
        // reserva do grafo não cresça desnecessariamente com n_batch.
        let mut cparams = context_params_with_offload(cfg, offload_kqv);
        // The MTP graph exposes logits and `h_nextn`, not generic embeddings.
        // Keeping this explicit protects the context if the shared config
        // later starts requesting residual embeddings for a MoE target.
        cparams.embeddings = false;
        // Mesmo orçamento de sequências do contexto alvo: `mtp_draft` recebe o
        // `seq_id` de lá, que a partir do desenho de duas sequências por sessão
        // pode ser o da geração (`prompt_seq + max_sessions`). Sem isto o
        // rascunho decodificaria numa sequência fora da faixa deste contexto.
        // Ver `llm::kv_seq_budget`.

        let ctx = unsafe { sys::llama_init_from_model(model_mtp.model_ptr(), cparams) };
        if ctx.is_null() {
            bail!("llama_init_from_model falhou para o contexto MTP (ctx_type=MTP)");
        }

        unsafe {
            // Alvo: emite `h` para toda posição do lote (`masked = false`),
            // porque `observe` lê uma linha por posição decodificada.
            pom_llama_set_embeddings_nextn(ctx_tgt, true, false);
            // Rascunho: só nas posições com logits (`masked = true`) - são as
            // únicas cujo `h` alimenta o passo seguinte do rascunho.
            pom_llama_set_embeddings_nextn(ctx, true, true);
        }

        let Some(allocation) = crate::llm::context_allocation(ctx) else {
            unsafe { sys::llama_free(ctx) };
            bail!("não foi possível medir a alocação do contexto MTP");
        };

        tracing::info!(
            "MTP pronto: {} cabeça(s) nextn, contexto de rascunho com n_ctx={}",
            nextn_layers(model_mtp.model_ptr()),
            cfg.n_ctx
        );
        tracing::info!(
            mtp_allocation = allocation.as_str(),
            "alocação do contexto MTP"
        );

        Ok(Some(MtpDrafter {
            ctx,
            _mtp_model: mtp_model,
            hidden_width,
            n_vocab,
            pending_h: vec![0.0; hidden_width],
            verify_h: Vec::new(),
            verify_rows: 0,
            allocation,
            external_moe_executor,
        }))
    }

    /// Instala no contexto MTP o mesmo callback que substitui os experts no
    /// contexto alvo. O modelo externo não contém buffers dos experts por
    /// desenho; deixar este contexto sem callback faz cada `llama_decode`
    /// falhar com `rc=-1` e desliga MTP para a sessão.
    pub fn set_moe_callback(
        &self,
        callback: sys::llama_moe_ffn_callback,
        userdata: *mut std::os::raw::c_void,
    ) {
        if !self.external_moe_executor {
            return;
        }
        unsafe { sys::llama_set_moe_ffn_callback(self.ctx, callback, userdata) };
    }

    pub fn allocation(&self) -> crate::llm::MtpAllocation {
        self.allocation
    }

    /// Acompanha um lote que o contexto **alvo** acabou de decodificar.
    ///
    /// Faz duas coisas, nessa ordem:
    ///
    /// 1. Repete o lote no contexto de rascunho, deslocando o estado oculto uma
    ///    posição para a direita - a linha `k` recebe `h` da posição `k-1`, e a
    ///    linha 0 recebe o `pending_h` guardado da chamada anterior. É esse
    ///    pareamento `(h_p, x_{p+1})` que o bloco `nextn` foi treinado a
    ///    consumir. Sem logits em posição nenhuma: aqui só se quer o cache KV
    ///    da camada `nextn` em dia, e pedir logits pagaria a cabeça de saída
    ///    (150k de vocabulário) por token de prompt.
    /// 2. Guarda as linhas de `h` do alvo, que `accept` usará para escolher o
    ///    `pending_h` da próxima rodada.
    ///
    /// O `seq_rm` antes do decode não é zelo: a rodada anterior deixou no cache
    /// do rascunho as posições que ela **apostou**, e a verificação pode ter
    /// recusado parte delas. Reescrever por cima sem apagar deixaria duas
    /// células para a mesma posição, e a atenção leria as duas.
    pub fn observe(
        &mut self,
        ctx_tgt: *mut sys::llama_context,
        tokens: &[i32],
        pos_offset: u32,
        seq_id: u32,
    ) -> Result<()> {
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }

        // 1. Alcança o alvo no contexto de rascunho.
        unsafe {
            let mem = sys::llama_get_memory(self.ctx);
            if !mem.is_null() {
                sys::llama_memory_seq_rm(mem, seq_id as i32, pos_offset as i32, -1);
            }
        }

        // `llama_batch_init` aloca **ou** `token` **ou** `embd`, nunca os dois;
        // pedindo `embd` ele devolve `token` nulo. O MTP precisa dos dois (o
        // token entra pelo embedding do próprio bloco, o `h` vem do tronco),
        // então o vetor de tokens é nosso e o ponteiro é devolvido ao que era
        // antes de `llama_batch_free` - que libera `token` se ele não for
        // nulo, e liberar memória do Rust ali seria corrupção de heap.
        let mut batch = unsafe { sys::llama_batch_init(n as i32, self.hidden_width as i32, 1) };
        let mut owned_tokens: Vec<i32> = tokens.to_vec();
        let restore_token = batch.token;
        batch.n_tokens = n as i32;
        batch.token = owned_tokens.as_mut_ptr();

        let rc = unsafe {
            for i in 0..n {
                *batch.pos.add(i) = pos_offset as i32 + i as i32;
                *batch.n_seq_id.add(i) = 1;
                *(*batch.seq_id.add(i)).add(0) = seq_id as i32;
                *batch.logits.add(i) = 0;
            }
            // Linha 0: o `h` que sobrou da chamada anterior. Linhas 1..n: o `h`
            // que o alvo acabou de produzir para as posições 0..n-1.
            ptr::copy_nonoverlapping(self.pending_h.as_ptr(), batch.embd, self.hidden_width);
            let mut missing = None;
            for i in 1..n {
                let src = pom_llama_get_embeddings_nextn_ith(ctx_tgt, (i - 1) as i32);
                if src.is_null() {
                    missing = Some(i - 1);
                    break;
                }
                ptr::copy_nonoverlapping(
                    src,
                    batch.embd.add(i * self.hidden_width),
                    self.hidden_width,
                );
            }
            if let Some(i) = missing {
                batch.token = restore_token;
                sys::llama_batch_free(batch);
                bail!("llama_get_embeddings_nextn_ith devolveu NULL para i={i}");
            }

            let rc = sys::llama_decode(self.ctx, batch);
            batch.token = restore_token;
            sys::llama_batch_free(batch);
            rc
        };
        if rc != 0 {
            bail!("llama_decode no contexto MTP falhou rc={rc}");
        }

        // 2. Guarda as linhas do alvo para a escolha de `accept`.
        self.verify_h.clear();
        self.verify_h.reserve(n * self.hidden_width);
        for i in 0..n {
            let src = unsafe { pom_llama_get_embeddings_nextn_ith(ctx_tgt, i as i32) };
            if src.is_null() {
                self.verify_rows = 0;
                bail!("llama_get_embeddings_nextn_ith devolveu NULL para i={i}");
            }
            self.verify_h
                .extend_from_slice(unsafe { std::slice::from_raw_parts(src, self.hidden_width) });
        }
        self.verify_rows = n;
        self.pending_h
            .copy_from_slice(&self.verify_h[(n - 1) * self.hidden_width..n * self.hidden_width]);
        Ok(())
    }

    /// Descarta a sequência inteira do cache do contexto MTP e as linhas de
    /// verificação pendentes.
    ///
    /// É o par de um lote que o alvo decodificou **sem** `observe`: a partir
    /// dali o cache `nextn` desta sequência teria um buraco de posições, e um
    /// `observe` posterior por cima de um buraco é recusado pelo llama.cpp
    /// ("inconsistent sequence positions"), o que desligaria o MTP do motor
    /// inteiro. Remoção total é aceita por todo motor e deixa a sequência
    /// vazia, de onde qualquer posição inicial é válida. O custo é só de
    /// metadado; o rascunho de uma sequência esquecida só perde aceitação,
    /// nunca correção, porque todo rascunho é verificado pelo alvo.
    pub fn forget(&mut self, seq_id: u32) {
        unsafe {
            let mem = sys::llama_get_memory(self.ctx);
            if !mem.is_null() {
                sys::llama_memory_seq_rm(mem, seq_id as i32, -1, -1);
            }
        }
        self.verify_rows = 0;
    }

    /// Anota quantos tokens **do rascunho** a verificação aceitou, e move o
    /// `pending_h` para a linha certa.
    ///
    /// `accepted` é o `longest_match` da rodada, não o total emitido: o token
    /// emitido a mais é sempre a amostragem real da primeira posição não
    /// confirmada, e o `h` que a próxima rodada precisa é o da última posição
    /// que **entrou** no cache, que é `accepted`.
    pub fn accept(&mut self, accepted: usize) {
        if self.verify_rows == 0 {
            return;
        }
        let row = accepted.min(self.verify_rows - 1);
        self.pending_h.copy_from_slice(
            &self.verify_h[row * self.hidden_width..(row + 1) * self.hidden_width],
        );
    }

    /// Rascunha até `k` tokens a partir do último token confirmado.
    ///
    /// Um decode do bloco `nextn` por token rascunhado: o primeiro parte de
    /// `(pending_h, id_last)`, e cada seguinte reaproveita o `h` que o próprio
    /// bloco produziu. `p_min` corta o rascunho no primeiro token de que a
    /// cabeça não tem certeza - continuar depois disso só alarga o lote de
    /// todos os hops para uma aposta que a verificação vai recusar.
    pub fn draft(
        &mut self,
        id_last: i32,
        n_past: u32,
        seq_id: u32,
        k: u32,
        p_min: f32,
    ) -> Vec<i32> {
        if k == 0 {
            return Vec::new();
        }
        // O cache do rascunho pode ter sobras da rodada anterior nas posições
        // que a verificação recusou; ver `observe`.
        unsafe {
            let mem = sys::llama_get_memory(self.ctx);
            if !mem.is_null() {
                sys::llama_memory_seq_rm(mem, seq_id as i32, n_past as i32, -1);
            }
        }

        let mut out: Vec<i32> = Vec::with_capacity(k as usize);
        let mut token = id_last;
        let mut h: Vec<f32> = self.pending_h.clone();

        for i in 0..k {
            match self.step_once(token, n_past + i, seq_id, &h) {
                Some((id, p, next_h)) => {
                    if p < p_min {
                        break;
                    }
                    out.push(id);
                    token = id;
                    h = next_h;
                }
                None => break,
            }
        }
        out
    }

    /// Um decode do bloco `nextn`: devolve o token mais provável, a
    /// probabilidade dele e o `h` que alimenta o passo seguinte.
    fn step_once(
        &mut self,
        token: i32,
        pos: u32,
        seq_id: u32,
        h: &[f32],
    ) -> Option<(i32, f32, Vec<f32>)> {
        let mut batch = unsafe { sys::llama_batch_init(1, self.hidden_width as i32, 1) };
        // Ver `observe`: o ponteiro de tokens é nosso e volta ao original
        // antes do `llama_batch_free`.
        let mut owned = [token];
        let restore_token = batch.token;
        batch.n_tokens = 1;
        batch.token = owned.as_mut_ptr();

        let rc = unsafe {
            *batch.pos = pos as i32;
            *batch.n_seq_id = 1;
            *(*batch.seq_id).add(0) = seq_id as i32;
            *batch.logits = 1;
            ptr::copy_nonoverlapping(h.as_ptr(), batch.embd, self.hidden_width);
            let rc = sys::llama_decode(self.ctx, batch);
            batch.token = restore_token;
            sys::llama_batch_free(batch);
            rc
        };
        if rc != 0 {
            tracing::warn!("rascunho MTP: llama_decode falhou rc={rc}, seguindo sem rascunho");
            return None;
        }

        let logits = unsafe { sys::llama_get_logits_ith(self.ctx, 0) };
        if logits.is_null() {
            return None;
        }
        let logits = unsafe { std::slice::from_raw_parts(logits, self.n_vocab) };
        let (id, p) = argmax_with_probability(logits);

        let next_h = unsafe { pom_llama_get_embeddings_nextn_ith(self.ctx, 0) };
        if next_h.is_null() {
            return None;
        }
        let next_h = unsafe { std::slice::from_raw_parts(next_h, self.hidden_width) }.to_vec();
        Some((id, p, next_h))
    }
}

/// O token mais provável e a probabilidade dele, num passo só sobre os logits.
///
/// A probabilidade sai de um softmax estável (subtrai o máximo antes de
/// exponenciar); sem isso um logit alto estoura em `inf` e a razão vira `NaN`,
/// que compararia `false` contra qualquer `p_min` e desligaria o corte em
/// silêncio.
fn argmax_with_probability(logits: &[f32]) -> (i32, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }
    let mut best = 0usize;
    let mut max = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > max {
            max = l;
            best = i;
        }
    }
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    (best as i32, if sum > 0.0 { 1.0 / sum } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> crate::llm::EngineConfig {
        crate::llm::EngineConfig {
            n_ctx: 4096,
            offload_kqv: true,
            cache_type_k: crate::KvCacheType::F16,
            cache_type_v: crate::KvCacheType::F16,
            fit_memory: true,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 4096,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 99,
            n_threads: 0,
            max_sessions: 1,
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
    fn mtp_context_keeps_512_with_an_external_moe_executor() {
        let mut cfg = cfg();
        cfg.moe_external_executor = true;
        let params = context_params(&cfg);
        assert_eq!(params.n_batch, 4096);
        // 512 é deliberado: o contexto MTP não herda a exceção de 128 do
        // contexto principal MoE.
        assert_eq!(params.n_ubatch, 512);

        cfg.fit_memory = false;
        let params = context_params(&cfg);
        assert_eq!(params.n_batch, 4096);
        assert_eq!(params.n_ubatch, 4096);
    }

    #[test]
    fn mtp_context_to_ram_turns_off_the_cache_offload() {
        let cfg = cfg();
        let params = context_params_with_offload(&cfg, false);

        assert!(!params.offload_kqv);
        assert!(!params.op_offload);
    }

    #[test]
    fn an_architecture_with_nextn_but_no_mtp_graph_is_refused_before_the_context() {
        let reason = mtp_graph_unavailable_reason(Some("graniteswitch"))
            .expect("arquitetura sem grafo MTP precisa ser recusada");

        assert!(reason.contains("graniteswitch"));
        assert!(reason.contains("grafo MTP"));
    }

    #[test]
    fn qwen35moe_is_a_known_mtp_graph_architecture() {
        assert_eq!(mtp_graph_unavailable_reason(Some("qwen35moe")), None);
    }

    #[test]
    fn the_generated_list_contains_the_known_mtp_architectures_of_the_fork() {
        let known = [
            "bailingmoe3",
            "cohere2moe",
            "deepseek2",
            "deepseek32",
            "deepseek4",
            "glm-dsa",
            "hy_v3",
            "mimo2",
            "nemotron_h_moe",
            "qwen35",
            "qwen35moe",
            "qwen3next",
            "step35",
        ];

        assert!(
            MTP_GRAPH_ARCHITECTURES.len() >= 13,
            "a lista gerada ficou pequena: {:?}",
            MTP_GRAPH_ARCHITECTURES
        );
        for architecture in known {
            assert!(
                MTP_GRAPH_ARCHITECTURES.contains(&architecture),
                "arquitetura MTP ausente: {architecture}"
            );
        }
        assert!(!MTP_GRAPH_ARCHITECTURES.contains(&"hy-v3"));
    }

    #[test]
    fn argmax_with_probability_finds_the_peak_and_its_share() {
        let (id, p) = argmax_with_probability(&[0.0, 2.0, 1.0]);
        assert_eq!(id, 1);
        // e^0 + e^2 + e^1 = 1 + 7.389 + 2.718 → e^2 / soma ≈ 0,665
        assert!((p - 0.665).abs() < 0.01, "p = {p}");
    }

    /// Logits grandes acontecem (um modelo confiante), e o softmax ingênuo
    /// devolveria `inf/inf = NaN` — que passaria por qualquer `p_min` sem
    /// ninguém notar, porque `NaN < x` é `false`.
    #[test]
    fn argmax_with_probability_survives_logits_that_would_overflow() {
        let (id, p) = argmax_with_probability(&[100.0, 200.0, 150.0]);
        assert_eq!(id, 1);
        assert!(p.is_finite() && (p - 1.0).abs() < 1e-6, "p = {p}");
    }

    #[test]
    fn argmax_with_probability_of_a_flat_distribution_is_one_over_n() {
        let (_, p) = argmax_with_probability(&[1.0, 1.0, 1.0, 1.0]);
        assert!((p - 0.25).abs() < 1e-6, "p = {p}");
    }

    /// Sonda ao vivo: carrega um GGUF **com** bloco `nextn` e exerce o ciclo
    /// inteiro - anexar, `observe` de um prompt, `draft`, `accept`.
    ///
    /// `#[ignore]` e dirigida por `POM_MTP_MODEL=<caminho.gguf>` porque
    /// precisa de um modelo em disco (o de referência é
    /// `unsloth/Qwen3.5-4B-MTP-GGUF`, 33 blocos com `nextn` em `blk.32`) e de
    /// backend de GPU compilado. Rode com:
    ///
    /// ```text
    /// POM_MTP_MODEL=/caminho/Qwen3.5-4B-Q4_K_M.gguf \
    ///   cargo test --release --features metal --bin pom \
    ///   mtp::tests::the_nextn_head_drafts_from_a_real_model -- --ignored --nocapture
    /// ```
    ///
    /// O que ela prova, e que nenhum teste de unidade prova: que os quatro
    /// símbolos da ponte C++ **linkam e executam**, que o contexto
    /// `ctx_type = MTP` sobe sobre os mesmos pesos, e que a cabeça devolve
    /// tokens plausíveis em vez de lixo. O prompt passa de um microlote quando
    /// o fit automático está ligado, então esta sonda também exerce o
    /// pareamento `(h_p, x_{p+1})` atravessando a divisão interna do lote.
    /// O que ela **não** prova é ganho de tok/s - isso é medição de ponta a
    /// ponta, e está pendente.
    #[test]
    #[ignore]
    fn the_nextn_head_drafts_from_a_real_model() {
        let Ok(path) = std::env::var("POM_MTP_MODEL") else {
            eprintln!("defina POM_MTP_MODEL para rodar esta sonda");
            return;
        };
        let cfg = crate::llm::EngineConfig {
            n_ctx: 4096,
            offload_kqv: true,
            cache_type_k: crate::KvCacheType::F16,
            cache_type_v: crate::KvCacheType::F16,
            fit_memory: true,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 99,
            n_threads: 0,
            max_sessions: 1,
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
        };
        let mut engine = crate::llm::LlmEngine::load_with_cache(
            &path,
            &cfg,
            &crate::llm::ModelCache::new(),
            false,
        )
        .expect("carregar o modelo");
        assert!(
            engine.mtp_ready(),
            "o modelo em {path} não trouxe bloco nextn — a sonda precisa de um que traga"
        );
        assert!(
            matches!(engine.mtp_allocation(), Some("vram" | "ram")),
            "o runtime precisa publicar onde o MTP foi alocado: {:?}",
            engine.mtp_allocation()
        );

        let prompt_text =
            std::iter::repeat("The capital of France is Paris. The capital of Italy is Rome. ")
                .take(20)
                .collect::<String>();
        let prompt = engine.tokenize(&prompt_text, true, false);
        assert!(
            prompt.len() > 128,
            "prompt curto demais para cruzar o microlote"
        );
        engine
            .decode_tokens(&prompt, 0, 0, false)
            .expect("prefill do alvo (e o observe da cabeça, dentro dele)");

        // O último token confirmado é o próprio fim do prompt; a cabeça
        // rascunha a partir dele exatamente como `step_blocking` faria.
        let n_past = prompt.len() as u32;
        let seed = *prompt.last().unwrap();
        let draft = engine.mtp_draft(seed, n_past, 0, 3, 0.0);
        assert!(
            !draft.is_empty(),
            "a cabeça nextn não rascunhou nada — sem rascunho o MTP não acelera coisa nenhuma"
        );
        let text: String = draft
            .iter()
            .map(|&t| String::from_utf8_lossy(&engine.token_to_piece_bytes(t, false)).into_owned())
            .collect();
        eprintln!("rascunho MTP: {draft:?} → {text:?}");
        assert!(
            draft
                .iter()
                .all(|&t| t >= 0 && (t as usize) < engine.n_vocab),
            "token fora do vocabulário: {draft:?}"
        );

        // Aceitar move o `pending_h`; a rodada seguinte tem que continuar
        // rascunhando em vez de degenerar.
        engine.mtp_accept(1);
        let again = engine.mtp_draft(seed, n_past, 0, 3, 0.0);
        assert!(
            !again.is_empty(),
            "a segunda rodada parou de rascunhar — o estado entre rodadas não sobreviveu"
        );
    }

    /// Sonda ao vivo do rascunho MTP **externo**: um modelo alvo que não
    /// embute `nextn` (ex. `qwen4exp`/Qwen3.8-Flash-Next) mais um GGUF de
    /// rascunho separado ("shared MTP module" da Unsloth, ex.
    /// `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`).
    ///
    /// `#[ignore]`, dirigida por `POM_MTP_TARGET_MODEL` (o alvo) e
    /// `POM_MTP_DRAFT_MODEL` (o rascunho). Diferente da sonda acima, o que
    /// esta prova primeiro é o **pré-requisito arquitetural** que a integração
    /// nova introduziu: o alvo precisa exportar `h_nextn` do próprio grafo
    /// mesmo quando o rascunho é um arquivo à parte - `qwen4exp.cpp` ganhou
    /// isso em `pieceofmindcloud/llama.cpp` (`pom`), mas um alvo de
    /// outra arquitetura sem esse suporte ainda faz `engine.mtp_ready()`
    /// continuar falso mesmo com um rascunho válido (degrada com aviso, não
    /// aborta - ver o cabeçalho deste módulo). A asserção abaixo distingue os
    /// dois desfechos em vez de tratar qualquer `mtp_ready() == false` como
    /// erro de sonda.
    ///
    /// Rode com:
    ///
    /// ```text
    /// POM_MTP_TARGET_MODEL=/caminho/Qwen3.8-Flash-Next-Q8_0.gguf \
    /// POM_MTP_DRAFT_MODEL=/caminho/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
    ///   cargo test --release --features metal --bin pom \
    ///   mtp::tests::an_external_mtp_draft_attaches_to_a_target_without_embedded_nextn \
    ///   -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn an_external_mtp_draft_attaches_to_a_target_without_embedded_nextn() {
        let (Ok(target_path), Ok(draft_path)) = (
            std::env::var("POM_MTP_TARGET_MODEL"),
            std::env::var("POM_MTP_DRAFT_MODEL"),
        ) else {
            eprintln!("defina POM_MTP_TARGET_MODEL e POM_MTP_DRAFT_MODEL para rodar esta sonda");
            return;
        };
        let cfg = crate::llm::EngineConfig {
            n_ctx: 4096,
            offload_kqv: true,
            cache_type_k: crate::KvCacheType::F16,
            cache_type_v: crate::KvCacheType::F16,
            fit_memory: true,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 99,
            n_threads: 0,
            max_sessions: 1,
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
            mtp_draft_path: Some(draft_path.clone()),
        };
        let engine = crate::llm::LlmEngine::load_with_cache(
            &target_path,
            &cfg,
            &crate::llm::ModelCache::new(),
            false,
        )
        .expect("carregar o alvo");

        if !engine.mtp_ready() {
            eprintln!(
                "MTP não ficou pronto: {:?}. Se a razão citar arquitetura sem grafo MTP, o \
                 alvo ainda precisa da correção no fork (emitir h_nextn do próprio grafo) - \
                 ver a nota de limitação em `crate::mtp` e o AGENTS.md de models/. Se citar \
                 outra coisa (arquivo não encontrado, largura incompatível), é o rascunho ou \
                 o caminho configurado que está errado.",
                engine.mtp_unavailable_reason()
            );
            panic!("MTP não ficou pronto - ver a mensagem acima para qual dos dois motivos");
        }
        assert!(
            matches!(engine.mtp_allocation(), Some("vram" | "ram")),
            "o runtime precisa publicar onde o MTP foi alocado: {:?}",
            engine.mtp_allocation()
        );
        eprintln!("MTP externo pronto: alocação {:?}", engine.mtp_allocation());
    }
}

/// Where a deployment wants the MTP block (its weights, KV and compute).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtpPlacement {
    /// In VRAM when the whole main model still fits there with it; otherwise
    /// in host memory, so the main model keeps every layer on the GPU.
    #[default]
    Auto,
    /// Always in VRAM, reserved first: the main model takes what is left and
    /// gives up layers to host memory if it has to.
    Vram,
    /// Always in host memory, run by the CPU: no VRAM is spent on MTP.
    Ram,
}
