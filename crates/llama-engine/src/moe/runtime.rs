//! Runtime local que entrega estados ocultos ao bloco MoE Q4_0.
//!
//! O runtime não cria um segundo protocolo: recebe o mesmo vetor f32 que
//! `decode_hidden` já aceitou do peer e devolve outro vetor da mesma forma.
//! Atenção e o restante das camadas continuam no contexto llama.cpp; aqui a
//! fronteira própria é o router GGUF mais o FFN esparso da camada local.

#[cfg(feature = "cuda")]
use super::block::HeterogeneousLayerWorkingSetDiagnostics;
use super::block::{
    HeterogeneousMoeBlock, IqQuantMoeBlock, LegacyQuantMoeBlock, LowBitMoeBlock, MoeBlockError,
    Nvfp4MoeBlock, PlainMoeBlock, Q4_0MoeBlock,
};
use super::cache::{CacheError, ExpertCache};
#[cfg(feature = "cuda")]
use super::cache::{DeviceWarmupOptions, DeviceWarmupReport, DeviceWarmupStop, ExpertKey};
#[cfg(feature = "cuda")]
use super::cuda::{CudaMoeDeviceDiagnostics, CudaTransferProfile, HeterogeneousCudaExecutor};
use super::format_heterogeneous::{HeterogeneousMoeSource, TensorRole};
use super::format_iq_quants::{IqQuantFormat, IqQuantSource};
use super::format_k_quants::{KQuantFormat, KQuantSource};
use super::format_legacy_quants::{LegacyQuantFormat, LegacyQuantSource};
use super::format_low_bits::{LowBitFormat, LowBitSource};
use super::format_nvfp4::Nvfp4Source;
use super::format_plain::{PlainFormat, PlainSource};
use super::format_q4_0::{
    detect_expert_dtypes, detect_expert_format, detect_expert_layer_count, ggml_type_name,
    ExpertFormat,
};
use super::format_q4_0::{Q4Error, Q4LoadError, Q4_0Source};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use std::collections::VecDeque;
use std::fmt;
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "cuda")]
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExecutorKind {
    CpuReference,
    Cuda,
}

impl MoeExecutorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuReference => "cpu-reference",
            Self::Cuda => "cuda",
        }
    }
}

/// Evidência do executor que realmente processou os experts de um forward.
///
/// `dtypes` é uma lista, não um único valor: arquivos UD podem misturar IQ3_S,
/// IQ2_S e IQ4_XS no mesmo modelo. `reason` fica preenchido quando o caminho
/// CPUReference é escolhido por falta de ponte CUDA, para que a telemetria não
/// transforme o caminho lento em uma escolha invisível.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeExecutorTelemetry {
    pub executor: MoeExecutorKind,
    pub dtypes: Vec<u32>,
    pub reason: Option<String>,
    pub diagnostics: Option<MoeExecutorDiagnostics>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceCacheBudgetDiagnostics {
    pub baseline_free_bytes: u64,
    pub compute_bytes: u64,
    pub weights_bytes: u64,
    pub fixed_bytes: u64,
    pub reserve_bytes: u64,
    pub current_free_bytes: u64,
    pub resident_bytes: u64,
    pub baseline_limit_bytes: Option<u64>,
    pub dynamic_limit_bytes: Option<u64>,
    pub effective_limit_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MoeExecutorDiagnostics {
    pub profiled: bool,
    pub calls: u64,
    pub cpu_expert_calls: u64,
    pub gpu_expert_calls: u64,
    pub coexecution_gpu_fraction: Option<f32>,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_admissions: u64,
    pub cache_evictions: u64,
    pub cache_rejections: u64,
    pub cache_entries: usize,
    pub cache_resident_bytes: usize,
    pub device_cache_hits: u64,
    pub device_cache_misses: u64,
    pub device_cache_admissions: u64,
    pub device_cache_evictions: u64,
    pub device_cache_rejections: u64,
    pub device_cache_uploads: u64,
    pub device_cache_upload_bytes: u64,
    pub device_cache_upload_ms: f64,
    pub device_cache_resident_bytes: usize,
    pub device_cache_devices: Vec<String>,
    pub device_cache_resident_bytes_by_device: Vec<usize>,
    /// Limite efetivo atual de cada shard, na mesma ordem de
    /// `device_cache_devices`. `None` indica que a sonda CUDA falhou.
    #[cfg(feature = "cuda")]
    #[serde(default)]
    pub device_cache_capacity_bytes: Option<u64>,
    #[cfg(feature = "cuda")]
    #[serde(default)]
    pub device_cache_capacity_bytes_by_device: Vec<Option<u64>>,
    #[cfg(feature = "cuda")]
    #[serde(default)]
    pub device_cache_entries_by_device: Vec<usize>,
    /// Entradas que formaram o limite, na mesma ordem de
    /// `device_cache_devices`. `None` indica que a sonda falhou.
    #[cfg(feature = "cuda")]
    #[serde(default)]
    pub device_cache_budget_by_device: Vec<Option<DeviceCacheBudgetDiagnostics>>,
    /// Working set observado por camada. `distinct_experts_per_token` é a
    /// soma por token e deve ser dividida por `tokens` para obter a média.
    #[cfg(feature = "cuda")]
    #[serde(default)]
    pub cache_working_set_by_layer: Vec<HeterogeneousLayerWorkingSetDiagnostics>,
    /// Chamadas efetivamente executadas por ordinal CUDA. Resident bytes não
    /// provam que uma placa participou da rota; estes contadores provam.
    #[cfg(feature = "cuda")]
    #[serde(default)]
    pub cuda_device_calls: Vec<CudaMoeDeviceDiagnostics>,
    pub payload_validation_ms: f64,
    pub cuda_alloc_ms: f64,
    pub cuda_h2d_ms: f64,
    pub cuda_h2d_bytes: u64,
    pub cuda_kernel_ms: f64,
    pub cuda_kernel_span_ms: f64,
    pub cuda_launch_ms: f64,
    pub cuda_sync_ms: f64,
    pub cuda_d2h_ms: f64,
    pub cuda_d2h_bytes: u64,
    pub cuda_transfer_span_ms: f64,
    pub cuda_free_ms: f64,
    pub cuda_buffer_mask: u64,
    pub cuda_buffer_switches: u64,
    pub host_cache_lookup_ms: f64,
    pub host_expert_load_ms: f64,
    pub host_cache_admit_ms: f64,
    pub host_batch_prepare_ms: f64,
    pub host_result_copy_ms: f64,
    pub host_aggregation_ms: f64,
    pub cuda_call_wall_ms: f64,
    pub runtime_forward_ms: f64,
    pub warmup_status: String,
    pub warmup_candidates: u64,
    pub warmup_admitted: u64,
    pub warmup_admitted_bytes: u64,
    pub warmup_resident_bytes: usize,
    pub warmup_elapsed_ms: f64,
    pub warmup_error: Option<String>,
}

#[derive(Debug, Clone, Copy, Default)]
struct MoeRuntimeDiagnostics {
    runtime_forward_ms: f64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default)]
struct MoeWarmupState {
    status: String,
    candidates: u64,
    admitted: u64,
    admitted_bytes: u64,
    resident_bytes: usize,
    elapsed_ms: f64,
    error: Option<String>,
    cancellation: Option<Arc<AtomicBool>>,
}

#[cfg(feature = "cuda")]
impl MoeWarmupState {
    fn new() -> Self {
        Self {
            status: "disabled".to_owned(),
            ..Self::default()
        }
    }

    fn apply(&mut self, report: DeviceWarmupReport) {
        self.status = match report.stop {
            DeviceWarmupStop::Completed => "completed",
            DeviceWarmupStop::BudgetExhausted => "budget-exhausted",
            DeviceWarmupStop::Cancelled => "cancelled",
            DeviceWarmupStop::TimedOut => "timed-out",
            DeviceWarmupStop::Failed => "failed",
        }
        .to_owned();
        self.candidates = report.candidates;
        self.admitted = report.admitted;
        self.admitted_bytes = report.admitted_bytes;
        self.resident_bytes = report.resident_bytes;
        self.elapsed_ms = report.elapsed_ms;
        self.error = report.error;
        self.cancellation = None;
    }
}

impl From<super::format_nvfp4::Nvfp4BackendKind> for MoeExecutorKind {
    fn from(kind: super::format_nvfp4::Nvfp4BackendKind) -> Self {
        match kind {
            super::format_nvfp4::Nvfp4BackendKind::CpuReference => Self::CpuReference,
            super::format_nvfp4::Nvfp4BackendKind::Cuda => Self::Cuda,
        }
    }
}

#[derive(Debug)]
pub enum MoeRuntimeError {
    Source(Q4LoadError),
    Cache(CacheError),
    Format(Q4Error),
    Block(String),
    FormatMismatch {
        source: String,
        block: String,
    },
    InvalidRoute {
        layer: u32,
        source: String,
        expert_count: usize,
        expected_expert_count: usize,
        expert_count_used: usize,
        hidden_values: usize,
        expected_hidden_values: usize,
        selected_values: usize,
        expected_selected_values: usize,
        weight_values: usize,
        expected_weight_values: usize,
    },
    MissingLayer {
        layer: u32,
        layers: u32,
    },
    InvalidShape {
        values: usize,
        tokens: usize,
        hidden: usize,
    },
}

impl fmt::Display for MoeRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => error.fmt(formatter),
            Self::Cache(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
            Self::Block(error) => formatter.write_str(error),
            Self::FormatMismatch { source, block } => write!(
                formatter,
                "formatos MoE diferentes: fonte={source}, bloco={block}"
            ),
            Self::InvalidRoute {
                layer,
                source,
                expert_count,
                expected_expert_count,
                expert_count_used,
                hidden_values,
                expected_hidden_values,
                selected_values,
                expected_selected_values,
                weight_values,
                expected_weight_values,
            } => write!(
                formatter,
                "rota MoE inválida na camada {layer} (fonte={source}): expert_count={expert_count}, esperado={expected_expert_count}; expert_count_used={expert_count_used}; hidden={hidden_values}, esperado={expected_hidden_values}; selected={selected_values}, esperado={expected_selected_values}; weights={weight_values}, esperado={expected_weight_values}"
            ),
            Self::MissingLayer { layer, layers } => {
                write!(formatter, "camada MoE local {layer} fora de [0, {layers})")
            }
            Self::InvalidShape {
                values,
                tokens,
                hidden,
            } => write!(
                formatter,
                "runtime MoE recebeu {values} floats, esperado {tokens} × {hidden}"
            ),
        }
    }
}

impl std::error::Error for MoeRuntimeError {}

impl From<Q4LoadError> for MoeRuntimeError {
    fn from(error: Q4LoadError) -> Self {
        Self::Source(error)
    }
}

impl From<CacheError> for MoeRuntimeError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

impl From<Q4Error> for MoeRuntimeError {
    fn from(error: Q4Error) -> Self {
        Self::Format(error)
    }
}

impl From<MoeBlockError> for MoeRuntimeError {
    fn from(error: MoeBlockError) -> Self {
        Self::Block(error.to_string())
    }
}

#[cfg(any(feature = "cuda", test))]
fn accept_cpu_replay_after_cuda_nonfinite(
    cuda_error: &str,
    output: Vec<f32>,
) -> Result<Vec<f32>, MoeRuntimeError> {
    if let Some((index, count, value)) = nonfinite_summary(&output) {
        let kind = if value.is_nan() { "NaN" } else { "infinito" };
        return Err(MoeRuntimeError::Block(format!(
            "{cuda_error}; replay CPU também devolveu {count} valores não finitos, primeiro índice {index} ({kind})"
        )));
    }
    tracing::warn!(
        error = cuda_error,
        "replay CPU do MoE produziu saída finita; resposta preservada, ponte CUDA precisa de revisão"
    );
    Ok(output)
}

#[cfg(any(feature = "cuda", test))]
fn nonfinite_summary(values: &[f32]) -> Option<(usize, usize, f32)> {
    let mut first = None;
    let mut count = 0;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            first.get_or_insert(index);
            count += 1;
        }
    }
    first.map(|index| (index, count, values[index]))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LayerInputSummary {
    elements: usize,
    non_finite_count: usize,
    first_non_finite: Option<usize>,
    finite_min: Option<f32>,
    finite_max: Option<f32>,
}

fn summarize_layer_input(values: &[f32]) -> LayerInputSummary {
    let mut summary = LayerInputSummary {
        elements: values.len(),
        non_finite_count: 0,
        first_non_finite: None,
        finite_min: None,
        finite_max: None,
    };
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            summary.non_finite_count += 1;
            summary.first_non_finite.get_or_insert(index);
            continue;
        }
        summary.finite_min = Some(
            summary
                .finite_min
                .map_or(value, |current| current.min(value)),
        );
        summary.finite_max = Some(
            summary
                .finite_max
                .map_or(value, |current| current.max(value)),
        );
    }
    summary
}

fn moe_layer_input_diagnostic_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn moe_layer_input_diagnostic_requested() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        moe_layer_input_diagnostic_enabled(
            std::env::var("POM_MOE_LAYER_INPUT_DIAGNOSTIC")
                .ok()
                .as_deref(),
        )
    })
}

fn log_moe_layer_input_diagnostic(
    source: &MoeSource,
    layer: u32,
    hidden: &[f32],
    selected_experts: &[i32],
    weights: &[f32],
    tokens: usize,
    expert_count_used: usize,
    weight_before_ffn: bool,
) {
    if layer != 0 || !moe_layer_input_diagnostic_requested() {
        return;
    }
    let summary = summarize_layer_input(hidden);
    tracing::warn!(
        target: "moe.layer_input_diagnostic",
        layer,
        source = source.kind(),
        dtypes = ?source.dtypes(),
        tokens,
        expert_count_used,
        hidden_width = source.hidden(),
        hidden_elements = summary.elements,
        hidden_non_finite_count = summary.non_finite_count,
        hidden_first_non_finite = ?summary.first_non_finite,
        hidden_finite_min = ?summary.finite_min,
        hidden_finite_max = ?summary.finite_max,
        hidden_first_values = ?hidden.iter().take(4).copied().collect::<Vec<_>>(),
        selected_experts_first = ?selected_experts.iter().take(8).copied().collect::<Vec<_>>(),
        weights_first = ?weights.iter().take(8).copied().collect::<Vec<_>>(),
        weight_before_ffn,
        "POM_MOE_LAYER_INPUT layer 0 antes dos executores"
    );
}

#[cfg(any(feature = "cuda", test))]
fn require_cuda_compute_bytes(compute_bytes: Option<&[(u32, u64)]>) -> Result<(), MoeRuntimeError> {
    compute_bytes
        .filter(|devices| !devices.is_empty() && devices.iter().all(|(_, bytes)| *bytes > 0))
        .ok_or_else(|| {
        MoeRuntimeError::Block(
            "executor CUDA recusado: llama.cpp não reportou um compute buffer CUDA medido por dispositivo".into(),
        )
    })
    .map(|_| ())
}

#[cfg(feature = "cuda")]
fn create_measured_device_cache(
    compute_bytes: Option<&[(u32, u64)]>,
) -> Result<Arc<super::cache::DeviceExpertCache>, MoeRuntimeError> {
    let compute_bytes = compute_bytes.ok_or_else(|| {
        MoeRuntimeError::Block(
            "executor CUDA recusado: llama.cpp não reportou compute buffers CUDA medidos por dispositivo".into(),
        )
    })?;
    super::cuda::create_device_cache_for_devices(compute_bytes)
        .map_err(|error| MoeRuntimeError::Block(error.to_string()))
}

enum MoeSource {
    Legacy(LegacyQuantSource),
    Plain(PlainSource),
    Q4(Q4_0Source),
    K(KQuantSource),
    Iq(IqQuantSource),
    Nvfp4(Nvfp4Source),
    LowBit(LowBitSource),
    Heterogeneous(HeterogeneousMoeSource),
}

impl MoeSource {
    fn kind(&self) -> &'static str {
        match self {
            Self::Legacy(_) => "legacy",
            Self::Plain(_) => "plain",
            Self::Q4(_) => "q4_0",
            Self::K(_) => "k_quant",
            Self::Iq(_) => "iq_quant",
            Self::Nvfp4(_) => "nvfp4",
            Self::LowBit(_) => "low_bit",
            Self::Heterogeneous(_) => "heterogeneous",
        }
    }

    fn hidden(&self) -> usize {
        match self {
            Self::Legacy(source) => source.hidden(),
            Self::Plain(source) => source.hidden(),
            Self::Q4(source) => source.hidden(),
            Self::K(source) => source.hidden(),
            Self::Iq(source) => source.hidden(),
            Self::Nvfp4(source) => source.hidden(),
            Self::LowBit(source) => source.hidden(),
            Self::Heterogeneous(source) => source.hidden(),
        }
    }

    fn geometry(&self, layer: u32) -> Result<(usize, usize), MoeRuntimeError> {
        match self {
            Self::Legacy(source) => Ok((source.hidden(), source.intermediate())),
            Self::Plain(source) => Ok((source.hidden(), source.intermediate())),
            Self::Q4(source) => Ok((source.hidden(), source.intermediate())),
            Self::K(source) => Ok((source.hidden(), source.intermediate())),
            Self::Iq(source) => Ok((source.hidden(), source.intermediate())),
            Self::Nvfp4(source) => Ok((source.hidden(), source.intermediate())),
            Self::LowBit(source) => Ok((source.hidden(), source.intermediate())),
            Self::Heterogeneous(source) => source
                .layer_geometry(layer)
                .map_err(MoeRuntimeError::Source),
        }
    }

    fn experts(&self) -> u32 {
        match self {
            Self::Legacy(source) => source.experts(),
            Self::Plain(source) => source.experts(),
            Self::Q4(source) => source.experts(),
            Self::K(source) => source.experts(),
            Self::Iq(source) => source.experts(),
            Self::Nvfp4(source) => source.experts(),
            Self::LowBit(source) => source.experts(),
            Self::Heterogeneous(source) => source.experts(),
        }
    }

    fn dtype(&self) -> u32 {
        match self {
            Self::Legacy(source) => source.format().dtype(),
            Self::Plain(source) => source.format().dtype(),
            Self::Q4(_) => 2,
            Self::K(source) => source.format().dtype(),
            Self::Iq(source) => source.format().dtype(),
            Self::Nvfp4(_) => 40,
            Self::LowBit(source) => source.format().dtype(),
            Self::Heterogeneous(_) => 0,
        }
    }

    fn dtypes(&self) -> Vec<u32> {
        let mut dtypes = match self {
            Self::Legacy(source) => vec![source.format().dtype()],
            Self::Plain(source) => vec![source.format().dtype()],
            Self::Q4(_) => vec![2],
            Self::K(source) => vec![source.format().dtype()],
            Self::Iq(source) => vec![source.format().dtype()],
            Self::Nvfp4(_) => vec![40],
            Self::LowBit(source) => vec![source.format().dtype()],
            Self::Heterogeneous(source) => source.dtypes(),
        };
        dtypes.sort_unstable();
        dtypes.dedup();
        dtypes
    }

    fn activation(&self) -> Option<i32> {
        match self {
            Self::Nvfp4(_) => Some(2), // LLM_FFN_GELU, as implemented by NVFP4Expert
            Self::Heterogeneous(_) => None,
            _ => Some(1), // LLM_FFN_SILU, as implemented by the other decoders
        }
    }

    #[cfg(feature = "cuda")]
    fn warmup_payloads(
        &self,
        layer: u32,
        expert: u32,
        model: &str,
    ) -> Result<Vec<(ExpertKey, Vec<u8>)>, String> {
        match self {
            Self::Plain(source) => source
                .load_expert(layer, expert)
                .map(|payload| {
                    vec![(
                        ExpertKey::new(model, layer, expert, source.format().name()),
                        payload,
                    )]
                })
                .map_err(|error| error.to_string()),
            Self::K(source) => source
                .load_expert(layer, expert)
                .map(|payload| {
                    vec![(
                        ExpertKey::new(model, layer, expert, source.format().name()),
                        payload,
                    )]
                })
                .map_err(|error| error.to_string()),
            Self::Iq(source) => source
                .load_expert(layer, expert)
                .map(|payload| {
                    vec![(
                        ExpertKey::new(model, layer, expert, source.format().name()),
                        payload,
                    )]
                })
                .map_err(|error| error.to_string()),
            Self::Nvfp4(source) => source
                .load_expert(layer, expert)
                .map(|payload| vec![(ExpertKey::new(model, layer, expert, "nvfp4"), payload)])
                .map_err(|error| error.to_string()),
            Self::Heterogeneous(source) => source
                .load_expert_tensors(layer, expert)
                .map(|payloads| {
                    payloads
                        .into_iter()
                        .map(|payload| {
                            (
                                ExpertKey::new(
                                    model,
                                    layer,
                                    expert,
                                    format!(
                                        "heterogeneous:{}:{}",
                                        payload.role.as_str(),
                                        payload.dtype
                                    ),
                                ),
                                payload.bytes,
                            )
                        })
                        .collect()
                })
                .map_err(|error| error.to_string()),
            Self::Legacy(_) | Self::Q4(_) | Self::LowBit(_) => Err(
                "aquecimento de device recusado: a fonte não possui ponte CUDA residente".into(),
            ),
        }
    }
}

#[cfg(any(feature = "cuda", test))]
/// Percorre a matriz expert/camada em ordem de expert para espalhar o working
/// set residente por todas as camadas antes de esgotar o orçamento de VRAM.
/// A ordem por camada aquecia apenas o prefixo do modelo e deixava a primeira
/// requisição pagar misses concentrados nas camadas finais.
fn next_warmup_position(
    layer: &mut u32,
    expert: &mut u32,
    layers: u32,
    experts: u32,
) -> Option<(u32, u32)> {
    if *layer >= layers || *expert >= experts || layers == 0 || experts == 0 {
        return None;
    }
    let position = (*layer, *expert);
    *layer = layer.saturating_add(1);
    if *layer >= layers {
        *layer = 0;
        *expert = expert.saturating_add(1);
    }
    Some(position)
}

#[cfg(feature = "cuda")]
struct DeviceWarmupCandidates<'a> {
    source: &'a MoeSource,
    model: &'a str,
    cancellation: Arc<AtomicBool>,
    layers: u32,
    experts: u32,
    layer: u32,
    expert: u32,
    pending: VecDeque<Result<(ExpertKey, Vec<u8>), String>>,
}

#[cfg(feature = "cuda")]
impl<'a> DeviceWarmupCandidates<'a> {
    fn new(
        source: &'a MoeSource,
        model: &'a str,
        cancellation: Arc<AtomicBool>,
        layers: u32,
        experts: u32,
    ) -> Self {
        Self {
            source,
            model,
            cancellation,
            layers,
            experts,
            layer: 0,
            expert: 0,
            pending: VecDeque::new(),
        }
    }
}

#[cfg(feature = "cuda")]
impl Iterator for DeviceWarmupCandidates<'_> {
    type Item = Result<(ExpertKey, Vec<u8>), String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cancellation.load(Ordering::Acquire) {
            return None;
        }
        if let Some(candidate) = self.pending.pop_front() {
            return Some(candidate);
        }
        let (layer, expert) =
            next_warmup_position(&mut self.layer, &mut self.expert, self.layers, self.experts)?;

        if self.cancellation.load(Ordering::Acquire) {
            return None;
        }
        match self.source.warmup_payloads(layer, expert, self.model) {
            Ok(payloads) => {
                if self.cancellation.load(Ordering::Acquire) {
                    return None;
                }
                self.pending.extend(payloads.into_iter().map(Ok));
                self.pending.pop_front()
            }
            Err(error) => Some(Err(format!(
                "aquecimento do expert {expert} da camada {layer}: {error}"
            ))),
        }
    }
}

enum MoeBlock {
    Legacy(LegacyQuantMoeBlock),
    Plain(PlainMoeBlock),
    Q4(Q4_0MoeBlock),
    K(super::block::KQuantMoeBlock),
    Iq(IqQuantMoeBlock),
    Nvfp4(Nvfp4MoeBlock),
    LowBit(LowBitMoeBlock),
    Heterogeneous(HeterogeneousMoeBlock),
}

impl MoeBlock {
    fn kind(&self) -> &'static str {
        match self {
            Self::Legacy(_) => "legacy",
            Self::Plain(_) => "plain",
            Self::Q4(_) => "q4_0",
            Self::K(_) => "k_quant",
            Self::Iq(_) => "iq_quant",
            Self::Nvfp4(_) => "nvfp4",
            Self::LowBit(_) => "low_bit",
            Self::Heterogeneous(_) => "heterogeneous",
        }
    }
}

pub struct MoeRuntime {
    source: MoeSource,
    blocks: Vec<MoeBlock>,
    cache: Arc<ExpertCache>,
    runtime_diagnostics: Mutex<MoeRuntimeDiagnostics>,
    #[cfg(feature = "cuda")]
    model: String,
    #[cfg(feature = "cuda")]
    device_cache: Option<Arc<super::cache::DeviceExpertCache>>,
    #[cfg(feature = "cuda")]
    warmup: Arc<Mutex<MoeWarmupState>>,
    #[cfg(feature = "cuda")]
    warmup_thread: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
}

pub struct MoeRouteBatch<'a> {
    pub layer: u32,
    pub hidden: &'a [f32],
    pub selected_experts: &'a [i32],
    pub weights: &'a [f32],
    pub tokens: usize,
    pub expert_count: usize,
    pub expert_count_used: usize,
    pub activation: i32,
    pub weight_before_ffn: bool,
}

/// Ownership decision made before loading the llama.cpp model.
///
/// A `1` means that every expert tensor in that local layer has an external
/// executor, CPU or device, and may be metadata-only in llama.cpp. A `0` means
/// that the combination has no supported external executor and must be
/// rejected before loading the model. Keeping this list data-driven prevents
/// an unavailable dtype from turning into a silent native fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeExternalExecutorPlan {
    pub layers: Vec<u8>,
    pub dtypes: Vec<u32>,
    pub reason: Option<String>,
}

impl MoeExternalExecutorPlan {
    pub fn enabled(&self) -> bool {
        self.layers.iter().any(|value| *value != 0)
    }
}

fn make_external_executor_plan(
    detected: ExpertFormat,
    dtypes: Vec<u32>,
    layers: usize,
    cuda_compiled: bool,
    nvfp4_cpu_requested: bool,
) -> MoeExternalExecutorPlan {
    let dtype_list = dtypes
        .iter()
        .map(|dtype| format!("{dtype} ({})", ggml_type_name(*dtype)))
        .collect::<Vec<_>>()
        .join(", ");
    let unsupported_dtype_list = dtypes
        .iter()
        .filter(|dtype| !cuda_heterogeneous_dtype_supported(**dtype))
        .map(|dtype| format!("{dtype} ({})", ggml_type_name(*dtype)))
        .collect::<Vec<_>>()
        .join(", ");
    let unsupported_dtype_list = if unsupported_dtype_list.is_empty() {
        dtype_list.clone()
    } else {
        unsupported_dtype_list
    };
    let single_dtype = dtypes.len() == 1;
    let bridge_supported = match detected {
        ExpertFormat::F32 | ExpertFormat::F16 | ExpertFormat::Bf16 => cuda_compiled,
        ExpertFormat::Q4_0 => cuda_compiled,
        ExpertFormat::Nvfp4 => cuda_compiled && !nvfp4_cpu_requested,
        ExpertFormat::Q2K
        | ExpertFormat::Q3K
        | ExpertFormat::Q4K
        | ExpertFormat::Q5K
        | ExpertFormat::Q6K => {
            cuda_compiled
                && single_dtype
                && KQuantFormat::from_dtype(dtypes[0]).is_ok_and(|format| format.cuda_supported())
        }
        ExpertFormat::Iq3XXS | ExpertFormat::Iq3S | ExpertFormat::Iq2S | ExpertFormat::Iq4XS => {
            cuda_compiled
                && single_dtype
                && IqQuantFormat::from_dtype(dtypes[0]).is_ok_and(|format| format.cuda_supported())
        }
        ExpertFormat::Heterogeneous => {
            cuda_compiled
                && !dtypes.is_empty()
                && dtypes
                    .iter()
                    .all(|dtype| cuda_heterogeneous_dtype_supported(*dtype))
        }
        _ => false,
    };
    if bridge_supported {
        return MoeExternalExecutorPlan {
            layers: vec![1; layers],
            dtypes,
            reason: None,
        };
    }

    // The CPU reference blocks are also executors. A missing CUDA bridge must
    // not hand the layer back to llama.cpp when the Rust runtime can execute
    // the exact GGML format itself. NVFP4 is the one exception: without CUDA
    // it requires the explicit CPU opt-in enforced by its constructor.
    let cpu_reference_supported = match detected {
        ExpertFormat::Nvfp4 => nvfp4_cpu_requested,
        ExpertFormat::Heterogeneous => dtypes
            .iter()
            .all(|dtype| heterogeneous_cpu_dtype_supported(*dtype)),
        _ => true,
    };
    if cpu_reference_supported {
        return MoeExternalExecutorPlan {
            layers: vec![1; layers],
            dtypes,
            reason: None,
        };
    }

    let cpu_unsupported_dtype_list = dtypes
        .iter()
        .filter(|dtype| !heterogeneous_cpu_dtype_supported(**dtype))
        .map(|dtype| format!("{dtype} ({})", ggml_type_name(*dtype)))
        .collect::<Vec<_>>()
        .join(", ");
    let reason = if !cpu_unsupported_dtype_list.is_empty() {
        format!(
            "dtypes heterogêneos [{cpu_unsupported_dtype_list}] não têm decoder CPU externo; a carga MoE deve ser recusada antes do modelo"
        )
    } else if !cuda_compiled {
        format!(
            "executor externo recusado para dtype(s) [{dtype_list}]: binário sem feature cuda; a carga MoE deve ser recusada antes do modelo"
        )
    } else if detected == ExpertFormat::Heterogeneous {
        format!(
            "executor externo recusado para dtypes heterogêneos [{unsupported_dtype_list}]: tensor(es) com esses dtypes não têm decoder externo suportado; a carga MoE deve ser recusada antes do modelo"
        )
    } else if detected == ExpertFormat::Nvfp4 && nvfp4_cpu_requested {
        format!(
            "executor externo recusado para dtype [{dtype_list}]: POM_MOE_NVFP4_BACKEND=cpu; a carga MoE deve ser recusada antes do modelo"
        )
    } else {
        format!(
            "executor externo recusado para dtype [{dtype_list}]: ponte device não suportada; a carga MoE deve ser recusada antes do modelo"
        )
    };
    MoeExternalExecutorPlan {
        layers: vec![0; layers],
        dtypes,
        reason: Some(reason),
    }
}

fn cuda_heterogeneous_dtype_supported(dtype: u32) -> bool {
    dtype == 2
        || KQuantFormat::from_dtype(dtype).is_ok_and(|format| format.cuda_supported())
        || IqQuantFormat::from_dtype(dtype).is_ok_and(|format| format.cuda_supported())
        || LowBitFormat::from_dtype(dtype).is_ok_and(|format| matches!(format, LowBitFormat::Mxfp4))
}

fn heterogeneous_cpu_dtype_supported(dtype: u32) -> bool {
    matches!(
        dtype,
        0 | 1 | 2 | 3 | 6 | 7 | 8 | 10..=23 | 29 | 30 | 34 | 35 | 39 | 41 | 42
    )
}

impl MoeRuntime {
    /// Decides ownership before `llama_model_load_from_file` so every format
    /// with a usable external CPU or device executor has its experts removed
    /// from the native model, while a truly unavailable format stays native
    /// with a concrete reason.
    pub fn external_executor_plan(
        path: impl AsRef<Path>,
    ) -> Result<MoeExternalExecutorPlan, MoeRuntimeError> {
        let path = path.as_ref();
        let detected = detect_expert_format(path)?;
        let dtypes = detect_expert_dtypes(path)?;
        let layers = detect_expert_layer_count(path)?;
        #[cfg(feature = "cuda")]
        let profile_requests_cpu = matches!(detected, ExpertFormat::Nvfp4)
            && CudaTransferProfile::load_for_model(path)
                .map_err(MoeRuntimeError::Block)?
                .is_some_and(|profile| !profile.prefers_cuda());
        #[cfg(not(feature = "cuda"))]
        let profile_requests_cpu = false;
        let mut plan = make_external_executor_plan(
            detected,
            dtypes,
            layers,
            cfg!(feature = "cuda"),
            std::env::var("POM_MOE_NVFP4_BACKEND")
                .is_ok_and(|value| value.eq_ignore_ascii_case("cpu")),
        );
        if profile_requests_cpu {
            // Keep ownership external while letting Nvfp4MoeBlock select its
            // measured CPU backend. Zero would silently return the experts to
            // llama.cpp and make a performance policy look like a fallback.
            plan.layers.fill(1);
            plan.reason = None;
        }
        Ok(plan)
    }

    /// Carrega o primeiro formato próprio suportado e compartilha o cache
    /// entre todas as camadas locais desta fatia.
    pub fn open(
        path: impl AsRef<Path>,
        model: impl Into<String>,
        experts: u32,
        top_k: usize,
        weights_bytes: u64,
        cuda_compute_bytes: Option<Vec<(u32, u64)>>,
    ) -> Result<Self, MoeRuntimeError> {
        #[cfg(not(feature = "cuda"))]
        let _ = cuda_compute_bytes;
        #[cfg(feature = "cuda")]
        let cuda_compute_bytes = cuda_compute_bytes.as_deref();
        #[cfg(feature = "cuda")]
        require_cuda_compute_bytes(cuda_compute_bytes)?;
        let cache = ExpertCache::measured_system(weights_bytes, 0, 0)?;
        let model = model.into();
        let detected = detect_expert_format(&path)?;
        match detected {
            ExpertFormat::Q4_1 | ExpertFormat::Q5_0 | ExpertFormat::Q5_1 | ExpertFormat::Q8_0 => {
                let format = match detected {
                    ExpertFormat::Q4_1 => LegacyQuantFormat::Q4_1,
                    ExpertFormat::Q5_0 => LegacyQuantFormat::Q5_0,
                    ExpertFormat::Q5_1 => LegacyQuantFormat::Q5_1,
                    ExpertFormat::Q8_0 => LegacyQuantFormat::Q8_0,
                    _ => unreachable!("detect_expert_format mudou após seleção legacy"),
                };
                let source = LegacyQuantSource::open_infer(&path, experts)?;
                if source.format() != format {
                    return Err(MoeRuntimeError::FormatMismatch {
                        source: source.format().name().into(),
                        block: format.name().into(),
                    });
                }
                if !source.has_router() {
                    return Err(MoeRuntimeError::Source(Q4LoadError::MissingTensor(
                        "blk.<layer>.ffn_gate_inp.weight".into(),
                    )));
                }
                let layers = source.layer_count();
                if layers == 0 {
                    return Err(MoeRuntimeError::Source(Q4LoadError::InvalidTensor(
                        "GGUF MoE sem camadas",
                    )));
                }
                let mut blocks = Vec::with_capacity(layers as usize);
                for layer in 0..layers {
                    blocks.push(MoeBlock::Legacy(
                        LegacyQuantMoeBlock::new(
                            Arc::clone(&cache),
                            &model,
                            layer,
                            top_k,
                            format,
                            source.hidden(),
                            source.intermediate(),
                        )
                        .map_err(MoeRuntimeError::Block)?,
                    ));
                }
                Ok(Self {
                    source: MoeSource::Legacy(source),
                    blocks,
                    cache: Arc::clone(&cache),
                    runtime_diagnostics: Mutex::new(MoeRuntimeDiagnostics::default()),
                    #[cfg(feature = "cuda")]
                    model: model.clone(),
                    #[cfg(feature = "cuda")]
                    device_cache: None,
                    #[cfg(feature = "cuda")]
                    warmup: Arc::new(Mutex::new(MoeWarmupState::new())),
                    #[cfg(feature = "cuda")]
                    warmup_thread: Arc::new(Mutex::new(None)),
                })
            }
            ExpertFormat::F32 | ExpertFormat::F16 | ExpertFormat::Bf16 => {
                let format = match detected {
                    ExpertFormat::F32 => PlainFormat::F32,
                    ExpertFormat::F16 => PlainFormat::F16,
                    ExpertFormat::Bf16 => PlainFormat::Bf16,
                    _ => unreachable!("detect_expert_format mudou após seleção escalar"),
                };
                let source = PlainSource::open_infer(&path, experts)?;
                if source.format() != format {
                    return Err(MoeRuntimeError::FormatMismatch {
                        source: source.format().name().into(),
                        block: format.name().into(),
                    });
                }
                if !source.has_router() {
                    return Err(MoeRuntimeError::Source(Q4LoadError::MissingTensor(
                        "blk.<layer>.ffn_gate_inp.weight".into(),
                    )));
                }
                let layers = source.layer_count();
                if layers == 0 {
                    return Err(MoeRuntimeError::Source(Q4LoadError::InvalidTensor(
                        "GGUF MoE sem camadas",
                    )));
                }
                #[cfg(feature = "cuda")]
                let device_cache = Some(create_measured_device_cache(cuda_compute_bytes)?);
                let mut blocks = Vec::with_capacity(layers as usize);
                for layer in 0..layers {
                    #[cfg(feature = "cuda")]
                    let block = PlainMoeBlock::new_with_device_cache(
                        Arc::clone(&cache),
                        Arc::clone(device_cache.as_ref().expect("cache CUDA escalar")),
                        &model,
                        layer,
                        top_k,
                        format,
                        source.hidden(),
                        source.intermediate(),
                    );
                    #[cfg(not(feature = "cuda"))]
                    let block = PlainMoeBlock::new(
                        Arc::clone(&cache),
                        &model,
                        layer,
                        top_k,
                        format,
                        source.hidden(),
                        source.intermediate(),
                    );
                    blocks.push(MoeBlock::Plain(block.map_err(MoeRuntimeError::Block)?));
                }
                Ok(Self {
                    source: MoeSource::Plain(source),
                    blocks,
                    cache: Arc::clone(&cache),
                    runtime_diagnostics: Mutex::new(MoeRuntimeDiagnostics::default()),
                    #[cfg(feature = "cuda")]
                    model: model.clone(),
                    #[cfg(feature = "cuda")]
                    device_cache: device_cache.clone(),
                    #[cfg(feature = "cuda")]
                    warmup: Arc::new(Mutex::new(MoeWarmupState::new())),
                    #[cfg(feature = "cuda")]
                    warmup_thread: Arc::new(Mutex::new(None)),
                })
            }
            ExpertFormat::Q4_0 => {
                let source = Q4_0Source::open_infer(&path, experts)?;
                if !source.has_router() {
                    return Err(MoeRuntimeError::Source(Q4LoadError::MissingTensor(
                        "blk.<layer>.ffn_gate_inp.weight".into(),
                    )));
                }
                let layers = source.layer_count();
                if layers == 0 {
                    return Err(MoeRuntimeError::Source(Q4LoadError::InvalidTensor(
                        "GGUF MoE sem camadas",
                    )));
                }
                #[cfg(feature = "cuda")]
                let device_cache = Some(create_measured_device_cache(cuda_compute_bytes)?);
                let mut blocks = Vec::with_capacity(layers as usize);
                for layer in 0..layers {
                    #[cfg(feature = "cuda")]
                    let block = Q4_0MoeBlock::new_with_device_cache(
                        Arc::clone(&cache),
                        Arc::clone(device_cache.as_ref().expect("cache CUDA Q4_0")),
                        &model,
                        layer,
                        top_k,
                        source.hidden(),
                        source.intermediate(),
                    );
                    #[cfg(not(feature = "cuda"))]
                    let block = Q4_0MoeBlock::new(
                        Arc::clone(&cache),
                        &model,
                        layer,
                        top_k,
                        source.hidden(),
                        source.intermediate(),
                    );
                    #[cfg(feature = "cuda")]
                    let block = block.map_err(MoeRuntimeError::Block)?;
                    #[cfg(not(feature = "cuda"))]
                    let block = block?;
                    blocks.push(MoeBlock::Q4(block));
                }
                Ok(Self {
                    source: MoeSource::Q4(source),
                    blocks,
                    cache: Arc::clone(&cache),
                    runtime_diagnostics: Mutex::new(MoeRuntimeDiagnostics::default()),
                    #[cfg(feature = "cuda")]
                    model: model.clone(),
                    #[cfg(feature = "cuda")]
                    device_cache: device_cache.clone(),
                    #[cfg(feature = "cuda")]
                    warmup: Arc::new(Mutex::new(MoeWarmupState::new())),
                    #[cfg(feature = "cuda")]
                    warmup_thread: Arc::new(Mutex::new(None)),
                })
            }
            ExpertFormat::Nvfp4 => {
                let source = Nvfp4Source::open_infer(&path, experts)?;
                if !source.has_router() {
                    return Err(MoeRuntimeError::Source(Q4LoadError::MissingTensor(
                        "blk.<layer>.ffn_gate_inp.weight".into(),
                    )));
                }
                let layers = source.layer_count();
                if layers == 0 {
                    return Err(MoeRuntimeError::Source(Q4LoadError::InvalidTensor(
                        "GGUF MoE sem camadas",
                    )));
                }
                #[cfg(feature = "cuda")]
                let cuda_profile =
                    CudaTransferProfile::load_for_model(&path).map_err(MoeRuntimeError::Block)?;
                #[cfg(feature = "cuda")]
                let profile_requests_cpu = cuda_profile
                    .as_ref()
                    .is_some_and(|profile| !profile.prefers_cuda());
                #[cfg(feature = "cuda")]
                let coexecution_profile = cuda_profile.and_then(|profile| profile.bandwidth);
                #[cfg(feature = "cuda")]
                let device_cache = if std::env::var("POM_MOE_NVFP4_BACKEND")
                    .is_ok_and(|value| value.eq_ignore_ascii_case("cpu"))
                    || profile_requests_cpu
                {
                    None
                } else {
                    Some(create_measured_device_cache(cuda_compute_bytes)?)
                };
                let mut blocks = Vec::with_capacity(layers as usize);
                for layer in 0..layers {
                    #[cfg(feature = "cuda")]
                    let block = if let Some(device_cache) = device_cache.as_ref() {
                        Nvfp4MoeBlock::new_with_device_cache(
                            Arc::clone(&cache),
                            Arc::clone(device_cache),
                            &model,
                            layer,
                            top_k,
                            source.hidden(),
                            source.intermediate(),
                            coexecution_profile,
                        )
                    } else if profile_requests_cpu {
                        Nvfp4MoeBlock::new_cpu(
                            Arc::clone(&cache),
                            &model,
                            layer,
                            top_k,
                            source.hidden(),
                            source.intermediate(),
                        )
                    } else {
                        Nvfp4MoeBlock::new(
                            Arc::clone(&cache),
                            &model,
                            layer,
                            top_k,
                            source.hidden(),
                            source.intermediate(),
                        )
                    }
                    .map_err(MoeRuntimeError::Block)?;
                    #[cfg(not(feature = "cuda"))]
                    let block = Nvfp4MoeBlock::new(
                        Arc::clone(&cache),
                        &model,
                        layer,
                        top_k,
                        source.hidden(),
                        source.intermediate(),
                    )
                    .map_err(MoeRuntimeError::Block)?;
                    blocks.push(MoeBlock::Nvfp4(block));
                }
                Ok(Self {
                    source: MoeSource::Nvfp4(source),
                    blocks,
                    cache: Arc::clone(&cache),
                    runtime_diagnostics: Mutex::new(MoeRuntimeDiagnostics::default()),
                    #[cfg(feature = "cuda")]
                    model: model.clone(),
                    #[cfg(feature = "cuda")]
                    device_cache: device_cache.clone(),
                    #[cfg(feature = "cuda")]
                    warmup: Arc::new(Mutex::new(MoeWarmupState::new())),
                    #[cfg(feature = "cuda")]
                    warmup_thread: Arc::new(Mutex::new(None)),
                })
            }
            ExpertFormat::Mxfp4
            | ExpertFormat::Tq1_0
            | ExpertFormat::Tq2_0
            | ExpertFormat::Q1_0
            | ExpertFormat::Q2_0 => {
                let format = match detected {
                    ExpertFormat::Mxfp4 => LowBitFormat::Mxfp4,
                    ExpertFormat::Tq1_0 => LowBitFormat::Tq1_0,
                    ExpertFormat::Tq2_0 => LowBitFormat::Tq2_0,
                    ExpertFormat::Q1_0 => LowBitFormat::Q1_0,
                    ExpertFormat::Q2_0 => LowBitFormat::Q2_0,
                    _ => unreachable!("detect_expert_format mudou após seleção low-bit"),
                };
                let source = LowBitSource::open_infer(&path, experts)?;
                if source.format() != format {
                    return Err(MoeRuntimeError::FormatMismatch {
                        source: source.format().name().into(),
                        block: format.name().into(),
                    });
                }
                if !source.has_router() {
                    return Err(MoeRuntimeError::Source(Q4LoadError::MissingTensor(
                        "blk.<layer>.ffn_gate_inp.weight".into(),
                    )));
                }
                let layers = source.layer_count();
                if layers == 0 {
                    return Err(MoeRuntimeError::Source(Q4LoadError::InvalidTensor(
                        "GGUF MoE sem camadas",
                    )));
                }
                let mut blocks = Vec::with_capacity(layers as usize);
                for layer in 0..layers {
                    blocks.push(MoeBlock::LowBit(
                        LowBitMoeBlock::new(
                            Arc::clone(&cache),
                            &model,
                            layer,
                            top_k,
                            format,
                            source.hidden(),
                            source.intermediate(),
                        )
                        .map_err(MoeRuntimeError::Block)?,
                    ));
                }
                Ok(Self {
                    source: MoeSource::LowBit(source),
                    blocks,
                    cache: Arc::clone(&cache),
                    runtime_diagnostics: Mutex::new(MoeRuntimeDiagnostics::default()),
                    #[cfg(feature = "cuda")]
                    model: model.clone(),
                    #[cfg(feature = "cuda")]
                    device_cache: None,
                    #[cfg(feature = "cuda")]
                    warmup: Arc::new(Mutex::new(MoeWarmupState::new())),
                    #[cfg(feature = "cuda")]
                    warmup_thread: Arc::new(Mutex::new(None)),
                })
            }
            ExpertFormat::Q2K
            | ExpertFormat::Q3K
            | ExpertFormat::Q4K
            | ExpertFormat::Q5K
            | ExpertFormat::Q6K
            | ExpertFormat::Q8K => {
                let format = match detected {
                    ExpertFormat::Q2K => KQuantFormat::Q2K,
                    ExpertFormat::Q3K => KQuantFormat::Q3K,
                    ExpertFormat::Q4K => KQuantFormat::Q4K,
                    ExpertFormat::Q5K => KQuantFormat::Q5K,
                    ExpertFormat::Q6K => KQuantFormat::Q6K,
                    ExpertFormat::Q8K => KQuantFormat::Q8K,
                    _ => unreachable!("detect_expert_format mudou após a seleção K-quant"),
                };
                let source = KQuantSource::open_infer(&path, experts)?;
                if source.format() != format {
                    return Err(MoeRuntimeError::FormatMismatch {
                        source: source.format().name().into(),
                        block: format.name().into(),
                    });
                }
                if !source.has_router() {
                    return Err(MoeRuntimeError::Source(Q4LoadError::MissingTensor(
                        "blk.<layer>.ffn_gate_inp.weight".into(),
                    )));
                }
                let layers = source.layer_count();
                if layers == 0 {
                    return Err(MoeRuntimeError::Source(Q4LoadError::InvalidTensor(
                        "GGUF MoE sem camadas",
                    )));
                }
                #[cfg(feature = "cuda")]
                let device_cache = if format.cuda_supported() {
                    Some(create_measured_device_cache(cuda_compute_bytes)?)
                } else {
                    None
                };
                let mut blocks = Vec::with_capacity(layers as usize);
                for layer in 0..layers {
                    #[cfg(feature = "cuda")]
                    let block = if let Some(device_cache) = device_cache.as_ref() {
                        super::block::KQuantMoeBlock::new_with_device_cache(
                            Arc::clone(&cache),
                            Arc::clone(device_cache),
                            &model,
                            layer,
                            top_k,
                            format,
                            source.hidden(),
                            source.intermediate(),
                        )
                    } else {
                        super::block::KQuantMoeBlock::new(
                            Arc::clone(&cache),
                            &model,
                            layer,
                            top_k,
                            format,
                            source.hidden(),
                            source.intermediate(),
                        )
                    };
                    #[cfg(not(feature = "cuda"))]
                    let block = super::block::KQuantMoeBlock::new(
                        Arc::clone(&cache),
                        &model,
                        layer,
                        top_k,
                        format,
                        source.hidden(),
                        source.intermediate(),
                    );
                    blocks.push(MoeBlock::K(
                        block.map_err(|error| MoeRuntimeError::Block(error.to_string()))?,
                    ));
                }
                Ok(Self {
                    source: MoeSource::K(source),
                    blocks,
                    cache: Arc::clone(&cache),
                    runtime_diagnostics: Mutex::new(MoeRuntimeDiagnostics::default()),
                    #[cfg(feature = "cuda")]
                    model: model.clone(),
                    #[cfg(feature = "cuda")]
                    device_cache: device_cache.clone(),
                    #[cfg(feature = "cuda")]
                    warmup: Arc::new(Mutex::new(MoeWarmupState::new())),
                    #[cfg(feature = "cuda")]
                    warmup_thread: Arc::new(Mutex::new(None)),
                })
            }
            ExpertFormat::Iq2XXS
            | ExpertFormat::Iq2XS
            | ExpertFormat::Iq3XXS
            | ExpertFormat::Iq1S
            | ExpertFormat::Iq4NL
            | ExpertFormat::Iq3S
            | ExpertFormat::Iq2S
            | ExpertFormat::Iq4XS
            | ExpertFormat::Iq1M => {
                let format = match detected {
                    ExpertFormat::Iq2XXS => IqQuantFormat::Iq2XXS,
                    ExpertFormat::Iq2XS => IqQuantFormat::Iq2XS,
                    ExpertFormat::Iq3XXS => IqQuantFormat::Iq3XXS,
                    ExpertFormat::Iq1S => IqQuantFormat::Iq1S,
                    ExpertFormat::Iq4NL => IqQuantFormat::Iq4NL,
                    ExpertFormat::Iq3S => IqQuantFormat::Iq3S,
                    ExpertFormat::Iq2S => IqQuantFormat::Iq2S,
                    ExpertFormat::Iq4XS => IqQuantFormat::Iq4XS,
                    ExpertFormat::Iq1M => IqQuantFormat::Iq1M,
                    _ => unreachable!("detect_expert_format mudou após a seleção IQ-quant"),
                };
                let source = IqQuantSource::open_infer(&path, experts)?;
                if source.format() != format {
                    return Err(MoeRuntimeError::FormatMismatch {
                        source: source.format().name().into(),
                        block: format.name().into(),
                    });
                }
                if !source.has_router() {
                    return Err(MoeRuntimeError::Source(Q4LoadError::MissingTensor(
                        "blk.<layer>.ffn_gate_inp.weight".into(),
                    )));
                }
                let layers = source.layer_count();
                if layers == 0 {
                    return Err(MoeRuntimeError::Source(Q4LoadError::InvalidTensor(
                        "GGUF MoE sem camadas",
                    )));
                }
                #[cfg(feature = "cuda")]
                let device_cache = if format.cuda_supported() {
                    Some(create_measured_device_cache(cuda_compute_bytes)?)
                } else {
                    None
                };
                let mut blocks = Vec::with_capacity(layers as usize);
                for layer in 0..layers {
                    #[cfg(feature = "cuda")]
                    let block = if let Some(device_cache) = device_cache.as_ref() {
                        IqQuantMoeBlock::new_with_device_cache(
                            Arc::clone(&cache),
                            Arc::clone(device_cache),
                            &model,
                            layer,
                            top_k,
                            format,
                            source.hidden(),
                            source.intermediate(),
                        )
                    } else {
                        IqQuantMoeBlock::new(
                            Arc::clone(&cache),
                            &model,
                            layer,
                            top_k,
                            format,
                            source.hidden(),
                            source.intermediate(),
                        )
                    };
                    #[cfg(not(feature = "cuda"))]
                    let block = IqQuantMoeBlock::new(
                        Arc::clone(&cache),
                        &model,
                        layer,
                        top_k,
                        format,
                        source.hidden(),
                        source.intermediate(),
                    );
                    blocks.push(MoeBlock::Iq(
                        block.map_err(|error| MoeRuntimeError::Block(error.to_string()))?,
                    ));
                }
                Ok(Self {
                    source: MoeSource::Iq(source),
                    blocks,
                    cache: Arc::clone(&cache),
                    runtime_diagnostics: Mutex::new(MoeRuntimeDiagnostics::default()),
                    #[cfg(feature = "cuda")]
                    model: model.clone(),
                    #[cfg(feature = "cuda")]
                    device_cache: device_cache.clone(),
                    #[cfg(feature = "cuda")]
                    warmup: Arc::new(Mutex::new(MoeWarmupState::new())),
                    #[cfg(feature = "cuda")]
                    warmup_thread: Arc::new(Mutex::new(None)),
                })
            }
            ExpertFormat::Heterogeneous => {
                let source = HeterogeneousMoeSource::open_infer(&path, experts)?;
                let layers = source.layer_count();
                if layers == 0 {
                    return Err(MoeRuntimeError::Source(Q4LoadError::InvalidTensor(
                        "GGUF MoE sem camadas",
                    )));
                }
                #[cfg(feature = "cuda")]
                let device_cache = if source
                    .dtypes()
                    .iter()
                    .all(|dtype| cuda_heterogeneous_dtype_supported(*dtype))
                {
                    Some(create_measured_device_cache(cuda_compute_bytes)?)
                } else {
                    None
                };
                #[cfg(feature = "cuda")]
                let cuda_executor = device_cache.as_ref().map(|device_cache| {
                    Arc::new(HeterogeneousCudaExecutor::new_with_cache(
                        0,
                        0,
                        Arc::clone(device_cache),
                    ))
                });
                #[cfg(feature = "cuda")]
                let coexecution_profile = CudaTransferProfile::load_for_model(&path)
                    .map_err(MoeRuntimeError::Block)?
                    .and_then(|profile| profile.bandwidth);
                let mut blocks = Vec::with_capacity(layers as usize);
                for layer in 0..layers {
                    #[cfg(feature = "cuda")]
                    let block = if let Some(cuda_executor) = cuda_executor.as_ref() {
                        HeterogeneousMoeBlock::new_with_device_executor(
                            Arc::clone(&cache),
                            Arc::clone(cuda_executor),
                            &model,
                            layer,
                            top_k,
                            coexecution_profile,
                        )
                    } else {
                        HeterogeneousMoeBlock::new(Arc::clone(&cache), &model, layer, top_k)
                    };
                    #[cfg(not(feature = "cuda"))]
                    let block =
                        HeterogeneousMoeBlock::new(Arc::clone(&cache), &model, layer, top_k);
                    blocks.push(MoeBlock::Heterogeneous(block));
                }
                Ok(Self {
                    source: MoeSource::Heterogeneous(source),
                    blocks,
                    cache: Arc::clone(&cache),
                    runtime_diagnostics: Mutex::new(MoeRuntimeDiagnostics::default()),
                    #[cfg(feature = "cuda")]
                    model: model.clone(),
                    #[cfg(feature = "cuda")]
                    device_cache: device_cache.clone(),
                    #[cfg(feature = "cuda")]
                    warmup: Arc::new(Mutex::new(MoeWarmupState::new())),
                    #[cfg(feature = "cuda")]
                    warmup_thread: Arc::new(Mutex::new(None)),
                })
            }
        }
    }

    pub fn layers(&self) -> u32 {
        self.blocks.len() as u32
    }

    #[cfg(feature = "cuda")]
    fn warmup_enabled() -> bool {
        // O carregamento inicial deve deixar a VRAM aquecida, como no caminho
        // FreeToken. O orçamento medido ainda limita quantos experts cabem e
        // a primeira rota cancela o restante antes de executar. O operador
        // pode desligar explicitamente o aquecimento para uma medição fria.
        !matches!(
            std::env::var("POM_MOE_WARMUP")
                .ok()
                .map(|value| value.trim().to_ascii_lowercase()),
            Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "no")
        )
    }

    #[cfg(feature = "cuda")]
    fn warmup_max_duration() -> Duration {
        const DEFAULT_MAX_MS: u64 = 30_000;
        match std::env::var("POM_MOE_WARMUP_MAX_MS") {
            Ok(value) => match value.parse::<u64>() {
                Ok(milliseconds) if milliseconds > 0 => Duration::from_millis(milliseconds),
                _ => {
                    tracing::warn!(
                        value = %value,
                        default_ms = DEFAULT_MAX_MS,
                        "POM_MOE_WARMUP_MAX_MS inválido; usando o prazo padrão"
                    );
                    Duration::from_millis(DEFAULT_MAX_MS)
                }
            },
            Err(std::env::VarError::NotPresent) => Duration::from_millis(DEFAULT_MAX_MS),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    default_ms = DEFAULT_MAX_MS,
                    "não foi possível ler POM_MOE_WARMUP_MAX_MS; usando o prazo padrão"
                );
                Duration::from_millis(DEFAULT_MAX_MS)
            }
        }
    }

    /// Inicia o preenchimento assíncrono do cache residente no device.
    ///
    /// O carregamento do modelo não espera o banco inteiro: a thread percorre
    /// a fonte fria até a reserva medida ser atingida, o prazo vencer ou o
    /// cancelamento ser solicitado. O estado fica nos diagnósticos do runtime
    /// e cada payload é admitido pela mesma política de memória usada no
    /// caminho quente. Se um preenchimento anterior ainda existir, ele é
    /// encerrado antes de iniciar outro para nunca haver duas fontes
    /// concorrendo pelo mesmo cache.
    #[cfg(feature = "cuda")]
    pub fn start_device_warmup(self: &Arc<Self>) {
        self.cancel_device_warmup();
        let Some(device_cache) = self.device_cache.clone() else {
            return;
        };
        if !Self::warmup_enabled() {
            tracing::debug!(
                model = %self.model,
                "aquecimento do cache MoE desabilitado explicitamente"
            );
            return;
        }
        let options = DeviceWarmupOptions::new(Self::warmup_max_duration());
        let cancellation = Arc::clone(&options.cancellation);
        {
            let mut state = self
                .warmup
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.cancellation = Some(Arc::clone(&cancellation));
            state.status = "running".to_owned();
            state.candidates = 0;
            state.admitted = 0;
            state.admitted_bytes = 0;
            state.resident_bytes = device_cache.stats().resident_bytes;
            state.elapsed_ms = 0.0;
            state.error = None;
        }

        tracing::info!(
            model = %self.model,
            max_ms = options.max_duration.as_millis() as u64,
            "aquecimento do cache MoE residente iniciado"
        );
        let runtime = Arc::clone(self);
        let options_for_thread = options.clone();
        let spawn = std::thread::Builder::new()
            .name("pom-moe-cache-warmup".to_owned())
            .spawn(move || {
                let candidates = DeviceWarmupCandidates::new(
                    &runtime.source,
                    &runtime.model,
                    Arc::clone(&options_for_thread.cancellation),
                    runtime.layers(),
                    runtime.source.experts(),
                );
                let report = device_cache.warm(candidates, &options_for_thread);
                let status = report.stop;
                let admitted_bytes = report.admitted_bytes;
                let remaining_reason = match status {
                    DeviceWarmupStop::BudgetExhausted => "o orçamento medido foi atingido",
                    DeviceWarmupStop::Cancelled => "o cancelamento foi solicitado",
                    DeviceWarmupStop::TimedOut => "o prazo configurado venceu",
                    DeviceWarmupStop::Completed => "todos os candidatos visitados foram aquecidos",
                    DeviceWarmupStop::Failed => "a fonte ou o upload falhou",
                };
                {
                    let mut state = runtime
                        .warmup
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    state.apply(report);
                }
                tracing::info!(
                    model = %runtime.model,
                    status = ?status,
                    admitted_bytes,
                    reason = remaining_reason,
                    "aquecimento do cache MoE residente terminou"
                );
            });
        match spawn {
            Ok(handle) => {
                *self
                    .warmup_thread
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(handle);
            }
            Err(error) => {
                let mut state = self
                    .warmup
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                state.status = "failed".to_owned();
                state.error = Some(format!(
                    "não foi possível iniciar a thread de aquecimento: {error}"
                ));
                state.cancellation = None;
                tracing::warn!(error = %error, "aquecimento do cache MoE não iniciou");
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn start_device_warmup(&self) {}

    /// Solicita o fim do aquecimento sem desalojar as entradas já admitidas.
    ///
    /// O caminho de inferência não pode aguardar a thread de warmup: se ela
    /// estiver em uma admissão CUDA, `join` transforma uma cópia de peso em
    /// latência de TTFT. A admissão corrente é serializada pelo shard e termina
    /// com segurança; a flag impede qualquer candidato seguinte. O handle só
    /// é aguardado por `cancel_device_warmup`, fora do caminho quente, quando o
    /// runtime vai iniciar outro aquecimento.
    #[cfg(feature = "cuda")]
    pub fn cancel_device_warmup(&self) {
        self.signal_device_warmup_cancel();
        let handle = self
            .warmup_thread
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(handle) = handle {
            if handle.thread().id() == std::thread::current().id() {
                tracing::error!("thread de aquecimento tentou aguardar a si própria");
            } else if handle.join().is_err() {
                tracing::warn!("thread de aquecimento terminou com panic");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cancel_device_warmup_for_forward(&self) {
        self.signal_device_warmup_cancel();
    }

    #[cfg(feature = "cuda")]
    fn signal_device_warmup_cancel(&self) {
        let cancellation = self
            .warmup
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .cancellation
            .clone();
        if let Some(cancellation) = cancellation {
            cancellation.store(true, Ordering::Release);
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cancel_device_warmup(&self) {}

    #[cfg(not(feature = "cuda"))]
    pub fn cancel_device_warmup_for_forward(&self) {}

    pub fn executor_kind(&self) -> MoeExecutorKind {
        match self.blocks.first() {
            Some(MoeBlock::Nvfp4(block)) => block.executor_kind().into(),
            #[cfg(feature = "cuda")]
            Some(MoeBlock::K(block)) if block.uses_cuda() => MoeExecutorKind::Cuda,
            #[cfg(feature = "cuda")]
            Some(MoeBlock::Iq(block)) if block.uses_cuda() => MoeExecutorKind::Cuda,
            #[cfg(feature = "cuda")]
            Some(MoeBlock::Plain(block)) if block.uses_cuda() => MoeExecutorKind::Cuda,
            #[cfg(feature = "cuda")]
            Some(MoeBlock::Q4(block)) if block.uses_cuda() => MoeExecutorKind::Cuda,
            #[cfg(feature = "cuda")]
            Some(MoeBlock::Heterogeneous(block)) if block.uses_cuda() => MoeExecutorKind::Cuda,
            Some(_) | None => MoeExecutorKind::CpuReference,
        }
    }

    fn executor_diagnostics(&self) -> MoeExecutorDiagnostics {
        let cache = self.cache.stats();
        let diagnostics = MoeExecutorDiagnostics {
            cache_hits: cache.hits,
            cache_misses: cache.misses,
            cache_admissions: cache.admissions,
            cache_evictions: cache.evictions,
            cache_rejections: cache.rejections,
            cache_entries: cache.entries,
            cache_resident_bytes: cache.resident_bytes,
            runtime_forward_ms: self
                .runtime_diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .runtime_forward_ms,
            warmup_status: "disabled".to_owned(),
            ..MoeExecutorDiagnostics::default()
        };

        #[cfg(feature = "cuda")]
        let mut diagnostics = diagnostics;

        #[cfg(feature = "cuda")]
        {
            let warmup = self
                .warmup
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone();
            diagnostics.warmup_status = if warmup.status.is_empty() {
                "disabled".to_owned()
            } else {
                warmup.status
            };
            diagnostics.warmup_candidates = warmup.candidates;
            diagnostics.warmup_admitted = warmup.admitted;
            diagnostics.warmup_admitted_bytes = warmup.admitted_bytes;
            diagnostics.warmup_resident_bytes = warmup.resident_bytes;
            diagnostics.warmup_elapsed_ms = warmup.elapsed_ms;
            diagnostics.warmup_error = warmup.error;

            let mut device_cache_recorded = false;
            for block in &self.blocks {
                let (cuda, device_cache, capacity_by_device, budget_by_device) = match block {
                    MoeBlock::Nvfp4(block) => (
                        block.cuda_diagnostics(),
                        block.resident_cache_stats(),
                        Vec::new(),
                        Vec::new(),
                    ),
                    MoeBlock::K(block) if block.uses_cuda() => (
                        block.cuda_diagnostics(),
                        block.resident_cache_stats(),
                        Vec::new(),
                        Vec::new(),
                    ),
                    MoeBlock::Iq(block) if block.uses_cuda() => (
                        block.cuda_diagnostics(),
                        block.resident_cache_stats(),
                        Vec::new(),
                        Vec::new(),
                    ),
                    MoeBlock::Plain(block) if block.uses_cuda() => (
                        block.cuda_diagnostics(),
                        block.resident_cache_stats(),
                        Vec::new(),
                        Vec::new(),
                    ),
                    MoeBlock::Q4(block) if block.uses_cuda() => (
                        block.cuda_diagnostics(),
                        block.resident_cache_stats(),
                        Vec::new(),
                        Vec::new(),
                    ),
                    MoeBlock::Heterogeneous(block) if block.uses_cuda() => (
                        block.cuda_diagnostics(),
                        block.resident_cache_stats(),
                        block.resident_cache_capacity_bytes_by_device(),
                        block.resident_cache_budget_snapshots_by_device(),
                    ),
                    _ => continue,
                };
                diagnostics.profiled |= cuda.calls > 0;
                diagnostics.calls = diagnostics.calls.saturating_add(cuda.calls);
                diagnostics.gpu_expert_calls =
                    diagnostics.gpu_expert_calls.saturating_add(cuda.calls);
                diagnostics.payload_validation_ms += cuda.payload_validation_ms;
                diagnostics.cuda_alloc_ms += cuda.alloc_ms;
                diagnostics.cuda_h2d_ms += cuda.h2d_ms;
                diagnostics.cuda_h2d_bytes =
                    diagnostics.cuda_h2d_bytes.saturating_add(cuda.h2d_bytes);
                diagnostics.cuda_kernel_ms += cuda.kernel_ms;
                diagnostics.cuda_kernel_span_ms += cuda.kernel_span_ms;
                diagnostics.cuda_launch_ms += cuda.launch_ms;
                diagnostics.cuda_sync_ms += cuda.sync_ms;
                diagnostics.cuda_d2h_ms += cuda.d2h_ms;
                diagnostics.cuda_d2h_bytes =
                    diagnostics.cuda_d2h_bytes.saturating_add(cuda.d2h_bytes);
                diagnostics.cuda_transfer_span_ms += cuda.transfer_span_ms;
                diagnostics.cuda_free_ms += cuda.free_ms;
                diagnostics.cuda_buffer_mask |= cuda.buffer_mask;
                diagnostics.cuda_buffer_switches = diagnostics
                    .cuda_buffer_switches
                    .saturating_add(cuda.buffer_switches);
                diagnostics.cuda_call_wall_ms += cuda.cuda_call_wall_ms;
                diagnostics.host_batch_prepare_ms += cuda.host_batch_prepare_ms;
                diagnostics.host_result_copy_ms += cuda.host_result_copy_ms;
                for device in cuda.device_calls {
                    if let Some(existing) = diagnostics
                        .cuda_device_calls
                        .iter_mut()
                        .find(|existing| existing.device_index == device.device_index)
                    {
                        existing.calls = existing.calls.saturating_add(device.calls);
                        existing.expert_calls =
                            existing.expert_calls.saturating_add(device.expert_calls);
                        existing.kernel_ms += device.kernel_ms;
                        existing.cuda_call_wall_ms += device.cuda_call_wall_ms;
                    } else {
                        diagnostics.cuda_device_calls.push(device);
                    }
                }
                if !device_cache_recorded {
                    diagnostics.device_cache_hits = device_cache.hits;
                    diagnostics.device_cache_misses = device_cache.misses;
                    diagnostics.device_cache_admissions = device_cache.admissions;
                    diagnostics.device_cache_evictions = device_cache.evictions;
                    diagnostics.device_cache_rejections = device_cache.rejections;
                    diagnostics.device_cache_uploads = device_cache.uploads;
                    diagnostics.device_cache_upload_bytes = device_cache.upload_bytes;
                    diagnostics.device_cache_upload_ms = device_cache.upload_ms;
                    diagnostics.device_cache_resident_bytes = device_cache.resident_bytes;
                    diagnostics.device_cache_devices = device_cache.device_names;
                    diagnostics.device_cache_resident_bytes_by_device =
                        device_cache.resident_bytes_by_device;
                    diagnostics.device_cache_entries_by_device = device_cache.entries_by_device;
                    diagnostics.device_cache_capacity_bytes_by_device = capacity_by_device;
                    diagnostics.device_cache_capacity_bytes = diagnostics
                        .device_cache_capacity_bytes_by_device
                        .iter()
                        .copied()
                        .try_fold(0u64, |total, capacity| total.checked_add(capacity?));
                    diagnostics.device_cache_budget_by_device = budget_by_device
                        .into_iter()
                        .map(|snapshot| {
                            snapshot.map(|snapshot| DeviceCacheBudgetDiagnostics {
                                baseline_free_bytes: snapshot.baseline_free_bytes,
                                compute_bytes: snapshot.compute_bytes,
                                weights_bytes: snapshot.weights_bytes,
                                fixed_bytes: snapshot.fixed_bytes,
                                reserve_bytes: snapshot.reserve_bytes,
                                current_free_bytes: snapshot.current_free_bytes,
                                resident_bytes: snapshot.resident_bytes,
                                baseline_limit_bytes: snapshot.baseline_limit_bytes,
                                dynamic_limit_bytes: snapshot.dynamic_limit_bytes,
                                effective_limit_bytes: snapshot.effective_limit_bytes,
                            })
                        })
                        .collect();
                    device_cache_recorded = true;
                }
                if let MoeBlock::Nvfp4(block) = block {
                    let host = block.host_diagnostics();
                    diagnostics.host_cache_lookup_ms += host.cache_lookup_ms;
                    diagnostics.host_expert_load_ms += host.expert_load_ms;
                    diagnostics.host_cache_admit_ms += host.cache_admit_ms;
                    diagnostics.host_aggregation_ms += host.aggregation_ms;
                    diagnostics.cpu_expert_calls = diagnostics
                        .cpu_expert_calls
                        .saturating_add(host.cpu_expert_calls);
                    diagnostics.coexecution_gpu_fraction = host.coexecution_gpu_fraction;
                }
                #[cfg(feature = "cuda")]
                if let MoeBlock::Heterogeneous(block) = block {
                    let host = block.host_diagnostics();
                    diagnostics.cpu_expert_calls = diagnostics
                        .cpu_expert_calls
                        .saturating_add(host.cpu_expert_calls);
                    diagnostics.coexecution_gpu_fraction = host.coexecution_gpu_fraction;
                    diagnostics
                        .cache_working_set_by_layer
                        .push(block.working_set_diagnostics());
                }
            }
            diagnostics
                .cuda_device_calls
                .sort_by_key(|device| device.device_index);
            diagnostics
                .cache_working_set_by_layer
                .sort_by_key(|layer| layer.layer);
        }

        diagnostics
    }

    pub fn executor_telemetry(&self) -> MoeExecutorTelemetry {
        let executor = self.executor_kind();
        let dtypes = self.source.dtypes();
        let reason = match executor {
            MoeExecutorKind::Cuda => None,
            MoeExecutorKind::CpuReference => {
                let unsupported = if matches!(&self.source, MoeSource::Heterogeneous(_))
                    && cfg!(feature = "cuda")
                {
                    dtypes
                        .iter()
                        .filter(|dtype| !cuda_heterogeneous_dtype_supported(**dtype))
                        .copied()
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                let reason_dtypes = if unsupported.is_empty() {
                    &dtypes
                } else {
                    &unsupported
                };
                let names = reason_dtypes
                    .iter()
                    .map(|dtype| {
                        format!("{} ({dtype})", super::format_q4_0::ggml_type_name(*dtype))
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let explicit_cpu = matches!(self.source, MoeSource::Nvfp4(_))
                    && std::env::var("POM_MOE_NVFP4_BACKEND")
                        .is_ok_and(|value| value.eq_ignore_ascii_case("cpu"));
                if explicit_cpu {
                    Some(format!(
                        "dtype(s) {names}: CPUReference solicitado por POM_MOE_NVFP4_BACKEND=cpu"
                    ))
                } else {
                    Some(format!(
                        "dtype(s) {names} sem ponte CUDA; CPUReference selecionado"
                    ))
                }
            }
        };
        MoeExecutorTelemetry {
            executor,
            dtypes,
            reason,
            diagnostics: Some(self.executor_diagnostics()),
        }
    }

    pub fn validate_callback_metadata(
        &self,
        layer: u32,
        metadata: &[(TensorRole, &str, u32, u32, [i64; 4])],
    ) -> Result<(), MoeRuntimeError> {
        if let MoeSource::Heterogeneous(source) = &self.source {
            return source
                .validate_metadata(layer, metadata)
                .map_err(MoeRuntimeError::Source);
        }
        let expected = self.source.dtype();
        let (hidden, intermediate) = self.source.geometry(layer)?;
        let experts = self.source.experts() as u64;
        for (role, name, dtype, n_dims, dims) in metadata {
            if *dtype != expected {
                return Err(MoeRuntimeError::Block(format!(
                    "tensor {name} ({role}) usa dtype GGML {dtype}, mas o runtime homogêneo abriu dtype GGML {expected}"
                )));
            }
            let actual_dims = usize::try_from(*n_dims)
                .ok()
                .filter(|count| *count <= dims.len())
                .map_or_else(Vec::new, |count| dims[..count].to_vec());
            let expected_dims = match role {
                TensorRole::GateUp => vec![hidden as u64, (2 * intermediate) as u64, experts],
                TensorRole::Gate | TensorRole::Up => {
                    vec![hidden as u64, intermediate as u64, experts]
                }
                TensorRole::Down => vec![intermediate as u64, hidden as u64, experts],
            };
            let geometry_matches = actual_dims.len() == expected_dims.len()
                && actual_dims
                    .iter()
                    .zip(&expected_dims)
                    .all(|(actual, expected)| u64::try_from(*actual).ok() == Some(*expected));
            if !geometry_matches {
                return Err(MoeRuntimeError::Source(
                    Q4LoadError::TensorGeometryMismatch {
                        name: (*name).to_string(),
                        dtype: *dtype,
                        dims: actual_dims,
                        expected_dims,
                    },
                ));
            }
        }
        Ok(())
    }

    pub fn forward_batch_with_route(
        &self,
        route: MoeRouteBatch<'_>,
    ) -> Result<Vec<f32>, MoeRuntimeError> {
        let runtime_started = std::time::Instant::now();
        let MoeRouteBatch {
            layer,
            hidden,
            selected_experts,
            weights,
            tokens,
            expert_count,
            expert_count_used,
            activation,
            weight_before_ffn,
        } = route;
        if weight_before_ffn {
            return Err(MoeRuntimeError::Block(
                "hook MoE não suporta pesos aplicados antes do FFN".into(),
            ));
        }
        let expected_ids =
            tokens
                .checked_mul(expert_count_used)
                .ok_or(MoeRuntimeError::InvalidShape {
                    values: selected_experts.len(),
                    tokens,
                    hidden: expert_count_used,
                })?;
        if expert_count != self.source.experts() as usize
            || expert_count_used == 0
            || expert_count_used > expert_count
            || hidden.len() != tokens.saturating_mul(self.source.hidden())
            || selected_experts.len() != expected_ids
            || weights.len() != expected_ids
        {
            return Err(MoeRuntimeError::InvalidRoute {
                layer,
                source: format!("{} dtypes={:?}", self.source.kind(), self.source.dtypes()),
                expert_count,
                expected_expert_count: self.source.experts() as usize,
                expert_count_used,
                hidden_values: hidden.len(),
                expected_hidden_values: tokens.saturating_mul(self.source.hidden()),
                selected_values: selected_experts.len(),
                expected_selected_values: expected_ids,
                weight_values: weights.len(),
                expected_weight_values: expected_ids,
            });
        }
        if let Some(expected_activation) = self.source.activation() {
            if activation != expected_activation {
                return Err(MoeRuntimeError::Block(format!(
                    "ativação MoE não suportada para dtype {}: {} (esperado {})",
                    self.source.dtype(),
                    activation,
                    expected_activation
                )));
            }
        }

        // O aquecimento é oportunista. Quando chega uma requisição real, ele
        // precisa ceder o cache e a GPU para o working set dessa requisição;
        // caso contrário, a thread de fundo pode continuar admitindo experts
        // de camadas futuras e expulsar os experts usados no turno anterior.
        #[cfg(feature = "cuda")]
        self.cancel_device_warmup_for_forward();

        log_moe_layer_input_diagnostic(
            &self.source,
            layer,
            hidden,
            selected_experts,
            weights,
            tokens,
            expert_count_used,
            weight_before_ffn,
        );

        let resolved_ids = selected_experts
            .iter()
            .map(|&expert| {
                u32::try_from(expert).map_err(|_| {
                    MoeRuntimeError::Block(format!(
                        "id de expert negativo no runtime MoE (dtype {})",
                        self.source.dtype()
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let block = self
            .blocks
            .get(layer as usize)
            .ok_or(MoeRuntimeError::MissingLayer {
                layer,
                layers: self.layers(),
            })?;

        #[cfg(feature = "cuda")]
        if let (MoeSource::Nvfp4(source), MoeBlock::Nvfp4(block)) = (&self.source, block) {
            if block.executor_kind() == super::format_nvfp4::Nvfp4BackendKind::Cuda {
                let token_ids = (0..tokens)
                    .flat_map(|token| std::iter::repeat(token as u32).take(expert_count_used))
                    .collect::<Vec<_>>();
                let token_offsets = (0..=tokens)
                    .map(|token| token.saturating_mul(expert_count_used))
                    .collect::<Vec<_>>();
                let output = block.forward_batch_from_source_with_route(
                    source,
                    hidden,
                    &resolved_ids,
                    &token_ids,
                    weights,
                    &token_offsets,
                    tokens,
                )?;
                let elapsed_ms = runtime_started.elapsed().as_secs_f64() * 1_000.0;
                self.runtime_diagnostics
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .runtime_forward_ms += elapsed_ms;
                return Ok(output);
            }
        }

        #[cfg(feature = "cuda")]
        if let (MoeSource::Q4(source), MoeBlock::Q4(block)) = (&self.source, block) {
            if block.uses_cuda() {
                let token_ids = (0..tokens)
                    .flat_map(|token| std::iter::repeat(token as u32).take(expert_count_used))
                    .collect::<Vec<_>>();
                let token_offsets = (0..=tokens)
                    .map(|token| token.saturating_mul(expert_count_used))
                    .collect::<Vec<_>>();
                let output = block.forward_batch_from_source_with_route(
                    source,
                    &resolved_ids,
                    &token_ids,
                    weights,
                    &token_offsets,
                    hidden,
                    tokens,
                )?;
                let elapsed_ms = runtime_started.elapsed().as_secs_f64() * 1_000.0;
                self.runtime_diagnostics
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .runtime_forward_ms += elapsed_ms;
                return Ok(output);
            }
        }

        #[cfg(feature = "cuda")]
        if let (MoeSource::K(source), MoeBlock::K(block)) = (&self.source, block) {
            if block.uses_cuda() {
                let token_ids = (0..tokens)
                    .flat_map(|token| std::iter::repeat(token as u32).take(expert_count_used))
                    .collect::<Vec<_>>();
                let token_offsets = (0..=tokens)
                    .map(|token| token.saturating_mul(expert_count_used))
                    .collect::<Vec<_>>();
                let output = match block.forward_batch_from_source_with_route(
                    source,
                    hidden,
                    &resolved_ids,
                    &token_ids,
                    weights,
                    &token_offsets,
                    tokens,
                ) {
                    Ok(output) => output,
                    Err(MoeBlockError::CudaNonFinite(error)) => {
                        tracing::error!(
                            layer,
                            model = %self.model,
                            source = self.source.kind(),
                            dtypes = ?self.source.dtypes(),
                            tokens,
                            experts = expert_count_used,
                            hidden_width = self.source.hidden(),
                            error = %error,
                            "CUDA MoE devolveu valor não finito; iniciando replay CPU"
                        );
                        accept_cpu_replay_after_cuda_nonfinite(
                            &error,
                            block.forward_batch_from_source_with_route_cpu(
                                source,
                                &resolved_ids,
                                &token_ids,
                                weights,
                                &token_offsets,
                                hidden,
                                tokens,
                            )?,
                        )?
                    }
                    Err(error) => return Err(error.into()),
                };
                let elapsed_ms = runtime_started.elapsed().as_secs_f64() * 1_000.0;
                self.runtime_diagnostics
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .runtime_forward_ms += elapsed_ms;
                return Ok(output);
            }
        }

        #[cfg(feature = "cuda")]
        if let (MoeSource::Iq(source), MoeBlock::Iq(block)) = (&self.source, block) {
            if block.uses_cuda() {
                let token_ids = (0..tokens)
                    .flat_map(|token| std::iter::repeat(token as u32).take(expert_count_used))
                    .collect::<Vec<_>>();
                let token_offsets = (0..=tokens)
                    .map(|token| token.saturating_mul(expert_count_used))
                    .collect::<Vec<_>>();
                let output = block.forward_batch_from_source_with_route(
                    source,
                    hidden,
                    &resolved_ids,
                    &token_ids,
                    weights,
                    &token_offsets,
                    tokens,
                )?;
                let elapsed_ms = runtime_started.elapsed().as_secs_f64() * 1_000.0;
                self.runtime_diagnostics
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .runtime_forward_ms += elapsed_ms;
                return Ok(output);
            }
        }

        #[cfg(feature = "cuda")]
        if let (MoeSource::Plain(source), MoeBlock::Plain(block)) = (&self.source, block) {
            if block.uses_cuda() {
                let token_ids = (0..tokens)
                    .flat_map(|token| std::iter::repeat(token as u32).take(expert_count_used))
                    .collect::<Vec<_>>();
                let token_offsets = (0..=tokens)
                    .map(|token| token.saturating_mul(expert_count_used))
                    .collect::<Vec<_>>();
                let output = block.forward_batch_from_source_with_route(
                    source,
                    hidden,
                    &resolved_ids,
                    &token_ids,
                    weights,
                    &token_offsets,
                    tokens,
                )?;
                let elapsed_ms = runtime_started.elapsed().as_secs_f64() * 1_000.0;
                self.runtime_diagnostics
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .runtime_forward_ms += elapsed_ms;
                return Ok(output);
            }
        }

        #[cfg(feature = "cuda")]
        if let (MoeSource::Heterogeneous(source), MoeBlock::Heterogeneous(block)) =
            (&self.source, block)
        {
            if block.uses_cuda() {
                let token_ids = (0..tokens)
                    .flat_map(|token| std::iter::repeat(token as u32).take(expert_count_used))
                    .collect::<Vec<_>>();
                let token_offsets = (0..=tokens)
                    .map(|token| token.saturating_mul(expert_count_used))
                    .collect::<Vec<_>>();
                let output = match block.forward_batch_from_source_with_route(
                    source,
                    &resolved_ids,
                    &token_ids,
                    weights,
                    &token_offsets,
                    hidden,
                    tokens,
                    activation,
                ) {
                    Ok(output) => output,
                    Err(MoeBlockError::CudaNonFinite(error)) => {
                        tracing::error!(
                            layer,
                            model = %self.model,
                            source = self.source.kind(),
                            dtypes = ?self.source.dtypes(),
                            tokens,
                            experts = expert_count_used,
                            hidden_width = self.source.hidden(),
                            error = %error,
                            "CUDA MoE heterogêneo devolveu valor não finito; iniciando replay CPU"
                        );
                        accept_cpu_replay_after_cuda_nonfinite(
                            &error,
                            block.forward_batch_from_source_with_route_cpu(
                                source,
                                &resolved_ids,
                                &token_ids,
                                weights,
                                &token_offsets,
                                hidden,
                                tokens,
                                activation,
                            )?,
                        )?
                    }
                    Err(error) => return Err(error.into()),
                };
                let elapsed_ms = runtime_started.elapsed().as_secs_f64() * 1_000.0;
                self.runtime_diagnostics
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .runtime_forward_ms += elapsed_ms;
                return Ok(output);
            }
        }

        if let (MoeSource::Heterogeneous(source), MoeBlock::Heterogeneous(block)) =
            (&self.source, block)
        {
            let token_ids = (0..tokens)
                .flat_map(|token| std::iter::repeat(token as u32).take(expert_count_used))
                .collect::<Vec<_>>();
            let token_offsets = (0..=tokens)
                .map(|token| token.saturating_mul(expert_count_used))
                .collect::<Vec<_>>();
            let output = block.forward_batch_from_source_with_route_cpu(
                source,
                &resolved_ids,
                &token_ids,
                weights,
                &token_offsets,
                hidden,
                tokens,
                activation,
            )?;
            let elapsed_ms = runtime_started.elapsed().as_secs_f64() * 1_000.0;
            self.runtime_diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .runtime_forward_ms += elapsed_ms;
            return Ok(output);
        }

        let hidden_width = self.source.hidden();
        let mut output = Vec::with_capacity(hidden.len());
        for token in 0..tokens {
            let token_hidden = &hidden[token * hidden_width..(token + 1) * hidden_width];
            let route_start = token * expert_count_used;
            let ids = &resolved_ids[route_start..route_start + expert_count_used];
            let route_weights = &weights[route_start..route_start + expert_count_used];
            let values = match (&self.source, block) {
                (MoeSource::Legacy(source), MoeBlock::Legacy(block)) => block
                    .forward_from_source_with_route(source, token_hidden, &ids, route_weights)?,
                (MoeSource::Plain(source), MoeBlock::Plain(block)) => block
                    .forward_from_source_with_route(source, token_hidden, &ids, route_weights)?,
                (MoeSource::Q4(source), MoeBlock::Q4(block)) => block
                    .forward_from_source_with_route(source, token_hidden, &ids, route_weights)?,
                (MoeSource::K(source), MoeBlock::K(block)) => block
                    .forward_from_source_with_route(source, token_hidden, &ids, route_weights)?,
                (MoeSource::Iq(source), MoeBlock::Iq(block)) => block
                    .forward_from_source_with_route(source, token_hidden, &ids, route_weights)?,
                (MoeSource::Nvfp4(source), MoeBlock::Nvfp4(block)) => block
                    .forward_from_source_with_route(source, token_hidden, &ids, route_weights)?,
                (MoeSource::LowBit(source), MoeBlock::LowBit(block)) => block
                    .forward_from_source_with_route(source, token_hidden, &ids, route_weights)?,
                (MoeSource::Heterogeneous(source), MoeBlock::Heterogeneous(block)) => block
                    .forward_from_source_with_route(
                        source,
                        token_hidden,
                        &ids,
                        route_weights,
                        activation,
                    )?,
                _ => {
                    return Err(MoeRuntimeError::FormatMismatch {
                        source: self.source.kind().into(),
                        block: block.kind().into(),
                    });
                }
            };
            output.extend(values);
        }
        let elapsed_ms = runtime_started.elapsed().as_secs_f64() * 1_000.0;
        self.runtime_diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .runtime_forward_ms += elapsed_ms;
        Ok(output)
    }

    pub fn forward_batch(
        &self,
        layer: u32,
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, MoeRuntimeError> {
        let block = self
            .blocks
            .get(layer as usize)
            .ok_or(MoeRuntimeError::MissingLayer {
                layer,
                layers: self.layers(),
            })?;
        let hidden_width = self.source.hidden();
        let expected = tokens
            .checked_mul(hidden_width)
            .ok_or(MoeRuntimeError::InvalidShape {
                values: hidden.len(),
                tokens,
                hidden: hidden_width,
            })?;
        if tokens == 0 || hidden.len() != expected {
            return Err(MoeRuntimeError::InvalidShape {
                values: hidden.len(),
                tokens,
                hidden: hidden_width,
            });
        }
        let mut output = Vec::with_capacity(hidden.len());
        for token_hidden in hidden.chunks_exact(hidden_width) {
            let values = match (&self.source, block) {
                (MoeSource::Legacy(source), MoeBlock::Legacy(block)) => {
                    block.forward_from_source(source, token_hidden)?
                }
                (MoeSource::Plain(source), MoeBlock::Plain(block)) => {
                    block.forward_from_source(source, token_hidden)?
                }
                (MoeSource::Q4(source), MoeBlock::Q4(block)) => {
                    block.forward_from_source(source, token_hidden)?
                }
                (MoeSource::K(source), MoeBlock::K(block)) => {
                    block.forward_from_source(source, token_hidden)?
                }
                (MoeSource::Iq(source), MoeBlock::Iq(block)) => {
                    block.forward_from_source(source, token_hidden)?
                }
                (MoeSource::Nvfp4(source), MoeBlock::Nvfp4(block)) => {
                    block.forward_from_source(source, token_hidden)?
                }
                (MoeSource::LowBit(source), MoeBlock::LowBit(block)) => {
                    block.forward_from_source(source, token_hidden)?
                }
                (MoeSource::Heterogeneous(_), MoeBlock::Heterogeneous(_)) => {
                    return Err(MoeRuntimeError::Block(
                        "runtime heterogêneo exige a rota resolvida do llama.cpp".into(),
                    ))
                }
                _ => {
                    return Err(MoeRuntimeError::FormatMismatch {
                        source: self.source.kind().into(),
                        block: block.kind().into(),
                    });
                }
            };
            output.extend(values);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::super::format_iq_quants::{tests::write_iq_source, IqQuantFormat};
    use super::super::format_k_quants::tests::write_k_source;
    use super::super::format_legacy_quants::{tests::write_legacy_source, LegacyQuantFormat};
    use super::super::format_low_bits::{tests::write_low_bit_source, LowBitFormat};
    use super::super::format_plain::{tests::write_plain_source, PlainFormat};
    use super::super::format_q4_0::tests::write_q4_source;
    use super::*;

    #[test]
    fn warmup_spreads_candidates_across_all_layers() {
        let mut layer = 0;
        let mut expert = 0;
        let positions = (0..6)
            .filter_map(|_| next_warmup_position(&mut layer, &mut expert, 3, 2))
            .collect::<Vec<_>>();
        assert_eq!(
            positions,
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        );
    }

    #[test]
    fn cuda_executor_refuses_a_report_without_measured_compute() {
        let error = require_cuda_compute_bytes(None).unwrap_err();
        assert!(error.to_string().contains("compute buffer CUDA medido"));
    }

    #[test]
    fn non_finite_diagnostic_distinguishes_first_index_and_count() {
        let summary = nonfinite_summary(&[1.0, f32::NAN, 2.0, f32::INFINITY]).unwrap();
        assert_eq!(summary.0, 1);
        assert_eq!(summary.1, 2);
        assert!(summary.2.is_nan());
        assert!(nonfinite_summary(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn layer_input_diagnostic_records_nan_and_the_finite_range() {
        let summary = summarize_layer_input(&[2.0, f32::NAN, -3.5, f32::INFINITY]);

        assert_eq!(summary.elements, 4);
        assert_eq!(summary.non_finite_count, 2);
        assert_eq!(summary.first_non_finite, Some(1));
        assert_eq!(summary.finite_min, Some(-3.5));
        assert_eq!(summary.finite_max, Some(2.0));
    }

    #[test]
    fn input_diagnostic_requires_an_explicit_gate() {
        assert!(!moe_layer_input_diagnostic_enabled(None));
        assert!(!moe_layer_input_diagnostic_enabled(Some("0")));
        assert!(!moe_layer_input_diagnostic_enabled(Some("false")));
        for value in ["1", "true", "TRUE", "yes", "on"] {
            assert!(moe_layer_input_diagnostic_enabled(Some(value)), "{value}");
        }
    }

    #[test]
    fn a_finite_cpu_replay_replaces_a_non_finite_cuda_output() {
        let output = vec![1.0, 2.0];

        assert_eq!(
            accept_cpu_replay_after_cuda_nonfinite("estágio agregação", output.clone()).unwrap(),
            output
        );
    }

    #[test]
    fn a_non_finite_cpu_replay_keeps_the_global_diagnostic() {
        let error = accept_cpu_replay_after_cuda_nonfinite(
            "estágio gate_up",
            vec![f32::NAN, 2.0, f32::INFINITY],
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("replay CPU"), "{error}");
        assert!(error.contains("2 valores não finitos"), "{error}");
        assert!(error.contains("primeiro índice 0"), "{error}");
        assert!(error.contains("NaN"), "{error}");
    }

    #[test]
    fn q4_0_with_a_cuda_bridge_uses_the_device_executor() {
        assert!(cuda_heterogeneous_dtype_supported(2));
        let plan = make_external_executor_plan(ExpertFormat::Q4_0, vec![2], 4, true, false);
        assert!(plan.enabled());
        assert_eq!(plan.layers, vec![1, 1, 1, 1]);
        assert!(plan.reason.is_none());
    }

    #[test]
    fn nvfp4_without_an_available_backend_is_refused_with_a_reason() {
        let plan = make_external_executor_plan(ExpertFormat::Nvfp4, vec![39], 2, false, false);
        assert!(!plan.enabled());
        assert_eq!(plan.layers, vec![0, 0]);
        assert!(plan
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("feature cuda")));

        let cpu_plan = make_external_executor_plan(ExpertFormat::Nvfp4, vec![39], 2, false, true);
        assert!(cpu_plan.enabled());
        assert_eq!(cpu_plan.layers, vec![1, 1]);
        assert!(cpu_plan.reason.is_none());
    }

    #[test]
    fn heterogeneous_dtypes_without_a_cuda_bridge_use_the_cpu_reference() {
        let plan = make_external_executor_plan(
            ExpertFormat::Heterogeneous,
            vec![12, 23, 34],
            3,
            true,
            false,
        );
        assert!(plan.enabled());
        assert_eq!(plan.layers, vec![1, 1, 1]);
        assert!(plan.reason.is_none());
    }

    #[test]
    fn a_heterogeneous_dtype_without_an_external_decoder_is_refused_before_loading() {
        let plan = make_external_executor_plan(
            ExpertFormat::Heterogeneous,
            vec![18, 39, 40],
            3,
            true,
            false,
        );

        assert!(!plan.enabled());
        assert_eq!(plan.layers, vec![0, 0, 0]);
        let reason = plan.reason.as_deref().unwrap();
        assert!(reason.contains("40 (NVFP4)"), "{reason}");
        assert!(reason.contains("antes do modelo"), "{reason}");
    }

    #[test]
    fn deepseek_heterogeneous_dtypes_have_a_cuda_bridge() {
        let plan =
            make_external_executor_plan(ExpertFormat::Heterogeneous, vec![18, 39], 14, true, false);
        assert!(plan.enabled());
        assert_eq!(plan.layers, vec![1; 14]);
        assert!(plan.reason.is_none());
    }

    #[test]
    fn heterogeneous_k_dtypes_with_a_bridge_externalize_all_layers() {
        let plan =
            make_external_executor_plan(ExpertFormat::Heterogeneous, vec![12, 14], 3, true, false);
        assert!(plan.enabled());
        assert_eq!(plan.layers, vec![1, 1, 1]);
        assert!(plan.reason.is_none());
    }

    #[test]
    fn q2k_q3k_and_q5k_enter_the_heterogeneous_cuda_bridge() {
        assert!(KQuantFormat::Q2K.cuda_supported());
        assert!(KQuantFormat::Q3K.cuda_supported());
        assert!(KQuantFormat::Q5K.cuda_supported());
        assert!(!KQuantFormat::Q8K.cuda_supported());
        assert!(IqQuantFormat::Iq3XXS.cuda_supported());
        assert!(cuda_heterogeneous_dtype_supported(10));
        assert!(cuda_heterogeneous_dtype_supported(11));
        assert!(cuda_heterogeneous_dtype_supported(18));
        assert!(cuda_heterogeneous_dtype_supported(13));
        let plan = make_external_executor_plan(
            ExpertFormat::Heterogeneous,
            vec![10, 11, 21, 22, 23],
            40,
            true,
            false,
        );
        assert!(plan.enabled());
        assert!(plan.reason.is_none());

        for (format, dtype) in [
            (ExpertFormat::Q2K, 10),
            (ExpertFormat::Q3K, 11),
            (ExpertFormat::Q5K, 13),
        ] {
            let plan = make_external_executor_plan(format, vec![dtype], 2, true, false);
            assert!(plan.enabled(), "{format:?}");
            assert!(plan.reason.is_none(), "{format:?}: {:?}", plan.reason);
        }
    }

    /// Forma medida do `Ornith-1.5-35B-A3B-Q5_K_S`: os experts misturam Q4_0 e
    /// Q5_K por papel, e um único dtype sem ponte derrubava a rota inteira para
    /// o `CPUReference`. Confere o dtype concreto, nunca o nome do ficheiro.
    #[test]
    fn q4_0_mixed_with_q5k_stays_on_the_cuda_bridge() {
        assert!(cuda_heterogeneous_dtype_supported(2));
        assert!(cuda_heterogeneous_dtype_supported(13));
        let plan =
            make_external_executor_plan(ExpertFormat::Heterogeneous, vec![2, 13], 40, true, false);
        assert!(plan.enabled());
        assert_eq!(plan.layers.len(), 40);
        assert!(plan.layers.iter().all(|layer| *layer == 1));
        assert!(plan.reason.is_none(), "{:?}", plan.reason);
    }

    #[test]
    fn heterogeneous_dtypes_with_iq_and_no_bridge_use_the_cpu_reference() {
        let plan = make_external_executor_plan(
            ExpertFormat::Heterogeneous,
            vec![16, 21, 23],
            3,
            true,
            false,
        );
        assert!(plan.enabled());
        assert_eq!(plan.layers, vec![1, 1, 1]);
        assert!(plan.reason.is_none());
    }

    #[test]
    fn a_dtype_with_a_device_bridge_externalizes_all_layers_of_the_plan() {
        let plan = make_external_executor_plan(ExpertFormat::Q4K, vec![12], 3, true, false);
        assert!(plan.enabled());
        assert_eq!(plan.layers, vec![1, 1, 1]);
        assert!(plan.reason.is_none());
    }

    #[test]
    fn scalars_with_a_device_bridge_externalize_all_layers() {
        for (format, dtype) in [
            (ExpertFormat::F32, 0),
            (ExpertFormat::F16, 1),
            (ExpertFormat::Bf16, 30),
        ] {
            let plan = make_external_executor_plan(format, vec![dtype], 2, true, false);
            assert!(plan.enabled(), "dtype {dtype} não foi externalizado");
            assert_eq!(plan.layers, vec![1, 1], "dtype {dtype}");
            assert!(plan.reason.is_none(), "dtype {dtype}");
        }
    }

    #[test]
    fn runtime_keeps_the_hidden_states_batch_shape() {
        let path =
            std::env::temp_dir().join(format!("pom-ftk-moe-runtime-{}.gguf", std::process::id()));
        let (hidden, _) = write_q4_source(&path);
        let plan = make_external_executor_plan(ExpertFormat::Q4_0, vec![2], 1, false, false);
        assert!(plan.enabled());
        assert_eq!(plan.layers, vec![1]);
        assert!(plan.reason.is_none());
        let runtime =
            MoeRuntime::open(&path, "fixture", 2, 1, 0, Some(vec![(0, 1_000_000)])).unwrap();
        assert_eq!(
            runtime.executor_telemetry().executor,
            MoeExecutorKind::CpuReference
        );
        let output = runtime.forward_batch(0, &[1.0; 64], 2).unwrap();
        assert_eq!(output.len(), hidden * 2);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(runtime.forward_batch(1, &[1.0; 32], 1).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn runtime_loads_and_runs_k_quants() {
        for format in [
            KQuantFormat::Q2K,
            KQuantFormat::Q3K,
            KQuantFormat::Q4K,
            KQuantFormat::Q5K,
            KQuantFormat::Q6K,
            KQuantFormat::Q8K,
        ] {
            let dtype = format.dtype();
            let expected = format.name();
            let path = std::env::temp_dir().join(format!(
                "pom-ftk-k-runtime-{dtype}-{}.gguf",
                std::process::id()
            ));
            write_k_source(&path, dtype, format);
            let runtime =
                MoeRuntime::open(&path, "fixture", 2, 1, 0, Some(vec![(0, 1_000_000)])).unwrap();
            assert_eq!(runtime.layers(), 1, "formato {expected}");
            let output = runtime.forward_batch(0, &[1.0; 512], 2).unwrap();
            assert_eq!(output.len(), 512, "formato {expected}");
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn runtime_loads_and_runs_iq_quants() {
        for format in [
            IqQuantFormat::Iq2XXS,
            IqQuantFormat::Iq2XS,
            IqQuantFormat::Iq3XXS,
            IqQuantFormat::Iq1S,
            IqQuantFormat::Iq4NL,
            IqQuantFormat::Iq3S,
            IqQuantFormat::Iq2S,
            IqQuantFormat::Iq4XS,
            IqQuantFormat::Iq1M,
        ] {
            let dtype = format.dtype();
            let path = std::env::temp_dir().join(format!(
                "pom-ftk-iq-runtime-{dtype}-{}.gguf",
                std::process::id()
            ));
            write_iq_source(&path, dtype, format, false);
            let runtime =
                MoeRuntime::open(&path, "fixture", 2, 1, 0, Some(vec![(0, 1_000_000)])).unwrap();
            assert_eq!(runtime.layers(), 1, "formato {}", format.name());
            let output = runtime.forward_batch(0, &[1.0; 512], 2).unwrap();
            assert_eq!(output.len(), 512, "formato {}", format.name());
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn runtime_loads_and_runs_f32_f16_and_bf16() {
        for format in [PlainFormat::F32, PlainFormat::F16, PlainFormat::Bf16] {
            let path = std::env::temp_dir().join(format!(
                "pom-ftk-plain-runtime-{}-{}.gguf",
                format.name(),
                std::process::id()
            ));
            write_plain_source(&path, format, false);
            let runtime =
                MoeRuntime::open(&path, "fixture", 2, 1, 0, Some(vec![(0, 1_000_000)])).unwrap();
            let output = runtime.forward_batch(0, &[1.0; 4], 2).unwrap();
            assert_eq!(output.len(), 4, "formato {}", format.name());
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn runtime_loads_and_runs_q4_1_q5_0_q5_1_and_q8_0() {
        for format in [
            LegacyQuantFormat::Q4_1,
            LegacyQuantFormat::Q5_0,
            LegacyQuantFormat::Q5_1,
            LegacyQuantFormat::Q8_0,
        ] {
            let path = std::env::temp_dir().join(format!(
                "pom-ftk-legacy-runtime-{}-{}.gguf",
                format.name(),
                std::process::id()
            ));
            write_legacy_source(&path, format, false);
            let runtime =
                MoeRuntime::open(&path, "fixture", 2, 1, 0, Some(vec![(0, 1_000_000)])).unwrap();
            let output = runtime.forward_batch(0, &[1.0; 64], 2).unwrap();
            assert_eq!(output.len(), 64, "formato {}", format.name());
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn runtime_loads_and_runs_mxfp4_tq1_tq2_q1_and_q2() {
        for format in [
            LowBitFormat::Mxfp4,
            LowBitFormat::Tq1_0,
            LowBitFormat::Tq2_0,
            LowBitFormat::Q1_0,
            LowBitFormat::Q2_0,
        ] {
            let path = std::env::temp_dir().join(format!(
                "pom-ftk-low-bit-runtime-{}-{}.gguf",
                format.name(),
                std::process::id()
            ));
            write_low_bit_source(&path, format, false);
            let runtime =
                MoeRuntime::open(&path, "fixture", 2, 1, 0, Some(vec![(0, 1_000_000)])).unwrap();
            let output = runtime.forward_batch(0, &[1.0; 512], 2).unwrap();
            assert_eq!(output.len(), 512, "formato {}", format.name());
            std::fs::remove_file(path).unwrap();
        }
    }
}
