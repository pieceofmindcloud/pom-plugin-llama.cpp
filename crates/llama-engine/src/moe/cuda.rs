//! Ponte Rust -> CUDA para o executor NVFP4 dos experts.

#![cfg(feature = "cuda")]

use super::admission::{CacheBudgetSnapshot, MemoryAccount};
use super::bandwidth::BandwidthProfile;
use super::cache::{DeviceAllocation, DeviceAllocator, DeviceExpertCache, DeviceExpertLease};
use super::format_k_quants::{KQuantExpert, KQuantFormat};
use super::format_nvfp4::{Nvfp4Error, Nvfp4Expert};
use super::format_plain::{PlainExpert, PlainFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const CUDA_TRANSFER_PROFILE_VERSION: u32 = 1;

// CUDA MoE diagnostics are part of the runtime contract. They must not depend
// on an operator-only environment variable: the hot path decides its
// execution policy from the loaded model and hardware profile, while the
// diagnostics describe what actually ran.
fn cuda_diagnostics_enabled() -> bool {
    true
}

/// Perfil persistido por uma carga e uma GPU concretas.
///
/// `cpu_wall_ms` é opcional para que um perfil de transferência possa ser
/// coletado antes da comparação CPU/GPU. Quando presente, a política só
/// escolhe CUDA se o tempo medido da chamada CUDA não for maior que o tempo
/// medido da referência CPU.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaTransferProfile {
    pub version: u32,
    pub hidden_width: usize,
    pub intermediate_width: usize,
    pub cuda_wall_ms: f64,
    pub cuda_kernel_span_ms: f64,
    pub cuda_transfer_span_ms: f64,
    pub cpu_wall_ms: Option<f64>,
    pub buffer_mask: u64,
    #[serde(default)]
    pub bandwidth: Option<BandwidthProfile>,
}

impl CudaTransferProfile {
    #[cfg(test)]
    pub fn new(
        hidden_width: usize,
        intermediate_width: usize,
        cuda_wall_ms: f64,
        cuda_kernel_span_ms: f64,
        cuda_transfer_span_ms: f64,
        cpu_wall_ms: Option<f64>,
        buffer_mask: u64,
    ) -> Result<Self, String> {
        let profile = Self {
            version: CUDA_TRANSFER_PROFILE_VERSION,
            hidden_width,
            intermediate_width,
            cuda_wall_ms,
            cuda_kernel_span_ms,
            cuda_transfer_span_ms,
            cpu_wall_ms,
            buffer_mask,
            bandwidth: None,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn prefers_cuda(&self) -> bool {
        match self.cpu_wall_ms {
            Some(cpu_wall_ms) => self.cuda_wall_ms <= cpu_wall_ms,
            None => true,
        }
    }

    #[cfg(test)]
    pub fn with_bandwidth(mut self, bandwidth: BandwidthProfile) -> Result<Self, String> {
        bandwidth
            .gpu_fraction()
            .map_err(|error| error.to_string())?;
        self.bandwidth = Some(bandwidth);
        self.validate()?;
        Ok(self)
    }

    pub fn sidecar_path(model_path: impl AsRef<Path>) -> PathBuf {
        model_path.as_ref().with_extension("cuda-profile.json")
    }

    pub fn load_for_model(model_path: impl AsRef<Path>) -> Result<Option<Self>, String> {
        let path = Self::sidecar_path(model_path);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(format!(
                    "não foi possível ler o perfil CUDA {}: {error}",
                    path.display()
                ))
            }
        };
        let profile: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("perfil CUDA inválido em {}: {error}", path.display()))?;
        profile
            .validate()
            .map_err(|error| format!("perfil CUDA inválido em {}: {error}", path.display()))?;
        Ok(Some(profile))
    }

    #[cfg(test)]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        self.validate()?;
        let path = path.as_ref();
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| format!("não foi possível serializar o perfil CUDA: {error}"))?;
        fs::write(path, bytes).map_err(|error| {
            format!(
                "não foi possível gravar o perfil CUDA {}: {error}",
                path.display()
            )
        })
    }

    fn validate(&self) -> Result<(), String> {
        if self.version != CUDA_TRANSFER_PROFILE_VERSION {
            return Err(format!(
                "versão {}, esperada {}",
                self.version, CUDA_TRANSFER_PROFILE_VERSION
            ));
        }
        if self.hidden_width == 0
            || self.intermediate_width == 0
            || self.hidden_width % 64 != 0
            || self.intermediate_width % 64 != 0
        {
            return Err("geometria deve ser positiva e múltipla de 64".into());
        }
        for (name, value) in [
            ("cuda_wall_ms", self.cuda_wall_ms),
            ("cuda_kernel_span_ms", self.cuda_kernel_span_ms),
            ("cuda_transfer_span_ms", self.cuda_transfer_span_ms),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} não é um tempo finito"));
            }
        }
        if let Some(value) = self.cpu_wall_ms {
            if !value.is_finite() || value < 0.0 {
                return Err("cpu_wall_ms não é um tempo finito".into());
            }
        }
        if self.buffer_mask == 0 {
            return Err("buffer_mask vazio".into());
        }
        if let Some(bandwidth) = self.bandwidth {
            bandwidth
                .gpu_fraction()
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CudaMoeCallTiming {
    pub expert_calls: u64,
    pub kernel_launches: u64,
    pub sync_count: u64,
    pub payload_validation_ms: f64,
    pub alloc_ms: f64,
    pub h2d_ms: f64,
    pub h2d_bytes: u64,
    pub kernel_ms: f64,
    pub kernel_span_ms: f64,
    pub launch_ms: f64,
    pub sync_ms: f64,
    pub d2h_ms: f64,
    pub d2h_bytes: u64,
    pub transfer_span_ms: f64,
    pub free_ms: f64,
    pub buffer_mask: u64,
    pub buffer_switches: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Nvfp4CudaDiagnostics {
    pub calls: u64,
    pub kernel_launches: u64,
    pub sync_count: u64,
    pub payload_validation_ms: f64,
    pub alloc_ms: f64,
    pub h2d_ms: f64,
    pub h2d_bytes: u64,
    pub kernel_ms: f64,
    pub kernel_span_ms: f64,
    pub launch_ms: f64,
    pub sync_ms: f64,
    pub d2h_ms: f64,
    pub d2h_bytes: u64,
    pub transfer_span_ms: f64,
    pub free_ms: f64,
    pub host_batch_prepare_ms: f64,
    pub host_result_copy_ms: f64,
    /// Parede externa da chamada FFI, sem somar submedições internas.
    pub cuda_call_wall_ms: f64,
    /// Bits dos slots de buffer usados desde o início do executor.
    pub buffer_mask: u64,
    /// Quantidade de alternâncias entre slots observada pelo executor.
    pub buffer_switches: u64,
    /// Medições separadas por ordinal CUDA. O cache residente mostra onde os
    /// bytes estão; esta lista mostra em quais placas a rota realmente rodou.
    pub device_calls: Vec<CudaMoeDeviceDiagnostics>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CudaMoeDeviceDiagnostics {
    pub device_index: u32,
    pub calls: u64,
    pub expert_calls: u64,
    pub kernel_ms: f64,
    pub cuda_call_wall_ms: f64,
}

impl Nvfp4CudaDiagnostics {
    pub fn add_call(&mut self, call: CudaMoeCallTiming) {
        self.calls = self.calls.saturating_add(call.expert_calls);
        self.kernel_launches = self.kernel_launches.saturating_add(call.kernel_launches);
        self.sync_count = self.sync_count.saturating_add(call.sync_count);
        self.payload_validation_ms += call.payload_validation_ms;
        self.alloc_ms += call.alloc_ms;
        self.h2d_ms += call.h2d_ms;
        self.h2d_bytes = self.h2d_bytes.saturating_add(call.h2d_bytes);
        self.kernel_ms += call.kernel_ms;
        self.kernel_span_ms += call.kernel_span_ms;
        self.launch_ms += call.launch_ms;
        self.sync_ms += call.sync_ms;
        self.d2h_ms += call.d2h_ms;
        self.d2h_bytes = self.d2h_bytes.saturating_add(call.d2h_bytes);
        self.transfer_span_ms += call.transfer_span_ms;
        self.free_ms += call.free_ms;
        self.buffer_mask |= call.buffer_mask;
        self.buffer_switches = self.buffer_switches.saturating_add(call.buffer_switches);
    }

    pub fn add_device_call(
        &mut self,
        device_index: u32,
        expert_calls: u64,
        kernel_ms: f64,
        cuda_call_wall_ms: f64,
    ) {
        let Some(device) = self
            .device_calls
            .iter_mut()
            .find(|device| device.device_index == device_index)
        else {
            self.device_calls.push(CudaMoeDeviceDiagnostics {
                device_index,
                calls: 1,
                expert_calls,
                kernel_ms,
                cuda_call_wall_ms,
            });
            return;
        };
        device.calls = device.calls.saturating_add(1);
        device.expert_calls = device.expert_calls.saturating_add(expert_calls);
        device.kernel_ms += kernel_ms;
        device.cuda_call_wall_ms += cuda_call_wall_ms;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaMoeError {
    Geometry(String),
    Payload(String),
    PayloadLength {
        expected: usize,
        actual: usize,
    },
    BackendUnavailable(String),
    Cache(String),
    HiddenWidth {
        expected: usize,
        actual: usize,
    },
    NonFinite {
        stage: &'static str,
        index: usize,
        count: usize,
        kind: &'static str,
    },
    Launch(i32),
}

impl fmt::Display for CudaMoeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(reason) => write!(formatter, "geometria CUDA MoE inválida: {reason}"),
            Self::Payload(reason) => write!(formatter, "payload CUDA MoE inválido: {reason}"),
            Self::PayloadLength { expected, actual } => write!(
                formatter,
                "payload CUDA MoE tem {actual} bytes, mas a geometria exige {expected}"
            ),
            Self::BackendUnavailable(reason) => {
                write!(formatter, "backend CUDA MoE indisponível: {reason}")
            }
            Self::Cache(reason) => write!(formatter, "cache CUDA MoE recusou o expert: {reason}"),
            Self::HiddenWidth { expected, actual } => {
                write!(
                    formatter,
                    "hidden CUDA MoE tem {actual}, esperado {expected}"
                )
            }
            Self::NonFinite {
                stage,
                index,
                count,
                kind,
            } => write!(
                formatter,
                "executor CUDA MoE produziu {count} valores não finitos ({kind}) no estágio {stage}, primeiro índice da saída {index}"
            ),
            Self::Launch(code) => write!(formatter, "kernel CUDA MoE falhou (código {code})"),
        }
    }
}

impl std::error::Error for CudaMoeError {}

fn cuda_device_index_i32(device_index: u32) -> Result<i32, CudaMoeError> {
    i32::try_from(device_index).map_err(|_| {
        CudaMoeError::BackendUnavailable(format!(
            "ordinal CUDA fora do limite suportado: {device_index}"
        ))
    })
}

struct DeviceRouteIndexGroup {
    device_index: u32,
    indices: Vec<usize>,
    token_ids: Vec<u32>,
    weights: Vec<f32>,
    token_offsets: Vec<usize>,
}

/// Divide uma rota por GPU sem alterar a ordem dos pares dentro de cada GPU.
/// Os offsets continuam indexados pelo lote global, portanto cada chamada CUDA
/// pode produzir um vetor completo e os resultados podem ser somados em
/// paralelo sem uma cópia de hidden entre placas.
fn group_route_indices(
    device_indices: &[u32],
    token_ids: &[u32],
    weights: &[f32],
    token_offsets: &[usize],
    tokens: usize,
) -> Result<Vec<DeviceRouteIndexGroup>, CudaMoeError> {
    if device_indices.is_empty()
        || device_indices.len() != token_ids.len()
        || device_indices.len() != weights.len()
        || tokens == 0
        || token_offsets.len() != tokens + 1
        || token_offsets.first() != Some(&0)
        || token_offsets.last() != Some(&device_indices.len())
        || token_offsets.windows(2).any(|pair| pair[0] > pair[1])
        || token_ids.iter().any(|&token| token as usize >= tokens)
        || weights.iter().any(|weight| !weight.is_finite())
    {
        return Err(CudaMoeError::Payload(
            "batch CUDA tem rotas ou offsets inválidos".into(),
        ));
    }

    let mut groups = Vec::<DeviceRouteIndexGroup>::new();
    let mut positions = HashMap::<u32, usize>::new();
    for token in 0..tokens {
        for route in token_offsets[token]..token_offsets[token + 1] {
            let device_index = device_indices[route];
            let group_index = if let Some(index) = positions.get(&device_index) {
                *index
            } else {
                let index = groups.len();
                positions.insert(device_index, index);
                groups.push(DeviceRouteIndexGroup {
                    device_index,
                    indices: Vec::new(),
                    token_ids: Vec::new(),
                    weights: Vec::new(),
                    // Fill offsets for tokens already traversed. The offset
                    // for the current token's end is appended below.
                    token_offsets: vec![0; token + 1],
                });
                index
            };
            let group = &mut groups[group_index];
            group.indices.push(route);
            group.token_ids.push(token_ids[route]);
            group.weights.push(weights[route]);
        }
        for group in &mut groups {
            group.token_offsets.push(group.indices.len());
        }
    }
    Ok(groups)
}

fn sum_parallel_device_outputs<T, F>(
    groups: Vec<T>,
    output_len: usize,
    run: F,
) -> Result<Vec<f32>, CudaMoeError>
where
    T: Send,
    F: Fn(T) -> Result<Vec<f32>, CudaMoeError> + Sync + Send,
{
    if groups.len() == 1 {
        let Some(group) = groups.into_iter().next() else {
            return Ok(vec![0.0; output_len]);
        };
        let values = run(group)?;
        if values.len() != output_len {
            return Err(CudaMoeError::Payload(
                "GPU CUDA MoE devolveu hidden com tamanho inesperado".into(),
            ));
        }
        return Ok(values);
    }
    let outputs = std::thread::scope(|scope| {
        let handles = groups
            .into_iter()
            .map(|group| {
                let run = &run;
                scope.spawn(move || run(group))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    CudaMoeError::BackendUnavailable("thread CUDA MoE entrou em pânico".into())
                })?
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut output = vec![0.0; output_len];
    for values in outputs {
        if values.len() != output_len {
            return Err(CudaMoeError::Payload(
                "GPU CUDA MoE devolveu hidden com tamanho inesperado".into(),
            ));
        }
        for (target, value) in output.iter_mut().zip(values) {
            *target += value;
        }
    }
    Ok(output)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaHeterogeneousTensor {
    pub data: *const u8,
    pub bytes: usize,
    pub rows: usize,
    pub columns: usize,
    pub dtype: u32,
    pub scale: f32,
    pub input_scale: f32,
    pub secondary_scale: f32,
    pub secondary_input_scale: f32,
}

// Os ponteiros apontam para alocações CUDA de experts que ficam imutáveis após
// o upload. A posse é mantida pelos leases durante toda a rota; as threads
// somente copiam os descritores e fazem leituras no device correspondente.
unsafe impl Sync for CudaHeterogeneousTensor {}

const _: () = {
    assert!(std::mem::offset_of!(CudaHeterogeneousTensor, data) == 0);
    assert!(std::mem::offset_of!(CudaHeterogeneousTensor, bytes) == 8);
    assert!(std::mem::offset_of!(CudaHeterogeneousTensor, rows) == 16);
    assert!(std::mem::offset_of!(CudaHeterogeneousTensor, columns) == 24);
    assert!(std::mem::offset_of!(CudaHeterogeneousTensor, dtype) == 32);
    assert!(std::mem::offset_of!(CudaHeterogeneousTensor, scale) == 36);
    assert!(std::mem::offset_of!(CudaHeterogeneousTensor, input_scale) == 40);
    assert!(std::mem::offset_of!(CudaHeterogeneousTensor, secondary_scale) == 44);
    assert!(std::mem::offset_of!(CudaHeterogeneousTensor, secondary_input_scale) == 48);
    assert!(std::mem::size_of::<CudaHeterogeneousTensor>() == 56);
};

impl Default for CudaHeterogeneousTensor {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            bytes: 0,
            rows: 0,
            columns: 0,
            dtype: 0,
            scale: 1.0,
            input_scale: 1.0,
            secondary_scale: 1.0,
            secondary_input_scale: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CudaHeterogeneousExpert {
    pub gate_up: CudaHeterogeneousTensor,
    pub gate: CudaHeterogeneousTensor,
    pub up: CudaHeterogeneousTensor,
    pub down: CudaHeterogeneousTensor,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct HeterogeneousTensorKey {
    data: usize,
    bytes: usize,
    rows: usize,
    columns: usize,
    dtype: u32,
    scale: u32,
    input_scale: u32,
    secondary_scale: u32,
    secondary_input_scale: u32,
}

impl From<CudaHeterogeneousTensor> for HeterogeneousTensorKey {
    fn from(tensor: CudaHeterogeneousTensor) -> Self {
        Self {
            data: tensor.data as usize,
            bytes: tensor.bytes,
            rows: tensor.rows,
            columns: tensor.columns,
            dtype: tensor.dtype,
            scale: tensor.scale.to_bits(),
            input_scale: tensor.input_scale.to_bits(),
            secondary_scale: tensor.secondary_scale.to_bits(),
            secondary_input_scale: tensor.secondary_input_scale.to_bits(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct HeterogeneousExpertTokenKey {
    gate_up: HeterogeneousTensorKey,
    gate: HeterogeneousTensorKey,
    up: HeterogeneousTensorKey,
    down: HeterogeneousTensorKey,
    token_id: u32,
}

impl HeterogeneousExpertTokenKey {
    fn new(expert: CudaHeterogeneousExpert, token_id: u32) -> Self {
        Self {
            gate_up: expert.gate_up.into(),
            gate: expert.gate.into(),
            up: expert.up.into(),
            down: expert.down.into(),
            token_id,
        }
    }
}

/// Deduplica o trabalho matemático sem alterar a rota lógica. Rotas que
/// apontam para o mesmo expert e token compartilham o resultado do FFN, mas
/// continuam no mapa para que seus pesos sejam somados individualmente.
fn deduplicate_heterogeneous_route_pairs(
    experts: &[CudaHeterogeneousExpert],
    token_ids: &[u32],
) -> Result<(Vec<CudaHeterogeneousExpert>, Vec<u32>, Vec<u32>), CudaMoeError> {
    if experts.len() != token_ids.len() {
        return Err(CudaMoeError::Payload(
            "rota CUDA heterogênea tem experts e tokens desalinhados".into(),
        ));
    }

    let mut positions = HashMap::<HeterogeneousExpertTokenKey, u32>::new();
    let mut unique_experts = Vec::new();
    let mut unique_token_ids = Vec::new();
    let mut route_to_pair = Vec::with_capacity(experts.len());
    for (&expert, &token_id) in experts.iter().zip(token_ids) {
        let key = HeterogeneousExpertTokenKey::new(expert, token_id);
        let pair = if let Some(&pair) = positions.get(&key) {
            pair
        } else {
            let pair = u32::try_from(unique_experts.len()).map_err(|_| {
                CudaMoeError::Payload(
                    "rota CUDA heterogênea excede o limite de pares deduplicados".into(),
                )
            })?;
            positions.insert(key, pair);
            unique_experts.push(expert);
            unique_token_ids.push(token_id);
            pair
        };
        route_to_pair.push(pair);
    }
    Ok((unique_experts, unique_token_ids, route_to_pair))
}

impl From<Nvfp4Error> for CudaMoeError {
    fn from(error: Nvfp4Error) -> Self {
        match error {
            Nvfp4Error::Geometry(reason) => Self::Geometry(reason.to_string()),
            Nvfp4Error::Payload(reason) => Self::Payload(reason.to_string()),
            Nvfp4Error::PayloadLength { expected, actual } => {
                Self::PayloadLength { expected, actual }
            }
            Nvfp4Error::BackendUnavailable(reason) => Self::BackendUnavailable(reason),
        }
    }
}

struct Nvfp4RoutePipelineHandle {
    handle: *mut c_void,
}

unsafe impl Send for Nvfp4RoutePipelineHandle {}

impl Nvfp4RoutePipelineHandle {
    fn new(device_index: u32, hidden: usize, intermediate: usize) -> Result<Self, CudaMoeError> {
        let device_index = cuda_device_index_i32(device_index)?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            pom_moe_nvfp4_route_pipeline_create(device_index, hidden, intermediate, &mut handle)
        };
        if status != 0 {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "pipeline CUDA MoE não pôde ser criado (código {status})"
            )));
        }
        if handle.is_null() {
            return Err(CudaMoeError::BackendUnavailable(
                "pipeline CUDA MoE retornou handle nulo".into(),
            ));
        }
        Ok(Self { handle })
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        payloads: *const *const u8,
        payload_count: usize,
        token_ids: *const u32,
        weights: *const f32,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32 {
        unsafe {
            pom_moe_nvfp4_route_pipeline_run(
                self.handle,
                payloads,
                payload_count,
                token_ids,
                weights,
                token_offsets,
                token_offsets_len,
                token_count,
                hidden,
                hidden_len,
                output,
                output_len,
                timing,
            )
        }
    }
}

impl Drop for Nvfp4RoutePipelineHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { pom_moe_nvfp4_route_pipeline_destroy(self.handle) };
        }
    }
}

pub struct Nvfp4CudaExecutor {
    hidden: usize,
    intermediate: usize,
    profile: bool,
    resident_cache: Arc<DeviceExpertCache>,
    route_pipelines: Vec<(u32, Mutex<Nvfp4RoutePipelineHandle>)>,
    diagnostics: Mutex<Nvfp4CudaDiagnostics>,
}

/// Executor CUDA para payloads K-quant residentes no device.
///
/// O dtype continua sendo dado na chamada FFI. A ponte não conhece modelo,
/// arquitetura nem convenção de nomes do GGUF. Os decodificadores
/// numericamente provados são Q2_K (10), Q3_K (11), Q4_K (12) e Q6_K (14);
/// os demais K-quants permanecem no caminho CPUReference.
pub struct KQuantCudaExecutor {
    hidden: usize,
    intermediate: usize,
    format: KQuantFormat,
    profile: bool,
    resident_cache: Arc<DeviceExpertCache>,
    route_pipelines: Vec<(u32, Mutex<KQuantRoutePipelineHandle>)>,
    diagnostics: Mutex<Nvfp4CudaDiagnostics>,
}

struct KQuantRoutePipelineHandle {
    handle: *mut c_void,
}

unsafe impl Send for KQuantRoutePipelineHandle {}

impl KQuantRoutePipelineHandle {
    fn new(device_index: u32, hidden: usize, intermediate: usize) -> Result<Self, CudaMoeError> {
        let device_index = cuda_device_index_i32(device_index)?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            pom_moe_kquant_route_pipeline_create(device_index, hidden, intermediate, &mut handle)
        };
        if status != 0 {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "pipeline CUDA K-quant não pôde ser criado (código {status})"
            )));
        }
        if handle.is_null() {
            return Err(CudaMoeError::BackendUnavailable(
                "pipeline CUDA K-quant retornou handle nulo".into(),
            ));
        }
        Ok(Self { handle })
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        dtype: u32,
        payloads: *const *const u8,
        payload_count: usize,
        payload_lengths: *const usize,
        token_ids: *const u32,
        weights: *const f32,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        nonfinite_stage: *mut i32,
        nonfinite_index: *mut usize,
        nonfinite_count: *mut usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32 {
        unsafe {
            pom_moe_kquant_route_pipeline_run(
                self.handle,
                dtype,
                payloads,
                payload_count,
                payload_lengths,
                token_ids,
                weights,
                token_offsets,
                token_offsets_len,
                token_count,
                hidden,
                hidden_len,
                output,
                output_len,
                nonfinite_stage,
                nonfinite_index,
                nonfinite_count,
                timing,
            )
        }
    }
}

impl Drop for KQuantRoutePipelineHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { pom_moe_kquant_route_pipeline_destroy(self.handle) };
        }
    }
}

struct IquantRoutePipelineHandle {
    handle: *mut c_void,
}

unsafe impl Send for IquantRoutePipelineHandle {}

impl IquantRoutePipelineHandle {
    fn new(device_index: u32, hidden: usize, intermediate: usize) -> Result<Self, CudaMoeError> {
        let device_index = cuda_device_index_i32(device_index)?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            pom_moe_iquant_route_pipeline_create(device_index, hidden, intermediate, &mut handle)
        };
        if status != 0 {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "pipeline CUDA IQ-quant não pôde ser criado (código {status})"
            )));
        }
        if handle.is_null() {
            return Err(CudaMoeError::BackendUnavailable(
                "pipeline CUDA IQ-quant retornou handle nulo".into(),
            ));
        }
        Ok(Self { handle })
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        dtype: u32,
        payloads: *const *const u8,
        payload_count: usize,
        payload_lengths: *const usize,
        token_ids: *const u32,
        weights: *const f32,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        nonfinite_stage: *mut i32,
        nonfinite_index: *mut usize,
        nonfinite_count: *mut usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32 {
        unsafe {
            pom_moe_iquant_route_pipeline_run(
                self.handle,
                dtype,
                payloads,
                payload_count,
                payload_lengths,
                token_ids,
                weights,
                token_offsets,
                token_offsets_len,
                token_count,
                hidden,
                hidden_len,
                output,
                output_len,
                nonfinite_stage,
                nonfinite_index,
                nonfinite_count,
                timing,
            )
        }
    }
}

impl Drop for IquantRoutePipelineHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { pom_moe_iquant_route_pipeline_destroy(self.handle) };
        }
    }
}

struct HeterogeneousRoutePipelineHandle {
    handle: *mut c_void,
}

unsafe impl Send for HeterogeneousRoutePipelineHandle {}

impl HeterogeneousRoutePipelineHandle {
    fn new(device_index: u32, hidden: usize, intermediate: usize) -> Result<Self, CudaMoeError> {
        let device_index = cuda_device_index_i32(device_index)?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            pom_moe_heterogeneous_route_pipeline_create(
                device_index,
                hidden,
                intermediate,
                &mut handle,
            )
        };
        if status != 0 {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "pipeline CUDA heterogêneo não pôde ser criado (código {status})"
            )));
        }
        if handle.is_null() {
            return Err(CudaMoeError::BackendUnavailable(
                "pipeline CUDA heterogêneo retornou handle nulo".into(),
            ));
        }
        Ok(Self { handle })
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        experts: *const CudaHeterogeneousExpert,
        expert_count: usize,
        token_ids: *const u32,
        weights: *const f32,
        route_to_pair: *const u32,
        route_count: usize,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        activation: i32,
        nonfinite_stage: *mut i32,
        nonfinite_index: *mut usize,
        nonfinite_count: *mut usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32 {
        unsafe {
            pom_moe_heterogeneous_route_pipeline_run(
                self.handle,
                experts,
                expert_count,
                token_ids,
                weights,
                route_to_pair,
                route_count,
                token_offsets,
                token_offsets_len,
                token_count,
                hidden,
                hidden_len,
                output,
                output_len,
                activation,
                nonfinite_stage,
                nonfinite_index,
                nonfinite_count,
                timing,
            )
        }
    }
}

impl Drop for HeterogeneousRoutePipelineHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { pom_moe_heterogeneous_route_pipeline_destroy(self.handle) };
        }
    }
}

/// Executor CUDA para os payloads escalares F32, F16 e BF16.
///
/// O payload continua no formato produzido por `PlainSource`, incluindo os
/// dois comprimentos de seção. O dtype é validado na fronteira e enviado ao
/// decoder CUDA como dado, para que F32, F16 e BF16 compartilhem o mesmo
/// caminho de cache e de roteamento.
pub struct PlainCudaExecutor {
    hidden: usize,
    intermediate: usize,
    format: PlainFormat,
    profile: bool,
    resident_cache: Arc<DeviceExpertCache>,
    diagnostics: Mutex<Nvfp4CudaDiagnostics>,
}

impl PlainCudaExecutor {
    pub fn new_with_cache(
        format: PlainFormat,
        hidden: usize,
        intermediate: usize,
        resident_cache: Arc<DeviceExpertCache>,
    ) -> Result<Self, CudaMoeError> {
        if !matches!(
            format,
            PlainFormat::F32 | PlainFormat::F16 | PlainFormat::Bf16
        ) {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "dtype GGML {} não tem ponte CUDA escalar",
                format.dtype()
            )));
        }
        Ok(Self {
            hidden,
            intermediate,
            format,
            profile: cuda_diagnostics_enabled(),
            resident_cache,
            diagnostics: Mutex::new(Nvfp4CudaDiagnostics::default()),
        })
    }

    pub fn resident_lookup(
        &self,
        key: &super::cache::ExpertKey,
    ) -> Option<super::cache::DeviceExpertLease> {
        self.resident_cache.lookup(key)
    }

    pub fn resident_admit(
        &self,
        key: super::cache::ExpertKey,
        payload: &[u8],
    ) -> Result<super::cache::DeviceExpertLease, CudaMoeError> {
        PlainExpert::from_payload(self.format, self.hidden, self.intermediate, payload)
            .map_err(|error| CudaMoeError::Payload(error.to_string()))?;
        self.resident_cache
            .admit(key, payload)
            .map_err(|error| CudaMoeError::Cache(error.to_string()))
    }

    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.resident_cache.stats()
    }

    pub fn run_route_batch_resident(
        &self,
        payloads: &[super::cache::DeviceExpertLease],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, CudaMoeError> {
        if payloads.is_empty()
            || payloads.len() != token_ids.len()
            || payloads.len() != weights.len()
            || tokens == 0
            || token_offsets.len() != tokens + 1
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&payloads.len())
            || token_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || token_ids.iter().any(|&token| token as usize >= tokens)
        {
            return Err(CudaMoeError::Payload(
                "batch CUDA escalar tem rotas inválidas".into(),
            ));
        }
        if hidden.len() != tokens.saturating_mul(self.hidden) {
            return Err(CudaMoeError::HiddenWidth {
                expected: tokens.saturating_mul(self.hidden),
                actual: hidden.len(),
            });
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(CudaMoeError::Payload(
                "batch CUDA escalar tem peso de rota não finito".into(),
            ));
        }

        let device_indices = payloads
            .iter()
            .map(DeviceExpertLease::device_index)
            .collect::<Vec<_>>();
        let groups =
            group_route_indices(&device_indices, token_ids, weights, token_offsets, tokens)?;
        sum_parallel_device_outputs(groups, hidden.len(), |group| {
            let grouped_payloads = group
                .indices
                .iter()
                .map(|&index| &payloads[index])
                .collect::<Vec<_>>();
            self.run_route_batch_on_device(&group, &grouped_payloads, hidden, tokens)
        })
    }

    fn run_route_batch_on_device(
        &self,
        group: &DeviceRouteIndexGroup,
        payloads: &[&DeviceExpertLease],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, CudaMoeError> {
        let handles = payloads
            .iter()
            .map(|payload| payload.handle() as *const u8)
            .collect::<Vec<_>>();
        let mut output = vec![0.0f32; hidden.len()];
        let mut timing = CudaMoeCallTiming {
            expert_calls: payloads.len() as u64,
            ..CudaMoeCallTiming::default()
        };
        let cuda_started = Instant::now();
        let status = unsafe {
            pom_moe_plain_forward_route_batch_resident(
                group.device_index as i32,
                self.format.dtype(),
                handles.as_ptr(),
                payloads.len(),
                group.token_ids.as_ptr(),
                group.weights.as_ptr(),
                group.token_offsets.as_ptr(),
                group.token_offsets.len(),
                tokens,
                hidden.as_ptr(),
                hidden.len(),
                output.as_mut_ptr(),
                output.len(),
                self.hidden,
                self.intermediate,
                if self.profile {
                    &mut timing
                } else {
                    std::ptr::null_mut()
                },
            )
        };
        let cuda_call_wall_ms = cuda_started.elapsed().as_secs_f64() * 1_000.0;
        if status != 0 {
            return Err(CudaMoeError::Launch(status));
        }
        if self.profile {
            let mut diagnostics = self
                .diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            diagnostics.add_device_call(
                group.device_index,
                payloads.len() as u64,
                timing.kernel_ms,
                cuda_call_wall_ms,
            );
            diagnostics.add_call(timing);
            diagnostics.cuda_call_wall_ms += cuda_call_wall_ms;
        }
        Ok(output)
    }

    pub fn diagnostics(&self) -> Nvfp4CudaDiagnostics {
        self.diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .to_owned()
    }
}

impl KQuantCudaExecutor {
    pub fn new_with_cache(
        format: KQuantFormat,
        hidden: usize,
        intermediate: usize,
        resident_cache: Arc<DeviceExpertCache>,
    ) -> Result<Self, CudaMoeError> {
        if !format.cuda_supported() {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "dtype GGML {} ({}) ainda não tem ponte CUDA K-quant",
                format.dtype(),
                format.name()
            )));
        }
        let route_pipelines = resident_cache
            .device_indices()
            .into_iter()
            .map(|device_index| {
                KQuantRoutePipelineHandle::new(device_index, hidden, intermediate)
                    .map(|pipeline| (device_index, Mutex::new(pipeline)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            hidden,
            intermediate,
            format,
            profile: cuda_diagnostics_enabled(),
            resident_cache,
            route_pipelines,
            diagnostics: Mutex::new(Nvfp4CudaDiagnostics::default()),
        })
    }

    pub fn resident_lookup(
        &self,
        key: &super::cache::ExpertKey,
    ) -> Option<super::cache::DeviceExpertLease> {
        self.resident_cache.lookup(key)
    }

    pub fn resident_admit(
        &self,
        key: super::cache::ExpertKey,
        payload: &[u8],
    ) -> Result<super::cache::DeviceExpertLease, CudaMoeError> {
        KQuantExpert::from_payload(self.format, self.hidden, self.intermediate, payload)
            .map_err(|error| CudaMoeError::Payload(error.to_string()))?;
        self.resident_cache
            .admit(key, payload)
            .map_err(|error| CudaMoeError::Cache(error.to_string()))
    }

    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.resident_cache.stats()
    }

    pub fn run_route_batch_resident(
        &self,
        payloads: &[super::cache::DeviceExpertLease],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, CudaMoeError> {
        if payloads.is_empty()
            || payloads.len() != token_ids.len()
            || payloads.len() != weights.len()
            || tokens == 0
            || token_offsets.len() != tokens + 1
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&payloads.len())
            || token_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || token_ids.iter().any(|&token| token as usize >= tokens)
        {
            return Err(CudaMoeError::Payload(
                "batch CUDA K-quant tem rotas inválidas".into(),
            ));
        }
        if hidden.len() != tokens.saturating_mul(self.hidden) {
            return Err(CudaMoeError::HiddenWidth {
                expected: tokens.saturating_mul(self.hidden),
                actual: hidden.len(),
            });
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(CudaMoeError::Payload(
                "batch CUDA K-quant tem peso de rota não finito".into(),
            ));
        }

        for payload in payloads {
            KQuantExpert::validate_packed_payload_length(
                self.format,
                self.hidden,
                self.intermediate,
                payload.bytes(),
            )
            .map_err(|error| match error {
                super::format_k_quants::KQuantError::PayloadLength { expected, actual } => {
                    CudaMoeError::PayloadLength { expected, actual }
                }
                error => CudaMoeError::Geometry(error.to_string()),
            })?;
        }

        let device_indices = payloads
            .iter()
            .map(DeviceExpertLease::device_index)
            .collect::<Vec<_>>();
        let groups =
            group_route_indices(&device_indices, token_ids, weights, token_offsets, tokens)?;
        sum_parallel_device_outputs(groups, hidden.len(), |group| {
            let grouped_payloads = group
                .indices
                .iter()
                .map(|&index| &payloads[index])
                .collect::<Vec<_>>();
            self.run_route_batch_on_device(&group, &grouped_payloads, hidden, tokens)
        })
    }

    fn run_route_batch_on_device(
        &self,
        group: &DeviceRouteIndexGroup,
        payloads: &[&DeviceExpertLease],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, CudaMoeError> {
        let handles = payloads
            .iter()
            .map(|payload| payload.handle() as *const u8)
            .collect::<Vec<_>>();
        let payload_lengths = payloads
            .iter()
            .map(|payload| payload.bytes())
            .collect::<Vec<_>>();
        let mut output = vec![0.0f32; hidden.len()];
        let mut timing = CudaMoeCallTiming {
            expert_calls: payloads.len() as u64,
            ..CudaMoeCallTiming::default()
        };
        let mut nonfinite_stage = 0;
        let mut nonfinite_index = 0;
        let mut nonfinite_count = 0;
        let cuda_started = Instant::now();
        let Some((_, route_pipeline)) = self
            .route_pipelines
            .iter()
            .find(|(device_index, _)| *device_index == group.device_index)
        else {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "pipeline K-quant ausente para CUDA{}",
                group.device_index
            )));
        };
        let status = route_pipeline
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .run(
                self.format.dtype(),
                handles.as_ptr(),
                payloads.len(),
                payload_lengths.as_ptr(),
                group.token_ids.as_ptr(),
                group.weights.as_ptr(),
                group.token_offsets.as_ptr(),
                group.token_offsets.len(),
                tokens,
                hidden.as_ptr(),
                hidden.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut nonfinite_stage,
                &mut nonfinite_index,
                &mut nonfinite_count,
                if self.profile {
                    &mut timing
                } else {
                    std::ptr::null_mut()
                },
            );
        let cuda_call_wall_ms = cuda_started.elapsed().as_secs_f64() * 1_000.0;
        if status != 0 {
            return Err(CudaMoeError::Launch(status));
        }
        let stage = match nonfinite_stage {
            1 => Some("gate_up"),
            2 => Some("ativação"),
            3 => Some("down"),
            4 => Some("agregação"),
            _ => None,
        };
        if let Some(stage) = stage {
            let kind = output
                .get(nonfinite_index)
                .map(|value| if value.is_nan() { "NaN" } else { "infinito" })
                .unwrap_or("não identificado");
            return Err(CudaMoeError::NonFinite {
                stage,
                index: nonfinite_index,
                count: nonfinite_count,
                kind,
            });
        }
        if self.profile {
            let mut diagnostics = self
                .diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            diagnostics.add_device_call(
                group.device_index,
                payloads.len() as u64,
                timing.kernel_ms,
                cuda_call_wall_ms,
            );
            diagnostics.add_call(timing);
            diagnostics.cuda_call_wall_ms += cuda_call_wall_ms;
        }
        Ok(output)
    }

    pub fn diagnostics(&self) -> Nvfp4CudaDiagnostics {
        self.diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .to_owned()
    }
}

/// Executor CUDA para os IQ usados nos modelos UD priorizados.
///
/// O formato permanece dado na chamada FFI. O pipeline residente cobre
/// IQ3_XXS (18), IQ3_S (21), IQ2_S (22) e IQ4_XS (23); um IQ diferente é
/// recusado antes de chegar ao kernel.
pub struct IqQuantCudaExecutor {
    hidden: usize,
    intermediate: usize,
    format: super::format_iq_quants::IqQuantFormat,
    profile: bool,
    resident_cache: Arc<DeviceExpertCache>,
    route_pipelines: Vec<(u32, Mutex<IquantRoutePipelineHandle>)>,
    diagnostics: Mutex<Nvfp4CudaDiagnostics>,
}

impl IqQuantCudaExecutor {
    pub fn new_with_cache(
        format: super::format_iq_quants::IqQuantFormat,
        hidden: usize,
        intermediate: usize,
        resident_cache: Arc<DeviceExpertCache>,
    ) -> Result<Self, CudaMoeError> {
        if !format.cuda_supported() {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "dtype GGML {} ({}) ainda não tem ponte CUDA IQ-quant",
                format.dtype(),
                format.name()
            )));
        }
        let route_pipelines = resident_cache
            .device_indices()
            .into_iter()
            .map(|device_index| {
                IquantRoutePipelineHandle::new(device_index, hidden, intermediate)
                    .map(|pipeline| (device_index, Mutex::new(pipeline)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            hidden,
            intermediate,
            format,
            profile: cuda_diagnostics_enabled(),
            resident_cache,
            route_pipelines,
            diagnostics: Mutex::new(Nvfp4CudaDiagnostics::default()),
        })
    }

    pub fn resident_lookup(
        &self,
        key: &super::cache::ExpertKey,
    ) -> Option<super::cache::DeviceExpertLease> {
        self.resident_cache.lookup(key)
    }

    pub fn resident_admit(
        &self,
        key: super::cache::ExpertKey,
        payload: &[u8],
    ) -> Result<super::cache::DeviceExpertLease, CudaMoeError> {
        self.resident_cache
            .admit(key, payload)
            .map_err(|error| CudaMoeError::Cache(error.to_string()))
    }

    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.resident_cache.stats()
    }

    pub fn run_route_batch_resident(
        &self,
        payloads: &[super::cache::DeviceExpertLease],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, CudaMoeError> {
        if payloads.is_empty()
            || payloads.len() != token_ids.len()
            || payloads.len() != weights.len()
            || tokens == 0
            || token_offsets.len() != tokens + 1
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&payloads.len())
            || token_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || token_ids.iter().any(|&token| token as usize >= tokens)
        {
            return Err(CudaMoeError::Payload(
                "batch CUDA IQ-quant tem rotas inválidas".into(),
            ));
        }
        if hidden.len() != tokens.saturating_mul(self.hidden) {
            return Err(CudaMoeError::HiddenWidth {
                expected: tokens.saturating_mul(self.hidden),
                actual: hidden.len(),
            });
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(CudaMoeError::Payload(
                "batch CUDA IQ-quant tem peso de rota não finito".into(),
            ));
        }

        let device_indices = payloads
            .iter()
            .map(DeviceExpertLease::device_index)
            .collect::<Vec<_>>();
        let groups =
            group_route_indices(&device_indices, token_ids, weights, token_offsets, tokens)?;
        sum_parallel_device_outputs(groups, hidden.len(), |group| {
            let grouped_payloads = group
                .indices
                .iter()
                .map(|&index| &payloads[index])
                .collect::<Vec<_>>();
            self.run_route_batch_on_device(&group, &grouped_payloads, hidden, tokens)
        })
    }

    fn run_route_batch_on_device(
        &self,
        group: &DeviceRouteIndexGroup,
        payloads: &[&DeviceExpertLease],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, CudaMoeError> {
        let handles = payloads
            .iter()
            .map(|payload| payload.handle() as *const u8)
            .collect::<Vec<_>>();
        let payload_lengths = payloads
            .iter()
            .map(|payload| payload.bytes())
            .collect::<Vec<_>>();
        let mut output = vec![0.0f32; hidden.len()];
        let mut timing = CudaMoeCallTiming {
            expert_calls: payloads.len() as u64,
            ..CudaMoeCallTiming::default()
        };
        let mut nonfinite_stage = 0;
        let mut nonfinite_index = 0;
        let mut nonfinite_count = 0;
        let Some((_, route_pipeline)) = self
            .route_pipelines
            .iter()
            .find(|(device_index, _)| *device_index == group.device_index)
        else {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "pipeline IQ-quant ausente para CUDA{}",
                group.device_index
            )));
        };
        let cuda_started = Instant::now();
        let status = route_pipeline
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .run(
                self.format.dtype(),
                handles.as_ptr(),
                payloads.len(),
                payload_lengths.as_ptr(),
                group.token_ids.as_ptr(),
                group.weights.as_ptr(),
                group.token_offsets.as_ptr(),
                group.token_offsets.len(),
                tokens,
                hidden.as_ptr(),
                hidden.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut nonfinite_stage,
                &mut nonfinite_index,
                &mut nonfinite_count,
                if self.profile {
                    &mut timing
                } else {
                    std::ptr::null_mut()
                },
            );
        let cuda_call_wall_ms = cuda_started.elapsed().as_secs_f64() * 1_000.0;
        if status != 0 {
            return Err(CudaMoeError::Launch(status));
        }
        if nonfinite_stage != 0 {
            let stage = match nonfinite_stage {
                1 => "gate_up",
                2 => "ativação",
                3 => "down",
                4 => "agregação",
                _ => "desconhecido",
            };
            return Err(CudaMoeError::NonFinite {
                stage,
                index: nonfinite_index,
                count: nonfinite_count,
                kind: "saída IQ-quant CUDA",
            });
        }
        if self.profile {
            let mut diagnostics = self
                .diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            diagnostics.add_device_call(
                group.device_index,
                payloads.len() as u64,
                timing.kernel_ms,
                cuda_call_wall_ms,
            );
            diagnostics.add_call(timing);
            diagnostics.cuda_call_wall_ms += cuda_call_wall_ms;
        }
        Ok(output)
    }

    pub fn diagnostics(&self) -> Nvfp4CudaDiagnostics {
        self.diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .to_owned()
    }
}

/// Executor CUDA para uma rota cujos tensores podem usar Q4_0, Q4_K, Q6_K,
/// IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS ou MXFP4 independentemente por papel. Os descritores
/// carregam ponteiro, tamanho, geometria de uma fatia e dtype de cada tensor;
/// o cache residente mantém os bytes no device entre tokens.
pub struct HeterogeneousCudaExecutor {
    profile: bool,
    resident_cache: Arc<DeviceExpertCache>,
    route_pipelines: Mutex<
        Vec<(
            u32,
            usize,
            usize,
            Arc<Mutex<HeterogeneousRoutePipelineHandle>>,
        )>,
    >,
    diagnostics: Mutex<Nvfp4CudaDiagnostics>,
}

impl HeterogeneousCudaExecutor {
    pub fn new_with_cache(
        _hidden: usize,
        _intermediate: usize,
        resident_cache: Arc<DeviceExpertCache>,
    ) -> Self {
        Self {
            profile: cuda_diagnostics_enabled(),
            resident_cache,
            route_pipelines: Mutex::new(Vec::new()),
            diagnostics: Mutex::new(Nvfp4CudaDiagnostics::default()),
        }
    }

    fn route_pipeline(
        &self,
        device_index: u32,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Arc<Mutex<HeterogeneousRoutePipelineHandle>>, CudaMoeError> {
        let mut pipelines = self
            .route_pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some((_, _, _, pipeline)) = pipelines.iter().find(|(device, width, inner, _)| {
            *device == device_index && *width == hidden && *inner == intermediate
        }) {
            return Ok(Arc::clone(pipeline));
        }
        let pipeline = Arc::new(Mutex::new(HeterogeneousRoutePipelineHandle::new(
            device_index,
            hidden,
            intermediate,
        )?));
        pipelines.push((device_index, hidden, intermediate, Arc::clone(&pipeline)));
        Ok(pipeline)
    }

    pub fn resident_lookup(
        &self,
        key: &super::cache::ExpertKey,
    ) -> Option<super::cache::DeviceExpertLease> {
        self.resident_cache.lookup(key)
    }

    pub fn resident_contains(&self, key: &super::cache::ExpertKey) -> bool {
        self.resident_cache.contains(key)
    }

    /// Retorna a GPU escolhida pela mesma afinidade usada pela admissão.
    ///
    /// O bloco heterogêneo usa esta informação antes de carregar os bytes
    /// frios para agrupar misses por shard e sobrepor uploads em GPUs
    /// diferentes.
    pub fn device_index_for_key(&self, key: &super::cache::ExpertKey) -> u32 {
        self.resident_cache.device_index_for_key(key)
    }

    pub fn resident_admit(
        &self,
        key: super::cache::ExpertKey,
        payload: &[u8],
    ) -> Result<super::cache::DeviceExpertLease, CudaMoeError> {
        self.resident_cache
            .admit(key, payload)
            .map_err(|error| CudaMoeError::Cache(error.to_string()))
    }

    pub fn resident_admit_many<'a, I>(
        &self,
        device_index: u32,
        candidates: I,
    ) -> Result<Vec<super::cache::DeviceExpertLease>, CudaMoeError>
    where
        I: IntoIterator<Item = (super::cache::ExpertKey, &'a [u8])>,
    {
        self.resident_cache
            .admit_many_on_device(device_index, candidates)
            .map_err(|error| CudaMoeError::Cache(error.to_string()))
    }

    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.resident_cache.stats()
    }

    pub fn resident_cache_capacity_bytes_by_device(&self) -> Vec<Option<u64>> {
        self.resident_cache
            .effective_capacity_bytes_by_device()
            .unwrap_or_else(|error| {
                tracing::debug!(
                    error = %error,
                    "não foi possível obter a capacidade efetiva do cache CUDA para diagnóstico"
                );
                vec![None; self.resident_cache.device_indices().len()]
            })
    }

    pub fn resident_cache_budget_snapshots_by_device(&self) -> Vec<Option<CacheBudgetSnapshot>> {
        self.resident_cache
            .budget_snapshots_by_device()
            .map(|snapshots| snapshots.into_iter().map(Some).collect())
            .unwrap_or_else(|error| {
                tracing::debug!(
                    error = %error,
                    "não foi possível obter o orçamento do cache CUDA para diagnóstico"
                );
                vec![None; self.resident_cache.device_indices().len()]
            })
    }

    pub fn run_route_batch_resident(
        &self,
        experts: &[CudaHeterogeneousExpert],
        device_indices: &[u32],
        expert_ids: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
        activation: i32,
    ) -> Result<Vec<f32>, CudaMoeError> {
        if experts.is_empty()
            || experts.len() != device_indices.len()
            || experts.len() != expert_ids.len()
            || experts.len() != token_ids.len()
            || experts.len() != weights.len()
            || tokens == 0
            || token_offsets.len() != tokens + 1
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&experts.len())
            || token_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || token_ids.iter().any(|&token| token as usize >= tokens)
        {
            return Err(CudaMoeError::Payload(
                "batch CUDA heterogêneo tem rotas inválidas".into(),
            ));
        }
        if !matches!(activation, 1 | 2) {
            return Err(CudaMoeError::Payload(format!(
                "ativação heterogênea {activation} não suportada pelo kernel CUDA"
            )));
        }
        let (hidden_width, intermediate_width) = heterogeneous_route_geometry(experts)?;
        if hidden.len() != tokens.saturating_mul(hidden_width) {
            return Err(CudaMoeError::HiddenWidth {
                expected: tokens.saturating_mul(hidden_width),
                actual: hidden.len(),
            });
        }
        if let Some((route, weight)) = weights
            .iter()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite())
        {
            return Err(CudaMoeError::Payload(format!(
                "tensor ffn_moe_weights tem valor não finito na rota {route}, token {}, expert {} ({})",
                token_ids[route],
                expert_ids[route],
                if weight.is_nan() { "NaN" } else { "infinito" }
            )));
        }

        for (index, expert) in experts.iter().enumerate() {
            validate_heterogeneous_tensor(
                expert.down,
                hidden_width,
                intermediate_width,
                &format!("rota {index} down"),
            )?;
            if !expert.gate_up.data.is_null() {
                if !expert.gate.data.is_null() || !expert.up.data.is_null() {
                    return Err(CudaMoeError::Payload(format!(
                        "rota {index} mistura gate_up fundido com gate/up separados"
                    )));
                }
                validate_heterogeneous_tensor(
                    expert.gate_up,
                    2 * intermediate_width,
                    hidden_width,
                    &format!("rota {index} gate_up"),
                )?;
            } else {
                validate_heterogeneous_tensor(
                    expert.gate,
                    intermediate_width,
                    hidden_width,
                    &format!("rota {index} gate"),
                )?;
                validate_heterogeneous_tensor(
                    expert.up,
                    intermediate_width,
                    hidden_width,
                    &format!("rota {index} up"),
                )?;
            }
        }

        let groups =
            group_route_indices(device_indices, token_ids, weights, token_offsets, tokens)?;
        sum_parallel_device_outputs(groups, hidden.len(), |group| {
            let grouped_experts = group
                .indices
                .iter()
                .map(|&index| experts[index])
                .collect::<Vec<_>>();
            self.run_route_batch_on_device(
                &group,
                &grouped_experts,
                hidden,
                tokens,
                activation,
                hidden_width,
                intermediate_width,
            )
        })
    }

    fn run_route_batch_on_device(
        &self,
        group: &DeviceRouteIndexGroup,
        experts: &[CudaHeterogeneousExpert],
        hidden: &[f32],
        tokens: usize,
        activation: i32,
        hidden_width: usize,
        intermediate_width: usize,
    ) -> Result<Vec<f32>, CudaMoeError> {
        let (unique_experts, unique_token_ids, route_to_pair) =
            deduplicate_heterogeneous_route_pairs(experts, &group.token_ids)?;
        let route_count = experts.len();
        let mut output = vec![0.0f32; hidden.len()];
        let mut timing = CudaMoeCallTiming {
            expert_calls: unique_experts.len() as u64,
            ..CudaMoeCallTiming::default()
        };
        let mut nonfinite_stage = 0;
        let mut nonfinite_index = 0;
        let mut nonfinite_count = 0;
        let cuda_started = Instant::now();
        let route_pipeline =
            self.route_pipeline(group.device_index, hidden_width, intermediate_width)?;
        let status = route_pipeline
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .run(
                unique_experts.as_ptr(),
                unique_experts.len(),
                unique_token_ids.as_ptr(),
                group.weights.as_ptr(),
                route_to_pair.as_ptr(),
                route_count,
                group.token_offsets.as_ptr(),
                group.token_offsets.len(),
                tokens,
                hidden.as_ptr(),
                hidden.len(),
                output.as_mut_ptr(),
                output.len(),
                activation,
                &mut nonfinite_stage,
                &mut nonfinite_index,
                &mut nonfinite_count,
                if self.profile {
                    &mut timing
                } else {
                    std::ptr::null_mut()
                },
            );
        let cuda_call_wall_ms = cuda_started.elapsed().as_secs_f64() * 1_000.0;
        if status != 0 {
            return Err(CudaMoeError::Launch(status));
        }
        let stage = match nonfinite_stage {
            1 => Some("gate_up"),
            2 => Some("ativação"),
            3 => Some("down"),
            4 => Some("agregação"),
            _ => None,
        };
        if let Some(stage) = stage {
            let kind = output
                .get(nonfinite_index)
                .map(|value| if value.is_nan() { "NaN" } else { "infinito" })
                .unwrap_or("não identificado");
            return Err(CudaMoeError::NonFinite {
                stage,
                index: nonfinite_index,
                count: nonfinite_count,
                kind,
            });
        }
        if self.profile {
            let mut diagnostics = self
                .diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            diagnostics.add_device_call(
                group.device_index,
                unique_experts.len() as u64,
                timing.kernel_ms,
                cuda_call_wall_ms,
            );
            diagnostics.add_call(timing);
            diagnostics.cuda_call_wall_ms += cuda_call_wall_ms;
        }
        Ok(output)
    }

    pub fn diagnostics(&self) -> Nvfp4CudaDiagnostics {
        self.diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .to_owned()
    }
}

fn heterogeneous_route_geometry(
    experts: &[CudaHeterogeneousExpert],
) -> Result<(usize, usize), CudaMoeError> {
    let down = experts
        .first()
        .map(|expert| expert.down)
        .ok_or_else(|| CudaMoeError::Geometry("rota heterogênea sem experts".into()))?;
    if down.data.is_null() || down.rows == 0 || down.columns == 0 {
        return Err(CudaMoeError::Geometry(
            "down heterogêneo não declara a geometria de um expert".into(),
        ));
    }
    Ok((down.rows, down.columns))
}

fn validate_heterogeneous_tensor(
    tensor: CudaHeterogeneousTensor,
    rows: usize,
    columns: usize,
    label: &str,
) -> Result<(), CudaMoeError> {
    if !tensor.scale.is_finite()
        || !tensor.input_scale.is_finite()
        || !tensor.secondary_scale.is_finite()
        || !tensor.secondary_input_scale.is_finite()
    {
        return Err(CudaMoeError::Payload(format!(
            "{label} contém escala não finita"
        )));
    }
    if tensor.rows != rows || tensor.columns != columns {
        return Err(CudaMoeError::Geometry(format!(
            "{label} declara matriz {}x{}, esperada {}x{} para um expert",
            tensor.rows, tensor.columns, rows, columns
        )));
    }
    let expected = match super::format_heterogeneous::cuda_tensor_expected_bytes(
        tensor.dtype,
        tensor.rows,
        tensor.columns,
    ) {
        Some(expected) => expected,
        None if !super::format_heterogeneous::cuda_tensor_dtype_supported(tensor.dtype) => {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "dtype GGML {} em {label} não tem ponte CUDA heterogênea",
                tensor.dtype
            )))
        }
        None => {
            return Err(CudaMoeError::Geometry(format!(
                "{label} tem colunas que não fecham blocos do dtype GGML {}",
                tensor.dtype
            )))
        }
    };
    if tensor.data.is_null() || columns == 0 {
        return Err(CudaMoeError::Geometry(format!(
            "{label} tem ponte nula ou colunas que não fecham blocos do dtype GGML {}",
            tensor.dtype
        )));
    }
    if tensor.bytes != expected {
        return Err(CudaMoeError::PayloadLength {
            expected,
            actual: tensor.bytes,
        });
    }
    Ok(())
}

impl Nvfp4CudaExecutor {
    #[cfg(test)]
    pub fn new(
        hidden: usize,
        intermediate: usize,
        compute_bytes: u64,
    ) -> Result<Self, CudaMoeError> {
        debug_assert!(Nvfp4Expert::packed_bytes(hidden, intermediate).is_ok());
        let resident_cache = measured_device_cache(compute_bytes)?;
        Self::new_with_cache(hidden, intermediate, resident_cache)
    }

    pub fn new_with_cache(
        hidden: usize,
        intermediate: usize,
        resident_cache: Arc<DeviceExpertCache>,
    ) -> Result<Self, CudaMoeError> {
        debug_assert!(Nvfp4Expert::packed_bytes(hidden, intermediate).is_ok());
        let route_pipelines = resident_cache
            .device_indices()
            .into_iter()
            .map(|device_index| {
                Nvfp4RoutePipelineHandle::new(device_index, hidden, intermediate)
                    .map(|pipeline| (device_index, Mutex::new(pipeline)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            hidden,
            intermediate,
            profile: cuda_diagnostics_enabled(),
            resident_cache,
            route_pipelines,
            diagnostics: Mutex::new(Nvfp4CudaDiagnostics::default()),
        })
    }

    pub fn resident_lookup(
        &self,
        key: &super::cache::ExpertKey,
    ) -> Option<super::cache::DeviceExpertLease> {
        self.resident_cache.lookup(key)
    }

    pub fn resident_admit(
        &self,
        key: super::cache::ExpertKey,
        payload: &[u8],
    ) -> Result<super::cache::DeviceExpertLease, CudaMoeError> {
        self.resident_cache
            .admit(key, payload)
            .map_err(|error| CudaMoeError::Cache(error.to_string()))
    }

    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.resident_cache.stats()
    }

    pub fn run_resident(
        &self,
        payload: &super::cache::DeviceExpertLease,
        hidden: &[f32],
    ) -> Result<Vec<f32>, CudaMoeError> {
        self.run_batch_resident(std::slice::from_ref(payload), hidden)
            .map(|mut outputs| outputs.remove(0))
    }

    pub fn run_batch_resident(
        &self,
        payloads: &[super::cache::DeviceExpertLease],
        hidden: &[f32],
    ) -> Result<Vec<Vec<f32>>, CudaMoeError> {
        if payloads.is_empty() {
            return Err(CudaMoeError::Payload(
                "batch residente de experts vazio".into(),
            ));
        }
        if hidden.len() != self.hidden {
            return Err(CudaMoeError::HiddenWidth {
                expected: self.hidden,
                actual: hidden.len(),
            });
        }
        let device_index = payloads[0].device_index();
        if payloads
            .iter()
            .any(|payload| payload.device_index() != device_index)
        {
            return Err(CudaMoeError::Payload(
                "batch residente de experts mistura dispositivos CUDA".into(),
            ));
        }
        let handles = payloads
            .iter()
            .map(|payload| payload.handle() as *const u8)
            .collect::<Vec<_>>();
        let mut flat_output = vec![0.0f32; self.hidden * payloads.len()];
        let mut timing = CudaMoeCallTiming {
            expert_calls: payloads.len() as u64,
            ..CudaMoeCallTiming::default()
        };
        let cuda_started = Instant::now();
        let status = unsafe {
            pom_moe_nvfp4_forward_batch_resident(
                device_index as i32,
                handles.as_ptr(),
                payloads.len(),
                hidden.as_ptr(),
                hidden.len(),
                flat_output.as_mut_ptr(),
                flat_output.len(),
                self.hidden,
                self.intermediate,
                if self.profile {
                    &mut timing
                } else {
                    std::ptr::null_mut()
                },
            )
        };
        let cuda_call_wall_ms = cuda_started.elapsed().as_secs_f64() * 1_000.0;
        if status != 0 {
            return Err(CudaMoeError::Launch(status));
        }
        if self.profile {
            let mut diagnostics = self
                .diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            diagnostics.add_device_call(
                device_index,
                payloads.len() as u64,
                timing.kernel_ms,
                cuda_call_wall_ms,
            );
            diagnostics.add_call(timing);
            diagnostics.cuda_call_wall_ms += cuda_call_wall_ms;
        }
        Ok(flat_output
            .chunks_exact(self.hidden)
            .map(ToOwned::to_owned)
            .collect())
    }

    pub fn run_route_batch_resident(
        &self,
        payloads: &[super::cache::DeviceExpertLease],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, CudaMoeError> {
        if hidden.len() != tokens.saturating_mul(self.hidden) {
            return Err(CudaMoeError::HiddenWidth {
                expected: tokens.saturating_mul(self.hidden),
                actual: hidden.len(),
            });
        }
        let device_indices = payloads
            .iter()
            .map(DeviceExpertLease::device_index)
            .collect::<Vec<_>>();
        let groups =
            group_route_indices(&device_indices, token_ids, weights, token_offsets, tokens)?;
        sum_parallel_device_outputs(groups, hidden.len(), |group| {
            let grouped_payloads = group
                .indices
                .iter()
                .map(|&index| &payloads[index])
                .collect::<Vec<_>>();
            self.run_route_batch_on_device(&group, &grouped_payloads, hidden, tokens)
        })
    }

    fn run_route_batch_on_device(
        &self,
        group: &DeviceRouteIndexGroup,
        payloads: &[&DeviceExpertLease],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, CudaMoeError> {
        let handles = payloads
            .iter()
            .map(|payload| payload.handle() as *const u8)
            .collect::<Vec<_>>();
        let mut output = vec![0.0f32; hidden.len()];
        let mut timing = CudaMoeCallTiming {
            expert_calls: payloads.len() as u64,
            ..CudaMoeCallTiming::default()
        };
        let Some((_, route_pipeline)) = self
            .route_pipelines
            .iter()
            .find(|(device_index, _)| *device_index == group.device_index)
        else {
            return Err(CudaMoeError::BackendUnavailable(format!(
                "pipeline NVFP4 ausente para CUDA{}",
                group.device_index
            )));
        };
        let cuda_started = Instant::now();
        let status = route_pipeline
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .run(
                handles.as_ptr(),
                payloads.len(),
                group.token_ids.as_ptr(),
                group.weights.as_ptr(),
                group.token_offsets.as_ptr(),
                group.token_offsets.len(),
                tokens,
                hidden.as_ptr(),
                hidden.len(),
                output.as_mut_ptr(),
                output.len(),
                if self.profile {
                    &mut timing
                } else {
                    std::ptr::null_mut()
                },
            );
        let cuda_call_wall_ms = cuda_started.elapsed().as_secs_f64() * 1_000.0;
        if status != 0 {
            return Err(CudaMoeError::Launch(status));
        }
        if self.profile {
            let mut diagnostics = self
                .diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            diagnostics.add_device_call(
                group.device_index,
                payloads.len() as u64,
                timing.kernel_ms,
                cuda_call_wall_ms,
            );
            diagnostics.add_call(timing);
            diagnostics.cuda_call_wall_ms += cuda_call_wall_ms;
        }
        Ok(output)
    }

    pub fn diagnostics(&self) -> Nvfp4CudaDiagnostics {
        self.diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .to_owned()
    }
}

struct CudaDeviceAllocation {
    device_index: u32,
    handle: usize,
    bytes: usize,
}

// The CUDA allocation is independent of the Rust thread that owns its lease;
// the CUDA runtime accepts freeing it from another host thread after the
// kernel call has completed.
unsafe impl Send for CudaDeviceAllocation {}
unsafe impl Sync for CudaDeviceAllocation {}

impl DeviceAllocation for CudaDeviceAllocation {
    fn bytes(&self) -> usize {
        self.bytes
    }

    fn handle(&self) -> usize {
        self.handle
    }
}

impl Drop for CudaDeviceAllocation {
    fn drop(&mut self) {
        unsafe { pom_moe_device_free(self.device_index as i32, self.handle as *mut c_void) };
    }
}

struct CudaDeviceAllocator {
    device_index: u32,
    name: String,
}

impl CudaDeviceAllocator {
    fn new(device_index: u32) -> Self {
        Self {
            device_index,
            name: format!("CUDA{device_index}"),
        }
    }
}

impl DeviceAllocator for CudaDeviceAllocator {
    fn device_name(&self) -> &str {
        &self.name
    }

    fn free_bytes(&self) -> io::Result<u64> {
        let mut free = 0u64;
        let status = unsafe { pom_moe_device_free_bytes(self.device_index as i32, &mut free) };
        if status == 0 {
            Ok(free)
        } else {
            Err(io::Error::other(format!(
                "cudaMemGetInfo retornou {status}"
            )))
        }
    }

    fn upload(&self, payload: &[u8]) -> Result<Box<dyn DeviceAllocation>, String> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            pom_moe_device_upload(
                self.device_index as i32,
                payload.as_ptr(),
                payload.len(),
                &mut handle,
            )
        };
        if status != 0 {
            return Err(format!("upload CUDA retornou {status}"));
        }
        if handle.is_null() {
            return Err("upload CUDA retornou handle nulo".into());
        }
        Ok(Box::new(CudaDeviceAllocation {
            device_index: self.device_index,
            handle: handle as usize,
            bytes: payload.len(),
        }))
    }

    fn upload_many(&self, payloads: &[&[u8]]) -> Result<Vec<Box<dyn DeviceAllocation>>, String> {
        if payloads.is_empty() {
            return Ok(Vec::new());
        }
        let payload_ptrs = payloads
            .iter()
            .map(|payload| payload.as_ptr())
            .collect::<Vec<_>>();
        let payload_lengths = payloads
            .iter()
            .map(|payload| payload.len())
            .collect::<Vec<_>>();
        let mut handles = vec![std::ptr::null_mut(); payloads.len()];
        let status = unsafe {
            pom_moe_device_upload_many(
                self.device_index as i32,
                payload_ptrs.as_ptr(),
                payload_lengths.as_ptr(),
                payloads.len(),
                handles.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(format!("upload CUDA em lote retornou {status}"));
        }
        handles
            .into_iter()
            .zip(payloads)
            .map(|(handle, payload)| {
                if handle.is_null() {
                    return Err("upload CUDA em lote retornou handle nulo".into());
                }
                Ok(Box::new(CudaDeviceAllocation {
                    device_index: self.device_index,
                    handle: handle as usize,
                    bytes: payload.len(),
                }) as Box<dyn DeviceAllocation>)
            })
            .collect()
    }
}

fn measured_device_cache(compute_bytes: u64) -> Result<Arc<DeviceExpertCache>, CudaMoeError> {
    measured_device_cache_for_devices(&[(0, compute_bytes)])
}

fn measured_device_cache_for_devices(
    compute_bytes: &[(u32, u64)],
) -> Result<Arc<DeviceExpertCache>, CudaMoeError> {
    if compute_bytes.is_empty() {
        return Err(CudaMoeError::BackendUnavailable(
            "llama.cpp não reportou compute buffers CUDA por dispositivo".into(),
        ));
    }
    let mut shards = Vec::with_capacity(compute_bytes.len());
    for &(device_index, compute) in compute_bytes {
        let allocator = Arc::new(CudaDeviceAllocator::new(device_index));
        let baseline_free_bytes = allocator
            .free_bytes()
            .map_err(|error| CudaMoeError::BackendUnavailable(error.to_string()))?;
        let account =
            MemoryAccount::for_device_cache(baseline_free_bytes, compute).ok_or_else(|| {
                CudaMoeError::BackendUnavailable(format!(
                    "compute buffer CUDA medido inválido em CUDA{device_index}: {compute} bytes"
                ))
            })?;
        shards.push((device_index, account, allocator as Arc<dyn DeviceAllocator>));
    }
    Ok(DeviceExpertCache::new_sharded(shards))
}

pub fn create_device_cache(compute_bytes: u64) -> Result<Arc<DeviceExpertCache>, CudaMoeError> {
    measured_device_cache(compute_bytes)
}

pub fn create_device_cache_for_devices(
    compute_bytes: &[(u32, u64)],
) -> Result<Arc<DeviceExpertCache>, CudaMoeError> {
    measured_device_cache_for_devices(compute_bytes)
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn pom_moe_nvfp4_forward_batch_resident(
        device_index: i32,
        payloads: *const *const u8,
        payload_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        hidden_width: usize,
        intermediate_width: usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32;

    fn pom_moe_nvfp4_route_pipeline_create(
        device_index: i32,
        hidden_width: usize,
        intermediate_width: usize,
        out: *mut *mut c_void,
    ) -> i32;

    fn pom_moe_nvfp4_route_pipeline_run(
        handle: *mut c_void,
        payloads: *const *const u8,
        payload_count: usize,
        token_ids: *const u32,
        weights: *const f32,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32;

    fn pom_moe_nvfp4_route_pipeline_destroy(handle: *mut c_void);

    fn pom_moe_kquant_route_pipeline_create(
        device_index: i32,
        hidden_width: usize,
        intermediate_width: usize,
        out: *mut *mut c_void,
    ) -> i32;

    fn pom_moe_kquant_route_pipeline_run(
        handle: *mut c_void,
        dtype: u32,
        payloads: *const *const u8,
        payload_count: usize,
        payload_lengths: *const usize,
        token_ids: *const u32,
        weights: *const f32,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        nonfinite_stage: *mut i32,
        nonfinite_index: *mut usize,
        nonfinite_count: *mut usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32;

    fn pom_moe_kquant_route_pipeline_destroy(handle: *mut c_void);

    fn pom_moe_iquant_route_pipeline_create(
        device_index: i32,
        hidden_width: usize,
        intermediate_width: usize,
        out: *mut *mut c_void,
    ) -> i32;

    fn pom_moe_iquant_route_pipeline_run(
        handle: *mut c_void,
        dtype: u32,
        payloads: *const *const u8,
        payload_count: usize,
        payload_lengths: *const usize,
        token_ids: *const u32,
        weights: *const f32,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        nonfinite_stage: *mut i32,
        nonfinite_index: *mut usize,
        nonfinite_count: *mut usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32;

    fn pom_moe_iquant_route_pipeline_destroy(handle: *mut c_void);

    fn pom_moe_plain_forward_route_batch_resident(
        device_index: i32,
        dtype: u32,
        payloads: *const *const u8,
        payload_count: usize,
        token_ids: *const u32,
        weights: *const f32,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        hidden_width: usize,
        intermediate_width: usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32;

    fn pom_moe_iquant_forward_route_batch_resident(
        device_index: i32,
        dtype: u32,
        payloads: *const *const u8,
        payload_count: usize,
        token_ids: *const u32,
        weights: *const f32,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        hidden_width: usize,
        intermediate_width: usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32;

    fn pom_moe_heterogeneous_quant_forward_route_batch_resident(
        device_index: i32,
        experts: *const CudaHeterogeneousExpert,
        expert_count: usize,
        token_ids: *const u32,
        weights: *const f32,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        hidden_width: usize,
        intermediate_width: usize,
        activation: i32,
        nonfinite_stage: *mut i32,
        nonfinite_index: *mut usize,
        nonfinite_count: *mut usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32;

    fn pom_moe_heterogeneous_route_pipeline_create(
        device_index: i32,
        hidden_width: usize,
        intermediate_width: usize,
        out: *mut *mut c_void,
    ) -> i32;

    fn pom_moe_heterogeneous_route_pipeline_run(
        handle: *mut c_void,
        experts: *const CudaHeterogeneousExpert,
        expert_count: usize,
        token_ids: *const u32,
        weights: *const f32,
        route_to_pair: *const u32,
        route_count: usize,
        token_offsets: *const usize,
        token_offsets_len: usize,
        token_count: usize,
        hidden: *const f32,
        hidden_len: usize,
        output: *mut f32,
        output_len: usize,
        activation: i32,
        nonfinite_stage: *mut i32,
        nonfinite_index: *mut usize,
        nonfinite_count: *mut usize,
        timing: *mut CudaMoeCallTiming,
    ) -> i32;

    fn pom_moe_heterogeneous_route_pipeline_destroy(handle: *mut c_void);

    fn pom_moe_device_free_bytes(device_index: i32, free_bytes: *mut u64) -> i32;
    fn pom_moe_device_upload(
        device_index: i32,
        payload: *const u8,
        payload_len: usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn pom_moe_device_upload_many(
        device_index: i32,
        payloads: *const *const u8,
        payload_lengths: *const usize,
        payload_count: usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn pom_moe_device_free(device_index: i32, handle: *mut c_void);
}

#[cfg(test)]
unsafe extern "C" {
    fn pom_moe_iquant_trace_single_resident(
        device_index: i32,
        dtype: u32,
        payload: *const u8,
        hidden: *const f32,
        hidden_len: usize,
        gate_up: *mut f32,
        gate_up_len: usize,
        activated: *mut f32,
        activated_len: usize,
        output: *mut f32,
        output_len: usize,
        hidden_width: usize,
        intermediate_width: usize,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::super::admission::MemoryAccount;
    use super::super::cache::{DeviceAllocation, DeviceAllocator, DeviceExpertCache, ExpertKey};
    use super::super::format_iq_quants::{
        dequantize_row as dequantize_iq_row, row_bytes, IqQuantExpert, IqQuantFormat,
    };
    use super::super::format_k_quants::{
        dequantize_row as dequantize_k_row, KQuantExpert, KQuantFormat,
    };
    use super::super::format_low_bits::{decode_row as decode_low_bit_row, LowBitFormat};
    use super::super::format_nvfp4::{Nvfp4Expert, Nvfp4Source};
    use super::super::format_plain::{PlainExpert, PlainFormat};
    use super::super::format_q4_0::{pack_payload, parse_tensor_table, Q4_0Expert};
    use super::*;

    #[test]
    fn moe_cuda_diagnostics_are_automatic() {
        assert!(cuda_diagnostics_enabled());
    }
    use memmap2::Mmap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct NonCloneState(Arc<AtomicUsize>);

    #[test]
    fn parallel_sum_reuses_the_fn_closure_across_all_groups() {
        let calls = Arc::new(AtomicUsize::new(0));
        let state = NonCloneState(Arc::clone(&calls));
        let output = sum_parallel_device_outputs(vec![1u8, 2, 3], 2, move |group| {
            state.0.fetch_add(1, Ordering::Relaxed);
            Ok(vec![group as f32, 1.0])
        })
        .unwrap();

        assert_eq!(output, vec![6.0, 3.0]);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn a_single_group_sum_creates_no_waiting_thread() {
        let caller = std::thread::current().id();
        let observed = Arc::new(Mutex::new(None));
        let output = sum_parallel_device_outputs(vec![1u8], 1, {
            let observed = Arc::clone(&observed);
            move |_| {
                *observed.lock().unwrap() = Some(std::thread::current().id());
                Ok(vec![1.0])
            }
        })
        .unwrap();

        assert_eq!(output, vec![1.0]);
        assert_eq!(*observed.lock().unwrap(), Some(caller));
    }

    #[test]
    fn grouping_by_gpu_preserves_offsets_when_a_gpu_only_shows_up_later() {
        let groups = group_route_indices(
            &[0, 1, 1, 0],
            &[0, 1, 1, 2],
            &[1.0, 1.0, 1.0, 1.0],
            &[0, 1, 3, 4],
            3,
        )
        .unwrap();

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].token_offsets, vec![0, 1, 1, 2]);
        assert_eq!(groups[1].token_offsets, vec![0, 0, 2, 2]);
    }

    #[test]
    fn deduplicating_expert_token_pairs_preserves_the_order_and_the_map() {
        let expert_a = CudaHeterogeneousExpert {
            down: CudaHeterogeneousTensor {
                data: 0x1000usize as *const u8,
                ..CudaHeterogeneousTensor::default()
            },
            ..CudaHeterogeneousExpert::default()
        };
        let expert_b = CudaHeterogeneousExpert {
            down: CudaHeterogeneousTensor {
                data: 0x2000usize as *const u8,
                ..CudaHeterogeneousTensor::default()
            },
            ..CudaHeterogeneousExpert::default()
        };

        let (unique_experts, unique_tokens, route_to_pair) = deduplicate_heterogeneous_route_pairs(
            &[expert_a, expert_a, expert_a, expert_b],
            &[0, 0, 1, 0],
        )
        .unwrap();

        assert_eq!(unique_experts.len(), 3);
        assert_eq!(unique_experts[0].down.data, expert_a.down.data);
        assert_eq!(unique_experts[1].down.data, expert_a.down.data);
        assert_eq!(unique_experts[2].down.data, expert_b.down.data);
        assert_eq!(unique_tokens, vec![0, 1, 0]);
        assert_eq!(route_to_pair, vec![0, 0, 1, 2]);
    }

    struct TestDeviceAllocation {
        bytes: usize,
    }

    impl DeviceAllocation for TestDeviceAllocation {
        fn bytes(&self) -> usize {
            self.bytes
        }

        fn handle(&self) -> usize {
            1
        }
    }

    struct TestDeviceAllocator;

    impl DeviceAllocator for TestDeviceAllocator {
        fn device_name(&self) -> &str {
            "test-device"
        }

        fn free_bytes(&self) -> std::io::Result<u64> {
            Ok(1 << 30)
        }

        fn upload(&self, payload: &[u8]) -> Result<Box<dyn DeviceAllocation>, String> {
            Ok(Box::new(TestDeviceAllocation {
                bytes: payload.len(),
            }))
        }
    }

    fn packed_rows(rows: usize, columns: usize, nibble: u8) -> Vec<u8> {
        let row_size = columns / 64 * 36;
        let mut output = vec![0u8; rows * row_size];
        for row in output.chunks_exact_mut(row_size) {
            for block in row.chunks_exact_mut(36) {
                block[..4].fill(0x38);
                block[4..].fill(nibble | (nibble << 4));
            }
        }
        output
    }

    fn payload(hidden: usize, intermediate: usize) -> Vec<u8> {
        payload_with_nibble(hidden, intermediate, 0x22)
    }

    fn payload_with_nibble(hidden: usize, intermediate: usize, nibble: u8) -> Vec<u8> {
        let gate_up = packed_rows(2 * intermediate, hidden, nibble);
        let down = packed_rows(hidden, intermediate, nibble);
        let mut payload = Vec::with_capacity(32 + gate_up.len() + down.len());
        payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
        for _ in 0..6 {
            payload.extend_from_slice(&1.0f32.to_le_bytes());
        }
        payload.extend_from_slice(&gate_up);
        payload.extend_from_slice(&down);
        payload
    }

    fn q4_rows(rows: usize, columns: usize, nibble: u8) -> Vec<u8> {
        let row_bytes = columns / 32 * 18;
        let mut output = vec![0u8; rows * row_bytes];
        for row in output.chunks_exact_mut(row_bytes) {
            for block in row.chunks_exact_mut(18) {
                block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                block[2..].fill(nibble | (nibble << 4));
            }
        }
        output
    }

    fn q4_payload(hidden: usize, intermediate: usize) -> Vec<u8> {
        pack_payload(
            &q4_rows(2 * intermediate, hidden, 9),
            &q4_rows(hidden, intermediate, 9),
        )
        .unwrap()
    }

    fn k_payload(format: KQuantFormat, hidden: usize, intermediate: usize) -> Vec<u8> {
        let block_bytes = match format {
            KQuantFormat::Q2K => 84,
            KQuantFormat::Q3K => 110,
            KQuantFormat::Q4K => 144,
            KQuantFormat::Q5K => 176,
            KQuantFormat::Q6K => 210,
            _ => panic!("fixture CUDA K-quant não suportada"),
        };
        let row_bytes = |columns: usize| columns / 256 * block_bytes;
        let matrix = |rows: usize, columns: usize| {
            let mut bytes = vec![0u8; rows * row_bytes(columns)];
            for block in bytes.chunks_exact_mut(block_bytes) {
                block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                if matches!(format, KQuantFormat::Q4K) {
                    block[4..16].fill(1);
                    block[16..].fill(0x22);
                } else if matches!(format, KQuantFormat::Q6K) {
                    block[..128].fill(0x11);
                    block[0] = 0;
                    block[1] = 0x7e;
                    block[128..192].fill(0xaa);
                    block[192..208].fill(1);
                    block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
                } else if matches!(format, KQuantFormat::Q5K) {
                    // block_q5_K: d[2], dmin[2], scales[12], qh[32], ql[128].
                    block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                    block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
                    for (index, value) in block[4..16].iter_mut().enumerate() {
                        *value = (index as u8).wrapping_mul(19).wrapping_add(3);
                    }
                    for (index, value) in block[16..48].iter_mut().enumerate() {
                        *value = (index as u8).wrapping_mul(23).wrapping_add(11);
                    }
                    for (index, value) in block[48..176].iter_mut().enumerate() {
                        *value = (index as u8).wrapping_mul(31).wrapping_add(5);
                    }
                } else if matches!(format, KQuantFormat::Q2K) {
                    for (index, value) in block[..16].iter_mut().enumerate() {
                        *value = (index as u8).wrapping_mul(7).wrapping_add(3);
                    }
                    for (index, value) in block[16..80].iter_mut().enumerate() {
                        *value = (index as u8).wrapping_mul(13).wrapping_add(5);
                    }
                    block[80..82].copy_from_slice(&0x3c00u16.to_le_bytes());
                    block[82..84].copy_from_slice(&0x3800u16.to_le_bytes());
                } else {
                    for (index, value) in block[..32].iter_mut().enumerate() {
                        *value = (index as u8).wrapping_mul(17).wrapping_add(9);
                    }
                    for (index, value) in block[32..96].iter_mut().enumerate() {
                        *value = (index as u8).wrapping_mul(29).wrapping_add(7);
                    }
                    for (index, value) in block[96..108].iter_mut().enumerate() {
                        *value = (index as u8).wrapping_mul(11).wrapping_add(1);
                    }
                    block[108..110].copy_from_slice(&0x3c00u16.to_le_bytes());
                }
            }
            bytes
        };
        let gate_up = matrix(2 * intermediate, hidden);
        let down = matrix(hidden, intermediate);
        let mut payload = Vec::with_capacity(32 + gate_up.len() + down.len());
        payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
        for _ in 0..6 {
            payload.extend_from_slice(&1.0f32.to_le_bytes());
        }
        payload.extend_from_slice(&gate_up);
        payload.extend_from_slice(&down);
        payload
    }

    #[test]
    fn cuda_k_quant_executor_rejects_a_lease_smaller_than_the_geometry_before_the_kernel() {
        let format = KQuantFormat::Q6K;
        let hidden = 256;
        let intermediate = 256;
        let expected = KQuantExpert::packed_bytes(format, hidden, intermediate).unwrap();
        let actual = expected / 8;
        let cache = DeviceExpertCache::new(
            MemoryAccount {
                baseline_free_bytes: 1 << 30,
                compute_bytes: 0,
                weights_bytes: 0,
                fixed_bytes: 0,
                reserve_bytes: 0,
            },
            std::sync::Arc::new(TestDeviceAllocator),
        );
        let lease = cache
            .admit(
                ExpertKey::new("cuda-k-guard", 0, 0, format.name()),
                &vec![0; actual],
            )
            .unwrap();
        let executor =
            KQuantCudaExecutor::new_with_cache(format, hidden, intermediate, cache).unwrap();

        let error = executor
            .run_route_batch_resident(&[lease], &[0], &[1.0], &[0, 1], &vec![0.0; hidden], 1)
            .unwrap_err();

        assert_eq!(error, CudaMoeError::PayloadLength { expected, actual });
    }

    #[test]
    fn cuda_q4_0_executor_matches_the_cpu_decoder_with_geometry_32() {
        let hidden = 64;
        let intermediate = 64;
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 + 1.0) * 0.01)
            .collect::<Vec<_>>();
        let payload = q4_payload(hidden, intermediate);
        let expected = Q4_0Expert::from_payload(hidden, intermediate, &payload)
            .unwrap()
            .run(&hidden_values)
            .unwrap();
        let executor = HeterogeneousCudaExecutor::new_with_cache(
            hidden,
            intermediate,
            measured_device_cache(4_000_000).unwrap(),
        );
        let lease = executor
            .resident_admit(ExpertKey::new("cuda-q4-0-test", 0, 0, "q4_0"), &payload)
            .unwrap();
        let gate_up_bytes = 2 * intermediate * (hidden / 32 * 18);
        let descriptor = CudaHeterogeneousExpert {
            gate_up: CudaHeterogeneousTensor {
                data: (lease.handle() as *const u8).wrapping_add(8),
                bytes: gate_up_bytes,
                rows: 2 * intermediate,
                columns: hidden,
                dtype: 2,
                ..CudaHeterogeneousTensor::default()
            },
            down: CudaHeterogeneousTensor {
                data: (lease.handle() as *const u8).wrapping_add(8 + gate_up_bytes),
                bytes: hidden * (intermediate / 32 * 18),
                rows: hidden,
                columns: intermediate,
                dtype: 2,
                ..CudaHeterogeneousTensor::default()
            },
            ..CudaHeterogeneousExpert::default()
        };
        let actual = executor
            .run_route_batch_resident(
                &[descriptor],
                &[0],
                &[0],
                &[1.0],
                &[0, 1],
                &hidden_values,
                1,
                1,
            )
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-3,
                "{actual} != {expected}"
            );
        }
    }

    fn plain_payload(format: PlainFormat, hidden: usize, intermediate: usize) -> Vec<u8> {
        let element_bytes = match format {
            PlainFormat::F32 => 4,
            PlainFormat::F16 | PlainFormat::Bf16 => 2,
        };
        let encode = |value: f32, output: &mut Vec<u8>| match format {
            PlainFormat::F32 => output.extend_from_slice(&value.to_le_bytes()),
            PlainFormat::F16 => {
                output.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes())
            }
            PlainFormat::Bf16 => {
                output.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes())
            }
        };
        let matrix = |rows: usize, columns: usize| {
            let mut output = Vec::with_capacity(rows * columns * element_bytes);
            for row in 0..rows {
                for column in 0..columns {
                    let value = ((row.wrapping_mul(3) + column) % 9) as f32 * 0.0625 - 0.25;
                    encode(value, &mut output);
                }
            }
            output
        };
        let gate_up = matrix(2 * intermediate, hidden);
        let down = matrix(hidden, intermediate);
        let mut payload = Vec::with_capacity(8 + gate_up.len() + down.len());
        payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
        payload.extend_from_slice(&gate_up);
        payload.extend_from_slice(&down);
        payload
    }

    fn iq_payload(format: IqQuantFormat, hidden: usize, intermediate: usize) -> Vec<u8> {
        let block_bytes = match format {
            IqQuantFormat::Iq3XXS => 98,
            IqQuantFormat::Iq3S => 110,
            IqQuantFormat::Iq2S => 82,
            IqQuantFormat::Iq4XS => 136,
            _ => panic!("fixture CUDA IQ não suportada"),
        };
        let row_bytes = |columns: usize| columns / 256 * block_bytes;
        let matrix = |rows: usize, columns: usize| {
            let mut bytes = vec![0u8; rows * row_bytes(columns)];
            for block in bytes.chunks_exact_mut(block_bytes) {
                block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                match format {
                    IqQuantFormat::Iq3S => {
                        block[2..66].iter_mut().enumerate().for_each(|(i, value)| {
                            *value = (i as u8).wrapping_mul(17) & 0x7f;
                        });
                        block[66..74].fill(0x55);
                        block[74..106].fill(0x33);
                        block[106..110].fill(0x21);
                    }
                    IqQuantFormat::Iq3XXS => {
                        block[2..66].iter_mut().enumerate().for_each(|(i, value)| {
                            *value = (i as u8).wrapping_mul(19);
                        });
                        block[66..98].iter_mut().enumerate().for_each(|(i, value)| {
                            *value = (i as u8).wrapping_mul(23);
                        });
                    }
                    IqQuantFormat::Iq2S => {
                        block[2..66].iter_mut().enumerate().for_each(|(i, value)| {
                            *value = (i as u8).wrapping_mul(29);
                        });
                        block[66..74].fill(0x55);
                        block[74..82].fill(0x21);
                    }
                    IqQuantFormat::Iq4XS => {
                        block[2..4].copy_from_slice(&0x02aau16.to_le_bytes());
                        block[4..8].fill(0x21);
                        block[8..136].iter_mut().enumerate().for_each(|(i, value)| {
                            *value = (i as u8).wrapping_mul(13);
                        });
                    }
                    _ => unreachable!(),
                }
            }
            bytes
        };
        let gate_up = matrix(2 * intermediate, hidden);
        let down = matrix(hidden, intermediate);
        let mut payload = Vec::with_capacity(8 + gate_up.len() + down.len());
        payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
        payload.extend_from_slice(&gate_up);
        payload.extend_from_slice(&down);
        payload
    }

    fn split_iq_payload(
        format: IqQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let payload = iq_payload(format, hidden, intermediate);
        let gate_up_len = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
        let down_len = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
        let gate_up = &payload[8..8 + gate_up_len];
        let gate_bytes = gate_up.len() / 2;
        assert_eq!(gate_bytes * 2, gate_up.len());
        assert_eq!(down_len, payload.len() - 8 - gate_up_len);
        (
            gate_up[..gate_bytes].to_vec(),
            gate_up[gate_bytes..].to_vec(),
            payload[8 + gate_up_len..].to_vec(),
        )
    }

    fn mxfp4_matrix(rows: usize, columns: usize, nibble: u8) -> Vec<u8> {
        let row_size = columns / 32 * 17;
        let mut bytes = vec![0u8; rows * row_size];
        for block in bytes.chunks_exact_mut(17) {
            block[0] = 128;
            block[1..].fill(nibble | (nibble << 4));
        }
        bytes
    }

    fn mxfp4_mat_vec(matrix: &[u8], rows: usize, columns: usize, input: &[f32]) -> Vec<f32> {
        let row_size = LowBitFormat::Mxfp4.row_bytes(columns).unwrap();
        (0..rows)
            .map(|row| {
                decode_low_bit_row(
                    LowBitFormat::Mxfp4,
                    &matrix[row * row_size..(row + 1) * row_size],
                    columns,
                )
                .unwrap()
                .into_iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum()
            })
            .collect()
    }

    fn iq_mat_vec(
        format: IqQuantFormat,
        matrix: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let row_size = row_bytes(format, columns).unwrap();
        (0..rows)
            .map(|row| {
                dequantize_iq_row(format, &matrix[row * row_size..(row + 1) * row_size])
                    .unwrap()
                    .iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect()
    }

    fn k_mat_vec(
        format: KQuantFormat,
        matrix: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let row_size = super::super::format_k_quants::row_bytes(format, columns).unwrap();
        (0..rows)
            .map(|row| {
                dequantize_k_row(format, &matrix[row * row_size..(row + 1) * row_size])
                    .unwrap()
                    .iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect()
    }

    fn run_resident(
        executor: &Nvfp4CudaExecutor,
        payloads: &[&[u8]],
        hidden: &[f32],
    ) -> Vec<Vec<f32>> {
        let leases = payloads
            .iter()
            .enumerate()
            .map(|(expert, payload)| {
                executor
                    .resident_admit(
                        super::super::cache::ExpertKey::new("cuda-test", 0, expert as u32, "nvfp4"),
                        payload,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        executor.run_batch_resident(&leases, hidden).unwrap()
    }

    #[test]
    fn cuda_nvfp4_executor_matches_the_cpu_decoder_forward() {
        let hidden = 64;
        let intermediate = 64;
        let payload = payload(hidden, intermediate);
        let hidden_values = vec![1.0f32; hidden];
        let cpu = Nvfp4Expert::from_payload(hidden, intermediate, &payload)
            .unwrap()
            .run(&hidden_values)
            .unwrap();
        let executor = Nvfp4CudaExecutor::new(hidden, intermediate, 1_000_000).unwrap();
        let gpu = run_resident(&executor, &[payload.as_slice()], &hidden_values)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu) {
            assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
        }
    }

    #[test]
    fn cuda_k_quants_executor_matches_cpu_decoder_q2k_q3k_q4k_q5k_and_q6k() {
        let hidden = 256;
        let intermediate = 256;
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 - 127.0) * 0.0078125)
            .collect::<Vec<_>>();
        for format in [
            KQuantFormat::Q2K,
            KQuantFormat::Q3K,
            KQuantFormat::Q4K,
            KQuantFormat::Q5K,
            KQuantFormat::Q6K,
        ] {
            let payload = k_payload(format, hidden, intermediate);
            let expected = KQuantExpert::from_payload(format, hidden, intermediate, &payload)
                .unwrap()
                .run(&hidden_values)
                .unwrap();
            let executor = KQuantCudaExecutor::new_with_cache(
                format,
                hidden,
                intermediate,
                measured_device_cache(1_000_000).unwrap(),
            )
            .unwrap();
            let lease = executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-k-test", 0, 0, format.name()),
                    &payload,
                )
                .unwrap();
            let actual = executor
                .run_route_batch_resident(&[lease], &[0], &[1.0], &[0, 1], &hidden_values, 1)
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                let absolute = (actual - expected).abs();
                let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
                assert!(
                    absolute <= 1.0e-3 || relative <= 1.0e-5,
                    "{actual} != {expected} (absolute={absolute}, relative={relative})"
                );
            }
        }
    }

    #[test]
    fn cuda_q4k_executor_matches_the_cpu_decoder_prefill_batch() {
        let format = KQuantFormat::Q4K;
        let hidden = 256;
        let intermediate = 256;
        let tokens = 3;
        let payloads = (0..2)
            .map(|_| k_payload(format, hidden, intermediate))
            .collect::<Vec<_>>();
        let hidden_values = (0..tokens * hidden)
            .map(|index| ((index as f32) * 0.013).sin())
            .collect::<Vec<_>>();
        let expected = (0..tokens)
            .flat_map(|token| {
                let token_hidden = &hidden_values[token * hidden..(token + 1) * hidden];
                let experts = payloads
                    .iter()
                    .map(|payload| {
                        KQuantExpert::from_payload(format, hidden, intermediate, payload)
                            .unwrap()
                            .run(token_hidden)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                (0..hidden).map(move |index| 0.25 * experts[0][index] + 0.75 * experts[1][index])
            })
            .collect::<Vec<_>>();
        let executor = KQuantCudaExecutor::new_with_cache(
            format,
            hidden,
            intermediate,
            measured_device_cache(4_000_000).unwrap(),
        )
        .unwrap();
        let model = "cuda-q4-batch";
        let leases = payloads
            .iter()
            .enumerate()
            .map(|(expert, payload)| {
                executor
                    .resident_admit(
                        super::super::cache::ExpertKey::new(model, 0, expert as u32, format.name()),
                        payload,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let route_leases = (0..tokens)
            .flat_map(|_| {
                (0..leases.len()).map(|expert| {
                    executor
                        .resident_lookup(&super::super::cache::ExpertKey::new(
                            model,
                            0,
                            expert as u32,
                            format.name(),
                        ))
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        let token_ids = (0..tokens)
            .flat_map(|token| [token as u32, token as u32])
            .collect::<Vec<_>>();
        let weights = (0..tokens)
            .flat_map(|_| [0.25f32, 0.75f32])
            .collect::<Vec<_>>();
        let token_offsets = (0..=tokens).map(|token| token * 2).collect::<Vec<_>>();
        let actual = executor
            .run_route_batch_resident(
                &route_leases,
                &token_ids,
                &weights,
                &token_offsets,
                &hidden_values,
                tokens,
            )
            .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            let absolute = (actual - expected).abs();
            let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
            assert!(
                absolute <= 1.0e-3 || relative <= 1.0e-5,
                "{actual} != {expected} (absolute={absolute}, relative={relative})"
            );
        }
    }

    #[test]
    fn cuda_q6_k_executor_gate_up_eight_experts_hidden_2048_is_finite() {
        let format = KQuantFormat::Q6K;
        let hidden = 2048;
        let intermediate = 512;
        let payloads = (0..8)
            .map(|_| k_payload(format, hidden, intermediate))
            .collect::<Vec<_>>();
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 * 0.001).sin())
            .collect::<Vec<_>>();
        let executor = KQuantCudaExecutor::new_with_cache(
            format,
            hidden,
            intermediate,
            measured_device_cache(64 * 1024 * 1024).unwrap(),
        )
        .unwrap();
        let leases = payloads
            .iter()
            .enumerate()
            .map(|(expert, payload)| {
                executor
                    .resident_admit(
                        super::super::cache::ExpertKey::new(
                            "cuda-q6-gate-up",
                            0,
                            expert as u32,
                            format.name(),
                        ),
                        payload,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let token_ids = vec![0u32; leases.len()];
        let weights = vec![1.0 / leases.len() as f32; leases.len()];

        let output = executor
            .run_route_batch_resident(
                &leases,
                &token_ids,
                &weights,
                &[0, leases.len()],
                &hidden_values,
                1,
            )
            .unwrap();

        assert_eq!(output.len(), hidden);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    #[ignore = "benchmark explícito do lote CUDA Q4_K e Q6_K"]
    fn benchmark_cuda_k_quants_prefill_batch() {
        for (label, format) in [("Q4_K", KQuantFormat::Q4K), ("Q6_K", KQuantFormat::Q6K)] {
            let hidden = 2048;
            let intermediate = 512;
            let tokens = 28;
            let model = format!("cuda-{label}-benchmark");
            let payloads = (0..8)
                .map(|_| k_payload(format, hidden, intermediate))
                .collect::<Vec<_>>();
            let executor = KQuantCudaExecutor::new_with_cache(
                format,
                hidden,
                intermediate,
                measured_device_cache(64 * 1024 * 1024).unwrap(),
            )
            .unwrap();
            let leases = payloads
                .iter()
                .enumerate()
                .map(|(expert, payload)| {
                    executor
                        .resident_admit(
                            super::super::cache::ExpertKey::new(
                                &model,
                                0,
                                expert as u32,
                                format.name(),
                            ),
                            payload,
                        )
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let mut hidden_values = Vec::with_capacity(tokens * hidden);
            let route_leases = (0..tokens)
                .flat_map(|_| {
                    (0..leases.len()).map(|expert| {
                        executor
                            .resident_lookup(&super::super::cache::ExpertKey::new(
                                &model,
                                0,
                                expert as u32,
                                format.name(),
                            ))
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            let mut token_ids = Vec::with_capacity(tokens * leases.len());
            let mut weights = Vec::with_capacity(tokens * leases.len());
            let mut token_offsets = Vec::with_capacity(tokens + 1);
            token_offsets.push(0);
            for token in 0..tokens {
                hidden_values.extend(
                    (0..hidden).map(|index| ((token * hidden + index) as f32 * 0.001).sin()),
                );
                token_ids.extend(std::iter::repeat_n(token as u32, leases.len()));
                weights.extend(std::iter::repeat_n(1.0 / leases.len() as f32, leases.len()));
                token_offsets.push(token_ids.len());
            }
            let started = Instant::now();
            let output = executor
                .run_route_batch_resident(
                    &route_leases,
                    &token_ids,
                    &weights,
                    &token_offsets,
                    &hidden_values,
                    tokens,
                )
                .unwrap();
            let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let diagnostics = executor.diagnostics();
            eprintln!(
                "CUDA {label} prefill benchmark tokens={tokens} routes={} wall_ms={wall_ms:.3} alloc_ms={:.3} kernel_ms={:.3} h2d_ms={:.3} d2h_ms={:.3}",
                token_ids.len(),
                diagnostics.alloc_ms,
                diagnostics.kernel_ms,
                diagnostics.h2d_ms,
                diagnostics.d2h_ms,
            );
            assert_eq!(output.len(), tokens * hidden);
            assert!(output.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn heterogeneous_cuda_k_quants_executor_matches_cpu_decoder_q2k_q3k_q4k() {
        let hidden = 256;
        let intermediate = 256;
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 - 127.0) * 0.0078125)
            .collect::<Vec<_>>();
        let gate_payload = k_payload(KQuantFormat::Q2K, hidden, intermediate);
        let up_payload = k_payload(KQuantFormat::Q3K, hidden, intermediate);
        let down_payload = k_payload(KQuantFormat::Q4K, hidden, intermediate);
        fn split(payload: &[u8]) -> (&[u8], &[u8], &[u8]) {
            let gate_up_len = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
            let matrix_start = 32;
            let down_start = matrix_start + gate_up_len;
            (
                &payload[matrix_start..matrix_start + gate_up_len / 2],
                &payload[matrix_start + gate_up_len / 2..down_start],
                &payload[down_start..],
            )
        }
        let (gate, _, _) = split(&gate_payload);
        let (_, up, _) = split(&up_payload);
        let (_, _, down) = split(&down_payload);
        let gate_values = k_mat_vec(
            KQuantFormat::Q2K,
            gate,
            intermediate,
            hidden,
            &hidden_values,
        );
        let up_values = k_mat_vec(KQuantFormat::Q3K, up, intermediate, hidden, &hidden_values);
        let activated = gate_values
            .iter()
            .zip(up_values)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        let expected = k_mat_vec(KQuantFormat::Q4K, down, hidden, intermediate, &activated);

        let executor = HeterogeneousCudaExecutor::new_with_cache(
            hidden,
            intermediate,
            measured_device_cache(1_000_000).unwrap(),
        );
        let gate_lease = executor
            .resident_admit(
                super::super::cache::ExpertKey::new("cuda-heterogeneous-k-test", 0, 0, "gate-10"),
                gate,
            )
            .unwrap();
        let up_lease = executor
            .resident_admit(
                super::super::cache::ExpertKey::new("cuda-heterogeneous-k-test", 0, 0, "up-11"),
                up,
            )
            .unwrap();
        let down_lease = executor
            .resident_admit(
                super::super::cache::ExpertKey::new("cuda-heterogeneous-k-test", 0, 0, "down-12"),
                down,
            )
            .unwrap();
        let descriptor = CudaHeterogeneousExpert {
            gate: CudaHeterogeneousTensor {
                data: gate_lease.handle() as *const u8,
                bytes: gate.len(),
                rows: intermediate,
                columns: hidden,
                dtype: 10,
                ..CudaHeterogeneousTensor::default()
            },
            up: CudaHeterogeneousTensor {
                data: up_lease.handle() as *const u8,
                bytes: up.len(),
                rows: intermediate,
                columns: hidden,
                dtype: 11,
                ..CudaHeterogeneousTensor::default()
            },
            down: CudaHeterogeneousTensor {
                data: down_lease.handle() as *const u8,
                bytes: down.len(),
                rows: hidden,
                columns: intermediate,
                dtype: 12,
                ..CudaHeterogeneousTensor::default()
            },
            ..CudaHeterogeneousExpert::default()
        };
        let actual = executor
            .run_route_batch_resident(
                &[descriptor],
                &[0],
                &[0],
                &[1.0],
                &[0, 1],
                &hidden_values,
                1,
                1,
            )
            .unwrap();
        let (max_absolute, max_relative) = actual.iter().zip(&expected).fold(
            (0.0f32, 0.0f32),
            |(max_absolute, max_relative), (actual, expected)| {
                let absolute = (actual - expected).abs();
                let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
                (max_absolute.max(absolute), max_relative.max(relative))
            },
        );
        const TOLERANCE: f32 = 1.0e-4;
        eprintln!(
            "CUDA K-quant heterogêneo: max_absolute={max_absolute:.6e}, max_relative={max_relative:.6e}, absolute_tolerance={TOLERANCE:.1e}"
        );
        assert_eq!(actual.len(), expected.len());
        assert!(max_absolute <= TOLERANCE || max_relative <= 1.0e-5);
    }

    #[test]
    fn cuda_plain_executor_matches_cpu_decoder_f32_f16_and_bf16() {
        let hidden = 64;
        let intermediate = 64;
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 - 31.0) * 0.03125)
            .collect::<Vec<_>>();
        for format in [PlainFormat::F32, PlainFormat::F16, PlainFormat::Bf16] {
            let payload = plain_payload(format, hidden, intermediate);
            let expected = PlainExpert::from_payload(format, hidden, intermediate, &payload)
                .unwrap()
                .run(&hidden_values)
                .unwrap();
            let executor = PlainCudaExecutor::new_with_cache(
                format,
                hidden,
                intermediate,
                measured_device_cache(1_000_000).unwrap(),
            )
            .unwrap();
            let lease = executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-plain-test", 0, 0, format.name()),
                    &payload,
                )
                .unwrap();
            let actual = executor
                .run_route_batch_resident(&[lease], &[0], &[1.0], &[0, 1], &hidden_values, 1)
                .unwrap();
            let tolerance = if matches!(format, PlainFormat::F32) {
                1e-5
            } else {
                2e-3
            };
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn cuda_iq_executor_matches_cpu_decoder_iq3xxs_iq3s_iq2s_and_iq4xs() {
        let hidden = 256;
        let intermediate = 256;
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 - 127.0) * 0.0078125)
            .collect::<Vec<_>>();
        let mut failures = Vec::new();
        for format in [
            IqQuantFormat::Iq3XXS,
            IqQuantFormat::Iq3S,
            IqQuantFormat::Iq2S,
            IqQuantFormat::Iq4XS,
        ] {
            let payload = iq_payload(format, hidden, intermediate);
            let expected = IqQuantExpert::from_payload(format, hidden, intermediate, &payload)
                .unwrap()
                .run(&hidden_values)
                .unwrap();
            let executor = IqQuantCudaExecutor::new_with_cache(
                format,
                hidden,
                intermediate,
                measured_device_cache(1_000_000).unwrap(),
            )
            .unwrap();
            let lease = executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-iq-test", 0, 0, format.name()),
                    &payload,
                )
                .unwrap();
            let actual = executor
                .run_route_batch_resident(&[lease], &[0], &[1.0], &[0, 1], &hidden_values, 1)
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            if let Some(difference) = first_numeric_difference(&actual, &expected, 1e-4) {
                eprintln!(
                    "IQ {}: primeiro erro índice={} actual={} esperado={} abs={} rel={}",
                    format.name(),
                    difference.index,
                    difference.actual,
                    difference.expected,
                    difference.absolute,
                    difference.relative
                );
                failures.push(format.name());
            }
        }
        assert!(failures.is_empty(), "IQ formats diverged: {failures:?}");
    }

    #[test]
    fn cuda_iq_diagnostic_divergence_per_stage() {
        let hidden = 256;
        let intermediate = 256;
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 - 127.0) * 0.0078125)
            .collect::<Vec<_>>();
        for format in [
            IqQuantFormat::Iq3XXS,
            IqQuantFormat::Iq3S,
            IqQuantFormat::Iq2S,
            IqQuantFormat::Iq4XS,
        ] {
            let payload = iq_payload(format, hidden, intermediate);
            let (gate, up, down) = split_iq_payload(format, hidden, intermediate);
            let gate_values = iq_mat_vec(format, &gate, intermediate, hidden, &hidden_values);
            let up_values = iq_mat_vec(format, &up, intermediate, hidden, &hidden_values);
            let expected_gate_up = gate_values
                .iter()
                .chain(&up_values)
                .copied()
                .collect::<Vec<_>>();
            let expected_activated = gate_values
                .iter()
                .zip(&up_values)
                .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
                .collect::<Vec<_>>();
            let expected_output =
                iq_mat_vec(format, &down, hidden, intermediate, &expected_activated);
            let executor = IqQuantCudaExecutor::new_with_cache(
                format,
                hidden,
                intermediate,
                measured_device_cache(1_000_000).unwrap(),
            )
            .unwrap();
            let lease = executor
                .resident_admit(
                    super::super::cache::ExpertKey::new(
                        "cuda-iq-stage-diagnostic",
                        0,
                        0,
                        format.name(),
                    ),
                    &payload,
                )
                .unwrap();
            let mut actual_gate_up = vec![0.0f32; 2 * intermediate];
            let mut actual_activated = vec![0.0f32; intermediate];
            let mut actual_output = vec![0.0f32; hidden];
            let status = unsafe {
                pom_moe_iquant_trace_single_resident(
                    0,
                    format.dtype(),
                    lease.handle() as *const u8,
                    hidden_values.as_ptr(),
                    hidden_values.len(),
                    actual_gate_up.as_mut_ptr(),
                    actual_gate_up.len() * std::mem::size_of::<f32>(),
                    actual_activated.as_mut_ptr(),
                    actual_activated.len() * std::mem::size_of::<f32>(),
                    actual_output.as_mut_ptr(),
                    actual_output.len() * std::mem::size_of::<f32>(),
                    hidden,
                    intermediate,
                )
            };
            assert_eq!(status, 0, "trace CUDA falhou para {}", format.name());
            let gate_up_error = first_numeric_difference(&actual_gate_up, &expected_gate_up, 1e-4);
            let activation_error =
                first_numeric_difference(&actual_activated, &expected_activated, 1e-4);
            let output_error = first_numeric_difference(&actual_output, &expected_output, 1e-4);
            eprintln!(
                "IQ {} estágios: gate_up={gate_up_error:?}, activation={activation_error:?}, down={output_error:?}",
                format.name(),
            );
        }
    }

    #[derive(Debug)]
    struct NumericDifference {
        index: usize,
        actual: f32,
        expected: f32,
        absolute: f32,
        relative: f32,
    }

    fn first_numeric_difference(
        actual: &[f32],
        expected: &[f32],
        tolerance: f32,
    ) -> Option<NumericDifference> {
        actual
            .iter()
            .zip(expected)
            .enumerate()
            .find_map(|(index, (&actual, &expected))| {
                let absolute = (actual - expected).abs();
                let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
                if !actual.is_finite()
                    || !expected.is_finite()
                    || absolute > tolerance && relative > tolerance
                {
                    Some(NumericDifference {
                        index,
                        actual,
                        expected,
                        absolute,
                        relative,
                    })
                } else {
                    None
                }
            })
    }

    #[test]
    fn heterogeneous_cuda_iq_executor_preserves_gate_up_down_per_dtype() {
        let hidden = 256;
        let intermediate = 256;
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 - 127.0) * 0.0078125)
            .collect::<Vec<_>>();
        let (gate, _, _) = split_iq_payload(IqQuantFormat::Iq3S, hidden, intermediate);
        let (_, up, _) = split_iq_payload(IqQuantFormat::Iq2S, hidden, intermediate);
        let (_, _, down) = split_iq_payload(IqQuantFormat::Iq4XS, hidden, intermediate);
        let gate_values = iq_mat_vec(
            IqQuantFormat::Iq3S,
            &gate,
            intermediate,
            hidden,
            &hidden_values,
        );
        let up_values = iq_mat_vec(
            IqQuantFormat::Iq2S,
            &up,
            intermediate,
            hidden,
            &hidden_values,
        );
        let activated = gate_values
            .iter()
            .zip(up_values)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        let expected = iq_mat_vec(
            IqQuantFormat::Iq4XS,
            &down,
            hidden,
            intermediate,
            &activated,
        );

        let executor = HeterogeneousCudaExecutor::new_with_cache(
            hidden,
            intermediate,
            measured_device_cache(1_000_000).unwrap(),
        );
        let gate_lease = executor
            .resident_admit(
                super::super::cache::ExpertKey::new("cuda-heterogeneous-test", 0, 0, "gate-21"),
                &gate,
            )
            .unwrap();
        let up_lease = executor
            .resident_admit(
                super::super::cache::ExpertKey::new("cuda-heterogeneous-test", 0, 0, "up-22"),
                &up,
            )
            .unwrap();
        let down_lease = executor
            .resident_admit(
                super::super::cache::ExpertKey::new("cuda-heterogeneous-test", 0, 0, "down-23"),
                &down,
            )
            .unwrap();
        let descriptor = CudaHeterogeneousExpert {
            gate: CudaHeterogeneousTensor {
                data: gate_lease.handle() as *const u8,
                bytes: gate.len(),
                rows: intermediate,
                columns: hidden,
                dtype: 21,
                ..CudaHeterogeneousTensor::default()
            },
            up: CudaHeterogeneousTensor {
                data: up_lease.handle() as *const u8,
                bytes: up.len(),
                rows: intermediate,
                columns: hidden,
                dtype: 22,
                ..CudaHeterogeneousTensor::default()
            },
            down: CudaHeterogeneousTensor {
                data: down_lease.handle() as *const u8,
                bytes: down.len(),
                rows: hidden,
                columns: intermediate,
                dtype: 23,
                ..CudaHeterogeneousTensor::default()
            },
            ..CudaHeterogeneousExpert::default()
        };
        let actual = executor
            .run_route_batch_resident(
                &[descriptor],
                &[0],
                &[0],
                &[1.0],
                &[0, 1],
                &hidden_values,
                1,
                1,
            )
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
        }
    }

    #[test]
    fn heterogeneous_cuda_executor_preserves_deepseek_iq3xxs_and_mxfp4() {
        let hidden = 256;
        let intermediate = 256;
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 - 127.0) * 0.0078125)
            .collect::<Vec<_>>();
        let (gate, _, _) = split_iq_payload(IqQuantFormat::Iq3XXS, hidden, intermediate);
        let (_, up, _) = split_iq_payload(IqQuantFormat::Iq3XXS, hidden, intermediate);
        let down = mxfp4_matrix(hidden, intermediate, 0x22);
        let gate_values = iq_mat_vec(
            IqQuantFormat::Iq3XXS,
            &gate,
            intermediate,
            hidden,
            &hidden_values,
        );
        let up_values = iq_mat_vec(
            IqQuantFormat::Iq3XXS,
            &up,
            intermediate,
            hidden,
            &hidden_values,
        );
        let activated = gate_values
            .iter()
            .zip(up_values)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        let expected = mxfp4_mat_vec(&down, hidden, intermediate, &activated);

        let executor = HeterogeneousCudaExecutor::new_with_cache(
            hidden,
            intermediate,
            measured_device_cache(1_000_000).unwrap(),
        );
        let gate_lease = executor
            .resident_admit(
                super::super::cache::ExpertKey::new("cuda-deepseek-heterogeneous", 0, 0, "gate-18"),
                &gate,
            )
            .unwrap();
        let up_lease = executor
            .resident_admit(
                super::super::cache::ExpertKey::new("cuda-deepseek-heterogeneous", 0, 0, "up-18"),
                &up,
            )
            .unwrap();
        let down_lease = executor
            .resident_admit(
                super::super::cache::ExpertKey::new("cuda-deepseek-heterogeneous", 0, 0, "down-39"),
                &down,
            )
            .unwrap();
        let descriptor = CudaHeterogeneousExpert {
            gate: CudaHeterogeneousTensor {
                data: gate_lease.handle() as *const u8,
                bytes: gate.len(),
                rows: intermediate,
                columns: hidden,
                dtype: 18,
                ..CudaHeterogeneousTensor::default()
            },
            up: CudaHeterogeneousTensor {
                data: up_lease.handle() as *const u8,
                bytes: up.len(),
                rows: intermediate,
                columns: hidden,
                dtype: 18,
                ..CudaHeterogeneousTensor::default()
            },
            down: CudaHeterogeneousTensor {
                data: down_lease.handle() as *const u8,
                bytes: down.len(),
                rows: hidden,
                columns: intermediate,
                dtype: 39,
                ..CudaHeterogeneousTensor::default()
            },
            ..CudaHeterogeneousExpert::default()
        };
        let actual = executor
            .run_route_batch_resident(
                &[descriptor],
                &[0],
                &[0],
                &[1.0],
                &[0, 1],
                &hidden_values,
                1,
                1,
            )
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
        }
    }

    #[test]
    fn cuda_nvfp4_executor_preserves_q8_quantization_and_fp16_gelu() {
        let hidden = 64;
        let intermediate = 64;
        let payload = payload(hidden, intermediate);
        let hidden_values = vec![0.125f32; hidden];
        let cpu = Nvfp4Expert::from_payload(hidden, intermediate, &payload)
            .unwrap()
            .run(&hidden_values)
            .unwrap();
        let executor = Nvfp4CudaExecutor::new(hidden, intermediate, 1_000_000).unwrap();
        let gpu = run_resident(&executor, &[payload.as_slice()], &hidden_values)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu) {
            assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
        }
    }

    #[test]
    fn cuda_nvfp4_batch_executor_preserves_each_expert() {
        let hidden = 64;
        let intermediate = 64;
        let first = payload_with_nibble(hidden, intermediate, 0x22);
        let second = payload_with_nibble(hidden, intermediate, 0x33);
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 - 31.0) * 0.03125)
            .collect::<Vec<_>>();
        let cpu = [&first, &second]
            .iter()
            .map(|payload| {
                Nvfp4Expert::from_payload(hidden, intermediate, payload)
                    .unwrap()
                    .run(&hidden_values)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let executor = Nvfp4CudaExecutor::new(hidden, intermediate, 1_000_000).unwrap();
        let gpu = run_resident(
            &executor,
            &[first.as_slice(), second.as_slice()],
            &hidden_values,
        );
        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu) {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn cuda_nvfp4_executor_route_batch_aggregates_tokens_on_the_device() {
        let hidden = 64;
        let intermediate = 64;
        let first = payload_with_nibble(hidden, intermediate, 0x22);
        let second = payload_with_nibble(hidden, intermediate, 0x33);
        let hidden_values = [
            (0..hidden)
                .map(|index| (index as f32 - 31.0) * 0.03125)
                .collect::<Vec<_>>(),
            (0..hidden)
                .map(|index| (15.0 - index as f32) * 0.0625)
                .collect::<Vec<_>>(),
        ];
        let cpu = [
            (&first, &hidden_values[0], [0.25f32, 0.75f32]),
            (&second, &hidden_values[0], [0.60f32, 0.40f32]),
        ];
        let expected = [
            [
                Nvfp4Expert::from_payload(hidden, intermediate, cpu[0].0)
                    .unwrap()
                    .run(cpu[0].1)
                    .unwrap(),
                Nvfp4Expert::from_payload(hidden, intermediate, cpu[1].0)
                    .unwrap()
                    .run(cpu[1].1)
                    .unwrap(),
            ],
            [
                Nvfp4Expert::from_payload(hidden, intermediate, &second)
                    .unwrap()
                    .run(&hidden_values[1])
                    .unwrap(),
                Nvfp4Expert::from_payload(hidden, intermediate, &first)
                    .unwrap()
                    .run(&hidden_values[1])
                    .unwrap(),
            ],
        ];
        let expected = [
            expected[0][0]
                .iter()
                .zip(&expected[0][1])
                .map(|(first, second)| 0.25 * first + 0.75 * second)
                .collect::<Vec<_>>(),
            expected[1][0]
                .iter()
                .zip(&expected[1][1])
                .map(|(first, second)| 0.60 * first + 0.40 * second)
                .collect::<Vec<_>>(),
        ];
        let executor = Nvfp4CudaExecutor::new(hidden, intermediate, 1_000_000).unwrap();
        let leases = vec![
            executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-route-test", 0, 0, "nvfp4"),
                    &first,
                )
                .unwrap(),
            executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-route-test", 0, 1, "nvfp4"),
                    &second,
                )
                .unwrap(),
            executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-route-test", 0, 1, "nvfp4"),
                    &second,
                )
                .unwrap(),
            executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-route-test", 0, 0, "nvfp4"),
                    &first,
                )
                .unwrap(),
        ];
        let actual = executor
            .run_route_batch_resident(
                &leases,
                &[0, 0, 1, 1],
                &[0.25, 0.75, 0.60, 0.40],
                &[0, 2, 4],
                &hidden_values.concat(),
                2,
            )
            .unwrap();
        let diagnostics = executor.diagnostics();
        assert_eq!(diagnostics.buffer_mask, 3);
        assert!(diagnostics.buffer_switches > 0);
        for (actual, expected) in actual.chunks_exact(hidden).zip(expected) {
            eprintln!(
                "NVFP4 route: actual[0..4]={:?} expected[0..4]={:?}",
                &actual[..4],
                &expected[..4]
            );
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn cuda_nvfp4_executor_one_expert_per_token_route_isolates_multitoken() {
        let hidden = 64;
        let intermediate = 64;
        let first = payload_with_nibble(hidden, intermediate, 0x22);
        let second = payload_with_nibble(hidden, intermediate, 0x33);
        let hidden_values = [
            (0..hidden)
                .map(|index| (index as f32 - 31.0) * 0.03125)
                .collect::<Vec<_>>(),
            (0..hidden)
                .map(|index| (15.0 - index as f32) * 0.0625)
                .collect::<Vec<_>>(),
        ];
        let expected = [
            Nvfp4Expert::from_payload(hidden, intermediate, &first)
                .unwrap()
                .run(&hidden_values[0])
                .unwrap(),
            Nvfp4Expert::from_payload(hidden, intermediate, &second)
                .unwrap()
                .run(&hidden_values[1])
                .unwrap(),
        ];
        let executor = Nvfp4CudaExecutor::new(hidden, intermediate, 1_000_000).unwrap();
        let leases = vec![
            executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-route-one", 0, 0, "nvfp4"),
                    &first,
                )
                .unwrap(),
            executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-route-one", 0, 1, "nvfp4"),
                    &second,
                )
                .unwrap(),
        ];
        let actual = executor
            .run_route_batch_resident(
                &leases,
                &[0, 1],
                &[1.0, 1.0],
                &[0, 1, 2],
                &hidden_values.concat(),
                2,
            )
            .unwrap();
        for (token_index, (actual, expected)) in
            actual.chunks_exact(hidden).zip(expected).enumerate()
        {
            eprintln!(
                "NVFP4 one route token {token_index}: actual[0..4]={:?} expected[0..4]={:?}",
                &actual[..4],
                &expected[..4]
            );
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn cuda_nvfp4_executor_multitoken_route_without_repeated_keys() {
        let hidden = 64;
        let intermediate = 64;
        let first = payload_with_nibble(hidden, intermediate, 0x22);
        let second = payload_with_nibble(hidden, intermediate, 0x33);
        let hidden_values = [
            (0..hidden)
                .map(|index| (index as f32 - 31.0) * 0.03125)
                .collect::<Vec<_>>(),
            (0..hidden)
                .map(|index| (15.0 - index as f32) * 0.0625)
                .collect::<Vec<_>>(),
        ];
        let expected = [
            (
                [
                    Nvfp4Expert::from_payload(hidden, intermediate, &first)
                        .unwrap()
                        .run(&hidden_values[0])
                        .unwrap(),
                    Nvfp4Expert::from_payload(hidden, intermediate, &second)
                        .unwrap()
                        .run(&hidden_values[0])
                        .unwrap(),
                ],
                [0.25, 0.75],
            ),
            (
                [
                    Nvfp4Expert::from_payload(hidden, intermediate, &second)
                        .unwrap()
                        .run(&hidden_values[1])
                        .unwrap(),
                    Nvfp4Expert::from_payload(hidden, intermediate, &first)
                        .unwrap()
                        .run(&hidden_values[1])
                        .unwrap(),
                ],
                [0.60, 0.40],
            ),
        ]
        .map(|(pair, weights)| {
            pair[0]
                .iter()
                .zip(&pair[1])
                .map(|(first, second)| weights[0] * first + weights[1] * second)
                .collect::<Vec<_>>()
        });
        let executor = Nvfp4CudaExecutor::new(hidden, intermediate, 1_000_000).unwrap();
        let leases = [(&first, 0), (&second, 1), (&second, 2), (&first, 3)]
            .into_iter()
            .map(|(payload, expert)| {
                executor
                    .resident_admit(
                        super::super::cache::ExpertKey::new(
                            "cuda-route-unique",
                            0,
                            expert,
                            "nvfp4",
                        ),
                        payload,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let actual = executor
            .run_route_batch_resident(
                &leases,
                &[0, 0, 1, 1],
                &[0.25, 0.75, 0.60, 0.40],
                &[0, 2, 4],
                &hidden_values.concat(),
                2,
            )
            .unwrap();
        for (actual, expected) in actual.chunks_exact(hidden).zip(expected) {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn cuda_nvfp4_executor_two_experts_per_token_route_isolates_the_batch() {
        let hidden = 64;
        let intermediate = 64;
        let first = payload_with_nibble(hidden, intermediate, 0x22);
        let second = payload_with_nibble(hidden, intermediate, 0x33);
        let hidden_values = [
            (0..hidden)
                .map(|index| (index as f32 - 31.0) * 0.03125)
                .collect::<Vec<_>>(),
            (0..hidden)
                .map(|index| (15.0 - index as f32) * 0.0625)
                .collect::<Vec<_>>(),
        ];
        let executor = Nvfp4CudaExecutor::new(hidden, intermediate, 1_000_000).unwrap();
        let leases = vec![
            executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-route-isolation", 0, 0, "nvfp4"),
                    &first,
                )
                .unwrap(),
            executor
                .resident_admit(
                    super::super::cache::ExpertKey::new("cuda-route-isolation", 0, 1, "nvfp4"),
                    &second,
                )
                .unwrap(),
        ];
        for (token_index, token) in hidden_values.iter().enumerate() {
            let expected = [
                Nvfp4Expert::from_payload(hidden, intermediate, &first)
                    .unwrap()
                    .run(token)
                    .unwrap(),
                Nvfp4Expert::from_payload(hidden, intermediate, &second)
                    .unwrap()
                    .run(token)
                    .unwrap(),
            ];
            let expected = expected[0]
                .iter()
                .zip(&expected[1])
                .map(|(first, second)| 0.25 * first + 0.75 * second)
                .collect::<Vec<_>>();
            let actual = executor
                .run_route_batch_resident(&leases, &[0, 0], &[0.25, 0.75], &[0, 2], token, 1)
                .unwrap();
            assert_eq!(actual.len(), expected.len(), "token {token_index}");
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 1e-4,
                    "token {token_index}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a real NVFP4 GGUF and a CUDA device"]
    fn cuda_nvfp4_executor_batch_preserves_the_real_gguf_geometry() {
        let path = std::env::var_os("POM_MOE_NUMERIC_MODEL")
            .expect("POM_MOE_NUMERIC_MODEL must name a real MoE GGUF");
        let path = std::path::PathBuf::from(path);
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { Mmap::map(&file) }.unwrap();
        let (_, tensors) = parse_tensor_table(&mmap).unwrap();
        let experts = tensors
            .iter()
            .find(|tensor| {
                tensor.name.ends_with("ffn_gate_up_exps.weight")
                    || tensor.name.ends_with("ffn_gate_exps.weight")
            })
            .and_then(|tensor| tensor.dims.get(2).copied())
            .and_then(|count| u32::try_from(count).ok())
            .expect("real GGUF must expose expert count in the gate tensor");
        drop(mmap);

        let source = Nvfp4Source::open_infer(&path, experts).unwrap();
        let hidden = (0..source.hidden())
            .map(|index| (index as f32 % 17.0 - 8.0) * 0.03125)
            .collect::<Vec<_>>();
        let payloads = [
            source.load_expert(0, 0).unwrap(),
            source.load_expert(0, 1).unwrap(),
        ];
        let cpu = payloads
            .iter()
            .map(|payload| {
                Nvfp4Expert::from_payload(source.hidden(), source.intermediate(), payload)
                    .unwrap()
                    .run(&hidden)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let executor =
            Nvfp4CudaExecutor::new(source.hidden(), source.intermediate(), 1_000_000).unwrap();
        let gpu = run_resident(
            &executor,
            &[payloads[0].as_slice(), payloads[1].as_slice()],
            &hidden,
        );

        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.iter().zip(cpu) {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 2.0e-3,
                    "real GGUF CUDA batch diverged: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn cuda_diagnostic_accumulates_phases_and_bytes_per_call() {
        let mut total = Nvfp4CudaDiagnostics::default();
        total.add_call(CudaMoeCallTiming {
            expert_calls: 1,
            kernel_launches: 2,
            sync_count: 2,
            payload_validation_ms: 1.0,
            alloc_ms: 2.0,
            h2d_ms: 3.0,
            h2d_bytes: 10,
            kernel_ms: 4.0,
            kernel_span_ms: 4.0,
            launch_ms: 5.0,
            sync_ms: 6.0,
            d2h_ms: 7.0,
            d2h_bytes: 8,
            transfer_span_ms: 10.0,
            free_ms: 9.0,
            buffer_mask: 1,
            buffer_switches: 1,
        });
        total.add_call(CudaMoeCallTiming {
            expert_calls: 1,
            kernel_launches: 2,
            sync_count: 2,
            payload_validation_ms: 10.0,
            alloc_ms: 20.0,
            h2d_ms: 30.0,
            h2d_bytes: 100,
            kernel_ms: 40.0,
            kernel_span_ms: 40.0,
            launch_ms: 50.0,
            sync_ms: 60.0,
            d2h_ms: 70.0,
            d2h_bytes: 80,
            transfer_span_ms: 100.0,
            free_ms: 90.0,
            buffer_mask: 2,
            buffer_switches: 1,
        });

        assert_eq!(total.calls, 2);
        assert_eq!(total.kernel_launches, 4);
        assert_eq!(total.sync_count, 4);
        assert_eq!(total.h2d_bytes, 110);
        assert_eq!(total.d2h_bytes, 88);
        assert_eq!(total.h2d_ms, 33.0);
        assert_eq!(total.kernel_ms, 44.0);
        assert_eq!(total.kernel_span_ms, 44.0);
        assert_eq!(total.sync_ms, 66.0);
        assert_eq!(total.transfer_span_ms, 110.0);
        assert_eq!(total.buffer_mask, 3);
        assert_eq!(total.buffer_switches, 2);
    }

    #[test]
    fn cuda_diagnostic_separates_execution_by_ordinal() {
        let mut diagnostics = Nvfp4CudaDiagnostics::default();
        diagnostics.add_device_call(1, 2, 3.0, 4.0);
        diagnostics.add_device_call(0, 1, 5.0, 6.0);
        diagnostics.add_device_call(1, 4, 7.0, 8.0);

        assert_eq!(
            diagnostics.device_calls,
            vec![
                CudaMoeDeviceDiagnostics {
                    device_index: 1,
                    calls: 2,
                    expert_calls: 6,
                    kernel_ms: 10.0,
                    cuda_call_wall_ms: 12.0,
                },
                CudaMoeDeviceDiagnostics {
                    device_index: 0,
                    calls: 1,
                    expert_calls: 1,
                    kernel_ms: 5.0,
                    cuda_call_wall_ms: 6.0,
                },
            ]
        );
    }

    #[test]
    fn cuda_profile_persists_and_refuses_cuda_when_the_measured_cpu_is_faster() {
        let model =
            std::env::temp_dir().join(format!("pom-cuda-profile-{}.gguf", std::process::id()));
        let sidecar = CudaTransferProfile::sidecar_path(&model);
        let profile = CudaTransferProfile::new(2_816, 704, 12.0, 8.0, 2.0, Some(8.0), 3).unwrap();
        profile.save(&sidecar).unwrap();

        let loaded = CudaTransferProfile::load_for_model(&model)
            .unwrap()
            .unwrap();
        assert_eq!(loaded, profile);
        assert!(!loaded.prefers_cuda());

        std::fs::remove_file(sidecar).unwrap();
    }

    #[test]
    fn cuda_profile_persists_bands_for_coexecution() {
        let model = std::env::temp_dir().join(format!(
            "pom-cuda-bandwidth-profile-{}.gguf",
            std::process::id()
        ));
        let sidecar = CudaTransferProfile::sidecar_path(&model);
        let bandwidth = crate::moe::bandwidth::BandwidthProfile {
            cpu_overlap_bytes_per_sec: 3.0,
            transfer_overlap_bytes_per_sec: 1.0,
            cpu_bytes_per_sec: 30.0,
            transfer_bytes_per_sec: 10.0,
        };
        let profile = CudaTransferProfile::new(2_816, 704, 12.0, 8.0, 2.0, None, 3)
            .unwrap()
            .with_bandwidth(bandwidth)
            .unwrap();
        profile.save(&sidecar).unwrap();
        let loaded = CudaTransferProfile::load_for_model(&model)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.bandwidth, Some(bandwidth));
        assert!((loaded.bandwidth.unwrap().gpu_fraction().unwrap() - 0.25).abs() < 1e-6);
        std::fs::remove_file(sidecar).unwrap();
    }

    #[test]
    fn cuda_profile_refuses_an_invalid_band_and_absence_means_no_profile() {
        let model = std::env::temp_dir().join(format!(
            "pom-cuda-invalid-bandwidth-profile-{}.gguf",
            std::process::id()
        ));
        let sidecar = CudaTransferProfile::sidecar_path(&model);
        assert_eq!(CudaTransferProfile::load_for_model(&model).unwrap(), None);
        let invalid = serde_json::json!({
            "version": 1,
            "hidden_width": 2816,
            "intermediate_width": 704,
            "cuda_wall_ms": 12.0,
            "cuda_kernel_span_ms": 8.0,
            "cuda_transfer_span_ms": 2.0,
            "cpu_wall_ms": null,
            "buffer_mask": 3,
            "bandwidth": {
                "cpu_overlap_bytes_per_sec": -1.0,
                "transfer_overlap_bytes_per_sec": 1.0,
                "cpu_bytes_per_sec": -1.0,
                "transfer_bytes_per_sec": 1.0
            }
        });
        std::fs::write(&sidecar, serde_json::to_vec(&invalid).unwrap()).unwrap();
        let error = CudaTransferProfile::load_for_model(&model).unwrap_err();
        assert!(error.contains("perfil CUDA inválido"));
        std::fs::remove_file(sidecar).unwrap();
    }

    #[test]
    fn heterogeneous_kernel_does_not_dispatch_vec_dot_through_an_indirect_pointer() {
        let source = include_str!("cuda_kernels.cu");
        assert!(
            !source.contains("using HeterogeneousVecDotQ8_1"),
            "o hot path não pode chamar vec_dot por ponteiro de função CUDA"
        );
        assert!(
            source.contains("heterogeneous_quant_partial_dot_q8_1_direct"),
            "o kernel deve manter um dispatcher CUDA com chamadas diretas por dtype"
        );
    }

    #[test]
    #[ignore = "medição CUDA explícita; execute com POM_MOE_CUDA_PROFILE=1"]
    fn measures_the_h2d_d2h_band_of_the_nvfp4_batch() {
        let cases = [
            ("tiny", 64, 64, 2, 2),
            ("gemma4_token", 2_816, 704, 1, 8),
            ("gemma4_batch8", 2_816, 704, 8, 8),
        ];

        for (label, hidden, intermediate, tokens, active_experts) in cases {
            let executor = Nvfp4CudaExecutor::new(hidden, intermediate, 1_000_000).unwrap();
            let model = format!("cuda-bandwidth-test-{label}");
            let mut keys = Vec::with_capacity(active_experts);
            let mut admitted = Vec::with_capacity(active_experts);
            for expert in 0..active_experts {
                let key =
                    super::super::cache::ExpertKey::new(model.clone(), 0, expert as u32, "nvfp4");
                let payload = payload_with_nibble(
                    hidden,
                    intermediate,
                    if expert % 2 == 0 { 0x22 } else { 0x33 },
                );
                admitted.push(executor.resident_admit(key.clone(), &payload).unwrap());
                keys.push(key);
            }

            let mut route_leases = Vec::with_capacity(tokens * active_experts);
            let mut token_ids = Vec::with_capacity(tokens * active_experts);
            let mut weights = Vec::with_capacity(tokens * active_experts);
            let mut token_offsets = Vec::with_capacity(tokens + 1);
            token_offsets.push(0);
            for token in 0..tokens {
                for key in &keys {
                    route_leases.push(executor.resident_lookup(key).unwrap());
                    token_ids.push(token as u32);
                    weights.push(1.0 / active_experts as f32);
                }
                token_offsets.push(route_leases.len());
            }
            drop(admitted);

            let hidden_values = (0..tokens * hidden)
                .map(|index| (index as f32 % 31.0 - 15.0) * 0.03125)
                .collect::<Vec<_>>();
            for _ in 0..100 {
                executor
                    .run_route_batch_resident(
                        &route_leases,
                        &token_ids,
                        &weights,
                        &token_offsets,
                        &hidden_values,
                        tokens,
                    )
                    .unwrap();
            }

            let diagnostics = executor.diagnostics();
            assert!(diagnostics.h2d_bytes > 0);
            assert!(diagnostics.d2h_bytes > 0);
            assert!(diagnostics.h2d_ms > 0.0);
            assert!(diagnostics.d2h_ms > 0.0);
            let h2d_gb_s = diagnostics.h2d_bytes as f64 / (diagnostics.h2d_ms * 1_000_000.0);
            let d2h_gb_s = diagnostics.d2h_bytes as f64 / (diagnostics.d2h_ms * 1_000_000.0);
            eprintln!(
                "CUDA_BANDWIDTH case={label} hidden={hidden} intermediate={intermediate} tokens={tokens} active_experts={active_experts} calls={} buffer_mask={} buffer_switches={} h2d_bytes={} h2d_ms={:.3} effective_h2d_GB_s={:.3} d2h_bytes={} d2h_ms={:.3} effective_d2h_GB_s={:.3} kernel_ms={:.3} kernel_span_ms={:.3} transfer_span_ms={:.3} wall_ms={:.3}",
                diagnostics.calls,
                diagnostics.buffer_mask,
                diagnostics.buffer_switches,
                diagnostics.h2d_bytes,
                diagnostics.h2d_ms,
                h2d_gb_s,
                diagnostics.d2h_bytes,
                diagnostics.d2h_ms,
                d2h_gb_s,
                diagnostics.kernel_ms,
                diagnostics.kernel_span_ms,
                diagnostics.transfer_span_ms,
                diagnostics.cuda_call_wall_ms,
            );
        }
    }

    #[test]
    #[ignore = "medição CUDA explícita; execute com POM_MOE_CUDA_PROFILE=1"]
    fn measures_the_nvfp4_coexecution_profile() {
        let hidden = 2_816;
        let intermediate = 704;
        let active_experts = 8;
        let payloads = (0..active_experts)
            .map(|expert| {
                payload_with_nibble(
                    hidden,
                    intermediate,
                    if expert % 2 == 0 { 0x22 } else { 0x33 },
                )
            })
            .collect::<Vec<_>>();
        let payload_bytes = payloads[0].len() as u64;
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 % 31.0 - 15.0) * 0.03125)
            .collect::<Vec<_>>();
        let shared_payloads = std::sync::Arc::new(payloads);
        let cpu_payloads = std::sync::Arc::clone(&shared_payloads);
        let gpu_payloads = std::sync::Arc::clone(&shared_payloads);
        let cpu_hidden = hidden_values.clone();
        let gpu_hidden = hidden_values;
        let (cpu_ms, (gpu_ms, upload_ms, upload_bytes, cuda)) =
            super::super::coexecution::run_overlap(
                move || {
                    let started = Instant::now();
                    let mut checksum = 0.0f32;
                    for payload in cpu_payloads.iter() {
                        let output = Nvfp4Expert::from_payload(hidden, intermediate, payload)
                            .unwrap()
                            .run(&cpu_hidden)
                            .unwrap();
                        checksum += output[0];
                    }
                    std::hint::black_box(checksum);
                    Ok::<_, String>(started.elapsed().as_secs_f64() * 1_000.0)
                },
                move || {
                    let started = Instant::now();
                    let executor = Nvfp4CudaExecutor::new(hidden, intermediate, 1_000_000)
                        .map_err(|error| error.to_string())?;
                    let model = "cuda-coexecution-bandwidth";
                    let mut leases = Vec::with_capacity(active_experts);
                    for (expert, payload) in gpu_payloads.iter().enumerate() {
                        leases.push(
                            executor
                                .resident_admit(
                                    super::super::cache::ExpertKey::new(
                                        model,
                                        0,
                                        expert as u32,
                                        "nvfp4",
                                    ),
                                    payload,
                                )
                                .map_err(|error| error.to_string())?,
                        );
                    }
                    let route_ids = vec![0u32; active_experts];
                    let route_weights = vec![1.0 / active_experts as f32; active_experts];
                    executor
                        .run_route_batch_resident(
                            &leases,
                            &route_ids,
                            &route_weights,
                            &[0, active_experts],
                            &gpu_hidden,
                            1,
                        )
                        .map_err(|error| error.to_string())?;
                    let stats = executor.resident_cache_stats();
                    Ok::<_, String>((
                        started.elapsed().as_secs_f64() * 1_000.0,
                        stats.upload_ms,
                        stats.upload_bytes,
                        executor.diagnostics(),
                    ))
                },
            )
            .unwrap();
        assert_eq!(upload_bytes, payload_bytes * active_experts as u64);
        assert!(cpu_ms > 0.0 && gpu_ms > 0.0 && upload_ms > 0.0);
        let bandwidth = BandwidthProfile {
            cpu_overlap_bytes_per_sec: payload_bytes as f64 * active_experts as f64
                / (cpu_ms / 1_000.0),
            transfer_overlap_bytes_per_sec: upload_bytes as f64 / (upload_ms / 1_000.0),
            cpu_bytes_per_sec: payload_bytes as f64 * active_experts as f64 / (cpu_ms / 1_000.0),
            transfer_bytes_per_sec: upload_bytes as f64 / (upload_ms / 1_000.0),
        };
        let profile = CudaTransferProfile::new(
            hidden,
            intermediate,
            gpu_ms,
            cuda.kernel_span_ms,
            cuda.transfer_span_ms,
            Some(cpu_ms),
            cuda.buffer_mask,
        )
        .unwrap()
        .with_bandwidth(bandwidth)
        .unwrap();
        eprintln!(
            "COEXEC_PROFILE cpu_ms={cpu_ms:.3} gpu_ms={gpu_ms:.3} upload_ms={upload_ms:.3} payload_bytes={payload_bytes} gpu_fraction={:.4} json={}",
            bandwidth.gpu_fraction().unwrap(),
            serde_json::to_string(&profile).unwrap()
        );
    }

    #[test]
    #[ignore = "requer uma GPU CUDA para exercitar a ABI do pipeline"]
    fn resident_heterogeneous_pipeline_exposes_a_persistent_handle() {
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { pom_moe_heterogeneous_route_pipeline_create(0, 256, 256, &mut handle) };
        assert_eq!(status, 0);
        assert!(!handle.is_null());
        unsafe { pom_moe_heterogeneous_route_pipeline_destroy(handle) };
    }
}
