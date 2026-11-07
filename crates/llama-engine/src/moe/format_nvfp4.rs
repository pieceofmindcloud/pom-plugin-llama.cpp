//! Executor CPU de experts NVFP4.
//!
//! O layout é o `block_nvfp4` do ggml: quatro escalas UE4M3 por bloco de 64
//! valores, uma para cada subbloco de 16, seguidas por 32 bytes de nibbles
//! E2M1. As escalas `.scale` e `.input_scale` por expert são preservadas no
//! payload do cache, em vez de serem confundidas com os bytes quantizados.
//! Nesta operação, a semântica nativa do llama.cpp usa apenas `.scale` depois
//! da multiplicação. `.input_scale` continua obrigatório e preservado, mas não
//! é aplicado ao hidden neste FFN.

#[cfg(feature = "cuda")]
use super::bandwidth::BandwidthProfile;
#[cfg(feature = "cuda")]
use super::cache::ExpertLease;
use super::cache::{CacheError, ExpertCache, ExpertKey};
#[cfg(feature = "cuda")]
use super::coexecution::{combine_lane_outputs, filter_route, run_overlap, split_route_experts};
use super::format_q4_0::{parse_tensor_table, Q4LoadError, TensorSpec};
use memmap2::Mmap;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::Mutex;
#[cfg(feature = "cuda")]
use std::time::Instant;

const BLOCK_ELEMENTS: usize = 64;
const SUBBLOCK_ELEMENTS: usize = 16;
const BLOCK_BYTES: usize = 36;
const PAYLOAD_HEADER_BYTES: usize = 32;
const SCALE_COUNT: usize = 6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Nvfp4Error {
    Geometry(&'static str),
    Payload(&'static str),
    PayloadLength { expected: usize, actual: usize },
    BackendUnavailable(String),
}

impl fmt::Display for Nvfp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(reason) => write!(formatter, "geometria NVFP4 inválida: {reason}"),
            Self::Payload(reason) => write!(formatter, "payload NVFP4 inválido: {reason}"),
            Self::PayloadLength { expected, actual } => write!(
                formatter,
                "payload NVFP4 tem {actual} bytes, mas a geometria exige {expected}"
            ),
            Self::BackendUnavailable(reason) => {
                write!(formatter, "executor CUDA NVFP4 indisponível: {reason}")
            }
        }
    }
}

impl std::error::Error for Nvfp4Error {}

#[derive(Debug)]
pub enum Nvfp4ExecuteError {
    Cache(CacheError),
    #[cfg(feature = "cuda")]
    Coexecution(String),
    Format(Nvfp4Error),
    Source(Q4LoadError),
    #[cfg(feature = "cuda")]
    Cuda(String),
    HiddenWidth {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for Nvfp4ExecuteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => error.fmt(formatter),
            #[cfg(feature = "cuda")]
            Self::Coexecution(error) => write!(formatter, "co-execução NVFP4 falhou: {error}"),
            Self::Format(error) => error.fmt(formatter),
            Self::Source(error) => error.fmt(formatter),
            #[cfg(feature = "cuda")]
            Self::Cuda(error) => write!(formatter, "executor CUDA NVFP4 falhou: {error}"),
            Self::HiddenWidth { expected, actual } => {
                write!(
                    formatter,
                    "hidden NVFP4 tem {actual} elementos, esperado {expected}"
                )
            }
        }
    }
}

impl std::error::Error for Nvfp4ExecuteError {}

impl From<CacheError> for Nvfp4ExecuteError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

impl From<Nvfp4Error> for Nvfp4ExecuteError {
    fn from(error: Nvfp4Error) -> Self {
        Self::Format(error)
    }
}

impl From<Q4LoadError> for Nvfp4ExecuteError {
    fn from(error: Q4LoadError) -> Self {
        Self::Source(error)
    }
}

pub struct Nvfp4Expert<'a> {
    hidden: usize,
    intermediate: usize,
    gate_up: &'a [u8],
    down: &'a [u8],
    gate_scale: f32,
    up_scale: f32,
    down_scale: f32,
}

/// Valores intermediários de um expert, usados pela prova diferencial contra
/// o caminho nativo do llama.cpp. A ordem é deliberada: a primeira divergência
/// identifica se o defeito está na decodificação, ativação ou projeção down.
#[derive(Debug)]
#[cfg(test)]
pub struct Nvfp4Trace {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub activated: Vec<f32>,
    pub output: Vec<f32>,
}

impl<'a> Nvfp4Expert<'a> {
    pub fn packed_bytes(hidden: usize, intermediate: usize) -> Result<usize, Nvfp4Error> {
        if hidden == 0 || intermediate == 0 {
            return Err(Nvfp4Error::Geometry("dimensão zero"));
        }
        if !hidden.is_multiple_of(BLOCK_ELEMENTS) || !intermediate.is_multiple_of(BLOCK_ELEMENTS) {
            return Err(Nvfp4Error::Geometry(
                "hidden e intermediate precisam fechar blocos de 64",
            ));
        }
        let gate_up = (2 * intermediate)
            .checked_mul(row_bytes(hidden)?)
            .ok_or(Nvfp4Error::Geometry("overflow em gate_up"))?;
        let down = hidden
            .checked_mul(row_bytes(intermediate)?)
            .ok_or(Nvfp4Error::Geometry("overflow em down"))?;
        PAYLOAD_HEADER_BYTES
            .checked_add(gate_up)
            .and_then(|bytes| bytes.checked_add(down))
            .ok_or(Nvfp4Error::Geometry("overflow no payload"))
    }

    pub fn from_payload(
        hidden: usize,
        intermediate: usize,
        payload: &'a [u8],
    ) -> Result<Self, Nvfp4Error> {
        let expected = Self::packed_bytes(hidden, intermediate)?;
        if payload.len() != expected {
            return Err(Nvfp4Error::PayloadLength {
                expected,
                actual: payload.len(),
            });
        }
        let gate_up_len = u32::from_le_bytes(
            payload[..4]
                .try_into()
                .map_err(|_| Nvfp4Error::Payload("cabeçalho gate_up ausente"))?,
        ) as usize;
        let down_len = u32::from_le_bytes(
            payload[4..8]
                .try_into()
                .map_err(|_| Nvfp4Error::Payload("cabeçalho down ausente"))?,
        ) as usize;
        let expected_gate_up = (2 * intermediate) * row_bytes(hidden)?;
        let expected_down = hidden * row_bytes(intermediate)?;
        if gate_up_len != expected_gate_up || down_len != expected_down {
            return Err(Nvfp4Error::Payload(
                "tamanhos gate_up/down não correspondem à geometria",
            ));
        }
        let mut scales = [1.0f32; SCALE_COUNT];
        for (index, scale) in scales.iter_mut().enumerate() {
            let start = 8 + index * 4;
            *scale = f32::from_le_bytes(
                payload[start..start + 4]
                    .try_into()
                    .map_err(|_| Nvfp4Error::Payload("escala ausente"))?,
            );
            if !scale.is_finite() {
                return Err(Nvfp4Error::Payload("escala não finita"));
            }
        }
        let gate_up_start = PAYLOAD_HEADER_BYTES;
        let down_start = gate_up_start + gate_up_len;
        if down_start + down_len != payload.len() {
            return Err(Nvfp4Error::Payload("seções não cobrem o payload"));
        }
        Ok(Self {
            hidden,
            intermediate,
            gate_up: &payload[gate_up_start..down_start],
            down: &payload[down_start..],
            gate_scale: scales[0],
            up_scale: scales[1],
            down_scale: scales[2],
        })
    }

    pub fn run(&self, hidden: &[f32]) -> Result<Vec<f32>, Nvfp4ExecuteError> {
        let (_, _, _, output) = self.run_stages(hidden)?;
        Ok(output)
    }

    #[cfg(test)]
    pub fn run_trace(&self, hidden: &[f32]) -> Result<Nvfp4Trace, Nvfp4ExecuteError> {
        let (gate, up, activated, output) = self.run_stages(hidden)?;
        Ok(Nvfp4Trace {
            gate,
            up,
            activated,
            output,
        })
    }

    fn run_stages(
        &self,
        hidden: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), Nvfp4ExecuteError> {
        if hidden.len() != self.hidden {
            return Err(Nvfp4ExecuteError::HiddenWidth {
                expected: self.hidden,
                actual: hidden.len(),
            });
        }
        // llama.cpp's native expert matmul consumes the normalized hidden
        // directly and applies the per-expert `.scale` after the matmul.
        // The GGUF `.input_scale` sidecars are metadata used by other
        // quantized paths; the native MoE graph does not feed them into this
        // FFN operation, so applying them here would change the model.
        let gate = mat_vec_nvfp4(
            &self.gate_up[..self.intermediate * row_bytes(self.hidden).unwrap()],
            self.intermediate,
            self.hidden,
            hidden,
        )
        .into_iter()
        .map(|value| value * self.gate_scale)
        .collect::<Vec<_>>();
        let up = mat_vec_nvfp4(
            &self.gate_up[self.intermediate * row_bytes(self.hidden).unwrap()..],
            self.intermediate,
            self.hidden,
            hidden,
        )
        .into_iter()
        .map(|value| value * self.up_scale)
        .collect::<Vec<_>>();
        let gated = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| ggml_geglu_value(gate, up))
            .collect::<Vec<_>>();
        let output = mat_vec_nvfp4(self.down, self.hidden, self.intermediate, &gated)
            .into_iter()
            .map(|value| value * self.down_scale)
            .collect();
        Ok((gate, up, gated, output))
    }
}

fn pack_payload(
    gate_up: &[u8],
    down: &[u8],
    scales: [f32; SCALE_COUNT],
) -> Result<Vec<u8>, Nvfp4Error> {
    if gate_up.len() > u32::MAX as usize || down.len() > u32::MAX as usize {
        return Err(Nvfp4Error::Payload("bank excede u32"));
    }
    let mut payload = Vec::with_capacity(PAYLOAD_HEADER_BYTES + gate_up.len() + down.len());
    payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
    for scale in scales {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    payload.extend_from_slice(gate_up);
    payload.extend_from_slice(down);
    Ok(payload)
}

enum GateUpTensors {
    Fused(Vec<TensorSpec>),
    Split {
        gate: Vec<TensorSpec>,
        up: Vec<TensorSpec>,
    },
}

pub struct Nvfp4Source {
    mmap: Mmap,
    data_start: usize,
    tensors: Vec<TensorSpec>,
    experts: u32,
    hidden: usize,
    intermediate: usize,
    gate_up: GateUpTensors,
    down: Vec<TensorSpec>,
    router: Vec<TensorSpec>,
}

impl Nvfp4Source {
    pub fn open_infer(path: impl AsRef<Path>, experts: u32) -> Result<Self, Q4LoadError> {
        let file = File::open(path.as_ref()).map_err(|error| Q4LoadError::Io(error.to_string()))?;
        // SAFETY: the descriptor remains valid while the mapping is created.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
        let (_, tensors) = parse_tensor_table(&mmap)?;
        let fused = tensors
            .iter()
            .find(|tensor| tensor.name.ends_with("ffn_gate_up_exps.weight"));
        let (hidden, intermediate, tensor_experts) = if let Some(tensor) = fused {
            ensure_nvfp4(tensor)?;
            if tensor.dims.len() != 3 {
                return Err(Q4LoadError::InvalidTensor(
                    "gate_up NVFP4 precisa ser rank 3",
                ));
            }
            let hidden = dimension(tensor, 0, "hidden")?;
            let two_intermediate = dimension(tensor, 1, "intermediate")?;
            let tensor_experts = dimension_u32(tensor, 2, "experts")?;
            if !two_intermediate.is_multiple_of(2) {
                return Err(Q4LoadError::InvalidTensor(
                    "gate_up NVFP4 tem linhas ímpares",
                ));
            }
            (hidden, two_intermediate / 2, tensor_experts)
        } else {
            let gate = tensors
                .iter()
                .find(|tensor| tensor.name.ends_with("ffn_gate_exps.weight"))
                .ok_or_else(|| Q4LoadError::MissingTensor("ffn_gate_exps.weight".into()))?;
            let up = tensors
                .iter()
                .find(|tensor| tensor.name.ends_with("ffn_up_exps.weight"))
                .ok_or_else(|| Q4LoadError::MissingTensor("ffn_up_exps.weight".into()))?;
            ensure_nvfp4(gate)?;
            ensure_nvfp4(up)?;
            if gate.dims != up.dims || gate.dims.len() != 3 {
                return Err(Q4LoadError::InvalidTensor(
                    "gate e up NVFP4 têm geometrias diferentes",
                ));
            }
            let (hidden, intermediate) = split_geometry(gate)?;
            (hidden, intermediate, dimension_u32(gate, 2, "experts")?)
        };
        if tensor_experts != experts {
            return Err(Q4LoadError::InvalidTensor(
                "quantidade de experts não corresponde aos metadados MoE",
            ));
        }
        drop(mmap);
        Self::open(path, experts, hidden, intermediate)
    }

    fn open(
        path: impl AsRef<Path>,
        experts: u32,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, Q4LoadError> {
        Nvfp4Expert::packed_bytes(hidden, intermediate)
            .map_err(|_| Q4LoadError::InvalidTensor("geometria não fecha NVFP4"))?;
        let (mmap, data_start, tensors) = super::format_q4_0::map_tensor_source(path.as_ref())?;
        let mut fused = Vec::new();
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        let mut router = Vec::new();
        for tensor in &tensors {
            if tensor.name.starts_with("blk.") && tensor.name.ends_with("ffn_gate_up_exps.weight") {
                fused.push(tensor.clone());
            } else if tensor.name.starts_with("blk.")
                && tensor.name.ends_with("ffn_gate_exps.weight")
            {
                gate.push(tensor.clone());
            } else if tensor.name.starts_with("blk.") && tensor.name.ends_with("ffn_up_exps.weight")
            {
                up.push(tensor.clone());
            } else if tensor.name.starts_with("blk.")
                && tensor.name.ends_with("ffn_down_exps.weight")
            {
                down.push(tensor.clone());
            } else if tensor.name.starts_with("blk.")
                && tensor.name.ends_with("ffn_gate_inp.weight")
            {
                router.push(tensor.clone());
            }
        }
        if down.is_empty() {
            return Err(Q4LoadError::MissingTensor(
                "blk.<layer>.ffn_down_exps.weight".into(),
            ));
        }
        let gate_up = if !fused.is_empty() {
            if !gate.is_empty() || !up.is_empty() {
                return Err(Q4LoadError::InvalidTensor(
                    "convenções fundida e separada de experts misturadas",
                ));
            }
            GateUpTensors::Fused(fused)
        } else if gate.is_empty() || up.is_empty() {
            return Err(Q4LoadError::MissingTensor(
                "ffn_gate_up_exps.weight ou ffn_gate_exps.weight + ffn_up_exps.weight".into(),
            ));
        } else {
            GateUpTensors::Split { gate, up }
        };
        Ok(Self {
            mmap,
            data_start,
            tensors,
            experts,
            hidden,
            intermediate,
            gate_up,
            down,
            router,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn intermediate(&self) -> usize {
        self.intermediate
    }

    pub fn experts(&self) -> u32 {
        self.experts
    }

    pub fn has_router(&self) -> bool {
        !self.router.is_empty()
    }

    pub fn layer_count(&self) -> u32 {
        let tensors = match &self.gate_up {
            GateUpTensors::Fused(tensors) => tensors,
            GateUpTensors::Split { gate, .. } => gate,
        };
        tensors
            .iter()
            .filter_map(|tensor| {
                tensor
                    .name
                    .strip_prefix("blk.")?
                    .split('.')
                    .next()?
                    .parse::<u32>()
                    .ok()
            })
            .max()
            .map_or(0, |last| last.saturating_add(1))
    }

    pub fn load_expert(&self, layer: u32, expert: u32) -> Result<Vec<u8>, Q4LoadError> {
        if expert >= self.experts {
            return Err(Q4LoadError::ExpertOutOfRange {
                expert,
                count: self.experts,
            });
        }
        let down = self.tensor_for_layer(&self.down, layer, "ffn_down_exps.weight")?;
        // GGUF stores matrix dimensions as [input, output, experts]. The
        // down projection is therefore [intermediate, hidden, experts].
        validate_tensor(down, self.experts, self.intermediate, self.hidden)?;
        let down_per_expert = self
            .hidden
            .checked_mul(
                row_bytes(self.intermediate)
                    .map_err(|_| Q4LoadError::InvalidTensor("geometria da linha down NVFP4"))?,
            )
            .ok_or(Q4LoadError::InvalidTensor("overflow em down NVFP4"))?;
        let down_bytes = self.expert_slice(down, expert, down_per_expert)?;
        let (gate_bytes, up_bytes, gate_tensor, up_tensor) = match &self.gate_up {
            GateUpTensors::Fused(tensors) => {
                let tensor = self.tensor_for_layer(tensors, layer, "ffn_gate_up_exps.weight")?;
                validate_tensor(tensor, self.experts, self.hidden, 2 * self.intermediate)?;
                let bytes = (2 * self.intermediate)
                    .checked_mul(row_bytes(self.hidden).map_err(|_| {
                        Q4LoadError::InvalidTensor("geometria da linha gate_up NVFP4")
                    })?)
                    .ok_or(Q4LoadError::InvalidTensor("overflow em gate_up NVFP4"))?;
                let bytes = self.expert_slice(tensor, expert, bytes)?;
                (bytes, &[][..], tensor, tensor)
            }
            GateUpTensors::Split { gate, up } => {
                let gate = self.tensor_for_layer(gate, layer, "ffn_gate_exps.weight")?;
                let up = self.tensor_for_layer(up, layer, "ffn_up_exps.weight")?;
                validate_tensor(gate, self.experts, self.hidden, self.intermediate)?;
                validate_tensor(up, self.experts, self.hidden, self.intermediate)?;
                let bytes =
                    self.intermediate
                        .checked_mul(row_bytes(self.hidden).map_err(|_| {
                            Q4LoadError::InvalidTensor("geometria da linha gate NVFP4")
                        })?)
                        .ok_or(Q4LoadError::InvalidTensor("overflow em gate/up NVFP4"))?;
                (
                    self.expert_slice(gate, expert, bytes)?,
                    self.expert_slice(up, expert, bytes)?,
                    gate,
                    up,
                )
            }
        };
        let mut gate_up = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        gate_up.extend_from_slice(gate_bytes);
        gate_up.extend_from_slice(up_bytes);
        let scales = [
            self.sidecar_scale(gate_tensor, "scale", expert)?,
            self.sidecar_scale(up_tensor, "scale", expert)?,
            self.sidecar_scale(down, "scale", expert)?,
            self.sidecar_scale(gate_tensor, "input_scale", expert)?,
            self.sidecar_scale(up_tensor, "input_scale", expert)?,
            self.sidecar_scale(down, "input_scale", expert)?,
        ];
        pack_payload(&gate_up, down_bytes, scales)
            .map_err(|_| Q4LoadError::InvalidTensor("payload NVFP4"))
    }

    pub fn router_logits(&self, layer: u32, hidden: &[f32]) -> Result<Vec<f32>, Q4LoadError> {
        if hidden.len() != self.hidden {
            return Err(Q4LoadError::InvalidTensor(
                "hidden não corresponde à largura do router",
            ));
        }
        let tensor = self.tensor_for_layer(&self.router, layer, "ffn_gate_inp.weight")?;
        let elements = (self.experts as u64)
            .checked_mul(self.hidden as u64)
            .ok_or(Q4LoadError::InvalidTensor("overflow no router"))?;
        let declared = tensor
            .dims
            .iter()
            .try_fold(1u64, |total, dimension| total.checked_mul(*dimension))
            .ok_or(Q4LoadError::InvalidTensor(
                "overflow nas dimensões do router",
            ))?;
        if declared != elements || tensor.dims.len() != 2 {
            return Err(Q4LoadError::InvalidTensor(
                "dimensões do router não correspondem ao modelo",
            ));
        }
        let element_bytes = scalar_bytes(tensor.dtype)?;
        let total_bytes = usize::try_from(elements)
            .ok()
            .and_then(|count| count.checked_mul(element_bytes))
            .ok_or(Q4LoadError::InvalidTensor("tamanho do router excede usize"))?;
        let raw = self.tensor_bytes(tensor, total_bytes)?;
        let input_scale = match self.tensor_sidecar(tensor, "scale") {
            Ok(scale) => self.read_scale_vector(scale)?,
            Err(Q4LoadError::MissingTensor(_)) if matches!(tensor.dtype, 0 | 1 | 30) => {
                vec![1.0; self.hidden]
            }
            Err(error) => return Err(error),
        };
        if input_scale.len() != self.hidden {
            return Err(Q4LoadError::InvalidTensor(
                "escala do router tem geometria errada",
            ));
        }
        let mut logits = Vec::with_capacity(self.experts as usize);
        for expert in 0..self.experts as usize {
            let row = &raw
                [expert * self.hidden * element_bytes..(expert + 1) * self.hidden * element_bytes];
            let value = row
                .chunks_exact(element_bytes)
                .zip(hidden.iter().zip(&input_scale))
                .map(|(encoded, (&input, &scale))| {
                    decode_scalar(encoded, tensor.dtype) * input * scale
                })
                .sum();
            logits.push(value);
        }
        Ok(logits)
    }

    fn tensor_for_layer<'a>(
        &self,
        tensors: &'a [TensorSpec],
        layer: u32,
        suffix: &str,
    ) -> Result<&'a TensorSpec, Q4LoadError> {
        tensors
            .iter()
            .find(|tensor| tensor.name == format!("blk.{layer}.{suffix}"))
            .ok_or_else(|| Q4LoadError::MissingTensor(format!("blk.{layer}.{suffix}")))
    }

    fn tensor_sidecar<'a>(
        &'a self,
        tensor: &TensorSpec,
        suffix: &str,
    ) -> Result<&'a TensorSpec, Q4LoadError> {
        find_sidecar(&self.tensors, tensor, suffix)
    }

    fn sidecar_scale(
        &self,
        tensor: &TensorSpec,
        suffix: &str,
        expert: u32,
    ) -> Result<f32, Q4LoadError> {
        let sidecar = self.tensor_sidecar(tensor, suffix)?;
        let values = self.read_scale_vector(sidecar)?;
        values
            .get(expert as usize)
            .copied()
            .ok_or(Q4LoadError::InvalidTensor("escala de expert ausente"))
    }

    fn read_scale_vector(&self, tensor: &TensorSpec) -> Result<Vec<f32>, Q4LoadError> {
        let count = tensor
            .dims
            .iter()
            .try_fold(1u64, |total, dimension| total.checked_mul(*dimension))
            .ok_or(Q4LoadError::InvalidTensor("overflow na escala"))?;
        let count = usize::try_from(count)
            .map_err(|_| Q4LoadError::InvalidTensor("escala excede usize"))?;
        let element_bytes = scalar_bytes(tensor.dtype)?;
        let bytes = count
            .checked_mul(element_bytes)
            .ok_or(Q4LoadError::InvalidTensor("bytes da escala excedem usize"))?;
        let raw = self.tensor_bytes(tensor, bytes)?;
        Ok(raw
            .chunks_exact(element_bytes)
            .map(|value| decode_scalar(value, tensor.dtype))
            .collect())
    }

    fn tensor_bytes<'a>(
        &'a self,
        tensor: &TensorSpec,
        total_bytes: usize,
    ) -> Result<&'a [u8], Q4LoadError> {
        let offset = usize::try_from(tensor.offset)
            .ok()
            .and_then(|value| self.data_start.checked_add(value))
            .ok_or(Q4LoadError::InvalidTensor("offset GGUF inválido"))?;
        let end = offset
            .checked_add(total_bytes)
            .ok_or(Q4LoadError::InvalidTensor("tamanho do tensor excede usize"))?;
        self.mmap
            .get(offset..end)
            .ok_or(Q4LoadError::InvalidTensor("tensor excede o arquivo"))
    }

    fn expert_slice<'a>(
        &'a self,
        tensor: &TensorSpec,
        expert: u32,
        bytes_per_expert: usize,
    ) -> Result<&'a [u8], Q4LoadError> {
        ensure_nvfp4(tensor)?;
        let total = bytes_per_expert
            .checked_mul(self.experts as usize)
            .ok_or(Q4LoadError::InvalidTensor("overflow no tensor NVFP4"))?;
        let offset = usize::try_from(tensor.offset)
            .ok()
            .and_then(|value| self.data_start.checked_add(value))
            .and_then(|value| {
                bytes_per_expert
                    .checked_mul(expert as usize)
                    .and_then(|delta| value.checked_add(delta))
            })
            .ok_or(Q4LoadError::InvalidTensor("offset GGUF inválido"))?;
        let end = offset
            .checked_add(bytes_per_expert)
            .ok_or(Q4LoadError::InvalidTensor("fim do tensor excede usize"))?;
        let expert_delta = bytes_per_expert
            .checked_mul(expert as usize)
            .ok_or(Q4LoadError::InvalidTensor("expert excede o tensor"))?;
        let tensor_end = offset
            .checked_sub(expert_delta)
            .and_then(|start| start.checked_add(total))
            .ok_or(Q4LoadError::InvalidTensor("tamanho GGUF inválido"))?;
        if tensor_end > self.mmap.len() || end > tensor_end {
            return Err(Q4LoadError::InvalidTensor("tensor excede o arquivo"));
        }
        Ok(&self.mmap[offset..end])
    }
}

fn sidecar_name(tensor: &TensorSpec, suffix: &str) -> String {
    let stem = tensor.name.strip_suffix(".weight").unwrap_or(&tensor.name);
    format!("{stem}.{suffix}")
}

fn find_sidecar<'a>(
    tensors: &'a [TensorSpec],
    tensor: &TensorSpec,
    suffix: &str,
) -> Result<&'a TensorSpec, Q4LoadError> {
    let name = sidecar_name(tensor, suffix);
    tensors
        .iter()
        .find(|candidate| candidate.name == name)
        .ok_or_else(|| Q4LoadError::MissingTensor(name))
}

fn ensure_nvfp4(tensor: &TensorSpec) -> Result<(), Q4LoadError> {
    if tensor.dtype != 40 {
        return Err(Q4LoadError::UnsupportedTensorType(tensor.dtype));
    }
    Ok(())
}

fn validate_tensor(
    tensor: &TensorSpec,
    experts: u32,
    rows: usize,
    columns: usize,
) -> Result<(), Q4LoadError> {
    ensure_nvfp4(tensor)?;
    if tensor.dims.len() != 3 || tensor.dims[0] != rows as u64 || tensor.dims[1] != columns as u64 {
        return Err(Q4LoadError::InvalidTensor(
            "dimensões não correspondem ao modelo",
        ));
    }
    if tensor.dims[2] != experts as u64 {
        return Err(Q4LoadError::InvalidTensor("quantidade de experts inválida"));
    }
    Ok(())
}

fn dimension(tensor: &TensorSpec, index: usize, name: &'static str) -> Result<usize, Q4LoadError> {
    tensor
        .dims
        .get(index)
        .copied()
        .ok_or(Q4LoadError::InvalidTensor("rank do tensor NVFP4"))
        .and_then(|value| usize::try_from(value).map_err(|_| Q4LoadError::InvalidTensor(name)))
}

fn dimension_u32(
    tensor: &TensorSpec,
    index: usize,
    name: &'static str,
) -> Result<u32, Q4LoadError> {
    tensor
        .dims
        .get(index)
        .copied()
        .ok_or(Q4LoadError::InvalidTensor("rank do tensor NVFP4"))
        .and_then(|value| u32::try_from(value).map_err(|_| Q4LoadError::InvalidTensor(name)))
}

fn split_geometry(gate: &TensorSpec) -> Result<(usize, usize), Q4LoadError> {
    if gate.dims.len() != 3 {
        return Err(Q4LoadError::InvalidTensor("gate NVFP4 precisa ser rank 3"));
    }
    // GGUF/ggml matrix weights use [input, output, experts]. In Gemma4 the
    // real tensors are [2816, 704, 128], so hidden is dims[0] and the FFN
    // intermediate width is dims[1].
    Ok((
        dimension(gate, 0, "hidden")?,
        dimension(gate, 1, "intermediate")?,
    ))
}

fn row_bytes(columns: usize) -> Result<usize, Nvfp4Error> {
    if !columns.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(Nvfp4Error::Geometry("linha não fecha bloco de 64"));
    }
    columns
        .checked_div(BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(BLOCK_BYTES))
        .ok_or(Nvfp4Error::Geometry("overflow na linha"))
}

fn scalar_bytes(dtype: u32) -> Result<usize, Q4LoadError> {
    match dtype {
        0 => Ok(4),
        1 | 30 => Ok(2),
        other => Err(Q4LoadError::UnsupportedTensorType(other)),
    }
}

fn decode_scalar(encoded: &[u8], dtype: u32) -> f32 {
    match dtype {
        0 => f32::from_le_bytes(encoded.try_into().unwrap()),
        1 => f16_to_f32(u16::from_le_bytes(encoded.try_into().unwrap())),
        30 => f32::from_bits(u32::from(u16::from_le_bytes(encoded.try_into().unwrap())) << 16),
        _ => 0.0,
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;
    match exponent {
        0 if fraction == 0 => sign * 0.0,
        0 => sign * (fraction as f32 / 1024.0) * 2.0f32.powi(-14),
        31 => {
            if fraction == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        exponent => sign * (1.0 + fraction as f32 / 1024.0) * 2.0f32.powi(exponent - 15),
    }
}

#[derive(Clone, Copy)]
struct Q8Block {
    d: f32,
    qs: [i8; 32],
}

fn quantize_q8_0(input: &[f32]) -> Vec<Q8Block> {
    assert!(input.len().is_multiple_of(32));
    input
        .chunks_exact(32)
        .map(|chunk| {
            let amax = chunk
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let d = f16_to_f32(ggml_f32_to_f16_bits(d));
            let mut qs = [0_i8; 32];
            for (quantized, &value) in qs.iter_mut().zip(chunk) {
                *quantized = (value * id).round() as i8;
            }
            Q8Block { d, qs }
        })
        .collect()
}

// GGML_TYPE_NVFP4's CPU dot product is NVFP4 x Q8_0, not a direct F32 dot.
// Keep the quantization, integer accumulation and block order identical to
// ggml_vec_dot_nvfp4_q8_0_generic so the Rust reference is numerically
// comparable to the native backend.
fn mat_vec_nvfp4(matrix: &[u8], rows: usize, columns: usize, input: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // llama.cpp selects the AVX2 NVFP4 kernel on this host. Its eight
        // lane accumulators and fused multiply-add are observably different
        // from the portable scalar reduction, so mirror that order when the
        // same runtime capabilities are present.
        return unsafe { mat_vec_nvfp4_avx2_order(matrix, rows, columns, input) };
    }
    mat_vec_nvfp4_scalar(matrix, rows, columns, input)
}

fn mat_vec_nvfp4_scalar(matrix: &[u8], rows: usize, columns: usize, input: &[f32]) -> Vec<f32> {
    let row_size = columns / BLOCK_ELEMENTS * BLOCK_BYTES;
    let input_q8 = quantize_q8_0(input);
    (0..rows)
        .map(|row| {
            let row_bytes = &matrix[row * row_size..(row + 1) * row_size];
            let mut result = 0.0_f32;
            for (block, packed) in row_bytes.chunks_exact(BLOCK_BYTES).enumerate() {
                for subblock in 0..(BLOCK_ELEMENTS / SUBBLOCK_ELEMENTS) {
                    let scale = ue4m3_to_f32(packed[subblock]);
                    let q8_block = block * 2 + subblock / 2;
                    let q8_offset = (subblock % 2) * SUBBLOCK_ELEMENTS;
                    let q8 = input_q8[q8_block];
                    let values = &packed[4 + subblock * (SUBBLOCK_ELEMENTS / 2)
                        ..4 + (subblock + 1) * (SUBBLOCK_ELEMENTS / 2)];
                    let mut sum_lo = 0_i32;
                    let mut sum_hi = 0_i32;
                    for (index, &quant) in values.iter().enumerate() {
                        let quant_lo = i32::from(fp4_value_i8(quant & 0x0f));
                        let quant_hi = i32::from(fp4_value_i8(quant >> 4));
                        sum_lo += i32::from(q8.qs[q8_offset + index]) * quant_lo;
                        sum_hi +=
                            i32::from(q8.qs[q8_offset + index + SUBBLOCK_ELEMENTS / 2]) * quant_hi;
                    }
                    result += q8.d * scale * (sum_lo + sum_hi) as f32;
                }
            }
            result
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn mat_vec_nvfp4_avx2_order(
    matrix: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Vec<f32> {
    let row_size = columns / BLOCK_ELEMENTS * BLOCK_BYTES;
    let input_q8 = quantize_q8_0(input);
    (0..rows)
        .map(|row| {
            let row_bytes = &matrix[row * row_size..(row + 1) * row_size];
            let mut accum = [0.0_f32; 8];
            for (block, packed) in row_bytes.chunks_exact(BLOCK_BYTES).enumerate() {
                let q8_blocks = [input_q8[block * 2], input_q8[block * 2 + 1]];
                for half in 0..2 {
                    let q8 = q8_blocks[half];
                    for lane in 0..8 {
                        let section = lane / 2;
                        let subblock = section / 2;
                        let nibble = section % 2;
                        let values = &packed[4 + (half * 2 + subblock) * (SUBBLOCK_ELEMENTS / 2)
                            ..4 + (half * 2 + subblock + 1) * (SUBBLOCK_ELEMENTS / 2)];
                        let value_start = (lane % 2) * 4;
                        let q8_start = section * 8 + value_start;
                        let mut dot = 0_i32;
                        for index in 0..4 {
                            let quant = values[value_start + index];
                            let fp4 = if nibble == 0 {
                                fp4_value_i8(quant & 0x0f)
                            } else {
                                fp4_value_i8(quant >> 4)
                            };
                            dot += i32::from(q8.qs[q8_start + index]) * i32::from(fp4);
                        }
                        let scale = ue4m3_to_f32(packed[half * 2 + subblock]);
                        let factor = scale * q8.d;
                        accum[lane] = factor.mul_add(dot as f32, accum[lane]);
                    }
                }
            }

            // This is hsum_float_8 from ggml-cpu/arch/x86/quants.c:
            // (x4+x0+x7+x3) + (x5+x1+x6+x2), preserving its pairwise order.
            let pair0 = accum[4] + accum[0];
            let pair1 = accum[5] + accum[1];
            let pair2 = accum[6] + accum[2];
            let pair3 = accum[7] + accum[3];
            (pair0 + pair3) + (pair1 + pair2)
        })
        .collect()
}

fn fp4_value_i8(value: u8) -> i8 {
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12][value as usize]
}

#[cfg(test)]
fn fp4_value(value: u8) -> f32 {
    f32::from(fp4_value_i8(value))
}

fn ue4m3_to_f32(value: u8) -> f32 {
    if value == 0 || value == 0x7f {
        return 0.0;
    }
    let exponent = ((value >> 3) & 0x0f) as i32;
    let mantissa = (value & 0x07) as f32;
    let raw = if exponent == 0 {
        mantissa * 2.0f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
    };
    raw * 0.5
}

fn ggml_f32_to_f16_bits(value: f32) -> u16 {
    // This is the conversion used by ggml_compute_fp32_to_fp16. In
    // particular, it preserves ggml's round-to-nearest behavior instead of
    // silently substituting a different half implementation in the oracle.
    let scale_to_inf = f32::from_bits(0x7780_0000);
    let scale_to_zero = f32::from_bits(0x0880_0000);
    let base = (value.abs() * scale_to_inf) * scale_to_zero;
    let bits = value.to_bits();
    let shl1_bits = bits.wrapping_add(bits);
    let sign = bits & 0x8000_0000;
    let mut bias = shl1_bits & 0xff00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }
    let base = f32::from_bits((bias >> 1) + 0x0780_0000) + base;
    let bits = base.to_bits();
    let exponent_bits = (bits >> 13) & 0x0000_7c00;
    let mantissa_bits = bits & 0x0000_0fff;
    let nonsign = exponent_bits + mantissa_bits;
    ((sign >> 16) as u16)
        | if shl1_bits > 0xff00_0000 {
            0x7e00
        } else {
            nonsign as u16
        }
}

fn ggml_gelu_f32(value: f32) -> f32 {
    0.5_f32
        * value
        * (1.0_f32
            + (0.797_884_560_802_865_4_f32 * value * (1.0_f32 + 0.044_715_f32 * value * value))
                .tanh())
}

fn ggml_gelu_f16(value: f32) -> f32 {
    let input = f16_to_f32(ggml_f32_to_f16_bits(value));
    let gelu = ggml_gelu_f32(input);
    f16_to_f32(ggml_f32_to_f16_bits(gelu))
}

fn ggml_geglu_value(gate: f32, up: f32) -> f32 {
    if gate <= -10.0 {
        0.0
    } else if gate >= 10.0 {
        gate * up
    } else {
        ggml_gelu_f16(gate) * up
    }
}

pub struct Nvfp4Executor {
    cache: Arc<ExpertCache>,
    hidden: usize,
    intermediate: usize,
    backend: Nvfp4Backend,
    #[cfg(feature = "cuda")]
    coexecution_profile: Option<BandwidthProfile>,
    #[cfg(feature = "cuda")]
    host_diagnostics: Mutex<Nvfp4HostDiagnostics>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Nvfp4HostDiagnostics {
    pub cache_lookup_ms: f64,
    pub expert_load_ms: f64,
    pub cache_admit_ms: f64,
    pub aggregation_ms: f64,
    pub cpu_expert_calls: u64,
    pub coexecution_gpu_fraction: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nvfp4BackendKind {
    CpuReference,
    Cuda,
}

#[cfg(feature = "cuda")]
enum Nvfp4Backend {
    Cpu,
    Cuda(super::cuda::Nvfp4CudaExecutor),
}

#[cfg(not(feature = "cuda"))]
enum Nvfp4Backend {
    Cpu,
}

impl Nvfp4Executor {
    #[cfg(feature = "cuda")]
    pub fn new_cpu(
        cache: Arc<ExpertCache>,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, Nvfp4Error> {
        Nvfp4Expert::packed_bytes(hidden, intermediate)?;
        Ok(Self {
            cache,
            hidden,
            intermediate,
            backend: Nvfp4Backend::Cpu,
            #[cfg(feature = "cuda")]
            coexecution_profile: None,
            #[cfg(feature = "cuda")]
            host_diagnostics: Mutex::new(Nvfp4HostDiagnostics::default()),
        })
    }

    pub fn new(
        cache: Arc<ExpertCache>,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, Nvfp4Error> {
        Nvfp4Expert::packed_bytes(hidden, intermediate)?;
        let cpu_requested = std::env::var("POM_MOE_NVFP4_BACKEND")
            .map(|value| value.eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        let backend_kind = select_backend_kind(cfg!(feature = "cuda"), cpu_requested)?;
        #[cfg(feature = "cuda")]
        let backend = match backend_kind {
            Nvfp4BackendKind::CpuReference => Nvfp4Backend::Cpu,
            Nvfp4BackendKind::Cuda => {
                return Err(Nvfp4Error::BackendUnavailable(
                    "compute buffer CUDA medido ausente; use o construtor com orçamento do contexto"
                        .into(),
                ));
            }
        };
        #[cfg(not(feature = "cuda"))]
        let backend = match backend_kind {
            Nvfp4BackendKind::CpuReference => Nvfp4Backend::Cpu,
            Nvfp4BackendKind::Cuda => unreachable!("CUDA não compilado não pode ser selecionado"),
        };

        Ok(Self {
            cache,
            hidden,
            intermediate,
            backend,
            #[cfg(feature = "cuda")]
            coexecution_profile: None,
            #[cfg(feature = "cuda")]
            host_diagnostics: Mutex::new(Nvfp4HostDiagnostics::default()),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_device_cache(
        cache: Arc<ExpertCache>,
        device_cache: Arc<super::cache::DeviceExpertCache>,
        hidden: usize,
        intermediate: usize,
        coexecution_profile: Option<BandwidthProfile>,
    ) -> Result<Self, Nvfp4Error> {
        Nvfp4Expert::packed_bytes(hidden, intermediate)?;
        let cpu_requested = std::env::var("POM_MOE_NVFP4_BACKEND")
            .map(|value| value.eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        let backend_kind = select_backend_kind(true, cpu_requested)?;
        let backend = match backend_kind {
            Nvfp4BackendKind::CpuReference => Nvfp4Backend::Cpu,
            Nvfp4BackendKind::Cuda => Nvfp4Backend::Cuda(
                super::cuda::Nvfp4CudaExecutor::new_with_cache(hidden, intermediate, device_cache)
                    .map_err(|error| Nvfp4Error::BackendUnavailable(error.to_string()))?,
            ),
        };
        Ok(Self {
            cache,
            hidden,
            intermediate,
            backend,
            coexecution_profile,
            #[cfg(feature = "cuda")]
            host_diagnostics: Mutex::new(Nvfp4HostDiagnostics::default()),
        })
    }

    pub fn backend_kind(&self) -> Nvfp4BackendKind {
        match &self.backend {
            Nvfp4Backend::Cpu => Nvfp4BackendKind::CpuReference,
            #[cfg(feature = "cuda")]
            Nvfp4Backend::Cuda(_) => Nvfp4BackendKind::Cuda,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_diagnostics(&self) -> super::cuda::Nvfp4CudaDiagnostics {
        match &self.backend {
            Nvfp4Backend::Cpu => super::cuda::Nvfp4CudaDiagnostics::default(),
            Nvfp4Backend::Cuda(executor) => executor.diagnostics(),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn host_diagnostics(&self) -> Nvfp4HostDiagnostics {
        *self
            .host_diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        match &self.backend {
            Nvfp4Backend::Cpu => super::cache::DeviceCacheStats::default(),
            Nvfp4Backend::Cuda(executor) => executor.resident_cache_stats(),
        }
    }

    pub fn run_from_source(
        &self,
        source: &Nvfp4Source,
        key: ExpertKey,
        layer: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>, Nvfp4ExecuteError> {
        let expert = key.expert;
        self.run_with_loader(
            key,
            || {
                source
                    .load_expert(layer, expert)
                    .map_err(Nvfp4ExecuteError::Source)
            },
            hidden,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn run_route_batch_from_source(
        &self,
        source: &Nvfp4Source,
        model: &str,
        layer: u32,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Nvfp4ExecuteError> {
        self.validate_route_batch(experts, token_ids, weights, token_offsets, hidden, tokens)?;
        let Nvfp4Backend::Cuda(_) = &self.backend else {
            return Err(Nvfp4Error::BackendUnavailable(
                "lote de rotas requer executor CUDA residente".into(),
            )
            .into());
        };

        if let Some(profile) = self.coexecution_profile {
            return self.run_route_batch_coexecuted_from_source(
                source,
                model,
                layer,
                experts,
                token_ids,
                weights,
                token_offsets,
                hidden,
                tokens,
                profile,
            );
        }

        self.run_route_batch_cuda_from_source(
            source,
            model,
            layer,
            experts,
            token_ids,
            weights,
            token_offsets,
            hidden,
            tokens,
        )
    }

    #[cfg(feature = "cuda")]
    fn validate_route_batch(
        &self,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<(), Nvfp4ExecuteError> {
        if experts.is_empty()
            || experts.len() != token_ids.len()
            || experts.len() != weights.len()
            || tokens == 0
            || token_offsets.len() != tokens + 1
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&experts.len())
            || token_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || token_ids.iter().any(|&token| token as usize >= tokens)
        {
            return Err(Nvfp4Error::Payload("batch de rotas NVFP4 inválido").into());
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(Nvfp4Error::Payload("batch de rotas tem peso não finito").into());
        }
        let expected = tokens.saturating_mul(self.hidden);
        if hidden.len() != expected {
            return Err(Nvfp4ExecuteError::HiddenWidth {
                expected,
                actual: hidden.len(),
            });
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_route_batch_cuda_from_source(
        &self,
        source: &Nvfp4Source,
        model: &str,
        layer: u32,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Nvfp4ExecuteError> {
        let Nvfp4Backend::Cuda(executor) = &self.backend else {
            return Err(
                Nvfp4Error::BackendUnavailable("executor CUDA NVFP4 ausente".into()).into(),
            );
        };
        let mut resident = Vec::with_capacity(experts.len());
        for &expert in experts {
            let key = ExpertKey::new(model, layer, expert, "nvfp4");
            if let Some(lease) = executor.resident_lookup(&key) {
                resident.push(lease);
                continue;
            }

            let load_started = Instant::now();
            let payload = source.load_expert(layer, expert)?;
            let expert_load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
            self.host_diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .expert_load_ms += expert_load_ms;
            Nvfp4Expert::from_payload(self.hidden, self.intermediate, &payload)?;
            resident.push(
                executor
                    .resident_admit(key, &payload)
                    .map_err(|error| Nvfp4ExecuteError::Cuda(error.to_string()))?,
            );
        }
        executor
            .run_route_batch_resident(&resident, token_ids, weights, token_offsets, hidden, tokens)
            .map_err(|error| Nvfp4ExecuteError::Cuda(error.to_string()))
    }

    #[cfg(feature = "cuda")]
    fn run_route_batch_coexecuted_from_source(
        &self,
        source: &Nvfp4Source,
        model: &str,
        layer: u32,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
        profile: BandwidthProfile,
    ) -> Result<Vec<f32>, Nvfp4ExecuteError> {
        let Nvfp4Backend::Cuda(executor) = &self.backend else {
            return Err(
                Nvfp4Error::BackendUnavailable("executor CUDA NVFP4 ausente".into()).into(),
            );
        };
        let mut resident_gpu_experts = Vec::new();
        for &expert in experts {
            if resident_gpu_experts.contains(&expert) {
                continue;
            }
            let key = ExpertKey::new(model, layer, expert, "nvfp4");
            if executor.resident_lookup(&key).is_some() {
                resident_gpu_experts.push(expert);
            }
        }
        let split = split_route_experts(experts, &resident_gpu_experts, Some(profile))
            .map_err(|error| Nvfp4ExecuteError::Coexecution(error.to_string()))?;
        self.host_diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .coexecution_gpu_fraction = Some(split.gpu_fraction);

        let cpu_route = filter_route(
            experts,
            token_ids,
            weights,
            token_offsets,
            &split.cpu_experts,
            tokens,
        );
        let gpu_route = filter_route(
            experts,
            token_ids,
            weights,
            token_offsets,
            &split.gpu_experts,
            tokens,
        );
        let cpu_route = cpu_route;
        let gpu_route = gpu_route;
        let (cpu_output, gpu_output) = run_overlap(
            || {
                if cpu_route.0.is_empty() {
                    Ok(vec![0.0; hidden.len()])
                } else {
                    self.run_cpu_route_batch_from_source(
                        source,
                        model,
                        layer,
                        &cpu_route.0,
                        &cpu_route.1,
                        &cpu_route.2,
                        hidden,
                    )
                }
            },
            || {
                if gpu_route.0.is_empty() {
                    Ok(vec![0.0; hidden.len()])
                } else {
                    self.run_route_batch_cuda_from_source(
                        source,
                        model,
                        layer,
                        &gpu_route.0,
                        &gpu_route.1,
                        &gpu_route.2,
                        &gpu_route.3,
                        hidden,
                        tokens,
                    )
                }
            },
        )
        .map_err(|error| Nvfp4ExecuteError::Coexecution(error.to_string()))?;
        combine_lane_outputs(&cpu_output, &gpu_output, tokens, self.hidden)
            .map_err(|error| Nvfp4ExecuteError::Coexecution(error.to_string()))
    }

    #[cfg(feature = "cuda")]
    fn run_cpu_route_batch_from_source(
        &self,
        source: &Nvfp4Source,
        model: &str,
        layer: u32,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        hidden: &[f32],
    ) -> Result<Vec<f32>, Nvfp4ExecuteError> {
        let mut leases = HashMap::<u32, ExpertLease>::new();
        for &expert in experts {
            if leases.contains_key(&expert) {
                continue;
            }
            let key = ExpertKey::new(model, layer, expert, "nvfp4");
            let lookup_started = Instant::now();
            let cached = self.cache.lookup(&key);
            let cache_lookup_ms = lookup_started.elapsed().as_secs_f64() * 1_000.0;
            let (lease, expert_load_ms, cache_admit_ms) = match cached {
                Some(lease) => (lease, 0.0, 0.0),
                None => {
                    let load_started = Instant::now();
                    let payload = source.load_expert(layer, expert)?;
                    let expert_load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
                    let admit_started = Instant::now();
                    let lease = self
                        .cache
                        .admit(key, payload)
                        .map_err(Nvfp4ExecuteError::Cache)?;
                    let cache_admit_ms = admit_started.elapsed().as_secs_f64() * 1_000.0;
                    (lease, expert_load_ms, cache_admit_ms)
                }
            };
            let mut host = self
                .host_diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            host.cache_lookup_ms += cache_lookup_ms;
            host.expert_load_ms += expert_load_ms;
            host.cache_admit_ms += cache_admit_ms;
            leases.insert(expert, lease);
        }

        let mut output = vec![0.0; hidden.len()];
        for index in 0..experts.len() {
            let token = token_ids[index] as usize;
            let token_start = token * self.hidden;
            let token_hidden = &hidden[token_start..token_start + self.hidden];
            let expert = experts[index];
            let lease = leases
                .get(&expert)
                .ok_or(Nvfp4ExecuteError::Coexecution("expert CPU ausente".into()))?;
            let values = Nvfp4Expert::from_payload(self.hidden, self.intermediate, lease.bytes())?
                .run(token_hidden)?;
            let target = &mut output[token_start..token_start + self.hidden];
            let weight = weights[index];
            for (slot, value) in target.iter_mut().zip(values) {
                *slot += weight * value;
            }
        }
        let mut host = self
            .host_diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        host.cpu_expert_calls = host.cpu_expert_calls.saturating_add(experts.len() as u64);
        Ok(output)
    }

    fn run_with_loader<F>(
        &self,
        key: ExpertKey,
        load: F,
        hidden: &[f32],
    ) -> Result<Vec<f32>, Nvfp4ExecuteError>
    where
        F: FnOnce() -> Result<Vec<u8>, Nvfp4ExecuteError>,
    {
        #[cfg(feature = "cuda")]
        if let Nvfp4Backend::Cuda(executor) = &self.backend {
            let resident = match executor.resident_lookup(&key) {
                Some(lease) => lease,
                None => {
                    let payload = load()?;
                    Nvfp4Expert::from_payload(self.hidden, self.intermediate, &payload)?;
                    executor
                        .resident_admit(key, &payload)
                        .map_err(|error| Nvfp4ExecuteError::Cuda(error.to_string()))?
                }
            };
            return executor
                .run_resident(&resident, hidden)
                .map_err(|error| Nvfp4ExecuteError::Cuda(error.to_string()));
        }

        let lease = match self.cache.lookup(&key) {
            Some(lease) => lease,
            None => self
                .cache
                .admit(key, load()?)
                .map_err(Nvfp4ExecuteError::Cache)?,
        };
        match &self.backend {
            Nvfp4Backend::Cpu => {
                let expert =
                    Nvfp4Expert::from_payload(self.hidden, self.intermediate, lease.bytes())?;
                expert.run(hidden)
            }
            #[cfg(feature = "cuda")]
            Nvfp4Backend::Cuda(_) => unreachable!("o caminho CUDA usa o cache residente"),
        }
    }
}

fn select_backend_kind(
    cuda_compiled: bool,
    cpu_requested: bool,
) -> Result<Nvfp4BackendKind, Nvfp4Error> {
    if cpu_requested {
        Ok(Nvfp4BackendKind::CpuReference)
    } else if cuda_compiled {
        Ok(Nvfp4BackendKind::Cuda)
    } else {
        Err(Nvfp4Error::BackendUnavailable(
            "o binário não foi compilado com a feature cuda; compile com --features cuda ou peça o fallback explícito com POM_MOE_NVFP4_BACKEND=cpu".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn llama_rs_moe_reference_expert(
            model: *const llama_cpp_sys_2::llama_model,
            layer: i32,
            expert: u32,
            hidden: *const f32,
            hidden_count: usize,
            activation: i32,
            gate: *mut f32,
            gate_capacity: usize,
            up: *mut f32,
            up_capacity: usize,
            activated: *mut f32,
            activated_capacity: usize,
            output: *mut f32,
            output_capacity: usize,
        ) -> i32;
    }

    fn assert_close(stage: &str, layer: u32, ours: &[f32], reference: &[f32]) {
        assert_eq!(ours.len(), reference.len(), "{stage} layer {layer} length");
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = 0usize;
        let mut signed_sum = 0.0f64;
        let mut abs_sum = 0.0f64;
        let mut positive = 0usize;
        let mut negative = 0usize;
        let mut exact = 0usize;
        for (index, (&actual, &expected)) in ours.iter().zip(reference).enumerate() {
            assert!(
                actual.is_finite() && expected.is_finite(),
                "{stage} layer {layer} non-finite at {index}: ours={actual} reference={expected}"
            );
            let abs = (actual - expected).abs();
            let rel = abs / actual.abs().max(expected.abs()).max(1.0e-12);
            let signed = (actual - expected) as f64;
            signed_sum += signed;
            abs_sum += f64::from(abs);
            if signed > 0.0 {
                positive += 1;
            } else if signed < 0.0 {
                negative += 1;
            } else {
                exact += 1;
            }
            if abs > max_abs {
                max_abs = abs;
                worst = index;
            }
            max_rel = max_rel.max(rel);
        }
        if max_abs > 1.0e-4 && max_rel > 2.0e-3 {
            eprintln!(
                "{stage} layer {layer} error distribution: n={} mean_signed={} mean_abs={} positive={} negative={} exact={}",
                ours.len(),
                signed_sum / ours.len() as f64,
                abs_sum / ours.len() as f64,
                positive,
                negative,
                exact,
            );
        }
        assert!(
            max_abs <= 1.0e-4 || max_rel <= 2.0e-3,
            "{stage} layer {layer} diverged at {worst}: max_abs={max_abs} max_rel={max_rel} ours={} reference={}",
            ours[worst],
            reference[worst]
        );
    }

    #[test]
    #[ignore = "requires a real GGUF and the llama.cpp reference helper"]
    fn nvfp4_matches_llama_cpp_from_first_layer_and_stage() {
        let path = std::env::var_os("POM_MOE_NUMERIC_MODEL")
            .expect("POM_MOE_NUMERIC_MODEL must name a real MoE GGUF");
        let path = std::path::PathBuf::from(path);
        let file = std::fs::File::open(&path).unwrap();
        // The test derives the count from the real GGUF instead of encoding a
        // model's expert count in the harness.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
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
        let hidden = if std::env::var("POM_MOE_NUMERIC_INPUT").as_deref() == Ok("ones") {
            // Every Q8_0 block represents this input exactly. This diagnostic
            // removes input quantization from the native ggml matmul.
            vec![1.0; source.hidden()]
        } else {
            (0..source.hidden())
                .map(|index| (index as f32 % 17.0 - 8.0) * 0.03125)
                .collect::<Vec<_>>()
        };

        unsafe { llama_cpp_sys_2::llama_backend_init() };
        let path_c = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let mut params = unsafe { llama_cpp_sys_2::llama_model_default_params() };
        params.n_gpu_layers = 0;
        params.moe_external_executor = false;
        let model = unsafe { llama_cpp_sys_2::llama_model_load_from_file(path_c.as_ptr(), params) };
        assert!(!model.is_null(), "llama.cpp reference model failed to load");

        let activation = std::env::var("POM_MOE_NUMERIC_ACTIVATION")
            .expect("POM_MOE_NUMERIC_ACTIVATION must name the GGUF FFN activation")
            .parse::<i32>()
            .unwrap();
        for layer in 0..source.layer_count() {
            let payload = source.load_expert(layer, 0).unwrap();
            let payload_scales = (0..SCALE_COUNT)
                .map(|index| {
                    let start = 8 + index * 4;
                    f32::from_le_bytes(payload[start..start + 4].try_into().unwrap())
                })
                .collect::<Vec<_>>();
            if layer == 0 {
                eprintln!(
                    "llama.cpp oracle diagnostic layer={layer} expert=0 scales={payload_scales:?}"
                );
            }
            let expert =
                Nvfp4Expert::from_payload(source.hidden(), source.intermediate(), &payload)
                    .unwrap();
            let ours = expert.run_trace(&hidden).unwrap();
            let mut gate = vec![0.0; source.intermediate()];
            let mut up = vec![0.0; source.intermediate()];
            let mut activated = vec![0.0; source.intermediate()];
            let mut output = vec![0.0; source.hidden()];
            let status = unsafe {
                llama_rs_moe_reference_expert(
                    model,
                    layer as i32,
                    0,
                    hidden.as_ptr(),
                    hidden.len(),
                    activation,
                    gate.as_mut_ptr(),
                    gate.len(),
                    up.as_mut_ptr(),
                    up.len(),
                    activated.as_mut_ptr(),
                    activated.len(),
                    output.as_mut_ptr(),
                    output.len(),
                )
            };
            assert_eq!(status, 0, "llama.cpp reference failed at layer {layer}");

            assert_close("gate", layer, &ours.gate, &gate);
            assert_close("up", layer, &ours.up, &up);
            assert_close("activated", layer, &ours.activated, &activated);
            assert_close("output", layer, &ours.output, &output);
        }

        unsafe { llama_cpp_sys_2::llama_free_model(model) };
    }

    fn packed_rows(rows: usize, columns: usize, nibble: u8) -> Vec<u8> {
        let row_size = columns / BLOCK_ELEMENTS * BLOCK_BYTES;
        let mut output = vec![0u8; rows * row_size];
        for row in output.chunks_exact_mut(row_size) {
            for block in row.chunks_exact_mut(BLOCK_BYTES) {
                block[..4].fill(0x38);
                block[4..].fill(nibble | (nibble << 4));
            }
        }
        output
    }

    #[test]
    fn nvfp4_decodes_ue4m3_subblocks_and_e2m1_nibbles() {
        let hidden = 64;
        let intermediate = 64;
        let gate_up = packed_rows(2 * intermediate, hidden, 0x22);
        let down = packed_rows(hidden, intermediate, 0x22);
        let payload = pack_payload(&gate_up, &down, [1.0; SCALE_COUNT]).unwrap();
        let expert = Nvfp4Expert::from_payload(hidden, intermediate, &payload).unwrap();
        let output = expert.run(&[1.0; 64]).unwrap();
        assert_eq!(output.len(), hidden);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn nvfp4_follows_the_ggml_native_convention() {
        assert_eq!(fp4_value(0), 0.0);
        assert_eq!(fp4_value(1), 1.0);
        assert_eq!(fp4_value(7), 12.0);
        assert_eq!(fp4_value(9), -1.0);
        assert_eq!(ue4m3_to_f32(0x38), 0.5);
    }

    #[test]
    fn nvfp4_reproduces_the_ggml_q8_0_quantization() {
        let mut input = [0.0_f32; 32];
        input[0] = 1.0;
        input[1] = 0.5;
        input[2] = -0.5;
        let block = quantize_q8_0(&input)[0];

        assert_eq!(block.d, f16_to_f32(ggml_f32_to_f16_bits(1.0 / 127.0)));
        assert_eq!(block.qs[0], 127);
        assert_eq!(block.qs[1], 64);
        assert_eq!(block.qs[2], -64);
    }

    #[test]
    fn nvfp4_reproduces_the_ggml_fp16_geglu() {
        assert_eq!(ggml_f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(ggml_gelu_f16(1.0), 0.84130859375);
        assert_eq!(ggml_geglu_value(1.0, 2.0), 1.6826171875);
    }

    #[test]
    fn nvfp4_rejects_a_width_that_does_not_close_a_block() {
        assert!(matches!(
            Nvfp4Expert::packed_bytes(32, 64),
            Err(Nvfp4Error::Geometry(_))
        ));
    }

    #[test]
    fn nvfp4_rejects_a_truncated_payload_before_the_forward() {
        let result = Nvfp4Expert::from_payload(64, 64, &[0u8; 16]);
        assert!(matches!(result, Err(Nvfp4Error::PayloadLength { .. })));
    }

    #[test]
    fn nvfp4_does_not_fall_to_cpu_without_cuda_unless_explicitly_asked() {
        #[cfg(feature = "cuda")]
        assert_eq!(
            select_backend_kind(true, false).unwrap(),
            Nvfp4BackendKind::Cuda
        );

        #[cfg(not(feature = "cuda"))]
        assert!(matches!(
            select_backend_kind(false, false),
            Err(Nvfp4Error::BackendUnavailable(_))
        ));
    }

    #[test]
    fn nvfp4_split_uses_the_gguf_dimension_order() {
        let gate = TensorSpec {
            name: "blk.0.ffn_gate_exps.weight".into(),
            dims: vec![2816, 704, 128],
            dtype: 40,
            offset: 0,
        };

        assert_eq!(split_geometry(&gate).unwrap(), (2816, 704));
    }

    #[test]
    fn nvfp4_refuses_a_missing_scale_sidecar() {
        let gate = TensorSpec {
            name: "blk.0.ffn_gate_exps.weight".into(),
            dims: vec![2816, 704, 128],
            dtype: 40,
            offset: 0,
        };
        let result = find_sidecar(&[], &gate, "scale");
        assert!(matches!(
            result,
            Err(Q4LoadError::MissingTensor(name))
                if name == "blk.0.ffn_gate_exps.scale"
        ));
    }
}
