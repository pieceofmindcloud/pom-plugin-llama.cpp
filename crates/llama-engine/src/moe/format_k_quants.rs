use super::cache::{CacheError, ExpertCache, ExpertKey};
#[cfg(feature = "cuda")]
use super::cuda::CudaMoeError;
use super::format_q4_0::{parse_tensor_table, read_scalar_vector, Q4LoadError, TensorSpec};
use memmap2::Mmap;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const BLOCK_ELEMENTS: usize = 256;
const Q2_K_BLOCK_BYTES: usize = 84;
const Q3_K_BLOCK_BYTES: usize = 110;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;
const Q8_K_BLOCK_BYTES: usize = 292;
const PAYLOAD_HEADER_BYTES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
struct KQuantScales {
    gate: f32,
    up: f32,
    down: f32,
    gate_input: f32,
    up_input: f32,
    down_input: f32,
}

impl Default for KQuantScales {
    fn default() -> Self {
        Self {
            gate: 1.0,
            up: 1.0,
            down: 1.0,
            gate_input: 1.0,
            up_input: 1.0,
            down_input: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuantFormat {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KQuantError {
    Geometry(&'static str),
    Payload(&'static str),
    PayloadLength {
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "cuda")]
    BackendUnavailable(String),
}

impl fmt::Display for KQuantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(reason) => write!(formatter, "geometria K-quant inválida: {reason}"),
            Self::Payload(reason) => write!(formatter, "payload K-quant inválido: {reason}"),
            Self::PayloadLength { expected, actual } => write!(
                formatter,
                "payload K-quant tem {actual} bytes, mas a geometria exige {expected}"
            ),
            #[cfg(feature = "cuda")]
            Self::BackendUnavailable(reason) => {
                write!(formatter, "backend CUDA K-quant indisponível: {reason}")
            }
        }
    }
}

impl std::error::Error for KQuantError {}

#[derive(Debug)]
pub enum KQuantExecuteError {
    Cache(CacheError),
    Format(KQuantError),
    Source(Q4LoadError),
    HiddenWidth {
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
    #[cfg(feature = "cuda")]
    CudaNonFinite {
        stage: &'static str,
        index: usize,
        count: usize,
        kind: &'static str,
    },
}

impl fmt::Display for KQuantExecuteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
            Self::Source(error) => error.fmt(formatter),
            Self::HiddenWidth { expected, actual } => {
                write!(
                    formatter,
                    "hidden K-quant tem {actual} elementos, esperado {expected}"
                )
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(error) => write!(formatter, "executor CUDA K-quant recusou: {error}"),
            #[cfg(feature = "cuda")]
            Self::CudaNonFinite {
                stage,
                index,
                count,
                kind,
            } => write!(
                formatter,
                "executor CUDA MoE produziu {count} valores não finitos ({kind}) no estágio {stage}, primeiro índice da saída {index}"
            ),
        }
    }
}

impl std::error::Error for KQuantExecuteError {}

impl From<CacheError> for KQuantExecuteError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

impl From<KQuantError> for KQuantExecuteError {
    fn from(error: KQuantError) -> Self {
        Self::Format(error)
    }
}

impl From<Q4LoadError> for KQuantExecuteError {
    fn from(error: Q4LoadError) -> Self {
        Self::Source(error)
    }
}

impl KQuantFormat {
    pub fn from_dtype(dtype: u32) -> Result<Self, Q4LoadError> {
        match dtype {
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            other => Err(Q4LoadError::UnsupportedTensorType(other)),
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Q2K => Q2_K_BLOCK_BYTES,
            Self::Q3K => Q3_K_BLOCK_BYTES,
            Self::Q4K => Q4_K_BLOCK_BYTES,
            Self::Q5K => Q5_K_BLOCK_BYTES,
            Self::Q6K => Q6_K_BLOCK_BYTES,
            Self::Q8K => Q8_K_BLOCK_BYTES,
        }
    }

    pub fn dtype(self) -> u32 {
        match self {
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q2K => "q2_k",
            Self::Q3K => "q3_k",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::Q8K => "q8_k",
        }
    }

    /// A ponte CUDA é dirigida pelo dtype, não pelo modelo. Q2_K, Q3_K, Q4_K,
    /// Q5_K e Q6_K têm decodificação coberta pelos kernels residentes. Q8_K
    /// fica de fora porque só aparece como formato de ativação, nunca de peso
    /// de expert num GGUF servido.
    pub fn cuda_supported(self) -> bool {
        matches!(
            self,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K
        )
    }
}

pub struct KQuantExpert<'a> {
    format: KQuantFormat,
    hidden: usize,
    intermediate: usize,
    gate_up: &'a [u8],
    down: &'a [u8],
    scales: KQuantScales,
}

impl<'a> KQuantExpert<'a> {
    pub fn packed_bytes(
        format: KQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<usize, KQuantError> {
        let gate_up = (2 * intermediate)
            .checked_mul(row_bytes(format, hidden)?)
            .ok_or(KQuantError::Geometry("overflow em gate_up"))?;
        let down = hidden
            .checked_mul(row_bytes(format, intermediate)?)
            .ok_or(KQuantError::Geometry("overflow em down"))?;
        PAYLOAD_HEADER_BYTES
            .checked_add(gate_up)
            .and_then(|bytes| bytes.checked_add(down))
            .ok_or(KQuantError::Geometry("overflow no payload"))
    }

    pub fn validate_packed_payload_length(
        format: KQuantFormat,
        hidden: usize,
        intermediate: usize,
        actual: usize,
    ) -> Result<(), KQuantError> {
        let expected = Self::packed_bytes(format, hidden, intermediate)?;
        if actual != expected {
            return Err(KQuantError::PayloadLength { expected, actual });
        }
        Ok(())
    }

    pub fn from_payload(
        format: KQuantFormat,
        hidden: usize,
        intermediate: usize,
        payload: &'a [u8],
    ) -> Result<Self, KQuantError> {
        Self::validate_packed_payload_length(format, hidden, intermediate, payload.len())?;
        let gate_up_len = u32::from_le_bytes(
            payload[..4]
                .try_into()
                .map_err(|_| KQuantError::Payload("cabeçalho gate_up ausente"))?,
        ) as usize;
        let down_len = u32::from_le_bytes(
            payload[4..8]
                .try_into()
                .map_err(|_| KQuantError::Payload("cabeçalho down ausente"))?,
        ) as usize;
        let expected_gate_up = (2 * intermediate) * row_bytes(format, hidden)?;
        let expected_down = hidden * row_bytes(format, intermediate)?;
        if gate_up_len != expected_gate_up || down_len != expected_down {
            return Err(KQuantError::Payload(
                "tamanhos gate_up/down não correspondem à geometria",
            ));
        }
        let mut scales = [0.0f32; 6];
        for (index, scale) in scales.iter_mut().enumerate() {
            let start = 8 + index * 4;
            *scale = f32::from_le_bytes(
                payload[start..start + 4]
                    .try_into()
                    .map_err(|_| KQuantError::Payload("escala ausente"))?,
            );
            if !scale.is_finite() {
                return Err(KQuantError::Payload("escala não finita"));
            }
        }
        let gate_up_start = PAYLOAD_HEADER_BYTES;
        let down_start = gate_up_start + gate_up_len;
        if down_start + down_len != payload.len() {
            return Err(KQuantError::Payload("seções não cobrem o payload"));
        }
        Ok(Self {
            format,
            hidden,
            intermediate,
            gate_up: &payload[gate_up_start..down_start],
            down: &payload[down_start..],
            scales: KQuantScales {
                gate: scales[0],
                up: scales[1],
                down: scales[2],
                gate_input: scales[3],
                up_input: scales[4],
                down_input: scales[5],
            },
        })
    }

    pub fn run(&self, hidden: &[f32]) -> Result<Vec<f32>, KQuantExecuteError> {
        if hidden.len() != self.hidden {
            return Err(KQuantExecuteError::HiddenWidth {
                expected: self.hidden,
                actual: hidden.len(),
            });
        }
        let gate_up = mat_vec(
            self.format,
            self.gate_up,
            2 * self.intermediate,
            self.hidden,
            hidden,
        )?;
        let gate = gate_up[..self.intermediate]
            .iter()
            .map(|value| value * self.scales.gate * self.scales.gate_input)
            .collect::<Vec<_>>();
        let up = gate_up[self.intermediate..]
            .iter()
            .map(|value| value * self.scales.up * self.scales.up_input)
            .collect::<Vec<_>>();
        let gated: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| silu(gate) * up)
            .collect();
        let output = mat_vec(
            self.format,
            self.down,
            self.hidden,
            self.intermediate,
            &gated,
        )?;
        Ok(output
            .into_iter()
            .map(|value| value * self.scales.down * self.scales.down_input)
            .collect())
    }
}

fn pack_payload(gate_up: &[u8], down: &[u8], scales: KQuantScales) -> Result<Vec<u8>, KQuantError> {
    if gate_up.len() > u32::MAX as usize || down.len() > u32::MAX as usize {
        return Err(KQuantError::Payload("bank excede u32"));
    }
    let mut payload = Vec::with_capacity(PAYLOAD_HEADER_BYTES + gate_up.len() + down.len());
    payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
    for scale in [
        scales.gate,
        scales.up,
        scales.down,
        scales.gate_input,
        scales.up_input,
        scales.down_input,
    ] {
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

pub struct KQuantSource {
    mmap: Mmap,
    data_start: usize,
    experts: u32,
    hidden: usize,
    intermediate: usize,
    format: KQuantFormat,
    tensors: Vec<TensorSpec>,
    gate_up: GateUpTensors,
    down: Vec<TensorSpec>,
    router: Vec<TensorSpec>,
}

impl KQuantSource {
    pub fn open_infer(path: impl AsRef<Path>, experts: u32) -> Result<Self, Q4LoadError> {
        let file = File::open(path.as_ref()).map_err(|error| Q4LoadError::Io(error.to_string()))?;
        // SAFETY: the descriptor remains valid while the mapping is created.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
        let (_, tensors) = parse_tensor_table(&mmap)?;
        let fused = tensors
            .iter()
            .find(|tensor| tensor.name.ends_with("ffn_gate_up_exps.weight"));
        let (format, hidden, intermediate, tensor_experts) = if let Some(tensor) = fused {
            let format = KQuantFormat::from_dtype(tensor.dtype)?;
            if tensor.dims.len() != 3 {
                return Err(Q4LoadError::InvalidTensor(
                    "gate_up K-quant precisa ser rank 3",
                ));
            }
            let hidden = dimension(tensor, 0, "hidden")?;
            let two_intermediate = dimension(tensor, 1, "intermediate")?;
            let tensor_experts = dimension_u32(tensor, 2, "experts")?;
            if !two_intermediate.is_multiple_of(2) {
                return Err(Q4LoadError::InvalidTensor(
                    "gate_up K-quant tem linhas ímpares",
                ));
            }
            (format, hidden, two_intermediate / 2, tensor_experts)
        } else {
            let gate = tensors
                .iter()
                .find(|tensor| tensor.name.ends_with("ffn_gate_exps.weight"))
                .ok_or_else(|| Q4LoadError::MissingTensor("ffn_gate_exps.weight".into()))?;
            let up = tensors
                .iter()
                .find(|tensor| tensor.name.ends_with("ffn_up_exps.weight"))
                .ok_or_else(|| Q4LoadError::MissingTensor("ffn_up_exps.weight".into()))?;
            let format = KQuantFormat::from_dtype(gate.dtype)?;
            if up.dtype != gate.dtype || gate.dims != up.dims || gate.dims.len() != 3 {
                return Err(Q4LoadError::InvalidTensor(
                    "gate e up K-quant têm geometrias ou dtypes diferentes",
                ));
            }
            (
                format,
                dimension(gate, 0, "hidden")?,
                dimension(gate, 1, "intermediate")?,
                dimension_u32(gate, 2, "experts")?,
            )
        };
        if tensor_experts != experts {
            return Err(Q4LoadError::InvalidTensor(
                "quantidade de experts não corresponde aos metadados MoE",
            ));
        }
        drop(mmap);
        Self::open(path, experts, format, hidden, intermediate)
    }

    fn open(
        path: impl AsRef<Path>,
        experts: u32,
        format: KQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, Q4LoadError> {
        KQuantExpert::packed_bytes(format, hidden, intermediate)
            .map_err(|_| Q4LoadError::InvalidTensor("geometria não fecha K-quant"))?;
        if experts == 0 {
            return Err(Q4LoadError::InvalidTensor("contagem de experts zero"));
        }
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
            experts,
            hidden,
            intermediate,
            format,
            tensors,
            gate_up,
            down,
            router,
        })
    }

    pub fn format(&self) -> KQuantFormat {
        self.format
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
        validate_tensor(
            down,
            self.format,
            self.experts,
            self.hidden,
            self.intermediate,
        )?;
        let down_per_expert = self
            .hidden
            .checked_mul(
                row_bytes(self.format, self.intermediate)
                    .map_err(|_| Q4LoadError::InvalidTensor("geometria da linha down K-quant"))?,
            )
            .ok_or(Q4LoadError::InvalidTensor("overflow em down K-quant"))?;
        let down_bytes = self.expert_slice(down, expert, down_per_expert)?;
        let (gate_bytes, up_bytes, scales) = match &self.gate_up {
            GateUpTensors::Fused(tensors) => {
                let tensor = self.tensor_for_layer(tensors, layer, "ffn_gate_up_exps.weight")?;
                validate_tensor(
                    tensor,
                    self.format,
                    self.experts,
                    2 * self.intermediate,
                    self.hidden,
                )?;
                let bytes = (2 * self.intermediate)
                    .checked_mul(row_bytes(self.format, self.hidden).map_err(|_| {
                        Q4LoadError::InvalidTensor("geometria da linha gate_up K-quant")
                    })?)
                    .ok_or(Q4LoadError::InvalidTensor("overflow em gate_up K-quant"))?;
                let scales = KQuantScales {
                    gate: self.sidecar_scale_with_fallback(
                        tensor,
                        "scale",
                        expert,
                        0,
                        self.split_tensor(layer, "ffn_gate_exps.weight"),
                    )?,
                    up: self.sidecar_scale_with_fallback(
                        tensor,
                        "scale",
                        expert,
                        1,
                        self.split_tensor(layer, "ffn_up_exps.weight"),
                    )?,
                    down: self.sidecar_scale(down, "scale", expert, 0)?,
                    gate_input: self.sidecar_scale_with_fallback(
                        tensor,
                        "input_scale",
                        expert,
                        0,
                        self.split_tensor(layer, "ffn_gate_exps.weight"),
                    )?,
                    up_input: self.sidecar_scale_with_fallback(
                        tensor,
                        "input_scale",
                        expert,
                        1,
                        self.split_tensor(layer, "ffn_up_exps.weight"),
                    )?,
                    down_input: self.sidecar_scale(down, "input_scale", expert, 0)?,
                };
                (
                    self.expert_slice(tensor, expert, bytes)?.to_vec(),
                    Vec::new(),
                    scales,
                )
            }
            GateUpTensors::Split { gate, up } => {
                let gate = self.tensor_for_layer(gate, layer, "ffn_gate_exps.weight")?;
                let up = self.tensor_for_layer(up, layer, "ffn_up_exps.weight")?;
                validate_tensor(
                    gate,
                    self.format,
                    self.experts,
                    self.intermediate,
                    self.hidden,
                )?;
                validate_tensor(
                    up,
                    self.format,
                    self.experts,
                    self.intermediate,
                    self.hidden,
                )?;
                let bytes = self
                    .intermediate
                    .checked_mul(row_bytes(self.format, self.hidden).map_err(|_| {
                        Q4LoadError::InvalidTensor("geometria da linha gate K-quant")
                    })?)
                    .ok_or(Q4LoadError::InvalidTensor("overflow em gate/up K-quant"))?;
                let scales = KQuantScales {
                    gate: self.sidecar_scale(gate, "scale", expert, 0)?,
                    up: self.sidecar_scale(up, "scale", expert, 0)?,
                    down: self.sidecar_scale(down, "scale", expert, 0)?,
                    gate_input: self.sidecar_scale(gate, "input_scale", expert, 0)?,
                    up_input: self.sidecar_scale(up, "input_scale", expert, 0)?,
                    down_input: self.sidecar_scale(down, "input_scale", expert, 0)?,
                };
                (
                    self.expert_slice(gate, expert, bytes)?.to_vec(),
                    self.expert_slice(up, expert, bytes)?.to_vec(),
                    scales,
                )
            }
        };
        let mut gate_up = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        gate_up.extend_from_slice(&gate_bytes);
        gate_up.extend_from_slice(&up_bytes);
        pack_payload(&gate_up, down_bytes, scales)
            .map_err(|_| Q4LoadError::InvalidTensor("payload K-quant"))
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
        let mut logits = Vec::with_capacity(self.experts as usize);
        for expert in 0..self.experts as usize {
            let row = &raw
                [expert * self.hidden * element_bytes..(expert + 1) * self.hidden * element_bytes];
            logits.push(
                row.chunks_exact(element_bytes)
                    .zip(hidden)
                    .map(|(encoded, &input)| decode_scalar(encoded, tensor.dtype) * input)
                    .sum(),
            );
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
        ensure_format(tensor, self.format)?;
        let total = bytes_per_expert
            .checked_mul(self.experts as usize)
            .ok_or(Q4LoadError::InvalidTensor("overflow no tensor K-quant"))?;
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

    fn sidecar_scale(
        &self,
        tensor: &TensorSpec,
        suffix: &str,
        expert: u32,
        component: usize,
    ) -> Result<f32, Q4LoadError> {
        Ok(self
            .sidecar_scale_value(tensor, suffix, expert, component)?
            .unwrap_or(1.0))
    }

    fn sidecar_scale_with_fallback(
        &self,
        tensor: &TensorSpec,
        suffix: &str,
        expert: u32,
        component: usize,
        fallback: Option<&TensorSpec>,
    ) -> Result<f32, Q4LoadError> {
        let value = self.sidecar_scale_value(tensor, suffix, expert, component)?;
        let value = match value {
            Some(value) => Some(value),
            None => fallback
                .map(|tensor| self.sidecar_scale_value(tensor, suffix, expert, 0))
                .transpose()?
                .flatten(),
        };
        Ok(value.unwrap_or(1.0))
    }

    fn sidecar_scale_value(
        &self,
        tensor: &TensorSpec,
        suffix: &str,
        expert: u32,
        component: usize,
    ) -> Result<Option<f32>, Q4LoadError> {
        let stem = tensor.name.strip_suffix(".weight").unwrap_or(&tensor.name);
        let name = format!("{stem}.{suffix}");
        let Some(sidecar) = self.tensors.iter().find(|candidate| candidate.name == name) else {
            return Ok(None);
        };
        let values = read_scalar_vector(&self.mmap, self.data_start, sidecar)?;
        let experts = self.experts as usize;
        let index = if values.len() == experts {
            expert as usize
        } else if values.len() == 2 * experts {
            component
                .checked_mul(experts)
                .and_then(|offset| offset.checked_add(expert as usize))
                .ok_or(Q4LoadError::InvalidTensor("índice da escala excede usize"))?
        } else {
            return Err(Q4LoadError::InvalidTensor(
                "escala de expert tem geometria errada",
            ));
        };
        values
            .get(index)
            .copied()
            .map(Some)
            .ok_or(Q4LoadError::InvalidTensor("escala de expert ausente"))
    }

    fn split_tensor(&self, layer: u32, suffix: &str) -> Option<&TensorSpec> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == format!("blk.{layer}.{suffix}"))
    }
}

fn ensure_format(tensor: &TensorSpec, format: KQuantFormat) -> Result<(), Q4LoadError> {
    let expected = format.dtype();
    if tensor.dtype != expected {
        return Err(Q4LoadError::UnsupportedTensorType(tensor.dtype));
    }
    Ok(())
}

fn validate_tensor(
    tensor: &TensorSpec,
    format: KQuantFormat,
    experts: u32,
    rows: usize,
    columns: usize,
) -> Result<(), Q4LoadError> {
    ensure_format(tensor, format)?;
    if tensor.dims.len() != 3
        // GGUF stores the fastest-changing matrix dimension first: [columns,
        // rows, experts]. The executor's arguments are intentionally rows,
        // columns because that is the mat-vec contract.
        || tensor.dims[0] != columns as u64
        || tensor.dims[1] != rows as u64
        || tensor.dims[2] != experts as u64
    {
        return Err(Q4LoadError::InvalidTensor(
            "dimensões não correspondem ao modelo",
        ));
    }
    Ok(())
}

fn dimension(tensor: &TensorSpec, index: usize, name: &'static str) -> Result<usize, Q4LoadError> {
    tensor
        .dims
        .get(index)
        .copied()
        .ok_or(Q4LoadError::InvalidTensor("rank do tensor K-quant"))
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
        .ok_or(Q4LoadError::InvalidTensor("rank do tensor K-quant"))
        .and_then(|value| u32::try_from(value).map_err(|_| Q4LoadError::InvalidTensor(name)))
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

pub struct KQuantExecutor {
    cache: Arc<ExpertCache>,
    format: KQuantFormat,
    hidden: usize,
    intermediate: usize,
    #[cfg(feature = "cuda")]
    cuda: Option<super::cuda::KQuantCudaExecutor>,
}

impl KQuantExecutor {
    pub fn new(
        cache: Arc<ExpertCache>,
        format: KQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, KQuantError> {
        KQuantExpert::packed_bytes(format, hidden, intermediate)?;
        Ok(Self {
            cache,
            format,
            hidden,
            intermediate,
            #[cfg(feature = "cuda")]
            cuda: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_device_cache(
        cache: Arc<ExpertCache>,
        device_cache: Arc<super::cache::DeviceExpertCache>,
        format: KQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, KQuantError> {
        KQuantExpert::packed_bytes(format, hidden, intermediate)?;
        let cuda = super::cuda::KQuantCudaExecutor::new_with_cache(
            format,
            hidden,
            intermediate,
            device_cache,
        )
        .map_err(|error| KQuantError::BackendUnavailable(error.to_string()))?;
        Ok(Self {
            cache,
            format,
            hidden,
            intermediate,
            cuda: Some(cuda),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn uses_cuda(&self) -> bool {
        self.cuda.is_some()
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_diagnostics(&self) -> super::cuda::Nvfp4CudaDiagnostics {
        self.cuda.as_ref().map_or_else(
            Default::default,
            super::cuda::KQuantCudaExecutor::diagnostics,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.cuda.as_ref().map_or_else(
            Default::default,
            super::cuda::KQuantCudaExecutor::resident_cache_stats,
        )
    }

    pub fn run_from_source(
        &self,
        source: &KQuantSource,
        key: ExpertKey,
        layer: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>, KQuantExecuteError> {
        let expert = key.expert;
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() {
            return self.run_route_batch_from_source(
                source,
                &key.model,
                layer,
                &[expert],
                &[0],
                &[1.0],
                &[0, 1],
                hidden,
                1,
            );
        }
        self.run_with_loader(
            key,
            || {
                source
                    .load_expert(layer, expert)
                    .map_err(KQuantExecuteError::Source)
            },
            hidden,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn run_route_batch_from_source(
        &self,
        source: &KQuantSource,
        model: &str,
        layer: u32,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, KQuantExecuteError> {
        let Some(cuda) = self.cuda.as_ref() else {
            return Err(KQuantExecuteError::Cuda(
                "backend CUDA K-quant não foi selecionado".into(),
            ));
        };
        let mut resident = Vec::with_capacity(experts.len());
        for &expert in experts {
            let key = ExpertKey::new(model, layer, expert, self.format.name());
            if let Some(lease) = cuda.resident_lookup(&key) {
                resident.push(lease);
                continue;
            }
            let payload = source
                .load_expert(layer, expert)
                .map_err(KQuantExecuteError::Source)?;
            KQuantExpert::from_payload(self.format, self.hidden, self.intermediate, &payload)?;
            resident.push(
                cuda.resident_admit(key, &payload)
                    .map_err(|error| KQuantExecuteError::Cuda(error.to_string()))?,
            );
        }
        cuda.run_route_batch_resident(&resident, token_ids, weights, token_offsets, hidden, tokens)
            .map_err(|error| match error {
                CudaMoeError::NonFinite {
                    stage,
                    index,
                    count,
                    kind,
                } => KQuantExecuteError::CudaNonFinite {
                    stage,
                    index,
                    count,
                    kind,
                },
                error => KQuantExecuteError::Cuda(error.to_string()),
            })
    }

    fn run_with_loader<F, E>(
        &self,
        key: ExpertKey,
        load: F,
        hidden: &[f32],
    ) -> Result<Vec<f32>, KQuantExecuteError>
    where
        F: FnOnce() -> Result<Vec<u8>, E>,
        E: Into<KQuantExecuteError>,
    {
        let lease = match self.cache.lookup(&key) {
            Some(lease) => lease,
            None => self
                .cache
                .admit(key, load().map_err(Into::into)?)
                .map_err(KQuantExecuteError::Cache)?,
        };
        let expert =
            KQuantExpert::from_payload(self.format, self.hidden, self.intermediate, lease.bytes())?;
        expert.run(hidden)
    }

    #[cfg(feature = "cuda")]
    pub fn run_from_source_cpu(
        &self,
        source: &KQuantSource,
        key: ExpertKey,
        layer: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>, KQuantExecuteError> {
        let expert = key.expert;
        self.run_with_loader(
            key,
            || {
                source
                    .load_expert(layer, expert)
                    .map_err(KQuantExecuteError::Source)
            },
            hidden,
        )
    }
}

pub fn dequantize_row(format: KQuantFormat, packed: &[u8]) -> Result<Vec<f32>, KQuantError> {
    let block_bytes = format.block_bytes();
    if packed.len() < block_bytes || !packed.len().is_multiple_of(block_bytes) {
        return Err(KQuantError::PayloadLength {
            expected: block_bytes,
            actual: packed.len(),
        });
    }
    let mut values = Vec::with_capacity(packed.len() / block_bytes * 256);
    for block in packed.chunks_exact(block_bytes) {
        match format {
            KQuantFormat::Q2K => dequantize_q2_k_block(block, &mut values),
            KQuantFormat::Q3K => dequantize_q3_k_block(block, &mut values),
            KQuantFormat::Q4K => dequantize_q4_k_block(block, &mut values),
            KQuantFormat::Q5K => dequantize_q5_k_block(block, &mut values),
            KQuantFormat::Q6K => dequantize_q6_k_block(block, &mut values),
            KQuantFormat::Q8K => dequantize_q8_k_block(block, &mut values),
        }
    }
    Ok(values)
}

fn dequantize_q2_k_block(block: &[u8], output: &mut Vec<f32>) {
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let min = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
    let scales = &block[..16];
    let quants = &block[16..80];
    let mut scale_index = 0;
    for chunk in quants.chunks_exact(32).take(2) {
        let mut shift = 0;
        for _ in 0..4 {
            let scale = scales[scale_index];
            let dl = d * f32::from(scale & 0x0f);
            let ml = min * f32::from(scale >> 4);
            output.extend(
                chunk
                    .iter()
                    .take(16)
                    .map(|packed| dl * f32::from((packed >> shift) & 3) - ml),
            );
            scale_index += 1;
            let scale = scales[scale_index];
            let dl = d * f32::from(scale & 0x0f);
            let ml = min * f32::from(scale >> 4);
            output.extend(
                chunk
                    .iter()
                    .skip(16)
                    .map(|packed| dl * f32::from((packed >> shift) & 3) - ml),
            );
            scale_index += 1;
            shift += 2;
        }
    }
}

fn dequantize_q3_k_block(block: &[u8], output: &mut Vec<f32>) {
    let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
    let high = &block[..32];
    let quants = &block[32..96];
    let mut aux = [0u32; 4];
    for (word, bytes) in aux.iter_mut().zip(block[96..108].chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    let mask1 = 0x0303_0303u32;
    let mask2 = 0x0f0f_0f0fu32;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & mask2) | (((tmp >> 4) & mask1) << 4);
    aux[3] = ((aux[1] >> 4) & mask2) | (((tmp >> 6) & mask1) << 4);
    aux[0] = (aux[0] & mask2) | ((tmp & mask1) << 4);
    aux[1] = (aux[1] & mask2) | (((tmp >> 2) & mask1) << 4);
    let scales = aux
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let mut scale_index = 0;
    let mut high_mask = 1u8;
    for chunk in quants.chunks_exact(32).take(2) {
        let mut shift = 0;
        for _ in 0..4 {
            let scale = i32::from(scales[scale_index] as i8) - 32;
            let factor = d * scale as f32;
            output.extend(chunk.iter().take(16).enumerate().map(|(index, packed)| {
                factor
                    * (i32::from((packed >> shift) & 3)
                        - if high[index] & high_mask != 0 { 0 } else { 4 })
                        as f32
            }));
            scale_index += 1;
            let scale = i32::from(scales[scale_index] as i8) - 32;
            let factor = d * scale as f32;
            output.extend(chunk.iter().skip(16).enumerate().map(|(index, packed)| {
                factor
                    * (i32::from((packed >> shift) & 3)
                        - if high[index + 16] & high_mask != 0 {
                            0
                        } else {
                            4
                        }) as f32
            }));
            scale_index += 1;
            shift += 2;
            high_mask <<= 1;
        }
    }
}

fn dequantize_q4_k_block(block: &[u8], output: &mut Vec<f32>) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let quants = &block[16..144];
    let mut scale_index = 0;
    for chunk in quants.chunks_exact(32).take(4) {
        let (scale0, min0) = get_scale_min_k4(scale_index, scales);
        let (scale1, min1) = get_scale_min_k4(scale_index + 1, scales);
        output.extend(
            chunk
                .iter()
                .map(|packed| d * scale0 * f32::from(packed & 0x0f) - min * min0),
        );
        output.extend(
            chunk
                .iter()
                .map(|packed| d * scale1 * f32::from(packed >> 4) - min * min1),
        );
        scale_index += 2;
    }
}

fn get_scale_min_k4(index: usize, scales: &[u8]) -> (f32, f32) {
    let (scale, min) = if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    };
    (f32::from(scale), f32::from(min))
}

fn dequantize_q6_k_block(block: &[u8], output: &mut Vec<f32>) {
    // GGML block_q6_K: ql[128], qh[64], scales[16], d[2].
    let ql = &block[..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    for half in 0..2 {
        let ql = &ql[half * 64..(half + 1) * 64];
        let qh = &qh[half * 32..(half + 1) * 32];
        let scales = &scales[half * 8..(half + 1) * 8];
        let mut values = [0.0f32; 128];
        for lane in 0..32 {
            let group = lane / 16;
            let high = qh[lane];
            let q1 = i32::from((ql[lane] & 0x0f) | ((high & 3) << 4)) - 32;
            let q2 = i32::from((ql[lane + 32] & 0x0f) | (((high >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((ql[lane] >> 4) | (((high >> 4) & 3) << 4)) - 32;
            let q4 = i32::from((ql[lane + 32] >> 4) | (((high >> 6) & 3) << 4)) - 32;
            values[lane] = d * f32::from(scales[group] as i8) * q1 as f32;
            values[lane + 32] = d * f32::from(scales[group + 2] as i8) * q2 as f32;
            values[lane + 64] = d * f32::from(scales[group + 4] as i8) * q3 as f32;
            values[lane + 96] = d * f32::from(scales[group + 6] as i8) * q4 as f32;
        }
        output.extend(values);
    }
}

fn dequantize_q5_k_block(block: &[u8], output: &mut Vec<f32>) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let high = &block[16..48];
    let low = &block[48..176];
    let mut scale_index = 0;
    let mut mask_low = 1u8;
    let mut mask_high = 2u8;
    for chunk in low.chunks_exact(32).take(4) {
        let (scale0, min0) = get_scale_min_k4(scale_index, scales);
        let (scale1, min1) = get_scale_min_k4(scale_index + 1, scales);
        output.extend(chunk.iter().enumerate().map(|(index, packed)| {
            d * scale0
                * (f32::from(packed & 0x0f)
                    + if high[index] & mask_low != 0 {
                        16.0
                    } else {
                        0.0
                    })
                - min * min0
        }));
        output.extend(chunk.iter().enumerate().map(|(index, packed)| {
            d * scale1
                * (f32::from(packed >> 4)
                    + if high[index] & mask_high != 0 {
                        16.0
                    } else {
                        0.0
                    })
                - min * min1
        }));
        scale_index += 2;
        mask_low <<= 2;
        mask_high <<= 2;
    }
}

fn dequantize_q8_k_block(block: &[u8], output: &mut Vec<f32>) {
    let d = f32::from_le_bytes(block[..4].try_into().unwrap());
    output.extend(
        block[4..260]
            .iter()
            .map(|value| d * f32::from(*value as i8)),
    );
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;
    match exponent {
        0 if fraction == 0 => sign * 0.0,
        0 => sign * (fraction as f32 / 1024.0) * 2.0f32.powi(-14),
        31 if fraction == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        exponent => sign * (1.0 + fraction as f32 / 1024.0) * 2.0f32.powi(exponent - 15),
    }
}

pub fn row_bytes(format: KQuantFormat, columns: usize) -> Result<usize, KQuantError> {
    if columns == 0 || !columns.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(KQuantError::Geometry(
            "dimensão precisa fechar superblocos de 256",
        ));
    }
    columns
        .checked_div(BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(format.block_bytes()))
        .ok_or(KQuantError::Geometry("overflow na linha"))
}

fn mat_vec(
    format: KQuantFormat,
    matrix: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>, KQuantError> {
    if input.len() != columns {
        return Err(KQuantError::Geometry("entrada não corresponde às colunas"));
    }
    let row_size = row_bytes(format, columns)?;
    let expected = rows
        .checked_mul(row_size)
        .ok_or(KQuantError::Geometry("overflow na matriz"))?;
    if matrix.len() != expected {
        return Err(KQuantError::Payload("tamanho da matriz não corresponde"));
    }
    (0..rows)
        .map(|row| {
            let decoded = dequantize_row(format, &matrix[row * row_size..(row + 1) * row_size])?;
            Ok(decoded
                .iter()
                .zip(input)
                .map(|(&weight, &value)| weight * value)
                .sum())
        })
        .collect()
}

fn silu(value: f32) -> f32 {
    if value == f32::NEG_INFINITY {
        return 0.0;
    }
    let value_exp = if value < 0.0 {
        value.exp()
    } else {
        (-value).exp()
    };
    if value < 0.0 {
        value * value_exp / (1.0 + value_exp)
    } else {
        value / (1.0 + value_exp)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn stable_silu_handles_negative_infinity() {
        assert_eq!(silu(f32::NEG_INFINITY), 0.0);
        assert!(silu(-1_000.0).is_finite());
    }

    #[test]
    fn q4_k_decodes_scale_minimum_and_nibbles_in_ggml_order() {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        block[4] = 2;
        block[5] = 3;
        block[8] = 1;
        block[9] = 2;
        block[16..48].fill(0x43);

        let values = dequantize_row(KQuantFormat::Q4K, &block).unwrap();
        assert_eq!(values.len(), 256);
        assert!((values[0] - 5.5).abs() < 1e-6);
        assert!((values[31] - 5.5).abs() < 1e-6);
        assert!((values[32] - 11.0).abs() < 1e-6);
        assert!((values[63] - 11.0).abs() < 1e-6);
        assert_eq!(values[64], 0.0);
    }

    #[test]
    fn q6_k_decodes_high_low_bits_and_per_group_scales() {
        let mut block = vec![0u8; 210];
        block[..128].fill(0x11);
        block[128..192].fill(0xaa);
        block[192] = 2;
        block[194] = 3;
        block[196] = 4;
        block[198] = 5;
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());

        let values = dequantize_row(KQuantFormat::Q6K, &block).unwrap();
        assert_eq!(values.len(), 256);
        assert_eq!(values[0], 2.0);
        assert_eq!(values[15], 2.0);
        assert_eq!(values[16], 0.0);
        assert_eq!(values[32], 3.0);
        assert_eq!(values[64], 4.0);
        assert_eq!(values[96], 5.0);
        assert_eq!(values[128], 0.0);
    }

    #[test]
    fn q6_k_decodes_a_real_ggml_block_against_the_reference() {
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        block[..128].fill(0x11);
        block[128..192].fill(0xaa);
        for (index, scale) in block[192..208].iter_mut().enumerate() {
            *scale = (index + 2) as u8;
        }
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());

        let values = dequantize_row(KQuantFormat::Q6K, &block).unwrap();

        assert_eq!(values.len(), 256);
        assert_eq!(values[0], 2.0);
        assert_eq!(values[15], 2.0);
        assert_eq!(values[16], 3.0);
        assert_eq!(values[31], 3.0);
        assert_eq!(values[32], 4.0);
        assert_eq!(values[64], 6.0);
        assert_eq!(values[96], 8.0);
        assert_eq!(values[128], 10.0);
    }

    #[test]
    fn q6_k_gate_up_eight_experts_hidden_2048_is_finite() {
        let format = KQuantFormat::Q6K;
        let hidden = 2048;
        let intermediate = 256;
        let gate_up = q6_matrix(2 * intermediate, hidden, true);
        let down = q6_matrix(hidden, intermediate, true);
        let payload = pack_payload(&gate_up, &down, KQuantScales::default()).unwrap();
        let input = (0..hidden)
            .map(|index| (index as f32 * 0.001).sin())
            .collect::<Vec<_>>();

        for expert_index in 0..8 {
            let expert =
                KQuantExpert::from_payload(format, hidden, intermediate, &payload).unwrap();
            let gate_up =
                mat_vec(format, expert.gate_up, 2 * intermediate, hidden, &input).unwrap();
            assert!(
                gate_up.iter().all(|value| value.is_finite()),
                "expert {expert_index} produziu gate_up não finito"
            );
            let output = expert.run(&input).unwrap();
            assert_eq!(output.len(), hidden);
            assert!(
                output.iter().all(|value| value.is_finite()),
                "expert {expert_index} produziu saída não finita"
            );
        }
    }

    #[test]
    fn q6_k_source_loads_eight_experts_hidden_2048_and_runs_a_finite_output() {
        let path = std::env::temp_dir().join(format!(
            "pom-ftk-k-source-eight-q6-{}.gguf",
            uuid::Uuid::new_v4()
        ));
        let format = KQuantFormat::Q6K;
        let hidden = 2048;
        let intermediate = 256;
        write_k_source_with_dimensions_and_experts(
            &path,
            format.dtype(),
            format,
            GateUpLayout::Fused,
            false,
            hidden,
            intermediate,
            8,
        );
        let source = KQuantSource::open_infer(&path, 8).unwrap();
        let input = (0..hidden)
            .map(|index| (index as f32 * 0.001).sin())
            .collect::<Vec<_>>();

        for expert_index in 0..8 {
            let payload = source.load_expert(0, expert_index).unwrap();
            let expert =
                KQuantExpert::from_payload(format, hidden, intermediate, &payload).unwrap();
            let output = expert.run(&input).unwrap();
            assert_eq!(output.len(), hidden);
            assert!(
                output.iter().all(|value| value.is_finite()),
                "expert {expert_index} produziu saída não finita"
            );
        }

        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn remaining_k_quants_decode_complete_blocks() {
        for format in [
            KQuantFormat::Q2K,
            KQuantFormat::Q3K,
            KQuantFormat::Q5K,
            KQuantFormat::Q8K,
        ] {
            let values = dequantize_row(format, &test_block(format)).unwrap();
            assert_eq!(values.len(), 256, "formato {}", format.name());
            assert!(values.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn k_quant_rejects_a_truncated_row() {
        let error = dequantize_row(KQuantFormat::Q4K, &[0; 143]).unwrap_err();
        assert!(matches!(error, KQuantError::PayloadLength { .. }));
    }

    #[test]
    fn cuda_k_quant_payload_of_one_expert_rejects_a_shrunk_size() {
        let format = KQuantFormat::Q6K;
        let hidden = 256;
        let intermediate = 256;
        let expected = KQuantExpert::packed_bytes(format, hidden, intermediate).unwrap();
        let actual = expected / 8;

        let error =
            KQuantExpert::validate_packed_payload_length(format, hidden, intermediate, actual)
                .unwrap_err();

        assert_eq!(error, KQuantError::PayloadLength { expected, actual });
    }

    #[test]
    fn cuda_k_quant_payload_of_one_expert_accepts_the_exact_size() {
        let format = KQuantFormat::Q6K;
        let expected = KQuantExpert::packed_bytes(format, 256, 256).unwrap();

        KQuantExpert::validate_packed_payload_length(format, 256, 256, expected).unwrap();
    }

    #[test]
    fn k_quant_applies_scale_and_input_scale_to_the_payload() {
        let format = KQuantFormat::Q4K;
        let hidden = 256;
        let intermediate = 256;
        let gate_up = packed_rows(2 * intermediate, hidden, format);
        let down = packed_rows(hidden, intermediate, format);
        let scales = KQuantScales {
            gate: 2.0,
            up: 3.0,
            down: 5.0,
            gate_input: 7.0,
            up_input: 11.0,
            down_input: 13.0,
        };
        let payload = pack_payload(&gate_up, &down, scales).unwrap();
        let expert = KQuantExpert::from_payload(format, hidden, intermediate, &payload).unwrap();
        let input = vec![1.0; hidden];
        let raw_gate_up = mat_vec(format, &gate_up, 2 * intermediate, hidden, &input).unwrap();
        let expected_gate = raw_gate_up[..intermediate]
            .iter()
            .map(|value| value * scales.gate * scales.gate_input)
            .collect::<Vec<_>>();
        let expected_up = raw_gate_up[intermediate..]
            .iter()
            .map(|value| value * scales.up * scales.up_input)
            .collect::<Vec<_>>();
        let gated = expected_gate
            .iter()
            .zip(&expected_up)
            .map(|(gate, up)| silu(*gate) * up)
            .collect::<Vec<_>>();
        let expected = mat_vec(format, &down, hidden, intermediate, &gated)
            .unwrap()
            .into_iter()
            .map(|value| value * scales.down * scales.down_input)
            .collect::<Vec<_>>();
        assert_eq!(expert.run(&input).unwrap(), expected);
    }

    #[test]
    fn mmap_source_loads_k_quants_per_expert() {
        for format in [
            KQuantFormat::Q2K,
            KQuantFormat::Q3K,
            KQuantFormat::Q4K,
            KQuantFormat::Q5K,
            KQuantFormat::Q6K,
            KQuantFormat::Q8K,
        ] {
            let dtype = format.dtype();
            let path = std::env::temp_dir().join(format!(
                "pom-ftk-k-source-{dtype}-{}.gguf",
                std::process::id()
            ));
            write_k_source(&path, dtype, format);
            let source = KQuantSource::open_infer(&path, 2).unwrap();
            assert_eq!(source.format(), format);
            let payload = source.load_expert(0, 1).unwrap();
            let expert = KQuantExpert::from_payload(format, 256, 256, &payload).unwrap();
            assert_eq!(expert.run(&[1.0; 256]).unwrap().len(), 256);
            drop(source);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn mmap_source_loads_split_gate_and_up_in_k_quants() {
        for format in [
            KQuantFormat::Q2K,
            KQuantFormat::Q3K,
            KQuantFormat::Q4K,
            KQuantFormat::Q5K,
            KQuantFormat::Q6K,
            KQuantFormat::Q8K,
        ] {
            let dtype = format.dtype();
            let path = std::env::temp_dir().join(format!(
                "pom-ftk-k-source-split-{dtype}-{}.gguf",
                std::process::id()
            ));
            write_k_source_with_layout(&path, dtype, format, GateUpLayout::Split, false);
            let source = KQuantSource::open_infer(&path, 2).unwrap();
            assert_eq!(source.format(), format);
            let payload = source.load_expert(0, 1).unwrap();
            let expert = KQuantExpert::from_payload(format, 256, 256, &payload).unwrap();
            assert_eq!(expert.run(&[1.0; 256]).unwrap().len(), 256);
            drop(source);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn mmap_source_preserves_hidden_and_intermediate_in_the_ggml_split_layout() {
        let path = std::env::temp_dir().join(format!(
            "pom-ftk-k-source-split-rectangular-{}.gguf",
            std::process::id()
        ));
        let hidden = 2048;
        let intermediate = 512;
        write_k_source_with_dimensions(
            &path,
            KQuantFormat::Q6K.dtype(),
            KQuantFormat::Q6K,
            GateUpLayout::Split,
            false,
            hidden,
            intermediate,
        );

        let source = KQuantSource::open_infer(&path, 2).unwrap();

        assert_eq!(source.hidden(), hidden);
        assert_eq!(source.intermediate(), intermediate);
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mmap_source_prefers_fused_when_the_gguf_has_both_conventions() {
        let path = std::env::temp_dir().join(format!(
            "pom-ftk-k-source-mixed-{}.gguf",
            std::process::id()
        ));
        write_k_source_with_layout(
            &path,
            KQuantFormat::Q4K.dtype(),
            KQuantFormat::Q4K,
            GateUpLayout::Both,
            false,
        );
        let source = KQuantSource::open_infer(&path, 2).unwrap();
        let payload = source.load_expert(0, 1).unwrap();
        let expert = KQuantExpert::from_payload(KQuantFormat::Q4K, 256, 256, &payload).unwrap();
        assert_eq!(expert.run(&[1.0; 256]).unwrap().len(), 256);
        assert!(matches!(source.gate_up, GateUpTensors::Fused(_)));
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mmap_source_loads_the_fused_expert_sidecars() {
        let path = std::env::temp_dir().join(format!(
            "pom-ftk-k-source-scales-{}.gguf",
            std::process::id()
        ));
        write_k_source_with_layout(
            &path,
            KQuantFormat::Q4K.dtype(),
            KQuantFormat::Q4K,
            GateUpLayout::Fused,
            true,
        );
        let source = KQuantSource::open_infer(&path, 2).unwrap();
        let payload = source.load_expert(0, 1).unwrap();
        let expert = KQuantExpert::from_payload(KQuantFormat::Q4K, 256, 256, &payload).unwrap();
        assert_eq!(
            expert.scales,
            KQuantScales {
                gate: 2.0,
                up: 3.0,
                down: 5.0,
                gate_input: 7.0,
                up_input: 11.0,
                down_input: 13.0,
            }
        );
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    pub fn write_k_source(path: &std::path::Path, dtype: u32, format: KQuantFormat) {
        write_k_source_with_layout(path, dtype, format, GateUpLayout::Fused, false);
    }

    #[derive(Clone, Copy)]
    enum GateUpLayout {
        Fused,
        Split,
        Both,
    }

    fn write_k_source_with_layout(
        path: &std::path::Path,
        dtype: u32,
        format: KQuantFormat,
        layout: GateUpLayout,
        with_sidecars: bool,
    ) {
        write_k_source_with_dimensions(path, dtype, format, layout, with_sidecars, 256, 256);
    }

    fn write_k_source_with_dimensions(
        path: &std::path::Path,
        dtype: u32,
        format: KQuantFormat,
        layout: GateUpLayout,
        with_sidecars: bool,
        hidden: usize,
        intermediate: usize,
    ) {
        write_k_source_with_dimensions_and_experts(
            path,
            dtype,
            format,
            layout,
            with_sidecars,
            hidden,
            intermediate,
            2,
        );
    }

    fn write_k_source_with_dimensions_and_experts(
        path: &std::path::Path,
        dtype: u32,
        format: KQuantFormat,
        layout: GateUpLayout,
        with_sidecars: bool,
        hidden: usize,
        intermediate: usize,
        experts: usize,
    ) {
        use std::fs::File;
        use std::io::Write;

        let row_bytes = format.block_bytes();
        let gate_up = packed_rows(experts * 2 * intermediate, hidden, format);
        let down = packed_rows(experts * hidden, intermediate, format);
        let mut router = Vec::with_capacity(experts * hidden * 4);
        for expert in 0..experts {
            for _ in 0..hidden {
                router.extend_from_slice(&((expert + 1) as f32).to_le_bytes());
            }
        }
        assert_eq!(
            gate_up.len(),
            experts * 2 * intermediate * (hidden / BLOCK_ELEMENTS) * row_bytes
        );
        let gate = packed_rows(experts * intermediate, hidden, format);
        let up = packed_rows(experts * intermediate, hidden, format);
        let mut file = Vec::new();
        file.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        file.extend_from_slice(&2u32.to_le_bytes());
        let base_tensor_count = match layout {
            GateUpLayout::Fused => 3u64,
            GateUpLayout::Split => 4u64,
            GateUpLayout::Both => 5u64,
        };
        let tensor_count = base_tensor_count + u64::from(with_sidecars) * 4;
        file.extend_from_slice(&tensor_count.to_le_bytes());
        file.extend_from_slice(&1u64.to_le_bytes());
        gguf_string(&mut file, "general.alignment");
        file.extend_from_slice(&4u32.to_le_bytes());
        file.extend_from_slice(&32u32.to_le_bytes());
        if matches!(layout, GateUpLayout::Split | GateUpLayout::Both) {
            gguf_tensor(
                &mut file,
                "blk.0.ffn_gate_exps.weight",
                &[hidden as u64, intermediate as u64, experts as u64],
                0,
                dtype,
            );
            gguf_tensor(
                &mut file,
                "blk.0.ffn_up_exps.weight",
                &[hidden as u64, intermediate as u64, experts as u64],
                gate.len() as u64,
                dtype,
            );
        }
        if matches!(layout, GateUpLayout::Fused | GateUpLayout::Both) {
            gguf_tensor(
                &mut file,
                "blk.0.ffn_gate_up_exps.weight",
                &[hidden as u64, (2 * intermediate) as u64, experts as u64],
                if matches!(layout, GateUpLayout::Both) {
                    (gate.len() + up.len()) as u64
                } else {
                    0
                },
                dtype,
            );
        }
        let gate_up_bytes = if matches!(layout, GateUpLayout::Split) {
            gate.len() + up.len()
        } else if matches!(layout, GateUpLayout::Both) {
            gate.len() + up.len() + gate_up.len()
        } else {
            gate_up.len()
        };
        gguf_tensor(
            &mut file,
            "blk.0.ffn_down_exps.weight",
            &[intermediate as u64, hidden as u64, experts as u64],
            gate_up_bytes as u64,
            dtype,
        );
        gguf_tensor(
            &mut file,
            "blk.0.ffn_gate_inp.weight",
            &[hidden as u64, experts as u64],
            (gate_up_bytes + down.len()) as u64,
            0,
        );
        let base_data_bytes = gate_up_bytes + down.len() + router.len();
        if with_sidecars {
            assert!(matches!(layout, GateUpLayout::Fused));
            for (name, count, offset) in [
                ("blk.0.ffn_gate_up_exps.scale", 4u64, base_data_bytes as u64),
                (
                    "blk.0.ffn_gate_up_exps.input_scale",
                    4,
                    (base_data_bytes + 16) as u64,
                ),
                (
                    "blk.0.ffn_down_exps.scale",
                    2,
                    (base_data_bytes + 32) as u64,
                ),
                (
                    "blk.0.ffn_down_exps.input_scale",
                    2,
                    (base_data_bytes + 40) as u64,
                ),
            ] {
                gguf_tensor(&mut file, name, &[count], offset, 0);
            }
        }
        let data_start = (file.len() + 31) / 32 * 32;
        file.resize(data_start, 0);
        if matches!(layout, GateUpLayout::Split) {
            file.extend_from_slice(&gate);
            file.extend_from_slice(&up);
        } else if matches!(layout, GateUpLayout::Fused) {
            file.extend_from_slice(&gate_up);
        } else {
            file.extend_from_slice(&gate);
            file.extend_from_slice(&up);
            file.extend_from_slice(&gate_up);
        }
        file.extend_from_slice(&down);
        file.extend_from_slice(&router);
        if with_sidecars {
            for value in [
                2.0f32, 2.0, 3.0, 3.0, 7.0, 7.0, 11.0, 11.0, 5.0, 5.0, 13.0, 13.0,
            ] {
                file.extend_from_slice(&value.to_le_bytes());
            }
        }
        File::create(path).unwrap().write_all(&file).unwrap();
    }

    fn packed_rows(rows: usize, columns: usize, format: KQuantFormat) -> Vec<u8> {
        let row_bytes = format.block_bytes();
        let blocks_per_row = columns / BLOCK_ELEMENTS;
        assert!(columns.is_multiple_of(BLOCK_ELEMENTS));
        let row_size = blocks_per_row * row_bytes;
        let mut output = vec![0u8; rows * row_size];
        for row in output.chunks_exact_mut(row_size) {
            for block in row.chunks_exact_mut(row_bytes) {
                match format {
                    KQuantFormat::Q2K => {
                        block[80..82].copy_from_slice(&0x3c00u16.to_le_bytes());
                        block[82..84].copy_from_slice(&0x3c00u16.to_le_bytes());
                        block[0..16].fill(1);
                        block[16..80].fill(0x55);
                    }
                    KQuantFormat::Q3K => {
                        block[96..108].fill(0);
                        block[108..110].copy_from_slice(&0x3c00u16.to_le_bytes());
                        block[32..96].fill(0x55);
                    }
                    KQuantFormat::Q4K => {
                        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                        block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
                        block[4..16].fill(1);
                        block[16..].fill(0x11);
                    }
                    KQuantFormat::Q5K => {
                        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                        block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
                        block[4..16].fill(1);
                        block[16..48].fill(0);
                        block[48..].fill(0x11);
                    }
                    KQuantFormat::Q6K => {
                        block[..128].fill(0x11);
                        block[0] = 0;
                        block[1] = 0x7e;
                        block[128..192].fill(0xaa);
                        block[192..208].fill(1);
                        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
                    }
                    KQuantFormat::Q8K => {
                        block[0..4].copy_from_slice(&1.0f32.to_le_bytes());
                        block[4..260].fill(1);
                    }
                }
            }
        }
        output
    }

    fn q6_matrix(rows: usize, columns: usize, trap_wrong_d_offset: bool) -> Vec<u8> {
        let row_size = columns / BLOCK_ELEMENTS * Q6_K_BLOCK_BYTES;
        let mut output = vec![0u8; rows * row_size];
        for block in output.chunks_exact_mut(Q6_K_BLOCK_BYTES) {
            block[..128].fill(0x11);
            if trap_wrong_d_offset {
                block[0] = 0;
                block[1] = 0x7e;
            }
            block[128..192].fill(0xaa);
            block[192..208].fill(1);
            block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        output
    }

    fn test_block(format: KQuantFormat) -> Vec<u8> {
        packed_rows(1, BLOCK_ELEMENTS, format)
    }

    fn gguf_string(output: &mut Vec<u8>, value: &str) {
        output.extend_from_slice(&(value.len() as u64).to_le_bytes());
        output.extend_from_slice(value.as_bytes());
    }

    fn gguf_tensor(output: &mut Vec<u8>, name: &str, dims: &[u64], offset: u64, dtype: u32) {
        gguf_string(output, name);
        output.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dimension in dims {
            output.extend_from_slice(&dimension.to_le_bytes());
        }
        output.extend_from_slice(&dtype.to_le_bytes());
        output.extend_from_slice(&offset.to_le_bytes());
    }
}
