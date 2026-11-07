//! Executor CPU mínimo para experts GGML Q4_0.
//!
//! O payload contém duas matrizes packed, `gate_up` e `down`, na ordem de
//! blocos do GGML. O executor faz o gate-up, SiLU gated e down projection. A
//! seleção top-k e a soma ponderada continuam sendo responsabilidade do bloco
//! MoE que o chamar.

use super::cache::{CacheError, ExpertCache, ExpertKey};
use memmap2::Mmap;
use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const BLOCK_ELEMENTS: usize = 32;
const BLOCK_BYTES: usize = 18;
const PAYLOAD_HEADER_BYTES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertFormat {
    Heterogeneous,
    F32,
    F16,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4_0,
    Q4K,
    Q2K,
    Q3K,
    Q5K,
    Q6K,
    Q8K,
    Iq3S,
    Iq2S,
    Iq4XS,
    Iq2XXS,
    Iq2XS,
    Iq3XXS,
    Iq1S,
    Iq4NL,
    Iq1M,
    Nvfp4,
    Bf16,
    Mxfp4,
    Tq1_0,
    Tq2_0,
    Q1_0,
    Q2_0,
}

pub fn ggml_type_name(dtype: u32) -> &'static str {
    match dtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        4 => "Q4_2 (removido)",
        5 => "Q4_3 (removido)",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        31 => "Q4_0_4_4 (removido)",
        32 => "Q4_0_4_8 (removido)",
        33 => "Q4_0_8_8 (removido)",
        34 => "TQ1_0",
        35 => "TQ2_0",
        36 => "IQ4_NL_4_4 (removido)",
        37 => "IQ4_NL_4_8 (removido)",
        38 => "IQ4_NL_8_8 (removido)",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        42 => "Q2_0",
        _ => "desconhecido",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Q4Error {
    Geometry(&'static str),
    Payload(&'static str),
    PayloadLength { expected: usize, actual: usize },
}

impl fmt::Display for Q4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(reason) => write!(formatter, "geometria Q4_0 inválida: {reason}"),
            Self::Payload(reason) => write!(formatter, "payload Q4_0 inválido: {reason}"),
            Self::PayloadLength { expected, actual } => write!(
                formatter,
                "payload Q4_0 tem {actual} bytes, mas a geometria exige {expected}"
            ),
        }
    }
}

impl std::error::Error for Q4Error {}

#[derive(Debug)]
pub enum Q4ExecuteError {
    Cache(CacheError),
    Format(Q4Error),
    Source(Q4LoadError),
    HiddenWidth { expected: usize, actual: usize },
}

impl fmt::Display for Q4ExecuteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
            Self::Source(error) => error.fmt(formatter),
            Self::HiddenWidth { expected, actual } => {
                write!(
                    formatter,
                    "hidden Q4_0 tem {actual} elementos, esperado {expected}"
                )
            }
        }
    }
}

impl std::error::Error for Q4ExecuteError {}

impl From<CacheError> for Q4ExecuteError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

impl From<Q4Error> for Q4ExecuteError {
    fn from(error: Q4Error) -> Self {
        Self::Format(error)
    }
}

impl From<Q4LoadError> for Q4ExecuteError {
    fn from(error: Q4LoadError) -> Self {
        Self::Source(error)
    }
}

pub struct Q4_0Expert<'a> {
    hidden: usize,
    intermediate: usize,
    gate_up: &'a [u8],
    down: &'a [u8],
}

impl<'a> Q4_0Expert<'a> {
    pub fn packed_bytes(hidden: usize, intermediate: usize) -> Result<usize, Q4Error> {
        if hidden == 0 || intermediate == 0 {
            return Err(Q4Error::Geometry("dimensão zero"));
        }
        if !hidden.is_multiple_of(BLOCK_ELEMENTS) || !intermediate.is_multiple_of(BLOCK_ELEMENTS) {
            return Err(Q4Error::Geometry(
                "hidden e intermediate precisam fechar blocos de 32",
            ));
        }
        let gate_up = (2 * intermediate)
            .checked_mul(row_bytes(hidden)?)
            .ok_or(Q4Error::Geometry("overflow em gate_up"))?;
        let down = hidden
            .checked_mul(row_bytes(intermediate)?)
            .ok_or(Q4Error::Geometry("overflow em down"))?;
        PAYLOAD_HEADER_BYTES
            .checked_add(gate_up)
            .and_then(|bytes| bytes.checked_add(down))
            .ok_or(Q4Error::Geometry("overflow no payload"))
    }

    pub fn from_payload(
        hidden: usize,
        intermediate: usize,
        payload: &'a [u8],
    ) -> Result<Self, Q4Error> {
        let expected = Self::packed_bytes(hidden, intermediate)?;
        if payload.len() != expected {
            return Err(Q4Error::PayloadLength {
                expected,
                actual: payload.len(),
            });
        }
        let gate_up_len = u32::from_le_bytes(
            payload[..4]
                .try_into()
                .map_err(|_| Q4Error::Payload("cabeçalho gate_up ausente"))?,
        ) as usize;
        let expected_gate_up = (2 * intermediate) * row_bytes(hidden)?;
        if gate_up_len != expected_gate_up {
            return Err(Q4Error::Payload(
                "tamanho gate_up não corresponde à geometria",
            ));
        }
        let down_len = u32::from_le_bytes(
            payload[4..8]
                .try_into()
                .map_err(|_| Q4Error::Payload("cabeçalho down ausente"))?,
        ) as usize;
        let expected_down = hidden * row_bytes(intermediate)?;
        if down_len != expected_down {
            return Err(Q4Error::Payload("tamanho down não corresponde à geometria"));
        }
        let gate_up_start = PAYLOAD_HEADER_BYTES;
        let down_start = gate_up_start + gate_up_len;
        if down_start + down_len != payload.len() {
            return Err(Q4Error::Payload("seções não cobrem o payload"));
        }
        Ok(Self {
            hidden,
            intermediate,
            gate_up: &payload[gate_up_start..down_start],
            down: &payload[down_start..],
        })
    }

    pub fn run(&self, hidden: &[f32]) -> Result<Vec<f32>, Q4ExecuteError> {
        if hidden.len() != self.hidden {
            return Err(Q4ExecuteError::HiddenWidth {
                expected: self.hidden,
                actual: hidden.len(),
            });
        }
        let gate_up = mat_vec(self.gate_up, 2 * self.intermediate, self.hidden, hidden);
        let gated: Vec<f32> = gate_up[..self.intermediate]
            .iter()
            .zip(&gate_up[self.intermediate..])
            .map(|(&gate, &up)| silu(gate) * up)
            .collect();
        Ok(mat_vec(self.down, self.hidden, self.intermediate, &gated))
    }
}

pub fn section_lengths(hidden: usize, intermediate: usize) -> Result<(usize, usize), Q4Error> {
    let gate_up = (2 * intermediate)
        .checked_mul(row_bytes(hidden)?)
        .ok_or(Q4Error::Geometry("overflow em gate_up"))?;
    let down = hidden
        .checked_mul(row_bytes(intermediate)?)
        .ok_or(Q4Error::Geometry("overflow em down"))?;
    Ok((gate_up, down))
}

/// Valida o payload e retorna os tamanhos das duas matrizes que a ponte CUDA
/// recebe sem o cabeçalho interno.
pub fn payload_section_lengths(
    hidden: usize,
    intermediate: usize,
    payload: &[u8],
) -> Result<(usize, usize), Q4Error> {
    let expert = Q4_0Expert::from_payload(hidden, intermediate, payload)?;
    Ok((expert.gate_up.len(), expert.down.len()))
}

/// Internal payload format used by the cache provider. It keeps the two bank
/// lengths explicit so a corrupt or mismatched expert cannot be interpreted as
/// a different geometry.
pub fn pack_payload(gate_up: &[u8], down: &[u8]) -> Result<Vec<u8>, Q4Error> {
    if gate_up.len() > u32::MAX as usize || down.len() > u32::MAX as usize {
        return Err(Q4Error::Payload("bank excede u32"));
    }
    let mut payload = Vec::with_capacity(PAYLOAD_HEADER_BYTES + gate_up.len() + down.len());
    payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
    payload.extend_from_slice(gate_up);
    payload.extend_from_slice(down);
    Ok(payload)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Q4LoadError {
    Io(String),
    InvalidGguf(&'static str),
    UnsupportedTensorType(u32),
    UnsupportedTensor {
        name: String,
        dtype: u32,
    },
    TensorMetadataMismatch {
        name: String,
        dtype: u32,
        expected_name: String,
        expected_dtype: u32,
    },
    TensorGeometryMismatch {
        name: String,
        dtype: u32,
        dims: Vec<i64>,
        expected_dims: Vec<u64>,
    },
    MissingTensor(String),
    InvalidTensor(&'static str),
    ExpertOutOfRange {
        expert: u32,
        count: u32,
    },
}

impl fmt::Display for Q4LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "falha ao ler GGUF: {error}"),
            Self::InvalidGguf(reason) => write!(formatter, "GGUF inválido: {reason}"),
            Self::UnsupportedTensorType(kind) => {
                write!(
                    formatter,
                    "tensor MoE usa dtype GGML não suportado: {kind} ({})",
                    ggml_type_name(*kind)
                )
            }
            Self::UnsupportedTensor { name, dtype } => write!(
                formatter,
                "tensor de expert não suportado: {name} usa dtype GGML {dtype} ({})",
                ggml_type_name(*dtype)
            ),
            Self::TensorMetadataMismatch {
                name,
                dtype,
                expected_name,
                expected_dtype,
            } => write!(
                formatter,
                "metadata do tensor de expert {name} (dtype GGML {dtype}, {}) não corresponde ao inventário {expected_name} (dtype GGML {expected_dtype}, {})",
                ggml_type_name(*dtype),
                ggml_type_name(*expected_dtype)
            ),
            Self::TensorGeometryMismatch {
                name,
                dtype,
                dims,
                expected_dims,
            } => write!(
                formatter,
                "geometria do tensor de expert {name} (dtype GGML {dtype}, {}) é {:?}, mas o inventário exige {:?}",
                ggml_type_name(*dtype),
                dims,
                expected_dims
            ),
            Self::MissingTensor(name) => write!(formatter, "tensor de expert ausente: {name}"),
            Self::InvalidTensor(reason) => write!(formatter, "tensor de expert inválido: {reason}"),
            Self::ExpertOutOfRange { expert, count } => {
                write!(formatter, "expert {expert} fora de [0, {count})")
            }
        }
    }
}

impl std::error::Error for Q4LoadError {}

#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

enum GateUpTensors {
    Fused(Vec<TensorSpec>),
    Split {
        gate: Vec<TensorSpec>,
        up: Vec<TensorSpec>,
    },
}

pub fn detect_expert_format(path: impl AsRef<Path>) -> Result<ExpertFormat, Q4LoadError> {
    let file = File::open(path.as_ref()).map_err(|error| Q4LoadError::Io(error.to_string()))?;
    // SAFETY: the file descriptor stays alive until the mmap is created and
    // the mapping is used only during this inspection.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
    let (_, tensors) = parse_tensor_table(&mmap)?;
    let expert_dtypes: BTreeSet<u32> = tensors
        .iter()
        .filter(|tensor| {
            tensor.name.starts_with("blk.")
                && tensor.name.ends_with(".weight")
                && [
                    "ffn_gate_up_exps.weight",
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ]
                .iter()
                .any(|suffix| tensor.name.ends_with(suffix))
        })
        .map(|tensor| tensor.dtype)
        .collect();
    let Some(&dtype) = expert_dtypes.iter().next() else {
        return Err(Q4LoadError::MissingTensor(
            "blk.<layer>.ffn_*_exps.weight".into(),
        ));
    };
    if expert_dtypes.len() > 1 {
        return Ok(ExpertFormat::Heterogeneous);
    }
    match dtype {
        2 => Ok(ExpertFormat::Q4_0),
        0 => Ok(ExpertFormat::F32),
        1 => Ok(ExpertFormat::F16),
        3 => Ok(ExpertFormat::Q4_1),
        6 => Ok(ExpertFormat::Q5_0),
        7 => Ok(ExpertFormat::Q5_1),
        8 => Ok(ExpertFormat::Q8_0),
        30 => Ok(ExpertFormat::Bf16),
        12 => Ok(ExpertFormat::Q4K),
        10 => Ok(ExpertFormat::Q2K),
        11 => Ok(ExpertFormat::Q3K),
        13 => Ok(ExpertFormat::Q5K),
        14 => Ok(ExpertFormat::Q6K),
        15 => Ok(ExpertFormat::Q8K),
        21 => Ok(ExpertFormat::Iq3S),
        22 => Ok(ExpertFormat::Iq2S),
        23 => Ok(ExpertFormat::Iq4XS),
        16 => Ok(ExpertFormat::Iq2XXS),
        17 => Ok(ExpertFormat::Iq2XS),
        18 => Ok(ExpertFormat::Iq3XXS),
        19 => Ok(ExpertFormat::Iq1S),
        20 => Ok(ExpertFormat::Iq4NL),
        29 => Ok(ExpertFormat::Iq1M),
        40 => Ok(ExpertFormat::Nvfp4),
        34 => Ok(ExpertFormat::Tq1_0),
        35 => Ok(ExpertFormat::Tq2_0),
        39 => Ok(ExpertFormat::Mxfp4),
        41 => Ok(ExpertFormat::Q1_0),
        42 => Ok(ExpertFormat::Q2_0),
        other => Err(Q4LoadError::UnsupportedTensorType(other)),
    }
}

/// Returns the distinct GGML dtypes used by routed expert weight tensors.
/// Heterogeneous GGUF files are reported as data instead of being collapsed
/// to a model-specific branch.
pub fn detect_expert_dtypes(path: impl AsRef<Path>) -> Result<Vec<u32>, Q4LoadError> {
    let file = File::open(path.as_ref()).map_err(|error| Q4LoadError::Io(error.to_string()))?;
    // SAFETY: the file descriptor stays alive until the mmap is created and
    // the mapping is used only during this inspection.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
    let (_, tensors) = parse_tensor_table(&mmap)?;
    let mut dtypes = tensors
        .iter()
        .filter(|tensor| {
            tensor.name.starts_with("blk.")
                && tensor.name.ends_with(".weight")
                && [
                    "ffn_gate_up_exps.weight",
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ]
                .iter()
                .any(|suffix| tensor.name.ends_with(suffix))
        })
        .map(|tensor| tensor.dtype)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if dtypes.is_empty() {
        return Err(Q4LoadError::MissingTensor(
            "blk.<layer>.ffn_*_exps.weight".into(),
        ));
    }
    dtypes.shrink_to_fit();
    Ok(dtypes)
}

/// Returns the number of local layers represented by expert tensors.
pub fn detect_expert_layer_count(path: impl AsRef<Path>) -> Result<usize, Q4LoadError> {
    let file = File::open(path.as_ref()).map_err(|error| Q4LoadError::Io(error.to_string()))?;
    // SAFETY: the file descriptor stays alive until the mmap is created and
    // the mapping is used only during this inspection.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
    let (_, tensors) = parse_tensor_table(&mmap)?;
    let last = tensors
        .iter()
        .filter(|tensor| {
            tensor.name.starts_with("blk.")
                && tensor.name.ends_with(".weight")
                && [
                    "ffn_gate_up_exps.weight",
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ]
                .iter()
                .any(|suffix| tensor.name.ends_with(suffix))
        })
        .filter_map(|tensor| {
            tensor
                .name
                .strip_prefix("blk.")?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .max()
        .ok_or_else(|| Q4LoadError::MissingTensor("blk.<layer>.ffn_*_exps.weight".into()))?;
    last.checked_add(1).ok_or(Q4LoadError::InvalidTensor(
        "overflow no número de camadas MoE",
    ))
}

/// Fonte mmap de experts Q4_0 em GGUF.
///
/// O arquivo permanece mapeado e cada miss copia apenas os dois bancos do
/// expert solicitado para o payload admitido pelo LRU. O parser aceita tanto
/// a convenção fundida `ffn_gate_up_exps.weight` quanto a separada
/// `ffn_gate_exps.weight` + `ffn_up_exps.weight`, além de `ffn_down_exps.weight`.
pub struct Q4_0Source {
    mmap: Mmap,
    data_start: usize,
    experts: u32,
    hidden: usize,
    intermediate: usize,
    gate_up: GateUpTensors,
    down: Vec<TensorSpec>,
    router: Vec<TensorSpec>,
}

impl Q4_0Source {
    pub fn open(
        path: impl AsRef<Path>,
        experts: u32,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, Q4LoadError> {
        Q4_0Expert::packed_bytes(hidden, intermediate)
            .map_err(|_| Q4LoadError::InvalidTensor("geometria não fecha Q4_0"))?;
        if experts == 0 {
            return Err(Q4LoadError::InvalidTensor("contagem de experts zero"));
        }
        let (mmap, data_start, tensors) = map_tensor_source(path.as_ref())?;
        let mut fused = Vec::new();
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        let mut router = Vec::new();
        for tensor in tensors {
            if tensor.name.starts_with("blk.") && tensor.name.ends_with("ffn_gate_up_exps.weight") {
                fused.push(tensor);
            } else if tensor.name.starts_with("blk.")
                && tensor.name.ends_with("ffn_gate_exps.weight")
            {
                gate.push(tensor);
            } else if tensor.name.starts_with("blk.") && tensor.name.ends_with("ffn_up_exps.weight")
            {
                up.push(tensor);
            } else if tensor.name.starts_with("blk.")
                && tensor.name.ends_with("ffn_down_exps.weight")
            {
                down.push(tensor);
            } else if tensor.name.starts_with("blk.")
                && tensor.name.ends_with("ffn_gate_inp.weight")
            {
                router.push(tensor);
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
                "blk.<layer>.ffn_gate_up_exps.weight ou ffn_gate_exps.weight + ffn_up_exps.weight"
                    .into(),
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
            gate_up,
            down,
            router,
        })
    }

    /// Infere a geometria Q4_0 a partir do primeiro banco de experts.
    ///
    /// GGUF guarda as dimensões lógicas `[hidden, 2*intermediate, experts]`,
    /// embora os bytes de cada linha sejam Q4_0 packed. O tamanho é validado
    /// novamente por `open`, portanto uma tabela truncada não vira geometria
    /// inventada.
    pub fn open_infer(path: impl AsRef<Path>, experts: u32) -> Result<Self, Q4LoadError> {
        let file = File::open(path.as_ref()).map_err(|error| Q4LoadError::Io(error.to_string()))?;
        // SAFETY: the file descriptor stays alive until the mmap is created.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
        let (_, tensors) = parse_tensor_table(&mmap)?;
        let fused = tensors
            .iter()
            .find(|tensor| tensor.name.ends_with("ffn_gate_up_exps.weight"));
        let (hidden, intermediate, tensor_experts) = if let Some(tensor) = fused {
            if tensor.dtype != 2 || tensor.dims.len() != 3 {
                return Err(Q4LoadError::InvalidTensor(
                    "gate_up precisa ser um tensor Q4_0 de rank 3",
                ));
            }
            let hidden = usize::try_from(tensor.dims[0])
                .map_err(|_| Q4LoadError::InvalidTensor("hidden excede usize"))?;
            let two_intermediate = usize::try_from(tensor.dims[1])
                .map_err(|_| Q4LoadError::InvalidTensor("intermediate excede usize"))?;
            let tensor_experts = u32::try_from(tensor.dims[2])
                .map_err(|_| Q4LoadError::InvalidTensor("experts excede u32"))?;
            if two_intermediate == 0 || !two_intermediate.is_multiple_of(2) {
                return Err(Q4LoadError::InvalidTensor(
                    "geometria gate_up não corresponde aos metadados MoE",
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
            if gate.dtype != 2 || up.dtype != 2 || gate.dims.len() != 3 || up.dims.len() != 3 {
                return Err(Q4LoadError::InvalidTensor(
                    "gate e up precisam ser tensores Q4_0 de rank 3",
                ));
            }
            if gate.dims != up.dims {
                return Err(Q4LoadError::InvalidTensor(
                    "gate e up têm geometrias diferentes",
                ));
            }
            let intermediate = usize::try_from(gate.dims[0])
                .map_err(|_| Q4LoadError::InvalidTensor("intermediate excede usize"))?;
            let hidden = usize::try_from(gate.dims[1])
                .map_err(|_| Q4LoadError::InvalidTensor("hidden excede usize"))?;
            let tensor_experts = u32::try_from(gate.dims[2])
                .map_err(|_| Q4LoadError::InvalidTensor("experts excede u32"))?;
            (hidden, intermediate, tensor_experts)
        };
        if tensor_experts != experts {
            return Err(Q4LoadError::InvalidTensor(
                "quantidade de experts não corresponde aos metadados MoE",
            ));
        }
        Self::open(path, experts, hidden, intermediate)
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
        validate_tensor(down, self.experts, self.hidden, self.intermediate)?;
        let down_row =
            row_bytes(self.intermediate).map_err(|_| Q4LoadError::InvalidTensor("down"))?;
        let down_per_expert = self
            .hidden
            .checked_mul(down_row)
            .ok_or(Q4LoadError::InvalidTensor("overflow em down"))?;
        let down = self.expert_slice(down, expert, down_per_expert)?;
        let gate_up = match &self.gate_up {
            GateUpTensors::Fused(tensors) => {
                let gate = self.tensor_for_layer(tensors, layer, "ffn_gate_up_exps.weight")?;
                validate_tensor(gate, self.experts, 2 * self.intermediate, self.hidden)?;
                let gate_row =
                    row_bytes(self.hidden).map_err(|_| Q4LoadError::InvalidTensor("gate_up"))?;
                let gate_per_expert = 2usize
                    .checked_mul(self.intermediate)
                    .and_then(|rows| rows.checked_mul(gate_row))
                    .ok_or(Q4LoadError::InvalidTensor("overflow em gate_up"))?;
                self.expert_slice(gate, expert, gate_per_expert)?.to_vec()
            }
            GateUpTensors::Split {
                gate: gates,
                up: ups,
            } => {
                let gate = self.tensor_for_layer(gates, layer, "ffn_gate_exps.weight")?;
                let up = self.tensor_for_layer(ups, layer, "ffn_up_exps.weight")?;
                validate_tensor(gate, self.experts, self.intermediate, self.hidden)?;
                validate_tensor(up, self.experts, self.intermediate, self.hidden)?;
                let gate_row =
                    row_bytes(self.hidden).map_err(|_| Q4LoadError::InvalidTensor("gate"))?;
                let bank_per_expert = self
                    .intermediate
                    .checked_mul(gate_row)
                    .ok_or(Q4LoadError::InvalidTensor("overflow em gate/up"))?;
                let gate = self.expert_slice(gate, expert, bank_per_expert)?;
                let up = self.expert_slice(up, expert, bank_per_expert)?;
                let mut gate_up = Vec::with_capacity(gate.len() + up.len());
                gate_up.extend_from_slice(gate);
                gate_up.extend_from_slice(up);
                gate_up
            }
        };
        pack_payload(&gate_up, down).map_err(|_| Q4LoadError::InvalidTensor("payload Q4_0"))
    }

    /// Calcula os logits do router local para uma ativação.
    ///
    /// No GGUF Gemma4 o tensor lógico é `[hidden, experts]`, que corresponde
    /// a uma linha contiguous por expert na representação de bytes. O router
    /// é mantido pequeno e residente; só os experts passam pelo LRU.
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
        let element_bytes = match tensor.dtype {
            0 => 4,
            1 | 30 => 2,
            kind => return Err(Q4LoadError::UnsupportedTensorType(kind)),
        };
        let total_bytes = usize::try_from(elements)
            .ok()
            .and_then(|count| count.checked_mul(element_bytes))
            .ok_or(Q4LoadError::InvalidTensor("tamanho do router excede usize"))?;
        let raw = self.tensor_bytes(tensor, total_bytes)?;
        let mut logits = Vec::with_capacity(self.experts as usize);
        for expert in 0..self.experts as usize {
            let row = &raw
                [expert * self.hidden * element_bytes..(expert + 1) * self.hidden * element_bytes];
            let value = row
                .chunks_exact(element_bytes)
                .zip(hidden)
                .map(|(encoded, &input)| decode_scalar(encoded, tensor.dtype) * input)
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

    fn expert_slice<'a>(
        &'a self,
        tensor: &TensorSpec,
        expert: u32,
        bytes_per_expert: usize,
    ) -> Result<&'a [u8], Q4LoadError> {
        if tensor.dtype != 2 {
            return Err(Q4LoadError::UnsupportedTensorType(tensor.dtype));
        }
        let total = bytes_per_expert
            .checked_mul(self.experts as usize)
            .ok_or(Q4LoadError::InvalidTensor("overflow no tensor"))?;
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
}

fn decode_scalar(encoded: &[u8], dtype: u32) -> f32 {
    match dtype {
        0 => f32::from_le_bytes(encoded.try_into().expect("F32 tem quatro bytes")),
        1 => f16_to_f32(u16::from_le_bytes(
            encoded.try_into().expect("F16 tem dois bytes"),
        )),
        30 => f32::from_bits(
            u32::from(u16::from_le_bytes(
                encoded.try_into().expect("BF16 tem dois bytes"),
            )) << 16,
        ),
        _ => 0.0,
    }
}

pub fn read_scalar_vector(
    mmap: &[u8],
    data_start: usize,
    tensor: &TensorSpec,
) -> Result<Vec<f32>, Q4LoadError> {
    let count = tensor
        .dims
        .iter()
        .try_fold(1u64, |total, dimension| total.checked_mul(*dimension))
        .ok_or(Q4LoadError::InvalidTensor(
            "overflow nas dimensões escalares",
        ))?;
    let count =
        usize::try_from(count).map_err(|_| Q4LoadError::InvalidTensor("escala excede usize"))?;
    let element_bytes = match tensor.dtype {
        0 => 4,
        1 | 30 => 2,
        dtype => return Err(Q4LoadError::UnsupportedTensorType(dtype)),
    };
    let bytes = count
        .checked_mul(element_bytes)
        .ok_or(Q4LoadError::InvalidTensor("bytes da escala excedem usize"))?;
    let offset = usize::try_from(tensor.offset)
        .ok()
        .and_then(|value| data_start.checked_add(value))
        .ok_or(Q4LoadError::InvalidTensor("offset GGUF inválido"))?;
    let end = offset
        .checked_add(bytes)
        .ok_or(Q4LoadError::InvalidTensor("escala excede usize"))?;
    let raw = mmap
        .get(offset..end)
        .ok_or(Q4LoadError::InvalidTensor("escala excede o arquivo"))?;
    let values = raw
        .chunks_exact(element_bytes)
        .map(|encoded| decode_scalar(encoded, tensor.dtype))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Q4LoadError::InvalidTensor("escala não finita"));
    }
    Ok(values)
}

fn validate_tensor(
    tensor: &TensorSpec,
    experts: u32,
    rows: usize,
    columns: usize,
) -> Result<(), Q4LoadError> {
    if tensor.dtype != 2 {
        return Err(Q4LoadError::UnsupportedTensorType(tensor.dtype));
    }
    let elements = tensor
        .dims
        .iter()
        .try_fold(1u64, |total, dimension| total.checked_mul(*dimension))
        .ok_or(Q4LoadError::InvalidTensor("overflow nas dimensões"))?;
    let expected = (experts as u64)
        .checked_mul(rows as u64)
        .and_then(|value| value.checked_mul(columns as u64))
        .ok_or(Q4LoadError::InvalidTensor("overflow na geometria"))?;
    if elements != expected {
        return Err(Q4LoadError::InvalidTensor(
            "dimensões não correspondem ao modelo",
        ));
    }
    Ok(())
}

/// Maps the bytes behind a GGUF's tensors: the file itself, or, for a
/// header-only virtual slice (`general.external_data.*`), the one source file
/// its tensors point into. Returns the mapping, the data start and the tensor
/// table, with offsets relative to that data start, so every reader keeps
/// indexing `data_start + offset`. A tensor stored in the header itself (the
/// identity norm of a non-final range, never an expert) gets an offset no read
/// can reach.
pub fn map_tensor_source(path: &Path) -> Result<(Mmap, usize, Vec<TensorSpec>), Q4LoadError> {
    let file = File::open(path).map_err(|error| Q4LoadError::Io(error.to_string()))?;
    // SAFETY: Mmap owns the mapping and outlives every returned tensor slice.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
    let (data_start, mut tensors) = parse_tensor_table(&mmap)?;
    let Some(external) = read_external_data(&mmap)? else {
        return Ok((mmap, data_start, tensors));
    };
    if external.files.len() != 1 {
        return Err(Q4LoadError::InvalidGguf(
            "external tensor data in more than one file: the MoE executor reads one file",
        ));
    }
    if external.tensor_files.len() != tensors.len()
        || external.tensor_offsets.len() != tensors.len()
    {
        return Err(Q4LoadError::InvalidGguf(
            "external tensor data does not cover the tensor table",
        ));
    }
    let source = Path::new(&external.files[0]);
    let source = if source.is_absolute() {
        source.to_path_buf()
    } else {
        path.parent().unwrap_or(Path::new(".")).join(source)
    };
    for ((tensor, &file), &offset) in tensors
        .iter_mut()
        .zip(&external.tensor_files)
        .zip(&external.tensor_offsets)
    {
        tensor.offset = if file == 0 { offset } else { u64::MAX };
    }
    let file = File::open(&source)
        .map_err(|error| Q4LoadError::Io(format!("{}: {error}", source.display())))?;
    // SAFETY: as above, for the source file the header points into.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
    Ok((mmap, 0, tensors))
}

struct ExternalData {
    files: Vec<String>,
    tensor_files: Vec<i32>,
    tensor_offsets: Vec<u64>,
}

/// The `general.external_data.*` keys of a GGUF header, when it has them.
fn read_external_data(raw: &[u8]) -> Result<Option<ExternalData>, Q4LoadError> {
    let mut cursor = 8usize;
    let _tensor_count = read_u64(raw, &mut cursor)?;
    let kv_count = read_u64(raw, &mut cursor)?;
    let (mut files, mut tensor_files, mut tensor_offsets) = (None, None, None);
    for _ in 0..kv_count {
        let key = read_string(raw, &mut cursor)?;
        let kind = read_u32(raw, &mut cursor)?;
        let wanted = matches!(
            key.as_str(),
            "general.external_data.files"
                | "general.external_data.tensor_files"
                | "general.external_data.tensor_offsets"
        );
        if !wanted || kind != 9 {
            skip_value(raw, &mut cursor, kind)?;
            continue;
        }
        let element_kind = read_u32(raw, &mut cursor)?;
        let count = usize::try_from(read_u64(raw, &mut cursor)?)
            .map_err(|_| Q4LoadError::InvalidGguf("array too long"))?;
        if count > raw.len() {
            return Err(Q4LoadError::InvalidGguf("array exceeds the file"));
        }
        match (key.as_str(), element_kind) {
            ("general.external_data.files", 8) => {
                files = Some(
                    (0..count)
                        .map(|_| read_string(raw, &mut cursor))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            ("general.external_data.tensor_files", 5) => {
                tensor_files = Some(
                    (0..count)
                        .map(|_| read_u32(raw, &mut cursor).map(|value| value as i32))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            ("general.external_data.tensor_offsets", 10) => {
                tensor_offsets = Some(
                    (0..count)
                        .map(|_| read_u64(raw, &mut cursor))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            _ => {
                return Err(Q4LoadError::InvalidGguf(
                    "external tensor data has the wrong type",
                ))
            }
        }
    }
    match (files, tensor_files, tensor_offsets) {
        (None, None, None) => Ok(None),
        (Some(files), Some(tensor_files), Some(tensor_offsets)) => Ok(Some(ExternalData {
            files,
            tensor_files,
            tensor_offsets,
        })),
        _ => Err(Q4LoadError::InvalidGguf(
            "external tensor data is incomplete",
        )),
    }
}

pub fn parse_tensor_table(raw: &[u8]) -> Result<(usize, Vec<TensorSpec>), Q4LoadError> {
    let mut cursor = 0usize;
    let magic = read_u32(raw, &mut cursor)?;
    if magic != 0x4655_4747 {
        return Err(Q4LoadError::InvalidGguf("magic"));
    }
    let _version = read_u32(raw, &mut cursor)?;
    let tensor_count = read_u64(raw, &mut cursor)?;
    let kv_count = read_u64(raw, &mut cursor)?;
    if tensor_count > 1_000_000 || kv_count > 100_000 {
        return Err(Q4LoadError::InvalidGguf("contagem de entradas"));
    }
    let mut alignment = 32u64;
    for _ in 0..kv_count {
        let key = read_string(raw, &mut cursor)?;
        let kind = read_u32(raw, &mut cursor)?;
        if key == "general.alignment" && kind == 4 {
            alignment = read_u32(raw, &mut cursor)? as u64;
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(Q4LoadError::InvalidGguf("alinhamento"));
            }
        } else {
            skip_value(raw, &mut cursor, kind)?;
        }
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(raw, &mut cursor)?;
        let rank = read_u32(raw, &mut cursor)?;
        if rank == 0 || rank > 8 {
            return Err(Q4LoadError::InvalidGguf("rank do tensor"));
        }
        let mut dims = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            dims.push(read_u64(raw, &mut cursor)?);
        }
        let dtype = read_u32(raw, &mut cursor)?;
        let offset = read_u64(raw, &mut cursor)?;
        tensors.push(TensorSpec {
            name,
            dims,
            dtype,
            offset,
        });
    }
    let align = usize::try_from(alignment).map_err(|_| Q4LoadError::InvalidGguf("alinhamento"))?;
    let data_start = cursor
        .checked_add(align - 1)
        .map(|value| value / align * align)
        .ok_or(Q4LoadError::InvalidGguf("offset da seção de dados"))?;
    if data_start > raw.len() {
        return Err(Q4LoadError::InvalidGguf("seção de dados ausente"));
    }
    Ok((data_start, tensors))
}

fn read_u32(raw: &[u8], cursor: &mut usize) -> Result<u32, Q4LoadError> {
    let end = cursor
        .checked_add(4)
        .ok_or(Q4LoadError::InvalidGguf("cursor"))?;
    let bytes = raw
        .get(*cursor..end)
        .ok_or(Q4LoadError::InvalidGguf("fim do arquivo"))?;
    *cursor = end;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64(raw: &[u8], cursor: &mut usize) -> Result<u64, Q4LoadError> {
    let end = cursor
        .checked_add(8)
        .ok_or(Q4LoadError::InvalidGguf("cursor"))?;
    let bytes = raw
        .get(*cursor..end)
        .ok_or(Q4LoadError::InvalidGguf("fim do arquivo"))?;
    *cursor = end;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_string(raw: &[u8], cursor: &mut usize) -> Result<String, Q4LoadError> {
    let length = usize::try_from(read_u64(raw, cursor)?)
        .map_err(|_| Q4LoadError::InvalidGguf("string longa"))?;
    let end = cursor
        .checked_add(length)
        .ok_or(Q4LoadError::InvalidGguf("string"))?;
    let bytes = raw
        .get(*cursor..end)
        .ok_or(Q4LoadError::InvalidGguf("string truncada"))?;
    *cursor = end;
    String::from_utf8(bytes.to_vec()).map_err(|_| Q4LoadError::InvalidGguf("string UTF-8"))
}

fn skip_value(raw: &[u8], cursor: &mut usize, kind: u32) -> Result<(), Q4LoadError> {
    let scalar = match kind {
        0 | 1 | 7 => Some(1usize),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    };
    if let Some(bytes) = scalar {
        *cursor = cursor
            .checked_add(bytes)
            .ok_or(Q4LoadError::InvalidGguf("valor"))?;
        if *cursor > raw.len() {
            return Err(Q4LoadError::InvalidGguf("valor truncado"));
        }
        return Ok(());
    }
    match kind {
        8 => {
            let length = usize::try_from(read_u64(raw, cursor)?)
                .map_err(|_| Q4LoadError::InvalidGguf("string longa"))?;
            *cursor = cursor
                .checked_add(length)
                .ok_or(Q4LoadError::InvalidGguf("string"))?;
            if *cursor > raw.len() {
                return Err(Q4LoadError::InvalidGguf("string truncada"));
            }
            Ok(())
        }
        9 => {
            let element_kind = read_u32(raw, cursor)?;
            let count = read_u64(raw, cursor)?;
            let available = u64::try_from(raw.len().saturating_sub(*cursor))
                .map_err(|_| Q4LoadError::InvalidGguf("array excede o arquivo"))?;
            if count > available {
                return Err(Q4LoadError::InvalidGguf("array excede o arquivo"));
            }
            for _ in 0..count {
                skip_value(raw, cursor, element_kind)?;
            }
            Ok(())
        }
        _ => Err(Q4LoadError::InvalidGguf("tipo KV")),
    }
}

/// Executor de um expert Q4_0 com cache compartilhado entre chamadas.
pub struct Q4_0Executor {
    cache: Arc<ExpertCache>,
    hidden: usize,
    intermediate: usize,
}

impl Q4_0Executor {
    pub fn new(
        cache: Arc<ExpertCache>,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, Q4Error> {
        Q4_0Expert::packed_bytes(hidden, intermediate)?;
        Ok(Self {
            cache,
            hidden,
            intermediate,
        })
    }

    pub fn run_from_source(
        &self,
        source: &Q4_0Source,
        key: ExpertKey,
        layer: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>, Q4ExecuteError> {
        let expert = key.expert;
        self.run_with_loader(
            key,
            || {
                source
                    .load_expert(layer, expert)
                    .map_err(Q4ExecuteError::Source)
            },
            hidden,
        )
    }

    fn run_with_loader<F, E>(
        &self,
        key: ExpertKey,
        load: F,
        hidden: &[f32],
    ) -> Result<Vec<f32>, Q4ExecuteError>
    where
        F: FnOnce() -> Result<Vec<u8>, E>,
        E: Into<Q4ExecuteError>,
    {
        let lease = match self.cache.lookup(&key) {
            Some(lease) => lease,
            None => self
                .cache
                .admit(key, load().map_err(Into::into)?)
                .map_err(Q4ExecuteError::Cache)?,
        };
        let expert = Q4_0Expert::from_payload(self.hidden, self.intermediate, lease.bytes())?;
        expert.run(hidden)
    }
}

pub fn row_bytes(columns: usize) -> Result<usize, Q4Error> {
    if !columns.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(Q4Error::Geometry("linha não fecha bloco de 32"));
    }
    columns
        .checked_div(BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(BLOCK_BYTES))
        .ok_or(Q4Error::Geometry("overflow na linha"))
}

pub fn decode_row(row: &[u8], columns: usize) -> Result<Vec<f32>, Q4Error> {
    let expected = row_bytes(columns)?;
    if row.len() != expected {
        return Err(Q4Error::PayloadLength {
            expected,
            actual: row.len(),
        });
    }
    Ok(row
        .chunks_exact(BLOCK_BYTES)
        .flat_map(|packed| {
            let scale = f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
            (0..BLOCK_ELEMENTS).map(move |offset| {
                let q = if offset < 16 {
                    packed[2 + offset] & 0x0f
                } else {
                    packed[2 + offset - 16] >> 4
                };
                scale * (q as f32 - 8.0)
            })
        })
        .collect())
}

fn mat_vec(matrix: &[u8], rows: usize, columns: usize, input: &[f32]) -> Vec<f32> {
    let row_size = columns / BLOCK_ELEMENTS * BLOCK_BYTES;
    (0..rows)
        .map(|row| {
            let bytes = &matrix[row * row_size..(row + 1) * row_size];
            bytes
                .chunks_exact(BLOCK_BYTES)
                .enumerate()
                .map(|(block, packed)| {
                    let scale = f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
                    (0..BLOCK_ELEMENTS)
                        .map(|offset| {
                            let q = if offset < 16 {
                                packed[2 + offset] & 0x0f
                            } else {
                                packed[2 + offset - 16] >> 4
                            };
                            scale * (q as f32 - 8.0) * input[block * BLOCK_ELEMENTS + offset]
                        })
                        .sum::<f32>()
                })
                .sum()
        })
        .collect()
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
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

#[cfg(test)]
pub mod tests {
    use super::super::admission::MemoryAccount;
    use super::super::cache::{ExpertCache, FreeMemoryProbe};
    use super::*;
    use std::fs::File;
    use std::io::{self, Write};
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct FixedProbe(AtomicU64);

    impl FixedProbe {
        fn new(bytes: u64) -> Arc<Self> {
            Arc::new(Self(AtomicU64::new(bytes)))
        }
    }

    impl FreeMemoryProbe for FixedProbe {
        fn free_bytes(&self) -> io::Result<u64> {
            Ok(self.0.load(Ordering::Relaxed))
        }
    }

    fn packed_rows(rows: usize, columns: usize, nibble: u8) -> Vec<u8> {
        let row_size = columns / BLOCK_ELEMENTS * BLOCK_BYTES;
        let mut out = vec![0u8; rows * row_size];
        for row in out.chunks_exact_mut(row_size) {
            for block in row.chunks_exact_mut(BLOCK_BYTES) {
                block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                for byte in &mut block[2..] {
                    *byte = nibble | (nibble << 4);
                }
            }
        }
        out
    }

    fn gguf_string(output: &mut Vec<u8>, value: &str) {
        output.extend_from_slice(&(value.len() as u64).to_le_bytes());
        output.extend_from_slice(value.as_bytes());
    }

    fn gguf_tensor(output: &mut Vec<u8>, name: &str, dims: &[u64], offset: u64) {
        gguf_tensor_dtype(output, name, dims, offset, 2);
    }

    fn gguf_tensor_dtype(output: &mut Vec<u8>, name: &str, dims: &[u64], offset: u64, dtype: u32) {
        gguf_string(output, name);
        output.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dimension in dims {
            output.extend_from_slice(&dimension.to_le_bytes());
        }
        output.extend_from_slice(&dtype.to_le_bytes());
        output.extend_from_slice(&offset.to_le_bytes());
    }

    pub fn write_q4_source(path: &Path) -> (usize, usize) {
        let hidden = 32;
        let intermediate = 32;
        let experts = 2;
        let gate_up = packed_rows(experts * 2 * intermediate, hidden, 9);
        let down = packed_rows(experts * hidden, intermediate, 9);
        let mut router = Vec::with_capacity(experts * hidden * 4);
        for expert in 0..experts {
            for _ in 0..hidden {
                router.extend_from_slice(&((expert + 1) as f32).to_le_bytes());
            }
        }
        let mut file = Vec::new();
        file.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        file.extend_from_slice(&2u32.to_le_bytes());
        file.extend_from_slice(&3u64.to_le_bytes());
        file.extend_from_slice(&1u64.to_le_bytes());
        gguf_string(&mut file, "general.alignment");
        file.extend_from_slice(&4u32.to_le_bytes());
        file.extend_from_slice(&32u32.to_le_bytes());
        gguf_tensor(
            &mut file,
            "blk.0.ffn_gate_up_exps.weight",
            &[hidden as u64, 2 * intermediate as u64, experts as u64],
            0,
        );
        gguf_tensor(
            &mut file,
            "blk.0.ffn_down_exps.weight",
            &[intermediate as u64, hidden as u64, experts as u64],
            gate_up.len() as u64,
        );
        gguf_tensor_dtype(
            &mut file,
            "blk.0.ffn_gate_inp.weight",
            &[hidden as u64, experts as u64],
            (gate_up.len() + down.len()) as u64,
            0,
        );
        let data_start = (file.len() + 31) / 32 * 32;
        file.resize(data_start, 0);
        file.extend_from_slice(&gate_up);
        file.extend_from_slice(&down);
        file.extend_from_slice(&router);
        File::create(path).unwrap().write_all(&file).unwrap();
        (hidden, intermediate)
    }

    fn write_expert_dtype_source(path: &Path, dtype: u32) {
        let mut file = Vec::new();
        file.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        file.extend_from_slice(&2u32.to_le_bytes());
        file.extend_from_slice(&1u64.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        gguf_tensor_dtype(
            &mut file,
            "blk.0.ffn_gate_up_exps.weight",
            &[32, 64, 1],
            0,
            dtype,
        );
        let data_start = (file.len() + 31) / 32 * 32;
        file.resize(data_start, 0);
        File::create(path).unwrap().write_all(&file).unwrap();
    }

    fn write_q4_split_source(path: &Path) -> (usize, usize) {
        let hidden = 32;
        let intermediate = 64;
        let experts = 2;
        let gate = packed_rows(experts * intermediate, hidden, 9);
        let up = packed_rows(experts * intermediate, hidden, 10);
        let down = packed_rows(experts * hidden, intermediate, 11);
        let mut router = Vec::with_capacity(experts * hidden * 4);
        for expert in 0..experts {
            for _ in 0..hidden {
                router.extend_from_slice(&((expert + 1) as f32).to_le_bytes());
            }
        }
        let mut file = Vec::new();
        file.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        file.extend_from_slice(&2u32.to_le_bytes());
        file.extend_from_slice(&4u64.to_le_bytes());
        file.extend_from_slice(&1u64.to_le_bytes());
        gguf_string(&mut file, "general.alignment");
        file.extend_from_slice(&4u32.to_le_bytes());
        file.extend_from_slice(&32u32.to_le_bytes());
        gguf_tensor(
            &mut file,
            "blk.0.ffn_gate_exps.weight",
            &[intermediate as u64, hidden as u64, experts as u64],
            0,
        );
        gguf_tensor(
            &mut file,
            "blk.0.ffn_up_exps.weight",
            &[intermediate as u64, hidden as u64, experts as u64],
            gate.len() as u64,
        );
        gguf_tensor(
            &mut file,
            "blk.0.ffn_down_exps.weight",
            &[hidden as u64, intermediate as u64, experts as u64],
            (gate.len() + up.len()) as u64,
        );
        gguf_tensor_dtype(
            &mut file,
            "blk.0.ffn_gate_inp.weight",
            &[hidden as u64, experts as u64],
            (gate.len() + up.len() + down.len()) as u64,
            0,
        );
        let data_start = (file.len() + 31) / 32 * 32;
        file.resize(data_start, 0);
        file.extend_from_slice(&gate);
        file.extend_from_slice(&up);
        file.extend_from_slice(&down);
        file.extend_from_slice(&router);
        File::create(path).unwrap().write_all(&file).unwrap();
        (hidden, intermediate)
    }

    #[test]
    fn parser_accepts_a_gguf_array_larger_than_an_arbitrary_limit() {
        let count = 100_001u64;
        let mut raw = Vec::new();
        raw.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        raw.extend_from_slice(&2u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&1u64.to_le_bytes());
        gguf_string(&mut raw, "tokenizer.ggml.tokens");
        raw.extend_from_slice(&9u32.to_le_bytes());
        raw.extend_from_slice(&8u32.to_le_bytes());
        raw.extend_from_slice(&count.to_le_bytes());
        for _ in 0..count {
            gguf_string(&mut raw, "");
        }
        let aligned = (raw.len() + 31) / 32 * 32;
        raw.resize(aligned, 0);

        let parsed = parse_tensor_table(&raw);
        assert!(parsed.is_ok(), "{parsed:?}");
    }

    #[test]
    fn an_unknown_dtype_reports_id_and_name() {
        let error = Q4LoadError::UnsupportedTensorType(999).to_string();
        assert!(error.contains("999"));
        assert!(error.contains("desconhecido"));
    }

    #[test]
    fn an_active_dtype_without_a_decoder_reports_id_and_name() {
        let error = Q4LoadError::UnsupportedTensorType(9).to_string();
        assert!(error.contains("9"));
        assert!(error.contains("Q8_1"));
    }

    #[test]
    fn detects_q4_k_and_q6_k_as_supportable_expert_formats() {
        for dtype in [12u32, 14u32] {
            let path = std::env::temp_dir().join(format!(
                "pom-ftk-detect-k-{dtype}-{}.gguf",
                std::process::id()
            ));
            write_expert_dtype_source(&path, dtype);
            assert!(
                detect_expert_format(&path).is_ok(),
                "dtype {dtype} deveria entrar no dispatch K-quant"
            );
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn q4_0_runs_gate_up_silu_and_down() {
        let hidden = 32;
        let intermediate = 32;
        let gate_up = packed_rows(2 * intermediate, hidden, 9);
        let down = packed_rows(hidden, intermediate, 9);
        let payload = pack_payload(&gate_up, &down).unwrap();
        let expert = Q4_0Expert::from_payload(hidden, intermediate, &payload).unwrap();
        let output = expert.run(&[1.0; 32]).unwrap();
        assert_eq!(output.len(), hidden);
        assert!(output
            .iter()
            .all(|value| value.is_finite() && *value > 30_000.0));
    }

    #[test]
    fn payload_rejects_a_geometry_that_does_not_close_a_block() {
        assert!(matches!(
            Q4_0Expert::packed_bytes(31, 32),
            Err(Q4Error::Geometry(_))
        ));
    }

    #[test]
    fn executor_reuses_the_admitted_expert_on_the_second_forward() {
        let hidden = 32;
        let intermediate = 32;
        let probe = FixedProbe::new(100_000);
        let cache = ExpertCache::new(
            MemoryAccount {
                baseline_free_bytes: 100_000,
                compute_bytes: 0,
                weights_bytes: 0,
                fixed_bytes: 0,
                reserve_bytes: 0,
            },
            probe,
        );
        let executor = Q4_0Executor::new(Arc::clone(&cache), hidden, intermediate).unwrap();
        let gate_up = packed_rows(2 * intermediate, hidden, 9);
        let down = packed_rows(hidden, intermediate, 9);
        let payload = pack_payload(&gate_up, &down).unwrap();
        let key = ExpertKey::new("fixture-moe", 0, 7, "q4_0");
        let first = executor
            .run_with_loader(
                key.clone(),
                || -> Result<Vec<u8>, CacheError> { Ok(payload.clone()) },
                &[1.0; 32],
            )
            .unwrap();
        let second = executor
            .run_with_loader(
                key,
                || -> Result<Vec<u8>, CacheError> { panic!("cache hit não deve chamar o loader") },
                &[1.0; 32],
            )
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn mmap_source_loads_only_the_requested_expert() {
        let path = std::env::temp_dir().join(format!(
            "pom-ftk-q4-source-{}-{}.gguf",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let (hidden, intermediate) = write_q4_source(&path);
        let source = Q4_0Source::open_infer(&path, 2).unwrap();
        let expert = source.load_expert(0, 1).unwrap();
        let expected = Q4_0Expert::packed_bytes(hidden, intermediate).unwrap();
        assert_eq!(expert.len(), expected);
        Q4_0Expert::from_payload(hidden, intermediate, &expert).unwrap();
        assert_eq!(
            source.router_logits(0, &[1.0; 32]).unwrap(),
            vec![32.0, 64.0]
        );
        assert!(matches!(
            source.load_expert(0, 2),
            Err(Q4LoadError::ExpertOutOfRange {
                expert: 2,
                count: 2
            })
        ));
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    /// A header-only copy of `source`: its KV section and tensor table, plus
    /// `general.external_data.*` pointing every tensor at its bytes in `source`.
    fn write_external_data_header(source: &Path, header: &Path) {
        let raw = std::fs::read(source).unwrap();
        let (data_start, tensors) = parse_tensor_table(&raw).unwrap();
        let mut cursor = 8usize;
        let tensor_count = read_u64(&raw, &mut cursor).unwrap();
        let kv_count = read_u64(&raw, &mut cursor).unwrap();
        let kv_begin = cursor;
        for _ in 0..kv_count {
            read_string(&raw, &mut cursor).unwrap();
            let kind = read_u32(&raw, &mut cursor).unwrap();
            skip_value(&raw, &mut cursor, kind).unwrap();
        }
        let kv_end = cursor;
        let mut table_end = kv_end;
        for _ in 0..tensor_count {
            read_string(&raw, &mut table_end).unwrap();
            let rank = read_u32(&raw, &mut table_end).unwrap();
            table_end += 8 * rank as usize + 4 + 8;
        }
        let key = |name: &str| {
            let mut out = (name.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(name.as_bytes());
            out
        };
        let mut extra = key("general.external_data.files");
        extra.extend_from_slice(&9u32.to_le_bytes());
        extra.extend_from_slice(&8u32.to_le_bytes());
        extra.extend_from_slice(&1u64.to_le_bytes());
        extra.extend_from_slice(&key(source.to_str().unwrap()));
        extra.extend_from_slice(&key("general.external_data.tensor_files"));
        extra.extend_from_slice(&9u32.to_le_bytes());
        extra.extend_from_slice(&5u32.to_le_bytes());
        extra.extend_from_slice(&tensor_count.to_le_bytes());
        for _ in 0..tensor_count {
            extra.extend_from_slice(&0i32.to_le_bytes());
        }
        extra.extend_from_slice(&key("general.external_data.tensor_offsets"));
        extra.extend_from_slice(&9u32.to_le_bytes());
        extra.extend_from_slice(&10u32.to_le_bytes());
        extra.extend_from_slice(&tensor_count.to_le_bytes());
        for tensor in &tensors {
            extra.extend_from_slice(&(data_start as u64 + tensor.offset).to_le_bytes());
        }
        let mut out = raw[..8].to_vec();
        out.extend_from_slice(&tensor_count.to_le_bytes());
        out.extend_from_slice(&(kv_count + 3).to_le_bytes());
        out.extend_from_slice(&raw[kv_begin..kv_end]);
        out.extend_from_slice(&extra);
        out.extend_from_slice(&raw[kv_end..table_end]);
        out.resize(out.len().div_ceil(32) * 32, 0);
        std::fs::write(header, out).unwrap();
    }

    #[test]
    fn a_header_with_external_data_reads_the_experts_from_the_source_file() {
        let base = std::env::temp_dir().join(format!("pom-ftk-q4-external-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let path = base.join("source.gguf");
        let header = base.join("header.gguf");
        let (hidden, intermediate) = write_q4_source(&path);
        write_external_data_header(&path, &header);
        assert!(
            std::fs::metadata(&header).unwrap().len() < std::fs::metadata(&path).unwrap().len()
        );

        let direct = Q4_0Source::open_infer(&path, 2).unwrap();
        let virtual_source = Q4_0Source::open_infer(&header, 2).unwrap();
        for expert in 0..2 {
            assert_eq!(
                virtual_source.load_expert(0, expert).unwrap(),
                direct.load_expert(0, expert).unwrap()
            );
        }
        assert_eq!(
            virtual_source.router_logits(0, &[1.0; 32]).unwrap(),
            direct.router_logits(0, &[1.0; 32]).unwrap()
        );
        let expert = virtual_source.load_expert(0, 1).unwrap();
        Q4_0Expert::from_payload(hidden, intermediate, &expert).unwrap();
        drop((direct, virtual_source));
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn mmap_source_accepts_gate_and_up_in_separate_tensors() {
        let path = std::env::temp_dir().join(format!(
            "pom-ftk-q4-split-source-{}-{}.gguf",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let (hidden, intermediate) = write_q4_split_source(&path);
        let source = Q4_0Source::open_infer(&path, 2).unwrap();
        let expert = source.load_expert(0, 1).unwrap();
        let expected = Q4_0Expert::packed_bytes(hidden, intermediate).unwrap();
        assert_eq!(expert.len(), expected);
        let bank_len = intermediate * row_bytes(hidden).unwrap();
        assert_eq!(expert[PAYLOAD_HEADER_BYTES + 2], 0x99);
        assert_eq!(expert[PAYLOAD_HEADER_BYTES + bank_len + 2], 0xaa);
        Q4_0Expert::from_payload(hidden, intermediate, &expert).unwrap();
        drop(source);
        std::fs::remove_file(path).unwrap();
    }
}
