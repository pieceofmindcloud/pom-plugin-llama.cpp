//! Executor CPU dos blocos clássicos Q4_1, Q5_0, Q5_1 e Q8_0.
//!
//! Os layouts são os definidos em `ggml-common.h` e as fórmulas de
//! dequantização seguem `dequantize_row_q*_ref` do ggml. O parser de experts
//! permanece separado do decoder para que um dtype nunca seja lido como outro.

use super::cache::{CacheError, ExpertCache, ExpertKey};
use super::format_q4_0::{Q4LoadError, TensorSpec};
use memmap2::Mmap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

const BLOCK_ELEMENTS: usize = 32;
const PAYLOAD_HEADER_BYTES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyQuantFormat {
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
}

impl LegacyQuantFormat {
    pub fn from_dtype(dtype: u32) -> Result<Self, Q4LoadError> {
        match dtype {
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            other => Err(Q4LoadError::UnsupportedTensorType(other)),
        }
    }

    pub fn dtype(self) -> u32 {
        match self {
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
        }
    }

    pub fn row_bytes(self, columns: usize) -> Result<usize, LegacyQuantError> {
        row_bytes(self, columns)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegacyQuantError {
    Geometry(&'static str),
    Payload(&'static str),
    PayloadLength { expected: usize, actual: usize },
}

impl fmt::Display for LegacyQuantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(reason) => write!(formatter, "geometria legacy inválida: {reason}"),
            Self::Payload(reason) => write!(formatter, "payload legacy inválido: {reason}"),
            Self::PayloadLength { expected, actual } => write!(
                formatter,
                "payload legacy tem {actual} bytes, mas a geometria exige {expected}"
            ),
        }
    }
}

impl std::error::Error for LegacyQuantError {}

#[derive(Debug)]
pub enum LegacyQuantExecuteError {
    Cache(CacheError),
    Format(LegacyQuantError),
    Source(Q4LoadError),
    HiddenWidth { expected: usize, actual: usize },
}

impl fmt::Display for LegacyQuantExecuteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
            Self::Source(error) => error.fmt(formatter),
            Self::HiddenWidth { expected, actual } => write!(
                formatter,
                "hidden legacy tem {actual} elementos, esperado {expected}"
            ),
        }
    }
}

impl std::error::Error for LegacyQuantExecuteError {}

impl From<CacheError> for LegacyQuantExecuteError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

impl From<LegacyQuantError> for LegacyQuantExecuteError {
    fn from(error: LegacyQuantError) -> Self {
        Self::Format(error)
    }
}

impl From<Q4LoadError> for LegacyQuantExecuteError {
    fn from(error: Q4LoadError) -> Self {
        Self::Source(error)
    }
}

enum GateUpTensors {
    Fused(Vec<TensorSpec>),
    Split {
        gate: Vec<TensorSpec>,
        up: Vec<TensorSpec>,
    },
}

pub struct LegacyQuantSource {
    mmap: Mmap,
    data_start: usize,
    experts: u32,
    hidden: usize,
    intermediate: usize,
    format: LegacyQuantFormat,
    gate_up: GateUpTensors,
    down: Vec<TensorSpec>,
    router: Vec<TensorSpec>,
}

impl LegacyQuantSource {
    pub fn open_infer(path: impl AsRef<Path>, experts: u32) -> Result<Self, Q4LoadError> {
        if experts == 0 {
            return Err(Q4LoadError::InvalidTensor("contagem de experts zero"));
        }
        let (mmap, data_start, tensors) = super::format_q4_0::map_tensor_source(path.as_ref())?;
        let format = detect_format(&tensors)?;
        let down = tensors_with_suffix(&tensors, "ffn_down_exps.weight");
        if down.is_empty() {
            return Err(Q4LoadError::MissingTensor("ffn_down_exps.weight".into()));
        }
        let fused = tensors_with_suffix(&tensors, "ffn_gate_up_exps.weight");
        let gate = tensors_with_suffix(&tensors, "ffn_gate_exps.weight");
        let up = tensors_with_suffix(&tensors, "ffn_up_exps.weight");
        let (hidden, intermediate, gate_up) = if !fused.is_empty() {
            if !gate.is_empty() || !up.is_empty() {
                return Err(Q4LoadError::InvalidTensor(
                    "GGUF mistura gate_up fundido e bancos separados",
                ));
            }
            let first = &fused[0];
            validate_dtype(first, format)?;
            if first.dims.len() != 3 || first.dims[2] != u64::from(experts) {
                return Err(Q4LoadError::InvalidTensor(
                    "gate_up legacy precisa ser rank 3 com o número de experts",
                ));
            }
            let hidden = dimension(first, 0)?;
            let two_intermediate = dimension(first, 1)?;
            if hidden == 0 || two_intermediate == 0 || !two_intermediate.is_multiple_of(2) {
                return Err(Q4LoadError::InvalidTensor(
                    "geometria gate_up legacy inválida",
                ));
            }
            let intermediate = two_intermediate / 2;
            validate_all_same_dtype(&fused, format)?;
            (hidden, intermediate, GateUpTensors::Fused(fused))
        } else {
            if gate.is_empty() || up.is_empty() {
                return Err(Q4LoadError::MissingTensor(
                    "ffn_gate_exps.weight ou ffn_up_exps.weight".into(),
                ));
            }
            let first = &gate[0];
            validate_dtype(first, format)?;
            validate_all_same_dtype(&gate, format)?;
            validate_all_same_dtype(&up, format)?;
            if first.dims.len() != 3 || first.dims[2] != u64::from(experts) {
                return Err(Q4LoadError::InvalidTensor(
                    "gate legacy precisa ser rank 3 com o número de experts",
                ));
            }
            let intermediate = dimension(first, 0)?;
            let hidden = dimension(first, 1)?;
            if intermediate == 0 || hidden == 0 {
                return Err(Q4LoadError::InvalidTensor("geometria gate legacy vazia"));
            }
            for tensor in gate.iter().chain(up.iter()) {
                if tensor.dims != first.dims {
                    return Err(Q4LoadError::InvalidTensor(
                        "gate e up têm geometrias diferentes",
                    ));
                }
            }
            (hidden, intermediate, GateUpTensors::Split { gate, up })
        };
        validate_all_same_dtype(&down, format)?;
        for tensor in &down {
            validate_matrix_tensor(tensor, experts, intermediate, hidden, format)?;
        }
        match &gate_up {
            GateUpTensors::Fused(items) => {
                for tensor in items {
                    validate_matrix_tensor(tensor, experts, 2 * intermediate, hidden, format)?;
                }
            }
            GateUpTensors::Split { gate, up } => {
                for tensor in gate.iter().chain(up) {
                    validate_matrix_tensor(tensor, experts, intermediate, hidden, format)?;
                }
            }
        }
        let layers = layer_count(match &gate_up {
            GateUpTensors::Fused(items) => items,
            GateUpTensors::Split { gate, .. } => gate,
        });
        if layers == 0 {
            return Err(Q4LoadError::InvalidTensor("GGUF legacy sem camadas"));
        }
        Ok(Self {
            mmap,
            data_start,
            experts,
            hidden,
            intermediate,
            format,
            gate_up,
            down,
            router: tensors_with_suffix(&tensors, "ffn_gate_inp.weight"),
        })
    }

    pub fn format(&self) -> LegacyQuantFormat {
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

    pub fn layer_count(&self) -> u32 {
        layer_count(match &self.gate_up {
            GateUpTensors::Fused(items) => items,
            GateUpTensors::Split { gate, .. } => gate,
        })
    }

    pub fn has_router(&self) -> bool {
        !self.router.is_empty()
    }

    pub fn router_logits(&self, layer: u32, hidden: &[f32]) -> Result<Vec<f32>, Q4LoadError> {
        if hidden.len() != self.hidden {
            return Err(Q4LoadError::InvalidTensor(
                "hidden não corresponde à largura do router legacy",
            ));
        }
        let tensor = tensor_for_layer(&self.router, layer, "ffn_gate_inp.weight")?;
        if tensor.dims.len() != 2
            || tensor.dims[0] != self.hidden as u64
            || tensor.dims[1] != u64::from(self.experts)
        {
            return Err(Q4LoadError::InvalidTensor(
                "dimensões do router legacy não correspondem ao modelo",
            ));
        }
        let router_format = scalar_router_format(tensor.dtype)?;
        let bytes = router_format.1;
        let total = self
            .hidden
            .checked_mul(self.experts as usize)
            .and_then(|count| count.checked_mul(bytes))
            .ok_or(Q4LoadError::InvalidTensor("tamanho do router excede usize"))?;
        let raw = self.tensor_bytes(tensor, total)?;
        let mut logits = Vec::with_capacity(self.experts as usize);
        for expert in 0..self.experts as usize {
            let row = &raw[expert * self.hidden * bytes..(expert + 1) * self.hidden * bytes];
            logits.push(
                row.chunks_exact(bytes)
                    .zip(hidden)
                    .map(|(encoded, &input)| decode_router(encoded, router_format.0) * input)
                    .sum(),
            );
        }
        Ok(logits)
    }

    pub fn load_expert(&self, layer: u32, expert: u32) -> Result<Vec<u8>, Q4LoadError> {
        if expert >= self.experts {
            return Err(Q4LoadError::ExpertOutOfRange {
                expert,
                count: self.experts,
            });
        }
        let down = tensor_for_layer(&self.down, layer, "ffn_down_exps.weight")?;
        let down_per_expert = self
            .hidden
            .checked_mul(
                row_bytes(self.format, self.intermediate)
                    .map_err(|_| Q4LoadError::InvalidTensor("geometria down legacy"))?,
            )
            .ok_or(Q4LoadError::InvalidTensor("overflow no down legacy"))?;
        let down = self.expert_slice(down, expert, down_per_expert)?.to_vec();
        let gate_up = match &self.gate_up {
            GateUpTensors::Fused(items) => {
                let tensor = tensor_for_layer(items, layer, "ffn_gate_up_exps.weight")?;
                let per_expert = 2usize
                    .checked_mul(self.intermediate)
                    .and_then(|rows| {
                        row_bytes(self.format, self.hidden)
                            .ok()
                            .and_then(|b| rows.checked_mul(b))
                    })
                    .ok_or(Q4LoadError::InvalidTensor("overflow no gate_up legacy"))?;
                self.expert_slice(tensor, expert, per_expert)?.to_vec()
            }
            GateUpTensors::Split { gate, up } => {
                let gate = tensor_for_layer(gate, layer, "ffn_gate_exps.weight")?;
                let up = tensor_for_layer(up, layer, "ffn_up_exps.weight")?;
                let per_expert = self
                    .intermediate
                    .checked_mul(
                        row_bytes(self.format, self.hidden)
                            .map_err(|_| Q4LoadError::InvalidTensor("geometria gate legacy"))?,
                    )
                    .ok_or(Q4LoadError::InvalidTensor("overflow no gate/up legacy"))?;
                let gate = self.expert_slice(gate, expert, per_expert)?;
                let up = self.expert_slice(up, expert, per_expert)?;
                let mut fused = Vec::with_capacity(gate.len() + up.len());
                fused.extend_from_slice(gate);
                fused.extend_from_slice(up);
                fused
            }
        };
        pack_payload(&gate_up, &down)
    }

    fn expert_slice<'a>(
        &'a self,
        tensor: &TensorSpec,
        expert: u32,
        bytes_per_expert: usize,
    ) -> Result<&'a [u8], Q4LoadError> {
        validate_dtype(tensor, self.format)?;
        let total = bytes_per_expert
            .checked_mul(self.experts as usize)
            .ok_or(Q4LoadError::InvalidTensor("overflow no tensor legacy"))?;
        let start = usize::try_from(tensor.offset)
            .ok()
            .and_then(|offset| self.data_start.checked_add(offset))
            .and_then(|offset| {
                bytes_per_expert
                    .checked_mul(expert as usize)
                    .and_then(|delta| offset.checked_add(delta))
            })
            .ok_or(Q4LoadError::InvalidTensor("offset GGUF inválido"))?;
        let end = start
            .checked_add(bytes_per_expert)
            .ok_or(Q4LoadError::InvalidTensor("fim do tensor excede usize"))?;
        let tensor_start = start
            .checked_sub(bytes_per_expert * expert as usize)
            .ok_or(Q4LoadError::InvalidTensor("expert excede o tensor"))?;
        let tensor_end = tensor_start
            .checked_add(total)
            .ok_or(Q4LoadError::InvalidTensor("tamanho GGUF inválido"))?;
        if end > tensor_end || tensor_end > self.mmap.len() {
            return Err(Q4LoadError::InvalidTensor("tensor excede o arquivo"));
        }
        Ok(&self.mmap[start..end])
    }

    fn tensor_bytes<'a>(
        &'a self,
        tensor: &TensorSpec,
        total_bytes: usize,
    ) -> Result<&'a [u8], Q4LoadError> {
        let start = usize::try_from(tensor.offset)
            .ok()
            .and_then(|offset| self.data_start.checked_add(offset))
            .ok_or(Q4LoadError::InvalidTensor("offset GGUF inválido"))?;
        let end = start
            .checked_add(total_bytes)
            .ok_or(Q4LoadError::InvalidTensor("tamanho do tensor excede usize"))?;
        self.mmap
            .get(start..end)
            .ok_or(Q4LoadError::InvalidTensor("tensor excede o arquivo"))
    }
}

pub struct LegacyQuantExpert<'a> {
    format: LegacyQuantFormat,
    hidden: usize,
    intermediate: usize,
    gate_up: &'a [u8],
    down: &'a [u8],
}

impl<'a> LegacyQuantExpert<'a> {
    pub fn packed_bytes(
        format: LegacyQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<usize, LegacyQuantError> {
        let gate_up = 2usize
            .checked_mul(intermediate)
            .and_then(|rows| {
                row_bytes(format, hidden)
                    .ok()
                    .and_then(|b| rows.checked_mul(b))
            })
            .ok_or(LegacyQuantError::Geometry("overflow em gate_up"))?;
        let down = hidden
            .checked_mul(row_bytes(format, intermediate)?)
            .ok_or(LegacyQuantError::Geometry("overflow em down"))?;
        PAYLOAD_HEADER_BYTES
            .checked_add(gate_up)
            .and_then(|bytes| bytes.checked_add(down))
            .ok_or(LegacyQuantError::Geometry("overflow no payload"))
    }

    pub fn from_payload(
        format: LegacyQuantFormat,
        hidden: usize,
        intermediate: usize,
        payload: &'a [u8],
    ) -> Result<Self, LegacyQuantError> {
        let expected = Self::packed_bytes(format, hidden, intermediate)?;
        if payload.len() != expected {
            return Err(LegacyQuantError::PayloadLength {
                expected,
                actual: payload.len(),
            });
        }
        let gate_up_len = u32::from_le_bytes(
            payload[..4]
                .try_into()
                .map_err(|_| LegacyQuantError::Payload("cabeçalho gate_up ausente"))?,
        ) as usize;
        let down_len = u32::from_le_bytes(
            payload[4..8]
                .try_into()
                .map_err(|_| LegacyQuantError::Payload("cabeçalho down ausente"))?,
        ) as usize;
        let expected_gate_up = 2 * intermediate * row_bytes(format, hidden)?;
        let expected_down = hidden * row_bytes(format, intermediate)?;
        if gate_up_len != expected_gate_up || down_len != expected_down {
            return Err(LegacyQuantError::Payload(
                "tamanhos gate_up/down não correspondem à geometria",
            ));
        }
        let down_start = PAYLOAD_HEADER_BYTES + gate_up_len;
        if down_start + down_len != payload.len() {
            return Err(LegacyQuantError::Payload("seções não cobrem o payload"));
        }
        Ok(Self {
            format,
            hidden,
            intermediate,
            gate_up: &payload[PAYLOAD_HEADER_BYTES..down_start],
            down: &payload[down_start..],
        })
    }

    pub fn run(&self, hidden: &[f32]) -> Result<Vec<f32>, LegacyQuantExecuteError> {
        if hidden.len() != self.hidden {
            return Err(LegacyQuantExecuteError::HiddenWidth {
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
        );
        let gated = gate_up[..self.intermediate]
            .iter()
            .zip(&gate_up[self.intermediate..])
            .map(|(&gate, &up)| silu(gate) * up)
            .collect::<Vec<_>>();
        Ok(mat_vec(
            self.format,
            self.down,
            self.hidden,
            self.intermediate,
            &gated,
        ))
    }
}

pub struct LegacyQuantExecutor {
    cache: Arc<ExpertCache>,
    format: LegacyQuantFormat,
    hidden: usize,
    intermediate: usize,
}

impl LegacyQuantExecutor {
    pub fn new(
        cache: Arc<ExpertCache>,
        format: LegacyQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, LegacyQuantError> {
        LegacyQuantExpert::packed_bytes(format, hidden, intermediate)?;
        Ok(Self {
            cache,
            format,
            hidden,
            intermediate,
        })
    }

    pub fn run_from_source(
        &self,
        source: &LegacyQuantSource,
        key: ExpertKey,
        layer: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>, LegacyQuantExecuteError> {
        let expert = key.expert;
        let lease = match self.cache.lookup(&key) {
            Some(lease) => lease,
            None => self.cache.admit(key, source.load_expert(layer, expert)?)?,
        };
        LegacyQuantExpert::from_payload(self.format, self.hidden, self.intermediate, lease.bytes())?
            .run(hidden)
    }
}

fn detect_format(tensors: &[TensorSpec]) -> Result<LegacyQuantFormat, Q4LoadError> {
    let tensor = tensors
        .iter()
        .find(|tensor| {
            tensor.name.starts_with("blk.")
                && (tensor.name.ends_with("ffn_gate_up_exps.weight")
                    || tensor.name.ends_with("ffn_gate_exps.weight"))
        })
        .ok_or_else(|| Q4LoadError::MissingTensor("ffn_gate[_up]_exps.weight".into()))?;
    LegacyQuantFormat::from_dtype(tensor.dtype)
}

fn tensors_with_suffix(tensors: &[TensorSpec], suffix: &str) -> Vec<TensorSpec> {
    tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("blk.") && tensor.name.ends_with(suffix))
        .cloned()
        .collect()
}

fn validate_dtype(tensor: &TensorSpec, format: LegacyQuantFormat) -> Result<(), Q4LoadError> {
    if tensor.dtype != format.dtype() {
        return Err(Q4LoadError::UnsupportedTensorType(tensor.dtype));
    }
    Ok(())
}

fn validate_all_same_dtype(
    tensors: &[TensorSpec],
    format: LegacyQuantFormat,
) -> Result<(), Q4LoadError> {
    for tensor in tensors {
        validate_dtype(tensor, format)?;
    }
    Ok(())
}

fn validate_matrix_tensor(
    tensor: &TensorSpec,
    experts: u32,
    rows: usize,
    columns: usize,
    format: LegacyQuantFormat,
) -> Result<(), Q4LoadError> {
    validate_dtype(tensor, format)?;
    if tensor.dims.len() != 3
        || tensor.dims[0] != columns as u64
        || tensor.dims[1] != rows as u64
        || tensor.dims[2] != u64::from(experts)
    {
        return Err(Q4LoadError::InvalidTensor(
            "dimensões não correspondem ao expert legacy",
        ));
    }
    Ok(())
}

fn dimension(tensor: &TensorSpec, index: usize) -> Result<usize, Q4LoadError> {
    usize::try_from(
        *tensor
            .dims
            .get(index)
            .ok_or(Q4LoadError::InvalidTensor("dimensão ausente"))?,
    )
    .map_err(|_| Q4LoadError::InvalidTensor("dimensão excede usize"))
}

fn layer_count(tensors: &[TensorSpec]) -> u32 {
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

fn tensor_for_layer<'a>(
    tensors: &'a [TensorSpec],
    layer: u32,
    suffix: &str,
) -> Result<&'a TensorSpec, Q4LoadError> {
    tensors
        .iter()
        .find(|tensor| tensor.name == format!("blk.{layer}.{suffix}"))
        .ok_or_else(|| Q4LoadError::MissingTensor(format!("blk.{layer}.{suffix}")))
}

fn pack_payload(gate_up: &[u8], down: &[u8]) -> Result<Vec<u8>, Q4LoadError> {
    if gate_up.len() > u32::MAX as usize || down.len() > u32::MAX as usize {
        return Err(Q4LoadError::InvalidTensor("payload legacy excede u32"));
    }
    let mut payload = Vec::with_capacity(PAYLOAD_HEADER_BYTES + gate_up.len() + down.len());
    payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
    payload.extend_from_slice(gate_up);
    payload.extend_from_slice(down);
    Ok(payload)
}

fn row_bytes(format: LegacyQuantFormat, columns: usize) -> Result<usize, LegacyQuantError> {
    if columns == 0 || !columns.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(LegacyQuantError::Geometry("linha não fecha bloco de 32"));
    }
    columns
        .checked_div(BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(format.block_bytes()))
        .ok_or(LegacyQuantError::Geometry("overflow no tamanho da linha"))
}

fn mat_vec(
    format: LegacyQuantFormat,
    matrix: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Vec<f32> {
    let row_size = row_bytes(format, columns).expect("geometria validada antes do mat-vec");
    (0..rows)
        .map(|row| {
            let bytes = &matrix[row * row_size..(row + 1) * row_size];
            dequantize_row(format, bytes, columns)
                .into_iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect()
}

pub fn decode_row(
    format: LegacyQuantFormat,
    row: &[u8],
    columns: usize,
) -> Result<Vec<f32>, LegacyQuantError> {
    let expected = row_bytes(format, columns)?;
    if row.len() != expected {
        return Err(LegacyQuantError::PayloadLength {
            expected,
            actual: row.len(),
        });
    }
    Ok(dequantize_row(format, row, columns))
}

fn dequantize_row(format: LegacyQuantFormat, row: &[u8], columns: usize) -> Vec<f32> {
    let block_bytes = format.block_bytes();
    row.chunks_exact(block_bytes)
        .flat_map(|block| dequantize_block(format, block))
        .take(columns)
        .collect()
}

fn dequantize_block(format: LegacyQuantFormat, block: &[u8]) -> [f32; BLOCK_ELEMENTS] {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut output = [0.0f32; BLOCK_ELEMENTS];
    match format {
        LegacyQuantFormat::Q4_1 => {
            let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            for j in 0..16 {
                output[j] = (block[4 + j] & 0x0f) as f32 * d + m;
                output[j + 16] = (block[4 + j] >> 4) as f32 * d + m;
            }
        }
        LegacyQuantFormat::Q5_0 | LegacyQuantFormat::Q5_1 => {
            let (qh_start, qs_start, m) = match format {
                LegacyQuantFormat::Q5_0 => (2, 6, 0.0),
                LegacyQuantFormat::Q5_1 => {
                    (4, 8, f16_to_f32(u16::from_le_bytes([block[2], block[3]])))
                }
                _ => unreachable!(),
            };
            let qh = u32::from_le_bytes(block[qh_start..qh_start + 4].try_into().unwrap());
            for j in 0..16 {
                let high0 = ((qh >> j) << 4) & 0x10;
                let high1 = (qh >> (j + 12)) & 0x10;
                let q0 = (u32::from(block[qs_start + j] & 0x0f) | high0) as f32;
                let q1 = (u32::from(block[qs_start + j] >> 4) | high1) as f32;
                output[j] =
                    (q0 - if matches!(format, LegacyQuantFormat::Q5_0) {
                        16.0
                    } else {
                        0.0
                    }) * d
                        + m;
                output[j + 16] =
                    (q1 - if matches!(format, LegacyQuantFormat::Q5_0) {
                        16.0
                    } else {
                        0.0
                    }) * d
                        + m;
            }
        }
        LegacyQuantFormat::Q8_0 => {
            for j in 0..BLOCK_ELEMENTS {
                output[j] = i8::from_ne_bytes([block[2 + j]]) as f32 * d;
            }
        }
    }
    output
}

fn scalar_router_format(dtype: u32) -> Result<(u32, usize), Q4LoadError> {
    match dtype {
        0 => Ok((0, 4)),
        1 | 30 => Ok((dtype, 2)),
        other => Err(Q4LoadError::UnsupportedTensorType(other)),
    }
}

fn decode_router(encoded: &[u8], dtype: u32) -> f32 {
    match dtype {
        0 => f32::from_le_bytes(encoded.try_into().unwrap()),
        1 => f16_to_f32(u16::from_le_bytes(encoded.try_into().unwrap())),
        30 => f32::from_bits(u32::from(u16::from_le_bytes(encoded.try_into().unwrap())) << 16),
        _ => unreachable!("scalar_router_format validou o dtype"),
    }
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

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
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

    impl FreeMemoryProbe for FixedProbe {
        fn free_bytes(&self) -> io::Result<u64> {
            Ok(self.0.load(Ordering::Relaxed))
        }
    }

    fn test_block(format: LegacyQuantFormat) -> Vec<u8> {
        let mut block = Vec::new();
        block.extend_from_slice(&0x3c00u16.to_le_bytes());
        match format {
            LegacyQuantFormat::Q4_1 => {
                block.extend_from_slice(&0u16.to_le_bytes());
                block.extend(std::iter::repeat_n(0x11, 16));
            }
            LegacyQuantFormat::Q5_0 => {
                block.extend([0u8; 4]);
                block.extend(std::iter::repeat_n(0x88, 16));
            }
            LegacyQuantFormat::Q5_1 => {
                block.extend_from_slice(&0u16.to_le_bytes());
                block.extend([0xffu8; 4]);
                block.extend(std::iter::repeat_n(0x00, 16));
            }
            LegacyQuantFormat::Q8_0 => {
                block.extend(std::iter::repeat_n(1u8, 32));
            }
        }
        block
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

    pub fn write_legacy_source(path: &Path, format: LegacyQuantFormat, split: bool) {
        let hidden = 32usize;
        let intermediate = 32usize;
        let experts = 2usize;
        let row = test_block(format);
        let gate = row.repeat(experts * intermediate);
        let up = gate.clone();
        let down = row.repeat(experts * hidden);
        let mut router = Vec::new();
        for _ in 0..experts * hidden {
            router.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let tensor_count = if split { 4u64 } else { 3u64 };
        let mut file = Vec::new();
        file.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        file.extend_from_slice(&2u32.to_le_bytes());
        file.extend_from_slice(&tensor_count.to_le_bytes());
        file.extend_from_slice(&1u64.to_le_bytes());
        gguf_string(&mut file, "general.alignment");
        file.extend_from_slice(&4u32.to_le_bytes());
        file.extend_from_slice(&32u32.to_le_bytes());
        let mut offset = 0u64;
        if split {
            gguf_tensor(
                &mut file,
                "blk.0.ffn_gate_exps.weight",
                &[intermediate as u64, hidden as u64, experts as u64],
                offset,
                format.dtype(),
            );
            offset += gate.len() as u64;
            gguf_tensor(
                &mut file,
                "blk.0.ffn_up_exps.weight",
                &[intermediate as u64, hidden as u64, experts as u64],
                offset,
                format.dtype(),
            );
            offset += up.len() as u64;
        } else {
            let mut fused = gate.clone();
            fused.extend_from_slice(&up);
            gguf_tensor(
                &mut file,
                "blk.0.ffn_gate_up_exps.weight",
                &[hidden as u64, (2 * intermediate) as u64, experts as u64],
                offset,
                format.dtype(),
            );
            offset += fused.len() as u64;
        }
        gguf_tensor(
            &mut file,
            "blk.0.ffn_down_exps.weight",
            &[intermediate as u64, hidden as u64, experts as u64],
            offset,
            format.dtype(),
        );
        offset += down.len() as u64;
        gguf_tensor(
            &mut file,
            "blk.0.ffn_gate_inp.weight",
            &[hidden as u64, experts as u64],
            offset,
            0,
        );
        let data_start = (file.len() + 31) / 32 * 32;
        file.resize(data_start, 0);
        if split {
            file.extend_from_slice(&gate);
            file.extend_from_slice(&up);
        } else {
            file.extend_from_slice(&gate);
            file.extend_from_slice(&up);
        }
        file.extend_from_slice(&down);
        file.extend_from_slice(&router);
        File::create(path).unwrap().write_all(&file).unwrap();
    }

    #[test]
    fn legacy_q4_1_q5_0_q5_1_q8_0_decode_a_known_vector() {
        let expected = [
            (LegacyQuantFormat::Q4_1, 1.0f32),
            (LegacyQuantFormat::Q5_0, -8.0),
            (LegacyQuantFormat::Q5_1, 16.0),
            (LegacyQuantFormat::Q8_0, 1.0),
        ];
        for (format, value) in expected {
            let decoded = dequantize_row(format, &test_block(format), 32);
            assert!(decoded.iter().all(|actual| (*actual - value).abs() < 1e-6));
        }
    }

    #[test]
    fn legacy_source_and_executor_cover_fused_and_split() {
        for format in [
            LegacyQuantFormat::Q4_1,
            LegacyQuantFormat::Q5_0,
            LegacyQuantFormat::Q5_1,
            LegacyQuantFormat::Q8_0,
        ] {
            for split in [false, true] {
                let path = std::env::temp_dir().join(format!(
                    "pom-ftk-legacy-{}-{}-{}.gguf",
                    format.name(),
                    split,
                    std::process::id()
                ));
                write_legacy_source(&path, format, split);
                let source = LegacyQuantSource::open_infer(&path, 2).unwrap();
                let cache = ExpertCache::new(
                    MemoryAccount {
                        baseline_free_bytes: 10_000_000,
                        compute_bytes: 0,
                        weights_bytes: 0,
                        fixed_bytes: 0,
                        reserve_bytes: 0,
                    },
                    Arc::new(FixedProbe(AtomicU64::new(10_000_000))),
                );
                let executor =
                    LegacyQuantExecutor::new(cache, format, source.hidden(), source.intermediate())
                        .unwrap();
                let output = executor
                    .run_from_source(
                        &source,
                        ExpertKey::new("fixture", 0, 0, format.name()),
                        0,
                        &[1.0; 32],
                    )
                    .unwrap();
                assert_eq!(output.len(), 32);
                assert!(output.iter().all(|value| value.is_finite()));
                std::fs::remove_file(path).unwrap();
            }
        }
    }
}
