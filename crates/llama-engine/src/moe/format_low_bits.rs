//! Executor CPU para MXFP4, TQ1_0, TQ2_0, Q1_0 e Q2_0.
//!
//! Cada formato mantém seu próprio tamanho de bloco e rotina de unpacking.
//! Isso evita que a semelhança superficial entre os formatos de baixa largura
//! vire uma leitura silenciosamente errada do payload.

use super::cache::{CacheError, ExpertCache, ExpertKey};
use super::format_q4_0::{parse_tensor_table, Q4LoadError, TensorSpec};
use memmap2::Mmap;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const PAYLOAD_HEADER_BYTES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowBitFormat {
    Mxfp4,
    Tq1_0,
    Tq2_0,
    Q1_0,
    Q2_0,
}

impl LowBitFormat {
    pub fn from_dtype(dtype: u32) -> Result<Self, Q4LoadError> {
        match dtype {
            34 => Ok(Self::Tq1_0),
            35 => Ok(Self::Tq2_0),
            39 => Ok(Self::Mxfp4),
            41 => Ok(Self::Q1_0),
            42 => Ok(Self::Q2_0),
            other => Err(Q4LoadError::UnsupportedTensorType(other)),
        }
    }

    pub fn dtype(self) -> u32 {
        match self {
            Self::Tq1_0 => 34,
            Self::Tq2_0 => 35,
            Self::Mxfp4 => 39,
            Self::Q1_0 => 41,
            Self::Q2_0 => 42,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mxfp4 => "mxfp4",
            Self::Tq1_0 => "tq1_0",
            Self::Tq2_0 => "tq2_0",
            Self::Q1_0 => "q1_0",
            Self::Q2_0 => "q2_0",
        }
    }

    fn block_elements(self) -> usize {
        match self {
            Self::Mxfp4 => 32,
            Self::Tq1_0 | Self::Tq2_0 => 256,
            Self::Q1_0 => 128,
            Self::Q2_0 => 64,
        }
    }

    pub fn row_bytes(self, columns: usize) -> Result<usize, LowBitError> {
        row_bytes(self, columns)
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Mxfp4 => 17,
            Self::Tq1_0 => 54,
            Self::Tq2_0 => 66,
            Self::Q1_0 | Self::Q2_0 => 18,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowBitError {
    Geometry(&'static str),
    Payload(&'static str),
    PayloadLength { expected: usize, actual: usize },
}

impl fmt::Display for LowBitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(reason) => write!(formatter, "geometria low-bit inválida: {reason}"),
            Self::Payload(reason) => write!(formatter, "payload low-bit inválido: {reason}"),
            Self::PayloadLength { expected, actual } => write!(
                formatter,
                "payload low-bit tem {actual} bytes, mas a geometria exige {expected}"
            ),
        }
    }
}

impl std::error::Error for LowBitError {}

#[derive(Debug)]
pub enum LowBitExecuteError {
    Cache(CacheError),
    Format(LowBitError),
    Source(Q4LoadError),
    HiddenWidth { expected: usize, actual: usize },
}

impl fmt::Display for LowBitExecuteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
            Self::Source(error) => error.fmt(formatter),
            Self::HiddenWidth { expected, actual } => write!(
                formatter,
                "hidden low-bit tem {actual} elementos, esperado {expected}"
            ),
        }
    }
}

impl std::error::Error for LowBitExecuteError {}

impl From<CacheError> for LowBitExecuteError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

impl From<LowBitError> for LowBitExecuteError {
    fn from(error: LowBitError) -> Self {
        Self::Format(error)
    }
}

impl From<Q4LoadError> for LowBitExecuteError {
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

pub struct LowBitSource {
    mmap: Mmap,
    data_start: usize,
    experts: u32,
    hidden: usize,
    intermediate: usize,
    format: LowBitFormat,
    gate_up: GateUpTensors,
    down: Vec<TensorSpec>,
    router: Vec<TensorSpec>,
}

impl LowBitSource {
    pub fn open_infer(path: impl AsRef<Path>, experts: u32) -> Result<Self, Q4LoadError> {
        if experts == 0 {
            return Err(Q4LoadError::InvalidTensor("contagem de experts zero"));
        }
        let file = File::open(path.as_ref()).map_err(|error| Q4LoadError::Io(error.to_string()))?;
        // SAFETY: Mmap owns the mapping and outlives every returned tensor slice.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
        let (data_start, tensors) = parse_tensor_table(&mmap)?;
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
                    "gate_up low-bit precisa ser rank 3 com o número de experts",
                ));
            }
            let hidden = dimension(first, 0)?;
            let two_intermediate = dimension(first, 1)?;
            if hidden == 0 || two_intermediate == 0 || !two_intermediate.is_multiple_of(2) {
                return Err(Q4LoadError::InvalidTensor(
                    "geometria gate_up low-bit inválida",
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
                    "gate low-bit precisa ser rank 3 com o número de experts",
                ));
            }
            let intermediate = dimension(first, 0)?;
            let hidden = dimension(first, 1)?;
            if intermediate == 0 || hidden == 0 {
                return Err(Q4LoadError::InvalidTensor("geometria gate low-bit vazia"));
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
            return Err(Q4LoadError::InvalidTensor("GGUF low-bit sem camadas"));
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

    pub fn format(&self) -> LowBitFormat {
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
                "hidden não corresponde à largura do router low-bit",
            ));
        }
        let tensor = tensor_for_layer(&self.router, layer, "ffn_gate_inp.weight")?;
        if tensor.dims.len() != 2
            || tensor.dims[0] != self.hidden as u64
            || tensor.dims[1] != u64::from(self.experts)
        {
            return Err(Q4LoadError::InvalidTensor(
                "dimensões do router low-bit não correspondem ao modelo",
            ));
        }
        let (router_dtype, bytes) = scalar_router_format(tensor.dtype)?;
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
                    .map(|(encoded, &input)| decode_scalar(encoded, router_dtype) * input)
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
                    .map_err(|_| Q4LoadError::InvalidTensor("geometria down low-bit"))?,
            )
            .ok_or(Q4LoadError::InvalidTensor("overflow no down low-bit"))?;
        let down = self.expert_slice(down, expert, down_per_expert)?.to_vec();
        let gate_up = match &self.gate_up {
            GateUpTensors::Fused(items) => {
                let tensor = tensor_for_layer(items, layer, "ffn_gate_up_exps.weight")?;
                let per_expert = 2usize
                    .checked_mul(self.intermediate)
                    .and_then(|rows| {
                        row_bytes(self.format, self.hidden)
                            .ok()
                            .and_then(|bytes| rows.checked_mul(bytes))
                    })
                    .ok_or(Q4LoadError::InvalidTensor("overflow no gate_up low-bit"))?;
                self.expert_slice(tensor, expert, per_expert)?.to_vec()
            }
            GateUpTensors::Split { gate, up } => {
                let gate = tensor_for_layer(gate, layer, "ffn_gate_exps.weight")?;
                let up = tensor_for_layer(up, layer, "ffn_up_exps.weight")?;
                let per_expert = self
                    .intermediate
                    .checked_mul(
                        row_bytes(self.format, self.hidden)
                            .map_err(|_| Q4LoadError::InvalidTensor("geometria gate low-bit"))?,
                    )
                    .ok_or(Q4LoadError::InvalidTensor("overflow no gate/up low-bit"))?;
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
            .ok_or(Q4LoadError::InvalidTensor("overflow no tensor low-bit"))?;
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

pub struct LowBitExpert<'a> {
    format: LowBitFormat,
    hidden: usize,
    intermediate: usize,
    gate_up: &'a [u8],
    down: &'a [u8],
}

impl<'a> LowBitExpert<'a> {
    pub fn packed_bytes(
        format: LowBitFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<usize, LowBitError> {
        let gate_up = 2usize
            .checked_mul(intermediate)
            .and_then(|rows| {
                row_bytes(format, hidden)
                    .ok()
                    .and_then(|bytes| rows.checked_mul(bytes))
            })
            .ok_or(LowBitError::Geometry("overflow em gate_up"))?;
        let down = hidden
            .checked_mul(row_bytes(format, intermediate)?)
            .ok_or(LowBitError::Geometry("overflow em down"))?;
        PAYLOAD_HEADER_BYTES
            .checked_add(gate_up)
            .and_then(|bytes| bytes.checked_add(down))
            .ok_or(LowBitError::Geometry("overflow no payload"))
    }

    pub fn from_payload(
        format: LowBitFormat,
        hidden: usize,
        intermediate: usize,
        payload: &'a [u8],
    ) -> Result<Self, LowBitError> {
        let expected = Self::packed_bytes(format, hidden, intermediate)?;
        if payload.len() != expected {
            return Err(LowBitError::PayloadLength {
                expected,
                actual: payload.len(),
            });
        }
        let gate_up_len = u32::from_le_bytes(
            payload[..4]
                .try_into()
                .map_err(|_| LowBitError::Payload("cabeçalho gate_up ausente"))?,
        ) as usize;
        let down_len = u32::from_le_bytes(
            payload[4..8]
                .try_into()
                .map_err(|_| LowBitError::Payload("cabeçalho down ausente"))?,
        ) as usize;
        let expected_gate_up = 2 * intermediate * row_bytes(format, hidden)?;
        let expected_down = hidden * row_bytes(format, intermediate)?;
        if gate_up_len != expected_gate_up || down_len != expected_down {
            return Err(LowBitError::Payload(
                "tamanhos gate_up/down não correspondem à geometria",
            ));
        }
        let down_start = PAYLOAD_HEADER_BYTES + gate_up_len;
        if down_start + down_len != payload.len() {
            return Err(LowBitError::Payload("seções não cobrem o payload"));
        }
        Ok(Self {
            format,
            hidden,
            intermediate,
            gate_up: &payload[PAYLOAD_HEADER_BYTES..down_start],
            down: &payload[down_start..],
        })
    }

    pub fn run(&self, hidden: &[f32]) -> Result<Vec<f32>, LowBitExecuteError> {
        if hidden.len() != self.hidden {
            return Err(LowBitExecuteError::HiddenWidth {
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

pub struct LowBitExecutor {
    cache: Arc<ExpertCache>,
    format: LowBitFormat,
    hidden: usize,
    intermediate: usize,
}

impl LowBitExecutor {
    pub fn new(
        cache: Arc<ExpertCache>,
        format: LowBitFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, LowBitError> {
        LowBitExpert::packed_bytes(format, hidden, intermediate)?;
        Ok(Self {
            cache,
            format,
            hidden,
            intermediate,
        })
    }

    pub fn run_from_source(
        &self,
        source: &LowBitSource,
        key: ExpertKey,
        layer: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>, LowBitExecuteError> {
        let expert = key.expert;
        let lease = match self.cache.lookup(&key) {
            Some(lease) => lease,
            None => self.cache.admit(key, source.load_expert(layer, expert)?)?,
        };
        LowBitExpert::from_payload(self.format, self.hidden, self.intermediate, lease.bytes())?
            .run(hidden)
    }
}

fn detect_format(tensors: &[TensorSpec]) -> Result<LowBitFormat, Q4LoadError> {
    let tensor = tensors
        .iter()
        .find(|tensor| {
            tensor.name.starts_with("blk.")
                && (tensor.name.ends_with("ffn_gate_up_exps.weight")
                    || tensor.name.ends_with("ffn_gate_exps.weight"))
        })
        .ok_or_else(|| Q4LoadError::MissingTensor("ffn_gate[_up]_exps.weight".into()))?;
    LowBitFormat::from_dtype(tensor.dtype)
}

fn tensors_with_suffix(tensors: &[TensorSpec], suffix: &str) -> Vec<TensorSpec> {
    tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("blk.") && tensor.name.ends_with(suffix))
        .cloned()
        .collect()
}

fn validate_dtype(tensor: &TensorSpec, format: LowBitFormat) -> Result<(), Q4LoadError> {
    if tensor.dtype != format.dtype() {
        return Err(Q4LoadError::UnsupportedTensorType(tensor.dtype));
    }
    Ok(())
}

fn validate_all_same_dtype(
    tensors: &[TensorSpec],
    format: LowBitFormat,
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
    format: LowBitFormat,
) -> Result<(), Q4LoadError> {
    validate_dtype(tensor, format)?;
    if tensor.dims.len() != 3
        || tensor.dims[0] != columns as u64
        || tensor.dims[1] != rows as u64
        || tensor.dims[2] != u64::from(experts)
    {
        return Err(Q4LoadError::InvalidTensor(
            "dimensões não correspondem ao expert low-bit",
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
        return Err(Q4LoadError::InvalidTensor("payload low-bit excede u32"));
    }
    let mut payload = Vec::with_capacity(PAYLOAD_HEADER_BYTES + gate_up.len() + down.len());
    payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
    payload.extend_from_slice(gate_up);
    payload.extend_from_slice(down);
    Ok(payload)
}

fn row_bytes(format: LowBitFormat, columns: usize) -> Result<usize, LowBitError> {
    if columns == 0 || !columns.is_multiple_of(format.block_elements()) {
        return Err(LowBitError::Geometry("linha não fecha o bloco do dtype"));
    }
    columns
        .checked_div(format.block_elements())
        .and_then(|blocks| blocks.checked_mul(format.block_bytes()))
        .ok_or(LowBitError::Geometry("overflow no tamanho da linha"))
}

fn mat_vec(
    format: LowBitFormat,
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
    format: LowBitFormat,
    row: &[u8],
    columns: usize,
) -> Result<Vec<f32>, LowBitError> {
    let expected = row_bytes(format, columns)?;
    if row.len() != expected {
        return Err(LowBitError::PayloadLength {
            expected,
            actual: row.len(),
        });
    }
    Ok(dequantize_row(format, row, columns))
}

fn dequantize_row(format: LowBitFormat, row: &[u8], columns: usize) -> Vec<f32> {
    row.chunks_exact(format.block_bytes())
        .flat_map(|block| dequantize_block(format, block))
        .take(columns)
        .collect()
}

fn dequantize_block(format: LowBitFormat, block: &[u8]) -> Vec<f32> {
    match format {
        LowBitFormat::Mxfp4 => {
            let scale = e8m0_half(block[0]);
            let values = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
            let mut output = Vec::with_capacity(32);
            for byte in &block[1..] {
                output.push(scale * values[usize::from(byte & 0x0f)] as f32);
            }
            for byte in &block[1..] {
                output.push(scale * values[usize::from(byte >> 4)] as f32);
            }
            output
        }
        LowBitFormat::Q1_0 => {
            let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            (0..128)
                .map(|index| {
                    if (block[2 + index / 8] >> (index % 8)) & 1 != 0 {
                        scale
                    } else {
                        -scale
                    }
                })
                .collect()
        }
        LowBitFormat::Q2_0 => {
            let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            (0..64)
                .map(|index| {
                    let q = (block[2 + index / 4] >> ((index % 4) * 2)) & 3;
                    (i32::from(q) - 1) as f32 * scale
                })
                .collect()
        }
        LowBitFormat::Tq1_0 => dequantize_tq1(block),
        LowBitFormat::Tq2_0 => dequantize_tq2(block),
    }
}

fn dequantize_tq1(block: &[u8]) -> Vec<f32> {
    let scale = f16_to_f32(u16::from_le_bytes([block[52], block[53]]));
    let pow3 = [1u16, 3, 9, 27, 81, 243];
    let mut output = Vec::with_capacity(256);
    for (start, count) in [(0usize, 32usize), (32, 16)] {
        for &power in pow3.iter().take(5) {
            for m in 0..count {
                let q = u16::from(block[start + m]) * power;
                let xi = ((u32::from(q) * 3) >> 8) as i32;
                output.push((xi - 1) as f32 * scale);
            }
        }
    }
    for &power in pow3.iter().take(4) {
        for j in 0..4 {
            let q = u16::from(block[48 + j]) * power;
            let xi = ((u32::from(q) * 3) >> 8) as i32;
            output.push((xi - 1) as f32 * scale);
        }
    }
    output
}

fn dequantize_tq2(block: &[u8]) -> Vec<f32> {
    let scale = f16_to_f32(u16::from_le_bytes([block[64], block[65]]));
    let mut output = Vec::with_capacity(256);
    for start in [0usize, 32] {
        for shift in [0usize, 2, 4, 6] {
            for m in 0..32 {
                let q = (block[start + m] >> shift) & 3;
                output.push((i32::from(q) - 1) as f32 * scale);
            }
        }
    }
    output
}

fn e8m0_half(value: u8) -> f32 {
    if value < 2 {
        f32::from_bits(0x0020_0000u32 << value)
    } else {
        f32::from_bits(u32::from(value - 1) << 23)
    }
}

fn scalar_router_format(dtype: u32) -> Result<(u32, usize), Q4LoadError> {
    match dtype {
        0 => Ok((0, 4)),
        1 | 30 => Ok((dtype, 2)),
        other => Err(Q4LoadError::UnsupportedTensorType(other)),
    }
}

fn decode_scalar(encoded: &[u8], dtype: u32) -> f32 {
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

    fn test_block(format: LowBitFormat) -> Vec<u8> {
        let mut block = match format {
            LowBitFormat::Mxfp4 => vec![128u8],
            LowBitFormat::Tq1_0 => vec![0u8; 54],
            LowBitFormat::Tq2_0 => vec![0u8; 66],
            LowBitFormat::Q1_0 => vec![0u8; 18],
            LowBitFormat::Q2_0 => vec![0u8; 18],
        };
        match format {
            LowBitFormat::Mxfp4 => block.extend(std::iter::repeat_n(0x11, 16)),
            LowBitFormat::Tq1_0 => block[48..52].fill(0),
            LowBitFormat::Tq2_0 => block[..64].fill(0xaa),
            LowBitFormat::Q1_0 => block[2..].fill(0xff),
            LowBitFormat::Q2_0 => block[2..].fill(0xaa),
        }
        let scale_offset = match format {
            LowBitFormat::Tq1_0 => 52,
            LowBitFormat::Tq2_0 => 64,
            _ => 0,
        };
        if matches!(format, LowBitFormat::Tq1_0 | LowBitFormat::Tq2_0) {
            block[scale_offset..scale_offset + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        } else if !matches!(format, LowBitFormat::Mxfp4) {
            block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
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

    pub fn write_low_bit_source(path: &Path, format: LowBitFormat, split: bool) {
        let hidden = 256usize;
        let intermediate = 256usize;
        let experts = 2usize;
        let row = test_block(format).repeat(hidden / format.block_elements());
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
    fn low_bits_mxfp4_tq1_tq2_q1_q2_decode_vectors() {
        for format in [
            LowBitFormat::Mxfp4,
            LowBitFormat::Tq1_0,
            LowBitFormat::Tq2_0,
            LowBitFormat::Q1_0,
            LowBitFormat::Q2_0,
        ] {
            let decoded = dequantize_block(format, &test_block(format));
            assert_eq!(decoded.len(), format.block_elements());
            assert!(decoded.iter().all(|value| value.is_finite()));
            match format {
                LowBitFormat::Mxfp4 | LowBitFormat::Q1_0 | LowBitFormat::Q2_0 => {
                    assert!(decoded.iter().all(|value| (*value - 1.0).abs() < 1e-6));
                }
                LowBitFormat::Tq1_0 => assert!(decoded.iter().all(|value| *value == -1.0)),
                LowBitFormat::Tq2_0 => assert!(decoded.iter().all(|value| *value == 1.0)),
            }
        }
    }

    #[test]
    fn low_bit_source_and_executor_cover_fused_and_split() {
        for format in [
            LowBitFormat::Mxfp4,
            LowBitFormat::Tq1_0,
            LowBitFormat::Tq2_0,
            LowBitFormat::Q1_0,
            LowBitFormat::Q2_0,
        ] {
            for split in [false, true] {
                let path = std::env::temp_dir().join(format!(
                    "pom-ftk-low-bit-{}-{}-{}.gguf",
                    format.name(),
                    split,
                    std::process::id()
                ));
                write_low_bit_source(&path, format, split);
                let source = LowBitSource::open_infer(&path, 2).unwrap();
                let cache = ExpertCache::new(
                    MemoryAccount {
                        baseline_free_bytes: 100_000_000,
                        compute_bytes: 0,
                        weights_bytes: 0,
                        fixed_bytes: 0,
                        reserve_bytes: 0,
                    },
                    Arc::new(FixedProbe(AtomicU64::new(100_000_000))),
                );
                let executor =
                    LowBitExecutor::new(cache, format, source.hidden(), source.intermediate())
                        .unwrap();
                let output = executor
                    .run_from_source(
                        &source,
                        ExpertKey::new("fixture", 0, 0, format.name()),
                        0,
                        &[1.0; 256],
                    )
                    .unwrap();
                assert_eq!(output.len(), 256);
                assert!(output.iter().all(|value| value.is_finite()));
                std::fs::remove_file(path).unwrap();
            }
        }
    }
}
