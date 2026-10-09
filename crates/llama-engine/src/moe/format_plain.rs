//! Executor CPU para experts escalares F32, F16 e BF16.
//!
//! Esses formatos não têm blocos de quantização: cada expert é um banco de
//! escalares little-endian. A geometria GGUF ainda é validada na fronteira e
//! o payload cacheado usa o mesmo cabeçalho de comprimentos dos decoders
//! quantizados, para que um expert de outra forma não seja reinterpretado.

use super::cache::{CacheError, ExpertCache, ExpertKey};
use super::format_q4_0::{parse_tensor_table, Q4LoadError, TensorSpec};
use memmap2::Mmap;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const PAYLOAD_HEADER_BYTES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlainFormat {
    F32,
    F16,
    Bf16,
}

impl PlainFormat {
    pub fn from_dtype(dtype: u32) -> Result<Self, Q4LoadError> {
        match dtype {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            30 => Ok(Self::Bf16),
            other => Err(Q4LoadError::UnsupportedTensorType(other)),
        }
    }

    pub fn dtype(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 30,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    pub fn row_bytes(self, columns: usize) -> Result<usize, PlainError> {
        columns
            .checked_mul(self.element_bytes())
            .ok_or(PlainError::Geometry("overflow no tamanho da linha"))
    }

    pub fn decode_row(self, row: &[u8], columns: usize) -> Result<Vec<f32>, PlainError> {
        let expected = self.row_bytes(columns)?;
        if row.len() != expected {
            return Err(PlainError::PayloadLength {
                expected,
                actual: row.len(),
            });
        }
        Ok(row
            .chunks_exact(self.element_bytes())
            .map(|encoded| decode_scalar(encoded, self))
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlainError {
    Geometry(&'static str),
    Payload(&'static str),
    PayloadLength { expected: usize, actual: usize },
}

impl fmt::Display for PlainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(reason) => write!(formatter, "geometria escalar inválida: {reason}"),
            Self::Payload(reason) => write!(formatter, "payload escalar inválido: {reason}"),
            Self::PayloadLength { expected, actual } => write!(
                formatter,
                "payload escalar tem {actual} bytes, mas a geometria exige {expected}"
            ),
        }
    }
}

impl std::error::Error for PlainError {}

#[derive(Debug)]
pub enum PlainExecuteError {
    Cache(CacheError),
    Format(PlainError),
    Source(Q4LoadError),
    HiddenWidth { expected: usize, actual: usize },
}

impl fmt::Display for PlainExecuteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
            Self::Source(error) => error.fmt(formatter),
            Self::HiddenWidth { expected, actual } => write!(
                formatter,
                "hidden escalar tem {actual} elementos, esperado {expected}"
            ),
        }
    }
}

impl std::error::Error for PlainExecuteError {}

impl From<CacheError> for PlainExecuteError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

impl From<PlainError> for PlainExecuteError {
    fn from(error: PlainError) -> Self {
        Self::Format(error)
    }
}

impl From<Q4LoadError> for PlainExecuteError {
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

pub struct PlainSource {
    mmap: Mmap,
    data_start: usize,
    experts: u32,
    hidden: usize,
    intermediate: usize,
    format: PlainFormat,
    gate_up: GateUpTensors,
    down: Vec<TensorSpec>,
    router: Vec<TensorSpec>,
}

impl PlainSource {
    pub fn open_infer(path: impl AsRef<Path>, experts: u32) -> Result<Self, Q4LoadError> {
        if experts == 0 {
            return Err(Q4LoadError::InvalidTensor("contagem de experts zero"));
        }
        let file = File::open(path.as_ref()).map_err(|error| Q4LoadError::Io(error.to_string()))?;
        // SAFETY: Mmap owns the mapping and outlives every returned tensor slice.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|error| Q4LoadError::Io(error.to_string()))?;
        let (data_start, tensors) = parse_tensor_table(&mmap)?;
        let format = detect_plain_format(&tensors)?;
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
                    "gate_up escalar precisa ser rank 3 com o número de experts",
                ));
            }
            let hidden = dimension(first, 0)?;
            let two_intermediate = dimension(first, 1)?;
            if hidden == 0 || two_intermediate == 0 || !two_intermediate.is_multiple_of(2) {
                return Err(Q4LoadError::InvalidTensor(
                    "geometria gate_up escalar inválida",
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
                    "gate escalar precisa ser rank 3 com o número de experts",
                ));
            }
            let intermediate = dimension(first, 0)?;
            let hidden = dimension(first, 1)?;
            if intermediate == 0 || hidden == 0 {
                return Err(Q4LoadError::InvalidTensor("geometria gate escalar vazia"));
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
        for tensor in match &gate_up {
            GateUpTensors::Fused(items) => items,
            GateUpTensors::Split { gate, .. } => gate,
        } {
            if matches!(&gate_up, GateUpTensors::Fused(_)) {
                validate_matrix_tensor(tensor, experts, 2 * intermediate, hidden, format)?;
            } else {
                validate_matrix_tensor(tensor, experts, intermediate, hidden, format)?;
            }
        }
        let layers = layer_count(match &gate_up {
            GateUpTensors::Fused(items) => items,
            GateUpTensors::Split { gate, .. } => gate,
        });
        if layers == 0 {
            return Err(Q4LoadError::InvalidTensor("GGUF escalar sem camadas"));
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

    pub fn format(&self) -> PlainFormat {
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
                "hidden não corresponde à largura do router escalar",
            ));
        }
        let tensor = tensor_for_layer(&self.router, layer, "ffn_gate_inp.weight")?;
        if tensor.dims.len() != 2
            || tensor.dims[0] != self.hidden as u64
            || tensor.dims[1] != u64::from(self.experts)
        {
            return Err(Q4LoadError::InvalidTensor(
                "dimensões do router escalar não correspondem ao modelo",
            ));
        }
        let router_format = PlainFormat::from_dtype(tensor.dtype)?;
        let total = self
            .hidden
            .checked_mul(self.experts as usize)
            .and_then(|count| count.checked_mul(router_format.element_bytes()))
            .ok_or(Q4LoadError::InvalidTensor("tamanho do router excede usize"))?;
        let raw = self.tensor_bytes(tensor, total)?;
        let element_bytes = router_format.element_bytes();
        let mut logits = Vec::with_capacity(self.experts as usize);
        for expert in 0..self.experts as usize {
            let row = &raw
                [expert * self.hidden * element_bytes..(expert + 1) * self.hidden * element_bytes];
            logits.push(
                row.chunks_exact(element_bytes)
                    .zip(hidden)
                    .map(|(encoded, &input)| decode_scalar(encoded, router_format) * input)
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
        let bytes = self.format.element_bytes();
        let down = tensor_for_layer(&self.down, layer, "ffn_down_exps.weight")?;
        let down_per_expert = self
            .hidden
            .checked_mul(self.intermediate)
            .and_then(|elements| elements.checked_mul(bytes))
            .ok_or(Q4LoadError::InvalidTensor("overflow no down escalar"))?;
        let down = self.expert_slice(down, expert, down_per_expert)?.to_vec();
        let gate_up = match &self.gate_up {
            GateUpTensors::Fused(items) => {
                let tensor = tensor_for_layer(items, layer, "ffn_gate_up_exps.weight")?;
                let per_expert = 2usize
                    .checked_mul(self.intermediate)
                    .and_then(|elements| elements.checked_mul(self.hidden))
                    .and_then(|elements| elements.checked_mul(bytes))
                    .ok_or(Q4LoadError::InvalidTensor("overflow no gate_up escalar"))?;
                self.expert_slice(tensor, expert, per_expert)?.to_vec()
            }
            GateUpTensors::Split { gate, up } => {
                let gate = tensor_for_layer(gate, layer, "ffn_gate_exps.weight")?;
                let up = tensor_for_layer(up, layer, "ffn_up_exps.weight")?;
                let per_expert = self
                    .intermediate
                    .checked_mul(self.hidden)
                    .and_then(|elements| elements.checked_mul(bytes))
                    .ok_or(Q4LoadError::InvalidTensor("overflow no gate/up escalar"))?;
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
            .ok_or(Q4LoadError::InvalidTensor("overflow no tensor escalar"))?;
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

pub struct PlainExpert<'a> {
    format: PlainFormat,
    hidden: usize,
    intermediate: usize,
    gate_up: &'a [u8],
    down: &'a [u8],
}

impl<'a> PlainExpert<'a> {
    pub fn packed_bytes(
        format: PlainFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<usize, PlainError> {
        if hidden == 0 || intermediate == 0 {
            return Err(PlainError::Geometry("dimensão zero"));
        }
        let element_bytes = format.element_bytes();
        let gate_up = 2usize
            .checked_mul(intermediate)
            .and_then(|elements| elements.checked_mul(hidden))
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or(PlainError::Geometry("overflow em gate_up"))?;
        let down = hidden
            .checked_mul(intermediate)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or(PlainError::Geometry("overflow em down"))?;
        PAYLOAD_HEADER_BYTES
            .checked_add(gate_up)
            .and_then(|bytes| bytes.checked_add(down))
            .ok_or(PlainError::Geometry("overflow no payload"))
    }

    pub fn from_payload(
        format: PlainFormat,
        hidden: usize,
        intermediate: usize,
        payload: &'a [u8],
    ) -> Result<Self, PlainError> {
        let expected = Self::packed_bytes(format, hidden, intermediate)?;
        if payload.len() != expected {
            return Err(PlainError::PayloadLength {
                expected,
                actual: payload.len(),
            });
        }
        let gate_up_len = u32::from_le_bytes(
            payload[..4]
                .try_into()
                .map_err(|_| PlainError::Payload("cabeçalho gate_up ausente"))?,
        ) as usize;
        let down_len = u32::from_le_bytes(
            payload[4..8]
                .try_into()
                .map_err(|_| PlainError::Payload("cabeçalho down ausente"))?,
        ) as usize;
        let expected_gate_up = 2 * intermediate * hidden * format.element_bytes();
        let expected_down = hidden * intermediate * format.element_bytes();
        if gate_up_len != expected_gate_up || down_len != expected_down {
            return Err(PlainError::Payload(
                "tamanhos gate_up/down não correspondem à geometria",
            ));
        }
        let down_start = PAYLOAD_HEADER_BYTES + gate_up_len;
        if down_start + down_len != payload.len() {
            return Err(PlainError::Payload("seções não cobrem o payload"));
        }
        Ok(Self {
            format,
            hidden,
            intermediate,
            gate_up: &payload[PAYLOAD_HEADER_BYTES..down_start],
            down: &payload[down_start..],
        })
    }

    pub fn run(&self, hidden: &[f32]) -> Result<Vec<f32>, PlainExecuteError> {
        if hidden.len() != self.hidden {
            return Err(PlainExecuteError::HiddenWidth {
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

pub struct PlainExecutor {
    cache: Arc<ExpertCache>,
    format: PlainFormat,
    hidden: usize,
    intermediate: usize,
}

impl PlainExecutor {
    pub fn new(
        cache: Arc<ExpertCache>,
        format: PlainFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, PlainError> {
        PlainExpert::packed_bytes(format, hidden, intermediate)?;
        Ok(Self {
            cache,
            format,
            hidden,
            intermediate,
        })
    }

    pub fn run_from_source(
        &self,
        source: &PlainSource,
        key: ExpertKey,
        layer: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>, PlainExecuteError> {
        let expert = key.expert;
        self.run_with_loader(
            key,
            || {
                source
                    .load_expert(layer, expert)
                    .map_err(PlainExecuteError::Source)
            },
            hidden,
        )
    }

    fn run_with_loader<F>(
        &self,
        key: ExpertKey,
        load: F,
        hidden: &[f32],
    ) -> Result<Vec<f32>, PlainExecuteError>
    where
        F: FnOnce() -> Result<Vec<u8>, PlainExecuteError>,
    {
        let lease = match self.cache.lookup(&key) {
            Some(lease) => lease,
            None => self.cache.admit(key, load()?)?,
        };
        PlainExpert::from_payload(self.format, self.hidden, self.intermediate, lease.bytes())?
            .run(hidden)
    }
}

fn detect_plain_format(tensors: &[TensorSpec]) -> Result<PlainFormat, Q4LoadError> {
    let tensor = tensors
        .iter()
        .find(|tensor| {
            tensor.name.starts_with("blk.")
                && (tensor.name.ends_with("ffn_gate_up_exps.weight")
                    || tensor.name.ends_with("ffn_gate_exps.weight"))
        })
        .ok_or_else(|| Q4LoadError::MissingTensor("ffn_gate[_up]_exps.weight".into()))?;
    PlainFormat::from_dtype(tensor.dtype)
}

fn tensors_with_suffix(tensors: &[TensorSpec], suffix: &str) -> Vec<TensorSpec> {
    tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("blk.") && tensor.name.ends_with(suffix))
        .cloned()
        .collect()
}

fn validate_dtype(tensor: &TensorSpec, format: PlainFormat) -> Result<(), Q4LoadError> {
    if tensor.dtype != format.dtype() {
        return Err(Q4LoadError::UnsupportedTensorType(tensor.dtype));
    }
    Ok(())
}

fn validate_all_same_dtype(tensors: &[TensorSpec], format: PlainFormat) -> Result<(), Q4LoadError> {
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
    format: PlainFormat,
) -> Result<(), Q4LoadError> {
    validate_dtype(tensor, format)?;
    if tensor.dims.len() != 3
        || tensor.dims[0] != columns as u64
        || tensor.dims[1] != rows as u64
        || tensor.dims[2] != u64::from(experts)
    {
        return Err(Q4LoadError::InvalidTensor(
            "dimensões não correspondem ao expert escalar",
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
        return Err(Q4LoadError::InvalidTensor("payload escalar excede u32"));
    }
    let mut payload = Vec::with_capacity(PAYLOAD_HEADER_BYTES + gate_up.len() + down.len());
    payload.extend_from_slice(&(gate_up.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(down.len() as u32).to_le_bytes());
    payload.extend_from_slice(gate_up);
    payload.extend_from_slice(down);
    Ok(payload)
}

fn decode_scalar(encoded: &[u8], format: PlainFormat) -> f32 {
    match format {
        PlainFormat::F32 => f32::from_le_bytes(encoded.try_into().unwrap()),
        PlainFormat::F16 => f16_to_f32(u16::from_le_bytes(encoded.try_into().unwrap())),
        PlainFormat::Bf16 => {
            f32::from_bits(u32::from(u16::from_le_bytes(encoded.try_into().unwrap())) << 16)
        }
    }
}

fn mat_vec(
    format: PlainFormat,
    matrix: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Vec<f32> {
    let element_bytes = format.element_bytes();
    (0..rows)
        .map(|row| {
            let start = row * columns * element_bytes;
            matrix[start..start + columns * element_bytes]
                .chunks_exact(element_bytes)
                .zip(input)
                .map(|(encoded, &value)| decode_scalar(encoded, format) * value)
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
        31 if fraction == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
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

    impl FreeMemoryProbe for FixedProbe {
        fn free_bytes(&self) -> io::Result<u64> {
            Ok(self.0.load(Ordering::Relaxed))
        }
    }

    fn scalar_one(format: PlainFormat) -> Vec<u8> {
        match format {
            PlainFormat::F32 => 1.0f32.to_le_bytes().to_vec(),
            PlainFormat::F16 => 0x3c00u16.to_le_bytes().to_vec(),
            PlainFormat::Bf16 => (1.0f32.to_bits() >> 16).to_le_bytes().to_vec(),
        }
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

    pub fn write_plain_source(path: &Path, format: PlainFormat, split: bool) {
        let hidden = 2usize;
        let intermediate = 2usize;
        let experts = 2usize;
        let scalar = scalar_one(format);
        let gate = vec![scalar.clone(); experts * intermediate * hidden]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let up = gate.clone();
        let down = gate.clone();
        let mut router = Vec::new();
        for _ in 0..experts * hidden {
            router.extend_from_slice(&scalar);
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
            format.dtype(),
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
    fn plain_f32_f16_bf16_source_executes_real_payload_contract() {
        for format in [PlainFormat::F32, PlainFormat::F16, PlainFormat::Bf16] {
            for split in [false, true] {
                let path = std::env::temp_dir().join(format!(
                    "pom-ftk-plain-{}-{}-{}.gguf",
                    format.name(),
                    split,
                    std::process::id()
                ));
                write_plain_source(&path, format, split);
                let source = PlainSource::open_infer(&path, 2).unwrap();
                let cache = ExpertCache::new(
                    MemoryAccount {
                        baseline_free_bytes: 1_000_000,
                        compute_bytes: 0,
                        weights_bytes: 0,
                        fixed_bytes: 0,
                        reserve_bytes: 0,
                    },
                    Arc::new(FixedProbe(AtomicU64::new(1_000_000))),
                );
                let executor =
                    PlainExecutor::new(cache, format, source.hidden(), source.intermediate())
                        .unwrap();
                let output = executor
                    .run_from_source(
                        &source,
                        ExpertKey::new("fixture", 0, 0, format.name()),
                        0,
                        &[1.0, 1.0],
                    )
                    .unwrap();
                assert_eq!(output.len(), 2);
                assert!(output.iter().all(|value| value.is_finite() && *value > 0.0));
                std::fs::remove_file(path).unwrap();
            }
        }
    }

    #[test]
    fn plain_dispatch_refuses_a_quantized_dtype_by_name() {
        let error = PlainFormat::from_dtype(3).unwrap_err();
        assert!(error.to_string().contains("3 (Q4_1)"));
    }
}
