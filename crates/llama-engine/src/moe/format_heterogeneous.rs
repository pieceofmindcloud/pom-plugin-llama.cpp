//! Inventário e execução de experts com dtype por tensor.
//!
//! Um GGUF UD pode usar, por exemplo, IQ3_S no gate, IQ2_S no up e IQ4_XS no
//! down. Este módulo não escolhe um formato para o arquivo: cada tensor entra
//! no inventário com o seu dtype, recebe o decoder correspondente e ganha uma
//! entrada própria no cache. A convenção de nomes só identifica o papel GGUF
//! do tensor; arquitetura e quantização não são codificadas neste caminho.

use super::cache::{CacheError, ExpertCache, ExpertKey};
use super::format_iq_quants::{row_bytes as iq_row_bytes, IqQuantFormat};
use super::format_k_quants::{row_bytes as k_row_bytes, KQuantFormat};
use super::format_legacy_quants::{decode_row as legacy_decode_row, LegacyQuantFormat};
use super::format_low_bits::{decode_row as low_bit_decode_row, LowBitFormat};
use super::format_plain::PlainFormat;
use super::format_q4_0::{
    decode_row as q4_decode_row, read_scalar_vector, Q4LoadError, TensorSpec,
};
use super::router::dispatch_route_parallel;
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    GateUp,
    Gate,
    Up,
    Down,
}

impl TensorRole {
    fn suffix(self) -> &'static str {
        match self {
            Self::GateUp => "ffn_gate_up_exps.weight",
            Self::Gate => "ffn_gate_exps.weight",
            Self::Up => "ffn_up_exps.weight",
            Self::Down => "ffn_down_exps.weight",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::GateUp => "gate_up",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

impl fmt::Display for TensorRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorDecoder {
    Plain(PlainFormat),
    Q4_0,
    Legacy(LegacyQuantFormat),
    K(KQuantFormat),
    Iq(IqQuantFormat),
    LowBit(LowBitFormat),
}

impl TensorDecoder {
    fn from_tensor(tensor: &TensorSpec) -> Result<Self, Q4LoadError> {
        let decoder = match tensor.dtype {
            0 => Self::Plain(PlainFormat::F32),
            1 => Self::Plain(PlainFormat::F16),
            2 => Self::Q4_0,
            3 => Self::Legacy(LegacyQuantFormat::Q4_1),
            6 => Self::Legacy(LegacyQuantFormat::Q5_0),
            7 => Self::Legacy(LegacyQuantFormat::Q5_1),
            8 => Self::Legacy(LegacyQuantFormat::Q8_0),
            10 => Self::K(KQuantFormat::Q2K),
            11 => Self::K(KQuantFormat::Q3K),
            12 => Self::K(KQuantFormat::Q4K),
            13 => Self::K(KQuantFormat::Q5K),
            14 => Self::K(KQuantFormat::Q6K),
            15 => Self::K(KQuantFormat::Q8K),
            16 => Self::Iq(IqQuantFormat::Iq2XXS),
            17 => Self::Iq(IqQuantFormat::Iq2XS),
            18 => Self::Iq(IqQuantFormat::Iq3XXS),
            19 => Self::Iq(IqQuantFormat::Iq1S),
            20 => Self::Iq(IqQuantFormat::Iq4NL),
            21 => Self::Iq(IqQuantFormat::Iq3S),
            22 => Self::Iq(IqQuantFormat::Iq2S),
            23 => Self::Iq(IqQuantFormat::Iq4XS),
            29 => Self::Iq(IqQuantFormat::Iq1M),
            30 => Self::Plain(PlainFormat::Bf16),
            34 => Self::LowBit(LowBitFormat::Tq1_0),
            35 => Self::LowBit(LowBitFormat::Tq2_0),
            39 => Self::LowBit(LowBitFormat::Mxfp4),
            41 => Self::LowBit(LowBitFormat::Q1_0),
            42 => Self::LowBit(LowBitFormat::Q2_0),
            dtype => {
                return Err(Q4LoadError::UnsupportedTensor {
                    name: tensor.name.clone(),
                    dtype,
                })
            }
        };
        Ok(decoder)
    }

    fn row_bytes(self, columns: usize) -> Result<usize, Q4LoadError> {
        match self {
            Self::Plain(format) => format
                .row_bytes(columns)
                .map_err(|_| Q4LoadError::InvalidTensor("geometria escalar inválida")),
            Self::Q4_0 => super::format_q4_0::row_bytes(columns)
                .map_err(|_| Q4LoadError::InvalidTensor("geometria Q4_0 inválida")),
            Self::Legacy(format) => format
                .row_bytes(columns)
                .map_err(|_| Q4LoadError::InvalidTensor("geometria legacy inválida")),
            Self::K(format) => k_row_bytes(format, columns)
                .map_err(|_| Q4LoadError::InvalidTensor("geometria K-quant inválida")),
            Self::Iq(format) => iq_row_bytes(format, columns)
                .map_err(|_| Q4LoadError::InvalidTensor("geometria IQ-quant inválida")),
            Self::LowBit(format) => format
                .row_bytes(columns)
                .map_err(|_| Q4LoadError::InvalidTensor("geometria low-bit inválida")),
        }
    }

    fn decode_row(self, row: &[u8], columns: usize) -> Result<Vec<f32>, Q4LoadError> {
        match self {
            Self::Plain(format) => format
                .decode_row(row, columns)
                .map_err(|_| Q4LoadError::InvalidTensor("payload escalar inválido")),
            Self::Q4_0 => q4_decode_row(row, columns)
                .map_err(|_| Q4LoadError::InvalidTensor("payload Q4_0 inválido")),
            Self::Legacy(format) => legacy_decode_row(format, row, columns)
                .map_err(|_| Q4LoadError::InvalidTensor("payload legacy inválido")),
            Self::K(format) => super::format_k_quants::dequantize_row(format, row)
                .map_err(|_| Q4LoadError::InvalidTensor("payload K-quant inválido")),
            Self::Iq(format) => super::format_iq_quants::dequantize_row(format, row)
                .map_err(|_| Q4LoadError::InvalidTensor("payload IQ-quant inválido")),
            Self::LowBit(format) => low_bit_decode_row(format, row, columns)
                .map_err(|_| Q4LoadError::InvalidTensor("payload low-bit inválido")),
        }
    }

    fn dtype(self) -> u32 {
        match self {
            Self::Plain(format) => format.dtype(),
            Self::Q4_0 => 2,
            Self::Legacy(format) => format.dtype(),
            Self::K(format) => format.dtype(),
            Self::Iq(format) => format.dtype(),
            Self::LowBit(format) => format.dtype(),
        }
    }
}

/// Tamanho de uma linha no payload que a ponte CUDA heterogênea entende.
///
/// Q4_0 e MXFP4 usam blocos de 32 elementos, enquanto os K/IQ-quants que a
/// ponte aceita usam blocos de 256. Manter essa tabela junto do inventário
/// impede que o validador Rust confunda os tamanhos entre dtypes.
#[cfg(any(feature = "cuda", test))]
pub fn cuda_tensor_row_bytes(dtype: u32, columns: usize) -> Option<usize> {
    let (block_elements, block_bytes) = match dtype {
        2 => (32, 18),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        18 => (256, 98),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        39 => (32, 17),
        _ => return None,
    };
    (columns > 0 && columns.is_multiple_of(block_elements))
        .then(|| columns / block_elements * block_bytes)
}

#[cfg(any(feature = "cuda", test))]
pub fn cuda_tensor_expected_bytes(dtype: u32, rows: usize, columns: usize) -> Option<usize> {
    rows.checked_mul(cuda_tensor_row_bytes(dtype, columns)?)
}

#[cfg(test)]
pub fn cuda_tensor_bank_expected_bytes(
    dtype: u32,
    rows: usize,
    columns: usize,
    experts: usize,
) -> Option<usize> {
    cuda_tensor_expected_bytes(dtype, rows, columns)?.checked_mul(experts)
}

#[cfg(feature = "cuda")]
pub fn cuda_tensor_dtype_supported(dtype: u32) -> bool {
    matches!(dtype, 2 | 10 | 11 | 12 | 13 | 14 | 18 | 21 | 22 | 23 | 39)
}

#[derive(Debug, Clone, Copy)]
struct ExpertTensor {
    spec_index: usize,
    decoder: TensorDecoder,
    role: TensorRole,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorScales {
    pub primary: f32,
    pub secondary: f32,
    pub input_primary: f32,
    pub input_secondary: f32,
}

impl Default for TensorScales {
    fn default() -> Self {
        Self {
            primary: 1.0,
            secondary: 1.0,
            input_primary: 1.0,
            input_secondary: 1.0,
        }
    }
}

struct LayerInventory {
    gate_up: Option<ExpertTensor>,
    gate: Option<ExpertTensor>,
    up: Option<ExpertTensor>,
    down: Option<ExpertTensor>,
    hidden: usize,
    intermediate: usize,
}

/// Payload de um tensor de expert individual, pronto para uma ponte device.
///
/// O dtype e a geometria acompanham cada papel porque um GGUF UD pode trocar
/// o formato entre gate, up e down. O cache de device usa uma entrada por
/// papel e expert, nunca uma premissa de dtype único para o modelo.
#[cfg(feature = "cuda")]
pub struct HeterogeneousTensorPayload {
    pub role: TensorRole,
    pub dtype: u32,
    /// Geometria de uma única fatia de expert, sem a dimensão do banco.
    pub rows: usize,
    pub columns: usize,
    pub bytes: Vec<u8>,
    pub scales: TensorScales,
}

/// Metadados de um tensor de expert, sem materializar a fatia do mmap.
///
/// O caminho CUDA consulta o cache residente com esta especificação antes de
/// pedir os bytes. O campo interno mantém o descritor validado do tensor GGUF
/// para que uma admissão posterior não precise procurar a tabela novamente.
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy)]
pub struct HeterogeneousTensorSpec {
    pub role: TensorRole,
    pub dtype: u32,
    /// Geometria de uma única fatia de expert, sem a dimensão do banco.
    pub rows: usize,
    pub columns: usize,
    pub scales: TensorScales,
    expert: u32,
    entry: ExpertTensor,
}

pub struct HeterogeneousMoeSource {
    mmap: Mmap,
    data_start: usize,
    experts: u32,
    tensors: Vec<TensorSpec>,
    sidecars: BTreeMap<String, Vec<f32>>,
    layers: Vec<LayerInventory>,
}

impl HeterogeneousMoeSource {
    pub fn open_infer(path: impl AsRef<Path>, experts: u32) -> Result<Self, Q4LoadError> {
        if experts == 0 {
            return Err(Q4LoadError::InvalidTensor("contagem de experts zero"));
        }
        let (mmap, data_start, tensors) = super::format_q4_0::map_tensor_source(path.as_ref())?;
        let sidecars = load_sidecars(&mmap, data_start, &tensors)?;
        let mut max_layer = None;
        let mut expert_entries = Vec::new();
        for (index, tensor) in tensors.iter().enumerate() {
            let Some(role) = role_for_name(&tensor.name) else {
                continue;
            };
            let layer = layer_for_name(&tensor.name)
                .ok_or(Q4LoadError::InvalidTensor("tensor de expert sem camada"))?;
            let decoder = TensorDecoder::from_tensor(tensor)?;
            max_layer = Some(max_layer.map_or(layer, |last: u32| last.max(layer)));
            expert_entries.push((
                layer,
                ExpertTensor {
                    spec_index: index,
                    decoder,
                    role,
                },
            ));
        }
        let layer_count = max_layer
            .map(|last| last.saturating_add(1) as usize)
            .ok_or_else(|| Q4LoadError::MissingTensor("blk.<layer>.ffn_*_exps.weight".into()))?;
        let mut layers: Vec<Option<LayerInventory>> = (0..layer_count).map(|_| None).collect();
        for layer in 0..layer_count as u32 {
            let find = |role: TensorRole| {
                expert_entries
                    .iter()
                    .find(|(entry_layer, entry)| *entry_layer == layer && entry.role == role)
                    .map(|(_, entry)| *entry)
            };
            let gate_up = find(TensorRole::GateUp);
            let gate = find(TensorRole::Gate);
            let up = find(TensorRole::Up);
            let down = find(TensorRole::Down);
            let (hidden, intermediate) = if let Some(entry) = gate_up {
                let tensor = &tensors[entry.spec_index];
                if tensor.dims.len() != 3
                    || tensor.dims[2] != u64::from(experts)
                    || tensor.dims[1] % 2 != 0
                {
                    return Err(Q4LoadError::InvalidTensor(
                        "geometria gate_up heterogênea inválida",
                    ));
                }
                (dimension(tensor, 0)?, dimension(tensor, 1)? / 2)
            } else {
                let gate = gate.ok_or_else(|| {
                    Q4LoadError::MissingTensor(format!("blk.{layer}.ffn_gate_exps.weight"))
                })?;
                let up = up.ok_or_else(|| {
                    Q4LoadError::MissingTensor(format!("blk.{layer}.ffn_up_exps.weight"))
                })?;
                let gate_tensor = &tensors[gate.spec_index];
                let up_tensor = &tensors[up.spec_index];
                if gate_tensor.dims != up_tensor.dims {
                    return Err(Q4LoadError::InvalidTensor(
                        "gate e up têm geometrias diferentes",
                    ));
                }
                // GGML matrices use ne[0] as the input width and ne[1] as
                // the output width. The down tensor below cross-checks this
                // geometry before any payload is decoded.
                (dimension(gate_tensor, 0)?, dimension(gate_tensor, 1)?)
            };
            let down = down.ok_or_else(|| {
                Q4LoadError::MissingTensor(format!("blk.{layer}.ffn_down_exps.weight"))
            })?;
            let down_tensor = &tensors[down.spec_index];
            validate_matrix(down_tensor, experts, hidden, intermediate, down.decoder)?;
            if let Some(entry) = gate_up {
                validate_matrix(
                    &tensors[entry.spec_index],
                    experts,
                    2 * intermediate,
                    hidden,
                    entry.decoder,
                )?;
            } else {
                let gate = gate.ok_or(Q4LoadError::InvalidTensor("gate ausente"))?;
                let up = up.ok_or(Q4LoadError::InvalidTensor("up ausente"))?;
                validate_matrix(
                    &tensors[gate.spec_index],
                    experts,
                    intermediate,
                    hidden,
                    gate.decoder,
                )?;
                validate_matrix(
                    &tensors[up.spec_index],
                    experts,
                    intermediate,
                    hidden,
                    up.decoder,
                )?;
            }
            layers[layer as usize] = Some(LayerInventory {
                gate_up,
                gate,
                up,
                down: Some(down),
                hidden,
                intermediate,
            });
        }
        let layers = layers
            .into_iter()
            .enumerate()
            .map(|(layer, inventory)| {
                inventory.ok_or_else(|| {
                    Q4LoadError::MissingTensor(format!("camada MoE {layer} sem experts"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            mmap,
            data_start,
            experts,
            tensors,
            sidecars,
            layers,
        })
    }

    pub fn hidden(&self) -> usize {
        self.layers[0].hidden
    }

    pub fn experts(&self) -> u32 {
        self.experts
    }

    pub fn layer_count(&self) -> u32 {
        self.layers.len() as u32
    }

    pub fn dtypes(&self) -> Vec<u32> {
        let mut dtypes = self
            .layers
            .iter()
            .flat_map(|layer| {
                let gate_up = layer.gate_up;
                let gate = if gate_up.is_none() { layer.gate } else { None };
                let up = if gate_up.is_none() { layer.up } else { None };
                [gate_up, gate, up, layer.down]
                    .into_iter()
                    .flatten()
                    .map(|tensor| tensor.decoder.dtype())
            })
            .collect::<Vec<_>>();
        dtypes.sort_unstable();
        dtypes.dedup();
        dtypes
    }

    pub fn layer_geometry(&self, layer: u32) -> Result<(usize, usize), Q4LoadError> {
        let inventory = self
            .layers
            .get(layer as usize)
            .ok_or(Q4LoadError::InvalidTensor(
                "camada MoE fora do inventário heterogêneo",
            ))?;
        Ok((inventory.hidden, inventory.intermediate))
    }

    pub fn validate_metadata(
        &self,
        layer: u32,
        metadata: &[(TensorRole, &str, u32, u32, [i64; 4])],
    ) -> Result<(), Q4LoadError> {
        let inventory = self
            .layers
            .get(layer as usize)
            .ok_or(Q4LoadError::InvalidTensor(
                "camada MoE fora do inventário heterogêneo",
            ))?;
        for (role, name, dtype, n_dims, dims) in metadata {
            let expected = self.entry_for_role(inventory, *role).ok_or_else(|| {
                Q4LoadError::MissingTensor(format!("tensor {role} ausente no inventário"))
            })?;
            let tensor = &self.tensors[expected.spec_index];
            let actual_dims = usize::try_from(*n_dims)
                .ok()
                .filter(|count| *count <= dims.len())
                .map_or_else(Vec::new, |count| dims[..count].to_vec());
            let geometry_matches = actual_dims.len() == tensor.dims.len()
                && actual_dims
                    .iter()
                    .zip(&tensor.dims)
                    .all(|(actual, expected)| u64::try_from(*actual) == Ok(*expected));
            if tensor.name != *name || expected.decoder.dtype() != *dtype {
                return Err(Q4LoadError::TensorMetadataMismatch {
                    name: (*name).to_string(),
                    dtype: *dtype,
                    expected_name: tensor.name.clone(),
                    expected_dtype: expected.decoder.dtype(),
                });
            }
            if !geometry_matches {
                return Err(Q4LoadError::TensorGeometryMismatch {
                    name: (*name).to_string(),
                    dtype: *dtype,
                    dims: actual_dims,
                    expected_dims: tensor.dims.clone(),
                });
            }
        }
        Ok(())
    }

    #[cfg(any(feature = "cuda", test))]
    pub fn expert_tensor_specs(
        &self,
        layer: u32,
        expert: u32,
    ) -> Result<Vec<HeterogeneousTensorSpec>, Q4LoadError> {
        self.expert_tensor_specs_batch(layer, std::slice::from_ref(&expert))?
            .into_iter()
            .next()
            .ok_or(Q4LoadError::InvalidTensor(
                "lote de specs heterogêneas vazio",
            ))
    }

    /// Monta as especificações de vários experts da mesma camada em uma só
    /// travessia do inventário. O callback normalmente recebe vários experts
    /// por lote; repetir `expert_entries(layer)` e as buscas de geometria para
    /// cada um fazia essa preparação crescer desnecessariamente no prefill.
    #[cfg(any(feature = "cuda", test))]
    pub fn expert_tensor_specs_batch(
        &self,
        layer: u32,
        experts: &[u32],
    ) -> Result<Vec<Vec<HeterogeneousTensorSpec>>, Q4LoadError> {
        let inventory = self
            .layers
            .get(layer as usize)
            .ok_or(Q4LoadError::InvalidTensor(
                "camada MoE fora do inventário heterogêneo",
            ))?;
        let entries = self.expert_entries(layer)?;
        experts
            .iter()
            .copied()
            .map(|expert| {
                if expert >= self.experts {
                    return Err(Q4LoadError::ExpertOutOfRange {
                        expert,
                        count: self.experts,
                    });
                }
                entries
                    .iter()
                    .map(|&(role, entry, rows, columns)| {
                        let scales = if role == TensorRole::GateUp {
                            TensorScales {
                                primary: self.sidecar_scale(
                                    entry,
                                    "scale",
                                    expert,
                                    0,
                                    inventory.gate,
                                )?,
                                secondary: self.sidecar_scale(
                                    entry,
                                    "scale",
                                    expert,
                                    1,
                                    inventory.up,
                                )?,
                                input_primary: self.sidecar_scale(
                                    entry,
                                    "input_scale",
                                    expert,
                                    0,
                                    inventory.gate,
                                )?,
                                input_secondary: self.sidecar_scale(
                                    entry,
                                    "input_scale",
                                    expert,
                                    1,
                                    inventory.up,
                                )?,
                            }
                        } else {
                            TensorScales {
                                primary: self.sidecar_scale(entry, "scale", expert, 0, None)?,
                                secondary: 1.0,
                                input_primary: self.sidecar_scale(
                                    entry,
                                    "input_scale",
                                    expert,
                                    0,
                                    None,
                                )?,
                                input_secondary: 1.0,
                            }
                        };
                        Ok(HeterogeneousTensorSpec {
                            role,
                            dtype: entry.decoder.dtype(),
                            rows,
                            columns,
                            scales,
                            expert,
                            entry,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    #[cfg(any(feature = "cuda", test))]
    pub fn load_expert_tensor(
        &self,
        spec: HeterogeneousTensorSpec,
    ) -> Result<Vec<u8>, Q4LoadError> {
        self.expert_tensor_slice(spec).map(ToOwned::to_owned)
    }

    /// Retorna diretamente a fatia do mmap para o upload CUDA.
    ///
    /// O allocator CUDA copia o payload de forma síncrona antes de retornar.
    /// Portanto o caminho quente não precisa materializar uma segunda cópia
    /// de cada tensor em `Vec`; a fatia continua válida enquanto a fonte
    /// estiver viva, que é exatamente a duração da admissão.
    #[cfg(any(feature = "cuda", test))]
    pub fn expert_tensor_slice(&self, spec: HeterogeneousTensorSpec) -> Result<&[u8], Q4LoadError> {
        self.tensor_slice(spec.entry, spec.expert, spec.rows, spec.columns)
    }

    #[cfg(feature = "cuda")]
    pub fn load_expert_tensors(
        &self,
        layer: u32,
        expert: u32,
    ) -> Result<Vec<HeterogeneousTensorPayload>, Q4LoadError> {
        self.expert_tensor_specs(layer, expert)?
            .into_iter()
            .map(|spec| {
                Ok(HeterogeneousTensorPayload {
                    role: spec.role,
                    dtype: spec.dtype,
                    rows: spec.rows,
                    columns: spec.columns,
                    bytes: self.load_expert_tensor(spec)?,
                    scales: spec.scales,
                })
            })
            .collect()
    }

    #[cfg(any(feature = "cuda", test))]
    fn expert_entries(
        &self,
        layer: u32,
    ) -> Result<Vec<(TensorRole, ExpertTensor, usize, usize)>, Q4LoadError> {
        let inventory = self
            .layers
            .get(layer as usize)
            .ok_or(Q4LoadError::InvalidTensor(
                "camada MoE fora do inventário heterogêneo",
            ))?;
        let entries = if inventory.gate_up.is_some() {
            vec![
                (
                    TensorRole::GateUp,
                    inventory.gate_up,
                    2 * inventory.intermediate,
                    inventory.hidden,
                ),
                (
                    TensorRole::Down,
                    inventory.down,
                    inventory.hidden,
                    inventory.intermediate,
                ),
            ]
        } else {
            vec![
                (
                    TensorRole::Gate,
                    inventory.gate,
                    inventory.intermediate,
                    inventory.hidden,
                ),
                (
                    TensorRole::Up,
                    inventory.up,
                    inventory.intermediate,
                    inventory.hidden,
                ),
                (
                    TensorRole::Down,
                    inventory.down,
                    inventory.hidden,
                    inventory.intermediate,
                ),
            ]
        };
        Ok(entries
            .into_iter()
            .filter_map(|(role, entry, rows, columns)| {
                entry.map(|entry| (role, entry, rows, columns))
            })
            .collect())
    }

    fn entry_for_role(&self, inventory: &LayerInventory, role: TensorRole) -> Option<ExpertTensor> {
        match role {
            TensorRole::GateUp => inventory.gate_up,
            TensorRole::Gate => inventory.gate,
            TensorRole::Up => inventory.up,
            TensorRole::Down => inventory.down,
        }
    }

    fn tensor_slice(
        &self,
        entry: ExpertTensor,
        expert: u32,
        rows: usize,
        columns: usize,
    ) -> Result<&[u8], Q4LoadError> {
        if expert >= self.experts {
            return Err(Q4LoadError::ExpertOutOfRange {
                expert,
                count: self.experts,
            });
        }
        let tensor = &self.tensors[entry.spec_index];
        let row_bytes = entry.decoder.row_bytes(columns)?;
        let bytes_per_expert = rows
            .checked_mul(row_bytes)
            .ok_or(Q4LoadError::InvalidTensor("overflow no tensor heterogêneo"))?;
        let total = bytes_per_expert
            .checked_mul(self.experts as usize)
            .ok_or(Q4LoadError::InvalidTensor("overflow no banco de experts"))?;
        let start = usize::try_from(tensor.offset)
            .ok()
            .and_then(|offset| self.data_start.checked_add(offset))
            .and_then(|offset| offset.checked_add(bytes_per_expert * expert as usize))
            .ok_or(Q4LoadError::InvalidTensor("offset GGUF inválido"))?;
        let end = start
            .checked_add(bytes_per_expert)
            .ok_or(Q4LoadError::InvalidTensor("fim do tensor excede usize"))?;
        let tensor_end = start
            .checked_sub(bytes_per_expert * expert as usize)
            .and_then(|begin| begin.checked_add(total))
            .ok_or(Q4LoadError::InvalidTensor("tamanho do tensor excede usize"))?;
        if tensor_end > self.mmap.len() || end > tensor_end {
            return Err(Q4LoadError::InvalidTensor(
                "tensor de expert excede o arquivo",
            ));
        }
        Ok(&self.mmap[start..end])
    }

    fn sidecar_scale(
        &self,
        tensor: ExpertTensor,
        suffix: &str,
        expert: u32,
        component: usize,
        fallback: Option<ExpertTensor>,
    ) -> Result<f32, Q4LoadError> {
        if let Some(value) = self.sidecar_value(tensor, suffix, expert, component)? {
            return Ok(value);
        }
        if let Some(entry) = fallback {
            if let Some(value) = self.sidecar_value(entry, suffix, expert, 0)? {
                return Ok(value);
            }
        }
        Ok(1.0)
    }

    fn sidecar_value(
        &self,
        entry: ExpertTensor,
        suffix: &str,
        expert: u32,
        component: usize,
    ) -> Result<Option<f32>, Q4LoadError> {
        let tensor = &self.tensors[entry.spec_index];
        let stem = tensor.name.strip_suffix(".weight").unwrap_or(&tensor.name);
        let name = format!("{stem}.{suffix}");
        let Some(values) = self.sidecars.get(&name) else {
            return Ok(None);
        };
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

    fn packed_tensor(
        &self,
        cache: &Arc<ExpertCache>,
        model: &str,
        layer: u32,
        expert: u32,
        entry: ExpertTensor,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<u8>, String> {
        let tensor = &self.tensors[entry.spec_index];
        let key = ExpertKey::new(
            model,
            layer,
            expert,
            format!("role={};dtype={}", entry.role, entry.decoder.dtype()),
        );
        let lease = match cache.lookup(&key) {
            Some(lease) => lease,
            None => cache
                .admit(
                    key,
                    self.tensor_slice(entry, expert, rows, columns)
                        .map_err(|e| format!("tensor {}: {e}", tensor.name))?
                        .to_vec(),
                )
                .map_err(|error: CacheError| error.to_string())?,
        };
        Ok(lease.bytes().to_vec())
    }

    fn decode_payload_row(
        &self,
        entry: ExpertTensor,
        payload: &[u8],
        row: usize,
        columns: usize,
    ) -> Result<Vec<f32>, String> {
        let tensor = &self.tensors[entry.spec_index];
        let row_bytes = entry
            .decoder
            .row_bytes(columns)
            .map_err(|error| error.to_string())?;
        let start = row
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("tensor {}: offset de linha excede usize", tensor.name))?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| format!("tensor {}: fim de linha excede usize", tensor.name))?;
        let encoded = payload.get(start..end).ok_or_else(|| {
            format!(
                "tensor {}: linha {row} exige bytes [{start}, {end}), payload tem {}",
                tensor.name,
                payload.len()
            )
        })?;
        entry
            .decoder
            .decode_row(encoded, columns)
            .map_err(|error| format!("tensor {}: {error}", tensor.name))
    }

    fn matrix(
        &self,
        cache: &Arc<ExpertCache>,
        model: &str,
        layer: u32,
        expert: u32,
        entry: ExpertTensor,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, String> {
        let payload = self.packed_tensor(cache, model, layer, expert, entry, rows, columns)?;
        (0..rows)
            .map(|row| self.decode_payload_row(entry, &payload, row, columns))
            .collect::<Result<Vec<_>, _>>()
            .map(|rows| rows.into_iter().flatten().collect())
    }

    fn accumulate_matrix_rows(
        &self,
        entry: ExpertTensor,
        payload: &[u8],
        first_row: usize,
        rows: usize,
        columns: usize,
        routes: &[(usize, f32)],
        hidden: &[f32],
        hidden_width: usize,
        values: &mut [f32],
        scale: f32,
    ) -> Result<(), String> {
        for row in 0..rows {
            let decoded = self.decode_payload_row(entry, payload, first_row + row, columns)?;
            for (route_index, &(token, _)) in routes.iter().enumerate() {
                let token_start = token
                    .checked_mul(hidden_width)
                    .ok_or_else(|| "índice de token excede usize".to_string())?;
                let token_hidden = hidden
                    .get(token_start..token_start + hidden_width)
                    .ok_or_else(|| "hidden não contém o token da rota heterogênea".to_string())?;
                values[route_index * rows + row] = dot(&decoded, token_hidden) * scale;
            }
        }
        Ok(())
    }

    fn run_expert_batch(
        &self,
        cache: &Arc<ExpertCache>,
        model: &str,
        layer: u32,
        expert: u32,
        routes: &[(usize, f32)],
        hidden: &[f32],
        output_len: usize,
        activation: i32,
    ) -> Result<Vec<f32>, String> {
        let inventory = self
            .layers
            .get(layer as usize)
            .ok_or_else(|| "camada MoE fora do inventário heterogêneo".to_string())?;
        let hidden_width = inventory.hidden;
        let intermediate = inventory.intermediate;
        let mut output = vec![0.0; output_len];
        let route_values = routes
            .len()
            .checked_mul(intermediate)
            .ok_or_else(|| "shape da rota heterogênea excede usize".to_string())?;
        let mut gate = vec![0.0; route_values];
        let mut up = vec![0.0; route_values];
        if let Some(entry) = inventory.gate_up {
            let payload = self.packed_tensor(
                cache,
                model,
                layer,
                expert,
                entry,
                2 * intermediate,
                hidden_width,
            )?;
            let scales = TensorScales {
                primary: self
                    .sidecar_scale(entry, "scale", expert, 0, inventory.gate)
                    .map_err(|error| error.to_string())?,
                secondary: self
                    .sidecar_scale(entry, "scale", expert, 1, inventory.up)
                    .map_err(|error| error.to_string())?,
                input_primary: self
                    .sidecar_scale(entry, "input_scale", expert, 0, inventory.gate)
                    .map_err(|error| error.to_string())?,
                input_secondary: self
                    .sidecar_scale(entry, "input_scale", expert, 1, inventory.up)
                    .map_err(|error| error.to_string())?,
            };
            self.accumulate_matrix_rows(
                entry,
                &payload,
                0,
                intermediate,
                hidden_width,
                routes,
                hidden,
                hidden_width,
                &mut gate,
                scales.primary * scales.input_primary,
            )?;
            self.accumulate_matrix_rows(
                entry,
                &payload,
                intermediate,
                intermediate,
                hidden_width,
                routes,
                hidden,
                hidden_width,
                &mut up,
                scales.secondary * scales.input_secondary,
            )?;
        } else {
            let gate_entry = inventory.gate.ok_or("gate ausente")?;
            let up_entry = inventory.up.ok_or("up ausente")?;
            let gate_payload = self.packed_tensor(
                cache,
                model,
                layer,
                expert,
                gate_entry,
                intermediate,
                hidden_width,
            )?;
            let up_payload = self.packed_tensor(
                cache,
                model,
                layer,
                expert,
                up_entry,
                intermediate,
                hidden_width,
            )?;
            let gate_scale = self
                .sidecar_scale(gate_entry, "scale", expert, 0, None)
                .map_err(|error| error.to_string())?;
            let up_scale = self
                .sidecar_scale(up_entry, "scale", expert, 0, None)
                .map_err(|error| error.to_string())?;
            let gate_input_scale = self
                .sidecar_scale(gate_entry, "input_scale", expert, 0, None)
                .map_err(|error| error.to_string())?;
            let up_input_scale = self
                .sidecar_scale(up_entry, "input_scale", expert, 0, None)
                .map_err(|error| error.to_string())?;
            self.accumulate_matrix_rows(
                gate_entry,
                &gate_payload,
                0,
                intermediate,
                hidden_width,
                routes,
                hidden,
                hidden_width,
                &mut gate,
                gate_scale * gate_input_scale,
            )?;
            self.accumulate_matrix_rows(
                up_entry,
                &up_payload,
                0,
                intermediate,
                hidden_width,
                routes,
                hidden,
                hidden_width,
                &mut up,
                up_scale * up_input_scale,
            )?;
        }
        for route_index in 0..routes.len() {
            for row in 0..intermediate {
                gate[route_index * intermediate + row] = match activation {
                    1 => {
                        silu(gate[route_index * intermediate + row])
                            * up[route_index * intermediate + row]
                    }
                    2 => {
                        gelu(gate[route_index * intermediate + row])
                            * up[route_index * intermediate + row]
                    }
                    other => return Err(format!("ativação MoE não suportada: {other}")),
                };
            }
        }
        let down = inventory.down.ok_or("down ausente")?;
        let down_payload = self.packed_tensor(
            cache,
            model,
            layer,
            expert,
            down,
            hidden_width,
            intermediate,
        )?;
        let down_scale = self
            .sidecar_scale(down, "scale", expert, 0, None)
            .map_err(|error| error.to_string())?;
        let down_input_scale = self
            .sidecar_scale(down, "input_scale", expert, 0, None)
            .map_err(|error| error.to_string())?;
        for row in 0..hidden_width {
            let decoded = self.decode_payload_row(down, &down_payload, row, intermediate)?;
            for (route_index, &(token, weight)) in routes.iter().enumerate() {
                let gated = &gate[route_index * intermediate..(route_index + 1) * intermediate];
                let output_index = token
                    .checked_mul(hidden_width)
                    .and_then(|index| index.checked_add(row))
                    .ok_or_else(|| "índice de saída excede usize".to_string())?;
                output[output_index] +=
                    weight * dot(&decoded, gated) * down_scale * down_input_scale;
            }
        }
        Ok(output)
    }

    pub fn run_expert(
        &self,
        cache: &Arc<ExpertCache>,
        model: &str,
        layer: u32,
        expert: u32,
        hidden: &[f32],
        activation: i32,
    ) -> Result<Vec<f32>, String> {
        let inventory = self
            .layers
            .get(layer as usize)
            .ok_or_else(|| "camada MoE fora do inventário heterogêneo".to_string())?;
        let (hidden_width, intermediate) = (inventory.hidden, inventory.intermediate);
        if hidden.len() != hidden_width {
            return Err(format!(
                "hidden heterogêneo tem {}, esperado {}",
                hidden.len(),
                hidden_width
            ));
        }
        let (gate, up): (Vec<f32>, Vec<f32>) = if let Some(entry) = inventory.gate_up {
            let values = self.matrix(
                cache,
                model,
                layer,
                expert,
                entry,
                2 * intermediate,
                hidden_width,
            )?;
            let scales = TensorScales {
                primary: self
                    .sidecar_scale(entry, "scale", expert, 0, inventory.gate)
                    .map_err(|error| error.to_string())?,
                secondary: self
                    .sidecar_scale(entry, "scale", expert, 1, inventory.up)
                    .map_err(|error| error.to_string())?,
                input_primary: self
                    .sidecar_scale(entry, "input_scale", expert, 0, inventory.gate)
                    .map_err(|error| error.to_string())?,
                input_secondary: self
                    .sidecar_scale(entry, "input_scale", expert, 1, inventory.up)
                    .map_err(|error| error.to_string())?,
            };
            (
                values[..intermediate]
                    .iter()
                    .map(|value| value * scales.primary * scales.input_primary)
                    .collect(),
                values[intermediate..]
                    .iter()
                    .map(|value| value * scales.secondary * scales.input_secondary)
                    .collect(),
            )
        } else {
            let gate_entry = inventory.gate.ok_or("gate ausente")?;
            let up_entry = inventory.up.ok_or("up ausente")?;
            let gate_scale = self
                .sidecar_scale(gate_entry, "scale", expert, 0, None)
                .map_err(|error| error.to_string())?;
            let up_scale = self
                .sidecar_scale(up_entry, "scale", expert, 0, None)
                .map_err(|error| error.to_string())?;
            let gate_input_scale = self
                .sidecar_scale(gate_entry, "input_scale", expert, 0, None)
                .map_err(|error| error.to_string())?;
            let up_input_scale = self
                .sidecar_scale(up_entry, "input_scale", expert, 0, None)
                .map_err(|error| error.to_string())?;
            (
                mat_vec(
                    &self.matrix(
                        cache,
                        model,
                        layer,
                        expert,
                        gate_entry,
                        intermediate,
                        hidden_width,
                    )?,
                    intermediate,
                    hidden_width,
                    hidden,
                )
                .into_iter()
                .map(|value| value * gate_scale * gate_input_scale)
                .collect(),
                mat_vec(
                    &self.matrix(
                        cache,
                        model,
                        layer,
                        expert,
                        up_entry,
                        intermediate,
                        hidden_width,
                    )?,
                    intermediate,
                    hidden_width,
                    hidden,
                )
                .into_iter()
                .map(|value| value * up_scale * up_input_scale)
                .collect(),
            )
        };
        let gated = gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| {
                Ok(match activation {
                    1 => silu(gate) * up,
                    2 => gelu(gate) * up,
                    other => return Err(format!("ativação MoE não suportada: {other}")),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let down = inventory.down.ok_or("down ausente")?;
        let down_matrix = self.matrix(
            cache,
            model,
            layer,
            expert,
            down,
            hidden_width,
            intermediate,
        )?;
        let down_scale = self
            .sidecar_scale(down, "scale", expert, 0, None)
            .map_err(|error| error.to_string())?;
        let down_input_scale = self
            .sidecar_scale(down, "input_scale", expert, 0, None)
            .map_err(|error| error.to_string())?;
        Ok(mat_vec(&down_matrix, hidden_width, intermediate, &gated)
            .into_iter()
            .map(|value| value * down_scale * down_input_scale)
            .collect())
    }

    pub fn run_from_source_with_route(
        &self,
        cache: &Arc<ExpertCache>,
        model: &str,
        layer: u32,
        hidden: &[f32],
        ids: &[u32],
        weights: &[f32],
        activation: i32,
    ) -> Result<Vec<f32>, String> {
        if ids.is_empty() || ids.len() != weights.len() {
            return Err("rota heterogênea sem ids/pesos correspondentes".into());
        }
        dispatch_route_parallel(ids, weights, |expert| {
            self.run_expert(cache, model, layer, expert, hidden, activation)
        })
        .map_err(|error| error.to_string())
    }

    pub fn run_route_batch_from_source(
        &self,
        cache: &Arc<ExpertCache>,
        model: &str,
        layer: u32,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
        activation: i32,
    ) -> Result<Vec<f32>, String> {
        let (hidden_width, _) = self
            .layer_geometry(layer)
            .map_err(|error| error.to_string())?;
        let expected_hidden = tokens
            .checked_mul(hidden_width)
            .ok_or_else(|| "shape hidden heterogêneo excede usize".to_string())?;
        if tokens == 0 || hidden.len() != expected_hidden {
            return Err(format!(
                "lote heterogêneo recebeu hidden com {} floats, esperado {} × {}",
                hidden.len(),
                tokens,
                hidden_width
            ));
        }
        if experts.is_empty() || experts.len() != token_ids.len() || experts.len() != weights.len()
        {
            return Err(format!(
                "lote heterogêneo tem rotas incompatíveis: experts={}, token_ids={}, weights={}",
                experts.len(),
                token_ids.len(),
                weights.len()
            ));
        }
        if token_offsets.len() != tokens.checked_add(1).unwrap_or(usize::MAX)
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&experts.len())
            || token_offsets.windows(2).any(|window| window[0] > window[1])
        {
            return Err(format!(
                "lote heterogêneo tem offsets inválidos: {} offsets para {} tokens e {} rotas",
                token_offsets.len(),
                tokens,
                experts.len()
            ));
        }
        if !matches!(activation, 1 | 2) {
            return Err(format!("ativação MoE não suportada: {activation}"));
        }

        let mut by_expert = BTreeMap::<u32, Vec<(usize, f32)>>::new();
        for token in 0..tokens {
            let start = token_offsets[token];
            let end = token_offsets[token + 1];
            for index in start..end {
                if token_ids[index] != token as u32 {
                    return Err(format!(
                        "rota heterogênea fora do formato token-major: índice {index} declara token {}, esperado {token}",
                        token_ids[index]
                    ));
                }
                let expert = experts[index];
                if expert >= self.experts {
                    return Err(format!(
                        "id de expert {} fora de [0, {}) na rota heterogênea",
                        expert, self.experts
                    ));
                }
                let weight = weights[index];
                if !weight.is_finite() {
                    return Err(format!(
                        "peso de rota heterogênea não finito no token {token}, slot {}",
                        index - start
                    ));
                }
                by_expert.entry(expert).or_default().push((token, weight));
            }
        }
        let results = std::thread::scope(|scope| {
            by_expert
                .into_iter()
                .map(|(expert, routes)| {
                    scope.spawn(move || {
                        self.run_expert_batch(
                            cache,
                            model,
                            layer,
                            expert,
                            &routes,
                            hidden,
                            expected_hidden,
                            activation,
                        )
                        .map(|output| (expert, output))
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .map_err(|_| "thread de execução do expert entrou em panic".to_string())?
                })
                .collect::<Result<BTreeMap<_, _>, String>>()
        })?;
        let mut output = vec![0.0; expected_hidden];
        for expert_output in results.into_values() {
            for (out, value) in output.iter_mut().zip(expert_output) {
                *out += value;
            }
        }
        Ok(output)
    }
}

fn load_sidecars(
    mmap: &Mmap,
    data_start: usize,
    tensors: &[TensorSpec],
) -> Result<BTreeMap<String, Vec<f32>>, Q4LoadError> {
    tensors
        .iter()
        .filter(|tensor| tensor.name.ends_with(".scale") || tensor.name.ends_with(".input_scale"))
        .map(|tensor| {
            read_scalar_vector(mmap, data_start, tensor).map(|values| (tensor.name.clone(), values))
        })
        .collect()
}

fn role_for_name(name: &str) -> Option<TensorRole> {
    [
        TensorRole::GateUp,
        TensorRole::Gate,
        TensorRole::Up,
        TensorRole::Down,
    ]
    .into_iter()
    .find(|role| name.starts_with("blk.") && name.ends_with(role.suffix()))
}

fn layer_for_name(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
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

fn validate_matrix(
    tensor: &TensorSpec,
    experts: u32,
    rows: usize,
    columns: usize,
    decoder: TensorDecoder,
) -> Result<(), Q4LoadError> {
    if tensor.dims.as_slice() != [columns as u64, rows as u64, experts as u64] {
        return Err(Q4LoadError::InvalidTensor(
            "dimensões não correspondem ao tensor de experts",
        ));
    }
    decoder.row_bytes(columns)?;
    Ok(())
}

fn mat_vec(matrix: &[f32], rows: usize, columns: usize, input: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            matrix[row * columns..(row + 1) * columns]
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(weight, value)| weight * value)
        .sum()
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

fn gelu(value: f32) -> f32 {
    if value == f32::NEG_INFINITY {
        return 0.0;
    }
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
}

#[cfg(test)]
mod tests {
    use super::super::admission::MemoryAccount;
    use super::super::cache::{ExpertCache, FreeMemoryProbe};
    use super::*;
    use std::collections::BTreeSet;
    use std::io;
    use std::io::Write;

    #[test]
    fn stable_activations_do_not_turn_limits_into_nan() {
        assert_eq!(silu(f32::NEG_INFINITY), 0.0);
        assert_eq!(gelu(f32::NEG_INFINITY), 0.0);
        assert!(silu(-1_000.0).is_finite());
    }

    #[test]
    fn heterogeneous_cuda_geometry_separates_one_expert_from_the_full_bank() {
        let one_expert = cuda_tensor_expected_bytes(18, 32_768, 256).unwrap();
        let expert_bank = cuda_tensor_bank_expected_bytes(18, 32_768, 256, 8).unwrap();

        assert_eq!(expert_bank, one_expert * 8);
    }

    #[test]
    fn heterogeneous_cuda_geometry_covers_q4_0_q5k_q2k_and_q3k() {
        assert_eq!(cuda_tensor_row_bytes(2, 2_048), Some(1_152));
        assert_eq!(cuda_tensor_expected_bytes(2, 256, 2_048), Some(294_912));
        assert_eq!(cuda_tensor_row_bytes(13, 2_048), Some(1_408));
        assert_eq!(cuda_tensor_expected_bytes(13, 256, 2_048), Some(360_448));
        assert_eq!(cuda_tensor_row_bytes(10, 2_048), Some(672));
        assert_eq!(cuda_tensor_row_bytes(11, 2_048), Some(880));
        assert_eq!(cuda_tensor_expected_bytes(10, 256, 2_048), Some(172_032));
        assert_eq!(cuda_tensor_expected_bytes(11, 256, 2_048), Some(225_280));
    }

    #[test]
    fn heterogeneous_q6_k_gate_up_eight_experts_hidden_2048_is_finite() {
        let experts = 8u32;
        let hidden = 2048usize;
        let intermediate = 256usize;
        let format = KQuantFormat::Q6K;
        let gate_up_rows = 2 * intermediate;
        let gate_up_expert_bytes = gate_up_rows * k_row_bytes(format, hidden).unwrap();
        let down_expert_bytes = hidden * k_row_bytes(format, intermediate).unwrap();
        let mut data =
            Vec::with_capacity(experts as usize * (gate_up_expert_bytes + down_expert_bytes));
        for _ in 0..experts {
            data.extend(q6_matrix(gate_up_rows, hidden));
        }
        for _ in 0..experts {
            data.extend(q6_matrix(hidden, intermediate));
        }

        let tensors = vec![
            TensorSpec {
                name: "blk.0.ffn_gate_up_exps.weight".into(),
                dims: vec![hidden as u64, gate_up_rows as u64, u64::from(experts)],
                dtype: format.dtype(),
                offset: 0,
            },
            TensorSpec {
                name: "blk.0.ffn_down_exps.weight".into(),
                dims: vec![intermediate as u64, hidden as u64, u64::from(experts)],
                dtype: format.dtype(),
                offset: (gate_up_expert_bytes * experts as usize) as u64,
            },
        ];
        let source = test_source_with_bytes(
            tensors,
            &data,
            experts,
            vec![LayerInventory {
                gate_up: Some(ExpertTensor {
                    spec_index: 0,
                    decoder: TensorDecoder::K(format),
                    role: TensorRole::GateUp,
                }),
                gate: None,
                up: None,
                down: Some(ExpertTensor {
                    spec_index: 1,
                    decoder: TensorDecoder::K(format),
                    role: TensorRole::Down,
                }),
                hidden,
                intermediate,
            }],
        );
        let cache = ExpertCache::new(
            MemoryAccount {
                baseline_free_bytes: 1 << 30,
                compute_bytes: 0,
                weights_bytes: 0,
                fixed_bytes: 0,
                reserve_bytes: 0,
            },
            Arc::new(FixedProbe),
        );
        let hidden_values = (0..hidden)
            .map(|index| (index as f32 * 0.001).sin())
            .collect::<Vec<_>>();
        let expert_ids = (0..experts).collect::<Vec<_>>();
        let token_ids = vec![0; experts as usize];
        let weights = vec![1.0 / experts as f32; experts as usize];

        let output = source
            .run_route_batch_from_source(
                &cache,
                "fixture",
                0,
                &expert_ids,
                &token_ids,
                &weights,
                &[0, experts as usize],
                &hidden_values,
                1,
                1,
            )
            .unwrap();

        assert_eq!(output.len(), hidden);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    fn test_source_with_scalar_data(
        tensors: Vec<TensorSpec>,
        values: &[f32],
    ) -> HeterogeneousMoeSource {
        let path = std::env::temp_dir().join(format!(
            "pom-ftk-heterogeneous-sidecars-{}.bin",
            uuid::Uuid::new_v4()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        drop(file);
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: the mapping owns the file contents independently after the
        // descriptor is closed; the test removes only its unique path.
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        std::fs::remove_file(path).unwrap();
        let sidecars = load_sidecars(&mmap, 0, &tensors).unwrap();
        HeterogeneousMoeSource {
            mmap,
            data_start: 0,
            experts: 2,
            tensors,
            sidecars,
            layers: Vec::new(),
        }
    }

    fn test_source_with_bytes(
        tensors: Vec<TensorSpec>,
        data: &[u8],
        experts: u32,
        layers: Vec<LayerInventory>,
    ) -> HeterogeneousMoeSource {
        let path = std::env::temp_dir().join(format!(
            "pom-ftk-heterogeneous-q6-{}.bin",
            uuid::Uuid::new_v4()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(data).unwrap();
        drop(file);
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: the mapping owns the file contents independently after the
        // descriptor is closed; the test removes only its unique path.
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        std::fs::remove_file(path).unwrap();
        let sidecars = load_sidecars(&mmap, 0, &tensors).unwrap();
        HeterogeneousMoeSource {
            mmap,
            data_start: 0,
            experts,
            tensors,
            sidecars,
            layers,
        }
    }

    fn q6_matrix(rows: usize, columns: usize) -> Vec<u8> {
        let block_bytes = 210usize;
        let row_bytes = columns / 256 * block_bytes;
        let mut output = vec![0u8; rows * row_bytes];
        for block in output.chunks_exact_mut(block_bytes) {
            block[..128].fill(0x11);
            // These bytes are valid Q6_K low quants but spell an f16 NaN if
            // mistaken for the super-block scale at the old offset.
            block[0] = 0;
            block[1] = 0x7e;
            block[128..192].fill(0xaa);
            block[192..208].fill(1);
            block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        output
    }

    fn scalar_tensor_with_len(name: &str, offset: usize, elements: usize) -> TensorSpec {
        TensorSpec {
            name: name.to_owned(),
            dims: vec![elements as u64],
            dtype: 0,
            offset: (offset * std::mem::size_of::<f32>()) as u64,
        }
    }

    fn scalar_tensor(name: &str, offset: usize) -> TensorSpec {
        scalar_tensor_with_len(name, offset, 2)
    }

    fn test_expert_tensor(spec_index: usize, role: TensorRole) -> ExpertTensor {
        ExpertTensor {
            spec_index,
            decoder: TensorDecoder::Q4_0,
            role,
        }
    }

    fn test_plain_expert_tensor(spec_index: usize, role: TensorRole) -> ExpertTensor {
        ExpertTensor {
            spec_index,
            decoder: TensorDecoder::Plain(PlainFormat::F32),
            role,
        }
    }

    struct FixedProbe;

    impl FreeMemoryProbe for FixedProbe {
        fn free_bytes(&self) -> io::Result<u64> {
            Ok(1 << 30)
        }
    }

    fn scalar_execution_source() -> HeterogeneousMoeSource {
        let tensors = vec![
            TensorSpec {
                name: "blk.0.ffn_gate_exps.weight".into(),
                dims: vec![2, 2, 2],
                dtype: 0,
                offset: 0,
            },
            TensorSpec {
                name: "blk.0.ffn_up_exps.weight".into(),
                dims: vec![2, 2, 2],
                dtype: 0,
                offset: 32,
            },
            TensorSpec {
                name: "blk.0.ffn_down_exps.weight".into(),
                dims: vec![2, 2, 2],
                dtype: 0,
                offset: 64,
            },
        ];
        let mut source = test_source_with_scalar_data(
            tensors,
            &[
                1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            ],
        );
        source.layers = vec![LayerInventory {
            gate_up: None,
            gate: Some(test_plain_expert_tensor(0, TensorRole::Gate)),
            up: Some(test_plain_expert_tensor(1, TensorRole::Up)),
            down: Some(test_plain_expert_tensor(2, TensorRole::Down)),
            hidden: 2,
            intermediate: 2,
        }];
        source
    }

    #[test]
    fn expert_spec_separates_metadata_from_the_payload() {
        let source = scalar_execution_source();

        let specs = source.expert_tensor_specs(0, 0).unwrap();

        assert_eq!(
            specs
                .iter()
                .map(|spec| (spec.role, spec.dtype, spec.rows, spec.columns))
                .collect::<Vec<_>>(),
            vec![
                (TensorRole::Gate, 0, 2, 2),
                (TensorRole::Up, 0, 2, 2),
                (TensorRole::Down, 0, 2, 2),
            ]
        );
        assert!(specs
            .iter()
            .all(|spec| spec.scales == TensorScales::default()));

        let payloads = specs
            .iter()
            .map(|spec| source.load_expert_tensor(*spec).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            payloads.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![16, 16, 16]
        );
    }

    #[test]
    fn the_shared_expert_stays_out_of_the_route_inventory() {
        assert_eq!(
            role_for_name("blk.1.ffn_gate_exps.weight"),
            Some(TensorRole::Gate)
        );
        assert_eq!(
            role_for_name("blk.1.ffn_down_exps.weight"),
            Some(TensorRole::Down)
        );
        assert_eq!(role_for_name("blk.1.ffn_gate_shexp.weight"), None);
        assert_eq!(role_for_name("blk.1.ffn_up_shexp.weight"), None);
        assert_eq!(role_for_name("blk.1.ffn_down_shexp.weight"), None);
    }

    #[test]
    fn route_inventory_separates_exps_from_shexp_and_the_sidecars() {
        let names = [
            "blk.1.ffn_gate_up_exps.weight",
            "blk.1.ffn_gate_exps.weight",
            "blk.1.ffn_up_exps.weight",
            "blk.1.ffn_down_exps.weight",
            "blk.1.ffn_gate_shexp.weight",
            "blk.1.ffn_up_shexp.weight",
            "blk.1.ffn_down_shexp.weight",
            "blk.1.ffn_gate_exps.weight.scale",
            "blk.1.ffn_gate_exps.weight.input_scale",
        ];

        let routed = names
            .iter()
            .filter(|name| role_for_name(name).is_some())
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(
            routed,
            vec![
                "blk.1.ffn_gate_up_exps.weight",
                "blk.1.ffn_gate_exps.weight",
                "blk.1.ffn_up_exps.weight",
                "blk.1.ffn_down_exps.weight",
            ]
        );
        for role in [
            TensorRole::GateUp,
            TensorRole::Gate,
            TensorRole::Up,
            TensorRole::Down,
        ] {
            assert_eq!(
                routed
                    .iter()
                    .filter_map(|name| role_for_name(name))
                    .filter(|found| *found == role)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn fused_sidecars_use_the_right_component_and_split_fallback() {
        let tensors = vec![
            scalar_tensor("blk.0.ffn_gate_up_exps.weight", 0),
            scalar_tensor_with_len("blk.0.ffn_gate_up_exps.scale", 0, 4),
            scalar_tensor_with_len("blk.0.ffn_gate_up_exps.input_scale", 8, 4),
            scalar_tensor("blk.0.ffn_gate_exps.weight", 0),
            scalar_tensor("blk.0.ffn_up_exps.weight", 0),
            scalar_tensor("blk.0.ffn_gate_exps.scale", 4),
            scalar_tensor("blk.0.ffn_up_exps.scale", 6),
            scalar_tensor("blk.0.ffn_gate_exps.input_scale", 8),
            scalar_tensor("blk.0.ffn_up_exps.input_scale", 10),
        ];
        let source = test_source_with_scalar_data(
            tensors,
            &[
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
            ],
        );
        assert_eq!(source.sidecars.len(), 6);
        let fused = test_expert_tensor(0, TensorRole::GateUp);

        assert_eq!(
            source.sidecar_scale(fused, "scale", 1, 0, None).unwrap(),
            2.0
        );
        assert_eq!(
            source.sidecar_scale(fused, "scale", 1, 1, None).unwrap(),
            4.0
        );
        assert_eq!(
            source
                .sidecar_scale(fused, "input_scale", 1, 0, None)
                .unwrap(),
            200.0
        );
        assert_eq!(
            source
                .sidecar_scale(fused, "input_scale", 1, 1, None)
                .unwrap(),
            400.0
        );

        let fallback_tensors = vec![
            scalar_tensor("blk.0.ffn_gate_up_exps.weight", 0),
            scalar_tensor("blk.0.ffn_gate_exps.weight", 0),
            scalar_tensor("blk.0.ffn_up_exps.weight", 0),
            scalar_tensor("blk.0.ffn_gate_exps.scale", 4),
            scalar_tensor("blk.0.ffn_up_exps.scale", 6),
            scalar_tensor("blk.0.ffn_gate_exps.input_scale", 8),
            scalar_tensor("blk.0.ffn_up_exps.input_scale", 10),
        ];
        let fallback_source = test_source_with_scalar_data(
            fallback_tensors,
            &[
                0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
            ],
        );
        let fallback_fused = test_expert_tensor(0, TensorRole::GateUp);
        let fallback_gate = test_expert_tensor(1, TensorRole::Gate);
        let fallback_up = test_expert_tensor(2, TensorRole::Up);

        assert_eq!(
            fallback_source
                .sidecar_scale(fallback_fused, "scale", 1, 0, Some(fallback_gate))
                .unwrap(),
            20.0
        );
        assert_eq!(
            fallback_source
                .sidecar_scale(fallback_fused, "scale", 1, 1, Some(fallback_up))
                .unwrap(),
            40.0
        );
        assert_eq!(
            fallback_source
                .sidecar_scale(fallback_fused, "input_scale", 1, 0, Some(fallback_gate))
                .unwrap(),
            200.0
        );
        assert_eq!(
            fallback_source
                .sidecar_scale(fallback_fused, "input_scale", 1, 1, Some(fallback_up))
                .unwrap(),
            400.0
        );
    }

    #[test]
    fn heterogeneous_batch_preserves_the_per_token_traversal() {
        let source = scalar_execution_source();
        let cache = ExpertCache::new(
            MemoryAccount {
                baseline_free_bytes: 1 << 30,
                compute_bytes: 0,
                weights_bytes: 0,
                fixed_bytes: 0,
                reserve_bytes: 0,
            },
            Arc::new(FixedProbe),
        );
        let hidden = vec![1.0, 2.0, 2.0, 3.0];
        let experts = vec![0, 1, 1, 0];
        let token_ids = vec![0, 0, 1, 1];
        let weights = vec![0.25, 0.75, 0.6, 0.4];
        let token_offsets = vec![0, 2, 4];

        let mut expected = Vec::new();
        for token in 0..2 {
            let start = token * 2;
            expected.extend(
                source
                    .run_from_source_with_route(
                        &cache,
                        "fixture",
                        0,
                        &hidden[token * 2..(token + 1) * 2],
                        &experts[start..start + 2],
                        &weights[start..start + 2],
                        1,
                    )
                    .unwrap(),
            );
        }
        let actual = source
            .run_route_batch_from_source(
                &cache,
                "fixture",
                0,
                &experts,
                &token_ids,
                &weights,
                &token_offsets,
                &hidden,
                2,
                1,
            )
            .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn heterogeneous_metadata_explains_the_incompatible_geometry() {
        let source = scalar_execution_source();
        let error = source
            .validate_metadata(
                0,
                &[(
                    TensorRole::Gate,
                    "blk.0.ffn_gate_exps.weight",
                    0,
                    3,
                    [512, 2, 2, 0],
                )],
            )
            .unwrap_err()
            .to_string();

        assert!(error.contains("[512, 2, 2]"), "{error}");
        assert!(error.contains("[2, 2, 2]"), "{error}");
    }

    #[test]
    fn heterogeneous_cuda_layout_computes_mxfp4_rows_for_down() {
        assert_eq!(cuda_tensor_row_bytes(39, 2_048), Some(1_088));
    }

    #[test]
    #[ignore = "requer o GGUF real Qwen3.6-35B-A3B-UD-IQ3_S"]
    fn real_qwen_ud_iq3_s_inventories_dtypes_per_tensor() {
        let path = std::env::var_os("POM_REAL_MOE_GGUF")
            .expect("POM_REAL_MOE_GGUF deve apontar para o GGUF real");
        let source = HeterogeneousMoeSource::open_infer(path, 256)
            .expect("o GGUF real UD-IQ3_S deve abrir com inventário heterogêneo");
        assert_eq!(source.experts(), 256);
        assert_eq!(source.layer_count(), 40);
        let dtypes = source
            .tensors
            .iter()
            .filter(|tensor| role_for_name(&tensor.name).is_some())
            .map(|tensor| tensor.dtype)
            .collect::<BTreeSet<_>>();
        assert_eq!(dtypes, BTreeSet::from([21, 22, 23]));
        for layer in 0..source.layer_count() {
            let (hidden, intermediate) = source
                .layer_geometry(layer)
                .expect("cada camada deve ter geometria heterogênea");
            assert!(hidden > 0);
            assert!(intermediate > 0);
        }
    }
}
