//! Minimal GGUF metadata reader.
//!
//! Reads only the key/value header of a `.gguf` file — enough to learn the
//! architecture and the real block (layer) count without loading any weights.
//! Mirrors the reader in `split.py`, so both agree on what a model looks like.
//!
//! Only the KV section is parsed; the tensor table and tensor data are skipped.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

// GGUF value type codes.
const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

/// Fixed byte width of every scalar GGUF value type.
fn scalar_size(t: u32) -> Option<u64> {
    match t {
        T_UINT8 | T_INT8 | T_BOOL => Some(1),
        T_UINT16 | T_INT16 => Some(2),
        T_UINT32 | T_INT32 | T_FLOAT32 => Some(4),
        T_UINT64 | T_INT64 | T_FLOAT64 => Some(8),
        _ => None,
    }
}

/// Metadata arrays used by Qwen4Exp's prompt lookup embedding (PLE) do not
/// follow the generic per-block rule. The three table-description arrays are
/// copied intact; `ple.layers` is a selector list and is filtered/rebased by
/// the splitter. Keeping it in this category also exempts it from the generic
/// short-array refusal.
///
/// DeepSeek-V4.1 has the same shape and more of it: its Engram token map and
/// hash tables can be longer than the block count, which the generic rule would
/// cut by layer without any error, and its KV and index source lists name
/// layers.
pub fn is_structural_array(key: &str, architecture: &str) -> bool {
    match architecture {
        "qwen4exp" => matches!(
            key,
            "qwen4exp.ple.layers"
                | "qwen4exp.ple.layer_multipliers"
                | "qwen4exp.ple.head_offsets"
                | "qwen4exp.ple.head_vocab_sizes"
        ),
        "deepseek41" => matches!(
            key,
            "deepseek41.ple.layers"
                | "deepseek41.ple.layer_multipliers"
                | "deepseek41.ple.head_offsets"
                | "deepseek41.ple.head_vocab_sizes"
                | "deepseek41.engram.token_map"
                | "deepseek41.attention.kv_source_layers"
                | "deepseek41.attention.indexer.source_layers"
        ),
        _ => false,
    }
}

/// Structural arrays whose values are absolute layer indexes: a slice keeps the
/// ones in its range, renumbered to its own blocks.
pub fn is_layer_selector_array(key: &str, architecture: &str) -> bool {
    match architecture {
        "qwen4exp" => key == "qwen4exp.ple.layers",
        "deepseek41" => matches!(
            key,
            "deepseek41.ple.layers"
                | "deepseek41.attention.kv_source_layers"
                | "deepseek41.attention.indexer.source_layers"
        ),
        _ => false,
    }
}

/// The DeepSeek-V4.1 scalar naming the layer that pins the candidate blocks.
/// llama.cpp treats its absence as "no candidate selection", which is right for
/// a slice that ends before that layer; inside the range it is renumbered.
pub const DEEPSEEK41_CANDIDATE_SOURCE_KEY: &str =
    "deepseek41.attention.indexer.candidate_source_layer";

/// Where a DeepSeek-V4.1 slice can start. `deepseek41.cpp` resolves these
/// dependencies inside one graph, so each one limits the cut:
/// - a compressed layer reads the KV and the top-k of the latest source layer
///   at or before it, so the slice has to start at a layer that is both;
/// - the layers after the candidate source consume its pin, so the cut cannot
///   come after that source.
///
/// Engram layers no longer limit it: a slice that starts mid-model gets the
/// layer-boundary state with the token ids of its rows (`Engine::state_tokens`,
/// the fork's `deepseek41.cpp` hashing them), so an Engram layer runs in any
/// slice.
fn deepseek41_slice_starts(
    kv_sources: &[u32],
    index_sources: &[u32],
    candidate_source: Option<u32>,
    total_layers: u32,
) -> Vec<u32> {
    let mut starts: Vec<u32> = kv_sources
        .iter()
        .copied()
        .filter(|&layer| layer > 0 && layer < total_layers)
        .filter(|layer| index_sources.contains(layer))
        .filter(|&layer| candidate_source.map_or(true, |source| layer <= source))
        .collect();
    starts.sort_unstable();
    starts.dedup();
    starts
}

/// What we care about from a GGUF header.
#[derive(Debug, Clone, Default)]
pub struct GgufInfo {
    /// `general.architecture`, e.g. "qwen3" / "llama".
    pub architecture: Option<String>,
    /// `general.name` as written by the producer.
    pub name: Option<String>,
    /// GGUF-declared model identifiers that can link a projector to its text
    /// model. Includes `general.name`, `general.basename`, and base-model
    /// names/repository URLs when present.
    pub identity_names: Vec<String>,
    /// `<arch>.block_count` — layers present *in this file*.
    pub block_count: Option<u32>,
    /// `pom.original_block_count` — layers in the model this file was sliced from.
    /// Only present on shards written by `split_model.py`.
    pub original_block_count: Option<u32>,
    /// `pom.start_layer` / `pom.end_layer` on shards.
    pub shard_range: Option<(u32, u32)>,
    /// `<arch>.embedding_length`.
    pub embedding_length: Option<u32>,
    /// `clip.vision.projection_dim`, the output width expected by the text
    /// model. It is distinct from the vision encoder's embedding length.
    pub projector_projection_dim: Option<u32>,
    /// `<arch>.attention.head_count_kv`, scalar ou valor por camada.
    pub attention_head_count_kv: Option<Vec<u32>>,
    /// `<arch>.attention.key_length`, scalar ou valor por camada.
    pub attention_key_length: Option<Vec<u32>>,
    /// `<arch>.attention.value_length`, scalar ou valor por camada.
    pub attention_value_length: Option<Vec<u32>>,
    /// `general.size_label`, e.g. "35B".
    pub size_label: Option<String>,
    /// `<arch>.expert_count`, present only when the architecture has routed
    /// feed-forward experts.
    pub expert_count: Option<u32>,
    /// `<arch>.expert_used_count`, the number selected by the router per token.
    pub expert_used_count: Option<u32>,
    /// Template Jinja real do tokenizer, quando o GGUF o declara.
    pub chat_template: Option<String>,
    /// Tokens de controle de thinking encontrados no vocabulário real. `None`
    /// significa que o GGUF não permitiu confirmar o vocabulário.
    pub thinking_control_tokens: Option<Vec<String>>,
    /// `<arch>.nextn_predict_layers`, quando presente no cabeçalho.
    pub nextn_predict_layers: Option<u32>,
    /// `pom.tp_format`/`pom.tp_size`/`pom.tp_rank` de um shard de tensor
    /// parallelism: qual versão da regra de fatiamento escreveu este arquivo, e
    /// que posição do grupo ele é.
    ///
    /// `None` em qualquer arquivo que não seja shard de TP - **e também** nos
    /// shards escritos antes de `tp_format` existir, que é o ponto: um shard
    /// sem carimbo é um shard de regra desconhecida, e regra desconhecida se
    /// regenera. Ver `splitter::TP_SHARD_FORMAT`.
    pub tp_stamp: Option<TpStamp>,
    /// Contagem de tensores declarada no cabeçalho. O catálogo usa isto para
    /// descrever uma parte sem abrir a fonte completa.
    pub tensor_count: u64,
    /// Campos `split.*` do cabeçalho, quando presentes. Eles são suficientes
    /// para identificar a coleção multipart sem ler a tabela de tensores.
    pub split_count: Option<u64>,
    pub split_no: Option<u64>,
    pub split_tensors_count: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArchitecture {
    Dense {
        architecture: Option<String>,
    },
    Moe {
        architecture: Option<String>,
        expert_count: u32,
        expert_used_count: Option<u32>,
    },
}

impl ModelArchitecture {
    pub fn is_moe(&self) -> bool {
        matches!(self, Self::Moe { .. })
    }
}

/// Carimbo de origem de um shard de tensor parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpStamp {
    pub format: u32,
    pub size: u32,
    pub rank: u32,
}

impl GgufInfo {
    /// Classify only from explicit GGUF metadata. A missing or zero expert
    /// count is dense, even when the architecture name looks unfamiliar.
    pub fn model_architecture(&self) -> ModelArchitecture {
        match self.expert_count.filter(|count| *count > 1) {
            Some(expert_count) => ModelArchitecture::Moe {
                architecture: self.architecture.clone(),
                expert_count,
                expert_used_count: self.expert_used_count,
            },
            None => ModelArchitecture::Dense {
                architecture: self.architecture.clone(),
            },
        }
    }

    /// Whether the real GGUF metadata describes a mixture of experts model.
    /// This delegates to the explicit expert metadata and never to the
    /// architecture name.
    pub fn is_moe(&self) -> bool {
        self.model_architecture().is_moe()
    }

    /// A capacidade de desativar thinking só é afirmada quando o template e
    /// os tokens de controle usados pelo runtime estão ambos no GGUF.
    pub fn thinking_capability(&self) -> Option<bool> {
        let template = self.chat_template.as_deref()?;
        let tokens = self.thinking_control_tokens.as_ref()?;
        let has = |token: &str| tokens.iter().any(|value| value == token);
        let generic = template.contains("<think>") && template.contains("</think>");
        let gemma4 = template.contains("enable_thinking") && template.contains("<|think|>");
        if generic {
            Some(has("<think>") && has("</think>"))
        } else if gemma4 {
            Some(has("<|think|>"))
        } else {
            Some(false)
        }
    }

    /// Layers of the *whole* model: the shard metadata when present, else the
    /// block count of the file itself.
    pub fn total_layers(&self) -> Option<u32> {
        self.original_block_count.or(self.block_count)
    }

    /// Geometria de KV de cada camada do intervalo pedido.
    ///
    /// GGUF pode declarar uma geometria escalar ou um array por camada. O
    /// planejador nunca reduz um array ao primeiro valor: quando a forma
    /// varia, cada camada recebe a sua própria conta.
    pub fn kv_geometry_for_range(
        &self,
        start_layer: u32,
        end_layer: u32,
    ) -> Result<Vec<crate::kv_budget::KvLayerGeometry>> {
        if end_layer < start_layer {
            bail!("intervalo de camadas KV invertido");
        }
        let range_layers = end_layer
            .checked_sub(start_layer)
            .and_then(|layers| layers.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("overflow ao contar camadas do KV"))?;
        (start_layer..=end_layer)
            .map(|layer| {
                Ok(crate::kv_budget::KvLayerGeometry {
                    kv_heads: self.kv_value(
                        &self.attention_head_count_kv,
                        "attention.head_count_kv",
                        layer,
                        start_layer,
                        range_layers,
                    )? as u64,
                    key_head_dim: self.kv_value(
                        &self.attention_key_length,
                        "attention.key_length",
                        layer,
                        start_layer,
                        range_layers,
                    )? as u64,
                    value_head_dim: self.kv_value(
                        &self.attention_value_length,
                        "attention.value_length",
                        layer,
                        start_layer,
                        range_layers,
                    )? as u64,
                })
            })
            .collect()
    }

    fn kv_value(
        &self,
        values: &Option<Vec<u32>>,
        name: &str,
        layer: u32,
        start_layer: u32,
        range_layers: u32,
    ) -> Result<u32> {
        let values = values
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GGUF não declara {name}; orçamento KV recusado"))?;
        let index = if values.len() == 1 {
            0
        } else if values.len() == range_layers as usize {
            (layer - start_layer) as usize
        } else {
            layer as usize
        };
        let value = values.get(index).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "GGUF declara {name} para {} camadas, mas falta a camada {layer}",
                values.len()
            )
        })?;
        if value == 0 {
            bail!("GGUF declara {name}=0 na camada {layer}; orçamento KV recusado");
        }
        Ok(value)
    }
}

/// Evidência de capacidade lida diretamente da metadata e da tabela de
/// tensores do GGUF. `None` em uma capacidade significa que a leitura não
/// conseguiu provar o estado, não que a capacidade está presente.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilityEvidence {
    pub thinking: Option<bool>,
    pub mtp: Option<bool>,
}

struct Reader<R: Read + Seek> {
    inner: R,
}

#[derive(Debug, Clone)]
struct LayoutKv {
    key: String,
    raw_bytes: u64,
    array: Option<ArrayShape>,
}

/// How llama.cpp interprets a metadata count when it chooses tensors or
/// graph branches by the local zero-based layer index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerCountSemantics {
    FromStart,
    FromEnd,
}

#[derive(Debug, Clone, Copy)]
struct ArrayShape {
    element_type: u32,
    count: u64,
}

#[derive(Debug, Clone)]
struct LayoutTensor {
    name: String,
    dims: Vec<u64>,
    bytes: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn position(&mut self) -> Result<u64> {
        Ok(self.inner.stream_position()?)
    }

    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.inner.read_exact(&mut b)?;
        Ok(b[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.inner.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn i8(&mut self) -> Result<i8> {
        let mut b = [0u8; 1];
        self.inner.read_exact(&mut b)?;
        Ok(i8::from_le_bytes(b))
    }

    fn i16(&mut self) -> Result<i16> {
        let mut b = [0u8; 2];
        self.inner.read_exact(&mut b)?;
        Ok(i16::from_le_bytes(b))
    }

    fn i32(&mut self) -> Result<i32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }

    fn i64(&mut self) -> Result<i64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        // Guard against a corrupt length turning into a huge allocation.
        if len > 64 * 1024 * 1024 {
            bail!("implausible GGUF string length {len}");
        }
        let mut buf = vec![0u8; len as usize];
        self.inner.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        self.inner.seek(SeekFrom::Current(n as i64))?;
        Ok(())
    }

    /// Skip a value of the given type without materialising it.
    fn skip_value(&mut self, vtype: u32) -> Result<()> {
        if let Some(sz) = scalar_size(vtype) {
            return self.skip(sz);
        }
        match vtype {
            T_STRING => {
                let len = self.u64()?;
                self.skip(len)
            }
            T_ARRAY => {
                let elem = self.u32()?;
                let count = self.u64()?;
                if let Some(sz) = scalar_size(elem) {
                    self.skip(sz.saturating_mul(count))
                } else {
                    for _ in 0..count {
                        self.skip_value(elem)?;
                    }
                    Ok(())
                }
            }
            other => bail!("unknown GGUF value type {other}"),
        }
    }

    fn skip_array_payload(&mut self, element_type: u32, count: u64) -> Result<u64> {
        let start = self.position()?;
        if let Some(sz) = scalar_size(element_type) {
            self.skip(sz.saturating_mul(count))?;
        } else {
            for _ in 0..count {
                self.skip_value(element_type)?;
            }
        }
        Ok(self.position()? - start)
    }

    fn value_as_u64(&mut self, vtype: u32) -> Result<Option<u64>> {
        let v = match vtype {
            T_UINT8 => Some(self.u8()? as u64),
            T_UINT16 => Some(self.u16()? as u64),
            T_UINT32 => Some(self.u32()? as u64),
            T_UINT64 => Some(self.u64()?),
            T_INT8 => Some(nonnegative_integer(i128::from(self.i8()?))?),
            T_INT16 => Some(nonnegative_integer(i128::from(self.i16()?))?),
            T_INT32 => Some(nonnegative_integer(i128::from(self.i32()?))?),
            T_INT64 => Some(nonnegative_integer(i128::from(self.i64()?))?),
            _ => {
                self.skip_value(vtype)?;
                None
            }
        };
        Ok(v)
    }

    /// Read a scalar as u32 when it is a nonnegative integer; otherwise skip
    /// it. GGUF producers use signed integer types for some numeric metadata.
    fn value_as_u32(&mut self, vtype: u32) -> Result<Option<u32>> {
        self.value_as_u64(vtype)?
            .map(|value| {
                u32::try_from(value)
                    .map_err(|_| anyhow::anyhow!("inteiro GGUF grande demais para u32: {value}"))
            })
            .transpose()
    }

    fn values_as_u32(&mut self, vtype: u32) -> Result<Option<Vec<u32>>> {
        if vtype != T_ARRAY {
            return Ok(self.value_as_u32(vtype)?.map(|value| vec![value]));
        }

        let element_type = self.u32()?;
        let count = self.u64()?;
        if count > 100_000 {
            bail!("implausible GGUF integer array length {count}");
        }
        let mut values = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let Some(value) = self.value_as_u32(element_type)? else {
                return Ok(None);
            };
            values.push(value);
        }
        Ok(Some(values))
    }

    fn string_array_matches(&mut self, vtype: u32, wanted: &[&str]) -> Result<Option<Vec<String>>> {
        if vtype != T_ARRAY {
            self.skip_value(vtype)?;
            return Ok(None);
        }
        let element_type = self.u32()?;
        let count = self.u64()?;
        if element_type != T_STRING {
            for _ in 0..count {
                self.skip_value(element_type)?;
            }
            return Ok(None);
        }
        if count > 1_000_000 {
            bail!("implausible GGUF string array length {count}");
        }
        let mut matches = Vec::new();
        for _ in 0..count {
            let value = self.string()?;
            if wanted.iter().any(|candidate| *candidate == value) {
                matches.push(value);
            }
        }
        Ok(Some(matches))
    }
}

fn nonnegative_integer(value: i128) -> Result<u64> {
    u64::try_from(value).map_err(|_| anyhow::anyhow!("inteiro GGUF não pode ser negativo: {value}"))
}

/// Parse the KV header of a GGUF file.
///
/// Cheap: reads a few hundred KB at most, never the tensor data.
pub fn read_info(path: &Path) -> Result<GgufInfo> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = Reader {
        inner: BufReader::with_capacity(1 << 16, file),
    };

    let magic = r.u32()?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (magic {magic:#x})");
    }
    let _version = r.u32()?;
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_kv > 100_000 {
        bail!("implausible GGUF kv count {n_kv}");
    }

    let mut info = GgufInfo::default();
    let mut start_layer: Option<u32> = None;
    let mut end_layer: Option<u32> = None;
    let mut tp_format: Option<u32> = None;
    let mut tp_size: Option<u32> = None;
    let mut tp_rank: Option<u32> = None;

    for _ in 0..n_kv {
        let key = r.string()?;
        let vtype = r.u32()?;

        match key.as_str() {
            "general.architecture" if vtype == T_STRING => info.architecture = Some(r.string()?),
            "general.name" if vtype == T_STRING => {
                let name = r.string()?;
                info.identity_names.push(name.clone());
                info.name = Some(name);
            }
            "general.basename" if vtype == T_STRING => {
                info.identity_names.push(r.string()?);
            }
            k if k.starts_with("general.base_model.")
                && (k.ends_with(".name") || k.ends_with(".repo_url"))
                && vtype == T_STRING =>
            {
                info.identity_names.push(r.string()?);
            }
            "general.repo_url" if vtype == T_STRING => {
                info.identity_names.push(r.string()?);
            }
            "general.size_label" if vtype == T_STRING => info.size_label = Some(r.string()?),
            "tokenizer.chat_template" if vtype == T_STRING => {
                info.chat_template = Some(r.string()?)
            }
            "tokenizer.ggml.tokens" => {
                info.thinking_control_tokens = Some(
                    r.string_array_matches(vtype, &["<think>", "</think>", "<|think|>"])?
                        .unwrap_or_default(),
                )
            }
            // Chaves nossas: o prefixo é `crate::GGUF_PREFIX`, uma constante só,
            // para renomear o produto não deixar metade do leitor procurando o
            // nome velho.
            k if k == crate::gguf_key("original_block_count") => {
                info.original_block_count = r.value_as_u32(vtype)?
            }
            k if k == crate::gguf_key("tp_format") => tp_format = r.value_as_u32(vtype)?,
            k if k == crate::gguf_key("tp_size") => tp_size = r.value_as_u32(vtype)?,
            k if k == crate::gguf_key("tp_rank") => tp_rank = r.value_as_u32(vtype)?,
            "split.count" => info.split_count = r.value_as_u32(vtype)?.map(u64::from),
            "split.no" => info.split_no = r.value_as_u32(vtype)?.map(u64::from),
            "split.tensors.count" => {
                info.split_tensors_count = r.value_as_u32(vtype)?.map(u64::from)
            }
            k if k == crate::gguf_key("start_layer") => start_layer = r.value_as_u32(vtype)?,
            k if k == crate::gguf_key("end_layer") => end_layer = r.value_as_u32(vtype)?,
            // `<arch>.block_count` / `<arch>.embedding_length` — the arch prefix
            // varies per model, so match on the suffix.
            k if k.ends_with(".block_count") => info.block_count = r.value_as_u32(vtype)?,
            k if k.ends_with(".embedding_length") => {
                info.embedding_length = r.value_as_u32(vtype)?
            }
            "clip.vision.projection_dim" => {
                info.projector_projection_dim = r.value_as_u32(vtype)?
            }
            k if k.ends_with(".expert_count") => info.expert_count = r.value_as_u32(vtype)?,
            k if k.ends_with(".expert_used_count") => {
                info.expert_used_count = r.value_as_u32(vtype)?
            }
            k if is_nextn_predict_layers_key(k) => {
                info.nextn_predict_layers = r.value_as_u32(vtype)?
            }
            k if k.ends_with(".attention.head_count_kv") => {
                info.attention_head_count_kv = r.values_as_u32(vtype)?
            }
            k if k.ends_with(".attention.key_length") => {
                info.attention_key_length = r.values_as_u32(vtype)?
            }
            k if k.ends_with(".attention.value_length") => {
                info.attention_value_length = r.values_as_u32(vtype)?
            }
            _ => r.skip_value(vtype)?,
        }
    }

    if let (Some(s), Some(e)) = (start_layer, end_layer) {
        info.shard_range = Some((s, e));
    }
    // Sem os tres, nao ha carimbo: um shard antigo trazia `tp_size`/`tp_rank`
    // mas nenhum `tp_format`, e e exatamente esse arquivo que nao pode passar
    // por atual.
    if let (Some(f), Some(sz), Some(rk)) = (tp_format, tp_size, tp_rank) {
        info.tp_stamp = Some(TpStamp {
            format: f,
            size: sz,
            rank: rk,
        });
    }

    info.tensor_count = n_tensors;

    Ok(info)
}

/// Read only tensor names, enough to prove structural MTP evidence.
///
/// This deliberately skips dimensions, dtypes, offsets, and tensor data. The
/// deployment catalog must not run the full layout accounting pass for every
/// model just to answer whether a `nextn` tensor exists.
pub fn has_nextn_tensors(path: &Path) -> Result<bool> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = Reader {
        inner: BufReader::with_capacity(1 << 16, file),
    };

    let magic = r.u32()?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (magic {magic:#x})");
    }
    let _version = r.u32()?;
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > 10_000_000 {
        bail!("implausible GGUF tensor count {n_tensors}");
    }
    if n_kv > 100_000 {
        bail!("implausible GGUF kv count {n_kv}");
    }

    for _ in 0..n_kv {
        let _key = r.string()?;
        let value_type = r.u32()?;
        r.skip_value(value_type)?;
    }

    for _ in 0..n_tensors {
        let name = r.string()?;
        let dimensions = r.u32()?;
        if dimensions > 8 {
            bail!("implausible tensor rank {dimensions} for {name}");
        }
        let dimension_bytes = u64::from(dimensions)
            .checked_mul(8)
            .ok_or_else(|| anyhow::anyhow!("tensor dimension byte count overflow"))?;
        r.skip(dimension_bytes)?;
        let _dtype = r.u32()?;
        let _offset = r.u64()?;
        if name.split('.').any(|part| part == "nextn") {
            return Ok(true);
        }
    }

    Ok(false)
}

// ── Tensor layout ────────────────────────────────────────────────────────────

/// Bytes-per-block and elements-per-block for every GGML tensor type.
///
/// Codes are `enum ggml_type`; sizes are what `ggml_type_size`/`ggml_blck_size`
/// report. Both were read from the llama.cpp in `./venv`, not from memory: the
/// IQ block used to start at 19 (every IQ code off by three, because the two
/// removed Q4_2/Q4_3 slots were counted away) which measured IQ3_S tensors with
/// IQ3_XXS's size and priced an IQ model's layers at roughly 60% of the truth.
///
/// This table and `_GGML_TYPE_INFO` in `split_model.py` must agree; the two
/// live in different languages, so `split_model_type_table_matches` below reads
/// the Python one and fails the build when they drift.
pub(crate) fn ggml_type_info(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        0 => (4, 1),      // F32
        1 => (2, 1),      // F16
        2 => (18, 32),    // Q4_0
        3 => (20, 32),    // Q4_1
        6 => (22, 32),    // Q5_0
        7 => (24, 32),    // Q5_1
        8 => (34, 32),    // Q8_0
        9 => (36, 32),    // Q8_1
        10 => (84, 256),  // Q2_K
        11 => (110, 256), // Q3_K
        12 => (144, 256), // Q4_K
        13 => (176, 256), // Q5_K  — was 112: qs is QK_K/2=128 bytes, not 64
        14 => (210, 256), // Q6_K
        15 => (292, 256), // Q8_K
        16 => (66, 256),  // IQ2_XXS — was labelled BF16 (2,1); BF16 is 30
        17 => (74, 256),  // IQ2_XS
        18 => (98, 256),  // IQ3_XXS
        19 => (50, 256),  // IQ1_S
        20 => (18, 32),   // IQ4_NL
        21 => (110, 256), // IQ3_S
        22 => (82, 256),  // IQ2_S
        23 => (136, 256), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (2, 1),     // I16
        26 => (4, 1),     // I32
        27 => (8, 1),     // I64
        28 => (8, 1),     // F64
        29 => (56, 256),  // IQ1_M
        30 => (2, 1),     // BF16
        34 => (54, 256),  // TQ1_0
        35 => (66, 256),  // TQ2_0
        39 => (17, 32),   // MXFP4
        40 => (36, 64),   // NVFP4
        41 => (18, 128),  // Q1_0
        42 => (18, 64),   // Q2_0
        _ => return None,
    })
}

/// Real on-disk weight cost of a model, broken down by transformer block.
///
/// This is what makes an honest memory budget possible: instead of guessing
/// "≈380 MB per layer", each layer's actual quantised size is summed from the
/// tensor table.
#[derive(Debug, Clone, Default)]
pub struct GgufLayout {
    /// `layer_bytes[n]` — total bytes of every `blk.<n>.*` tensor.
    pub layer_bytes: Vec<u64>,
    /// `token_embd.weight` — every slice carries it (see `split_model.py`).
    pub embedding_bytes: u64,
    /// Tensors de saída, incluindo cabeças específicas da arquitetura; só a
    /// fatia final carrega esse grupo.
    pub head_bytes: u64,
    /// Tensors belonging to none of the above.
    pub other_bytes: u64,
    /// Sum of everything, i.e. the size of the full model's weights.
    pub total_bytes: u64,
    /// Exact byte length of the source GGUF, when read from disk.
    pub source_file_bytes: u64,
    /// `layer_kinds[n]` — sorted tensor suffixes of `blk.<n>.*`, so two layers
    /// that are different *kinds* of layer compare unequal. Hybrid models
    /// (attention interleaved with Gated DeltaNet / Mamba) are not a stack of
    /// identical blocks, and that decides where a slice may start.
    pub layer_kinds: Vec<Vec<String>>,
    /// Values of the metadata counted from the first layer
    /// (`LayerCountSemantics::FromStart`, such as `leading_dense_block_count`).
    /// llama.cpp decides a layer's kind by comparing its local index with that
    /// count, and the splitter rebases it by `start_layer`, so a kind change
    /// exactly at that index survives the renumbering and is not a cut limit.
    pub start_relative_layer_counts: Vec<u32>,
    /// `<arch>.nextn_predict_layers` — how many trailing layers form the
    /// NextN/MTP (multi-token prediction) block. On Qwen3.5+ the predictor
    /// tensors live only in the very last layer; a slice that does not end on
    /// the model's last layer must carry this hyperparameter zeroed or
    /// llama.cpp aborts the load looking for `blk.<last>.nextn.*`.
    pub nextn_predict_layers: u32,
    /// Há tensores `nextn` reais na tabela do arquivo.
    pub has_nextn_tensors: bool,
    /// Metadata and tensor table needed to price the exact serialized slice.
    /// It is populated by `read_layout`; hand-built layouts in unit tests use
    /// the legacy payload-only fallback.
    pub serialized: Option<SerializedLayout>,
    /// Slice starts an architecture allows no matter what the tensor suffixes
    /// say. `None` leaves the decision to `layer_kinds`; an empty list means the
    /// model has no valid cut and stays whole.
    pub restricted_slice_starts: Option<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct SerializedLayout {
    architecture: String,
    hidden_dim: u32,
    original_block_count: u32,
    ple_layers: Vec<u32>,
    /// Values of the `is_layer_selector_array` keys.
    layer_selectors: HashMap<String, Vec<u32>>,
    deepseek41_candidate_source: Option<u32>,
    kv: Vec<LayoutKv>,
    tensors: Vec<LayoutTensor>,
}

impl GgufLayout {
    pub fn layer_count(&self) -> u32 {
        self.layer_bytes.len() as u32
    }

    /// Indices where the layer kind changes — the architectural boundaries.
    ///
    /// llama.cpp decides a layer's kind from its *index*, and a slice renumbers
    /// `blk.<start+i>` to `blk.<i>`. A slice boundary is only safe where the
    /// kind sequence is continuous: cutting a hybrid model mid-group shifts every
    /// subsequent layer onto the wrong kind — sometimes a hard `missing tensor`
    /// error, sometimes silently wrong answers.
    ///
    /// For [8 Mamba, 8 Attention, 8 Mamba, 8 Attention] this returns [8, 16, 24].
    ///
    /// A change at an index declared by a start-relative count (for example the
    /// dense prefix of a MoE, `leading_dense_block_count = 2`) is not a
    /// boundary: the splitter rebases that count, so every cut keeps each
    /// layer on its kind.
    pub fn arch_boundaries(&self) -> Vec<u32> {
        if let Some(starts) = &self.restricted_slice_starts {
            return starts.clone();
        }
        let n = self.layer_kinds.len();
        (1..n)
            .filter(|&i| self.layer_kinds[i] != self.layer_kinds[i - 1])
            .map(|i| i as u32)
            .filter(|i| !self.start_relative_layer_counts.contains(i))
            .collect()
    }

    /// Ends at which a prefix can be cut without splitting an architectural
    /// run. The final layer is always valid because it leaves no suffix.
    ///
    /// This is the same boundary rule consumed by `snap_end_layer`; exposing
    /// the actual cut points lets the deploy UI use a discrete slider without
    /// guessing a model-specific multiple.
    pub fn valid_prefix_end_layers(&self) -> Vec<u32> {
        let total = self.layer_count();
        if total == 0 {
            return Vec::new();
        }
        let boundaries = self.arch_boundaries();
        let mut ends = if boundaries.is_empty() && self.restricted_slice_starts.is_none() {
            (0..total).collect::<Vec<_>>()
        } else {
            boundaries
                .into_iter()
                .filter(|boundary| *boundary > 0 && *boundary < total)
                .map(|boundary| boundary - 1)
                .collect::<Vec<_>>()
        };
        ends.push(total - 1);
        ends.sort_unstable();
        ends.dedup();
        ends
    }

    /// Whether a slice can begin at this layer without changing the kind
    /// sequence when the fatiador renumbers its blocks.
    pub fn is_valid_slice_start(&self, start: u32) -> bool {
        if start >= self.layer_count() {
            return false;
        }
        if start == 0 {
            return true;
        }
        let boundaries = self.arch_boundaries();
        (boundaries.is_empty() && self.restricted_slice_starts.is_none())
            || boundaries.contains(&start)
    }

    /// Uniform layer distance when the model's valid prefix cuts are evenly
    /// spaced. `None` means the architecture has irregular boundaries; callers
    /// should use `valid_prefix_end_layers` rather than inventing a step.
    pub fn prefix_cut_granularity(&self) -> Option<u32> {
        let ends = self.valid_prefix_end_layers();
        let mut distances = ends.windows(2).map(|window| window[1] - window[0]);
        let first = distances.next()?;
        let gcd = distances.fold(first, gcd_u32);
        (ends.windows(2).all(|window| window[1] - window[0] == gcd)).then_some(gcd)
    }

    /// Snap `end` so that `end + 1` falls on a valid slice-start position.
    ///
    /// Valid starts are layer 0, every architectural boundary, and `total_layers`
    /// (end of model). When a budget's natural cut falls mid-group, this walks
    /// back to the last boundary that is ≤ `end + 1` — the slice shrinks to stay
    /// whole-group, rather than handing split_model.py a cut it will refuse.
    pub fn snap_end_layer(&self, start: u32, end: u32, total: u32) -> u32 {
        if end + 1 >= total {
            return end;
        }
        let boundaries = self.arch_boundaries();
        if self.restricted_slice_starts.is_some() {
            // There is no shorter valid slice to fall back on here: the walk
            // back below keeps `end` when it finds nothing, which is a cut the
            // model does not allow.
            let next_start = end + 1;
            return boundaries
                .iter()
                .rev()
                .find(|&&boundary| boundary <= next_start && boundary > start)
                .or_else(|| boundaries.iter().find(|&&boundary| boundary > start))
                .map_or(total - 1, |&boundary| boundary - 1);
        }
        if boundaries.is_empty() {
            return end;
        }
        let next_start = end + 1;
        if let Some(&b) = boundaries
            .iter()
            .rev()
            .find(|&&b| b <= next_start && b > start)
        {
            b - 1
        } else {
            end
        }
    }

    /// Bytes a slice covering `start..=end` would occupy on disk.
    ///
    /// Mirrors the Rust splitter. For a layout read from a real GGUF this
    /// includes the rewritten header, tensor table, padding, global tensors,
    /// duplicated embeddings and the synthetic norm on non-final slices. A
    /// hand-built test layout retains the older payload-only fallback.
    pub fn slice_bytes(&self, start: u32, end: u32, is_last: bool) -> u64 {
        if let Some(serialized) = &self.serialized {
            return serialized.slice_bytes(start, end, is_last);
        }
        let lo = start as usize;
        let hi = (end as usize).min(self.layer_bytes.len().saturating_sub(1));
        let layers: u64 = self
            .layer_bytes
            .get(lo..=hi)
            .map(|s| s.iter().sum())
            .unwrap_or(0);
        layers + self.embedding_bytes + if is_last { self.head_bytes } else { 0 }
    }

    /// Bytes of real architecture-required head tensors repeated in each
    /// non-final slice. The synthetic identity norm is intentionally excluded:
    /// this reports the extra source payload an operator must keep on disk.
    pub fn required_head_bytes(&self) -> u64 {
        self.serialized
            .as_ref()
            .map(|serialized| {
                serialized
                    .tensors
                    .iter()
                    .filter(|tensor| {
                        is_required_non_final_tensor(&serialized.architecture, &tensor.name)
                    })
                    .map(|tensor| align_up_layout(tensor.bytes))
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Valida a cabeça arquitetural antes de oferecer ou escrever faixas. O
    /// layout já contém a tensor table completa, então a checagem é barata e
    /// não toca nos payloads dos tensores.
    pub fn validate_slice_ranges(&self, ranges: &[(u32, u32)]) -> Result<()> {
        let Some(serialized) = &self.serialized else {
            return Ok(());
        };
        let tensor_names: HashSet<&str> = serialized
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        let total_layers = self.layer_count();
        for &(start, end) in ranges {
            if start > end || end >= total_layers {
                bail!("intervalo de camadas inválido L{start}-{end}");
            }
            if self.restricted_slice_starts.is_some() && !self.is_valid_slice_start(start) {
                bail!(
                    "{} não pode começar um slice em L{start}: cortes válidos em {:?}",
                    serialized.architecture,
                    self.arch_boundaries()
                );
            }
            let is_last = end.checked_add(1) == Some(total_layers);
            validate_slice_tensors(&serialized.architecture, &tensor_names, is_last)?;
        }
        Ok(())
    }
}

/// Tensores que o llama.cpp cria fora de `blk.N.*` para esta arquitetura.
///
/// Os nomes vêm dos loaders arquiteturais do llama.cpp, onde são criados com
/// flags obrigatórias. Em particular, os tensores HC são uma transformação
/// arquitetural antes de `output_norm`, não uma projeção de saída opcional.
pub fn required_output_tensors(architecture: &str) -> &'static [&'static str] {
    match architecture {
        "qwen4exp" => &[
            "token_embd.weight",
            "output_hc_norm.weight",
            "output_hc_down.weight",
            "output_hc_up.weight",
        ],
        "deepseek4" => &[
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "output_hc_fn.weight",
            "output_hc_base.weight",
            "output_hc_scale.weight",
        ],
        "deepseek41" => &["token_embd.weight", "output_norm.weight", "output.weight"],
        _ => &[],
    }
}

/// Whether this architecture's loader ever creates `output_norm.weight`.
/// True for the generic dense-transformer contract and for `deepseek4`
/// (which lists it explicitly). `qwen4exp` is the one exception: its final
/// hyper-connection mixer (`output_hc_norm.weight`) carries that role
/// instead, and the loader never calls `create_tensor` for `output_norm`.
/// A non-final slice must not receive the synthetic identity norm that
/// every other architecture needs, or the physical tensor count in the file
/// outruns what the loader actually claims.
pub fn uses_output_norm(architecture: &str) -> bool {
    architecture != "qwen4exp"
}

pub fn is_output_tensor(architecture: &str, name: &str) -> bool {
    matches!(
        name,
        "output_norm.weight" | "output_norm.bias" | "output.weight"
    ) || required_output_tensors(architecture)
        .iter()
        .any(|required| *required == name && *required != "token_embd.weight")
}

/// Real required tensors copied into every non-final slice. `token_embd` is
/// selected by the common embedding rule, while `output_norm` keeps the
/// established synthetic identity used by trunk slices.
pub fn is_required_non_final_tensor(architecture: &str, name: &str) -> bool {
    name != "token_embd.weight"
        && name != "output_norm.weight"
        && required_output_tensors(architecture)
            .iter()
            .any(|required| *required == name)
}

/// Verifica o contrato exato do loader para uma fatia candidata.
pub fn validate_slice_tensors(
    architecture: &str,
    tensor_names: &HashSet<&str>,
    is_last: bool,
) -> Result<()> {
    let required = required_output_tensors(architecture);
    if required.is_empty() {
        return Ok(());
    }

    if !is_last {
        let missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|name| *name != "output_norm.weight" && !tensor_names.contains(name))
            .collect();
        if !missing.is_empty() {
            bail!(
                "fatiamento da arquitetura '{architecture}' recusado para faixa não final: não foi possível copiar da origem os tensores obrigatórios {}",
                missing.join(", ")
            );
        }
        return Ok(());
    }

    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|name| !tensor_names.contains(name))
        .collect();
    if !missing.is_empty() {
        bail!(
            "fatiamento da arquitetura '{architecture}' recusado: a fatia final exige os tensores não-bloco {}, mas a origem não os contém",
            missing.join(", ")
        );
    }
    Ok(())
}

fn gcd_u32(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn align_up_layout(n: u64) -> u64 {
    (n + 31) & !31
}

fn tensor_header_size_layout(name: &str, dims: &[u64]) -> u64 {
    8 + name.len() as u64 + 4 + 8 * dims.len() as u64 + 4 + 8
}

fn remap_layout_name(name: &str, start: u32) -> String {
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some(dot) = rest.find('.') else {
        return name.to_string();
    };
    let Ok(index) = rest[..dot].parse::<u32>() else {
        return name.to_string();
    };
    format!("blk.{}{}", index - start, &rest[dot..])
}

fn layout_block_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    rest[..dot].parse().ok()
}

fn layout_wants_tensor(
    name: &str,
    architecture: &str,
    start: u32,
    end: u32,
    is_last: bool,
    ple_layers: &[u32],
) -> bool {
    if let Some(index) = layout_block_index(name) {
        return (start..=end).contains(&index);
    }
    match name {
        "token_embd.weight" => true,
        "per_layer_token_embd.weight"
            if matches!(architecture, "qwen4exp" | "deepseek41")
                && ple_layers
                    .iter()
                    .any(|&layer| (start..=end).contains(&layer)) =>
        {
            true
        }
        _ if is_required_non_final_tensor(architecture, name) => true,
        _ if is_output_tensor(architecture, name) => is_last,
        n if n.contains("rope") || n.contains("pos_embd") => true,
        _ => false,
    }
}

impl SerializedLayout {
    fn encoded_u32_size(key: &str) -> u64 {
        8 + key.len() as u64 + 4 + 4
    }

    fn encoded_u32_array_size(key: &str, count: u32) -> u64 {
        8 + key.len() as u64 + 4 + 4 + 8 + 4 * count as u64
    }

    fn kv_bytes(&self, start: u32, end: u32, is_last: bool) -> u64 {
        let slice_layers = end.saturating_sub(start).saturating_add(1);
        let pom_prefix = format!("{}.", crate::GGUF_PREFIX);
        let mut bytes = 0u64;
        let mut had_recurrent_key = false;

        for entry in &self.kv {
            if entry.key == crate::gguf_key("original_block_count")
                || entry.key.starts_with(&pom_prefix)
            {
                continue;
            }
            if entry.key.ends_with(".attention.recurrent_layers") {
                had_recurrent_key = true;
                continue;
            }
            if matches!(
                layer_count_semantics(&entry.key),
                Some(LayerCountSemantics::FromStart)
            ) {
                bytes += Self::encoded_u32_size(&entry.key);
                continue;
            }
            if matches!(
                layer_count_semantics(&entry.key),
                Some(LayerCountSemantics::FromEnd)
            ) && !is_last
            {
                bytes += Self::encoded_u32_size(&entry.key);
                continue;
            }
            if entry.key.ends_with(".block_count") {
                bytes += Self::encoded_u32_size(&entry.key);
                continue;
            }
            if entry.key == DEEPSEEK41_CANDIDATE_SOURCE_KEY
                && self.architecture == "deepseek41"
                && self
                    .deepseek41_candidate_source
                    .is_some_and(|layer| !(start..=end).contains(&layer))
            {
                continue;
            }
            if let Some(selected) = self.layer_selectors.get(&entry.key) {
                let local_count = selected
                    .iter()
                    .filter(|&&layer| (start..=end).contains(&layer))
                    .count();
                bytes += Self::encoded_u32_array_size(&entry.key, local_count as u32);
                continue;
            }
            if let Some(array) = entry.array {
                if !is_structural_array(&entry.key, &self.architecture)
                    && array.count >= self.original_block_count as u64
                {
                    if let Some(element_size) = scalar_size(array.element_type) {
                        bytes += 8
                            + entry.key.len() as u64
                            + 4
                            + 4
                            + 8
                            + element_size * slice_layers as u64;
                        continue;
                    }
                }
            }
            bytes += entry.raw_bytes;
        }

        // The splitter derives this array from actual ssm_conv1d tensors. A
        // source that has the key but no such tensor gets no replacement.
        if had_recurrent_key || self.has_recurrent_tensors() {
            bytes += Self::encoded_u32_array_size(
                &format!("{}.attention.recurrent_layers", self.architecture),
                slice_layers,
            );
        }
        bytes += Self::encoded_u32_size(&crate::gguf_key("original_block_count"));
        bytes += Self::encoded_u32_size(&crate::gguf_key("start_layer"));
        bytes += Self::encoded_u32_size(&crate::gguf_key("end_layer"));
        bytes
    }

    fn has_recurrent_tensors(&self) -> bool {
        self.tensors
            .iter()
            .any(|tensor| tensor.name.ends_with(".ssm_conv1d.weight"))
    }

    fn slice_bytes(&self, start: u32, end: u32, is_last: bool) -> u64 {
        let slice_layers = end.saturating_sub(start).saturating_add(1);
        let mut headers = 0u64;
        let mut payload = 0u64;
        let mut has_output_norm = false;

        for tensor in &self.tensors {
            if !layout_wants_tensor(
                &tensor.name,
                &self.architecture,
                start,
                end,
                is_last,
                &self.ple_layers,
            ) {
                continue;
            }
            headers +=
                tensor_header_size_layout(&remap_layout_name(&tensor.name, start), &tensor.dims);
            payload += align_up_layout(tensor.bytes);
            has_output_norm |= tensor.name == "output_norm.weight";
        }

        // Non-final slices carry an F32 identity norm exactly as splitter::split
        // does. The source norm's dtype is deliberately not reused here.
        // `qwen4exp` never creates `output_norm.weight` (see `uses_output_norm`),
        // so it must not be charged or emitted for that architecture.
        if !is_last && uses_output_norm(&self.architecture) {
            let hidden = self.hidden_dim as u64;
            headers += tensor_header_size_layout("output_norm.weight", &[hidden]);
            payload += align_up_layout(hidden * 4);
        }

        let _ = (slice_layers, has_output_norm);
        let meta_size = 4 + 4 + 8 + 8 + self.kv_bytes(start, end, is_last) + headers;
        align_up_layout(meta_size) + payload
    }
}

/// Read the tensor table and total up the bytes each transformer block costs.
///
/// Reads the header only — the tensor *data* is never touched.
pub fn read_layout(path: &Path) -> Result<GgufLayout> {
    let source = crate::multipart::GgufSource::open(path)?;
    let source_file_bytes = source.source_bytes();
    let first_map = source
        .parts()
        .first()
        .ok_or_else(|| anyhow::anyhow!("GGUF sem partes"))?
        .map();
    let mut r = Reader {
        inner: Cursor::new(first_map),
    };

    let magic = r.u32()?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (magic {magic:#x})");
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        bail!("unsupported GGUF version {version} (expected 2 or 3)");
    }
    let n_tensors = r.u64()? as usize;
    let n_kv = r.u64()? as usize;
    if n_tensors > 1_000_000 || n_kv > 100_000 {
        bail!("implausible GGUF header (tensors={n_tensors}, kv={n_kv})");
    }

    let mut layout = GgufLayout::default();
    layout.source_file_bytes = source_file_bytes;
    let mut architecture = "llama".to_string();
    let mut hidden_dim = 4096u32;
    let mut source_block_count = None;
    let mut layer_selectors: HashMap<String, Vec<u32>> = HashMap::new();
    let mut deepseek41_candidate_source = None;
    let mut kv = Vec::with_capacity(n_kv);
    for _ in 0..n_kv {
        let kv_start = r.position()?;
        let key = r.string()?;
        let vtype = r.u32()?;
        let mut scalar_u32 = None;
        let mut array = None;
        match vtype {
            T_STRING => {
                let value = r.string()?;
                if key == "general.architecture" {
                    architecture = value;
                }
            }
            T_ARRAY => {
                let element_type = r.u32()?;
                let count = r.u64()?;
                // By the key's own namespace: GGUF does not promise that
                // `general.architecture` comes before this array.
                if is_layer_selector_array(&key, key.split('.').next().unwrap_or_default()) {
                    if count > 100_000 {
                        bail!("implausible layer selector array '{key}' length {count}");
                    }
                    let selected = (0..count)
                        .map(|_| {
                            r.value_as_u32(element_type)?.ok_or_else(|| {
                                anyhow::anyhow!(
                                    "metadata '{}' tem elementos que não são inteiros não negativos",
                                    key
                                )
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    layer_selectors.insert(key.clone(), selected);
                } else {
                    r.skip_array_payload(element_type, count)?;
                }
                array = Some(ArrayShape {
                    element_type,
                    count,
                });
            }
            _ => {
                scalar_u32 = r.value_as_u32(vtype)?;
                if key == DEEPSEEK41_CANDIDATE_SOURCE_KEY {
                    deepseek41_candidate_source = scalar_u32;
                }
            }
        }
        if matches!(
            layer_count_semantics(&key),
            Some(LayerCountSemantics::FromStart)
        ) {
            let Some(count) = scalar_u32 else {
                bail!(
                    "metadata '{key}' conta camadas desde o início, mas não é um inteiro escalar não negativo que caiba em u32"
                );
            };
            layout.start_relative_layer_counts.push(count);
        }
        if is_nextn_predict_layers_key(&key) {
            layout.nextn_predict_layers = scalar_u32.unwrap_or(0);
        }
        if key.ends_with(".block_count") {
            source_block_count = scalar_u32;
        }
        if key.ends_with(".embedding_length") {
            hidden_dim = scalar_u32.unwrap_or(hidden_dim);
        }
        kv.push(LayoutKv {
            key,
            raw_bytes: r.position()? - kv_start,
            array,
        });
    }

    let layout_tensors: Vec<LayoutTensor> = source
        .tensors()
        .iter()
        .map(|tensor| {
            if ggml_type_info(tensor.dtype).is_none() {
                return Err(anyhow::anyhow!(
                    "dtype GGML {} desconhecido no tensor '{}'; estimativa de tamanho recusada",
                    tensor.dtype,
                    tensor.name
                ));
            }
            Ok(LayoutTensor {
                name: tensor.name.clone(),
                dims: tensor.dims.clone(),
                bytes: tensor.byte_size,
            })
        })
        .collect::<Result<_>>()?;

    let mut layers: HashMap<u32, u64> = HashMap::new();
    let mut kinds: HashMap<u32, Vec<String>> = HashMap::new();
    let mut max_layer: i64 = -1;
    for tensor in &layout_tensors {
        let bytes = tensor.bytes;
        layout.total_bytes += bytes;
        if tensor.name.split('.').any(|part| part == "nextn") {
            layout.has_nextn_tensors = true;
        }

        if let Some(n) = block_index(&tensor.name) {
            *layers.entry(n).or_insert(0) += bytes;
            if let Some(suffix) = tensor.name.splitn(3, '.').nth(2) {
                kinds.entry(n).or_default().push(suffix.to_string());
            }
            max_layer = max_layer.max(n as i64);
        } else if tensor.name == "token_embd.weight" {
            layout.embedding_bytes += bytes;
        } else if is_output_tensor(&architecture, &tensor.name) {
            layout.head_bytes += bytes;
        } else {
            layout.other_bytes += bytes;
        }
    }

    if max_layer >= 0 {
        layout.layer_bytes = (0..=max_layer as u32)
            .map(|n| layers.get(&n).copied().unwrap_or(0))
            .collect();
        layout.layer_kinds = (0..=max_layer as u32)
            .map(|n| {
                let mut v = kinds.remove(&n).unwrap_or_default();
                v.sort(); // tensor order in the file is not meaningful here
                v
            })
            .collect();
    }

    let original_block_count = source_block_count.unwrap_or_else(|| {
        if max_layer >= 0 {
            max_layer as u32 + 1
        } else {
            0
        }
    });
    let ple_layers = layer_selectors
        .get(&format!("{architecture}.ple.layers"))
        .cloned()
        .unwrap_or_default();
    if matches!(architecture.as_str(), "qwen4exp" | "deepseek41") {
        if let Some(&layer) = ple_layers
            .iter()
            .find(|&&layer| layer >= original_block_count)
        {
            bail!(
                "{architecture}.ple.layers contém a camada {layer}, mas a origem declara apenas {original_block_count} blocos"
            );
        }
        if !ple_layers.is_empty()
            && !layout_tensors
                .iter()
                .any(|tensor| tensor.name == "per_layer_token_embd.weight")
        {
            bail!(
                "a origem {architecture} declara PLE, mas não contém per_layer_token_embd.weight"
            );
        }
    }
    if architecture == "deepseek41" {
        let selected = |name: &str| {
            layer_selectors
                .get(&format!("deepseek41.attention.{name}"))
                .map_or(&[][..], Vec::as_slice)
        };
        layout.restricted_slice_starts = Some(deepseek41_slice_starts(
            selected("kv_source_layers"),
            selected("indexer.source_layers"),
            deepseek41_candidate_source,
            original_block_count,
        ));
    }
    let arch_prefix = format!("{architecture}.");
    for entry in &kv {
        let Some(array) = entry.array else {
            continue;
        };
        if !entry.key.starts_with(&arch_prefix)
            || entry.key.contains(".rope.")
            || is_structural_array(&entry.key, &architecture)
        {
            continue;
        }
        if array.count != 1 && array.count < original_block_count as u64 {
            bail!(
                "array metadata '{}' tem comprimento {}; esperado 1 ou pelo menos {} (contagem original de blocos)",
                entry.key,
                array.count,
                original_block_count
            );
        }
        if array.count >= original_block_count as u64 && scalar_size(array.element_type).is_none() {
            bail!(
                "array metadata '{}' tem elementos aninhados e não pode ser fatiado por camada",
                entry.key
            );
        }
    }
    layout.serialized = Some(SerializedLayout {
        architecture,
        hidden_dim,
        original_block_count,
        ple_layers,
        layer_selectors,
        deepseek41_candidate_source,
        kv,
        tensors: layout_tensors,
    });

    Ok(layout)
}

/// Lê a evidência completa usada pelas provas de capacidades do GGUF.
// O catálogo faz a mesma leitura de forma leve durante o enriquecimento; este
// helper fica compilado em testes para manter a prova estrutural reutilizável.
#[cfg(test)]
pub fn read_capability_evidence(path: &Path) -> Result<ModelCapabilityEvidence> {
    let info = read_info(path)?;
    let mtp = match (info.nextn_predict_layers, has_nextn_tensors(path)?) {
        // The tensor table is the strongest evidence: it is the actual
        // predictor payload, and covers producers that omit or rename the
        // architecture-prefixed hyperparameter.
        (_, true) => Some(true),
        (Some(_), false) => Some(false),
        (None, false) => None,
    };
    Ok(ModelCapabilityEvidence {
        thinking: info.thinking_capability(),
        mtp,
    })
}

fn is_nextn_predict_layers_key(key: &str) -> bool {
    matches!(
        layer_count_semantics(key),
        Some(LayerCountSemantics::FromEnd)
    )
}

pub fn layer_count_semantics(key: &str) -> Option<LayerCountSemantics> {
    if is_metadata_key(key, "nextn_predict_layers") {
        Some(LayerCountSemantics::FromEnd)
    } else if is_metadata_key(key, "leading_dense_block_count")
        || is_metadata_key(key, "hash_layer_count")
        || is_metadata_key(key, "n_deepstack_layers")
    {
        Some(LayerCountSemantics::FromStart)
    } else {
        None
    }
}

fn is_metadata_key(key: &str, suffix: &str) -> bool {
    key == suffix
        || key
            .strip_suffix(suffix)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

/// `blk.17.attn_q.weight` → `Some(17)`.
fn block_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let idx = rest.split('.').next()?;
    idx.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_architecture_requires_an_explicit_expert_count() {
        let dense = GgufInfo {
            architecture: Some("llama".into()),
            ..GgufInfo::default()
        };
        assert!(matches!(
            dense.model_architecture(),
            ModelArchitecture::Dense { .. }
        ));

        let moe = GgufInfo {
            architecture: Some("qwen3moe".into()),
            expert_count: Some(128),
            expert_used_count: Some(8),
            ..GgufInfo::default()
        };
        assert!(moe.model_architecture().is_moe());
        assert!(matches!(
            moe.model_architecture(),
            ModelArchitecture::Moe {
                expert_count: 128,
                expert_used_count: Some(8),
                ..
            }
        ));
    }

    #[test]
    fn parses_block_indices() {
        assert_eq!(block_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(block_index("blk.31.ffn_up.weight"), Some(31));
        assert_eq!(block_index("token_embd.weight"), None);
        assert_eq!(block_index("blk.x.weight"), None);
    }

    #[test]
    fn slice_bytes_accounts_for_embedding_and_head() {
        let layout = GgufLayout {
            layer_bytes: vec![100; 4],
            embedding_bytes: 50,
            head_bytes: 30,
            other_bytes: 0,
            total_bytes: 480,
            source_file_bytes: 0,
            layer_kinds: vec![],
            start_relative_layer_counts: vec![],
            nextn_predict_layers: 0,
            has_nextn_tensors: false,
            serialized: None,
            restricted_slice_starts: None,
        };
        // Middle slice: 2 layers + embedding.
        assert_eq!(layout.slice_bytes(0, 1, false), 250);
        // Last slice also carries the output head.
        assert_eq!(layout.slice_bytes(2, 3, true), 280);
    }

    #[test]
    fn valid_prefix_cuts_follow_dense_and_hybrid_layouts() {
        let dense = GgufLayout {
            layer_bytes: vec![100; 4],
            layer_kinds: vec![vec!["dense".into()]; 4],
            ..Default::default()
        };
        assert_eq!(dense.valid_prefix_end_layers(), vec![0, 1, 2, 3]);
        assert_eq!(dense.prefix_cut_granularity(), Some(1));

        let hybrid = GgufLayout {
            layer_bytes: vec![100; 8],
            layer_kinds: vec![
                vec!["mamba".into()],
                vec!["mamba".into()],
                vec!["attention".into()],
                vec!["attention".into()],
                vec!["mamba".into()],
                vec!["mamba".into()],
                vec!["attention".into()],
                vec!["attention".into()],
            ],
            ..Default::default()
        };
        assert_eq!(hybrid.valid_prefix_end_layers(), vec![1, 3, 5, 7]);
        assert_eq!(hybrid.prefix_cut_granularity(), Some(2));
    }

    #[test]
    fn dense_prefix_declared_by_leading_dense_count_is_not_a_boundary() {
        let dense = vec!["ffn_up.weight".to_string()];
        let moe = vec!["ffn_up_exps.weight".to_string()];
        let mut layer_kinds = vec![dense.clone(), dense];
        layer_kinds.extend(std::iter::repeat_n(moe, 6));
        let mut layout = GgufLayout {
            layer_bytes: vec![100; 8],
            layer_kinds,
            ..Default::default()
        };
        assert_eq!(layout.valid_prefix_end_layers(), vec![1, 7]);

        layout.start_relative_layer_counts = vec![2];
        assert!(layout.arch_boundaries().is_empty());
        assert_eq!(layout.valid_prefix_end_layers(), (0..8).collect::<Vec<_>>());
        assert!(layout.is_valid_slice_start(1));
        assert_eq!(layout.snap_end_layer(0, 4, 8), 4);

        // A count that does not match the kind change explains nothing.
        layout.start_relative_layer_counts = vec![3];
        assert_eq!(layout.arch_boundaries(), vec![2]);
    }

    #[test]
    fn rejects_non_gguf() {
        let mut p = std::env::temp_dir();
        p.push("pom-not-a-gguf.bin");
        std::fs::write(&p, b"hello world, definitely not gguf").unwrap();
        assert!(read_info(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }

    /// A header-only GGUF: `n_tensors = 0` and whatever KV pairs are given.
    fn header_only_gguf(kvs: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, vtype, v) in kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&vtype.to_le_bytes());
            if *vtype == T_STRING {
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            }
            b.extend_from_slice(v);
        }
        b.resize((b.len() + 31) & !31, 0);
        b
    }

    #[test]
    fn accepts_nonnegative_signed_integer_metadata() {
        let nonce = uuid::Uuid::new_v4().simple();
        let path = std::env::temp_dir().join(format!("pom-signed-gguf-{nonce}.gguf"));
        let bytes = header_only_gguf(&[
            ("split.tensors.count", T_INT32, 0i32.to_le_bytes().to_vec()),
            ("llama.block_count", T_INT32, 32i32.to_le_bytes().to_vec()),
        ]);
        std::fs::write(&path, bytes).unwrap();

        let info = read_info(&path).unwrap();

        assert_eq!(info.split_tensors_count, Some(0));
        assert_eq!(info.block_count, Some(32));
        let layout = read_layout(&path).unwrap();
        assert_eq!(layout.serialized.as_ref().unwrap().original_block_count, 32);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_negative_signed_integer_metadata() {
        let nonce = uuid::Uuid::new_v4().simple();
        let path = std::env::temp_dir().join(format!("pom-negative-gguf-{nonce}.gguf"));
        let bytes = header_only_gguf(&[(
            "split.tensors.count",
            T_INT32,
            (-1i32).to_le_bytes().to_vec(),
        )]);
        std::fs::write(&path, bytes).unwrap();

        assert!(read_info(&path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reads_model_identity_and_multimodal_projection_dimension() {
        let string = |value: &str| value.as_bytes().to_vec();
        let path = std::env::temp_dir().join(format!(
            "pom-mmproj-metadata-{}.gguf",
            uuid::Uuid::new_v4().simple()
        ));
        let bytes = header_only_gguf(&[
            ("general.name", T_STRING, string("Ornith 1.5")),
            ("general.basename", T_STRING, string("ornith-1.5")),
            (
                "general.base_model.0.name",
                T_STRING,
                string("publisher/ornith-1.5"),
            ),
            (
                "llama.embedding_length",
                T_UINT32,
                3072u32.to_le_bytes().to_vec(),
            ),
            (
                "clip.vision.projection_dim",
                T_UINT32,
                3072u32.to_le_bytes().to_vec(),
            ),
        ]);
        std::fs::write(&path, bytes).unwrap();

        let info = read_info(&path).unwrap();

        assert_eq!(info.embedding_length, Some(3072));
        assert_eq!(info.projector_projection_dim, Some(3072));
        assert_eq!(
            info.identity_names,
            vec!["Ornith 1.5", "ornith-1.5", "publisher/ornith-1.5"]
        );
        let _ = std::fs::remove_file(path);
    }

    fn gguf_with_tensors(kvs: &[(&str, u32, Vec<u8>)], names: &[&str]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&(names.len() as u64).to_le_bytes());
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, vtype, v) in kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&vtype.to_le_bytes());
            if *vtype == T_STRING {
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            }
            b.extend_from_slice(v);
        }
        for name in names {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
        }
        b
    }

    #[test]
    fn mtp_is_proven_by_real_nextn_tensors_even_without_metadata() {
        let dir = std::env::temp_dir();
        let path = dir.join("pom-mtp-structural-evidence.gguf");
        std::fs::write(
            &path,
            gguf_with_tensors(
                &[],
                &["blk.64.nextn.eh_proj.weight", "blk.64.nextn.hnorm.weight"],
            ),
        )
        .unwrap();

        assert_eq!(read_capability_evidence(&path).unwrap().mtp, Some(true));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn canonical_qwen35_nextn_metadata_and_tensor_names_prove_mtp() {
        let dir = std::env::temp_dir();
        let path = dir.join("pom-mtp-qwen35-canonical.gguf");
        std::fs::write(
            &path,
            gguf_with_tensors(
                &[(
                    "qwen35.nextn_predict_layers",
                    T_UINT32,
                    1u32.to_le_bytes().to_vec(),
                )],
                &[
                    "blk.64.nextn.eh_proj.weight",
                    "blk.64.nextn.enorm.weight",
                    "blk.64.nextn.hnorm.weight",
                    "blk.64.nextn.shared_head_norm.weight",
                ],
            ),
        )
        .unwrap();

        let info = read_info(&path).unwrap();
        assert_eq!(info.nextn_predict_layers, Some(1));
        assert_eq!(read_capability_evidence(&path).unwrap().mtp, Some(true));
        let _ = std::fs::remove_file(path);
    }

    fn string_array(values: &[&str]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&T_STRING.to_le_bytes());
        b.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            b.extend_from_slice(&(value.len() as u64).to_le_bytes());
            b.extend_from_slice(value.as_bytes());
        }
        b
    }

    #[test]
    fn thinking_capability_requires_real_template_and_vocabulary_tokens() {
        let dir = std::env::temp_dir();
        let supported = dir.join("pom-thinking-supported.gguf");
        std::fs::write(
            &supported,
            header_only_gguf(&[
                (
                    "tokenizer.chat_template",
                    T_STRING,
                    "{% if enable_thinking %}<|think|>{% endif %}".into(),
                ),
                (
                    "tokenizer.ggml.tokens",
                    T_ARRAY,
                    string_array(&["<|think|>"]),
                ),
            ]),
        )
        .unwrap();
        assert_eq!(
            read_info(&supported).unwrap().thinking_capability(),
            Some(true)
        );

        let unsupported = dir.join("pom-thinking-unsupported.gguf");
        std::fs::write(
            &unsupported,
            header_only_gguf(&[
                (
                    "tokenizer.chat_template",
                    T_STRING,
                    "{% if enable_thinking %}<|think|>{% endif %}".into(),
                ),
                ("tokenizer.ggml.tokens", T_ARRAY, string_array(&["foo"])),
            ]),
        )
        .unwrap();
        assert_eq!(
            read_info(&unsupported).unwrap().thinking_capability(),
            Some(false)
        );

        let unknown = dir.join("pom-thinking-unknown.gguf");
        std::fs::write(&unknown, header_only_gguf(&[])).unwrap();
        assert_eq!(read_info(&unknown).unwrap().thinking_capability(), None);

        let _ = std::fs::remove_file(supported);
        let _ = std::fs::remove_file(unsupported);
        let _ = std::fs::remove_file(unknown);
    }

    fn u32_array(values: &[u32]) -> Vec<u8> {
        let mut out = T_UINT32.to_le_bytes().to_vec();
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    #[test]
    fn read_info_preserves_scalar_and_per_layer_kv_geometry() {
        let dir = std::env::temp_dir();
        let mut p = dir.clone();
        p.push("pom-kv-geometry.gguf");
        std::fs::write(
            &p,
            header_only_gguf(&[
                (
                    "llama.attention.head_count_kv",
                    T_UINT32,
                    8u32.to_le_bytes().to_vec(),
                ),
                ("llama.attention.key_length", T_ARRAY, u32_array(&[128, 64])),
                (
                    "llama.attention.value_length",
                    T_UINT32,
                    128u32.to_le_bytes().to_vec(),
                ),
            ]),
        )
        .unwrap();

        let info = read_info(&p).unwrap();
        assert_eq!(info.attention_head_count_kv, Some(vec![8]));
        assert_eq!(info.attention_key_length, Some(vec![128, 64]));
        assert_eq!(info.attention_value_length, Some(vec![128]));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn read_layout_sees_nextn_predict_layers() {
        let dir = std::env::temp_dir();
        let mut p = dir.clone();
        p.push("pom-nextn-layout.gguf");
        std::fs::write(
            &p,
            header_only_gguf(&[
                ("general.architecture", T_STRING, "qwen35".into()),
                (
                    "qwen35.nextn_predict_layers",
                    T_UINT32,
                    1u32.to_le_bytes().to_vec(),
                ),
            ]),
        )
        .unwrap();
        let layout = read_layout(&p).unwrap();
        assert_eq!(layout.nextn_predict_layers, 1);
        let _ = std::fs::remove_file(&p);

        // Absent or non-scalar means "no predictor", not a broken plan.
        let mut q = dir.clone();
        q.push("pom-no-nextn-layout.gguf");
        std::fs::write(
            &q,
            header_only_gguf(&[("general.architecture", T_STRING, "llama".into())]),
        )
        .unwrap();
        let layout = read_layout(&q).unwrap();
        assert_eq!(layout.nextn_predict_layers, 0);
        let _ = std::fs::remove_file(&q);
    }

    #[test]
    fn classifies_layer_count_origins_by_metadata_suffix() {
        for key in [
            "alpha.leading_dense_block_count",
            "beta.hash_layer_count",
            "gamma.n_deepstack_layers",
        ] {
            assert_eq!(
                layer_count_semantics(key),
                Some(LayerCountSemantics::FromStart),
                "a contagem usada com índice local deve ser rebased: {key}"
            );
        }
        assert_eq!(
            layer_count_semantics("delta.nextn_predict_layers"),
            Some(LayerCountSemantics::FromEnd)
        );
        assert_eq!(layer_count_semantics("epsilon.block_count"), None);
    }

    fn u32_array_kv(values: &[u32]) -> Vec<u8> {
        let mut payload = T_UINT32.to_le_bytes().to_vec();
        payload.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        payload
    }

    /// Eight blocks shaped like the real DeepSeek-V4.1: the short lists name
    /// layers, the long Engram lists describe the table, not the blocks.
    fn deepseek41_layout_path(ple: &[u32], candidate: u32) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "pom-deepseek41-{}.gguf",
            uuid::Uuid::new_v4().simple()
        ));
        let mut names: Vec<String> = (0..8)
            .map(|layer| format!("blk.{layer}.attn.weight"))
            .collect();
        for name in [
            "per_layer_token_embd.weight",
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
        ] {
            names.push(name.into());
        }
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut bytes = gguf_with_tensors(
            &[
                ("general.architecture", T_STRING, b"deepseek41".to_vec()),
                (
                    "deepseek41.block_count",
                    T_UINT32,
                    8u32.to_le_bytes().to_vec(),
                ),
                (
                    "deepseek41.attention.kv_source_layers",
                    T_ARRAY,
                    u32_array_kv(&[2, 4]),
                ),
                (
                    "deepseek41.attention.indexer.source_layers",
                    T_ARRAY,
                    u32_array_kv(&[2, 4, 6]),
                ),
                (
                    "deepseek41.attention.indexer.candidate_source_layer",
                    T_UINT32,
                    candidate.to_le_bytes().to_vec(),
                ),
                ("deepseek41.ple.layers", T_ARRAY, u32_array_kv(ple)),
                (
                    "deepseek41.ple.layer_multipliers",
                    T_ARRAY,
                    u32_array_kv(&[1, 2, 3, 4]),
                ),
                (
                    "deepseek41.ple.head_offsets",
                    T_ARRAY,
                    u32_array_kv(&[0; 12]),
                ),
                (
                    "deepseek41.ple.head_vocab_sizes",
                    T_ARRAY,
                    u32_array_kv(&[7; 12]),
                ),
                (
                    "deepseek41.engram.token_map",
                    T_ARRAY,
                    u32_array_kv(&[0; 16]),
                ),
            ],
            &names,
        );
        bytes.resize(((bytes.len() + 31) & !31) + 32, 0);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn deepseek41_only_cuts_where_every_cross_layer_source_stays_reachable() {
        // KV sources 2 and 4, indexer sources 2, 4 and 6, Engram at 1 and 3,
        // candidate source 4: the cuts are the shared sources up to 4.
        let path = deepseek41_layout_path(&[1, 3], 4);
        let layout = read_layout(&path).unwrap();
        let _ = std::fs::remove_file(path);

        assert_eq!(layout.arch_boundaries(), vec![2, 4]);
        assert_eq!(layout.valid_prefix_end_layers(), vec![1, 3, 7]);
        assert!(
            layout.is_valid_slice_start(2),
            "Engram at 3 gets the token ids beside the state"
        );
        assert!(layout.is_valid_slice_start(4));
        assert!(
            !layout.is_valid_slice_start(3),
            "layer 3 reads layer 2's compressed KV"
        );
        assert!(!layout.is_valid_slice_start(6), "past the candidate source");
        assert_eq!(layout.snap_end_layer(0, 5, 8), 3);
        assert!(layout
            .validate_slice_ranges(&[(0, 1), (2, 3), (4, 7)])
            .is_ok());
        let error = layout
            .validate_slice_ranges(&[(0, 2), (3, 7)])
            .unwrap_err()
            .to_string();
        assert!(error.contains("L3"), "{error}");
    }

    #[test]
    fn deepseek41_engram_layers_do_not_hold_the_cut_back() {
        // Engram at 5, after the candidate source at 4: before, no cut was left.
        let path = deepseek41_layout_path(&[1, 5], 4);
        let layout = read_layout(&path).unwrap();
        let _ = std::fs::remove_file(path);

        assert_eq!(layout.arch_boundaries(), vec![2, 4]);
        assert!(layout.is_valid_slice_start(4));
        assert!(layout.is_valid_slice_start(0));
    }

    #[test]
    fn deepseek41_cuts_at_every_shared_source_up_to_the_candidate_source() {
        // DeepSeek-V4.1-Flash: KV sources 2, 8, 14, 20; Engram at 1 and 14; candidate source 20.
        assert_eq!(
            deepseek41_slice_starts(
                &[2, 8, 14, 20],
                &[2, 8, 14, 20, 24, 28, 32, 36],
                Some(20),
                40
            ),
            vec![2, 8, 14, 20]
        );
        // Nothing past the candidate source, nor a source the indexer does not share.
        assert_eq!(
            deepseek41_slice_starts(&[2, 8, 30], &[2, 30], Some(20), 40),
            vec![2]
        );
    }

    #[test]
    fn deepseek41_engram_table_only_counts_in_the_slice_with_engram_layers() {
        let path = deepseek41_layout_path(&[1, 3], 4);
        let layout = read_layout(&path).unwrap();
        let _ = std::fs::remove_file(path);
        let serialized = layout.serialized.as_ref().unwrap();

        assert!(layout_wants_tensor(
            "per_layer_token_embd.weight",
            "deepseek41",
            0,
            3,
            false,
            &serialized.ple_layers,
        ));
        assert!(!layout_wants_tensor(
            "per_layer_token_embd.weight",
            "deepseek41",
            4,
            7,
            true,
            &serialized.ple_layers,
        ));
    }

    #[test]
    fn deepseek41_requires_its_real_output_tensors_in_every_slice() {
        assert_eq!(
            required_output_tensors("deepseek41"),
            &["token_embd.weight", "output_norm.weight", "output.weight"]
        );
        assert!(is_required_non_final_tensor("deepseek41", "output.weight"));
    }

    #[test]
    fn rejects_non_scalar_from_start_layer_count_metadata() {
        let nonce = uuid::Uuid::new_v4().simple();
        let path = std::env::temp_dir().join(format!("pom-layer-count-{nonce}.gguf"));
        let bytes = header_only_gguf(&[("deepseek4.hash_layer_count", T_STRING, b"3".to_vec())]);
        std::fs::write(&path, bytes).unwrap();

        let error = read_layout(&path).unwrap_err().to_string();
        assert!(error.contains("deepseek4.hash_layer_count"), "{error}");
        assert!(error.contains("inteiro escalar"), "{error}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn moe_detection_uses_real_expert_metadata_not_architecture_name() {
        let dir = std::env::temp_dir();
        let moe = dir.join("pom-moe-metadata.gguf");
        std::fs::write(
            &moe,
            header_only_gguf(&[
                ("general.architecture", T_STRING, "qwen3moe".into()),
                (
                    "qwen3moe.expert_count",
                    T_UINT32,
                    128u32.to_le_bytes().to_vec(),
                ),
            ]),
        )
        .unwrap();
        let info = read_info(&moe).unwrap();
        assert!(info.is_moe());

        let dense = dir.join("pom-dense-with-moe-name.gguf");
        std::fs::write(
            &dense,
            header_only_gguf(&[("general.architecture", T_STRING, "qwen3moe".into())]),
        )
        .unwrap();
        let info = read_info(&dense).unwrap();
        assert!(!info.is_moe(), "nome de arquitetura não substitui metadata");

        let _ = std::fs::remove_file(moe);
        let _ = std::fs::remove_file(dense);
    }

    /// Every type this project claims to measure, with the `(type_size,
    /// blck_size)` ggml itself reports. Read out of the llama.cpp shipped in
    /// `./venv` via `ggml_type_size`/`ggml_blck_size` — the point of writing
    /// them down again here is that a typo has to be made twice to survive.
    const GGML_TRUTH: &[(u32, u64, u64, &str)] = &[
        (0, 4, 1, "F32"),
        (1, 2, 1, "F16"),
        (2, 18, 32, "Q4_0"),
        (3, 20, 32, "Q4_1"),
        (6, 22, 32, "Q5_0"),
        (7, 24, 32, "Q5_1"),
        (8, 34, 32, "Q8_0"),
        (9, 36, 32, "Q8_1"),
        (10, 84, 256, "Q2_K"),
        (11, 110, 256, "Q3_K"),
        (12, 144, 256, "Q4_K"),
        (13, 176, 256, "Q5_K"),
        (14, 210, 256, "Q6_K"),
        (15, 292, 256, "Q8_K"),
        (16, 66, 256, "IQ2_XXS"),
        (17, 74, 256, "IQ2_XS"),
        (18, 98, 256, "IQ3_XXS"),
        (19, 50, 256, "IQ1_S"),
        (20, 18, 32, "IQ4_NL"),
        (21, 110, 256, "IQ3_S"),
        (22, 82, 256, "IQ2_S"),
        (23, 136, 256, "IQ4_XS"),
        (24, 1, 1, "I8"),
        (25, 2, 1, "I16"),
        (26, 4, 1, "I32"),
        (27, 8, 1, "I64"),
        (28, 8, 1, "F64"),
        (29, 56, 256, "IQ1_M"),
        (30, 2, 1, "BF16"),
        (34, 54, 256, "TQ1_0"),
        (35, 66, 256, "TQ2_0"),
        (39, 17, 32, "MXFP4"),
        (40, 36, 64, "NVFP4"),
        (41, 18, 128, "Q1_0"),
        (42, 18, 64, "Q2_0"),
    ];

    #[test]
    fn ggml_type_table_matches_ggml() {
        for &(code, bytes, elems, name) in GGML_TRUTH {
            assert_eq!(
                ggml_type_info(code),
                Some((bytes, elems)),
                "ggml type {code} ({name}) has the wrong block size"
            );
        }
    }

    /// The codes ggml removed, plus the ones whose layout we chose not to cite.
    /// They must stay unknown: a file using one has to be refused, and a size
    /// invented here would be spent slicing it into garbage instead.
    #[test]
    fn removed_and_uncited_ggml_types_stay_unknown() {
        for code in [4, 5, 31, 32, 33, 36, 37, 38, 99] {
            assert_eq!(ggml_type_info(code), None, "type {code} should be unknown");
        }
    }

    /// IQ3_S is what the captain's IQ3_M model is mostly made of, and the two
    /// numbers this pins are exactly the ones that were wrong: a 5120×248320
    /// embedding priced with IQ3_XXS's 66-byte block instead of IQ3_S's 110.
    #[test]
    fn tensor_bytes_per_type() {
        let bytes = |code: u32, elements: u64| {
            let (b, e) = ggml_type_info(code).expect("known type");
            elements / e * b
        };
        // token_embd.weight of the real IQ3_M model: 546 MB, not the 327 MB the
        // old table reported for it by handing IQ3_S a 66-byte block.
        assert_eq!(bytes(21, 5120 * 248320), 546_304_000);
        // BF16 output.weight — 2 bytes per element, not a 256-element block.
        assert_eq!(bytes(30, 5120 * 248320), 2_542_796_800);
        // A Q5_K layer, the entry that was booked at 64% of its true size
        // (11_534_336 here against the 7_340_032 the 112-byte block gave).
        assert_eq!(bytes(13, 4096 * 4096), 11_534_336);
        assert_eq!(bytes(12, 4096 * 4096), 9_437_184); // Q4_K, always correct
    }

    #[test]
    fn scalar_sizes_cover_all_scalar_types() {
        for t in [
            T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_UINT64,
            T_INT64, T_FLOAT64,
        ] {
            assert!(scalar_size(t).is_some(), "type {t} has no size");
        }
        assert!(scalar_size(T_STRING).is_none());
        assert!(scalar_size(T_ARRAY).is_none());
    }
}
