//! Shared, read-only view of a GGUF file or GGUF multipart set.
//!
//! A multipart GGUF repeats its metadata in every part but distributes the
//! tensor table and payloads across the files. This module maps the parts and
//! exposes one tensor table without creating a temporary monolithic file.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

const GGUF_MAGIC: u32 = 0x4655_4747;
const ALIGN: usize = 32;
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STR: u32 = 8;
const T_ARR: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

#[derive(Debug, Default, Clone, Copy)]
struct SplitFields {
    count: Option<u64>,
    no: Option<u64>,
    tensors_count: Option<u64>,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| anyhow::anyhow!("overflow ao ler cabeçalho GGUF"))?;
        if end > self.data.len() {
            bail!("GGUF truncado em byte {}", self.pos);
        }
        let value = &self.data[self.pos..end];
        self.pos = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let length = self.u64()?;
        let length = usize::try_from(length)
            .map_err(|_| anyhow::anyhow!("string GGUF grande demais: {length} bytes"))?;
        if length > 64 << 20 {
            bail!("string GGUF implausível: {length} bytes");
        }
        Ok(String::from_utf8_lossy(self.take(length)?).into_owned())
    }

    fn skip_value(&mut self, value_type: u32) -> Result<()> {
        match value_type {
            T_U8 | T_I8 | T_BOOL => {
                self.take(1)?;
            }
            T_U16 | T_I16 => {
                self.take(2)?;
            }
            T_U32 | T_I32 | T_F32 => {
                self.take(4)?;
            }
            T_U64 | T_I64 | T_F64 => {
                self.take(8)?;
            }
            T_STR => {
                let length = usize::try_from(self.u64()?)
                    .map_err(|_| anyhow::anyhow!("string GGUF grande demais"))?;
                self.take(length)?;
            }
            T_ARR => {
                let element_type = self.u32()?;
                let count = self.u64()?;
                if let Some(size) = scalar_size(element_type) {
                    let bytes = count
                        .checked_mul(size as u64)
                        .ok_or_else(|| anyhow::anyhow!("overflow no array GGUF"))?;
                    self.take(usize::try_from(bytes).map_err(|_| {
                        anyhow::anyhow!("array GGUF grande demais: {bytes} bytes")
                    })?)?;
                } else {
                    for _ in 0..count {
                        self.skip_value(element_type)?;
                    }
                }
            }
            other => bail!("tipo GGUF desconhecido {other}"),
        }
        Ok(())
    }
}

fn scalar_size(value_type: u32) -> Option<usize> {
    match value_type {
        T_U8 | T_I8 | T_BOOL => Some(1),
        T_U16 | T_I16 => Some(2),
        T_U32 | T_I32 | T_F32 => Some(4),
        T_U64 | T_I64 | T_F64 => Some(8),
        _ => None,
    }
}

fn nonnegative_integer(reader: &mut Reader<'_>, value_type: u32, key: &str) -> Result<u64> {
    match value_type {
        T_U8 => Ok(reader.u8()? as u64),
        T_U16 => Ok(reader.u16()? as u64),
        T_U32 => Ok(reader.u32()? as u64),
        T_U64 => reader.u64(),
        T_I8 => {
            let value = i8::from_le_bytes(reader.take(1)?.try_into().unwrap());
            u64::try_from(value).map_err(|_| anyhow::anyhow!("'{key}' tem valor negativo: {value}"))
        }
        T_I16 => {
            let value = i16::from_le_bytes(reader.take(2)?.try_into().unwrap());
            u64::try_from(value).map_err(|_| anyhow::anyhow!("'{key}' tem valor negativo: {value}"))
        }
        T_I32 => {
            let value = i32::from_le_bytes(reader.take(4)?.try_into().unwrap());
            u64::try_from(value).map_err(|_| anyhow::anyhow!("'{key}' tem valor negativo: {value}"))
        }
        T_I64 => {
            let value = i64::from_le_bytes(reader.take(8)?.try_into().unwrap());
            u64::try_from(value).map_err(|_| anyhow::anyhow!("'{key}' tem valor negativo: {value}"))
        }
        _ => bail!("'{key}' não é um inteiro GGUF não negativo"),
    }
}

#[derive(Debug, Clone)]
struct HeaderTensor {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
}

struct ParsedHeader {
    tensor_count: u64,
    fields: SplitFields,
    tensors: Vec<HeaderTensor>,
    data_start: usize,
}

fn align_up(value: usize) -> Result<usize> {
    value
        .checked_add(ALIGN - 1)
        .map(|value| value & !(ALIGN - 1))
        .ok_or_else(|| anyhow::anyhow!("overflow ao alinhar cabeçalho GGUF"))
}

fn parse_header(data: &[u8]) -> Result<ParsedHeader> {
    let mut reader = Reader { data, pos: 0 };
    let magic = reader.u32()?;
    if magic != GGUF_MAGIC {
        bail!("não é um arquivo GGUF (magic={magic:#x})");
    }
    let version = reader.u32()?;
    if version != 2 && version != 3 {
        bail!("versão GGUF {version} não suportada (esperado 2 ou 3)");
    }
    let tensor_count = reader.u64()?;
    let kv_count = reader.u64()?;
    if tensor_count > 1_000_000 || kv_count > 100_000 {
        bail!("cabeçalho GGUF implausível ({tensor_count} tensors, {kv_count} kv)");
    }

    let mut fields = SplitFields::default();
    for _ in 0..kv_count {
        let key = reader.string()?;
        let value_type = reader.u32()?;
        match key.as_str() {
            "split.count" => {
                fields.count = Some(nonnegative_integer(&mut reader, value_type, &key)?)
            }
            "split.no" => fields.no = Some(nonnegative_integer(&mut reader, value_type, &key)?),
            "split.tensors.count" => {
                fields.tensors_count = Some(nonnegative_integer(&mut reader, value_type, &key)?)
            }
            _ => reader.skip_value(value_type)?,
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let dimensions = reader.u32()?;
        if dimensions > 8 {
            bail!("rank {dimensions} implausível para tensor '{name}'");
        }
        let dims = (0..dimensions)
            .map(|_| reader.u64())
            .collect::<Result<Vec<_>>>()?;
        let dtype = reader.u32()?;
        let offset = reader.u64()?;
        tensors.push(HeaderTensor {
            name,
            dims,
            dtype,
            offset,
        });
    }

    let data_start = align_up(reader.pos)?;
    if data_start > data.len() {
        bail!("GGUF truncado antes da seção de dados");
    }
    Ok(ParsedHeader {
        tensor_count,
        fields,
        tensors,
        data_start,
    })
}

fn tensor_sizes(header: &ParsedHeader, file_bytes: usize) -> Result<Vec<u64>> {
    let total_data = file_bytes
        .checked_sub(header.data_start)
        .ok_or_else(|| anyhow::anyhow!("GGUF truncado antes da seção de dados"))?;
    let total_data = total_data as u64;
    let mut offsets: Vec<u64> = header.tensors.iter().map(|tensor| tensor.offset).collect();
    offsets.sort_unstable();
    offsets.dedup();
    let mut sizes = Vec::with_capacity(offsets.len());
    for (index, offset) in offsets.iter().copied().enumerate() {
        if offset > total_data {
            bail!("offset de tensor {offset} excede a seção de dados ({total_data})");
        }
        let next = offsets.get(index + 1).copied().unwrap_or(total_data);
        sizes.push((
            offset,
            next.checked_sub(offset)
                .ok_or_else(|| anyhow::anyhow!("offsets de tensor GGUF fora de ordem"))?,
        ));
    }
    header
        .tensors
        .iter()
        .map(|tensor| {
            sizes
                .binary_search_by_key(&tensor.offset, |(offset, _)| *offset)
                .map(|index| sizes[index].1)
                .map_err(|_| anyhow::anyhow!("offset de tensor ausente na tabela"))
        })
        .collect()
}

#[derive(Debug)]
pub struct GgufPart {
    /// Mantido para que o `Debug` derivado identifique a parte nos diagnósticos.
    #[allow(dead_code)]
    pub path: PathBuf,
    pub index: u32,
    pub bytes: u64,
    pub tensor_count: u64,
    data_start: usize,
    map: Mmap,
}

impl GgufPart {
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    pub fn map(&self) -> &[u8] {
        &self.map
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub part_index: usize,
    pub offset: u64,
    pub byte_size: u64,
}

pub struct GgufSource {
    parts: Vec<GgufPart>,
    tensors: Vec<GgufTensor>,
    /// API explícita para distinguir uma fonte lógica multipart sem inferir
    /// isso pela quantidade de partes.
    #[allow(dead_code)]
    multipart: bool,
}

/// Resumo de uma fonte usado pelo catálogo. Diferente de [`GgufSource`], não
/// mantém mappings nem valida offsets dos pesos: cada parte só é lida até o
/// próprio cabeçalho e o restante vem de `stat`.
#[derive(Debug, Clone)]
pub struct CatalogPart {
    pub index: u32,
    pub path: PathBuf,
    pub bytes: u64,
    pub tensor_count: u64,
}

#[derive(Debug, Clone, Default)]
pub struct CatalogMetadata {
    pub multipart: bool,
    pub parts: Vec<CatalogPart>,
    pub source_bytes: u64,
}

/// Read only the metadata needed to list a GGUF multipart source.
///
/// The full [`GgufSource::open`] path remains reserved for slicing and layout
/// consumers. In particular, catalog enrichment must not map or validate every
/// tensor payload merely to show the model architecture and part sizes.
pub fn catalog_metadata(
    path: &Path,
    first_info: &crate::gguf::GgufInfo,
) -> Result<CatalogMetadata> {
    let split_count = first_info.split_count.unwrap_or(1);
    if split_count == 0 {
        bail!("split.count não pode ser zero em {}", path.display());
    }
    if split_count == 1 {
        if first_info.split_no.unwrap_or(0) != 0 {
            bail!(
                "split.no={:?} não é uma parte única válida",
                first_info.split_no
            );
        }
        return Ok(CatalogMetadata::default());
    }
    if first_info.split_no.unwrap_or(0) != 0 {
        bail!(
            "o GGUF multipart foi aberto pela parte split.no={:?}; abra a primeira parte (split.no=0)",
            first_info.split_no
        );
    }
    let expected_tensors = first_info
        .split_tensors_count
        .ok_or_else(|| anyhow::anyhow!("GGUF multipart sem split.tensors.count no cabeçalho"))?;
    let (prefix, width, filename_total, filename_index) = multipart_filename(path).ok_or_else(|| {
        anyhow::anyhow!(
            "GGUF declara split.count={split_count}, mas o nome '{}' não identifica as partes como N-of-M.gguf",
            path.display()
        )
    })?;
    if filename_total != split_count {
        bail!("nome do GGUF declara {filename_total} partes, mas split.count={split_count}");
    }
    let part_count = usize::try_from(split_count)
        .map_err(|_| anyhow::anyhow!("split.count GGUF grande demais: {split_count}"))?;
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("gguf");
    let mut parts = Vec::with_capacity(part_count);
    for index in 0..part_count {
        let part_path = if index == 0 {
            path.to_path_buf()
        } else {
            let part_number = filename_index
                .checked_add(index as u64)
                .ok_or_else(|| anyhow::anyhow!("índice da parte GGUF excede u64"))?;
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .join(format!(
                    "{prefix}-{part_number:0width$}-of-{split_count:0width$}.{extension}"
                ))
        };
        let part_info = if index == 0 {
            first_info.clone()
        } else {
            crate::gguf::read_info(&part_path).with_context(|| {
                format!(
                    "ler cabeçalho da parte {} de {} do GGUF multipart ({})",
                    index + 1,
                    split_count,
                    part_path.display()
                )
            })?
        };
        let bytes = std::fs::metadata(&part_path)
            .with_context(|| format!("stat da parte GGUF {}", part_path.display()))?
            .len();
        if part_info.split_count != Some(split_count)
            || part_info.split_no != Some(index as u64)
            || part_info.split_tensors_count != Some(expected_tensors)
        {
            bail!(
                "parte {} de {} tem split.count={:?}, split.no={:?}, split.tensors.count={:?}; esperado ({split_count}, {index}, {expected_tensors})",
                index + 1,
                split_count,
                part_info.split_count,
                part_info.split_no,
                part_info.split_tensors_count
            );
        }
        parts.push(CatalogPart {
            index: u32::try_from(index)
                .map_err(|_| anyhow::anyhow!("índice da parte GGUF grande demais"))?,
            path: part_path,
            bytes,
            tensor_count: part_info.tensor_count,
        });
    }
    let source_bytes = parts.iter().try_fold(0u64, |total, part| {
        total
            .checked_add(part.bytes)
            .ok_or_else(|| anyhow::anyhow!("tamanho total do GGUF multipart excede u64"))
    })?;
    Ok(CatalogMetadata {
        multipart: true,
        parts,
        source_bytes,
    })
}

impl GgufSource {
    pub fn open(path: &Path) -> Result<Self> {
        let first = map_part(path)?;
        let first_header = parse_header(first.map())?;
        let split_count = first_header.fields.count.unwrap_or(1);
        if split_count == 0 {
            bail!("split.count não pode ser zero em {}", path.display());
        }
        let first_no = first_header.fields.no.unwrap_or(0);
        if split_count == 1 {
            if first_no != 0 {
                bail!("split.no={first_no} não é uma parte única válida");
            }
            let expected_tensors = first_header.fields.tensors_count;
            return Self::from_headers(vec![(first, first_header)], false, expected_tensors);
        }
        if first_no != 0 {
            bail!(
                "o GGUF multipart foi aberto pela parte split.no={first_no}; abra a primeira parte (split.no=0)"
            );
        }
        let (prefix, width, filename_total, filename_index) = multipart_filename(path).ok_or_else(|| {
            anyhow::anyhow!(
                "GGUF declara split.count={split_count}, mas o nome '{}' não identifica as partes como N-of-M.gguf",
                path.display()
            )
        })?;
        if filename_total != split_count {
            bail!("nome do GGUF declara {filename_total} partes, mas split.count={split_count}");
        }
        let expected_tensors = first_header.fields.tensors_count.ok_or_else(|| {
            anyhow::anyhow!("GGUF multipart sem split.tensors.count no cabeçalho")
        })?;
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("gguf");
        let mut headers = Vec::with_capacity(split_count as usize);
        headers.push((first, first_header));
        for index in 1..split_count {
            let part_path = path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(format!(
                    "{prefix}-{part_number:0width$}-of-{split_count:0width$}.{extension}",
                    part_number = filename_index + index
                ));
            let part = map_part(&part_path).with_context(|| {
                format!(
                    "abrir parte {} de {} do GGUF multipart ({})",
                    index + 1,
                    split_count,
                    part_path.display()
                )
            })?;
            let header = parse_header(part.map()).with_context(|| {
                format!(
                    "ler cabeçalho da parte {} ({})",
                    index + 1,
                    part_path.display()
                )
            })?;
            if header.fields.count != Some(split_count)
                || header.fields.no != Some(index)
                || header.fields.tensors_count != Some(expected_tensors)
            {
                bail!(
                    "parte {} do GGUF multipart tem split.count={:?}, split.no={:?}, split.tensors.count={:?}; esperado ({split_count}, {index}, {expected_tensors})",
                    index + 1,
                    header.fields.count,
                    header.fields.no,
                    header.fields.tensors_count
                );
            }
            headers.push((part, header));
        }
        Self::from_headers(headers, true, Some(expected_tensors))
    }

    fn from_headers(
        headers: Vec<(GgufPart, ParsedHeader)>,
        multipart: bool,
        expected_tensors: Option<u64>,
    ) -> Result<Self> {
        let mut parts = Vec::with_capacity(headers.len());
        let mut tensors = Vec::new();
        let mut names = HashSet::new();
        for (part_index, (mut part, header)) in headers.into_iter().enumerate() {
            let sizes = tensor_sizes(&header, part.map.len())?;
            if header.tensor_count != header.tensors.len() as u64 {
                bail!("contagem de tensores GGUF inconsistente na parte {part_index}");
            }
            part.index = u32::try_from(header.fields.no.unwrap_or(part_index as u64))
                .map_err(|_| anyhow::anyhow!("índice da parte GGUF grande demais"))?;
            part.tensor_count = header.tensor_count;
            part.data_start = header.data_start;
            for (tensor, byte_size) in header.tensors.into_iter().zip(sizes) {
                if !names.insert(tensor.name.clone()) {
                    bail!("tensor GGUF duplicado entre partes: '{}'", tensor.name);
                }
                tensors.push(GgufTensor {
                    name: tensor.name,
                    dims: tensor.dims,
                    dtype: tensor.dtype,
                    part_index,
                    offset: tensor.offset,
                    byte_size,
                });
            }
            parts.push(part);
        }
        if let Some(expected) = expected_tensors {
            if tensors.len() as u64 != expected {
                bail!(
                    "split.tensors.count={expected}, mas a tabela unificada tem {} tensores",
                    tensors.len()
                );
            }
        }
        Ok(Self {
            parts,
            tensors,
            multipart,
        })
    }

    /// Conveniência pública para callers que precisam da forma da fonte.
    #[allow(dead_code)]
    pub fn is_multipart(&self) -> bool {
        self.multipart
    }

    pub fn parts(&self) -> &[GgufPart] {
        &self.parts
    }

    pub fn tensors(&self) -> &[GgufTensor] {
        &self.tensors
    }

    pub fn source_bytes(&self) -> u64 {
        self.parts.iter().map(|part| part.bytes).sum()
    }

    /// Conveniência pública para calcular o hash sem callback de progresso.
    #[allow(dead_code)]
    pub fn hash_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        for part in &self.parts {
            hasher.update(part.map());
        }
        hex::encode(hasher.finalize())
    }
}

pub fn declared_split_count(path: &Path) -> Result<Option<u32>> {
    crate::gguf::read_info(path)?
        .split_count
        .map(|count| {
            u32::try_from(count)
                .map_err(|_| anyhow::anyhow!("split.count GGUF grande demais: {count}"))
        })
        .transpose()
        .map(|count| count.filter(|count| *count > 1))
}

fn map_part(path: &Path) -> Result<GgufPart> {
    let file = File::open(path).with_context(|| format!("abrir {}", path.display()))?;
    let bytes = file.metadata()?.len();
    let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
    Ok(GgufPart {
        path: path.to_owned(),
        index: 0,
        bytes,
        tensor_count: 0,
        data_start: 0,
        map,
    })
}

fn multipart_filename(path: &Path) -> Option<(String, usize, u64, u64)> {
    let filename = path.file_name()?.to_str()?;
    let extension = path.extension()?.to_str()?;
    let stem = filename.strip_suffix(&format!(".{extension}"))?;
    let (prefix_and_index, total_text) = stem.rsplit_once("-of-")?;
    let (prefix, index_text) = prefix_and_index.rsplit_once('-')?;
    let index: u64 = index_text.parse().ok()?;
    let total: u64 = total_text.parse().ok()?;
    (matches!(index, 0 | 1) && total > 1).then_some((
        prefix.to_owned(),
        index_text.len(),
        total,
        index,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn string(value: &str) -> Vec<u8> {
        let mut out = (value.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(value.as_bytes());
        out
    }

    fn header(no: u16, total_tensor_count: u64, tensors: &[(&str, u64)]) -> Vec<u8> {
        typed_header(no, 2, T_U64, total_tensor_count, tensors)
    }

    fn typed_header(
        no: u16,
        split_count: u16,
        tensor_count_type: u32,
        total_tensor_count: u64,
        tensors: &[(&str, u64)],
    ) -> Vec<u8> {
        let mut kv = Vec::new();
        for (key, value) in [("split.count", split_count), ("split.no", no)] {
            kv.extend_from_slice(&string(key));
            kv.extend_from_slice(&T_U16.to_le_bytes());
            kv.extend_from_slice(&value.to_le_bytes());
        }
        kv.extend_from_slice(&string("split.tensors.count"));
        kv.extend_from_slice(&tensor_count_type.to_le_bytes());
        match tensor_count_type {
            T_I32 => kv.extend_from_slice(&(total_tensor_count as i32).to_le_bytes()),
            T_U64 => kv.extend_from_slice(&total_tensor_count.to_le_bytes()),
            other => panic!("tipo de teste não suportado: {other}"),
        }

        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&3u64.to_le_bytes());
        out.extend_from_slice(&kv);
        for (name, offset) in tensors {
            out.extend_from_slice(&string(name));
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&4u64.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }
        while out.len() % ALIGN != 0 {
            out.push(0);
        }
        out
    }

    #[test]
    fn accepts_reference_signed_split_tensor_count() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("pom-multipart-signed-count-{nonce}"));
        std::fs::create_dir_all(&dir).unwrap();

        for no in 0..4u16 {
            let path = dir.join(format!("model-{no:05}-of-00004.gguf"));
            std::fs::write(&path, typed_header(no, 4, T_I32, 0, &[])).unwrap();
        }
        let first = dir.join("model-00000-of-00004.gguf");
        let info = crate::gguf::read_info(&first).unwrap();
        let metadata = catalog_metadata(&first, &info).unwrap();
        let source = GgufSource::open(&first).unwrap();

        assert_eq!(metadata.parts.len(), 4);
        assert_eq!(source.parts().len(), 4);
        assert_eq!(
            metadata.source_bytes,
            (0..4)
                .map(|no| {
                    std::fs::metadata(dir.join(format!("model-{no:05}-of-00004.gguf")))
                        .unwrap()
                        .len()
                })
                .sum::<u64>()
        );

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_negative_signed_split_count() {
        let bytes = (-1i32).to_le_bytes();
        let mut reader = Reader {
            data: &bytes,
            pos: 0,
        };

        assert!(nonnegative_integer(&mut reader, T_I32, "split.tensors.count").is_err());
    }

    #[test]
    fn unifies_tensor_tables_without_remounting_weights() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("pom-multipart-{nonce}"));
        std::fs::create_dir_all(&dir).unwrap();
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        let first_bytes = header(0, 1, &[]);
        std::fs::write(&first, &first_bytes).unwrap();
        let mut second_bytes = header(1, 1, &[("blk.0.weight", 0)]);
        second_bytes.extend_from_slice(&[7u8; 16]);
        std::fs::write(&second, &second_bytes).unwrap();

        let source = GgufSource::open(&first).unwrap();
        assert!(source.is_multipart());
        assert_eq!(source.parts().len(), 2);
        assert_eq!(source.tensors().len(), 1);
        assert_eq!(source.tensors()[0].part_index, 1);
        assert_eq!(source.tensors()[0].byte_size, 16);
        assert_eq!(
            source.source_bytes(),
            (first_bytes.len() + second_bytes.len()) as u64
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn catalog_metadata_reads_headers_without_opening_tensor_source() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("pom-multipart-catalog-{nonce}"));
        std::fs::create_dir_all(&dir).unwrap();
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        let first_bytes = header(0, 1, &[]);
        std::fs::write(&first, &first_bytes).unwrap();
        let mut second_bytes = header(1, 1, &[("blk.0.weight", 0)]);
        second_bytes.extend_from_slice(&[7u8; 16]);
        std::fs::write(&second, &second_bytes).unwrap();

        let info = crate::gguf::read_info(&first).unwrap();
        let metadata = catalog_metadata(&first, &info).unwrap();
        assert!(metadata.multipart);
        assert_eq!(metadata.parts.len(), 2);
        assert_eq!(metadata.parts[0].tensor_count, 0);
        assert_eq!(metadata.parts[1].tensor_count, 1);
        assert_eq!(
            metadata.source_bytes,
            (first_bytes.len() + second_bytes.len()) as u64
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}
