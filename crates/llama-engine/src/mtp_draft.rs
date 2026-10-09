//! Shared MTP drafts: completing a draft that borrows tensors from its target.
//!
//! Some published MTP drafts (the "shared" Unsloth variants, flagged with
//! `<arch>.nextn_shared_target_tensors = true`) ship only the `nextn` block and
//! leave `token_embd.weight` and `output.weight` out, because the target model
//! already has them. llama.cpp loads a draft as an independent model, and its
//! `mtp_only` loaders require `token_embd`, so such a file never loads on its
//! own.
//!
//! [`ready_draft_path`] keeps the fork untouched: when the draft lacks
//! `token_embd.weight`, it writes a completed copy that appends the missing
//! tensors read from the target GGUF (a slice or a whole, possibly multipart,
//! model). The copy is cached in [`CACHE_DIR`] next to the target and reused
//! while the draft and the target keep their size and mtime. A draft that
//! already carries `token_embd.weight` is returned as is, and the original
//! files are never written.

use crate::multipart::GgufSource;
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Folder, next to the target model, that holds the completed drafts. The
/// model catalog skips it, so a completed copy never shows up as a file of its
/// own.
pub const CACHE_DIR: &str = ".mtp-drafts";

/// Tensors a shared draft takes from its target, in the order they are
/// appended. `token_embd` is required; `output` is optional because the
/// loaders fall back to the tied embedding when it is missing.
pub const SHARED_TARGET_TENSORS: [&str; 2] = ["token_embd.weight", "output.weight"];
const REQUIRED_TENSOR: &str = "token_embd.weight";

const GGUF_MAGIC: u32 = 0x4655_4747;
const DEFAULT_ALIGNMENT: u64 = 32;
const T_UINT32: u32 = 4;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;

/// Returns the draft path llama.cpp can load for `target`: the draft itself
/// when it is complete, or the cached completed copy of a shared draft.
pub fn ready_draft_path(draft: &Path, target: &Path) -> Result<PathBuf> {
    let header = DraftHeader::read(draft)?;
    if header.has_tensor(REQUIRED_TENSOR) {
        return Ok(draft.to_path_buf());
    }
    let cache_dir = target
        .parent()
        .map(|dir| dir.join(CACHE_DIR))
        .ok_or_else(|| anyhow::anyhow!("o modelo alvo {} não tem pasta", target.display()))?;
    let stem = draft
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mtp-draft".to_string());
    let output = cache_dir.join(format!("{stem}-{}.gguf", fingerprint(draft, target)?));
    if output.is_file() {
        return Ok(output);
    }
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("criar {}", cache_dir.display()))?;
    static NEXT_TMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = cache_dir.join(format!(
        ".{}.{}-{}.tmp",
        output.file_name().unwrap().to_string_lossy(),
        std::process::id(),
        NEXT_TMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let written = write_completed(&header, draft, target, &tmp);
    if let Err(error) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(error.context(format!(
            "completar o rascunho MTP compartilhado {} com os tensores de {}",
            draft.display(),
            target.display()
        )));
    }
    std::fs::rename(&tmp, &output)
        .with_context(|| format!("publicar {}", output.display()))?;
    prune_older_copies(&cache_dir, &stem, &output);
    tracing::info!(
        draft = %draft.display(),
        target = %target.display(),
        completed = %output.display(),
        "rascunho MTP compartilhado completado com os tensores do alvo"
    );
    Ok(output)
}

/// Identity of a (draft, target) pair: canonical path, size and mtime of both.
/// A changed file produces a new name instead of reusing a stale copy.
fn fingerprint(draft: &Path, target: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    for path in [draft, target] {
        let canonical = path
            .canonicalize()
            .with_context(|| format!("resolver {}", path.display()))?;
        let meta = std::fs::metadata(&canonical)?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);
        hasher.update(canonical.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(meta.len().to_le_bytes());
        hasher.update(mtime.to_le_bytes());
    }
    Ok(hex::encode(&hasher.finalize()[..8]))
}

fn prune_older_copies(cache_dir: &Path, stem: &str, keep: &Path) {
    let Ok(entries) = std::fs::read_dir(cache_dir) else {
        return;
    };
    let prefix = format!("{stem}-");
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path != keep && name.starts_with(&prefix) && name.ends_with(".gguf") {
            let _ = std::fs::remove_file(&path);
        }
    }
}

struct DraftTensor {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
}

/// The parts of the draft header the completed copy reuses verbatim.
struct DraftHeader {
    version: u32,
    kv_count: u64,
    kv_bytes: Vec<u8>,
    tensors: Vec<DraftTensor>,
    alignment: u64,
    data_start: u64,
    file_len: u64,
}

impl DraftHeader {
    fn read(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("abrir {}", path.display()))?;
        let file_len = file.metadata()?.len();
        // The header is read from a map: only the pages it touches are read.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mapear {}", path.display()))?;
        let mut reader = SliceReader { data: &map, pos: 0 };
        if reader.u32()? != GGUF_MAGIC {
            bail!("{} não é um arquivo GGUF", path.display());
        }
        let version = reader.u32()?;
        if version != 2 && version != 3 {
            bail!("versão GGUF {version} não suportada em {}", path.display());
        }
        let tensor_count = reader.u64()?;
        let kv_count = reader.u64()?;
        if tensor_count > 1_000_000 || kv_count > 100_000 {
            bail!("cabeçalho GGUF implausível em {}", path.display());
        }
        let kv_start = reader.pos;
        let mut alignment = DEFAULT_ALIGNMENT;
        for _ in 0..kv_count {
            let key = reader.string()?;
            let value_type = reader.u32()?;
            if key == "general.alignment" && value_type == T_UINT32 {
                alignment = u64::from(reader.u32()?);
            } else {
                reader.skip_value(value_type)?;
            }
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            bail!("general.alignment={alignment} inválido em {}", path.display());
        }
        let kv_bytes = map[kv_start..reader.pos].to_vec();
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let rank = reader.u32()?;
            if rank > 8 {
                bail!("rank {rank} implausível para o tensor {name}");
            }
            let dims = (0..rank).map(|_| reader.u64()).collect::<Result<Vec<_>>>()?;
            let dtype = reader.u32()?;
            let offset = reader.u64()?;
            tensors.push(DraftTensor {
                name,
                dims,
                dtype,
                offset,
            });
        }
        let data_start = align_up(reader.pos as u64, alignment);
        if data_start > file_len {
            bail!("{} termina antes da seção de dados", path.display());
        }
        Ok(Self {
            version,
            kv_count,
            kv_bytes,
            tensors,
            alignment,
            data_start,
            file_len,
        })
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensors.iter().any(|tensor| tensor.name == name)
    }
}

fn write_completed(header: &DraftHeader, draft: &Path, target: &Path, output: &Path) -> Result<()> {
    let source = GgufSource::open(target)?;
    let borrowed: Vec<_> = SHARED_TARGET_TENSORS
        .iter()
        .filter(|name| !header.has_tensor(name))
        .filter_map(|name| source.tensors().iter().find(|tensor| tensor.name == *name))
        .collect();
    if !borrowed.iter().any(|tensor| tensor.name == REQUIRED_TENSOR) {
        bail!(
            "o modelo alvo {} não traz {REQUIRED_TENSOR}; um rascunho MTP compartilhado \
             precisa da fatia final ou do modelo inteiro",
            target.display()
        );
    }

    let alignment = header.alignment;
    let draft_data = header.file_len - header.data_start;
    let mut appended = Vec::with_capacity(borrowed.len());
    let mut next = align_up(draft_data, alignment);
    for tensor in &borrowed {
        let bytes = exact_tensor_bytes(&tensor.dims, tensor.dtype).unwrap_or(tensor.byte_size);
        if bytes > tensor.byte_size {
            bail!("o tensor {} do alvo está truncado", tensor.name);
        }
        appended.push((*tensor, next, bytes));
        next = align_up(next + bytes, alignment);
    }

    let file = File::create(output).with_context(|| format!("criar {}", output.display()))?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    let mut head = Vec::new();
    head.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    head.extend_from_slice(&header.version.to_le_bytes());
    head.extend_from_slice(&((header.tensors.len() + appended.len()) as u64).to_le_bytes());
    head.extend_from_slice(&header.kv_count.to_le_bytes());
    head.extend_from_slice(&header.kv_bytes);
    for tensor in &header.tensors {
        head.extend_from_slice(&tensor_info(&tensor.name, &tensor.dims, tensor.dtype, tensor.offset));
    }
    for (tensor, offset, _) in &appended {
        head.extend_from_slice(&tensor_info(&tensor.name, &tensor.dims, tensor.dtype, *offset));
    }
    head.resize(align_up(head.len() as u64, alignment) as usize, 0);
    out.write_all(&head)?;

    // The draft's own data keeps its relative offsets.
    let mut draft_file = File::open(draft)?;
    std::io::Seek::seek(&mut draft_file, std::io::SeekFrom::Start(header.data_start))?;
    let copied = std::io::copy(&mut std::io::Read::take(draft_file, draft_data), &mut out)?;
    if copied != draft_data {
        bail!("{} mudou durante a cópia", draft.display());
    }
    let mut relative = draft_data;
    for (tensor, offset, bytes) in &appended {
        out.write_all(&vec![0u8; (offset - relative) as usize])?;
        let part = &source.parts()[tensor.part_index];
        let start = part.data_start() + tensor.offset as usize;
        let data = part
            .map()
            .get(start..start + *bytes as usize)
            .ok_or_else(|| anyhow::anyhow!("o tensor {} do alvo está truncado", tensor.name))?;
        out.write_all(data)?;
        relative = offset + bytes;
    }
    let file = out.into_inner().map_err(|error| error.into_error())?;
    file.sync_all()?;
    Ok(())
}

fn exact_tensor_bytes(dims: &[u64], dtype: u32) -> Option<u64> {
    let (block_bytes, block_elements) = crate::gguf::ggml_type_info(dtype)?;
    let elements = dims.iter().try_fold(1u64, |acc, dim| acc.checked_mul(*dim))?;
    (elements % block_elements == 0).then(|| elements / block_elements * block_bytes)
}

fn tensor_info(name: &str, dims: &[u64], dtype: u32, offset: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(name.len() + 24 + dims.len() * 8);
    out.extend_from_slice(&(name.len() as u64).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for dim in dims {
        out.extend_from_slice(&dim.to_le_bytes());
    }
    out.extend_from_slice(&dtype.to_le_bytes());
    out.extend_from_slice(&offset.to_le_bytes());
    out
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

struct SliceReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl SliceReader<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|end| *end <= self.data.len())
            .ok_or_else(|| anyhow::anyhow!("cabeçalho GGUF truncado"))?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    fn skip_value(&mut self, value_type: u32) -> Result<()> {
        match value_type {
            T_STRING => {
                let len = self.u64()? as usize;
                self.take(len)?;
            }
            T_ARRAY => {
                let element_type = self.u32()?;
                let count = self.u64()?;
                if let Some(size) = scalar_size(element_type) {
                    let bytes = count
                        .checked_mul(size)
                        .ok_or_else(|| anyhow::anyhow!("array GGUF implausível"))?;
                    self.take(bytes as usize)?;
                } else {
                    for _ in 0..count {
                        self.skip_value(element_type)?;
                    }
                }
            }
            other => {
                let size = scalar_size(other)
                    .ok_or_else(|| anyhow::anyhow!("tipo de valor GGUF desconhecido {other}"))?;
                self.take(size as usize)?;
            }
        }
        Ok(())
    }
}

fn scalar_size(value_type: u32) -> Option<u64> {
    Some(match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: u32 = 0;

    fn string(value: &str) -> Vec<u8> {
        let mut out = (value.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(value.as_bytes());
        out
    }

    /// A GGUF with one string KV, `general.alignment` and F32 tensors whose
    /// payload is the tensor's index repeated.
    fn gguf(alignment: u32, tensors: &[(&str, u64)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes());
        out.extend_from_slice(&string("general.architecture"));
        out.extend_from_slice(&T_STRING.to_le_bytes());
        out.extend_from_slice(&string("qwen4exp"));
        out.extend_from_slice(&string("general.alignment"));
        out.extend_from_slice(&T_UINT32.to_le_bytes());
        out.extend_from_slice(&alignment.to_le_bytes());
        let alignment = u64::from(alignment);
        let mut offset = 0u64;
        let mut payload = Vec::new();
        for (index, (name, elements)) in tensors.iter().enumerate() {
            out.extend_from_slice(&tensor_info(name, &[*elements], F32, offset));
            payload.resize(offset as usize, 0);
            payload.extend(std::iter::repeat(index as u8 + 1).take(*elements as usize * 4));
            offset = align_up(offset + elements * 4, alignment);
        }
        out.resize(align_up(out.len() as u64, alignment) as usize, 0);
        out.extend_from_slice(&payload);
        out
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pom-mtp-draft-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Reads a tensor honouring `general.alignment`, which the multipart
    /// reader does not (it assumes the default 32).
    fn tensor_bytes(path: &Path, name: &str) -> Vec<u8> {
        let header = DraftHeader::read(path).unwrap();
        let tensor = header.tensors.iter().find(|tensor| tensor.name == name).unwrap();
        let start = (header.data_start + tensor.offset) as usize;
        let len = exact_tensor_bytes(&tensor.dims, tensor.dtype).unwrap() as usize;
        std::fs::read(path).unwrap()[start..start + len].to_vec()
    }

    #[test]
    fn a_complete_draft_is_used_as_is() {
        let dir = temp_dir("complete");
        let draft = dir.join("mtp-complete.gguf");
        let target = dir.join("tail.gguf");
        std::fs::write(&draft, gguf(32, &[("blk.0.nextn.enorm.weight", 4), ("token_embd.weight", 8)]))
            .unwrap();
        std::fs::write(&target, gguf(32, &[("token_embd.weight", 8)])).unwrap();

        assert_eq!(ready_draft_path(&draft, &target).unwrap(), draft);
        assert!(!dir.join(CACHE_DIR).exists());
    }

    #[test]
    fn a_shared_draft_gets_the_target_embedding_and_output() {
        let dir = temp_dir("shared");
        let draft = dir.join("mtp-shared.gguf");
        let target = dir.join("tail.gguf");
        // An odd element count leaves the draft's data unaligned at the end.
        let draft_bytes = gguf(64, &[("blk.0.nextn.enorm.weight", 5), ("blk.0.nextn.hnorm.weight", 3)]);
        std::fs::write(&draft, &draft_bytes).unwrap();
        std::fs::write(
            &target,
            gguf(32, &[("blk.0.ffn.weight", 6), ("output.weight", 7), ("token_embd.weight", 9)]),
        )
        .unwrap();

        let completed = ready_draft_path(&draft, &target).unwrap();
        assert_eq!(completed.parent().unwrap(), dir.join(CACHE_DIR));
        assert_eq!(std::fs::read(&draft).unwrap(), draft_bytes, "the draft is never written");

        let names: Vec<_> = DraftHeader::read(&completed)
            .unwrap()
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();
        assert_eq!(
            names,
            [
                "blk.0.nextn.enorm.weight",
                "blk.0.nextn.hnorm.weight",
                "token_embd.weight",
                "output.weight"
            ]
        );
        for name in ["blk.0.nextn.enorm.weight", "blk.0.nextn.hnorm.weight"] {
            assert_eq!(tensor_bytes(&completed, name), tensor_bytes(&draft, name));
        }
        for name in ["token_embd.weight", "output.weight"] {
            assert_eq!(tensor_bytes(&completed, name), tensor_bytes(&target, name));
        }
        let header = DraftHeader::read(&completed).unwrap();
        assert_eq!(header.alignment, 64, "the draft's metadata is kept");

        // A second load reuses the copy instead of writing it again.
        let modified = std::fs::metadata(&completed).unwrap().modified().unwrap();
        assert_eq!(ready_draft_path(&draft, &target).unwrap(), completed);
        assert_eq!(std::fs::metadata(&completed).unwrap().modified().unwrap(), modified);
    }

    #[test]
    fn a_target_without_the_embedding_is_refused_by_name() {
        let dir = temp_dir("middle");
        let draft = dir.join("mtp-shared.gguf");
        let target = dir.join("middle.gguf");
        std::fs::write(&draft, gguf(32, &[("blk.0.nextn.enorm.weight", 4)])).unwrap();
        std::fs::write(&target, gguf(32, &[("blk.3.ffn.weight", 4)])).unwrap();

        let error = format!("{:#}", ready_draft_path(&draft, &target).unwrap_err());
        assert!(error.contains("token_embd.weight"), "{error}");
        let leftovers = std::fs::read_dir(dir.join(CACHE_DIR)).unwrap().count();
        assert_eq!(leftovers, 0, "a failed copy leaves nothing behind");
    }

    /// Real files, when present: `POM_MTP_SHARED_DRAFT` (a shared draft) and
    /// `POM_MTP_TARGET_SLICE` (the slice that holds the output head).
    #[test]
    #[ignore]
    fn a_real_shared_draft_is_completed_against_its_target() {
        let (Ok(draft), Ok(target)) = (
            std::env::var("POM_MTP_SHARED_DRAFT"),
            std::env::var("POM_MTP_TARGET_SLICE"),
        ) else {
            return;
        };
        let completed = ready_draft_path(Path::new(&draft), Path::new(&target)).unwrap();
        let source = GgufSource::open(&completed).unwrap();
        for name in SHARED_TARGET_TENSORS {
            assert!(source.tensors().iter().any(|tensor| tensor.name == name), "{name}");
        }
    }
}
