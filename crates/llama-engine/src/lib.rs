//! Motor de inferência llama.cpp do POM, isolado de `crates/node`.
//!
//! Aqui vive **todo** o código que fala com `llama-cpp-sys-2`: o wrapper FFI
//! (`llm`), a predição multi-token (`mtp`), o executor MoE próprio (`moe`), a
//! barreira de reduce do tensor parallelism (`collective`), a leitura de
//! cabeçalhos GGUF (`gguf` + `multipart`), o cálculo de orçamento de KV
//! (`kv_budget`) e o monitor de sistema/GPU (`sysmon`). `crates/node` depende
//! deste crate e re-exporta estes módulos, então editar o backend/rede/
//! roteamento de `node` não reconstrói nada disto.
//!
//! O que **não** mora aqui: `inference/tp.rs` (a coletiva fala com o primary
//! por HTTP/túnel libp2p) fica em `node`; `llm` alcança o estado dela pela
//! ponte fina `tp_probe`, que `node` instala no arranque.
//!
//! Ver `docs/subsystems/build-and-run.md` (o preparo do fork em `/llama.cpp` e
//! as features `cuda`/`metal`/`vulkan` continuam valendo, agora neste crate).

pub mod collective;
pub mod device_placement;
pub mod gguf;
pub mod kv_budget;
pub mod kv_cache;
#[path = "llm/mod.rs"]
pub mod llm;
pub mod moe;
pub mod mtp;
pub mod mtp_draft;
pub mod multipart;
pub mod sysmon;
pub mod tp_probe;
pub mod wire;

/// Prefixo das chaves GGUF próprias do POM (`pom.<sufixo>`). Tem que ser igual
/// ao `slug` de `brand.json` no repositório do POM: o `pom-jair` tem um teste
/// que confere essa igualdade, para que um rename não deixe as duas pontas
/// lendo nomes diferentes.
pub const GGUF_PREFIX: &str = "pom";

/// Chave de metadado GGUF do POM: `gguf_key("tp_size")` -> `pom.tp_size`.
pub fn gguf_key(suffix: &str) -> String {
    format!("{GGUF_PREFIX}.{suffix}")
}

pub use kv_cache::KvCacheType;
pub use mtp::MtpPlacement;

/// The micro-batch a context uses under the memory fit when a deployment names
/// none: the value that was fixed in the code before it became a setting.
pub const DEFAULT_N_UBATCH: u32 = 512;

/// Folga conservadora para a projeção de alocação CUDA: 256 MiB mantém o ganho
/// de capacidade observado no modelo híbrido sem voltar a consumir 1 GiB.
/// (Era `crate::runtime::DEFAULT_FIT_MEMORY_MARGIN_MB` em `crates/node`.)
pub const DEFAULT_FIT_MEMORY_MARGIN_MB: u32 = 256;

/// Re-export da FFI crua, para quem depende do engine alcançá-la sem declarar
/// `llama-cpp-sys-2` de novo (os testes de `models/splitter.rs` em `crates/node`).
pub use llama_cpp_sys_2;
