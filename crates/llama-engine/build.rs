//! Compila as pontes C que o bindgen do `llama-cpp-sys-2` não emite.
//!
//! Estava tudo no `build.rs` de `crates/node`; mudou para cá junto com o código
//! Rust que fala com o llama.cpp, para que editar o backend/rede de `node` não
//! recompile este lado. Ver `docs/subsystems/build-and-run.md`.

use std::path::PathBuf;

#[path = "build/mtp_arch.rs"]
mod mtp_arch;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root");

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    println!("cargo:rerun-if-changed=build/mtp_arch.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_SOURCE");
    let prepared_source = std::env::var_os("DEP_LLAMA_LLAMA_SOURCE").map(PathBuf::from);
    let source_override = std::env::var_os("LLAMA_SOURCE").map(PathBuf::from);
    let llama_cpp_root = mtp_arch::llama_cpp_source(
        &root,
        prepared_source.as_deref(),
        source_override.as_deref(),
    );
    mtp_arch::generate(&llama_cpp_root, &out_dir);
    build_mtp_shim();
}

/// Compila a ponte C para as entradas NextN/MTP do llama.cpp.
///
/// Ver o cabeçalho de `src/llama_mtp_shim.cpp` para o porquê de existir uma
/// ponte em vez de um `#[link_name]` com o nome mangled. Não acrescenta
/// requisito de build: o `llama-cpp-sys-2` já exige um compilador C++.
///
/// A ordem do link importa em `ld` do GNU (esquerda para a direita): esta
/// biblioteca referencia símbolos do `libllama`, então tem que aparecer antes
/// dele na linha. O rustc emite as bibliotecas nativas do crate atual antes das
/// dos dependentes; como `llama-cpp-sys-2` é dependência deste crate, essa
/// ordem se mantém.
fn build_mtp_shim() {
    println!("cargo:rerun-if-changed=src/llama_mtp_shim.cpp");
    cc::Build::new()
        .cpp(true)
        .file("src/llama_mtp_shim.cpp")
        .compile("pom_llama_mtp_shim");
}
