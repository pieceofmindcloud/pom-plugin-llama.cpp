//! Compila as pontes C/CUDA que o bindgen do `llama-cpp-sys-2` não emite.
//!
//! Estava tudo no `build.rs` de `crates/node`; mudou para cá junto com o código
//! Rust que fala com o llama.cpp, para que editar o backend/rede de `node` não
//! recompile este lado. Ver `docs/subsystems/build-and-run.md`.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "build/cuda_arch.rs"]
mod cuda_arch;
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
    build_moe_cuda(&llama_cpp_root);
}

/// Compila os kernels do executor MoE próprio quando o binário usa CUDA.
///
/// O host continua em Rust e só chama a ABI C mínima exportada pelo arquivo
/// CUDA. Não há Python, Triton ou processo externo no caminho de inferência.
fn build_moe_cuda(llama_cpp_root: &Path) {
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=CCACHE");
    println!("cargo:rerun-if-env-changed=CUDAARCHS");

    let source = Path::new("src/moe/cuda_kernels.cu");
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    // MSVC links `<name>.lib` archives built by lib.exe; everything else takes
    // the Unix `lib<name>.a` from `ar`.
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let (object, library) = if msvc {
        (
            out_dir.join("pom_moe_cuda.obj"),
            out_dir.join("pom_moe_cuda.lib"),
        )
    } else {
        (
            out_dir.join("pom_moe_cuda.o"),
            out_dir.join("libpom_moe_cuda.a"),
        )
    };
    println!("cargo:rerun-if-changed={}", source.display());

    let nvcc = std::env::var_os("CUDACXX")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("CUDA_PATH")
                .map(PathBuf::from)
                .map(|path| path.join("bin/nvcc"))
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    // `CCACHE` is a command wrapper, not a ccache-specific feature. In
    // particular, `CCACHE=sccache` produces `sccache nvcc ...` and uses the
    // same shared cache as the Rust and CMake builds.
    let mut compile = if let Some(wrapper) =
        std::env::var_os("CCACHE").filter(|wrapper| !wrapper.to_string_lossy().is_empty())
    {
        let mut command = Command::new(wrapper);
        command.arg(&nvcc);
        command
    } else {
        Command::new(&nvcc)
    };
    compile.args([
        "-c",
        source.to_str().expect("caminho CUDA UTF-8"),
        "-o",
        object.to_str().expect("objeto CUDA UTF-8"),
        "-O3",
        "--std=c++17",
    ]);
    if msvc {
        // Rust links the dynamic CRT; a static-CRT object would clash at link.
        compile.args(["-Xcompiler", "/MD"]);
    }
    // The IQ3_XXS route reuses llama.cpp's canonical q8_1 vector-dot
    // implementation (MMVQ/dp4a). Keep the include roots tied to the exact
    // fork selected by DEP_LLAMA_LLAMA_SOURCE/LLAMA_SOURCE rather than to the
    // ignored vendored checkout.
    let ggml_root = llama_cpp_root.join("ggml");
    for include in [ggml_root.join("include"), ggml_root.join("src")] {
        compile.arg("-I");
        compile.arg(without_verbatim_prefix(&include));
    }
    for dependency in [
        "src/ggml-common.h",
        "src/ggml-cuda/common.cuh",
        "src/ggml-cuda/vecdotq.cuh",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            ggml_root.join(dependency).display()
        );
    }
    if let Some(architectures) = std::env::var_os("CUDAARCHS") {
        let generations = cuda_arch::nvcc_code_generations(&architectures.to_string_lossy())
            .unwrap_or_else(|error| panic!("invalid CUDAARCHS: {error}"));
        for generation in generations {
            compile.arg("--generate-code").arg(generation);
        }
    }
    let status = compile
        .status()
        .unwrap_or_else(|error| panic!("não foi possível executar {}: {error}", nvcc.display()));
    assert!(status.success(), "nvcc falhou ao compilar cuda_kernels.cu");

    let status = if msvc {
        let target = std::env::var("TARGET").expect("TARGET");
        cc::windows_registry::find(&target, "lib.exe")
            .unwrap_or_else(|| Command::new("lib.exe"))
            .arg("/nologo")
            .arg(format!("/OUT:{}", library.display()))
            .arg(&object)
            .status()
            .expect("não foi possível executar lib.exe para a biblioteca CUDA")
    } else {
        Command::new("ar")
            .args([
                "crus",
                library.to_str().expect("biblioteca CUDA UTF-8"),
                object.to_str().expect("objeto CUDA UTF-8"),
            ])
            .status()
            .expect("não foi possível executar ar para a biblioteca CUDA")
    };
    assert!(status.success(), "falha ao empacotar os kernels CUDA");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=pom_moe_cuda");
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

/// A Windows path without the `\\?\` verbatim prefix `canonicalize` adds.
///
/// The fork's source path reaches this build already canonicalized
/// (`DEP_LLAMA_LLAMA_SOURCE`), and neither nvcc nor cl.exe understands a
/// verbatim path in `-I`: the Windows CUDA build failed with "Cannot open
/// include file: 'ggml-common.h'". Other paths pass through unchanged.
fn without_verbatim_prefix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        path.to_path_buf()
    }
}
