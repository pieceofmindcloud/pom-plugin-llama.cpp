use std::env;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr as _;

use cmake::Config;
use glob::glob;
use walkdir::DirEntry;

#[path = "src/build_support.rs"]
mod build_support;
#[path = "src/prebuilt.rs"]
mod prebuilt;
use build_support::{
    build_tool_path, clone_destination, destination_can_be_replaced, source_can_be_used_directly,
    source_include_dirs, source_root,
};
use prebuilt::{prepare_prebuilt_libraries, PrebuiltLayout};

fn cuda_option_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

const DEFAULT_LLAMA_CPP_REPOSITORY: &str = "https://github.com/pieceofmindcloud/llama.cpp.git";
const DEFAULT_LLAMA_CPP_REF: &str = "pom";

fn missing_llama_source_error(source: &Path) -> String {
    format!(
        "fork do llama.cpp não está disponível em {}.\n\n\
         O build tentou preparar automaticamente a branch `pom` de\n\
         https://github.com/pieceofmindcloud/llama.cpp.git, mas não encontrou um checkout\n\
         com `ggml`. Verifique a rede e o Git, ou prepare manualmente com:\n\
         rm -f {}\n\
         git clone --depth 1 --branch pom https://github.com/pieceofmindcloud/llama.cpp.git {}\n\n\
         Se você já tem um checkout local, use `LLAMA_SOURCE=/caminho/para/llama.cpp`\n\
         ao executar o build. O diretório informado precisa conter `ggml`.\n\
         Consulte docs/subsystems/build-and-run.md, seção `Preparar um worktree local`.",
        source.display(),
        source.display(),
        source.display(),
    )
}

fn canonical_local_source(source: PathBuf) -> Result<PathBuf, String> {
    if !source.join("ggml").is_dir() {
        return Err(format!(
            "LLAMA_SOURCE aponta para um checkout inválido: {}.\n\
             O diretório precisa existir e conter `ggml`.",
            source.display(),
        ));
    }
    source.canonicalize().map_err(|error| {
        format!(
            "não foi possível resolver o checkout indicado por LLAMA_SOURCE em {}: {error}",
            source.display()
        )
    })
}

fn source_symlink_target(source: &Path) -> Result<Option<PathBuf>, String> {
    let metadata = match std::fs::symlink_metadata(source) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "não foi possível inspecionar {}: {error}",
                source.display()
            ));
        }
    };
    if !metadata.file_type().is_symlink() {
        return Ok(None);
    }
    std::fs::read_link(source).map(Some).map_err(|error| {
        format!(
            "não foi possível ler o symlink {}: {error}",
            source.display()
        )
    })
}

fn ensure_llama_source(manifest_dir: &Path) -> Result<PathBuf, String> {
    // An explicit source must take precedence over the vendored symlink. In
    // Docker release mode that symlink may resolve to an empty named context,
    // while LLAMA_SOURCE points to the matching release source archive.
    if let Ok(local_source) = env::var("LLAMA_SOURCE") {
        return canonical_local_source(PathBuf::from(local_source));
    }

    let source = manifest_dir.join("llama.cpp");
    let symlink_target = source_symlink_target(&source)?;
    // Windows checkouts may leave the versioned symlink as a reparse point or
    // a text placeholder. WalkDir must receive a real directory on that host.
    if source_can_be_used_directly(
        source.join("ggml").is_dir(),
        symlink_target.is_some(),
        cfg!(windows),
    ) {
        return Ok(source);
    }

    if let Ok(metadata) = std::fs::metadata(&source) {
        if metadata.is_dir() {
            return Err(format!(
                "o checkout de llama.cpp em {} está incompleto: falta o diretório `ggml`.\n\
                 Remova o checkout incompleto ou aponte `LLAMA_SOURCE` para um checkout\n\
                 do fork `pieceofmindcloud/llama.cpp` que contenha `ggml`.",
                source.display(),
            ));
        }
    }

    let destination = clone_destination(
        manifest_dir,
        &source,
        symlink_target.as_deref(),
        cfg!(windows),
    );
    // A Windows placeholder at the versioned path is safe to replace; an
    // incomplete checkout anywhere else must remain operator-managed.
    if destination.exists()
        && !destination.join("ggml").is_dir()
        && !destination_can_be_replaced(&destination, &source, cfg!(windows))
    {
        return Err(format!(
            "o caminho de llama.cpp em {} está incompleto: falta o diretório `ggml`.\n\
             Não será sobrescrito automaticamente. Remova esse checkout ou use\n\
             `LLAMA_SOURCE=/caminho/para/llama.cpp` com um checkout válido.",
            destination.display(),
        ));
    }
    let repository = env::var("POM_LLAMA_CPP_REPOSITORY")
        .unwrap_or_else(|_| DEFAULT_LLAMA_CPP_REPOSITORY.to_owned());
    let reference =
        env::var("POM_LLAMA_CPP_REF").unwrap_or_else(|_| DEFAULT_LLAMA_CPP_REF.to_owned());

    println!("cargo:rerun-if-env-changed=POM_LLAMA_CPP_REPOSITORY");
    println!("cargo:rerun-if-env-changed=POM_LLAMA_CPP_REF");
    println!("cargo:rerun-if-env-changed=POM_LLAMA_CPP_AUTOCLONE");

    if env::var("POM_LLAMA_CPP_AUTOCLONE").as_deref() == Ok("0") {
        return Err(missing_llama_source_error(&source));
    }

    let parent = destination.parent().ok_or_else(|| {
        format!(
            "não foi possível determinar onde preparar o fork: {}",
            destination.display()
        )
    })?;
    std::fs::create_dir_all(parent).map_err(|error| {
        format!(
            "não foi possível criar o diretório de preparo {}: {error}",
            parent.display()
        )
    })?;

    let temporary = parent.join(format!(".llama.cpp.clone-{}", std::process::id()));
    if temporary.exists() {
        return Err(format!(
            "já existe um preparo temporário do fork em {}. Remova-o após confirmar que\
             nenhum outro build está rodando e tente novamente.",
            temporary.display(),
        ));
    }

    println!("cargo:warning=llama.cpp ausente; preparando branch `{reference}` de {repository}");
    let clone_status = Command::new("git")
        .arg("clone")
        .arg("--depth")
        .arg("1")
        .arg("--branch")
        .arg(&reference)
        .arg(&repository)
        .arg(&temporary)
        .output()
        .map_err(|error| {
            format!(
                "não foi possível executar Git para preparar o fork: {error}.\n\n{}",
                missing_llama_source_error(&source)
            )
        })?;
    if !clone_status.status.success() {
        let details = String::from_utf8_lossy(&clone_status.stderr);
        let _ = std::fs::remove_dir_all(&temporary);
        return Err(format!(
            "não foi possível baixar o fork do llama.cpp (git clone saiu com {}).\n\
             Git informou: {}\n\n{}",
            clone_status.status,
            details.trim(),
            missing_llama_source_error(&source),
        ));
    }

    if !temporary.join("ggml").is_dir() {
        let _ = std::fs::remove_dir_all(&temporary);
        return Err(format!(
            "o fork baixado não contém o diretório `ggml`; a referência `{reference}`\
             não parece ser um checkout compatível.\n\n{}",
            missing_llama_source_error(&source),
        ));
    }

    if destination == source {
        if let Ok(metadata) = std::fs::symlink_metadata(&source) {
            if metadata.file_type().is_symlink() || metadata.is_file() {
                std::fs::remove_file(&source).map_err(|error| {
                    format!(
                        "não foi possível substituir o arquivo de symlink {}: {error}",
                        source.display()
                    )
                })?;
            }
        }
    }
    std::fs::rename(&temporary, &destination).map_err(|error| {
        let _ = std::fs::remove_dir_all(&temporary);
        format!(
            "não foi possível instalar o fork preparado em {}: {error}.\n\n{}",
            destination.display(),
            missing_llama_source_error(&source),
        )
    })?;

    Ok(source)
}

enum WindowsVariant {
    Msvc,
    Other,
}

enum AppleVariant {
    MacOS,
    WatchOS,
    Other,
}

enum TargetOs {
    Windows(WindowsVariant),
    Apple(AppleVariant),
    Linux,
    Android,
}

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("BUILD_DEBUG").is_ok() {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

fn emit_compiler_static_archive_search_path(archive: &str) {
    let compiler = cc::Build::new().get_compiler();
    let Ok(output) = Command::new(compiler.path())
        .arg(format!("--print-file-name={archive}"))
        .output()
    else {
        return;
    };

    if !output.status.success() {
        return;
    }

    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let path = Path::new(&path);
    if path.is_file() {
        if let Some(parent) = path.parent() {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
    }
}

fn parse_target_os() -> Result<(TargetOs, String), String> {
    let target = env::var("TARGET").unwrap();

    if target.contains("windows") {
        if target.ends_with("-windows-msvc") {
            Ok((TargetOs::Windows(WindowsVariant::Msvc), target))
        } else {
            Ok((TargetOs::Windows(WindowsVariant::Other), target))
        }
    } else if target.contains("apple") {
        if target.ends_with("-apple-darwin") {
            Ok((TargetOs::Apple(AppleVariant::MacOS), target))
        } else if target.contains("watchos") {
            Ok((TargetOs::Apple(AppleVariant::WatchOS), target))
        } else {
            Ok((TargetOs::Apple(AppleVariant::Other), target))
        }
    } else if target.contains("android")
        || target == "aarch64-linux-android"
        || target == "armv7-linux-androideabi"
        || target == "i686-linux-android"
        || target == "x86_64-linux-android"
    {
        // Handle both full android targets and short names like arm64-v8a that cargo ndk might use
        Ok((TargetOs::Android, target))
    } else if target.contains("linux") {
        Ok((TargetOs::Linux, target))
    } else {
        Err(target)
    }
}

fn get_cargo_target_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let path = PathBuf::from(out_dir);
    let target_dir = path
        .ancestors()
        .nth(3)
        .ok_or("OUT_DIR is not deep enough")?;
    Ok(target_dir.to_path_buf())
}

fn extract_lib_names(out_dir: &Path, build_shared_libs: bool, target_os: &TargetOs) -> Vec<String> {
    let lib_pattern = match target_os {
        // MSVC emits .lib; the GNU (MinGW) toolchain emits .a static archives.
        TargetOs::Windows(WindowsVariant::Msvc) => "*.lib",
        TargetOs::Windows(_) => "*.a",
        TargetOs::Apple(_) => {
            if build_shared_libs {
                "*.dylib"
            } else {
                "*.a"
            }
        }
        TargetOs::Linux | TargetOs::Android => {
            if build_shared_libs {
                "*.so"
            } else {
                "*.a"
            }
        }
    };
    let libs_dir = out_dir.join("lib*");
    let pattern = libs_dir.join(lib_pattern);
    debug_log!("Extract libs {}", pattern.display());

    let mut lib_names: Vec<String> = Vec::new();

    // Process the libraries based on the pattern
    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                let stem = path.file_stem().unwrap();
                let stem_str = stem.to_str().unwrap();

                // Remove the "lib" prefix if present
                let lib_name = if stem_str.starts_with("lib") {
                    stem_str.strip_prefix("lib").unwrap_or(stem_str)
                } else {
                    if path.extension() == Some(std::ffi::OsStr::new("a")) {
                        let target = path.parent().unwrap().join(format!("lib{}.a", stem_str));
                        std::fs::rename(&path, &target).unwrap_or_else(|e| {
                            panic!("Failed to rename {path:?} to {target:?}: {e:?}");
                        })
                    }
                    stem_str
                };
                lib_names.push(lib_name.to_string());
            }
            Err(e) => println!("cargo:warning=error={}", e),
        }
    }
    lib_names
}

fn extract_lib_assets(out_dir: &Path, target_os: &TargetOs) -> Vec<PathBuf> {
    let shared_lib_pattern = match target_os {
        TargetOs::Windows(_) => "*.dll",
        TargetOs::Apple(_) => "*.dylib",
        TargetOs::Linux | TargetOs::Android => "*.so",
    };

    let shared_libs_dir = match target_os {
        TargetOs::Windows(_) => "bin",
        _ => "lib",
    };
    let libs_dir = out_dir.join(shared_libs_dir);
    let pattern = libs_dir.join(shared_lib_pattern);
    debug_log!("Extract lib assets {}", pattern.display());
    let mut files = Vec::new();

    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                files.push(path);
            }
            Err(e) => eprintln!("cargo:warning=error={}", e),
        }
    }

    files
}

fn library_file_exists(
    search_dirs: &[PathBuf],
    lib_name: &str,
    build_shared_libs: bool,
    target_os: &TargetOs,
) -> bool {
    let (prefixes, extensions): (&[&str], &[&str]) = match target_os {
        TargetOs::Windows(_) => (&["", "lib"], &["lib"]),
        TargetOs::Apple(_) => {
            if build_shared_libs {
                (&["lib"], &["dylib"])
            } else {
                (&["lib"], &["a"])
            }
        }
        TargetOs::Linux | TargetOs::Android => {
            if build_shared_libs {
                (&["lib"], &["so"])
            } else {
                (&["lib"], &["a"])
            }
        }
    };

    search_dirs.iter().any(|dir| {
        prefixes.iter().any(|prefix| {
            extensions.iter().any(|extension| {
                dir.join(format!("{prefix}{lib_name}.{extension}"))
                    .is_file()
            })
        })
    })
}

fn macos_link_search_path() -> Option<String> {
    let output = Command::new("clang")
        .arg("--print-search-dirs")
        .output()
        .ok()?;
    if !output.status.success() {
        println!(
            "failed to run 'clang --print-search-dirs', continuing without a link search path"
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;
            return Some(format!("{}/lib/darwin", path));
        }
    }

    println!("failed to determine link search path, continuing without it");
    None
}

fn validate_android_ndk(ndk_path: &str) -> Result<(), String> {
    let ndk_path = Path::new(ndk_path);

    if !ndk_path.exists() {
        return Err(format!(
            "Android NDK path does not exist: {}",
            ndk_path.display()
        ));
    }

    let toolchain_file = ndk_path.join("build/cmake/android.toolchain.cmake");
    if !toolchain_file.exists() {
        return Err(format!(
            "Android NDK toolchain file not found: {}\n\
             This indicates an incomplete NDK installation.",
            toolchain_file.display()
        ));
    }

    Ok(())
}

/// Android NDK toolchain facts derived from the environment for a given Rust target
/// triple. Shared by the bindgen setup (clang args) and the Vulkan backend build (CMake
/// cache vars) so the two agree on which NDK, sysroot, arch and API level to use.
struct AndroidToolchain {
    /// NDK root (e.g. the value of `ANDROID_NDK`).
    ndk: String,
    /// `<ndk>/toolchains/llvm/prebuilt/<host_tag>`
    toolchain_path: String,
    /// `<toolchain_path>/sysroot`
    sysroot: String,
    /// NDK sysroot arch directory / triple, e.g. `aarch64-linux-android`.
    arch_triple: &'static str,
    /// CMake `ANDROID_ABI` name, e.g. `arm64-v8a`.
    abi: &'static str,
    /// Android API level (numeric), e.g. `28`.
    api: String,
}

fn android_toolchain(target_triple: &str) -> AndroidToolchain {
    // NDK path: explicit env vars first, then auto-detect the newest NDK under the SDK.
    let ndk = env::var("ANDROID_NDK")
        .or_else(|_| env::var("ANDROID_NDK_ROOT"))
        .or_else(|_| env::var("NDK_ROOT"))
        .or_else(|_| env::var("CARGO_NDK_ANDROID_NDK"))
        .or_else(|_| {
            if let Some(home) = env::home_dir() {
                let android_home = env::var("ANDROID_HOME")
                    .or_else(|_| env::var("ANDROID_SDK_ROOT"))
                    .unwrap_or_else(|_| format!("{}/Android/Sdk", home.display()));
                let ndk_dir = format!("{}/ndk", android_home);
                if let Ok(entries) = std::fs::read_dir(&ndk_dir) {
                    let mut versions: Vec<_> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                        .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                        .collect();
                    versions.sort();
                    if let Some(latest) = versions.last() {
                        return Ok(format!("{}/{}", ndk_dir, latest));
                    }
                }
            }
            Err(env::VarError::NotPresent)
        })
        .unwrap_or_else(|_| {
            panic!(
                "Android NDK not found. Please set one of: ANDROID_NDK, NDK_ROOT, ANDROID_NDK_ROOT\n\
                 Current target: {target_triple}\n\
                 Download from: https://developer.android.com/ndk/downloads"
            )
        });

    if let Err(e) = validate_android_ndk(&ndk) {
        panic!("{e}");
    }

    // API level: ANDROID_API_LEVEL, else the numeric part of ANDROID_PLATFORM (`android-NN`).
    let api = env::var("ANDROID_API_LEVEL")
        .or_else(|_| env::var("ANDROID_PLATFORM").map(|p| p.replace("android-", "")))
        .or_else(|_| env::var("CARGO_NDK_ANDROID_PLATFORM").map(|p| p.replace("android-", "")))
        .unwrap_or_else(|_| "28".to_string());

    let host_tag = if cfg!(target_os = "macos") {
        "darwin-x86_64"
    } else if cfg!(target_os = "linux") {
        "linux-x86_64"
    } else if cfg!(target_os = "windows") {
        "windows-x86_64"
    } else {
        panic!("Unsupported host platform for Android NDK");
    };

    // Both the sysroot arch triple (used for header/lib paths) and the CMake ABI name.
    let (arch_triple, abi) = if target_triple.contains("aarch64") {
        ("aarch64-linux-android", "arm64-v8a")
    } else if target_triple.contains("armv7") {
        ("arm-linux-androideabi", "armeabi-v7a")
    } else if target_triple.contains("x86_64") {
        ("x86_64-linux-android", "x86_64")
    } else if target_triple.contains("i686") {
        ("i686-linux-android", "x86")
    } else {
        panic!(
            "Unsupported Android target: {target_triple}\n\
             Supported: aarch64-linux-android, armv7-linux-androideabi, \
             i686-linux-android, x86_64-linux-android"
        );
    };

    let toolchain_path = format!("{ndk}/toolchains/llvm/prebuilt/{host_tag}");
    if !Path::new(&toolchain_path).exists() {
        panic!(
            "Android NDK toolchain not found at: {toolchain_path}\n\
             Please ensure you have the correct Android NDK for your platform."
        );
    }
    let sysroot = format!("{toolchain_path}/sysroot");

    // Any NDK / platform env change should re-trigger the build (all Android steps depend
    // on these), so emit the directives here rather than in one specific call site.
    for var in [
        "ANDROID_NDK",
        "ANDROID_NDK_ROOT",
        "NDK_ROOT",
        "CARGO_NDK_ANDROID_NDK",
        "ANDROID_PLATFORM",
        "ANDROID_API_LEVEL",
        "CARGO_NDK_ANDROID_PLATFORM",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    AndroidToolchain {
        ndk,
        toolchain_path,
        sysroot,
        arch_triple,
        abi,
        api,
    }
}

fn is_hidden(e: &DirEntry) -> bool {
    e.file_name()
        .to_str()
        .map(|s| s.starts_with('.'))
        .unwrap_or_default()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_SOURCE");

    let (target_os, target_triple) =
        parse_target_os().unwrap_or_else(|t| panic!("Failed to parse target os {t}"));
    let target_is_windows = matches!(&target_os, TargetOs::Windows(_));
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let target_dir = get_cargo_target_dir().unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let llama_src =
        ensure_llama_source(Path::new(&manifest_dir)).unwrap_or_else(|error| panic!("{error}"));
    let llama_include_root = source_root(&llama_src);
    // O `llama-engine` também inspeciona o checkout para gerar sua lista MTP.
    // O metadado mantém os dois build scripts no caminho exato escolhido aqui,
    // inclusive o caminho materializado no Windows e um `LLAMA_SOURCE` explícito.
    println!("cargo:llama_source={}", llama_src.display());
    let build_shared_libs = cfg!(feature = "dynamic-link");

    let build_shared_libs = std::env::var("LLAMA_BUILD_SHARED_LIBS")
        .map(|v| v == "1")
        .unwrap_or(build_shared_libs);
    println!("cargo:rerun-if-env-changed=POM_LLAMA_CPP_PREBUILT_LIB_DIR");
    let prebuilt_lib_dir = env::var_os("POM_LLAMA_CPP_PREBUILT_LIB_DIR").map(PathBuf::from);
    if prebuilt_lib_dir.is_some() {
        assert!(build_shared_libs, "prebuilt llama.cpp requires shared linking");
        assert!(
            matches!(
                &target_os,
                TargetOs::Linux | TargetOs::Windows(WindowsVariant::Msvc)
            ),
            "prebuilt llama.cpp is supported only on Linux and Windows (MSVC)"
        );
    }
    let profile = env::var("LLAMA_LIB_PROFILE").unwrap_or("Release".to_string());
    let static_crt = env::var("LLAMA_STATIC_CRT")
        .map(|v| v == "1")
        .unwrap_or(false);

    println!("cargo:rerun-if-env-changed=LLAMA_LIB_PROFILE");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_SHARED_LIBS");
    println!("cargo:rerun-if-env-changed=LLAMA_STATIC_CRT");
    for variable in [
        "CMAKE_C_COMPILER_LAUNCHER",
        "CMAKE_CXX_COMPILER_LAUNCHER",
        "CMAKE_CUDA_COMPILER_LAUNCHER",
        "CMAKE_Vulkan_LIBRARY",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    for variable in ["Vulkan_LIBRARY", "VULKAN_SDK"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    debug_log!("TARGET: {}", target_triple);
    debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir);
    debug_log!("TARGET_DIR: {}", target_dir.display());
    debug_log!("OUT_DIR: {}", out_dir.display());
    debug_log!("BUILD_SHARED: {}", build_shared_libs);

    // Make sure that changes to the llama.cpp project trigger a rebuild.
    let rebuild_on_children_of = [
        llama_src.join("src"),
        llama_src.join("ggml/src"),
        llama_src.join("common"),
    ];
    for entry in walkdir::WalkDir::new(&llama_src)
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
    {
        let entry = entry.expect("Failed to obtain entry");
        let rebuild = entry
            .file_name()
            .to_str()
            .map(|f| f.starts_with("CMake"))
            .unwrap_or_default()
            || rebuild_on_children_of
                .iter()
                .any(|src_folder| entry.path().starts_with(src_folder));
        if rebuild {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    // Use all available cores except 2 to
    let cmake_build_parallelism_level =
        match env::var("CMAKE_BUILD_PARALLEL_LEVEL").map(|v| NonZeroUsize::from_str(&v)) {
            Ok(Ok(v)) => v.to_string(),
            _ => std::thread::available_parallelism()
                .expect("failed to load available parallelism")
                .get()
                .to_string(),
        };
    env::set_var("CMAKE_BUILD_PARALLEL_LEVEL", cmake_build_parallelism_level);

    // Bindings
    let mut bindings_builder = bindgen::Builder::default()
        .header("wrapper.h")
        // The wrappers use leaf header names. The explicit source subdirectories
        // below work for both the versioned symlink and an external LLAMA_SOURCE
        // used by CI, including the mtmd-only wrapper.
        .clang_arg(format!("-I{}", llama_include_root.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .prepend_enum_name(false);

    for include_dir in source_include_dirs(&llama_src) {
        bindings_builder = bindings_builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    // The `llama_rs_*` symbols are emitted by `wrapper_common.cpp`, which is
    // only compiled (and only has its header included from `wrapper.h`) when
    // the `common` feature is enabled.
    if cfg!(feature = "common") {
        bindings_builder = bindings_builder
            .clang_arg("-DLLAMA_RS_BUILD_COMMON")
            .allowlist_function("llama_rs_.*")
            .allowlist_type("llama_rs_.*");
    }

    // Configure mtmd feature if enabled
    if cfg!(feature = "mtmd") {
        bindings_builder = bindings_builder
            .header("wrapper_mtmd.h")
            .allowlist_function("mtmd_.*")
            .allowlist_type("mtmd_.*");
    }

    // Configure Android-specific bindgen settings
    if matches!(target_os, TargetOs::Android) {
        let tc = android_toolchain(&target_triple);

        // Find clang builtin includes
        let clang_builtin_includes = {
            let clang_lib_path = format!("{}/lib/clang", tc.toolchain_path);
            std::fs::read_dir(&clang_lib_path).ok().and_then(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .find(|entry| {
                        entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                            && entry
                                .file_name()
                                .to_str()
                                .map(|name| name.chars().next().unwrap_or('0').is_ascii_digit())
                                .unwrap_or(false)
                    })
                    .and_then(|entry| {
                        let include_path =
                            format!("{}/{}/include", clang_lib_path, entry.file_name().to_str()?);
                        if std::path::Path::new(&include_path).exists() {
                            Some(include_path)
                        } else {
                            None
                        }
                    })
            })
        };

        // Configure bindgen for Android
        bindings_builder = bindings_builder
            .clang_arg(format!("--sysroot={}", tc.sysroot))
            .clang_arg(format!("-D__ANDROID_API__={}", tc.api))
            .clang_arg("-D__ANDROID__");

        // Add include paths in correct order
        if let Some(ref builtin_includes) = clang_builtin_includes {
            bindings_builder = bindings_builder
                .clang_arg("-isystem")
                .clang_arg(builtin_includes);
        }

        bindings_builder = bindings_builder
            .clang_arg("-isystem")
            .clang_arg(format!("{}/usr/include/{}", tc.sysroot, tc.arch_triple))
            .clang_arg("-isystem")
            .clang_arg(format!("{}/usr/include", tc.sysroot))
            .clang_arg("-include")
            .clang_arg("stdbool.h")
            .clang_arg("-include")
            .clang_arg("stdint.h");

        // Set additional clang args for cargo ndk compatibility
        if env::var("CARGO_SUBCOMMAND").as_deref() == Ok("ndk") {
            std::env::set_var(
                "BINDGEN_EXTRA_CLANG_ARGS",
                format!("--target={}", target_triple),
            );
        }
    }

    // Fix bindgen header discovery on Windows MSVC
    // Use cc crate to discover MSVC include paths by compiling a dummy file
    if matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc)) {
        // Create a minimal dummy C file to extract compiler flags
        let out_dir = env::var("OUT_DIR").unwrap();
        let dummy_c = Path::new(&out_dir).join("dummy.c");
        std::fs::write(&dummy_c, "int main() { return 0; }").unwrap();

        // Use cc crate to get compiler with proper environment setup
        let mut build = cc::Build::new();
        build.file(build_tool_path(&dummy_c, target_is_windows));

        // Get the actual compiler command cc would use
        let compiler = build.try_get_compiler().unwrap();

        // Extract include paths by checking compiler's environment
        // cc crate sets up MSVC environment internally
        let env_include = compiler
            .env()
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("INCLUDE"))
            .map(|(_, v)| v);

        if let Some(include_paths) = env_include {
            for include_path in include_paths
                .to_string_lossy()
                .split(';')
                .filter(|s| !s.is_empty())
            {
                bindings_builder = bindings_builder
                    .clang_arg("-isystem")
                    .clang_arg(include_path);
                debug_log!("Added MSVC include path: {}", include_path);
            }
        }

        // Add MSVC compatibility flags
        bindings_builder = bindings_builder
            .clang_arg(format!("--target={}", target_triple))
            .clang_arg("-fms-compatibility")
            .clang_arg("-fms-extensions");

        debug_log!(
            "Configured bindgen with MSVC toolchain for target: {}",
            target_triple
        );
    }
    let bindings = bindings_builder
        .generate()
        .expect("Failed to generate bindings");

    // Write the generated bindings to an output file
    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(bindings_path)
        .expect("Failed to write bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper_common.h");
    println!("cargo:rerun-if-changed=wrapper_common.cpp");
    println!("cargo:rerun-if-changed=wrapper_utils.h");
    println!("cargo:rerun-if-changed=wrapper_mtmd.h");

    debug_log!("Bindings Created");

    if cfg!(feature = "common") {
        let mut common_wrapper_build = cc::Build::new();
        common_wrapper_build
            .cpp(true)
            .file("wrapper_common.cpp")
            .flag_if_supported("-std=c++17")
            .pic(true);

        // Clang and CMake accept Windows extended paths, but some MSVC
        // versions reject the same `\\?\` prefix in `/I` arguments.
        common_wrapper_build.include(build_tool_path(&llama_include_root, target_is_windows));
        for include_dir in source_include_dirs(&llama_src) {
            common_wrapper_build.include(build_tool_path(&include_dir, target_is_windows));
        }

        if matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc)) {
            common_wrapper_build.flag("/std:c++17");
        }

        // When static-stdcxx is enabled on Android, suppress the cc crate's automatic
        // C++ stdlib linking (which defaults to c++_shared) so we can link c++_static instead.
        if matches!(target_os, TargetOs::Android) && cfg!(feature = "static-stdcxx") {
            common_wrapper_build.cpp_link_stdlib(None);
        }

        common_wrapper_build.compile("llama_cpp_sys_2_common_wrapper");
    }

    // Build with Cmake

    // Bindgen accepts the Windows extended path, but the CMake crate forwards
    // its source path to CMake/MSBuild, which may not accept the `\\?\` prefix.
    let cmake_source = build_tool_path(&llama_src, target_is_windows);
    let mut config = Config::new(&cmake_source);

    // Would require extra source files to pointlessly
    // be included in what's uploaded to and downloaded from
    // crates.io, so deactivating these instead
    config.define("LLAMA_BUILD_TESTS", "OFF");
    config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    config.define("LLAMA_BUILD_SERVER", "OFF");
    config.define("LLAMA_BUILD_TOOLS", "OFF");
    // `app` (the unified `llama` binary) defaults to ON when llama.cpp is the
    // top-level CMake project; it pulls in server/tool internals we don't build.
    config.define("LLAMA_BUILD_APP", "OFF");
    config.define(
        "LLAMA_BUILD_COMMON",
        if cfg!(feature = "common") {
            "ON"
        } else {
            "OFF"
        },
    );
    config.define("LLAMA_CURL", "OFF");

    // Pass CMAKE_ environment variables down to CMake. The launcher variables
    // above are declared explicitly as rerun inputs because `define` only
    // creates CMake cache entries and does not make Cargo observe the env vars.
    for (key, value) in env::vars() {
        if key.starts_with("CMAKE_") {
            config.define(&key, &value);
        }
    }

    // Also forward GGML_ environment variables as CMake cache entries, so a
    // downstream can toggle any ggml build option without patching this build
    // script. For example `GGML_CPU_REPACK=OFF` keeps Q4_0 weights in their
    // original (mmap-friendly) layout instead of the runtime-repacked, anon-RAM
    // one — needed when mmap-streaming a model that is larger than RAM.
    //
    // Precedence: options this build script sets explicitly further down
    // (GGML_NATIVE, GGML_AVX*, GGML_CUDA, …) are defined after this loop and so
    // win over any env value; only options left unset here are overridable.
    for (key, value) in env::vars() {
        if key.starts_with("GGML_") {
            println!("cargo:rerun-if-env-changed={key}");
            config.define(&key, &value);
        }
    }

    // extract the target-cpu config value, if specified
    let target_cpu = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .ok()
        .and_then(|rustflags| {
            rustflags
                .split('\x1f')
                .find(|f| f.contains("target-cpu="))
                .and_then(|f| f.split("target-cpu=").nth(1))
                .map(|s| s.to_string())
        });

    if target_cpu == Some("native".into()) {
        debug_log!("Detected target-cpu=native, compiling with GGML_NATIVE");
        config.define("GGML_NATIVE", "ON");
    }
    // if native isn't specified, enable specific features for ggml instead
    else {
        // rust code isn't using `target-cpu=native`, so llama.cpp shouldn't use GGML_NATIVE either
        config.define("GGML_NATIVE", "OFF");

        // if `target-cpu` is set set, also set -march for llama.cpp to the same value
        if let Some(ref cpu) = target_cpu {
            debug_log!("Setting baseline architecture: -march={}", cpu);
            config.cflag(format!("-march={}", cpu));
            config.cxxflag(format!("-march={}", cpu));
        }

        // I expect this env var to always be present
        let features = std::env::var("CARGO_CFG_TARGET_FEATURE")
            .expect("Env var CARGO_CFG_TARGET_FEATURE not found.");
        debug_log!("Compiling with target features: {}", features);

        // list of rust target_features here:
        //   https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute
        // GGML config flags have been found by looking at:
        //   llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt
        for feature in features.split(',') {
            match feature {
                "avx" => {
                    config.define("GGML_AVX", "ON");
                }
                "avx2" => {
                    config.define("GGML_AVX2", "ON");
                }
                "avx512bf16" => {
                    config.define("GGML_AVX512_BF16", "ON");
                }
                "avx512vbmi" => {
                    config.define("GGML_AVX512_VBMI", "ON");
                }
                "avx512vnni" => {
                    config.define("GGML_AVX512_VNNI", "ON");
                }
                "avxvnni" => {
                    config.define("GGML_AVX_VNNI", "ON");
                }
                "bmi2" => {
                    config.define("GGML_BMI2", "ON");
                }
                "f16c" => {
                    config.define("GGML_F16C", "ON");
                }
                "fma" => {
                    config.define("GGML_FMA", "ON");
                }
                "sse4.2" => {
                    config.define("GGML_SSE42", "ON");
                }
                _ => {
                    debug_log!(
                        "Unrecognized cpu feature: '{}' - skipping GGML config for it.",
                        feature
                    );
                    continue;
                }
            };
        }
    }

    config.define(
        "BUILD_SHARED_LIBS",
        if build_shared_libs { "ON" } else { "OFF" },
    );

    if matches!(target_os, TargetOs::Apple(_)) {
        config.define("GGML_BLAS", "OFF");
    }

    // watchOS has no Metal framework, so disable the Metal backend there.
    // Also define _DARWIN_C_SOURCE so BSD types (u_int, u_char, u_short) used by
    // some sources are visible — implicit on macOS/iOS but not on watchOS.
    if matches!(target_os, TargetOs::Apple(AppleVariant::WatchOS)) {
        config.define("GGML_METAL", "OFF");
        config.cflag("-D_DARWIN_C_SOURCE");
        config.cxxflag("-D_DARWIN_C_SOURCE");
    }

    if (matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc))
        && matches!(
            profile.as_str(),
            "Release" | "RelWithDebInfo" | "MinSizeRel"
        ))
    {
        // Debug Rust builds under MSVC turn off optimization even though we're ideally building the release profile of llama.cpp.
        // Looks like an upstream bug:
        // https://github.com/rust-lang/cmake-rs/issues/240
        // For now explicitly reinject the optimization flags that a CMake Release build is expected to have on in this scenario.
        // This fixes CPU inference performance when part of a Rust debug build.
        for flag in &["/O2", "/DNDEBUG", "/Ob2"] {
            config.cflag(flag);
            config.cxxflag(flag);
        }
    }

    config.static_crt(static_crt);

    if matches!(target_os, TargetOs::Android) {
        if cfg!(feature = "shared-stdcxx") && cfg!(feature = "static-stdcxx") {
            panic!("Features 'shared-stdcxx' and 'static-stdcxx' are mutually exclusive");
        }

        let tc = android_toolchain(&target_triple);

        // CMake cross-compile toolchain file + target platform/ABI selection.
        config.define(
            "CMAKE_TOOLCHAIN_FILE",
            format!("{}/build/cmake/android.toolchain.cmake", tc.ndk),
        );
        config.define("ANDROID_PLATFORM", format!("android-{}", tc.api));
        config.define("ANDROID_ABI", tc.abi);

        // Configure C++ standard library linkage for Android.
        // By default, the NDK toolchain uses c++_shared.
        // The shared-stdcxx and static-stdcxx features allow explicit control.
        if cfg!(feature = "static-stdcxx") {
            config.define("ANDROID_STL", "c++_static");
        } else if cfg!(feature = "shared-stdcxx") {
            config.define("ANDROID_STL", "c++_shared");
        }

        // Configure architecture-specific compiler flags
        match tc.abi {
            "arm64-v8a" => {
                config.cflag("-march=armv8-a");
                config.cxxflag("-march=armv8-a");
                // Allow overriding ggml-cpu's Arm architecture, e.g. to enable the
                // int8 dot-product / matmul kernels (`armv8.2-a+dotprod`,
                // `armv8.6-a+i8mm`) on capable devices — Q4_K_M is several times
                // faster with them. Opt-in via the `GGML_CPU_ARM_ARCH` env var so the
                // default (`armv8-a`) is unchanged; this mirrors the Linux aarch64
                // path, which already sets `GGML_CPU_ARM_ARCH`.
                println!("cargo:rerun-if-env-changed=GGML_CPU_ARM_ARCH");
                if let Ok(arch) = std::env::var("GGML_CPU_ARM_ARCH") {
                    config.define("GGML_CPU_ARM_ARCH", &arch);
                }
            }
            "armeabi-v7a" => {
                config.cflag("-march=armv7-a");
                config.cxxflag("-march=armv7-a");
                config.cflag("-mfpu=neon");
                config.cxxflag("-mfpu=neon");
                config.cflag("-mthumb");
                config.cxxflag("-mthumb");
            }
            "x86_64" => {
                config.cflag("-march=x86-64");
                config.cxxflag("-march=x86-64");
            }
            "x86" => {
                config.cflag("-march=i686");
                config.cxxflag("-march=i686");
            }
            _ => {}
        }

        // Android-specific CMake configurations
        config.define("GGML_LLAMAFILE", "OFF");

        // Link Android system libraries
        println!("cargo:rustc-link-lib=log");
        println!("cargo:rustc-link-lib=android");
    }

    if matches!(target_os, TargetOs::Linux)
        && target_triple.contains("aarch64")
        && target_cpu != Some("native".into())
    {
        // If the target-cpu is not specified as native, we take off the native ARM64 support.
        // It is useful in docker environments where the native feature is not enabled.
        config.define("GGML_NATIVE", "OFF");
        config.define("GGML_CPU_ARM_ARCH", "armv8-a");
    }

    if cfg!(feature = "vulkan") {
        config.define("GGML_VULKAN", "ON");
        match target_os {
            TargetOs::Windows(_) => {
                let vulkan_path = env::var("VULKAN_SDK").expect(
                    "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set",
                );
                let vulkan_lib_path = Path::new(&vulkan_path).join("Lib");
                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
                println!("cargo:rustc-link-lib=vulkan-1");

                // workaround for this error: "FileTracker : error FTK1011: could not create the new file tracking log file"
                // it has to do with MSBuild FileTracker not respecting the path
                // limit configuration set in the windows registry.
                // I'm not sure why that's a thing, but this makes my builds work.
                // (crates that depend on llama-cpp-rs w/ vulkan easily exceed the default PATH_MAX on windows)
                // MSVC-only: MSBuild FileTracker and /FS do not exist on the GNU (MinGW)
                // toolchain — gcc parses "/FS" as a linker input path and fails.
                if matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc)) {
                    env::set_var("TrackFileAccess", "false");
                    // since we disabled TrackFileAccess, we can now run into problems with parallel
                    // access to pdb files. /FS solves this.
                    config.cflag("/FS");
                    config.cxxflag("/FS");
                }
            }
            TargetOs::Linux => {
                // If we are not using system provided vulkan SDK, add vulkan libs for linking
                if let Ok(vulkan_path) = env::var("VULKAN_SDK") {
                    let vulkan_lib_path = Path::new(&vulkan_path).join("lib");
                    println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
                }
                println!("cargo:rustc-link-lib=vulkan");
            }
            TargetOs::Android => {
                // Cross-compiling the Vulkan backend for Android needs a few things the NDK
                // toolchain can't supply on its own, because its CMake toolchain re-roots
                // find_package() into the NDK sysroot (so FindVulkan / find_package(SPIRV-Headers)
                // can't discover host packages):
                //
                //   * Vulkan headers WITH the C++ bindings (`vulkan/vulkan.hpp`). The NDK ships
                //     only the C headers, but ggml-vulkan is C++ — so the caller points
                //     VULKAN_INCLUDE_DIR at a full Vulkan-Hpp header set (e.g. a `vulkan-headers`
                //     package). It may be newer than the device loader; the loader is compatible.
                //   * SPIRV-Headers (ggml-vulkan does `find_package(SPIRV-Headers CONFIG)` and
                //     `#include <spirv/unified1/spirv.hpp>`). SPIRV_HEADERS_DIR points at the dir
                //     holding SPIRV-HeadersConfig.cmake; the headers are added to the compile line
                //     (defaulting to alongside the Vulkan headers, e.g. /usr/include).
                //
                // The Vulkan *loader* to link against is in the NDK sysroot (per-API
                // libvulkan.so) and `glslc` is a host tool on PATH, so both are auto-detected.
                // The device provides the real driver at runtime.
                let tc = android_toolchain(&target_triple);

                // Headers with the C++ bindings — the NDK lacks vulkan.hpp.
                let vk_include = env::var("VULKAN_INCLUDE_DIR").expect(
                    "the Vulkan backend for Android requires VULKAN_INCLUDE_DIR to point at \
                     Vulkan headers that include <vulkan/vulkan.hpp> (the NDK ships only the C \
                     headers)",
                );
                config.define("Vulkan_INCLUDE_DIR", &vk_include);

                // Loader from the NDK sysroot; the device supplies the driver at runtime.
                config.define(
                    "Vulkan_LIBRARY",
                    format!(
                        "{}/usr/lib/{}/{}/libvulkan.so",
                        tc.sysroot, tc.arch_triple, tc.api
                    ),
                );

                // Host shader compiler for the vulkan-shaders-gen build tool.
                let glslc = env::var("VULKAN_GLSLC")
                    .ok()
                    .or_else(|| {
                        env::var_os("PATH").and_then(|paths| {
                            env::split_paths(&paths)
                                .map(|p| p.join("glslc"))
                                .find(|p| p.exists())
                                .map(|p| p.to_string_lossy().into_owned())
                        })
                    })
                    .expect(
                        "the Vulkan backend for Android requires `glslc` on PATH or VULKAN_GLSLC",
                    );
                config.define("Vulkan_GLSLC_EXECUTABLE", glslc);

                // SPIRV-Headers: the CONFIG package plus its include dir on the compile line
                // (the CONFIG target's interface include is not propagated under the toolchain).
                let spirv_dir = env::var("SPIRV_HEADERS_DIR").expect(
                    "the Vulkan backend for Android requires SPIRV_HEADERS_DIR to point at the \
                     directory containing SPIRV-HeadersConfig.cmake",
                );
                config.define("SPIRV-Headers_DIR", spirv_dir);
                let spirv_include = env::var("SPIRV_HEADERS_INCLUDE_DIR").unwrap_or(vk_include);
                config.cflag(format!("-I{spirv_include}"));
                config.cxxflag(format!("-I{spirv_include}"));

                // Resolved from the device's libvulkan.so at load time.
                println!("cargo:rustc-link-lib=vulkan");
            }
            TargetOs::Apple(AppleVariant::MacOS) => {
                // CMake links the static ggml-vulkan objects, so the final Rust
                // binary also needs the Vulkan loader. Homebrew and the Vulkan
                // SDK expose it as libvulkan.dylib rather than a framework.
                let search_dir = env::var_os("CMAKE_Vulkan_LIBRARY")
                    .or_else(|| env::var_os("Vulkan_LIBRARY"))
                    .map(PathBuf::from)
                    .and_then(|path| path.parent().map(Path::to_path_buf))
                    .or_else(|| {
                        env::var_os("VULKAN_SDK").map(|sdk| PathBuf::from(sdk).join("lib"))
                    });
                if let Some(search_dir) = search_dir {
                    println!("cargo:rustc-link-search=native={}", search_dir.display());
                }
                println!("cargo:rustc-link-lib=dylib=vulkan");
            }
            _ => (),
        }
    }

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
        println!("cargo:rerun-if-env-changed=GGML_CUDA_NCCL");
        println!("cargo:rerun-if-env-changed=NCCL_ROOT");
        // Keep NCCL opt-in for CUDA builds because it adds a runtime dependency
        // on libnccl.so. Operators that have installed NCCL can enable the
        // collective path with GGML_CUDA_NCCL=ON; this must override the
        // previous unconditional OFF below the environment forwarding pass.
        config.define(
            "GGML_CUDA_NCCL",
            if cuda_option_enabled(env::var("GGML_CUDA_NCCL").ok().as_deref()) {
                "ON"
            } else {
                "OFF"
            },
        );
        // CUDA Graphs reserve extra driver state and can fail at graph launch
        // after the normal warmup already succeeded. That is fatal in the
        // fork's CUDA error path and is especially easy to hit when `--fit`
        // intentionally fills a GPU. Keep the safe path as the default; this
        // opt-in is for operators who have verified spare VRAM on their GPU.
        println!("cargo:rerun-if-env-changed=POM_CUDA_GRAPHS");
        let graphs = cuda_option_enabled(env::var("POM_CUDA_GRAPHS").ok().as_deref());
        config.define("GGML_CUDA_GRAPHS", if graphs { "ON" } else { "OFF" });

        if cfg!(feature = "cuda-no-vmm") {
            config.define("GGML_CUDA_NO_VMM", "ON");
        }
    }

    if cfg!(feature = "rocm") {
        config.define("GGML_HIP", "ON");
    }

    if cfg!(feature = "opencl") {
        // The Qualcomm-supported GPU backend for Adreno. EMBED_KERNELS and
        // USE_ADRENO_KERNELS are ON by default upstream, so no extra defines are
        // needed for those.
        config.define("GGML_OPENCL", "ON");

        // ggml-opencl/CMakeLists.txt runs `find_package(OpenCL REQUIRED)`. When
        // cross-compiling (e.g. Android, whose NDK ships no OpenCL SDK) CMake's
        // FindOpenCL can't locate one, so let the caller hand us the header dir
        // and the import library directly — FindOpenCL skips its own search when
        // these result variables are already set.
        println!("cargo:rerun-if-env-changed=OPENCL_INCLUDE_DIR");
        println!("cargo:rerun-if-env-changed=OPENCL_LIBRARY");
        if let Ok(include_dir) = env::var("OPENCL_INCLUDE_DIR") {
            config.define("OpenCL_INCLUDE_DIR", include_dir);
        }
        if let Ok(library) = env::var("OPENCL_LIBRARY") {
            config.define("OpenCL_LIBRARY", library);
        }

        // The backend embeds its kernels at build time with a Python helper
        // (`find_package(Python3 REQUIRED)`); allow pinning the interpreter so a
        // cross-build doesn't pick a broken stub `python3` (e.g. the Windows
        // Store alias). When unset, CMake's FindPython3 runs as usual.
        println!("cargo:rerun-if-env-changed=PYTHON3_EXECUTABLE");
        if let Ok(python3) = env::var("PYTHON3_EXECUTABLE") {
            config.define("Python3_EXECUTABLE", python3);
        }

        // The final `-lOpenCL` link is left to the top-level crate (mirroring how
        // the Android branch above leaves `-lvulkan` to it), keeping this fork
        // minimal: at runtime the device's own ICD provides the implementation.
    }

    if cfg!(feature = "mkl") {
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        assert_eq!(
            target_arch, "x86_64",
            "The `mkl` feature requires an x86_64 target; Intel MKL is unavailable for {target_arch}."
        );
        config.define("GGML_BLAS", "ON");
        config.define("GGML_BLAS_VENDOR", "Intel10_64lp");
    }

    // Android doesn't have OpenMP support AFAICT and openmp is a default feature. Do this here
    // rather than modifying the defaults in Cargo.toml just in case someone enables the OpenMP feature
    // and tries to build for Android anyway.
    if cfg!(feature = "openmp") && !matches!(target_os, TargetOs::Android) {
        config.define("GGML_OPENMP", "ON");
    } else {
        config.define("GGML_OPENMP", "OFF");
    }

    if cfg!(feature = "system-ggml") {
        config.define("LLAMA_USE_SYSTEM_GGML", "ON");
    }

    if cfg!(feature = "dynamic-backends") {
        // Pre-create the backends directory so CMake can install MODULE libs there.
        // GGML_BACKEND_DIR causes backends to install to this known path instead of
        // CMAKE_INSTALL_BINDIR, making them easy to locate in downstream build scripts.
        let backends_dir = out_dir.join("backends");
        std::fs::create_dir_all(&backends_dir).unwrap();
        config.define("GGML_BACKEND_DL", "ON");
        config.define("GGML_CPU_ALL_VARIANTS", "ON");
        config.define("GGML_BACKEND_DIR", backends_dir.to_str().unwrap());
        // BUILD_SHARED_LIBS=ON is already set above via the dynamic-link feature.
    }

    // General
    config
        .profile(&profile)
        .very_verbose(std::env::var("CMAKE_VERBOSE").is_ok()) // Not verbose by default
        // The llama.cpp fork collects model sources with CMake's GLOB. The
        // target directory is deliberately cacheable (including in Docker),
        // so reusing a CMakeCache without configuring can leave a newly
        // checked-out model source out of libllama.a.
        .always_configure(true);

    let build_dir = if let Some(prebuilt_dir) = prebuilt_lib_dir.as_deref() {
        let layout = if matches!(&target_os, TargetOs::Windows(_)) {
            PrebuiltLayout::WindowsMsvc
        } else {
            PrebuiltLayout::Linux
        };
        prepare_prebuilt_libraries(prebuilt_dir, &out_dir, layout)
            .unwrap_or_else(|error| panic!("invalid prebuilt llama.cpp libraries: {error}"));
        out_dir.clone()
    } else {
        config.build()
    };

    // The CMake targets are static archives. Their NCCL dependency is therefore
    // not propagated to the final Rust link, so add the imported library here
    // only when the operator explicitly enabled the NCCL path. Resolve the
    // search directory from the same CMake cache entry used by FindNCCL.
    if cfg!(feature = "cuda")
        && matches!(target_os, TargetOs::Linux)
        && cuda_option_enabled(env::var("GGML_CUDA_NCCL").ok().as_deref())
    {
        let cmake_cache = build_dir.join("build").join("CMakeCache.txt");
        if let Ok(cache_contents) = std::fs::read_to_string(&cmake_cache) {
            if let Some(library) = cache_contents
                .lines()
                .find_map(|line| line.strip_prefix("NCCL_LIBRARY:FILEPATH="))
                .filter(|path| !path.is_empty() && Path::new(path).is_file())
            {
                if let Some(parent) = Path::new(library).parent() {
                    println!("cargo:rustc-link-search=native={}", parent.display());
                }
            }
        }
    }

    if cfg!(feature = "dynamic-backends") {
        println!("cargo:backends_dir={}", out_dir.join("backends").display());
    }

    // The CMake build installs a ggml package config. Tell the dependent crates where it
    // is, in the DEP_LLAMA_GGML_CMAKE_DIR variable. A dependent crate that also builds
    // ggml can put this path in CMAKE_PREFIX_PATH and use this ggml. Then there is only
    // one ggml in the program. Two copies of ggml in one program cause duplicate symbols.
    for libdir in ["lib64", "lib"] {
        let cmake_dir = out_dir.join(libdir).join("cmake");
        if cmake_dir.join("ggml").is_dir() {
            println!("cargo:ggml_cmake_dir={}", cmake_dir.display());
            break;
        }
    }

    // Build mtmd directly with cc::Build, bypassing the cmake tools build.
    // Using LLAMA_BUILD_TOOLS=ON would pull in all tools (batched-bench, quantize, etc.)
    // and their CMakeLists.txt files, which are not included in the crate package.
    if cfg!(feature = "mtmd") {
        let mtmd_src = llama_src.join("tools/mtmd");
        let mut mtmd_build = cc::Build::new();
        mtmd_build
            .cpp(true)
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-Wno-cast-qual")
            .pic(true);

        for include_dir in [
            mtmd_src.clone(),
            llama_src.clone(),
            llama_src.join("include"),
            llama_src.join("ggml/include"),
            llama_src.join("common"),
            llama_src.join("vendor/hash"),
            llama_src.join("vendor"),
        ] {
            mtmd_build.include(build_tool_path(&include_dir, target_is_windows));
        }
        mtmd_build.file(build_tool_path(
            &llama_src.join("vendor/hash/hash.cpp"),
            target_is_windows,
        ));

        if matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc)) {
            mtmd_build.flag("/std:c++17");
        }

        // When static-stdcxx is enabled on Android, suppress the cc crate's automatic
        // C++ stdlib linking (which defaults to c++_shared) so we can link c++_static instead.
        if matches!(target_os, TargetOs::Android) && cfg!(feature = "static-stdcxx") {
            mtmd_build.cpp_link_stdlib(None);
        }

        // Collect all .cpp files in tools/mtmd and its subdirectories
        for entry in glob(mtmd_src.join("**/*.cpp").to_str().unwrap()).unwrap() {
            match entry {
                Ok(path) => {
                    // Skip the CLI / debug / deprecation-warning binaries (each has a
                    // `main`) — we only want the library sources
                    let filename = path.file_name().unwrap().to_str().unwrap();
                    if matches!(
                        filename,
                        "mtmd-cli.cpp" | "mtmd-debug.cpp" | "deprecation-warning.cpp"
                    ) {
                        continue;
                    }
                    mtmd_build.file(build_tool_path(&path, target_is_windows));
                }
                Err(e) => println!("cargo:warning=mtmd glob error: {}", e),
            }
        }

        mtmd_build.compile("mtmd");

        // hash.cpp is C++, but the bundled SHA-256 implementation must keep
        // its C symbol name for the wrapper's extern "C" declaration.
        cc::Build::new()
            .include(build_tool_path(
                &llama_src.join("vendor/hash"),
                target_is_windows,
            ))
            .file(build_tool_path(
                &llama_src.join("vendor/hash/sha256/sha256.c"),
                target_is_windows,
            ))
            .compile("mtmd_sha256");
    }

    // Search paths
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!(
        "cargo:rustc-link-search={}",
        out_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());

    if cfg!(feature = "system-ggml") {
        // Extract library directory from CMake's found GGML package
        let cmake_cache = build_dir.join("build").join("CMakeCache.txt");
        if let Ok(cache_contents) = std::fs::read_to_string(&cmake_cache) {
            let mut ggml_lib_dirs = std::collections::HashSet::new();

            // Parse CMakeCache.txt to find where GGML libraries were found
            for line in cache_contents.lines() {
                if line.starts_with("GGML_LIBRARY:")
                    || line.starts_with("GGML_BASE_LIBRARY:")
                    || line.starts_with("GGML_CPU_LIBRARY:")
                {
                    if let Some(lib_path) = line.split('=').nth(1) {
                        if let Some(parent) = Path::new(lib_path).parent() {
                            ggml_lib_dirs.insert(parent.to_path_buf());
                        }
                    }
                }
            }

            // Add each unique library directory to the search path
            for lib_dir in ggml_lib_dirs {
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
                debug_log!("Added system GGML library path: {}", lib_dir.display());
            }
        }
    }

    if cfg!(feature = "cuda") {
        // Re-run build script if CUDA_PATH environment variable changes
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        // Add CUDA library directories to the linker search path
        for lib_dir in find_cuda_helper::find_cuda_lib_dirs() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }

        // Platform-specific linking
        if cfg!(target_os = "windows") {
            // ✅ On Windows, use dynamic linking.
            // Static linking is problematic because NVIDIA does not provide culibos.lib,
            // and static CUDA libraries (like cublas_static.lib) are usually not shipped.

            println!("cargo:rustc-link-lib=cudart"); // Links to cudart64_*.dll
            println!("cargo:rustc-link-lib=cublas"); // Links to cublas64_*.dll
            println!("cargo:rustc-link-lib=cublasLt"); // Links to cublasLt64_*.dll

            // Link to CUDA driver API (nvcuda.dll via cuda.lib)
            if !cfg!(feature = "cuda-no-vmm") {
                println!("cargo:rustc-link-lib=cuda");
            }
        } else if build_shared_libs {
            // The backend is a runtime-loaded module, but POM's own CUDA kernels
            // are linked into the executable and need the CUDA runtime.
            println!("cargo:rustc-link-lib=dylib=cudart");
            if !cfg!(feature = "cuda-no-vmm") {
                println!("cargo:rustc-link-lib=dylib=cuda");
            }
        } else {
            // Static libraries like cudart_static and cublas_static depend on culibos.
            println!("cargo:rustc-link-lib=static=cudart_static");
            println!("cargo:rustc-link-lib=static=cublas_static");
            println!("cargo:rustc-link-lib=static=cublasLt_static");
            if !cfg!(feature = "cuda-no-vmm") {
                println!("cargo:rustc-link-lib=cuda");
            }
            println!("cargo:rustc-link-lib=static=culibos");
        }
    }

    if cfg!(feature = "rocm") && !build_shared_libs {
        // Re-run build script if ROCM_PATH environment variable changes
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-env-changed=HIP_PATH");

        // Find ROCm installation
        let rocm_path = env::var("ROCM_PATH")
            .or_else(|_| env::var("HIP_PATH"))
            .unwrap_or_else(|_| {
                if cfg!(target_os = "windows") {
                    "C:\\Program Files\\AMD\\ROCm".to_string()
                } else {
                    "/opt/rocm".to_string()
                }
            });

        let rocm_lib = Path::new(&rocm_path).join("lib");
        if !rocm_lib.exists() {
            panic!(
                "ROCm libraries not found at: {}\n\
                 Please install ROCm or set ROCM_PATH/HIP_PATH environment variable.\n\
                 Download from: https://rocm.docs.amd.com/",
                rocm_lib.display()
            );
        }

        println!("cargo:rustc-link-search=native={}", rocm_lib.display());

        // Link ROCm libraries
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-lib=dylib=hipblas");
    }

    if cfg!(feature = "mkl") && !build_shared_libs {
        println!("cargo:rerun-if-env-changed=MKLROOT");

        let mkl_root = env::var("MKLROOT")
            .expect("Intel MKL not found. Please install Intel oneAPI/MKL and set MKLROOT.");

        let mut found = false;
        for sub in ["lib/intel64", "lib"] {
            let dir = Path::new(&mkl_root).join(sub);
            if dir.is_dir() {
                println!("cargo:rustc-link-search=native={}", dir.display());
                found = true;
            }
        }
        assert!(
            found,
            "No MKL library directory found under MKLROOT={mkl_root}"
        );

        println!("cargo:rustc-link-lib=dylib=mkl_rt");
    }

    // Link libraries
    let llama_libs_kind = if build_shared_libs
        || (cfg!(feature = "system-ggml") && !cfg!(feature = "system-ggml-static"))
    {
        "dylib"
    } else {
        "static"
    };

    let llama_libs = extract_lib_names(&out_dir, build_shared_libs, &target_os);

    assert_ne!(llama_libs.len(), 0);

    // `ggml_cpu_built_with` (src/lib.rs) asks the CPU backend which instruction
    // sets it was compiled for. Those symbols only exist in this link when the
    // CPU backend is one of the linked libraries: a dynamic-backends build
    // loads it at run time, already picking a variant the processor supports.
    println!("cargo:rustc-check-cfg=cfg(ggml_cpu_linked)");
    if llama_libs.iter().any(|lib| lib == "ggml-cpu") {
        println!("cargo:rustc-cfg=ggml_cpu_linked");
    }

    let common_lib_dir = out_dir.join("build").join("common");
    if cfg!(feature = "common") && common_lib_dir.is_dir() {
        println!(
            "cargo:rustc-link-search=native={}",
            common_lib_dir.display()
        );
        let mut common_search_dirs = vec![common_lib_dir.clone()];
        let common_profile_dir = common_lib_dir.join(&profile);
        if common_profile_dir.is_dir() {
            println!(
                "cargo:rustc-link-search=native={}",
                common_profile_dir.display()
            );
            common_search_dirs.push(common_profile_dir);
        }

        if library_file_exists(
            &common_search_dirs,
            "llama-common",
            build_shared_libs,
            &target_os,
        ) {
            println!("cargo:rustc-link-lib={llama_libs_kind}=llama-common");
            if library_file_exists(
                &common_search_dirs,
                "llama-common-base",
                build_shared_libs,
                &target_os,
            ) {
                println!("cargo:rustc-link-lib={llama_libs_kind}=llama-common-base");
            }
        } else if library_file_exists(&common_search_dirs, "common", build_shared_libs, &target_os)
        {
            println!("cargo:rustc-link-lib={llama_libs_kind}=common");
        } else {
            println!(
                "cargo:warning=common feature was enabled, but no common library was found in {}",
                common_lib_dir.display()
            );
        }
    }

    if cfg!(feature = "system-ggml") {
        println!("cargo:rustc-link-lib={llama_libs_kind}=ggml");
        println!("cargo:rustc-link-lib={llama_libs_kind}=ggml-base");
        println!("cargo:rustc-link-lib={llama_libs_kind}=ggml-cpu");
    }
    for lib in llama_libs {
        let link = format!("cargo:rustc-link-lib={}={}", llama_libs_kind, lib);
        debug_log!("LINK {link}",);
        println!("{link}",);
    }

    if cfg!(feature = "cuda")
        && matches!(target_os, TargetOs::Linux)
        && cuda_option_enabled(env::var("GGML_CUDA_NCCL").ok().as_deref())
    {
        println!("cargo:rustc-link-lib=dylib=nccl");
    }

    // OpenMP
    if cfg!(feature = "openmp") && target_triple.contains("gnu") {
        if cfg!(feature = "static-openmp") {
            emit_compiler_static_archive_search_path("libgomp.a");
            println!("cargo:rustc-link-lib=static=gomp");
        } else {
            println!("cargo:rustc-link-lib=gomp");
        }
    }

    match target_os {
        TargetOs::Windows(WindowsVariant::Msvc) => {
            println!("cargo:rustc-link-lib=advapi32");
            let lib_is_debug = profile.eq_ignore_ascii_case("debug");
            if lib_is_debug {
                if static_crt {
                    println!("cargo:rustc-link-lib=libcmtd");
                } else {
                    println!("cargo:rustc-link-lib=dylib=msvcrtd");
                }
            }
        }
        TargetOs::Linux => {
            if cfg!(feature = "static-stdcxx") {
                emit_compiler_static_archive_search_path("libstdc++.a");
                println!("cargo:rustc-link-lib=static=stdc++");
            } else {
                println!("cargo:rustc-link-lib=dylib=stdc++");
            }
        }
        TargetOs::Apple(ref variant) => {
            println!("cargo:rustc-link-lib=framework=Foundation");
            // watchOS has no Metal; skip the Metal frameworks there.
            if !matches!(variant, AppleVariant::WatchOS) {
                println!("cargo:rustc-link-lib=framework=Metal");
                println!("cargo:rustc-link-lib=framework=MetalKit");
            }
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=c++");

            match variant {
                AppleVariant::MacOS => {
                    // On (older) OSX we need to link against the clang runtime,
                    // which is hidden in some non-default path.
                    //
                    // More details at https://github.com/alexcrichton/curl-rust/issues/279.
                    if let Some(path) = macos_link_search_path() {
                        println!("cargo:rustc-link-lib=clang_rt.osx");
                        println!("cargo:rustc-link-search={}", path);
                    }
                }
                AppleVariant::WatchOS | AppleVariant::Other => (),
            }
        }
        TargetOs::Android => {
            if cfg!(feature = "static-stdcxx") {
                println!("cargo:rustc-link-lib=c++_static");
                println!("cargo:rustc-link-lib=c++abi");
            } else if cfg!(feature = "shared-stdcxx") {
                println!("cargo:rustc-link-lib=c++_shared");
            }
            // When neither feature is set, the cc crate handles C++ stdlib
            // linking automatically (defaults to c++_shared on Android).
        }
        _ => (),
    }

    // copy DLLs to target
    if build_shared_libs {
        let libs_assets = extract_lib_assets(&out_dir, &target_os);
        for asset in libs_assets {
            let asset_clone = asset.clone();
            let filename = asset_clone.file_name().unwrap();
            let filename = filename.to_str().unwrap();
            let dst = target_dir.join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                std::fs::hard_link(asset.clone(), dst).unwrap();
            }

            // Copy DLLs to examples as well
            if target_dir.join("examples").exists() {
                let dst = target_dir.join("examples").join(filename);
                debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
                if !dst.exists() {
                    std::fs::hard_link(asset.clone(), dst).unwrap();
                }
            }

            // Copy DLLs to target/profile/deps as well for tests
            let dst = target_dir.join("deps").join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                std::fs::hard_link(asset.clone(), dst).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{cuda_option_enabled, missing_llama_source_error};

    #[test]
    fn cuda_options_are_opt_in() {
        assert!(!cuda_option_enabled(None));
        assert!(!cuda_option_enabled(Some("0")));
        assert!(!cuda_option_enabled(Some("off")));
        assert!(cuda_option_enabled(Some("1")));
        assert!(cuda_option_enabled(Some(" On ")));
    }

    #[test]
    fn missing_fork_error_explains_recovery() {
        let error = missing_llama_source_error(Path::new("vendor/llama-cpp-sys-2/llama.cpp"));

        assert!(error.contains("fork do llama.cpp não está disponível"));
        assert!(error.contains("git clone --depth 1 --branch pom"));
        assert!(error.contains("LLAMA_SOURCE=/caminho/para/llama.cpp"));
        assert!(error.contains("vendor/llama-cpp-sys-2/llama.cpp"));
    }
}
