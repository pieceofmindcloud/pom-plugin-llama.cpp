use std::path::Path;

/// Where a prebuilt package's files go, per target.
pub enum PrebuiltLayout {
    /// `lib*.so`, staged in `lib/` for the link and the runtime.
    Linux,
    /// MSVC: the `.lib` import libraries are staged in `lib/` for the link,
    /// and the `.dll`s in `bin/`, which the build copies next to the binary.
    WindowsMsvc,
}

const LINUX_LIBS: [&str; 4] = [
    "libllama.so",
    "libggml.so",
    "libggml-base.so",
    "libllama-common.so",
];
const WINDOWS_STEMS: [&str; 4] = ["llama", "ggml", "ggml-base", "llama-common"];

fn stage(source: &Path, target_dir: &Path, name: &str) -> Result<(), String> {
    let file = source.join(name);
    if !file.is_file() {
        return Err(format!("required prebuilt library is missing: {}", file.display()));
    }
    println!("cargo:rerun-if-changed={}", file.display());
    std::fs::create_dir_all(target_dir).map_err(|error| error.to_string())?;
    std::fs::copy(&file, target_dir.join(name))
        .map_err(|error| format!("could not stage prebuilt library {}: {error}", file.display()))?;
    Ok(())
}

pub fn prepare_prebuilt_libraries(
    source_dir: &Path,
    out_dir: &Path,
    layout: PrebuiltLayout,
) -> Result<(), String> {
    match layout {
        PrebuiltLayout::Linux => {
            for name in LINUX_LIBS {
                stage(source_dir, &out_dir.join("lib"), name)?;
            }
        }
        PrebuiltLayout::WindowsMsvc => {
            for stem in WINDOWS_STEMS {
                stage(source_dir, &out_dir.join("lib"), &format!("{stem}.lib"))?;
                stage(source_dir, &out_dir.join("bin"), &format!("{stem}.dll"))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{prepare_prebuilt_libraries, PrebuiltLayout};
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "pom-llama-prebuilt-{name}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn stages_the_required_shared_libraries() {
        let root = temp_dir("complete");
        let source = root.join("source");
        let out = root.join("out");
        fs::create_dir_all(&source).unwrap();
        for name in [
            "libllama.so",
            "libggml.so",
            "libggml-base.so",
            "libllama-common.so",
        ] {
            fs::write(source.join(name), name).unwrap();
        }

        prepare_prebuilt_libraries(&source, &out, PrebuiltLayout::Linux).unwrap();
        for name in [
            "libllama.so",
            "libggml.so",
            "libggml-base.so",
            "libllama-common.so",
        ] {
            assert_eq!(fs::read_to_string(out.join("lib").join(name)).unwrap(), name);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_a_package_missing_a_required_library() {
        let root = temp_dir("missing");
        fs::create_dir_all(&root).unwrap();
        let error = prepare_prebuilt_libraries(&root, &root.join("out"), PrebuiltLayout::Linux).unwrap_err();
        assert!(error.contains("libllama.so"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stages_import_libraries_and_dlls_on_windows() {
        let root = temp_dir("windows");
        let source = root.join("source");
        let out = root.join("out");
        fs::create_dir_all(&source).unwrap();
        for stem in ["llama", "ggml", "ggml-base", "llama-common"] {
            fs::write(source.join(format!("{stem}.lib")), stem).unwrap();
            fs::write(source.join(format!("{stem}.dll")), stem).unwrap();
        }

        prepare_prebuilt_libraries(&source, &out, PrebuiltLayout::WindowsMsvc).unwrap();
        assert!(out.join("lib").join("llama.lib").is_file());
        assert!(out.join("bin").join("llama-common.dll").is_file());

        fs::remove_file(source.join("ggml.lib")).unwrap();
        let error = prepare_prebuilt_libraries(&source, &root.join("out2"), PrebuiltLayout::WindowsMsvc)
            .unwrap_err();
        assert!(error.contains("ggml.lib"));
        fs::remove_dir_all(root).unwrap();
    }
}
