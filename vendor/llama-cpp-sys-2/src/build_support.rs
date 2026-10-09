use std::path::{Component, Path, PathBuf};

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }
    normalized
}

pub(crate) fn clone_destination(
    manifest_dir: &Path,
    source: &Path,
    symlink_target: Option<&Path>,
    is_windows: bool,
) -> PathBuf {
    if is_windows {
        return source.to_path_buf();
    }

    let Some(target) = symlink_target else {
        return source.to_path_buf();
    };
    let base = source.parent().unwrap_or(manifest_dir);
    normalize_path(&base.join(target))
}

pub(crate) fn destination_can_be_replaced(
    destination: &Path,
    source: &Path,
    is_windows: bool,
) -> bool {
    is_windows && destination == source
}

pub(crate) fn source_can_be_used_directly(
    has_ggml: bool,
    is_symlink: bool,
    is_windows: bool,
) -> bool {
    has_ggml && !(is_windows && is_symlink)
}

pub(crate) fn source_root(source: &Path) -> PathBuf {
    source.to_path_buf()
}

pub(crate) fn build_tool_path(path: &Path, is_windows: bool) -> PathBuf {
    if !is_windows {
        return path.to_path_buf();
    }

    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = text.strip_prefix(r"\\?\") {
        return PathBuf::from(rest);
    }

    path.to_path_buf()
}

pub(crate) fn source_include_dirs(source: &Path) -> [PathBuf; 6] {
    [
        source.join("include"),
        source.join("ggml/include"),
        source.join("src"),
        source.join("common"),
        source.join("tools/mtmd"),
        source.join("vendor"),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        build_tool_path, clone_destination, destination_can_be_replaced,
        source_can_be_used_directly, source_include_dirs, source_root,
    };
    use std::path::Path;

    #[test]
    fn windows_installs_clone_at_source_instead_of_following_symlink() {
        let manifest_dir = Path::new(r"D:\a\rdma-p2p\rdma-p2p\vendor\llama-cpp-sys-2");
        let source = manifest_dir.join("llama.cpp");

        assert_eq!(
            clone_destination(
                manifest_dir,
                &source,
                Some(Path::new(r"..\..\llama.cpp")),
                true,
            ),
            source
        );
    }

    #[test]
    fn unix_installs_clone_into_relative_symlink_target() {
        let manifest_dir = Path::new("/worktree/vendor/llama-cpp-sys-2");
        let source = manifest_dir.join("llama.cpp");

        assert_eq!(
            clone_destination(
                manifest_dir,
                &source,
                Some(Path::new("../../llama.cpp")),
                false,
            ),
            Path::new("/worktree/llama.cpp")
        );
    }

    #[test]
    fn windows_allows_replacing_placeholder_at_source() {
        let manifest_dir = Path::new(r"D:\a\rdma-p2p\rdma-p2p\vendor\llama-cpp-sys-2");
        let source = manifest_dir.join("llama.cpp");

        assert!(destination_can_be_replaced(&source, &source, true));
        assert!(!destination_can_be_replaced(
            &manifest_dir.join(r"..\..\llama.cpp"),
            &source,
            true
        ));
        assert!(!destination_can_be_replaced(&source, &source, false));
    }

    #[test]
    fn windows_does_not_reuse_a_valid_symlink_as_the_build_root() {
        assert!(!source_can_be_used_directly(true, true, true));
        assert!(source_can_be_used_directly(true, false, true));
        assert!(source_can_be_used_directly(true, true, false));
    }

    #[test]
    fn external_source_is_include_root_for_wrapper_paths() {
        assert_eq!(
            source_root(Path::new("/worktree/.ci-cache/llama.cpp")),
            Path::new("/worktree/.ci-cache/llama.cpp")
        );
    }

    #[test]
    fn mtmd_wrapper_uses_an_explicit_include_directory() {
        let wrapper = include_str!("../wrapper_mtmd.h");

        assert!(!wrapper.contains("\"tools/mtmd/"));
        assert!(wrapper.contains("\"mtmd.h\""));
        assert!(wrapper.contains("\"mtmd-helper.h\""));
    }

    #[test]
    fn wrapper_include_directories_cover_all_source_subdirectories() {
        let source = Path::new("/worktree/.ci-cache/llama.cpp");
        let include_dirs = source_include_dirs(source);

        for relative in [
            "include",
            "ggml/include",
            "src",
            "common",
            "tools/mtmd",
            "vendor",
        ] {
            assert!(include_dirs.contains(&source.join(relative)));
        }
    }

    #[test]
    fn msvc_path_drops_extended_drive_prefix() {
        assert_eq!(
            build_tool_path(Path::new(r"\\?\D:\a\rdma-p2p\include"), true),
            Path::new(r"D:\a\rdma-p2p\include"),
        );
    }

    #[test]
    fn msvc_path_converts_extended_unc_prefix() {
        assert_eq!(
            build_tool_path(Path::new(r"\\?\UNC\build\share\include"), true),
            Path::new(r"\\build\share\include"),
        );
    }

    #[test]
    fn non_windows_path_is_unchanged() {
        let path = Path::new("/worktree/.ci-cache/llama.cpp/include");

        assert_eq!(build_tool_path(path, false), path);
    }
}
