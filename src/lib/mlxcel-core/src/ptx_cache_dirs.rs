// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Mode repair for the PTX cache directories mlxcel manages (issue #2539).
//!
//! The `jit_module.cpp` overlay refuses a cache directory with a group- or
//! other-writable component. Under a umask of 002 (user private groups, the
//! Ubuntu default for regular users) earlier mlxcel releases created
//! `~/.cache/mlxcel/cuda-ptx/<mlx-commit>` as 0775, so an upgrade would turn
//! the persistent cache off on exactly the hosts it was built for.
//! [`ensure_persistent_ptx_cache`](crate::ensure_persistent_ptx_cache) repairs
//! the directories it owns: `cuda-ptx`, `cuda-ptx/<scope>`, and the cache root
//! only when it is the default `~/.cache/mlxcel` (an explicit
//! `MLXCEL_CACHE_DIR` is the user's choice). Nothing above the root is touched,
//! so a group-writable `~/.cache` is still refused.

use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;

/// Clear the group- and other-write bits of `path`, keeping every other bit,
/// when it is a real directory (not a symlink) owned by `euid`. Errors are
/// ignored: a directory left as it was is refused by the overlay with a
/// notice that names it.
pub(crate) fn make_private_dir(path: &Path, euid: u32) {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return;
    };
    if !meta.file_type().is_dir() || meta.uid() != euid {
        return;
    }
    let mode = meta.permissions().mode() & 0o7777;
    if mode & 0o022 == 0 {
        return;
    }
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode & !0o022));
}

/// The effective uid of this process, read from the owner of a probe file
/// created in `dir` (mlxcel-core has no libc dependency). `None` when `dir`
/// is not writable.
fn current_euid(dir: &Path) -> Option<u32> {
    let probe = dir.join(format!(".mlxcel-uid-probe-{}", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .ok()?;
    let uid = file.metadata().ok().map(|m| m.uid());
    drop(file);
    let _ = std::fs::remove_file(&probe);
    uid
}

/// Repair the modes of the PTX cache directories under `root`: `scope_dir`
/// (`root/cuda-ptx/<scope>`), its parent `root/cuda-ptx`, and `root` itself
/// only when `repair_root` is set (the default cache root).
pub(crate) fn tighten_persistent_ptx_dirs(root: &Path, scope_dir: &Path, repair_root: bool) {
    let Some(euid) = current_euid(scope_dir) else {
        return;
    };
    make_private_dir(scope_dir, euid);
    if let Some(cuda_ptx) = scope_dir.parent() {
        make_private_dir(cuda_ptx, euid);
    }
    if repair_root {
        make_private_dir(root, euid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode(path: &Path) -> u32 {
        std::fs::symlink_metadata(path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o7777
    }

    fn dir_with_mode(parent: &Path, name: &str, mode: u32) -> std::path::PathBuf {
        let dir = parent.join(name);
        std::fs::create_dir(&dir).expect("create dir");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).expect("chmod");
        dir
    }

    fn euid(tmp: &Path) -> u32 {
        current_euid(tmp).expect("temp dir is writable")
    }

    #[test]
    fn clears_only_group_and_other_write() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let euid = euid(tmp.path());
        let umask_002 = dir_with_mode(tmp.path(), "umask002", 0o775);
        let world = dir_with_mode(tmp.path(), "world", 0o2777);
        let private = dir_with_mode(tmp.path(), "private", 0o700);
        for dir in [&umask_002, &world, &private] {
            make_private_dir(dir, euid);
        }
        assert_eq!(mode(&umask_002), 0o755);
        assert_eq!(
            mode(&world),
            0o2755,
            "setgid and read/execute bits are kept"
        );
        assert_eq!(mode(&private), 0o700);
    }

    #[test]
    fn leaves_foreign_symlinked_and_non_directories_alone() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let euid = euid(tmp.path());
        let foreign = dir_with_mode(tmp.path(), "foreign", 0o775);
        make_private_dir(&foreign, euid.wrapping_add(1));
        assert_eq!(mode(&foreign), 0o775, "a directory owned by another uid");

        let target = dir_with_mode(tmp.path(), "target", 0o775);
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        make_private_dir(&link, euid);
        assert_eq!(mode(&target), 0o775, "a symlink is not followed");

        let file = tmp.path().join("file");
        std::fs::write(&file, b"").expect("file");
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o664)).expect("chmod");
        make_private_dir(&file, euid);
        assert_eq!(mode(&file), 0o664, "only directories are repaired");
    }

    #[test]
    fn tightens_the_managed_chain_and_the_root_only_when_asked() {
        for repair_root in [true, false] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let above = dir_with_mode(tmp.path(), "home-cache", 0o775);
            let root = dir_with_mode(&above, "mlxcel", 0o775);
            let cuda_ptx = dir_with_mode(&root, "cuda-ptx", 0o775);
            let scope = dir_with_mode(&cuda_ptx, "0123456789ab", 0o775);
            tighten_persistent_ptx_dirs(&root, &scope, repair_root);
            assert_eq!(mode(&scope), 0o755);
            assert_eq!(mode(&cuda_ptx), 0o755);
            assert_eq!(mode(&root), if repair_root { 0o755 } else { 0o775 });
            assert_eq!(mode(&above), 0o775, "nothing above the root is touched");
            let leftovers: Vec<_> = std::fs::read_dir(&scope)
                .expect("scope dir")
                .filter_map(|e| e.ok())
                .collect();
            assert!(leftovers.is_empty(), "the uid probe must be removed");
        }
    }
}
