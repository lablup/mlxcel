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

use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;

/// Clear the group- and other-write bits of `path`, keeping every other bit,
/// when it is a real directory (not a symlink) owned by `euid`. Errors are
/// ignored: a directory left as it was is refused by the overlay with a
/// notice that names it.
pub(crate) fn make_private_dir(path: &Path, euid: u32) {
    repair_dir(path, euid, || {});
}

/// [`make_private_dir`], with `after_check` run between the check and the
/// mode change so a test can swap the path in that window.
///
/// The mode is changed through a descriptor (`fchmod`), and only when the
/// opened object is a directory, reached without following a symlink, that
/// is the one the `lstat` approved (same device and inode). A chmod by path
/// follows a symlink swapped in by anyone who can write the parent, and would
/// apply this directory's mode (typically 0755) to whatever the link points
/// at, such as a private file of this user.
fn repair_dir(path: &Path, euid: u32, after_check: impl FnOnce()) {
    let Ok(checked) = std::fs::symlink_metadata(path) else {
        return;
    };
    if !checked.file_type().is_dir() || checked.uid() != euid || checked.mode() & 0o022 == 0 {
        return;
    }
    after_check();
    // O_DIRECTORY and O_NOFOLLOW make the open fail on anything but a real
    // directory, and O_NONBLOCK keeps it from waiting on a FIFO swapped in
    // since the check: a plain open blocks there until a writer appears, and
    // this runs at startup (#2765).
    let Ok(dir) = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
    else {
        return;
    };
    let Ok(opened) = dir.metadata() else {
        return;
    };
    if opened.dev() != checked.dev() || opened.ino() != checked.ino() || opened.uid() != euid {
        return;
    }
    let mode = opened.mode() & 0o7777;
    let _ = dir.set_permissions(std::fs::Permissions::from_mode(mode & !0o022));
}

/// Repair the modes of the PTX cache directories under `root`: `scope_dir`
/// (`root/cuda-ptx/<scope>`), its parent `root/cuda-ptx`, and `root` itself
/// only when `repair_root` is set (the default cache root).
pub(crate) fn tighten_persistent_ptx_dirs(root: &Path, scope_dir: &Path, repair_root: bool) {
    // SAFETY: geteuid has no preconditions and cannot fail. It replaces a
    // probe file created in the scope dir, which failed on a scope dir the
    // owner cannot write and could be left behind by a crash (#2765).
    let euid = unsafe { libc::geteuid() };
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

    #[test]
    fn clears_only_group_and_other_write() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let euid = process_euid();
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
        let euid = process_euid();
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
    fn a_path_swapped_after_the_check_is_left_alone() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let euid = process_euid();
        let secret = tmp.path().join("secret");
        std::fs::write(&secret, b"private").expect("file");
        std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        let other = dir_with_mode(tmp.path(), "other", 0o775);
        for target in [&secret, &other] {
            let managed = dir_with_mode(tmp.path(), "managed", 0o775);
            let moved = tmp.path().join("moved");
            let before = mode(target);
            repair_dir(&managed, euid, || {
                std::fs::rename(&managed, &moved).expect("move the checked dir away");
                std::os::unix::fs::symlink(target, &managed).expect("swap in a symlink");
            });
            assert_eq!(
                mode(target),
                before,
                "{} must keep its mode when a symlink replaces the checked dir",
                target.display()
            );
            assert_eq!(
                mode(&moved),
                0o775,
                "the moved dir is no longer at the path"
            );
            std::fs::remove_file(&managed).expect("remove symlink");
            std::fs::remove_dir(&moved).expect("remove moved dir");
        }
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
            assert!(
                leftovers.is_empty(),
                "nothing may be created in the scope dir"
            );
        }
    }

    /// The effective uid, read without touching the filesystem.
    fn process_euid() -> u32 {
        // SAFETY: geteuid has no preconditions and cannot fail.
        unsafe { libc::geteuid() }
    }

    fn make_fifo(path: &Path) {
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).expect("no NUL in path");
        // SAFETY: c_path is a NUL-terminated string that outlives the call.
        let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(
            rc,
            0,
            "mkfifo {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
    }

    /// Open `fifo` for writing without blocking, which completes an open for
    /// reading that is blocked on it. Retries for a bounded time while no
    /// reader has arrived yet (ENXIO).
    fn release_fifo_reader(fifo: &Path) {
        for _ in 0..100 {
            if std::fs::OpenOptions::new()
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(fifo)
                .is_ok()
            {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    /// Every path under `dir`, recursively, without following symlinks.
    fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
        let mut found = Vec::new();
        let mut pending = vec![dir.to_path_buf()];
        while let Some(next) = pending.pop() {
            for entry in std::fs::read_dir(&next).expect("read_dir").flatten() {
                let path = entry.path();
                if entry.file_type().is_ok_and(|t| t.is_dir()) {
                    pending.push(path.clone());
                }
                found.push(path);
            }
        }
        found
    }

    /// The repair needs the effective uid, not a writable scope directory:
    /// a scope directory without owner write is still tightened, and nothing
    /// is created to learn the uid (#2765).
    #[test]
    fn repair_runs_on_a_scope_dir_the_owner_cannot_write() {
        if process_euid() == 0 {
            eprintln!("skipping: root writes any directory, so the case cannot be set up");
            return;
        }
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = dir_with_mode(tmp.path(), "mlxcel", 0o755);
        let cuda_ptx = dir_with_mode(&root, "cuda-ptx", 0o755);
        let scope = dir_with_mode(&cuda_ptx, "0123456789ab", 0o575);
        tighten_persistent_ptx_dirs(&root, &scope, true);
        assert_eq!(
            mode(&scope),
            0o555,
            "group and other write must be cleared on a scope dir the owner cannot write"
        );
        let probes: Vec<_> = walk(tmp.path())
            .into_iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(".mlxcel-uid-probe-"))
            })
            .collect();
        assert!(probes.is_empty(), "no uid probe may be created: {probes:?}");
    }

    /// A directory swapped for a FIFO between the check and the open must not
    /// block the repair (and with it mlxcel's startup) until a writer appears
    /// (#2765).
    #[test]
    fn a_directory_swapped_for_a_fifo_does_not_block() {
        use std::os::unix::fs::FileTypeExt;
        use std::sync::mpsc::{RecvTimeoutError, channel};

        let tmp = tempfile::tempdir().expect("tempdir");
        let euid = process_euid();
        let managed = dir_with_mode(tmp.path(), "managed", 0o775);
        let moved = tmp.path().join("moved");
        let (done_tx, done_rx) = channel();
        let worker = {
            let (managed, moved) = (managed.clone(), moved.clone());
            std::thread::spawn(move || {
                repair_dir(&managed, euid, || {
                    std::fs::rename(&managed, &moved).expect("move the checked dir away");
                    make_fifo(&managed);
                });
                let _ = done_tx.send(());
            })
        };
        let outcome = done_rx.recv_timeout(std::time::Duration::from_secs(2));
        if outcome == Err(RecvTimeoutError::Timeout) {
            // Unblock the helper thread so a failing run ends instead of hanging.
            release_fifo_reader(&managed);
        }
        // Bounded join: a worker that stays blocked fails the test instead of
        // hanging it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !worker.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "the repair thread is still blocked 5 s after the FIFO was opened"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        worker.join().expect("the repair thread must not panic");
        assert!(
            outcome.is_ok(),
            "repair_dir must return within 2 s when the checked dir became a FIFO"
        );
        let fifo = std::fs::symlink_metadata(&managed).expect("the FIFO is still there");
        assert!(fifo.file_type().is_fifo(), "the FIFO must be left in place");
        assert_eq!(fifo.mode() & 0o7777, 0o600, "the FIFO's mode is untouched");
        assert_eq!(
            mode(&moved),
            0o775,
            "the moved dir is no longer at the path"
        );
    }
}
