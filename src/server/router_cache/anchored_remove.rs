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

use super::anchored_delete::delete_contents;
use super::anchored_publish::{
    child_exists, cstring_segment, ensure_child_dir, open_child_dir, open_dir_path,
    regular_file_nonempty_at, rename_no_replace, repo_segments, verify_child_identity,
    verify_private_staging_parent, verify_same_file_identity,
};
use anyhow::Context;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const QUARANTINE_DIR: &str = ".mlxcel-quarantine";
static REMOVE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ManagedRemoval {
    pub(super) path: PathBuf,
    pub(super) size_bytes: u64,
}

pub(super) fn remove_managed_snapshot(
    root_path: &Path,
    repo_id: &str,
) -> anyhow::Result<Option<ManagedRemoval>> {
    remove_managed_snapshot_impl(root_path, repo_id, RemovalHooks::default())
}

#[derive(Default)]
struct RemovalHooks {
    pre_rename: Option<Box<dyn FnOnce() + Send>>,
    post_rename: Option<Box<dyn FnOnce() + Send>>,
    before_final_unlink: Option<Box<dyn FnOnce() + Send>>,
}

#[cfg(test)]
pub(super) fn remove_managed_snapshot_with_pre_rename_hook<F>(
    root_path: &Path,
    repo_id: &str,
    hook: F,
) -> anyhow::Result<Option<ManagedRemoval>>
where
    F: FnOnce() + Send + 'static,
{
    remove_managed_snapshot_impl(
        root_path,
        repo_id,
        RemovalHooks {
            pre_rename: Some(Box::new(hook)),
            ..RemovalHooks::default()
        },
    )
}

#[cfg(test)]
pub(super) fn remove_managed_snapshot_with_post_rename_hook<F>(
    root_path: &Path,
    repo_id: &str,
    hook: F,
) -> anyhow::Result<Option<ManagedRemoval>>
where
    F: FnOnce() + Send + 'static,
{
    remove_managed_snapshot_impl(
        root_path,
        repo_id,
        RemovalHooks {
            post_rename: Some(Box::new(hook)),
            ..RemovalHooks::default()
        },
    )
}

#[cfg(test)]
pub(super) fn remove_managed_snapshot_with_final_unlink_hook<F>(
    root_path: &Path,
    repo_id: &str,
    hook: F,
) -> anyhow::Result<Option<ManagedRemoval>>
where
    F: FnOnce() + Send + 'static,
{
    remove_managed_snapshot_impl(
        root_path,
        repo_id,
        RemovalHooks {
            before_final_unlink: Some(Box::new(hook)),
            ..RemovalHooks::default()
        },
    )
}

fn remove_managed_snapshot_impl(
    root_path: &Path,
    repo_id: &str,
    mut hooks: RemovalHooks,
) -> anyhow::Result<Option<ManagedRemoval>> {
    let (owner, final_name) = repo_segments(repo_id)?;
    let root = match open_dir_path(root_path) {
        Ok(root) => root,
        Err(err) if err.raw_os_error() == Some(libc::ENOENT) => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "model-store root {} must be a real directory, not a symlink",
                    root_path.display()
                )
            });
        }
    };
    let final_parent = match open_child_dir(&root, &owner) {
        Ok(owner) => owner,
        Err(err) if err.raw_os_error() == Some(libc::ENOENT) => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| {
                format!("cache owner directory for repo '{repo_id}' must be a real directory")
            });
        }
    };
    let target = match open_child_dir(&final_parent, &final_name) {
        Ok(target) => target,
        Err(err) if err.raw_os_error() == Some(libc::ENOENT) => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| {
                format!("cache snapshot for repo '{repo_id}' must be a real directory")
            });
        }
    };
    if !regular_file_nonempty_at(&target, c"config.json")? {
        anyhow::bail!(
            "refusing to remove cache snapshot for '{repo_id}' because it is missing a non-empty regular config.json"
        );
    }

    let quarantine_name = cstring_segment(QUARANTINE_DIR)?;
    ensure_child_dir(&root, &quarantine_name, 0o700)
        .context("failed to create router removal quarantine directory")?;
    let quarantine_parent = open_child_dir(&root, &quarantine_name)
        .context("router removal quarantine path must be a real directory")?;
    verify_private_staging_parent(&quarantine_parent)
        .context("router removal quarantine directory must be private to the server user")?;
    verify_child_identity(&root, &quarantine_name, &quarantine_parent)
        .context("router removal quarantine directory identity changed before removal")?;
    let quarantine_child = unique_available_child_name(&quarantine_parent, repo_id)?;

    verify_child_identity(&root, &owner, &final_parent)
        .context("cache owner directory identity changed before removal")?;
    verify_child_identity(&final_parent, &final_name, &target)
        .context("cache snapshot directory identity changed before removal")?;
    if let Some(hook) = hooks.pre_rename.take() {
        hook();
    }
    verify_child_identity(&root, &owner, &final_parent)
        .context("cache owner directory identity changed before removal")?;
    rename_no_replace(
        &final_parent,
        &final_name,
        &quarantine_parent,
        &quarantine_child,
    )
    .context("failed to move cache snapshot into private removal quarantine")?;
    if let Some(hook) = hooks.post_rename.take() {
        hook();
    }
    verify_child_identity(&root, &owner, &final_parent).context(
        "cache owner directory identity changed after removal quarantine; leaving quarantined snapshot for inspection",
    )?;
    let moved = open_child_dir(&quarantine_parent, &quarantine_child)
        .context("quarantined cache snapshot vanished before identity check")?;
    verify_same_file_identity(&target, &moved).context(
        "quarantined cache snapshot identity changed; leaving quarantine for inspection",
    )?;

    // Deletion walks the held directory fd and unlinks children relative to that
    // fd, so symlinks inside a managed snapshot are unlinked rather than
    // followed. The final directory unlink still names the private quarantine
    // child, so a same-UID process with write access to the 0700 quarantine can
    // race that last name lookup; we re-check the child identity immediately
    // before unlink and leave any mismatch undeleted. Stronger protection would
    // require a platform API that removes a directory by fd.
    let size_bytes = delete_contents(&target)?;
    verify_child_identity(&quarantine_parent, &quarantine_child, &target)
        .context("quarantined cache snapshot identity changed before final unlink")?;
    if let Some(hook) = hooks.before_final_unlink.take() {
        hook();
    }
    verify_child_identity(&quarantine_parent, &quarantine_child, &target)
        .context("quarantined cache snapshot identity changed before final unlink")?;
    let rc = unsafe {
        libc::unlinkat(
            quarantine_parent.as_raw_fd(),
            quarantine_child.as_ptr(),
            libc::AT_REMOVEDIR,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error())
            .context("failed to remove quarantined cache directory");
    }

    Ok(Some(ManagedRemoval {
        path: managed_snapshot_path(root_path, &owner, &final_name),
        size_bytes,
    }))
}

fn unique_available_child_name(parent: &File, repo_id: &str) -> anyhow::Result<CString> {
    for _ in 0..64 {
        let seq = REMOVE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = safe_remove_name(repo_id, seq);
        if !child_exists(parent, &name)? {
            return Ok(name);
        }
    }
    anyhow::bail!("failed to allocate a unique private quarantine name")
}

fn safe_remove_name(repo_id: &str, seq: u64) -> CString {
    let safe = repo_id
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-') {
                b as char
            } else {
                '_'
            }
        })
        .collect::<String>();
    CString::new(format!(
        "{}-{}-{}-{safe}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        seq
    ))
    .expect("generated quarantine name has no NUL")
}

fn managed_snapshot_path(root: &Path, owner: &CStr, final_name: &CStr) -> PathBuf {
    root.join(std::ffi::OsStr::from_bytes(owner.to_bytes()))
        .join(std::ffi::OsStr::from_bytes(final_name.to_bytes()))
}

#[cfg(test)]
#[path = "anchored_remove_tests.rs"]
mod tests;
