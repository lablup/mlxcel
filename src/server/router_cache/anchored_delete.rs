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

use super::anchored_publish::open_child_dir;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io;
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;

pub(super) fn delete_tree_at(parent: &File, name: &CStr) -> io::Result<u64> {
    let expected = stat_child(parent, name)?;
    let mut hook = None;
    delete_tree_at_verified(parent, name, &expected, &mut hook)
}

#[cfg(test)]
pub(super) enum DeleteHookEvent {
    BeforeOpenDir,
    BeforeRemoveDir,
}

#[cfg(test)]
pub(super) fn delete_contents_with_hook<F>(dir: &File, mut hook: F) -> io::Result<u64>
where
    F: FnMut(DeleteHookEvent, &CStr),
{
    let mut hook: Option<&mut dyn FnMut(DeleteHookEvent, &CStr)> = Some(&mut hook);
    delete_contents_internal(dir, &mut hook)
}

pub(super) fn delete_contents(dir: &File) -> io::Result<u64> {
    let mut hook = None;
    delete_contents_internal(dir, &mut hook)
}

#[cfg(not(test))]
type DeleteHook<'a> = Option<&'a mut dyn FnMut((), &CStr)>;
#[cfg(test)]
type DeleteHook<'a> = Option<&'a mut dyn FnMut(DeleteHookEvent, &CStr)>;

fn delete_tree_at_verified(
    parent: &File,
    name: &CStr,
    expected: &libc::stat,
    hook: &mut DeleteHook<'_>,
) -> io::Result<u64> {
    #[cfg(test)]
    if let Some(hook) = hook.as_deref_mut() {
        hook(DeleteHookEvent::BeforeOpenDir, name);
    }
    let child = open_child_dir(parent, name)?;
    verify_file_matches_stat(&child, expected)?;
    let size_bytes = delete_contents_internal(&child, hook)?;
    #[cfg(test)]
    if let Some(hook) = hook.as_deref_mut() {
        hook(DeleteHookEvent::BeforeRemoveDir, name);
    }
    verify_child_matches_file(parent, name, &child)?;
    let rc = unsafe { libc::unlinkat(parent.as_raw_fd(), name.as_ptr(), libc::AT_REMOVEDIR) };
    if rc == 0 {
        Ok(size_bytes)
    } else {
        Err(io::Error::last_os_error())
    }
}

fn delete_contents_internal(dir: &File, hook: &mut DeleteHook<'_>) -> io::Result<u64> {
    let dup_fd = unsafe { libc::dup(dir.as_raw_fd()) };
    if dup_fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let dir_ptr = unsafe { libc::fdopendir(dup_fd) };
    if dir_ptr.is_null() {
        let err = io::Error::last_os_error();
        unsafe {
            libc::close(dup_fd);
        }
        return Err(err);
    }
    let _guard = DirCloser(dir_ptr);
    let mut total = 0u64;
    loop {
        errno_reset();
        let entry = unsafe { libc::readdir(dir_ptr) };
        if entry.is_null() {
            let err = io::Error::last_os_error();
            if err.raw_os_error().unwrap_or(0) == 0 {
                return Ok(total);
            }
            return Err(err);
        }
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
        if name.to_bytes() == b"." || name.to_bytes() == b".." {
            continue;
        }
        let name = CString::new(name.to_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "NUL in dir entry"))?;
        let stat = match stat_child(dir, &name) {
            Ok(stat) => stat,
            Err(err) if err.raw_os_error() == Some(libc::ENOENT) => continue,
            Err(err) => return Err(err),
        };
        if is_dir_mode(stat.st_mode) {
            total = total.saturating_add(delete_tree_at_verified(dir, &name, &stat, hook)?);
        } else {
            if is_regular_mode(stat.st_mode) && stat.st_size > 0 {
                total = total.saturating_add(stat.st_size as u64);
            }
            let rc = unsafe { libc::unlinkat(dir.as_raw_fd(), name.as_ptr(), 0) };
            if rc != 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::ENOENT) {
                    return Err(err);
                }
            }
        }
    }
}

fn stat_child(parent: &File, name: &CStr) -> io::Result<libc::stat> {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    let rc = unsafe {
        libc::fstatat(
            parent.as_raw_fd(),
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if rc == 0 {
        Ok(unsafe { stat.assume_init() })
    } else {
        Err(io::Error::last_os_error())
    }
}

fn verify_file_matches_stat(file: &File, expected: &libc::stat) -> io::Result<()> {
    let actual = file.metadata()?;
    if actual.dev() == expected.st_dev as u64 && actual.ino() == expected.st_ino {
        Ok(())
    } else {
        Err(identity_mismatch())
    }
}

fn verify_child_matches_file(parent: &File, name: &CStr, held: &File) -> io::Result<()> {
    let current = stat_child(parent, name)?;
    let held = held.metadata()?;
    if held.dev() == current.st_dev as u64 && held.ino() == current.st_ino {
        Ok(())
    } else {
        Err(identity_mismatch())
    }
}

fn identity_mismatch() -> io::Error {
    io::Error::other("directory identity mismatch")
}

fn errno_reset() {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    unsafe {
        *libc::__error() = 0;
    }
    #[cfg(target_os = "linux")]
    unsafe {
        *libc::__errno_location() = 0;
    }
}

fn is_dir_mode(mode: libc::mode_t) -> bool {
    (mode & libc::S_IFMT) == libc::S_IFDIR
}

fn is_regular_mode(mode: libc::mode_t) -> bool {
    (mode & libc::S_IFMT) == libc::S_IFREG
}

struct DirCloser(*mut libc::DIR);

impl Drop for DirCloser {
    fn drop(&mut self) {
        unsafe {
            libc::closedir(self.0);
        }
    }
}

#[cfg(test)]
#[path = "anchored_delete_tests.rs"]
mod tests;
