// Copyright 2026 Lablup Inc.
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

//! Trust of the CUDA JIT PTX disk cache on Windows (issue #2797).
//!
//! The `jit_module.cpp` overlay loads compiled modules from the cache, so a
//! directory another account can write is a code-planting path. Every scenario
//! sets real ACLs with `icacls`, plants entries a first run compiled (so a run
//! that trusts the directory would load them without compiling), and re-runs
//! this binary as a child, since the overlay reads its environment once per
//! process. The child launches the fused MoE kernel pair, after calling
//! `ensure_persistent_ptx_cache` as `mlxcel` and `mlxcel-server` do when the
//! scenario asks for it. A hit is a run that prints no JIT notice.
//!
//! ```sh
//! cargo test --release --features cuda --test cuda_ptx_cache_trust_windows -- --test-threads=1
//! ```

#![cfg(all(feature = "cuda", windows))]

mod common;

use std::ffi::{OsStr, OsString};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::process::Command;

use common::cuda_kernels::{
    JIT_NOTICE, device_dir, emit, files_with_extension, launch_fused_moe_pair, parse_out,
    private_tempdir,
};

const CHILD_TEST: &str = "ptx_cache_trust_windows_child";
const CHILD_ENV: &str = "MLXCEL_TEST_PTX_TRUST_CHILD";
const PERSISTENT_ENV: &str = "MLXCEL_TEST_PTX_TRUST_PERSISTENT";
const NO_CUDA_TAG: &str = "MLXCEL_PTX_TRUST_NO_CUDA";
const REFUSAL: &str = "mlxcel: CUDA kernel disk cache disabled: ";

const USERS: &str = "S-1-5-32-545";
const EVERYONE: &str = "S-1-1-0";
const AUTHENTICATED_USERS: &str = "S-1-5-11";

/// The child side of every scenario. A no-op in the parent run.
#[test]
fn ptx_cache_trust_windows_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    if std::env::var_os(PERSISTENT_ENV).is_some() {
        mlxcel_core::ensure_persistent_ptx_cache();
    }
    if mlxcel_core::hardware::gpu_backend_kind() != mlxcel_core::hardware::GpuBackendKind::Cuda {
        eprintln!("{NO_CUDA_TAG}");
        return;
    }
    emit(&launch_fused_moe_pair());
}

struct ChildRun {
    status: std::process::ExitStatus,
    stderr: String,
}

impl ChildRun {
    fn refusals(&self) -> Vec<&str> {
        self.stderr
            .lines()
            .filter(|line| line.contains(REFUSAL))
            .collect()
    }

    fn compiled(&self) -> bool {
        self.stderr.contains(JIT_NOTICE)
    }

    fn assert_success(&self, what: &str) {
        assert!(
            self.status.success(),
            "{what}: the child must succeed: {}\n{}",
            self.status,
            self.stderr
        );
    }

    /// The directory was accepted and every kernel came from it.
    fn assert_hit(&self, what: &str) {
        self.assert_success(what);
        assert!(
            self.refusals().is_empty(),
            "{what}: the directory must be accepted:\n{}",
            self.stderr
        );
        assert!(
            !self.compiled(),
            "{what}: every kernel must come from the cache:\n{}",
            self.stderr
        );
    }

    /// The directory was accepted and the kernels were compiled.
    fn assert_miss(&self, what: &str) {
        self.assert_success(what);
        assert!(
            self.refusals().is_empty(),
            "{what}: the directory must be accepted:\n{}",
            self.stderr
        );
        assert!(
            self.compiled(),
            "{what}: the kernels must be compiled:\n{}",
            self.stderr
        );
    }

    /// One notice holding every piece of `expected`, and the kernels compiled
    /// rather than taken from the directory.
    fn assert_refused(&self, what: &str, expected: &[&str]) {
        self.assert_success(what);
        let refusals = self.refusals();
        assert_eq!(
            refusals.len(),
            1,
            "{what}: the notice must be printed exactly once:\n{}",
            self.stderr
        );
        for piece in expected {
            assert!(
                refusals[0].contains(piece),
                "{what}: expected `{piece}` in `{}`",
                refusals[0]
            );
        }
        assert!(
            self.compiled(),
            "{what}: a planted module must not load, so the kernels are compiled:\n{}",
            self.stderr
        );
        parse_out(&self.stderr);
    }
}

/// A child with the cache variables and the temp directory it may fall back
/// to under `tmp`, then `envs` applied. `None` when it found no CUDA device.
fn run_child_in(cwd: &Path, tmp: &Path, envs: &[(&str, &OsStr)]) -> Option<ChildRun> {
    let fallback = tmp.join("unused-fallback-temp");
    std::fs::create_dir_all(&fallback).expect("fallback temp");
    let mut command = Command::new(std::env::current_exe().expect("test binary path"));
    command
        .args([CHILD_TEST, "--exact", "--nocapture", "--test-threads=1"])
        .current_dir(cwd)
        .env(CHILD_ENV, "1")
        .env_remove("MLX_PTX_CACHE_DIR")
        .env_remove("MLXCEL_CACHE_DIR")
        .env_remove("MLXCEL_QUIET_JIT")
        .env("TEMP", &fallback)
        .env("TMP", &fallback);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command.output().expect("re-run the test binary as a child");
    let run = ChildRun {
        status: output.status,
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    };
    if run.stderr.contains(NO_CUDA_TAG) {
        eprintln!("skipping: no CUDA device");
        return None;
    }
    Some(run)
}

fn run_with_cache(tmp: &Path, cache: &Path) -> Option<ChildRun> {
    run_child_in(tmp, tmp, &[("MLX_PTX_CACHE_DIR", cache.as_os_str())])
}

/// `icacls <path> <args>`. A failed setup fails the test: a refusal could
/// otherwise come from anything, and an acceptance from a grant that never
/// happened.
fn icacls(path: &Path, args: &[&str]) {
    let output = Command::new("icacls")
        .arg(path)
        .args(args)
        .output()
        .expect("run icacls");
    assert!(
        output.status.success(),
        "icacls {} {args:?}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn grant(path: &Path, sid: &str, rights: &str) {
    icacls(path, &["/grant", &format!("*{sid}:{rights}")]);
}

/// A junction needs no privilege, unlike a symlink.
fn junction(link: &Path, target: &Path) {
    let status = Command::new("cmd")
        .arg("/c")
        .arg("mklink")
        .arg("/J")
        .arg(link)
        .arg(target)
        .stdout(std::process::Stdio::null())
        .status()
        .expect("run mklink");
    assert!(status.success(), "mklink /J {}", link.display());
}

/// `path` as `canonical()` in the overlay reports it, without the `\\?\` that
/// `std::fs::canonicalize` adds.
fn shown(path: &Path) -> String {
    let canonical = std::fs::canonicalize(path).expect("canonicalize");
    let text = canonical.display().to_string();
    text.strip_prefix(r"\\?\").unwrap_or(&text).to_string()
}

fn verbatim(path: &Path) -> PathBuf {
    let wide: Vec<u16> = OsStr::new(r"\\?\")
        .encode_wide()
        .chain(path.as_os_str().encode_wide())
        .collect();
    PathBuf::from(OsString::from_wide(&wide))
}

/// Everything under `dir` with its bytes, to show a refused run wrote nothing.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).expect("read_dir").flatten() {
            let path = entry.path();
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                pending.push(path);
            } else {
                let bytes = std::fs::read(&path).expect("read entry");
                found.push((path, bytes));
            }
        }
    }
    found.sort();
    found
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create plant dir");
    for entry in std::fs::read_dir(from).expect("read_dir").flatten() {
        let target = to.join(entry.file_name());
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).expect("copy entry");
        }
    }
}

/// Compile the kernels once into a private cache under `tmp`, and return that
/// run and the device directory holding its entries.
fn reference(tmp: &Path) -> Option<(ChildRun, PathBuf)> {
    let cache = tmp.join("reference");
    std::fs::create_dir(&cache).expect("reference cache");
    let run = run_with_cache(tmp, &cache)?;
    run.assert_miss("reference run");
    let device = device_dir(&cache).unwrap_or_else(|| {
        panic!(
            "entries must sit in one device subdirectory of {}:\n{}",
            cache.display(),
            run.stderr
        )
    });
    Some((run, device))
}

/// Copy the reference entries into `cache`, under the name a run looks for.
fn plant(reference_device: &Path, cache: &Path) -> PathBuf {
    let planted = cache.join(reference_device.file_name().expect("device name"));
    copy_tree(reference_device, &planted);
    planted
}

#[test]
fn private_cache_dir_hits_on_the_second_run() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    std::fs::create_dir(&cache).expect("cache");
    let Some(first) = run_with_cache(tmp.path(), &cache) else {
        return;
    };
    first.assert_miss("first run");
    let device = device_dir(&cache).expect("device subdirectory");
    assert!(
        files_with_extension(&device, "ptx").len() >= 2,
        "both fused MoE kernels must be published at least"
    );
    let second = run_with_cache(tmp.path(), &cache).expect("device");
    second.assert_hit("second run");
    assert_eq!(parse_out(&second.stderr), parse_out(&first.stderr));
}

/// The control for every refusal below: the same planted entries in a private
/// directory do load.
#[test]
fn planted_entries_load_from_a_private_dir() {
    let tmp = private_tempdir();
    let Some((reference_run, device)) = reference(tmp.path()) else {
        return;
    };
    let cache = tmp.path().join("cache");
    std::fs::create_dir(&cache).expect("cache");
    plant(&device, &cache);
    let run = run_with_cache(tmp.path(), &cache).expect("device");
    run.assert_hit("planted in a private directory");
    assert_eq!(parse_out(&run.stderr), parse_out(&reference_run.stderr));
}

#[test]
fn shared_write_entry_on_the_leaf_or_a_parent_is_refused() {
    let tmp = private_tempdir();
    let Some((reference_run, device)) = reference(tmp.path()) else {
        return;
    };
    for sid in [USERS, EVERYONE, AUTHENTICATED_USERS] {
        for (level, rights) in [(0, "M"), (0, "(OI)(CI)(M)"), (1, "M"), (2, "(WD)")] {
            let top = tmp.path().join(format!("{sid}-{level}-{}", rights.len()));
            let middle = top.join("middle");
            let cache = middle.join("cache");
            std::fs::create_dir_all(&cache).expect("cache");
            plant(&device, &cache);
            let target = [&cache, &middle, &top][level];
            grant(target, sid, rights);
            let before = snapshot(&cache);

            let run = run_with_cache(tmp.path(), &cache).expect("device");
            let what = format!("{sid} {rights} at level {level}");
            eprintln!("{}", run.refusals().join("\n"));
            let writable = format!("{} is writable by {sid}", shown(target));
            run.assert_refused(&what, &[&writable]);
            assert_eq!(
                parse_out(&run.stderr),
                parse_out(&reference_run.stderr),
                "{what}: a refused cache must not change the kernel output"
            );
            assert_eq!(
                snapshot(&cache),
                before,
                "{what}: a refused cache must not be written"
            );
        }
    }
}

#[test]
fn a_write_entry_inherited_by_the_leaf_is_refused() {
    let tmp = private_tempdir();
    let Some((_, device)) = reference(tmp.path()) else {
        return;
    };
    let top = tmp.path().join("top");
    std::fs::create_dir(&top).expect("top");
    grant(&top, AUTHENTICATED_USERS, "(OI)(CI)(M)");
    let cache = top.join("cache");
    std::fs::create_dir(&cache).expect("cache");
    plant(&device, &cache);
    let run = run_with_cache(tmp.path(), &cache).expect("device");
    eprintln!("{}", run.refusals().join("\n"));
    run.assert_refused(
        "inherited from a folder such as D:\\mlxcel",
        &[&format!(
            "{} is writable by {AUTHENTICATED_USERS}",
            shown(&cache)
        )],
    );
}

/// The entry reaches every file written in the leaf, so it counts even though
/// it does not apply to the leaf itself.
#[test]
fn an_inherit_only_write_entry_on_the_leaf_is_refused() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    std::fs::create_dir(&cache).expect("cache");
    grant(&cache, USERS, "(OI)(IO)(M)");
    let Some(run) = run_with_cache(tmp.path(), &cache) else {
        return;
    };
    run.assert_refused("inherit-only on the leaf", &[USERS]);
}

/// Read access for others, and a deny entry, grant no one a way to plant.
#[test]
fn read_only_and_deny_entries_are_accepted() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    std::fs::create_dir(&cache).expect("cache");
    grant(&cache, USERS, "(OI)(CI)(RX)");
    grant(tmp.path(), EVERYONE, "(RX)");
    icacls(&cache, &["/deny", &format!("*{AUTHENTICATED_USERS}:(WD)")]);
    let Some(first) = run_with_cache(tmp.path(), &cache) else {
        return;
    };
    first.assert_miss("first run over read-only entries");
    let second = run_with_cache(tmp.path(), &cache).expect("device");
    second.assert_hit("second run over read-only entries");
}

#[test]
fn a_missing_cache_dir_under_a_shared_parent_is_not_created() {
    let tmp = private_tempdir();
    let top = tmp.path().join("top");
    std::fs::create_dir(&top).expect("top");
    grant(&top, USERS, "(M)");
    let cache = top.join("new").join("cache");
    let Some(run) = run_with_cache(tmp.path(), &cache) else {
        return;
    };
    run.assert_refused("missing under a shared parent", &[USERS]);
    assert!(
        !top.join("new").exists(),
        "nothing may be created below a refused directory"
    );
}

#[test]
fn a_file_in_place_of_the_cache_dir_is_refused() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    std::fs::write(&cache, b"").expect("file");
    let Some(run) = run_with_cache(tmp.path(), &cache) else {
        return;
    };
    run.assert_refused("a file", &["is not a directory"]);
}

/// `jit_module.cpp` keeps the path with a `\\?\` prefix for long paths, which
/// changes how `std::filesystem` splits it: the drive root stops being the
/// root. A variable may hold that spelling too.
#[test]
fn a_verbatim_spelled_cache_dir_is_checked_like_any_other() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    std::fs::create_dir(&cache).expect("cache");
    let spelled = verbatim(&cache);
    let Some(first) = run_with_cache(tmp.path(), &spelled) else {
        return;
    };
    first.assert_miss("first run");
    let second = run_with_cache(tmp.path(), &spelled).expect("device");
    second.assert_hit("second run");

    grant(&cache, EVERYONE, "(M)");
    let refused = run_with_cache(tmp.path(), &spelled).expect("device");
    refused.assert_refused(
        "verbatim spelling",
        &[&format!("{} is writable by {EVERYONE}", shown(&cache))],
    );
    assert!(
        !refused.refusals()[0].contains(r"\\?\"),
        "the notice shows the plain path: {}",
        refused.refusals()[0]
    );
}

/// Past `MAX_PATH`, which is why upstream added the prefix.
#[test]
fn a_cache_dir_past_max_path_works_and_is_checked() {
    let tmp = private_tempdir();
    let mut cache = tmp.path().to_path_buf();
    while cache.as_os_str().len() < 300 {
        cache.push("a-directory-name-of-some-length");
    }
    std::fs::create_dir_all(&cache).expect("long cache");
    let Some(first) = run_with_cache(tmp.path(), &cache) else {
        return;
    };
    first.assert_miss("first run past MAX_PATH");
    let second = run_with_cache(tmp.path(), &cache).expect("device");
    second.assert_hit("second run past MAX_PATH");

    // Set on a short parent and inherited, since icacls stops at MAX_PATH.
    let top = tmp.path().join("a-directory-name-of-some-length");
    grant(&top, USERS, "(M)");
    let refused = run_with_cache(tmp.path(), &cache).expect("device");
    refused.assert_refused("a parent past MAX_PATH", &[USERS]);
}

#[path = "cuda_ptx_cache_trust_windows/location.rs"]
mod location;
