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

//! Trust and keying of the CUDA JIT PTX disk cache (issue #2539).
//!
//! The `jit_module.cpp` overlay loads compiled modules from the cache, so a
//! directory another user can write is a code-injection path into every
//! process that shares it. The overlay now refuses a cache directory unless
//! every component from the leaf up to `/` is owned by the current user or by
//! root and is not group- or other-writable (a root-owned sticky directory
//! such as `/tmp` excepted), and prints one notice when it does. The POSIX
//! fallback moved to a per-user `$TMPDIR/mlx-<euid>/` root, entries are keyed
//! by a `sm<cc>-nvrtc<version>` subdirectory, orphaned publish temporaries are
//! swept, and temporaries are opened with `O_CREAT|O_EXCL|O_NOFOLLOW`.
//!
//! `MLX_PTX_CACHE_DIR` and `TMPDIR` are read once per process, so every
//! scenario re-runs this binary as a child with its own environment; the
//! child launches the fused MoE kernel pair and prints its output. The parent
//! tests never touch CUDA themselves.
//!
//! ```sh
//! cargo test --profile test-fast --features cuda --test cuda_ptx_cache_trust --no-run
//! gpu-lock run --tag ptx-cache -- target/test-fast/deps/cuda_ptx_cache_trust-<hash> --test-threads=1 --nocapture
//! ```

#![cfg(all(feature = "cuda", unix))]

mod common;

use std::ffi::OsStr;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

use common::cuda_kernels::{
    JIT_NOTICE, device_dir, emit, files_with_extension, launch_fused_moe_pair, parse_out,
    private_tempdir,
};

const CHILD_TEST: &str = "ptx_cache_trust_child";
const CHILD_ENV: &str = "MLXCEL_TEST_PTX_TRUST_CHILD";
const NO_CUDA_TAG: &str = "MLXCEL_PTX_TRUST_NO_CUDA";
const CC_TAG: &str = "MLXCEL_PTX_TRUST_CC=";
const REFUSAL: &str = "mlxcel: CUDA kernel disk cache disabled: ";
/// The overlay's test-only hook that fixes the publish temporaries' suffix.
const TMP_SUFFIX_ENV: &str = "MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX";
const TWO_HOURS: Duration = Duration::from_secs(2 * 60 * 60);

/// The child side of every scenario: launch the fused MoE pair and print its
/// output and the device's compute capability. A no-op in the parent run.
#[test]
fn ptx_cache_trust_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    if mlxcel_core::hardware::gpu_backend_kind() != mlxcel_core::hardware::GpuBackendKind::Cuda {
        eprintln!("{NO_CUDA_TAG}");
        return;
    }
    let (major, minor) =
        mlxcel_core::cuda_arch::cuda_compute_capability().expect("CUDA compute capability");
    eprintln!("{CC_TAG}{major}.{minor}");
    emit(&launch_fused_moe_pair());
}

struct ChildRun {
    status: std::process::ExitStatus,
    stderr: String,
}

impl ChildRun {
    fn assert_success(&self, what: &str) {
        assert!(
            self.status.success(),
            "{what}: the child must succeed: {}\n{}",
            self.status,
            self.stderr
        );
    }

    fn refusals(&self) -> Vec<&str> {
        self.stderr
            .lines()
            .filter(|line| line.contains(REFUSAL))
            .collect()
    }

    /// `(major, minor)` as the child reported it.
    fn compute_capability(&self) -> (u32, u32) {
        let cc = self
            .stderr
            .lines()
            .find_map(|l| l.strip_prefix(CC_TAG))
            .expect("child printed its compute capability");
        let (major, minor) = cc.split_once('.').expect("major.minor");
        (major.parse().expect("major"), minor.parse().expect("minor"))
    }
}

/// Re-run this binary as a child with `MLX_PTX_CACHE_DIR`, `TMPDIR` and the
/// temporary-suffix hook cleared, then `envs` applied. `None` when the child
/// found no CUDA device (the scenario is skipped).
fn run_child(envs: &[(&str, &OsStr)]) -> Option<ChildRun> {
    let mut command = Command::new(std::env::current_exe().expect("test binary path"));
    command
        .args([CHILD_TEST, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env_remove("MLX_PTX_CACHE_DIR")
        .env_remove("MLXCEL_QUIET_JIT")
        .env_remove(TMP_SUFFIX_ENV);
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

fn run_with_cache(cache: &Path) -> Option<ChildRun> {
    run_child(&[("MLX_PTX_CACHE_DIR", cache.as_os_str())])
}

fn make_dir(path: &Path, mode: u32) {
    std::fs::create_dir(path).expect("create test dir");
    // Set the mode explicitly: create_dir is subject to the umask.
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).expect("chmod");
}

fn canonical(path: &Path) -> String {
    std::fs::canonicalize(path)
        .expect("canonicalize")
        .display()
        .to_string()
}

fn set_mtime(path: &Path, when: SystemTime) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .expect("open for set_modified")
        .set_modified(when)
        .expect("set_modified");
}

fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
}

/// Regular `.ptx` files under `dir`, symlinks excluded.
fn published_ptx(dir: &Path) -> Vec<PathBuf> {
    files_with_extension(dir, "ptx")
        .into_iter()
        .filter(|p| std::fs::symlink_metadata(p).is_ok_and(|m| m.file_type().is_file()))
        .collect()
}

/// A refused cache must leave compilation working and print exactly one
/// notice naming the offending component.
fn assert_refused(run: &ChildRun, cache: &Path, component: &Path, reason: &str) {
    run.assert_success("refused cache");
    let refusals = run.refusals();
    assert_eq!(
        refusals.len(),
        1,
        "the refusal notice must be printed exactly once:\n{}",
        run.stderr
    );
    let expected = format!("{REFUSAL}{} {reason}", canonical(component));
    assert!(
        refusals[0].contains(&expected),
        "expected `{expected}` in the refusal notice, got `{}`",
        refusals[0]
    );
    assert!(
        run.stderr.contains(JIT_NOTICE),
        "kernels must still be compiled without a cache:\n{}",
        run.stderr
    );
    parse_out(&run.stderr);
    let leaked = files_with_extension(cache, "ptx");
    assert!(
        leaked.is_empty(),
        "a refused cache dir must receive no entries: {leaked:?}"
    );
}

#[test]
fn group_writable_cache_dir_is_refused() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o770);
    let Some(run) = run_with_cache(&cache) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    assert_refused(&run, &cache, &cache, "is group- or other-writable");
}

#[test]
fn cache_dir_under_world_writable_parent_is_refused() {
    let tmp = private_tempdir();
    let open = tmp.path().join("open");
    make_dir(&open, 0o777);
    let cache = open.join("cache");
    make_dir(&cache, 0o700);
    let Some(run) = run_with_cache(&cache) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    assert_refused(&run, &cache, &open, "is group- or other-writable");
}

#[test]
fn private_cache_dir_is_accepted() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o700);
    let Some(run) = run_with_cache(&cache) else {
        return;
    };
    run.assert_success("private cache");
    assert!(
        run.refusals().is_empty(),
        "a private cache dir must be accepted:\n{}",
        run.stderr
    );
    assert!(
        published_ptx(&cache).len() >= 2,
        "the accepted cache must be populated (both fused MoE kernels at least)"
    );
}

#[test]
fn fallback_cache_is_per_user_under_tmpdir() {
    let tmp = private_tempdir();
    let probe = tmp.path().join("euid-probe");
    std::fs::write(&probe, b"").expect("write probe file");
    let euid = std::fs::metadata(&probe).expect("probe metadata").uid();
    let Some(run) = run_child(&[("TMPDIR", tmp.path().as_os_str())]) else {
        return;
    };
    run.assert_success("fallback cache");
    assert!(
        run.refusals().is_empty(),
        "the per-user fallback must be trusted:\n{}",
        run.stderr
    );
    let user_root = tmp.path().join(format!("mlx-{euid}"));
    let mode = std::fs::metadata(&user_root)
        .unwrap_or_else(|e| panic!("{} must exist: {e}", user_root.display()))
        .permissions()
        .mode();
    assert_eq!(
        mode & 0o077,
        0,
        "the per-user fallback root must be private, mode {mode:o}"
    );
    assert!(
        !published_ptx(&user_root).is_empty(),
        "entries must land under {}",
        user_root.display()
    );
    assert!(
        !tmp.path().join("mlx").exists(),
        "the shared $TMPDIR/mlx root must not be created"
    );
}

#[test]
fn entries_are_keyed_by_device_and_nvrtc() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o700);
    let Some(run) = run_with_cache(&cache) else {
        return;
    };
    run.assert_success("keyed cache");
    let (major, minor) = run.compute_capability();
    let device = device_dir(&cache).unwrap_or_else(|| {
        panic!(
            "the cache root must hold one sm<cc>-nvrtc<version> subdirectory: {:?}",
            std::fs::read_dir(&cache).map(|d| d
                .filter_map(|e| e.ok())
                .map(|e| e.file_name())
                .collect::<Vec<_>>())
        )
    });
    let name = device
        .file_name()
        .and_then(|n| n.to_str())
        .expect("utf-8 name");
    assert!(
        name.starts_with(&format!("sm{major}{minor}-nvrtc")),
        "device subdirectory {name} must match compute capability {major}.{minor}"
    );
    let root_entries: Vec<_> = std::fs::read_dir(&cache)
        .expect("cache root")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    assert_eq!(
        root_entries,
        vec![device.clone()],
        "nothing but the device subdirectory may sit at the cache root"
    );
    let entries = published_ptx(&cache);
    assert!(!entries.is_empty(), "the cache must be populated");
    for entry in &entries {
        assert!(
            entry.starts_with(&device),
            "{} is outside the device subdirectory",
            entry.display()
        );
    }
}

#[test]
fn stale_publish_temporaries_are_swept() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o700);
    let Some(first) = run_with_cache(&cache) else {
        return;
    };
    first.assert_success("populating run");
    let device = device_dir(&cache).expect("device subdirectory");
    let stale = SystemTime::now() - TWO_HOURS;

    let stale_ptx = device.join(".tmp1_1.ptx");
    let stale_txt = device.join(".tmp3_3.txt");
    let fresh_ptx = device.join(".tmp2_2.ptx");
    let unrelated = device.join("notes.tmp4_4");
    for path in [&stale_ptx, &stale_txt, &fresh_ptx, &unrelated] {
        std::fs::write(path, b"orphan").expect("plant temporary");
    }
    for path in [&stale_ptx, &stale_txt, &unrelated] {
        set_mtime(path, stale);
    }
    // A matching name that is a symlink: the sweep must neither follow nor
    // remove it, even when its target is old.
    let target = tmp.path().join("outside-target");
    std::fs::write(&target, b"keep me").expect("symlink target");
    set_mtime(&target, stale);
    let stale_link = device.join(".tmp5_5.ptx");
    std::os::unix::fs::symlink(&target, &stale_link).expect("plant symlink");

    let Some(second) = run_with_cache(&cache) else {
        return;
    };
    second.assert_success("sweeping run");
    assert!(
        !stale_ptx.exists() && !stale_txt.exists(),
        "temporaries older than an hour must be swept"
    );
    assert!(
        fresh_ptx.exists(),
        "a fresh temporary may be a live publish"
    );
    assert!(
        unrelated.exists(),
        "only the publish temporary pattern is swept"
    );
    assert!(
        is_symlink(&stale_link),
        "the sweep must not remove a symlink"
    );
    assert_eq!(
        std::fs::read(&target).expect("target"),
        b"keep me",
        "the sweep must not follow a symlink"
    );
}

#[test]
fn symlinked_temporary_is_not_written_through() {
    let tmp = private_tempdir();
    let reference_cache = tmp.path().join("reference");
    make_dir(&reference_cache, 0o700);
    let Some(reference) = run_with_cache(&reference_cache) else {
        return;
    };
    reference.assert_success("reference run");
    let device_name = device_dir(&reference_cache)
        .expect("device subdirectory")
        .file_name()
        .expect("name")
        .to_owned();

    // A fresh cache whose device directory already holds symlinks at the
    // paths the hook makes every publish use.
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o700);
    let device = cache.join(&device_name);
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&device)
        .expect("device dir");
    let txt_target = tmp.path().join("txt-target");
    let ptx_target = tmp.path().join("ptx-target");
    std::fs::write(&ptx_target, b"original\n").expect("ptx target");
    let txt_link = device.join(".tmp1_1.txt");
    let ptx_link = device.join(".tmp1_1.ptx");
    std::os::unix::fs::symlink(&txt_target, &txt_link).expect("txt symlink");
    std::os::unix::fs::symlink(&ptx_target, &ptx_link).expect("ptx symlink");

    let Some(run) = run_child(&[
        ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
        (TMP_SUFFIX_ENV, OsStr::new(".tmp1_1")),
    ]) else {
        return;
    };
    run.assert_success("planted symlinks");
    assert!(
        run.stderr.contains(JIT_NOTICE),
        "the kernels must be compiled:\n{}",
        run.stderr
    );
    assert_eq!(
        parse_out(&run.stderr),
        parse_out(&reference.stderr),
        "a failed publish must not change the kernel output"
    );
    assert!(
        !txt_target.exists() && std::fs::symlink_metadata(&txt_target).is_err(),
        "the publish must not create a file through the planted .txt symlink"
    );
    assert_eq!(
        std::fs::read(&ptx_target).expect("ptx target"),
        b"original\n",
        "the publish must not write through the planted .ptx symlink"
    );
    assert!(
        is_symlink(&txt_link) && is_symlink(&ptx_link),
        "the planted paths are not ours: they must be left in place"
    );
    assert_eq!(std::fs::read_link(&txt_link).expect("link"), txt_target);
    // Every publish used the colliding name and every kernel map publish
    // failed, so no entry was published: the .ptx goes out only after its map.
    let published = published_ptx(&cache);
    assert!(
        published.is_empty(),
        "a kernel whose temporary collided must stay unpublished: {published:?}"
    );

    // The hook accepts only `.tmp<hex>_<hex>`: any other value, such as one
    // that would leave the directory, is ignored and publishing works.
    let Some(ignored) = run_child(&[
        ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
        (TMP_SUFFIX_ENV, OsStr::new(".tmp1_1/../escape")),
    ]) else {
        return;
    };
    ignored.assert_success("invalid hook value");
    assert!(
        !published_ptx(&device).is_empty(),
        "an invalid suffix hook value must be ignored, not break publishing"
    );
    assert!(
        !cache.join("escape.txt").exists() && !device.join("escape.txt").exists(),
        "an invalid suffix hook value must not name a temporary"
    );
}

/// Assert a run that refused a symlinked cache component: it succeeded, said
/// so exactly once naming `component`, still compiled, matched `reference`,
/// and wrote nothing to the symlink's private `target`.
fn assert_symlink_refused(run: &ChildRun, reference: &ChildRun, component: &Path, target: &Path) {
    run.assert_success("symlinked cache component");
    let refusals = run.refusals();
    assert_eq!(
        refusals.len(),
        1,
        "the refusal notice must be printed exactly once:\n{}",
        run.stderr
    );
    let expected = format!("{REFUSAL}{} is a symbolic link", component.display());
    assert!(
        refusals[0].contains(&expected),
        "expected `{expected}` in the refusal notice, got `{}`",
        refusals[0]
    );
    assert!(
        run.stderr.contains(JIT_NOTICE),
        "kernels must still be compiled without a cache:\n{}",
        run.stderr
    );
    assert_eq!(
        parse_out(&run.stderr),
        parse_out(&reference.stderr),
        "a refused cache must not change the kernel output"
    );
    assert!(
        is_symlink(component),
        "the planted symlink is not ours: it must be left in place"
    );
    let leaked = published_ptx(target);
    assert!(
        leaked.is_empty(),
        "nothing may be written through the symlink: {leaked:?}"
    );
}

/// The device subdirectory is created by the overlay, so a symlink in its
/// place was planted by someone else. Following it would load whatever the
/// target holds as long as the target passes the directory checks, for
/// example a directory that was group-writable until the persistent-cache
/// repair in `ensure_persistent_ptx_cache` tightened it, with another user's
/// modules still inside.
#[test]
fn symlinked_device_dir_is_refused() {
    let tmp = private_tempdir();
    let reference_cache = tmp.path().join("reference");
    make_dir(&reference_cache, 0o700);
    let Some(reference) = run_with_cache(&reference_cache) else {
        return;
    };
    reference.assert_success("reference run");
    let device_name = device_dir(&reference_cache)
        .expect("device subdirectory")
        .file_name()
        .expect("name")
        .to_owned();

    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o700);
    let target = tmp.path().join("elsewhere");
    make_dir(&target, 0o700);
    let link = cache.join(&device_name);
    std::os::unix::fs::symlink(&target, &link).expect("plant device symlink");

    let Some(run) = run_with_cache(&cache) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    let component = Path::new(&canonical(&cache)).join(&device_name);
    assert_symlink_refused(&run, &reference, &component, &target);
}

/// The per-user fallback root `$TMPDIR/mlx-<euid>` sits in a shared
/// directory, where another user can create the name first. A symlink there
/// must not decide where this user's cache goes.
#[test]
fn symlinked_per_user_fallback_root_is_refused() {
    let tmp = private_tempdir();
    let reference_cache = tmp.path().join("reference");
    make_dir(&reference_cache, 0o700);
    let Some(reference) = run_with_cache(&reference_cache) else {
        return;
    };
    reference.assert_success("reference run");

    let probe = tmp.path().join("euid-probe");
    std::fs::write(&probe, b"").expect("write probe file");
    let euid = std::fs::metadata(&probe).expect("probe metadata").uid();
    let target = tmp.path().join("victim-private");
    make_dir(&target, 0o700);
    let link = tmp.path().join(format!("mlx-{euid}"));
    std::os::unix::fs::symlink(&target, &link).expect("plant fallback symlink");

    let Some(run) = run_child(&[("TMPDIR", tmp.path().as_os_str())]) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    assert_symlink_refused(&run, &reference, &link, &target);
    assert!(
        std::fs::read_dir(&target).expect("target").next().is_none(),
        "no directory may be created through the symlink"
    );
}
