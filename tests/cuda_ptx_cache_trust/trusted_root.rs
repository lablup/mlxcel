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

//! The trusted group and the trusted root of the PTX cache (issue #3503).
//!
//! A container volume root is root-owned and group-writable (`fsGroup`) or
//! mode 0777, and the process cannot change it, so the walk of #2539 refused
//! every cache under one. Two rules accept it:
//!
//! - Rule 1: a component only its group can write, besides the owner, passes
//!   when that group is the gid the Rust side handed over as provably private.
//!   The scenarios set it through the `MLXCEL_PTX_CACHE_TEST_TRUSTED_GID`
//!   hook, so they do not depend on the host's groups, and skip in a profile
//!   that does not compile the hook in.
//! - Rule 2: `MLXCEL_CACHE_TRUSTED_ROOT` names an ancestor that is not judged;
//!   everything below it must be a real directory owned by the current user
//!   that no one else can write.
//!
//! The same rules run without a GPU in
//! `scripts/mlxcelverse/cuda_runtime_dir_test.sh`.

use super::*;

const IGNORED: &str = "mlxcel: MLXCEL_CACHE_TRUSTED_ROOT ignored: ";

fn hooks_compiled_in() -> bool {
    if !mlxcel_core::cuda_arch::PTX_CACHE_TEST_HOOKS {
        eprintln!(
            "skipping: {TRUSTED_GID_ENV} is compiled in only for the debug and test-fast \
             profiles, not this one"
        );
    }
    mlxcel_core::cuda_arch::PTX_CACHE_TEST_HOOKS
}

fn gid_of(path: &Path) -> u32 {
    std::fs::metadata(path).expect("metadata").gid()
}

/// Every file under `dir` with its inode and modification time. A cache hit
/// leaves this unchanged; a miss republishes the entry, which is a new inode.
fn snapshot(dir: &Path) -> Vec<(PathBuf, u64, SystemTime)> {
    let mut files: Vec<_> = ["ptx", "txt", "cu"]
        .iter()
        .flat_map(|ext| files_with_extension(dir, ext))
        .map(|path| {
            let meta = std::fs::symlink_metadata(&path).expect("entry metadata");
            (path, meta.ino(), meta.modified().expect("mtime"))
        })
        .collect();
    files.sort();
    files
}

/// `first` populated `cache` and `second` hit it: no refusal, no compile, the
/// same output, and not one file written.
fn assert_populated_then_hit(
    first: &ChildRun,
    second: &ChildRun,
    before: &[(PathBuf, u64, SystemTime)],
    cache: &Path,
) {
    for (run, what) in [(first, "populating run"), (second, "second run")] {
        run.assert_success(what);
        assert!(
            run.refusals().is_empty() && !run.stderr.contains(IGNORED),
            "{what}: the cache must be accepted:\n{}",
            run.stderr
        );
    }
    assert!(
        first.stderr.contains(JIT_NOTICE),
        "the first run compiles:\n{}",
        first.stderr
    );
    assert!(
        before.len() >= 4,
        "the first run must publish both fused MoE kernels: {before:?}"
    );
    assert!(
        !second.stderr.contains(JIT_NOTICE),
        "the second run must hit, not compile:\n{}",
        second.stderr
    );
    assert_eq!(
        snapshot(cache),
        before,
        "a hit must not write any cache file"
    );
    assert_eq!(
        parse_out(&second.stderr),
        parse_out(&first.stderr),
        "a hit must not change the kernel output"
    );
}

/// Run twice with `envs` and require the second run to hit `cache`.
fn assert_two_runs_hit(envs: &[(&str, &OsStr)], cache: &Path) {
    let Some(first) = run_child(envs) else {
        return;
    };
    let before = snapshot(cache);
    let Some(second) = run_child(envs) else {
        return;
    };
    eprintln!("{}\n{}", first.stderr, second.stderr);
    assert_populated_then_hit(&first, &second, &before, cache);
}

/// The directory components under `root` down to each published entry are
/// private (mode 0700) and owned by the owner of `owned_like`.
fn assert_private_chain(root: &Path, owned_like: &Path) {
    let uid = std::fs::metadata(owned_like).expect("metadata").uid();
    let entries = published_ptx(root);
    assert!(!entries.is_empty(), "entries under {}", root.display());
    for entry in entries {
        let mut dir = entry.parent().expect("parent");
        while dir != root {
            let meta = std::fs::symlink_metadata(dir).expect("component metadata");
            assert!(
                meta.file_type().is_dir(),
                "{} is a directory",
                dir.display()
            );
            assert_eq!(meta.uid(), uid, "{} is owned by the user", dir.display());
            // The permission bits only: a directory made under a setgid
            // parent (a volume root with `fsGroup`) inherits that bit.
            assert_eq!(
                meta.mode() & 0o777,
                0o700,
                "{} must be created 0700",
                dir.display()
            );
            dir = dir.parent().expect("parent");
        }
    }
}

// Rule 1.

#[test]
fn group_writable_dir_of_the_trusted_group_is_accepted() {
    if !hooks_compiled_in() {
        return;
    }
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o770);
    let gid = gid_of(&cache).to_string();
    assert_two_runs_hit(
        &[
            ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
            (TRUSTED_GID_ENV, OsStr::new(&gid)),
        ],
        &cache,
    );
}

/// The shape of a volume root with `fsGroup`: the group-writable directory is
/// above the cache, and the cache below it is private.
#[test]
fn cache_under_group_writable_parent_of_the_trusted_group_is_accepted() {
    if !hooks_compiled_in() {
        return;
    }
    let tmp = private_tempdir();
    let volume = tmp.path().join("volume");
    make_dir(&volume, 0o2775);
    let cache = volume.join("cache");
    let gid = gid_of(&volume).to_string();
    assert_two_runs_hit(
        &[
            ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
            (TRUSTED_GID_ENV, OsStr::new(&gid)),
        ],
        &cache,
    );
    assert_private_chain(&volume, &volume);
}

#[test]
fn group_writable_dir_of_another_group_is_refused() {
    if !hooks_compiled_in() {
        return;
    }
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o770);
    let other = (gid_of(&cache) + 1).to_string();
    let Some(run) = run_child(&[
        ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
        (TRUSTED_GID_ENV, OsStr::new(&other)),
    ]) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    assert_refused(&run, &cache, &cache, "is group- or other-writable");
}

/// Rule 1 excuses group write only: an other-writable ancestor stays refused
/// even when its group is the trusted one.
#[test]
fn other_writable_parent_is_refused_despite_the_trusted_group() {
    if !hooks_compiled_in() {
        return;
    }
    let tmp = private_tempdir();
    let open = tmp.path().join("open");
    make_dir(&open, 0o777);
    let cache = open.join("cache");
    make_dir(&cache, 0o700);
    let gid = gid_of(&open).to_string();
    let Some(run) = run_child(&[
        ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
        (TRUSTED_GID_ENV, OsStr::new(&gid)),
    ]) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    assert_refused(&run, &cache, &open, "is group- or other-writable");
}

// Rule 2.

#[test]
fn cache_under_other_writable_trusted_root_is_accepted() {
    let tmp = private_tempdir();
    let open = tmp.path().join("open");
    make_dir(&open, 0o777);
    let cache = open.join("cache");
    assert_two_runs_hit(
        &[
            ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
            (TRUSTED_ROOT_ENV, open.as_os_str()),
        ],
        &cache,
    );
    assert_private_chain(&open, &open);
}

/// The image and chart default: the trusted root is `MLXCEL_CACHE_DIR` itself
/// and `ensure_persistent_ptx_cache` places the cache under it.
#[test]
fn persistent_cache_under_trusted_cache_root_is_accepted() {
    let tmp = private_tempdir();
    let volume = tmp.path().join("volume");
    make_dir(&volume, 0o777);
    assert_two_runs_hit(
        &[
            ("MLXCEL_CACHE_DIR", volume.as_os_str()),
            (TRUSTED_ROOT_ENV, volume.as_os_str()),
            (PERSISTENT_ENV, OsStr::new("1")),
            // Nothing in this scenario may lean on Rule 1.
            (TRUSTED_GID_ENV, OsStr::new("none")),
        ],
        &volume,
    );
    let cuda_ptx = volume.join("cuda-ptx");
    assert!(
        published_ptx(&cuda_ptx).len() >= 2,
        "the persistent default must sit under cuda-ptx"
    );
    assert_private_chain(&volume, &volume);
}

/// Below the trusted root the rule is stricter than the walk's: group write
/// is refused whatever the group, the trusted one included.
#[test]
fn group_writable_leaf_below_trusted_root_is_refused() {
    let tmp = private_tempdir();
    let open = tmp.path().join("open");
    make_dir(&open, 0o777);
    let cache = open.join("cache");
    make_dir(&cache, 0o770);
    let gid = gid_of(&cache).to_string();
    let Some(run) = run_child(&[
        ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
        (TRUSTED_ROOT_ENV, open.as_os_str()),
        (TRUSTED_GID_ENV, OsStr::new(&gid)),
    ]) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    assert_refused(&run, &cache, &cache, "is group- or other-writable");
}

/// Root owns `/usr/share`, which the walk accepts as an owner. Below a
/// trusted root only the current user counts: there the process created
/// every component itself, so anything else was put there by someone else.
#[test]
fn root_owned_leaf_below_trusted_root_is_refused() {
    let probe = private_tempdir();
    if std::fs::metadata(probe.path()).expect("metadata").uid() == 0 {
        eprintln!("skipping: running as root, who owns /usr/share");
        return;
    }
    let Some(run) = run_child(&[
        ("MLX_PTX_CACHE_DIR", OsStr::new("/usr/share")),
        (TRUSTED_ROOT_ENV, OsStr::new("/usr")),
    ]) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    run.assert_success("root-owned leaf");
    assert_eq!(
        run.refusals(),
        vec![format!(
            "{REFUSAL}/usr/share is not owned by the current user"
        )],
        "exactly one refusal naming /usr/share:\n{}",
        run.stderr
    );
    assert!(
        run.stderr.contains(JIT_NOTICE),
        "kernels must still be compiled without a cache:\n{}",
        run.stderr
    );
}

/// Whoever can write the trusted root can replace the cache's first component
/// with a symlink to another directory this user owns, here one shaped like a
/// cache and full of entries. The path is split as given and opened without
/// following a link, so that is refused rather than resolved and loaded.
#[test]
fn symlink_planted_below_trusted_root_is_refused() {
    let tmp = private_tempdir();
    let open = tmp.path().join("open");
    make_dir(&open, 0o777);
    // A populated cache elsewhere under the root, standing in for a
    // directory of this user that holds files of someone else's choosing.
    let shaped = open.join("models");
    let Some(reference) = run_child(&[
        ("MLX_PTX_CACHE_DIR", shaped.as_os_str()),
        (TRUSTED_ROOT_ENV, open.as_os_str()),
    ]) else {
        return;
    };
    reference.assert_success("reference run");
    let before = snapshot(&shaped);
    assert!(before.len() >= 4, "the shaped directory holds entries");

    let cache = open.join("cache");
    std::os::unix::fs::symlink(&shaped, &cache).expect("plant the symlink");
    let Some(run) = run_child(&[
        ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
        (TRUSTED_ROOT_ENV, open.as_os_str()),
    ]) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    let component = Path::new(&canonical(&open)).join("cache");
    assert_symlink_refused(&run, &reference, &component, &tmp.path().join("nowhere"));
    assert_eq!(
        snapshot(&shaped),
        before,
        "nothing may be read into or written through the symlink"
    );
}

/// The opt-in changes which directories are trusted, not which files: a
/// group-writable entry is still a miss.
#[test]
fn group_writable_entry_below_trusted_root_is_not_loaded() {
    let tmp = private_tempdir();
    let open = tmp.path().join("open");
    make_dir(&open, 0o777);
    let cache = open.join("cache");
    let envs = [
        ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
        (TRUSTED_ROOT_ENV, open.as_os_str()),
    ];
    let Some(first) = run_child(&envs) else {
        return;
    };
    first.assert_success("populating run");
    let entries = published_ptx(&cache);
    assert!(entries.len() >= 2, "both kernels published: {entries:?}");
    for entry in &entries {
        std::fs::set_permissions(entry, std::fs::Permissions::from_mode(0o664))
            .expect("make the entry group-writable");
    }
    let Some(second) = run_child(&envs) else {
        return;
    };
    entries::assert_recompiled_and_republished(&second, &first, &entries);
}

/// An unusable trusted root is ignored with one line of its own, and the
/// walk then decides: the other-writable parent is refused as without it.
#[test]
fn unusable_trusted_root_is_ignored_and_the_walk_applies() {
    let tmp = private_tempdir();
    let open = tmp.path().join("open");
    make_dir(&open, 0o777);
    let cache = open.join("cache");
    make_dir(&cache, 0o700);
    let file = tmp.path().join("file");
    std::fs::write(&file, b"").expect("write file");
    let elsewhere = tmp.path().join("elsewhere");
    make_dir(&elsewhere, 0o700);
    let missing = tmp.path().join("missing");
    // The notice names the cache directory as it was given.
    let cache_text = cache.display().to_string();

    let cases: Vec<(&OsStr, String)> = vec![
        (
            OsStr::new("rel/root"),
            "must be an absolute path, got \"rel/root\"".to_string(),
        ),
        (
            missing.as_os_str(),
            format!("{} cannot be resolved (", missing.display()),
        ),
        (
            file.as_os_str(),
            format!("{} is not a directory", canonical(&file)),
        ),
        (
            elsewhere.as_os_str(),
            format!(
                "{} is not an ancestor of {cache_text}",
                canonical(&elsewhere)
            ),
        ),
        // The cache directory itself is not a proper ancestor.
        (
            cache.as_os_str(),
            format!("{} is not an ancestor of {cache_text}", canonical(&cache)),
        ),
    ];
    for (value, reason) in cases {
        let Some(run) = run_child(&[
            ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
            (TRUSTED_ROOT_ENV, value),
        ]) else {
            return;
        };
        let ignored: Vec<&str> = run
            .stderr
            .lines()
            .filter(|line| line.contains(IGNORED))
            .collect();
        eprintln!("{}\n{}", ignored.join("\n"), run.refusals().join("\n"));
        assert_eq!(
            ignored.len(),
            1,
            "one ignored line for {value:?}:\n{}",
            run.stderr
        );
        let expected = format!("{IGNORED}{reason}");
        assert!(
            ignored[0].starts_with(&expected),
            "expected `{expected}`, got `{}`",
            ignored[0]
        );
        assert_refused(&run, &cache, &open, "is group- or other-writable");
    }
}

/// An empty value counts as unset and prints nothing.
#[test]
fn empty_trusted_root_counts_as_unset() {
    let tmp = private_tempdir();
    let open = tmp.path().join("open");
    make_dir(&open, 0o777);
    let cache = open.join("cache");
    make_dir(&cache, 0o700);
    let Some(run) = run_child(&[
        ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
        (TRUSTED_ROOT_ENV, OsStr::new("")),
    ]) else {
        return;
    };
    assert!(
        !run.stderr.contains(IGNORED),
        "an empty value prints nothing:\n{}",
        run.stderr
    );
    assert_refused(&run, &cache, &open, "is group- or other-writable");
}
