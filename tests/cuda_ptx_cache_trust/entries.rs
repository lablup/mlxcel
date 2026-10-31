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

//! Cache path normalization and per-entry checks (issue #2765).
//!
//! The directory checks of #2539 decide whether a cache directory is trusted;
//! these scenarios cover what is inside it and how its path is spelled. A
//! `..` in `MLX_PTX_CACHE_DIR` is resolved before any directory is created,
//! and a `.ptx` entry is loaded only when it is a regular file owned by the
//! current user or by root that no one else can write. Anything else is a
//! miss: the kernel is compiled again and the entry republished in its place.

use super::*;

/// Run once to populate `cache`, and return that run and the published
/// `.ptx` entries of its device subdirectory.
fn populate(cache: &Path) -> Option<(ChildRun, Vec<PathBuf>)> {
    let first = run_with_cache(cache)?;
    first.assert_success("populating run");
    let device = device_dir(cache).expect("device subdirectory");
    let entries = published_ptx(&device);
    assert!(
        entries.len() >= 2,
        "both fused MoE kernels must be published at least: {entries:?}"
    );
    Some((first, entries))
}

/// The second run found every planted entry unusable: it compiled the
/// kernels again, produced the first run's output, and republished each
/// entry as a private regular file.
fn assert_recompiled_and_republished(second: &ChildRun, first: &ChildRun, entries: &[PathBuf]) {
    second.assert_success("run over the planted entries");
    assert!(
        second.refusals().is_empty(),
        "the cache directory itself is trusted:\n{}",
        second.stderr
    );
    assert!(
        second.stderr.contains(JIT_NOTICE),
        "an untrusted entry must be a miss, so the kernels are compiled again:\n{}",
        second.stderr
    );
    assert_eq!(
        parse_out(&second.stderr),
        parse_out(&first.stderr),
        "recompiling must not change the kernel output"
    );
    for entry in entries {
        let meta = std::fs::symlink_metadata(entry)
            .unwrap_or_else(|e| panic!("{} must be republished: {e}", entry.display()));
        assert!(
            meta.file_type().is_file(),
            "{} must be republished as a regular file",
            entry.display()
        );
        assert_eq!(
            meta.mode() & 0o022,
            0,
            "{} must be republished without group or other write, mode {:o}",
            entry.display(),
            meta.mode() & 0o7777
        );
    }
}

#[test]
fn dotdot_in_cache_dir_is_normalized() {
    let tmp = private_tempdir();
    let spelled = tmp.path().join("missing").join("..").join("cache");
    let Some(run) = run_with_cache(&spelled) else {
        return;
    };
    run.assert_success("cache path with ..");
    assert!(
        run.refusals().is_empty(),
        "a private cache dir spelled with .. must be accepted:\n{}",
        run.stderr
    );
    let cache = tmp.path().join("cache");
    let device = device_dir(&cache).unwrap_or_else(|| {
        panic!(
            "entries must be published under {}/<device dir>",
            cache.display()
        )
    });
    assert!(
        !published_ptx(&device).is_empty(),
        "the normalized cache must be populated"
    );
    assert!(
        !tmp.path().join("missing").exists(),
        "the component before .. must not be created"
    );
}

/// A relative path resolves against the current directory, which anyone can
/// pick for a victim (#2797), so it is refused and nothing is created there.
fn assert_relative_refused(run: &ChildRun, name: &str, value: &str, cwd: &Path) {
    run.assert_success("relative cache path");
    let refusals = run.refusals();
    assert_eq!(
        refusals.len(),
        1,
        "the refusal notice must be printed exactly once:\n{}",
        run.stderr
    );
    let expected = format!("{REFUSAL}{name} must be an absolute path, got \"{value}\"");
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
    let created: Vec<_> = std::fs::read_dir(cwd)
        .expect("cwd")
        .flatten()
        .map(|e| e.file_name())
        .collect();
    assert!(
        created.is_empty(),
        "nothing may be created there: {created:?}"
    );
}

#[test]
fn relative_mlx_ptx_cache_dir_is_refused() {
    let tmp = private_tempdir();
    let cwd = tmp.path().join("cwd");
    make_dir(&cwd, 0o700);
    let Some(run) = run_child_in(&cwd, &[("MLX_PTX_CACHE_DIR", OsStr::new("rel-cache"))]) else {
        return;
    };
    eprintln!("{}", run.refusals().join("\n"));
    assert_relative_refused(&run, "MLX_PTX_CACHE_DIR", "rel-cache", &cwd);
}

#[test]
fn relative_mlxcel_cache_dir_is_refused() {
    let tmp = private_tempdir();
    let cwd = tmp.path().join("cwd");
    make_dir(&cwd, 0o700);
    let fallback = tmp.path().join("temp");
    make_dir(&fallback, 0o700);
    for persistent in [false, true] {
        let mut envs = vec![
            ("MLXCEL_CACHE_DIR", OsStr::new("rel-root")),
            ("TMPDIR", fallback.as_os_str()),
        ];
        if persistent {
            envs.push((PERSISTENT_ENV, OsStr::new("1")));
        }
        let Some(run) = run_child_in(&cwd, &envs) else {
            return;
        };
        eprintln!("{}", run.refusals().join("\n"));
        assert_relative_refused(&run, "MLXCEL_CACHE_DIR", "rel-root", &cwd);
    }
}

#[test]
fn group_writable_cache_entry_is_not_loaded() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o700);
    let Some((first, entries)) = populate(&cache) else {
        return;
    };
    for entry in &entries {
        std::fs::set_permissions(entry, std::fs::Permissions::from_mode(0o664))
            .expect("make the entry group-writable");
    }
    let Some(second) = run_with_cache(&cache) else {
        return;
    };
    assert_recompiled_and_republished(&second, &first, &entries);
}

#[test]
fn symlinked_cache_entry_is_not_loaded() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o700);
    let Some((first, entries)) = populate(&cache) else {
        return;
    };
    // Each entry becomes a symlink to an identical private copy elsewhere, so
    // only the symlink itself can make the entry a miss.
    let copies = tmp.path().join("copies");
    make_dir(&copies, 0o700);
    let mut planted = Vec::new();
    for entry in &entries {
        let copy = copies.join(entry.file_name().expect("entry name"));
        std::fs::copy(entry, &copy).expect("copy the entry");
        std::fs::remove_file(entry).expect("remove the entry");
        std::os::unix::fs::symlink(&copy, entry).expect("plant the entry symlink");
        planted.push((copy.clone(), std::fs::read(&copy).expect("copy contents")));
    }
    let Some(second) = run_with_cache(&cache) else {
        return;
    };
    assert_recompiled_and_republished(&second, &first, &entries);
    for (copy, contents) in planted {
        assert_eq!(
            std::fs::read(&copy).expect("symlink target"),
            contents,
            "republishing must replace the symlink, not write through it"
        );
    }
}

#[test]
fn group_writable_kernel_map_is_not_loaded() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    make_dir(&cache, 0o700);
    let Some((first, entries)) = populate(&cache) else {
        return;
    };
    // The .ptx files stay valid; only the .txt kernel maps are tampered with.
    let maps: Vec<PathBuf> = entries.iter().map(|e| e.with_extension("txt")).collect();
    for map in &maps {
        std::fs::set_permissions(map, std::fs::Permissions::from_mode(0o664))
            .unwrap_or_else(|e| panic!("make {} group-writable: {e}", map.display()));
    }
    let Some(second) = run_with_cache(&cache) else {
        return;
    };
    assert_recompiled_and_republished(&second, &first, &maps);
}
