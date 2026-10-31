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

//! Where the cache may be on Windows (issue #2797): junctions, the persistent
//! default of `mlxcel` and `mlxcel-server`, the temp fallback of other binaries,
//! and relative paths.

use super::*;

/// A grant on the device directory alone, under a root that passes.
#[test]
fn a_shared_write_entry_on_the_device_dir_is_refused() {
    let tmp = private_tempdir();
    let Some((_, device)) = reference(tmp.path()) else {
        return;
    };
    let cache = tmp.path().join("cache");
    std::fs::create_dir(&cache).expect("cache");
    let planted = plant(&device, &cache);
    grant(&planted, USERS, "(M)");
    let before = snapshot(&cache);
    let run = run_with_cache(tmp.path(), &cache).expect("device");
    eprintln!("{}", run.refusals().join("\n"));
    run.assert_refused(
        "a grant on the device directory",
        &[&format!("{} is writable by {USERS}", shown(&planted))],
    );
    assert_eq!(
        snapshot(&cache),
        before,
        "a refused cache must not be written"
    );
}

/// A directory that was shared and later made private passes its own check and
/// still holds what was planted. Each file of an entry is checked on load, and a
/// refusal is a miss that recompiles and republishes.
#[test]
fn a_planted_entry_with_a_foreign_write_entry_is_not_loaded() {
    let tmp = private_tempdir();
    let Some((reference_run, device)) = reference(tmp.path()) else {
        return;
    };
    for ext in ["ptx", "txt"] {
        let cache = tmp.path().join(format!("cache-{ext}"));
        std::fs::create_dir(&cache).expect("cache");
        let planted = plant(&device, &cache);

        let control = run_with_cache(tmp.path(), &cache).expect("device");
        control.assert_hit("before the grant");

        for entry in files_with_extension(&planted, ext) {
            grant(&entry, USERS, "(M)");
        }
        let run = run_with_cache(tmp.path(), &cache).expect("device");
        run.assert_miss(&format!("a foreign write entry on the .{ext}"));
        assert_eq!(parse_out(&run.stderr), parse_out(&reference_run.stderr));

        let republished = run_with_cache(tmp.path(), &cache).expect("device");
        republished.assert_hit(&format!("the .{ext} is republished with a private ACL"));
    }
}

/// An empty value counts as unset, like `MLXCEL_CACHE_DIR` and the ROCm variable.
#[test]
fn an_empty_mlx_ptx_cache_dir_counts_as_unset() {
    let tmp = private_tempdir();
    let temp = tmp.path().join("temp");
    std::fs::create_dir(&temp).expect("temp");
    let envs = [
        ("MLX_PTX_CACHE_DIR", OsStr::new("")),
        ("TEMP", temp.as_os_str()),
        ("TMP", temp.as_os_str()),
    ];
    let Some(first) = run_child_in(tmp.path(), tmp.path(), &envs) else {
        return;
    };
    first.assert_miss("first run with an empty variable");
    assert!(
        files_with_extension(&temp.join("mlx"), "ptx").len() >= 2,
        "the temp fallback must be used:\n{}",
        first.stderr
    );
    let second = run_child_in(tmp.path(), tmp.path(), &envs).expect("device");
    second.assert_hit("second run with an empty variable");

    // The persistent default replaces the empty value.
    let root = tmp.path().join("root");
    let envs = [
        ("MLX_PTX_CACHE_DIR", OsStr::new("")),
        (PERSISTENT_ENV, OsStr::new("1")),
        ("MLXCEL_CACHE_DIR", root.as_os_str()),
    ];
    let run = run_child_in(tmp.path(), tmp.path(), &envs).expect("device");
    run.assert_miss("the persistent default under an empty variable");
    assert!(
        files_with_extension(&root.join("cuda-ptx"), "ptx").len() >= 2,
        "entries must land under the persistent default:\n{}",
        run.stderr
    );
}

/// A junction at the device subdirectory, a name only the overlay creates, was
/// planted by someone else, even when it points at a private directory.
#[test]
fn a_junction_at_the_device_dir_is_refused() {
    let tmp = private_tempdir();
    let Some((_, device)) = reference(tmp.path()) else {
        return;
    };
    let cache = tmp.path().join("cache");
    std::fs::create_dir(&cache).expect("cache");
    let elsewhere = tmp.path().join("elsewhere");
    copy_tree(&device, &elsewhere);
    junction(
        &cache.join(device.file_name().expect("device name")),
        &elsewhere,
    );
    let run = run_with_cache(tmp.path(), &cache).expect("device");
    eprintln!("{}", run.refusals().join("\n"));
    run.assert_refused("junction at the device dir", &["is a symbolic link"]);
}

/// A junction in the path the user named is judged by its target, as a symlink
/// is on POSIX.
#[test]
fn a_junction_in_the_named_path_is_judged_by_its_target() {
    let tmp = private_tempdir();
    let target = tmp.path().join("target");
    std::fs::create_dir(&target).expect("target");
    let link = tmp.path().join("link");
    junction(&link, &target);
    let Some(first) = run_with_cache(tmp.path(), &link) else {
        return;
    };
    first.assert_miss("first run through a junction");
    let second = run_with_cache(tmp.path(), &link).expect("device");
    second.assert_hit("second run through a junction");

    grant(&target, EVERYONE, "(M)");
    let refused = run_with_cache(tmp.path(), &link).expect("device");
    refused.assert_refused(
        "a shared target",
        &[&format!("{} is writable by {EVERYONE}", shown(&target))],
    );
}

// ---------------------------------------------------------------------------
// The persistent default of mlxcel and mlxcel-server.
// ---------------------------------------------------------------------------

/// `ensure_persistent_ptx_cache` sets `MLX_PTX_CACHE_DIR` with `set_var`, which
/// the C runtime's `getenv` does not see on Windows, so the overlay has to read
/// the process environment itself.
#[test]
fn the_persistent_default_is_used_and_hits_on_the_second_run() {
    let tmp = private_tempdir();
    let root = tmp.path().join("root");
    let envs = [
        (PERSISTENT_ENV, OsStr::new("1")),
        ("MLXCEL_CACHE_DIR", root.as_os_str()),
    ];
    let Some(first) = run_child_in(tmp.path(), tmp.path(), &envs) else {
        return;
    };
    first.assert_miss("first run with the persistent default");
    let published = files_with_extension(&root.join("cuda-ptx"), "ptx");
    assert!(
        published.len() >= 2,
        "entries must land under {}\\cuda-ptx: {published:?}\n{}",
        root.display(),
        first.stderr
    );
    let second = run_child_in(tmp.path(), tmp.path(), &envs).expect("device");
    second.assert_hit("second run with the persistent default");
}

#[test]
fn a_shared_mlxcel_cache_dir_is_refused_for_the_persistent_default() {
    let tmp = private_tempdir();
    let Some((_, device)) = reference(tmp.path()) else {
        return;
    };
    // What D:\mlxcel looks like: inherited write for Authenticated Users.
    let root = tmp.path().join("root");
    std::fs::create_dir(&root).expect("root");
    let envs = [
        (PERSISTENT_ENV, OsStr::new("1")),
        ("MLXCEL_CACHE_DIR", root.as_os_str()),
    ];
    let populated = run_child_in(tmp.path(), tmp.path(), &envs).expect("device");
    populated.assert_success("creating the managed directories");
    let scope_dir = std::fs::read_dir(root.join("cuda-ptx"))
        .expect("cuda-ptx")
        .flatten()
        .next()
        .expect("scope dir")
        .path();
    plant(&device, &scope_dir);
    grant(&root, AUTHENTICATED_USERS, "(OI)(CI)(M)");
    icacls(&scope_dir, &["/inheritance:e"]);
    let run = run_child_in(tmp.path(), tmp.path(), &envs).expect("device");
    eprintln!("{}", run.refusals().join("\n"));
    run.assert_refused(
        "MLXCEL_CACHE_DIR like D:\\mlxcel",
        &["is writable by", AUTHENTICATED_USERS],
    );
}

// ---------------------------------------------------------------------------
// The temp fallback of other binaries.
// ---------------------------------------------------------------------------

#[test]
fn the_temp_fallback_is_checked_and_hits_on_the_second_run() {
    let tmp = private_tempdir();
    let temp = tmp.path().join("temp");
    std::fs::create_dir(&temp).expect("temp");
    let envs = [("TEMP", temp.as_os_str()), ("TMP", temp.as_os_str())];
    let Some(first) = run_child_in(tmp.path(), tmp.path(), &envs) else {
        return;
    };
    first.assert_miss("first run with the temp fallback");
    assert!(
        files_with_extension(&temp.join("mlx"), "ptx").len() >= 2,
        "entries must land under {}\\mlx",
        temp.display()
    );
    let second = run_child_in(tmp.path(), tmp.path(), &envs).expect("device");
    second.assert_hit("second run with the temp fallback");

    // Now the temp directory is shared, with the entries still in it.
    grant(&temp, USERS, "(M)");
    let refused = run_child_in(tmp.path(), tmp.path(), &envs).expect("device");
    eprintln!("{}", refused.refusals().join("\n"));
    refused.assert_refused(
        "a shared temp directory",
        &[&format!("{} is writable by {USERS}", shown(&temp))],
    );
}

#[test]
fn a_junction_at_the_temp_fallback_root_is_refused() {
    let tmp = private_tempdir();
    let temp = tmp.path().join("temp");
    let target = tmp.path().join("target");
    std::fs::create_dir(&temp).expect("temp");
    std::fs::create_dir(&target).expect("target");
    junction(&temp.join("mlx"), &target);
    let envs = [("TEMP", temp.as_os_str()), ("TMP", temp.as_os_str())];
    let Some(run) = run_child_in(tmp.path(), tmp.path(), &envs) else {
        return;
    };
    run.assert_refused("junction at <temp>\\mlx", &["is a symbolic link"]);
    assert!(
        std::fs::read_dir(&target).expect("target").next().is_none(),
        "nothing may be created through the junction"
    );
}

// ---------------------------------------------------------------------------
// Relative paths resolve against a directory anyone can pick for a victim.
// ---------------------------------------------------------------------------

fn assert_cwd_untouched(cwd: &Path) {
    let left: Vec<_> = std::fs::read_dir(cwd)
        .expect("cwd")
        .flatten()
        .map(|e| e.file_name())
        .collect();
    assert!(left.is_empty(), "nothing may be created there: {left:?}");
}

#[test]
fn a_relative_mlx_ptx_cache_dir_is_refused() {
    let tmp = private_tempdir();
    let Some((_, device)) = reference(tmp.path()) else {
        return;
    };
    // Plain, rooted without a drive, and drive-relative: the last two are
    // relative too, since they depend on the current drive or its directory.
    let drive = tmp
        .path()
        .to_str()
        .and_then(|p| p.get(..2))
        .expect("drive prefix");
    for spelled in [
        "rel-cache".to_string(),
        r".\rel-cache".to_string(),
        r"\rel-cache-rooted".to_string(),
        format!("{drive}rel-cache"),
    ] {
        let cwd = tempfile::tempdir_in(tmp.path()).expect("cwd");
        let run = run_child_in(
            cwd.path(),
            tmp.path(),
            &[("MLX_PTX_CACHE_DIR", OsStr::new(&spelled))],
        )
        .expect("device");
        eprintln!("{}", run.refusals().join("\n"));
        run.assert_refused(
            &spelled,
            &["MLX_PTX_CACHE_DIR must be an absolute path", &spelled],
        );
        assert_cwd_untouched(cwd.path());
    }

    // A planted module at the relative path stays unloaded.
    let cwd = tempfile::tempdir_in(tmp.path()).expect("cwd");
    plant(&device, &cwd.path().join("rel-cache"));
    let run = run_child_in(
        cwd.path(),
        tmp.path(),
        &[("MLX_PTX_CACHE_DIR", OsStr::new("rel-cache"))],
    )
    .expect("device");
    run.assert_refused("planted", &["MLX_PTX_CACHE_DIR must be an absolute path"]);
}

#[test]
fn a_relative_mlxcel_cache_dir_is_refused() {
    let tmp = private_tempdir();
    let Some((_, device)) = reference(tmp.path()) else {
        return;
    };
    // Without and with the persistent default, which would otherwise create
    // the directories under the current directory.
    for persistent in [false, true] {
        let cwd = tempfile::tempdir_in(tmp.path()).expect("cwd");
        let mut envs = vec![("MLXCEL_CACHE_DIR", OsStr::new("rel-root"))];
        if persistent {
            envs.push((PERSISTENT_ENV, OsStr::new("1")));
        }
        let run = run_child_in(cwd.path(), tmp.path(), &envs).expect("device");
        eprintln!("{}", run.refusals().join("\n"));
        run.assert_refused(
            "relative MLXCEL_CACHE_DIR",
            &["MLXCEL_CACHE_DIR must be an absolute path", "rel-root"],
        );
        assert_cwd_untouched(cwd.path());
    }

    // An explicit MLX_PTX_CACHE_DIR means the root is not used, as for ROCm.
    let cwd = tempfile::tempdir_in(tmp.path()).expect("cwd");
    let cache = tmp.path().join("explicit");
    std::fs::create_dir(&cache).expect("cache");
    plant(&device, &cache);
    let run = run_child_in(
        cwd.path(),
        tmp.path(),
        &[
            ("MLXCEL_CACHE_DIR", OsStr::new("rel-root")),
            ("MLX_PTX_CACHE_DIR", cache.as_os_str()),
        ],
    )
    .expect("device");
    run.assert_hit("an unused relative root");
}
