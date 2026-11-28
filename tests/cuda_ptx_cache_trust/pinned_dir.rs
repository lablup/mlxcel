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

//! Housekeeping of a cache directory held by descriptor (issue #3503).
//!
//! Under `MLXCEL_CACHE_TRUSTED_ROOT` the device subdirectory is pinned by a
//! descriptor and the stale-temporary sweep lists and deletes through it,
//! a separate code path from the sweep by path that
//! `stale_publish_temporaries_are_swept` covers.

use super::*;

#[test]
fn stale_publish_temporaries_are_swept_below_trusted_root() {
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
    // A matching name that is a symlink to an old file, and one that is a
    // directory: neither is a regular file, so neither is removed.
    let target = tmp.path().join("outside-target");
    std::fs::write(&target, b"keep me").expect("symlink target");
    set_mtime(&target, stale);
    let stale_link = device.join(".tmp5_5.ptx");
    std::os::unix::fs::symlink(&target, &stale_link).expect("plant symlink");
    let stale_dir = device.join(".tmp6_6.ptx");
    make_dir(&stale_dir, 0o700);

    let Some(second) = run_child(&envs) else {
        return;
    };
    second.assert_success("sweeping run");
    assert!(
        second.refusals().is_empty() && !second.stderr.contains(JIT_NOTICE),
        "the sweeping run must still hit the cache:\n{}",
        second.stderr
    );
    assert!(
        !stale_ptx.exists() && !stale_txt.exists(),
        "temporaries older than an hour must be swept through the descriptor"
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
        is_symlink(&stale_link) && stale_dir.is_dir(),
        "the sweep removes regular files only"
    );
    assert_eq!(
        std::fs::read(&target).expect("target"),
        b"keep me",
        "the sweep must not follow a symlink"
    );
}
