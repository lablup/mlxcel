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

use super::super::anchored_publish::open_dir_path;
use super::*;
use std::fs;

#[test]
fn delete_contents_detects_nested_directory_swap_before_open() {
    let root = tempfile::tempdir().expect("root");
    let nested = root.path().join("nested");
    fs::create_dir(&nested).expect("nested");
    fs::write(nested.join("file"), b"original").expect("file");
    let parent = open_dir_path(root.path()).expect("open root");
    let root_path = root.path().to_path_buf();
    let mut swapped = false;

    let err = delete_contents_with_hook(&parent, move |event, name| {
        if !swapped
            && matches!(event, DeleteHookEvent::BeforeOpenDir)
            && name.to_bytes() == b"nested"
        {
            swapped = true;
            fs::rename(root_path.join("nested"), root_path.join("nested-old"))
                .expect("move original nested");
            fs::create_dir(root_path.join("nested")).expect("replacement nested");
            fs::write(root_path.join("nested/file"), b"replacement").expect("replacement file");
        }
    })
    .expect_err("nested identity swap must fail");

    assert!(
        err.to_string().contains("identity mismatch"),
        "unexpected error: {err}"
    );
    assert_eq!(
        fs::read(root.path().join("nested-old/file")).expect("original remains"),
        b"original"
    );
    assert_eq!(
        fs::read(root.path().join("nested/file")).expect("replacement remains"),
        b"replacement"
    );
}

#[test]
fn identity_checks_preserve_device_and_inode_without_narrowing() {
    let root = tempfile::tempdir().expect("root");
    fs::create_dir(root.path().join("child")).expect("child");
    let parent = open_dir_path(root.path()).expect("parent");
    let name = CString::new("child").expect("name");
    let held = open_child_dir(&parent, &name).expect("held");
    let mut expected = stat_child(&parent, &name).expect("stat");
    assert_eq!(stat_device_id(&expected), held.metadata().unwrap().dev());
    verify_file_matches_stat(&held, &expected).expect("same identity");
    verify_child_matches_file(&parent, &name, &held).expect("same child");

    expected.st_dev ^= 1;
    assert!(verify_file_matches_stat(&held, &expected).is_err());
    expected.st_dev ^= 1;
    expected.st_ino ^= 1;
    assert!(verify_file_matches_stat(&held, &expected).is_err());

    fs::rename(root.path().join("child"), root.path().join("original")).expect("retain original");
    fs::create_dir(root.path().join("child")).expect("replacement");
    assert!(verify_child_matches_file(&parent, &name, &held).is_err());
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn darwin_signed_device_id_preserves_metadata_representation() {
    let root = tempfile::tempdir().expect("root");
    let parent = open_dir_path(root.path()).expect("parent");
    let name = CString::new(".").expect("name");
    let mut stat = stat_child(&parent, &name).expect("stat");
    stat.st_dev = -1;
    assert_eq!(stat_device_id(&stat), u64::MAX);
    stat.st_dev = i32::MIN;
    assert_eq!(stat_device_id(&stat), u64::MAX - i32::MAX as u64);
}
