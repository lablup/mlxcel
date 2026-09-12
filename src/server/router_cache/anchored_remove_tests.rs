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

use super::*;
use std::fs;

#[test]
fn managed_removal_deletes_complete_snapshot_from_private_quarantine() {
    let root = tempfile::tempdir().expect("root");
    let snapshot = root.path().join("owner/model");
    fs::create_dir_all(&snapshot).expect("snapshot");
    fs::write(snapshot.join("config.json"), b"{}").expect("config");
    fs::write(snapshot.join("model.safetensors"), b"weights").expect("weights");

    let removed = remove_managed_snapshot(root.path(), "owner/model")
        .expect("remove")
        .expect("removed");

    assert_eq!(removed.path, snapshot);
    assert!(
        removed.size_bytes >= 9,
        "removed size should include regular files"
    );
    assert!(!root.path().join("owner/model").exists());
    assert!(root.path().join(QUARANTINE_DIR).is_dir());
    assert!(
        fs::read_dir(root.path().join(QUARANTINE_DIR))
            .expect("quarantine")
            .next()
            .is_none(),
        "private quarantine should be empty after successful delete"
    );
}

#[test]
fn managed_removal_unlinks_symlink_child_without_deleting_target() {
    let root = tempfile::tempdir().expect("root");
    let snapshot = root.path().join("owner/model");
    fs::create_dir_all(&snapshot).expect("snapshot");
    fs::write(snapshot.join("config.json"), b"{}").expect("config");
    let victim = root.path().join("external-victim");
    fs::write(&victim, b"keep").expect("victim");
    std::os::unix::fs::symlink(&victim, snapshot.join("link")).expect("symlink");

    remove_managed_snapshot(root.path(), "owner/model")
        .expect("remove")
        .expect("removed");

    assert_eq!(fs::read(&victim).expect("victim remains"), b"keep");
    assert!(!snapshot.exists());
}

#[test]
fn managed_removal_detects_source_swap_after_verify_and_leaves_quarantine() {
    let root = tempfile::tempdir().expect("root");
    let owner = root.path().join("owner");
    let snapshot = owner.join("model");
    fs::create_dir_all(&snapshot).expect("snapshot");
    fs::write(snapshot.join("config.json"), b"original").expect("config");
    let owner_for_hook = owner.clone();

    let err = remove_managed_snapshot_with_pre_rename_hook(root.path(), "owner/model", move || {
        fs::rename(
            owner_for_hook.join("model"),
            owner_for_hook.join("original"),
        )
        .expect("move original");
        fs::create_dir(owner_for_hook.join("model")).expect("replacement");
        fs::write(owner_for_hook.join("model/config.json"), b"replacement")
            .expect("replacement config");
    })
    .expect_err("identity mismatch after rename must fail");

    assert!(
        err.to_string().contains("identity changed"),
        "unexpected error: {err:#}"
    );
    assert_eq!(
        fs::read(owner.join("original/config.json")).expect("original remains"),
        b"original"
    );
    let quarantine = root.path().join(QUARANTINE_DIR);
    let quarantined: Vec<_> = fs::read_dir(&quarantine)
        .expect("quarantine")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect();
    assert_eq!(
        quarantined.len(),
        1,
        "wrong inode should be left quarantined"
    );
    assert_eq!(
        fs::read(quarantined[0].join("config.json")).expect("replacement quarantined"),
        b"replacement"
    );
}

#[test]
fn managed_removal_refuses_parent_swap_before_quarantine_rename() {
    let root = tempfile::tempdir().expect("root");
    let owner = root.path().join("owner");
    let snapshot = owner.join("model");
    fs::create_dir_all(&snapshot).expect("snapshot");
    fs::write(snapshot.join("config.json"), b"original").expect("config");
    let root_for_hook = root.path().to_path_buf();

    let err = remove_managed_snapshot_with_pre_rename_hook(root.path(), "owner/model", move || {
        fs::rename(root_for_hook.join("owner"), root_for_hook.join("owner-old"))
            .expect("move owner");
        fs::create_dir_all(root_for_hook.join("owner/model")).expect("replacement snapshot");
        fs::write(
            root_for_hook.join("owner/model/config.json"),
            b"replacement",
        )
        .expect("replacement config");
    })
    .expect_err("owner identity mismatch must fail before rename");

    assert!(
        err.to_string().contains("owner directory identity changed"),
        "unexpected error: {err:#}"
    );
    assert_eq!(
        fs::read(root.path().join("owner-old/model/config.json")).expect("old owner remains"),
        b"original"
    );
    assert_eq!(
        fs::read(root.path().join("owner/model/config.json")).expect("replacement remains"),
        b"replacement"
    );
}

#[test]
fn managed_removal_refuses_parent_swap_after_quarantine_rename() {
    let root = tempfile::tempdir().expect("root");
    let owner = root.path().join("owner");
    let snapshot = owner.join("model");
    fs::create_dir_all(&snapshot).expect("snapshot");
    fs::write(snapshot.join("config.json"), b"original").expect("config");
    let root_for_hook = root.path().to_path_buf();

    let err =
        remove_managed_snapshot_with_post_rename_hook(root.path(), "owner/model", move || {
            fs::rename(root_for_hook.join("owner"), root_for_hook.join("owner-old"))
                .expect("move owner");
            fs::create_dir(root_for_hook.join("owner")).expect("replacement owner");
        })
        .expect_err("owner identity mismatch after rename must fail");

    assert!(
        err.to_string().contains("owner directory identity changed"),
        "unexpected error: {err:#}"
    );
    let quarantined: Vec<_> = fs::read_dir(root.path().join(QUARANTINE_DIR))
        .expect("quarantine")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect();
    assert_eq!(quarantined.len(), 1, "snapshot should be left quarantined");
    assert_eq!(
        fs::read(quarantined[0].join("config.json")).expect("quarantined original"),
        b"original"
    );
}

#[test]
fn managed_removal_refuses_quarantine_child_swap_before_final_unlink() {
    let root = tempfile::tempdir().expect("root");
    let snapshot = root.path().join("owner/model");
    fs::create_dir_all(&snapshot).expect("snapshot");
    fs::write(snapshot.join("config.json"), b"original").expect("config");
    fs::write(snapshot.join("model.safetensors"), b"weights").expect("weights");
    let root_for_hook = root.path().to_path_buf();

    let err =
        remove_managed_snapshot_with_final_unlink_hook(root.path(), "owner/model", move || {
            let quarantine = root_for_hook.join(QUARANTINE_DIR);
            let entry = fs::read_dir(&quarantine)
                .expect("quarantine")
                .next()
                .expect("quarantine child")
                .expect("quarantine child");
            let child_name = entry.file_name();
            fs::rename(entry.path(), quarantine.join("saved-original"))
                .expect("move quarantined child");
            fs::create_dir(quarantine.join(&child_name)).expect("replacement quarantine child");
            fs::write(quarantine.join(&child_name).join("marker"), b"replacement")
                .expect("replacement marker");
        })
        .expect_err("quarantine child identity mismatch must fail");

    assert!(
        err.to_string().contains("identity changed"),
        "unexpected error: {err:#}"
    );
    let quarantine = root.path().join(QUARANTINE_DIR);
    assert!(quarantine.join("saved-original").is_dir());
    let replacement = fs::read_dir(&quarantine)
        .expect("quarantine")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .find(|path| path.join("marker").exists())
        .expect("replacement child remains");
    assert_eq!(
        fs::read(replacement.join("marker")).expect("replacement marker"),
        b"replacement"
    );
}
