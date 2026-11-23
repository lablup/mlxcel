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

use std::os::unix::fs::PermissionsExt;

use super::*;

fn chmod(path: &Path, mode: u32) {
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).expect("chmod");
}

fn mode(path: &Path) -> u32 {
    std::fs::symlink_metadata(path).expect("metadata").mode() & 0o7777
}

#[test]
fn creates_the_chain_owner_only_and_accepts_it() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("mlxcel");
    let leaf = ensure_private_dir(&root, Path::new("xla/vmfb"), true, current_euid())
        .expect("a fresh chain is private");
    assert_eq!(leaf, root.join("xla").join("vmfb"));
    for dir in [&root, &root.join("xla"), &leaf] {
        assert_eq!(mode(dir), 0o700, "{}", dir.display());
    }
}

#[test]
fn refuses_a_relative_root_and_an_escaping_subpath() {
    let euid = current_euid();
    assert!(matches!(
        ensure_private_dir(Path::new("rel"), Path::new("xla"), true, euid),
        Err(PrivateDirError::RelativeRoot(_))
    ));
    let tmp = tempfile::tempdir().expect("tempdir");
    for sub in ["../x", "/abs"] {
        assert!(matches!(
            ensure_private_dir(tmp.path(), Path::new(sub), false, euid),
            Err(PrivateDirError::InvalidSubpath(_))
        ));
    }
}

#[test]
fn a_directory_owned_by_another_uid_is_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let other = current_euid().wrapping_add(1);
    assert!(matches!(
        verify_private_dir(tmp.path(), other),
        Err(PrivateDirError::ForeignOwner { .. })
    ));
    assert!(matches!(
        ensure_private_dir(tmp.path(), Path::new("xla"), false, other),
        Err(PrivateDirError::ForeignOwner { .. })
    ));
}

#[test]
fn group_or_world_writable_directories_are_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let euid = current_euid();
    for bits in [0o770, 0o707, 0o1777] {
        let dir = tmp.path().join(format!("d{bits:o}"));
        std::fs::create_dir(&dir).expect("dir");
        chmod(&dir, bits);
        assert!(
            matches!(
                verify_private_dir(&dir, euid),
                Err(PrivateDirError::Writable { .. })
            ),
            "mode {bits:o}"
        );
    }
}

#[test]
fn the_managed_chain_loses_group_write_but_an_explicit_root_does_not() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let euid = current_euid();
    let root = tmp.path().join("root");
    std::fs::create_dir_all(root.join("xla")).expect("dirs");
    chmod(&root, 0o775);
    chmod(&root.join("xla"), 0o775);
    assert!(matches!(
        ensure_private_dir(&root, Path::new("xla"), false, euid),
        Err(PrivateDirError::Writable { path, .. }) if path == root
    ));
    assert_eq!(mode(&root.join("xla")), 0o755, "managed dir repaired");
    ensure_private_dir(&root, Path::new("xla"), true, euid).expect("default root repaired");
    assert_eq!(mode(&root), 0o755);

    let world = tmp.path().join("world");
    std::fs::create_dir_all(world.join("xla")).expect("dirs");
    chmod(&world.join("xla"), 0o777);
    assert!(
        ensure_private_dir(&world, Path::new("xla"), true, euid).is_err(),
        "other write is never repaired"
    );
}

#[test]
fn symlinked_directories_are_refused_anywhere_in_the_chain() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let euid = current_euid();
    let target = tmp.path().join("elsewhere");
    std::fs::create_dir(&target).expect("dir");
    chmod(&target, 0o700);
    let root = tmp.path().join("root");
    std::fs::create_dir(&root).expect("root");
    chmod(&root, 0o700);
    std::os::unix::fs::symlink(&target, root.join("xla")).expect("symlink");
    assert!(matches!(
        ensure_private_dir(&root, Path::new("xla/vmfb"), false, euid),
        Err(PrivateDirError::Symlink(_))
    ));
    let link_root = tmp.path().join("link-root");
    std::os::unix::fs::symlink(&target, &link_root).expect("symlink");
    assert!(matches!(
        verify_private_dir(&link_root, euid),
        Err(PrivateDirError::Symlink(_))
    ));
}

#[test]
fn files_are_checked_on_the_descriptor() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let euid = current_euid();
    let missing = tmp.path().join("missing");
    assert!(open_private_file(&missing, euid).expect("absent").is_none());

    let file = tmp.path().join("artifact.vmfb");
    std::fs::write(&file, b"bytes").expect("write");
    chmod(&file, 0o664);
    assert!(matches!(
        open_private_file(&file, euid),
        Err(PrivateDirError::Writable { .. })
    ));
    seal_private_file(&file).expect("seal");
    assert_eq!(mode(&file), 0o600);
    assert!(open_private_file(&file, euid).expect("trusted").is_some());
    assert!(matches!(
        open_private_file(&file, euid.wrapping_add(1)),
        Err(PrivateDirError::ForeignOwner { .. })
    ));

    let link = tmp.path().join("link.vmfb");
    std::os::unix::fs::symlink(&file, &link).expect("symlink");
    assert!(matches!(
        open_private_file(&link, euid),
        Err(PrivateDirError::Symlink(_))
    ));
    assert!(seal_private_file(&link).is_err(), "sealing never follows");
    assert!(matches!(
        open_private_file(tmp.path(), euid),
        Err(PrivateDirError::NotARegularFile(_))
    ));
}

#[test]
fn fresh_directories_are_new_and_owner_only() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let first = create_fresh_private_dir(tmp.path(), "mlxcel-xla-").expect("fresh");
    let second = create_fresh_private_dir(tmp.path(), "mlxcel-xla-").expect("fresh");
    assert_ne!(first, second);
    for dir in [&first, &second] {
        assert_eq!(mode(dir), 0o700);
        verify_private_dir(dir, current_euid()).expect("private");
    }
}

/// Whoever can write a directory above the root can rename the verified root
/// away after the check, so an other-writable (non-sticky) ancestor is
/// refused, while the user's own group-writable `~/.cache` (umask 002 under
/// user private groups) is not (#3248).
#[test]
fn a_directory_above_the_root_that_others_can_write_is_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let euid = current_euid();
    let shared = tmp.path().join("shared");
    std::fs::create_dir(&shared).expect("dir");
    chmod(&shared, 0o777);
    assert!(matches!(
        ensure_private_dir(&shared.join("mlxcel"), Path::new("xla/vmfb"), true, euid),
        Err(PrivateDirError::Writable { path, .. }) if path.ends_with("shared")
    ));
    assert_eq!(mode(&shared), 0o777, "an ancestor is never modified");

    let own_group = tmp.path().join("own-group");
    std::fs::create_dir(&own_group).expect("dir");
    chmod(&own_group, 0o775);
    verify_trusted_ancestors_with(&own_group, euid, |_| true)
        .expect("a provably private own group may write an ancestor");
    assert_eq!(mode(&own_group), 0o775, "an ancestor is never modified");
}

/// The user's own group is trusted on an ancestor only when it is provably
/// private; otherwise the strict rule applies and the message names the fix.
#[test]
fn a_group_writable_ancestor_is_refused_unless_the_group_is_private() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let euid = current_euid();
    let dir = tmp.path().join("cache");
    std::fs::create_dir(&dir).expect("dir");
    chmod(&dir, 0o775);
    let error = verify_trusted_ancestors_with(&dir, euid, |_| false).unwrap_err();
    assert!(
        matches!(&error, PrivateDirError::GroupWritable { path, .. } if *path == dir),
        "{error}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!("chmod g-w {}", dir.display())),
        "{message}"
    );
    assert!(message.contains("MLXCEL_CACHE_DIR"), "{message}");

    // Other write is refused even when the group is private.
    chmod(&dir, 0o777);
    assert!(matches!(
        verify_trusted_ancestors_with(&dir, euid, |_| true),
        Err(PrivateDirError::Writable { .. })
    ));
    // The relaxation never applies to a uid other than this process's.
    chmod(&dir, 0o775);
    assert!(verify_trusted_ancestors(&dir, euid.wrapping_add(1)).is_err());
}

/// A symlink above the root is resolved once, and the returned leaf names the
/// real directories, so swapping that symlink later cannot redirect a load.
#[test]
fn the_leaf_is_spelled_through_the_canonical_parent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let euid = current_euid();
    let real = tmp.path().join("real");
    std::fs::create_dir(&real).expect("dir");
    chmod(&real, 0o700);
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(&real, &link).expect("symlink");
    let leaf =
        ensure_private_dir(&link.join("mlxcel"), Path::new("xla/vmfb"), true, euid).expect("leaf");
    let canonical = std::fs::canonicalize(&real).expect("canonical");
    assert_eq!(leaf, canonical.join("mlxcel").join("xla").join("vmfb"));
    assert!(matches!(
        ensure_private_dir(Path::new("/tmp/a/../b"), Path::new("xla"), true, euid),
        Err(PrivateDirError::InvalidSubpath(_))
    ));
}
