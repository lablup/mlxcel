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

//! Issue #3248: an XLA cache location that is not private to the current user
//! is refused, the artifact is compiled elsewhere, and nothing planted there is
//! loaded. "Owned by another uid" is modeled by checking against a different
//! effective uid, since creating such a directory needs root.

use std::os::unix::fs::{MetadataExt, PermissionsExt};

use super::*;
use crate::aux_manifest::{
    AuxiliaryArtifactContract, auxiliary_manifest_path, qualify_auxiliary_artifact_for,
    write_auxiliary_manifest,
};
use crate::auxiliary::{AuxiliaryWeight, AuxiliaryWeightDType, AuxiliaryWeightStorage};

fn chmod(path: &Path, mode: u32) {
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).expect("chmod");
}

fn mode(path: &Path) -> u32 {
    std::fs::symlink_metadata(path).expect("metadata").mode() & 0o7777
}

fn private_tempdir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    chmod(dir.path(), 0o700);
    dir
}

fn weights() -> Vec<AuxiliaryWeight> {
    vec![AuxiliaryWeight {
        name: "weight".to_string(),
        storage: AuxiliaryWeightStorage::Float32(vec![1.0]),
        dtype: AuxiliaryWeightDType::Float32,
        shape: vec![1],
    }]
}

fn contract() -> AuxiliaryArtifactContract {
    AuxiliaryArtifactContract::new_legacy_unqualified("aux.main", "config=v1", "compiler=v1")
        .expect("contract")
}

#[test]
fn the_default_location_is_created_owner_only_and_reused() {
    let tmp = private_tempdir();
    let root = tmp.path().join("mlxcel");
    let euid = current_euid();
    let dir = resolve_cache_dir(Some(root.clone()), true, XlaCache::Vmfb, euid).expect("dir");
    assert_eq!(dir, root.join("xla").join("vmfb"));
    assert_eq!(mode(&dir), 0o700);
    assert_eq!(
        resolve_cache_dir(Some(root), true, XlaCache::Vmfb, euid).expect("dir"),
        dir,
        "the owning user keeps the same location"
    );
}

#[test]
fn a_directory_owned_by_another_uid_is_refused() {
    let tmp = private_tempdir();
    let root = tmp.path().join("mlxcel");
    std::fs::create_dir_all(root.join("xla").join("qmv4")).expect("pre-created");
    let other = current_euid().wrapping_add(1);
    let dir = resolve_cache_dir(Some(root.clone()), true, XlaCache::Qmv4, other).expect("dir");
    assert!(
        !dir.starts_with(&root),
        "{} must not be used",
        root.display()
    );
    verify_private_dir(&dir, current_euid()).expect("the fallback is private");
}

#[test]
fn group_or_world_writable_directories_are_refused() {
    let tmp = private_tempdir();
    let euid = current_euid();

    // Another user may already have planted files in a world-writable cache
    // directory, so it is never repaired.
    let root = tmp.path().join("world");
    let planted = root.join("xla").join("vision");
    std::fs::create_dir_all(&planted).expect("pre-created");
    chmod(&root, 0o700);
    chmod(&root.join("xla"), 0o700);
    chmod(&planted, 0o777);
    let dir = resolve_cache_dir(Some(root.clone()), true, XlaCache::Vision, euid).expect("dir");
    assert!(!dir.starts_with(&root));
    assert_eq!(
        mode(&planted) & 0o002,
        0o002,
        "other write is never repaired, so the directory stays refused"
    );
    assert!(
        std::fs::read_dir(&planted).expect("read").next().is_none(),
        "nothing is compiled into the refused directory"
    );

    // An explicit MLXCEL_CACHE_DIR is checked as found.
    let explicit = tmp.path().join("explicit");
    std::fs::create_dir(&explicit).expect("root");
    chmod(&explicit, 0o770);
    let dir =
        resolve_cache_dir(Some(explicit.clone()), false, XlaCache::Vision, euid).expect("dir");
    assert!(!dir.starts_with(&explicit));
}

#[test]
fn a_symlinked_cache_directory_is_refused() {
    let tmp = private_tempdir();
    let euid = current_euid();
    let attacker = tmp.path().join("attacker");
    std::fs::create_dir(&attacker).expect("dir");
    chmod(&attacker, 0o700);
    let root = tmp.path().join("mlxcel");
    std::fs::create_dir_all(root.join("xla")).expect("dirs");
    std::os::unix::fs::symlink(&attacker, root.join("xla").join("vmfb")).expect("symlink");
    let dir = resolve_cache_dir(Some(root.clone()), true, XlaCache::Vmfb, euid).expect("dir");
    assert!(!dir.starts_with(&root));
    assert!(!dir.starts_with(&attacker));
}

#[test]
fn a_root_below_a_directory_others_can_write_is_refused() {
    let tmp = private_tempdir();
    let shared = tmp.path().join("shared");
    std::fs::create_dir(&shared).expect("dir");
    chmod(&shared, 0o777);
    let root = shared.join("mlxcel");
    let dir =
        resolve_cache_dir(Some(root.clone()), false, XlaCache::Vmfb, current_euid()).expect("dir");
    assert!(!dir.starts_with(&shared), "{}", dir.display());
    verify_private_dir(&dir, current_euid()).expect("the fallback is private");
}

#[test]
fn a_relative_or_missing_root_uses_a_fresh_private_directory() {
    let euid = current_euid();
    for root in [Some(PathBuf::from("relative/cache")), None] {
        let dir = resolve_cache_dir(root, true, XlaCache::Vmfb, euid).expect("dir");
        assert!(dir.is_absolute());
        verify_private_dir(&dir, euid).expect("private");
    }
    assert!(
        !Path::new("relative").exists(),
        "nothing is created under the cwd"
    );
}

#[test]
fn a_symlinked_artifact_is_refused_and_rebuilt_elsewhere() {
    let tmp = private_tempdir();
    let cache = tmp.path().join("cache");
    std::fs::create_dir(&cache).expect("dir");
    chmod(&cache, 0o700);
    let foreign = tmp.path().join("foreign.vmfb");
    std::fs::write(&foreign, b"planted").expect("write");
    chmod(&foreign, 0o600);
    let vmfb = cache.join("decode.vmfb");
    std::os::unix::fs::symlink(&foreign, &vmfb).expect("symlink");

    let slot = artifact_slot(&vmfb, &[]).expect("slot");
    assert_ne!(slot, vmfb);
    assert!(
        !slot.exists(),
        "the fallback starts empty, so it is compiled"
    );
    assert_eq!(slot.file_name(), vmfb.file_name());
    assert_eq!(std::fs::read(&foreign).expect("read"), b"planted");
    assert!(
        std::fs::symlink_metadata(&vmfb)
            .expect("link")
            .file_type()
            .is_symlink(),
        "the refused entry is not touched"
    );
}

#[test]
fn a_group_writable_artifact_in_a_private_directory_is_refused() {
    let tmp = private_tempdir();
    let vmfb = tmp.path().join("prefill.vmfb");
    std::fs::write(&vmfb, b"planted").expect("write");
    chmod(&vmfb, 0o664);
    assert_ne!(artifact_slot(&vmfb, &[]).expect("slot"), vmfb);
    assert!(
        read_private(&vmfb).unwrap_err().contains("writable"),
        "the native GEMV read refuses it too"
    );
}

/// A planted VMFB whose manifest is valid for the contract (every manifest field
/// is derivable from public inputs) is not loaded from an untrusted directory:
/// the artifact is compiled again into a private one.
#[test]
fn a_planted_vmfb_with_a_valid_manifest_is_not_loaded() {
    let contract = contract();
    let resident = weights();
    for case in ["world-writable", "other-uid"] {
        let tmp = private_tempdir();
        let cache = tmp.path().join("cache");
        std::fs::create_dir(&cache).expect("dir");
        let planted = cache.join("vision-256.vmfb");
        std::fs::write(&planted, b"attacker-vmfb").expect("plant");
        write_auxiliary_manifest(&planted, &contract, &resident).expect("valid manifest");
        let euid = match case {
            "world-writable" => {
                chmod(&cache, 0o777);
                current_euid()
            }
            _ => {
                chmod(&cache, 0o700);
                current_euid().wrapping_add(1)
            }
        };
        let mut compiles = 0usize;
        let loaded = qualify_auxiliary_artifact_for(
            &planted,
            &contract,
            &resident,
            |output| {
                compiles += 1;
                std::fs::write(output, b"compiled-vmfb").map_err(|error| error.to_string())
            },
            euid,
        )
        .expect("qualify");
        assert_eq!(compiles, 1, "{case}: the artifact is recompiled");
        assert!(!loaded.starts_with(&cache), "{case}: {}", loaded.display());
        assert_eq!(std::fs::read(&loaded).expect("read"), b"compiled-vmfb");
        assert_eq!(
            std::fs::read(&planted).expect("read"),
            b"attacker-vmfb",
            "{case}: the planted file is left alone"
        );
        chmod(&cache, 0o700);
    }
}

#[test]
fn the_owner_reuses_a_qualified_artifact() {
    let tmp = private_tempdir();
    let vmfb = tmp.path().join("vision-256.vmfb");
    let contract = contract();
    let resident = weights();
    let mut compiles = 0usize;
    let mut compile = |output: &Path| {
        compiles += 1;
        std::fs::write(output, b"compiled").map_err(|error| error.to_string())
    };
    let euid = current_euid();
    let first = qualify_auxiliary_artifact_for(&vmfb, &contract, &resident, &mut compile, euid)
        .expect("first");
    let second = qualify_auxiliary_artifact_for(&vmfb, &contract, &resident, &mut compile, euid)
        .expect("second");
    assert_eq!(first, vmfb);
    assert_eq!(second, vmfb, "the second load reuses the artifact");
    assert_eq!(compiles, 1);
    assert_eq!(mode(&vmfb), 0o600, "published owner-only under any umask");
    assert_eq!(mode(&auxiliary_manifest_path(&vmfb)), 0o600);
}

#[test]
fn private_writes_are_atomic_and_owner_only() {
    let tmp = private_tempdir();
    let path = tmp.path().join("spec.mlir");
    write_private_atomic(&path, b"one").expect("write");
    assert_eq!(mode(&path), 0o600);
    // A symlink planted at the name is replaced, not written through.
    let target = tmp.path().join("target");
    std::fs::write(&target, b"keep").expect("write");
    std::fs::remove_file(&path).expect("remove");
    std::os::unix::fs::symlink(&target, &path).expect("symlink");
    write_private_atomic(&path, b"two").expect("write");
    assert_eq!(std::fs::read(&target).expect("read"), b"keep");
    assert_eq!(std::fs::read(&path).expect("read"), b"two");
    assert!(
        !std::fs::symlink_metadata(&path)
            .expect("meta")
            .file_type()
            .is_symlink()
    );
    assert_eq!(read_private(&path).expect("read"), Some(b"two".to_vec()));
}
