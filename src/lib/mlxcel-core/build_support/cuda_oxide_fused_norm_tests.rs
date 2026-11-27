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

//! Unit tests for the cuda-oxide fused RMSNorm build gate (issue #2833). Run
//! by `cargo test -p mlxcel-mlx-pin`, which needs no MLX build.

use super::*;

/// `spike/cuda-oxide-fused-norm` in this checkout. `mlxcel-mlx-pin` sits at
/// `src/lib/mlxcel-mlx-pin`, three levels below the repository root.
fn committed_spike_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../spike/cuda-oxide-fused-norm")
}

/// A copy of the committed artifacts, manifests and schemas that a test may
/// damage.
fn scratch_spike_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = committed_spike_dir();
    for sub in ["artifacts", "abi"] {
        fs::create_dir_all(dir.path().join(sub)).expect("mkdir");
    }
    let mut files = vec![("artifacts", PTX_FILE.to_string())];
    for entry in ENTRIES {
        files.push(("artifacts", manifest_file(entry)));
        files.push(("abi", schema_file(entry)));
    }
    for (sub, name) in files {
        fs::copy(
            source.join(sub).join(&name),
            dir.path().join(sub).join(&name),
        )
        .expect("copy");
    }
    dir
}

#[test]
fn sha256_matches_the_fips_180_4_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // Two blocks: the padding does not fit after a 56-byte message.
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    assert_eq!(
        sha256_hex(&vec![b'a'; 1_000_000]),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn the_switch_is_off_unless_spelled_on() {
    assert_eq!(requested(None), Ok(false));
    for off in ["", "0", "false", "OFF", " no "] {
        assert_eq!(requested(Some(off)), Ok(false), "{off:?}");
    }
    for on in ["1", "true", "On", " yes "] {
        assert_eq!(requested(Some(on)), Ok(true), "{on:?}");
    }
    let error = requested(Some("enable")).expect_err("a typo must not build silently");
    assert!(error.contains(BUILD_ENV), "{error}");
}

#[test]
fn only_a_linux_cuda_build_is_supported() {
    assert_eq!(check_supported("linux", true), Ok(()));
    for (os, cuda) in [("linux", false), ("windows", true), ("macos", false)] {
        let error = check_supported(os, cuda).expect_err("unsupported build");
        assert!(error.contains(BUILD_ENV), "{error}");
    }
}

#[test]
fn string_fields_are_read_after_their_anchors() {
    let text = r#"{"container": {"payload_kind": "ptx", "payload_sha256": "aa"},
        "artifacts": {"ptx": {"path": "p", "sha256": "bb"}, "cubin": {"sha256": "cc"}}}"#;
    assert_eq!(
        string_field_after(text, &["container"], "payload_sha256"),
        Some("aa")
    );
    assert_eq!(
        string_field_after(text, &["artifacts", "ptx"], "sha256"),
        Some("bb")
    );
    assert_eq!(
        string_field_after(text, &["artifacts", "cubin"], "sha256"),
        Some("cc")
    );
    assert_eq!(string_field_after(text, &["missing"], "sha256"), None);
    assert_eq!(string_field_after(text, &["artifacts"], "absent"), None);
}

#[test]
fn the_committed_artifacts_verify_and_stage() {
    let stage = tempfile::tempdir().expect("tempdir");
    let sources = verify_and_stage(&committed_spike_dir(), stage.path()).expect("verify");
    assert_eq!(sources.len(), 1 + 2 * ENTRIES.len());

    let staged_ptx = fs::read(stage.path().join(PTX_FILE)).expect("staged ptx");
    let committed = fs::read(committed_spike_dir().join("artifacts").join(PTX_FILE)).expect("ptx");
    assert_eq!(staged_ptx, committed);
    for entry in ENTRIES {
        assert!(stage.path().join(manifest_file(entry)).is_file());
        assert!(stage.path().join(schema_file(entry)).is_file());
    }
}

#[test]
fn a_ptx_that_does_not_match_its_manifests_is_not_staged() {
    let spike = scratch_spike_dir();
    let ptx_path = spike.path().join("artifacts").join(PTX_FILE);
    let mut ptx = fs::read(&ptx_path).expect("ptx");
    ptx.extend_from_slice(b"\n// tampered\n");
    fs::write(&ptx_path, ptx).expect("write");

    let stage = tempfile::tempdir().expect("tempdir");
    let error = verify_and_stage(spike.path(), stage.path()).expect_err("digest mismatch");
    assert!(error.contains("hashes to"), "{error}");
    assert!(
        !stage.path().join(PTX_FILE).exists(),
        "a rejected artifact must not be staged"
    );
}

#[test]
fn a_manifest_pinning_another_digest_is_refused() {
    let spike = scratch_spike_dir();
    let name = manifest_file(ENTRIES[3]);
    let path = spike.path().join("artifacts").join(&name);
    let text = fs::read_to_string(&path).expect("manifest");
    let pinned = string_field_after(&text, &["artifacts", "ptx"], "sha256")
        .expect("digest")
        .to_string();
    let wrong = "0".repeat(pinned.len());
    fs::write(&path, text.replace(&pinned, &wrong)).expect("write");

    let stage = tempfile::tempdir().expect("tempdir");
    let error = verify_and_stage(spike.path(), stage.path()).expect_err("wrong manifest");
    assert!(error.contains(&name), "{error}");
    assert!(!stage.path().join(PTX_FILE).exists());
}

#[test]
fn a_missing_artifact_manifest_or_schema_is_refused() {
    for (sub, name) in [
        ("artifacts", PTX_FILE.to_string()),
        ("artifacts", manifest_file(ENTRIES[0])),
        ("abi", schema_file(ENTRIES[4])),
    ] {
        let spike = scratch_spike_dir();
        fs::remove_file(spike.path().join(sub).join(&name)).expect("remove");
        let stage = tempfile::tempdir().expect("tempdir");
        let error = verify_and_stage(spike.path(), stage.path()).expect_err("missing file");
        assert!(error.contains(&name), "{error}");
        assert!(!stage.path().join(PTX_FILE).exists());
    }
}
