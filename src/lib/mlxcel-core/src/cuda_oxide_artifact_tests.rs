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

//! Manifest, schema and digest validation of the cuda-oxide fused RMSNorm
//! artifact (#2833). No GPU: these run on any opted-in build.

use std::path::Path;

use serde_json::{Value, json};

use crate::cuda_oxide_artifact::{
    ARG_COUNT, ArgRole, ArtifactError, ENTRIES, EntryDocuments, Lane, PTX_FILE, TARGET_ARCH,
    module_name, read_dir, validate,
};

/// The committed artifact, read from the spike package in this checkout.
pub(crate) fn committed() -> (Vec<u8>, Vec<EntryDocuments>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../spike/cuda-oxide-fused-norm");
    read_dir(&dir).expect("the committed artifact is readable")
}

/// Write an artifact in the directory layout [`read_dir`] expects.
pub(crate) fn write_layout(dir: &Path, ptx: &[u8], documents: &[EntryDocuments]) {
    std::fs::create_dir_all(dir.join("artifacts")).expect("mkdir");
    std::fs::create_dir_all(dir.join("abi")).expect("mkdir");
    std::fs::write(dir.join("artifacts").join(PTX_FILE), ptx).expect("write ptx");
    for docs in documents {
        std::fs::write(
            dir.join("artifacts")
                .join(format!("{}.sm_121.manifest.json", docs.entry)),
            &docs.manifest,
        )
        .expect("write manifest");
        std::fs::write(
            dir.join("abi").join(format!("{}.schema.json", docs.entry)),
            &docs.schema,
        )
        .expect("write schema");
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Apply `edit` to the manifest of entry `index`.
fn edit_manifest(documents: &mut [EntryDocuments], index: usize, edit: impl FnOnce(&mut Value)) {
    let mut value: Value = serde_json::from_str(&documents[index].manifest).expect("manifest");
    edit(&mut value);
    documents[index].manifest = value.to_string();
}

fn edit_schema(documents: &mut [EntryDocuments], index: usize, edit: impl FnOnce(&mut Value)) {
    let mut value: Value = serde_json::from_str(&documents[index].schema).expect("schema");
    edit(&mut value);
    documents[index].schema = value.to_string();
}

/// Re-pin every manifest to `ptx`, the way a rebuilt artifact's manifests
/// would be regenerated.
pub(crate) fn repin(documents: &mut [EntryDocuments], ptx: &[u8]) {
    let digest = sha256_hex(ptx);
    for index in 0..documents.len() {
        edit_manifest(documents, index, |m| {
            m["artifacts"]["ptx"]["sha256"] = json!(digest);
            m["artifacts"]["ptx"]["bytes"] = json!(ptx.len());
            m["container"]["payload_sha256"] = json!(digest);
            m["container"]["payload_bytes"] = json!(ptx.len());
        });
    }
}

fn manifest_error(result: Result<impl std::fmt::Debug, ArtifactError>, needle: &str) {
    match result {
        Err(ArtifactError::Manifest { detail, .. }) => {
            assert!(detail.contains(needle), "{detail:?} lacks {needle:?}");
        }
        other => panic!("expected a manifest refusal mentioning {needle:?}, got {other:?}"),
    }
}

#[test]
fn the_committed_artifact_validates_with_the_abi_v1_layout() {
    let (ptx, documents) = committed();
    let artifact = validate(&ptx, &documents).expect("valid");

    assert_eq!(artifact.entries.len(), ENTRIES.len());
    let pairs: Vec<(Lane, Lane)> = artifact
        .entries
        .iter()
        .map(|e| (e.activation, e.weight))
        .collect();
    assert_eq!(
        pairs,
        [
            (Lane::Bf16, Lane::Bf16),
            (Lane::F16, Lane::F16),
            (Lane::F32, Lane::F32),
            (Lane::F16, Lane::F32),
            (Lane::Bf16, Lane::F32),
        ]
    );
    // ABI version 1: the roles are numbered in parameter order.
    for entry in &artifact.entries {
        let order: Vec<u8> = entry.roles.iter().map(|role| *role as u8).collect();
        assert_eq!(
            order,
            (0..ARG_COUNT as u8).collect::<Vec<_>>(),
            "{}",
            entry.symbol
        );
    }
    assert_eq!(artifact.sha256, sha256_hex(&ptx));
    assert_eq!(artifact.compiler_revision.len(), 40);
}

#[test]
fn the_module_name_carries_the_whole_cache_identity() {
    let (ptx, documents) = committed();
    let artifact = validate(&ptx, &documents).expect("valid");
    let name = &artifact.module_name;
    assert!(name.contains("abi1"), "{name}");
    assert!(name.contains(TARGET_ARCH), "{name}");
    assert!(name.contains(&artifact.sha256[..32]), "{name}");
    assert!(name.contains(&artifact.compiler_revision[..12]), "{name}");
    // One short path component in the PTX disk cache.
    assert!(name.len() <= 245 && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'));

    assert_ne!(
        module_name(&"a".repeat(64), &artifact.compiler_revision),
        module_name(&"b".repeat(64), &artifact.compiler_revision)
    );
    assert_ne!(
        module_name(&artifact.sha256, &"1".repeat(40)),
        module_name(&artifact.sha256, &"2".repeat(40))
    );
}

#[test]
fn a_rebuilt_artifact_gets_a_different_module_name() {
    let (mut ptx, mut documents) = committed();
    let original = validate(&ptx, &documents).expect("valid");

    ptx.extend_from_slice(b"\n// rebuilt\n");
    repin(&mut documents, &ptx);
    let rebuilt = validate(&ptx, &documents).expect("a consistently re-pinned artifact is valid");
    assert_ne!(rebuilt.sha256, original.sha256);
    assert_ne!(
        rebuilt.module_name, original.module_name,
        "a changed payload must not share a PTX cache entry with the old one"
    );
}

#[test]
fn a_payload_its_manifests_do_not_pin_is_refused() {
    let (mut ptx, documents) = committed();
    ptx.extend_from_slice(b"\n// tampered\n");
    assert!(matches!(
        validate(&ptx, &documents),
        Err(ArtifactError::Digest { .. })
    ));
}

#[test]
fn a_manifest_pinning_another_digest_is_refused() {
    let (ptx, mut documents) = committed();
    edit_manifest(&mut documents, 2, |m| {
        m["artifacts"]["ptx"]["sha256"] = json!("0".repeat(64));
    });
    assert!(matches!(
        validate(&ptx, &documents),
        Err(ArtifactError::Digest { .. })
    ));

    let (ptx, mut documents) = committed();
    edit_manifest(&mut documents, 4, |m| {
        m["container"]["payload_sha256"] = json!("0".repeat(64));
    });
    assert!(matches!(
        validate(&ptx, &documents),
        Err(ArtifactError::Digest { .. })
    ));
}

#[test]
fn another_abi_version_target_format_or_entry_is_refused() {
    let (ptx, documents) = committed();
    let with = |index: usize, edit: &dyn Fn(&mut Value)| {
        let mut documents = documents.clone();
        edit_manifest(&mut documents, index, edit);
        validate(&ptx, &documents)
    };
    manifest_error(with(0, &|m| m["abi_version"] = json!(2)), "ABI version 2");
    manifest_error(
        with(1, &|m| m["target"]["device_arch"] = json!("sm_90")),
        "sm_90",
    );
    manifest_error(
        with(1, &|m| m["container"]["target"] = json!("sm_100")),
        "sm_100",
    );
    manifest_error(
        with(2, &|m| m["manifest"] = json!("something/v9")),
        "something/v9",
    );
    manifest_error(
        with(3, &|m| m["entry_symbol"] = json!("fused_add_rms_norm_f32")),
        "describes entry",
    );
    manifest_error(
        with(3, &|m| m["container"]["payload_kind"] = json!("cubin")),
        "not PTX",
    );
    manifest_error(
        with(4, &|m| m["launch"]["block_dim_rule"] = json!("256")),
        "launch rule",
    );
    manifest_error(
        with(4, &|m| m["launch"]["shared_mem_bytes"] = json!(4096)),
        "launch rule",
    );
    // The rule of the artifact before #3510, which summed in another order.
    manifest_error(
        with(4, &|m| {
            m["launch"]["block_dim_rule"] = json!("fused_norm_threads(dim)")
        }),
        "launch rule",
    );
    manifest_error(
        with(0, &|m| m["source"]["revision"] = json!("main")),
        "compiler revision",
    );
}

#[test]
fn a_manifest_that_disagrees_with_its_schema_is_refused() {
    let (ptx, mut documents) = committed();
    // The manifest alone reorders two parameters: the hand-written ABI claim
    // no longer describes what the manifest says was emitted.
    edit_manifest(&mut documents, 0, |m| {
        let args = m["arguments"].as_array_mut().expect("arguments");
        args.swap(6, 7);
        args[6]["ordinal"] = json!(6);
        args[7]["ordinal"] = json!(7);
    });
    assert!(matches!(
        validate(&ptx, &documents),
        Err(ArtifactError::Schema { .. })
    ));

    let (ptx, mut documents) = committed();
    edit_schema(&mut documents, 1, |s| s["abi_version"] = json!(3));
    assert!(matches!(
        validate(&ptx, &documents),
        Err(ArtifactError::Schema { .. })
    ));
}

#[test]
fn the_packing_order_is_read_from_the_manifest_not_assumed() {
    let (ptx, mut documents) = committed();
    // Manifest and schema agree on a different order: eps and weight_bias
    // trade places. The artifact is still valid, and the roles follow it.
    let swap = |value: &mut Value| {
        let args = value["arguments"].as_array_mut().expect("arguments");
        args.swap(6, 7);
        args[6]["ordinal"] = json!(6);
        args[7]["ordinal"] = json!(7);
    };
    edit_manifest(&mut documents, 2, swap);
    edit_schema(&mut documents, 2, swap);
    let artifact = validate(&ptx, &documents).expect("a consistent reordering is valid");
    let entry = &artifact.entries[2];
    assert_eq!(entry.roles[6], ArgRole::WeightBias);
    assert_eq!(entry.roles[7], ArgRole::Eps);
    assert_eq!(artifact.entries[0].roles[6], ArgRole::Eps);
}

#[test]
fn an_argument_the_adapter_would_pack_wrongly_is_refused() {
    let (ptx, documents) = committed();
    let with = |edit: &dyn Fn(&mut Value)| {
        let mut documents = documents.clone();
        // Applied to manifest and schema alike, so the refusal comes from the
        // ABI check and not from the two documents disagreeing.
        edit_manifest(&mut documents, 0, edit);
        edit_schema(&mut documents, 0, edit);
        validate(&ptx, &documents)
    };
    manifest_error(
        with(&|v| {
            v["arguments"][8]["cuda_param_type"] = json!("u64");
            v["arguments"][8]["width_bits"] = json!(64);
        }),
        "wrong scalar type",
    );
    manifest_error(
        with(&|v| v["arguments"][1]["cuda_param_type"] = json!("u32")),
        "64-bit element count",
    );
    manifest_error(
        with(&|v| v["arguments"][4]["pointee"] = json!("f32")),
        "another element type",
    );
    manifest_error(
        with(&|v| v["arguments"][9]["access"] = json!("read")),
        "wrong access",
    );
    manifest_error(
        with(&|v| v["arguments"][0]["length_argument"] = json!(3)),
        "its own length",
    );
    manifest_error(
        with(&|v| v["arguments"][12]["name"] = json!("stride")),
        "unknown parameter",
    );
    manifest_error(
        with(&|v| v["arguments"][3]["name"] = json!("x_len")),
        "appears twice",
    );
    manifest_error(
        with(&|v| {
            v["arguments"].as_array_mut().expect("arguments").pop();
        }),
        "12 kernel parameters",
    );
}

#[test]
fn an_unsupported_dtype_or_a_duplicate_pair_is_refused() {
    let (ptx, mut documents) = committed();
    edit_schema(&mut documents, 0, |s| s["activation_dtype"] = json!("f64"));
    assert!(matches!(
        validate(&ptx, &documents),
        Err(ArtifactError::Schema { .. })
    ));

    // The f16 entry claims the bf16 pair the first entry already serves. Its
    // 16-bit pointees are unchanged, so only the duplicate check can catch it.
    let (ptx, mut documents) = committed();
    edit_schema(&mut documents, 1, |s| {
        s["activation_dtype"] = json!("bf16");
        s["weight_dtype"] = json!("bf16");
    });
    manifest_error(validate(&ptx, &documents), "already serves");
}

/// The block width depends on the vector width, so an entry compiled for
/// another one is refused even when its manifest and schema agree. Entry 4 is
/// bf16 under an f32 weight, which computes in bf16 (8 per vector) under
/// mlxcel's promotion table, not in f32 (4).
#[test]
fn an_entry_compiled_for_another_vector_width_is_refused() {
    let (ptx, documents) = committed();
    let artifact = validate(&ptx, &documents).expect("valid");
    for entry in &artifact.entries {
        let sixteen = entry.activation != Lane::F32 && entry.weight != Lane::F32
            || entry.activation == Lane::Bf16;
        assert_eq!(
            entry.activation.n_reads_under(entry.weight),
            if sixteen { 8 } else { 4 },
            "{}",
            entry.symbol
        );
    }
    assert_eq!(Lane::F32.n_reads_under(Lane::Bf16), 8);
    assert_eq!(Lane::F32.n_reads_under(Lane::F16), 4);
    assert_eq!(Lane::F16.n_reads_under(Lane::Bf16), 4);

    let mut documents = documents;
    edit_manifest(&mut documents, 4, |m| m["launch"]["n_reads"] = json!(4));
    edit_schema(&mut documents, 4, |s| s["launch"]["n_reads"] = json!(4));
    manifest_error(validate(&ptx, &documents), "n_reads 4");
}

#[test]
fn a_missing_file_or_entry_is_refused() {
    let (ptx, documents) = committed();
    manifest_error(validate(&ptx, &documents[..4]), "no manifest was supplied");

    let dir = tempfile::tempdir().expect("tempdir");
    write_layout(dir.path(), &ptx, &documents);
    assert!(validate_dir(dir.path()).is_ok());

    for relative in [
        format!("artifacts/{PTX_FILE}"),
        format!("artifacts/{}.sm_121.manifest.json", ENTRIES[1]),
        format!("abi/{}.schema.json", ENTRIES[3]),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        write_layout(dir.path(), &ptx, &documents);
        std::fs::remove_file(dir.path().join(&relative)).expect("remove");
        match validate_dir(dir.path()) {
            Err(ArtifactError::Missing { path, .. }) => {
                assert!(path.ends_with(&relative), "{path} is not {relative}");
            }
            other => panic!("expected a missing-file refusal for {relative}, got {other:?}"),
        }
    }
}

#[test]
fn malformed_json_and_a_non_ptx_payload_are_refused() {
    let (ptx, mut documents) = committed();
    documents[0].manifest.truncate(40);
    assert!(matches!(
        validate(&ptx, &documents),
        Err(ArtifactError::Json { .. })
    ));

    // A payload with the right digest and size that is not the PTX module:
    // the manifests pin it, but it has no such target or entries.
    let (_, mut documents) = committed();
    let payload = b"OXIDEART not a ptx module".to_vec();
    repin(&mut documents, &payload);
    manifest_error(validate(&payload, &documents), "the PTX has no");
}

fn validate_dir(dir: &Path) -> Result<(), ArtifactError> {
    let (ptx, documents) = read_dir(dir)?;
    validate(&ptx, &documents).map(|_| ())
}
