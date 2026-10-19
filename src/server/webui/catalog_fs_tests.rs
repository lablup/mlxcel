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

use super::{content_fingerprint, indexed_completeness};
use std::fs;
use std::path::Path;

fn write_index(dir: &Path, shards: &[&str]) {
    let map: serde_json::Map<String, serde_json::Value> = shards
        .iter()
        .enumerate()
        .map(|(i, name)| (format!("layer.{i}.weight"), serde_json::json!(name)))
        .collect();
    fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap(),
    )
    .unwrap();
}

/// A repackaged mlx-community quant ships an index naming shards that were never
/// written while the directory holds one differently-named `*.safetensors`. The
/// loader globs that file and runs the model, so the catalog must not report the
/// checkpoint unusable. Nine of two hundred entries on the M1 Ultra store were
/// blocked this way while `mlxcel generate` ran them (#1848).
#[test]
fn repackaged_quant_with_a_stale_shard_index_is_complete() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_index(
        dir,
        &[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    );
    fs::write(dir.join("model.safetensors"), b"quant").unwrap();
    let verdict = indexed_completeness(dir, &dir.join("model.safetensors.index.json"));
    assert!(verdict.ok, "unexpected reason: {}", verdict.reason);
    assert!(verdict.reason.is_empty());
}

/// The interrupted-download case must still be reported: every on-disk shard is
/// one the index names, so there is no unindexed weight file to fall back to.
#[test]
fn interrupted_download_missing_an_indexed_shard_is_incomplete() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_index(
        dir,
        &[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    );
    fs::write(dir.join("model-00001-of-00002.safetensors"), b"a").unwrap();
    let verdict = indexed_completeness(dir, &dir.join("model.safetensors.index.json"));
    assert!(!verdict.ok);
    assert_eq!(
        verdict.reason,
        "required shard 'model-00002-of-00002.safetensors' is missing or empty"
    );
}

/// A zero-byte shard is not a present shard, and a zero-byte unindexed file is
/// not a fallback either.
#[test]
fn zero_byte_weight_files_do_not_satisfy_either_path() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_index(
        dir,
        &[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    );
    fs::write(dir.join("model-00001-of-00002.safetensors"), b"a").unwrap();
    fs::write(dir.join("model-00002-of-00002.safetensors"), b"").unwrap();
    fs::write(dir.join("model.safetensors"), b"").unwrap();
    let verdict = indexed_completeness(dir, &dir.join("model.safetensors.index.json"));
    assert!(!verdict.ok);
    assert_eq!(
        verdict.reason,
        "required shard 'model-00002-of-00002.safetensors' is missing or empty"
    );
}

const OPTIQ_SIDECAR: &str = "optiq/optiq_vision.safetensors";

/// An OptiQ conversion: the index maps the vision tensors to the sidecar that
/// `config.json` declares under `optiq/`, which the VLM loader merges.
fn write_optiq_layout(dir: &Path, declared: bool) {
    let config = if declared {
        format!(r#"{{"optiq_vision": {{"sidecar": "{OPTIQ_SIDECAR}"}}}}"#)
    } else {
        "{}".to_string()
    };
    fs::write(dir.join("config.json"), config).unwrap();
    write_index(dir, &["model.safetensors", OPTIQ_SIDECAR]);
    fs::write(dir.join("model.safetensors"), b"text").unwrap();
    fs::create_dir_all(dir.join("optiq")).unwrap();
    fs::write(dir.join(OPTIQ_SIDECAR), b"vision").unwrap();
}

/// The declared sidecar is not an unsafe shard path: the checkpoint is
/// complete when it is present and reports it by name when it is absent
/// (#2451, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`).
#[test]
fn optiq_layout_with_its_declared_sidecar_is_complete() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_optiq_layout(dir, true);
    let index = dir.join("model.safetensors.index.json");
    let verdict = indexed_completeness(dir, &index);
    assert!(verdict.ok, "unexpected reason: {}", verdict.reason);
    assert!(verdict.reason.is_empty());

    fs::remove_file(dir.join(OPTIQ_SIDECAR)).unwrap();
    let verdict = indexed_completeness(dir, &index);
    assert!(!verdict.ok);
    assert_eq!(
        verdict.reason,
        format!("required shard '{OPTIQ_SIDECAR}' is missing or empty")
    );
}

/// Without the declaration the subdirectory entry is still an unsafe path.
#[test]
fn undeclared_subdirectory_shard_is_still_unsafe() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_optiq_layout(dir, false);
    let index = dir.join("model.safetensors.index.json");
    let verdict = indexed_completeness(dir, &index);
    assert!(!verdict.ok);
    assert_eq!(
        verdict.reason,
        "safetensors index contains an unsafe shard path"
    );

    // A declared traversal path is not a sidecar either.
    fs::write(
        dir.join("config.json"),
        r#"{"optiq_vision": {"sidecar": "../outside.safetensors"}}"#,
    )
    .unwrap();
    write_index(dir, &["model.safetensors", "../outside.safetensors"]);
    let verdict = indexed_completeness(dir, &index);
    assert!(!verdict.ok);
    assert_eq!(
        verdict.reason,
        "safetensors index contains an unsafe shard path"
    );
}

/// The content fingerprint covers the declared sidecar, so replacing it
/// invalidates the catalog entry instead of hashing an "unsafe-path" fact.
#[test]
fn content_fingerprint_tracks_the_declared_optiq_sidecar() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_optiq_layout(dir, true);
    let before = content_fingerprint(dir).expect("fingerprint");
    fs::write(dir.join(OPTIQ_SIDECAR), b"re-downloaded vision tower").unwrap();
    let after = content_fingerprint(dir).expect("fingerprint");
    assert_ne!(before, after);
}
