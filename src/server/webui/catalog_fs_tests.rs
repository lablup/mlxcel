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

use super::indexed_completeness;
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
