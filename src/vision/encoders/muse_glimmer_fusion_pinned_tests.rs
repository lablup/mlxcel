// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Contract check against the pinned Muse Glimmer checkpoint, plus synthetic
//! coverage for the availability pre-check that guards it.
//!
//! The checkpoint is roughly 60 GB and is absent on most machines, so the check
//! skips when it cannot read what it needs. That skip is deliberately narrow:
//! only availability and readability problems (an absent file, an unreadable
//! file, JSON that does not parse, a truncated safetensors header) may skip. A
//! contract violation always fails, because turning a mismatched weight-root
//! set, a config `validate()` rejects, or a wrong recorded shape into a silent
//! skip would permanently disable the contract this module exists to enforce.

use crate::models::muse_glimmer::MuseGlimmerConfig;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Directory holding the pinned Muse Glimmer checkpoint the published contract
/// is validated against. Absent on machines that never downloaded it.
const PINNED_MODEL_DIR: &str = "models/mlx/muse-glimmer-30b";

/// The exact post-tower weight roots the published contract pins. Used both to
/// enumerate the shards the check will open and as the expected value of the
/// weight-root assertion itself.
const PINNED_POST_TOWER_WEIGHT_KEYS: [&str; 3] = [
    "model.vision_adapter.fc1.weight",
    "model.vision_adapter.fc2.weight",
    "model.vision_projection.weight",
];

/// Checkpoint inputs the post-tower contract assertions read, parsed once so
/// the availability pre-check and the assertion body share a single read of
/// `model.safetensors.index.json` rather than opening it twice.
#[derive(Debug)]
struct PinnedCheckpoint {
    config: MuseGlimmerConfig,
    weight_map: serde_json::Map<String, Value>,
}

/// Read every file the post-tower contract assertions need.
///
/// `Err` is returned for availability and readability problems only: a file
/// that is absent, a file that cannot be read, or a file whose JSON does not
/// parse. The message always names the offending path, so a partially
/// materialized checkpoint (an interrupted `hf download` or `mlxcel download`)
/// skips with a reason instead of panicking on an opaque `unwrap`.
///
/// Contract violations are deliberately NOT detected here. An absent
/// `weight_map`, an omitted pinned weight root, a config `validate()` rejects,
/// and a wrong recorded shape all flow through to the caller's assertions so
/// they still fail the test. Availability is the only thing this may veto.
fn load_pinned_checkpoint(model_dir: &Path) -> Result<PinnedCheckpoint, String> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let index_text = read_checkpoint_file(&index_path)?;
    let index: Value = serde_json::from_str(&index_text)
        .map_err(|err| format!("{} does not parse as JSON: {err}", index_path.display()))?;

    let config_path = model_dir.join("config.json");
    let config_text = read_checkpoint_file(&config_path)?;
    let config: MuseGlimmerConfig = serde_json::from_str(&config_text).map_err(|err| {
        format!(
            "{} does not parse as a Muse Glimmer config: {err}",
            config_path.display()
        )
    })?;

    // A missing or non-object `weight_map` is a malformed index rather than a
    // missing file: hand the caller an empty map so the weight-root assertion
    // reports it as the contract violation it is.
    let weight_map = index["weight_map"].as_object().cloned().unwrap_or_default();
    for key in PINNED_POST_TOWER_WEIGHT_KEYS {
        // A pinned key absent from an index that otherwise parsed is a contract
        // violation, not an availability problem. Leave it for the assertion.
        let Some(shard) = weight_map.get(key).and_then(Value::as_str) else {
            continue;
        };
        let shard_path = model_dir.join(shard);
        if !shard_path.exists() {
            return Err(format!(
                "shard {} holding {key} is not present",
                shard_path.display()
            ));
        }
    }

    Ok(PinnedCheckpoint { config, weight_map })
}

/// Read a checkpoint file, distinguishing "absent" from "present but
/// unreadable" and naming the path in both cases.
fn read_checkpoint_file(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            format!("{} is not present", path.display())
        } else {
            format!("{} could not be read: {err}", path.display())
        }
    })
}

/// Report that the pinned checkpoint is not usable. Machines that never
/// downloaded it skip, so the rest of the suite still runs there. Machines that
/// own it can set `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` to turn the skip into a
/// failure, so a half-downloaded or corrupted checkpoint cannot quietly disable
/// this contract check forever on the one machine positioned to enforce it.
fn skip_or_fail_pinned_checkpoint(reason: &str) {
    // The crate-wide env lock serializes this read against the tests that
    // mutate the process environment with `unsafe set_var`; on Rust 2024 an
    // unsynchronized concurrent read of the env block is undefined behavior.
    let required = {
        let _env_guard = crate::test_support::env_lock::env_lock();
        std::env::var("MLXCEL_REQUIRE_PINNED_CHECKPOINTS").is_ok_and(|value| value == "1")
    };
    assert!(
        !required,
        "MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1 but the pinned Muse Glimmer checkpoint is not usable: \
         {reason}"
    );
    eprintln!(
        "Skipping pinned_post_tower_weight_roots_and_shapes_match_published_contract: {reason}"
    );
}

#[test]
fn pinned_post_tower_weight_roots_and_shapes_match_published_contract() {
    let model_dir = Path::new(PINNED_MODEL_DIR);
    match load_pinned_checkpoint(model_dir) {
        Ok(checkpoint) => assert_post_tower_contract(model_dir, checkpoint),
        Err(reason) => skip_or_fail_pinned_checkpoint(&reason),
    }
}

/// The published post-tower contract itself. Every check is an unconditional
/// assertion: reaching this function means every file it needs was already
/// confirmed readable, so anything failing here is the checkpoint disagreeing
/// with the contract. Split out of the test body so the synthetic checkpoints
/// below can prove a mismatched weight root or shape still fails, never skips.
fn assert_post_tower_contract(model_dir: &Path, checkpoint: PinnedCheckpoint) {
    let PinnedCheckpoint { config, weight_map } = checkpoint;

    config.validate().unwrap();
    let mut actual_keys = weight_map
        .keys()
        .filter(|key| {
            key.starts_with("model.vision_adapter.") || key.starts_with("model.vision_projection.")
        })
        .cloned()
        .collect::<Vec<_>>();
    actual_keys.sort();
    assert_eq!(actual_keys, PINNED_POST_TOWER_WEIGHT_KEYS);

    let expected = BTreeMap::from([
        (
            "model.vision_adapter.fc1.weight",
            vec![config.projector_hidden_size, config.out_hidden_size],
        ),
        (
            "model.vision_adapter.fc2.weight",
            vec![config.projector_hidden_size, config.projector_hidden_size],
        ),
        (
            "model.vision_projection.weight",
            vec![config.text_config.hidden_size, config.projector_hidden_size],
        ),
    ]);
    for (key, expected_shape) in expected {
        let shard = weight_map
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("{key} must name a shard in the pinned weight index"));
        let shape = match safetensors_shape(&model_dir.join(shard), key) {
            Ok(shape) => shape,
            Err(reason) => {
                skip_or_fail_pinned_checkpoint(&reason);
                return;
            }
        };
        assert_eq!(shape, expected_shape, "{key}");
    }
}

/// Read the recorded shape of `key` out of the safetensors header at `path`.
///
/// `Err` covers availability and readability only: a shard that cannot be
/// opened, a header truncated by an interrupted download, or a header that does
/// not parse as JSON. A header that parses but does not describe `key` as a
/// shaped tensor is a wrong checkpoint rather than a missing one, so it panics
/// and fails the test. Only the file header is read, never the tensor payload.
fn safetensors_shape(path: &Path, key: &str) -> Result<Vec<usize>, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|err| format!("shard {} could not be opened: {err}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|err| format!("shard {} could not be inspected: {err}", path.display()))?
        .len();

    let mut header_len = [0u8; 8];
    file.read_exact(&mut header_len).map_err(|err| {
        format!(
            "shard {} is truncated before its header length: {err}",
            path.display()
        )
    })?;
    let header_len = u64::from_le_bytes(header_len);
    // Bound the declared header against the file itself before allocating, so a
    // corrupt length cannot turn into a multi-gigabyte allocation.
    if header_len > file_len.saturating_sub(8) {
        return Err(format!(
            "shard {} declares a {header_len}-byte safetensors header but holds {file_len} bytes",
            path.display()
        ));
    }

    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header).map_err(|err| {
        format!(
            "shard {} is truncated inside its header: {err}",
            path.display()
        )
    })?;
    let header: Value = serde_json::from_slice(&header).map_err(|err| {
        format!(
            "shard {} has a safetensors header that does not parse as JSON: {err}",
            path.display()
        )
    })?;

    Ok(header[key]["shape"]
        .as_array()
        .unwrap_or_else(|| panic!("shard {} records no shape for {key}", path.display()))
        .iter()
        .map(|dim| {
            dim.as_u64().unwrap_or_else(|| {
                panic!(
                    "shard {} records a non-integer dimension for {key}",
                    path.display()
                )
            }) as usize
        })
        .collect())
}

// Synthetic partial-checkpoint coverage. The real pinned checkpoint is either
// complete or absent, so the partial cases cannot be exercised against it, and
// nothing under `models/` may be moved or truncated to fake one. These tests
// build throwaway checkpoints in a temp dir, which is what makes the skip path
// observable on machines that do not own the checkpoint.

/// Smallest `config.json` body that both deserializes into a
/// `MuseGlimmerConfig` and passes `validate()`, so the contract tests below
/// reach the weight-root and shape assertions rather than tripping on the
/// config. Every field outside `text_config` carries a serde default.
const SYNTHETIC_CONFIG_JSON: &str = r#"{
  "text_config": {
    "model_type": "muse_glimmer_text", "hidden_size": 6, "intermediate_size": 8,
    "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
    "head_dim": 8, "vocab_size": 32, "rms_norm_eps": 1e-6,
    "layer_types": ["full_attention"]
  }
}"#;

/// Shard name every synthetic index below points its pinned keys at.
const SYNTHETIC_SHARD: &str = "model-00001-of-00002.safetensors";

fn write_synthetic_config(dir: &Path) {
    std::fs::write(dir.join("config.json"), SYNTHETIC_CONFIG_JSON).unwrap();
}

/// Write an index whose `weight_map` sends every key in `keys` to `shard`.
fn write_synthetic_index(dir: &Path, keys: &[&str], shard: &str) {
    let weight_map = keys
        .iter()
        .map(|key| (*key, shard))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::json!({ "weight_map": weight_map }).to_string(),
    )
    .unwrap();
}

/// Write a shard whose 8-byte prefix declares `declared_len` bytes of header
/// followed by `body`, and return its path. A `declared_len` larger than `body`
/// reproduces the truncation an interrupted download leaves behind.
fn write_shard(dir: &Path, declared_len: u64, body: &[u8]) -> PathBuf {
    let mut bytes = declared_len.to_le_bytes().to_vec();
    bytes.extend_from_slice(body);
    let path = dir.join(SYNTHETIC_SHARD);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// Write a well-formed shard whose header records `shape` for every key in
/// `keys`. The tensor payload is omitted because only the header is ever read.
fn write_synthetic_shard(dir: &Path, keys: &[&str], shape: &[usize]) -> PathBuf {
    let header = keys
        .iter()
        .map(|key| {
            (
                (*key).to_string(),
                serde_json::json!({ "dtype": "F32", "shape": shape, "data_offsets": [0, 0] }),
            )
        })
        .collect::<serde_json::Map<String, Value>>();
    let header = Value::Object(header).to_string();
    write_shard(dir, header.len() as u64, header.as_bytes())
}

/// Materialize a complete synthetic checkpoint: config, an index over `keys`,
/// and the shard those keys point at recording `shape` for each of them.
fn complete_synthetic_checkpoint(keys: &[&str], shape: &[usize]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_config(dir.path());
    write_synthetic_index(dir.path(), keys, SYNTHETIC_SHARD);
    write_synthetic_shard(dir.path(), keys, shape);
    dir
}

#[test]
fn pinned_precheck_names_an_absent_index() {
    let dir = tempfile::tempdir().unwrap();

    let err = load_pinned_checkpoint(dir.path()).unwrap_err();

    assert!(err.contains("model.safetensors.index.json"), "{err}");
    assert!(err.contains("is not present"), "{err}");
}

#[test]
fn pinned_precheck_names_a_missing_config() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_index(dir.path(), &PINNED_POST_TOWER_WEIGHT_KEYS, SYNTHETIC_SHARD);

    let err = load_pinned_checkpoint(dir.path()).unwrap_err();

    assert!(err.contains("config.json"), "{err}");
    assert!(err.contains("is not present"), "{err}");
}

#[test]
fn pinned_precheck_names_a_missing_shard() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_config(dir.path());
    write_synthetic_index(dir.path(), &PINNED_POST_TOWER_WEIGHT_KEYS, SYNTHETIC_SHARD);

    let err = load_pinned_checkpoint(dir.path()).unwrap_err();

    assert!(err.contains(SYNTHETIC_SHARD), "{err}");
    assert!(err.contains("is not present"), "{err}");
}

#[test]
fn pinned_precheck_names_an_unparseable_index() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_config(dir.path());
    // An index whose write was cut short mid-object.
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        "{\"weight_map\": {\"model.vision_projection.weight\"",
    )
    .unwrap();

    let err = load_pinned_checkpoint(dir.path()).unwrap_err();

    assert!(err.contains("model.safetensors.index.json"), "{err}");
    assert!(err.contains("does not parse as JSON"), "{err}");
}

#[test]
fn pinned_precheck_names_an_unparseable_config() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_index(dir.path(), &PINNED_POST_TOWER_WEIGHT_KEYS, SYNTHETIC_SHARD);
    std::fs::write(dir.path().join("config.json"), "{\"text_config\": {").unwrap();

    let err = load_pinned_checkpoint(dir.path()).unwrap_err();

    assert!(err.contains("config.json"), "{err}");
    assert!(
        err.contains("does not parse as a Muse Glimmer config"),
        "{err}"
    );
}

#[test]
fn pinned_precheck_accepts_a_complete_synthetic_checkpoint() {
    let dir = complete_synthetic_checkpoint(&PINNED_POST_TOWER_WEIGHT_KEYS, &[6, 4]);

    let checkpoint = load_pinned_checkpoint(dir.path()).unwrap();

    assert_eq!(checkpoint.config.text_config.hidden_size, 6);
    assert_eq!(
        checkpoint.weight_map.len(),
        PINNED_POST_TOWER_WEIGHT_KEYS.len()
    );
}

#[test]
fn pinned_precheck_leaves_a_missing_weight_root_to_the_contract_assertion() {
    // An index that parses but omits a pinned weight root is a contract
    // violation, not an availability problem, so the pre-check must pass it
    // through for the weight-root assertion to fail on.
    let dir = complete_synthetic_checkpoint(&PINNED_POST_TOWER_WEIGHT_KEYS[..2], &[4, 8]);

    let checkpoint = load_pinned_checkpoint(dir.path()).unwrap();

    assert!(
        !checkpoint
            .weight_map
            .contains_key("model.vision_projection.weight")
    );
}

#[test]
fn pinned_precheck_leaves_an_absent_weight_map_to_the_contract_assertion() {
    // Same boundary for an index object carrying no `weight_map` at all: the
    // empty map drives the weight-root assertion to fail, not to skip.
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_config(dir.path());
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        "{\"metadata\": {}}",
    )
    .unwrap();

    let checkpoint = load_pinned_checkpoint(dir.path()).unwrap();

    assert!(checkpoint.weight_map.is_empty());
}

#[test]
#[should_panic(expected = "model.vision_projection.weight")]
fn contract_assertion_still_fails_on_a_missing_weight_root() {
    let dir = complete_synthetic_checkpoint(&PINNED_POST_TOWER_WEIGHT_KEYS[..2], &[4, 8]);
    let checkpoint = load_pinned_checkpoint(dir.path()).unwrap();

    assert_post_tower_contract(dir.path(), checkpoint);
}

#[test]
#[should_panic(expected = "model.vision_adapter.fc1.weight")]
fn contract_assertion_still_fails_on_a_wrong_recorded_shape() {
    // Every file is present and readable, so the pre-check passes. A recorded
    // shape that disagrees with the config must fail, never skip.
    let dir = complete_synthetic_checkpoint(&PINNED_POST_TOWER_WEIGHT_KEYS, &[1, 1]);
    let checkpoint = load_pinned_checkpoint(dir.path()).unwrap();

    assert_post_tower_contract(dir.path(), checkpoint);
}

#[test]
fn safetensors_shape_reads_a_well_formed_header() {
    let dir = tempfile::tempdir().unwrap();
    let shard = write_synthetic_shard(dir.path(), &["model.vision_projection.weight"], &[6, 4]);

    let shape = safetensors_shape(&shard, "model.vision_projection.weight").unwrap();

    assert_eq!(shape, vec![6, 4]);
}

#[test]
fn safetensors_shape_reports_a_truncated_header() {
    // Declare a 4 KiB header, then stop four bytes in, the shape an interrupted
    // download leaves behind.
    let dir = tempfile::tempdir().unwrap();
    let shard = write_shard(dir.path(), 4096, b"{\"mo");

    let err = safetensors_shape(&shard, "model.vision_projection.weight").unwrap_err();

    assert!(err.contains(SYNTHETIC_SHARD), "{err}");
    assert!(
        err.contains("declares a 4096-byte safetensors header"),
        "{err}"
    );
}

#[test]
fn safetensors_shape_reports_an_unparseable_header() {
    let dir = tempfile::tempdir().unwrap();
    let shard = write_shard(dir.path(), 16, b"not json at all!");

    let err = safetensors_shape(&shard, "model.vision_projection.weight").unwrap_err();

    assert!(err.contains(SYNTHETIC_SHARD), "{err}");
    assert!(err.contains("does not parse as JSON"), "{err}");
}
