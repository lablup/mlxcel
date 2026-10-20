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

//! DeBERTa-v2 encoder tests: Hugging Face relative-position buckets, config
//! refusals, a tiny encoder against mlx-vlm's `DebertaModel`, and the real
//! `gliner2.5-small-v1` encoder against `transformers` on an input longer
//! than the 128 tokens where mlx-vlm stops agreeing with Hugging Face.
//!
//! Fixtures: `tests/fixtures/token_classification/deberta_v2_layer_fixture.json`
//! and `deberta_v2_hf_long.json`, both written by
//! `tests/fixtures/token_classification/generate_gliner2_5_fixture.py`; each
//! records its oracle versions in an `oracle` field.

use std::path::PathBuf;

use mlxcel_core::weights::WeightMap;
use serde_json::{Value, json};

use super::{DebertaV2Args, DebertaV2Encoder, relative_position_buckets};
use crate::models::embedding_test_support::{max_abs_diff, mlx_test_guard, to_vec};

fn layer_fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/deberta_v2_layer_fixture.json"
    ))
    .expect("fixture parses")
}

fn long_fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/deberta_v2_hf_long.json"
    ))
    .expect("fixture parses")
}

fn ints(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_i64().expect("integer") as i32)
        .collect()
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("number") as f32)
        .collect()
}

/// The live gliner2.5-small-v1 encoder config, as committed in the fixture
/// generator's `TINY` but at the published geometry.
fn live_config() -> Value {
    json!({
        "model_type": "deberta-v2",
        "vocab_size": 128011,
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "max_position_embeddings": 512,
        "max_relative_positions": -1,
        "position_buckets": 256,
        "relative_attention": true,
        "pos_att_type": ["p2c", "c2p"],
        "share_att_key": true,
        "position_biased_input": false,
        "type_vocab_size": 0,
        "layer_norm_eps": 1e-7,
        "norm_rel_ebd": "layer_norm",
    })
}

#[test]
fn relative_buckets_match_hugging_face_for_lengths_8_600_and_4096() {
    let fixture = layer_fixture();
    let expected = &fixture["hf_relative_buckets"];
    for length in [8usize, 600, 4096] {
        let want = ints(&expected[length.to_string()]);
        let got = relative_position_buckets(length, 256, 512);
        assert_eq!(got.len(), 2 * length - 1);
        if let Some(index) = (0..want.len()).find(|&i| got[i] != want[i]) {
            panic!(
                "length {length}: bucket of distance {} is {} in mlxcel, {} in Hugging Face",
                index as i64 - (length as i64 - 1),
                got[index],
                want[index]
            );
        }
    }
    // Distances within half the bucket count are the identity; beyond, the
    // value grows logarithmically, reaching the table edge (255) at
    // `max_position - 1` and overshooting past it, where the gather index
    // clips, exactly as Hugging Face does.
    let buckets = relative_position_buckets(600, 256, 512);
    let at = |distance: i64| buckets[(distance + 599) as usize];
    assert_eq!(at(128), 128);
    assert_eq!(at(-128), -128);
    assert!(at(129) > 128 && at(129) < 256, "{}", at(129));
    assert_eq!(at(511), 255);
    assert_eq!(at(599), -at(-599));
    assert_eq!(at(599), 270);
}

#[test]
fn live_config_resolves_max_relative_positions_to_max_position_embeddings() {
    let args = DebertaV2Args::from_config(&live_config()).unwrap();
    assert_eq!(args.max_relative_positions, 512);
    assert_eq!(args.position_buckets, 256);
    assert!(args.c2p && args.p2c);
    assert!(args.norm_rel_ebd);
    assert_eq!(args.head_dim(), 64);
    assert!((args.attention_scale() - (64.0f32 * 3.0).sqrt()).abs() < 1e-6);

    let mut explicit = live_config();
    explicit["max_relative_positions"] = json!(1024);
    assert_eq!(
        DebertaV2Args::from_config(&explicit)
            .unwrap()
            .max_relative_positions,
        1024
    );
}

#[test]
fn unported_options_are_refused_at_config_time() {
    for (key, value, needle) in [
        (
            "position_biased_input",
            json!(true),
            "position_biased_input",
        ),
        ("type_vocab_size", json!(2), "type_vocab_size"),
        ("share_att_key", json!(false), "share_att_key"),
        ("conv_kernel_size", json!(3), "conv_kernel_size"),
        ("embedding_size", json!(128), "embedding_size"),
        ("relative_attention", json!(false), "relative_attention"),
        ("position_buckets", json!(-1), "position_buckets"),
        ("pos_att_type", json!(["c2c"]), "pos_att_type"),
        ("norm_rel_ebd", json!("rms"), "norm_rel_ebd"),
    ] {
        let mut config = live_config();
        config[key] = value;
        let err = DebertaV2Args::from_config(&config)
            .err()
            .unwrap_or_else(|| panic!("{key} should be refused"));
        assert!(err.to_string().contains(needle), "{key}: {err}");
    }
    // Hugging Face defaults `share_att_key` to false, which is the unported
    // separate-projection layout, so an absent key is refused rather than
    // silently read as shared.
    let mut absent = live_config();
    absent.as_object_mut().unwrap().remove("share_att_key");
    let err = DebertaV2Args::from_config(&absent).err().unwrap();
    assert!(err.to_string().contains("share_att_key"), "{err}");
    // A missing required field reports the config it came from.
    let mut missing = live_config();
    missing.as_object_mut().unwrap().remove("hidden_size");
    let err = format!("{:#}", DebertaV2Args::from_config(&missing).err().unwrap());
    assert!(
        err.contains("deberta-v2 config") && err.contains("hidden_size"),
        "{err}"
    );
}

#[test]
fn pos_att_type_accepts_the_legacy_joined_string() {
    let mut config = live_config();
    config["pos_att_type"] = json!("p2c|c2p");
    let args = DebertaV2Args::from_config(&config).unwrap();
    assert!(args.c2p && args.p2c);
    config["pos_att_type"] = json!("c2p");
    let args = DebertaV2Args::from_config(&config).unwrap();
    assert!(args.c2p && !args.p2c);
    assert!((args.attention_scale() - (64.0f32 * 2.0).sqrt()).abs() < 1e-6);
}

/// Weights of the tiny fixture as the encoder reads them.
fn fixture_weights(fixture: &Value) -> WeightMap {
    fixture["weights"]
        .as_object()
        .expect("weights object")
        .iter()
        .map(|(key, tensor)| {
            let shape = ints(&tensor["shape"]);
            let data = floats(&tensor["data"]);
            (key.clone(), mlxcel_core::from_slice_f32(&data, &shape))
        })
        .collect()
}

#[test]
fn tiny_encoder_matches_mlx_vlm_with_a_padded_row() {
    let fixture = layer_fixture();
    let _guard = mlx_test_guard();
    let args = DebertaV2Args::from_config(&fixture["config"]).unwrap();
    assert_eq!(
        args.position_buckets, 4,
        "length 6 must exceed the identity range"
    );
    let encoder = DebertaV2Encoder::from_weights(&fixture_weights(&fixture), args, 64, 4).unwrap();

    let ids: Vec<i32> = fixture["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(ints)
        .collect();
    let mask: Vec<i32> = fixture["attention_mask"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(ints)
        .collect();
    let input_ids = mlxcel_core::from_slice_i32(&ids, &[2, 6]);
    let attention_mask = mlxcel_core::from_slice_i32(&mask, &[2, 6]);
    let hidden = encoder.encode(&input_ids, Some(&attention_mask)).unwrap();
    assert_eq!(mlxcel_core::array_shape(&hidden), vec![2, 6, 8]);

    let want: Vec<f32> = fixture["output"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().flat_map(floats))
        .collect();
    let got = to_vec(&hidden);
    let diff = max_abs_diff(&got, &want);
    assert!(diff < 1e-4, "max |mlxcel - mlx-vlm| = {diff}");

    // Without a mask the full first row is unchanged, which pins the
    // `None` path GLiNER2.5 takes.
    let unmasked = encoder
        .encode(&mlxcel_core::from_slice_i32(&ids[..6], &[1, 6]), None)
        .unwrap();
    let diff = max_abs_diff(&to_vec(&unmasked), &want[..48]);
    assert!(diff < 1e-4, "max |unmasked - mlx-vlm row 0| = {diff}");
}

#[test]
fn encoder_refuses_mismatched_inputs() {
    let fixture = layer_fixture();
    let _guard = mlx_test_guard();
    let args = DebertaV2Args::from_config(&fixture["config"]).unwrap();
    let encoder = DebertaV2Encoder::from_weights(&fixture_weights(&fixture), args, 64, 4).unwrap();
    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let mask = mlxcel_core::from_slice_i32(&[1, 1], &[1, 2]);
    let err = encoder
        .encode(&ids, Some(&mask))
        .err()
        .expect("mask shape is checked");
    assert!(err.to_string().contains("attention_mask"), "{err}");
    let flat = mlxcel_core::from_slice_i32(&[1, 2, 3], &[3]);
    let err = encoder.encode(&flat, None).err().expect("rank is checked");
    assert!(err.to_string().contains("[B, L]"), "{err}");
}

/// `models/mlx/gliner2.5-small-v1` through the repository's `models` link, or
/// the download cache; `None` soft-skips.
pub(crate) fn gliner2_5_small_dir() -> Option<PathBuf> {
    let mlx = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("mlx")
        .join("gliner2.5-small-v1");
    if mlx.join("config.json").is_file() {
        return Some(mlx);
    }
    crate::models::embedding_test_support::local_checkpoint("fastino/gliner2.5-small-v1")
}

/// The real gliner2.5-small-v1 trunk, as `GLiNER2.5` loads it.
fn load_real_encoder(dir: &std::path::Path) -> DebertaV2Encoder {
    let config = crate::embeddings::loader::read_embedding_config(dir).unwrap();
    let encoder_config =
        crate::token_classification::gliner2_5::resolve_encoder_config(dir, &config).unwrap();
    let args = DebertaV2Args::from_config(&encoder_config).unwrap();
    let weights: WeightMap = mlxcel_core::weights::load_weights_from_dir(dir)
        .expect("weights load")
        .into_iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("encoder.")
                .map(|rest| (rest.to_string(), value))
        })
        .collect();
    DebertaV2Encoder::from_weights(&weights, args, 64, 4).unwrap()
}

#[test]
fn real_encoder_matches_transformers_past_128_tokens() {
    let Some(dir) = gliner2_5_small_dir() else {
        eprintln!("gliner2.5-small-v1 not found; skipping the transformers comparison");
        return;
    };
    let _guard = mlx_test_guard();
    let fixture = long_fixture();
    let encoder = load_real_encoder(&dir);

    let ids = ints(&fixture["input_ids"]);
    assert!(
        ids.len() > 256,
        "the fixture must exceed the 128-token agreement range"
    );
    let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);
    let hidden = to_vec(&encoder.encode(&input_ids, None).unwrap());
    // The second encode is the replay of the same CUDA graphs; the two must
    // agree bit for bit (`real_encoder_first_run_equals_replay` is the
    // fresh-process version of this check, issue #2462).
    let replay = to_vec(&encoder.encode(&input_ids, None).unwrap());
    let drift = max_abs_diff(&hidden, &replay);
    assert_eq!(
        drift, 0.0,
        "first encode differs from its replay by {drift}"
    );
    let width = 384;
    let mut worst = 0.0f32;
    for row in ints(&fixture["rows"]) {
        let want = floats(&fixture["hidden_rows"][row.to_string()]);
        let got = &hidden[row as usize * width..(row as usize + 1) * width];
        let diff = max_abs_diff(got, &want);
        let mlx_vlm_diff = fixture["mlx_vlm_max_abs_diff_per_row"][row.to_string()]
            .as_f64()
            .unwrap();
        eprintln!(
            "row {row}: max |mlxcel - transformers| = {diff:.2e} (mlx-vlm differs from transformers by {mlx_vlm_diff:.2e})"
        );
        worst = worst.max(diff);
    }
    assert!(worst < 2e-3, "max |mlxcel - transformers| = {worst}");
    eprintln!(
        "transformers parity ok on {} tokens: max diff {worst:.2e}",
        ids.len()
    );
}

/// Issue #2462 A2: the whole 12-layer trunk through MLX's own CUDA graph
/// chunking on the 306-token fixture input. The first `encode` of a process
/// must equal its replay bit for bit. Process-level: run it in N >= 20 fresh
/// processes, since one process proves nothing when a race fails about 40%
/// of them (a cuBLASLt GEMM overlapping unrelated kernel nodes of the same
/// graph, fixed by the capture barrier in the CUDA device overlay).
#[test]
fn real_encoder_first_run_equals_replay() {
    let Some(dir) = gliner2_5_small_dir() else {
        eprintln!("gliner2.5-small-v1 not found; skipping the graph replay check");
        return;
    };
    let _guard = mlx_test_guard();
    let fixture = long_fixture();
    let encoder = load_real_encoder(&dir);

    let ids = ints(&fixture["input_ids"]);
    assert_eq!(
        ids.len(),
        306,
        "the fixture is the 306-token input the defect was seen on"
    );
    let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);
    let first = to_vec(&encoder.encode(&input_ids, None).unwrap());
    let replay = to_vec(&encoder.encode(&input_ids, None).unwrap());
    let drift = max_abs_diff(&first, &replay);
    let differing = first.iter().zip(&replay).filter(|(a, b)| a != b).count();
    assert_eq!(
        drift,
        0.0,
        "first encode differs from its replay by {drift} on {differing} of {} values",
        first.len()
    );
}
