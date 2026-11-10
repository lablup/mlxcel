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

//! `qwen4_exp_mtp` drafter tests (issue #2880): config resolution, upstream
//! sanitize semantics, parity against mlx-vlm's `Qwen4ExpMTPDraftModel` on
//! the recorded fixture (`tests/fixtures/qwen4_exp/mtp.json`), and the
//! stateful prefill / draft / accept lifecycle.

use serde_json::{Value, json};

use mlxcel_core::drafter::{Drafter, DrafterError};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::Qwen4ExpMtpDraftModel;
use crate::models::qwen4_exp::parity_tests::{fixture, hashed, shape_of, weights};
use crate::models::qwen4_exp_mtp_config::Qwen4ExpMtpConfig;

const RTOL: f32 = 1e-5;

fn values(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect()
}

fn expected(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("float list")
        .iter()
        .map(|v| v.as_f64().expect("float") as f32)
        .collect()
}

fn ints(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("int list")
        .iter()
        .map(|v| v.as_i64().expect("int") as i32)
        .collect()
}

#[track_caller]
fn assert_close(label: &str, actual: &MlxArray, expected: &[f32]) {
    let actual = values(actual);
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let diff = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        diff <= RTOL * scale,
        "{label}: max |actual - upstream| = {diff:e} exceeds {RTOL:e} x max |upstream| ({scale:e})"
    );
}

fn drafter_config(fixture: &Value) -> Qwen4ExpMtpConfig {
    Qwen4ExpMtpConfig::from_json(&json!({
        "model_type": "qwen4_exp_mtp",
        "text_config": fixture["tiny_config"].clone(),
    }))
    .expect("tiny drafter config")
}

/// The target pieces the drafter borrows, hashed under `target.*`.
struct MockTarget {
    embed: UnifiedEmbedding,
    lm_head: UnifiedLinear,
    residual_streams: Option<usize>,
}

impl MockTarget {
    fn new(weights: &WeightMap) -> Self {
        Self {
            embed: UnifiedEmbedding::from_weights(weights, "target.embed_tokens", 64, 4)
                .expect("embedding"),
            lm_head: UnifiedLinear::from_weights(weights, "target.lm_head", 64, 4).expect("head"),
            residual_streams: None,
        }
    }
}

impl LanguageModel for MockTarget {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        unreachable!("drafter tests do not invoke the target forward")
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        0
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed.forward(input_ids))
    }

    fn embed_tokens_module(&self) -> Option<UnifiedEmbedding> {
        Some(self.embed.clone_shared())
    }

    fn lm_head_module(&self) -> Option<UnifiedLinear> {
        Some(self.lm_head.clone_shared())
    }

    fn residual_stream_count(&self) -> Option<usize> {
        self.residual_streams
    }
}

fn bound_drafter(fixture: &Value) -> (Qwen4ExpMtpDraftModel, MockTarget) {
    let weights = weights(fixture);
    let mut drafter =
        Qwen4ExpMtpDraftModel::from_weights(&weights, drafter_config(fixture)).expect("drafter");
    let target = MockTarget::new(&weights);
    drafter.validate_target_compat(&target).expect("compatible");
    drafter.bind(&target).expect("bind");
    (drafter, target)
}

fn greedy() -> SamplingConfig {
    SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    }
}

fn last_position(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last = shape[1] - 1;
    mlxcel_core::slice(x, &[0, last, 0], &[shape[0], last + 1, shape[2]])
}

// ── Config ─────────────────────────────────────────────────────────────────

/// The released drafter's `config.json` shape: top-level `block_size`, the
/// checkpoint spelling `full_attention` of the one QSA layer, no PLE.
#[test]
fn config_resolves_the_one_qsa_layer_head_and_reads_the_block_size() {
    let fixture = fixture("mtp");
    let mut text = fixture["tiny_config"].clone();
    text["layer_types"] = json!(["full_attention"]);
    text["num_hidden_layers"] = json!(1);
    text["mtp_num_hidden_layers"] = json!(1);
    let config = Qwen4ExpMtpConfig::from_json(&json!({
        "block_size": 4,
        "model_type": "qwen4_exp_mtp",
        "text_config": text,
    }))
    .expect("config");
    assert_eq!(config.block_size, 4);
    assert_eq!(config.layer.raw.num_hidden_layers, 1);
    assert!(!config.layer.is_linear_layer(0));
    assert!(config.layer.ple_layer_ids.is_empty());
    assert_eq!(config.hc_count(), 4);
    assert_eq!(config.stream_width(), 4 * 32);
    // Upstream's default: `mtp_num_hidden_layers + 1`.
    assert_eq!(drafter_config(&fixture).block_size, 2);
}

#[test]
fn config_refuses_a_deeper_head_a_wrong_model_type_and_a_missing_text_config() {
    let fixture = fixture("mtp");
    let mut text = fixture["tiny_config"].clone();
    text["mtp_num_hidden_layers"] = json!(2);
    let err =
        Qwen4ExpMtpConfig::from_json(&json!({"model_type": "qwen4_exp_mtp", "text_config": text}))
            .expect_err("two layers");
    assert!(err.contains("mtp_num_hidden_layers = 2"), "{err}");

    let err = Qwen4ExpMtpConfig::from_json(&json!({
        "model_type": "qwen3_5_mtp",
        "text_config": fixture["tiny_config"].clone(),
    }))
    .expect_err("wrong family");
    assert!(err.contains("model_type"), "{err}");

    let err = Qwen4ExpMtpConfig::from_json(&json!({"model_type": "qwen4_exp_mtp"}))
        .expect_err("no text config");
    assert!(err.contains("text_config"), "{err}");
}

// ── Sanitize ───────────────────────────────────────────────────────────────

fn tensor(shape: &[i32], offset: f32) -> UniquePtr<MlxArray> {
    let count: i32 = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|i| i as f32 + offset).collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

/// Upstream `sanitize`: the `mtp.` prefix is stripped, a fused
/// `experts.gate_up_proj` (with `.scales` / `.biases`) splits along the row
/// axis into `switch_mlp.gate_proj` (first half) and `switch_mlp.up_proj`
/// (second half), `down_proj` moves alongside, and the router keeps its
/// unquantized tensor untouched.
#[test]
fn sanitize_splits_the_fused_gate_up_with_its_sidecars_and_keeps_the_router_dense() {
    let (experts, inter, hidden) = (2, 3, 4);
    let mut weights = WeightMap::new();
    weights.insert(
        "mtp.layers.0.mlp.experts.gate_up_proj.weight".into(),
        tensor(&[experts, 2 * inter, hidden], 0.0),
    );
    weights.insert(
        "mtp.layers.0.mlp.experts.gate_up_proj.scales".into(),
        tensor(&[experts, 2 * inter, 1], 100.0),
    );
    weights.insert(
        "mtp.layers.0.mlp.experts.gate_up_proj.biases".into(),
        tensor(&[experts, 2 * inter, 1], 200.0),
    );
    weights.insert(
        "mtp.layers.0.mlp.experts.down_proj.weight".into(),
        tensor(&[experts, hidden, inter], 300.0),
    );
    weights.insert(
        "mtp.layers.0.mlp.experts.down_proj.scales".into(),
        tensor(&[experts, hidden, 1], 400.0),
    );
    weights.insert(
        "mtp.layers.0.mlp.gate.weight".into(),
        tensor(&[experts, hidden], 500.0),
    );
    weights.insert(
        "model.mtp.fc_hidden.weight".into(),
        tensor(&[hidden, hidden], 600.0),
    );
    weights.insert(
        "language_model.mtp.pre_fc_norm_hidden.weight".into(),
        tensor(&[hidden], 700.0),
    );

    Qwen4ExpMtpDraftModel::sanitize_weights(&mut weights, experts as usize).expect("sanitize");

    let mut keys: Vec<&String> = weights.keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "fc_hidden.weight",
            "layers.0.mlp.gate.weight",
            "layers.0.mlp.switch_mlp.down_proj.scales",
            "layers.0.mlp.switch_mlp.down_proj.weight",
            "layers.0.mlp.switch_mlp.gate_proj.biases",
            "layers.0.mlp.switch_mlp.gate_proj.scales",
            "layers.0.mlp.switch_mlp.gate_proj.weight",
            "layers.0.mlp.switch_mlp.up_proj.biases",
            "layers.0.mlp.switch_mlp.up_proj.scales",
            "layers.0.mlp.switch_mlp.up_proj.weight",
            "pre_fc_norm_hidden.weight",
        ]
    );
    // Axis -2 of `[experts, 2 * inter, hidden]`: expert 0 rows 0..3 are the
    // gate half, rows 3..6 the up half.
    let gate = values(&weights["layers.0.mlp.switch_mlp.gate_proj.weight"]);
    let up = values(&weights["layers.0.mlp.switch_mlp.up_proj.weight"]);
    let fused = values(&tensor(&[experts, 2 * inter, hidden], 0.0));
    let rows = (inter * hidden) as usize;
    assert_eq!(gate[..rows], fused[..rows]);
    assert_eq!(up[..rows], fused[rows..2 * rows]);
    assert_eq!(gate[rows..], fused[2 * rows..3 * rows]);
    assert_eq!(up[rows..], fused[3 * rows..]);
    assert_eq!(
        mlxcel_core::array_shape(&weights["layers.0.mlp.switch_mlp.gate_proj.scales"]),
        vec![experts, inter, 1]
    );
    assert_eq!(
        values(&weights["layers.0.mlp.switch_mlp.up_proj.biases"])[..(inter as usize)],
        values(&tensor(&[experts, 2 * inter, 1], 200.0))[(inter as usize)..(2 * inter as usize)]
    );
    assert_eq!(
        mlxcel_core::array_shape(&weights["layers.0.mlp.switch_mlp.down_proj.weight"]),
        vec![experts, hidden, inter]
    );
    // The router: same values, float32, no quantization sidecars appeared.
    let router = &weights["layers.0.mlp.gate.weight"];
    assert_eq!(mlxcel_core::array_dtype(router), dtype::FLOAT32);
    assert_eq!(values(router), values(&tensor(&[experts, hidden], 500.0)));
    assert!(!weights.contains_key("layers.0.mlp.gate.scales"));
}

/// The released conversion already stores stacked `switch_mlp.*` tensors
/// under stripped keys (`sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16`);
/// sanitize leaves that layout alone.
#[test]
fn sanitize_leaves_an_already_split_layout_untouched() {
    let mut weights = WeightMap::new();
    for name in [
        "layers.0.mlp.switch_mlp.gate_proj.weight",
        "layers.0.mlp.switch_mlp.up_proj.weight",
        "layers.0.mlp.switch_mlp.down_proj.weight",
        "layers.0.mlp.gate.weight",
        "fc_embedding.weight",
    ] {
        weights.insert(name.into(), tensor(&[2, 2], 0.0));
    }
    let before: Vec<String> = weights.keys().cloned().collect();
    Qwen4ExpMtpDraftModel::sanitize_weights(&mut weights, 2).expect("sanitize");
    let mut after: Vec<String> = weights.keys().cloned().collect();
    let mut before = before;
    before.sort();
    after.sort();
    assert_eq!(before, after);
}

// ── Parity against mlx-vlm ─────────────────────────────────────────────────

/// `fuse_inputs`, the one-layer forward (pre-mixer streams and LM-head
/// logits) and a cached single-token step match upstream on the recorded
/// fixture, and the post-mixer hidden is refused as the head's input.
#[test]
fn drafter_matches_upstream_fusion_forward_and_logits() {
    let fixture = fixture("mtp");
    let (mut drafter, target) = bound_drafter(&fixture);
    let tokens = ints(&fixture["tokens"]);
    let spec = &fixture["hidden"];
    let hidden = hashed(
        spec["name"].as_str().expect("name"),
        &shape_of(&spec["shape"]),
        spec["scale"].as_f64().expect("scale"),
        0.0,
    );
    let ids = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);
    let token_embed = target.embed.forward(&ids);

    let fused = drafter.fuse_inputs(&token_embed, &hidden).expect("fuse");
    assert_close("fuse_inputs", &fused, &expected(&fixture["fused"]));

    let (mixed, streams) = drafter
        .forward_hidden_stack(&tokens, &hidden)
        .expect("forward");
    assert_close(
        "pre-mixer streams",
        &streams,
        &expected(&fixture["streams"]),
    );
    let logits = drafter.project_logits(&mixed).expect("logits");
    assert_close("logits", &logits, &expected(&fixture["logits"]));
    assert_eq!(
        drafter.state_probe().0,
        tokens.len() as i32,
        "cache holds every position"
    );

    let next = fixture["next_token"].as_i64().expect("next token") as i32;
    let (next_mixed, next_streams) = drafter
        .forward_hidden_stack(&[next], &last_position(&streams))
        .expect("cached step");
    assert_close(
        "next pre-mixer streams",
        &next_streams,
        &expected(&fixture["next_streams"]),
    );
    assert_close(
        "next logits",
        &drafter.project_logits(&next_mixed).expect("logits"),
        &expected(&fixture["next_logits"]),
    );

    // The 4-stream contract: the post-mixer `[1, S, hidden]` state the
    // Qwen 3.5 adapter hands out is not this head's input.
    let err = match drafter.fuse_inputs(&token_embed, &mixed) {
        Err(err) => err,
        Ok(_) => panic!("the post-mixer hidden must be refused"),
    };
    assert!(
        matches!(err, DrafterError::DraftFailed { ref reason } if reason.contains("hc_count * hidden_size")),
        "{err}"
    );
}

// ── Lifecycle ──────────────────────────────────────────────────────────────

fn hidden_block(positions: i32, width: i32, seed: f32) -> UniquePtr<MlxArray> {
    let n = (positions * width) as usize;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect();
    mlxcel_core::from_slice_f32(&data, &[1, positions, width])
}

/// Prompt prefill seeds the first proposal, a draft round appends one
/// in-round entry per extra proposal, and the accept hook trims the rejected
/// tail and re-seeds from the target's verify streams.
#[test]
fn prefill_draft_and_accept_keep_the_cache_aligned_with_the_target() {
    let fixture = fixture("mtp");
    let (mut drafter, _target) = bound_drafter(&fixture);
    let width = drafter.config().stream_width() as i32;
    let prompt = [3, 7, 11, 5, 9];
    drafter
        .prefill_from_target_hidden(&prompt, &hidden_block(5, width, 0.1), 21, &greedy())
        .expect("prefill");
    assert_eq!(drafter.state_probe(), (5, 5, 0, true));

    let drafts = drafter.draft_block(21, None, 4, &greedy()).expect("draft");
    assert_eq!(drafts.len(), 3, "block 4 is three proposals");
    // The seed came from the prefill; two steps appended in-round entries.
    assert_eq!(drafter.state_probe(), (7, 7, 2, false));

    // One accepted: keep one in-round entry, drop one, append the bonus.
    drafter
        .accept_verified_tokens(&hidden_block(4, width, 0.2), &drafts, 1, &[13], &greedy())
        .expect("accept");
    assert_eq!(drafter.state_probe(), (7, 7, 0, true));

    // Nothing accepted after a round that appended two entries: both go,
    // only the bonus is appended.
    let drafts = drafter.draft_block(13, None, 4, &greedy()).expect("draft");
    assert_eq!(drafter.state_probe().2, 2);
    drafter
        .accept_verified_tokens(&hidden_block(4, width, 0.3), &drafts, 0, &[17], &greedy())
        .expect("accept");
    assert_eq!(drafter.state_probe(), (8, 8, 0, true));

    // An empty cache re-anchors at the target's offset.
    drafter.reset(&_target).expect("reset");
    assert_eq!(drafter.state_probe(), (0, 0, 0, false));
    drafter
        .set_shared_kv(mlxcel_core::drafter::SharedKv::new(&[]), 40, 39, 0)
        .expect("anchor");
    assert_eq!(drafter.state_probe().1, 40);
    assert_eq!(drafter.configured_block_size(), Some(2));
    assert!(!drafter.prefer_requested_block_size());
}

#[test]
fn hidden_of_the_wrong_width_is_refused_at_prefill() {
    let fixture = fixture("mtp");
    let (mut drafter, _target) = bound_drafter(&fixture);
    let err = drafter
        .prefill_from_target_hidden(&[3, 7], &hidden_block(2, 32, 0.0), 1, &greedy())
        .expect_err("post-mixer width");
    assert!(err.to_string().contains("hc_count * hidden_size"), "{err}");
}

/// A target with another stream count is refused at bind, by name: past the
/// bind the mismatch is a draft error that ends the generation at the seed.
#[test]
fn validate_target_compat_refuses_a_different_hc_count() {
    let fixture = fixture("mtp");
    let (drafter, mut target) = bound_drafter(&fixture);
    target.residual_streams = Some(drafter.config().hc_count());
    drafter.validate_target_compat(&target).expect("same count");
    target.residual_streams = Some(drafter.config().hc_count() + 1);
    let err = drafter
        .validate_target_compat(&target)
        .expect_err("different count");
    assert!(
        matches!(err, DrafterError::BindFailed { ref reason } if reason.contains("hc_count")),
        "{err}"
    );
}

// ── Embedded head (issue #2917) ────────────────────────────────────────────

use std::path::Path;

use crate::models::embedding_test_support::write_f32_safetensors;

/// The fixture's head tensors (`target.*` excluded) as `(name, shape, values)`,
/// each name passed through `rename`.
fn head_tensors(
    fixture: &Value,
    rename: impl Fn(&str) -> String,
) -> Vec<(String, Vec<i32>, Vec<f32>)> {
    let weights = weights(fixture);
    let mut names: Vec<&String> = weights
        .keys()
        .filter(|name| !name.starts_with("target."))
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let tensor = &weights[name];
            (
                rename(name),
                mlxcel_core::array_shape(tensor),
                values(tensor),
            )
        })
        .collect()
}

/// The released target layout: the head under `mtp.*` in its own shard, the
/// target's tensors in another shard the index points at. That shard holds
/// invalid bytes, so any attempt to open it fails the load.
fn write_target_checkpoint(dir: &Path, fixture: &Value, with_head: bool) {
    let mut text = fixture["tiny_config"].clone();
    text["model_type"] = json!("qwen4_exp_text");
    text["mtp_num_hidden_layers"] = json!(1);
    std::fs::write(
        dir.join("config.json"),
        json!({
            "model_type": "qwen4_exp",
            "tie_word_embeddings": false,
            "text_config": text,
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("model-00001-of-00001.safetensors"),
        b"not a safetensors file",
    )
    .unwrap();
    let mut weight_map = serde_json::Map::new();
    for name in [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    ] {
        weight_map.insert(name.to_string(), json!("model-00001-of-00001.safetensors"));
    }
    if with_head {
        let tensors = head_tensors(fixture, |name| format!("mtp.{name}"));
        for (name, _, _) in &tensors {
            weight_map.insert(name.clone(), json!("model-mtp-00001-of-00001.safetensors"));
        }
        write_f32_safetensors(&dir.join("model-mtp-00001-of-00001.safetensors"), &tensors);
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        json!({"metadata": {}, "weight_map": weight_map}).to_string(),
    )
    .unwrap();
}

/// The split drafter layout: stripped names, the drafter's own config.
fn write_split_drafter(dir: &Path, fixture: &Value) {
    std::fs::write(
        dir.join("config.json"),
        json!({
            "model_type": "qwen4_exp_mtp",
            "block_size": 4,
            "text_config": fixture["tiny_config"].clone(),
        })
        .to_string(),
    )
    .unwrap();
    write_f32_safetensors(
        &dir.join("model.safetensors"),
        &head_tensors(fixture, str::to_string),
    );
}

fn bind(drafter: &mut Qwen4ExpMtpDraftModel, target: &MockTarget) {
    drafter.validate_target_compat(target).expect("compatible");
    drafter.bind(target).expect("bind");
}

/// One forward over the fixture prompt plus one cached step; returns both
/// logits.
fn draft_logits(drafter: &mut Qwen4ExpMtpDraftModel, fixture: &Value) -> (Vec<f32>, Vec<f32>) {
    let tokens = ints(&fixture["tokens"]);
    let spec = &fixture["hidden"];
    let hidden = hashed(
        spec["name"].as_str().expect("name"),
        &shape_of(&spec["shape"]),
        spec["scale"].as_f64().expect("scale"),
        0.0,
    );
    let (mixed, streams) = drafter
        .forward_hidden_stack(&tokens, &hidden)
        .expect("forward");
    let logits = values(&drafter.project_logits(&mixed).expect("logits"));
    let next = fixture["next_token"].as_i64().expect("next token") as i32;
    let (next_mixed, _) = drafter
        .forward_hidden_stack(&[next], &last_position(&streams))
        .expect("cached step");
    let next_logits = values(&drafter.project_logits(&next_mixed).expect("logits"));
    (logits, next_logits)
}

/// The same head written as a target checkpoint and as a split drafter
/// directory loads to the same drafter: identical draft logits, both equal
/// to upstream's, and the same block size.
#[test]
fn embedded_head_and_split_directory_produce_identical_draft_logits() {
    let fixture = fixture("mtp");
    let target_dir = tempfile::tempdir().unwrap();
    let split_dir = tempfile::tempdir().unwrap();
    write_target_checkpoint(target_dir.path(), &fixture, true);
    write_split_drafter(split_dir.path(), &fixture);
    let target = MockTarget::new(&weights(&fixture));

    let mut embedded =
        Qwen4ExpMtpDraftModel::from_target_checkpoint(target_dir.path()).expect("embedded head");
    let mut split = Qwen4ExpMtpDraftModel::from_path(split_dir.path()).expect("split drafter");
    bind(&mut embedded, &target);
    bind(&mut split, &target);
    assert_eq!(embedded.configured_block_size(), Some(4));
    assert_eq!(split.configured_block_size(), Some(4));

    let (embedded_logits, embedded_next) = draft_logits(&mut embedded, &fixture);
    let (split_logits, split_next) = draft_logits(&mut split, &fixture);
    assert_eq!(embedded_logits, split_logits, "prompt logits");
    assert_eq!(embedded_next, split_next, "cached-step logits");
    let upstream = expected(&fixture["logits"]);
    let scale = upstream.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = embedded_logits
        .iter()
        .zip(&upstream)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    assert!(diff <= RTOL * scale, "embedded head vs upstream: {diff:e}");
}

/// The target shard holds invalid bytes and the load still succeeds: the
/// index-filtered loader never opens a shard without a head tensor. The
/// binary-side factory auto-detects the head with no `--draft-kind`.
#[test]
fn embedded_head_loads_without_opening_the_target_shards() {
    let fixture = fixture("mtp");
    let dir = tempfile::tempdir().unwrap();
    write_target_checkpoint(dir.path(), &fixture, true);
    assert!(
        mlxcel_core::weights::load_weights_from_dir(dir.path()).is_err(),
        "the target shard must be unreadable for this test to mean anything"
    );
    Qwen4ExpMtpDraftModel::from_target_checkpoint(dir.path()).expect("head only");

    let (drafter, kind) =
        crate::models::drafter_loader::load_drafter(dir.path(), None).expect("factory");
    assert_eq!(kind, mlxcel_core::drafter::DrafterKind::Mtp);
    assert_eq!(drafter.configured_block_size(), Some(4));
}

/// `from_target_json` keeps the default pair and the `mtp.*` overrides under
/// their stripped names, drops the target's own overrides, and starts the
/// head at block size 4.
#[test]
fn from_target_json_strips_the_mtp_overrides_and_sets_block_size_4() {
    let fixture = fixture("mtp");
    let mut text = fixture["tiny_config"].clone();
    text["mtp_num_hidden_layers"] = json!(1);
    let full = json!({
        "model_type": "qwen4_exp",
        "tie_word_embeddings": false,
        "quantization": {
            "group_size": 32,
            "bits": 4,
            "mode": "affine",
            "mtp.layers.0.mlp.switch_mlp.gate_proj": {"group_size": 64, "bits": 8, "mode": "affine"},
            "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8, "mode": "affine"},
            "language_model.model.layers.0.mlp.switch_mlp.up_proj": {"group_size": 64, "bits": 2},
        },
        "text_config": text,
    });
    let config = Qwen4ExpMtpConfig::from_target_json(&full).expect("embedded config");
    assert_eq!(config.block_size, 4);
    let quantization = &config.layer.quantization;
    assert_eq!(
        quantization.for_module("layers.0.mlp.switch_mlp.gate_proj"),
        (64, 8)
    );
    // Target overrides never reach the head, whatever their name.
    assert_eq!(quantization.for_module("layers.0.mlp.gate"), (32, 4));
    assert_eq!(quantization.for_module("model.layers.0.mlp.gate"), (32, 4));
    assert_eq!(
        quantization.for_module("layers.0.mlp.switch_mlp.up_proj"),
        (32, 4)
    );
    assert_eq!(config.layer.raw.num_hidden_layers, 1);
    assert!(!config.layer.is_linear_layer(0));
    assert!(config.layer.ple_layer_ids.is_empty());

    // The target's own family only, and the existing depth refusal.
    let mut split = full.clone();
    split["model_type"] = json!("qwen4_exp_mtp");
    let err = Qwen4ExpMtpConfig::from_target_json(&split).expect_err("not a target config");
    assert!(err.contains("qwen4_exp"), "{err}");
    let mut deep = full;
    deep["text_config"]["mtp_num_hidden_layers"] = json!(2);
    let err = Qwen4ExpMtpConfig::from_target_json(&deep).expect_err("two layers");
    assert!(err.contains("mtp_num_hidden_layers = 2"), "{err}");
}

/// A `qwen4_exp` checkpoint without `mtp.*`: `--draft-kind mtp` is refused
/// by name through the factory, and the constructor itself refuses it the
/// same way rather than building an empty head.
#[test]
fn a_target_without_the_head_is_refused_as_no_embedded_mtp_head() {
    let fixture = fixture("mtp");
    let dir = tempfile::tempdir().unwrap();
    write_target_checkpoint(dir.path(), &fixture, false);
    let err = crate::models::drafter_loader::load_drafter(
        dir.path(),
        Some(mlxcel_core::drafter::DrafterKind::Mtp),
    )
    .expect_err("no head");
    assert!(
        matches!(err, DrafterError::NoEmbeddedMtpHead { .. }),
        "{err}"
    );
    let err = Qwen4ExpMtpDraftModel::from_target_checkpoint(dir.path()).expect_err("no head");
    assert!(
        matches!(err, DrafterError::NoEmbeddedMtpHead { .. }),
        "{err}"
    );
    // Without `--draft-kind` the directory resolves as it did before #2917.
    assert_eq!(
        mlxcel_core::drafter::resolve_drafter_kind(dir.path(), None).unwrap(),
        mlxcel_core::drafter::DEFAULT_DRAFTER_KIND
    );
}

/// The embedded head keeps the split drafter's `hc_count` guard.
#[test]
fn embedded_head_refuses_a_target_with_a_different_hc_count() {
    let fixture = fixture("mtp");
    let dir = tempfile::tempdir().unwrap();
    write_target_checkpoint(dir.path(), &fixture, true);
    let drafter = Qwen4ExpMtpDraftModel::from_target_checkpoint(dir.path()).expect("head");
    let mut target = MockTarget::new(&weights(&fixture));
    target.residual_streams = Some(drafter.config().hc_count() + 1);
    let err = drafter
        .validate_target_compat(&target)
        .expect_err("different count");
    assert!(
        matches!(err, DrafterError::BindFailed { ref reason } if reason.contains("hc_count")),
        "{err}"
    );
}
