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

//! Unit tests for the Qwen 3.5 MTP drafter: sanitize semantics, the strict
//! weight inventory, and the stateful cache/position lifecycle
//! (prompt prefill → draft → accept) that upstream drives implicitly.

use super::config::Qwen35MtpConfig;
use super::model::Qwen35MtpDraftModel;
use crate::drafter::{Drafter, DrafterError, SharedKv};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::{UnifiedEmbedding, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;

const HIDDEN: i32 = 8;
const VOCAB: i32 = 16;

fn tiny_config() -> Qwen35MtpConfig {
    let cfg: Qwen35MtpConfig = serde_json::from_str(
        r#"{
        "model_type": "qwen3_5_mtp",
        "block_size": 3,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "vocab_size": 16,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
            "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": false
        }
    }"#,
    )
    .expect("tiny config parses");
    cfg.normalize().expect("tiny config normalizes")
}

fn insert_val(w: &mut WeightMap, key: &str, shape: &[i32], val: f32) {
    let n: i32 = shape.iter().product();
    let data = vec![val; n as usize];
    w.insert(key.to_string(), ffi::from_slice_f32(&data, shape));
}

/// The published 15-tensor inventory (already `mtp.`-prefix stripped), with
/// deterministic small values so real MLX forwards run.
fn tiny_weights() -> WeightMap {
    let mut w: WeightMap = WeightMap::new();
    insert_val(&mut w, "fc.weight", &[8, 16], 0.05);
    insert_val(&mut w, "pre_fc_norm_embedding.weight", &[8], 1.0);
    insert_val(&mut w, "pre_fc_norm_hidden.weight", &[8], 1.0);
    insert_val(&mut w, "norm.weight", &[8], 1.0);
    insert_val(&mut w, "layers.0.input_layernorm.weight", &[8], 1.0);
    insert_val(&mut w, "layers.0.post_attention_layernorm.weight", &[8], 1.0);
    insert_val(&mut w, "layers.0.self_attn.q_proj.weight", &[16, 8], 0.05);
    insert_val(&mut w, "layers.0.self_attn.k_proj.weight", &[4, 8], 0.05);
    insert_val(&mut w, "layers.0.self_attn.v_proj.weight", &[4, 8], 0.05);
    insert_val(&mut w, "layers.0.self_attn.o_proj.weight", &[8, 8], 0.05);
    insert_val(&mut w, "layers.0.self_attn.q_norm.weight", &[4], 1.0);
    insert_val(&mut w, "layers.0.self_attn.k_norm.weight", &[4], 1.0);
    insert_val(&mut w, "layers.0.mlp.gate_proj.weight", &[16, 8], 0.05);
    insert_val(&mut w, "layers.0.mlp.up_proj.weight", &[16, 8], 0.05);
    insert_val(&mut w, "layers.0.mlp.down_proj.weight", &[8, 16], 0.05);
    w
}

fn build_drafter() -> Qwen35MtpDraftModel {
    Qwen35MtpDraftModel::from_weights(&tiny_weights(), tiny_config()).expect("tiny drafter builds")
}

/// Mock Qwen-style target: real embedding table + untied LM head so the
/// drafter's bind captures working shared-buffer modules.
struct MockQwenTarget {
    embed: UnifiedEmbedding,
    lm_head: UnifiedLinear,
}

impl MockQwenTarget {
    fn new() -> Self {
        let mut w: WeightMap = WeightMap::new();
        insert_val(&mut w, "embed_tokens.weight", &[VOCAB, HIDDEN], 0.1);
        insert_val(&mut w, "lm_head.weight", &[VOCAB, HIDDEN], 0.1);
        let embed =
            UnifiedEmbedding::from_weights(&w, "embed_tokens", 64, 4).expect("mock embed builds");
        let lm_head = UnifiedLinear::from_weights(&w, "lm_head", 64, 4).expect("mock head builds");
        Self { embed, lm_head }
    }
}

impl LanguageModel for MockQwenTarget {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [crate::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        unreachable!("drafter tests do not invoke the target forward")
    }

    fn make_caches(&self) -> Vec<crate::layers::KVCache> {
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
}

fn greedy() -> SamplingConfig {
    SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    }
}

fn hidden_block(seq: i32) -> UniquePtr<MlxArray> {
    let n = (seq * HIDDEN) as usize;
    let data: Vec<f32> = (0..n).map(|i| 0.01 * (i % 7) as f32).collect();
    ffi::from_slice_f32(&data, &[1, seq, HIDDEN])
}

fn first_f32(arr: &MlxArray) -> f32 {
    ffi::eval(arr);
    let bytes = ffi::array_to_raw_bytes(arr);
    f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

// ── Sanitize ──────────────────────────────────────────────────────────────

/// Raw-HF layout (`mtp.` prefix) gets stripped and its 1-D norm weights get
/// the `+1.0` shift on exactly the seven upstream suffixes.
#[test]
fn sanitize_strips_mtp_prefix_and_shifts_norms() {
    let mut w: WeightMap = WeightMap::new();
    w.insert("mtp.norm.weight".into(), ffi::from_slice_f32(&[0.5], &[1]));
    w.insert(
        "mtp.layers.0.self_attn.q_norm.weight".into(),
        ffi::from_slice_f32(&[0.25], &[1]),
    );
    // Non-norm key: stripped but NOT shifted.
    w.insert(
        "mtp.fc.weight".into(),
        ffi::from_slice_f32(&[0.5, 0.5], &[1, 2]),
    );
    Qwen35MtpDraftModel::sanitize_weights(&mut w);

    assert!(!w.keys().any(|k| k.starts_with("mtp.")), "prefix stripped");
    assert_eq!(first_f32(w.get("norm.weight").unwrap()), 1.5);
    assert_eq!(
        first_f32(w.get("layers.0.self_attn.q_norm.weight").unwrap()),
        1.25
    );
    assert_eq!(first_f32(w.get("fc.weight").unwrap()), 0.5);
}

/// The published `mlx-community/Qwen3.8-27B-MTP-bf16` layout is already
/// stripped, i.e. the conversion already applied the shift: sanitize MUST
/// leave it untouched (a second `+1.0` corrupts every norm).
#[test]
fn sanitize_leaves_already_stripped_layout_unshifted() {
    let mut w: WeightMap = WeightMap::new();
    w.insert("norm.weight".into(), ffi::from_slice_f32(&[1.5], &[1]));
    w.insert(
        "pre_fc_norm_hidden.weight".into(),
        ffi::from_slice_f32(&[0.75], &[1]),
    );
    Qwen35MtpDraftModel::sanitize_weights(&mut w);
    assert_eq!(first_f32(w.get("norm.weight").unwrap()), 1.5);
    assert_eq!(first_f32(w.get("pre_fc_norm_hidden.weight").unwrap()), 0.75);
}

// ── Weight inventory ──────────────────────────────────────────────────────

/// The exact published 15-tensor inventory constructs.
#[test]
fn from_weights_accepts_the_published_inventory() {
    let _ = build_drafter();
}

/// A missing required tensor fails with a message naming it.
#[test]
fn from_weights_rejects_missing_tensor() {
    let mut w = tiny_weights();
    w.remove("layers.0.self_attn.k_norm.weight");
    let err = Qwen35MtpDraftModel::from_weights(&w, tiny_config()).expect_err("must fail");
    let msg = format!("{err}");
    assert!(msg.contains("k_norm"), "got: {msg}");
}

/// Unknown extra tensors mean the directory is not a standalone MTP head;
/// fail closed naming them instead of silently ignoring.
#[test]
fn from_weights_rejects_unknown_extra_tensor() {
    let mut w = tiny_weights();
    insert_val(&mut w, "model.layers.0.self_attn.q_proj.weight", &[4, 4], 0.0);
    let err = Qwen35MtpDraftModel::from_weights(&w, tiny_config()).expect_err("must fail");
    let msg = format!("{err}");
    assert!(msg.contains("unexpected tensor"), "got: {msg}");
    assert!(msg.contains("model.layers.0"), "got: {msg}");
}

// ── Lifecycle errors ──────────────────────────────────────────────────────

#[test]
fn draft_block_requires_bind_then_hidden() {
    let mut drafter = build_drafter();
    let err = drafter
        .draft_block(1, None, 3, &greedy())
        .expect_err("unbound must fail");
    assert!(matches!(err, DrafterError::BindNotCalled));

    let target = MockQwenTarget::new();
    drafter.bind(&target).expect("bind");
    // No seed, no hidden: the MTP path needs one of the two.
    let err = drafter
        .draft_block(1, None, 3, &greedy())
        .expect_err("hidden required without a seed");
    assert!(matches!(err, DrafterError::DraftBlockMissingHidden));
}

#[test]
fn validate_target_compat_accepts_matching_mock() {
    let drafter = build_drafter();
    let target = MockQwenTarget::new();
    drafter
        .validate_target_compat(&target)
        .expect("mock geometry matches the tiny config");
}

// ── Stateful lifecycle ────────────────────────────────────────────────────

/// The load-bearing state machine: prompt prefill, seeded draft, full and
/// partial accepts. After every phase the drafter's cache offset and logical
/// `next_position` must track the target cache offset exactly (that
/// invariant is what `set_shared_kv` re-anchoring checks against).
#[test]
fn prefill_draft_accept_keeps_positions_in_sync() {
    let mut drafter = build_drafter();
    let target = MockQwenTarget::new();
    drafter.bind(&target).expect("bind");
    let sampler = greedy();

    // Prompt prefill: 4 tokens + first bonus -> 4 cache entries, seed ready.
    let prompt = [3_i32, 5, 7, 2];
    drafter
        .prefill_from_target_hidden(&prompt, &hidden_block(4), 9, &sampler)
        .expect("prefill");
    assert_eq!(drafter.state_probe(), (4, 4, 0, true));

    // Round arm: target cache offset after target prefill = 4. Cache is
    // non-empty and in sync, so nothing is cleared.
    drafter
        .set_shared_kv(SharedKv::new(&[]), 4, 3, 0)
        .expect("arm");
    assert_eq!(drafter.state_probe(), (4, 4, 0, true));

    // Draft block (block_size 3): the seed supplies the first proposal, one
    // forward produces the second. One in-round cache append.
    let draft = drafter
        .draft_block(9, None, 3, &sampler)
        .expect("draft with seed");
    assert_eq!(draft.len(), 2);
    assert_eq!(drafter.state_probe(), (5, 5, 1, false));

    // Full accept: accepted = 2, new_tokens = both drafts + target bonus.
    // keep = min(2, 1) = 1 in-round entry, no trim; appends draft[1] and the
    // bonus paired with verify hidden -> cache 4 + 1 + 2 = 7. The target
    // cache after a full-accept verify is 4 + 3 = 7: in sync.
    let new_tokens = vec![draft[0], draft[1], 11];
    drafter
        .accept_verified_tokens(&hidden_block(3), &draft, 2, &new_tokens, &sampler)
        .expect("full accept");
    assert_eq!(drafter.state_probe(), (7, 7, 0, true));
    drafter
        .set_shared_kv(SharedKv::new(&[]), 7, 6, 0)
        .expect("re-arm stays in sync");
    assert_eq!(drafter.state_probe(), (7, 7, 0, true));

    // Next round: draft again (seed + 1 forward -> 8 entries), then a
    // zero-accept round: trim the 1 in-round entry back to 7, append only
    // the target bonus -> 8. Target: 7 + (0 + 1) = 8. In sync.
    let draft2 = drafter.draft_block(11, None, 3, &sampler).expect("draft 2");
    assert_eq!(draft2.len(), 2);
    assert_eq!(drafter.state_probe(), (8, 8, 1, false));
    drafter
        .accept_verified_tokens(&hidden_block(3), &draft2, 0, &[13], &sampler)
        .expect("zero accept");
    assert_eq!(drafter.state_probe(), (8, 8, 0, true));
}

/// Empty-cache mode: `set_shared_kv` anchors `next_position` to the target
/// offset, and a draft block runs from the caller's (bonus, hidden) pair
/// without any seed or history.
#[test]
fn set_shared_kv_anchors_empty_cache_and_drafts_without_history() {
    let mut drafter = build_drafter();
    let target = MockQwenTarget::new();
    drafter.bind(&target).expect("bind");
    drafter
        .set_shared_kv(SharedKv::new(&[]), 10, 9, 0)
        .expect("arm");
    assert_eq!(drafter.state_probe(), (0, 10, 0, false));

    let draft = drafter
        .draft_block(4, Some(hidden_block(1).as_ref().unwrap()), 3, &greedy())
        .expect("seedless draft");
    assert_eq!(draft.len(), 2);
    // Two forwards appended (no seed): cache 2, position 12.
    assert_eq!(drafter.state_probe(), (2, 12, 2, false));
}

/// A non-empty cache whose position disagrees with the target offset is
/// stale (out-of-band reset, hook failure): `set_shared_kv` clears it and
/// re-anchors instead of drafting from corrupt state.
#[test]
fn set_shared_kv_clears_stale_state_on_position_mismatch() {
    let mut drafter = build_drafter();
    let target = MockQwenTarget::new();
    drafter.bind(&target).expect("bind");
    drafter
        .prefill_from_target_hidden(&[3, 5, 7, 2], &hidden_block(4), 9, &greedy())
        .expect("prefill");
    assert_eq!(drafter.state_probe(), (4, 4, 0, true));

    // Target says the cache sits at 9; the drafter thinks 4 -> stale.
    drafter
        .set_shared_kv(SharedKv::new(&[]), 9, 8, 0)
        .expect("arm");
    assert_eq!(drafter.state_probe(), (0, 9, 0, false));
}

/// `reset` clears everything and re-binds, leaving the drafter ready for the
/// next session's empty-cache re-anchor (the slice-grant rotation path).
#[test]
fn reset_clears_state_and_rebinds() {
    let mut drafter = build_drafter();
    let target = MockQwenTarget::new();
    drafter.bind(&target).expect("bind");
    drafter
        .prefill_from_target_hidden(&[3, 5, 7, 2], &hidden_block(4), 9, &greedy())
        .expect("prefill");
    drafter.reset(&target).expect("reset");
    assert_eq!(drafter.state_probe(), (0, 0, 0, false));
    // Still bound: a seedless draft with an explicit hidden works.
    drafter
        .set_shared_kv(SharedKv::new(&[]), 4, 3, 0)
        .expect("arm");
    let draft = drafter
        .draft_block(4, Some(hidden_block(1).as_ref().unwrap()), 3, &greedy())
        .expect("post-reset draft");
    assert_eq!(draft.len(), 2);
}

/// `prefer_requested_block_size` and the configured block size mirror the
/// upstream drafter flags.
#[test]
fn block_size_flags_match_upstream() {
    let drafter = build_drafter();
    assert!(drafter.prefer_requested_block_size());
    assert_eq!(drafter.configured_block_size(), Some(3));
}
