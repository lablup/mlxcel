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

//! Unit tests for the GLM-4.7-Flash MTP drafter (issue #1326): config
//! normalization, the strict weight inventory, the `kv_b_proj` sanitizer,
//! and the stateful cache/position lifecycle (prompt prefill, seeded draft,
//! partial and full accepts) on a tiny synthetic block.

use super::*;
use mlxcel_core::drafter::{Drafter, DrafterError, SharedKv};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const HIDDEN: i32 = 8;
const VOCAB: i32 = 16;
const HEADS: i32 = 2;
const QK_NOPE: i32 = 4;
const QK_ROPE: i32 = 2;
const V_HEAD: i32 = 4;
const KV_LORA_RANK: i32 = 8;
const EXPERTS: i32 = 2;
const MOE_INTER: i32 = 8;

const TINY_CONFIG: &str = r#"{
    "model_type": "glm4_moe_lite_mtp",
    "block_size": 3,
    "tie_word_embeddings": false,
    "text_config": {
        "model_type": "glm4_moe_lite",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "kv_lora_rank": 8,
        "q_lora_rank": null,
        "qk_rope_head_dim": 2,
        "qk_nope_head_dim": 4,
        "v_head_dim": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0
    }
}"#;

fn tiny_config() -> Glm4MoeLiteMtpConfig {
    let cfg: Glm4MoeLiteMtpConfig = serde_json::from_str(TINY_CONFIG).expect("tiny config parses");
    cfg.normalize().expect("tiny config normalizes")
}

fn tiny_config_json() -> serde_json::Value {
    serde_json::from_str(TINY_CONFIG).expect("tiny config parses")
}

/// Deterministic, non-constant values so different tokens produce different
/// logits and the routed selection is not a tie.
fn insert_ramp(w: &mut WeightMap, key: &str, shape: &[i32], seed: f32) {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| 0.05 * ((i as f32 * 0.37 + seed).sin()))
        .collect();
    w.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
}

fn insert_ones(w: &mut WeightMap, key: &str, shape: &[i32]) {
    let n: i32 = shape.iter().product();
    w.insert(
        key.to_string(),
        mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape),
    );
}

/// The `split-mtp` layout for the tiny config, with `kv_b_proj` already
/// decomposed into the `embed_q` / `unembed_out` pair.
fn tiny_weights() -> WeightMap {
    let mut w = WeightMap::new();
    let b = BLOCK_PREFIX;
    insert_ramp(&mut w, "model.embed_tokens.weight", &[VOCAB, HIDDEN], 1.0);
    insert_ones(&mut w, "model.enorm.weight", &[HIDDEN]);
    insert_ones(&mut w, "model.hnorm.weight", &[HIDDEN]);
    insert_ramp(&mut w, "model.eh_proj.weight", &[HIDDEN, 2 * HIDDEN], 2.0);
    insert_ones(&mut w, "model.shared_head_norm.weight", &[HIDDEN]);
    insert_ramp(&mut w, "lm_head.weight", &[VOCAB, HIDDEN], 3.0);
    insert_ones(&mut w, &format!("{b}.input_layernorm.weight"), &[HIDDEN]);
    insert_ones(
        &mut w,
        &format!("{b}.post_attention_layernorm.weight"),
        &[HIDDEN],
    );
    insert_ramp(
        &mut w,
        &format!("{b}.self_attn.q_proj.weight"),
        &[HEADS * (QK_NOPE + QK_ROPE), HIDDEN],
        4.0,
    );
    insert_ramp(
        &mut w,
        &format!("{b}.self_attn.kv_a_proj_with_mqa.weight"),
        &[KV_LORA_RANK + QK_ROPE, HIDDEN],
        5.0,
    );
    insert_ones(
        &mut w,
        &format!("{b}.self_attn.kv_a_layernorm.weight"),
        &[KV_LORA_RANK],
    );
    insert_ramp(
        &mut w,
        &format!("{b}.self_attn.embed_q.weight"),
        &[HEADS, KV_LORA_RANK, QK_NOPE],
        6.0,
    );
    insert_ramp(
        &mut w,
        &format!("{b}.self_attn.unembed_out.weight"),
        &[HEADS, V_HEAD, KV_LORA_RANK],
        7.0,
    );
    insert_ramp(
        &mut w,
        &format!("{b}.self_attn.o_proj.weight"),
        &[HIDDEN, HEADS * V_HEAD],
        8.0,
    );
    insert_ramp(
        &mut w,
        &format!("{b}.mlp.gate.weight"),
        &[EXPERTS, HIDDEN],
        9.0,
    );
    w.insert(
        format!("{b}.mlp.gate.e_score_correction_bias"),
        mlxcel_core::from_slice_f32(&[0.0, 0.0], &[EXPERTS]),
    );
    insert_ramp(
        &mut w,
        &format!("{b}.mlp.switch_mlp.gate_proj.weight"),
        &[EXPERTS, MOE_INTER, HIDDEN],
        10.0,
    );
    insert_ramp(
        &mut w,
        &format!("{b}.mlp.switch_mlp.up_proj.weight"),
        &[EXPERTS, MOE_INTER, HIDDEN],
        11.0,
    );
    insert_ramp(
        &mut w,
        &format!("{b}.mlp.switch_mlp.down_proj.weight"),
        &[EXPERTS, HIDDEN, MOE_INTER],
        12.0,
    );
    insert_ramp(
        &mut w,
        &format!("{b}.mlp.shared_experts.gate_proj.weight"),
        &[MOE_INTER, HIDDEN],
        13.0,
    );
    insert_ramp(
        &mut w,
        &format!("{b}.mlp.shared_experts.up_proj.weight"),
        &[MOE_INTER, HIDDEN],
        14.0,
    );
    insert_ramp(
        &mut w,
        &format!("{b}.mlp.shared_experts.down_proj.weight"),
        &[HIDDEN, MOE_INTER],
        15.0,
    );
    w
}

fn build_drafter() -> Glm4MoeLiteMtpDraftModel {
    let _runtime = crate::initialize_runtime();
    Glm4MoeLiteMtpDraftModel::from_weights(&tiny_weights(), tiny_config())
        .expect("tiny drafter builds")
}

/// Mock target exposing only the two modules the drafter's compat check
/// reads: a token table (hidden width) and an untied head (vocabulary).
struct MockGlmTarget {
    embed: UnifiedEmbedding,
    lm_head: UnifiedLinear,
}

impl MockGlmTarget {
    fn new(hidden: i32, vocab: i32) -> Self {
        let mut w = WeightMap::new();
        insert_ramp(&mut w, "embed_tokens.weight", &[vocab, hidden], 20.0);
        insert_ramp(&mut w, "lm_head.weight", &[vocab, hidden], 21.0);
        let embed = UnifiedEmbedding::from_weights(&w, "embed_tokens", 64, 4).expect("mock embed");
        let lm_head = UnifiedLinear::from_weights(&w, "lm_head", 64, 4).expect("mock head");
        Self { embed, lm_head }
    }
}

impl LanguageModel for MockGlmTarget {
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
    mlxcel_core::from_slice_f32(&data, &[1, seq, HIDDEN])
}

fn bound_drafter() -> (Glm4MoeLiteMtpDraftModel, MockGlmTarget) {
    let mut drafter = build_drafter();
    let target = MockGlmTarget::new(HIDDEN, VOCAB);
    drafter.bind(&target).expect("bind");
    (drafter, target)
}

// ── Config ────────────────────────────────────────────────────────────────

#[test]
fn config_defaults_block_size_to_nextn_plus_one_and_floors_at_two() {
    let mut raw = tiny_config_json();
    raw.as_object_mut().unwrap().remove("block_size");
    let cfg: Glm4MoeLiteMtpConfig = serde_json::from_value(raw).unwrap();
    let cfg = cfg.normalize().expect("normalizes");
    assert_eq!(cfg.block_size(), 2);
    assert_eq!(cfg.runtime_block_size(), 2);

    // A wider requested block is honored by `block_size()` but the trained
    // depth caps `runtime_block_size()`.
    let cfg = tiny_config();
    assert_eq!(cfg.block_size(), 3);
    assert_eq!(cfg.runtime_block_size(), 2);
}

#[test]
fn config_rejects_a_block_of_one_and_a_foreign_text_config() {
    let mut raw = tiny_config_json();
    raw["block_size"] = serde_json::json!(1);
    let err = serde_json::from_value::<Glm4MoeLiteMtpConfig>(raw)
        .unwrap()
        .normalize()
        .expect_err("block_size 1 drafts nothing");
    assert!(err.contains("drafts nothing"), "{err}");

    let mut raw = tiny_config_json();
    raw["text_config"]["model_type"] = serde_json::json!("qwen3_moe");
    let err = serde_json::from_value::<Glm4MoeLiteMtpConfig>(raw)
        .unwrap()
        .normalize()
        .expect_err("foreign text_config");
    assert!(err.contains("qwen3_moe"), "{err}");

    let mut raw = tiny_config_json();
    raw["text_config"]["hidden_size"] = serde_json::json!(0);
    let err = serde_json::from_value::<Glm4MoeLiteMtpConfig>(raw)
        .unwrap()
        .normalize()
        .expect_err("zero hidden");
    assert!(err.contains("hidden_size"), "{err}");
}

#[test]
fn config_quantization_block_feeds_the_block_loader() {
    let mut raw = tiny_config_json();
    raw["quantization"] = serde_json::json!({"group_size": 32, "bits": 8});
    let cfg = serde_json::from_value::<Glm4MoeLiteMtpConfig>(raw)
        .unwrap()
        .normalize()
        .expect("quantized config normalizes");
    assert!(cfg.is_quantized());
    assert_eq!(cfg.text_config().group_size(), 32);
    assert_eq!(cfg.text_config().bits(), 8);
}

// ── Weight inventory and sanitize ─────────────────────────────────────────

#[test]
fn from_weights_accepts_the_split_mtp_inventory() {
    let _ = build_drafter();
}

#[test]
fn from_weights_rejects_unknown_extra_tensor() {
    let _runtime = crate::initialize_runtime();
    let mut w = tiny_weights();
    insert_ramp(
        &mut w,
        "model.layers.0.self_attn.q_proj.weight",
        &[4, 4],
        0.0,
    );
    let err = Glm4MoeLiteMtpDraftModel::from_weights(&w, tiny_config()).expect_err("must fail");
    let msg = err.to_string();
    assert!(msg.contains("unexpected tensor"), "got: {msg}");
    assert!(msg.contains("model.layers.0"), "got: {msg}");
}

#[test]
fn from_weights_rejects_eh_proj_that_does_not_project_to_hidden() {
    let _runtime = crate::initialize_runtime();
    let mut w = tiny_weights();
    insert_ramp(
        &mut w,
        "model.eh_proj.weight",
        &[HIDDEN + 1, 2 * HIDDEN],
        0.0,
    );
    let err = Glm4MoeLiteMtpDraftModel::from_weights(&w, tiny_config()).expect_err("must fail");
    assert!(err.to_string().contains("eh_proj"), "{err}");
}

#[test]
fn from_weights_rejects_missing_block_tensor() {
    let _runtime = crate::initialize_runtime();
    let mut w = tiny_weights();
    w.remove(&format!("{BLOCK_PREFIX}.self_attn.kv_a_layernorm.weight"));
    let err = Glm4MoeLiteMtpDraftModel::from_weights(&w, tiny_config()).expect_err("must fail");
    assert!(err.to_string().contains("kv_a_layernorm"), "{err}");
}

/// A hand-assembled directory that kept `kv_b_proj` under the block prefix is
/// decomposed by the same helper the target sanitizer uses, and a directory
/// that already ships the pair is left alone.
#[test]
fn sanitize_decomposes_a_kept_kv_b_proj_and_leaves_the_pair_alone() {
    let _runtime = crate::initialize_runtime();
    let cfg = tiny_config();
    let mut w = tiny_weights();
    let attn = format!("{BLOCK_PREFIX}.self_attn");
    w.remove(&format!("{attn}.embed_q.weight"));
    w.remove(&format!("{attn}.unembed_out.weight"));
    insert_ramp(
        &mut w,
        &format!("{attn}.kv_b_proj.weight"),
        &[HEADS * (QK_NOPE + V_HEAD), KV_LORA_RANK],
        30.0,
    );
    Glm4MoeLiteMtpDraftModel::sanitize_weights(&mut w, cfg.text_config()).expect("sanitize");
    assert!(!w.contains_key(&format!("{attn}.kv_b_proj.weight")));
    assert_eq!(
        mlxcel_core::array_shape(w.get(&format!("{attn}.embed_q.weight")).unwrap()),
        vec![HEADS, KV_LORA_RANK, QK_NOPE]
    );
    assert_eq!(
        mlxcel_core::array_shape(w.get(&format!("{attn}.unembed_out.weight")).unwrap()),
        vec![HEADS, V_HEAD, KV_LORA_RANK]
    );
    // The decomposed map builds.
    let _ = Glm4MoeLiteMtpDraftModel::from_weights(&w, cfg).expect("decomposed map builds");

    // Pre-decomposed: untouched.
    let mut w = tiny_weights();
    let before = w.len();
    Glm4MoeLiteMtpDraftModel::sanitize_weights(&mut w, tiny_config().text_config())
        .expect("sanitize no-op");
    assert_eq!(w.len(), before);
}

// ── Compat ────────────────────────────────────────────────────────────────

#[test]
fn validate_target_compat_accepts_matching_and_rejects_foreign_geometry() {
    let drafter = build_drafter();
    drafter
        .validate_target_compat(&MockGlmTarget::new(HIDDEN, VOCAB))
        .expect("matching geometry");

    let err = drafter
        .validate_target_compat(&MockGlmTarget::new(HIDDEN * 2, VOCAB))
        .expect_err("hidden mismatch");
    assert!(matches!(err, DrafterError::BindFailed { .. }), "{err}");
    assert!(err.to_string().contains("hidden"), "{err}");

    let err = drafter
        .validate_target_compat(&MockGlmTarget::new(HIDDEN, VOCAB * 2))
        .expect_err("vocab mismatch");
    assert!(err.to_string().contains("vocab"), "{err}");
}

#[test]
fn draft_block_requires_bind_then_hidden() {
    let mut drafter = build_drafter();
    let err = drafter
        .draft_block(1, None, 3, &greedy())
        .expect_err("unbound must fail");
    assert!(matches!(err, DrafterError::BindNotCalled));

    let target = MockGlmTarget::new(HIDDEN, VOCAB);
    drafter.bind(&target).expect("bind");
    let err = drafter
        .draft_block(1, None, 3, &greedy())
        .expect_err("hidden required without a seed");
    assert!(matches!(err, DrafterError::DraftBlockMissingHidden));
}

// ── Stateful lifecycle ────────────────────────────────────────────────────

/// One seedless draft step consumes the caller's `(bonus, hidden)` pair
/// through `embed_tokens -> enorm / hnorm -> eh_proj -> block` and yields one
/// in-vocabulary proposal per step plus one cache entry per step.
#[test]
fn draft_step_consumes_embedding_and_hidden() {
    let (mut drafter, _target) = bound_drafter();
    drafter
        .set_shared_kv(SharedKv::new(&[]), 10, 9, 0)
        .expect("arm");
    assert_eq!(drafter.state_probe(), (0, 10, 0, false));

    let hidden = hidden_block(1);
    let draft = drafter
        .draft_block(4, Some(hidden.as_ref().unwrap()), 3, &greedy())
        .expect("seedless draft");
    assert_eq!(draft.len(), 2);
    for tok in &draft {
        assert!(
            (0..VOCAB).contains(tok),
            "proposal {tok} outside the vocabulary"
        );
    }
    // Two forwards appended (no seed): cache 2, position 12, both in-round.
    assert_eq!(drafter.state_probe(), (2, 12, 2, false));
}

/// Prompt prefill runs `prompt[1..] ++ [first_bonus]` paired with the `P`
/// prompt hiddens: the cache holds exactly `P` entries afterwards and the
/// next-round seed is ready.
#[test]
fn history_prefill_pairs_shifted_tokens_with_hidden() {
    let (mut drafter, _target) = bound_drafter();
    let prompt = [3_i32, 5, 7, 2];
    drafter
        .prefill_from_target_hidden(&prompt, &hidden_block(4), 9, &greedy())
        .expect("prefill");
    assert_eq!(drafter.state_probe(), (4, 4, 0, true));

    // A hidden block that does not cover the prompt is refused.
    let err = drafter
        .prefill_from_target_hidden(&prompt, &hidden_block(3), 9, &greedy())
        .expect_err("hidden too short");
    assert!(matches!(err, DrafterError::DraftFailed { .. }), "{err}");

    // An empty prompt is a no-op.
    let mut fresh = build_drafter();
    fresh.bind(&MockGlmTarget::new(HIDDEN, VOCAB)).unwrap();
    fresh
        .prefill_from_target_hidden(&[], &hidden_block(1), 9, &greedy())
        .expect("empty prompt");
    assert_eq!(fresh.state_probe(), (0, 0, 0, false));
}

/// After a `block_size = 3` draft (one in-round append), the accept hook
/// trims the rejected in-round tail and appends the accepted tokens plus the
/// bonus, so the cache offset tracks the target's `P + accepted + 1`.
#[test]
fn accept_trims_rejected_tail_and_appends_accepted() {
    let sampler = greedy();
    let prompt = [3_i32, 5, 7, 2];
    let p = prompt.len() as i32;

    for accepted in 0..=2_usize {
        let (mut drafter, _target) = bound_drafter();
        drafter
            .prefill_from_target_hidden(&prompt, &hidden_block(p), 9, &sampler)
            .expect("prefill");
        drafter
            .set_shared_kv(SharedKv::new(&[]), p as usize, (p - 1) as usize, 0)
            .expect("arm");
        let draft = drafter.draft_block(9, None, 3, &sampler).expect("draft");
        assert_eq!(draft.len(), 2);
        // Seed supplied the first proposal; one forward appended the second.
        assert_eq!(drafter.state_probe(), (p + 1, p + 1, 1, false));

        let mut new_tokens: Vec<i32> = draft[..accepted].to_vec();
        new_tokens.push(11);
        drafter
            .accept_verified_tokens(&hidden_block(3), &draft, accepted, &new_tokens, &sampler)
            .unwrap_or_else(|e| panic!("accept {accepted}: {e}"));
        let expected = p + accepted as i32 + 1;
        assert_eq!(
            drafter.state_probe(),
            (expected, expected, 0, true),
            "accepted = {accepted}"
        );
        // The target's cache after the same round sits at the same offset, so
        // the next arm keeps the history.
        drafter
            .set_shared_kv(
                SharedKv::new(&[]),
                expected as usize,
                (expected - 1) as usize,
                0,
            )
            .expect("re-arm");
        assert_eq!(drafter.state_probe(), (expected, expected, 0, true));
    }
}

/// A malformed accept clears the drafter rather than leaving a half-updated
/// cache behind.
#[test]
fn accept_with_short_hidden_fails_closed() {
    let (mut drafter, _target) = bound_drafter();
    drafter
        .prefill_from_target_hidden(&[3, 5, 7, 2], &hidden_block(4), 9, &greedy())
        .expect("prefill");
    let err = drafter
        .accept_verified_tokens(&hidden_block(1), &[1, 2], 1, &[1, 11], &greedy())
        .expect_err("short hidden");
    assert!(matches!(err, DrafterError::DraftFailed { .. }), "{err}");
    assert_eq!(drafter.state_probe(), (0, 0, 0, false));
}

#[test]
fn set_shared_kv_clears_stale_state_on_position_mismatch() {
    let (mut drafter, _target) = bound_drafter();
    drafter
        .prefill_from_target_hidden(&[3, 5, 7, 2], &hidden_block(4), 9, &greedy())
        .expect("prefill");
    drafter
        .set_shared_kv(SharedKv::new(&[]), 9, 8, 0)
        .expect("arm");
    assert_eq!(drafter.state_probe(), (0, 9, 0, false));
}

#[test]
fn reset_clears_state_and_rebinds() {
    let (mut drafter, target) = bound_drafter();
    drafter
        .prefill_from_target_hidden(&[3, 5, 7, 2], &hidden_block(4), 9, &greedy())
        .expect("prefill");
    drafter.reset(&target).expect("reset");
    assert_eq!(drafter.state_probe(), (0, 0, 0, false));
    drafter
        .set_shared_kv(SharedKv::new(&[]), 4, 3, 0)
        .expect("arm");
    let draft = drafter
        .draft_block(4, Some(hidden_block(1).as_ref().unwrap()), 2, &greedy())
        .expect("post-reset draft");
    assert_eq!(draft.len(), 1);
}

#[test]
fn block_size_flags_report_the_trained_depth() {
    let drafter = build_drafter();
    assert!(drafter.prefer_requested_block_size());
    assert_eq!(drafter.configured_block_size(), Some(2));
    assert_eq!(drafter.kind(), DrafterKind::Mtp);
    assert_eq!(drafter.hidden_size(), HIDDEN as usize);
    assert!(drafter.make_cache().is_empty());
}
