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

//! Unit tests for the Gemma 4 MTP assistant drafter (`Gemma4AssistantDraftModel`).
//!
//! The tests build a deliberately tiny synthetic config (`hidden_size = 64`,
//! `num_hidden_layers = 2`) and feed in zero-initialised weights so that the
//! whole load / forward / sample path can be exercised on a unit-test budget.
//! Output magnitude is not asserted — the tests verify shapes, lifecycle
//! ordering, and error propagation.

use super::config::{DrafterRopeParameters, DrafterTextConfig, Gemma4AssistantConfig};
use super::model::Gemma4AssistantDraftModel;
use crate::drafter::{Drafter, DrafterError, DrafterKind, SharedKv};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::collections::HashMap;

/// Minimal drafter config used by every test. Uses `hidden_size=64`, 2 layers
/// in a SWA / full pattern, head_dim=32 → 2 heads, all KV-shared. Field
/// defaults match upstream Python where applicable.
fn make_test_config(num_hidden_layers: usize, tie_word_embeddings: bool) -> Gemma4AssistantConfig {
    let mut rope_parameters = HashMap::new();
    rope_parameters.insert(
        "full_attention".to_string(),
        DrafterRopeParameters {
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 1.0,
            rope_type: "proportional".to_string(),
        },
    );
    rope_parameters.insert(
        "sliding_attention".to_string(),
        DrafterRopeParameters {
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rope_type: "default".to_string(),
        },
    );
    let layer_types: Vec<String> = (0..num_hidden_layers)
        .map(|i| {
            if i + 1 == num_hidden_layers {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect();
    let text_config = DrafterTextConfig {
        model_type: "gemma4_text".into(),
        hidden_size: 64,
        num_hidden_layers,
        intermediate_size: 128,
        num_attention_heads: 2,
        head_dim: 32,
        global_head_dim: None,
        rms_norm_eps: 1e-6,
        vocab_size: 16,
        num_key_value_heads: 1,
        num_global_key_value_heads: None,
        num_kv_shared_layers: num_hidden_layers,
        rope_parameters,
        sliding_window: 64,
        sliding_window_pattern: 2,
        max_position_embeddings: 256,
        layer_types,
        attention_k_eq_v: false,
        final_logit_softcapping: None,
        use_double_wide_mlp: false,
        quantization: None,
    };
    Gemma4AssistantConfig {
        model_type: "gemma4_assistant".into(),
        backbone_hidden_size: 32,
        use_ordered_embeddings: false,
        num_centroids: 8,
        centroid_intermediate_top_k: 2,
        tie_word_embeddings,
        block_size: 4,
        target_layer_ids: vec![],
        target_layer_types: vec![],
        text_config: Some(text_config),
    }
    .normalize()
    .expect("normalize")
}

/// Build a fp32 zero-initialised tensor of the given shape.
fn zeros_f32(shape: &[i32]) -> UniquePtr<MlxArray> {
    ffi::zeros(shape, crate::dtype::FLOAT32)
}

/// Insert a fp32 zero-initialised tensor at the given weight-map key.
fn insert_zeros(weights: &mut WeightMap, key: &str, shape: &[i32]) {
    weights.insert(key.to_string(), zeros_f32(shape));
}

/// Build a complete weight map for the test config above. Every tensor is
/// fp32 zeros — values are not asserted by the tests, only shapes and
/// lifecycle.
fn make_test_weights(config: &Gemma4AssistantConfig) -> WeightMap {
    let tc = config.text_config();
    let hidden = tc.hidden_size as i32;
    let inter = tc.intermediate_size as i32;
    let vocab = tc.vocab_size as i32;
    let n_heads = tc.num_attention_heads as i32;
    let head_dim = tc.head_dim as i32;
    let n_kv = tc.num_key_value_heads as i32;
    let backbone = config.backbone_hidden_size as i32;
    let num_centroids = config.num_centroids as i32;

    let mut w = WeightMap::new();

    // model.embed_tokens.weight: [vocab, hidden].
    insert_zeros(&mut w, "model.embed_tokens.weight", &[vocab, hidden]);

    // model.norm.weight: [hidden].
    insert_zeros(&mut w, "model.norm.weight", &[hidden]);

    // pre_projection.weight: [hidden, 2 * backbone] (Linear.forward
    // transposes weight, so weight shape is [out, in] = [hidden,
    // 2*backbone]).
    insert_zeros(&mut w, "pre_projection.weight", &[hidden, 2 * backbone]);

    // post_projection.weight: [backbone, hidden].
    insert_zeros(&mut w, "post_projection.weight", &[backbone, hidden]);

    if !config.tie_word_embeddings {
        // lm_head.weight: [vocab, hidden].
        insert_zeros(&mut w, "lm_head.weight", &[vocab, hidden]);
    }

    // Centroid LM head weights — only for E-series (use_ordered_embeddings=true).
    if config.use_ordered_embeddings {
        // masked_embedding.centroids.weight: [num_centroids, hidden].
        insert_zeros(
            &mut w,
            "masked_embedding.centroids.weight",
            &[num_centroids, hidden],
        );
        // masked_embedding.token_ordering: [vocab] (int32).
        // Use sequential ordering so tests can verify basic correctness.
        let ordering: Vec<i32> = (0..vocab).collect();
        w.insert(
            "masked_embedding.token_ordering".to_string(),
            ffi::from_slice_i32(&ordering, &[vocab]),
        );
    }

    // Per-layer weights.
    for i in 0..tc.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let kv_dim = n_kv * head_dim;
        // Q proj: out = n_heads * head_dim
        insert_zeros(
            &mut w,
            &format!("{p}.self_attn.q_proj.weight"),
            &[n_heads * head_dim, hidden],
        );
        let _ = kv_dim; // KV-shared layer never has its own K/V proj.
        insert_zeros(&mut w, &format!("{p}.self_attn.q_norm.weight"), &[head_dim]);
        // o_proj: out = hidden, in = n_heads * head_dim
        insert_zeros(
            &mut w,
            &format!("{p}.self_attn.o_proj.weight"),
            &[hidden, n_heads * head_dim],
        );

        // MLP
        insert_zeros(
            &mut w,
            &format!("{p}.mlp.gate_proj.weight"),
            &[inter, hidden],
        );
        insert_zeros(&mut w, &format!("{p}.mlp.up_proj.weight"), &[inter, hidden]);
        insert_zeros(
            &mut w,
            &format!("{p}.mlp.down_proj.weight"),
            &[hidden, inter],
        );

        // Norms
        insert_zeros(&mut w, &format!("{p}.input_layernorm.weight"), &[hidden]);
        insert_zeros(
            &mut w,
            &format!("{p}.post_attention_layernorm.weight"),
            &[hidden],
        );
        insert_zeros(
            &mut w,
            &format!("{p}.pre_feedforward_layernorm.weight"),
            &[hidden],
        );
        insert_zeros(
            &mut w,
            &format!("{p}.post_feedforward_layernorm.weight"),
            &[hidden],
        );
    }

    w
}

/// Mock `LanguageModel` for `bind()` testing. Only `embed_tokens` is
/// meaningful; everything else falls through to the trait defaults or
/// `unreachable!()` panics if the drafter inadvertently calls them.
struct MockLanguageModel {
    hidden_size: i32,
}

impl MockLanguageModel {
    fn new(_vocab_size: i32, hidden_size: i32) -> Self {
        Self { hidden_size }
    }
}

impl LanguageModel for MockLanguageModel {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [crate::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        unreachable!("draft tests do not invoke target forward")
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
        // Simulate the gemma4 target's embed_tokens: gather rows from
        // `embed_weight` at the given `input_ids`. For unit tests we don't
        // need correctness of the lookup — just a non-None return so the
        // drafter accepts the bind.
        let shape = ffi::array_shape(input_ids);
        let b = shape[0];
        let l = shape[1];
        Some(ffi::zeros(&[b, l, self.hidden_size], crate::dtype::FLOAT32))
    }
}

// ── Sanitize tests ────────────────────────────────────────────────────────

#[test]
fn sanitize_drops_lm_head_when_tied() {
    let cfg = make_test_config(2, true);
    let mut weights = make_test_weights(&cfg);
    // Add an extra lm_head.weight as if the checkpoint shipped one despite
    // tie_word_embeddings=true (upstream Python explicitly handles this case).
    insert_zeros(&mut weights, "lm_head.weight", &[16, 64]);
    assert!(weights.contains_key("lm_head.weight"));
    Gemma4AssistantDraftModel::sanitize_weights(&mut weights, &cfg);
    assert!(
        !weights.contains_key("lm_head.weight"),
        "sanitize must drop lm_head.weight when tie_word_embeddings=true"
    );
}

#[test]
fn sanitize_keeps_lm_head_when_not_tied() {
    let cfg = make_test_config(2, false);
    let mut weights = make_test_weights(&cfg);
    assert!(weights.contains_key("lm_head.weight"));
    Gemma4AssistantDraftModel::sanitize_weights(&mut weights, &cfg);
    assert!(
        weights.contains_key("lm_head.weight"),
        "sanitize must keep lm_head.weight when tie_word_embeddings=false"
    );
}

// ── Construction / weight-loading tests ──────────────────────────────────

#[test]
fn from_weights_loads_tied_dense_drafter() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg);
    assert!(
        model.is_ok(),
        "tied-dense drafter must load: {:?}",
        model.err()
    );
}

#[test]
fn from_weights_loads_non_tied_drafter_with_explicit_lm_head() {
    let cfg = make_test_config(2, false);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg);
    assert!(
        model.is_ok(),
        "non-tied drafter must load with explicit lm_head: {:?}",
        model.err()
    );
}

#[test]
fn from_weights_errors_on_missing_pre_projection() {
    let cfg = make_test_config(2, true);
    let mut weights = make_test_weights(&cfg);
    weights.remove("pre_projection.weight");
    let err = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect_err("must fail");
    match err {
        DrafterError::WeightLoad { reason } => {
            assert!(
                reason.contains("pre_projection"),
                "error must point at the missing pre_projection: {reason}"
            );
        }
        other => panic!("expected WeightLoad, got {other:?}"),
    }
}

// ── Drafter trait surface tests ──────────────────────────────────────────

#[test]
fn drafter_kind_returns_mtp() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    assert_eq!(model.kind(), DrafterKind::Mtp);
}

#[test]
fn make_cache_returns_empty_vec() {
    // MTP drafter has no own KV cache (matches upstream Python
    // `Gemma4AssistantDraftModel.make_cache(self) -> []`).
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    assert!(model.make_cache().is_empty());
}

#[test]
fn draft_block_rejects_call_before_set_shared_kv() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    let target = MockLanguageModel::new(16, 64);
    model.bind(&target).expect("bind");

    let sampler = SamplingConfig::greedy();
    let err = model
        .draft_block(0, None, 4, &sampler)
        .expect_err("must fail");
    match err {
        DrafterError::SetSharedKvNotCalled => {}
        other => panic!("expected SetSharedKvNotCalled, got {other:?}"),
    }
}

#[test]
fn draft_block_rejects_call_before_bind() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");

    // Skip bind, set_shared_kv first (the order doesn't matter; both
    // pre-conditions must hold). draft_block must fail before it runs.
    let sampler = SamplingConfig::greedy();
    let err = model
        .draft_block(0, None, 4, &sampler)
        .expect_err("must fail");
    // Order check: set_shared_kv runs first inside draft_block, so the
    // first guard to trigger is SetSharedKvNotCalled. After that fixes,
    // BindNotCalled fires. The test pins the first-failure ordering.
    match err {
        DrafterError::SetSharedKvNotCalled => {}
        other => panic!("expected SetSharedKvNotCalled, got {other:?}"),
    }
}

#[test]
fn bind_rejects_target_without_embed_tokens_override() {
    /// LanguageModel that does NOT override embed_tokens (returns None).
    struct BareTarget;
    impl LanguageModel for BareTarget {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [crate::layers::KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            unreachable!()
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
        // No `embed_tokens` override; falls through to the trait default
        // which returns None.
    }

    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");

    let target = BareTarget;
    let err = model.bind(&target).expect_err("must fail");
    match err {
        DrafterError::TargetMissingFeature { feature } => {
            assert_eq!(feature, "embed_tokens");
        }
        other => panic!("expected TargetMissingFeature, got {other:?}"),
    }
}

// ── Centroid LM head (MaskedEmbedder, issue #627) ───────────────────────

#[test]
fn bind_accepts_centroid_lm_head_via_masked_embedder() {
    // E2B / E4B drafters: `use_ordered_embeddings=true` selects the centroid
    // (MaskedEmbedder) LM head. With #627 and #628 merged, this path must
    // now succeed — `bind()` constructs the real `MaskedEmbedder` from the
    // pre-built centroid head and sets `lm_head = Some(LmHead::Centroid(...))`.
    let mut cfg = make_test_config(2, true);
    cfg.use_ordered_embeddings = true;
    // make_test_weights automatically adds masked_embedding.* when
    // use_ordered_embeddings is true.
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");

    let target = MockLanguageModel::new(16, 64);
    model
        .bind(&target)
        .expect("bind must succeed when MaskedEmbedder is wired in (#627)");
}

/// Full E-series Centroid path: `bind` → `set_shared_kv` → `draft_block`.
///
/// Asserts:
/// 1. `bind()` succeeds for a synthetic E-series config.
/// 2. `draft_block()` returns exactly `block_size - 1` proposals.
/// 3. All sampled token ids are within `[0, vocab_size)`.
///
/// Value correctness is not asserted — the test fixture uses zero weights,
/// which drive the centroid logits and embed matmul through zero → the
/// `MaskedEmbedder` scatter fills with equal values → `argmax` (greedy) is
/// deterministic but not meaningful. Shape and range suffice for this test.
#[test]
fn centroid_path_bind_set_shared_kv_draft_block_end_to_end() {
    // Build a config where hidden_size == backbone_hidden_size so that
    // pre_projection's matmul shapes are consistent with the forward call.
    let mut cfg = make_test_config(2, true);
    cfg.use_ordered_embeddings = true;
    cfg.backbone_hidden_size = 64; // match text_config.hidden_size = 64

    // Must satisfy vocab_size % num_centroids == 0.
    // Default num_centroids = 8, vocab_size = 16 → 16 % 8 == 0. OK.
    assert_eq!(cfg.num_centroids, 8, "test relies on num_centroids == 8");
    let tc = cfg.text_config();
    let vocab = tc.vocab_size;
    let backbone = cfg.backbone_hidden_size as i32;
    assert_eq!(
        vocab % cfg.num_centroids,
        0,
        "vocab must be divisible by num_centroids"
    );

    // Rebuild pre/post projection weights for the corrected backbone size.
    let mut weights = make_test_weights(&cfg);
    weights.remove("pre_projection.weight");
    weights.remove("post_projection.weight");
    insert_zeros(&mut weights, "pre_projection.weight", &[64, 2 * backbone]);
    insert_zeros(&mut weights, "post_projection.weight", &[backbone, 64]);

    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg.clone()).expect("load");

    let target = MockLanguageModel::new(vocab as i32, 64);
    model
        .bind(&target)
        .expect("bind must succeed for centroid path");

    // Set up shared K/V: 4 tensors (full + SWA) at [B=1, n_kv=1, kv=4, head=32].
    let kv_shape = &[1_i32, 1, 4, 32];
    let k_full = ffi::zeros(kv_shape, crate::dtype::FLOAT32);
    let v_full = ffi::zeros(kv_shape, crate::dtype::FLOAT32);
    let k_swa = ffi::zeros(kv_shape, crate::dtype::FLOAT32);
    let v_swa = ffi::zeros(kv_shape, crate::dtype::FLOAT32);
    let tensors: Vec<&crate::ffi::MlxArray> = vec![
        k_full.as_ref().unwrap(),
        v_full.as_ref().unwrap(),
        k_swa.as_ref().unwrap(),
        v_swa.as_ref().unwrap(),
    ];
    let shared = crate::drafter::SharedKv::new(&tensors);
    model
        .set_shared_kv(
            shared, /*kv_offset=*/ 0, /*position=*/ 0, /*left_padding=*/ 0,
        )
        .expect("set_shared_kv");

    // Build a hidden tensor [1, 1, backbone].
    let hidden = ffi::zeros(&[1, 1, backbone], crate::dtype::FLOAT32);
    let sampler = crate::generate::SamplingConfig::greedy();
    let block_size = cfg.block_size; // 4

    let proposals = model
        .draft_block(0, Some(hidden.as_ref().unwrap()), block_size, &sampler)
        .expect("draft_block must succeed for centroid path");

    // Shape: block_size - 1 proposals.
    assert_eq!(
        proposals.len(),
        block_size - 1,
        "centroid path must emit block_size - 1 proposals"
    );

    // Range: every token id must be in [0, vocab_size).
    for (i, &tok) in proposals.iter().enumerate() {
        assert!(
            tok >= 0 && tok < vocab as i32,
            "proposal[{i}] = {tok} is out of [0, {vocab})"
        );
    }
}

// ── SharedKv shape validation ────────────────────────────────────────────

#[test]
fn set_shared_kv_rejects_unexpected_tensor_count() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");

    let target = MockLanguageModel::new(16, 64);
    model.bind(&target).expect("bind");

    // 3 tensors — not a valid count for Gemma 4 shared K/V.
    let t0 = ffi::zeros(&[1, 1, 2, 32], crate::dtype::FLOAT32);
    let t1 = ffi::zeros(&[1, 1, 2, 32], crate::dtype::FLOAT32);
    let t2 = ffi::zeros(&[1, 1, 2, 32], crate::dtype::FLOAT32);
    let tensors: Vec<&MlxArray> = vec![
        t0.as_ref().unwrap(),
        t1.as_ref().unwrap(),
        t2.as_ref().unwrap(),
    ];
    let shared = SharedKv::new(&tensors);
    let err = model.set_shared_kv(shared, 0, 0, 0).expect_err("must fail");
    match err {
        DrafterError::SharedKvShape { got, expected } => {
            assert_eq!(got, 3);
            assert_eq!(expected, &[2, 4]);
        }
        other => panic!("expected SharedKvShape, got {other:?}"),
    }
}

// ── Object-safety / dispatch test ────────────────────────────────────────

#[test]
fn gemma4_assistant_drafter_is_object_safe_via_box_dyn() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    let boxed: Box<dyn Drafter> = Box::new(model);
    assert_eq!(boxed.kind(), DrafterKind::Mtp);
}

// ── Batched draft path (issue #631) ──────────────────────────────────────

/// `draft_block_batched` produces `B` rows of `block_size - 1` proposals
/// each. Pins the shape contract.
///
/// This test exercises the actual forward path; the test fixture's
/// default config uses `hidden_size != backbone_hidden_size` which
/// breaks the `pre_projection` matmul. We rebuild a corrected weight
/// set sized for `hidden == backbone` so the forward runs.
#[test]
fn draft_block_batched_returns_b_rows_of_k_minus_one_proposals() {
    // Build a corrected config where `hidden_size == backbone_hidden_size`
    // so pre_projection in_dim = 2 * backbone matches `tok_embed + h_prev`.
    let mut cfg = make_test_config(2, true);
    cfg.backbone_hidden_size = 64; // match text_config.hidden_size = 64
    let backbone = cfg.backbone_hidden_size as i32;

    // Rebuild weights with the corrected `2 * backbone` pre_projection in.
    let mut weights = make_test_weights(&cfg);
    weights.remove("pre_projection.weight");
    weights.remove("post_projection.weight");
    insert_zeros(&mut weights, "pre_projection.weight", &[64, 2 * backbone]);
    insert_zeros(&mut weights, "post_projection.weight", &[backbone, 64]);

    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    let target = MockLanguageModel::new(16, 64);
    model.bind(&target).expect("bind");

    // Build a 4-tensor shared K/V at [B=3, n_kv=1, kv_len=8, head_dim=32].
    let batch_size = 3;
    let kv_len = 8;
    let head_dim = 32;
    let n_kv = 1;
    let k_full = ffi::zeros(&[batch_size, n_kv, kv_len, head_dim], crate::dtype::FLOAT32);
    let v_full = ffi::zeros(&[batch_size, n_kv, kv_len, head_dim], crate::dtype::FLOAT32);
    let k_swa = ffi::zeros(&[batch_size, n_kv, kv_len, head_dim], crate::dtype::FLOAT32);
    let v_swa = ffi::zeros(&[batch_size, n_kv, kv_len, head_dim], crate::dtype::FLOAT32);
    let tensors: Vec<&MlxArray> = vec![
        k_full.as_ref().unwrap(),
        v_full.as_ref().unwrap(),
        k_swa.as_ref().unwrap(),
        v_swa.as_ref().unwrap(),
    ];
    let shared = SharedKv::new(&tensors);
    model.set_shared_kv(shared, 0, 0, 0).expect("set_shared_kv");

    // hidden tensor: [B, 1, backbone].
    let hidden = ffi::zeros(&[batch_size, 1, backbone], crate::dtype::FLOAT32);
    let last_bonus = vec![1_i32, 2, 3];
    let sampler = SamplingConfig::greedy();
    let out = model
        .draft_block_batched(&last_bonus, Some(hidden.as_ref().unwrap()), 4, &sampler)
        .expect("draft_block_batched must succeed");

    assert_eq!(out.len(), batch_size as usize, "B rows expected");
    for (r, row) in out.iter().enumerate() {
        assert_eq!(
            row.len(),
            3,
            "row {r}: must produce block_size - 1 = 3 proposals"
        );
    }
}

/// `draft_block_batched` rejects empty bonus vector with empty output
/// (defensive: matches the B = 1 path's `block_size == 0` handling).
#[test]
fn draft_block_batched_rejects_block_size_zero_with_empty_rows() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    let target = MockLanguageModel::new(16, 64);
    model.bind(&target).expect("bind");

    // 2 tensors = full-attention only.
    let k = ffi::zeros(&[2, 1, 4, 32], crate::dtype::FLOAT32);
    let v = ffi::zeros(&[2, 1, 4, 32], crate::dtype::FLOAT32);
    let tensors: Vec<&MlxArray> = vec![k.as_ref().unwrap(), v.as_ref().unwrap()];
    let shared = SharedKv::new(&tensors);
    model.set_shared_kv(shared, 0, 0, 0).expect("set_shared_kv");

    let last_bonus = vec![1_i32, 2];
    let sampler = SamplingConfig::greedy();
    let out = model
        .draft_block_batched(&last_bonus, None, 0, &sampler)
        .expect("block_size = 0 returns empty rows");
    assert_eq!(out.len(), 2);
    assert!(out[0].is_empty());
    assert!(out[1].is_empty());
}

/// `draft_block_batched` rejects calls before `set_shared_kv`.
#[test]
fn draft_block_batched_rejects_call_before_set_shared_kv() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    let target = MockLanguageModel::new(16, 64);
    model.bind(&target).expect("bind");

    let last_bonus = vec![1_i32, 2];
    let sampler = SamplingConfig::greedy();
    let err = model
        .draft_block_batched(&last_bonus, None, 4, &sampler)
        .expect_err("must fail");
    match err {
        DrafterError::SetSharedKvNotCalled => {}
        other => panic!("expected SetSharedKvNotCalled, got {other:?}"),
    }
}

/// `set_shared_kv` with `left_padding > 0` routes through the masks
/// normalizer (issue #631). Pins the path: drafter accepts the call
/// without trying to forward through invalid shared K/V buffers.
#[test]
fn set_shared_kv_with_left_padding_routes_through_normalizer() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    let target = MockLanguageModel::new(16, 64);
    model.bind(&target).expect("bind");

    // B=2, kv_len=8 — non-trivial enough to make the normalizer's roll
    // observable.
    let kv_len = 8_i32;
    let k = ffi::zeros(&[2, 1, kv_len, 32], crate::dtype::FLOAT32);
    let v = ffi::zeros(&[2, 1, kv_len, 32], crate::dtype::FLOAT32);
    let tensors: Vec<&MlxArray> = vec![k.as_ref().unwrap(), v.as_ref().unwrap()];
    let shared = SharedKv::new(&tensors);
    model
        .set_shared_kv(shared, 0, 0, /* left_padding */ 3)
        .expect("set_shared_kv with left_padding must succeed");
}
