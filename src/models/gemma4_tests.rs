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

//! Unit tests for Gemma 4 configuration parsing and RoPE handling.
//!
//! These tests lock in the behavior described in GitHub
//!
//! 1. Real Gemma 4 checkpoints declare `rope_type: "proportional"` on every
//!    `full_attention` layer and `rope_type: "default"` on every
//!    `sliding_attention` layer.
//! 2. Under `rope_type: "proportional"`, RoPE exponents MUST be normalized by
//!    the full head dimension (not the rotated-only slice) — matching the
//!    upstream `mlx_vlm.models.gemma4.rope_utils.ProportionalRoPE` semantics.
//! 3. Under `rope_type: "default"` (sliding-attention layers), mlxcel keeps
//!    the historical `nn.RoPE(head_dim * partial_rotary_factor)` path.

use super::gemma4::{RopeParameters, TextConfig};

fn parse_text_config(json: serde_json::Value) -> TextConfig {
    serde_json::from_value(json).expect("TextConfig must deserialize")
}

/// Re-export the tiny synthetic 1-layer Gemma 4 wrapper builder so sibling
/// test modules (e.g. `gemma4_mtp_target_tests`) can drive the *real*
/// speculative forward against a hermetic in-process model instead of
/// requiring 31B checkpoints. The fixture is a single `sliding_attention`
/// layer (hidden=4, vocab=8, `sliding_window=8`), exactly the model used by
/// the cache-isolation tests below.
pub(crate) fn build_synthetic_wrapper() -> super::Gemma4Wrapper {
    cache_isolation::build_wrapper()
}

/// Like [`build_synthetic_wrapper`] but lets the caller pick the single layer's
/// attention family (`"sliding_attention"` or `"full_attention"`). The
/// full-attention variant is the one that exercises the *unbounded* KVCache
/// verify-round left-padding mask (the sliding fixture alone cannot, since it
/// only ever drives the windowed mask path).
pub(crate) fn build_synthetic_wrapper_with_layer(layer_type: &str) -> super::Gemma4Wrapper {
    cache_isolation::build_wrapper_with_layer(layer_type)
}

// -----------------------------------------------------------------
// per-`SequenceId` cache isolation tests for `Gemma4Wrapper`.
//
// These tests build a tiny synthetic Gemma 4 model (1 layer, hidden=4,
// vocab=8, sliding-attention only) and verify that
// `Gemma4Wrapper::forward_with_sequence_id` resolves to a distinct
// per-sequence `Vec<Cache>` so a mixed-length batch cannot leak cache
// state across rows. This is the runtime fix — the
// per-row dispatch helper added only routes correctly if
// the underlying wrapper isolates cache state per `SequenceId`.
//
// The fixture is duplicated from
// `distributed::tensor_parallel::llama_runtime_tests` (`make_test_gemma4_args`
// / `make_test_gemma4_weight_map`) so the two test surfaces stay
// independent — they're both small synthetic configs, not a real
// checkpoint, and divergence between the two would be caught by the
// shared logits comparison test
// (`tensor_parallel_gemma4_matches_full_model_logits`).
mod cache_isolation {
    use crate::models::gemma4::{ModelArgs, RopeParameters as Gemma4RopeParameters};
    use crate::models::{Gemma4Model, Gemma4Wrapper};
    use mlxcel_core::cache::{SequenceId, SequenceStateBackend, SequenceStateLayout};
    use mlxcel_core::generate::LanguageModel;
    use mlxcel_core::weights::WeightMap;
    use std::collections::HashMap;

    pub(super) fn make_test_gemma4_args_with_layer(layer_type: &str) -> ModelArgs {
        let mut rope_parameters: HashMap<String, Gemma4RopeParameters> = HashMap::new();
        rope_parameters.insert(
            "sliding_attention".to_string(),
            Gemma4RopeParameters {
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                rope_type: "default".to_string(),
            },
        );
        rope_parameters.insert(
            "full_attention".to_string(),
            Gemma4RopeParameters {
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                rope_type: "default".to_string(),
            },
        );

        ModelArgs {
            model_type: "gemma4".to_string(),
            text_config: serde_json::json!({
                "model_type": "gemma4_text",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "head_dim": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 8,
                "vocab_size_per_layer_input": 0,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": null,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "rope_traditional": false,
                "rope_parameters": rope_parameters,
                "sliding_window": 8,
                "sliding_window_pattern": 1,
                "max_position_embeddings": 4096,
                "attention_k_eq_v": false,
                "final_logit_softcapping": null,
                "use_double_wide_mlp": false,
                "enable_moe_block": false,
                "num_experts": null,
                "top_k_experts": null,
                "moe_intermediate_size": null,
                "layer_types": [layer_type],
                "quantization": null
            }),
            eos_token_id: Some(serde_json::json!([1])),
            quantization: None,
        }
    }

    fn insert_tensor(weights: &mut WeightMap, name: &str, values: &[f32], shape: &[i32]) {
        weights.insert(name.to_string(), mlxcel_core::from_slice_f32(values, shape));
    }

    fn seq_values(len: usize, start: f32, step: f32) -> Vec<f32> {
        (0..len).map(|idx| start + idx as f32 * step).collect()
    }

    fn make_test_gemma4_weight_map() -> WeightMap {
        let mut weights = HashMap::new();
        insert_tensor(
            &mut weights,
            "language_model.model.embed_tokens.weight",
            &seq_values(32, 0.0, 0.1),
            &[8, 4],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.self_attn.q_proj.weight",
            &[
                0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7,
            ],
            &[4, 4],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.self_attn.k_proj.weight",
            &[0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.3],
            &[2, 4],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.self_attn.v_proj.weight",
            &[0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.20, 0.25],
            &[2, 4],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.self_attn.o_proj.weight",
            &[
                0.20, 0.10, 0.30, 0.40, 0.10, 0.30, 0.20, 0.40, 0.40, 0.30, 0.10, 0.20, 0.30, 0.40,
                0.20, 0.10,
            ],
            &[4, 4],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.self_attn.q_norm.weight",
            &[1.0, 1.0],
            &[2],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.self_attn.k_norm.weight",
            &[1.0, 1.0],
            &[2],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.mlp.gate_proj.weight",
            &seq_values(32, 0.01, 0.01),
            &[8, 4],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.mlp.up_proj.weight",
            &seq_values(32, 0.02, 0.01),
            &[8, 4],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.mlp.down_proj.weight",
            &seq_values(32, 0.03, 0.01),
            &[4, 8],
        );
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            insert_tensor(
                &mut weights,
                &format!("language_model.model.layers.0.{norm}.weight"),
                &[1.0, 1.0, 1.0, 1.0],
                &[4],
            );
        }
        insert_tensor(
            &mut weights,
            "language_model.model.layers.0.layer_scalar",
            &[1.0],
            &[1],
        );
        insert_tensor(
            &mut weights,
            "language_model.model.norm.weight",
            &[1.0, 1.0, 1.0, 1.0],
            &[4],
        );
        weights
    }

    pub(super) fn build_wrapper() -> Gemma4Wrapper {
        build_wrapper_with_layer("sliding_attention")
    }

    pub(super) fn build_wrapper_with_layer(layer_type: &str) -> Gemma4Wrapper {
        let args = make_test_gemma4_args_with_layer(layer_type);
        let weights = make_test_gemma4_weight_map();
        Gemma4Wrapper::new(Gemma4Model::from_weights(&weights, &args).unwrap())
    }

    fn array_to_vec_f32(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
        let arr_f32 = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&arr_f32);
        let bytes = mlxcel_core::array_to_raw_bytes(&arr_f32);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// `Gemma4Wrapper` must declare `supports_batching == true`
    /// so the server scheduler actually drives the batched-decode dispatch
    /// path that calls `forward_with_sequence_id` per row.
    #[test]
    fn gemma4_wrapper_supports_batching() {
        let wrapper = build_wrapper();
        assert!(
            wrapper.supports_batching(),
            "Gemma4Wrapper must support batching after (server batched decode \
             requires per-`SequenceId` cache isolation, which the wrapper now provides via \
             ModelOwnedSequenceState<Cache>)"
        );
    }

    /// `Gemma4Wrapper` must declare a `ModelOwned` sequence
    /// state layout so the cache pool allocates a placeholder
    /// (empty `Vec<KVCache>`) per sequence and the wrapper itself owns
    /// the real `Vec<Cache>` keyed on `SequenceId`. A `DenseKvCache`
    /// layout would crash because Gemma 4's sliding-window
    /// `RotatingKVCache` cannot be wired to scheduler-managed `KVCache`
    /// slices.
    #[test]
    fn gemma4_wrapper_uses_model_owned_sequence_state_layout() {
        let wrapper = build_wrapper();
        let layout = wrapper.sequence_state_layout();
        assert_eq!(
            layout.backend,
            SequenceStateBackend::ModelOwned,
            "Gemma4Wrapper layout backend must be ModelOwned, got {:?}",
            layout.backend
        );
        let expected = SequenceStateLayout::model_owned(1);
        assert_eq!(
            layout.backend, expected.backend,
            "layout helper must emit a ModelOwned descriptor"
        );
        assert!(
            wrapper.make_caches().is_empty(),
            "Gemma4Wrapper::make_caches() must return an empty Vec for ModelOwned layout — \
             the scheduler-side cache pool relies on this to allocate placeholder entries \
             instead of full per-layer KV slices"
        );
    }

    /// per-`SequenceId` cache isolation. Two sequences with
    /// distinct `SequenceId`s must produce row-correct logits even when
    /// driven through the same wrapper instance back-to-back.
    ///
    /// Prior to the wrapper held a single `RefCell<Vec<Cache>>`
    /// shared across every call, so seq B's first decode step would
    /// inherit seq A's KV state and produce wrong logits. After the
    /// fix, each `SequenceId` resolves to its own slot in
    /// `ModelOwnedSequenceState<Cache>` and the two rows behave
    /// independently.
    ///
    /// We assert this by:
    /// 1. Prefilling seq A (offset 0 → 2) and seq B (offset 0 → 2)
    ///    independently with different prompts.
    /// 2. Running one decode step on seq A, then one on seq B.
    /// 3. Comparing seq A's *re-run* logits (in a fresh wrapper, single
    ///    sequence, identical prompt + decode token) against the seq A
    ///    logits computed in step 2. They must match within tolerance —
    ///    if seq B's prefill leaked into seq A's cache, the logits in
    ///    step 2 would be polluted.
    #[test]
    #[ignore = "requires serial MLX execution"]
    fn gemma4_per_sequence_cache_isolation_no_cross_contamination() {
        // ---- Reference (single-sequence) run for seq A only. ----
        let reference = build_wrapper();
        let seq_a_ref = SequenceId::from_raw(700);
        reference.prepare_sequence_state(seq_a_ref);

        // Prefill seq A: tokens [3, 4]
        let prefill_a = mlxcel_core::from_slice_i32(&[3, 4], &[1, 2]);
        let _ = reference.forward_with_sequence_id(&prefill_a, Some(seq_a_ref), &mut [], None);

        // Decode step: token [5]
        let decode_a = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
        let logits_a_reference =
            reference.forward_with_sequence_id(&decode_a, Some(seq_a_ref), &mut [], None);
        let logits_a_reference_vec = array_to_vec_f32(logits_a_reference.as_ref().unwrap());

        // ---- Mixed-batch wrapper running BOTH seq A and seq B. ----
        let mixed = build_wrapper();
        let seq_a = SequenceId::from_raw(701);
        let seq_b = SequenceId::from_raw(702);
        mixed.prepare_sequence_state(seq_a);
        mixed.prepare_sequence_state(seq_b);

        // Prefill seq A: tokens [3, 4]
        let prefill_a_mixed = mlxcel_core::from_slice_i32(&[3, 4], &[1, 2]);
        let _ = mixed.forward_with_sequence_id(&prefill_a_mixed, Some(seq_a), &mut [], None);

        // Prefill seq B: longer/different tokens [1, 2, 6]
        // explicitly requires the case where the two prompts differ in
        // length. If cache state leaks, seq A's offset would be stomped.
        let prefill_b_mixed = mlxcel_core::from_slice_i32(&[1, 2, 6], &[1, 3]);
        let _ = mixed.forward_with_sequence_id(&prefill_b_mixed, Some(seq_b), &mut [], None);

        // Now interleave decode: seq A decodes one step, seq B decodes
        // one step. The order does not matter — what matters is that
        // seq A's logits are unaffected by seq B's prior prefill.
        let decode_a_mixed = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
        let logits_a_mixed =
            mixed.forward_with_sequence_id(&decode_a_mixed, Some(seq_a), &mut [], None);
        let decode_b_mixed = mlxcel_core::from_slice_i32(&[7], &[1, 1]);
        let _logits_b_mixed =
            mixed.forward_with_sequence_id(&decode_b_mixed, Some(seq_b), &mut [], None);

        let logits_a_mixed_vec = array_to_vec_f32(logits_a_mixed.as_ref().unwrap());

        assert_eq!(
            logits_a_reference_vec.len(),
            logits_a_mixed_vec.len(),
            "logits length must match between reference and mixed runs"
        );
        for (i, (&got, &want)) in logits_a_mixed_vec
            .iter()
            .zip(logits_a_reference_vec.iter())
            .enumerate()
        {
            let abs_err = (got - want).abs();
            let rel_err = abs_err / want.abs().max(1.0);
            assert!(
                abs_err < 1e-3 || rel_err < 1e-3,
                "logit[{i}] differs: mixed={got} vs reference={want} \
                 (abs={abs_err}, rel={rel_err}); seq B prefill must not leak \
                 into seq A's per-sequence cache slot"
            );
        }

        // Cleanup so the per-sequence map does not hold the cache state.
        mixed.release_sequence_state_by_id(seq_a);
        mixed.release_sequence_state_by_id(seq_b);
        reference.release_sequence_state_by_id(seq_a_ref);
    }

    /// acceptance criterion 1: a batch of two Gemma 4
    /// requests with different prompt lengths produces output for each
    /// row that matches the unbatched baseline within tolerance.
    ///
    /// This drives the per-row dispatch helper
    /// (`crate::multimodal::batched_dispatch::forward_batched_with_seq_ids_dispatch`)
    /// directly against a real `Gemma4Wrapper` to demonstrate the
    /// end-to-end fix is reachable. After this PR the server scheduler
    /// reaches this same code path because
    /// `Gemma4VLModel::supports_batching() == true` and its
    /// `forward_batched_with_context_and_ids` override delegates to
    /// the same helper.
    #[test]
    #[ignore = "requires serial MLX execution"]
    fn gemma4_mixed_length_batched_decode_matches_unbatched_baseline() {
        use crate::multimodal::batched_dispatch::forward_batched_with_seq_ids_dispatch;

        // ---- Unbatched baselines for two distinct sequences. ----
        let baseline = build_wrapper();
        let seq_a = SequenceId::from_raw(900);
        let seq_b = SequenceId::from_raw(901);
        baseline.prepare_sequence_state(seq_a);
        baseline.prepare_sequence_state(seq_b);

        // Different-length prompts: the explicit reproducer.
        let prompt_a = mlxcel_core::from_slice_i32(&[3, 4], &[1, 2]);
        let prompt_b = mlxcel_core::from_slice_i32(&[1, 2, 6], &[1, 3]);
        let _ = baseline.forward_with_sequence_id(&prompt_a, Some(seq_a), &mut [], None);
        let _ = baseline.forward_with_sequence_id(&prompt_b, Some(seq_b), &mut [], None);

        // Decode one token per sequence under the baseline (sequential).
        let decode_a = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
        let decode_b = mlxcel_core::from_slice_i32(&[7], &[1, 1]);
        let baseline_a = baseline.forward_with_sequence_id(&decode_a, Some(seq_a), &mut [], None);
        let baseline_b = baseline.forward_with_sequence_id(&decode_b, Some(seq_b), &mut [], None);
        let baseline_a_vec = array_to_vec_f32(baseline_a.as_ref().unwrap());
        let baseline_b_vec = array_to_vec_f32(baseline_b.as_ref().unwrap());

        // ---- Batched run via the per-row dispatch helper. ----
        let batched = build_wrapper();
        let seq_a_b = SequenceId::from_raw(910);
        let seq_b_b = SequenceId::from_raw(911);
        batched.prepare_sequence_state(seq_a_b);
        batched.prepare_sequence_state(seq_b_b);

        // Prefill is still per-sequence — the batched-prefill path is
        // gated on `supports_batched_prefill` (false for Gemma 4) so the
        // scheduler walks each sequence individually first.
        let _ = batched.forward_with_sequence_id(&prompt_a, Some(seq_a_b), &mut [], None);
        let _ = batched.forward_with_sequence_id(&prompt_b, Some(seq_b_b), &mut [], None);

        // Now drive the BATCHED decode through the dispatch helper —
        // this is the exact code path the scheduler now reaches via
        // `execute_batched_decode -> forward_batched_with_context_and_ids`
        // -> `Gemma4VLModel`'s override -> this helper.
        let decode_batched = mlxcel_core::from_slice_i32(&[5, 7], &[2, 1]);
        let mut row_a_caches: Vec<mlxcel_core::layers::KVCache> = Vec::new();
        let mut row_b_caches: Vec<mlxcel_core::layers::KVCache> = Vec::new();
        let mut batch_caches: Vec<&mut [mlxcel_core::layers::KVCache]> =
            vec![row_a_caches.as_mut_slice(), row_b_caches.as_mut_slice()];
        let seq_ids = [seq_a_b, seq_b_b];
        let logits_batched = forward_batched_with_seq_ids_dispatch(
            &batched,
            &decode_batched,
            Some(&seq_ids),
            batch_caches.as_mut_slice(),
            None,
            None,
        );
        let shape = mlxcel_core::array_shape(&logits_batched);
        assert_eq!(
            shape,
            vec![2, 1, 8],
            "batched decode logits must be shape [B=2, T=1, V=8]"
        );

        // Slice out per-row logits and compare to the per-sequence baseline.
        let row_a = mlxcel_core::slice(&logits_batched, &[0, 0, 0], &[1, 1, 8]);
        let row_b = mlxcel_core::slice(&logits_batched, &[1, 0, 0], &[2, 1, 8]);
        let row_a_vec = array_to_vec_f32(row_a.as_ref().unwrap());
        let row_b_vec = array_to_vec_f32(row_b.as_ref().unwrap());

        for (i, (&got, &want)) in row_a_vec.iter().zip(baseline_a_vec.iter()).enumerate() {
            let abs_err = (got - want).abs();
            let rel_err = abs_err / want.abs().max(1.0);
            assert!(
                abs_err < 1e-3 || rel_err < 1e-3,
                "row A logit[{i}] differs: batched={got} vs unbatched={want} \
                 (abs={abs_err}, rel={rel_err}) — acceptance criterion 1 \
                 violated"
            );
        }
        for (i, (&got, &want)) in row_b_vec.iter().zip(baseline_b_vec.iter()).enumerate() {
            let abs_err = (got - want).abs();
            let rel_err = abs_err / want.abs().max(1.0);
            assert!(
                abs_err < 1e-3 || rel_err < 1e-3,
                "row B logit[{i}] differs: batched={got} vs unbatched={want} \
                 (abs={abs_err}, rel={rel_err}) — acceptance criterion 1 \
                 violated"
            );
        }

        baseline.release_sequence_state_by_id(seq_a);
        baseline.release_sequence_state_by_id(seq_b);
        batched.release_sequence_state_by_id(seq_a_b);
        batched.release_sequence_state_by_id(seq_b_b);
    }

    /// `release_sequence_state_by_id` must drop the per-sequence cache
    /// slot. After release, `forward_with_sequence_id` for the same
    /// `SequenceId` must behave as if it were a fresh sequence (cache
    /// offset 0 — i.e. a re-prefill).
    ///
    /// This guards the scheduler's cleanup path
    /// (`scheduler.rs::release_sequence_caches`) which calls
    /// `release_sequence_state_by_id` after a sequence finishes.
    #[test]
    #[ignore = "requires serial MLX execution"]
    fn gemma4_release_sequence_state_drops_cached_state() {
        let wrapper = build_wrapper();
        let seq_id = SequenceId::from_raw(800);
        wrapper.prepare_sequence_state(seq_id);

        // Prefill some tokens to populate the per-seq cache slot.
        let prefill = mlxcel_core::from_slice_i32(&[3, 4, 5], &[1, 3]);
        let _ = wrapper.forward_with_sequence_id(&prefill, Some(seq_id), &mut [], None);

        // Capture decode logits BEFORE release.
        let decode = mlxcel_core::from_slice_i32(&[6], &[1, 1]);
        let logits_before = wrapper.forward_with_sequence_id(&decode, Some(seq_id), &mut [], None);
        let logits_before_vec = array_to_vec_f32(logits_before.as_ref().unwrap());

        // Release the slot.
        wrapper.release_sequence_state_by_id(seq_id);

        // Re-running the SAME prefill + decode under the same SequenceId
        // (now stale) must produce identical logits to a brand-new
        // prefill+decode in a fresh wrapper, since the slot was dropped.
        wrapper.prepare_sequence_state(seq_id);
        let _ = wrapper.forward_with_sequence_id(&prefill, Some(seq_id), &mut [], None);
        let logits_after = wrapper.forward_with_sequence_id(&decode, Some(seq_id), &mut [], None);
        let logits_after_vec = array_to_vec_f32(logits_after.as_ref().unwrap());

        assert_eq!(
            logits_before_vec.len(),
            logits_after_vec.len(),
            "logits length must match across release+reprefill"
        );
        for (i, (&before, &after)) in logits_before_vec
            .iter()
            .zip(logits_after_vec.iter())
            .enumerate()
        {
            let abs_err = (before - after).abs();
            let rel_err = abs_err / before.abs().max(1.0);
            assert!(
                abs_err < 1e-3 || rel_err < 1e-3,
                "logit[{i}] differs after release+reprefill: before={before} vs after={after} \
                 (abs={abs_err}, rel={rel_err}); release_sequence_state_by_id must drop \
                 per-seq cache state cleanly"
            );
        }

        wrapper.release_sequence_state_by_id(seq_id);
    }
}

/// Minimal text_config mirroring the Gemma 4 E2B real checkpoint
/// (trimmed to fields relevant for RoPE / layer-type dispatch).
fn real_gemma4_e2b_text_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "intermediate_size": 6144,
        "num_attention_heads": 8,
        "head_dim": 256,
        "global_head_dim": 512,
        "rms_norm_eps": 1e-6,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 20,
        "hidden_size_per_layer_input": 256,
        "sliding_window": 512,
        "max_position_embeddings": 131072,
        "use_double_wide_mlp": true,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default"
            }
        },
        // Real layer pattern from the checkpoint — 4 sliding then 1 full, repeated.
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention"
        ]
    })
}

#[test]
fn gemma4_config_parses_real_checkpoint_rope_parameters() {
    // The primary regression target for make sure we can in fact
    // read `rope_type` out of the real checkpoint config without erroring,
    // and that both per-layer-type entries deserialize correctly.
    let cfg = parse_text_config(real_gemma4_e2b_text_config());

    let full = cfg
        .rope_parameters
        .get("full_attention")
        .expect("full_attention rope params must be present");
    assert_eq!(full.rope_type, "proportional");
    assert!((full.partial_rotary_factor - 0.25).abs() < 1e-6);
    assert!((full.rope_theta - 1_000_000.0).abs() < 1e-3);

    let sliding = cfg
        .rope_parameters
        .get("sliding_attention")
        .expect("sliding_attention rope params must be present");
    assert_eq!(sliding.rope_type, "default");
    // Sliding entries in the real checkpoint omit `partial_rotary_factor`;
    // the serde default should be 1.0.
    assert!((sliding.partial_rotary_factor - 1.0).abs() < 1e-6);
    assert!((sliding.rope_theta - 10_000.0).abs() < 1e-3);
}

#[test]
fn gemma4_rope_parameters_rope_type_defaults_when_absent() {
    // Older / simpler configs that omit `rope_type` entirely must still
    // deserialize and default to "default".
    let params: RopeParameters = serde_json::from_value(serde_json::json!({
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0
    }))
    .expect("RopeParameters must deserialize without rope_type");
    assert_eq!(params.rope_type, "default");
}

#[test]
fn gemma4_proportional_rope_freqs_match_python_semantics() {
    // Lock in the numerical semantics Case A:
    //
    //   freqs[i] = base^(2 * i / head_dim)   for i in [0, rope_angles)
    //
    // with rope_angles = int(partial_rotary_factor * head_dim / 2), followed
    // by an `inf` tail that disables rotation for the remaining pairs. The
    // denominator is the FULL head_dim — this is what distinguishes
    // "proportional" RoPE from the default `nn.RoPE(rope_dims)` form.
    //
    // If this test regresses, it means the RoPE frequencies diverged from
    // upstream `mlx_vlm.models.gemma4.rope_utils.ProportionalRoPE`, which
    // is exactly the hazard that motivated.
    let head_dim = 256_i32;
    let prf = 0.25_f32;
    let base = 1_000_000.0_f32;
    let factor = 1.0_f32;

    let freqs = mlxcel_core::rope_proportional::compute_proportional_rope_freqs(
        head_dim, prf, base, factor,
    )
    .expect("freqs must exist for prf=0.25");
    mlxcel_core::eval(&freqs);

    // For head_dim=256 and prf=0.25, rope_angles = 32, but upstream pads the
    // table to head_dim/2 with `inf`.
    assert_eq!(
        mlxcel_core::array_shape(&freqs),
        vec![128],
        "freqs length must equal head_dim / 2"
    );

    // Pull the values back to host and spot-check a handful of entries.
    let freqs_f32 = mlxcel_core::astype(&freqs, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&freqs_f32);
    let freq_bytes = mlxcel_core::array_to_raw_bytes(&freqs_f32);
    assert_eq!(freq_bytes.len(), 128 * 4);
    let freq_values: Vec<f32> = freq_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    assert_eq!(freq_values.len(), 128);

    for (i, &got) in freq_values.iter().take(32).enumerate() {
        let expected = base.powf((2 * i) as f32 / head_dim as f32);
        let rel = (got - expected).abs() / expected.max(1.0);
        assert!(
            rel < 1e-4,
            "freqs[{i}] expected {expected}, got {got} (rel err {rel})"
        );
    }
    for (i, &got) in freq_values.iter().enumerate().skip(32) {
        assert!(
            got.is_infinite() && got.is_sign_positive(),
            "freqs[{i}] must be +inf"
        );
    }

    // Sanity: the second-to-last entry is noticeably smaller than
    // `base^(rotated_dims/head_dim)` in the default (non-proportional) form.
    // If mlxcel regressed to the default form, exponents would be normalized
    // by `rope_dims = 64` instead of `head_dim = 256`, giving
    //     default[i=15] = base^(30/64)  ≈ 1148
    //     proportional[i=15] = base^(30/256) ≈ 14.7
    // i.e. a ~78x larger value — a regression would be immediately obvious.
    let default_formula = base.powf(30.0 / 64.0);
    assert!(
        freq_values[15] < default_formula / 10.0,
        "freqs[15]={}, should be far smaller than default-RoPE formula ({}); \
         likely regression to non-proportional semantics",
        freq_values[15],
        default_formula,
    );
}

// -----------------------------------------------------------------
// Gemma 4 MTP target hooks — `rollback_speculative_cache`
// + sink-aware forward.
//
// The tests below cover the cache-rewind primitives in isolation
// (`RotatingKVCache::trim` + `Cache::zero_partial_accept_tail`) and
// then exercise the sink path end-to-end through the synthetic
// 1-layer Gemma 4 fixture defined in the `cache_isolation` sub-module
// above (the same fixture's batching tests use). The
// sink-path test is gated on serial MLX execution because the
// fixture's forward pass touches the global MLX runtime.
mod mtp_hooks {
    use crate::models::gemma4::Cache;
    use mlxcel_core::layers::{KVCache, RotatingKVCache};

    /// `RotatingKVCache::trim(n)` must rewind both the monotonic
    /// `offset` and the internal write index (`buffer_write_idx`)
    /// — Python's `trim` rewinds both `self.offset` and `self._idx`.
    /// Without rewinding `_idx`, a subsequent `update_and_fetch` would
    /// overwrite the WRONG buffer slot after rollback.
    #[test]
    fn rotating_kv_cache_trim_rewinds_offset_and_idx() {
        let mut cache = RotatingKVCache::new(32);
        // Prefill 4 tokens via single-token updates (each goes through
        // `update_in_place`, advancing `offset` and `idx` by 1 apiece).
        let val = |x: f32| mlxcel_core::from_slice_f32(&[x], &[1, 1, 1, 1]);
        cache.update_and_fetch(val(1.0), val(10.0));
        cache.update_and_fetch(val(2.0), val(20.0));
        cache.update_and_fetch(val(3.0), val(30.0));
        cache.update_and_fetch(val(4.0), val(40.0));
        assert_eq!(cache.offset, 4);
        assert_eq!(cache.buffer_write_idx(), 4);

        // Trim the last 3 tokens (mirrors a verify pass where only 1
        // of 4 speculated tokens was accepted: trim = block_size - n =
        // 4 - 1 = 3).
        let n_trimmed = cache.trim(3);
        assert_eq!(n_trimmed, 3);
        assert_eq!(cache.offset, 1);
        assert_eq!(
            cache.buffer_write_idx(),
            1,
            "trim must rewind buffer_write_idx so the next update overwrites the correct slot"
        );

        // Trim more than what's live: clamps to the live window.
        let n_trimmed = cache.trim(10);
        assert_eq!(n_trimmed, 1);
        assert_eq!(cache.offset, 0);
        assert_eq!(cache.buffer_write_idx(), 0);

        // Trim on an empty cache: no-op.
        let n_trimmed = cache.trim(5);
        assert_eq!(n_trimmed, 0);
        assert_eq!(cache.offset, 0);
        assert_eq!(cache.buffer_write_idx(), 0);

        // Negative / zero argument: no-op (matches `if n <= 0` guard).
        let mut cache2 = RotatingKVCache::new(32);
        cache2.update_and_fetch(val(1.0), val(10.0));
        cache2.update_and_fetch(val(2.0), val(20.0));
        let pre_offset = cache2.offset;
        let pre_idx = cache2.buffer_write_idx();
        assert_eq!(cache2.trim(0), 0);
        assert_eq!(cache2.trim(-3), 0);
        assert_eq!(cache2.offset, pre_offset);
        assert_eq!(cache2.buffer_write_idx(), pre_idx);
    }

    /// Read the dense scalar at index `[b, h, t, d]` from a 4-D MLX
    /// array, casting to f32. The cache buffers in the tests below
    /// live in fp16 (the default `KVCacheMode::Fp16` path); fetching
    /// through f32 is the cheapest way to verify per-cell content
    /// without depending on bf16/f16 binary representations.
    fn at(arr: &mlxcel_core::MlxArray, indices: &[i32; 4]) -> f32 {
        let stops = [
            indices[0] + 1,
            indices[1] + 1,
            indices[2] + 1,
            indices[3] + 1,
        ];
        let cell = mlxcel_core::slice(arr, indices, &stops);
        let cell_f32 = mlxcel_core::astype(&cell, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::item_f32(&cell_f32)
    }

    /// Build a [B, H=1, max_size, D=1] rotating cache primed with
    /// monotonically increasing values per row so we can detect which
    /// cells got zeroed. Row 0 uses positive values (1..=max_size);
    /// row 1 uses negative values (-1..=-max_size). The distinct sign
    /// per row makes assertions readable.
    ///
    /// Strategy: drive `update_and_fetch` with junk values `write_idx`
    /// times so the internal `idx` advances to the desired write
    /// position (and the rotating-cache state machine is in a known
    /// good state), then overwrite the now-allocated `keys` / `values`
    /// buffers with our primed content. `idx == offset` while the
    /// buffer hasn't wrapped (the common rollback regime), which is
    /// what every Gemma 4 MTP rollback call hits.
    fn build_primed_rotating_cache(max_size: i32, write_idx: i32, batch: i32) -> RotatingKVCache {
        let total = (batch * max_size) as usize;
        let mut k_vals: Vec<f32> = Vec::with_capacity(total);
        let mut v_vals: Vec<f32> = Vec::with_capacity(total);
        for bi in 0..batch {
            for t in 0..max_size {
                let sign: f32 = if bi == 0 { 1.0 } else { -1.0 };
                k_vals.push(sign * (t as f32 + 1.0));
                v_vals.push(sign * 10.0 * (t as f32 + 1.0));
            }
        }
        let shape = [batch, 1, max_size, 1];

        let mut warmed = RotatingKVCache::new(max_size);
        let val = |x: f32| {
            let v: Vec<f32> = vec![x; batch as usize];
            mlxcel_core::from_slice_f32(&v, &[batch, 1, 1, 1])
        };
        for _ in 0..write_idx {
            warmed.update_and_fetch(val(0.0), val(0.0));
        }
        warmed.keys = Some(mlxcel_core::from_slice_f32(&k_vals, &shape));
        warmed.values = Some(mlxcel_core::from_slice_f32(&v_vals, &shape));
        debug_assert_eq!(warmed.offset, write_idx);
        debug_assert_eq!(warmed.buffer_write_idx(), write_idx);
        warmed
    }

    /// Per-row tail-zero must zero ONLY rows whose accept count is
    /// below `max(accepted)`. With block_size = 4, accepted = [3, 1]:
    /// max_a = 3, n = 4, verify_start = kv_len - 4 = 0, ve = [4, 2].
    /// Row 0: start = 0 + 4 = 4 == kv_len -> no zero (full-accept).
    /// Row 1: start = 0 + 2 = 2 < 4 -> zero positions [2, 4) in row 1.
    #[test]
    #[ignore = "requires serial MLX execution"]
    fn cache_zero_partial_accept_tail_zeros_only_partial_rows_in_rotating() {
        let warmed = build_primed_rotating_cache(
            /*max_size=*/ 8, /*write_idx=*/ 4, /*batch=*/ 2,
        );
        let mut cache = Cache::Rotating(warmed);
        let valid_ends = vec![4i32, 2i32];
        cache
            .zero_partial_accept_tail(&valid_ends, /*block_size=*/ 4)
            .expect("zero_partial_accept_tail must succeed on well-formed input");

        let rotating = match &cache {
            Cache::Rotating(c) => c,
            _ => unreachable!(),
        };
        let keys = rotating.keys.as_ref().unwrap();
        let values = rotating.values.as_ref().unwrap();

        // Row 0: untouched — original values at positions 0..4 are
        // 1.0, 2.0, 3.0, 4.0 in keys (positive sign).
        for t in 0..4 {
            let k = at(keys, &[0, 0, t, 0]);
            let v = at(values, &[0, 0, t, 0]);
            assert!(
                (k - (t as f32 + 1.0)).abs() < 1e-3,
                "row 0 K[{t}] must be unchanged, got {k}, expected {}",
                t as f32 + 1.0
            );
            assert!(
                (v - 10.0 * (t as f32 + 1.0)).abs() < 1e-3,
                "row 0 V[{t}] must be unchanged, got {v}"
            );
        }
        // Row 1: positions 0..2 untouched (still negative-signed),
        // positions 2..4 zeroed.
        for t in 0..2 {
            let k = at(keys, &[1, 0, t, 0]);
            let v = at(values, &[1, 0, t, 0]);
            assert!(
                (k - (-(t as f32 + 1.0))).abs() < 1e-3,
                "row 1 K[{t}] must be unchanged (pre-verify), got {k}"
            );
            assert!(
                (v - (-10.0 * (t as f32 + 1.0))).abs() < 1e-3,
                "row 1 V[{t}] must be unchanged (pre-verify), got {v}"
            );
        }
        for t in 2..4 {
            let k = at(keys, &[1, 0, t, 0]);
            let v = at(values, &[1, 0, t, 0]);
            assert!(
                k.abs() < 1e-3,
                "row 1 K[{t}] must be zeroed (partial-accept tail), got {k}"
            );
            assert!(
                v.abs() < 1e-3,
                "row 1 V[{t}] must be zeroed (partial-accept tail), got {v}"
            );
        }
    }

    /// `Cache::Standard` (dense `KVCache`) must be a no-op for
    /// `zero_partial_accept_tail` — Python's hook gates per-row
    /// zeroing on `hasattr(c, "_idx")`, which only `RotatingKVCache`
    /// exposes. The dense cache's monotonic trim is sufficient on its
    /// own because subsequent decode steps overwrite the trimmed
    /// slots.
    #[test]
    fn cache_zero_partial_accept_tail_is_no_op_for_standard_kv() {
        let mut cache = Cache::Standard(KVCache::new());
        // Empty standard cache: still a no-op (no err).
        let valid_ends = vec![1i32, 2i32];
        cache
            .zero_partial_accept_tail(&valid_ends, 4)
            .expect("Standard cache must accept zero_partial_accept_tail as a no-op");
    }

    /// Sink-aware forward must populate `hidden_sink` (last-layer
    /// pre-norm) and `shared_kv_sink["sliding_attention"]` (the only
    /// attention type in the synthetic 1-layer fixture). Asserts on
    /// shapes only — the synthetic weights are not tuned for any
    /// specific output value, but the shape contract is what the MTP
    /// drafter binds against.
    #[test]
    #[ignore = "requires serial MLX execution"]
    fn gemma4_speculative_sinks_populates_hidden_and_shared_kv() {
        use super::cache_isolation::build_wrapper;
        use crate::models::Gemma4SpeculativeSinks;
        use mlxcel_core::cache::SequenceId;
        use mlxcel_core::generate::LanguageModel;

        let wrapper = build_wrapper();
        let seq = SequenceId::from_raw(900);
        wrapper.prepare_sequence_state(seq);

        let mut sinks = Gemma4SpeculativeSinks::with_hidden_and_shared_kv();

        // Prefill 3 tokens so the sliding-attention KV cache has 3
        // entries to expose to the sink.
        let prefill = mlxcel_core::from_slice_i32(&[3, 4, 5], &[1, 3]);
        let logits = wrapper.forward_with_speculative_sinks(
            &prefill,
            /*input_embeddings=*/ None,
            /*per_layer_inputs=*/ None,
            /*mask=*/ None,
            Some(seq),
            /*capture_layer_ids=*/ None,
            Some(&mut sinks),
        );
        mlxcel_core::eval(&logits);

        // Logits shape: [B=1, L=3, vocab=8].
        let logit_shape = mlxcel_core::array_shape(&logits);
        assert_eq!(logit_shape, vec![1, 3, 8]);

        // Hidden sink: last-layer pre-norm, [B=1, L=3, hidden=4].
        let hidden = sinks
            .hidden_sink
            .as_ref()
            .expect("hidden_sink must be Some when with_hidden_and_shared_kv");
        assert_eq!(
            hidden.len(),
            1,
            "with no capture_layer_ids, exactly one hidden capture (last layer pre-norm) is expected"
        );
        assert_eq!(mlxcel_core::array_shape(&hidden[0]), vec![1, 3, 4]);

        // Shared K/V sink: sliding-attention only (the synthetic
        // fixture has 0 full-attention layers).
        let kv = sinks
            .shared_kv_sink
            .as_ref()
            .expect("shared_kv_sink must be Some when with_hidden_and_shared_kv");
        let (k, v) = kv
            .get("sliding_attention")
            .expect("sliding_attention K/V slab must be captured");
        // Rotating cache buffer shape: [B=1, num_kv_heads=1, sliding_window=8, head_dim=2].
        let k_shape = mlxcel_core::array_shape(k);
        let v_shape = mlxcel_core::array_shape(v);
        assert_eq!(k_shape[0], 1, "shared-K batch must be 1");
        assert_eq!(k_shape[1], 1, "shared-K num_kv_heads must be 1");
        assert_eq!(k_shape[3], 2, "shared-K head_dim must be 2");
        assert_eq!(v_shape, k_shape, "shared-V shape must match shared-K");
        assert!(
            !kv.contains_key("full_attention"),
            "fixture has zero full_attention layers, so the full_attention entry must be absent"
        );
    }
}
