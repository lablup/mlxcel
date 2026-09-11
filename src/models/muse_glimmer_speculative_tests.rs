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

//! Muse Glimmer DFlash target tests (issue #1343) on a synthetic two-layer
//! decoder (one sliding, one full layer) with non-constant weights.

use super::*;
use crate::models::embedding_test_support::mlx_test_guard;
use mlxcel_core::drafter::dflash::SpeculativeTarget;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;

const SPEC_HIDDEN: usize = 4;
const SPEC_VOCAB: usize = 8;

fn spec_config(window: usize) -> MuseGlimmerTextConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "muse_glimmer_text",
        "hidden_size": SPEC_HIDDEN,
        "intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "vocab_size": SPEC_VOCAB,
        "tie_word_embeddings": false,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": window,
        "qk_scale_factor": 1.0,
        "output_multiplier": 0.2,
        "final_logit_softcapping": 20.0,
        "layer_rope_theta": [500000.0, null]
    }))
    .unwrap()
}

/// Deterministic, non-constant fill so positions and tokens are told apart.
fn varied(shape: &[i32], seed: usize, scale: f32) -> UniquePtr<MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    let values: Vec<f32> = (0..len)
        .map(|i| (((i * 7919 + seed * 104_729) % 211) as f32 / 211.0 - 0.5) * scale)
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn spec_weights(config: &MuseGlimmerTextConfig) -> WeightMap {
    let h = config.hidden_size as i32;
    let v = config.vocab_size as i32;
    let inter = config.intermediate_size as i32;
    let q_out = (config.num_attention_heads * config.head_dim) as i32;
    let kv_out = (config.num_key_value_heads * config.head_dim) as i32;
    let root = "model.language_model";
    let mut w = WeightMap::new();
    w.insert(
        format!("{root}.embed_tokens.weight"),
        varied(&[v, h], 1, 1.0),
    );
    w.insert("lm_head.weight".to_string(), varied(&[v, h], 2, 1.0));
    w.insert(format!("{root}.norm.weight"), varied(&[h], 3, 0.2));
    for layer in 0..config.num_hidden_layers {
        let p = format!("{root}.layers.{layer}");
        for (i, norm) in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        .iter()
        .enumerate()
        {
            w.insert(
                format!("{p}.{norm}.weight"),
                varied(&[h], 10 + i + layer, 0.2),
            );
        }
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            varied(&[q_out, h], 20 + layer, 0.6),
        );
        w.insert(
            format!("{p}.self_attn.gate_proj.weight"),
            varied(&[q_out, h], 25 + layer, 0.6),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            varied(&[h, q_out], 30 + layer, 0.6),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            varied(&[kv_out, h], 35 + layer, 0.6),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            varied(&[kv_out, h], 40 + layer, 0.6),
        );
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            varied(&[inter, h], 45 + layer, 0.6),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            varied(&[inter, h], 50 + layer, 0.6),
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            varied(&[h, inter], 55 + layer, 0.6),
        );
    }
    w
}

fn spec_model(window: usize) -> MuseGlimmerTextModel {
    let config = spec_config(window);
    let weights = spec_weights(&config);
    MuseGlimmerTextModel::from_weights(
        &weights,
        &config,
        "model.language_model",
        "lm_head",
        vec![200_001, 200_008],
        vec![200_092, 200_091, 200_018],
    )
    .unwrap()
}

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let diff = mlxcel_core::subtract(a, b);
    mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(&diff)))
}

fn offsets(caches: &[MuseCache]) -> Vec<i32> {
    caches.iter().map(MuseCache::offset).collect()
}

#[test]
fn verify_forward_captures_layers_before_final_norm() {
    let _guard = mlx_test_guard();
    let model = spec_model(4);
    let mut caches = model.make_speculative_caches();
    let bs = 4;
    let out = model.forward_speculative(&ids(&[1, 2, 3, 4]), &mut caches, &[0, 1]);

    assert_eq!(
        mlxcel_core::array_shape(&out.logits),
        vec![1, bs, SPEC_VOCAB as i32],
        "full-block softcapped logits"
    );
    assert_eq!(out.hidden_states.len(), 2);
    for slab in &out.hidden_states {
        assert_eq!(
            mlxcel_core::array_shape(slab),
            vec![1, bs, SPEC_HIDDEN as i32]
        );
    }
    // Captured before the final norm: the last slab pushed through the norm,
    // the untied head and the softcap reproduces the logits, so the slab
    // itself is the pre-norm residual stream.
    let reprojected = model.head_for_test(&out.hidden_states[1]);
    assert!(
        max_abs_diff(&reprojected, &out.logits) < 1e-5,
        "the last captured slab must be the pre-norm residual stream"
    );
    // The two slabs differ: layer 1 did run on top of layer 0.
    assert!(max_abs_diff(&out.hidden_states[0], &out.hidden_states[1]) > 1e-4);
    assert_eq!(offsets(&caches), vec![bs, bs]);
    assert!(caches[0].is_sliding());
    assert!(!caches[1].is_sliding());
}

/// Prefill 3, verify 4, roll back to `accepted = 1`: every cache, sliding
/// and full, is left at offset 5 (3 prompt rows plus the 2 committed rows).
/// Repeated with a 16-token window and 40 prefilled tokens so the sliding
/// cache has wrapped and the speculative buffer is what makes the rewind
/// possible.
#[test]
fn rollback_after_partial_accept_rewinds_every_cache() {
    let _guard = mlx_test_guard();
    let model = spec_model(4);
    let mut caches = model.make_speculative_caches();
    model.enable_speculative_buffers(&mut caches, speculative_buffer_size(4));
    let _ = model.forward_speculative(&ids(&[1, 2, 3]), &mut caches, &[]);
    let out = model.forward_speculative(&ids(&[4, 5, 6, 7]), &mut caches, &[]);
    assert_eq!(offsets(&caches), vec![7, 7]);
    model.rollback_speculative_cache(&mut caches, 1, 4);
    assert_eq!(offsets(&caches), vec![5, 5]);
    drop(out);
    // The next block appends onto the committed prefix.
    let next = model.forward_speculative(&ids(&[0, 1, 2, 3]), &mut caches, &[]);
    assert_eq!(
        mlxcel_core::array_shape(&next.logits),
        vec![1, 4, SPEC_VOCAB as i32]
    );
    assert_eq!(offsets(&caches), vec![9, 9]);

    // Wrapped sliding cache: 40 prefilled tokens over a 16-token window.
    let model = spec_model(16);
    let mut caches = model.make_speculative_caches();
    model.enable_speculative_buffers(&mut caches, speculative_buffer_size(4));
    let prompt: Vec<i32> = (0..40).map(|i| (i * 3) % SPEC_VOCAB as i32).collect();
    let _ = model.forward_speculative(&ids(&prompt), &mut caches, &[]);
    assert_eq!(offsets(&caches), vec![40, 40]);
    let _ = model.forward_speculative(&ids(&[1, 2, 3, 4]), &mut caches, &[]);
    assert_eq!(offsets(&caches), vec![44, 44]);
    model.rollback_speculative_cache(&mut caches, 1, 4);
    assert_eq!(offsets(&caches), vec![42, 42]);
    let _ = model.forward_speculative(&ids(&[5, 6, 7, 0]), &mut caches, &[]);
    assert_eq!(offsets(&caches), vec![46, 46]);
    // A full accept rolls nothing back.
    model.rollback_speculative_cache(&mut caches, 3, 4);
    assert_eq!(offsets(&caches), vec![46, 46]);
}

/// The exactness premise on the synthetic model: a four-row verify block
/// (on buffered caches, as the burst runs) and four single-token decode
/// steps (on plain caches, as classic decode runs) from the same prefilled
/// state, with a prompt longer than the 4-token window so the sliding ring
/// has wrapped in the chain arm. The synthetic model is f32 and tiny, so
/// this pins the arithmetic path (window band, buffer conversion, rollback
/// offsets), not any kernel-selection effect; the real-checkpoint probe
/// measures those.
#[test]
fn speculative_block_logits_match_single_token_decode() {
    let _guard = mlx_test_guard();
    let model = spec_model(4);
    let prompt = [1, 2, 3, 4, 5, 6];
    let block = [7, 0, 5, 2];

    let mut chain_caches = model.make_speculative_caches();
    let _ = model.forward_speculative(&ids(&prompt), &mut chain_caches, &[]);
    let mut chain_rows: Vec<UniquePtr<MlxArray>> = Vec::new();
    for tok in block {
        let out = model.forward_speculative(&ids(&[tok]), &mut chain_caches, &[]);
        chain_rows.push(out.logits);
    }

    let mut block_caches = model.make_speculative_caches();
    model.enable_speculative_buffers(&mut block_caches, speculative_buffer_size(block.len()));
    let _ = model.forward_speculative(&ids(&prompt), &mut block_caches, &[]);
    let out = model.forward_speculative(&ids(&block), &mut block_caches, &[]);
    for (i, chain) in chain_rows.iter().enumerate() {
        let i = i as i32;
        let row = mlxcel_core::slice(&out.logits, &[0, i, 0], &[1, i + 1, SPEC_VOCAB as i32]);
        let diff = max_abs_diff(&row, chain);
        assert!(
            diff < 1e-4,
            "verify row {i} diverged from the single-token step: max|diff| = {diff}"
        );
    }
    assert_eq!(offsets(&chain_caches), offsets(&block_caches));

    // A second round after a partial rollback still matches the chain.
    model.rollback_speculative_cache(&mut block_caches, 1, block.len() as i32);
    let mut chain_caches = model.make_speculative_caches();
    let _ = model.forward_speculative(&ids(&prompt), &mut chain_caches, &[]);
    let _ = model.forward_speculative(&ids(&block[..2]), &mut chain_caches, &[]);
    let next = [3, 6, 1];
    let mut chain_rows: Vec<UniquePtr<MlxArray>> = Vec::new();
    for tok in next {
        let out = model.forward_speculative(&ids(&[tok]), &mut chain_caches, &[]);
        chain_rows.push(out.logits);
    }
    let out = model.forward_speculative(&ids(&next), &mut block_caches, &[]);
    for (i, chain) in chain_rows.iter().enumerate() {
        let i = i as i32;
        let row = mlxcel_core::slice(&out.logits, &[0, i, 0], &[1, i + 1, SPEC_VOCAB as i32]);
        let diff = max_abs_diff(&row, chain);
        assert!(
            diff < 1e-4,
            "post-rollback verify row {i} diverged from the single-token step: max|diff| = {diff}"
        );
    }
}

#[test]
fn exactness_probe_runs_on_the_synthetic_model() {
    let _guard = mlx_test_guard();
    let model = spec_model(4);
    let verdict = model.probe_block_chain_exactness(4);
    assert!(
        !matches!(
            verdict,
            crate::models::speculative_exactness::BlockChainExactness::NotRun(_)
        ),
        "probe must run: {verdict:?}"
    );
    assert!(matches!(
        model.probe_block_chain_exactness(1),
        crate::models::speculative_exactness::BlockChainExactness::NotRun(_)
    ));
}

#[test]
fn speculative_buffer_size_follows_the_gemma_rule() {
    assert_eq!(speculative_buffer_size(2), 32);
    assert_eq!(speculative_buffer_size(4), 32);
    assert_eq!(speculative_buffer_size(8), 64);
    assert_eq!(speculative_buffer_size(16), 128);
    assert_eq!(speculative_buffer_size(32), 128);
}

/// The buffer is the slack a verify block is appended into past the
/// window, so it can never be narrower than the block: `--draft-block-size`
/// is not bounded above on the way in, and the Gemma rule's 128-row cap
/// alone would hand a 200-row block 128 rows of slack, overwrite 72 rows
/// the window still shows, and let `trim` rewind the offsets over rows it
/// cannot restore.
#[test]
fn speculative_buffer_never_falls_below_the_block_it_buffers() {
    for block in [2_usize, 4, 8, 16, 32, 64, 128, 129, 200, 1024, 65536] {
        let buffer = speculative_buffer_size(block);
        assert!(
            buffer >= block as i32,
            "a {block}-row verify block was armed with only {buffer} rows of slack"
        );
    }
}

/// The exactness gate has to clear every width the adaptive round loop
/// runs at, not only the requested ceiling: the loop warms up at the
/// drafter's declared depth of 4 and widens from there, and the forward
/// width selects which quantized-matmul kernel MLX dispatches.
#[test]
fn probed_widths_cover_the_warm_up_depth_and_the_ceiling() {
    assert_eq!(super::speculative::probed_verify_widths(16), vec![4, 16]);
    assert_eq!(super::speculative::probed_verify_widths(8), vec![4, 8]);
    assert_eq!(super::speculative::probed_verify_widths(5), vec![4, 5]);
    // At or below the depth there is nothing to widen to.
    assert_eq!(super::speculative::probed_verify_widths(4), vec![4]);
    assert_eq!(super::speculative::probed_verify_widths(3), vec![3]);
    assert_eq!(super::speculative::probed_verify_widths(2), vec![2]);
}

/// The drafter binds the RAW table and the untied head: what the wrapper
/// hands out through `embed_tokens_module` is not the `embed_norm`-wrapped
/// lookup `embed_tokens` applies, and the head is the plain projection.
#[test]
fn wrapper_hands_out_raw_embedding_and_untied_head() {
    let _guard = mlx_test_guard();
    let wrapper = MuseGlimmerTextWrapper::new(spec_model(4));
    let tokens = ids(&[1, 5]);
    let raw = wrapper
        .embed_tokens_module()
        .expect("Muse hands out its embedding table")
        .forward(&tokens);
    let normed = LanguageModel::embed_tokens(&wrapper, &tokens).expect("normed lookup");
    assert_eq!(
        mlxcel_core::array_shape(&raw),
        vec![1, 2, SPEC_HIDDEN as i32]
    );
    assert!(
        max_abs_diff(&raw, &normed) > 1e-3,
        "the raw table must not carry the embed_norm"
    );
    let head = wrapper
        .lm_head_module()
        .expect("Muse hands out its untied head");
    let logits = head.forward(&raw);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 2, SPEC_VOCAB as i32]
    );

    // The trait surface the round loop drives.
    let mut caches = wrapper.model.make_speculative_caches();
    let out = wrapper.verify_forward_with_capture_layers(&ids(&[1, 2, 3]), &mut caches, &[0, 1]);
    let hidden = wrapper.concat_hidden_for_drafter(&out);
    assert_eq!(
        mlxcel_core::array_shape(&hidden),
        vec![1, 3, 2 * SPEC_HIDDEN as i32]
    );
    assert_eq!(
        mlxcel_core::array_shape(wrapper.verify_logits(&out)),
        vec![1, 3, SPEC_VOCAB as i32]
    );
    wrapper.rollback_partial(&mut caches, &out, 0, 3);
    assert_eq!(offsets(&caches), vec![1, 1]);
}
