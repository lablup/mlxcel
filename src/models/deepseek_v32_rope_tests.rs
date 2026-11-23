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

//! YaRN RoPE on the DeepSeek V3.2 MLA attention and DSA indexer (#3178).
//!
//! The forward-path tests feed one constant hidden row at every position. RoPE
//! at position 0 is the identity for any frequency table, so the cached row 0
//! is the unrotated vector, and every later cached row must equal that vector
//! rotated by the YaRN table at its own position. That reference is built from
//! `yarn_rope_setup` and `fast_rope_with_freqs` only, not from the model's
//! RoPE handle, so these tests observe what the model actually cached.
//!
//! The pinned table values were computed independently from upstream's
//! `YarnRoPE.__init__` formula (mlx-lm `rope_utils.py`) in float64 Python.

use super::deepseek_v32_tests::synthetic_mla_weights;
use super::indexer::Indexer;
use super::rope::DsaRope;
use super::{DeepSeekV32Model, MLAAttention, ModelArgs};
use crate::models::rope_utils::{RopeScalingSpec, yarn_rope_setup};
use mlxcel_core::layers::KVCache;
use mlxcel_core::utils::{array_to_vec_f32, slice_axis};
use mlxcel_core::{MlxArray, UniquePtr};

const PREFIX: &str = "model.layers.0.self_attn";
/// Rotated dims and indexer head width of the tiny config: the indexer
/// rotates 8 of its 12 features, as the checkpoint rotates 64 of 128.
const ROPE_DIMS: i32 = 8;
const INDEX_HEAD_DIM: i32 = 12;
const KV_LORA: i32 = 4;
/// The tiny config's `original_max_position_embeddings`. Kept small so a
/// prefill crosses it cheaply; the YaRN band arithmetic is the same.
const ORIGINAL_CTX: i32 = 16;

/// Tiny MLA geometry with a YaRN block, or none.
fn tiny_config(rope_scaling: Option<&str>) -> ModelArgs {
    let rope_scaling = rope_scaling.map_or(String::new(), |block| {
        format!(r#""rope_scaling": {block},"#)
    });
    let json = format!(
        r#"{{
        "model_type": "deepseek_v32",
        "vocab_size": 16,
        "hidden_size": 6,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 8,
        "kv_lora_rank": {KV_LORA},
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": {ROPE_DIMS},
        "v_head_dim": 2,
        "index_topk": 64,
        "index_n_heads": 2,
        "index_head_dim": {INDEX_HEAD_DIM},
        {rope_scaling}
        "rope_theta": 10000.0
    }}"#
    );
    serde_json::from_str(&json).expect("parse tiny YaRN config")
}

/// The checkpoint's block, with `original_max_position_embeddings` shrunk.
fn tiny_yarn_block() -> String {
    format!(
        r#"{{"type": "yarn", "factor": 40, "original_max_position_embeddings": {ORIGINAL_CTX},
            "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}}"#
    )
}

fn tiny_attention(args: &ModelArgs) -> MLAAttention {
    let weights = synthetic_mla_weights(args, PREFIX, true);
    let weights = DeepSeekV32Model::sanitize_weights(weights, args).expect("sanitize");
    super::load_mla_attention(&weights, args, PREFIX).expect("tiny attention loads")
}

/// `[1, len, hidden]` with the same row at every position.
fn constant_rows(len: usize, hidden: usize, seed: f32) -> UniquePtr<MlxArray> {
    let row: Vec<f32> = (0..hidden)
        .map(|i| seed + 0.07 * (i as f32 + 1.0))
        .collect();
    let values: Vec<f32> = (0..len).flat_map(|_| row.iter().copied()).collect();
    mlxcel_core::from_slice_f32(&values, &[1, len as i32, hidden as i32])
}

/// The YaRN table for [`tiny_yarn_block`], straight from the shared helper
/// and parsed from the JSON rather than from `args`, so the reference does not
/// depend on how the model reads its config.
fn reference_freqs(args: &ModelArgs) -> UniquePtr<MlxArray> {
    let spec: RopeScalingSpec = serde_json::from_str(&tiny_yarn_block()).expect("YaRN block");
    let spec = &spec;
    let (freqs, mscale) =
        yarn_rope_setup(spec, args.qk_rope_head_dim, args.rope_theta, None).expect("YaRN table");
    assert_eq!(
        mscale, 1.0,
        "the forward tests assume no magnitude multiplier"
    );
    freqs
}

/// `row0` (`[1, H, 1, D]`) repeated at positions `start..start + len` and
/// rotated there: what the cache must hold if YaRN was applied.
fn rotated_reference(
    row0: &MlxArray,
    start: i32,
    len: i32,
    traditional: bool,
    freqs: Option<&MlxArray>,
    base: f32,
) -> UniquePtr<MlxArray> {
    let rows = mlxcel_core::tile(row0, &[1, 1, len, 1]);
    match freqs {
        Some(freqs) => {
            mlxcel_core::fast_rope_with_freqs(&rows, ROPE_DIMS, traditional, 1.0, start, freqs)
        }
        None => mlxcel_core::fast_rope(&rows, ROPE_DIMS, traditional, base, 1.0, start),
    }
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let diff = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(a, b)));
    mlxcel_core::eval(&diff);
    mlxcel_core::item_f32(&diff)
}

/// The live `(k_pe, indexer key)` rows of a fp16 cache.
fn cached_rows(cache: &KVCache) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let live = cache.live_len();
    let keys = cache.keys.as_ref().expect("cached keys");
    let values = cache.values.as_ref().expect("cached k_pe");
    let k_pe = slice_axis(values, 2, 0, live);
    let idx_key = slice_axis(
        &slice_axis(keys, 2, 0, live),
        -1,
        KV_LORA,
        KV_LORA + INDEX_HEAD_DIM,
    );
    (k_pe, idx_key)
}

/// Assert the cached rows `from..to` are row 0 rotated by YaRN at their own
/// positions, and that plain RoPE would have cached something else.
fn assert_cache_holds_yarn_rows(args: &ModelArgs, cache: &KVCache, from: i32, to: i32) {
    let freqs = reference_freqs(args);
    let (k_pe, idx_key) = cached_rows(cache);
    let base = args.rope_theta;
    for (name, cached, traditional) in [
        ("k_pe", &k_pe, true),
        ("indexer key", &idx_key, args.indexer_rope_interleave),
    ] {
        let row0 = slice_axis(cached, 2, 0, 1);
        let got = slice_axis(cached, 2, from, to);
        let want = rotated_reference(&row0, from, to - from, traditional, Some(&freqs), base);
        let plain = rotated_reference(&row0, from, to - from, traditional, None, base);
        let err = max_abs_diff(&got, &want);
        assert!(
            err < 1e-4,
            "cached {name} at positions {from}..{to} must be YaRN-rotated (max diff {err})"
        );
        let gap = max_abs_diff(&want, &plain);
        assert!(
            gap > 1e-2,
            "YaRN and plain {name} must differ past the original context (gap {gap})"
        );
    }
}

#[test]
fn yarn_table_matches_the_upstream_formula_at_checkpoint_geometry() {
    let _runtime = crate::initialize_runtime();
    // DeepSeek-V3.2's shipped block at its real rope width.
    let block = r#"{"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096,
        "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}"#;
    let mut args = tiny_config(Some(block));
    args.qk_rope_head_dim = 64;
    args.index_head_dim = 128;

    for rope_type in ["yarn", "deepseek_yarn"] {
        args.rope_scaling.as_mut().expect("block").rope_type = Some(rope_type.to_string());
        let rope = DsaRope::from_args(&args).expect("valid rope");
        let freqs = array_to_vec_f32(rope.freqs().expect("{rope_type} must build a table"));
        assert_eq!(freqs.len(), 32);
        // (pair, upstream YarnRoPE._freqs) in float64. Pairs below 10
        // extrapolate (plain), pairs above 23 interpolate (x40).
        for (pair, want) in [
            (0, 1.0_f64),
            (10, 17.78279410038923),
            (16, 181.8181818181818),
            (20, 1264.911064067352),
            (24, 40000.0),
            (31, 299957.6837329823),
        ] {
            let got = f64::from(freqs[pair]);
            assert!(
                ((got - want) / want).abs() < 1e-5,
                "{rope_type} pair {pair}: got {got}, upstream {want}"
            );
        }
        assert_eq!(
            rope.mscale(),
            1.0,
            "mscale == mscale_all_dim resolves to 1.0"
        );
    }
}

#[test]
fn absent_rope_scaling_keeps_the_plain_rope_call() {
    let _runtime = crate::initialize_runtime();
    let args = tiny_config(None);
    let rope = DsaRope::from_args(&args).expect("valid rope");
    assert!(rope.freqs().is_none(), "no block, no table");

    let x = constant_rows(3 * INDEX_HEAD_DIM as usize, 1, 0.3);
    let x = mlxcel_core::reshape(&x, &[1, 1, 3, INDEX_HEAD_DIM]);
    for traditional in [true, false] {
        let got = rope.apply(&x, traditional, 5000);
        let want = mlxcel_core::fast_rope(&x, ROPE_DIMS, traditional, 10000.0, 1.0, 5000);
        assert_eq!(max_abs_diff(&got, &want), 0.0, "must stay byte-identical");
        let got = rope.apply_batched(&x, traditional, &[5000]);
        let want =
            mlxcel_core::fast_rope_batched(&x, ROPE_DIMS, traditional, 10000.0, 1.0, &[5000]);
        assert_eq!(
            max_abs_diff(&got, &want),
            0.0,
            "batched must stay byte-identical"
        );
    }
}

#[test]
fn yarn_magnitude_scales_only_the_rotated_dims() {
    let _runtime = crate::initialize_runtime();
    let block = format!(
        r#"{{"type": "yarn", "factor": 40, "original_max_position_embeddings": {ORIGINAL_CTX},
            "mscale": 2.0, "mscale_all_dim": 1.0}}"#
    );
    let rope = DsaRope::from_args(&tiny_config(Some(&block))).expect("valid rope");
    // Upstream: yarn_get_mscale(40, 2.0) / yarn_get_mscale(40, 1.0).
    let want_mscale = 1.269_480_015_985_188_f64;
    assert!(
        (f64::from(rope.mscale()) - want_mscale).abs() < 1e-6,
        "mscale {}",
        rope.mscale()
    );

    // Position 0 rotates by nothing, so only the magnitude shows.
    let values: Vec<f32> = (0..INDEX_HEAD_DIM).map(|i| 0.5 + i as f32).collect();
    let x = mlxcel_core::from_slice_f32(&values, &[1, 1, 1, INDEX_HEAD_DIM]);
    let got = array_to_vec_f32(&rope.apply(&x, false, 0));
    for (i, (&g, &v)) in got.iter().zip(&values).enumerate() {
        let want = if (i as i32) < ROPE_DIMS {
            v * want_mscale as f32
        } else {
            v
        };
        assert!((g - want).abs() < 1e-4, "feature {i}: got {g}, want {want}");
    }
}

/// Prefill across `original_max_position_embeddings`, then one decode step:
/// both the attention's `k_pe` and the indexer key in the cache carry YaRN.
#[test]
fn prefill_and_decode_cache_yarn_rotated_keys_past_the_original_context() {
    let _runtime = crate::initialize_runtime();
    // Forward passes bump the process-wide MLA decode-path counters that the
    // latent tests measure as deltas, so hold the shared MLX test lock.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = tiny_config(Some(&tiny_yarn_block()));
    let attn = tiny_attention(&args);
    let hidden = args.hidden_size;
    let prefill = ORIGINAL_CTX + 8;

    let mut cache = KVCache::new();
    let out = attn.forward(
        &constant_rows(prefill as usize, hidden, 0.1),
        None,
        &mut cache,
    );
    mlxcel_core::eval(&out);
    assert_cache_holds_yarn_rows(&args, &cache, ORIGINAL_CTX, prefill);

    let out = attn.forward(&constant_rows(1, hidden, 0.1), None, &mut cache);
    mlxcel_core::eval(&out);
    assert_eq!(cache.offset, prefill + 1);
    assert_cache_holds_yarn_rows(&args, &cache, prefill, prefill + 1);
}

/// The batched decode path rotates each row at its own offset with YaRN, and
/// still matches sequential decode.
#[test]
fn batched_decode_caches_yarn_rotated_keys_per_row() {
    let _runtime = crate::initialize_runtime();
    // Forward passes bump the process-wide MLA decode-path counters that the
    // latent tests measure as deltas, so hold the shared MLX test lock.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = tiny_config(Some(&tiny_yarn_block()));
    let attn = tiny_attention(&args);
    let hidden = args.hidden_size;
    let lens = [3, ORIGINAL_CTX + 5];

    let mut expected = [KVCache::new(), KVCache::new()];
    let mut actual = [KVCache::new(), KVCache::new()];
    for (row, &len) in lens.iter().enumerate() {
        for cache in [&mut expected[row], &mut actual[row]] {
            let out = attn.forward(&constant_rows(len as usize, hidden, 0.1), None, cache);
            mlxcel_core::eval(&out);
        }
    }

    let step = constant_rows(1, hidden, 0.1);
    let want: Vec<_> = expected
        .iter_mut()
        .map(|cache| attn.forward(&step, None, cache))
        .collect();
    let want = mlxcel_core::concatenate(&want[0], &want[1], 0);
    let batch = mlxcel_core::concatenate(&step, &step, 0);
    let got = {
        let [a, b] = &mut actual;
        let mut caches = [a, b];
        attn.forward_batched_decode(&batch, &mut caches)
    };
    mlxcel_core::eval(&got);
    let err = max_abs_diff(&got, &want);
    assert!(
        err < 1e-4,
        "batched decode must match sequential (max diff {err})"
    );

    for (row, &len) in lens.iter().enumerate() {
        assert_eq!(actual[row].offset, len + 1);
        assert_cache_holds_yarn_rows(&args, &actual[row], len, len + 1);
    }
}

/// The indexer's queries (not cached) rotate with YaRN too.
#[test]
fn indexer_queries_and_keys_rotate_with_yarn() {
    let _runtime = crate::initialize_runtime();
    // Forward passes bump the process-wide MLA decode-path counters that the
    // latent tests measure as deltas, so hold the shared MLX test lock.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = tiny_config(Some(&tiny_yarn_block()));
    let weights = synthetic_mla_weights(&args, PREFIX, true);
    let indexer = Indexer::load(&weights, &args, PREFIX)
        .expect("load ok")
        .expect("indexer present");
    let freqs = reference_freqs(&args);
    let traditional = args.indexer_rope_interleave;
    let position = ORIGINAL_CTX + 4;

    let qr = constant_rows(1, args.q_lora_rank, 0.2);
    let x = constant_rows(1, args.hidden_size, 0.2);
    for (name, at0, at_p) in [
        (
            "queries",
            indexer.queries(&qr, 0),
            indexer.queries(&qr, position),
        ),
        ("keys", indexer.keys(&x, 0), indexer.keys(&x, position)),
    ] {
        let want =
            mlxcel_core::fast_rope_with_freqs(&at0, ROPE_DIMS, traditional, 1.0, position, &freqs);
        let err = max_abs_diff(&at_p, &want);
        assert!(
            err < 1e-4,
            "indexer {name} must be YaRN-rotated (max diff {err})"
        );
        let plain = mlxcel_core::fast_rope(&at0, ROPE_DIMS, traditional, 10000.0, 1.0, position);
        assert!(
            max_abs_diff(&want, &plain) > 1e-2,
            "{name}: YaRN must differ from plain"
        );
    }
}

/// The whole model loads with a YaRN block and its batched decode matches
/// sequential decode across the original context.
#[test]
fn yarn_model_batched_decode_matches_sequential() {
    let _runtime = crate::initialize_runtime();
    // Forward passes bump the process-wide MLA decode-path counters that the
    // latent tests measure as deltas, so hold the shared MLX test lock.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = tiny_config(Some(&tiny_yarn_block()));
    let mut weights = synthetic_mla_weights(&args, PREFIX, true);
    let mut put = |key: &str, shape: &[i32]| {
        weights.insert(
            key.to_string(),
            super::deepseek_v32_tests::f32_weight(shape),
        );
    };
    put("model.embed_tokens.weight", &[16, 6]);
    put("model.norm.weight", &[6]);
    put("lm_head.weight", &[16, 6]);
    put("model.layers.0.input_layernorm.weight", &[6]);
    put("model.layers.0.post_attention_layernorm.weight", &[6]);
    put("model.layers.0.mlp.gate_proj.weight", &[16, 6]);
    put("model.layers.0.mlp.up_proj.weight", &[16, 6]);
    put("model.layers.0.mlp.down_proj.weight", &[6, 16]);
    let weights = DeepSeekV32Model::sanitize_weights(weights, &args).expect("sanitize");
    let model = DeepSeekV32Model::from_weights(&weights, &args).expect("YaRN model loads");

    let long: Vec<i32> = (0..ORIGINAL_CTX + 3).map(|t| t % 16).collect();
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        &[&[1, 2], &long],
        &[3, 4],
        1e-4,
    );
}

/// A cache with nothing in it whose next write lands at absolute `offset`.
fn empty_cache_at(offset: i32) -> KVCache {
    let mut cache = KVCache::new();
    cache
        .restore_fp16_live_window(None, None, offset)
        .expect("empty fp16 window");
    cache
}

/// The MLA queries rotate with the same YaRN table as the cached keys, in
/// single-sequence and batched decode alike.
///
/// The query is not cached, so it is observed through RoPE's relative-position
/// property: when `q_pe` and `k_pe` rotate with one table, every attention
/// score depends only on the distance between query and key, and the same two
/// tokens attend identically at positions `0, 1` and at `p, p + 1`. A query
/// rotated by a different table than its keys (plain RoPE against cached YaRN
/// keys, the pre-#3178 failure on the query side) breaks that invariance once
/// `p` is past the original context, where the two tables disagree.
#[test]
fn mla_queries_rotate_with_the_key_table_past_the_original_context() {
    let _runtime = crate::initialize_runtime();
    // Forward passes bump the process-wide MLA decode-path counters that the
    // latent tests measure as deltas, so hold the shared MLX test lock.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = tiny_config(Some(&tiny_yarn_block()));
    let attn = tiny_attention(&args);
    let hidden = args.hidden_size;
    let far = 8 * ORIGINAL_CTX;
    let tokens = |len: usize| {
        let a = constant_rows(1, hidden, 0.1);
        let b = constant_rows(len - 1, hidden, -0.4);
        mlxcel_core::concatenate(&a, &b, 1)
    };

    // Single-sequence: a two-token prefill at 0 and at `far`.
    let near_out = attn.forward(&tokens(2), None, &mut empty_cache_at(0));
    let far_out = attn.forward(&tokens(2), None, &mut empty_cache_at(far));
    let err = max_abs_diff(&far_out, &near_out);
    assert!(
        err < 1e-4,
        "prefill at {far} must attend as at 0 when q_pe and k_pe share a table (max diff {err})"
    );

    // Batched decode: one key cached per row, then one query per row, the two
    // rows `far` positions apart.
    let mut near = empty_cache_at(0);
    let mut far_cache = empty_cache_at(far);
    for cache in [&mut near, &mut far_cache] {
        let out = attn.forward(&constant_rows(1, hidden, 0.1), None, cache);
        mlxcel_core::eval(&out);
    }
    let step = constant_rows(1, hidden, -0.4);
    let batch = mlxcel_core::concatenate(&step, &step, 0);
    let got = {
        let mut caches = [&mut near, &mut far_cache];
        attn.forward_batched_decode(&batch, &mut caches)
    };
    mlxcel_core::eval(&got);
    let err = max_abs_diff(&slice_axis(&got, 0, 1, 2), &slice_axis(&got, 0, 0, 1));
    assert!(
        err < 1e-4,
        "batched decode at {} must attend as at 1 (max diff {err})",
        far + 1
    );
}

#[test]
fn malformed_rope_width_is_refused_before_the_table_is_built() {
    let _runtime = crate::initialize_runtime();
    let block = format!(
        r#"{{"type": "yarn", "factor": 40, "original_max_position_embeddings": {ORIGINAL_CTX}}}"#
    );
    // Zero, odd, and wider than the indexer head (12 here).
    for bad in [0_usize, 7, 14] {
        let mut args = tiny_config(Some(&block));
        args.qk_rope_head_dim = bad;
        let err = DsaRope::from_args(&args).err().expect("must be refused");
        assert!(err.contains("qk_rope_head_dim"), "width {bad}: {err}");
    }
}
