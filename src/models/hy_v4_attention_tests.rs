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

//! Hy4 attention tests: cross-layer top-k sharing and the sink semantics
//! (issue #2354). The blocks come from the synthetic two-layer model of
//! `hy_v4_tests.rs`: layer 0 is `full`, layer 1 is `shared`.

use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::hy_v4_tests::{
    HEADS, HIDDEN, Tiny, max_abs_diff, tiny_config, tiny_weights, values,
};
use super::super::{ParityArms, load_decoder_layer};
use super::HyV4Attention;

/// The attention block of `layer` from the tiny checkpoint.
fn attention(index_topk: i32, layer: usize) -> HyV4Attention {
    let tiny = Tiny::new(index_topk);
    let config = tiny_config(tiny);
    load_decoder_layer(
        &tiny_weights(tiny),
        &config,
        &config.base,
        layer,
        ParityArms::default(),
    )
    .expect("tiny layer loads")
    .self_attn
}

fn hidden(len: usize, seed: f32) -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..len * HIDDEN as usize)
        .map(|i| 0.8 * (0.61 * i as f32 + seed).sin())
        .collect();
    mlxcel_core::from_slice_f32(&data, &[1, len as i32, HIDDEN])
}

fn indices(positions: &[i32]) -> UniquePtr<MlxArray> {
    let array = mlxcel_core::from_slice_i32(positions, &[1, 1, 1, positions.len() as i32]);
    mlxcel_core::astype(&array, mlxcel_core::dtype::UINT32)
}

/// A cache holding a five-token dense prefill of `attn`.
fn prefilled(attn: &HyV4Attention) -> KVCache {
    let mut cache = KVCache::new();
    let mut topk = None;
    mlxcel_core::eval(&attn.forward(&hidden(5, 0.3), None, &mut cache, &mut topk));
    cache
}

/// One decode step of `attn` behind a fresh five-token prefill, with `topk`
/// as the selection handed down by the previous layer. Returns the output
/// and the selection state the layer left behind.
fn decode_with(attn: &HyV4Attention, topk: Option<&[i32]>) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut cache = prefilled(attn);
    let mut state = topk.map(indices);
    let out = attn.forward(&hidden(1, 1.7), None, &mut cache, &mut state);
    (values(&out), state.as_deref().map(values))
}

/// The shared layer attends to exactly the positions the previous layer
/// selected: different selections give different outputs, a selection of
/// every position equals dense attention, `None` is dense, and the layer
/// never replaces the selection it was handed.
#[test]
fn shared_layer_attends_to_the_previous_layers_selection() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let shared = attention(2, 1);
    assert!(shared.shares_indexer && shared.mla.indexer.is_none());

    let (first_two, state) = decode_with(&shared, Some(&[0, 1]));
    assert_eq!(
        state,
        Some(vec![0.0, 1.0]),
        "a shared layer keeps the state"
    );
    let (last_two, _) = decode_with(&shared, Some(&[3, 4]));
    let (dense, state) = decode_with(&shared, None);
    assert_eq!(state, None);
    let (everything, _) = decode_with(&shared, Some(&[0, 1, 2, 3, 4, 5]));

    assert!(max_abs_diff(&first_two, &last_two) > 1e-4);
    assert!(max_abs_diff(&first_two, &dense) > 1e-4);
    assert!(max_abs_diff(&last_two, &dense) > 1e-4);
    let diff = max_abs_diff(&everything, &dense);
    assert!(diff < 1e-5, "selecting every key must equal dense: {diff}");
}

/// A shared layer's multi-token step masks to the handed-down selection too.
#[test]
fn shared_layer_prefill_masks_to_the_previous_layers_selection() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let shared = attention(2, 1);
    let run = |rows: Option<[[i32; 2]; 3]>| {
        let mut cache = KVCache::new();
        let mut state = rows.map(|rows| {
            let flat: Vec<i32> = rows.iter().flatten().copied().collect();
            let array = mlxcel_core::from_slice_i32(&flat, &[1, 1, 3, 2]);
            mlxcel_core::astype(&array, mlxcel_core::dtype::UINT32)
        });
        values(&shared.forward(&hidden(3, 0.9), None, &mut cache, &mut state))
    };
    // Row 2 may see keys 0..=2; restrict it to {0, 2} or to {1, 2}.
    let a = run(Some([[0, 0], [0, 1], [0, 2]]));
    let b = run(Some([[0, 0], [0, 1], [1, 2]]));
    let dense = run(None);
    let last = |v: &[f32]| v[2 * HIDDEN as usize..].to_vec();
    assert!(max_abs_diff(&last(&a), &last(&b)) > 1e-4);
    assert!(max_abs_diff(&last(&a), &last(&dense)) > 1e-4);
    // Rows 0 and 1 see every causally valid key under both selections.
    let head = |v: &[f32]| v[..2 * HIDDEN as usize].to_vec();
    assert!(max_abs_diff(&head(&a), &head(&dense)) < 1e-5);
}

/// The full layer always overwrites the selection: with its own top-k when
/// `kv_len > index_topk`, and with `None` (dense fallback) at short context,
/// even when a stale selection is still in the state.
#[test]
fn full_layer_overwrites_the_selection_including_the_dense_fallback() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();

    let sparse = attention(2, 0);
    assert!(!sparse.shares_indexer && sparse.mla.indexer.is_some());
    let mut cache = KVCache::new();
    let mut state = None;
    mlxcel_core::eval(&sparse.forward(&hidden(5, 0.3), None, &mut cache, &mut state));
    let selection = state.expect("kv_len 5 > index_topk 2 selects");
    assert_eq!(mlxcel_core::array_shape(&selection), [1, 1, 5, 2]);
    // Row 4 may select any two of the five keys, and only those.
    let picked = values(&selection);
    assert!(picked.iter().all(|p| (0.0..5.0).contains(p)), "{picked:?}");

    let dense = attention(2048, 0);
    let mut cache = KVCache::new();
    let mut state = Some(indices(&[0, 1]));
    mlxcel_core::eval(&dense.forward(&hidden(5, 0.3), None, &mut cache, &mut state));
    assert!(state.is_none(), "kv_len <= index_topk must clear the state");
}

/// Sink semantics, on the three attention routes (causal prefill, masked
/// sparse prefill, absorbed decode): a sink is one extra logit per head in
/// the softmax denominator with no value row. A very negative sink therefore
/// equals no sink at all, and a very large one absorbs the attention mass and
/// drives the output to zero.
#[test]
fn sinks_are_a_valueless_logit_in_the_softmax_denominator() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let sink = |value: f32| {
        Some(mlxcel_core::from_slice_f32(
            &[value; HEADS as usize],
            &[HEADS],
        ))
    };

    // (layer, index_topk): causal dense prefill, then masked sparse prefill.
    for (layer, index_topk) in [(1, 2), (0, 2), (0, 2048)] {
        let run = |sinks: Option<UniquePtr<MlxArray>>| {
            let mut attn = attention(index_topk, layer);
            attn.sinks = sinks;
            let mut cache = KVCache::new();
            let mut state = None;
            let prefill = values(&attn.forward(&hidden(5, 0.3), None, &mut cache, &mut state));
            let decode = values(&attn.forward(&hidden(1, 1.7), None, &mut cache, &mut state));
            (prefill, decode)
        };
        let (none_prefill, none_decode) = run(None);
        let (off_prefill, off_decode) = run(sink(-1.0e4));
        let (on_prefill, on_decode) = run(sink(40.0));
        let (trained_prefill, trained_decode) = run(sink(0.5));
        let what = format!("layer {layer}, index_topk {index_topk}");

        assert!(max_abs_diff(&none_prefill, &off_prefill) < 1e-5, "{what}");
        assert!(max_abs_diff(&none_decode, &off_decode) < 1e-5, "{what}");
        let peak = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(peak(&on_prefill) < 1e-6, "{what}: {}", peak(&on_prefill));
        assert!(peak(&on_decode) < 1e-6, "{what}: {}", peak(&on_decode));
        assert!(
            max_abs_diff(&none_prefill, &trained_prefill) > 1e-4,
            "{what}"
        );
        assert!(max_abs_diff(&none_decode, &trained_decode) > 1e-4, "{what}");
    }
}

/// The gate multiplies the per-head output, so removing it changes the
/// block's output.
#[test]
fn gate_scales_the_per_head_output() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let run = |gated: bool| {
        let mut attn = attention(2, 1);
        if !gated {
            attn.linear_gate = None;
        }
        let mut cache = KVCache::new();
        let mut state = None;
        values(&attn.forward(&hidden(4, 0.3), None, &mut cache, &mut state))
    };
    assert!(max_abs_diff(&run(true), &run(false)) > 1e-4);
}

/// Deterministic pseudo-random values in `[-amp, amp]`.
fn noise(shape: &[i32], seed: u32, amp: f32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state = seed;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            amp * ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0)
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// `softmax([scale * q . k + mask, sinks])` without the sink column, times
/// `v`, written out in float32 from primitive ops.
fn reference_sink_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
    sinks: &MlxArray,
) -> Vec<f32> {
    let f32_dtype = mlxcel_core::dtype::FLOAT32;
    let cast = |a: &MlxArray| mlxcel_core::astype(a, f32_dtype);
    let k_t = mlxcel_core::transpose_axes(&cast(k), &[0, 1, 3, 2]);
    let scores = mlxcel_core::multiply_scalar(&mlxcel_core::matmul(&cast(q), &k_t), scale);
    let scores = match mask {
        Some(mask) => mlxcel_core::add(&scores, &cast(mask)),
        None => scores,
    };
    let shape = mlxcel_core::array_shape(&scores);
    let (b, h, l, s) = (shape[0], shape[1], shape[2], shape[3]);
    let sink = mlxcel_core::reshape(&cast(sinks), &[1, h, 1, 1]);
    let sink = mlxcel_core::broadcast_to(&sink, &[b, h, l, 1]);
    let probs = mlxcel_core::softmax_precise(&mlxcel_core::concatenate(&scores, &sink, -1), -1);
    let probs = mlxcel_core::utils::slice_axis(&probs, -1, 0, s);
    values(&mlxcel_core::matmul(&probs, &cast(v)))
}

/// The MLX kernels the Hy4 attention relies on, at the checkpoint's geometry
/// (64 heads, a 512-wide latent, 256-wide folded heads) rather than the tiny
/// fixture's, in float32 and in the checkpoint's bf16: the absorbed decode
/// with one key head for 64 query heads, the masked multi-token step, and the
/// maskless causal one. Each must equal the sink softmax written out by hand.
#[test]
fn sink_sdpa_matches_the_written_out_softmax_at_checkpoint_geometry() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    const HEADS_REAL: i32 = 64;
    let scale = 256f32.powf(-0.5);
    let sinks = noise(&[HEADS_REAL], 7, 1.0);
    let peak = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));

    for (dtype, tolerance) in [
        (mlxcel_core::dtype::FLOAT32, 1e-4),
        (mlxcel_core::dtype::BFLOAT16, 4e-2),
    ] {
        let cast = |a: UniquePtr<MlxArray>| mlxcel_core::astype(&a, dtype);

        // Absorbed decode: the latent is K and V, with one head.
        let q = cast(noise(&[1, HEADS_REAL, 1, 512], 1, 0.5));
        let latent = cast(noise(&[1, 1, 2048, 512], 2, 0.5));
        let pe_scores = cast(noise(&[1, HEADS_REAL, 1, 2048], 3, 2.0));
        let got = values(&super::sink_sdpa(
            &q,
            &latent,
            &latent,
            scale,
            Some(&pe_scores),
            Some(&sinks),
        ));
        let want = reference_sink_attention(&q, &latent, &latent, scale, Some(&pe_scores), &sinks);
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff < tolerance * peak(&want).max(1e-3),
            "decode, dtype {dtype}: {diff} of {}",
            peak(&want)
        );

        // Multi-token step over folded `[nope | rope]` heads.
        let (l, s) = (8, 512);
        let q = cast(noise(&[1, HEADS_REAL, l, 256], 4, 0.5));
        let k = cast(noise(&[1, HEADS_REAL, s, 256], 5, 0.5));
        let v = cast(noise(&[1, HEADS_REAL, s, 256], 6, 0.5));
        let causal = mlxcel_core::utils::create_causal_mask(l, s - l);
        let want = reference_sink_attention(&q, &k, &v, scale, Some(&causal), &sinks);

        let masked = values(&super::sink_sdpa(
            &q,
            &k,
            &v,
            scale,
            Some(&cast(mlxcel_core::copy(&causal))),
            Some(&sinks),
        ));
        let diff = max_abs_diff(&masked, &want);
        assert!(
            diff < tolerance * peak(&want),
            "masked prefill, dtype {dtype}: {diff}"
        );

        let sinks_cast = mlxcel_core::astype(&sinks, dtype);
        let maskless = values(&mlxcel_core::causal_prefill_attention(
            &q,
            &k,
            &v,
            scale,
            0,
            Some(&sinks_cast),
        ));
        let diff = max_abs_diff(&maskless, &want);
        assert!(
            diff < tolerance * peak(&want),
            "causal prefill, dtype {dtype}: {diff}"
        );
    }
}
