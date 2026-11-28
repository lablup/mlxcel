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

//! Unit tests for the Talkie model. Everything is checkpoint-free: the numeric
//! guards use CPU references, and the tiny-model tests use deterministic
//! synthetic weights.

use super::{
    Attention, ModelArgs, TALKIE_EOS_TOKEN_IDS, TalkieModel, TransformerBlock, sanitize_weights,
};
use crate::models::nanochat::mirrored_rope_freqs;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn to_vec(arr: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic pseudo-random values in `[-scale, scale)`.
fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn rms_ref(x: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * inv).collect()
}

/// Talkie's rotation, `y1 = x1 cos + x2 sin`, `y2 = -x1 sin + x2 cos`, with
/// `angle = pos / base^(i / half)` over the full head.
fn talkie_rope_ref(x: &[f32], pos: f32, base: f32) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = vec![0.0; x.len()];
    for i in 0..half {
        let a = pos / base.powf(i as f32 / half as f32);
        out[i] = x[i] * a.cos() + x[i + half] * a.sin();
        out[i + half] = -x[i] * a.sin() + x[i + half] * a.cos();
    }
    out
}

const EPS: f32 = f32::EPSILON;

// Config.

#[test]
fn config_defaults_match_13b() {
    let args: ModelArgs = serde_json::from_str(r#"{"model_type": "talkie"}"#).unwrap();
    assert_eq!(args.hidden_size, 5120);
    assert_eq!(args.num_hidden_layers, 40);
    assert_eq!(args.num_attention_heads, 40);
    assert_eq!(args.head_dim, 128);
    assert_eq!(args.intermediate_size, 13696);
    assert_eq!(args.vocab_size, 65540);
    assert_eq!(args.rope_theta, 1_000_000.0);
    assert_eq!(args.rms_norm_eps, 1.1920928955078125e-07);
    assert_eq!(args.eos_token_ids(), TALKIE_EOS_TOKEN_IDS.to_vec());
    args.validate().unwrap();
}

#[test]
fn config_validation_rejects_hostile_scalars() {
    for bad in [
        r#"{"hidden_size": 0}"#,
        r#"{"num_attention_heads": 0}"#,
        r#"{"head_dim": 7}"#,
        r#"{"vocab_size": 999999999}"#,
        r#"{"rope_theta": -1.0}"#,
        r#"{"rms_norm_eps": 0.0}"#,
    ] {
        let args: ModelArgs = serde_json::from_str(bad).unwrap();
        assert!(args.validate().is_err(), "{bad} must be rejected");
    }
}

// Sanitize.

#[test]
fn sanitize_renames_bare_lm_head_and_drops_rotary_buffers() {
    let t = || mlxcel_core::from_slice_f32(&[1.0], &[1]);
    let mut w = WeightMap::new();
    for k in [
        "lm_head",
        "lm_head_gain.w_g",
        "model.blocks.0.attn.rotary.inv_freq",
        "model.rope_cos",
        "model.rope_sin",
        "model.embed.weight",
    ] {
        w.insert(k.into(), t());
    }
    let s = sanitize_weights(&w);
    let mut keys: Vec<_> = s.keys().cloned().collect();
    keys.sort();
    assert_eq!(
        keys,
        ["lm_head.weight", "lm_head_gain.w_g", "model.embed.weight"]
    );
    // Idempotent.
    let mut again: Vec<_> = sanitize_weights(&s).keys().cloned().collect();
    again.sort();
    assert_eq!(again, keys);
}

// RoPE and the q/k pipeline, against CPU references.

#[test]
fn rope_uses_the_negated_table() {
    let (head_dim, base) = (8usize, 1_000_000.0f32);
    // The shared table equals upstream `-(theta ** (arange(0, D, 2) / D))`.
    let table = to_vec(&mirrored_rope_freqs(head_dim, base));
    for (j, got) in table.iter().enumerate() {
        let want = -(f64::from(base).powf((2 * j) as f64 / head_dim as f64)) as f32;
        assert!((got - want).abs() <= want.abs() * 1e-6, "{got} vs {want}");
    }

    let xv = lcg(head_dim, 21, 1.0);
    let x = mlxcel_core::from_slice_f32(&xv, &[1, 1, 1, head_dim as i32]);
    let offset = 5;
    let freqs = mirrored_rope_freqs(head_dim, base);
    let got = to_vec(&mlxcel_core::fast_rope_with_freqs(
        &x,
        head_dim as i32,
        false,
        1.0,
        offset,
        &freqs,
    ));
    let want = talkie_rope_ref(&xv, offset as f32, base);
    assert!(max_diff(&got, &want) < 1e-5, "{got:?} vs {want:?}");

    // The ordinary NeoX rotation with the same base turns the other way.
    let ordinary = to_vec(&mlxcel_core::fast_rope(
        &x,
        head_dim as i32,
        false,
        base,
        1.0,
        offset,
    ));
    assert!(max_diff(&ordinary, &got) > 0.1, "{ordinary:?} vs {got:?}");
}

fn tiny_attention(num_heads: usize, head_dim: usize, head_gain: &[f32]) -> Attention {
    let dim = (num_heads * head_dim) as i32;
    let lin = |seed| {
        let mut w = WeightMap::new();
        w.insert(
            "l.weight".into(),
            mlxcel_core::from_slice_f32(&lcg((dim * dim) as usize, seed, 0.3), &[dim, dim]),
        );
        UnifiedLinear::from_weights(&w, "l", 64, 4).unwrap()
    };
    Attention {
        attn_query: lin(3),
        attn_key: lin(4),
        attn_value: lin(5),
        attn_resid: lin(6),
        head_gain: mlxcel_core::from_slice_f32(head_gain, &[1, num_heads as i32, 1, 1]),
        freqs: mirrored_rope_freqs(head_dim, 10000.0),
        num_heads: num_heads as i32,
        head_dim: head_dim as i32,
        scale: (head_dim as f32).powf(-0.5),
        eps: EPS,
    }
}

#[test]
fn qk_norm_follows_rope_and_head_gain_follows_the_norm() {
    let (heads, head_dim) = (2usize, 8usize);
    let gains = [1.5f32, -0.5];
    let attn = tiny_attention(heads, head_dim, &gains);
    let qv = lcg(heads * head_dim, 11, 2.0);
    let kv = lcg(heads * head_dim, 12, 5.0);
    let shape = [1, heads as i32, 1, head_dim as i32];
    let q = mlxcel_core::from_slice_f32(&qv, &shape);
    let k = mlxcel_core::from_slice_f32(&kv, &shape);
    let offset = 3;
    let (qn, kn) = attn.rotate_norm_gain(&q, &k, offset);
    let (qn, kn) = (to_vec(&qn), to_vec(&kn));

    for h in 0..heads {
        let span = h * head_dim..(h + 1) * head_dim;
        let rq = rms_ref(
            &talkie_rope_ref(&qv[span.clone()], offset as f32, 10000.0),
            EPS,
        );
        let rk = rms_ref(
            &talkie_rope_ref(&kv[span.clone()], offset as f32, 10000.0),
            EPS,
        );
        let want_q: Vec<f32> = rq.iter().map(|v| v * gains[h]).collect();
        assert!(max_diff(&qn[span.clone()], &want_q) < 1e-4, "q head {h}");
        // Keys get the norm but no gain: unit RMS per head.
        assert!(max_diff(&kn[span.clone()], &rk) < 1e-4, "k head {h}");

        // The gain must not be normalized away: gain-then-norm would give a
        // unit-RMS query regardless of the gain.
        let gain_first = rms_ref(
            &talkie_rope_ref(&qv[span.clone()], offset as f32, 10000.0)
                .iter()
                .map(|v| v * gains[h])
                .collect::<Vec<_>>(),
            EPS,
        );
        assert!(max_diff(&qn[span.clone()], &gain_first) > 0.1, "head {h}");
    }

    // Norm-then-RoPE: RoPE is an orthogonal rotation over the full head_dim,
    // so it preserves the RMS and commutes with the weightless norm. The
    // order is still RoPE first, as upstream writes it; this pins that the
    // swap is numerically a no-op here, so the gain order above is the one
    // that changes the output.
    let normed_first = mlxcel_core::fast_rms_norm_no_weight(&q, EPS);
    let swapped = mlxcel_core::fast_rope_with_freqs(
        &normed_first,
        head_dim as i32,
        false,
        1.0,
        offset,
        &attn.freqs,
    );
    let swapped = to_vec(&mlxcel_core::multiply(&swapped, &attn.head_gain));
    assert!(max_diff(&swapped, &qn) < 1e-4);
}

// Tiny model.

const H: usize = 64;
const HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const INTER: usize = 128;
const VOCAB: usize = 32;

fn tiny_args(layers: usize) -> ModelArgs {
    serde_json::from_str(&format!(
        r#"{{"model_type": "talkie", "hidden_size": {H}, "num_hidden_layers": {layers},
            "num_attention_heads": {HEADS}, "head_dim": {HEAD_DIM},
            "intermediate_size": {INTER}, "vocab_size": {VOCAB}, "rope_theta": 10000.0}}"#
    ))
    .unwrap()
}

fn scalar(v: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(&[v], &[1])
}

/// Blocks, embedding and gains; the head is added per layout. `gains` are
/// `(attn_gain, mlp_gain, embed_skip)`, and `embed_scale` sizes the embedding.
fn body_weights(layers: usize, gains: (f32, f32, f32), embed_scale: f32) -> WeightMap {
    let mut w = WeightMap::new();
    let mut seed = 100u64;
    let mut put = |w: &mut WeightMap, name: String, rows: usize, cols: usize, scale: f32| {
        seed += 1;
        let data = lcg(rows * cols, seed, scale);
        w.insert(
            name,
            mlxcel_core::from_slice_f32(&data, &[rows as i32, cols as i32]),
        );
    };
    put(&mut w, "model.embed.weight".into(), VOCAB, H, embed_scale);
    for l in 0..layers {
        let p = format!("model.blocks.{l}");
        for n in ["attn_query", "attn_key", "attn_value", "attn_resid"] {
            put(&mut w, format!("{p}.attn.{n}.weight"), H, H, 0.2);
        }
        put(&mut w, format!("{p}.mlp.mlp_gate.weight"), INTER, H, 0.2);
        put(&mut w, format!("{p}.mlp.mlp_linear.weight"), INTER, H, 0.2);
        put(&mut w, format!("{p}.mlp.mlp_resid.weight"), H, INTER, 0.2);
        w.insert(
            format!("{p}.attn.head_gain.head_g"),
            mlxcel_core::from_slice_f32(&[1.3, 0.7], &[HEADS as i32]),
        );
        w.insert(format!("{p}.attn_gain.a_g"), scalar(gains.0));
        w.insert(format!("{p}.mlp_gain.a_g"), scalar(gains.1));
        w.insert(format!("{p}.embed_skip.a_g"), scalar(gains.2));
    }
    w
}

fn head_dense() -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(&lcg(VOCAB * H, 77, 0.3), &[VOCAB as i32, H as i32])
}

/// Layout (a): the upstream bare `lm_head` tensor plus `lm_head_gain.w_g`.
fn layout_a(layers: usize, gain: f32) -> WeightMap {
    let mut w = body_weights(layers, (0.4, 0.6, 0.25), 1.0);
    w.insert("lm_head".into(), head_dense());
    w.insert("lm_head_gain.w_g".into(), scalar(gain));
    w
}

/// Layout (b): a quantized `lm_head.{weight,scales,biases}` and no gain.
fn layout_b(layers: usize) -> WeightMap {
    let mut w = body_weights(layers, (0.4, 0.6, 0.25), 1.0);
    let q = mlxcel_core::quantize_weights(&head_dense(), 64, 4);
    w.insert(
        "lm_head.weight".into(),
        mlxcel_core::quantized_weights_w(&q),
    );
    w.insert(
        "lm_head.scales".into(),
        mlxcel_core::quantized_weights_scales(&q),
    );
    w.insert(
        "lm_head.biases".into(),
        mlxcel_core::quantized_weights_biases(&q),
    );
    w
}

fn prefill(model: &TalkieModel, ids: &[i32]) -> Vec<f32> {
    let mut caches = LanguageModel::make_caches(model);
    let x = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
    to_vec(&LanguageModel::forward(model, &x, &mut caches, None))
}

const IDS: [i32; 6] = [3, 1, 4, 1, 5, 9];

#[test]
fn layout_a_bare_lm_head_loads_and_applies_its_gain() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = tiny_args(2);
    let one = TalkieModel::from_weights(&layout_a(2, 1.0), &args).unwrap();
    let two = TalkieModel::from_weights(&layout_a(2, 2.0), &args).unwrap();
    assert!(one.lm_head_gain.is_some());
    let (l1, l2) = (prefill(&one, &IDS), prefill(&two, &IDS));
    assert_eq!(l1.len(), IDS.len() * VOCAB);
    assert!(l1.iter().all(|v| v.is_finite()));
    let doubled: Vec<f32> = l1.iter().map(|v| v * 2.0).collect();
    assert!(max_diff(&l2, &doubled) < 1e-4);

    // The same tensor named `lm_head.weight` without a gain is the gain-1 model.
    let mut renamed = layout_a(2, 1.0);
    let head = renamed.remove("lm_head").unwrap();
    renamed.remove("lm_head_gain.w_g");
    renamed.insert("lm_head.weight".into(), head);
    let plain = TalkieModel::from_weights(&renamed, &args).unwrap();
    assert!(max_diff(&prefill(&plain, &IDS), &l1) < 1e-5);
}

#[test]
fn layout_b_quantized_head_without_gain_uses_gain_one() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = tiny_args(2);
    let b = TalkieModel::from_weights(&layout_b(2), &args).unwrap();
    assert!(
        b.lm_head_gain.is_none(),
        "a missing gain must not be invented"
    );
    assert!(b.lm_head.quantized_weight().is_some());

    let mut with_gain = layout_b(2);
    with_gain.insert("lm_head_gain.w_g".into(), scalar(1.0));
    let explicit = TalkieModel::from_weights(&with_gain, &args).unwrap();
    let (lb, le) = (prefill(&b, &IDS), prefill(&explicit, &IDS));
    assert!(lb.iter().all(|v| v.is_finite()));
    assert!(max_diff(&lb, &le) < 1e-5);

    // And the 4-bit head tracks the dense one it was quantized from.
    let dense = TalkieModel::from_weights(&layout_a(2, 1.0), &args).unwrap();
    assert!(max_diff(&lb, &prefill(&dense, &IDS)) < 0.5);
}

#[test]
fn embed_skip_adds_the_normalized_embedding() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    // Zero sublayer gains leave only the skip: h = e_x + 2 * e_x with
    // e_x = rms(embed). The embedding is scaled well away from unit RMS, so
    // starting from or skipping the raw embedding changes the result.
    let args = tiny_args(1);
    let mut w = body_weights(1, (0.0, 0.0, 2.0), 3.0);
    w.insert("lm_head.weight".into(), head_dense());
    let model = TalkieModel::from_weights(&w, &args).unwrap();
    let got = prefill(&model, &IDS);

    let embed = lcg(VOCAB * H, 101, 3.0);
    let head = lcg(VOCAB * H, 77, 0.3);
    for (pos, &id) in IDS.iter().enumerate() {
        let raw = &embed[id as usize * H..(id as usize + 1) * H];
        let e_x = rms_ref(raw, EPS);
        let h: Vec<f32> = e_x.iter().map(|v| v * 3.0).collect();
        let out = rms_ref(&h, EPS);
        let want: Vec<f32> = (0..VOCAB)
            .map(|t| (0..H).map(|j| out[j] * head[t * H + j]).sum())
            .collect();
        let row = &got[pos * VOCAB..(pos + 1) * VOCAB];
        assert!(max_diff(row, &want) < 1e-3, "position {pos}");

        // The raw-embedding variant is far from it.
        let raw_skip: Vec<f32> = e_x.iter().zip(raw).map(|(e, r)| e + 2.0 * r).collect();
        let out = rms_ref(&raw_skip, EPS);
        let wrong: Vec<f32> = (0..VOCAB)
            .map(|t| (0..H).map(|j| out[j] * head[t * H + j]).sum())
            .collect();
        assert!(max_diff(&want, &wrong) > 0.05, "position {pos}");
    }
}

#[test]
fn block_adds_gained_skip_after_both_sublayers() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = tiny_args(1);
    let w = body_weights(1, (0.0, 0.0, 0.5), 1.0);
    let block = TransformerBlock::from_weights(&w, &args, 0).unwrap();
    let xv = lcg(2 * H, 5, 4.0);
    let ev = lcg(2 * H, 6, 1.0);
    let x = mlxcel_core::from_slice_f32(&xv, &[1, 2, H as i32]);
    let e = mlxcel_core::from_slice_f32(&ev, &[1, 2, H as i32]);
    let mut cache = KVCache::new();
    let got = to_vec(&block.forward(&x, &e, &mut cache, None));
    let want: Vec<f32> = xv.iter().zip(&ev).map(|(a, b)| a + 0.5 * b).collect();
    assert!(max_diff(&got, &want) < 1e-5);
}

fn tiny_model() -> TalkieModel {
    TalkieModel::from_weights(&layout_a(2, 1.7), &tiny_args(2)).unwrap()
}

#[test]
fn native_batched_decode_matches_three_unequal_rows() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    let prompts: &[&[i32]] = &[&[1], &[2, 3], &[4, 5, 6]];
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        prompts,
        &[7, 8, 9],
        1e-4,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        &model,
        &[1, 2],
        3,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
}

#[test]
fn prefill_is_causal_and_decode_matches_it() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    let ids: Vec<i32> = (0..24).map(|i| (i * 7 + 3) % VOCAB as i32).collect();
    let full = prefill(&model, &ids);
    let short = prefill(&model, &ids[..12]);
    assert!(max_diff(&full[..12 * VOCAB], &short) < 1e-4);

    let mut caches = LanguageModel::make_caches(&model);
    let head = mlxcel_core::from_slice_i32(&ids[..23], &[1, 23]);
    LanguageModel::forward(&model, &head, &mut caches, None);
    let tok = mlxcel_core::from_slice_i32(&ids[23..], &[1, 1]);
    let step = to_vec(&LanguageModel::forward(&model, &tok, &mut caches, None));
    assert!(max_diff(&full[23 * VOCAB..], &step) < 1e-3);
    assert_eq!(LanguageModel::num_layers(&model), 2);
    assert_eq!(
        LanguageModel::eos_token_ids(&model),
        TALKIE_EOS_TOKEN_IDS.to_vec()
    );
}

crate::models::last_row_logits_harness::last_row_case!(talkie_last_row_logits, tiny_model);
