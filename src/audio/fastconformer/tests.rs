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

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;

use super::*;

/// Deterministic pseudo-random values in `[-scale, scale]`.
pub(super) fn rand_vec(n: usize, seed: &mut u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

pub(super) fn put(w: &mut WeightMap, key: &str, shape: &[i32], seed: &mut u64, scale: f32) {
    let n = shape.iter().product::<i32>() as usize;
    w.insert(
        key.to_string(),
        mlxcel_core::from_slice_f32(&rand_vec(n, seed, scale), shape),
    );
}

pub(super) fn tiny_args() -> ConformerArgs {
    ConformerArgs {
        feat_in: 16,
        n_layers: 2,
        d_model: 8,
        n_heads: 2,
        ff_expansion_factor: 2,
        subsampling_conv_channels: 4,
        conv_kernel_size: 3,
        ..ConformerArgs::default()
    }
}

/// Torch-layout weights for [`tiny_args`] under `prefix`.
pub(super) fn tiny_weights(prefix: &str) -> WeightMap {
    let a = tiny_args();
    let (d, ch, ff) = (8, 4, 16);
    let freq = a.subsampled_length(a.feat_in) as i32;
    let mut s = 7u64;
    let mut w = WeightMap::new();
    let p = format!("{prefix}.pre_encode");
    put(
        &mut w,
        &format!("{p}.conv.0.weight"),
        &[ch, 1, 3, 3],
        &mut s,
        0.5,
    );
    put(&mut w, &format!("{p}.conv.0.bias"), &[ch], &mut s, 0.1);
    for base in [2, 5] {
        put(
            &mut w,
            &format!("{p}.conv.{base}.weight"),
            &[ch, 1, 3, 3],
            &mut s,
            0.5,
        );
        put(&mut w, &format!("{p}.conv.{base}.bias"), &[ch], &mut s, 0.1);
        let pw = base + 1;
        put(
            &mut w,
            &format!("{p}.conv.{pw}.weight"),
            &[ch, ch, 1, 1],
            &mut s,
            0.5,
        );
        put(&mut w, &format!("{p}.conv.{pw}.bias"), &[ch], &mut s, 0.1);
    }
    put(
        &mut w,
        &format!("{p}.out.weight"),
        &[d, ch * freq],
        &mut s,
        0.3,
    );
    put(&mut w, &format!("{p}.out.bias"), &[d], &mut s, 0.1);
    for layer in 0..a.n_layers {
        let l = format!("{prefix}.layers.{layer}");
        for norm in [
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "conv.batch_norm",
            "norm_feed_forward2",
            "norm_out",
        ] {
            put(&mut w, &format!("{l}.{norm}.weight"), &[d], &mut s, 1.0);
            put(&mut w, &format!("{l}.{norm}.bias"), &[d], &mut s, 0.1);
        }
        for ffn in ["feed_forward1", "feed_forward2"] {
            put(
                &mut w,
                &format!("{l}.{ffn}.linear1.weight"),
                &[ff, d],
                &mut s,
                0.3,
            );
            put(
                &mut w,
                &format!("{l}.{ffn}.linear2.weight"),
                &[d, ff],
                &mut s,
                0.3,
            );
        }
        for lin in [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_out",
            "linear_pos",
        ] {
            put(
                &mut w,
                &format!("{l}.self_attn.{lin}.weight"),
                &[d, d],
                &mut s,
                0.3,
            );
        }
        put(
            &mut w,
            &format!("{l}.self_attn.pos_bias_u"),
            &[2, 4],
            &mut s,
            0.2,
        );
        put(
            &mut w,
            &format!("{l}.self_attn.pos_bias_v"),
            &[2, 4],
            &mut s,
            0.2,
        );
        put(
            &mut w,
            &format!("{l}.conv.pointwise_conv1.weight"),
            &[2 * d, d, 1],
            &mut s,
            0.3,
        );
        put(
            &mut w,
            &format!("{l}.conv.depthwise_conv.weight"),
            &[d, 1, 3],
            &mut s,
            0.5,
        );
        put(
            &mut w,
            &format!("{l}.conv.pointwise_conv2.weight"),
            &[d, d, 1],
            &mut s,
            0.3,
        );
    }
    w
}

/// Convert every conv kernel of a torch-layout map to MLX layout.
fn to_mlx_layout(torch: &WeightMap) -> WeightMap {
    torch
        .iter()
        .map(|(k, v)| {
            let converted = match mlxcel_core::array_shape(v).len() {
                4 => mlxcel_core::transpose_axes(v, &[0, 2, 3, 1]),
                3 => mlxcel_core::transpose_axes(v, &[0, 2, 1]),
                _ => mlxcel_core::copy(v),
            };
            (k.clone(), converted)
        })
        .collect()
}

pub(super) fn mel(frames: i32, feat: i32) -> UniquePtr<MlxArray> {
    let mut s = 99u64;
    let n = (frames * feat) as usize;
    mlxcel_core::from_slice_f32(&rand_vec(n, &mut s, 2.0), &[1, frames, feat])
}

#[test]
fn subsampling_lengths_128_mel_frames_give_17() {
    let args = ConformerArgs::default();
    // Causal padding (2, 1): 128 -> 65 -> 33 -> 17 (both time and frequency).
    assert_eq!(args.subsampled_length(128), 17);
    assert_eq!(args.subsampled_length(182), 24);
    assert_eq!(args.subsampled_length(1), 1);
    let symmetric = ConformerArgs {
        causal_downsampling: false,
        ..ConformerArgs::default()
    };
    assert_eq!(symmetric.subsampled_length(128), 16);

    let enc =
        FastConformerEncoder::from_weights(&tiny_weights("enc"), "enc", &tiny_args(), (64, 4))
            .unwrap();
    let (out, len) = enc.forward(&mel(128, 16), 128).unwrap();
    assert_eq!(len, 17);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 17, 8]);
}

#[test]
fn chunked_limited_mask_70_0_allows_self_and_70_left() {
    let t = 100usize;
    let mask = array_to_vec_f32(&chunked_limited_mask(t, 70, 0));
    assert_eq!(mask.len(), t * t);
    for i in 0..t {
        for j in 0..t {
            let visible = j <= i && i - j <= 70;
            assert_eq!(chunked_limited_visible(i, j, 70, 0), visible, "({i},{j})");
            let expected = if visible { 0.0 } else { NEG_INF };
            assert_eq!(mask[i * t + j], expected, "({i},{j})");
        }
    }
    // Chunks of two frames with two chunks of history: frame 5 (chunk 2) sees
    // chunks 0..=2, i.e. frames 0..=5, and frame 4 also sees frame 5.
    assert!(chunked_limited_visible(4, 5, 4, 1));
    assert!(!chunked_limited_visible(6, 1, 4, 1));
    assert!(chunked_limited_visible(6, 2, 4, 1));
    let small = array_to_vec_f32(&chunked_limited_mask(6, 4, 1));
    for i in 0..6 {
        for j in 0..6 {
            let expected = if chunked_limited_visible(i, j, 4, 1) {
                0.0
            } else {
                NEG_INF
            };
            assert_eq!(small[i * 6 + j], expected);
        }
    }
}

#[test]
fn rel_shift_matches_hand_indexing() {
    let values: Vec<f32> = (0..3)
        .flat_map(|i| (0..5).map(move |j| (10 * i + j) as f32))
        .collect();
    let x = mlxcel_core::from_slice_f32(&values, &[1, 1, 3, 5]);
    let shifted = rel_shift(&x);
    assert_eq!(mlxcel_core::array_shape(&shifted), vec![1, 1, 3, 5]);
    assert_eq!(
        array_to_vec_f32(&shifted),
        vec![
            2., 3., 4., 0., 10., 11., 12., 13., 14., 0., 20., 21., 22., 23., 24.
        ]
    );
    // The first `Tq` columns are `x[i][j + (Tq - 1 - i)]`.
    let cut = mlxcel_core::slice(&shifted, &[0, 0, 0, 0], &[1, 1, 3, 3]);
    assert_eq!(
        array_to_vec_f32(&cut),
        vec![2., 3., 4., 11., 12., 13., 20., 21., 22.]
    );
}

#[test]
fn rel_pos_embedding_rows_run_from_l_minus_1_down() {
    let pe = rel_pos_embedding(4, 8);
    assert_eq!(mlxcel_core::array_shape(&pe), vec![1, 7, 8]);
    let v = array_to_vec_f32(&pe);
    // Row 3 is position 0: sin(0) = 0 at even, cos(0) = 1 at odd channels.
    assert_eq!(&v[3 * 8..4 * 8], &[0., 1., 0., 1., 0., 1., 0., 1.]);
    // Row 0 is position +3, row 6 is position -3 (odd symmetry of sin).
    assert!((v[0] - 3f32.sin()).abs() < 1e-6);
    assert!((v[6 * 8] + 3f32.sin()).abs() < 1e-6);
    assert!((v[1] - v[6 * 8 + 1]).abs() < 1e-6);
}

#[test]
fn block_output_shape_and_layout_gate_is_idempotent() {
    let torch = tiny_weights("enc");
    let mlx = to_mlx_layout(&torch);
    let args = tiny_args();
    let a = FastConformerEncoder::from_weights(&torch, "enc", &args, (64, 4)).unwrap();
    let b = FastConformerEncoder::from_weights(&mlx, "enc", &args, (64, 4)).unwrap();
    let input = mel(40, 16);
    let (out_a, len) = a.forward(&input, 40).unwrap();
    let (out_b, _) = b.forward(&input, 40).unwrap();
    assert_eq!(len, args.subsampled_length(40));
    assert_eq!(mlxcel_core::array_shape(&out_a), vec![1, len as i32, 8]);
    let (va, vb) = (array_to_vec_f32(&out_a), array_to_vec_f32(&out_b));
    assert!(va.iter().all(|v| v.is_finite()));
    assert_eq!(va, vb);

    let block = ConformerBlock::from_weights(&torch, "enc.layers.0", &args, (64, 4)).unwrap();
    let x = mlxcel_core::from_slice_f32(&rand_vec(5 * 8, &mut 3u64, 1.0), &[1, 5, 8]);
    let pe = rel_pos_embedding(5, 8);
    let mask = chunked_limited_mask(5, 70, 0);
    let y = block.forward(&x, &pe, Some(&*mask)).unwrap();
    assert_eq!(mlxcel_core::array_shape(&y), vec![1, 5, 8]);
}

#[test]
fn perception_projects_and_returns_encoder_output() {
    let mut w = tiny_weights("p.encoder");
    let mut s = 5u64;
    put(&mut w, "p.proj.weight", &[12, 8], &mut s, 0.3);
    put(&mut w, "p.proj.bias", &[12], &mut s, 0.1);
    let perception = VoiceChatPerception::from_weights(&w, "p", &tiny_args(), (64, 4)).unwrap();
    let out = perception.forward(&mel(24, 16), 24).unwrap();
    assert_eq!(out.length, 4);
    assert_eq!(mlxcel_core::array_shape(&out.projected), vec![1, 4, 12]);
    assert_eq!(mlxcel_core::array_shape(&out.encoded), vec![1, 4, 8]);
}

#[test]
fn bad_inputs_and_weights_fail_cleanly() {
    let args = tiny_args();
    let w = tiny_weights("enc");
    let enc = FastConformerEncoder::from_weights(&w, "enc", &args, (64, 4)).unwrap();
    assert!(enc.forward(&mel(10, 12), 10).is_err());
    // A config whose frequency width disagrees with `pre_encode.out`.
    let wrong = ConformerArgs {
        feat_in: 32,
        ..tiny_args()
    };
    let err = FastConformerEncoder::from_weights(&w, "enc", &wrong, (64, 4))
        .err()
        .unwrap();
    assert!(err.contains("pre_encode.out.weight"), "{err}");
    let batch_norm = ConformerArgs {
        conv_norm_type: "batch_norm".to_string(),
        ..tiny_args()
    };
    assert!(FastConformerEncoder::from_weights(&w, "enc", &batch_norm, (64, 4)).is_err());
}

#[test]
fn config_accepts_flat_and_nested_context_sizes() {
    let nested: ConformerArgs =
        serde_json::from_str(r#"{"att_context_size": [[70, 0]], "dropout": 0.1}"#).unwrap();
    let flat: ConformerArgs = serde_json::from_str(r#"{"att_context_size": [70, 0]}"#).unwrap();
    assert_eq!(nested.att_context_size, vec![[70, 0]]);
    assert_eq!(flat, nested);
    assert_eq!(nested, ConformerArgs::default());
    assert_eq!(nested.conv_padding().unwrap(), (8, 0));
    let explicit: ConformerArgs = serde_json::from_str(
        r#"{"conv_context_size": [4, 4], "att_context_size": [[56, 13], [70, 0]]}"#,
    )
    .unwrap();
    assert_eq!(explicit.conv_padding().unwrap(), (4, 4));
    assert_eq!(explicit.default_att_context(), [56, 13]);
}

/// Affine-dequantize a quantized triple to dense f32 weights.
pub(crate) fn dequantize_triple(
    w: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    group_size: i32,
    bits: i32,
) -> UniquePtr<MlxArray> {
    // SAFETY: `biases` is a valid array that outlives the call.
    unsafe { mlxcel_core::dequantize(w, scales, biases, group_size, bits, "affine") }
}

/// Quantize dense `w` and insert the `.weight` / `.scales` / `.biases` triple
/// under `key` in `quant`, plus the dequantized dense weight in `dense`.
pub(crate) fn insert_quantized(
    quant: &mut WeightMap,
    dense: &mut WeightMap,
    key: &str,
    w: &MlxArray,
    group_size: i32,
    bits: i32,
) {
    let q = mlxcel_core::quantize_weights(w, group_size, bits);
    let (qw, scales) = (
        mlxcel_core::quantized_weights_w(&q),
        mlxcel_core::quantized_weights_scales(&q),
    );
    let biases = mlxcel_core::quantized_weights_biases(&q);
    dense.insert(
        format!("{key}.weight"),
        dequantize_triple(&qw, &scales, &biases, group_size, bits),
    );
    quant.insert(format!("{key}.weight"), qw);
    quant.insert(format!("{key}.scales"), scales);
    quant.insert(format!("{key}.biases"), biases);
}

#[test]
fn attention_loads_non_default_quantization() {
    const D: i32 = 64;
    let mut seed = 21u64;
    let mut dense_src = WeightMap::new();
    let (mut quant, mut reference) = (WeightMap::new(), WeightMap::new());
    for name in [
        "linear_q",
        "linear_k",
        "linear_v",
        "linear_out",
        "linear_pos",
    ] {
        put(&mut dense_src, name, &[D, D], &mut seed, 0.3);
        let w = dense_src.remove(name).unwrap();
        insert_quantized(
            &mut quant,
            &mut reference,
            &format!("att.{name}"),
            &w,
            32,
            8,
        );
    }
    for (name, scale) in [("pos_bias_u", 0.2), ("pos_bias_v", 0.2)] {
        let mut b = WeightMap::new();
        put(&mut b, name, &[4, 16], &mut seed, scale);
        let b = b.remove(name).unwrap();
        quant.insert(format!("att.{name}"), mlxcel_core::copy(&b));
        reference.insert(format!("att.{name}"), b);
    }

    let t = 5;
    let x = mlxcel_core::from_slice_f32(&rand_vec((t * D) as usize, &mut seed, 1.0), &[1, t, D]);
    let pos = rel_pos_embedding(t as usize, D as usize);
    let run = |weights: &WeightMap, quantization: (i32, i32)| {
        RelPositionMultiHeadAttention::from_weights(weights, "att", 4, D as usize, quantization)
            .map(|att| array_to_vec_f32(&att.forward(&x, &pos, None)))
    };
    let expected = run(&reference, (64, 4)).unwrap();
    let got = run(&quant, (32, 8)).unwrap();
    let max_diff = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (p, q)| m.max((p - q).abs()))
    };
    assert!(max_diff(&expected, &got) < 1e-4);
    assert!(expected.iter().any(|v| v.abs() > 1e-3));

    // The old hard-coded (64, 4) cannot describe this checkpoint: the scales
    // hold one column per 32 inputs, not per 64. Running it through the
    // quantized matmul raises an uncatchable MLX C++ exception (abort), so
    // the mismatch is asserted on the stored layout instead.
    let scales = mlxcel_core::array_shape(quant.get("att.linear_q.scales").unwrap());
    assert_eq!(scales, vec![D, D / 32]);
    assert_ne!(scales[1], D / 64);
}

/// Issue #2045 (P1): perception holds its bf16 weights as f32, which must be
/// bit-identical to MLX promoting the bf16 weights inside every op. The
/// reference spells that promotion out as an in-graph cast, because CUDA
/// builds resolve bf16 + f32 to bf16 instead (issue #2087).
#[test]
fn perception_f32_weights_match_the_promoted_bf16_path() {
    let mut w = tiny_weights("p.encoder");
    let mut s = 5u64;
    put(&mut w, "p.proj.weight", &[12, 8], &mut s, 0.3);
    put(&mut w, "p.proj.bias", &[12], &mut s, 0.1);
    let cast = |w: &WeightMap, dt| -> WeightMap {
        w.iter()
            .map(|(k, v)| (k.clone(), mlxcel_core::astype(v, dt)))
            .collect()
    };
    let w = cast(&w, mlxcel_core::dtype::BFLOAT16);
    let perception = VoiceChatPerception::from_weights(&w, "p", &tiny_args(), (64, 4)).unwrap();
    let promoted = cast(&w, mlxcel_core::dtype::FLOAT32);
    let encoder =
        FastConformerEncoder::from_weights(&promoted, "p.encoder", &tiny_args(), (64, 4)).unwrap();
    let proj =
        mlxcel_core::layers::UnifiedLinear::from_weights(&promoted, "p.proj", 64, 4).unwrap();
    let input = mel(24, 16);
    let out = perception.forward(&input, 24).unwrap();
    let (encoded, _) = encoder.forward(&input, 24).unwrap();
    let projected = proj.forward(&encoded);
    let bytes = |a: &mlxcel_core::MlxArray| {
        mlxcel_core::eval(a);
        mlxcel_core::array_to_raw_bytes(a)
    };
    assert_eq!(
        mlxcel_core::array_dtype(&out.encoded),
        mlxcel_core::dtype::FLOAT32
    );
    assert_eq!(bytes(&out.encoded), bytes(&encoded));
    assert_eq!(bytes(&out.projected), bytes(&projected));
}
