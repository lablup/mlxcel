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

//! Raon-1 mini building blocks against hand computations: SiTU, the RDA
//! write ratio and decay, NoPE MLA, and the sanitize passes.

use super::Raon1MiniModel;
use super::tests::{
    Fixture, HIDDEN, KERNEL, KV_LORA, MLA_HEADS, NOPE, Q_LORA, RDA_DIM, RDA_HEADS, ROPE, V_HEAD,
    e4m3_bytes, raw_weights, tiny_config, values,
};
use crate::models::kimi_k3::{KimiK3MLAAttention, NopeMlaSpec};
use crate::models::raon_rda::{rda_core, rda_decay, write_ratio};
use crate::models::raon1_mini_sanitize::stack_mxfp8_experts;
use crate::models::switch_layers::situ_activation;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

fn guard() -> crate::models::embedding_test_support::MlxTestGuard {
    crate::models::embedding_test_support::mlx_test_guard()
}

fn arr(data: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(data, shape)
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn situ_is_zero_at_zero_and_saturates_at_both_ends() {
    let _guard = guard();
    let gate = arr(&[0.0, 100.0, -100.0, 2.0], &[4]);
    let up = arr(&[5.0, 100.0, 1.0, -3.0], &[4]);
    let out = values(&situ_activation(&gate, &up, 4.0, Some(25.0)));
    assert_eq!(out[0], 0.0);
    // 4 tanh(25) sigmoid(100) * 25 tanh(4): both tanh caps engage.
    let high = 4.0 * 25.0 * 4f64.tanh();
    assert!((out[1] as f64 - high).abs() < 1e-3, "{out:?}");
    assert!(
        out[2].abs() < 1e-30,
        "a large negative gate closes SiTU: {out:?}"
    );
    let mid = 4.0 * (0.5f64).tanh() * sig(2.0) * 25.0 * (-3.0f64 / 25.0).tanh();
    assert!((out[3] as f64 - mid).abs() < 1e-5, "{out:?}");
}

#[test]
fn write_ratio_and_lower_bounded_decay_match_a_hand_computation() {
    let _guard = guard();
    let b = [0.3f32, -2.0, 4.0];
    let d = [-1.2f32, 0.7, -9.0];
    let got = values(&write_ratio(&arr(&b, &[3]), &arr(&d, &[3])));
    for i in 0..3 {
        let want = sig((b[i] + d[i]) as f64) / sig(b[i] as f64);
        assert!((got[i] as f64 - want).abs() < 1e-6, "ratio {i}: {got:?}");
    }

    // g = exp(-5 sigmoid(exp(A_log) (a + dt_bias))), per head and channel.
    let a_log = [0.5f32, -1.0];
    let a = [0.2f32, -0.4, 1.5, 3.0];
    let dt = [0.1f32, 0.0, -0.5, 0.25];
    let decay = |bound: Option<f32>| {
        values(&rda_decay(
            &arr(&a_log, &[2, 1]),
            &arr(&a, &[1, 1, 2, 2]),
            &arr(&dt, &[2, 2]),
            bound,
        ))
    };
    let g = decay(Some(-5.0));
    // A null bound is `exp(-softplus(exp(A_log) (a + dt_bias)))`, the scale
    // inside the softplus as in the reference (not `compute_g`'s outside).
    let unbounded = decay(None);
    let softplus = |x: f64| (1.0 + x.exp()).ln();
    for (h, &a_log_h) in a_log.iter().enumerate() {
        for c in 0..2 {
            let i = h * 2 + c;
            let s = (a_log_h as f64).exp() * (a[i] + dt[i]) as f64;
            let want = (-5.0 * sig(s)).exp();
            assert!((g[i] as f64 - want).abs() < 1e-6, "decay {h},{c}: {g:?}");
            assert!(g[i] > (-5f32).exp() && g[i] < 1.0);
            let want = (-softplus(s)).exp();
            assert!(
                (unbounded[i] as f64 - want).abs() < 1e-6,
                "unbounded decay {h},{c}: {unbounded:?}"
            );
        }
    }
}

/// The RDA recurrence keeps the float32 decay. One write at t = 0 and
/// `beta = 0` after it leave `y_t = g^t`, so a decay of 0.998 read as bf16
/// (0.99609375) shows as 0.781 instead of 0.881 by t = 63; the reference
/// computes the decay and the state in fp32.
#[test]
fn rda_core_keeps_the_float32_decay_and_state_over_a_long_sequence() {
    let _guard = guard();
    let (t, d) = (64usize, 128usize);
    let g_value = 0.998f32;
    let mut unit = vec![0.0f32; t * d];
    for step in 0..t {
        unit[step * d] = 1.0;
    }
    let bf = |a: UniquePtr<MlxArray>| mlxcel_core::astype(&a, dtype::BFLOAT16);
    let q = bf(arr(&unit, &[1, t as i32, 1, d as i32]));
    let k = bf(arr(&unit, &[1, t as i32, 1, d as i32]));
    let v = bf(arr(&vec![1.0f32; t * d], &[1, t as i32, 1, d as i32]));
    let mut beta = vec![0.0f32; t];
    beta[0] = 1.0;
    let beta = bf(arr(&beta, &[1, t as i32, 1]));
    let g = arr(&vec![g_value; t * d], &[1, t as i32, 1, d as i32]);

    let (y, state) = rda_core(&q, &k, &v, &g, &beta, None);
    assert_eq!(mlxcel_core::array_dtype(&state), dtype::FLOAT32);
    let y = values(&y);
    for step in [1usize, 16, 63] {
        let want = (g_value as f64).powi(step as i32);
        let got = y[step * d] as f64;
        assert!((got - want).abs() < 4e-3, "y[{step}] = {got}, want {want}");
    }
    let s = values(&state);
    let want = (g_value as f64).powi(t as i32 - 1);
    assert!(
        (s[0] as f64 - want).abs() < 1e-5,
        "state = {}, want {want}",
        s[0]
    );
}

/// NoPE MLA attends without positions, so the last query's output is the
/// same for any order of the keys before it. A rotation of `k_pe` / `q_pe`
/// would break that.
#[test]
fn mla_applies_no_rotation_to_the_rope_half() {
    let _guard = guard();
    let mut f = Fixture::new(31);
    f.mla("attn");
    let spec = NopeMlaSpec {
        hidden_size: HIDDEN,
        num_heads: MLA_HEADS,
        q_lora_rank: Some(Q_LORA),
        kv_lora_rank: KV_LORA,
        qk_nope_head_dim: NOPE,
        qk_rope_head_dim: ROPE,
        v_head_dim: V_HEAD,
        use_output_gate: true,
        latent_norm_eps: 1e-6,
        group_size: 64,
        bits: 4,
        family: "raon1_mini",
    };
    let mut weights = f.into_weights();
    crate::models::kimi_linear::decompose_kv_b_proj(
        &mut weights,
        "attn",
        0,
        MLA_HEADS as i32,
        NOPE as i32,
        V_HEAD as i32,
        KV_LORA as i32,
    )
    .unwrap();
    let attn = KimiK3MLAAttention::from_spec(&weights, &spec, "attn").expect("mla loads");

    let rows = super::tests::noise(4 * HIDDEN, 77);
    let row = |i: usize| rows[i * HIDDEN..(i + 1) * HIDDEN].to_vec();
    let run = |order: [usize; 4]| {
        let x: Vec<f32> = order.iter().flat_map(|&i| row(i)).collect();
        let out = attn.forward(&arr(&x, &[1, 4, HIDDEN as i32]), None, &mut KVCache::new());
        values(&out)[3 * HIDDEN..].to_vec()
    };
    let a = run([0, 1, 2, 3]);
    let b = run([2, 0, 1, 3]);
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(scale > 1e-3, "degenerate output");
    assert!(
        worst <= 1e-5 * scale.max(1.0),
        "the last row depends on key order: {worst}"
    );
    // Distinct rows really do attend differently, so the check has teeth.
    assert_ne!(
        run([0, 1, 2, 3])[..HIDDEN.min(8)],
        run([0, 1, 3, 2])[..HIDDEN.min(8)]
    );
}

#[test]
fn sanitize_fuses_convs_in_qkv_order_for_both_source_shapes_and_splits_kv_b() {
    let _guard = guard();
    let config = tiny_config();
    let raw = raw_weights();
    let p = RDA_HEADS * RDA_DIM;
    let src: Vec<Vec<f32>> = [0usize, 3]
        .iter()
        .flat_map(|layer| {
            ["q", "k", "v"]
                .map(|c| values(&raw[&format!("model.layers.{layer}.self_attn.{c}_conv1d.weight")]))
        })
        .collect();
    let kv_b = values(&raw["model.layers.2.self_attn.kv_b_proj.weight"]);
    let weights = Raon1MiniModel::sanitize_weights(raw, &config).unwrap();

    for (n, layer) in [0usize, 3].iter().enumerate() {
        let key = format!("model.layers.{layer}.self_attn.qkv_conv1d.weight");
        let fused = &weights[&key];
        assert_eq!(
            mlxcel_core::array_shape(fused),
            vec![(3 * p) as i32, KERNEL as i32, 1]
        );
        let want: Vec<f32> = src[n * 3..n * 3 + 3].concat();
        assert_eq!(values(fused), want, "{key}");
        assert!(!weights.contains_key(&format!("model.layers.{layer}.self_attn.q_conv1d.weight")));
    }

    let attn = "model.layers.2.self_attn";
    assert!(!weights.contains_key(&format!("{attn}.kv_b_proj.weight")));
    let wk = &weights[&format!("{attn}.embed_q.weight")];
    let wv = &weights[&format!("{attn}.unembed_out.weight")];
    assert_eq!(
        mlxcel_core::array_shape(wk),
        vec![MLA_HEADS as i32, KV_LORA as i32, NOPE as i32]
    );
    assert_eq!(
        mlxcel_core::array_shape(wv),
        vec![MLA_HEADS as i32, V_HEAD as i32, KV_LORA as i32]
    );
    // kv_b rows per head: NOPE key rows, then V_HEAD value rows.
    let (wk, wv) = (values(wk), values(wv));
    for h in 0..MLA_HEADS {
        for r in 0..NOPE {
            for c in 0..KV_LORA {
                let src = kv_b[(h * (NOPE + V_HEAD) + r) * KV_LORA + c];
                assert_eq!(wk[(h * KV_LORA + c) * NOPE + r], src);
            }
        }
        for r in 0..V_HEAD {
            for c in 0..KV_LORA {
                let src = kv_b[(h * (NOPE + V_HEAD) + NOPE + r) * KV_LORA + c];
                assert_eq!(wv[(h * V_HEAD + r) * KV_LORA + c], src);
            }
        }
    }
}

fn u8_values(a: &MlxArray) -> Vec<u8> {
    let bytes = mlxcel_core::view(&mlxcel_core::contiguous(a, false), dtype::UINT8);
    values(&bytes).iter().map(|&v| v as u8).collect()
}

#[test]
fn mxfp8_experts_are_viewed_without_changing_a_byte() {
    let _guard = guard();
    let (experts, rows, cols) = (3usize, 4usize, 64usize);
    let mut weights = WeightMap::new();
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    for e in 0..experts {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let c = e4m3_bytes(rows * cols, 100 + e as u32);
            let s: Vec<u8> = (0..rows * cols / 32).map(|i| 120 + (i + e) as u8).collect();
            let key = format!("moe.experts.{e}.{proj}");
            let shape = [rows as i32, cols as i32];
            weights.insert(
                format!("{key}.weight"),
                mlxcel_core::from_bytes(&c, &shape, dtype::UINT8),
            );
            let s_shape = [rows as i32, (cols / 32) as i32];
            weights.insert(
                format!("{key}.weight_scale_inv"),
                mlxcel_core::from_bytes(&s, &s_shape, dtype::UINT8),
            );
            if proj == "up_proj" {
                codes.extend(c);
                scales.extend(s);
            }
        }
    }
    stack_mxfp8_experts(&mut weights, "moe", experts).unwrap();
    let packed = &weights["moe.switch_mlp.up_proj.weight"];
    assert_eq!(mlxcel_core::array_dtype(packed), dtype::UINT32);
    assert_eq!(
        mlxcel_core::array_shape(packed),
        vec![experts as i32, rows as i32, (cols / 4) as i32]
    );
    assert_eq!(u8_values(packed), codes);
    assert_eq!(u8_values(&weights["moe.switch_mlp.up_proj.scales"]), scales);
    assert!(!weights.keys().any(|k| k.contains(".experts.")));

    // A plane that is not uint8 (a loader that widened E4M3) is refused by name.
    let mut widened = WeightMap::new();
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let key = format!("moe.experts.0.{proj}");
        widened.insert(
            format!("{key}.weight"),
            mlxcel_core::zeros(&[4, 64], dtype::FLOAT16),
        );
        widened.insert(
            format!("{key}.weight_scale_inv"),
            mlxcel_core::zeros(&[4, 2], dtype::UINT8),
        );
    }
    let err = stack_mxfp8_experts(&mut widened, "moe", 1).unwrap_err();
    assert!(err.contains("uint8"), "{err}");

    // The native mxfp8 matmul equals dequantize-then-matmul on these rows.
    let x = arr(&super::tests::noise(2 * cols, 5), &[2, cols as i32]);
    let w0 = mlxcel_core::slice(packed, &[0, 0, 0], &[1, rows as i32, (cols / 4) as i32]);
    let w0 = mlxcel_core::squeeze_axis(&w0, 0);
    let s_all = &weights["moe.switch_mlp.up_proj.scales"];
    let s0 = mlxcel_core::squeeze_axis(
        &mlxcel_core::slice(s_all, &[0, 0, 0], &[1, rows as i32, 2]),
        0,
    );
    // SAFETY: every operand is a live array; a null bias is how mxfp8 is called.
    let native = unsafe {
        mlxcel_core::quantized_matmul(&x, &w0, &s0, std::ptr::null(), true, 32, 8, "mxfp8")
    };
    let native = values(&native);
    let dense: Vec<f32> = codes[..rows * cols]
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let scale = 2f32.powi(scales[(i / cols) * (cols / 32) + (i % cols) / 32] as i32 - 127);
            crate::models::f8_e4m3_to_f32(c) * scale
        })
        .collect();
    let xs = values(&x);
    for r in 0..2 {
        for o in 0..rows {
            let want: f32 = (0..cols)
                .map(|i| xs[r * cols + i] * dense[o * cols + i])
                .sum();
            let got = native[r * rows + o];
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "row {r} out {o}: {got} vs {want}"
            );
        }
    }
}
