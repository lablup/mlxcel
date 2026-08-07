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

//! Unit tests for the Ling / Ring linear-attention MoE loader.
//!
//! Everything here is checkpoint-free. The config tests parse the verbatim
//! `mlx-community/Ring-mini-linear-2.0-4bit` `config.json`; the shape tests
//! build synthetic weight maps whose tensor names and shapes mirror the real
//! export; the numerics tests drive the GLA kernels directly.
//!
//! Four groups carry more weight than the rest, because the real-checkpoint
//! gate cannot see what they cover:
//!
//! 1. **The chunked prefill against the recurrence it replaces.**
//!    [`super::gla_chunked`] is a closed form, not a transcription of upstream's
//!    `for t in range(L)` loop, so
//!    `chunked_gla_matches_the_sequential_recurrence` reimplements the loop
//!    naively and asserts the two agree, with and without an incoming state,
//!    across a chunk boundary. A wrong closed form still produces finite,
//!    plausible output.
//! 2. **The decay schedule**, which is read from no tensor at all. If
//!    [`super::alibi_slopes`] or [`super::layer_decay`] is off, every linear
//!    layer weights its history wrong and nothing in the checkpoint disagrees.
//!    Both are pinned against values computed from the upstream formula by hand.
//! 3. **`GroupRMSNorm` against a plain RMSNorm**, the substitution that looks
//!    right and is not.
//! 4. **The two fused-QKV widths.** A linear layer's `query_key_value` is
//!    `3 * H * head_dim` wide and a global layer's is `(H + 2 * H_kv) * head_dim`;
//!    on Ring-mini that is 6144 against 3072 in the same checkpoint, and MLX's
//!    `slice` clamps rather than throws, so validating both against one width
//!    would let a mislabeled layer read the wrong channels.

use super::{
    BailingLinearCache, BailingMoeLinearModel, GroupRMSNorm, LinearAttentionCache, ModelArgs,
    Quantization, TokenIdField, alibi_slopes, chunked_prefill_enabled_from, gla_chunked,
    gla_sequential, gla_step, layer_decay, validate_weights,
};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// The real checkpoint's config.

/// `mlx-community/Ring-mini-linear-2.0-4bit`'s `config.json`.
///
/// Reproduced field for field, including the keys this loader ignores and the
/// per-tensor quantization overrides, because the point of the parse test is
/// that serde accepts the file exactly as shipped. The override list is
/// truncated to three layers; the loader never reads those keys (per-tensor bits
/// are reconciled from the tensors themselves), and carrying all 19 would add
/// 100 lines of noise without testing anything more.
const RING_MINI_CONFIG: &str = r#"{
    "architectures": ["BailingMoeLinearV2ForCausalLM"],
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_bailing_moe_linear_v2.BailingMoeLinearV2Config"
    },
    "embedding_dropout": 0.0,
    "eos_token_id": 156892,
    "first_k_dense_replace": 1,
    "group_norm_size": 4,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 5120,
    "layer_group_size": 5,
    "linear_silu": false,
    "max_position_embeddings": 131072,
    "model_type": "bailing_moe_linear",
    "moe_intermediate_size": 512,
    "moe_router_enable_expert_bias": true,
    "moe_shared_expert_intermediate_size": 512,
    "n_group": 8,
    "norm_topk_prob": true,
    "num_attention_heads": 16,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 20,
    "num_key_value_heads": 4,
    "num_nextn_predict_layers": 0,
    "num_shared_experts": 1,
    "output_dropout": 0.0,
    "pad_token_id": 156892,
    "partial_rotary_factor": 0.5,
    "quantization": {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        "model.layers.1.mlp.gate.gate_proj": {"group_size": 64, "bits": 8},
        "model.layers.2.mlp.gate.gate_proj": {"group_size": 64, "bits": 8},
        "model.layers.3.mlp.gate.gate_proj": {"group_size": 64, "bits": 8}
    },
    "quantization_config": {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        "model.layers.1.mlp.gate.gate_proj": {"group_size": 64, "bits": 8}
    },
    "rms_norm_eps": 1e-06,
    "rope_scaling": null,
    "rope_theta": 1000000,
    "routed_scaling_factor": 2.5,
    "router_dtype": "fp32",
    "score_function": "sigmoid",
    "tie_word_embeddings": false,
    "topk_group": 4,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.56.1",
    "use_bias": false,
    "use_cache": true,
    "use_qk_norm": true,
    "use_qkv_bias": false,
    "use_rmsnorm": true,
    "vocab_size": 157184
}"#;

fn ring_mini_args() -> ModelArgs {
    serde_json::from_str(RING_MINI_CONFIG).expect("Ring-mini config parses")
}

// Config parsing.

#[test]
fn the_real_config_parses_and_validates() {
    let args = ring_mini_args();
    assert_eq!(args.model_type, "bailing_moe_linear");
    assert_eq!(args.hidden_size, 2048);
    assert_eq!(args.num_hidden_layers, 20);
    assert_eq!(args.num_attention_heads, 16);
    assert_eq!(args.num_key_value_heads, Some(4));
    assert_eq!(args.head_dim, Some(128));
    assert_eq!(args.layer_group_size, 5);
    assert_eq!(args.group_norm_size, 4);
    assert_eq!(args.first_k_dense_replace, 1);
    assert_eq!(args.num_experts, Some(256));
    assert_eq!(args.num_experts_per_tok, 8);
    assert_eq!(args.num_shared_experts, 1);
    assert_eq!(args.moe_shared_expert_intermediate_size, Some(512));
    assert_eq!(args.n_group, 8);
    assert_eq!(args.topk_group, 4);
    assert_eq!(args.score_function, "sigmoid");
    assert!(args.moe_router_enable_expert_bias);
    assert!(args.use_qk_norm);
    assert!(!args.tie_word_embeddings);
    assert_eq!(args.routed_scaling_factor, 2.5);
    assert_eq!(args.partial_rotary_factor, 0.5);
    assert_eq!(args.eos_token_ids(), vec![156892]);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    args.validate().expect("the shipped config is accepted");
}

#[test]
fn a_list_valued_eos_token_id_parses() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
            "intermediate_size": 8, "vocab_size": 16, "eos_token_id": [1, 2, 3]}"#,
    )
    .expect("a list eos_token_id parses");
    assert!(matches!(args.eos_token_id, Some(TokenIdField::Multiple(_))));
    assert_eq!(args.eos_token_ids(), vec![1, 2, 3]);
}

// The hybrid schedule.

#[test]
fn the_layer_schedule_makes_every_fifth_layer_global() {
    let args = ring_mini_args();
    let global: Vec<usize> = (0..args.num_hidden_layers)
        .filter(|&i| args.is_global_layer(i))
        .collect();
    assert_eq!(global, vec![4, 9, 14, 19]);
    assert_eq!(
        (0..args.num_hidden_layers)
            .filter(|&i| !args.is_global_layer(i))
            .count(),
        16
    );
    assert_eq!(args.global_layer_index(), Some(4));
}

#[test]
fn the_tail_clause_makes_the_leftover_layers_global() {
    // 22 layers in groups of 5: 22 // 5 * 5 == 20, so layers 20 and 21 are
    // global on top of the modulus's 4, 9, 14 and 19. This clause never fires on
    // Ring-mini and is the half of the predicate a real-checkpoint run cannot
    // reach.
    let mut args = ring_mini_args();
    args.num_hidden_layers = 22;
    let global: Vec<usize> = (0..22).filter(|&i| args.is_global_layer(i)).collect();
    assert_eq!(global, vec![4, 9, 14, 19, 20, 21]);
}

#[test]
fn a_stack_shorter_than_one_group_still_finds_a_global_layer() {
    // Upstream hardcodes `attn_idx = layer_group_size - 1` and would index a
    // 3-element cache list at 4.
    let mut args = ring_mini_args();
    args.num_hidden_layers = 3;
    assert!((0..3).all(|i| args.is_global_layer(i)));
    assert_eq!(args.global_layer_index(), Some(0));
}

// The two fused-QKV widths.

#[test]
fn the_linear_and_global_layers_have_different_qkv_widths() {
    let args = ring_mini_args();
    // (16 + 2 * 4) * 128
    assert_eq!(args.attention_qkv_out_features(), 3072);
    // 3 * 16 * 128: the linear path is MHA, not GQA.
    assert_eq!(args.linear_qkv_out_features(), 6144);
    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.linear_head_dim(), 128);
    // partial_rotary_factor 0.5 over a 128-wide head.
    assert_eq!(args.rope_dims(), 64);
    assert_eq!(args.linear_rope_dims(), 64);
}

#[test]
fn an_explicit_head_dim_does_not_reach_the_linear_layers() {
    // A config whose declared head_dim disagrees with hidden_size /
    // num_attention_heads has two head widths in one stack, because upstream's
    // LinearAttention never consults args.head_dim. Deriving both from one field
    // would size the linear split wrong, and MLX's slice clamps rather than
    // throws.
    let mut args = ring_mini_args();
    args.head_dim = Some(64);
    assert_eq!(args.head_dim(), 64);
    assert_eq!(args.linear_head_dim(), 128);
    assert_eq!(args.attention_qkv_out_features(), (16 + 8) * 64);
    assert_eq!(args.linear_qkv_out_features(), 3 * 16 * 128);
}

// The decay schedule, which is read from no tensor.

#[test]
fn the_alibi_slopes_match_the_upstream_formula_for_a_power_of_two() {
    let slopes = alibi_slopes(16);
    assert_eq!(slopes.len(), 16);
    // ratio = 2^(-2^-(log2(16) - 3)) = 2^(-2^-1) = 2^-0.5
    let ratio = 2f64.powf(-0.5);
    for (i, slope) in slopes.iter().enumerate() {
        let expected = ratio.powi(i as i32 + 1) as f32;
        assert!(
            (slope - expected).abs() < 1e-6,
            "slope[{i}] = {slope}, expected {expected}"
        );
    }
    // Strictly decreasing, all in (0, 1].
    for pair in slopes.windows(2) {
        assert!(pair[1] < pair[0]);
    }
    assert!(slopes[0] > 0.0 && slopes[0] <= 1.0);
}

#[test]
fn the_alibi_slopes_fall_back_for_a_non_power_of_two_head_count() {
    // Upstream: the 8-head schedule, then every other slope of the 16-head
    // schedule to fill the remaining 4. Unreachable on Ring-mini (16 heads).
    let slopes = alibi_slopes(12);
    assert_eq!(slopes.len(), 12);
    let base = alibi_slopes(8);
    assert_eq!(&slopes[..8], &base[..]);
    let doubled = alibi_slopes(16);
    for (k, slope) in slopes[8..].iter().enumerate() {
        assert!((slope - doubled[2 * k]).abs() < 1e-6);
    }
}

#[test]
fn the_layer_decay_is_negative_and_shrinks_with_depth() {
    let heads = 16;
    let layers = 20;
    let slopes = alibi_slopes(heads);

    // Layers 0 and 1 share a factor: max(0, layer_idx - 1) clamps both to 0.
    let g0 = layer_decay(heads, 0, layers);
    let g1 = layer_decay(heads, 1, layers);
    assert_eq!(g0, g1);
    for (i, g) in g0.iter().enumerate() {
        let expected = (-(slopes[i] as f64) * (1.0 + 1e-5)) as f32;
        assert!((g - expected).abs() < 1e-6, "g[{i}] = {g}");
    }

    // The factor shrinks monotonically with depth and bottoms out at
    // `1 - (L - 2) / (L - 1) + 1e-5`, which is `1 / (L - 1) + 1e-5` and NOT
    // `1e-5`: `layer_pos` is `layer_idx - 1`, so it reaches `denom` only at
    // `layer_idx == num_hidden_layers`, one past the last layer.
    let g_last = layer_decay(heads, layers - 1, layers);
    let last_factor = 1.0 - ((layers - 2) as f64 / (layers - 1) as f64) + 1e-5;
    for (i, g) in g_last.iter().enumerate() {
        let expected = (-(slopes[i] as f64) * last_factor) as f32;
        assert!((g - expected).abs() < 1e-7, "g[{i}] = {g}");
    }
    for layer in 1..layers {
        let prev = layer_decay(heads, layer - 1, layers)[0];
        let cur = layer_decay(heads, layer, layers)[0];
        assert!(
            cur >= prev,
            "decay must weaken with depth: layer {layer} gave {cur} after {prev}"
        );
    }

    // Every entry is negative, so exp(g) < 1 and the recurrence decays.
    for layer in 0..layers {
        for g in layer_decay(heads, layer, layers) {
            assert!(g < 0.0, "layer {layer} produced a non-negative decay {g}");
        }
    }
}

// The GLA recurrence.

/// The naive transcription of upstream's `recurrent_gla`, on the host.
///
/// `q`, `k` are `[H][L][D]`, `v` is `[H][L][Dv]`. Returns `y[H][L][Dv]` and the
/// final state `h[H][D][Dv]`. Batch is 1 throughout, which is the only shape the
/// generation path builds.
fn reference_gla(
    q: &[Vec<Vec<f32>>],
    k: &[Vec<Vec<f32>>],
    v: &[Vec<Vec<f32>>],
    g: &[f32],
    scale: f32,
    state: Option<&Vec<Vec<Vec<f32>>>>,
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>) {
    let heads = q.len();
    let len = q[0].len();
    let dk = q[0][0].len();
    let dv = v[0][0].len();

    let mut h: Vec<Vec<Vec<f32>>> = match state {
        Some(s) => s.clone(),
        None => vec![vec![vec![0.0; dv]; dk]; heads],
    };
    let seeded = state.is_some();
    let mut y = vec![vec![vec![0.0; dv]; len]; heads];

    for head in 0..heads {
        let decay = g[head].exp();
        for t in 0..len {
            for a in 0..dk {
                for b in 0..dv {
                    let carried = if t == 0 && !seeded {
                        0.0
                    } else {
                        h[head][a][b] * decay
                    };
                    h[head][a][b] = carried + k[head][t][a] * v[head][t][b];
                }
            }
            for b in 0..dv {
                let mut acc = 0.0;
                for a in 0..dk {
                    acc += q[head][t][a] * scale * h[head][a][b];
                }
                y[head][t][b] = acc;
            }
        }
    }
    (y, h)
}

/// Deterministic pseudo-random filler in `[-0.5, 0.5)`.
fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn nested(flat: &[f32], heads: usize, len: usize, dim: usize) -> Vec<Vec<Vec<f32>>> {
    (0..heads)
        .map(|h| {
            (0..len)
                .map(|t| {
                    let base = (h * len + t) * dim;
                    flat[base..base + dim].to_vec()
                })
                .collect()
        })
        .collect()
}

fn to_array(flat: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(flat, shape)
}

fn read_all(array: &MlxArray) -> Vec<f32> {
    let flat = mlxcel_core::reshape(array, &[-1]);
    let n = mlxcel_core::array_shape(&flat)[0];
    (0..n)
        .map(|i| mlxcel_core::item_f32(&mlxcel_core::slice(&flat, &[i], &[i + 1])))
        .collect()
}

#[test]
fn chunked_gla_matches_the_sequential_recurrence() {
    // A sequence longer than the chunk length, so the inter-chunk carry is
    // exercised rather than skipped.
    let (heads, len, dim) = (3usize, 10usize, 4usize);
    let chunk = 4;

    let q_flat = noise(heads * len * dim, 0x1234_5678);
    let k_flat = noise(heads * len * dim, 0x2345_6789);
    let v_flat = noise(heads * len * dim, 0x3456_789a);
    let g: Vec<f32> = vec![-0.05, -0.3, -1.1];
    let scale = 0.5;

    let shape = [1, heads as i32, len as i32, dim as i32];
    let (out, state) = gla_chunked(
        &to_array(&q_flat, &shape),
        &to_array(&k_flat, &shape),
        &to_array(&v_flat, &shape),
        &to_array(&g, &[heads as i32]),
        scale,
        None,
        chunk,
    );

    let (expected_y, expected_h) = reference_gla(
        &nested(&q_flat, heads, len, dim),
        &nested(&k_flat, heads, len, dim),
        &nested(&v_flat, heads, len, dim),
        &g,
        scale,
        None,
    );

    let got_y = read_all(&out);
    let flat_expected: Vec<f32> = expected_y.iter().flatten().flatten().copied().collect();
    assert_eq!(got_y.len(), flat_expected.len());
    for (i, (got, want)) in got_y.iter().zip(flat_expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "output[{i}]: chunked {got} vs recurrence {want}"
        );
    }

    let got_h = read_all(&state);
    let flat_h: Vec<f32> = expected_h.iter().flatten().flatten().copied().collect();
    assert_eq!(got_h.len(), flat_h.len());
    for (i, (got, want)) in got_h.iter().zip(flat_h.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "state[{i}]: chunked {got} vs recurrence {want}"
        );
    }
}

#[test]
fn chunked_gla_carries_an_incoming_state() {
    // The decode-then-prefill shape: a non-empty state entering a multi-token
    // call. Getting the `h_0 * exp(g * (i + 1))` term wrong is invisible without
    // this, because a prefill from an empty cache never reaches it.
    let (heads, len, dim) = (2usize, 5usize, 3usize);
    let q_flat = noise(heads * len * dim, 0xabcd_1234);
    let k_flat = noise(heads * len * dim, 0xbcde_2345);
    let v_flat = noise(heads * len * dim, 0xcdef_3456);
    let h_flat = noise(heads * dim * dim, 0xdef0_4567);
    let g: Vec<f32> = vec![-0.2, -0.9];
    let scale = 0.7;

    let shape = [1, heads as i32, len as i32, dim as i32];
    let state_shape = [1, heads as i32, dim as i32, dim as i32];
    let incoming = to_array(&h_flat, &state_shape);

    let (out, state) = gla_chunked(
        &to_array(&q_flat, &shape),
        &to_array(&k_flat, &shape),
        &to_array(&v_flat, &shape),
        &to_array(&g, &[heads as i32]),
        scale,
        Some(&incoming),
        3,
    );

    let incoming_nested: Vec<Vec<Vec<f32>>> = (0..heads)
        .map(|h| {
            (0..dim)
                .map(|a| {
                    let base = (h * dim + a) * dim;
                    h_flat[base..base + dim].to_vec()
                })
                .collect()
        })
        .collect();
    let (expected_y, expected_h) = reference_gla(
        &nested(&q_flat, heads, len, dim),
        &nested(&k_flat, heads, len, dim),
        &nested(&v_flat, heads, len, dim),
        &g,
        scale,
        Some(&incoming_nested),
    );

    for (i, (got, want)) in read_all(&out)
        .iter()
        .zip(expected_y.iter().flatten().flatten())
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-4,
            "output[{i}]: chunked {got} vs recurrence {want}"
        );
    }
    for (i, (got, want)) in read_all(&state)
        .iter()
        .zip(expected_h.iter().flatten().flatten())
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-4,
            "state[{i}]: chunked {got} vs recurrence {want}"
        );
    }
}

#[test]
fn the_decode_step_agrees_with_a_one_token_chunk() {
    // `gla_step` exists only to save ops on the hot path; it must be the same
    // arithmetic a single-token chunk performs.
    let (heads, dim) = (3usize, 4usize);
    let q_flat = noise(heads * dim, 0x1111_2222);
    let k_flat = noise(heads * dim, 0x3333_4444);
    let v_flat = noise(heads * dim, 0x5555_6666);
    let h_flat = noise(heads * dim * dim, 0x7777_8888);
    let g: Vec<f32> = vec![-0.1, -0.5, -1.5];
    let scale = 0.25;

    let shape = [1, heads as i32, 1, dim as i32];
    let state_shape = [1, heads as i32, dim as i32, dim as i32];
    let q = to_array(&q_flat, &shape);
    let k = to_array(&k_flat, &shape);
    let v = to_array(&v_flat, &shape);
    let state = to_array(&h_flat, &state_shape);
    let g_arr = to_array(&g, &[heads as i32]);

    let exp_g = mlxcel_core::exp(&mlxcel_core::reshape(&g_arr, &[1, heads as i32, 1, 1]));
    let (step_out, step_state) = gla_step(&q, &k, &v, &exp_g, scale, Some(&state));
    let (chunk_out, chunk_state) = gla_chunked(&q, &k, &v, &g_arr, scale, Some(&state), 64);

    for (i, (a, b)) in read_all(&step_out)
        .iter()
        .zip(read_all(&chunk_out).iter())
        .enumerate()
    {
        assert!((a - b).abs() < 1e-5, "output[{i}]: step {a} vs chunk {b}");
    }
    for (i, (a, b)) in read_all(&step_state)
        .iter()
        .zip(read_all(&chunk_state).iter())
        .enumerate()
    {
        assert!((a - b).abs() < 1e-5, "state[{i}]: step {a} vs chunk {b}");
    }
}

#[test]
fn the_sequential_prefill_is_the_recurrence_it_claims_to_be() {
    // `gla_sequential` is the default path and the one whose numbers must match
    // the reference, so it is pinned against the host-side transcription
    // directly rather than only through `gla_chunked`.
    let (heads, len, dim) = (3usize, 7usize, 4usize);
    let q_flat = noise(heads * len * dim, 0x0f0f_1111);
    let k_flat = noise(heads * len * dim, 0x0f0f_2222);
    let v_flat = noise(heads * len * dim, 0x0f0f_3333);
    let h_flat = noise(heads * dim * dim, 0x0f0f_4444);
    let g: Vec<f32> = vec![-0.15, -0.6, -1.4];
    let scale = 0.4;

    let shape = [1, heads as i32, len as i32, dim as i32];
    let state_shape = [1, heads as i32, dim as i32, dim as i32];
    let g_arr = to_array(&g, &[heads as i32]);
    let exp_g = mlxcel_core::exp(&mlxcel_core::reshape(&g_arr, &[1, heads as i32, 1, 1]));
    let incoming = to_array(&h_flat, &state_shape);

    for state in [None, Some(&incoming)] {
        let (out, carried) = gla_sequential(
            &to_array(&q_flat, &shape),
            &to_array(&k_flat, &shape),
            &to_array(&v_flat, &shape),
            &exp_g,
            scale,
            state.map(|s| s.as_ref().expect("array")),
        );
        assert_eq!(
            mlxcel_core::array_shape(&out),
            vec![1, heads as i32, len as i32, dim as i32],
            "the stacked output must restore the time axis in place"
        );

        let nested_state: Option<Vec<Vec<Vec<f32>>>> = state.map(|_| {
            (0..heads)
                .map(|h| {
                    (0..dim)
                        .map(|a| {
                            let base = (h * dim + a) * dim;
                            h_flat[base..base + dim].to_vec()
                        })
                        .collect()
                })
                .collect()
        });
        let (expected_y, expected_h) = reference_gla(
            &nested(&q_flat, heads, len, dim),
            &nested(&k_flat, heads, len, dim),
            &nested(&v_flat, heads, len, dim),
            &g,
            scale,
            nested_state.as_ref(),
        );
        for (i, (got, want)) in read_all(&out)
            .iter()
            .zip(expected_y.iter().flatten().flatten())
            .enumerate()
        {
            assert!((got - want).abs() < 1e-5, "output[{i}]: {got} vs {want}");
        }
        for (i, (got, want)) in read_all(&carried)
            .iter()
            .zip(expected_h.iter().flatten().flatten())
            .enumerate()
        {
            assert!((got - want).abs() < 1e-5, "state[{i}]: {got} vs {want}");
        }
    }
}

#[test]
fn the_first_token_of_a_fresh_state_is_not_decayed() {
    // Upstream skips the multiply on the first token (`h` is None), which is the
    // same as seeding with zeros only because 0 * anything is 0. A closed form
    // that applies `exp(g)^(t+1)` to a phantom initial state would shift every
    // output.
    let (heads, dim) = (1usize, 2usize);
    let q = to_array(&[1.0, 0.0], &[1, 1, 1, 2]);
    let k = to_array(&[1.0, 0.0], &[1, 1, 1, 2]);
    let v = to_array(&[3.0, 5.0], &[1, 1, 1, 2]);
    let g = to_array(&[-2.0], &[heads as i32]);

    let (out, _) = gla_chunked(&q, &k, &v, &g, 1.0, None, 8);
    let got = read_all(&out);
    // h = k^T v = [[3, 5], [0, 0]]; y = q h = [3, 5], with no decay applied.
    assert!((got[0] - 3.0).abs() < 1e-5, "{got:?}");
    assert!((got[1] - 5.0).abs() < 1e-5, "{got:?}");
    let _ = dim;
}

// GroupRMSNorm.

#[test]
fn group_rms_norm_normalizes_within_groups_not_across_them() {
    // Two groups of two, deliberately at very different magnitudes: a plain
    // RMSNorm over all four channels would leave the small group near zero,
    // while the grouped form brings both to unit RMS before the weight.
    let x = to_array(&[1.0, 1.0, 100.0, 100.0], &[1, 4]);
    let weight = to_array(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let norm = GroupRMSNorm::new(weight, 2, 1e-6).expect("groups divide the width");
    let got = read_all(&norm.forward(&x));
    for (i, value) in got.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < 1e-3,
            "channel {i} normalized to {value}, expected ~1.0"
        );
    }

    // The ungrouped form is a different function on the same input.
    let plain = GroupRMSNorm::new(to_array(&[1.0, 1.0, 1.0, 1.0], &[4]), 1, 1e-6)
        .expect("one group is valid");
    let plain_out = read_all(&plain.forward(&x));
    assert!(
        (plain_out[0] - 1.0).abs() > 0.9,
        "a plain RMSNorm must not reproduce the grouped result: {plain_out:?}"
    );
}

#[test]
fn group_rms_norm_applies_the_weight_after_flattening() {
    let x = to_array(&[1.0, 1.0, 2.0, 2.0], &[1, 4]);
    let weight = to_array(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let norm = GroupRMSNorm::new(weight, 2, 1e-6).expect("groups divide the width");
    let got = read_all(&norm.forward(&x));
    // Each group normalizes to [1, 1]; the full-width weight then scales.
    let expected = [1.0, 2.0, 3.0, 4.0];
    for (i, (got, want)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-3, "channel {i}: {got} vs {want}");
    }
}

#[test]
fn group_rms_norm_rejects_an_indivisible_width() {
    let err = match GroupRMSNorm::new(to_array(&[1.0; 6], &[6]), 4, 1e-6) {
        Ok(_) => panic!("6 channels do not split into 4 groups"),
        Err(err) => err,
    };
    assert!(err.contains("divisible"), "{err}");
}

// Config guards.

#[test]
fn a_group_norm_size_that_does_not_divide_the_width_is_rejected() {
    let mut args = ring_mini_args();
    args.group_norm_size = 3; // 2048 / 3 is not an integer
    let err = args.validate().expect_err("3 does not divide 2048");
    assert!(err.contains("group_norm_size"), "{err}");
}

#[test]
fn a_zero_layer_group_size_is_rejected() {
    let mut args = ring_mini_args();
    args.layer_group_size = 0;
    let err = args
        .validate()
        .expect_err("0 is the modulus and would divide by zero");
    assert!(err.contains("layer_group_size"), "{err}");
}

#[test]
fn the_dead_vendor_flags_are_rejected_rather_than_ignored() {
    let mut args = ring_mini_args();
    args.norm_softmax = true;
    let err = args.validate().expect_err("norm_softmax is dead upstream");
    assert!(err.contains("norm_softmax"), "{err}");

    let mut args = ring_mini_args();
    args.use_rmsnorm = false;
    let err = args.validate().expect_err("use_rmsnorm is dead upstream");
    assert!(err.contains("use_rmsnorm"), "{err}");
}

#[test]
fn a_rope_width_that_mlx_would_throw_on_is_rejected_on_either_head() {
    // MLX requires an even, positive `dims` no wider than the head; a violation
    // is an uncatchable std::terminate at the first forward pass rather than a
    // load error, so it has to be caught here.
    let mut args = ring_mini_args();
    args.partial_rotary_factor = 0.0;
    let err = args
        .validate()
        .expect_err("a zero rotary width is rejected");
    assert!(err.contains("rotary width"), "{err}");

    let mut args = ring_mini_args();
    args.partial_rotary_factor = 2.0;
    let err = args
        .validate()
        .expect_err("a rotary width wider than the head is rejected");
    assert!(err.contains("rotary width"), "{err}");

    let mut args = ring_mini_args();
    args.partial_rotary_factor = f32::NAN;
    let err = args
        .validate()
        .expect_err("a NaN partial_rotary_factor is rejected");
    assert!(err.contains("partial_rotary_factor"), "{err}");

    // An odd width on the LINEAR head only: head_dim 128 stays even at 0.5 while
    // linear_head_dim 6 gives 3. The global head is fine and the linear one is
    // not, which is exactly the case a single-width check would miss.
    let mut args = ring_mini_args();
    args.hidden_size = 96;
    args.num_attention_heads = 16;
    args.num_key_value_heads = Some(4);
    args.head_dim = Some(128);
    args.group_norm_size = 1;
    args.partial_rotary_factor = 0.5;
    let err = args
        .validate()
        .expect_err("an odd linear rotary width is rejected");
    assert!(err.contains("linear-attention"), "{err}");
    assert!(err.contains("odd"), "{err}");
}

#[test]
fn a_non_default_rope_scaling_block_is_rejected() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
            "intermediate_size": 8, "vocab_size": 16,
            "rope_scaling": {"rope_type": "linear", "factor": 4.0}}"#,
    )
    .expect("the config parses");
    let err = args
        .validate()
        .expect_err("a scaled rope block is rejected");
    assert!(err.contains("rope_scaling"), "{err}");

    let args: ModelArgs = serde_json::from_str(
        r#"{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
            "intermediate_size": 8, "vocab_size": 16,
            "rope_scaling": {"rope_type": "default"}}"#,
    )
    .expect("the config parses");
    args.validate().expect("a default block is a no-op");
}

#[test]
fn the_grouped_routing_guards_fire_on_a_config_this_family_actually_reaches() {
    // Unlike the dense Bailing family, Ring-mini declares n_group 8 with
    // topk_group 4, so these branches run on a real checkpoint.
    let mut args = ring_mini_args();
    args.topk_group = 8;
    let err = args
        .validate()
        .expect_err("topk_group == n_group puts argpartition out of range");
    assert!(err.contains("topk_group"), "{err}");

    let mut args = ring_mini_args();
    args.num_experts = Some(255); // not divisible by n_group 8
    let err = args
        .validate()
        .expect_err("an indivisible expert count cannot be regrouped");
    assert!(err.contains("n_group"), "{err}");

    let mut args = ring_mini_args();
    args.n_group = 256; // one expert per group
    args.topk_group = 4;
    let err = args
        .validate()
        .expect_err("a group of one cannot be scored by its top two");
    assert!(err.contains("per group"), "{err}");
}

#[test]
fn an_indivisible_hidden_size_is_rejected_even_with_an_explicit_head_dim() {
    // The linear layers derive their head width from the ratio regardless, so an
    // explicit head_dim does not rescue an indivisible pair.
    let mut args = ring_mini_args();
    args.hidden_size = 2049;
    args.head_dim = Some(128);
    args.group_norm_size = 1;
    let err = args
        .validate()
        .expect_err("2049 is not divisible by 16 heads");
    assert!(err.contains("divisible by num_attention_heads"), "{err}");
}

#[test]
fn a_zero_scalar_is_rejected_before_it_divides() {
    for mutate in [
        (|a: &mut ModelArgs| a.num_attention_heads = 0) as fn(&mut ModelArgs),
        |a: &mut ModelArgs| a.hidden_size = 0,
        |a: &mut ModelArgs| a.num_hidden_layers = 0,
        |a: &mut ModelArgs| a.intermediate_size = 0,
        |a: &mut ModelArgs| a.vocab_size = 0,
        |a: &mut ModelArgs| a.max_position_embeddings = 0,
        |a: &mut ModelArgs| a.rms_norm_eps = 0.0,
        |a: &mut ModelArgs| a.rope_theta = 0.0,
    ] {
        let mut args = ring_mini_args();
        mutate(&mut args);
        assert!(
            args.validate().is_err(),
            "a zero scalar must be rejected at load"
        );
    }
}

// Weight-shape validation.

fn lazy(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(shape, 0.0, mlxcel_core::dtype::FLOAT32)
}

/// A weight map matching the real export's names and shapes, unquantized.
///
/// Built from lazy MLX arrays that are never evaluated, so nothing is allocated
/// on device. Shrunk from the real checkpoint (4 layers, 4 experts) so the map
/// stays readable; the layout is identical.
fn synthetic_weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let mut weights = WeightMap::new();
    weights.insert(
        "model.word_embeddings.weight".into(),
        lazy(&[args.vocab_size as i32, hidden]),
    );
    weights.insert("model.norm.weight".into(), lazy(&[hidden]));
    weights.insert(
        "lm_head.weight".into(),
        lazy(&[args.vocab_size as i32, hidden]),
    );

    for layer in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        let attention = format!("{prefix}.attention");
        let is_global = args.is_global_layer(layer);
        let (qkv, head_dim, width) = if is_global {
            (
                args.attention_qkv_out_features() as i32,
                args.head_dim() as i32,
                (args.num_attention_heads * args.head_dim()) as i32,
            )
        } else {
            (
                args.linear_qkv_out_features() as i32,
                args.linear_head_dim() as i32,
                (args.num_attention_heads * args.linear_head_dim()) as i32,
            )
        };
        weights.insert(
            format!("{attention}.query_key_value.weight"),
            lazy(&[qkv, hidden]),
        );
        weights.insert(format!("{attention}.dense.weight"), lazy(&[hidden, width]));
        weights.insert(
            format!("{attention}.query_layernorm.weight"),
            lazy(&[head_dim]),
        );
        weights.insert(
            format!("{attention}.key_layernorm.weight"),
            lazy(&[head_dim]),
        );
        if !is_global {
            weights.insert(format!("{attention}.g_proj.weight"), lazy(&[width, hidden]));
            weights.insert(format!("{attention}.g_norm.weight"), lazy(&[width]));
        }
        weights.insert(format!("{prefix}.input_layernorm.weight"), lazy(&[hidden]));
        weights.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            lazy(&[hidden]),
        );

        let mlp = format!("{prefix}.mlp");
        if args.is_moe_layer(layer) {
            let experts = args.num_experts() as i32;
            let moe = args.moe_intermediate_size() as i32;
            weights.insert(
                format!("{mlp}.gate.gate_proj.weight"),
                lazy(&[experts, hidden]),
            );
            weights.insert(format!("{mlp}.gate.expert_bias"), lazy(&[experts]));
            weights.insert(
                format!("{mlp}.switch_mlp.gate_proj.weight"),
                lazy(&[experts, moe, hidden]),
            );
            weights.insert(
                format!("{mlp}.switch_mlp.up_proj.weight"),
                lazy(&[experts, moe, hidden]),
            );
            weights.insert(
                format!("{mlp}.switch_mlp.down_proj.weight"),
                lazy(&[experts, hidden, moe]),
            );
            let shared = args.shared_expert_intermediate_size() as i32;
            weights.insert(
                format!("{mlp}.shared_experts.gate_proj.weight"),
                lazy(&[shared, hidden]),
            );
            weights.insert(
                format!("{mlp}.shared_experts.up_proj.weight"),
                lazy(&[shared, hidden]),
            );
            weights.insert(
                format!("{mlp}.shared_experts.down_proj.weight"),
                lazy(&[hidden, shared]),
            );
        } else {
            let inter = args.intermediate_size as i32;
            weights.insert(format!("{mlp}.gate_proj.weight"), lazy(&[inter, hidden]));
            weights.insert(format!("{mlp}.up_proj.weight"), lazy(&[inter, hidden]));
            weights.insert(format!("{mlp}.down_proj.weight"), lazy(&[hidden, inter]));
        }
    }
    weights
}

/// A shrunk Ring-mini: 6 layers in groups of 3 (so layers 2 and 5 are global),
/// 4 experts, small dims.
fn small_args() -> ModelArgs {
    let mut args = ring_mini_args();
    args.hidden_size = 32;
    args.num_hidden_layers = 6;
    args.num_attention_heads = 4;
    args.num_key_value_heads = Some(2);
    args.head_dim = Some(8);
    args.layer_group_size = 3;
    args.group_norm_size = 4;
    args.intermediate_size = 16;
    args.moe_intermediate_size = Some(8);
    args.moe_shared_expert_intermediate_size = Some(8);
    args.num_experts = Some(4);
    args.num_experts_per_tok = 2;
    args.n_group = 2;
    args.topk_group = 1;
    args.vocab_size = 64;
    args.quantization = Some(Quantization {
        group_size: 64,
        bits: 4,
    });
    args
}

#[test]
fn a_well_formed_checkpoint_passes_validation() {
    let args = small_args();
    args.validate().expect("the shrunk config is valid");
    let weights = synthetic_weights(&args);
    validate_weights(&weights, &args).expect("the synthetic export validates");
}

#[test]
fn a_linear_layer_carrying_a_global_layers_qkv_width_is_rejected() {
    // This is the failure a shared validator could not see: 3 * H * d against
    // (H + 2 * H_kv) * d, both plausible 2-D projections into the same hidden
    // size. MLX's slice clamps an out-of-range stop rather than throwing, so the
    // model would load and read the wrong channels.
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    assert!(!args.is_global_layer(0));
    weights.insert(
        "model.layers.0.attention.query_key_value.weight".into(),
        lazy(&[
            args.attention_qkv_out_features() as i32,
            args.hidden_size as i32,
        ]),
    );
    let err = validate_weights(&weights, &args).expect_err("the narrow QKV is rejected");
    assert!(err.contains("query_key_value"), "{err}");
    assert!(
        err.contains(&args.linear_qkv_out_features().to_string()),
        "the message must name the width the linear layer needs: {err}"
    );
}

#[test]
fn a_missing_linear_gate_is_rejected() {
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    weights.remove("model.layers.0.attention.g_proj.weight");
    let err = validate_weights(&weights, &args).expect_err("g_proj is required on a linear layer");
    assert!(err.contains("g_proj"), "{err}");

    let mut weights = synthetic_weights(&args);
    weights.remove("model.layers.0.attention.g_norm.weight");
    let err = validate_weights(&weights, &args).expect_err("g_norm is required on a linear layer");
    assert!(err.contains("g_norm"), "{err}");
}

#[test]
fn a_short_expert_stack_is_rejected() {
    // The router emits indices below num_experts and the gather behind
    // gather_qmm does not range-check a positive index.
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    weights.insert(
        "model.layers.1.mlp.switch_mlp.gate_proj.weight".into(),
        lazy(&[
            args.num_experts() as i32 - 1,
            args.moe_intermediate_size() as i32,
            args.hidden_size as i32,
        ]),
    );
    let err = validate_weights(&weights, &args).expect_err("a short stack is rejected");
    assert!(err.contains("num_experts"), "{err}");
}

#[test]
fn a_dense_prefix_layer_is_validated_as_a_plain_mlp() {
    // first_k_dense_replace is 1 on Ring-mini, so layer 0 has no router at all
    // and must not be probed for one.
    let args = small_args();
    let weights = synthetic_weights(&args);
    assert!(!args.is_moe_layer(0));
    assert!(weights.contains_key("model.layers.0.mlp.gate_proj.weight"));
    assert!(!weights.contains_key("model.layers.0.mlp.gate.gate_proj.weight"));
    validate_weights(&weights, &args).expect("the dense prefix validates");
}

// End-to-end construction and a forward pass.

#[test]
fn a_synthetic_model_builds_and_produces_finite_logits() {
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    // Replace the lazy zeros with real values: an all-zero stack produces
    // technically-finite logits from a graph that never exercises the decay,
    // the router or the gate.
    let keys: Vec<String> = weights.keys().cloned().collect();
    let mut seed = 0x5EED_1234u32;
    for key in keys {
        let shape = mlxcel_core::array_shape(weights.get(&key).expect("key just listed"));
        let n: i32 = shape.iter().product();
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        weights.insert(
            key,
            mlxcel_core::from_slice_f32(&noise(n as usize, seed), &shape),
        );
    }

    let model = BailingMoeLinearModel::from_weights(&weights, &args).expect("the model builds");
    assert_eq!(model.num_layers(), args.num_hidden_layers);

    let tokens = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7], &[1, 7]);
    let mut caches = LanguageModel::make_caches(&model);
    let logits = LanguageModel::forward(&model, &tokens, &mut caches, None);
    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 7);
    assert_eq!(shape[2], args.vocab_size as i32);
    for (i, value) in read_all(&mlxcel_core::slice(&logits, &[0, 6, 0], &[1, 7, 16]))
        .iter()
        .enumerate()
    {
        assert!(value.is_finite(), "logit[{i}] is {value}");
    }

    // A decode step after the prefill must reuse the carried state rather than
    // start over: the second call sees only one token.
    let next = mlxcel_core::from_slice_i32(&[8], &[1, 1]);
    let step = LanguageModel::forward(&model, &next, &mut caches, None);
    let step_shape = mlxcel_core::array_shape(&step);
    assert_eq!(step_shape, vec![1, 1, args.vocab_size as i32]);
    for (i, value) in read_all(&mlxcel_core::slice(&step, &[0, 0, 0], &[1, 1, 16]))
        .iter()
        .enumerate()
    {
        assert!(value.is_finite(), "decode logit[{i}] is {value}");
    }
}

#[test]
fn the_cache_flavour_follows_the_layer_schedule() {
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    let keys: Vec<String> = weights.keys().cloned().collect();
    let mut seed = 0x1357_9BDFu32;
    for key in keys {
        let shape = mlxcel_core::array_shape(weights.get(&key).expect("key just listed"));
        let n: i32 = shape.iter().product();
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        weights.insert(
            key,
            mlxcel_core::from_slice_f32(&noise(n as usize, seed), &shape),
        );
    }
    let model = BailingMoeLinearModel::from_weights(&weights, &args).expect("the model builds");

    let caches = model.make_internal_caches();
    assert_eq!(caches.len(), args.num_hidden_layers);
    for (idx, cache) in caches.iter().enumerate() {
        let is_attention = matches!(cache, BailingLinearCache::Attention(_));
        assert_eq!(
            is_attention,
            args.is_global_layer(idx),
            "layer {idx} got the wrong cache flavour"
        );
    }
}

#[test]
fn a_fresh_linear_cache_holds_no_state() {
    let cache = LinearAttentionCache::new();
    assert!(cache.state.is_none());
    assert_eq!(cache.offset, 0);
}

#[test]
fn chunked_prefill_is_on_unless_explicitly_switched_off() {
    // #1040 promoted the chunked closed form to the default on measured
    // perplexity, so an unset variable must select it. This is the assertion
    // that would catch the default silently flipping back.
    assert!(chunked_prefill_enabled_from(None));

    // The variable inverted rather than being renamed, so a pre-#1040 `=1`
    // still means chunked instead of becoming a surprise opt-out.
    for on in ["1", "true", "TRUE", "yes", "on", "anything else"] {
        assert!(
            chunked_prefill_enabled_from(Some(on)),
            "{on:?} should keep the chunked default"
        );
    }

    // Only the documented off-spellings restore upstream's sequential
    // recurrence, which is what a mlx-lm reference diff needs.
    for off in ["0", "false", "FALSE", "off", "no", "  no  "] {
        assert!(
            !chunked_prefill_enabled_from(Some(off)),
            "{off:?} should select the sequential recurrence"
        );
    }
}
