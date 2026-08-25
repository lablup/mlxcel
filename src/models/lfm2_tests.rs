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

//! Unit tests for LFM2 / LFM2-MoE config parsing and the config-derived layer
//! typing and routing wiring.
//!
//! These cover the checkpoint-free surface: that the dense `lfm2` and sparse
//! `lfm2_moe` `config.json` files deserialize into `ModelArgs`, that the
//! full-attention layer indices derive correctly (explicit list for `lfm2`,
//! `layer_types` for `lfm2_moe`), that dense-vs-MoE feed-forward selection
//! follows `num_dense_layers`, and that the sigmoid-gating fields parse. The
//! numeric correctness of the short-conv and the sigmoid-gated routing is
//! validated end-to-end against real `mlx-community` checkpoints
//! (`LFM2-350M-8bit`, `LFM2-8B-A1B-4bit`) and is not exercised here (no Metal
//! device is assumed in unit tests).
//!
//! The exception is the short-convolution padding, which the LFM2.5-Embedding
//! port (#1325) made directional: `ModelArgs::conv_causal` chooses between the
//! generator's left pad and the embedder's split pad, and the difference is one
//! index of shift that sixteen layers of mixing would hide. The three tests at
//! the end drive `ShortConv` directly with a one-hot impulse instead.
//!
//! Every test here that evaluates an MLX graph takes the shared
//! `mlx_test_guard`. `cargo test` runs one thread per logical CPU, and two
//! concurrent MLX forward passes in one process either abort inside CUDA graph
//! capture or drift numerically; see the guard's own doc comment.

use super::lfm2::{ModelArgs, ShortConv};
use crate::models::embedding_test_support::mlx_test_guard;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// A trimmed `LFM2-350M` (dense) config with the fields the loader reads.
const LFM2_350M_CONFIG: &str = r#"{
    "model_type": "lfm2",
    "vocab_size": 65536,
    "hidden_size": 1024,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "max_position_embeddings": 128000,
    "norm_eps": 1e-05,
    "conv_bias": false,
    "conv_L_cache": 3,
    "rope_theta": 1000000.0,
    "full_attn_idxs": [2, 5, 8, 10, 12, 14],
    "eos_token_id": 7,
    "block_dim": 1024,
    "block_ff_dim": 6656,
    "quantization": { "group_size": 64, "bits": 8 }
}"#;

/// A trimmed `LFM2-8B-A1B` (MoE) config. `layer_types` drives the attention
/// layer derivation; the MoE block fields drive sigmoid-gated routing.
const LFM2_8B_MOE_CONFIG: &str = r#"{
    "model_type": "lfm2_moe",
    "vocab_size": 65536,
    "hidden_size": 2048,
    "intermediate_size": 7168,
    "moe_intermediate_size": 1792,
    "num_hidden_layers": 24,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 128000,
    "norm_eps": 1e-05,
    "conv_bias": false,
    "conv_L_cache": 3,
    "rope_theta": 1000000.0,
    "num_dense_layers": 2,
    "num_experts": 32,
    "num_experts_per_tok": 4,
    "norm_topk_prob": true,
    "use_expert_bias": true,
    "routed_scaling_factor": 1.0,
    "eos_token_id": 7,
    "layer_types": [
        "conv", "conv", "full_attention", "conv", "conv", "conv",
        "full_attention", "conv", "conv", "conv", "full_attention", "conv",
        "conv", "conv", "full_attention", "conv", "conv", "conv",
        "full_attention", "conv", "conv", "full_attention", "conv", "conv"
    ],
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

#[test]
fn lfm2_dense_config_parses() {
    let args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    assert_eq!(args.model_type, "lfm2");
    assert_eq!(args.hidden_size, 1024);
    assert_eq!(args.num_attention_heads, 16);
    assert_eq!(args.num_key_value_heads, 8);
    assert_eq!(args.num_hidden_layers, 16);
    assert_eq!(args.conv_l_cache, 3);
    assert!(!args.conv_bias);
    assert_eq!(args.rope_theta, 1_000_000.0);
    // head_dim derives from hidden/heads; the q/k layernorm weights are [64].
    assert_eq!(args.head_dim(), 64);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 8);
    assert_eq!(args.eos_token_ids(), vec![7]);
    // No MoE fields → a pure dense checkpoint.
    assert!(!args.is_moe());
}

#[test]
fn lfm2_dense_full_attn_idxs_are_explicit() {
    let args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    assert_eq!(args.full_attn_idxs(), vec![2, 5, 8, 10, 12, 14]);
    for idx in 0..args.num_hidden_layers {
        let expect_attention = [2, 5, 8, 10, 12, 14].contains(&idx);
        assert_eq!(
            args.is_attention_layer(idx),
            expect_attention,
            "layer {idx}"
        );
    }
}

#[test]
fn lfm2_dense_every_layer_is_dense_ffn() {
    let args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    // The dense checkpoint has no experts, so no layer routes to MoE and the
    // dense-layer boundary defaults to the full layer count.
    assert_eq!(args.num_dense_layers(), args.num_hidden_layers);
    for idx in 0..args.num_hidden_layers {
        assert!(!args.layer_is_moe(idx), "layer {idx} must use a dense MLP");
    }
}

#[test]
fn lfm2_moe_config_parses_sigmoid_gating_fields() {
    let args: ModelArgs = serde_json::from_str(LFM2_8B_MOE_CONFIG).expect("parse MoE config");
    assert_eq!(args.model_type, "lfm2_moe");
    assert_eq!(args.hidden_size, 2048);
    assert_eq!(args.num_attention_heads, 32);
    assert_eq!(args.head_dim(), 64);
    assert_eq!(args.bits(), 4);
    assert!(args.is_moe());
    assert_eq!(args.num_experts, Some(32));
    assert_eq!(args.num_experts_per_tok, Some(4));
    assert_eq!(args.moe_intermediate_size, Some(1792));
    assert_eq!(args.num_dense_layers, Some(2));
    // The load-bearing sigmoid-gating switches.
    assert_eq!(args.norm_topk_prob, Some(true));
    assert_eq!(args.use_expert_bias, Some(true));
    assert_eq!(args.routed_scaling_factor, 1.0);
    assert_eq!(args.eos_token_ids(), vec![7]);
}

#[test]
fn lfm2_moe_full_attn_idxs_derive_from_layer_types() {
    let args: ModelArgs = serde_json::from_str(LFM2_8B_MOE_CONFIG).expect("parse MoE config");
    // `full_attn_idxs` is absent, so the indices come from layer_types entries
    // equal to "full_attention".
    assert!(args.full_attn_idxs.is_none());
    assert_eq!(args.full_attn_idxs(), vec![2, 6, 10, 14, 18, 21]);
    assert!(args.is_attention_layer(2));
    assert!(args.is_attention_layer(21));
    assert!(!args.is_attention_layer(0));
    assert!(!args.is_attention_layer(23));
}

#[test]
fn lfm2_moe_feed_forward_selection_follows_num_dense_layers() {
    let args: ModelArgs = serde_json::from_str(LFM2_8B_MOE_CONFIG).expect("parse MoE config");
    assert_eq!(args.num_dense_layers(), 2);
    // Layers 0 and 1 use a dense MLP; every later layer routes to the sparse
    // MoE block, including the attention layer at index 2.
    assert!(!args.layer_is_moe(0));
    assert!(!args.layer_is_moe(1));
    assert!(args.layer_is_moe(2));
    assert!(args.is_attention_layer(2) && args.layer_is_moe(2));
    assert!(args.layer_is_moe(23));
}

#[test]
fn lfm2_conv_state_keep_length_invariant() {
    // The per-layer conv cache holds the last `L_cache - 1` time steps of `Bx`
    // (shape `[batch, L_cache - 1, hidden]`), which is what the depthwise
    // kernel-size-`L_cache` causal conv needs prepended on decode.
    let args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    assert_eq!(args.conv_l_cache, 3);
    assert_eq!(args.conv_l_cache - 1, 2);
}

#[test]
fn lfm2_eos_token_id_handles_scalar_array_and_missing() {
    // Scalar (the shape both shipped checkpoints use).
    let scalar: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    assert_eq!(scalar.eos_token_ids(), vec![7]);

    // Array form.
    let array_cfg = LFM2_350M_CONFIG.replace("\"eos_token_id\": 7", "\"eos_token_id\": [7, 1]");
    let array: ModelArgs = serde_json::from_str(&array_cfg).expect("parse array eos config");
    assert_eq!(array.eos_token_ids(), vec![7, 1]);

    // Missing → default `<|im_end|>` (id 7).
    let no_eos = LFM2_350M_CONFIG.replace("\"eos_token_id\": 7,", "");
    let missing: ModelArgs = serde_json::from_str(&no_eos).expect("parse no-eos config");
    assert_eq!(missing.eos_token_ids(), vec![7]);
}

#[test]
fn lfm2_short_conv_decode_matches_conv1d() {
    let _guard = mlx_test_guard();
    // Guard the decode fast path (issue #748): the single-step depthwise short
    // conv, computed as an explicit weighted sum of the L_cache taps, must be
    // numerically identical to the stride-1/no-pad/depthwise `conv1d` it
    // replaces. This is the checkpoint-free core of the regression fix; the
    // CUDA kernel-dispatch win is measured end-to-end against real checkpoints.
    use super::lfm2::{build_conv_decode_weight, short_conv_decode_step};

    let hidden = 4;
    let l_cache = 3;

    // conv_weight is [hidden, L_cache, 1] (MLX depthwise layout): for each
    // channel c, the three causal taps weight[c, 0..3, 0].
    let weight_data: Vec<f32> = vec![
        0.5, -0.25, 0.75, // channel 0
        -1.0, 0.5, 0.25, // channel 1
        0.1, 0.2, -0.3, // channel 2
        2.0, -0.5, 1.5, // channel 3
    ];
    let conv_weight = mlxcel_core::from_slice_f32(&weight_data, &[hidden, l_cache, 1]);

    // padded is [1, L_cache, hidden]: the cached conv-state tail prepended to
    // the current Bx step, one row per time index.
    let padded_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // t = 0
        -1.0, 0.5, 2.0, -2.0, // t = 1
        0.25, -0.75, 1.0, 0.5, // t = 2
    ];
    let padded = mlxcel_core::from_slice_f32(&padded_data, &[1, l_cache, hidden]);

    // Ground truth: stride-1, no-pad, dilation-1, groups==hidden conv1d.
    let reference = mlxcel_core::conv1d(&padded, &conv_weight, 1, 0, 1, hidden);
    assert_eq!(mlxcel_core::array_shape(&reference), vec![1, 1, hidden]);

    // Fast path: broadcast weighted sum over the precomputed time-major weight.
    let decode_weight = build_conv_decode_weight(&conv_weight);
    assert_eq!(
        mlxcel_core::array_shape(&decode_weight),
        vec![1, l_cache, hidden]
    );
    let elementwise = short_conv_decode_step(&padded, &decode_weight, mlxcel_core::dtype::FLOAT32);
    assert_eq!(mlxcel_core::array_shape(&elementwise), vec![1, 1, hidden]);

    let diff = mlxcel_core::subtract(&reference, &elementwise);
    let max_abs = mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(&diff)));
    assert!(
        max_abs < 1e-5,
        "decode short-conv diverged from conv1d: max|diff| = {max_abs}"
    );
}

#[test]
fn lfm2_short_conv_decode_matches_conv1d_bf16() {
    let _guard = mlx_test_guard();
    // bf16 variant of `lfm2_short_conv_decode_matches_conv1d`: this is the
    // dtype the decode fast path actually runs in on real (bf16) LFM2
    // checkpoints, and is defense-in-depth against a future MLX change to
    // how `sum_axis` accumulates for half dtypes. Values are built in f32
    // (same asymmetric kernel as the f32 test) and cast to bf16 for both
    // the `conv1d` reference and the fast path, so the comparison isolates
    // dtype-driven rounding rather than construction differences. bf16 has
    // ~3 decimal digits of precision, so the tolerance is loose relative to
    // the f32 test.
    use super::lfm2::{build_conv_decode_weight, short_conv_decode_step};
    use mlxcel_core::dtype;

    let hidden = 4;
    let l_cache = 3;

    let weight_data: Vec<f32> = vec![
        0.5, -0.25, 0.75, // channel 0
        -1.0, 0.5, 0.25, // channel 1
        0.1, 0.2, -0.3, // channel 2
        2.0, -0.5, 1.5, // channel 3
    ];
    let conv_weight_f32 = mlxcel_core::from_slice_f32(&weight_data, &[hidden, l_cache, 1]);
    let conv_weight = mlxcel_core::astype(&conv_weight_f32, dtype::BFLOAT16);

    let padded_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // t = 0
        -1.0, 0.5, 2.0, -2.0, // t = 1
        0.25, -0.75, 1.0, 0.5, // t = 2
    ];
    let padded_f32 = mlxcel_core::from_slice_f32(&padded_data, &[1, l_cache, hidden]);
    let padded = mlxcel_core::astype(&padded_f32, dtype::BFLOAT16);

    // Ground truth: stride-1, no-pad, dilation-1, groups==hidden conv1d, run
    // in bf16 (as it is on a bf16 checkpoint off Metal before this fix).
    let reference = mlxcel_core::conv1d(&padded, &conv_weight, 1, 0, 1, hidden);
    assert_eq!(mlxcel_core::array_shape(&reference), vec![1, 1, hidden]);
    assert_eq!(mlxcel_core::array_dtype(&reference), dtype::BFLOAT16);

    // Fast path, built from the bf16 conv weight exactly as `ShortConv`
    // does for a bf16 checkpoint's non-quantized conv weight.
    let decode_weight = build_conv_decode_weight(&conv_weight);
    assert_eq!(
        mlxcel_core::array_shape(&decode_weight),
        vec![1, l_cache, hidden]
    );
    let elementwise = short_conv_decode_step(&padded, &decode_weight, dtype::BFLOAT16);
    assert_eq!(mlxcel_core::array_shape(&elementwise), vec![1, 1, hidden]);

    // Compare in f32 (bf16 subtraction/abs would itself be lossy).
    let diff = mlxcel_core::subtract(
        &mlxcel_core::astype(&reference, dtype::FLOAT32),
        &mlxcel_core::astype(&elementwise, dtype::FLOAT32),
    );
    let max_abs = mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(&diff)));
    assert!(
        max_abs < 2e-2,
        "bf16 decode short-conv diverged from bf16 conv1d: max|diff| = {max_abs}"
    );
}

#[test]
fn lfm2_unquantized_config_uses_quantization_defaults() {
    // Drop the quantization block (a bf16 checkpoint) and confirm the defaults.
    let no_quant = LFM2_350M_CONFIG.replace(
        ",\n    \"quantization\": { \"group_size\": 64, \"bits\": 8 }",
        "",
    );
    let args: ModelArgs = serde_json::from_str(&no_quant).expect("parse unquantized config");
    assert!(args.quantization.is_none());
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

// Directional short convolution (#1325).

/// Channel count of the impulse fixture: channel 0 carries the impulse,
/// channel 1 is a constant 1 that lets the `C` gate and the `x` factor stay 1
/// everywhere, so the output is the raw convolution of the impulse.
const IMPULSE_HIDDEN: i32 = 2;
/// Taps, chosen distinct so the position each one lands at is unambiguous.
const IMPULSE_TAPS: [f32; 3] = [1.0, 2.0, 3.0];
/// Sequence length and the impulse position inside it.
const IMPULSE_LEN: i32 = 12;
const IMPULSE_AT: usize = 5;

fn impulse_args(conv_causal: bool) -> ModelArgs {
    let mut args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    args.hidden_size = IMPULSE_HIDDEN as usize;
    args.quantization = None;
    args.conv_causal = conv_causal;
    args
}

/// `in_proj` maps `[impulse, ones]` to `B = impulse` (both channels),
/// `C = 1` and `x = 1`, so `Bx = impulse` and `y = conv(impulse)`;
/// `out_proj` is the identity.
fn impulse_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    #[rustfmt::skip]
    let in_proj = [
        1.0, 0.0, // B channel 0 <- impulse
        1.0, 0.0, // B channel 1 <- impulse
        0.0, 1.0, // C channel 0 <- ones
        0.0, 1.0, // C channel 1 <- ones
        0.0, 1.0, // x channel 0 <- ones
        0.0, 1.0, // x channel 1 <- ones
    ];
    weights.insert(
        "conv.in_proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&in_proj, &[3 * IMPULSE_HIDDEN, IMPULSE_HIDDEN]),
    );
    weights.insert(
        "conv.out_proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0, 0.0, 0.0, 1.0], &[IMPULSE_HIDDEN, IMPULSE_HIDDEN]),
    );
    let taps: Vec<f32> = IMPULSE_TAPS
        .iter()
        .chain(IMPULSE_TAPS.iter())
        .copied()
        .collect();
    weights.insert(
        "conv.conv.weight".to_string(),
        mlxcel_core::from_slice_f32(&taps, &[IMPULSE_HIDDEN, IMPULSE_TAPS.len() as i32, 1]),
    );
    weights
}

/// `[1, IMPULSE_LEN, 2]`: a unit impulse on channel 0 at [`IMPULSE_AT`], ones
/// on channel 1.
fn impulse_input() -> UniquePtr<MlxArray> {
    let mut values = vec![0.0_f32; (IMPULSE_LEN * IMPULSE_HIDDEN) as usize];
    for t in 0..IMPULSE_LEN as usize {
        values[t * 2 + 1] = 1.0;
    }
    values[IMPULSE_AT * 2] = 1.0;
    mlxcel_core::from_slice_f32(&values, &[1, IMPULSE_LEN, IMPULSE_HIDDEN])
}

/// Channel-0 response of one `ShortConv` to [`impulse_input`], plus whether the
/// conv state was written.
fn impulse_response(conv_causal: bool) -> (Vec<f32>, bool) {
    let args = impulse_args(conv_causal);
    let conv = ShortConv::from_weights(&impulse_weights(), &args, "conv").expect("short conv");
    let mut state: Option<UniquePtr<MlxArray>> = None;
    let out = conv.forward(&impulse_input(), &mut state, None);
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![1, IMPULSE_LEN, IMPULSE_HIDDEN],
        "the short conv must preserve the sequence length"
    );
    let flat = mlxcel_core::utils::array_to_vec_f32(&out);
    let channel0 = (0..IMPULSE_LEN as usize).map(|t| flat[t * 2]).collect();
    (channel0, state.is_some())
}

#[test]
fn conv_causal_defaults_true_in_args() {
    // No published config declares the key, so the generator must keep its
    // causal mixer purely by the serde default.
    let args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    assert!(args.conv_causal, "conv_causal must default to true");
    let moe: ModelArgs = serde_json::from_str(LFM2_8B_MOE_CONFIG).expect("parse moe config");
    assert!(moe.conv_causal, "conv_causal must default to true for MoE");
}

#[test]
fn causal_short_conv_unchanged_when_flag_true() {
    let _guard = mlx_test_guard();
    // The causal branch is the pre-#1325 code: all L_cache - 1 zeros on the
    // left, and the conv state written from the padded tail. An impulse at t
    // therefore reaches only t, t + 1 and t + 2, never any earlier position.
    let args = impulse_args(true);
    let conv = ShortConv::from_weights(&impulse_weights(), &args, "conv").expect("short conv");
    assert_eq!(conv.conv_padding(), (IMPULSE_TAPS.len() as i32 - 1, 0));

    let (response, wrote_state) = impulse_response(true);
    assert!(wrote_state, "the causal mixer must persist its conv state");
    // out[t] = bx[t - 2] * w0 + bx[t - 1] * w1 + bx[t] * w2.
    let mut expected = vec![0.0_f32; IMPULSE_LEN as usize];
    expected[IMPULSE_AT] = IMPULSE_TAPS[2];
    expected[IMPULSE_AT + 1] = IMPULSE_TAPS[1];
    expected[IMPULSE_AT + 2] = IMPULSE_TAPS[0];
    for (t, (got, want)) in response.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "causal response at {t}: got {got}, want {want}; full response {response:?}"
        );
    }
}

#[test]
fn noncausal_short_conv_symmetric_pad_keeps_length() {
    let _guard = mlx_test_guard();
    // With conv_causal = false the same L_cache - 1 zeros are split evenly, so
    // the output length is unchanged and the impulse at t reaches t - 1, t and
    // t + 1: one position of look-ahead, which is what makes the mixer
    // bidirectional.
    let args = impulse_args(false);
    let conv = ShortConv::from_weights(&impulse_weights(), &args, "conv").expect("short conv");
    assert_eq!(conv.conv_padding(), (1, 1));

    let (response, wrote_state) = impulse_response(false);
    assert!(
        !wrote_state,
        "the bidirectional mixer never decodes and must leave the conv state unset"
    );
    let mut expected = vec![0.0_f32; IMPULSE_LEN as usize];
    expected[IMPULSE_AT - 1] = IMPULSE_TAPS[2];
    expected[IMPULSE_AT] = IMPULSE_TAPS[1];
    expected[IMPULSE_AT + 1] = IMPULSE_TAPS[0];
    for (t, (got, want)) in response.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "bidirectional response at {t}: got {got}, want {want}; full response {response:?}"
        );
    }

    // The two modes differ, and they differ exactly by the one-position shift
    // above rather than by a scale or a dropped tap.
    let (causal, _) = impulse_response(true);
    assert!(
        causal[IMPULSE_AT - 1] == 0.0 && response[IMPULSE_AT - 1] != 0.0,
        "only the bidirectional mixer may write behind the impulse"
    );
}
