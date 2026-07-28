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

//! Unit tests for the GPT-NeoX loader, its interleaved per-head QKV split, its
//! partial RoPE dimensioning and both of its residual layouts.
//!
//! Everything here is checkpoint-free: the config tests parse the real
//! `EleutherAI/pythia-1b` `config.json` field set, and the layout tests build
//! synthetic weight maps whose tensor names and shapes mirror the HuggingFace
//! export. The forward tests build tiny models and run them on the default
//! device.
//!
//! The tests that matter most are the interleaved-QKV ones. A flat three-way
//! split of the fused projection produces tensors of exactly the right shape, so
//! nothing about the model's shapes, its cache, or its logits distinguishes the
//! two; only the channel *values* do. `interleaved_qkv_split_pins_the_head_major_layout`
//! and `attention_forward_returns_the_interleaved_v_block` therefore assert
//! concrete numbers and additionally assert what the wrong split would have
//! produced, so the mistake cannot be reintroduced silently.

use super::{
    Attention, GptNeoxLayout, GptNeoxModel, ModelArgs, TokenIdField, load_linear,
    split_interleaved_qkv, strip_registered_buffers,
};
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// Config surface.

/// The `EleutherAI/pythia-1b` config, field-for-field.
const PYTHIA_1B_CONFIG: &str = r#"{
    "architectures": ["GPTNeoXForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 0,
    "hidden_act": "gelu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 2048,
    "model_type": "gpt_neox",
    "num_attention_heads": 8,
    "num_hidden_layers": 16,
    "rotary_emb_base": 10000,
    "rotary_pct": 0.25,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.24.0",
    "use_cache": true,
    "use_parallel_residual": true,
    "vocab_size": 50304
}"#;

#[test]
fn parses_the_real_pythia_1b_config() {
    let args: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("pythia-1b config parses");

    assert_eq!(args.model_type, "gpt_neox");
    assert_eq!(args.hidden_size, 2048);
    assert_eq!(args.num_attention_heads, 8);
    assert_eq!(args.num_hidden_layers, 16);
    assert_eq!(args.intermediate_size, Some(8192));
    assert_eq!(args.max_position_embeddings, 2048);
    assert_eq!(args.vocab_size, 50304);
    assert!(args.use_parallel_residual);
    assert!(!args.tie_word_embeddings);
    assert!((args.rotary_pct - 0.25).abs() < 1e-9);
    assert!((args.rotary_emb_base - 10000.0).abs() < 1e-6);
    assert!((args.layer_norm_eps - 1e-5).abs() < 1e-12);
    assert!(args.activation_is_gelu());

    // No `quantization` block in the raw HuggingFace export.
    assert!(args.quantization.is_none());
    assert_eq!(args.eos_token_ids(), vec![0]);
    assert!(args.validate().is_ok());
}

#[test]
fn config_defaults_cover_a_bare_gpt_neox_config() {
    let args: ModelArgs = serde_json::from_str(r#"{"model_type": "gpt_neox"}"#).expect("parses");

    assert_eq!(args.hidden_size, 2048);
    assert_eq!(args.num_attention_heads, 8);
    assert_eq!(args.num_hidden_layers, 16);
    assert_eq!(args.max_position_embeddings, 2048);
    assert_eq!(args.vocab_size, 50304);
    // The family defaults: parallel residual, untied head, quarter RoPE.
    assert!(args.use_parallel_residual);
    assert!(!args.tie_word_embeddings);
    assert!((args.rotary_pct - 0.25).abs() < 1e-9);
    // No config-declared stop token: guessing one could truncate at an ordinary
    // token, since NeoX-derived instruction models add their own.
    assert!(args.eos_token_ids().is_empty());
    // An absent `hidden_act` is not a mismatch.
    assert!(args.activation_is_gelu());
    assert!(args.validate().is_ok());
}

#[test]
fn token_id_fields_accept_a_scalar_or_a_list() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_neox", "eos_token_id": [0, 50278], "bos_token_id": [0]}"#,
    )
    .expect("parses");

    assert!(matches!(args.eos_token_id, Some(TokenIdField::Multiple(_))));
    assert_eq!(args.eos_token_ids(), vec![0, 50278]);

    // `bos_token_id` is the fallback when the config omits `eos_token_id`.
    let args: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt_neox", "bos_token_id": 0}"#).expect("parses");
    assert_eq!(args.eos_token_ids(), vec![0]);

    // An empty list is not a stop token, so the fallback still applies.
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_neox", "eos_token_id": [], "bos_token_id": 7}"#,
    )
    .expect("parses");
    assert_eq!(args.eos_token_ids(), vec![7]);
}

#[test]
fn partial_rope_rotates_only_the_rotary_pct_fraction_of_each_head() {
    let args: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("parses");

    // 2048 / 8 = 256 channels per head, of which int(256 * 0.25) = 64 rotate and
    // the remaining 192 pass through untouched. This is the value handed to
    // `fast_rope`'s `dims` argument.
    assert_eq!(args.head_dim(), 256);
    assert_eq!(args.rope_dims(), 64);
    assert_eq!(args.head_dim() - args.rope_dims() as usize, 192);
    assert_ne!(
        args.rope_dims(),
        args.head_dim() as i32,
        "partial RoPE must not rotate the whole head"
    );

    // `int()` truncates rather than rounds, matching upstream.
    let truncating: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_neox", "hidden_size": 2048, "num_attention_heads": 8,
            "rotary_pct": 0.26}"#,
    )
    .expect("parses");
    assert_eq!(truncating.rope_dims(), 66, "int(256 * 0.26) == 66");

    // A full-rotation config still works; it is just no longer partial.
    let full: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_neox", "hidden_size": 2048, "num_attention_heads": 8,
            "rotary_pct": 1.0}"#,
    )
    .expect("parses");
    assert_eq!(full.rope_dims(), 256);
    assert!(full.validate().is_ok());
}

#[test]
fn intermediate_size_comes_from_the_config_or_four_times_hidden_size() {
    // Pythia's `intermediate_size` happens to be 4 * hidden_size, so use a
    // config where the two genuinely disagree.
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_neox", "hidden_size": 2048, "intermediate_size": 5632}"#,
    )
    .expect("parses");
    assert_eq!(args.intermediate_size(), 5632);

    // Absent and explicit-null both fall back to 4 * hidden_size, which is what
    // upstream hardcodes.
    let args: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt_neox", "hidden_size": 2048}"#).expect("parses");
    assert_eq!(args.intermediate_size(), 8192);
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_neox", "hidden_size": 2048, "intermediate_size": null}"#,
    )
    .expect("parses");
    assert_eq!(args.intermediate_size(), 8192);
}

#[test]
fn a_non_gelu_activation_is_reported_rather_than_applied_silently() {
    let gelu: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt_neox", "hidden_act": "gelu_fast"}"#)
            .expect("parses");
    assert!(gelu.activation_is_gelu());

    let silu: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt_neox", "hidden_act": "silu"}"#)
            .expect("parses");
    assert!(
        !silu.activation_is_gelu(),
        "an activation this loader does not implement must be detectable"
    );
    // It is a diagnostic, not a rejection: the activation cannot corrupt memory,
    // and refusing an otherwise loadable checkpoint over it would be worse.
    assert!(silu.validate().is_ok());
}

// Config validation. `config.json` arrives from the model directory, which for
// `mlxcel generate -m <org>/<repo>` is a third-party HuggingFace repo the
// download layer never parses.

#[test]
fn config_validation_rejects_hostile_scalars() {
    let cases: [(&str, &str); 8] = [
        // Divides by zero in `head_dim()`. `0.is_multiple_of(0)` is true, so the
        // divisibility check alone would let this through.
        (
            r#"{"model_type": "gpt_neox", "num_attention_heads": 0}"#,
            "num_attention_heads",
        ),
        (
            r#"{"model_type": "gpt_neox", "hidden_size": 0, "num_attention_heads": 0}"#,
            "num_attention_heads",
        ),
        (
            r#"{"model_type": "gpt_neox", "hidden_size": 0}"#,
            "hidden_size",
        ),
        (
            r#"{"model_type": "gpt_neox", "hidden_size": 2050}"#,
            "divisible",
        ),
        (
            r#"{"model_type": "gpt_neox", "num_hidden_layers": 0}"#,
            "num_hidden_layers",
        ),
        // Sizes the `Vec::with_capacity` in `from_weights`.
        (
            r#"{"model_type": "gpt_neox", "num_hidden_layers": 18446744073709551615}"#,
            "num_hidden_layers",
        ),
        (
            r#"{"model_type": "gpt_neox", "vocab_size": 0}"#,
            "vocab_size",
        ),
        // Reaches the `as i32` cast that sizes the MLP projections.
        (
            r#"{"model_type": "gpt_neox", "intermediate_size": 18446744073709551615}"#,
            "intermediate_size",
        ),
    ];

    for (config, expected) in cases {
        let args: ModelArgs = serde_json::from_str(config).expect("parses");
        let err = args
            .validate()
            .expect_err(&format!("must be rejected: {config}"));
        assert!(
            err.contains(expected),
            "unhelpful error for {config}: {err}"
        );
    }

    // `max_position_embeddings` is not an index bound in this family (there is
    // no learned position table), but it is still reported as the context
    // window, so an absurd value is rejected rather than propagated.
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_neox", "max_position_embeddings": 4294967296}"#,
    )
    .expect("parses");
    let err = args.validate().expect_err("must be rejected");
    assert!(
        err.contains("max_position_embeddings"),
        "unhelpful error: {err}"
    );

    // The `4 * hidden_size` fallback must not overflow into a valid-looking
    // width when `hidden_size` itself is at the ceiling.
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_neox", "hidden_size": 65536, "num_attention_heads": 8}"#,
    )
    .expect("parses");
    assert_eq!(args.intermediate_size(), 262_144);
    assert!(args.validate().is_ok());
}

#[test]
fn config_validation_rejects_out_of_range_rope_parameters() {
    // `rotary_pct` is the first float this family lets a config author control.
    // MLX does check `rope`'s `dims` (positive, even, and no larger than the
    // last axis), but it checks by throwing, and an MLX C++ exception crossing
    // the cxx bridge is an uncatchable `std::terminate`: the process dies with
    // SIGABRT at the first forward pass, after the model has already loaded and
    // a server has already accepted it. Every value MLX would throw on has to be
    // rejected here instead, at load.
    let float_cases: [(&str, &str); 6] = [
        // int(256 * 0.0) == 0 rotary dimensions.
        (
            r#"{"model_type": "gpt_neox", "rotary_pct": 0.0}"#,
            "rotary_pct",
        ),
        // Truncates to 0 as well, so the boundary is exercised from both sides.
        (
            r#"{"model_type": "gpt_neox", "rotary_pct": 0.003}"#,
            "rotary_pct",
        ),
        (
            r#"{"model_type": "gpt_neox", "rotary_pct": -0.25}"#,
            "rotary_pct",
        ),
        // Past the end of the head: int(256 * 1.5) == 384 > 256.
        (
            r#"{"model_type": "gpt_neox", "rotary_pct": 1.5}"#,
            "rotary_pct",
        ),
        // Odd, which MLX refuses outright ("[rope] dims must be even"): RoPE
        // rotates channel pairs. On a 4-wide head int(4 * 0.75) == 3.
        (
            r#"{"model_type": "gpt_neox", "hidden_size": 8, "num_attention_heads": 2,
                "rotary_pct": 0.75}"#,
            "even",
        ),
        // The same at Pythia's head width: int(256 * (3 / 256)) == 3.
        (
            r#"{"model_type": "gpt_neox", "rotary_pct": 0.01171875}"#,
            "even",
        ),
    ];
    for (config, expected) in float_cases {
        let args: ModelArgs = serde_json::from_str(config).expect("parses");
        let err = args
            .validate()
            .expect_err(&format!("must be rejected: {config}"));
        assert!(
            err.contains(expected),
            "unhelpful error for {config}: {err}"
        );
    }

    // JSON has no NaN or infinity literal, so build those directly. The `as i32`
    // cast in `rope_dims` saturates, turning NaN into 0 and infinity into
    // `i32::MAX`; both are out of range, but the message must name the real
    // problem rather than the derived one.
    for pct in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut args: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("parses");
        args.rotary_pct = pct;
        let err = args
            .validate()
            .expect_err("a non-finite rotary_pct must be rejected");
        assert!(
            err.contains("finite"),
            "unhelpful error for rotary_pct {pct}: {err}"
        );
    }

    // RoPE takes the logarithm of the base, so a zero or negative base makes
    // every rotated channel NaN without anything throwing.
    for base in [0.0f32, -10000.0, f32::NAN, f32::INFINITY] {
        let mut args: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("parses");
        args.rotary_emb_base = base;
        let err = args
            .validate()
            .expect_err("a non-positive or non-finite rotary_emb_base must be rejected");
        assert!(
            err.contains("rotary_emb_base"),
            "unhelpful error for base {base}: {err}"
        );
    }

    // The exact boundaries stay accepted. The floor is two rotary dimensions,
    // not one: a single dimension is odd, and MLX refuses an odd `dims`.
    let mut args: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("parses");
    args.rotary_pct = 1.0;
    assert_eq!(args.rope_dims(), 256);
    assert!(args.validate().is_ok());
    args.rotary_pct = 2.0 / 256.0;
    assert_eq!(args.rope_dims(), 2);
    assert!(args.validate().is_ok());

    args.rotary_pct = 1.0 / 256.0;
    assert_eq!(args.rope_dims(), 1);
    let err = args
        .validate()
        .expect_err("one rotary dimension is odd, and MLX refuses an odd rope `dims`");
    assert!(err.contains("even"), "unhelpful error: {err}");
}

#[test]
fn config_validation_rejects_a_layer_norm_eps_that_would_nan_every_hidden_state() {
    // Unlike the rope `dims` contract, MLX's `fast::layer_norm` never looks at
    // `eps`, so none of these throws. They compute `x * rsqrt(mean(x^2) + eps)`
    // and hand back NaN, which propagates through every remaining layer into the
    // logits and then into the sampler. A checkpoint that does this loads
    // cleanly and generates uniform garbage, so the rejection has to happen at
    // load or not at all.
    //
    // Zero is refused for its own reason: an all-zero row gives `rsqrt(0)`, and
    // `0 * inf` is NaN again.
    for (config, label) in [
        (
            r#"{"model_type": "gpt_neox", "layer_norm_eps": 0.0}"#,
            "zero",
        ),
        (
            r#"{"model_type": "gpt_neox", "layer_norm_eps": -1e-5}"#,
            "negative",
        ),
    ] {
        let args: ModelArgs = serde_json::from_str(config).expect("parses");
        let err = args
            .validate()
            .expect_err(&format!("a {label} layer_norm_eps must be rejected"));
        assert!(
            err.contains("layer_norm_eps"),
            "unhelpful error for a {label} layer_norm_eps: {err}"
        );
    }

    // JSON has no NaN or infinity literal, so build those directly.
    for eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut args: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("parses");
        args.layer_norm_eps = eps;
        let err = args
            .validate()
            .expect_err("a non-finite layer_norm_eps must be rejected");
        assert!(
            err.contains("layer_norm_eps"),
            "unhelpful error for layer_norm_eps {eps}: {err}"
        );
    }

    // The real value stays accepted, and so does the whole small-positive range
    // a checkpoint might plausibly declare.
    for eps in [1e-5f32, 1e-12, 1e-3, 1.0] {
        let mut args: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("parses");
        args.layer_norm_eps = eps;
        assert!(
            args.validate().is_ok(),
            "layer_norm_eps {eps} is an ordinary value and must be accepted"
        );
    }
}

#[test]
fn config_validation_rejects_a_quantization_block_that_would_abort_an_mlx_kernel() {
    // MLX computes `packed_in * 32 / bits`, which divides by zero at 0 and
    // collapses to zero above 32, and the throw that follows crosses the cxx
    // bridge as an uncatchable `std::terminate`. Since issue #929
    // `reconcile_quantization_layout` refuses such a pair itself, so this is the
    // family-level early diagnostic rather than the only guard: it names GPT-NeoX
    // and fires during config validation, before any tensor is touched.
    let hostile = [
        (r#""group_size": 64, "bits": 0"#, "bits"),
        (r#""group_size": 64, "bits": -4"#, "bits"),
        (r#""group_size": 64, "bits": 33"#, "bits"),
        (r#""group_size": 0, "bits": 4"#, "group_size"),
        (r#""group_size": -64, "bits": 4"#, "group_size"),
    ];
    for (quantization, expected) in hostile {
        let config = format!(r#"{{"model_type": "gpt_neox", "quantization": {{{quantization}}}}}"#);
        let args: ModelArgs = serde_json::from_str(&config).expect("parses");
        let err = args
            .validate()
            .expect_err(&format!("must be rejected: {config}"));
        assert!(
            err.contains(expected),
            "unhelpful error for {config}: {err}"
        );
    }

    // Every bit width and group size a real export declares stays accepted. The
    // guard is a range rather than an allowlist because mlxcel re-derives an
    // effective bit width from the tensor shapes when the declared one
    // disagrees, and an allowlist would reject the mixed-precision exports that
    // relies on.
    for (group_size, bits) in [(32, 4), (64, 4), (128, 4), (64, 8), (64, 6), (16, 4)] {
        let config = format!(
            r#"{{"model_type": "gpt_neox", "quantization": {{"group_size": {group_size}, "bits": {bits}}}}}"#
        );
        let args: ModelArgs = serde_json::from_str(&config).expect("parses");
        assert!(
            args.validate().is_ok(),
            "group_size {group_size} / bits {bits} is a real export and must be accepted"
        );
        assert_eq!(args.group_size(), group_size);
        assert_eq!(args.bits(), bits);
    }

    // An absent block keeps the defaults and stays accepted.
    let args: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("parses");
    assert!(args.quantization.is_none());
    assert!(args.validate().is_ok());
}

#[test]
fn every_accepted_rotary_pct_survives_a_real_rope_call() {
    // `validate_rope` exists because MLX enforces its `dims` contract by
    // throwing, and a throw crossing the cxx bridge is an uncatchable
    // `std::terminate` rather than a Rust error. That makes the guard's
    // agreement with MLX load-bearing in a way an ordinary range check is not:
    // if it ever accepts a value MLX refuses, the production failure is a
    // SIGABRT on the first request, not a rejected load.
    //
    // So sweep `rotary_pct` and put every accepted value through a real
    // `fast_rope` call on a head-shaped array. Accepting too much aborts this
    // test binary, which is the loudest possible signal; accepting too little
    // trips the count assertions below.
    let mut args = tiny_args();
    let head_dim = args.head_dim() as i32;
    assert_eq!(head_dim, 4);

    let mut accepted = Vec::new();
    for step in 0..=400 {
        args.rotary_pct = step as f32 / 100.0;
        if args.validate().is_err() {
            continue;
        }
        accepted.push(args.rope_dims());
        let q = counting(&[1, 2, 1, head_dim]);
        let rotated =
            mlxcel_core::fast_rope(&q, args.rope_dims(), false, args.rotary_emb_base, 1.0, 3);
        assert_eq!(mlxcel_core::array_shape(&rotated), vec![1, 2, 1, head_dim]);
    }
    assert!(
        accepted.iter().all(|d| *d == 2 || *d == 4),
        "a 4-wide head admits exactly the even counts 2 and 4, got {accepted:?}"
    );
    assert!(accepted.contains(&2) && accepted.contains(&4));

    // The same sweep at Pythia's head width, walking the derived dimension
    // count one at a time so every odd value is actually offered to the guard.
    let mut pythia: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("parses");
    let head_dim = pythia.head_dim() as i32;
    assert_eq!(head_dim, 256);
    let mut accepted = 0;
    for dims in 1..=head_dim {
        pythia.rotary_pct = dims as f32 / head_dim as f32;
        if pythia.validate().is_err() {
            continue;
        }
        assert_eq!(
            pythia.rope_dims() % 2,
            0,
            "an odd dims must never be accepted"
        );
        accepted += 1;
        let q = counting(&[1, 1, 1, head_dim]);
        let rotated = mlxcel_core::fast_rope(
            &q,
            pythia.rope_dims(),
            false,
            pythia.rotary_emb_base,
            1.0,
            0,
        );
        assert_eq!(mlxcel_core::array_shape(&rotated), vec![1, 1, 1, head_dim]);
    }
    assert_eq!(accepted, 128, "exactly the even counts 2..=256 are legal");
}

// Synthetic checkpoints.

/// hidden_size 8, 2 heads of width 4, `rotary_pct` 0.5 so 2 of the 4 channels
/// per head rotate. The fused projection is 24 wide, laid out per head as 12
/// contiguous channels.
fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
            "model_type": "gpt_neox",
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "max_position_embeddings": 16,
            "vocab_size": 10,
            "layer_norm_eps": 1e-5,
            "rotary_emb_base": 10000,
            "rotary_pct": 0.5,
            "use_parallel_residual": true,
            "tie_word_embeddings": false,
            "eos_token_id": 9
        }"#,
    )
    .expect("tiny config parses")
}

/// Same shape as [`tiny_args`] but with two layers, so a checkpoint whose layers
/// disagree with each other can be built.
fn two_layer_args() -> ModelArgs {
    let mut args = tiny_args();
    args.num_hidden_layers = 2;
    args
}

/// Deterministic non-zero filler so LayerNorm and softmax see real values.
fn filled(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// `0, 1, 2, ...` in row-major order, so every element identifies its own
/// channel index. Used by the interleaved-QKV tests.
fn counting(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}

fn zeros(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![0.0; n as usize], shape)
}

fn identity(dim: i32) -> UniquePtr<MlxArray> {
    let mut data = vec![0.0f32; (dim * dim) as usize];
    for i in 0..dim {
        data[(i * dim + i) as usize] = 1.0;
    }
    mlxcel_core::from_slice_f32(&data, &[dim, dim])
}

/// Read one element out of an array by full index.
///
/// Slicing at explicit coordinates rather than flattening first keeps this
/// correct for the strided views `split_interleaved_qkv` returns.
fn element(x: &MlxArray, index: &[i32]) -> f32 {
    let stops: Vec<i32> = index.iter().map(|i| i + 1).collect();
    mlxcel_core::item_f32(&mlxcel_core::slice(x, index, &stops))
}

/// Every element of a rank-4 `[b, h, l, d]` array in row-major order.
fn values_4d(x: &MlxArray) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(x);
    assert_eq!(shape.len(), 4, "expected a rank-4 array, got {shape:?}");
    let mut out = Vec::new();
    for b in 0..shape[0] {
        for h in 0..shape[1] {
            for l in 0..shape[2] {
                for d in 0..shape[3] {
                    out.push(element(x, &[b, h, l, d]));
                }
            }
        }
    }
    out
}

/// Every element of a rank-3 `[b, l, d]` array in row-major order.
fn values_3d(x: &MlxArray) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(x);
    assert_eq!(shape.len(), 3, "expected a rank-3 array, got {shape:?}");
    let mut out = Vec::new();
    for b in 0..shape[0] {
        for l in 0..shape[1] {
            for d in 0..shape[2] {
                out.push(element(x, &[b, l, d]));
            }
        }
    }
    out
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(
        &mlxcel_core::subtract(a, b),
    )))
}

/// A raw HuggingFace GPT-NeoX export: `gpt_neox.`-prefixed decoder keys,
/// `gpt_neox.layers.N`, and the untied head `embed_out` at the top level.
/// Includes the three registered buffers a real checkpoint carries.
fn hf_weights(args: &ModelArgs) -> WeightMap {
    let h = args.hidden_size as i32;
    let ff = args.intermediate_size() as i32;
    let qkv = 3 * h;

    let mut w = WeightMap::new();
    w.insert(
        "gpt_neox.embed_in.weight".into(),
        filled(&[args.vocab_size as i32, h]),
    );
    w.insert("gpt_neox.final_layer_norm.weight".into(), ones(&[h]));
    w.insert("gpt_neox.final_layer_norm.bias".into(), zeros(&[h]));
    w.insert(
        "embed_out.weight".into(),
        filled(&[args.vocab_size as i32, h]),
    );

    for i in 0..args.num_hidden_layers {
        let p = format!("gpt_neox.layers.{i}");
        w.insert(format!("{p}.input_layernorm.weight"), ones(&[h]));
        w.insert(format!("{p}.input_layernorm.bias"), zeros(&[h]));
        w.insert(format!("{p}.post_attention_layernorm.weight"), ones(&[h]));
        w.insert(format!("{p}.post_attention_layernorm.bias"), zeros(&[h]));
        w.insert(
            format!("{p}.attention.query_key_value.weight"),
            filled(&[qkv, h]),
        );
        w.insert(
            format!("{p}.attention.query_key_value.bias"),
            filled(&[qkv]),
        );
        w.insert(format!("{p}.attention.dense.weight"), filled(&[h, h]));
        w.insert(format!("{p}.attention.dense.bias"), filled(&[h]));
        w.insert(format!("{p}.mlp.dense_h_to_4h.weight"), filled(&[ff, h]));
        w.insert(format!("{p}.mlp.dense_h_to_4h.bias"), filled(&[ff]));
        w.insert(format!("{p}.mlp.dense_4h_to_h.weight"), filled(&[h, ff]));
        w.insert(format!("{p}.mlp.dense_4h_to_h.bias"), filled(&[h]));

        // The PyTorch registered buffers a real export carries.
        w.insert(format!("{p}.attention.bias"), ones(&[1, 1, 4, 4]));
        w.insert(format!("{p}.attention.masked_bias"), zeros(&[1]));
        w.insert(format!("{p}.attention.rotary_emb.inv_freq"), ones(&[2]));
    }
    w
}

/// The same checkpoint after upstream mlx-lm's `sanitize`: `model.`-prefixed,
/// `model.h.N` blocks, `model.embed_out`.
fn mlx_converted_weights(args: &ModelArgs) -> WeightMap {
    let mut out = WeightMap::new();
    for (key, value) in hf_weights(args) {
        let renamed = if let Some(rest) = key.strip_prefix("gpt_neox.layers.") {
            format!("model.h.{rest}")
        } else if let Some(rest) = key.strip_prefix("gpt_neox.") {
            format!("model.{rest}")
        } else {
            format!("model.{key}")
        };
        out.insert(renamed, value);
    }
    // A conversion is produced from the already-sanitized module tree, so the
    // registered buffers never reach it.
    strip_registered_buffers(&mut out);
    out
}

// THE trap: the interleaved per-head QKV split.

#[test]
fn interleaved_qkv_split_pins_the_head_major_layout() {
    // 1 token, 2 heads of width 4, so the fused projection is 24 channels laid
    // out head-major:
    //
    //   head 0: q 0..3    k 4..7    v 8..11
    //   head 1: q 12..15  k 16..19  v 20..23
    //
    // The array values are their own channel indices, so the assertions below
    // are the layout itself rather than a shape.
    let qkv = counting(&[1, 1, 24]);
    let (q, k, v) = split_interleaved_qkv(&qkv, 1, 1, 2, 4);

    for (name, tensor) in [("q", &q), ("k", &k), ("v", &v)] {
        assert_eq!(
            mlxcel_core::array_shape(tensor),
            vec![1, 2, 1, 4],
            "{name} must be [batch, heads, seq, head_dim]"
        );
    }

    assert_eq!(
        values_4d(&q),
        vec![0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0]
    );
    assert_eq!(
        values_4d(&k),
        vec![4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]
    );
    assert_eq!(
        values_4d(&v),
        vec![8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0]
    );

    // The same layout stated as a pure function of the config: within one head's
    // `3 * head_dim` block, Q ends at `head_dim`, K at `2 * head_dim` and V at
    // `3 * head_dim`.
    let mut args = tiny_args();
    args.hidden_size = 8;
    args.num_attention_heads = 2;
    assert_eq!(args.interleaved_qkv_channel_offsets(), (4, 8, 12));

    // Pythia 1B: 8 heads of width 256, so each head contributes 768 channels.
    let pythia: ModelArgs = serde_json::from_str(PYTHIA_1B_CONFIG).expect("parses");
    assert_eq!(pythia.interleaved_qkv_channel_offsets(), (256, 512, 768));
}

#[test]
fn a_flat_three_way_split_would_take_different_channels() {
    // The mistake this family invites: reuse the GPT-2 / GPT-BigCode pattern of
    // slicing the *unreshaped* projection into three contiguous `hidden_size`
    // blocks. With the same 24-channel projection that would take channels
    // 0..7 as Q, 8..15 as K and 16..23 as V, spelled out here as literals so
    // the comparison does not depend on any array op.
    //
    // Under the real head-major layout those runs are something else entirely:
    // channels 0..7 are head 0's Q *and* K, and channels 8..15 are head 0's V
    // plus head 1's Q. The two splits produce tensors of identical shape, which
    // is exactly why no shape assertion anywhere in this suite would catch the
    // substitution and why the model would still decode fluent English.
    let flat_q = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let flat_k = vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let flat_v = vec![16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0];

    let qkv = counting(&[1, 1, 24]);
    let (q, k, v) = split_interleaved_qkv(&qkv, 1, 1, 2, 4);

    for (name, correct, wrong) in [("q", &q, flat_q), ("k", &k, flat_k), ("v", &v, flat_v)] {
        let correct_values = values_4d(correct);
        assert_eq!(
            correct_values.len(),
            wrong.len(),
            "the flat split is element-count identical for {name}"
        );
        assert_ne!(
            correct_values, wrong,
            "{name} must not be the flat three-way split of the projection"
        );
    }
}

#[test]
fn attention_forward_returns_the_interleaved_v_block() {
    // An end-to-end pin that does not depend on the internal helper. With a
    // single token, attention softmaxes over exactly one key, so its output is V
    // verbatim; with an identity `dense` and no bias, the block output *is* the
    // V channels. RoPE touches Q and K only, so it cannot perturb this.
    //
    // The fused projection is made constant by zeroing its weight and setting
    // its bias to 0..23, so the observed output names the channels V was taken
    // from.
    let args = tiny_args();
    let h = args.hidden_size as i32;
    let qkv = 3 * h;

    let mut weights = WeightMap::new();
    weights.insert("attention.query_key_value.weight".into(), zeros(&[qkv, h]));
    weights.insert("attention.query_key_value.bias".into(), counting(&[qkv]));
    weights.insert("attention.dense.weight".into(), identity(h));

    let attention =
        Attention::from_weights(&weights, &args, "attention").expect("attention builds");
    let mut cache = KVCache::new();
    let x = filled(&[1, 1, h]);
    let out = attention.forward(&x, &mut cache, None);

    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 1, h]);
    // head 0's V is channels 8..11, head 1's V is channels 20..23.
    assert_eq!(
        values_3d(&out),
        vec![8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0],
        "attention must read V from the interleaved per-head block"
    );
    // A flat three-way split would have taken the last third, 16..23.
    assert_ne!(
        values_3d(&out),
        vec![16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
    );
}

// Partial RoPE actually reaches the graph.

#[test]
fn partial_rope_leaves_the_unrotated_channels_untouched() {
    // With `rotary_pct` 0.5 the first 2 of each head's 4 channels rotate. At
    // offset 0 the rotation is the identity, so instead compare against a
    // deliberately wrong full rotation: if `rope_dims` were ignored and the
    // whole head rotated, the tail channels would change at a non-zero offset.
    let args = tiny_args();
    assert_eq!(args.head_dim(), 4);
    assert_eq!(args.rope_dims(), 2);

    let q = counting(&[1, 2, 1, 4]);
    let partial = mlxcel_core::fast_rope(&q, args.rope_dims(), false, args.rotary_emb_base, 1.0, 3);
    let full = mlxcel_core::fast_rope(
        &q,
        args.head_dim() as i32,
        false,
        args.rotary_emb_base,
        1.0,
        3,
    );

    let partial_values = values_4d(&partial);
    let original = values_4d(&q);
    let full_values = values_4d(&full);

    for head in 0..2usize {
        for channel in 2..4usize {
            let i = head * 4 + channel;
            assert!(
                (partial_values[i] - original[i]).abs() < 1e-5,
                "channel {channel} of head {head} is outside rope_dims and must pass through \
                 unrotated: {} vs {}",
                partial_values[i],
                original[i]
            );
        }
    }
    assert_ne!(
        partial_values, full_values,
        "a partial rotation must differ from rotating the whole head"
    );
}

// Key layout.

#[test]
fn detects_both_the_raw_and_the_converted_layout() {
    let args = tiny_args();

    let layout = GptNeoxLayout::detect(&hf_weights(&args)).expect("raw layout detected");
    assert_eq!(layout.prefix, "gpt_neox.");
    assert_eq!(layout.layers_key, "layers");
    assert_eq!(layout.embed_out, "embed_out");
    assert_eq!(layout.layer_prefix(3), "gpt_neox.layers.3");

    let layout =
        GptNeoxLayout::detect(&mlx_converted_weights(&args)).expect("converted layout detected");
    assert_eq!(layout.prefix, "model.");
    assert_eq!(layout.layers_key, "h");
    assert_eq!(layout.embed_out, "model.embed_out");
    assert_eq!(layout.layer_prefix(3), "model.h.3");
}

#[test]
fn detection_rejects_a_weight_map_with_no_known_layout() {
    let args = tiny_args();
    let mut weights = WeightMap::new();
    for (key, value) in hf_weights(&args) {
        weights.insert(key.replace("gpt_neox.", "transformer."), value);
    }

    let err = GptNeoxLayout::detect(&weights).expect_err("must not guess a layout");
    assert!(err.contains("embed_in.weight"), "unhelpful error: {err}");
}

#[test]
fn both_layouts_build_the_same_model() {
    let args = tiny_args();
    let raw = GptNeoxModel::from_weights(&hf_weights(&args), &args).expect("raw builds");
    let converted =
        GptNeoxModel::from_weights(&mlx_converted_weights(&args), &args).expect("converted builds");

    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let mut raw_caches = raw.make_caches();
    let mut converted_caches = converted.make_caches();
    let raw_logits = raw.forward(&prompt, &mut raw_caches, None);
    let converted_logits = converted.forward(&prompt, &mut converted_caches, None);

    assert!(
        max_abs_diff(&raw_logits, &converted_logits) < 1e-6,
        "the two key layouts describe the same weights and must produce the same logits"
    );
}

// Registered buffers.

#[test]
fn strips_the_registered_buffers_but_keeps_the_real_projection_biases() {
    let args = two_layer_args();
    let mut weights = hf_weights(&args);

    // Three buffers per layer, and `.attention.bias` collides in name with the
    // projection bias namespace exactly as GPT-2's `h.N.attn.bias` does.
    assert!(weights.contains_key("gpt_neox.layers.0.attention.bias"));
    assert!(weights.contains_key("gpt_neox.layers.1.attention.masked_bias"));
    assert!(weights.contains_key("gpt_neox.layers.0.attention.rotary_emb.inv_freq"));

    let removed = strip_registered_buffers(&mut weights);
    assert_eq!(removed, 3 * args.num_hidden_layers);

    for i in 0..args.num_hidden_layers {
        let p = format!("gpt_neox.layers.{i}");
        assert!(!weights.contains_key(&format!("{p}.attention.bias")));
        assert!(!weights.contains_key(&format!("{p}.attention.masked_bias")));
        assert!(!weights.contains_key(&format!("{p}.attention.rotary_emb.inv_freq")));

        // The real biases must survive: neither ends in `.attention.bias`.
        assert!(weights.contains_key(&format!("{p}.attention.query_key_value.bias")));
        assert!(weights.contains_key(&format!("{p}.attention.dense.bias")));
        assert!(weights.contains_key(&format!("{p}.input_layernorm.bias")));
    }

    // A second pass removes nothing.
    assert_eq!(strip_registered_buffers(&mut weights), 0);
}

#[test]
fn the_model_builds_whether_or_not_the_buffers_were_stripped() {
    // `load()` strips before constructing, but `from_weights` is also reachable
    // directly (the owned-weights route), and the graph must simply never ask
    // for those keys.
    let args = tiny_args();
    let with_buffers = hf_weights(&args);
    let mut without_buffers = hf_weights(&args);
    strip_registered_buffers(&mut without_buffers);

    let a = GptNeoxModel::from_weights(&with_buffers, &args).expect("builds with buffers");
    let b = GptNeoxModel::from_weights(&without_buffers, &args).expect("builds without buffers");

    let prompt = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut caches_a = a.make_caches();
    let mut caches_b = b.make_caches();
    let logits_a = a.forward(&prompt, &mut caches_a, None);
    let logits_b = b.forward(&prompt, &mut caches_b, None);
    assert!(max_abs_diff(&logits_a, &logits_b) < 1e-6);
}

// Projection loading.

#[test]
fn projections_load_without_a_transpose() {
    let args = tiny_args();
    let weights = hf_weights(&args);
    let model = GptNeoxModel::from_weights(&weights, &args).expect("model builds");

    let h = args.hidden_size as i32;
    let ff = args.intermediate_size() as i32;

    let checks: [(&UnifiedLinear, [i32; 2], Option<i32>); 4] = [
        (
            &model.h[0].attention.query_key_value,
            [3 * h, h],
            Some(3 * h),
        ),
        (&model.h[0].attention.dense, [h, h], Some(h)),
        (&model.h[0].mlp.dense_h_to_4h, [ff, h], Some(ff)),
        (&model.h[0].mlp.dense_4h_to_h, [h, ff], Some(h)),
    ];

    for (linear, expected_weight, expected_bias) in checks {
        let UnifiedLinear::Regular(inner) = linear else {
            panic!("an unquantized GPT-NeoX checkpoint must load as a regular Linear");
        };
        assert_eq!(
            mlxcel_core::array_shape(&inner.weight),
            expected_weight.to_vec(),
            "the stored [out, in] weight must reach the graph unchanged"
        );
        if let Some(width) = expected_bias {
            let bias = inner
                .bias
                .as_ref()
                .expect("GPT-NeoX ships a bias on every projection");
            assert_eq!(mlxcel_core::array_shape(bias), vec![width]);
        }
    }
}

#[test]
fn load_linear_rejects_a_transposed_degenerate_or_missized_projection() {
    let args = tiny_args();
    let h = args.hidden_size as i32;

    // The [in, out] orientation. GPT-NeoX uses `nn.Linear`, so a genuine
    // checkpoint is already [out, in]; transposing here would silently corrupt
    // the projection.
    let mut weights = hf_weights(&args);
    weights.insert(
        "gpt_neox.layers.0.attention.query_key_value.weight".into(),
        filled(&[h, 3 * h]),
    );
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a transposed projection must be rejected, not accepted"),
        Err(err) => err,
    };
    assert!(
        err.contains("query_key_value.weight") && err.contains("nn.Linear"),
        "the error must name the tensor and the orientation: {err}"
    );

    // Rank 3 instead of rank 2. Nothing else would notice until the matmul, and
    // an MLX C++ exception crossing the cxx bridge is an uncatchable
    // `std::terminate` rather than a load error.
    let mut weights = hf_weights(&args);
    weights.insert(
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight".into(),
        filled(&[1, args.intermediate_size() as i32, h]),
    );
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a rank-3 projection weight must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.contains("dense_h_to_4h.weight"),
        "unhelpful error: {err}"
    );

    // A bias that is not the width of its projection output.
    let mut weights = hf_weights(&args);
    weights.insert(
        "gpt_neox.layers.0.attention.dense.bias".into(),
        filled(&[h + 1]),
    );
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a mis-sized projection bias must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("dense.bias"), "unhelpful error: {err}");

    // A missing weight names the key it looked for.
    let mut weights = hf_weights(&args);
    weights.remove("gpt_neox.layers.0.attention.dense.weight");
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a missing projection must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("dense.weight"), "unhelpful error: {err}");
}

#[test]
fn every_layer_is_checked_not_just_layer_zero() {
    let args = two_layer_args();
    let mut weights = hf_weights(&args);
    weights.insert(
        "gpt_neox.layers.1.attention.query_key_value.weight".into(),
        filled(&[args.hidden_size as i32, 3 * args.hidden_size as i32]),
    );

    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a layer that disagrees with the config must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.contains("gpt_neox.layers.1.attention.query_key_value.weight"),
        "the error must name the offending tensor: {err}"
    );
}

#[test]
fn a_quantized_projection_skips_the_packed_input_width_but_not_the_output_width() {
    // Packing compresses the input axis, so a packed weight matches no float
    // input layout and that half of the contract must not be applied to it;
    // `UnifiedLinear` reconciles the quantization layout itself. Shapes here
    // follow the 4-bit affine packing (8 values per u32).
    let mut weights = WeightMap::new();
    weights.insert("q.weight".into(), zeros(&[24, 1]));
    weights.insert("q.scales".into(), ones(&[24, 1]));
    weights.insert("q.biases".into(), zeros(&[24, 1]));

    let loaded = load_linear(&weights, "q", 8, 24, 8, 4).expect("a quantized projection loads");
    assert!(
        matches!(loaded, UnifiedLinear::Quantized { .. }),
        "the packed path must not be routed through the float input-width check"
    );

    // The row count is untouched by packing, so the output width still has to be
    // the one the config implies. `Attention::forward` reshapes the fused
    // projection to `[batch, seq, num_heads, 3 * head_dim]` using config-derived
    // widths, so a packed `query_key_value` of a different real width makes the
    // reshape throw, which crosses the cxx bridge as an uncatchable
    // `std::terminate` rather than a load error.
    for rows in [32, 16] {
        let mut weights = WeightMap::new();
        weights.insert("q.weight".into(), zeros(&[rows, 1]));
        weights.insert("q.scales".into(), ones(&[rows, 1]));
        weights.insert("q.biases".into(), zeros(&[rows, 1]));

        let err = match load_linear(&weights, "q", 8, 24, 8, 4) {
            Ok(_) => panic!("a packed weight of output width {rows} must not load as width 24"),
            Err(err) => err,
        };
        assert!(err.contains("q.weight"), "unhelpful error: {err}");
    }

    // A quantized projection's bias is a plain 1-D vector, so it is checked on
    // the packed path too.
    let mut weights = WeightMap::new();
    weights.insert("q.weight".into(), zeros(&[24, 1]));
    weights.insert("q.scales".into(), ones(&[24, 1]));
    weights.insert("q.biases".into(), zeros(&[24, 1]));
    weights.insert("q.bias".into(), zeros(&[25]));

    let err = match load_linear(&weights, "q", 8, 24, 8, 4) {
        Ok(_) => panic!("a mis-sized bias must be rejected on the packed path too"),
        Err(err) => err,
    };
    assert!(err.contains("q.bias"), "unhelpful error: {err}");
}

// Tables and norms versus the config.

#[test]
fn from_weights_rejects_a_config_that_overstates_the_token_table() {
    // Token ids are bounded by `vocab_size`, not by the rows actually present.
    // The gather wraps a negative index but does not range-check a positive one,
    // so those reads return whatever follows the table in the buffer and the
    // values reach the logits.
    let args = tiny_args();
    let mut weights = hf_weights(&args);
    weights.insert(
        "gpt_neox.embed_in.weight".into(),
        filled(&[3, args.hidden_size as i32]),
    );
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a config that overstates the embed_in table must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("vocab_size"), "unhelpful error: {err}");

    // The other direction is safe: a padded table keeps the bound inside it.
    let mut weights = hf_weights(&args);
    weights.insert(
        "gpt_neox.embed_in.weight".into(),
        filled(&[64, args.hidden_size as i32]),
    );
    assert!(GptNeoxModel::from_weights(&weights, &args).is_ok());
}

#[test]
fn from_weights_rejects_tables_and_norms_of_the_wrong_width() {
    let args = tiny_args();

    let mut weights = hf_weights(&args);
    weights.insert(
        "gpt_neox.embed_in.weight".into(),
        filled(&[args.vocab_size as i32, 4]),
    );
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("an embedding width that disagrees with hidden_size must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.contains("gpt_neox.embed_in.weight") && err.contains("hidden_size"),
        "unhelpful error: {err}"
    );

    let mut weights = hf_weights(&args);
    weights.insert(
        "gpt_neox.layers.0.post_attention_layernorm.weight".into(),
        ones(&[4]),
    );
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a LayerNorm of the wrong width must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.contains("post_attention_layernorm.weight"),
        "unhelpful error: {err}"
    );

    let mut weights = hf_weights(&args);
    weights.insert("gpt_neox.final_layer_norm.weight".into(), ones(&[4]));
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a final norm of the wrong width must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.contains("final_layer_norm.weight"),
        "unhelpful error: {err}"
    );
}

// The output head.

#[test]
fn an_untied_config_requires_a_real_embed_out() {
    let args = tiny_args();
    assert!(!args.tie_word_embeddings, "the family default is untied");

    // Falling back to the tied path would produce logits from the wrong matrix.
    let mut weights = hf_weights(&args);
    weights.remove("embed_out.weight");
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("an untied config without an embed_out must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("embed_out.weight"), "unhelpful error: {err}");

    let model = GptNeoxModel::from_weights(&hf_weights(&args), &args).expect("model builds");
    assert!(model.embed_out.is_some());

    // A tied config uses the embedding as the head and needs no extra tensor.
    let mut tied = tiny_args();
    tied.tie_word_embeddings = true;
    let mut weights = hf_weights(&tied);
    weights.remove("embed_out.weight");
    let model = GptNeoxModel::from_weights(&weights, &tied).expect("tied model builds");
    assert!(model.embed_out.is_none());
}

// Both residual layouts.

#[test]
fn parallel_residual_feeds_both_sub_layers_the_same_pre_norm_input() {
    // `out = x + attn(input_layernorm(x)) + mlp(post_attention_layernorm(x))`.
    // Recomputing the sum from the block's own sub-modules pins the wiring
    // rather than the arithmetic: the MLP must read `x`, not the post-attention
    // residual.
    let args = tiny_args();
    assert!(args.use_parallel_residual);
    let model = GptNeoxModel::from_weights(&hf_weights(&args), &args).expect("model builds");
    let block = &model.h[0];

    let x = filled(&[1, 3, args.hidden_size as i32]);

    let mut cache = KVCache::new();
    let attn = block
        .attention
        .forward(&block.input_layernorm.forward(&x), &mut cache, None);
    let mlp = block
        .mlp
        .forward(&block.post_attention_layernorm.forward(&x));
    let expected = mlxcel_core::add(&mlxcel_core::add(&x, &attn), &mlp);

    let mut fresh = KVCache::new();
    let actual = block.forward(&x, &mut fresh, None);

    assert!(
        max_abs_diff(&expected, &actual) < 1e-5,
        "the parallel block must sum x, attn(ln1(x)) and mlp(ln2(x))"
    );
}

#[test]
fn sequential_residual_feeds_the_mlp_the_post_attention_residual() {
    // `h = x + attn(input_layernorm(x)); out = h + mlp(post_attention_layernorm(h))`.
    // The distinguishing input to `post_attention_layernorm` is `h`, not `x`.
    let mut args = tiny_args();
    args.use_parallel_residual = false;
    let model = GptNeoxModel::from_weights(&hf_weights(&args), &args).expect("model builds");
    let block = &model.h[0];
    assert!(!block.use_parallel_residual);

    let x = filled(&[1, 3, args.hidden_size as i32]);

    let mut cache = KVCache::new();
    let attn = block
        .attention
        .forward(&block.input_layernorm.forward(&x), &mut cache, None);
    let h = mlxcel_core::add(&x, &attn);
    let mlp = block
        .mlp
        .forward(&block.post_attention_layernorm.forward(&h));
    let expected = mlxcel_core::add(&h, &mlp);

    let mut fresh = KVCache::new();
    let actual = block.forward(&x, &mut fresh, None);

    assert!(
        max_abs_diff(&expected, &actual) < 1e-5,
        "the sequential block must chain attention into the MLP norm"
    );
}

#[test]
fn the_two_residual_layouts_are_not_interchangeable() {
    // Both produce identically shaped output from identical weights, so only the
    // values separate them. Running the wrong one is a silent quality
    // regression, not a crash.
    let parallel_args = tiny_args();
    let mut sequential_args = tiny_args();
    sequential_args.use_parallel_residual = false;

    let weights = hf_weights(&parallel_args);
    let parallel =
        GptNeoxModel::from_weights(&weights, &parallel_args).expect("parallel model builds");
    let sequential =
        GptNeoxModel::from_weights(&weights, &sequential_args).expect("sequential model builds");

    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let mut parallel_caches = parallel.make_caches();
    let mut sequential_caches = sequential.make_caches();
    let parallel_logits = parallel.forward(&prompt, &mut parallel_caches, None);
    let sequential_logits = sequential.forward(&prompt, &mut sequential_caches, None);

    assert_eq!(
        mlxcel_core::array_shape(&parallel_logits),
        mlxcel_core::array_shape(&sequential_logits)
    );
    assert!(
        max_abs_diff(&parallel_logits, &sequential_logits) > 1e-4,
        "the parallel and sequential residual layouts must not produce the same logits"
    );
}

// Construction and forward.

#[test]
fn from_weights_rejects_an_indivisible_head_split() {
    let mut args = tiny_args();
    args.num_attention_heads = 3; // 8 is not divisible by 3
    let weights = hf_weights(&tiny_args());

    // `GptNeoxModel` is not `Debug`, so `expect_err` is not available here.
    let err = match GptNeoxModel::from_weights(&weights, &args) {
        Ok(_) => panic!("an indivisible hidden_size / num_attention_heads split must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("divisible"), "unhelpful error: {err}");
}

#[test]
fn tiny_model_prefills_then_decodes() {
    use mlxcel_core::generate::LanguageModel;

    for parallel in [true, false] {
        let mut args = tiny_args();
        args.use_parallel_residual = parallel;
        let weights = hf_weights(&args);
        let model = GptNeoxModel::from_weights(&weights, &args).expect("model builds");

        assert_eq!(LanguageModel::num_layers(&model), args.num_hidden_layers);
        let mut caches = LanguageModel::make_caches(&model);
        assert_eq!(caches.len(), args.num_hidden_layers);
        assert_eq!(caches[0].offset, 0);

        // Prefill: 4 tokens at positions 0..4.
        let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);
        let logits = LanguageModel::forward(&model, &prompt, &mut caches, None);
        assert_eq!(
            mlxcel_core::array_shape(&logits),
            vec![1, 4, args.vocab_size as i32]
        );
        assert_eq!(caches[0].offset, 4, "prefill must advance the KV cache");

        // GPT-NeoX has no grouped-query attention: the fused projection is
        // reshaped to `(num_heads, 3 * head_dim)`, so K and V always carry one
        // head per query head.
        let keys = caches[0]
            .keys
            .as_ref()
            .expect("prefill populates the cache");
        let key_shape = mlxcel_core::array_shape(keys);
        assert_eq!(key_shape.len(), 4);
        assert_eq!(key_shape[0], 1, "batch");
        assert_eq!(key_shape[1], args.num_attention_heads as i32);
        assert_eq!(key_shape[3], args.head_dim() as i32);

        // Decode: one token, rotated at offset 4 and attending over the cache.
        let next = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
        let logits = LanguageModel::forward(&model, &next, &mut caches, None);
        assert_eq!(
            mlxcel_core::array_shape(&logits),
            vec![1, 1, args.vocab_size as i32]
        );
        assert_eq!(caches[0].offset, 5);

        assert_eq!(LanguageModel::eos_token_ids(&model), vec![9]);
    }
}
