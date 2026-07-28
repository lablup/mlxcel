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

//! Unit tests for the GPT-2 loader and its two checkpoint-layout traps.
//!
//! Everything here is checkpoint-free: the config tests parse the real
//! `openai-community/gpt2` `config.json` field set, and the layout tests build
//! synthetic weight maps whose tensor names and shapes mirror both a raw
//! HuggingFace export and an MLX conversion. The tiny forward test builds an
//! 8-wide, single-layer model and runs it on the default device.

use super::{
    EosTokenId, GPT2_EOS_TOKEN_ID, Gpt2Layout, Gpt2Model, ModelArgs, exceeds_position_table,
    position_ids, strip_causal_mask_buffers,
};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

// Config surface.

/// The `openai-community/gpt2` config, field-for-field.
const GPT2_CONFIG: &str = r#"{
    "activation_function": "gelu_new",
    "architectures": ["GPT2LMHeadModel"],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "summary_activation": null,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": true,
    "summary_type": "cls_index",
    "summary_use_proj": true,
    "task_specific_params": {"text-generation": {"do_sample": true, "max_length": 50}},
    "vocab_size": 50257
}"#;

#[test]
fn parses_the_real_gpt2_config() {
    let args: ModelArgs = serde_json::from_str(GPT2_CONFIG).expect("gpt2 config parses");

    assert_eq!(args.model_type, "gpt2");
    assert_eq!(args.n_embd, 768);
    assert_eq!(args.n_head, 12);
    assert_eq!(args.n_layer, 12);
    assert_eq!(args.n_positions, 1024);
    assert_eq!(args.n_ctx, 1024);
    assert_eq!(args.vocab_size, 50257);
    assert!((args.layer_norm_epsilon - 1e-5).abs() < 1e-12);

    // Derived: 768 / 12 = 64, and the MLP is always 4x the model width.
    assert_eq!(args.head_dim(), 64);
    assert_eq!(args.intermediate_size(), 3072);

    // No `quantization` block in the raw HuggingFace export.
    assert!(args.quantization.is_none());
    assert_eq!(args.eos_token_ids(), vec![50256]);
}

#[test]
fn config_defaults_cover_a_bare_gpt2_config() {
    let args: ModelArgs = serde_json::from_str(r#"{"model_type": "gpt2"}"#).expect("parses");

    assert_eq!(args.n_embd, 768);
    assert_eq!(args.n_head, 12);
    assert_eq!(args.n_layer, 12);
    assert_eq!(args.n_positions, 1024);
    assert_eq!(args.vocab_size, 50257);
    assert_eq!(args.eos_token_ids(), vec![GPT2_EOS_TOKEN_ID]);
}

#[test]
fn eos_token_id_accepts_a_list() {
    let args: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt2", "eos_token_id": [50256, 50257]}"#)
            .expect("parses");

    assert!(matches!(args.eos_token_id, Some(EosTokenId::Multiple(_))));
    assert_eq!(args.eos_token_ids(), vec![50256, 50257]);
}

// Config validation. `config.json` arrives from the model directory, which for
// `mlxcel generate -m <org>/<repo>` is a third-party HuggingFace repo the
// download layer never parses.

#[test]
fn config_validation_rejects_hostile_scalars() {
    let cases: [(&str, &str); 7] = [
        // Divides by zero in `head_dim()`. `0.is_multiple_of(0)` is true, so the
        // divisibility check alone would let this through.
        (r#"{"model_type": "gpt2", "n_head": 0}"#, "n_head"),
        (
            r#"{"model_type": "gpt2", "n_embd": 0, "n_head": 0}"#,
            "n_head",
        ),
        (r#"{"model_type": "gpt2", "n_embd": 0}"#, "n_embd"),
        (r#"{"model_type": "gpt2", "n_embd": 770}"#, "divisible"),
        (r#"{"model_type": "gpt2", "n_layer": 0}"#, "n_layer"),
        // Sizes the `Vec::with_capacity` in `from_weights`, and used to drive an
        // unbounded probe loop in `strip_causal_mask_buffers`.
        (
            r#"{"model_type": "gpt2", "n_layer": 18446744073709551615}"#,
            "n_layer",
        ),
        // 2^32: `(n_positions - 1) as i32` truncates to -1, and `clamp(0, -1)`
        // panics.
        (
            r#"{"model_type": "gpt2", "n_positions": 4294967296}"#,
            "n_positions",
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

    // The real checkpoint config is untouched by any of the ceilings.
    let real: ModelArgs = serde_json::from_str(GPT2_CONFIG).expect("parses");
    assert!(real.validate().is_ok(), "the real gpt2 config must pass");
}

// Learned absolute positions.

#[test]
fn position_ids_start_at_zero_during_prefill() {
    assert_eq!(position_ids(0, 5, 1024), vec![0, 1, 2, 3, 4]);
}

#[test]
fn position_ids_continue_from_the_cache_offset_during_decode() {
    // A decode step after a 5-token prefill must embed position 5, not 0.
    assert_eq!(position_ids(5, 1, 1024), vec![5]);
    // A multi-token verify/chunk step after the same prefill continues too.
    assert_eq!(position_ids(5, 3, 1024), vec![5, 6, 7]);
    // Every step past the prefill keeps advancing.
    assert_eq!(position_ids(1023, 1, 1024), vec![1023]);
}

#[test]
fn position_ids_clamp_at_the_end_of_the_learned_table() {
    // GPT-2's `wpe` has exactly `n_positions` rows; a generation that runs
    // past the context must not index off the end of the table.
    assert_eq!(position_ids(1022, 4, 1024), vec![1022, 1023, 1023, 1023]);
    assert_eq!(position_ids(4096, 2, 1024), vec![1023, 1023]);
    assert_eq!(position_ids(0, 3, 1), vec![0, 0, 0]);
}

#[test]
fn position_ids_of_an_empty_step_are_empty() {
    assert!(position_ids(7, 0, 1024).is_empty());
}

#[test]
fn position_ids_saturate_instead_of_panicking_on_an_absurd_table_size() {
    // Both helpers are public, so neither may depend on its caller having run
    // `ModelArgs::validate`. `(1usize << 40) - 1` truncates to -1 as an i32, and
    // `clamp(0, -1)` panics; on a server worker thread that panic is a process
    // abort.
    assert_eq!(position_ids(0, 3, 1usize << 40), vec![0, 1, 2]);
    assert_eq!(position_ids(0, 2, usize::MAX), vec![0, 1]);
    assert_eq!(position_ids(0, 2, 1usize << 32), vec![0, 1]);
    assert!(!exceeds_position_table(0, 2, usize::MAX));
}

#[test]
fn position_table_overflow_is_reported_exactly_at_the_boundary() {
    // A full-window prefill and the last in-table decode step both fit.
    assert!(!exceeds_position_table(0, 1024, 1024));
    assert!(!exceeds_position_table(1023, 1, 1024));

    // One token past the last row, and a chunk that straddles the end.
    assert!(exceeds_position_table(1024, 1, 1024));
    assert!(exceeds_position_table(1020, 5, 1024));

    // An empty step never overflows, however far the cache has advanced.
    assert!(!exceeds_position_table(4096, 0, 1024));
}

// Checkpoint layout detection.

fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
            "model_type": "gpt2",
            "n_embd": 8,
            "n_head": 2,
            "n_layer": 1,
            "n_positions": 16,
            "n_ctx": 16,
            "vocab_size": 10,
            "layer_norm_epsilon": 1e-5
        }"#,
    )
    .expect("tiny config parses")
}

/// Same shape as [`tiny_args`] but with two layers, so a checkpoint whose layers
/// disagree with each other can be built.
fn two_layer_args() -> ModelArgs {
    let mut args = tiny_args();
    args.n_layer = 2;
    args
}

/// Deterministic non-zero filler so LayerNorm and softmax see real values.
fn filled(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}

fn zeros(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![0.0; n as usize], shape)
}

/// Raw HuggingFace GPT-2 export: no key prefix, Conv1D `[in, out]` weights,
/// and the `h.N.attn.bias` causal-mask buffer present.
fn raw_hf_weights(args: &ModelArgs, prefix: &str) -> WeightMap {
    let h = args.n_embd as i32;
    let ff = args.intermediate_size() as i32;
    let ctx = args.n_ctx as i32;

    let mut w = WeightMap::new();
    w.insert(
        format!("{prefix}wte.weight"),
        filled(&[args.vocab_size as i32, h]),
    );
    w.insert(
        format!("{prefix}wpe.weight"),
        filled(&[args.n_positions as i32, h]),
    );
    w.insert(format!("{prefix}ln_f.weight"), ones(&[h]));
    w.insert(format!("{prefix}ln_f.bias"), zeros(&[h]));

    for i in 0..args.n_layer {
        let p = format!("{prefix}h.{i}");
        w.insert(format!("{p}.ln_1.weight"), ones(&[h]));
        w.insert(format!("{p}.ln_1.bias"), zeros(&[h]));
        w.insert(format!("{p}.ln_2.weight"), ones(&[h]));
        w.insert(format!("{p}.ln_2.bias"), zeros(&[h]));
        // The registered causal-mask buffer, [1, 1, n_ctx, n_ctx].
        w.insert(format!("{p}.attn.bias"), zeros(&[1, 1, ctx, ctx]));
        w.insert(format!("{p}.attn.c_attn.weight"), filled(&[h, 3 * h]));
        w.insert(format!("{p}.attn.c_attn.bias"), filled(&[3 * h]));
        w.insert(format!("{p}.attn.c_proj.weight"), filled(&[h, h]));
        w.insert(format!("{p}.attn.c_proj.bias"), filled(&[h]));
        w.insert(format!("{p}.mlp.c_fc.weight"), filled(&[h, ff]));
        w.insert(format!("{p}.mlp.c_fc.bias"), filled(&[ff]));
        w.insert(format!("{p}.mlp.c_proj.weight"), filled(&[ff, h]));
        w.insert(format!("{p}.mlp.c_proj.bias"), filled(&[h]));
    }
    w
}

/// MLX conversion: `model.` prefix, weights already `[out, in]`, no mask
/// buffer (the reference `sanitize` deleted it).
fn mlx_converted_weights(args: &ModelArgs) -> WeightMap {
    let h = args.n_embd as i32;
    let ff = args.intermediate_size() as i32;

    let mut w = WeightMap::new();
    w.insert(
        "model.wte.weight".into(),
        filled(&[args.vocab_size as i32, h]),
    );
    w.insert(
        "model.wpe.weight".into(),
        filled(&[args.n_positions as i32, h]),
    );
    w.insert("model.ln_f.weight".into(), ones(&[h]));
    w.insert("model.ln_f.bias".into(), zeros(&[h]));

    for i in 0..args.n_layer {
        let p = format!("model.h.{i}");
        w.insert(format!("{p}.ln_1.weight"), ones(&[h]));
        w.insert(format!("{p}.ln_1.bias"), zeros(&[h]));
        w.insert(format!("{p}.ln_2.weight"), ones(&[h]));
        w.insert(format!("{p}.ln_2.bias"), zeros(&[h]));
        w.insert(format!("{p}.attn.c_attn.weight"), filled(&[3 * h, h]));
        w.insert(format!("{p}.attn.c_attn.bias"), filled(&[3 * h]));
        w.insert(format!("{p}.attn.c_proj.weight"), filled(&[h, h]));
        w.insert(format!("{p}.attn.c_proj.bias"), filled(&[h]));
        w.insert(format!("{p}.mlp.c_fc.weight"), filled(&[ff, h]));
        w.insert(format!("{p}.mlp.c_fc.bias"), filled(&[ff]));
        w.insert(format!("{p}.mlp.c_proj.weight"), filled(&[h, ff]));
        w.insert(format!("{p}.mlp.c_proj.bias"), filled(&[h]));
    }
    w
}

#[test]
fn detects_raw_huggingface_layout() {
    let args = tiny_args();
    let weights = raw_hf_weights(&args, "");
    let layout = Gpt2Layout::detect(&weights, &args).expect("layout detected");

    assert_eq!(layout.prefix, "");
    assert!(
        layout.conv1d,
        "a raw HuggingFace export stores c_attn as [n_embd, 3*n_embd] and needs the transpose"
    );
}

#[test]
fn detects_transformer_prefixed_layout() {
    let args = tiny_args();
    let weights = raw_hf_weights(&args, "transformer.");
    let layout = Gpt2Layout::detect(&weights, &args).expect("layout detected");

    assert_eq!(layout.prefix, "transformer.");
    assert!(layout.conv1d);
}

#[test]
fn detects_mlx_converted_layout() {
    let args = tiny_args();
    let weights = mlx_converted_weights(&args);
    let layout = Gpt2Layout::detect(&weights, &args).expect("layout detected");

    assert_eq!(layout.prefix, "model.");
    assert!(
        !layout.conv1d,
        "an MLX conversion is produced from an already-sanitized module tree"
    );
}

#[test]
fn detection_rejects_a_weight_map_without_wte() {
    let args = tiny_args();
    let mut weights = raw_hf_weights(&args, "");
    weights.remove("wte.weight");

    let err = Gpt2Layout::detect(&weights, &args).expect_err("must not guess a layout");
    assert!(err.contains("wte.weight"), "unhelpful error: {err}");
}

#[test]
fn detection_rejects_an_unexpected_c_attn_shape() {
    let args = tiny_args();
    let mut weights = raw_hf_weights(&args, "");
    weights.insert("h.0.attn.c_attn.weight".into(), zeros(&[8, 9]));

    let err = Gpt2Layout::detect(&weights, &args).expect_err("must not guess a layout");
    assert!(err.contains("c_attn"), "unhelpful error: {err}");
}

// Conv1D transpose: weights only, never biases.

#[test]
fn conv1d_transpose_applies_to_weights_and_not_to_biases() {
    let args = tiny_args();
    let weights = raw_hf_weights(&args, "");
    let layout = Gpt2Layout::detect(&weights, &args).expect("layout detected");
    let model = Gpt2Model::from_weights(&weights, &args).expect("model builds");
    let h = args.n_embd as i32;
    let ff = args.intermediate_size() as i32;

    let checks: [(&UnifiedLinear, [i32; 2], i32); 4] = [
        (&model.h[0].attn.c_attn, [3 * h, h], 3 * h),
        (&model.h[0].attn.c_proj, [h, h], h),
        (&model.h[0].mlp.c_fc, [ff, h], ff),
        (&model.h[0].mlp.c_proj, [h, ff], h),
    ];

    for (linear, expected_weight, expected_bias) in checks {
        let UnifiedLinear::Regular(inner) = linear else {
            panic!("an unquantized GPT-2 checkpoint must load as a regular Linear");
        };
        assert_eq!(
            mlxcel_core::array_shape(&inner.weight),
            expected_weight.to_vec(),
            "Conv1D weight must be transposed to [out, in]"
        );
        let bias = inner
            .bias
            .as_ref()
            .expect("GPT-2 ships a bias on every projection");
        assert_eq!(
            mlxcel_core::array_shape(bias),
            vec![expected_bias],
            "a 1-D bias must never be transposed"
        );
    }

    // An already-transposed MLX conversion must load with the same shapes.
    let converted = mlx_converted_weights(&args);
    let converted_model = Gpt2Model::from_weights(&converted, &args).expect("model builds");
    let UnifiedLinear::Regular(c_attn) = &converted_model.h[0].attn.c_attn else {
        panic!("expected a regular Linear");
    };
    assert_eq!(mlxcel_core::array_shape(&c_attn.weight), vec![3 * h, h]);

    // `layout` is consumed by the assertions above; keep it referenced so the
    // detection result stays part of this test's contract.
    assert!(layout.conv1d);
}

// The `h.N.attn.bias` causal-mask buffer.

#[test]
fn strips_causal_mask_buffers_without_touching_projection_biases() {
    let args = tiny_args();
    let mut weights = raw_hf_weights(&args, "");
    assert!(weights.contains_key("h.0.attn.bias"));

    let removed = strip_causal_mask_buffers(&mut weights);

    assert_eq!(removed, args.n_layer);
    assert!(
        !weights.contains_key("h.0.attn.bias"),
        "the [1, 1, n_ctx, n_ctx] mask buffer must be dropped"
    );
    assert!(
        weights.contains_key("h.0.attn.c_attn.bias"),
        "the fused QKV bias is a real bias and must survive"
    );
    assert!(weights.contains_key("h.0.attn.c_proj.bias"));
    assert!(weights.contains_key("h.0.mlp.c_fc.bias"));
    assert!(weights.contains_key("h.0.mlp.c_proj.bias"));
}

#[test]
fn causal_mask_stripping_matches_key_shape_not_the_config_layer_count() {
    // One pass over the weight map, not `4 * n_layer` probes: `n_layer` is
    // attacker-controlled and the probing form had no early exit, so a config
    // declaring billions of layers spun here before any weight was looked at.
    let mut weights = WeightMap::new();
    for key in [
        "h.0.attn.bias",
        "transformer.h.7.attn.bias",
        "model.transformer.h.11.attn.bias",
    ] {
        weights.insert(key.into(), zeros(&[1, 1, 4, 4]));
    }
    for key in [
        "h.0.attn.c_attn.bias", // a real projection bias
        "blah.0.attn.bias",     // contains "h." but not at a segment start
        "h.x.attn.bias",        // non-numeric layer index
        "h.0.attn.weight",      // not a bias at all
    ] {
        weights.insert(key.into(), zeros(&[4]));
    }

    assert_eq!(strip_causal_mask_buffers(&mut weights), 3);
    assert_eq!(weights.len(), 4);
    assert!(weights.contains_key("h.0.attn.c_attn.bias"));
    assert!(weights.contains_key("blah.0.attn.bias"));
    assert!(weights.contains_key("h.x.attn.bias"));
    assert!(weights.contains_key("h.0.attn.weight"));
}

#[test]
fn from_weights_rejects_a_layer_that_disagrees_with_layer_zero_about_conv1d() {
    // The layout is probed once, from `h.0.attn.c_attn.weight`. Here layer 0 is
    // Conv1D `[in, out]` so everything gets transposed, but layer 1 already
    // arrives as `[out, in]`; transposing it again yields a weight the matmul
    // cannot consume, and that failure lands in MLX C++ as an uncatchable
    // `std::terminate` rather than a load error.
    let args = two_layer_args();
    let mut weights = raw_hf_weights(&args, "");
    let h = args.n_embd as i32;
    weights.insert("h.1.attn.c_attn.weight".into(), filled(&[3 * h, h]));

    let err = match Gpt2Model::from_weights(&weights, &args) {
        Ok(_) => panic!("layers that disagree about the Conv1D layout must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.contains("h.1.attn.c_attn.weight"),
        "the error must name the offending tensor: {err}"
    );
}

#[test]
fn from_weights_rejects_a_degenerate_projection_shape() {
    let args = tiny_args();
    let h = args.n_embd as i32;

    // Rank 3 instead of rank 2: `transpose` reverses all axes without
    // complaining, so nothing else would notice until the matmul.
    let mut weights = raw_hf_weights(&args, "");
    weights.insert("h.0.mlp.c_fc.weight".into(), filled(&[1, h, 4 * h]));
    let err = match Gpt2Model::from_weights(&weights, &args) {
        Ok(_) => panic!("a rank-3 projection weight must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("c_fc.weight"), "unhelpful error: {err}");

    // A bias that is not the width of its projection output.
    let mut weights = raw_hf_weights(&args, "");
    weights.insert("h.0.attn.c_proj.bias".into(), filled(&[h + 1]));
    let err = match Gpt2Model::from_weights(&weights, &args) {
        Ok(_) => panic!("a mis-sized projection bias must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("c_proj.bias"), "unhelpful error: {err}");
}

#[test]
fn model_builds_when_the_mask_buffer_is_still_present() {
    // The graph never asks for `h.N.attn.bias`, so a caller that skips the
    // strip step still gets a correct model rather than a mis-loaded Linear.
    let args = tiny_args();
    let weights = raw_hf_weights(&args, "");
    let model = Gpt2Model::from_weights(&weights, &args).expect("model builds");

    let UnifiedLinear::Regular(c_proj) = &model.h[0].attn.c_proj else {
        panic!("expected a regular Linear");
    };
    let bias = c_proj.bias.as_ref().expect("c_proj ships a bias");
    assert_eq!(
        mlxcel_core::array_shape(bias),
        vec![args.n_embd as i32],
        "c_proj must take its own [n_embd] bias, not the [1, 1, n_ctx, n_ctx] mask buffer"
    );
}

// The learned position table versus the config that describes it.

#[test]
fn from_weights_rejects_a_config_that_overstates_the_position_table() {
    // `wpe` holds 2 rows while the config claims 16. `position_ids` clamps to
    // 15, so an ordinary 8-token prompt gathers rows 2..7 from past the end of
    // the table. The gather behind the lookup wraps a negative index but does
    // not range-check a positive one, so those reads return whatever follows the
    // table in the buffer, and the values reach the logits.
    let args = tiny_args();
    let mut weights = raw_hf_weights(&args, "");
    weights.insert("wpe.weight".into(), filled(&[2, args.n_embd as i32]));

    let err = match Gpt2Model::from_weights(&weights, &args) {
        Ok(_) => panic!("a config that overstates the wpe table must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("n_positions"), "unhelpful error: {err}");

    // The other direction is safe: the clamp stays inside a longer table.
    let mut weights = raw_hf_weights(&args, "");
    weights.insert("wpe.weight".into(), filled(&[64, args.n_embd as i32]));
    assert!(Gpt2Model::from_weights(&weights, &args).is_ok());
}

#[test]
fn from_weights_rejects_a_position_table_of_the_wrong_width() {
    // A width mismatch only surfaces at the `add` against the token embeddings,
    // and an MLX C++ exception crossing the cxx bridge is `std::terminate`.
    let args = tiny_args();
    let mut weights = raw_hf_weights(&args, "");
    weights.insert("wpe.weight".into(), filled(&[args.n_positions as i32, 4]));

    let err = match Gpt2Model::from_weights(&weights, &args) {
        Ok(_) => panic!("a wpe width that disagrees with n_embd must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("n_embd"), "unhelpful error: {err}");
}

/// Replace `wpe` with an affine-quantized table at 4 bits and group_size 8.
///
/// `groups` is how many quantization groups the scales claim, so the table
/// describes a `groups * 8`-wide input. The honest value for the 8-wide tiny
/// config is 1; anything else is a table packed for a different model width,
/// which keeps exactly the right row count.
fn quantize_position_table(args: &ModelArgs, weights: &mut WeightMap, groups: i32) {
    let rows = args.n_positions as i32;
    let packed_in = args.n_embd as i32 * 4 / 32; // 4-bit packs 8 values per uint32
    weights.insert("wpe.weight".into(), ones(&[rows, packed_in]));
    weights.insert("wpe.scales".into(), filled(&[rows, groups]));
    weights.insert("wpe.biases".into(), filled(&[rows, groups]));
}

#[test]
fn from_weights_rejects_a_quantized_position_table_of_the_wrong_dequantized_width() {
    // Packing compresses the input axis only, so a table built for a different
    // model width keeps exactly the right row count and the packed width alone
    // says nothing without a bit depth. The width check used to be skipped
    // outright for a quantized table, so the mismatch first surfaced as a
    // wrong-width hidden state inside `fast::layer_norm`, whose throw crosses
    // the cxx bridge as an uncatchable abort at the first forward pass.
    let mut args = tiny_args();
    args.quantization = Some(super::Quantization {
        group_size: 8,
        bits: 4,
    });

    // The positive control first, so the width check cannot be passing by
    // rejecting everything quantized.
    let mut weights = raw_hf_weights(&args, "");
    quantize_position_table(&args, &mut weights, 1);
    assert!(
        Gpt2Model::from_weights(&weights, &args).is_ok(),
        "a consistently packed position table must load"
    );

    let mut weights = raw_hf_weights(&args, "");
    quantize_position_table(&args, &mut weights, 2);
    let err = match Gpt2Model::from_weights(&weights, &args) {
        Ok(_) => panic!("a quantized wpe packed for a different width must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("input width"), "unhelpful error: {err}");
    assert!(
        err.contains("n_embd"),
        "the message must name the field: {err}"
    );
}

#[test]
fn from_weights_rejects_a_quantization_block_that_would_abort_an_mlx_kernel() {
    // GPT-2 carries no family-level `validate_quantization`, so this exercises
    // the shared guard in `reconcile_quantization_layout` rather than a local
    // copy: a `bits` of 0 divides by zero inside MLX's `validate_quantized_input`
    // and a non-positive `group_size` can match no real tensor, and either throw
    // crosses the cxx bridge as an uncatchable `std::terminate` at the first
    // forward pass rather than a load error.
    for (group_size, bits, field) in [
        (8, 0, "bits"),
        (8, -4, "bits"),
        (8, 33, "bits"),
        (0, 4, "group_size"),
        (-8, 4, "group_size"),
    ] {
        let mut args = tiny_args();
        args.quantization = Some(super::Quantization { group_size, bits });
        let mut weights = raw_hf_weights(&args, "");
        quantize_position_table(&args, &mut weights, 1);

        let err = match Gpt2Model::from_weights(&weights, &args) {
            Ok(_) => panic!("group_size {group_size} / bits {bits} must be rejected at load"),
            Err(err) => err,
        };
        assert!(err.contains(field), "unhelpful error: {err}");
    }
}

// Construction and forward.

#[test]
fn from_weights_rejects_an_indivisible_head_split() {
    let mut args = tiny_args();
    args.n_head = 3; // 8 is not divisible by 3
    let weights = raw_hf_weights(&tiny_args(), "");

    // `Gpt2Model` is not `Debug`, so `expect_err` is not available here.
    let err = match Gpt2Model::from_weights(&weights, &args) {
        Ok(_) => panic!("an indivisible n_embd / n_head split must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("divisible"), "unhelpful error: {err}");
}

#[test]
fn tiny_model_forward_threads_the_cache_offset() {
    use mlxcel_core::generate::LanguageModel;

    let args = tiny_args();
    let weights = raw_hf_weights(&args, "");
    let model = Gpt2Model::from_weights(&weights, &args).expect("model builds");

    assert_eq!(LanguageModel::num_layers(&model), args.n_layer);
    let mut caches = LanguageModel::make_caches(&model);
    assert_eq!(caches.len(), args.n_layer);
    assert_eq!(caches[0].offset, 0);

    // Prefill: 4 tokens at positions 0..4.
    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);
    let logits = LanguageModel::forward(&model, &prompt, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 4, args.vocab_size as i32]
    );
    assert_eq!(caches[0].offset, 4, "prefill must advance the KV cache");

    // Decode: one token, which must be embedded at position 4.
    assert_eq!(position_ids(caches[0].offset, 1, args.n_positions), vec![4]);
    let next = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
    let logits = LanguageModel::forward(&model, &next, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 1, args.vocab_size as i32]
    );
    assert_eq!(caches[0].offset, 5);

    assert_eq!(
        LanguageModel::eos_token_ids(&model),
        vec![GPT2_EOS_TOKEN_ID]
    );
}
