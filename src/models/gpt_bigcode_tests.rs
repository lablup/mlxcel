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

//! Unit tests for the GPT-BigCode loader, its Multi-Query Attention split and
//! its `nn.Linear` (not `Conv1D`) weight layout.
//!
//! Everything here is checkpoint-free: the config tests parse the real
//! `bigcode/gpt_bigcode-santacoder` `config.json` field set, and the layout
//! tests build synthetic weight maps whose tensor names and shapes mirror the
//! HuggingFace export. The forward tests build tiny single-layer models and run
//! them on the default device.

use super::{GptBigCodeModel, ModelArgs, TokenIdField, detect_prefix, load_linear};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

// Config surface.

/// The `bigcode/gpt_bigcode-santacoder` config, field-for-field.
const SANTACODER_CONFIG: &str = r#"{
    "activation_function": "gelu_pytorch_tanh",
    "architectures": ["GPTBigCodeForCausalLM"],
    "attention_softmax_in_fp32": true,
    "multi_query": true,
    "attn_pdrop": 0.1,
    "bos_token_id": 49152,
    "embd_pdrop": 0.1,
    "eos_token_id": 49152,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt_bigcode",
    "n_embd": 2048,
    "n_head": 16,
    "n_inner": 8192,
    "n_layer": 24,
    "n_positions": 2048,
    "resid_pdrop": 0.1,
    "runner_max_sequence_length": null,
    "scale_attention_softmax_in_fp32": true,
    "scale_attn_weights": true,
    "summary_activation": null,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": true,
    "summary_type": "cls_index",
    "summary_use_proj": true,
    "transformers_version": "4.28.0.dev0",
    "use_cache": true,
    "vocab_size": 49280
}"#;

#[test]
fn parses_the_real_santacoder_config() {
    let args: ModelArgs =
        serde_json::from_str(SANTACODER_CONFIG).expect("santacoder config parses");

    assert_eq!(args.model_type, "gpt_bigcode");
    assert_eq!(args.n_embd, 2048);
    assert_eq!(args.n_head, 16);
    assert_eq!(args.n_layer, 24);
    assert_eq!(args.n_inner, Some(8192));
    assert_eq!(args.n_positions, 2048);
    assert_eq!(args.vocab_size, 49280);
    assert!(args.multi_query);
    assert!((args.layer_norm_epsilon - 1e-5).abs() < 1e-12);

    // No `quantization` block in the raw HuggingFace export.
    assert!(args.quantization.is_none());
    assert_eq!(args.eos_token_ids(), vec![49152]);
    assert!(args.validate().is_ok());
}

#[test]
fn config_defaults_cover_a_bare_gpt_bigcode_config() {
    let args: ModelArgs = serde_json::from_str(r#"{"model_type": "gpt_bigcode"}"#).expect("parses");

    assert_eq!(args.n_embd, 2048);
    assert_eq!(args.n_head, 16);
    assert_eq!(args.n_layer, 24);
    assert_eq!(args.n_positions, 2048);
    assert_eq!(args.vocab_size, 49280);
    // Multi-query and a tied head are the family defaults.
    assert!(args.multi_query);
    assert!(args.tie_word_embeddings);
    // No config-declared stop token: guessing one could truncate at an
    // ordinary token, since GPT-BigCode has no family-wide `<|endoftext|>` id.
    assert!(args.eos_token_ids().is_empty());
}

#[test]
fn token_id_fields_accept_a_scalar_or_a_list() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_bigcode", "eos_token_id": [0, 49152], "bos_token_id": [0]}"#,
    )
    .expect("parses");

    assert!(matches!(args.eos_token_id, Some(TokenIdField::Multiple(_))));
    assert_eq!(args.eos_token_ids(), vec![0, 49152]);

    // `bos_token_id` is the fallback when the config omits `eos_token_id`.
    let args: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt_bigcode", "bos_token_id": 49152}"#)
            .expect("parses");
    assert_eq!(args.eos_token_ids(), vec![49152]);

    // An empty list is not a stop token, so the fallback still applies.
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_bigcode", "eos_token_id": [], "bos_token_id": 7}"#,
    )
    .expect("parses");
    assert_eq!(args.eos_token_ids(), vec![7]);
}

// Multi-Query Attention: the split that defines the family.

#[test]
fn multi_query_gives_one_kv_head_and_an_uneven_qkv_split() {
    let args: ModelArgs = serde_json::from_str(SANTACODER_CONFIG).expect("parses");

    assert_eq!(
        args.n_kv_heads(),
        1,
        "multi_query means exactly one KV head"
    );
    assert_eq!(args.head_dim(), 128, "2048 / 16");
    assert_eq!(args.kv_dim(), 128, "one KV head of width 128");

    // 2048 + 2 * 128, which is what the checkpoint actually ships:
    // `transformer.h.0.attn.c_attn.weight` is [2304, 2048].
    assert_eq!(args.c_attn_out_features(), 2304);
    assert_ne!(
        args.c_attn_out_features(),
        3 * args.n_embd,
        "MQA must not widen c_attn to 3 * n_embd"
    );

    // Q is [0, 2048), K is [2048, 2176), V is [2176, 2304). Splitting the
    // projection into three equal parts (the GPT-2 pattern) would put query
    // channels into K and V and still produce correctly shaped garbage.
    assert_eq!(args.qkv_split_offsets(), (2048, 2176));
    let even_split = args.c_attn_out_features() / 3;
    assert_ne!(args.qkv_split_offsets().0, even_split);
    assert_ne!(args.qkv_split_offsets().1, 2 * even_split);
}

#[test]
fn multi_query_false_degenerates_to_multi_head_attention() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "gpt_bigcode", "n_embd": 2048, "n_head": 16, "multi_query": false}"#,
    )
    .expect("parses");

    assert_eq!(args.n_kv_heads(), 16);
    assert_eq!(args.kv_dim(), 2048);
    // With one KV head per query head the fused projection is the plain
    // three-way GPT-2 shape again, and the offsets fall on the even thirds.
    assert_eq!(args.c_attn_out_features(), 3 * 2048);
    assert_eq!(args.qkv_split_offsets(), (2048, 4096));
}

#[test]
fn intermediate_size_comes_from_n_inner_not_four_times_n_embd() {
    // Santacoder's `n_inner` happens to be 4 * n_embd, so use a config where
    // the two genuinely disagree.
    let args: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt_bigcode", "n_embd": 2048, "n_inner": 5632}"#)
            .expect("parses");
    assert_eq!(args.intermediate_size(), 5632);

    // Absent and explicit-null both fall back to 4 * n_embd, matching what
    // HuggingFace does with the same field.
    let args: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt_bigcode", "n_embd": 2048}"#).expect("parses");
    assert_eq!(args.intermediate_size(), 8192);
    let args: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt_bigcode", "n_embd": 2048, "n_inner": null}"#)
            .expect("parses");
    assert_eq!(args.intermediate_size(), 8192);
}

// Config validation. `config.json` arrives from the model directory, which for
// `mlxcel generate -m <org>/<repo>` is a third-party HuggingFace repo the
// download layer never parses.

#[test]
fn config_validation_rejects_hostile_scalars() {
    let cases: [(&str, &str); 9] = [
        // Divides by zero in `head_dim()`. `0.is_multiple_of(0)` is true, so the
        // divisibility check alone would let this through.
        (r#"{"model_type": "gpt_bigcode", "n_head": 0}"#, "n_head"),
        (
            r#"{"model_type": "gpt_bigcode", "n_embd": 0, "n_head": 0}"#,
            "n_head",
        ),
        (r#"{"model_type": "gpt_bigcode", "n_embd": 0}"#, "n_embd"),
        (
            r#"{"model_type": "gpt_bigcode", "n_embd": 2050}"#,
            "divisible",
        ),
        (r#"{"model_type": "gpt_bigcode", "n_layer": 0}"#, "n_layer"),
        // Sizes the `Vec::with_capacity` in `from_weights`.
        (
            r#"{"model_type": "gpt_bigcode", "n_layer": 18446744073709551615}"#,
            "n_layer",
        ),
        // 2^32: `(n_positions - 1) as i32` truncates to -1, and `clamp(0, -1)`
        // panics.
        (
            r#"{"model_type": "gpt_bigcode", "n_positions": 4294967296}"#,
            "n_positions",
        ),
        (
            r#"{"model_type": "gpt_bigcode", "vocab_size": 0}"#,
            "vocab_size",
        ),
        // Reaches the `as i32` cast that sizes the MLP projections.
        (
            r#"{"model_type": "gpt_bigcode", "n_inner": 18446744073709551615}"#,
            "n_inner",
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

    // The `4 * n_embd` fallback must not overflow into a valid-looking width
    // when `n_embd` itself is at the ceiling.
    let args: ModelArgs =
        serde_json::from_str(r#"{"model_type": "gpt_bigcode", "n_embd": 65536, "n_head": 16}"#)
            .expect("parses");
    assert_eq!(args.intermediate_size(), 262_144);
    assert!(args.validate().is_ok());
}

// Synthetic checkpoints.

fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
            "model_type": "gpt_bigcode",
            "n_embd": 8,
            "n_head": 2,
            "n_inner": 16,
            "n_layer": 1,
            "n_positions": 16,
            "vocab_size": 10,
            "layer_norm_epsilon": 1e-5,
            "multi_query": true,
            "eos_token_id": 9
        }"#,
    )
    .expect("tiny config parses")
}

/// Same shape as [`tiny_args`] but with two layers, so a checkpoint whose
/// layers disagree with each other can be built.
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

/// A HuggingFace GPT-BigCode export: `transformer.`-prefixed keys and
/// `nn.Linear` `[out, in]` weights throughout. No `h.N.attn.bias` mask buffer,
/// because HuggingFace registers GPT-BigCode's causal mask non-persistently.
fn hf_weights(args: &ModelArgs) -> WeightMap {
    let h = args.n_embd as i32;
    let ff = args.intermediate_size() as i32;
    let qkv = args.c_attn_out_features() as i32;

    let mut w = WeightMap::new();
    w.insert(
        "transformer.wte.weight".into(),
        filled(&[args.vocab_size as i32, h]),
    );
    w.insert(
        "transformer.wpe.weight".into(),
        filled(&[args.n_positions as i32, h]),
    );
    w.insert("transformer.ln_f.weight".into(), ones(&[h]));
    w.insert("transformer.ln_f.bias".into(), zeros(&[h]));

    for i in 0..args.n_layer {
        let p = format!("transformer.h.{i}");
        w.insert(format!("{p}.ln_1.weight"), ones(&[h]));
        w.insert(format!("{p}.ln_1.bias"), zeros(&[h]));
        w.insert(format!("{p}.ln_2.weight"), ones(&[h]));
        w.insert(format!("{p}.ln_2.bias"), zeros(&[h]));
        w.insert(format!("{p}.attn.c_attn.weight"), filled(&[qkv, h]));
        w.insert(format!("{p}.attn.c_attn.bias"), filled(&[qkv]));
        w.insert(format!("{p}.attn.c_proj.weight"), filled(&[h, h]));
        w.insert(format!("{p}.attn.c_proj.bias"), filled(&[h]));
        w.insert(format!("{p}.mlp.c_fc.weight"), filled(&[ff, h]));
        w.insert(format!("{p}.mlp.c_fc.bias"), filled(&[ff]));
        w.insert(format!("{p}.mlp.c_proj.weight"), filled(&[h, ff]));
        w.insert(format!("{p}.mlp.c_proj.bias"), filled(&[h]));
    }
    w
}

// Key layout.

#[test]
fn detects_the_transformer_prefix() {
    let args = tiny_args();
    let weights = hf_weights(&args);
    assert_eq!(
        detect_prefix(&weights).expect("prefix detected"),
        "transformer."
    );
}

#[test]
fn detection_rejects_a_weight_map_without_the_transformer_prefix() {
    // A bare-keyed map is not a GPT-BigCode export: both HuggingFace and
    // upstream mlx-lm nest the decoder under a `transformer` submodule, and
    // upstream ships no `sanitize` that could flatten it.
    let args = tiny_args();
    let mut weights = WeightMap::new();
    for (key, value) in hf_weights(&args) {
        weights.insert(key.trim_start_matches("transformer.").to_string(), value);
    }

    let err = detect_prefix(&weights).expect_err("must not guess a layout");
    assert!(
        err.contains("transformer.wte.weight"),
        "unhelpful error: {err}"
    );
}

// The Conv1D correction: GPT-BigCode weights are already [out, in].

#[test]
fn projections_load_without_a_transpose() {
    let args = tiny_args();
    let weights = hf_weights(&args);
    let model = GptBigCodeModel::from_weights(&weights, &args).expect("model builds");

    let h = args.n_embd as i32;
    let ff = args.intermediate_size() as i32;
    let qkv = args.c_attn_out_features() as i32;

    // Each loaded weight must have exactly the shape the checkpoint stored.
    // GPT-BigCode uses `nn.Linear`, so the stored orientation is already the
    // one `mlxcel_core::layers::Linear` wants; a transpose anywhere here would
    // silently corrupt the projection.
    let checks: [(&UnifiedLinear, [i32; 2], i32); 4] = [
        (&model.h[0].attn.c_attn, [qkv, h], qkv),
        (&model.h[0].attn.c_proj, [h, h], h),
        (&model.h[0].mlp.c_fc, [ff, h], ff),
        (&model.h[0].mlp.c_proj, [h, ff], h),
    ];

    for (linear, expected_weight, expected_bias) in checks {
        let UnifiedLinear::Regular(inner) = linear else {
            panic!("an unquantized GPT-BigCode checkpoint must load as a regular Linear");
        };
        assert_eq!(
            mlxcel_core::array_shape(&inner.weight),
            expected_weight.to_vec(),
            "the stored [out, in] weight must reach the graph unchanged"
        );
        let bias = inner
            .bias
            .as_ref()
            .expect("GPT-BigCode ships a bias on every projection");
        assert_eq!(mlxcel_core::array_shape(bias), vec![expected_bias]);
    }

    // c_attn is [dim + 2 * kv_dim, dim], never [3 * dim, dim]. Here head_dim is
    // 8 / 2 = 4 and one shared KV head makes kv_dim 4, so the fused output is
    // 8 + 2 * 4 = 16 features, not 24.
    let UnifiedLinear::Regular(c_attn) = &model.h[0].attn.c_attn else {
        panic!("expected a regular Linear");
    };
    assert_eq!(mlxcel_core::array_shape(&c_attn.weight), vec![16, 8]);
    assert_ne!(mlxcel_core::array_shape(&c_attn.weight), vec![24, 8]);
}

#[test]
fn a_conv1d_oriented_weight_is_rejected_and_named() {
    let args = tiny_args();
    let h = args.n_embd as i32;
    let qkv = args.c_attn_out_features() as i32;

    // The `[in, out]` orientation GPT-2 stores. Accepting it here (or worse,
    // transposing it as the GPT-2 loader does) would build a model that loads
    // and generates fluent-looking output from corrupted projections.
    let mut weights = hf_weights(&args);
    weights.insert(
        "transformer.h.0.attn.c_attn.weight".into(),
        filled(&[h, qkv]),
    );

    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a Conv1D-oriented weight must be rejected, not transposed"),
        Err(err) => err,
    };
    assert!(
        err.contains("transformer.h.0.attn.c_attn.weight"),
        "the error must name the offending tensor: {err}"
    );
    assert!(
        err.contains("Conv1D") && err.contains("nn.Linear"),
        "the error must explain why GPT-BigCode is not Conv1D: {err}"
    );
}

#[test]
fn load_linear_rejects_a_degenerate_shape_or_a_missized_bias() {
    let args = tiny_args();
    let h = args.n_embd as i32;

    // Rank 3 instead of rank 2. Nothing else would notice until the matmul, and
    // an MLX C++ exception crossing the cxx bridge is an uncatchable
    // `std::terminate` rather than a load error.
    let mut weights = hf_weights(&args);
    weights.insert(
        "transformer.h.0.mlp.c_fc.weight".into(),
        filled(&[1, args.intermediate_size() as i32, h]),
    );
    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a rank-3 projection weight must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("c_fc.weight"), "unhelpful error: {err}");

    // A bias that is not the width of its projection output.
    let mut weights = hf_weights(&args);
    weights.insert("transformer.h.0.attn.c_proj.bias".into(), filled(&[h + 1]));
    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a mis-sized projection bias must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("c_proj.bias"), "unhelpful error: {err}");

    // A missing weight names the key it looked for.
    let mut weights = hf_weights(&args);
    weights.remove("transformer.h.0.attn.c_proj.weight");
    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a missing projection must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("c_proj.weight"), "unhelpful error: {err}");
}

#[test]
fn every_layer_is_checked_not_just_layer_zero() {
    // The shape contract comes from the config, so it holds for every layer.
    // A checkpoint whose later layers disagree must be rejected by name rather
    // than reaching a matmul.
    let args = two_layer_args();
    let mut weights = hf_weights(&args);
    weights.insert(
        "transformer.h.1.attn.c_attn.weight".into(),
        filled(&[args.n_embd as i32, args.c_attn_out_features() as i32]),
    );

    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a layer that disagrees with the config must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.contains("transformer.h.1.attn.c_attn.weight"),
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
    weights.insert("q.weight".into(), zeros(&[10, 1]));
    weights.insert("q.scales".into(), ones(&[10, 1]));
    weights.insert("q.biases".into(), zeros(&[10, 1]));

    let loaded = load_linear(&weights, "q", 8, 10, 8, 4).expect("a quantized projection loads");
    assert!(
        matches!(loaded, UnifiedLinear::Quantized { .. }),
        "the packed path must not be routed through the float input-width check"
    );

    // The row count is untouched by packing, so the output width still has to
    // be the one the config implies. `Attention::forward` slices the fused
    // `c_attn` at config-derived offsets, so a packed projection wider than the
    // config claims keeps every slice in bounds and silently hands K and V the
    // wrong channels; a narrower one makes `slice` clamp and the following
    // `reshape` throw, which crosses the cxx bridge as `std::terminate`.
    for rows in [12, 8] {
        let mut weights = WeightMap::new();
        weights.insert("q.weight".into(), zeros(&[rows, 1]));
        weights.insert("q.scales".into(), ones(&[rows, 1]));
        weights.insert("q.biases".into(), zeros(&[rows, 1]));

        let err = match load_linear(&weights, "q", 8, 10, 8, 4) {
            Ok(_) => panic!("a packed weight of output width {rows} must not load as width 10"),
            Err(err) => err,
        };
        assert!(err.contains("q.weight"), "unhelpful error: {err}");
    }

    // A quantized projection's bias is a plain 1-D vector, so it is checked on
    // the packed path too.
    let mut weights = WeightMap::new();
    weights.insert("q.weight".into(), zeros(&[10, 1]));
    weights.insert("q.scales".into(), ones(&[10, 1]));
    weights.insert("q.biases".into(), zeros(&[10, 1]));
    weights.insert("q.bias".into(), zeros(&[11]));

    let err = match load_linear(&weights, "q", 8, 10, 8, 4) {
        Ok(_) => panic!("a mis-sized bias must be rejected on the packed path too"),
        Err(err) => err,
    };
    assert!(err.contains("q.bias"), "unhelpful error: {err}");
}

// The learned position table and the token table versus the config.

#[test]
fn from_weights_rejects_a_config_that_overstates_a_table() {
    // `wpe` holds 2 rows while the config claims 16. Position ids are clamped
    // to 15, so an ordinary 8-token prompt gathers rows past the end of the
    // table. The gather wraps a negative index but does not range-check a
    // positive one, so those reads return whatever follows the table in the
    // buffer, and the values reach the logits.
    let args = tiny_args();
    let mut weights = hf_weights(&args);
    weights.insert(
        "transformer.wpe.weight".into(),
        filled(&[2, args.n_embd as i32]),
    );
    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a config that overstates the wpe table must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("n_positions"), "unhelpful error: {err}");

    // The same hazard on the token table: token ids are bounded by
    // `vocab_size`, not by the rows actually present.
    let mut weights = hf_weights(&args);
    weights.insert(
        "transformer.wte.weight".into(),
        filled(&[3, args.n_embd as i32]),
    );
    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a config that overstates the wte table must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("vocab_size"), "unhelpful error: {err}");

    // The other direction is safe: a padded table keeps the bound inside it.
    let mut weights = hf_weights(&args);
    weights.insert(
        "transformer.wpe.weight".into(),
        filled(&[64, args.n_embd as i32]),
    );
    assert!(GptBigCodeModel::from_weights(&weights, &args).is_ok());
}

#[test]
fn from_weights_rejects_tables_and_norms_of_the_wrong_width() {
    // A width mismatch only surfaces at the `add` against the token embeddings
    // (or inside the norm), and an MLX C++ exception crossing the cxx bridge is
    // an uncatchable `std::terminate`.
    let args = tiny_args();

    let mut weights = hf_weights(&args);
    weights.insert(
        "transformer.wpe.weight".into(),
        filled(&[args.n_positions as i32, 4]),
    );
    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a wpe width that disagrees with n_embd must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("n_embd"), "unhelpful error: {err}");

    let mut weights = hf_weights(&args);
    weights.insert("transformer.h.0.ln_2.weight".into(), ones(&[4]));
    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("a LayerNorm of the wrong width must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("ln_2.weight"), "unhelpful error: {err}");
}

// The output head.

#[test]
fn an_untied_config_requires_a_real_lm_head() {
    let mut args = tiny_args();
    args.tie_word_embeddings = false;

    // Falling back to the tied path would produce logits from the wrong matrix.
    let weights = hf_weights(&args);
    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("an untied config without an lm_head must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("lm_head.weight"), "unhelpful error: {err}");

    // With the tensor present the separate head is used.
    let mut weights = hf_weights(&args);
    weights.insert(
        "lm_head.weight".into(),
        filled(&[args.vocab_size as i32, args.n_embd as i32]),
    );
    let model = GptBigCodeModel::from_weights(&weights, &args).expect("model builds");
    assert!(model.lm_head.is_some());

    // The default config ties the head and ships no `lm_head` tensor.
    let tied = GptBigCodeModel::from_weights(&hf_weights(&tiny_args()), &tiny_args())
        .expect("model builds");
    assert!(tied.lm_head.is_none());
}

// Construction and forward.

#[test]
fn from_weights_rejects_an_indivisible_head_split() {
    let mut args = tiny_args();
    args.n_head = 3; // 8 is not divisible by 3
    let weights = hf_weights(&tiny_args());

    // `GptBigCodeModel` is not `Debug`, so `expect_err` is not available here.
    let err = match GptBigCodeModel::from_weights(&weights, &args) {
        Ok(_) => panic!("an indivisible n_embd / n_head split must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("divisible"), "unhelpful error: {err}");
}

#[test]
fn tiny_model_forward_caches_a_single_kv_head() {
    use mlxcel_core::generate::LanguageModel;

    let args = tiny_args();
    let weights = hf_weights(&args);
    let model = GptBigCodeModel::from_weights(&weights, &args).expect("model builds");

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

    // The whole point of MQA: the cache holds one KV head, not `n_head`. The
    // sequence axis is the cache's step-padded capacity rather than the live
    // length (`offset` above is the live length), so only the batch, head and
    // head_dim axes are asserted here.
    let keys = caches[0]
        .keys
        .as_ref()
        .expect("prefill populates the cache");
    let key_shape = mlxcel_core::array_shape(keys);
    assert_eq!(key_shape.len(), 4);
    assert_eq!(key_shape[0], 1, "batch");
    assert_eq!(
        key_shape[1], 1,
        "multi_query must cache exactly one KV head, not n_head"
    );
    assert_eq!(key_shape[3], args.head_dim() as i32);

    // Decode: one token, embedded at position 4 and attending over the cache.
    let next = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
    let logits = LanguageModel::forward(&model, &next, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 1, args.vocab_size as i32]
    );
    assert_eq!(caches[0].offset, 5);

    assert_eq!(LanguageModel::eos_token_ids(&model), vec![9]);
}

#[test]
fn tiny_multi_head_model_forward_caches_every_kv_head() {
    use mlxcel_core::generate::LanguageModel;

    // `multi_query: false` must keep working: it is the same graph with
    // `n_kv_heads == n_head`, and it exercises the even three-way c_attn split.
    let mut args = tiny_args();
    args.multi_query = false;
    let weights = hf_weights(&args);
    let model = GptBigCodeModel::from_weights(&weights, &args).expect("model builds");

    let mut caches = LanguageModel::make_caches(&model);
    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let logits = LanguageModel::forward(&model, &prompt, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 3, args.vocab_size as i32]
    );

    let keys = caches[0]
        .keys
        .as_ref()
        .expect("prefill populates the cache");
    let key_shape = mlxcel_core::array_shape(keys);
    assert_eq!(
        key_shape[1], args.n_head as i32,
        "multi_query: false must cache one KV head per query head"
    );
    assert_eq!(key_shape[3], args.head_dim() as i32);
}
