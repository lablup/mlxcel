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

//! Unit tests for `youtu_vl_lm`.
//!
//! Avoid coupling to a checkpoint on disk so the tests can run anywhere
//! `mlxcel_core` does (Linux/CUDA CI included).

use super::*;

fn minimal_config() -> YoutuTextConfig {
    YoutuTextConfig {
        model_type: "youtu_vl".to_string(),
        vocab_size: 32,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(4),
        kv_lora_rank: 16,
        q_lora_rank: Some(32),
        qk_rope_head_dim: 8,
        v_head_dim: 16,
        qk_nope_head_dim: 16,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 500_000.0,
        rope_scaling: None,
        rope_traditional: true,
        rope_interleave: true,
        tie_word_embeddings: true,
        attention_bias: false,
        mlp_bias: false,
        n_shared_experts: None,
        n_routed_experts: None,
        moe_intermediate_size: None,
        num_experts_per_tok: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
        norm_topk_prob: true,
        moe_layer_freq: 1,
        first_k_dense_replace: 0,
        quantization: None,
    }
}

#[test]
fn config_defaults_match_upstream() {
    // Round-trip through serde_json with only the required fields set —
    // mirrors what the loader sees on a real `config.json`.
    let raw = serde_json::json!({
        "model_type": "youtu_vl",
        "vocab_size": 283386,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "qk_nope_head_dim": 128,
    });
    let config: YoutuTextConfig = serde_json::from_value(raw).unwrap();

    assert!(config.tie_word_embeddings);
    assert!(config.rope_traditional);
    assert!(config.rope_interleave);
    assert_eq!(config.rope_theta, 500_000.0);
    assert_eq!(config.max_position_embeddings, 32_768);
    assert!(config.n_routed_experts.is_none());
}

#[test]
fn sanitize_decomposes_kv_b_proj_per_head() {
    let config = minimal_config();
    let mut weights = WeightMap::new();

    // Build a fake non-quantized kv_b_proj weight per layer.
    let num_heads = config.num_attention_heads;
    let head_dim = config.qk_nope_head_dim + config.v_head_dim;
    let kv_lora_rank = config.kv_lora_rank;
    let total = num_heads * head_dim * kv_lora_rank;

    for layer_idx in 0..config.num_hidden_layers {
        let key = format!("model.layers.{}.self_attn.kv_b_proj.weight", layer_idx);
        let arange = mlxcel_core::arange_f32(0.0, total as f32, 1.0);
        let reshaped = mlxcel_core::reshape(
            &arange,
            &[(num_heads * head_dim) as i32, kv_lora_rank as i32],
        );
        weights.insert(key, mlxcel_core::copy(&reshaped));
    }

    // Tied lm_head should be dropped after sanitization.
    weights.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::zeros(&[1, 1], mlxcel_core::dtype::FLOAT32),
    );

    let sanitized = sanitize_text_weights(weights, &config).unwrap();

    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);
        assert!(!sanitized.contains_key(&format!("{}.kv_b_proj.weight", prefix)));

        let embed_q_key = format!("{}.embed_q.weight", prefix);
        let unembed_out_key = format!("{}.unembed_out.weight", prefix);
        assert!(sanitized.contains_key(&embed_q_key));
        assert!(sanitized.contains_key(&unembed_out_key));

        // Shapes: embed_q = [H, kv_rank, qk_nope], unembed_out = [H, v_head, kv_rank]
        let eq_shape = mlxcel_core::array_shape(sanitized.get(&embed_q_key).unwrap());
        assert_eq!(
            eq_shape,
            vec![
                num_heads as i32,
                kv_lora_rank as i32,
                config.qk_nope_head_dim as i32
            ]
        );
        let uo_shape = mlxcel_core::array_shape(sanitized.get(&unembed_out_key).unwrap());
        assert_eq!(
            uo_shape,
            vec![
                num_heads as i32,
                config.v_head_dim as i32,
                kv_lora_rank as i32
            ]
        );
    }

    assert!(!sanitized.contains_key("lm_head.weight"));
}

#[test]
fn kv_b_proj_decompose_returns_error_when_biases_missing() {
    // A quantized kv_b_proj (scales present) with no biases tensor must
    // produce a clear error rather than panic (M1 hardening).
    let config = minimal_config();
    let mut weights = WeightMap::new();

    // Insert a plausible scales tensor but deliberately omit biases.
    let layer_idx = 0;
    let key = format!("model.layers.{layer_idx}.self_attn.kv_b_proj.weight");
    let scales_key = format!("model.layers.{layer_idx}.self_attn.kv_b_proj.scales");

    // Minimal weight tensor: shape doesn't matter for the biases-missing path.
    weights.insert(
        key,
        mlxcel_core::zeros(&[1, 1], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        scales_key,
        mlxcel_core::zeros(&[1, 1], mlxcel_core::dtype::FLOAT32),
    );
    // biases are intentionally absent

    let result = sanitize_text_weights(weights, &config);
    assert!(
        result.is_err(),
        "expected Err when biases are missing for a quantized kv_b_proj"
    );
    let msg = result.err().unwrap();
    assert!(
        msg.contains("biases"),
        "error message should mention 'biases'; got: {msg}"
    );
    assert!(
        msg.contains(&format!("layer {layer_idx}")),
        "error message should identify the layer; got: {msg}"
    );
}

/// The MLA `kv_b_proj` pair is solved from `kv_lora_rank` and two tensor axes
/// rather than declared, and every input is untrusted: `kv_lora_rank` comes from
/// `config.json` and the axes from the checkpoint. Before issue #958 the naive
/// arithmetic divided by both without checking them, so a `kv_lora_rank` of 0
/// panicked on integer division and a solved pair outside anything MLX can
/// describe reached `dequantize`, which crosses the cxx bridge as
/// `UniquePtr<MlxArray>` rather than `Result` and therefore aborts during weight
/// sanitization rather than failing the load.
///
/// This drives the real `sanitize_text_weights` rather than the shared helper
/// directly, so the guard is exercised where a checkpoint reaches it. The
/// assertions are on the returned `Result`, so a regression fails cleanly here
/// rather than aborting the test binary.
#[test]
fn quantized_kv_b_proj_rejects_a_kv_lora_rank_no_packing_can_describe() {
    // Honest affine 4-bit geometry for the positive control: the packed axis is
    // kv_lora_rank / 8 and the scales axis is kv_lora_rank / group_size, so
    // packed_in * 32 == bits * num_groups * group_size holds (2 * 32 == 4 * 1 * 16).
    let build = |config: &YoutuTextConfig| {
        let num_heads = config.num_attention_heads;
        let head_dim = config.qk_nope_head_dim + config.v_head_dim;
        let rows = (num_heads * head_dim) as i32;
        let mut weights = WeightMap::new();
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}.self_attn.kv_b_proj", layer_idx);
            // UINT32, not FLOAT32: `dequantize` rejects any other packed dtype
            // by throwing, and that throw is the uncatchable abort this guard
            // exists to keep out of the forward path. A float fixture here would
            // take the test binary down on the positive control.
            weights.insert(
                format!("{prefix}.weight"),
                mlxcel_core::zeros(&[rows, 2], mlxcel_core::dtype::UINT32),
            );
            weights.insert(
                format!("{prefix}.scales"),
                mlxcel_core::zeros(&[rows, 1], mlxcel_core::dtype::FLOAT32),
            );
            weights.insert(
                format!("{prefix}.biases"),
                mlxcel_core::zeros(&[rows, 1], mlxcel_core::dtype::FLOAT32),
            );
        }
        weights
    };

    // Positive control first, so a guard that rejected every quantized
    // kv_b_proj could not pass this test.
    let honest = minimal_config();
    let sanitized = match sanitize_text_weights(build(&honest), &honest) {
        Ok(w) => w,
        Err(e) => panic!("an honest quantized kv_b_proj must still sanitize: {e}"),
    };
    assert!(sanitized.contains_key("model.layers.0.self_attn.embed_q.weight"));

    // A zero `kv_lora_rank` is Rust integer division by zero on the very first
    // solve, so it has to be refused before the division rather than after.
    let mut zero_rank = minimal_config();
    zero_rank.kv_lora_rank = 0;
    let err = sanitize_text_weights(build(&zero_rank), &zero_rank)
        .err()
        .unwrap_or_else(|| panic!("kv_lora_rank 0 must be refused, not divided by"));
    assert!(err.contains("kv_lora_rank"), "unhelpful error: {err}");

    // A large `kv_lora_rank` truncates the solved bit width to 0, which is the
    // divisor MLX would then divide by.
    let mut wide_rank = minimal_config();
    wide_rank.kv_lora_rank = 4096;
    let err = sanitize_text_weights(build(&wide_rank), &wide_rank)
        .err()
        .unwrap_or_else(|| panic!("a solved bit width of 0 must be refused"));
    assert!(err.contains("bits"), "unhelpful error: {err}");

    // A non-quantized kv_b_proj carries no packing, so the pair is never solved
    // and a `kv_lora_rank` that no packing could describe must not gate it. The
    // tensor is built at `wide_rank`'s own width so it still satisfies the
    // separate shape cross-check below the solve, which is what would otherwise
    // reject this for an unrelated reason.
    let mut float_only = WeightMap::new();
    let rows = (wide_rank.num_attention_heads * (wide_rank.qk_nope_head_dim + wide_rank.v_head_dim))
        as i32;
    for layer_idx in 0..wide_rank.num_hidden_layers {
        float_only.insert(
            format!("model.layers.{}.self_attn.kv_b_proj.weight", layer_idx),
            mlxcel_core::zeros(
                &[rows, wide_rank.kv_lora_rank as i32],
                mlxcel_core::dtype::FLOAT32,
            ),
        );
    }
    if let Err(e) = sanitize_text_weights(float_only, &wide_rank) {
        panic!("a float kv_b_proj must not be gated on quantization params: {e}");
    }
}

// ---- Issue #1371: the text-only Youtu-LLM route --------------------------
//
// The text-only checkpoint is the same decoder without a vision tower, so the
// tests below pin the three config properties the VLM sibling does not
// exercise: an identity YaRN block, a `deepseek_v2` label, and the
// `rope_interleave` switch actually reaching the rope call.

/// The published text-only field set, taken from
/// `mlx-community/Youtu-LLM-2B-4bit`'s `config.json`. It labels itself
/// `deepseek_v2` for mlx-lm compatibility, omits `rope_traditional`, and
/// carries a YaRN block whose factor is 1.
fn text_only_config_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v2",
        "architectures": ["YoutuForCausalLM"],
        "vocab_size": 128256,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {"type": "yarn", "factor": 1.0, "mscale_all_dim": 0},
        "rope_theta": 1600000,
        "rope_interleave": true,
        "tie_word_embeddings": true,
        "attention_bias": false,
        "mlp_bias": false,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    })
}

/// The identity YaRN block must parse and must leave the attention scale at the
/// plain `(qk_nope + qk_rope) ** -0.5`. The vendor applies its mscale only when
/// `mscale_all_dim` is truthy, and `yarn_get_mscale` returns 1 at factor 1, so
/// anything other than the unscaled value here would be a silent divergence
/// from the reference decoder.
#[test]
fn text_only_config_with_identity_yarn_parses() {
    let config: YoutuTextConfig = serde_json::from_value(text_only_config_json()).unwrap();

    assert_eq!(config.q_lora_rank, Some(1536));
    assert_eq!(config.rope_theta, 1_600_000.0);
    assert!(config.tie_word_embeddings);
    // `rope_traditional` is absent from this checkpoint and defaults to true,
    // so the interleaved layout is selected by `rope_interleave` alone.
    assert!(config.rope_interleave);
    assert!(config.rope_is_interleaved());
    assert_eq!(config.group_size(), 64);
    assert_eq!(config.bits(), 4);

    let with_yarn = config.to_deepseek_v3_config().get_attention_scale();

    let mut plain: YoutuTextConfig = serde_json::from_value(text_only_config_json()).unwrap();
    plain.rope_scaling = None;
    let without_yarn = plain.to_deepseek_v3_config().get_attention_scale();

    let expected = ((config.qk_nope_head_dim + config.qk_rope_head_dim) as f32).powf(-0.5);
    assert_eq!(with_yarn, without_yarn);
    assert_eq!(with_yarn, expected);

    config
        .validate_rope_scaling()
        .expect("an identity yarn block must be accepted");
}

/// A YaRN block that actually interpolates is refused at load. The decoder
/// applies plain RoPE at `rope_theta` and has no frequency interpolation, so
/// accepting one would produce fluent but positionally wrong long-context
/// output rather than a visible failure.
#[test]
fn rope_scaling_that_interpolates_is_refused() {
    let mut raw = text_only_config_json();
    raw["rope_scaling"] = serde_json::json!({"type": "yarn", "factor": 4.0, "mscale_all_dim": 1});
    let config: YoutuTextConfig = serde_json::from_value(raw).unwrap();

    let err = config
        .validate_rope_scaling()
        .expect_err("a factor above 1 must be refused, not silently ignored");
    assert!(err.contains("factor"), "unhelpful error: {err}");
    assert!(err.contains("yarn"), "error should name the scaling: {err}");
}

/// `q_lora_rank: null` selects the direct `q_proj` branch of the shared MLA
/// attention. No published Youtu checkpoint sets it, but the vendor config
/// declares the field optional, so a missing or null value must parse rather
/// than fail the whole load.
#[test]
fn null_q_lora_rank_parses_and_selects_the_direct_q_projection() {
    let mut raw = text_only_config_json();
    raw["q_lora_rank"] = serde_json::Value::Null;
    let config: YoutuTextConfig = serde_json::from_value(raw).unwrap();
    assert_eq!(config.q_lora_rank, None);
    assert_eq!(config.to_deepseek_v3_config().q_lora_rank, None);

    let mut absent = text_only_config_json();
    absent.as_object_mut().unwrap().remove("q_lora_rank");
    let config: YoutuTextConfig = serde_json::from_value(absent).unwrap();
    assert_eq!(config.q_lora_rank, None);

    // A numeric rank still selects the LoRA chain.
    let config: YoutuTextConfig = serde_json::from_value(text_only_config_json()).unwrap();
    assert_eq!(config.to_deepseek_v3_config().q_lora_rank, Some(1536));
}

/// `rope_is_interleaved` folds the two spellings of the same switch: the
/// vendor's `rope_interleave` and the mlx-vlm port's `rope_traditional`. Both
/// default to true, and either one turned off selects the half-split form.
#[test]
fn rope_interleave_and_rope_traditional_are_the_same_switch() {
    let mut config = minimal_config();
    assert!(config.rope_is_interleaved());

    config.rope_interleave = false;
    assert!(!config.rope_is_interleaved());

    config.rope_interleave = true;
    config.rope_traditional = false;
    assert!(!config.rope_is_interleaved());
}

/// Build the complete unquantized weight set for a `minimal_config` model,
/// including the `kv_b_proj` that `sanitize_text_weights` decomposes. No
/// `lm_head` is inserted: the config ties word embeddings, which is what every
/// published Youtu checkpoint does.
fn synthetic_tied_weights(config: &YoutuTextConfig) -> WeightMap {
    fn ramp(n: usize) -> Vec<f32> {
        // Small alternating values keep the forward pass numerically tame while
        // still making every weight distinct.
        (0..n)
            .map(|i| ((i % 17) as f32 - 8.0) / 64.0)
            .collect::<Vec<f32>>()
    }
    fn insert(weights: &mut WeightMap, key: &str, shape: &[i32]) {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        weights.insert(
            key.to_string(),
            mlxcel_core::from_slice_f32(&ramp(n), shape),
        );
    }

    let hidden = config.hidden_size as i32;
    let heads = config.num_attention_heads as i32;
    let nope = config.qk_nope_head_dim as i32;
    let rope = config.qk_rope_head_dim as i32;
    let v_dim = config.v_head_dim as i32;
    let kv_lora = config.kv_lora_rank as i32;
    let q_lora = config.q_lora_rank.expect("minimal_config uses a q LoRA") as i32;
    let inter = config.intermediate_size as i32;

    let mut weights = WeightMap::new();
    insert(
        &mut weights,
        "model.embed_tokens.weight",
        &[config.vocab_size as i32, hidden],
    );
    insert(&mut weights, "model.norm.weight", &[hidden]);

    for layer in 0..config.num_hidden_layers {
        let attn = format!("model.layers.{layer}.self_attn");
        insert(
            &mut weights,
            &format!("{attn}.q_a_proj.weight"),
            &[q_lora, hidden],
        );
        insert(
            &mut weights,
            &format!("{attn}.q_a_layernorm.weight"),
            &[q_lora],
        );
        insert(
            &mut weights,
            &format!("{attn}.q_b_proj.weight"),
            &[heads * (nope + rope), q_lora],
        );
        insert(
            &mut weights,
            &format!("{attn}.kv_a_proj_with_mqa.weight"),
            &[kv_lora + rope, hidden],
        );
        insert(
            &mut weights,
            &format!("{attn}.kv_a_layernorm.weight"),
            &[kv_lora],
        );
        insert(
            &mut weights,
            &format!("{attn}.kv_b_proj.weight"),
            &[heads * (nope + v_dim), kv_lora],
        );
        insert(
            &mut weights,
            &format!("{attn}.o_proj.weight"),
            &[hidden, heads * v_dim],
        );

        let mlp = format!("model.layers.{layer}.mlp");
        insert(
            &mut weights,
            &format!("{mlp}.gate_proj.weight"),
            &[inter, hidden],
        );
        insert(
            &mut weights,
            &format!("{mlp}.up_proj.weight"),
            &[inter, hidden],
        );
        insert(
            &mut weights,
            &format!("{mlp}.down_proj.weight"),
            &[hidden, inter],
        );

        let layer_prefix = format!("model.layers.{layer}");
        insert(
            &mut weights,
            &format!("{layer_prefix}.input_layernorm.weight"),
            &[hidden],
        );
        insert(
            &mut weights,
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
            &[hidden],
        );
    }

    weights
}

fn build_tied_model(config: &YoutuTextConfig) -> YoutuLanguageModel {
    let weights = sanitize_text_weights(synthetic_tied_weights(config), config)
        .expect("sanitize must succeed");
    YoutuLanguageModel::from_weights(&weights, config).expect("from_weights must succeed")
}

/// A tied checkpoint carries no `lm_head`, so the head has to come from the
/// embedding table. This is the whole-model shape of the text-only route:
/// build, forward, and land on `[batch, seq, vocab]` with finite logits.
#[test]
fn tied_embeddings_produce_logits_without_an_lm_head() {
    let config = minimal_config();
    assert!(config.tie_word_embeddings);
    let model = build_tied_model(&config);
    assert!(
        model.lm_head.is_none(),
        "a tied checkpoint must not build a separate lm_head"
    );

    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);
    let mut caches = model.make_caches_impl();
    let logits = model.forward_impl(&ids, None, &mut caches, None);

    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 4, config.vocab_size as i32]
    );
    let values = mlxcel_core::utils::array_to_vec_f32(&logits);
    assert!(
        values.iter().all(|v| v.is_finite()),
        "tied-head logits must be finite"
    );
}

/// The interleaved switch has to reach the rope call, not just parse. Before
/// issue #1371 the shared MLA attention hardcoded the interleaved layout, so a
/// checkpoint declaring `rope_interleave: false` would have been rotated the
/// wrong way with no error and no visible symptom beyond wrong text.
#[test]
fn rope_interleave_reaches_the_rope_call() {
    let interleaved = minimal_config();
    let mut half_split = minimal_config();
    half_split.rope_interleave = false;

    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);

    let model = build_tied_model(&interleaved);
    let mut caches = model.make_caches_impl();
    let a =
        mlxcel_core::utils::array_to_vec_f32(&model.forward_impl(&ids, None, &mut caches, None));

    let model = build_tied_model(&half_split);
    let mut caches = model.make_caches_impl();
    let b =
        mlxcel_core::utils::array_to_vec_f32(&model.forward_impl(&ids, None, &mut caches, None));

    assert!(a.iter().all(|v| v.is_finite()) && b.iter().all(|v| v.is_finite()));
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-5),
        "flipping rope_interleave must change the logits; the flag is being ignored"
    );
}
