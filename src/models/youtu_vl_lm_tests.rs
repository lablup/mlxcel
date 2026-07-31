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
        q_lora_rank: 32,
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
