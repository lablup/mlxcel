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

//! Unit tests for the DeepSeek Sparse Attention (DSA) lightning indexer.

use super::ModelArgs;
use super::indexer::{Indexer, indexer_top_indices};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Minimal deepseek_v32 config that exercises the indexer with tiny dims.
/// Everything not listed falls back to the serde default, so
/// `indexer_rope_interleave` is `false` here (the deepseek_v32 default).
fn tiny_deepseek_config() -> ModelArgs {
    let json = r#"{
        "model_type": "deepseek_v32",
        "vocab_size": 16,
        "hidden_size": 6,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 8,
        "qk_rope_head_dim": 2,
        "index_topk": 2,
        "index_n_heads": 2,
        "index_head_dim": 4
    }"#;
    serde_json::from_str(json).expect("parse tiny deepseek config")
}

fn f32_weight(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// Build synthetic (non-quantized) indexer weights for `{attn_prefix}.indexer.*`.
fn synthetic_indexer_weights(args: &ModelArgs, attn_prefix: &str) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let q_lora = args.q_lora_rank as i32;
    let n_heads = args.index_n_heads as i32;
    let head_dim = args.index_head_dim as i32;

    let mut weights: WeightMap = WeightMap::new();
    let prefix = format!("{}.indexer", attn_prefix);
    // Linear weights are stored [out, in].
    weights.insert(
        format!("{}.wk.weight", prefix),
        f32_weight(&[head_dim, hidden]),
    );
    weights.insert(
        format!("{}.wq_b.weight", prefix),
        f32_weight(&[n_heads * head_dim, q_lora]),
    );
    weights.insert(
        format!("{}.weights_proj.weight", prefix),
        f32_weight(&[n_heads, hidden]),
    );
    weights.insert(format!("{}.k_norm.weight", prefix), f32_weight(&[head_dim]));
    weights.insert(format!("{}.k_norm.bias", prefix), f32_weight(&[head_dim]));
    weights
}

fn index_values(indices: &MlxArray, count: i32) -> Vec<i32> {
    let f = mlxcel_core::astype(indices, mlxcel_core::dtype::FLOAT32);
    (0..count)
        .map(|i| {
            let elem = mlxcel_core::slice(&f, &[0, 0, 0, i], &[1, 1, 1, i + 1]);
            mlxcel_core::item_f32(&elem).round() as i32
        })
        .collect()
}

#[test]
fn deepseek_v32_indexer_rope_defaults_to_non_interleaved() {
    // deepseek_v32: indexer RoPE is non-interleaved (traditional = false).
    let args = tiny_deepseek_config();
    assert!(
        !args.indexer_rope_interleave,
        "deepseek_v32 default indexer_rope_interleave must be false"
    );
}

#[test]
fn deepseek_v32_indexer_config_defaults() {
    // Absent explicit index_* fields, the indexer defaults match upstream.
    let json = r#"{
        "model_type": "deepseek_v32",
        "vocab_size": 16,
        "hidden_size": 6,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2
    }"#;
    let args: ModelArgs = serde_json::from_str(json).expect("parse defaults config");
    assert_eq!(args.index_topk, 2048);
    assert_eq!(args.index_n_heads, 64);
    assert_eq!(args.index_head_dim, 128);
    assert!(!args.indexer_rope_interleave);
}

#[test]
fn indexer_load_threads_rope_flag_per_model() {
    // The RoPE `traditional` flag must be threaded from config into the loaded
    // Indexer: false for deepseek_v32, true for the glm_moe_dsa mapping. Wrong
    // value silently corrupts key selection, so pin it directly on the module.
    let _runtime = crate::initialize_runtime();

    let mut args = tiny_deepseek_config();
    let weights = synthetic_indexer_weights(&args, "block.self_attn");

    let ds = Indexer::load(&weights, &args, "block.self_attn")
        .expect("load ok")
        .expect("indexer present");
    assert!(
        !ds.rope_traditional,
        "deepseek_v32 indexer must be non-interleaved"
    );

    // Same weights, glm-style interleave flag.
    args.indexer_rope_interleave = true;
    let glm = Indexer::load(&weights, &args, "block.self_attn")
        .expect("load ok")
        .expect("indexer present");
    assert!(
        glm.rope_traditional,
        "glm_moe_dsa indexer must be interleaved"
    );
}

#[test]
fn indexer_load_absent_weights_is_dense_fallback() {
    // No indexer weights in the checkpoint -> dense fallback (Ok(None)), which
    // keeps the pre-#509 full-attention behavior for such models.
    let _runtime = crate::initialize_runtime();
    let args = tiny_deepseek_config();
    let empty: WeightMap = WeightMap::new();
    let loaded = Indexer::load(&empty, &args, "block.self_attn").expect("load ok");
    assert!(loaded.is_none(), "absent indexer weights must yield None");
}

#[test]
fn indexer_top_indices_none_at_short_context() {
    // Short-context parity: when kv_len <= index_topk the indexer selects
    // nothing and the caller reduces to dense attention (bit-for-bit with the
    // pre-#509 full-attention path).
    let _runtime = crate::initialize_runtime();

    // b=1, n_heads=1, s=1, head_dim=2, kv_len=3, index_topk=8.
    let q = mlxcel_core::from_slice_f32(&[1.0, 0.0], &[1, 1, 1, 2]);
    let k = mlxcel_core::from_slice_f32(&[1.0, 0.0, 5.0, 0.0, 2.0, 0.0], &[1, 1, 3, 2]);
    let w = mlxcel_core::from_slice_f32(&[1.0], &[1, 1, 1]);

    let out = indexer_top_indices(&q, &k, &w, None, 1, 2.0_f32.powf(-0.5), 8);
    assert!(out.is_none(), "kv_len <= index_topk must return None");
}

#[test]
fn indexer_top_indices_selects_highest_scoring_keys() {
    // Constructed long-context case: with q=[1,0] the score of key j reduces to
    // relu(k_j[0]); n_heads=1 and a positive weight preserve that ordering.
    // Keys' first component: [1, 5, 2, 4, 3] -> the two largest are at
    // positions 1 (=5) and 3 (=4). Pin that the top-2 index set is {1, 3}.
    let _runtime = crate::initialize_runtime();

    let q = mlxcel_core::from_slice_f32(&[1.0, 0.0], &[1, 1, 1, 2]);
    let k = mlxcel_core::from_slice_f32(
        &[1.0, 0.0, 5.0, 0.0, 2.0, 0.0, 4.0, 0.0, 3.0, 0.0],
        &[1, 1, 5, 2],
    );
    let w = mlxcel_core::from_slice_f32(&[1.0], &[1, 1, 1]);

    let indices = indexer_top_indices(&q, &k, &w, None, 1, 2.0_f32.powf(-0.5), 2)
        .expect("kv_len > index_topk selects top-k");
    let shape = mlxcel_core::array_shape(&indices);
    assert_eq!(shape, vec![1, 1, 1, 2], "top-k indices shape [b,1,s,topk]");

    let mut got = index_values(&indices, 2);
    got.sort_unstable();
    assert_eq!(got, vec![1, 3], "top-2 keys must be positions 1 and 3");
}

#[test]
fn indexer_top_indices_respects_causal_mask() {
    // The additive causal mask must push masked positions out of the top-k.
    // Same key ordering as above, but mask out position 1 (the top score) with
    // -inf; the top-2 set then becomes {3, 4} (values 4 and 3).
    let _runtime = crate::initialize_runtime();

    let q = mlxcel_core::from_slice_f32(&[1.0, 0.0], &[1, 1, 1, 2]);
    let k = mlxcel_core::from_slice_f32(
        &[1.0, 0.0, 5.0, 0.0, 2.0, 0.0, 4.0, 0.0, 3.0, 0.0],
        &[1, 1, 5, 2],
    );
    let w = mlxcel_core::from_slice_f32(&[1.0], &[1, 1, 1]);
    let neg_inf = f32::NEG_INFINITY;
    let mask = mlxcel_core::from_slice_f32(&[0.0, neg_inf, 0.0, 0.0, 0.0], &[1, 1, 1, 5]);

    let indices = indexer_top_indices(&q, &k, &w, Some(&mask), 1, 2.0_f32.powf(-0.5), 2)
        .expect("kv_len > index_topk selects top-k");
    let mut got = index_values(&indices, 2);
    got.sort_unstable();
    assert_eq!(
        got,
        vec![3, 4],
        "masking position 1 shifts top-2 to {{3,4}}"
    );
}

/// Tiny config with every MLA dimension shrunk so a synthetic attention block
/// is buildable in-test (hidden 6, 2 heads, q_head_dim 4 = nope 2 + rope 2,
/// kv_lora_rank 4, v_head_dim 2).
fn tiny_mla_config(index_topk: i32) -> ModelArgs {
    let json = format!(
        r#"{{
        "model_type": "deepseek_v32",
        "vocab_size": 16,
        "hidden_size": 6,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "index_topk": {index_topk},
        "index_n_heads": 2,
        "index_head_dim": 4
    }}"#
    );
    serde_json::from_str(&json).expect("parse tiny MLA config")
}

/// Synthetic (non-quantized) MLA attention weights for `{prefix}.*`, with
/// `kv_b_proj` in checkpoint form (sanitize decomposes it into
/// `embed_q` / `unembed_out`).
fn synthetic_mla_weights(args: &ModelArgs, prefix: &str, with_indexer: bool) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let q_lora = args.q_lora_rank as i32;
    let kv_lora = args.kv_lora_rank as i32;
    let heads = args.num_attention_heads as i32;
    let q_head_dim = args.q_head_dim() as i32;
    let rope = args.qk_rope_head_dim as i32;
    let nope = args.qk_nope_head_dim as i32;
    let v = args.v_head_dim as i32;

    let mut weights = WeightMap::new();
    let w = |shape: &[i32]| f32_weight(shape);
    weights.insert(format!("{prefix}.q_a_proj.weight"), w(&[q_lora, hidden]));
    weights.insert(format!("{prefix}.q_a_layernorm.weight"), w(&[q_lora]));
    weights.insert(
        format!("{prefix}.q_b_proj.weight"),
        w(&[heads * q_head_dim, q_lora]),
    );
    weights.insert(
        format!("{prefix}.kv_a_proj_with_mqa.weight"),
        w(&[kv_lora + rope, hidden]),
    );
    weights.insert(format!("{prefix}.kv_a_layernorm.weight"), w(&[kv_lora]));
    weights.insert(
        format!("{prefix}.kv_b_proj.weight"),
        w(&[heads * (nope + v), kv_lora]),
    );
    weights.insert(format!("{prefix}.o_proj.weight"), w(&[hidden, heads * v]));
    if with_indexer {
        for (k, v) in synthetic_indexer_weights(args, prefix) {
            weights.insert(k, v);
        }
    }
    weights
}

fn tiny_mla_attention(args: &ModelArgs, with_indexer: bool) -> super::MLAAttention {
    let prefix = "model.layers.0.self_attn";
    let weights = synthetic_mla_weights(args, prefix, with_indexer);
    let weights = super::DeepSeekV32Model::sanitize_weights(weights, args);
    super::load_mla_attention(&weights, args, prefix).expect("tiny MLA attention must load")
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let diff = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(a, b)));
    mlxcel_core::eval(&diff);
    mlxcel_core::item_f32(&diff)
}

/// Regression for issue #619 (dense path): a maskless multi-token prefill must
/// be causal. Position 0 of an N-token prefill must equal a single-token
/// forward of that token (the deepseek_v3 `prefill_is_causal_without_caller_mask`
/// assertion, post-#618).
#[test]
fn prefill_is_causal_without_caller_mask_dense() {
    let _runtime = crate::initialize_runtime();
    // index_topk large -> indexer returns None -> dense path.
    let args = tiny_mla_config(2048);
    let attn = tiny_mla_attention(&args, false);

    let hidden = args.hidden_size as i32;
    // Strongly position-asymmetric inputs (sign flip + growth) so a
    // bidirectional prefill visibly contaminates position 0.
    let x_data: Vec<f32> = (0..2 * args.hidden_size)
        .map(|i| if i < args.hidden_size { 1.0 } else { -3.0 } * (0.4 * (i as f32 % 6.0 + 1.0)))
        .collect();
    let x_two = mlxcel_core::from_slice_f32(&x_data, &[1, 2, hidden]);
    let x_first = mlxcel_core::from_slice_f32(&x_data[..args.hidden_size], &[1, 1, hidden]);

    let mut cache_prefill = mlxcel_core::layers::KVCache::new();
    let out_two = attn.forward(&x_two, None, &mut cache_prefill);
    let pos0_of_two = mlxcel_core::slice(&out_two, &[0, 0, 0], &[1, 1, hidden]);

    let mut cache_single = mlxcel_core::layers::KVCache::new();
    let out_one = attn.forward(&x_first, None, &mut cache_single);

    let diff = max_abs_diff(&pos0_of_two, &out_one);
    assert!(
        diff < 1e-4,
        "dense prefill: position 0 must not attend to position 1 (max diff {diff})"
    );
}

/// Regression for issue #619 (sparse-indexer path): with `kv_len > index_topk`
/// the lightning indexer's top-k selection and the sparse prefill mask must
/// both be causal when the caller passes no mask.
#[test]
fn prefill_is_causal_without_caller_mask_sparse() {
    let _runtime = crate::initialize_runtime();
    // index_topk 2 with a 4-token prefill -> kv_len 4 > 2 -> sparse path.
    let args = tiny_mla_config(2);
    let attn = tiny_mla_attention(&args, true);

    let hidden = args.hidden_size as i32;
    let x_data: Vec<f32> = (0..4 * args.hidden_size)
        .map(|i| {
            let sign = if (i / args.hidden_size).is_multiple_of(2) {
                1.0
            } else {
                -3.0
            };
            sign * (0.4 * (i as f32 % 6.0 + 1.0))
        })
        .collect();
    let x_four = mlxcel_core::from_slice_f32(&x_data, &[1, 4, hidden]);
    let x_first = mlxcel_core::from_slice_f32(&x_data[..args.hidden_size], &[1, 1, hidden]);

    let mut cache_prefill = mlxcel_core::layers::KVCache::new();
    let out_four = attn.forward(&x_four, None, &mut cache_prefill);
    let pos0_of_four = mlxcel_core::slice(&out_four, &[0, 0, 0], &[1, 1, hidden]);

    let mut cache_single = mlxcel_core::layers::KVCache::new();
    let out_one = attn.forward(&x_first, None, &mut cache_single);

    let diff = max_abs_diff(&pos0_of_four, &out_one);
    assert!(
        diff < 1e-4,
        "sparse prefill: position 0 must not attend to future keys (max diff {diff})"
    );
}

/// Decode (`l == 1`) must NOT get the causal fallback: with cached history,
/// every position is causally valid and a fallback mask of the wrong shape
/// would corrupt (or crash) the step. Pin that a prefill-then-decode run
/// produces finite output.
#[test]
fn decode_step_stays_maskless_after_prefill() {
    let _runtime = crate::initialize_runtime();
    let args = tiny_mla_config(2048);
    let attn = tiny_mla_attention(&args, false);

    let hidden = args.hidden_size as i32;
    let x_data: Vec<f32> = (0..3 * args.hidden_size)
        .map(|i| 0.05 * (i as f32 + 1.0))
        .collect();
    let x_two = mlxcel_core::from_slice_f32(&x_data[..2 * args.hidden_size], &[1, 2, hidden]);
    let x_next = mlxcel_core::from_slice_f32(&x_data[2 * args.hidden_size..], &[1, 1, hidden]);

    let mut cache = mlxcel_core::layers::KVCache::new();
    let _ = attn.forward(&x_two, None, &mut cache);
    let out = attn.forward(&x_next, None, &mut cache);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&out));
    mlxcel_core::eval(&m);
    assert!(
        mlxcel_core::item_f32(&m).is_finite(),
        "decode step after prefill must stay finite"
    );
}

/// Regression for the #618-class off-by-one: `sanitize_weights` must strip
/// ONLY the MTP trailer at `layer_idx == num_hidden_layers` and keep every
/// real decoder layer below it (a stripped real layer would make
/// `from_weights` fail with a missing weight, dropping the last layer's
/// computation as in deepseek_v3 pre-#618).
#[test]
fn sanitize_strips_only_the_mtp_trailer_layer() {
    let args = tiny_mla_config(2048); // num_hidden_layers = 1
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        f32_weight(&[6]),
    );
    weights.insert(
        "model.layers.1.input_layernorm.weight".to_string(), // MTP trailer
        f32_weight(&[6]),
    );
    weights.insert("model.norm.weight".to_string(), f32_weight(&[6]));

    let sanitized = super::DeepSeekV32Model::sanitize_weights(weights, &args);
    assert!(
        sanitized.contains_key("model.layers.0.input_layernorm.weight"),
        "real decoder layer 0 must be kept"
    );
    assert!(
        !sanitized.contains_key("model.layers.1.input_layernorm.weight"),
        "MTP trailer at index num_hidden_layers must be stripped"
    );
    assert!(
        sanitized.contains_key("model.norm.weight"),
        "non-layer keys must pass through"
    );
}
