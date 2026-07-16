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

//! Checkpoint-free unit tests for the MiniMax-M3 text decoder.
//!
//! These cover the surface reachable without the 427B checkpoint: real
//! `text_config` parsing and the derived layer plan, the sigmoid router's
//! bias-affects-selection-only invariant, the packed shared expert, partial
//! RoPE, per-head Q/K norm, and the block-sparse indexer degeneration property.
//! Run serially (`--test-threads=1`); the MLX ops touch the device.

use super::ModelArgs;
use super::indexer::build_block_drop_mask;
use super::moe::{append_shared_expert, route};
use crate::models::gemma::GemmaRMSNorm;
use mlxcel_core::MlxArray;

// The real MiniMaxAI/MiniMax-M3 `text_config` (60 layers, first 3 dense) laid
// out as the flat text config a text-only export / VL wrapper would present.
const REAL_TEXT_CONFIG: &str = r#"{
    "model_type": "minimax_m3",
    "hidden_size": 6144,
    "intermediate_size": 3072,
    "num_hidden_layers": 60,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 200064,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-06,
    "use_gemma_norm": true,
    "attention_output_gate": false,
    "rope_theta": 5000000,
    "rotary_dim": 64,
    "partial_rotary_factor": 0.5,
    "hidden_act": "swigluoai",
    "use_qk_norm": true,
    "qk_norm_type": "per_head",
    "tie_word_embeddings": false,
    "dense_intermediate_size": 12288,
    "shared_intermediate_size": 3072,
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "n_shared_experts": 1,
    "scoring_func": "sigmoid",
    "use_routing_bias": true,
    "routed_scaling_factor": 2.0,
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "num_mtp_modules": 7,
    "num_nextn_predict_layers": 1,
    "moe_layer_freq": [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    "sparse_attention_config": {
        "use_sparse_attention": true,
        "sparse_index_dim": 128,
        "sparse_num_index_heads": 4,
        "sparse_topk_blocks": 16,
        "sparse_block_size": 128,
        "sparse_score_type": "max",
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_attention_freq": [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        "sparse_disable_index_value": [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    }
}"#;

fn reduce_max_abs(a: &MlxArray) -> f32 {
    let flat = mlxcel_core::reshape(a, &[-1]);
    let m = mlxcel_core::max_axis(&mlxcel_core::abs(&flat), 0, false);
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

fn reduce_min(a: &MlxArray) -> f32 {
    let flat = mlxcel_core::reshape(a, &[-1]);
    let m = mlxcel_core::min_axis(&flat, 0, false);
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

fn reduce_sum(a: &MlxArray) -> f32 {
    let f32a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    let flat = mlxcel_core::reshape(&f32a, &[-1]);
    let s = mlxcel_core::sum_axis(&flat, 0, false);
    mlxcel_core::eval(&s);
    mlxcel_core::item_f32(&s)
}

fn scalar_i32(a: &MlxArray) -> i32 {
    mlxcel_core::eval(a);
    mlxcel_core::item_i32(a)
}

#[test]
fn config_parses_real_text_config_and_derives_layer_plan() {
    let args: ModelArgs = serde_json::from_str(REAL_TEXT_CONFIG).expect("real text_config parses");

    assert_eq!(args.model_type, "minimax_m3");
    assert_eq!(args.hidden_size, 6144);
    assert_eq!(args.intermediate_size, 3072);
    assert_eq!(args.num_hidden_layers, 60);
    assert_eq!(args.num_attention_heads, 64);
    assert_eq!(args.num_key_value_heads, 4);
    assert_eq!(args.head_dim, 128);
    assert_eq!(args.vocab_size, 200064);
    assert_eq!(args.rotary_dim, 64);
    assert!((args.rope_theta - 5_000_000.0).abs() < 1.0);
    assert_eq!(args.num_local_experts, 128);
    assert_eq!(args.num_experts_per_tok, 4);
    assert_eq!(args.n_shared_experts, 1);
    assert_eq!(args.dense_intermediate_size, 12288);
    assert_eq!(args.shared_intermediate_size, 3072);
    assert!((args.routed_scaling_factor - 2.0).abs() < 1e-6);
    assert_eq!(args.hidden_act, "swigluoai");
    assert_eq!(args.qk_norm_type, "per_head");
    assert!(args.use_qk_norm && args.use_gemma_norm && args.use_routing_bias);
    assert!(!args.attention_output_gate);
    assert!((args.swiglu_alpha - 1.702).abs() < 1e-6);
    assert!((args.swiglu_limit - 7.0).abs() < 1e-6);

    // MTP metadata parses but is otherwise ignored.
    assert_eq!(args.num_mtp_modules, Some(7));
    assert_eq!(args.num_nextn_predict_layers, Some(1));

    // Layer plan: first 3 layers dense, the rest MoE.
    assert!(!args.is_moe_layer(0));
    assert!(!args.is_moe_layer(2));
    assert!(args.is_moe_layer(3));
    assert!(args.is_moe_layer(59));

    // Shared expert packs into the switch tensors (widths equal).
    assert!(args.shared_expert_is_packed());

    // Sparse-attention layer plan aligns with the MoE plan (first 3 dense).
    let sparse = args
        .sparse_attention_config
        .as_ref()
        .expect("sparse_attention_config present");
    assert!(sparse.use_sparse_attention);
    assert_eq!(sparse.sparse_block_size, 128);
    assert_eq!(sparse.sparse_topk_blocks, 16);
    assert_eq!(sparse.sparse_num_index_heads, 4);
    assert_eq!(sparse.sparse_index_dim, 128);
    assert_eq!(sparse.sparse_init_block, 0);
    assert_eq!(sparse.sparse_local_block, 1);
    assert_eq!(sparse.sparse_score_type, "max");
    assert!(!args.is_sparse_layer(0));
    assert!(!args.is_sparse_layer(2));
    assert!(args.is_sparse_layer(3));
    assert!(args.is_sparse_layer(59));
}

#[test]
fn router_bias_changes_selection_not_mixture_weights() {
    // 1 token, 4 experts. Descending logits, so without bias the top-2 are
    // experts 0 and 1; a large bias on experts 2 and 3 flips the selection to
    // {2, 3}, but the returned mixture weights must be the UNBIASED sigmoids of
    // experts 2 and 3, not the bias-inflated ones.
    let logits = mlxcel_core::from_slice_f32(&[2.0, 1.0, 0.5, 0.0], &[1, 4]);
    let bias = mlxcel_core::from_slice_f32(&[0.0, 0.0, 5.0, 5.0], &[4]);

    let (idx, scores) = route(&logits, &bias, 2, false, 1.0);

    let i0 = scalar_i32(&mlxcel_core::utils::slice_axis(&idx, -1, 0, 1));
    let i1 = scalar_i32(&mlxcel_core::utils::slice_axis(&idx, -1, 1, 2));
    let mut selected = [i0, i1];
    selected.sort_unstable();
    assert_eq!(
        selected,
        [2, 3],
        "bias must steer selection to experts 2 and 3"
    );

    // Unbiased mixture weights: sigmoid(0.5) + sigmoid(0.0) ~= 1.1224. If the
    // bias had leaked into the weights it would be sigmoid(5.5) + sigmoid(5.0)
    // ~= 1.989, so this sum discriminates the two behaviors.
    let sum = reduce_sum(&scores);
    let expected = sigmoid(0.5) + sigmoid(0.0);
    assert!(
        (sum - expected).abs() < 1e-3,
        "mixture weights must be unbiased sigmoids: got {sum}, expected {expected}"
    );
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn packed_shared_expert_appended_with_index_and_unit_score() {
    // 2 tokens, top-2 routed. The shared expert (index 128) must be appended to
    // every token's selection with a fixed score of 1.0.
    let idx = mlxcel_core::from_slice_i32(&[0, 1, 5, 7], &[2, 2]);
    let scores = mlxcel_core::from_slice_f32(&[0.3, 0.7, 0.4, 0.6], &[2, 2]);

    let (idx_full, score_full) = append_shared_expert(&idx, &scores, 128);

    assert_eq!(mlxcel_core::array_shape(&idx_full), vec![2, 3]);
    assert_eq!(mlxcel_core::array_shape(&score_full), vec![2, 3]);

    // Last column: index 128 for both tokens -> sum 256; score 1.0 -> sum 2.0.
    let last_idx = mlxcel_core::utils::slice_axis(&idx_full, -1, 2, 3);
    let last_score = mlxcel_core::utils::slice_axis(&score_full, -1, 2, 3);
    assert!((reduce_sum(&last_idx) - 256.0).abs() < 1e-3);
    assert!((reduce_sum(&last_score) - 2.0).abs() < 1e-4);

    // The original routed columns are preserved unchanged.
    let kept_idx = mlxcel_core::utils::slice_axis(&idx_full, -1, 0, 2);
    let kept_diff = mlxcel_core::subtract(
        &mlxcel_core::astype(&kept_idx, mlxcel_core::dtype::FLOAT32),
        &mlxcel_core::astype(&idx, mlxcel_core::dtype::FLOAT32),
    );
    assert!(reduce_max_abs(&kept_diff) < 1e-4);
}

#[test]
fn partial_rope_leaves_the_non_rotary_tail_untouched() {
    // head_dim 128, rotary_dim 64: RoPE must rotate the first 64 dims and leave
    // dims [64, 128) exactly as they were. Shape [b=1, heads=1, seq=2, 128].
    let head_dim = 128usize;
    let rot = 64i32;
    let seq = 2usize;
    let n = head_dim * seq;
    let data: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let q = mlxcel_core::from_slice_f32(&data, &[1, 1, seq as i32, head_dim as i32]);

    let out = mlxcel_core::fast_rope(&q, rot, false, 5_000_000.0, 1.0, 0);

    let q_tail = mlxcel_core::utils::slice_axis(&q, -1, rot, head_dim as i32);
    let out_tail = mlxcel_core::utils::slice_axis(&out, -1, rot, head_dim as i32);
    let tail_diff = mlxcel_core::subtract(&out_tail, &q_tail);
    assert!(
        reduce_max_abs(&tail_diff) < 1e-4,
        "partial RoPE must not touch the non-rotary tail"
    );

    // The rotary head (dims [0, 64)) is actually rotated at position 1 (offset 0
    // leaves position 0 an identity rotation).
    let q_pos1 = mlxcel_core::utils::slice_axis(&q, 2, 1, 2);
    let out_pos1 = mlxcel_core::utils::slice_axis(&out, 2, 1, 2);
    let q_head = mlxcel_core::utils::slice_axis(&q_pos1, -1, 0, rot);
    let out_head = mlxcel_core::utils::slice_axis(&out_pos1, -1, 0, rot);
    let head_diff = mlxcel_core::subtract(&out_head, &q_head);
    assert!(
        reduce_max_abs(&head_diff) > 1e-3,
        "the rotary head must actually rotate"
    );
}

#[test]
fn per_head_qk_norm_normalizes_each_head_independently() {
    // Per-head norm operates on the last axis of [b, l, heads, head_dim]. Feed
    // head 1 = 2 * head 0; RMSNorm is scale-invariant, so if the norm is truly
    // per-head both heads become equal. A wrongly-flattened norm over all 8
    // values would not.
    let head_dim = 4usize;
    let input = mlxcel_core::from_slice_f32(
        &[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
        &[1, 1, 2, head_dim as i32],
    );
    // Zero weight -> GemmaRMSNorm uses (1 + weight) = 1, i.e. plain RMSNorm.
    let weight = mlxcel_core::zeros(&[head_dim as i32], mlxcel_core::dtype::FLOAT32);
    let norm = GemmaRMSNorm::new(weight, 1e-6);
    let out = norm.forward(&input);

    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![1, 1, 2, head_dim as i32]
    );
    let head0 = mlxcel_core::utils::slice_axis(&out, 2, 0, 1);
    let head1 = mlxcel_core::utils::slice_axis(&out, 2, 1, 2);
    let diff = mlxcel_core::subtract(&head0, &head1);
    assert!(
        reduce_max_abs(&diff) < 1e-4,
        "scale-invariant RMSNorm must normalize each head independently"
    );
}

#[test]
fn block_sparse_indexer_degenerates_to_dense_when_topk_covers_all_blocks() {
    // token_scores [1, 1, s=8, kv_len=8], block_size 4 -> 2 blocks.
    let s = 8i32;
    let kv_len = 8i32;
    let block_size = 4i32;
    let scores: Vec<f32> = (0..(s * kv_len)).map(|i| (i as f32 % 5.0) - 2.0).collect();
    let token_scores = mlxcel_core::from_slice_f32(&scores, &[1, 1, s, kv_len]);

    // topk_blocks (2) covers every block -> the additive mask is exactly zero,
    // so block-sparse attention is identical to dense.
    let dense = build_block_drop_mask(&token_scores, 0, block_size, 2, 0, 1);
    assert_eq!(mlxcel_core::array_shape(&dense), vec![1, 1, s, kv_len]);
    assert!(
        reduce_max_abs(&dense) < 1e-12,
        "full-coverage block mask must be all zeros (dense-equivalent)"
    );

    // With a smaller budget (topk_blocks 1 < 2 blocks), at least one block is
    // dropped, so some entries are -inf.
    let sparse = build_block_drop_mask(&token_scores, 0, block_size, 1, 0, 1);
    assert!(
        reduce_min(&sparse).is_infinite() && reduce_min(&sparse) < 0.0,
        "an under-budget block mask must drop at least one block to -inf"
    );
}
