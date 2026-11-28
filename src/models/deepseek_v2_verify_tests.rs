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

//! A DeepSeek V2 verify block against the single-token chain (issue #3421),
//! on both cache layouts: the absorbed latent (the default where the fused
//! MLA kernel runs) and the decompressed per-head K/V.

use mlxcel_core::cache::DecodeLookaheadAppendScope;
use mlxcel_core::layers::KVCache;
use mlxcel_core::mla::{MlaAbsorbedProjections, MlaDecodePath};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{DeepSeekV2Model, ModelArgs};
use crate::models::mla_batched::RowInvariantTestMode;

const VOCAB: i32 = 96;
const HIDDEN: i32 = 64;
const LAYERS: usize = 2;

/// Every product width a multiple of 4 (the matvec contract); the latent
/// rank and the per-head K/V width 64, a head dim MLX's fused SDPA takes;
/// rope 32, the smallest the fused MLA kernel takes. Layer 1 is a float MoE
/// layer with a shared expert.
fn args() -> ModelArgs {
    let json = format!(
        r#"{{
        "model_type": "deepseek_v2",
        "vocab_size": {VOCAB},
        "hidden_size": {HIDDEN},
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_hidden_layers": {LAYERS},
        "num_attention_heads": 4,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 32,
        "v_head_dim": 64,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "rope_theta": 10000.0
    }}"#
    );
    serde_json::from_str(&json).expect("parse tiny DeepSeek V2 verify config")
}

/// A varied, bounded fill, salted per tensor so no two weights repeat.
fn fill(shape: &[i32], salt: usize) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().product::<i32>() as usize;
    let data: Vec<f32> = (0..n)
        .map(|i| ((((i + salt * 131) * 37 + 11) % 97) as f32 - 48.0) * 0.01)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// The tiny model with every layer's attention on the absorbed latent
/// (`absorbed`) or the decompressed K/V.
fn model(absorbed: bool) -> DeepSeekV2Model {
    let args = args();
    let heads = 4;
    let (rank, nope, rope, v) = (64, 32, 32, 64);
    let mut shapes: Vec<(String, Vec<i32>)> = Vec::new();
    let mut put = |key: String, shape: &[i32]| shapes.push((key, shape.to_vec()));
    put("model.embed_tokens.weight".into(), &[VOCAB, HIDDEN]);
    put("model.norm.weight".into(), &[HIDDEN]);
    put("lm_head.weight".into(), &[VOCAB, HIDDEN]);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        put(format!("{p}.input_layernorm.weight"), &[HIDDEN]);
        put(format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]);
        let a = format!("{p}.self_attn");
        put(
            format!("{a}.q_proj.weight"),
            &[heads * (nope + rope), HIDDEN],
        );
        put(
            format!("{a}.kv_a_proj_with_mqa.weight"),
            &[rank + rope, HIDDEN],
        );
        put(format!("{a}.kv_a_layernorm.weight"), &[rank]);
        put(format!("{a}.kv_b_proj.weight"), &[heads * (nope + v), rank]);
        put(format!("{a}.o_proj.weight"), &[HIDDEN, heads * v]);
        if l == 0 {
            put(format!("{p}.mlp.gate_proj.weight"), &[128, HIDDEN]);
            put(format!("{p}.mlp.up_proj.weight"), &[128, HIDDEN]);
            put(format!("{p}.mlp.down_proj.weight"), &[HIDDEN, 128]);
        } else {
            put(format!("{p}.mlp.gate.weight"), &[4, HIDDEN]);
            put(
                format!("{p}.mlp.switch_mlp.gate_proj.weight"),
                &[4, 32, HIDDEN],
            );
            put(
                format!("{p}.mlp.switch_mlp.up_proj.weight"),
                &[4, 32, HIDDEN],
            );
            put(
                format!("{p}.mlp.switch_mlp.down_proj.weight"),
                &[4, HIDDEN, 32],
            );
            put(
                format!("{p}.mlp.shared_experts.gate_proj.weight"),
                &[32, HIDDEN],
            );
            put(
                format!("{p}.mlp.shared_experts.up_proj.weight"),
                &[32, HIDDEN],
            );
            put(
                format!("{p}.mlp.shared_experts.down_proj.weight"),
                &[HIDDEN, 32],
            );
        }
    }
    let weights: WeightMap = shapes
        .iter()
        .enumerate()
        .map(|(salt, (key, shape))| (key.clone(), fill(shape, salt)))
        .collect();
    let mut model = DeepSeekV2Model::from_weights(&weights, &args).expect("tiny model loads");
    for layer in &mut model.layers {
        let attn = &mut layer.self_attn;
        attn.absorbed = absorbed.then(|| {
            MlaAbsorbedProjections::from_kv_b_proj(&attn.kv_b_proj, attn.geometry()).expect("fold")
        });
    }
    model
}

fn input(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn row_bytes(logits: &MlxArray, row: i32) -> Vec<u8> {
    mlxcel_core::array_to_raw_bytes(&mlxcel_core::utils::slice_axis(logits, 1, row, row + 1))
}

fn prefilled(model: &DeepSeekV2Model, prompt: &[i32]) -> Vec<KVCache> {
    let mut caches = model.make_caches();
    let logits = model.forward(&input(prompt), &mut caches, None);
    mlxcel_core::eval(&logits);
    caches
}

/// Rows of a `width`-token block in the speculative append scope that differ
/// from the single-token chain over the same tokens.
fn differing_rows(model: &DeepSeekV2Model, width: usize) -> Vec<usize> {
    let prompt: Vec<i32> = (0..9).map(|i| (i * 7 + 3) % VOCAB).collect();
    let block: Vec<i32> = (0..width as i32).map(|i| (i * 13 + 5) % VOCAB).collect();

    let mut chain_caches = prefilled(model, &prompt);
    let chain: Vec<Vec<u8>> = block
        .iter()
        .map(|&token| row_bytes(&model.forward(&input(&[token]), &mut chain_caches, None), 0))
        .collect();

    let mut block_caches = prefilled(model, &prompt);
    let logits = {
        let _speculative = DecodeLookaheadAppendScope::enter();
        model.forward(&input(&block), &mut block_caches, None)
    };
    (0..width)
        .filter(|&row| row_bytes(&logits, row as i32) != chain[row])
        .collect()
}

#[test]
fn a_verify_block_matches_the_single_token_chain_on_metal() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    for absorbed in [true, false] {
        let model = model(absorbed);
        for kernel in [true, false] {
            let _mode = RowInvariantTestMode::set(Some(kernel));
            for width in [2, 4, 8, 12] {
                assert_eq!(
                    differing_rows(&model, width),
                    Vec::<usize>::new(),
                    "absorbed {absorbed}, width {width}, matvec kernel {kernel}: block rows \
                     differ from the chain"
                );
            }
        }
    }
}

/// The default: without the scope the same block parts from the chain.
///
/// The widths reach past the few-row GEMV route of the matmul overlay
/// (`MLXCEL_DENSE_GEMV_ROWS`, 6 rows by default): on GPU generations 13 and
/// 14 (M1, M2) a dense product of up to that many rows runs each row through
/// the one-row GEMV, so the decompressed layout's projections at widths 2
/// and 4 already equal the chain there (issue #3468). Wider blocks take the
/// steel GEMM tile on every generation, which sums in another order than the
/// decode step's GEMV.
#[test]
fn without_the_scope_a_block_differs_from_the_chain_on_metal() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    let _mode = RowInvariantTestMode::set(None);
    for absorbed in [true, false] {
        let model = model(absorbed);
        let differing: usize = [2, 4, 8, 12]
            .into_iter()
            .map(|width| differing_rows(&model, width).len())
            .sum();
        assert!(
            differing > 0,
            "absorbed {absorbed}: the block already equals the chain without the scope"
        );
    }
}

/// On the absorbed layout a verify block records one single-query call per
/// layer, the path its decode step takes, and a decode step records one.
#[test]
fn an_absorbed_verify_block_records_one_call_per_layer_on_metal() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    let _mode = RowInvariantTestMode::set(Some(true));
    let model = model(true);
    let mut caches = prefilled(&model, &[3, 5, 7, 2, 11, 6]);
    let layers = LAYERS as u64;
    let (_, decode) = mlxcel_core::mla::stats::scoped(|| {
        mlxcel_core::eval(&model.forward(&input(&[10]), &mut caches, None));
    });
    assert_eq!(decode.total(), layers, "{decode:?}");
    let (_, block) = mlxcel_core::mla::stats::scoped(|| {
        let _speculative = DecodeLookaheadAppendScope::enter();
        mlxcel_core::eval(&model.forward(&input(&[9, 4, 13, 8]), &mut caches, None));
    });
    assert_eq!(
        block, decode,
        "a block counts as the decode step it stands for"
    );
    assert_eq!(block.get(MlaDecodePath::AbsorbedPrefill), 0, "{block:?}");
}
