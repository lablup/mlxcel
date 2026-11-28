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

//! A DeepSeek V3 verify block against the single-token chain (issue #3418).
//!
//! The engine's token-only verify round (prompt lookup, a full-model
//! drafter) forwards K tokens at once inside the speculative append scope and
//! reads every row's logits. Those rows must be the logits K decode steps
//! produce, byte for byte.

use mlxcel_core::cache::DecodeLookaheadAppendScope;
use mlxcel_core::layers::KVCache;

use super::tests::{test_config_dense_direct_q, tiny_dense_model_weights};
use super::{DeepSeekV3Config, DeepSeekV3Model};
use crate::models::mla_batched::ROW_INVARIANT_TEST_MODE;

/// Wide enough for the row-invariant matvec (every product width a multiple
/// of 4). A rope width of 16 keeps a single query on the composed MLA path;
/// 32 is the smallest the fused kernel takes.
fn config() -> DeepSeekV3Config {
    config_with_rope(16)
}

fn config_with_rope(qk_rope_head_dim: usize) -> DeepSeekV3Config {
    DeepSeekV3Config {
        vocab_size: 96,
        hidden_size: 64,
        intermediate_size: 128,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        kv_lora_rank: 64,
        qk_rope_head_dim,
        qk_nope_head_dim: 32,
        v_head_dim: 32,
        num_hidden_layers: 3,
        ..test_config_dense_direct_q()
    }
}

fn model(cfg: &DeepSeekV3Config) -> DeepSeekV3Model {
    let weights = tiny_dense_model_weights(cfg);
    let weights = DeepSeekV3Model::sanitize_weights(weights, cfg).expect("sanitize must succeed");
    DeepSeekV3Model::from_weights(&weights, cfg).expect("tiny model must load")
}

fn input(tokens: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn row_bytes(logits: &mlxcel_core::MlxArray, row: i32) -> Vec<u8> {
    mlxcel_core::array_to_raw_bytes(&mlxcel_core::utils::slice_axis(logits, 1, row, row + 1))
}

fn prefilled(model: &DeepSeekV3Model, prompt: &[i32]) -> Vec<KVCache> {
    let mut caches: Vec<KVCache> = (0..model.layers.len()).map(|_| KVCache::new()).collect();
    let logits = model.forward_impl(&input(prompt), &mut caches, None);
    mlxcel_core::eval(&logits);
    caches
}

/// Rows of a `width`-token block in the speculative append scope that differ
/// from the single-token chain over the same tokens.
fn differing_rows(model: &DeepSeekV3Model, width: usize) -> Vec<usize> {
    let vocab = model.config.vocab_size;
    let prompt: Vec<i32> = (0..9).map(|i| ((i * 7 + 3) % vocab) as i32).collect();
    let block: Vec<i32> = (0..width).map(|i| ((i * 13 + 5) % vocab) as i32).collect();

    let mut chain_caches = prefilled(model, &prompt);
    let chain: Vec<Vec<u8>> = block
        .iter()
        .map(|&token| {
            row_bytes(
                &model.forward_impl(&input(&[token]), &mut chain_caches, None),
                0,
            )
        })
        .collect();

    let mut block_caches = prefilled(model, &prompt);
    let logits = {
        let _speculative = DecodeLookaheadAppendScope::enter();
        model.forward_impl(&input(&block), &mut block_caches, None)
    };
    (0..width)
        .filter(|&row| row_bytes(&logits, row as i32) != chain[row])
        .collect()
}

struct ModeGuard;

impl ModeGuard {
    fn set(mode: Option<bool>) -> Self {
        ROW_INVARIANT_TEST_MODE.with(|cell| cell.set(Some(mode)));
        Self
    }
}

impl Drop for ModeGuard {
    fn drop(&mut self) {
        ROW_INVARIANT_TEST_MODE.with(|cell| cell.set(None));
    }
}

#[test]
fn a_verify_block_matches_the_single_token_chain_on_metal() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    for rope in [16, 32] {
        let cfg = config_with_rope(rope);
        let model = model(&cfg);
        for kernel in [true, false] {
            let _mode = ModeGuard::set(Some(kernel));
            for width in [2, 4, 8, 12] {
                assert_eq!(
                    differing_rows(&model, width),
                    Vec::<usize>::new(),
                    "rope {rope}, width {width}, matvec kernel {kernel}: block rows differ from \
                     the chain"
                );
            }
        }
    }
}

/// The two geometries above cover both single-query paths: every row of a
/// verify block takes the fused kernel at rope 32 and the composed form at
/// rope 16, as its decode step does. The block counts as one call per layer
/// on that path (#3421), and a decode step still counts one.
#[test]
fn verify_block_rows_take_the_single_query_paths_on_metal() {
    use mlxcel_core::mla::MlaDecodePath;
    if !mlxcel_core::metal_is_available() {
        return;
    }
    let _mode = ModeGuard::set(Some(true));
    for (rope, path) in [
        (16, MlaDecodePath::AbsorbedComposed),
        (32, MlaDecodePath::AbsorbedFused),
    ] {
        let cfg = config_with_rope(rope);
        let model = model(&cfg);
        let mut caches = prefilled(&model, &[3, 5, 7, 2, 11, 6]);
        let (_, counts) = mlxcel_core::mla::stats::scoped(|| {
            let _speculative = DecodeLookaheadAppendScope::enter();
            let logits = model.forward_impl(&input(&[9, 4, 13, 8]), &mut caches, None);
            mlxcel_core::eval(&logits);
        });
        let layers = model.layers.len() as u64;
        assert_eq!(
            counts.get(path),
            layers,
            "rope {rope}: one record per verify block and layer, not one per row"
        );
        assert_eq!(counts.total(), layers, "rope {rope}: {counts:?}");

        let (_, decode) = mlxcel_core::mla::stats::scoped(|| {
            for token in [10, 12] {
                let logits = model.forward_impl(&input(&[token]), &mut caches, None);
                mlxcel_core::eval(&logits);
            }
        });
        assert_eq!(decode.get(path), 2 * layers, "rope {rope}: decode steps");
        assert_eq!(decode.total(), 2 * layers, "rope {rope}: {decode:?}");
    }
}

/// What the scope fixes, and the default: without it the same block parts
/// from the chain.
#[test]
fn without_the_scope_a_block_differs_from_the_chain_on_metal() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    let cfg = config();
    let model = model(&cfg);
    let _mode = ModeGuard::set(None);
    let differing: usize = [2, 4]
        .into_iter()
        .map(|width| differing_rows(&model, width).len())
        .sum();
    assert!(
        differing > 0,
        "the block already equals the chain without the scope"
    );
}

/// Outside the speculative append scope a short multi-token forward is a
/// prefill and keeps the block attention.
#[test]
fn a_forward_outside_the_append_scope_is_not_a_verify_block() {
    let shape = [1, 4, 64];
    assert!(!mlxcel_core::row_invariant::is_verify_block(&shape));
    let _speculative = DecodeLookaheadAppendScope::enter();
    assert!(mlxcel_core::row_invariant::is_verify_block(&shape));
}

#[test]
fn the_env_value_selects_the_mode() {
    use crate::models::mla_batched::row_invariant_mode_from as mode;
    for on in ["1", "true", "yes", "on", " ON "] {
        assert_eq!(mode(Some(on)), Some(true));
    }
    for off in [
        None,
        Some("0"),
        Some("false"),
        Some("off"),
        Some("verify"),
        Some(""),
    ] {
        assert_eq!(mode(off), None);
    }
}
