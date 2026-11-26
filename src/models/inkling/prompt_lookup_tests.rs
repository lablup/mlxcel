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

//! Prompt-lookup verify rounds on first-generation Inkling (issue #3422): a
//! token-only verify block runs as the engine runs it (a speculative append
//! through `forward_with_sequence_id`), its rejected tail is dropped through
//! `rewind_decode_appends`, and the decode steps after it must equal the
//! single-token chain that fed only the kept tokens.

use mlxcel_core::cache::{DecodeLookaheadAppendScope, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::{MlxArray, UniquePtr};

use super::InklingModel;
use super::tiny_tests::tiny_model_with_real_heads;
use crate::models::speculative_exactness_draws::position_bytes;

fn seq() -> SequenceId {
    SequenceId::from_raw(7)
}

fn opened(model: &InklingModel) -> Vec<mlxcel_core::layers::KVCache> {
    model.prepare_sequence_state(seq());
    model.make_caches()
}

fn step(
    model: &InklingModel,
    caches: &mut [mlxcel_core::layers::KVCache],
    tokens: &[i32],
) -> UniquePtr<MlxArray> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let logits = model.forward_with_sequence_id(&input, Some(seq()), caches, None);
    mlxcel_core::eval(&logits);
    logits
}

fn verify(
    model: &InklingModel,
    caches: &mut [mlxcel_core::layers::KVCache],
    block: &[i32],
) -> UniquePtr<MlxArray> {
    let _speculative = DecodeLookaheadAppendScope::enter();
    step(model, caches, block)
}

/// Every reject position of a width-4 and a width-10 block (past the 8
/// queries MLX's SDPA runs per query), the full reject (`kept == 0`, what a
/// failed round and the width warm-up unwind) included: two decode steps
/// after the rewind equal, byte for byte, a model that decoded the kept
/// tokens one at a time.
#[test]
fn a_rewound_verify_round_matches_the_single_token_chain_on_metal() {
    if !mlxcel_core::row_invariant::available() {
        eprintln!("skipping: the row-invariant kernel needs the Metal device");
        return;
    }
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let prompt: Vec<i32> = (0..8).map(|i| (i * 5 + 1) % 32).collect();
    for width in [4_usize, 10] {
        let block: Vec<i32> = (0..width as i32).map(|i| (i * 7 + 3) % 32).collect();
        for kept in 0..width {
            let model = tiny_model_with_real_heads();
            assert!(model.supports_speculative_verify_rewind());
            let mut caches = opened(&model);
            let _ = step(&model, &mut caches, &prompt);
            let _ = verify(&model, &mut caches, &block);
            model
                .rewind_decode_appends(seq(), (width - kept) as i32)
                .unwrap_or_else(|err| panic!("width {width}, kept {kept}: {err}"));

            let chain = tiny_model_with_real_heads();
            let mut chain_caches = opened(&chain);
            let _ = step(&chain, &mut chain_caches, &prompt);
            for &token in &block[..kept] {
                let _ = step(&chain, &mut chain_caches, &[token]);
            }
            for next in [9, 17] {
                assert_eq!(
                    position_bytes(&step(&model, &mut caches, &[next]), 0),
                    position_bytes(&step(&chain, &mut chain_caches, &[next]), 0),
                    "width {width}, kept {kept}, next {next}"
                );
            }
        }
    }
}

/// A rewind needs the windows of the last verify block: after a plain
/// decode step, past the block's length, or a second time, it is refused
/// and the state stays the chain's.
#[test]
fn a_rewind_without_a_recorded_block_is_refused_without_mutation() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model_with_real_heads();
    let mut caches = opened(&model);
    let chain = tiny_model_with_real_heads();
    let mut chain_caches = opened(&chain);
    for tokens in [&[1, 2, 3][..], &[4]] {
        let _ = step(&model, &mut caches, tokens);
        let _ = step(&chain, &mut chain_caches, tokens);
    }
    assert!(model.rewind_decode_appends(seq(), 1).is_err());
    assert_eq!(model.rewind_decode_appends(seq(), 0), Ok(()));

    let _ = verify(&model, &mut caches, &[5, 6, 7]);
    let error = model.rewind_decode_appends(seq(), 4).unwrap_err();
    assert!(error.contains("cannot drop 4 rows"), "{error}");
    model
        .rewind_decode_appends(seq(), 3)
        .expect("drop the whole block");
    assert!(model.rewind_decode_appends(seq(), 1).is_err());
    let missing = SequenceId::from_raw(99);
    assert!(model.rewind_decode_appends(missing, 1).is_err());

    let actual = step(&model, &mut caches, &[8]);
    let expected = step(&chain, &mut chain_caches, &[8]);
    let difference = mlxcel_core::subtract(&actual, &expected);
    let values = mlxcel_core::utils::array_to_vec_f32(&difference);
    assert!(values.iter().all(|v| v.abs() <= 1e-5), "{values:?}");
}

/// A block wider than the row-invariant scope (prompt lookup allows up to
/// 65 rows) still records its windows, so its rejected tail is dropped; its
/// rows are not byte-identical to the chain, so the check is a tolerance.
#[test]
fn a_verify_block_past_the_row_invariant_scope_still_rewinds() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let prompt: Vec<i32> = (0..8).map(|i| (i * 3 + 2) % 32).collect();
    let block: Vec<i32> = (0..20).map(|i| (i * 11 + 1) % 32).collect();
    let model = tiny_model_with_real_heads();
    let mut caches = opened(&model);
    let _ = step(&model, &mut caches, &prompt);
    let _ = verify(&model, &mut caches, &block);
    model
        .rewind_decode_appends(seq(), 15)
        .expect("drop 15 of 20 rows");

    let chain = tiny_model_with_real_heads();
    let mut chain_caches = opened(&chain);
    let _ = step(&chain, &mut chain_caches, &prompt);
    for &token in &block[..5] {
        let _ = step(&chain, &mut chain_caches, &[token]);
    }
    let actual = step(&model, &mut caches, &[9]);
    let expected = step(&chain, &mut chain_caches, &[9]);
    let difference = mlxcel_core::subtract(&actual, &expected);
    let values = mlxcel_core::utils::array_to_vec_f32(&difference);
    assert!(values.iter().all(|v| v.abs() <= 2e-4), "{values:?}");
}
