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

//! `--kv-cache-mode` on the batching families whose KV caches live in
//! `ModelOwnedSequenceState` (#2766): ExaOne 4, GPT-OSS, Ministral 3,
//! Step 3.5, Mellum and Laguna.
//!
//! The server and CLI hand the resolved per-layer table to the model through
//! `set_kv_cache_layer_modes`; the pool caches of these families are empty,
//! so the table is the only way a quantized mode reaches their storage. Each
//! test injects `Int8`, opens a sequence, prefills it, and reads the modes the
//! sequence's own caches were built with.

use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::utils::{array_to_vec_f32, slice_axis};
use mlxcel_core::{MlxArray, UniquePtr};

use super::mixed_cache_batched_decode_tests::{exaone, laguna, mellum, ministral, step};

const PROMPT: [i32; 6] = [1, 2, 3, 4, 5, 6];
const DECODE_STEPS: usize = 8;

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

/// The last position's logits of a `[1, L, V]` output, as f32.
fn last_row(logits: &MlxArray) -> Vec<f32> {
    let len = mlxcel_core::array_shape(logits)[1];
    let row = slice_axis(logits, 1, len - 1, len);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    array_to_vec_f32(&row)
}

fn argmax(values: &[f32]) -> i32 {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best as i32
}

/// Inject `Int8` for every layer, prefill one sequence and return the modes
/// its caches carry. Fails on a family that drops the table, where
/// `kv_cache_layer_modes()` is the trait default `None`.
fn int8_sequence_modes<M: LanguageModel>(
    model: &M,
    kv_modes: impl Fn(&M, SequenceId) -> Vec<KVCacheMode>,
) -> Vec<KVCacheMode> {
    let int8 = vec![KVCacheMode::Int8; model.num_layers()];
    model.set_kv_cache_layer_modes(int8.clone());
    assert_eq!(
        model.kv_cache_layer_modes(),
        Some(int8),
        "the model must keep the injected KV cache mode table"
    );
    let id = SequenceId::from_raw(7);
    model.prepare_sequence_state(id);
    let logits = model.forward_with_sequence_id(&ids(&PROMPT), Some(id), &mut [], None);
    mlxcel_core::eval(&logits);
    let modes = kv_modes(model, id);
    model.release_sequence_state_by_id(id);
    modes
}

/// Greedy decode of [`DECODE_STEPS`] tokens on one sequence under `mode`,
/// asserting every step's logits are finite.
fn greedy_tokens<M: LanguageModel>(model: &M, mode: KVCacheMode) -> Vec<i32> {
    let table = vec![mode; model.num_layers()];
    model.set_kv_cache_layer_modes(table.clone());
    assert_eq!(
        model.kv_cache_layer_modes(),
        Some(table),
        "the {mode} run must store {mode}"
    );
    let id = SequenceId::from_raw(11);
    model.prepare_sequence_state(id);
    let mut tokens = Vec::with_capacity(DECODE_STEPS);
    let mut input = ids(&PROMPT);
    for step in 0..DECODE_STEPS {
        let logits = model.forward_with_sequence_id(&input, Some(id), &mut [], None);
        let row = last_row(&logits);
        assert!(
            row.iter().all(|v| v.is_finite()),
            "{mode} step {step} produced non-finite logits"
        );
        let next = argmax(&row);
        tokens.push(next);
        input = ids(&[next]);
    }
    model.release_sequence_state_by_id(id);
    tokens
}

/// Under `Int8` the run stays finite and starts on the FP16 run's token.
fn assert_int8_greedy_matches_fp16_first_token<M: LanguageModel>(make: impl Fn() -> M) {
    let fp16 = greedy_tokens(&make(), KVCacheMode::Fp16);
    let int8 = greedy_tokens(&make(), KVCacheMode::Int8);
    assert_eq!(
        int8[0], fp16[0],
        "Int8 KV changed the first greedy token (fp16 {fp16:?}, int8 {int8:?})"
    );
}

fn all_int8(modes: &[KVCacheMode], n_layers: usize) -> bool {
    modes.len() == n_layers && modes.iter().all(|mode| *mode == KVCacheMode::Int8)
}

#[test]
fn exaone4_sequence_caches_take_the_injected_kv_mode() {
    let model = exaone();
    let modes = int8_sequence_modes(&model, |m, id| m.sequence_kv_modes(id));
    assert!(all_int8(&modes, model.num_layers()), "got {modes:?}");
}

#[test]
fn exaone4_int8_greedy_run_matches_fp16_first_token() {
    assert_int8_greedy_matches_fp16_first_token(exaone);
}

#[test]
fn gpt_oss_sequence_caches_take_the_injected_kv_mode() {
    let model = super::gpt_oss::batching_tests::tiny(false);
    let modes = int8_sequence_modes(&model, |m, id| m.sequence_kv_modes(id));
    assert!(all_int8(&modes, model.num_layers()), "got {modes:?}");
}

#[test]
fn gpt_oss_int8_greedy_run_matches_fp16_first_token() {
    assert_int8_greedy_matches_fp16_first_token(|| super::gpt_oss::batching_tests::tiny(false));
}

#[test]
fn ministral3_sequence_caches_take_the_injected_kv_mode() {
    let model = ministral();
    let modes = int8_sequence_modes(&model, |m, id| m.sequence_kv_modes(id));
    assert!(all_int8(&modes, model.num_layers()), "got {modes:?}");
}

#[test]
fn ministral3_int8_greedy_run_matches_fp16_first_token() {
    assert_int8_greedy_matches_fp16_first_token(ministral);
}

#[test]
fn step3p5_sequence_caches_take_the_injected_kv_mode() {
    let model = step();
    let modes = int8_sequence_modes(&model, |m, id| m.sequence_kv_modes(id));
    assert!(all_int8(&modes, model.num_layers()), "got {modes:?}");
}

#[test]
fn step3p5_int8_greedy_run_matches_fp16_first_token() {
    assert_int8_greedy_matches_fp16_first_token(step);
}

#[test]
fn mellum_sequence_caches_take_the_injected_kv_mode() {
    let model = mellum();
    let modes = int8_sequence_modes(&model, |m, id| m.sequence_kv_modes(id));
    assert!(all_int8(&modes, model.num_layers()), "got {modes:?}");
}

#[test]
fn mellum_int8_greedy_run_matches_fp16_first_token() {
    assert_int8_greedy_matches_fp16_first_token(mellum);
}

#[test]
fn laguna_sequence_caches_take_the_injected_kv_mode() {
    let model = laguna();
    let modes = int8_sequence_modes(&model, |m, id| m.sequence_kv_modes(id));
    assert!(all_int8(&modes, model.num_layers()), "got {modes:?}");
}

#[test]
fn laguna_int8_greedy_run_matches_fp16_first_token() {
    assert_int8_greedy_matches_fp16_first_token(laguna);
}

/// The table is per layer: a sequence opened after injection takes each
/// layer's own entry, and the fallback slot rebuilt by the injection does
/// too (the single-sequence `forward` the CLI warmup used to take).
#[test]
fn exaone4_mixed_table_is_applied_per_layer() {
    let model = exaone();
    let table = vec![KVCacheMode::Int8, KVCacheMode::Fp16];
    assert_eq!(table.len(), model.num_layers());
    model.set_kv_cache_layer_modes(table.clone());
    let id = SequenceId::from_raw(3);
    model.prepare_sequence_state(id);
    assert_eq!(model.sequence_kv_modes(id), table);
    model.release_sequence_state_by_id(id);
}
