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

fn gemma3(sliding_window: i32) -> super::gemma3::Gemma3Wrapper {
    super::gemma3::gemma3_tests::snapshot_prompt_cache::build_wrapper_with_window(sliding_window)
}

/// The fixtures put a sliding layer at index 0 and a full-attention layer at
/// index 1 (Gemma 3 `sliding_window_pattern` 2, GPT-OSS `layer_types`).
const SLIDING_THEN_FULL: usize = 2;

/// `kv_cache_layer_modes()` stays the injected table, while
/// `kv_cache_storage_modes()` reports what each cache stores: a sliding
/// `RotatingKVCache` keeps FP16 under `Int8` (issue #2783).
fn assert_sliding_layer_reports_fp16_storage_under_int8<M: LanguageModel>(model: &M) {
    assert_eq!(model.num_layers(), SLIDING_THEN_FULL);
    let int8 = vec![KVCacheMode::Int8; SLIDING_THEN_FULL];
    model.set_kv_cache_layer_modes(int8.clone());
    assert_eq!(model.kv_cache_layer_modes(), Some(int8));
    assert_eq!(
        model.kv_cache_storage_modes(),
        Some(vec![Some(KVCacheMode::Fp16), Some(KVCacheMode::Int8)]),
        "the sliding layer stores FP16, so it must not be reported as Int8"
    );
}

#[test]
fn gemma3_sliding_layers_report_fp16_storage_under_int8() {
    assert_sliding_layer_reports_fp16_storage_under_int8(&gemma3(8));
}

#[test]
fn gpt_oss_sliding_layers_report_fp16_storage_under_int8() {
    assert_sliding_layer_reports_fp16_storage_under_int8(&super::gpt_oss::batching_tests::tiny(
        false,
    ));
}

/// An aligned `Turbo4Asym` window is the one sliding case that really
/// quantizes; a misaligned one stores FP16.
#[test]
fn gemma3_storage_modes_follow_the_turbo4_asym_window_alignment() {
    let turbo = vec![KVCacheMode::Turbo4Asym; SLIDING_THEN_FULL];
    let aligned = gemma3(512);
    aligned.set_kv_cache_layer_modes(turbo.clone());
    assert_eq!(
        aligned.kv_cache_storage_modes(),
        Some(turbo.iter().copied().map(Some).collect())
    );

    let misaligned = gemma3(100);
    misaligned.set_kv_cache_layer_modes(turbo);
    assert_eq!(
        misaligned.kv_cache_storage_modes(),
        Some(vec![Some(KVCacheMode::Fp16), Some(KVCacheMode::Turbo4Asym)])
    );
}

/// `fp16+turbo4` on a sliding window that is not a multiple of
/// `turbo::BLOCK_SIZE` used to panic in `RotatingKVCache::new_with_mode`
/// while building the sequence's caches. The sliding layer now falls back to
/// FP16, the full layer keeps `Turbo4Asym`, and the forward stays finite.
fn assert_misaligned_turbo4_asym_window_falls_back_to_fp16<M: LanguageModel>(
    model: &M,
    kv_modes: impl Fn(&M, SequenceId) -> Vec<KVCacheMode>,
) {
    assert_eq!(model.num_layers(), SLIDING_THEN_FULL);
    model.set_kv_cache_layer_modes(vec![KVCacheMode::Turbo4Asym; SLIDING_THEN_FULL]);
    let id = SequenceId::from_raw(13);
    model.prepare_sequence_state(id);
    let logits = model.forward_with_sequence_id(&ids(&PROMPT), Some(id), &mut [], None);
    let row = last_row(&logits);
    assert!(
        row.iter().all(|v| v.is_finite()),
        "non-finite logits: {row:?}"
    );
    assert_eq!(
        kv_modes(model, id),
        vec![KVCacheMode::Fp16, KVCacheMode::Turbo4Asym],
        "the sliding cache must fall back to FP16 and the full cache keep Turbo4Asym"
    );
    model.release_sequence_state_by_id(id);
}

#[test]
fn turbo4_asym_on_a_misaligned_window_falls_back_to_fp16() {
    assert_misaligned_turbo4_asym_window_falls_back_to_fp16(&gemma3(100), |m, id| {
        m.sequence_kv_modes(id)
    });
    // The GPT-OSS fixture's window is 4, also not a multiple of 32.
    assert_misaligned_turbo4_asym_window_falls_back_to_fp16(
        &super::gpt_oss::batching_tests::tiny(false),
        |m, id| m.sequence_kv_modes(id),
    );
}

// Issue #2790 item 2: a zero sliding window or pattern is a load error.

#[test]
fn validate_sliding_window_accepts_exactly_the_i32_range_above_zero() {
    use super::model_owned::validate_sliding_window;
    let max = i32::MAX as usize;
    assert_eq!(validate_sliding_window("Fam", 1), Ok(1));
    assert_eq!(validate_sliding_window("Fam", max), Ok(i32::MAX));
    for window in [0, max + 1] {
        let err = validate_sliding_window("Fam", window).expect_err("out of range");
        assert_eq!(
            err,
            format!("Fam: sliding_window must be between 1 and {max}, got {window}")
        );
    }
}

#[test]
fn gemma3_rejects_zero_sliding_window_pattern_at_load() {
    let err = super::gemma3::gemma3_tests::snapshot_prompt_cache::try_load_with(8, 0)
        .err()
        .expect("a zero pattern must fail the load");
    assert!(
        err.contains("sliding_window_pattern must be positive"),
        "{err}"
    );
}

#[test]
fn sliding_window_zero_is_a_load_error() {
    let gemma3 = super::gemma3::gemma3_tests::snapshot_prompt_cache::try_load_with(0, 2)
        .err()
        .expect("Gemma 3 with a zero window must fail the load");
    assert!(
        gemma3.contains("Gemma 3") && gemma3.contains("got 0"),
        "{gemma3}"
    );
    let gpt_oss = super::gpt_oss::batching_tests::tiny_model_with_window(false, 0)
        .err()
        .expect("GPT-OSS with a zero window must fail the load");
    assert!(
        gpt_oss.contains("GPT-OSS") && gpt_oss.contains("got 0"),
        "{gpt_oss}"
    );
}

/// A config whose layers are all full attention never builds a rotating
/// cache, so its window is not checked.
#[test]
fn a_zero_window_without_sliding_layers_still_loads() {
    super::gemma3::gemma3_tests::snapshot_prompt_cache::try_load_with(0, 1)
        .map(|_| ())
        .expect("pattern 1 makes every layer global");
}
