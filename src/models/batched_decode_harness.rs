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

use std::collections::HashSet;

use mlxcel_core::cache::{SequenceId, SequenceStateBackend};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) fn assert_batched_decode_matches_sequential<M: LanguageModel>(
    model: &M,
    prompts: &[&[i32]],
    next_tokens: &[i32],
    tol: f64,
) {
    validate_rows(prompts, next_tokens);
    let expected = sequential_decode(model, prompts, next_tokens, None);
    let (actual, caches, before) = batched_decode(model, prompts, next_tokens, None);
    assert_rows_close(&actual, &expected, tol);
    assert_cache_advance(model.sequence_state_layout().backend, &caches, &before);
}

pub(crate) fn assert_single_row_batch_is_single_sequence_forward<M: LanguageModel>(
    model: &M,
    prompt: &[i32],
    next_token: i32,
) {
    let expected = sequential_decode(model, &[prompt], &[next_token], None);
    let (actual, caches, before) = batched_decode(model, &[prompt], &[next_token], None);
    assert_rows_close(&actual, &expected, 0.0);
    assert_cache_advance(model.sequence_state_layout().backend, &caches, &before);
}

pub(crate) fn assert_empty_batch_returns_empty_logits<M: LanguageModel>(model: &M) {
    let input = mlxcel_core::from_slice_i32(&[], &[0, 1]);
    let mut caches: Vec<&mut [KVCache]> = Vec::new();
    let logits = model.forward_batched(&input, &mut caches, None);
    assert_eq!(mlxcel_core::array_shape(&logits), vec![0, 1, 1]);
}

#[allow(dead_code)]
pub(crate) fn assert_batched_decode_matches_sequential_with_ids<M: LanguageModel>(
    model: &M,
    prompts: &[&[i32]],
    next_tokens: &[i32],
    tol: f64,
) {
    validate_rows(prompts, next_tokens);
    let expected_ids = sequence_ids(10_000, prompts.len());
    let actual_ids = sequence_ids(20_000, prompts.len());
    let expected = sequential_decode(model, prompts, next_tokens, Some(&expected_ids));
    let (actual, caches, before) = batched_decode(model, prompts, next_tokens, Some(&actual_ids));
    assert_rows_close(&actual, &expected, tol);
    assert_cache_advance(model.sequence_state_layout().backend, &caches, &before);
    release_ids(model, &expected_ids);
    release_ids(model, &actual_ids);
}

#[allow(dead_code)]
pub(crate) fn assert_single_row_batch_is_single_sequence_forward_with_ids<M: LanguageModel>(
    model: &M,
    prompt: &[i32],
    next_token: i32,
) {
    let expected_ids = sequence_ids(30_000, 1);
    let actual_ids = sequence_ids(40_000, 1);
    let expected = sequential_decode(model, &[prompt], &[next_token], Some(&expected_ids));
    let (actual, caches, before) =
        batched_decode(model, &[prompt], &[next_token], Some(&actual_ids));
    assert_rows_close(&actual, &expected, 0.0);
    assert_cache_advance(model.sequence_state_layout().backend, &caches, &before);
    release_ids(model, &expected_ids);
    release_ids(model, &actual_ids);
}

fn validate_rows(prompts: &[&[i32]], next_tokens: &[i32]) {
    assert!(
        prompts.len() >= 2,
        "batched decode comparison needs at least two rows"
    );
    assert_eq!(prompts.len(), next_tokens.len());
    assert!(prompts.iter().all(|prompt| !prompt.is_empty()));
    if prompts.len() > 1 {
        let lengths: HashSet<usize> = prompts.iter().map(|prompt| prompt.len()).collect();
        assert_eq!(
            lengths.len(),
            prompts.len(),
            "prompts must have pairwise distinct lengths"
        );
    }
}

#[allow(dead_code)]
fn sequence_ids(base: u64, rows: usize) -> Vec<SequenceId> {
    (0..rows)
        .map(|row| SequenceId::from_raw(base + row as u64))
        .collect()
}

#[allow(dead_code)]
fn release_ids<M: LanguageModel>(model: &M, ids: &[SequenceId]) {
    for &id in ids {
        model.release_sequence_state_by_id(id);
    }
}

fn prefill<M: LanguageModel>(
    model: &M,
    prompt: &[i32],
    seq_id: Option<SequenceId>,
) -> Vec<KVCache> {
    if let Some(id) = seq_id {
        model.prepare_sequence_state(id);
    }
    let mut caches = model.make_caches();
    let input = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let logits = model.forward_with_sequence_id(&input, seq_id, &mut caches, None);
    mlxcel_core::eval(&logits);
    caches
}

fn sequential_decode<M: LanguageModel>(
    model: &M,
    prompts: &[&[i32]],
    next_tokens: &[i32],
    seq_ids: Option<&[SequenceId]>,
) -> Vec<UniquePtr<MlxArray>> {
    prompts
        .iter()
        .zip(next_tokens)
        .enumerate()
        .map(|(row, (prompt, &token))| {
            let seq_id = seq_ids.map(|ids| ids[row]);
            let mut caches = prefill(model, prompt, seq_id);
            let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            let logits = model.forward_with_sequence_id(&input, seq_id, &mut caches, None);
            mlxcel_core::eval(&logits);
            logits
        })
        .collect()
}

fn batched_decode<M: LanguageModel>(
    model: &M,
    prompts: &[&[i32]],
    next_tokens: &[i32],
    seq_ids: Option<&[SequenceId]>,
) -> (UniquePtr<MlxArray>, Vec<Vec<KVCache>>, Vec<Vec<i32>>) {
    let mut caches: Vec<Vec<KVCache>> = prompts
        .iter()
        .enumerate()
        .map(|(row, prompt)| prefill(model, prompt, seq_ids.map(|ids| ids[row])))
        .collect();
    let before = caches
        .iter()
        .map(|layers| layers.iter().map(|cache| cache.offset).collect())
        .collect();
    let input = mlxcel_core::from_slice_i32(next_tokens, &[prompts.len() as i32, 1]);
    let logits = {
        let mut rows: Vec<&mut [KVCache]> = caches.iter_mut().map(Vec::as_mut_slice).collect();
        model.forward_batched_with_ids(&input, seq_ids, &mut rows, None)
    };
    mlxcel_core::eval(&logits);
    (logits, caches, before)
}

fn assert_rows_close(actual: &MlxArray, expected: &[UniquePtr<MlxArray>], tol: f64) {
    let shape = mlxcel_core::array_shape(actual);
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], expected.len() as i32);
    assert_eq!(shape[1], 1);
    for (row, expected) in expected.iter().enumerate() {
        let actual_row =
            mlxcel_core::slice(actual, &[row as i32, 0, 0], &[row as i32 + 1, 1, shape[2]]);
        let close = mlxcel_core::allclose(&actual_row, expected, tol, tol);
        let actual_values = mlxcel_core::utils::array_to_vec_f32(&actual_row);
        let expected_values = mlxcel_core::utils::array_to_vec_f32(expected);
        let max_abs = actual_values
            .iter()
            .zip(&expected_values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_rel = actual_values
            .iter()
            .zip(&expected_values)
            .map(|(a, b)| (a - b).abs() / b.abs().max(f32::EPSILON))
            .fold(0.0_f32, f32::max);
        assert!(
            mlxcel_core::item_bool(&close),
            "row {row} differs beyond {tol}: max_abs={max_abs}, max_rel={max_rel}"
        );
    }
}

fn assert_cache_advance(
    backend: SequenceStateBackend,
    caches: &[Vec<KVCache>],
    before: &[Vec<i32>],
) {
    // Model-owned families expose compatibility placeholders, not their live
    // state. Their model-specific tests must check internal offsets instead.
    if backend == SequenceStateBackend::ModelOwned {
        return;
    }
    for (row, (layers, offsets)) in caches.iter().zip(before).enumerate() {
        for (layer, (cache, &offset)) in layers.iter().zip(offsets).enumerate() {
            assert_eq!(
                cache.offset,
                offset + 1,
                "row {row} layer {layer} did not advance once"
            );
        }
    }
}

/// Exercise isolated and batched decode across different cache wrap/pooling positions.
pub(crate) fn assert_model_owned_multistep<M: LanguageModel>(make: impl Fn() -> M) {
    assert_model_owned_multistep_with_lengths(make, [2, 7, 19]);
}

pub(crate) fn assert_model_owned_multistep_with_lengths<M: LanguageModel>(
    make: impl Fn() -> M,
    lengths: [i32; 3],
) {
    let prompts: Vec<Vec<i32>> = lengths
        .into_iter()
        .map(|n| (0..n).map(|i| i % 7).collect())
        .collect();
    let refs: Vec<&[i32]> = prompts.iter().map(Vec::as_slice).collect();
    let model = make();
    assert!(model.supports_batching());
    assert_eq!(
        model.batched_decode_kind(),
        mlxcel_core::generate::BatchedDecodeKind::Native
    );
    assert_eq!(
        model.sequence_state_layout().backend,
        mlxcel_core::cache::SequenceStateBackend::ModelOwned
    );
    let input = mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]);
    let mut a = model.make_caches();
    let mut b = model.make_caches();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.forward_batched_with_ids(&input, None, &mut [&mut a, &mut b], None)
        }))
        .is_err(),
        "model-owned multi-row calls must require sequence ids"
    );
    assert_batched_decode_matches_sequential_with_ids(&model, &refs, &[2, 3, 4], 1e-5);
    assert_single_row_batch_is_single_sequence_forward_with_ids(&model, &prompts[0], 2);
    assert_empty_batch_returns_empty_logits(&model);
    for batched in [false, true] {
        let model = make();
        let solo: Vec<_> = (0..3).map(|_| make()).collect();
        let ids = sequence_ids(90_000, 3);
        let mut caches: Vec<_> = prompts
            .iter()
            .zip(&ids)
            .map(|(p, id)| prefill(&model, p, Some(*id)))
            .collect();
        let mut solo_caches: Vec<_> = solo
            .iter()
            .zip(&prompts)
            .zip(&ids)
            .map(|((m, p), id)| prefill(m, p, Some(*id)))
            .collect();
        for step in 0..6 {
            // Fallback cache initialization and admission of another sequence must not reset live rows.
            let _ = model.make_caches();
            let newcomer = SequenceId::from_raw(99_000 + step);
            model.prepare_sequence_state(newcomer);
            let input = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
            let _ = model.forward_with_sequence_id(&input, Some(newcomer), &mut [], None);
            model.release_sequence_state_by_id(newcomer);
            let tokens: Vec<_> = (0..3).map(|i| ((step + i) % 7) as i32).collect();
            let expected: Vec<_> = solo
                .iter()
                .zip(&ids)
                .zip(&mut solo_caches)
                .zip(&tokens)
                .map(|(((m, id), c), token)| {
                    let input = mlxcel_core::from_slice_i32(&[*token], &[1, 1]);
                    m.forward_with_sequence_id(&input, Some(*id), c, None)
                })
                .collect();
            if batched {
                let input = mlxcel_core::from_slice_i32(&tokens, &[3, 1]);
                let mut rows: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
                let actual = model.forward_batched_with_ids(&input, Some(&ids), &mut rows, None);
                assert_rows_absolute(&actual, &expected, 1e-5);
            } else {
                for row in [2, 0, 1] {
                    let input = mlxcel_core::from_slice_i32(&[tokens[row]], &[1, 1]);
                    let actual = model.forward_with_sequence_id(
                        &input,
                        Some(ids[row]),
                        &mut caches[row],
                        None,
                    );
                    assert_rows_absolute(&actual, &[mlxcel_core::copy(&expected[row])], 1e-5);
                }
            }
        }
        release_ids(&model, &ids);
    }
}

fn assert_rows_absolute(actual: &MlxArray, expected: &[UniquePtr<MlxArray>], tolerance: f32) {
    for (row, expected) in expected.iter().enumerate() {
        let actual = mlxcel_core::utils::slice_axis(actual, 0, row as i32, row as i32 + 1);
        let difference = mlxcel_core::subtract(&actual, expected);
        let values = mlxcel_core::utils::array_to_vec_f32(&difference);
        assert!(
            values
                .iter()
                .all(|value| value.is_finite() && value.abs() <= tolerance),
            "row {row} exceeds absolute tolerance {tolerance}: {values:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "did not advance once")]
    fn scheduler_owned_cache_must_still_advance() {
        assert_cache_advance(
            SequenceStateBackend::DenseKvCache,
            &[vec![KVCache::new()]],
            &[vec![0]],
        );
    }

    #[test]
    fn model_owned_placeholder_need_not_advance() {
        assert_cache_advance(
            SequenceStateBackend::ModelOwned,
            &[vec![KVCache::new()]],
            &[vec![0]],
        );
    }
}
