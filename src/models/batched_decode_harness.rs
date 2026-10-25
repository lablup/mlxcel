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

use mlxcel_core::cache::SequenceId;
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
    assert_cache_advance(&caches, &before);
}

pub(crate) fn assert_single_row_batch_is_single_sequence_forward<M: LanguageModel>(
    model: &M,
    prompt: &[i32],
    next_token: i32,
) {
    let expected = sequential_decode(model, &[prompt], &[next_token], None);
    let (actual, caches, before) = batched_decode(model, &[prompt], &[next_token], None);
    assert_rows_close(&actual, &expected, 0.0);
    assert_cache_advance(&caches, &before);
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
    assert_cache_advance(&caches, &before);
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
    assert_cache_advance(&caches, &before);
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
        assert!(
            mlxcel_core::item_bool(&close),
            "row {row} differs beyond {tol}"
        );
    }
}

fn assert_cache_advance(caches: &[Vec<KVCache>], before: &[Vec<i32>]) {
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
