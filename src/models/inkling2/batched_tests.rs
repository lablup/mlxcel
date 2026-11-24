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

//! Native batched decode and padded prefill on the tiny looped Inkling 2
//! (issue #3377). The tiny model computes in f32 and its channel counts keep
//! every Metal kernel off, so batched rows must equal their single-row runs
//! to f32 rounding.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel};
use mlxcel_core::layers::KVCache;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Inkling2Model;
use super::tiny_tests::model;

/// Batched rows and their single-row runs differ only by f32 rounding of
/// matmuls with a different row count (observed up to 3e-5 over the 10
/// visits); the bound the other tiny tests use for differently shaped
/// forwards. A padding row read by a convolution or a key from another row
/// moves logits by 1e-2 or more.
const TOLERANCE: f32 = 2e-4;
const VOCAB: i32 = 30;

fn tokens(count: usize, seed: usize) -> Vec<i32> {
    (0..count)
        .map(|i| ((i * 5 + seed * 11 + 2) % VOCAB as usize) as i32)
        .collect()
}

fn ids(values: &[i32], rows: usize) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(values, &[rows as i32, (values.len() / rows) as i32])
}

fn values(x: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(x);
    array_to_vec_f32(&mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn fresh(count: u64, base: u64) -> Vec<SequenceId> {
    (0..count).map(|i| SequenceId::from_raw(base + i)).collect()
}

fn single(model: &Inkling2Model, id: SequenceId, input: &[i32]) -> Vec<f32> {
    let mut caches = vec![KVCache::new()];
    values(&model.forward_with_sequence_id(&ids(input, 1), Some(id), &mut caches, None))
}

fn batched(model: &Inkling2Model, seqs: &[SequenceId], input: &[i32]) -> Vec<f32> {
    let mut storage: Vec<Vec<KVCache>> = seqs.iter().map(|_| vec![KVCache::new()]).collect();
    let mut caches: Vec<&mut [KVCache]> = storage.iter_mut().map(Vec::as_mut_slice).collect();
    values(&model.forward_batched_with_ids(&ids(input, seqs.len()), Some(seqs), &mut caches, None))
}

/// `[rows, length, vocab]` logits split per row.
fn split_rows(logits: &[f32], rows: usize) -> Vec<Vec<f32>> {
    logits
        .chunks(logits.len() / rows)
        .map(<[f32]>::to_vec)
        .collect()
}

#[test]
fn reports_native_batched_decode_and_padded_prefill() {
    let model = model(3, false);
    assert_eq!(model.batched_decode_kind(), BatchedDecodeKind::Native);
    assert!(model.supports_batched_prefill());
    assert!(model.supports_padded_prefill());
}

/// Rows at 3, 12 and 30 tokens (the last past the 4-token window, the first
/// two with convolution histories shorter than the margin), then: batched
/// steps, a trim of one row after a batched step, a row leaving mid-batch.
#[test]
fn batched_decode_rows_equal_single_row_decodes() {
    let model = model(3, false);
    let prompts = [tokens(3, 1), tokens(12, 2), tokens(30, 3)];
    let batch = fresh(3, 100);
    let reference = fresh(3, 200);
    for (row, prompt) in prompts.iter().enumerate() {
        let _ = single(&model, batch[row], prompt);
        let _ = single(&model, reference[row], prompt);
    }
    let mut next: Vec<i32> = vec![4, 9, 17];
    let check = |step: usize, live: &[usize], next: &[i32]| {
        let rows: Vec<SequenceId> = live.iter().map(|&row| batch[row]).collect();
        let out = split_rows(&batched(&model, &rows, next), live.len());
        for (slot, &row) in live.iter().enumerate() {
            let expected = single(&model, reference[row], &[next[slot]]);
            let diff = max_diff(&out[slot], &expected);
            assert!(diff < TOLERANCE, "step {step} row {row}: {diff}");
        }
    };
    for step in 0..6 {
        check(step, &[0, 1, 2], &next);
        next = next.iter().map(|t| (t * 3 + 1) % VOCAB).collect();
    }
    // A rejected tail on row 1 only.
    model.trim_state(Some(batch[1]), 2).unwrap();
    model.trim_state(Some(reference[1]), 2).unwrap();
    for step in 6..9 {
        check(step, &[0, 1, 2], &next);
        next = next.iter().map(|t| (t * 7 + 2) % VOCAB).collect();
    }
    // Row 0 finishes; the other two keep decoding as a batch of two.
    model.release_sequence_state_by_id(batch[0]);
    for step in 9..12 {
        check(step, &[1, 2], &next[1..]);
        next = next.iter().map(|t| (t * 5 + 3) % VOCAB).collect();
    }
}

/// A one-row batch keeps the single-sequence graph.
#[test]
fn one_row_batch_is_the_single_sequence_forward() {
    let model = model(3, false);
    let (a, b) = (SequenceId::from_raw(300), SequenceId::from_raw(301));
    let prompt = tokens(9, 4);
    let first = batched(&model, &[a], &prompt);
    assert_eq!(first, single(&model, b, &prompt));
    assert_eq!(batched(&model, &[a], &[5]), single(&model, b, &[5]));
}

/// Visit caches of `id`: every KV offset and visible key/value, and the
/// newest `kernel - 1 + min(len, margin)` rows of each convolution history
/// (a padded cohort keeps older rows a single prefill drops; the next decode
/// step drops them too).
fn state(model: &Inkling2Model, id: SequenceId, prompt_len: i32) -> Vec<(i32, Vec<f32>)> {
    let tail = 3 + prompt_len.min(super::conv::CONV_HISTORY_MARGIN);
    model
        .sequence_state
        .with_sequence_state_ref(id, |caches| {
            let mut out = Vec::new();
            for cache in caches.iter() {
                let (keys, vals) = cache.kv.visible_state().expect("prefilled");
                out.push((cache.kv.offset, values(&keys)));
                out.push((cache.kv.offset, values(&vals)));
                for history in cache.conv.iter().flatten() {
                    let rows = mlxcel_core::array_shape(history)[1];
                    assert!(rows >= tail, "history keeps {rows} rows, needs {tail}");
                    let newest = mlxcel_core::utils::slice_axis(history, 1, rows - tail, rows);
                    out.push((rows - tail, values(&newest)));
                }
            }
            out
        })
        .expect("sequence state exists")
}

/// The scheduler's cohort: rows of 5, 23 and 40 tokens right-padded with
/// token 0 to 48, one forward, the last real row's logits, then a per-row
/// trim of the padding (43 and 25 rows are past the convolution margin).
#[test]
fn padded_cohort_prefill_equals_unpadded_prefill() {
    let model = model(3, false);
    let lens = [5usize, 23, 40];
    let padded = 48usize;
    let prompts: Vec<Vec<i32>> = lens
        .iter()
        .enumerate()
        .map(|(row, &len)| tokens(len, row + 5))
        .collect();
    let cohort = fresh(3, 400);
    let reference = fresh(3, 500);
    let mut flat = Vec::new();
    for prompt in &prompts {
        flat.extend_from_slice(prompt);
        flat.extend(std::iter::repeat_n(0, padded - prompt.len()));
    }
    let logits = split_rows(&batched(&model, &cohort, &flat), 3);
    let width = VOCAB as usize;
    for (row, prompt) in prompts.iter().enumerate() {
        let expected = single(&model, reference[row], prompt);
        let len = prompt.len();
        let diff = max_diff(
            &logits[row][(len - 1) * width..len * width],
            &expected[(len - 1) * width..],
        );
        assert!(diff < TOLERANCE, "row {row} last prompt logits: {diff}");
        model
            .trim_state(Some(cohort[row]), (padded - len) as i32)
            .unwrap();
        let (got, want) = (
            state(&model, cohort[row], len as i32),
            state(&model, reference[row], len as i32),
        );
        assert_eq!(got.len(), want.len());
        for (part, ((got_offset, got), (want_offset, want))) in got.iter().zip(&want).enumerate() {
            // Six parts per visit: keys and values (tagged with the KV
            // offset), then four convolution tails.
            if part % 6 < 2 {
                assert_eq!(
                    got_offset,
                    want_offset,
                    "row {row} visit {} offset",
                    part / 6
                );
            }
            let diff = max_diff(got, want);
            assert!(diff < TOLERANCE, "row {row} state part {part}: {diff}");
        }
    }
    // Decoding on from the trimmed cohort matches decoding on unpadded.
    let mut next = vec![3, 8, 21];
    for step in 0..4 {
        let out = split_rows(&batched(&model, &cohort, &next), 3);
        for row in 0..3 {
            let expected = single(&model, reference[row], &[next[row]]);
            let diff = max_diff(&out[row], &expected);
            assert!(diff < TOLERANCE, "decode step {step} row {row}: {diff}");
        }
        next = next.iter().map(|t| (t * 3 + 2) % VOCAB).collect();
    }
}
