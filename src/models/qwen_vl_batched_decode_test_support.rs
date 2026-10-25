// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Clone, Copy)]
pub(crate) enum RowPositionState {
    Text,
    Delta(i32),
    Stored { delta: i32 },
}

const PROMPTS: [&[i32]; 3] = [&[1, 2], &[3, 4, 5], &[6, 7, 8, 9]];
const NEXT: [i32; 3] = [10, 11, 12];

pub(crate) fn assert_native_qwen_vl_contract<M>(
    model: &M,
    mut set_state: impl FnMut(SequenceId, RowPositionState, usize),
) where
    M: LanguageModel,
{
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential_with_ids(
        model, &PROMPTS, &NEXT, 2e-4,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward_with_ids(
        model,
        &[1, 3, 5],
        7,
    );

    let empty = mlxcel_core::from_slice_i32(&[], &[0, 1]);
    let ids: [SequenceId; 0] = [];
    let mut caches: Vec<&mut [KVCache]> = Vec::new();
    let logits = model.forward_batched_with_ids(&empty, Some(&ids), &mut caches, None);
    assert_eq!(mlxcel_core::array_shape(&logits), vec![0, 1, 1]);

    // Plain group absent, MRoPE group has three unequal deltas.
    run_case(
        model,
        [
            RowPositionState::Delta(2),
            RowPositionState::Delta(7),
            RowPositionState::Delta(-1),
        ],
        &mut set_state,
        50_000,
    );
    // Mixed groups: a singleton plain row is surrounded by MRoPE rows. This
    // exercises group-size-one exact dispatch and result scattering.
    run_case(
        model,
        [
            RowPositionState::Delta(5),
            RowPositionState::Text,
            RowPositionState::Delta(-2),
        ],
        &mut set_state,
        51_000,
    );
    // Opposite split: singleton MRoPE row and a two-row plain native group.
    run_case(
        model,
        [
            RowPositionState::Text,
            RowPositionState::Delta(4),
            RowPositionState::Text,
        ],
        &mut set_state,
        52_000,
    );
    // A stored tensor covering offset+1 makes the helper return None, keeping
    // the whole batch on exact per-row dispatch.
    run_case(
        model,
        [
            RowPositionState::Stored { delta: 9 },
            RowPositionState::Delta(3),
            RowPositionState::Text,
        ],
        &mut set_state,
        53_000,
    );
}

pub(crate) fn assert_decode_matches_rows_with_global_state<M: LanguageModel>(
    model: &M,
    install_global_state: impl FnOnce(),
) {
    let prompts: [&[i32]; 2] = [&[1, 2], &[3, 4, 5]];
    let next = [9, 10];
    let expected_ids = [SequenceId::from_raw(60_000), SequenceId::from_raw(60_001)];
    let actual_ids = [SequenceId::from_raw(60_100), SequenceId::from_raw(60_101)];
    let mut expected_caches: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(row, prompt)| prefill(model, expected_ids[row], prompt))
        .collect();
    let mut actual_caches: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(row, prompt)| prefill(model, actual_ids[row], prompt))
        .collect();
    install_global_state();

    let mut expected = Vec::new();
    for row in 0..2 {
        let token = mlxcel_core::from_slice_i32(&[next[row]], &[1, 1]);
        let logits = model.forward_with_sequence_id(
            &token,
            Some(expected_ids[row]),
            &mut expected_caches[row],
            None,
        );
        mlxcel_core::eval(&logits);
        expected.push(logits);
    }
    let input = mlxcel_core::from_slice_i32(&next, &[2, 1]);
    let actual = {
        let mut rows: Vec<&mut [KVCache]> =
            actual_caches.iter_mut().map(Vec::as_mut_slice).collect();
        model.forward_batched_with_ids(&input, Some(&actual_ids), &mut rows, None)
    };
    mlxcel_core::eval(&actual);
    assert_rows_close(&actual, &expected, 2e-4);
    for id in expected_ids.into_iter().chain(actual_ids) {
        model.release_sequence_state_by_id(id);
    }
}

fn run_case<M: LanguageModel>(
    model: &M,
    states: [RowPositionState; 3],
    set_state: &mut impl FnMut(SequenceId, RowPositionState, usize),
    id_base: u64,
) {
    let expected_ids: Vec<_> = (0..3)
        .map(|row| SequenceId::from_raw(id_base + row as u64))
        .collect();
    let actual_ids: Vec<_> = (0..3)
        .map(|row| SequenceId::from_raw(id_base + 100 + row as u64))
        .collect();
    let mut expected_caches: Vec<Vec<KVCache>> = PROMPTS
        .iter()
        .enumerate()
        .map(|(row, prompt)| prefill(model, expected_ids[row], prompt))
        .collect();
    let mut actual_caches: Vec<Vec<KVCache>> = PROMPTS
        .iter()
        .enumerate()
        .map(|(row, prompt)| prefill(model, actual_ids[row], prompt))
        .collect();

    for row in 0..3 {
        set_state(expected_ids[row], states[row], PROMPTS[row].len());
        set_state(actual_ids[row], states[row], PROMPTS[row].len());
    }

    let mut expected = Vec::new();
    for row in 0..3 {
        let token = mlxcel_core::from_slice_i32(&[NEXT[row]], &[1, 1]);
        let logits = model.forward_with_sequence_id(
            &token,
            Some(expected_ids[row]),
            &mut expected_caches[row],
            None,
        );
        mlxcel_core::eval(&logits);
        expected.push(logits);
    }

    let tokens = mlxcel_core::from_slice_i32(&NEXT, &[3, 1]);
    let actual = {
        let mut rows: Vec<&mut [KVCache]> =
            actual_caches.iter_mut().map(Vec::as_mut_slice).collect();
        model.forward_batched_with_ids(&tokens, Some(&actual_ids), &mut rows, None)
    };
    mlxcel_core::eval(&actual);
    assert_rows_close(&actual, &expected, 2e-4);

    for row in 0..3 {
        for cache in &actual_caches[row] {
            assert_eq!(cache.offset, PROMPTS[row].len() as i32 + 1);
        }
        model.release_sequence_state_by_id(expected_ids[row]);
        model.release_sequence_state_by_id(actual_ids[row]);
    }
}

fn prefill<M: LanguageModel>(model: &M, seq_id: SequenceId, prompt: &[i32]) -> Vec<KVCache> {
    let mut caches = model.make_caches();
    let input = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let logits = model.forward_with_sequence_id(&input, Some(seq_id), &mut caches, None);
    mlxcel_core::eval(&logits);
    caches
}

fn assert_rows_close(actual: &MlxArray, expected: &[UniquePtr<MlxArray>], tol: f64) {
    let shape = mlxcel_core::array_shape(actual);
    assert_eq!(shape[0], expected.len() as i32);
    for (row, expected) in expected.iter().enumerate() {
        let actual_row =
            mlxcel_core::slice(actual, &[row as i32, 0, 0], &[row as i32 + 1, 1, shape[2]]);
        let close = mlxcel_core::allclose(&actual_row, expected, tol, tol);
        assert!(
            mlxcel_core::item_bool(&close),
            "Qwen VL row {row} differs from isolated decode beyond {tol}"
        );
    }
}
