// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::models::embedding_test_support::mlx_test_guard;

#[test]
fn state_rows_round_trip_at_one_and_three_rows() {
    let _guard = mlx_test_guard();
    for batch in [1, 3] {
        let rows: Vec<_> = (0..batch)
            .map(|row| {
                mlxcel_core::full_f32(&[1, 2, 3], row as f32 + 0.5, mlxcel_core::dtype::FLOAT32)
            })
            .collect();
        let refs: Vec<_> = rows.iter().map(|row| row.as_ref().unwrap()).collect();
        let stacked = stack_rows(&refs);
        assert_eq!(mlxcel_core::array_shape(&stacked), vec![batch as i32, 2, 3]);
        let split = split_rows(&stacked, batch);
        for (actual, expected) in split.iter().zip(&rows) {
            assert_eq!(mlxcel_core::array_shape(actual), vec![1, 2, 3]);
            let delta =
                mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(actual, expected)));
            assert_eq!(mlxcel_core::item_f32(&delta), 0.0);
        }
    }
}

#[test]
fn optional_state_preserves_absence_and_rejects_mixed_rows() {
    let _guard = mlx_test_guard();
    let state = mlxcel_core::ones(&[1, 2, 3], mlxcel_core::dtype::FLOAT32);
    assert!(matches!(
        stack_optional_rows(&[None, None]),
        StackedRows::AllNone
    ));
    assert!(matches!(
        stack_optional_rows(&[Some(&state), None]),
        StackedRows::Mixed
    ));
    let StackedRows::Stacked(stacked) = stack_optional_rows(&[Some(&state), Some(&state)]) else {
        panic!("initialized rows should stack")
    };
    assert_eq!(mlxcel_core::array_shape(&stacked), vec![2, 2, 3]);
}

#[test]
fn mamba_state_stacking_requires_uniform_initialization_for_both_fields() {
    let _guard = mlx_test_guard();
    let empty = Mamba2Cache::new();
    let initialized = Mamba2Cache {
        conv_state: Some(mlxcel_core::ones(&[1, 3, 8], mlxcel_core::dtype::FLOAT32)),
        ssm_state: Some(mlxcel_core::ones(
            &[1, 2, 4, 3],
            mlxcel_core::dtype::FLOAT32,
        )),
    };
    assert!(stack_mamba2(&[&empty, &initialized]).is_none());
    let stacked = stack_mamba2(&[&initialized, &initialized]).unwrap();
    let split = split_mamba2(&stacked, 2);
    assert_eq!(
        mlxcel_core::array_shape(split[0].conv_state.as_ref().unwrap()),
        vec![1, 3, 8]
    );
    assert_eq!(
        mlxcel_core::array_shape(split[1].ssm_state.as_ref().unwrap()),
        vec![1, 2, 4, 3]
    );
}

#[test]
fn state_ring_never_writes_the_last_two_states() {
    // Steady state: each step writes a slot, which becomes the newest.
    let mut recent: [Option<usize>; 2] = [None, None];
    let mut written = Vec::new();
    for _ in 0..9 {
        let slot = super::state_ring_write_slot(&recent);
        assert!(!recent.contains(&Some(slot)), "wrote over {recent:?}");
        written.push(slot);
        recent = [Some(slot), recent[0]];
    }
    assert_eq!(written, [0, 1, 2, 0, 1, 2, 0, 1, 2]);
    // A step that allocated instead (no ring slot) still protects the slot
    // before it, which a two-step rewind can return to.
    let recent = [None, Some(1)];
    assert_ne!(super::state_ring_write_slot(&recent), 1);
    let recent = [Some(2), Some(0)];
    assert_eq!(super::state_ring_write_slot(&recent), 1);
}
