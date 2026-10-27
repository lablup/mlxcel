// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Stack recurrent state along the sequence-row axis without changing its recurrence.
//! Used by: Mamba1/2, Jamba, LFM2, RecurrentGemma, RWKV7, Nemotron-H,
//! GraniteMoeHybrid, Falcon-H1, PLaMo2, Qwen3-Next, Qwen 3.5, Kimi Linear,
//! Kimi K3, and Bailing MoE Linear.

use super::gated_delta::GatedDeltaCache;
use super::mamba2::Mamba2Cache;
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) enum StackedRows {
    AllNone,
    Stacked(UniquePtr<MlxArray>),
    Mixed,
}

impl StackedRows {
    /// Call only after preflighting the entire batch for mixed initialization.
    pub(crate) fn into_option(self) -> Option<UniquePtr<MlxArray>> {
        match self {
            Self::AllNone => None,
            Self::Stacked(state) => Some(state),
            Self::Mixed => panic!("mixed recurrent state requires per-row forwarding"),
        }
    }
}

pub(crate) fn optional_rows_uniform(rows: &[Option<&MlxArray>]) -> bool {
    let count = rows.iter().filter(|row| row.is_some()).count();
    count == 0 || count == rows.len()
}

pub(crate) fn stack_rows(rows: &[&MlxArray]) -> UniquePtr<MlxArray> {
    assert!(!rows.is_empty(), "cannot stack an empty recurrent batch");
    for row in rows {
        assert_eq!(
            mlxcel_core::array_shape(row).first(),
            Some(&1),
            "recurrent state must have one row"
        );
    }
    mlxcel_core::concatenate_many(rows, 0)
}

pub(crate) fn stack_optional_rows(rows: &[Option<&MlxArray>]) -> StackedRows {
    if !optional_rows_uniform(rows) {
        StackedRows::Mixed
    } else if rows.iter().all(Option::is_none) {
        StackedRows::AllNone
    } else {
        StackedRows::Stacked(stack_rows(
            &rows.iter().flatten().copied().collect::<Vec<_>>(),
        ))
    }
}

pub(crate) fn split_rows(stacked: &MlxArray, batch: usize) -> Vec<UniquePtr<MlxArray>> {
    assert!(batch > 0, "cannot split an empty recurrent batch");
    assert_eq!(
        mlxcel_core::array_shape(stacked).first().copied(),
        Some(batch as i32),
        "recurrent batch size mismatch"
    );
    (0..batch)
        .map(|row| {
            let slice = mlxcel_core::utils::slice_axis(stacked, 0, row as i32, row as i32 + 1);
            // Copy materializes each row, so a sequence cannot pin the other rows' buffer.
            mlxcel_core::copy(&slice)
        })
        .collect()
}

pub(crate) fn stack_mamba2(rows: &[&Mamba2Cache]) -> Option<Mamba2Cache> {
    let conv: Vec<_> = rows
        .iter()
        .map(|row| row.conv_state.as_ref().and_then(|s| s.as_ref()))
        .collect();
    let ssm: Vec<_> = rows
        .iter()
        .map(|row| row.ssm_state.as_ref().and_then(|s| s.as_ref()))
        .collect();
    if !optional_rows_uniform(&conv) || !optional_rows_uniform(&ssm) {
        return None;
    }
    Some(Mamba2Cache {
        conv_state: stack_optional_rows(&conv).into_option(),
        ssm_state: stack_optional_rows(&ssm).into_option(),
    })
}

pub(crate) fn split_mamba2(state: &Mamba2Cache, batch: usize) -> Vec<Mamba2Cache> {
    let split = |state: &Option<UniquePtr<MlxArray>>| match state {
        Some(state) => split_rows(state, batch)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>(),
        None => (0..batch).map(|_| None).collect(),
    };
    split(&state.conv_state)
        .into_iter()
        .zip(split(&state.ssm_state))
        .map(|(conv_state, ssm_state)| Mamba2Cache {
            conv_state,
            ssm_state,
        })
        .collect()
}

pub(crate) fn gated_delta_rows_uniform(rows: &[&GatedDeltaCache]) -> bool {
    optional_rows_uniform(
        &rows
            .iter()
            .map(|row| row.conv_state.as_deref())
            .collect::<Vec<_>>(),
    ) && optional_rows_uniform(
        &rows
            .iter()
            .map(|row| row.state_cache.as_deref())
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn stack_gated_delta(rows: &[&GatedDeltaCache]) -> Option<GatedDeltaCache> {
    if !gated_delta_rows_uniform(rows) {
        return None;
    }
    Some(GatedDeltaCache {
        conv_state: stack_optional_rows(
            &rows
                .iter()
                .map(|row| row.conv_state.as_deref())
                .collect::<Vec<_>>(),
        )
        .into_option(),
        state_cache: stack_optional_rows(
            &rows
                .iter()
                .map(|row| row.state_cache.as_deref())
                .collect::<Vec<_>>(),
        )
        .into_option(),
        // Recurrence math does not read the scalar offset. Callers restore
        // each row's own monotonic value after the batched step.
        offset: 0,
    })
}

pub(crate) fn split_gated_delta(
    state: &GatedDeltaCache,
    previous_offsets: &[i32],
) -> Vec<GatedDeltaCache> {
    let batch = previous_offsets.len();
    let split = |state: &Option<UniquePtr<MlxArray>>| match state {
        Some(state) => split_rows(state, batch)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>(),
        None => (0..batch).map(|_| None).collect(),
    };
    split(&state.conv_state)
        .into_iter()
        .zip(split(&state.state_cache))
        .zip(previous_offsets)
        .map(|((conv_state, state_cache), offset)| GatedDeltaCache {
            conv_state,
            state_cache,
            offset: offset + 1,
        })
        .collect()
}

#[cfg(test)]
#[path = "recurrent_batch_tests.rs"]
mod tests;
