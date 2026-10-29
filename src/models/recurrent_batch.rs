// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Stack recurrent state along the sequence-row axis without changing its recurrence.
//! Used by: Mamba1/2, Jamba, LFM2, RecurrentGemma, RWKV7, Nemotron-H,
//! GraniteMoeHybrid, Falcon-H1, PLaMo2, Qwen3-Next, Qwen 3.5, Kimi Linear,
//! Kimi K3, and Bailing MoE Linear.

use std::collections::HashMap;

use super::gated_delta::GatedDeltaCache;
use super::mamba2::Mamba2Cache;
use mlxcel_core::cache::SequenceId;
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

/// [`split_gated_delta`] without the per-row copy: each row gets a view of
/// the stacked state. Only for state the caller keeps track of through
/// [`GatedDeltaBatchReuse`], which materializes a view before it can outlive
/// the batch it came from.
pub(crate) fn split_gated_delta_views(
    state: &GatedDeltaCache,
    previous_offsets: &[i32],
) -> Vec<GatedDeltaCache> {
    let batch = previous_offsets.len() as i32;
    let view = |state: &Option<UniquePtr<MlxArray>>, row: i32| {
        state
            .as_ref()
            .map(|state| mlxcel_core::utils::slice_axis(state, 0, row, row + 1))
    };
    (0..batch)
        .zip(previous_offsets)
        .map(|(row, offset)| GatedDeltaCache {
            conv_state: view(&state.conv_state, row),
            state_cache: view(&state.state_cache, row),
            offset: offset + 1,
        })
        .collect()
}

struct ReusedGatedDeltaLayer {
    /// Shared handles to the stacked state the views below slice.
    stacked: GatedDeltaCache,
    /// `array_id` of each row's (conv, state) view, in batch order.
    row_ids: Vec<(Option<usize>, Option<usize>)>,
    /// Handles to those views, held so their ids stay unique.
    _views: Vec<UniquePtr<MlxArray>>,
}

/// Stacked gated-delta state carried from one batched decode step to the
/// next.
///
/// Stacking every row's state and copying each row back out costs two
/// launches per row per linear layer and step (about 9,000 launches per step
/// for 64 Qwen3.8-27B rows). When the next step runs the same sequences in the
/// same order and every row still holds the view it was handed, the stacked
/// state is reused as is and rows get views instead of copies.
///
/// A view keeps the whole stacked buffer alive, so a view must not outlive the
/// batch: rows that leave it and rows about to be snapshotted are materialized
/// first ([`Self::owns`] and [`materialize_gated_delta`]). Identity is checked
/// with `array_id` while this struct holds a handle to every current view, so
/// an id cannot be reused by an unrelated array.
#[derive(Default)]
pub(crate) struct GatedDeltaBatchReuse {
    ids: Vec<SequenceId>,
    layers: HashMap<usize, ReusedGatedDeltaLayer>,
}

fn optional_id(array: &Option<UniquePtr<MlxArray>>) -> Option<usize> {
    array.as_ref().map(|array| mlxcel_core::array_id(array))
}

impl GatedDeltaBatchReuse {
    /// The sequences of the last batched step, in batch order.
    pub(crate) fn ids(&self) -> &[SequenceId] {
        &self.ids
    }

    /// Whether `cache` still holds a view handed out by this batch.
    pub(crate) fn owns(&self, cache: &GatedDeltaCache) -> bool {
        let ids: Vec<usize> = [&cache.conv_state, &cache.state_cache]
            .into_iter()
            .filter_map(optional_id)
            .collect();
        self.layers.values().any(|layer| {
            layer.row_ids.iter().any(|(conv, state)| {
                ids.iter()
                    .any(|id| Some(*id) == *conv || Some(*id) == *state)
            })
        })
    }

    /// Forget the carried state and start tracking a batch of `ids`.
    pub(crate) fn reset(&mut self, ids: &[SequenceId]) {
        self.ids = ids.to_vec();
        self.layers.clear();
    }

    /// The stacked state for `layer_idx` when every row of `rows` still holds
    /// the view this batch handed it on the previous step.
    pub(crate) fn take(
        &self,
        layer_idx: usize,
        rows: &[&GatedDeltaCache],
    ) -> Option<GatedDeltaCache> {
        let layer = self.layers.get(&layer_idx)?;
        if layer.row_ids.len() != rows.len()
            || !rows.iter().zip(&layer.row_ids).all(|(row, ids)| {
                (optional_id(&row.conv_state), optional_id(&row.state_cache)) == *ids
            })
        {
            return None;
        }
        let share =
            |a: &Option<UniquePtr<MlxArray>>| a.as_ref().map(|a| mlxcel_core::array_share(a));
        Some(GatedDeltaCache {
            conv_state: share(&layer.stacked.conv_state),
            state_cache: share(&layer.stacked.state_cache),
            offset: 0,
        })
    }

    /// Record the stacked state of `layer_idx` and the views handed to rows.
    pub(crate) fn store(
        &mut self,
        layer_idx: usize,
        stacked: &GatedDeltaCache,
        views: &[GatedDeltaCache],
    ) {
        let share =
            |a: &Option<UniquePtr<MlxArray>>| a.as_ref().map(|a| mlxcel_core::array_share(a));
        let mut row_ids = Vec::with_capacity(views.len());
        let mut handles = Vec::with_capacity(views.len() * 2);
        for view in views {
            row_ids.push((
                optional_id(&view.conv_state),
                optional_id(&view.state_cache),
            ));
            handles.extend(
                [&view.conv_state, &view.state_cache]
                    .into_iter()
                    .flatten()
                    .map(|array| mlxcel_core::array_share(array)),
            );
        }
        self.layers.insert(
            layer_idx,
            ReusedGatedDeltaLayer {
                stacked: GatedDeltaCache {
                    conv_state: share(&stacked.conv_state),
                    state_cache: share(&stacked.state_cache),
                    offset: 0,
                },
                row_ids,
                _views: handles,
            },
        );
    }
}

/// Replace a row's state views with copies, scheduled for evaluation, so the
/// row stops holding the stacked buffer the views slice.
pub(crate) fn materialize_gated_delta(cache: &mut GatedDeltaCache) {
    let mut pending: Vec<*const MlxArray> = Vec::with_capacity(2);
    for array in [&mut cache.conv_state, &mut cache.state_cache]
        .into_iter()
        .flatten()
    {
        *array = mlxcel_core::copy(array);
        pending.push(&**array as *const MlxArray);
    }
    // SAFETY: the pointers refer to arrays owned by `cache`, alive for the call.
    unsafe { mlxcel_core::async_eval_all(&pending) };
}

#[cfg(test)]
#[path = "recurrent_batch_tests.rs"]
mod tests;
