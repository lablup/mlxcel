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

//! Native width-one batched decode for Z1T: stack every row's streaming state,
//! run the recurrence once, and give each row back an independent copy.

use mlxcel_core::{MlxArray, UniquePtr};

use super::Z1t;
use super::layers::Z1tLayerState;
use crate::models::recurrent_batch::{optional_rows_uniform, split_rows, stack_optional_rows};

impl Z1t {
    pub(super) fn batch_states_uniform(&self, rows: &[Vec<Z1tLayerState>]) -> bool {
        (0..self.model.layers.len()).all(|layer| {
            let column = |pick: fn(&Z1tLayerState) -> Option<&MlxArray>| -> bool {
                optional_rows_uniform(&rows.iter().map(|row| pick(&row[layer])).collect::<Vec<_>>())
            };
            column(|s| s.cum_ekv.as_deref())
                && column(|s| s.cum_ek.as_deref())
                && column(|s| s.win_ekv.as_deref())
                && column(|s| s.win_ek.as_deref())
        })
    }

    /// One width-one decode step over stacked per-row state, split back after.
    pub(super) fn forward_batched_decode(
        &self,
        input_ids: &MlxArray,
        rows: &mut [Vec<Z1tLayerState>],
    ) -> UniquePtr<MlxArray> {
        let batch = rows.len();
        let offsets: Vec<usize> = rows.iter().map(|row| row[0].offset).collect();
        let stack = |layer: usize, pick: fn(&Z1tLayerState) -> Option<&MlxArray>| {
            stack_optional_rows(&rows.iter().map(|row| pick(&row[layer])).collect::<Vec<_>>())
                .into_option()
        };
        let mut stacked: Vec<Z1tLayerState> = (0..self.model.layers.len())
            .map(|layer| Z1tLayerState {
                offset: 0,
                cum_ekv: stack(layer, |s| s.cum_ekv.as_deref()),
                cum_ek: stack(layer, |s| s.cum_ek.as_deref()),
                win_ekv: stack(layer, |s| s.win_ekv.as_deref()),
                win_ek: stack(layer, |s| s.win_ek.as_deref()),
            })
            .collect();
        let h = self
            .model
            .forward_rows(input_ids, &mut stacked, &offsets, None);
        let split = |array: &Option<UniquePtr<MlxArray>>| -> Vec<Option<UniquePtr<MlxArray>>> {
            match array.as_deref() {
                Some(array) => split_rows(array, batch).into_iter().map(Some).collect(),
                None => (0..batch).map(|_| None).collect(),
            }
        };
        for (layer, state) in stacked.iter().enumerate() {
            let [cum_ekv, cum_ek, win_ekv, win_ek] = [
                split(&state.cum_ekv),
                split(&state.cum_ek),
                split(&state.win_ekv),
                split(&state.win_ek),
            ];
            for (((((row, a), b), c), d), offset) in rows
                .iter_mut()
                .zip(cum_ekv)
                .zip(cum_ek)
                .zip(win_ekv)
                .zip(win_ek)
                .zip(&offsets)
            {
                row[layer] = Z1tLayerState {
                    offset: offset + 1,
                    cum_ekv: a,
                    cum_ek: b,
                    win_ekv: c,
                    win_ek: d,
                };
            }
        }
        self.lm_head.forward(&h)
    }
}
