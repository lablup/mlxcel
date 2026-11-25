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

use crate::weights::WeightMap;
use crate::{MlxArray, UniquePtr, dtype};

use super::{InklingLayerCache, weight};

/// Causal depth-wise convolution shared by Inkling target and MTP layers.
pub struct InklingShortConv {
    weight: UniquePtr<MlxArray>,
    decode_weight: UniquePtr<MlxArray>,
    kernel: i32,
}

impl InklingShortConv {
    pub(super) fn forward_batched_decode(
        &self,
        input: &MlxArray,
        caches: &mut [&mut InklingLayerCache],
        slot: usize,
        residual: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let channels = crate::array_shape(input)[2];
        // The scalar path pads absent state with zeros in float32 too.
        let states: Vec<_> = caches
            .iter()
            .map(|cache| match &cache.conv[slot] {
                Some(state) => crate::copy(state),
                None => crate::zeros(&[1, self.kernel - 1, channels], dtype::FLOAT32),
            })
            .collect();
        let state_refs: Vec<_> = states.iter().map(|state| &**state).collect();
        let mut stacked = Some(crate::concatenate_many(&state_refs, 0));
        let out = self.forward(input, &mut stacked, residual);
        let state = stacked.expect("convolution retains its tail");
        let rows = crate::array_shape(input)[1];
        for (i, cache) in caches.iter_mut().enumerate() {
            // Per-row slice along axis 0 of the already-scheduled tail, so it
            // stays a slice plus `contiguous` and is scheduled explicitly.
            let row = crate::contiguous(
                &crate::utils::slice_axis(&state, 0, i as i32, i as i32 + 1),
                false,
            );
            crate::utils::schedule_prefill_state(&row, rows);
            cache.conv[slot] = Some(row);
        }
        out
    }

    pub fn from_weights(weights: &WeightMap, name: &str, kernel: usize) -> Result<Self, String> {
        let raw = weight(weights, name)?;
        let shape = crate::array_shape(&raw);
        if shape.len() != 3 || shape[1] != kernel as i32 || shape[2] != 1 {
            return Err(format!(
                "{name}: expected [channels, {kernel}, 1], got {shape:?}"
            ));
        }
        let weight = crate::astype(&raw, dtype::FLOAT32);
        let decode_weight = crate::transpose_axes(&weight, &[2, 1, 0]);
        let decode_weight = crate::contiguous(&decode_weight, false);
        crate::eval(&decode_weight);
        Ok(Self {
            weight,
            decode_weight,
            kernel: kernel as i32,
        })
    }

    pub fn forward(
        &self,
        input: &MlxArray,
        state: &mut Option<UniquePtr<MlxArray>>,
        residual: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_recording(input, state, None, residual)
    }

    /// [`Self::forward`] on convolution `slot` of `cache`, recording the
    /// verify-block input window in `cache.block_tails[slot]` (see
    /// [`InklingLayerCache::keep_verified_prefix`]).
    pub(super) fn forward_in_cache(
        &self,
        input: &MlxArray,
        cache: &mut InklingLayerCache,
        slot: usize,
        residual: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let InklingLayerCache {
            conv, block_tails, ..
        } = cache;
        self.forward_recording(
            input,
            &mut conv[slot],
            Some(&mut block_tails[slot]),
            residual,
        )
    }

    fn forward_recording(
        &self,
        input: &MlxArray,
        state: &mut Option<UniquePtr<MlxArray>>,
        tail: Option<&mut Option<UniquePtr<MlxArray>>>,
        residual: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let input_dtype = crate::array_dtype(input);
        let input_f32 = crate::astype(input, dtype::FLOAT32);
        let shape = crate::array_shape(&input_f32);
        let keep = self.kernel - 1;
        let padded = match state.as_ref().and_then(|value| value.as_ref()) {
            Some(previous) => crate::concatenate(previous, &input_f32, 1),
            None => {
                let zeros = crate::zeros(&[shape[0], keep, shape[2]], dtype::FLOAT32);
                crate::concatenate(&zeros, &input_f32, 1)
            }
        };
        let padded_len = crate::array_shape(&padded)[1];
        // A fresh buffer, scheduled now on a long prefill so `padded` is not
        // held until the caches are evaluated.
        *state = Some(crate::utils::materialized_state_tail(&padded, keep));

        // One position: the window times the taps, summed. A verify block
        // repeats that per position instead of taking `conv1d`, whose
        // summation order is not the step's (issue #3409).
        let step = |end: i32| {
            let window = crate::utils::slice_axis(&padded, 1, end - self.kernel, end);
            let multiplied = crate::multiply(&window, &self.decode_weight);
            crate::sum_axis(&multiplied, 1, true)
        };
        let verify_block = shape[0] == 1
            && shape[1] > 1
            && shape[1] <= crate::row_invariant::MAX_ROWS
            && crate::row_invariant::in_verify_block();
        let convolved = if shape[1] == 1 {
            step(padded_len)
        } else if verify_block {
            let parts: Vec<_> = (0..shape[1]).map(|i| step(keep + i + 1)).collect();
            let refs: Vec<&MlxArray> = parts.iter().map(|part| &**part).collect();
            crate::concatenate_many(&refs, 1)
        } else {
            crate::conv1d(&padded, &self.weight, 1, 0, 1, shape[2])
        };
        if let Some(tail) = tail {
            *tail = verify_block.then_some(padded);
        }
        let inner = crate::add(&convolved, &input_f32);
        let inner = crate::astype(&inner, input_dtype);
        match residual {
            Some(residual) => crate::add(residual, &inner),
            None => inner,
        }
    }
}
