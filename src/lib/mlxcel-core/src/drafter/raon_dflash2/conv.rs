// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! The DFlash2 grouped convolution around each attention and MLP sublayer.
//!
//! ```text
//! delta        = kernel_projection(x).reshape(n, 2, taps, groups)
//! coef[stage]  = base_kernel[stage].reshape(taps, groups, gsize) + delta[:, stage, ..., None]
//! out[t]       = sum_tap coef[t, tap] * x[t - tap]     when t % block_size >= tap
//! ```
//!
//! The first coefficient set wraps the sublayer input, the second its
//! output; both come from the sublayer input. A shift never crosses a block
//! boundary, so a proposal row never reads the row of another block.

use crate::ffi::{self, MlxArray};
use crate::layers::UnifiedLinear;
use crate::weights::WeightMap;
use cxx::UniquePtr;

/// One `attention_conv` or `mlp_conv` module.
pub struct GroupedConv {
    /// `[2, taps, hidden]`: one base kernel per stage.
    base_kernel: UniquePtr<MlxArray>,
    /// `[2 * taps * groups, hidden]`.
    kernel_projection: UnifiedLinear,
    taps: i32,
    groups: i32,
    group_size: i32,
}

impl GroupedConv {
    /// Load `{prefix}.base_kernel` and `{prefix}.kernel_projection`.
    ///
    /// # Errors
    /// Returns a message naming a missing tensor.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        taps: usize,
        group_size: usize,
        hidden: usize,
    ) -> Result<Self, String> {
        let key = format!("{prefix}.base_kernel");
        let base_kernel = weights
            .get(&key)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Raon DFlash2: missing weight {key}"))?;
        let kernel_projection =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.kernel_projection"), 64, 4)?;
        Ok(Self {
            base_kernel,
            kernel_projection,
            taps: taps as i32,
            groups: (hidden / group_size) as i32,
            group_size: group_size as i32,
        })
    }

    /// Per-row coefficient deltas `[n, 2, taps, groups]` for rows `x [n, H]`.
    pub fn deltas(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let n = ffi::array_shape(x)[0];
        let d = self.kernel_projection.forward(x);
        ffi::reshape(&d, &[n, 2, self.taps, self.groups])
    }

    /// Stage `stage`'s `[n, taps, groups]` slice of `deltas`.
    pub fn stage(&self, deltas: &MlxArray, stage: i32) -> UniquePtr<MlxArray> {
        let n = ffi::array_shape(deltas)[0];
        let s = ffi::slice(
            deltas,
            &[0, stage, 0, 0],
            &[n, stage + 1, self.taps, self.groups],
        );
        ffi::reshape(&s, &[n, self.taps, self.groups])
    }

    /// Stage `stage`'s base kernel, `[taps, hidden]`.
    fn base(&self, stage: i32) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(&self.base_kernel);
        let s = ffi::slice(
            &self.base_kernel,
            &[stage, 0, 0],
            &[stage + 1, shape[1], shape[2]],
        );
        ffi::reshape(&s, &[shape[1], shape[2]])
    }

    /// The convolution of rows `x [n, H]` with stage `stage`'s coefficients.
    pub fn apply(
        &self,
        x: &MlxArray,
        delta: &MlxArray,
        stage: i32,
        block_size: i32,
    ) -> UniquePtr<MlxArray> {
        grouped_conv(
            x,
            delta,
            &self.base(stage),
            block_size,
            self.groups,
            self.group_size,
        )
    }
}

/// `out[t] = sum_tap (base[tap] + delta[t, tap]) * h[t - tap]`, with the
/// shifted term dropped where `t % block_size < tap`.
///
/// - `h`: `[n, groups * gsize]` rows.
/// - `delta`: `[n, taps, groups]` per-row coefficient deltas.
/// - `base`: `[taps, groups * gsize]` base kernel.
///
/// The vendor `_grouped_conv`, operation for operation, in `h`'s dtype.
pub fn grouped_conv(
    h: &MlxArray,
    delta: &MlxArray,
    base: &MlxArray,
    block_size: i32,
    groups: i32,
    gsize: i32,
) -> UniquePtr<MlxArray> {
    let n = ffi::array_shape(h)[0];
    let taps = ffi::array_shape(delta)[1];
    let dt = ffi::array_dtype(h);
    let blocks = ffi::reshape(h, &[n, groups, gsize]);
    let base = ffi::reshape(base, &[1, taps, groups, gsize]);
    let coef = ffi::add(&base, &ffi::expand_dims(delta, -1)); // [n, taps, groups, gsize]
    let tap_coef = |tap: i32| {
        let s = ffi::slice(&coef, &[0, tap, 0, 0], &[n, tap + 1, groups, gsize]);
        ffi::reshape(&s, &[n, groups, gsize])
    };
    let mut out = ffi::multiply(&tap_coef(0), &blocks);
    let pos = ffi::remainder(
        &ffi::arange_i32(0, n, 1),
        &ffi::from_slice_i32(&[block_size], &[1]),
    );
    for tap in 1..taps {
        if tap >= n {
            break;
        }
        let head = ffi::zeros(&[tap, groups, gsize], ffi::array_dtype(&blocks));
        let body = ffi::slice(&blocks, &[0, 0, 0], &[n - tap, groups, gsize]);
        let shifted = crate::ops::concatenate(&head, &body, 0);
        let keep = ffi::greater_equal(&pos, &ffi::from_slice_i32(&[tap], &[1]));
        let keep = ffi::astype(&ffi::reshape(&keep, &[n, 1, 1]), dt);
        let term = ffi::multiply(&ffi::multiply(&tap_coef(tap), &shifted), &keep);
        out = ffi::add(&out, &term);
    }
    ffi::reshape(&out, &[n, groups * gsize])
}
