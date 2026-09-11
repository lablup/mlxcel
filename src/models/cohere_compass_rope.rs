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

//! Cohere Compass 3-axis MRoPE.
//!
//! This is **not** the Qwen3-VL interleaved MRoPE, despite the checkpoint
//! carrying `mrope_interleaved: true` (transformers lists that key under
//! `ignore_keys_at_rope_validation` and never reads it). Upstream's
//! `CohereCompassRotaryEmbedding` does two things Qwen3-VL does not:
//!
//! 1. It **pre-permutes** the inverse frequencies. With `hw = section[0] +
//!    section[1]`, the first `hw` entries of `inv_freq` are re-laid-out as
//!    `concat(inv_freq[0..hw][0::2], inv_freq[0..hw][1::2])` and the trailing
//!    `section[2]` entries are left alone. Upstream calls this "(pre-)rotate to
//!    avoid another rotation during the forward".
//! 2. It then assigns axes in **contiguous sections**, ordered `[H, W, T]`, not
//!    by a step-3 interleave: channels `[0, s_h)` take H, `[s_h, s_h + s_w)`
//!    take W, and the rest take T.
//!
//! The two together are what makes the published `[24, 20, 20]` section give H
//! 24 channels, W 20 and T 20, where Qwen3-VL's interleave over the same list
//! would give T 24, H 20 and W 20 with unpermuted frequencies. Substituting one
//! for the other rotates every sliding layer against the wrong table.
//!
//! Reference: `CohereCompassRotaryEmbedding.compute_default_rope_parameters`
//! and `.recomposition_frequencies` in
//! `transformers/models/cohere_compass/modular_cohere_compass.py`.

use mlxcel_core::{MlxArray, UniquePtr};

/// Section used when `rope_parameters` names a layer type but omits
/// `mrope_section`, matching upstream's `rope_params.get("mrope_section",
/// [22, 22, 20])`.
pub(crate) const DEFAULT_MROPE_SECTION: [i32; 3] = [22, 22, 20];

/// Axis index into the `[3, batch, seq]` position-id tensor. Row order is
/// `(T, H, W)`, the same order `compute_qwen_vl_mrope_position_ids` emits.
const AXIS_T: i32 = 0;
const AXIS_H: i32 = 1;
const AXIS_W: i32 = 2;

pub(crate) struct CompassMRoPE {
    /// Upstream's `inv_freq_3d`: the natural `inv_freq` with its first
    /// `s_h + s_w` entries split into evens-then-odds.
    inv_freq: Vec<f32>,
    /// Source axis for each of the `head_dim / 2` frequency channels.
    axis_of_channel: Vec<i32>,
}

impl CompassMRoPE {
    /// `section` is `[h, w, t]` and must sum to `head_dim / 2`.
    pub(crate) fn new(head_dim: usize, base: f32, section: [i32; 3]) -> Result<Self, String> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(format!(
                "cohere_compass: head_dim must be a positive even number, got {head_dim}"
            ));
        }
        let half = head_dim / 2;
        if section.iter().any(|&s| s < 0) {
            return Err(format!(
                "cohere_compass: mrope_section entries must be non-negative, got {section:?}"
            ));
        }
        let [s_h, s_w, s_t] = section.map(|s| s as usize);
        if s_h + s_w + s_t != half {
            return Err(format!(
                "cohere_compass: mrope_section {section:?} sums to {} but head_dim / 2 is {half}",
                s_h + s_w + s_t
            ));
        }

        let natural: Vec<f32> = (0..half)
            .map(|j| 1.0 / base.powf(2.0 * j as f32 / head_dim as f32))
            .collect();

        let hw = s_h + s_w;
        let mut inv_freq = Vec::with_capacity(half);
        inv_freq.extend((0..hw).step_by(2).map(|j| natural[j]));
        inv_freq.extend((1..hw).step_by(2).map(|j| natural[j]));
        inv_freq.extend_from_slice(&natural[hw..]);
        debug_assert_eq!(inv_freq.len(), half);

        let axis_of_channel = (0..half)
            .map(|j| {
                if j < s_h {
                    AXIS_H
                } else if j < hw {
                    AXIS_W
                } else {
                    AXIS_T
                }
            })
            .collect();

        Ok(Self {
            inv_freq,
            axis_of_channel,
        })
    }

    /// Per-channel source axis. Test surface for the section layout.
    #[cfg(test)]
    pub(crate) fn axis_of_channel(&self) -> &[i32] {
        &self.axis_of_channel
    }

    /// Pre-permuted inverse frequencies. Test surface for the pre-rotation.
    #[cfg(test)]
    pub(crate) fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// `position_ids`: `[3, batch, seq]` (T, H, W), or `[batch, seq]` for a
    /// text-only sequence where all three axes share one position.
    ///
    /// Returns `(cos, sin)`, each `[batch, seq, head_dim]`, built in f32 the
    /// way upstream forces the table under `maybe_autocast(enabled=False)`.
    pub(crate) fn forward(
        &self,
        position_ids: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let pos_shape = mlxcel_core::array_shape(position_ids);
        let position_ids_3d = if pos_shape.len() == 2 {
            let expanded = mlxcel_core::expand_dims(position_ids, 0);
            mlxcel_core::broadcast_to(&expanded, &[3, pos_shape[0], pos_shape[1]])
        } else {
            mlxcel_core::copy(position_ids)
        };

        let pos_shape = mlxcel_core::array_shape(&position_ids_3d);
        let batch = pos_shape[1];
        let seq_len = pos_shape[2];
        let half_dim = self.inv_freq.len() as i32;

        // inv_freq: [half] -> [3, batch, half, 1]
        let inv_freq_arr = mlxcel_core::from_slice_f32(&self.inv_freq, &[half_dim]);
        let inv_freq_arr = mlxcel_core::astype(&inv_freq_arr, mlxcel_core::dtype::FLOAT32);
        let inv_freq_4d = mlxcel_core::reshape(&inv_freq_arr, &[1, 1, half_dim, 1]);
        let inv_freq_4d = mlxcel_core::broadcast_to(&inv_freq_4d, &[3, batch, half_dim, 1]);

        // position ids: [3, batch, seq] -> [3, batch, 1, seq]
        let pos_expanded = mlxcel_core::reshape(&position_ids_3d, &[3, batch, 1, seq_len]);
        let pos_expanded = mlxcel_core::astype(&pos_expanded, mlxcel_core::dtype::FLOAT32);

        // freqs: [3, batch, half, seq] -> [3, batch, seq, half]
        let freqs = mlxcel_core::matmul(&inv_freq_4d, &pos_expanded);
        let freqs = mlxcel_core::transpose_axes(&freqs, &[0, 1, 3, 2]);

        // Recomposition: pick the section's axis per channel.
        let idx = mlxcel_core::from_slice_i32(&self.axis_of_channel, &[1, 1, 1, half_dim]);
        let picked = mlxcel_core::take_along_axis(&freqs, &idx, 0);
        let picked = mlxcel_core::squeeze_axis(&picked, 0);

        // Split layout: the table is duplicated so `rotate_half` pairs channel
        // j with j + head_dim / 2.
        let emb = mlxcel_core::concatenate(&picked, &picked, -1);
        (mlxcel_core::cos(&emb), mlxcel_core::sin(&emb))
    }
}

#[cfg(test)]
#[path = "cohere_compass_rope_tests.rs"]
mod cohere_compass_rope_tests;
