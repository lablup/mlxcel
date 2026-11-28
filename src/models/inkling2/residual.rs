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

//! Inkling 2 attention residuals and loop message controllers.
//!
//! Derived from mlx-vlm `AttentionResidual`, `_attention_residual` and
//! `MessageController` (`mlx_vlm/models/inkling2/language.py`, MIT).

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::attention::inverse_sqrt;

fn f32(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(x, dtype::FLOAT32)
}

/// Softmax-weighted mix over the completed blocks plus the current partial.
pub struct AttentionResidual {
    /// f32 `[queries, hidden]`.
    query: UniquePtr<MlxArray>,
    normalize_values: bool,
}

impl AttentionResidual {
    pub fn from_weights(
        weights: &WeightMap,
        key: &str,
        queries: usize,
        hidden: usize,
        normalize_values: bool,
    ) -> Result<Self, String> {
        let value = weights
            .get(key)
            .ok_or_else(|| format!("Weight not found: {key}"))?;
        let size = mlxcel_core::array_size(value);
        if size != queries * hidden {
            return Err(format!(
                "{key} must hold {queries} x {hidden} values, got shape {:?}",
                mlxcel_core::array_shape(value)
            ));
        }
        Ok(Self {
            query: mlxcel_core::reshape(&f32(value), &[queries as i32, hidden as i32]),
            normalize_values,
        })
    }

    /// `[B, L, queries, hidden]` f32.
    pub fn forward(
        &self,
        blocks: &[UniquePtr<MlxArray>],
        partial: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let mut states: Vec<UniquePtr<MlxArray>> = blocks.iter().map(|block| f32(block)).collect();
        states.push(f32(partial));
        let states = mlxcel_core::stack_owned(&states, -2);
        // Both products depend on the block length. `[B, L, S, H] @ [H, Q]`
        // shares its right operand, so MLX folds the positions into
        // M = L * S, and the mix runs L batches where a decode step runs one.
        // On GPU generations 13 and 14 (M1, M2) the matmul overlay sends a
        // product of at most 6 rows in a single batch through per-row GEMV
        // and anything else through the steel GEMM tile, which sums in
        // another order, so a verify block parted from the chain there
        // (issue #3468). Inside a verify block on those generations each
        // position runs alone, at its decode step's shapes. Generation 15
        // and later have no such route, the block already matched the chain
        // there (M3 Ultra), and the split costs a verify round several
        // percent, so they keep the single mix. CUDA and ROCm have no such
        // route and keep it too.
        if mlxcel_core::hardware::get_hardware().few_row_dense_gemv() {
            mlxcel_core::row_invariant::per_position(&states, |states| self.mix(states))
        } else {
            self.mix(&states)
        }
    }

    /// The mix of one forward's `states [B, L, S, H]` (f32).
    fn mix(&self, states: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = *mlxcel_core::array_shape(states).last().expect("rank >= 1");
        let keys = mlxcel_core::fast_rms_norm_no_weight(states, 1e-6);
        let scores = mlxcel_core::matmul(&keys, &mlxcel_core::transpose(&self.query));
        let scores = mlxcel_core::multiply_scalar(&scores, inverse_sqrt(hidden));
        let weights = mlxcel_core::swap_axes(&mlxcel_core::softmax(&scores, -2), -1, -2);
        mlxcel_core::matmul(&weights, if self.normalize_values { &keys } else { states })
    }
}

/// Blend one previous-pass message into a sublayer input:
/// `(1 - tanh(h F^T)) * x + tanh(h G^T) * message`, `h = [rms(x), rms(m)] D^T`.
pub struct MessageController {
    down: UniquePtr<MlxArray>,
    /// f32.
    forget: UniquePtr<MlxArray>,
    gain: UniquePtr<MlxArray>,
    eps: f32,
}

impl MessageController {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden: usize,
        rank: usize,
        eps: f32,
    ) -> Result<Self, String> {
        let load = |leaf: &str, shape: [usize; 2]| -> Result<UniquePtr<MlxArray>, String> {
            let key = format!("{prefix}.{leaf}");
            let value = weights
                .get(&key)
                .ok_or_else(|| format!("Weight not found: {key}"))?;
            let expected = [shape[0] as i32, shape[1] as i32];
            if mlxcel_core::array_shape(value) != expected {
                return Err(format!(
                    "{key} must be {expected:?}, got {:?}",
                    mlxcel_core::array_shape(value)
                ));
            }
            Ok(mlxcel_core::copy(value))
        };
        let down = load("down_weight", [rank, 2 * hidden])?;
        let forget = load("forget_weight", [hidden, rank])?;
        let gain = load("gain_weight", [hidden, rank])?;
        Ok(Self {
            down,
            forget: f32(&forget),
            gain: f32(&gain),
            eps,
        })
    }

    pub fn forward(&self, x: &MlxArray, message: &MlxArray) -> UniquePtr<MlxArray> {
        let pair = mlxcel_core::concatenate(
            &mlxcel_core::fast_rms_norm_no_weight(x, self.eps),
            &mlxcel_core::fast_rms_norm_no_weight(message, self.eps),
            -1,
        );
        let pair = mlxcel_core::astype(&pair, mlxcel_core::array_dtype(&self.down));
        let rw = mlxcel_core::row_invariant::dense;
        let hidden = f32(&rw(&pair, &self.down));
        let forget = mlxcel_core::tanh(&rw(&hidden, &self.forget));
        let gain = mlxcel_core::tanh(&rw(&hidden, &self.gain));
        let keep = mlxcel_core::subtract(&mlxcel_core::from_slice_f32(&[1.0], &[1]), &forget);
        mlxcel_core::add(
            &mlxcel_core::multiply(&keep, x),
            &mlxcel_core::multiply(&gain, message),
        )
    }
}
