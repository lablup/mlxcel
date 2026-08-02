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

//! Load-time fold of `kv_b_proj` into the query and output paths (issue #907).
//!
//! `kv_b_proj` is one `[num_heads * (qk_nope_head_dim + v_head_dim),
//! kv_lora_rank]` matrix that up-projects the latent into per-head K and V. A
//! linear layer computes `y = x @ W^T`, so for latent row `c`
//!
//! ```text
//!   k_nope[h, d] = sum_r W[h * rows_per_head + d, r] * c[r]
//!   v[h, e]      = sum_r W[h * rows_per_head + qk_nope + e, r] * c[r]
//! ```
//!
//! Naming the two row blocks `W_UK[h, d, r]` and `W_UV[h, e, r]`, the two
//! absorption identities follow directly:
//!
//! ```text
//!   q_nope[h] . k_nope[h] = sum_r (sum_d q_nope[h, d] W_UK[h, d, r]) c[r]
//!   sum_t a[t] v[h]       = sum_r (sum_t a[t] c[t, r]) W_UV[h, e, r]
//! ```
//!
//! so the fold stores `W_UK` as `[1, H, qk_nope, r]` (right operand of the
//! query contraction) and `W_UV` transposed to `[1, H, r, v_head]` (right
//! operand of the output contraction). The leading `1` is the batch axis MLX
//! broadcasts against a `[B, H, L, ...]` left operand, so both applications are
//! a single batched `matmul` with no per-head loop.
//!
//! Nothing here mutates the checkpoint on disk. This is the same class of
//! load-time repack as `src/models/sanitize.rs`, applied to an already-loaded
//! [`crate::layers::UnifiedLinear`] rather than to the weight map, so a family
//! can fold without restructuring its loader.

use cxx::UniquePtr;

use crate::ffi::{self, MlxArray};
use crate::layers::UnifiedLinear;
use crate::mla::MlaGeometry;

/// `kv_b_proj` folded into a query-side and an output-side operand.
///
/// Both tensors are dense: the contractions are batched matmuls, and MLX has no
/// batched quantized matmul that would keep a 3-D per-head quantized operand.
/// [`MlaAbsorbedProjections::element_count`] is the fixed weight-memory price
/// that buys the per-token cache reduction.
pub struct MlaAbsorbedProjections {
    /// `W_UK` as `[1, num_heads, qk_nope_head_dim, kv_lora_rank]`.
    w_uk: UniquePtr<MlxArray>,
    /// `W_UV^T` as `[1, num_heads, kv_lora_rank, v_head_dim]`.
    w_uv: UniquePtr<MlxArray>,
    geometry: MlaGeometry,
}

impl MlaAbsorbedProjections {
    /// Fold an already-loaded `kv_b_proj`.
    ///
    /// Dequantizes when the layer is quantized, using the group size and bit
    /// width the loader already reconciled against the tensor shapes, so this
    /// path cannot disagree with what `quantized_matmul` would have used.
    ///
    /// Declines (as `Err`) rather than panicking, so a caller can fall back to
    /// the decompressed path for one family or one layer.
    pub fn from_kv_b_proj(
        kv_b_proj: &UnifiedLinear,
        geometry: MlaGeometry,
    ) -> Result<Self, String> {
        geometry.check()?;
        let dense = match kv_b_proj {
            UnifiedLinear::Regular(linear) => {
                if linear.bias.is_some() {
                    return Err(
                        "mla: kv_b_proj carries a bias, which the absorption identity does not \
                         cover; keeping the decompressed path"
                            .to_string(),
                    );
                }
                ffi::copy(&linear.weight)
            }
            UnifiedLinear::Quantized { weight, bias } => {
                if bias.is_some() {
                    return Err(
                        "mla: kv_b_proj carries a bias, which the absorption identity does not \
                         cover; keeping the decompressed path"
                            .to_string(),
                    );
                }
                let biases_ptr = weight
                    .biases
                    .as_ref()
                    .map(|b| &**b as *const MlxArray)
                    .unwrap_or(std::ptr::null());
                // SAFETY: `biases_ptr` is either null (block-float modes, which
                // the bridge accepts) or a pointer into `weight.biases`, which
                // outlives this call.
                let dequantized = unsafe {
                    ffi::dequantize(
                        &weight.weight,
                        &weight.scales,
                        biases_ptr,
                        weight.group_size,
                        weight.bits,
                        &weight.mode,
                    )
                };
                match weight.global_scale.as_ref() {
                    // NVFP4's per-tensor `weight_scale_2` is a scalar, and
                    // `(x @ W^T) * s2 == x @ (W * s2)^T`, so folding it into the
                    // dense weight here reproduces what `quantized_matmul`
                    // applies on its output.
                    Some(scale) => ffi::multiply(&dequantized, scale),
                    None => dequantized,
                }
            }
        };
        Self::from_dense(&dense, geometry)
    }

    /// Fold a dense `[H * (qk_nope + v_head), kv_lora_rank]` `kv_b_proj`.
    ///
    /// Split out from [`Self::from_kv_b_proj`] so the identity can be tested on
    /// a synthetic matrix without constructing a quantized layer.
    pub fn from_dense(weight: &MlxArray, geometry: MlaGeometry) -> Result<Self, String> {
        geometry.check()?;
        let shape = ffi::array_shape(weight);
        if shape.len() != 2 {
            return Err(format!(
                "mla: kv_b_proj must be 2-D [H*(qk_nope+v), kv_lora_rank], got {shape:?}"
            ));
        }
        let heads = geometry.num_heads as i32;
        let rows_per_head = geometry.kv_b_rows_per_head() as i32;
        let nope = geometry.qk_nope_head_dim as i32;
        let v_dim = geometry.v_head_dim as i32;
        let rank = geometry.kv_lora_rank as i32;
        if shape[0] != heads * rows_per_head || shape[1] != rank {
            return Err(format!(
                "mla: kv_b_proj shape {shape:?} disagrees with geometry \
                 [{heads}*{rows_per_head}, {rank}]"
            ));
        }

        // [H*(nope+v), r] -> [H, nope+v, r], then the two row blocks.
        let w3 = ffi::reshape(weight, &[heads, rows_per_head, rank]);
        let w_uk = ffi::slice(&w3, &[0, 0, 0], &[heads, nope, rank]);
        let w_uv = ffi::slice(&w3, &[0, nope, 0], &[heads, rows_per_head, rank]);

        // `contiguous` after the slice and the swap: both produce strided views,
        // and a strided operand forces MLX to materialize a copy on every
        // decode step instead of once here.
        let w_uk = ffi::contiguous(&ffi::reshape(&w_uk, &[1, heads, nope, rank]), false);
        let w_uv = ffi::swap_axes(&w_uv, -1, -2);
        let w_uv = ffi::contiguous(&ffi::reshape(&w_uv, &[1, heads, rank, v_dim]), false);

        ffi::eval(&w_uk);
        ffi::eval(&w_uv);
        Ok(Self {
            w_uk,
            w_uv,
            geometry,
        })
    }

    /// `W_UK` as `[1, H, qk_nope_head_dim, kv_lora_rank]`.
    #[must_use]
    pub fn w_uk(&self) -> &MlxArray {
        &self.w_uk
    }

    /// `W_UV^T` as `[1, H, kv_lora_rank, v_head_dim]`.
    #[must_use]
    pub fn w_uv(&self) -> &MlxArray {
        &self.w_uv
    }

    /// The geometry this fold was built for.
    #[must_use]
    pub const fn geometry(&self) -> MlaGeometry {
        self.geometry
    }

    /// Dense elements held by the fold, for one layer.
    ///
    /// Equal to the element count of `kv_b_proj` itself: the fold is a
    /// partition of its rows, not a copy of the whole with extra. Reported so
    /// the benchmark can state the fixed weight cost against the per-token
    /// cache saving instead of leaving it implicit.
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.geometry.num_heads * self.geometry.kv_b_rows_per_head() * self.geometry.kv_lora_rank
    }
}

impl std::fmt::Debug for MlaAbsorbedProjections {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlaAbsorbedProjections")
            .field("geometry", &self.geometry)
            .field("elements", &self.element_count())
            .finish()
    }
}

#[cfg(test)]
#[path = "absorb_tests.rs"]
mod absorb_tests;
