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
//! A dense `kv_b_proj` folds into `W_UK` as `[1, H, qk_nope, r]` (right operand
//! of the query contraction) and `W_UV` transposed to `[1, H, r, v_head]`
//! (right operand of the output contraction). The leading `1` is the batch axis
//! MLX broadcasts against a `[B, H, L, ...]` left operand, so both applications
//! are a single batched `matmul` with no per-head loop.
//!
//! A quantized `kv_b_proj` folds without dequantizing (issue #3121). Its packed
//! rows are already grouped along `r`, which is the output axis of the query
//! contraction and the input axis of the output contraction, so the per-head row
//! blocks `[H, qk_nope, r]` and `[H, v_head, r]` are used as they are:
//! `q_nope @ W_UK` is a `quantized_matmul` with `transpose = false` and
//! `o_latent @ W_UV^T` one with `transpose = true`. Both read the exact
//! quantized weights the decompressed path's `kv_b_proj` reads, at a quarter of
//! the bytes of the f16 fold that the decode step used to stream per token.
//!
//! Nothing here mutates the checkpoint on disk. This is the same class of
//! load-time repack as `src/models/sanitize.rs`, applied to an already-loaded
//! [`crate::layers::UnifiedLinear`] rather than to the weight map, so a family
//! can fold without restructuring its loader.

use cxx::UniquePtr;

use crate::ffi::{self, MlxArray};
use crate::layers::{QuantizedMultiLinear, QuantizedWeight, UnifiedLinear};
use crate::mla::MlaGeometry;

/// The two fold operands, in whichever form `kv_b_proj` arrived in.
enum FoldWeights {
    /// From an unquantized `kv_b_proj` (or one whose quantization the batched
    /// quantized matmul cannot express, see [`MlaAbsorbedProjections::from_kv_b_proj`]).
    Dense {
        /// `W_UK` as `[1, num_heads, qk_nope_head_dim, kv_lora_rank]`.
        w_uk: UniquePtr<MlxArray>,
        /// `W_UV^T` as `[1, num_heads, kv_lora_rank, v_head_dim]`.
        w_uv: UniquePtr<MlxArray>,
    },
    /// The quantized `kv_b_proj` row blocks, packed as the checkpoint ships them.
    Quantized {
        /// `W_UK` as `[num_heads, qk_nope_head_dim, kv_lora_rank / pack]`.
        w_uk: QuantizedMultiLinear,
        /// `W_UV` as `[num_heads, v_head_dim, kv_lora_rank / pack]`.
        w_uv: QuantizedMultiLinear,
    },
}

/// `kv_b_proj` folded into a query-side and an output-side operand.
///
/// A quantized `kv_b_proj` keeps its packed rows (issue #3121); an unquantized
/// one is held dense. [`MlaAbsorbedProjections::weight_bytes`] is the fixed
/// weight-memory price that buys the per-token cache reduction.
pub struct MlaAbsorbedProjections {
    weights: FoldWeights,
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
                if let Some(fold) = Self::quantized_fold(weight, geometry)? {
                    return Ok(fold);
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

    /// Fold a quantized `kv_b_proj` by slicing its packed planes per head.
    ///
    /// Returns `Ok(None)` when the layer carries something a per-head
    /// `quantized_matmul` cannot reproduce on its own (an NVFP4 tensor scale,
    /// a runtime LoRA term, or a stored mode that disagrees with the mode the
    /// planes imply), and the caller then folds the dequantized matrix as
    /// before. `Err` only for a shape that does not describe the geometry.
    fn quantized_fold(
        weight: &QuantizedWeight,
        geometry: MlaGeometry,
    ) -> Result<Option<Self>, String> {
        if weight.global_scale.is_some() || !weight.runtime_loras.is_empty() {
            return Ok(None);
        }
        let mode = crate::layers::infer_quantization_mode(
            weight.biases.is_some(),
            weight.group_size,
            weight.bits,
        );
        if mode != weight.mode {
            return Ok(None);
        }
        let heads = geometry.num_heads as i32;
        let rows_per_head = geometry.kv_b_rows_per_head() as i32;
        let nope = geometry.qk_nope_head_dim as i32;
        let w_shape = ffi::array_shape(&weight.weight);
        let s_shape = ffi::array_shape(&weight.scales);
        if w_shape.len() != 2 || s_shape.len() != 2 || w_shape[0] != heads * rows_per_head {
            return Err(format!(
                "mla: quantized kv_b_proj shape {w_shape:?} disagrees with geometry \
                 [{heads}*{rows_per_head}, ..]"
            ));
        }
        if s_shape[1] * weight.group_size != geometry.kv_lora_rank as i32 {
            return Err(format!(
                "mla: quantized kv_b_proj scales {s_shape:?} at group size {} do not cover \
                 kv_lora_rank {}",
                weight.group_size, geometry.kv_lora_rank
            ));
        }

        // [H*(nope+v), cols] -> [H, nope+v, cols], then the two row blocks,
        // made contiguous once here so no decode step copies a strided operand.
        let split = |plane: &MlxArray| {
            let cols = ffi::array_shape(plane)[1];
            let p3 = ffi::reshape(plane, &[heads, rows_per_head, cols]);
            let uk = ffi::contiguous(&ffi::slice(&p3, &[0, 0, 0], &[heads, nope, cols]), false);
            let uv = ffi::contiguous(
                &ffi::slice(&p3, &[0, nope, 0], &[heads, rows_per_head, cols]),
                false,
            );
            ffi::eval(&uk);
            ffi::eval(&uv);
            (uk, uv)
        };
        let (w_uk, w_uv) = split(&weight.weight);
        let (s_uk, s_uv) = split(&weight.scales);
        let (b_uk, b_uv) = match weight.biases.as_ref() {
            Some(b) => {
                let (uk, uv) = split(b);
                (Some(uk), Some(uv))
            }
            None => (None, None),
        };
        let w_uk = QuantizedMultiLinear::new(w_uk, s_uk, b_uk, weight.group_size, weight.bits)?;
        let w_uv = QuantizedMultiLinear::new(w_uv, s_uv, b_uv, weight.group_size, weight.bits)?;
        Ok(Some(Self {
            weights: FoldWeights::Quantized { w_uk, w_uv },
            geometry,
        }))
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
            weights: FoldWeights::Dense { w_uk, w_uv },
            geometry,
        })
    }

    /// Dense `W_UK` as `[1, H, qk_nope_head_dim, kv_lora_rank]`, or `None` for
    /// a quantized fold.
    #[must_use]
    pub fn w_uk(&self) -> Option<&MlxArray> {
        match &self.weights {
            FoldWeights::Dense { w_uk, .. } => Some(w_uk),
            FoldWeights::Quantized { .. } => None,
        }
    }

    /// Dense `W_UV^T` as `[1, H, kv_lora_rank, v_head_dim]`, or `None` for a
    /// quantized fold.
    #[must_use]
    pub fn w_uv(&self) -> Option<&MlxArray> {
        match &self.weights {
            FoldWeights::Dense { w_uv, .. } => Some(w_uv),
            FoldWeights::Quantized { .. } => None,
        }
    }

    /// Whether the fold kept `kv_b_proj`'s quantized planes.
    #[must_use]
    pub const fn is_quantized(&self) -> bool {
        matches!(self.weights, FoldWeights::Quantized { .. })
    }

    /// `q_nope @ W_UK`: `[B, H, L, qk_nope] -> [B, H, L, kv_lora_rank]`.
    #[must_use]
    pub fn absorb(&self, q_nope: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.weights {
            FoldWeights::Dense { w_uk, .. } => ffi::matmul(q_nope, w_uk),
            FoldWeights::Quantized { w_uk, .. } => w_uk.forward_no_transpose(q_nope),
        }
    }

    /// `o_latent @ W_UV^T`: `[B, H, L, kv_lora_rank] -> [B, H, L, v_head_dim]`.
    #[must_use]
    pub fn unabsorb(&self, o_latent: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.weights {
            FoldWeights::Dense { w_uv, .. } => ffi::matmul(o_latent, w_uv),
            FoldWeights::Quantized { w_uv, .. } => w_uv.forward(o_latent),
        }
    }

    /// Up-project a latent `[B, 1, S, kv_lora_rank]` into
    /// `(k_nope [B, H, S, qk_nope], v [B, H, S, v_head_dim])`.
    #[must_use]
    pub fn expand(&self, ckv: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match &self.weights {
            FoldWeights::Dense { w_uk, w_uv } => {
                let w_uk_t = ffi::swap_axes(w_uk, -1, -2);
                (ffi::matmul(ckv, &w_uk_t), ffi::matmul(ckv, w_uv))
            }
            FoldWeights::Quantized { w_uk, w_uv } => (w_uk.forward(ckv), w_uv.forward(ckv)),
        }
    }

    /// The geometry this fold was built for.
    #[must_use]
    pub const fn geometry(&self) -> MlaGeometry {
        self.geometry
    }

    /// Logical weight elements held by the fold, for one layer.
    ///
    /// Equal to the element count of `kv_b_proj` itself: the fold is a
    /// partition of its rows, not a copy of the whole with extra.
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.geometry.num_heads * self.geometry.kv_b_rows_per_head() * self.geometry.kv_lora_rank
    }

    /// Bytes the fold holds for one layer, as stored: packed weight, scales
    /// and biases for a quantized fold, the dense operands otherwise. Reported
    /// so the fixed weight cost can be stated against the per-token cache
    /// saving instead of left implicit.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        let bytes = |a: &MlxArray| ffi::array_nbytes(a);
        match &self.weights {
            FoldWeights::Dense { w_uk, w_uv } => bytes(w_uk) + bytes(w_uv),
            FoldWeights::Quantized { w_uk, w_uv } => [w_uk, w_uv]
                .iter()
                .map(|q| {
                    bytes(&q.weight) + bytes(&q.scales) + q.biases.as_ref().map_or(0, |b| bytes(b))
                })
                .sum(),
        }
    }
}

impl std::fmt::Debug for MlaAbsorbedProjections {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlaAbsorbedProjections")
            .field("geometry", &self.geometry)
            .field("elements", &self.element_count())
            .field("quantized", &self.is_quantized())
            .finish()
    }
}

#[cfg(test)]
#[path = "absorb_tests.rs"]
mod absorb_tests;
