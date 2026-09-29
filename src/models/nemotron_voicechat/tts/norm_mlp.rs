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

//! Small building blocks shared by the EAR-TTS modules.
//!
//! Ports `OffsetRMSNorm`, `MLP` and `MLPLayer` from
//! `mlx_vlm/models/nemotron_voicechat/tts.py`, plus the weight-map helpers
//! and the `nn.Linear` bias semantics the other TTS files rely on.
//!
//! `OffsetRMSNorm` is deliberately not [`mlxcel_core::layers::GemmaRMSNorm`]:
//! the reference upcasts the input to `f32`, builds `1 + weight` in `f32`,
//! normalizes, and casts back, whereas `GemmaRMSNorm` builds `1 + weight` in
//! the weight's dtype and normalizes in the input's. The two differ in the
//! last bits for bf16 weights, which is visible in the sampled RVQ codes.

use mlxcel_core::dtype;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gemma3_backbone::gelu_approx;

/// Copy `key` out of `weights`, or report it missing.
pub(crate) fn weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {key}"))
}

/// Copy `key` and check its shape.
pub(crate) fn weight_with_shape(
    weights: &WeightMap,
    key: &str,
    expected: &[i32],
) -> Result<UniquePtr<MlxArray>, String> {
    let w = weight(weights, key)?;
    let shape = mlxcel_core::array_shape(&w);
    if shape != expected {
        return Err(format!("{key}: expected shape {expected:?}, got {shape:?}"));
    }
    Ok(w)
}

/// A scalar array holding `value` in `dtype_id`, the equivalent of a Python
/// float literal (weakly typed) meeting an array of that dtype.
pub(crate) fn scalar(value: f64, dtype_id: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[], value as f32, dtype_id)
}

/// `mx.sum(x, axis=axis)` with MLX's native reduction for every dtype.
///
/// [`mlxcel_core::sum_axis`] widens half-precision inputs to `f32`, reduces
/// and rounds once, which differs from the native bf16 `mx.sum` the
/// reference runs (the codec port measured about 20% of bf16 codebook norms
/// differing, enough to flip RVQ `argmin` codes). A pure-reduction `einsum`
/// lowers to the native `sum`. Rank is limited to 8.
pub(crate) fn native_sum_axis(x: &MlxArray, axis: usize) -> UniquePtr<MlxArray> {
    const LETTERS: &str = "abcdefgh";
    let rank = mlxcel_core::array_shape(x).len().min(LETTERS.len());
    let input = &LETTERS[..rank];
    let output: String = input
        .chars()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, c)| c)
        .collect();
    let spec = format!("{input}->{output}");
    let operands = [x as *const MlxArray];
    // SAFETY: the operand pointer borrows `x`, which outlives the call.
    unsafe { mlxcel_core::einsum(&spec, &operands) }
}

/// `nn.Linear.__call__`: `addmm(bias, x, W.T)` when a bias exists, `x @ W.T`
/// otherwise.
///
/// The crate's dense [`mlxcel_core::layers::Linear`] adds the bias after the
/// matmul, which rounds twice for a bf16 output; MLX's `addmm` fuses the add
/// into the GEMM epilogue and rounds once. Quantized layers keep
/// [`UnifiedLinear::forward`].
pub(crate) fn linear_forward(layer: &UnifiedLinear, x: &MlxArray) -> UniquePtr<MlxArray> {
    match layer {
        UnifiedLinear::Regular(linear) => match &linear.bias {
            Some(bias) => {
                let wt = mlxcel_core::transpose(&linear.weight);
                mlxcel_core::addmm(bias, x, &wt, 1.0, 1.0)
            }
            None => layer.forward(x),
        },
        UnifiedLinear::Quantized { .. } => layer.forward(x),
    }
}

/// `OffsetRMSNorm`: RMSNorm in `f32` with weight `1 + w`, cast back to the
/// input dtype.
pub struct OffsetRmsNorm {
    adjusted_weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl OffsetRmsNorm {
    /// Build from the stored offset `w` (any float dtype).
    pub fn new(offset_weight: &MlxArray, eps: f32) -> Self {
        let w32 = mlxcel_core::astype(offset_weight, dtype::FLOAT32);
        let adjusted_weight = mlxcel_core::add(&scalar(1.0, dtype::FLOAT32), &w32);
        Self {
            adjusted_weight,
            eps,
        }
    }

    /// Load `{prefix}.weight` with `hidden` entries.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden: usize,
        eps: f32,
    ) -> Result<Self, String> {
        let w = weight_with_shape(weights, &format!("{prefix}.weight"), &[hidden as i32])?;
        Ok(Self::new(&w, eps))
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let dtype_id = mlxcel_core::array_dtype(x);
        let x32 = mlxcel_core::astype(x, dtype::FLOAT32);
        let y = mlxcel_core::fast_rms_norm(&x32, &self.adjusted_weight, self.eps);
        mlxcel_core::astype(&y, dtype_id)
    }
}

/// The reference `MLP`: `down_proj(gelu_approx(gate_proj(x)) * up_proj(x))`,
/// with the op-for-op [`gelu_approx`] rather than the crate's fused GeGLU.
pub struct GeluMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl GeluMlp {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let lin = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        Ok(Self {
            gate_proj: lin("gate_proj")?,
            up_proj: lin("up_proj")?,
            down_proj: lin("down_proj")?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gated = mlxcel_core::multiply(
            &gelu_approx(&self.gate_proj.forward(x)),
            &self.up_proj.forward(x),
        );
        self.down_proj.forward(&gated)
    }
}

/// `MLPLayer`: `x + post_norm(mlp(pre_norm(x)))`.
pub struct MlpLayer {
    pre_norm: OffsetRmsNorm,
    mlp: GeluMlp,
    post_norm: OffsetRmsNorm,
}

impl MlpLayer {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden: usize,
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            pre_norm: OffsetRmsNorm::from_weights(
                weights,
                &format!("{prefix}.pre_norm"),
                hidden,
                eps,
            )?,
            mlp: GeluMlp::from_weights(weights, &format!("{prefix}.mlp"), group_size, bits)?,
            post_norm: OffsetRmsNorm::from_weights(
                weights,
                &format!("{prefix}.post_norm"),
                hidden,
                eps,
            )?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.mlp.forward(&self.pre_norm.forward(x));
        mlxcel_core::add(x, &self.post_norm.forward(&h))
    }
}
