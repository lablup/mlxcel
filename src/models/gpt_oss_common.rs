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

//! Building blocks the GPT-OSS stack shares with its derivatives.
//!
//! OpenAI Privacy Filter is a bidirectional encoder built from the GPT-OSS
//! parts: YaRN RoPE, the clipped SwiGLU (alpha 1.702, limit 7), per-head
//! attention sinks, and biased per-expert projections over `gather_mm` /
//! `gather_qmm`. The pieces that do not depend on the decoder's cache or
//! layer schedule live here so both models run the same arithmetic.
//!
//! Used by: GptOss, OpenAIPrivacyFilter

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::switch_layers::validate_expert_quantization_params;

/// Inputs of [`yarn_rope_freqs`].
///
/// Used by: GptOss (`truncate: true`, mlx-lm `YarnRoPE`), OpenAIPrivacyFilter
/// (`truncate` from `rope_parameters`, default `false`), Ministral3
/// (`truncate` from `rope_parameters`, default `true`, mlx-lm `YarnRoPE`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct YarnRopeParams {
    /// Rotated dimensions (`head_dim` for both families).
    pub dims: usize,
    /// `rope_theta`.
    pub base: f32,
    /// YaRN `factor`.
    pub factor: f32,
    /// `original_max_position_embeddings`.
    pub original_max_position_embeddings: f32,
    /// YaRN `beta_fast`.
    pub beta_fast: f32,
    /// YaRN `beta_slow`.
    pub beta_slow: f32,
    /// `mscale` (GPT-OSS reads it from `rope_scaling`, default 1).
    pub mscale: f32,
    /// `mscale_all_dim` (default 0).
    pub mscale_all_dim: f32,
    /// Floor / ceil the correction range to whole dimensions (mlx-lm's
    /// `YarnRoPE` always does; transformers' `truncate` flag makes it
    /// optional, and Privacy Filter ships `truncate: false`).
    pub truncate: bool,
}

/// `yarn_get_mscale`: the attention factor YaRN applies to the rotated input.
fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// YaRN frequencies for `mx.fast.rope(freqs=...)` plus the attention factor.
///
/// Returns `dims / 2` wavelength-like denominators (the reciprocal of the
/// interpolated inverse frequency) and `yarn_get_mscale(factor, mscale) /
/// yarn_get_mscale(factor, mscale_all_dim)`, which multiplies q and k before
/// rotation. With `truncate` the correction range is floored / ceiled to whole
/// dimensions, exactly as `GptOssModel` always computed it; without it the
/// fractional bounds are kept, as `PrivacyFilterRoPE` does upstream.
///
/// Used by: GptOss, OpenAIPrivacyFilter, Ministral3
pub(crate) fn yarn_rope_freqs(params: &YarnRopeParams) -> (Vec<f32>, f32) {
    let dims = params.dims as f32;
    let base = params.base;
    let half_dims = params.dims / 2;

    // yarn_find_correction_dim
    let find_correction_dim = |num_rotations: f32| -> f32 {
        (dims
            * (params.original_max_position_embeddings
                / (num_rotations * 2.0 * std::f32::consts::PI))
                .ln())
            / (2.0 * base.ln())
    };

    // yarn_find_correction_range
    let (low, mut high) = if params.truncate {
        (
            find_correction_dim(params.beta_fast).floor().max(0.0),
            find_correction_dim(params.beta_slow).ceil().min(dims - 1.0),
        )
    } else {
        (
            find_correction_dim(params.beta_fast).max(0.0),
            find_correction_dim(params.beta_slow).min(dims - 1.0),
        )
    };
    if high == low {
        high += 0.001;
    }

    let attention_factor = yarn_get_mscale(params.factor, params.mscale)
        / yarn_get_mscale(params.factor, params.mscale_all_dim);

    let freqs = (0..half_dims)
        .map(|i| {
            let freq_extra = base.powf((2 * i) as f32 / dims);
            let freq_inter = params.factor * freq_extra;
            // yarn_linear_ramp_mask
            let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let freq_mask = 1.0 - ramp;
            // (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1 - freq_mask))
            (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1.0 - freq_mask))
        })
        .collect();
    (freqs, attention_factor)
}

/// Expert projection with an optional per-expert linear bias.
///
/// Handles affine quantization (with quant biases) and MXFP4 (E8M0 scales, no
/// quant biases). `switch_layers::SwitchLinear` has no per-expert `.bias`
/// slot, which every GPT-OSS expert carries.
///
/// Used by: GptOss, OpenAIPrivacyFilter
pub(crate) enum ExpertLinear {
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        quant_biases: Option<UniquePtr<MlxArray>>,
        linear_bias: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
    Regular {
        weight: UniquePtr<MlxArray>,
        linear_bias: Option<UniquePtr<MlxArray>>,
    },
}

impl ExpertLinear {
    /// `x` is `[tokens, 1, 1, in]` (or its sorted `[tokens * k, 1, in]`
    /// form) and `indices` the routed expert ids; the output dtype follows
    /// MLX promotion of `x` with the weight (or scales), so an f32 `x` gives
    /// an f32 result over bf16 / f16 planes.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
        sorted: bool,
    ) -> UniquePtr<MlxArray> {
        let out = match self {
            Self::Quantized {
                weight,
                scales,
                quant_biases,
                group_size,
                bits,
                mode,
                ..
            } => {
                let biases_ptr = quant_biases
                    .as_ref()
                    .map(|b| b.as_ref().unwrap() as *const _)
                    .unwrap_or(std::ptr::null());
                unsafe {
                    mlxcel_core::gather_qmm(
                        x,
                        weight,
                        scales,
                        biases_ptr,
                        std::ptr::null(), // lhs_indices
                        indices as *const _,
                        true,
                        *group_size,
                        *bits,
                        sorted,
                        mode,
                        std::ptr::null(),
                    )
                }
            }
            Self::Regular { weight, .. } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                unsafe {
                    mlxcel_core::gather_mm(x, &wt, std::ptr::null(), indices as *const _, sorted)
                }
            }
        };
        // Add per-expert linear bias if present
        let linear_bias = match self {
            Self::Quantized { linear_bias, .. } | Self::Regular { linear_bias, .. } => {
                linear_bias.as_ref()
            }
        };
        if let Some(bias) = linear_bias {
            let gathered_bias = mlxcel_core::take(bias, indices, 0);
            let gathered_bias = mlxcel_core::expand_dims(&gathered_bias, -2);
            mlxcel_core::add(&out, &gathered_bias)
        } else {
            out
        }
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Self, String> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing weight: {}", prefix))?;

        // Per-expert linear bias (e.g., gate_proj.bias [num_experts, out_dim])
        let linear_bias = weights
            .get(&format!("{}.bias", prefix))
            .map(|w| mlxcel_core::copy(w));

        // A per-expert NVFP4 tensor scale (mlx-lm `global_scale: true`, resolved
        // to `.native_global_scale` by the load funnel) or a bare `.global_scale`
        // has no slot in this expert path, which calls `gather_qmm` without
        // one; loading the plane anyway would decode every expert wrong
        // (issue #2351).
        for leaf in ["native_global_scale", "global_scale"] {
            if weights.contains_key(&format!("{prefix}.{leaf}")) {
                return Err(format!(
                    "{prefix}: carries a per-expert NVFP4 global scale ({prefix}.{leaf}), which \
                     the gpt-oss expert path does not apply"
                ));
            }
        }

        let scales_key = format!("{}.scales", prefix);
        if weights.contains_key(&scales_key) {
            // Bound the declared pair here, where it is stored: `ExpertLinear`
            // never reaches `reconcile_quantization_layout` and hands the stored
            // pair to `gather_qmm` (issue #958). Unlike every sibling projection
            // in this family, the expert pair comes from `Quantization::defaults`
            // rather than from `quant_for`, so the per-prefix overrides that the
            // guarded `UnifiedLinear` / `UnifiedEmbedding` loaders see do not
            // describe it, and a valid override on `model.embed_tokens` (the
            // shape every real gpt-oss checkpoint ships) leaves a hostile
            // top-level pair reaching the experts alone.
            validate_expert_quantization_params(prefix, group_size, bits)?;

            let scales = weights
                .get(&scales_key)
                .map(|w| mlxcel_core::copy(w))
                .unwrap();

            // Quantized biases (only present for affine mode)
            let quant_biases = weights
                .get(&format!("{}.biases", prefix))
                .map(|w| mlxcel_core::copy(w));

            // Bound the declared mode on the same terms as the pair, and on the
            // same reasoning: it is stored below and handed to `gather_qmm`
            // verbatim (issue #973). The bias-presence half matters here in a way
            // it does not for the shared loaders: this branch selects the
            // quantized path on `.scales` alone and makes `.biases` an `Option`
            // that becomes a null pointer when absent, so a plane with scales and
            // no zero points under a declared `affine` mode used to be built
            // without complaint and abort in `gather_qmm` with "Biases must be
            // provided for affine quantization". This is MLX's own precondition,
            // so it refuses only a load that was already going to abort; the
            // canonical mxfp4 checkpoint pairs its 72 mode-less expert planes with
            // block-float and its 122 `.biases`-carrying planes with affine, and
            // is unaffected.
            mlxcel_core::layers::validate_quantization_biases(mode, quant_biases.is_some())
                .map_err(|e| format!("{prefix}: {e}"))?;
            // A mode the running backend has no kernel for (issue #1806).
            mlxcel_core::layers::validate_quantization_mode_for_running_backend(mode)
                .map_err(|e| format!("{prefix}: {e}"))?;

            Ok(Self::Quantized {
                weight,
                scales,
                quant_biases,
                linear_bias,
                group_size,
                bits,
                mode: mode.to_string(),
            })
        } else {
            Ok(Self::Regular {
                weight,
                linear_bias,
            })
        }
    }
}

/// GPT-OSS clipped SwiGLU.
///
/// ```text
/// swiglu(x_linear, x_glu, alpha=1.702, limit):
///   x_glu = clamp(x_glu, max=limit)
///   x_linear = clamp(x_linear, min=-limit, max=limit)
///   out_glu = x_glu * sigmoid(alpha * x_glu)
///   return out_glu * (x_linear + 1)
/// ```
///
/// At the default limit of 7 this is the compiled fused kernel; any other
/// limit takes the op-by-op path. Both keep the input dtype, so f32 inputs
/// (Privacy Filter runs its experts in f32) come back f32.
///
/// Used by: GptOss, OpenAIPrivacyFilter
pub(crate) fn gpt_oss_swiglu(
    x_linear: &MlxArray,
    x_glu: &MlxArray,
    limit: f32,
) -> UniquePtr<MlxArray> {
    if (limit - 7.0).abs() <= f32::EPSILON {
        return mlxcel_core::compiled_gpt_oss_swiglu_activation(x_linear, x_glu);
    }

    let input_dtype = mlxcel_core::array_dtype(x_linear);
    let alpha = 1.702f32;

    // Clamp values
    let neg_limit = mlxcel_core::full_f32(&[1], -limit, input_dtype);
    let pos_limit = mlxcel_core::full_f32(&[1], limit, input_dtype);
    let x_glu = mlxcel_core::minimum(x_glu, &pos_limit);
    let x_linear = mlxcel_core::maximum(x_linear, &neg_limit);
    let x_linear = mlxcel_core::minimum(&x_linear, &pos_limit);

    // glu_scaled = alpha * x_glu -> sigmoid -> out_glu = x_glu * sig
    let alpha_arr = mlxcel_core::full_f32(&[1], alpha, input_dtype);
    let glu_scaled = mlxcel_core::multiply(&alpha_arr, &x_glu);
    let sig = mlxcel_core::sigmoid(&glu_scaled);
    let out_glu = mlxcel_core::multiply(&x_glu, &sig);

    // (x_linear + 1) * out_glu
    let one = mlxcel_core::full_f32(&[1], 1.0, input_dtype);
    let x_linear_plus_1 = mlxcel_core::add(&x_linear, &one);
    let result = mlxcel_core::multiply(&out_glu, &x_linear_plus_1);
    mlxcel_core::astype(&result, input_dtype)
}

#[cfg(test)]
#[path = "gpt_oss_common_tests.rs"]
mod tests;
