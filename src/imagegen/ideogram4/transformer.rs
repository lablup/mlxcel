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

//! `Ideogram4Transformer`: the single-stream DiT both `transformer/` and
//! `unconditional_transformer/` load into.
//!
//! A port of
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ideogram4/transformer.py>.
//! Text tokens (`LLM_TOKEN_INDICATOR`) carry the projected LLM features and
//! image tokens (`OUTPUT_IMAGE_INDICATOR`) the projected latents, in one
//! sequence; the unconditional transformer sees only the image tokens and no
//! LLM features. Everything runs in the weights' dtype (bf16) except the
//! timestep embedding's sinusoid, the RoPE tables before rounding, SDPA (f32)
//! and the f32 output.
//!
//! The LLM features do not change across steps, so their projection is
//! computed once per prompt ([`Ideogram4Transformer::project_llm_features`])
//! instead of on every step as upstream does, and only over the text rows
//! (the image rows are zero after upstream's mask); the arithmetic is the
//! same.

use anyhow::{Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::blocks::{Block, Dense, MRope, Norm, layer_norm_no_affine, tensor};
use super::config::Ideogram4TransformerConfig;

/// `indicator` value of a text token.
pub const LLM_TOKEN_INDICATOR: i32 = 3;
/// `indicator` value of an output-image token.
pub const OUTPUT_IMAGE_INDICATOR: i32 = 2;

/// Token-level inputs of one forward.
pub struct TokenInputs<'a> {
    /// `[B, L, 3]` int32 positions.
    pub position_ids: &'a MlxArray,
    /// `[B, L]` int32 token types.
    pub indicator: &'a MlxArray,
    /// Additive `[B, 1, L, L]` segment mask; `None` for a single segment.
    pub mask: Option<&'a MlxArray>,
}

/// The LLM conditioning of one prompt, projected to the model width and
/// zeroed outside text tokens (`[B, L, emb_dim]`).
pub struct ProjectedLlm(UniquePtr<MlxArray>);

impl ProjectedLlm {
    #[must_use]
    pub fn array(&self) -> &MlxArray {
        &self.0
    }
}

/// The loaded transformer.
pub struct Ideogram4Transformer {
    config: Ideogram4TransformerConfig,
    input_proj: Dense,
    /// Present on the conditional transformer; the unconditional one never
    /// reads text and ships without them in memory.
    llm_cond_norm: Option<Norm>,
    llm_cond_proj: Option<Dense>,
    t_mlp_in: Dense,
    t_mlp_out: Dense,
    adaln_proj: Dense,
    embed_image_indicator: UniquePtr<MlxArray>,
    rotary: MRope,
    layers: Vec<Block>,
    final_linear: Dense,
    final_adaln: Dense,
}

fn scalar(value: f32, dt: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, dt)
}

/// `(indicator == value)` as `[B, L, 1]` in `dt`.
fn token_mask(indicator: &MlxArray, value: i32, dt: i32) -> UniquePtr<MlxArray> {
    let eq = mlxcel_core::equal(
        indicator,
        &mlxcel_core::full_f32(&[1], value as f32, dtype::INT32),
    );
    mlxcel_core::expand_dims(&mlxcel_core::astype(&eq, dt), -1)
}

/// Upstream `_sinusoidal_embedding` (scale 1e4) in f32: `[sin, cos]` of
/// `t * exp(-ln(1e4) / (half - 1) * arange(half))`.
fn sinusoidal_embedding(t: &MlxArray, dim: i32) -> UniquePtr<MlxArray> {
    let half = dim / 2;
    let freq = (1e4_f64).ln() / f64::from(half - 1);
    let freq = mlxcel_core::exp(&mlxcel_core::multiply(
        &mlxcel_core::astype(&mlxcel_core::arange_i32(0, half, 1), dtype::FLOAT32),
        &scalar(-(freq as f32), dtype::FLOAT32),
    ));
    let emb = mlxcel_core::multiply(&mlxcel_core::expand_dims(t, -1), &freq);
    let emb = mlxcel_core::concatenate(&mlxcel_core::sin(&emb), &mlxcel_core::cos(&emb), -1);
    if dim % 2 == 1 {
        let rank = mlxcel_core::array_shape(&emb).len();
        let mut pads = vec![0; 2 * rank];
        pads[2 * rank - 1] = 1;
        return mlxcel_core::pad(&emb, &pads, 0.0);
    }
    emb
}

impl Ideogram4Transformer {
    /// Build from dequantized (bf16) weights under upstream's key names.
    pub fn from_weights(config: Ideogram4TransformerConfig, weights: &WeightMap) -> Result<Self> {
        let head_dim = config.head_dim();
        let (e, a, c, f) = (
            config.emb_dim,
            config.adaln_dim,
            config.in_channels,
            config.llm_features_dim,
        );
        let layers = (0..config.num_layers)
            .map(|i| Block::load(weights, &format!("layers.{i}"), &config))
            .collect::<Result<Vec<_>>>()?;
        let embed_image_indicator = tensor(weights, "embed_image_indicator.weight", &[2, e])?;
        // Either both LLM tensors are present or neither is.
        let has_llm = weights.contains_key("llm_cond_norm.weight")
            || weights.contains_key("llm_cond_proj.weight");
        let (llm_cond_norm, llm_cond_proj) = if has_llm {
            (
                Some(Norm::load(weights, "llm_cond_norm.weight", 1e-6, f)?),
                Some(Dense::load(weights, "llm_cond_proj", e, f)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            input_proj: Dense::load(weights, "input_proj", e, c)?,
            llm_cond_norm,
            llm_cond_proj,
            t_mlp_in: Dense::load(weights, "t_embedding.mlp_in", e, e)?,
            t_mlp_out: Dense::load(weights, "t_embedding.mlp_out", e, e)?,
            adaln_proj: Dense::load(weights, "adaln_proj", a, e)?,
            embed_image_indicator,
            rotary: MRope::new(head_dim, config.rope_theta, config.mrope_section),
            layers,
            final_linear: Dense::load(weights, "final_layer.linear", c, e)?,
            final_adaln: Dense::load(weights, "final_layer.adaln_modulation", e, a)?,
            config,
        })
    }

    /// Whether the LLM conditioning weights were loaded.
    #[must_use]
    pub fn has_llm_conditioning(&self) -> bool {
        self.llm_cond_proj.is_some()
    }

    #[must_use]
    pub fn config(&self) -> &Ideogram4TransformerConfig {
        &self.config
    }

    /// Weight dtype (upstream's `param_dtype`).
    #[must_use]
    pub fn param_dtype(&self) -> i32 {
        self.input_proj.dtype()
    }

    /// Upstream's LLM branch, `proj(norm(f * text_mask)) * text_mask`, for the
    /// `[B, T, llm_features_dim]` features of the text tokens, in the weight
    /// dtype, followed by `num_image_tokens` zero rows: `[B, T + N, emb_dim]`.
    ///
    /// Upstream feeds zero rows for the image tokens and masks them again
    /// after the projection. Every op is row-wise and the text mask is
    /// exactly 1 on text rows, so projecting only the text rows and
    /// appending zeros is the same arithmetic without the image rows' cost.
    pub fn project_llm_features(
        &self,
        text_features: &MlxArray,
        num_image_tokens: i32,
    ) -> Result<ProjectedLlm> {
        let shape = mlxcel_core::array_shape(text_features);
        if shape.len() != 3 || shape[2] != self.config.llm_features_dim {
            bail!(
                "llm_features must be [B, T, {}], got {shape:?}",
                self.config.llm_features_dim
            );
        }
        if num_image_tokens < 0 {
            bail!("num_image_tokens must be non-negative, got {num_image_tokens}");
        }
        let (Some(llm_cond_norm), Some(llm_cond_proj)) = (&self.llm_cond_norm, &self.llm_cond_proj)
        else {
            bail!("this Ideogram 4 transformer was loaded without llm_cond_norm/llm_cond_proj");
        };
        let dt = self.param_dtype();
        let features = mlxcel_core::astype(text_features, dt);
        let projected = llm_cond_proj.forward(&llm_cond_norm.forward(&features));
        let padding = mlxcel_core::zeros(&[shape[0], num_image_tokens, self.config.emb_dim], dt);
        Ok(ProjectedLlm(mlxcel_core::concatenate(
            &projected, &padding, 1,
        )))
    }

    /// `Ideogram4EmbedScalar` then `silu(adaln_proj(.))`: `[B, 1, adaln_dim]`
    /// for `t` (`[B]`, already in the weight dtype, range `[0, 1]`).
    fn adaln_input(&self, t: &MlxArray) -> UniquePtr<MlxArray> {
        let dim = self.config.emb_dim;
        // `1e4 * (x.f32 - 0) / (1 - 0)`.
        let scaled = mlxcel_core::multiply(
            &scalar(1e4, dtype::FLOAT32),
            &mlxcel_core::divide(
                &mlxcel_core::subtract(
                    &mlxcel_core::astype(t, dtype::FLOAT32),
                    &scalar(0.0, dtype::FLOAT32),
                ),
                &scalar(1.0, dtype::FLOAT32),
            ),
        );
        let emb = mlxcel_core::astype(&sinusoidal_embedding(&scaled, dim), self.t_mlp_in.dtype());
        let t_cond = self
            .t_mlp_out
            .forward(&mlxcel_core::silu(&self.t_mlp_in.forward(&emb)));
        let t_cond = mlxcel_core::expand_dims(&t_cond, 1);
        mlxcel_core::silu(&self.adaln_proj.forward(&t_cond))
    }

    /// One forward: velocity for every token, `[B, L, in_channels]` f32.
    ///
    /// `x` is `[B, L, in_channels]` (f32 is fine; it is cast to the weight
    /// dtype first), `t` is `[B]` f32. `llm` is `None` for the
    /// unconditional transformer.
    pub fn forward(
        &self,
        llm: Option<&ProjectedLlm>,
        x: &MlxArray,
        t: &MlxArray,
        inputs: &TokenInputs<'_>,
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(x);
        if shape.len() != 3 || shape[2] != self.config.in_channels {
            bail!(
                "x must be [B, L, {}], got {shape:?}",
                self.config.in_channels
            );
        }
        let dt = self.param_dtype();
        let x = mlxcel_core::astype(x, dt);
        let t = mlxcel_core::astype(t, dt);
        let image_mask = token_mask(inputs.indicator, OUTPUT_IMAGE_INDICATOR, dt);
        let x = mlxcel_core::multiply(&x, &image_mask);
        let x = mlxcel_core::multiply(&self.input_proj.forward(&x), &image_mask);
        let adaln_input = self.adaln_input(&t);
        let h = match llm {
            Some(llm) => mlxcel_core::add(&x, llm.array()),
            None => x,
        };
        let is_image = mlxcel_core::astype(
            &mlxcel_core::equal(
                inputs.indicator,
                &mlxcel_core::full_f32(&[1], OUTPUT_IMAGE_INDICATOR as f32, dtype::INT32),
            ),
            dtype::INT32,
        );
        let mut h = mlxcel_core::add(
            &h,
            &mlxcel_core::take(&self.embed_image_indicator, &is_image, 0),
        );
        let (cos, sin) = self.rotary.tables(inputs.position_ids);
        let table = |t: &MlxArray| {
            mlxcel_core::expand_dims(&mlxcel_core::astype(t, mlxcel_core::array_dtype(&h)), 1)
        };
        let (cos, sin) = (table(&cos), table(&sin));
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, inputs.mask, &adaln_input);
        }
        let scale = mlxcel_core::add(
            &scalar(1.0, dt),
            &self.final_adaln.forward(&mlxcel_core::silu(&adaln_input)),
        );
        let out = self.final_linear.forward(&mlxcel_core::multiply(
            &layer_norm_no_affine(&h, 1e-6),
            &scale,
        ));
        Ok(mlxcel_core::astype(&out, dtype::FLOAT32))
    }
}

/// Upstream's segment mask: `0` where two tokens share a segment, `-inf`
/// elsewhere, `[B, 1, L, L]` in `dt`. The pipeline always uses one segment,
/// for which this is all zeros and is skipped.
#[must_use]
pub fn segment_mask(segment_ids: &MlxArray, dt: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(segment_ids);
    let (b, l) = (shape[0], shape[1]);
    let rows = mlxcel_core::reshape(segment_ids, &[b, l, 1]);
    let cols = mlxcel_core::reshape(segment_ids, &[b, 1, l]);
    let same = mlxcel_core::equal(&rows, &cols);
    let masked = mlxcel_core::where_cond(
        &same,
        &mlxcel_core::zeros(&[b, l, l], dt),
        &mlxcel_core::full_f32(&[b, l, l], f32::NEG_INFINITY, dt),
    );
    mlxcel_core::expand_dims(&masked, 1)
}

#[cfg(test)]
#[path = "transformer_tests.rs"]
mod tests;
