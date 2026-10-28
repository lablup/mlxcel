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

//! Ideogram 4 sampling: tokenization, token layout, text features, the
//! guided denoising loop and decoding.
//!
//! A port of `Ideogram4ImagePipeline` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ideogram4/pipeline.py>
//! (`#L127-L404`). The conditional transformer sees `[text tokens, image
//! tokens]` with 3-axis positions (`(i, i, i)` for text, `(0, row, col) +
//! 65536` for image tokens); the unconditional transformer sees the image
//! tokens alone. Each step computes `v = g * pos + (1 - g) * neg` and
//! `z += v * (s - t)` in f32, walking the logit-normal intervals backwards.
//! Noise is upstream's `mx.random.seed(seed)` followed by `mx.random.normal`,
//! which draws from the second half of `split(key(seed))`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::json;

use super::config::SamplingPlan;
use super::latent_norm::{LATENT_DIM, denormalize};
use super::scheduler::{get_schedule_for_resolution, make_step_intervals};
use super::transformer::{
    Ideogram4Transformer, LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, ProjectedLlm, TokenInputs,
};
use crate::imagegen::Qwen3TextEncoder;
use crate::imagegen::vae::Flux2Vae;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

/// Hidden states fed to the transformer: `hidden_states[layer + 1]` of the
/// 36-layer Qwen3-VL text tower.
pub const QWEN3_VL_ACTIVATION_LAYERS: [usize; 13] =
    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35];
/// Offset added to every image-token position.
pub const IMAGE_POSITION_OFFSET: i32 = 65536;
/// Latent patch size.
pub const PATCH_SIZE: u32 = 2;
/// VAE downsampling factor.
pub const AE_SCALE_FACTOR: u32 = 8;
/// Longest tokenized prompt.
pub const MAX_TEXT_TOKENS: usize = 2048;

/// `tokenizer/` with its chat template.
pub struct PromptTokenizer {
    tokenizer: MlxcelTokenizer,
    template: ChatTemplateProcessor,
}

impl PromptTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let tokenizer = load_tokenizer(dir)
            .with_context(|| format!("cannot load tokenizer from {}", dir.display()))?;
        let template = ChatTemplateProcessor::from_model_path(dir)?
            .with_context(|| format!("{} ships no chat template", dir.display()))?;
        Ok(Self {
            tokenizer,
            template,
        })
    }

    /// Upstream `_tokenize`: one user turn whose content is a single text
    /// part, `add_generation_prompt=True`, encoded without special tokens.
    /// A prompt over [`MAX_TEXT_TOKENS`] errors with upstream's message.
    pub fn token_ids(&self, prompt: &str) -> Result<Vec<i32>> {
        let messages = json!([{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]);
        let text =
            self.template
                .apply_raw_with_kwargs(&messages, None, &ChatTemplateKwargs::new())?;
        let ids = self.tokenizer.encode(&text, false)?;
        if ids.len() > MAX_TEXT_TOKENS {
            bail!(
                "prompt has {} tokens, exceeds max_text_tokens={MAX_TEXT_TOKENS}",
                ids.len()
            );
        }
        ids.into_iter()
            .map(|id| i32::try_from(id).context("token id exceeds i32"))
            .collect()
    }
}

/// Upstream `_build_inputs` for one prompt (batch 1).
pub struct SequenceLayout {
    pub num_text_tokens: i32,
    pub num_image_tokens: i32,
    pub grid_h: i32,
    pub grid_w: i32,
    /// `[1, T + N, 3]` int32.
    pub position_ids: UniquePtr<MlxArray>,
    /// `[1, T + N]` int32.
    pub indicator: UniquePtr<MlxArray>,
    /// The image-token tail of the two, for the unconditional transformer.
    pub image_position_ids: UniquePtr<MlxArray>,
    pub image_indicator: UniquePtr<MlxArray>,
}

impl SequenceLayout {
    #[must_use]
    pub fn new(num_text_tokens: usize, width: u32, height: u32) -> Self {
        let patch = PATCH_SIZE * AE_SCALE_FACTOR;
        let (grid_h, grid_w) = ((height / patch) as i32, (width / patch) as i32);
        let t = num_text_tokens as i32;
        let n = grid_h * grid_w;
        let mut positions = Vec::with_capacity(3 * (t + n) as usize);
        for i in 0..t {
            positions.extend([i, i, i]);
        }
        let mut image_positions = Vec::with_capacity(3 * n as usize);
        for row in 0..grid_h {
            for col in 0..grid_w {
                image_positions.extend([
                    IMAGE_POSITION_OFFSET,
                    row + IMAGE_POSITION_OFFSET,
                    col + IMAGE_POSITION_OFFSET,
                ]);
            }
        }
        positions.extend_from_slice(&image_positions);
        let mut indicator = vec![LLM_TOKEN_INDICATOR; t as usize];
        indicator.extend(std::iter::repeat_n(OUTPUT_IMAGE_INDICATOR, n as usize));
        Self {
            num_text_tokens: t,
            num_image_tokens: n,
            grid_h,
            grid_w,
            position_ids: mlxcel_core::from_slice_i32(&positions, &[1, t + n, 3]),
            indicator: mlxcel_core::from_slice_i32(&indicator, &[1, t + n]),
            image_position_ids: mlxcel_core::from_slice_i32(&image_positions, &[1, n, 3]),
            image_indicator: mlxcel_core::from_slice_i32(&indicator[t as usize..], &[1, n]),
        }
    }
}

/// Upstream `_encode_text`: `hidden_states[layer + 1]` for each activation
/// layer, stacked on a new last axis and flattened hidden-major to
/// `[1, T, H * 13]`, zero over the `num_image_tokens` image tokens, f32.
///
/// The order differs from FLUX.2's `stacked_prompt_embeds` (layer-major), so
/// that helper is not used.
pub fn text_features(
    encoder: &Qwen3TextEncoder,
    token_ids: &[i32],
    num_image_tokens: i32,
) -> Result<UniquePtr<MlxArray>> {
    let t = token_ids.len() as i32;
    let ids = mlxcel_core::from_slice_i32(token_ids, &[1, t]);
    // Upstream's all-ones attention mask: causal only.
    let states = encoder.hidden_states(&ids, None)?;
    if states.len() <= *QWEN3_VL_ACTIVATION_LAYERS.last().unwrap_or(&0) + 1 {
        bail!(
            "the Ideogram 4 text encoder needs {} layers, got {}",
            QWEN3_VL_ACTIVATION_LAYERS.last().unwrap_or(&0) + 1,
            encoder.num_layers()
        );
    }
    let selected: Vec<UniquePtr<MlxArray>> = QWEN3_VL_ACTIVATION_LAYERS
        .iter()
        .map(|&layer| mlxcel_core::expand_dims(&states[layer + 1], -1))
        .collect();
    let stacked =
        mlxcel_core::concatenate_many(&selected.iter().map(|a| &**a).collect::<Vec<_>>(), -1);
    let h = encoder.hidden_size();
    let n = QWEN3_VL_ACTIVATION_LAYERS.len() as i32;
    let embeds = mlxcel_core::reshape(&stacked, &[1, t, h * n]);
    let padding = mlxcel_core::zeros(
        &[1, num_image_tokens, h * n],
        mlxcel_core::array_dtype(&embeds),
    );
    let features = mlxcel_core::astype(
        &mlxcel_core::concatenate(&embeds, &padding, 1),
        dtype::FLOAT32,
    );
    mlxcel_core::try_eval(&features).context("Ideogram 4 prompt encoding failed")?;
    Ok(features)
}

/// `mx.random.seed(seed); mx.random.normal((1, tokens, 128), float32)`.
#[must_use]
pub fn initial_noise(seed: u64, tokens: i32) -> UniquePtr<MlxArray> {
    let keys = mlxcel_core::random_split_key(&mlxcel_core::random_key(seed), 2);
    let key = mlxcel_core::slice(&keys, &[1, 0], &[2, 2]);
    let key = mlxcel_core::reshape(&key, &[2]);
    // SAFETY: `key` is alive for the duration of the call.
    unsafe { mlxcel_core::random_normal(&[1, tokens, LATENT_DIM as i32], dtype::FLOAT32, &*key) }
}

fn f32_scalar(value: f64) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value as f32, dtype::FLOAT32)
}

/// The two transformers and the prompt's projected features.
pub struct Denoiser<'a> {
    pub conditional: &'a Ideogram4Transformer,
    pub unconditional: &'a Ideogram4Transformer,
    pub llm: &'a ProjectedLlm,
    pub layout: &'a SequenceLayout,
}

impl Denoiser<'_> {
    /// Conditional and unconditional velocities of the image tokens at `t`.
    pub fn velocities(
        &self,
        z: &MlxArray,
        t: f64,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let layout = self.layout;
        let (t_len, n) = (layout.num_text_tokens, layout.num_image_tokens);
        let t_arr = f32_scalar(t);
        let padding = mlxcel_core::zeros(&[1, t_len, LATENT_DIM as i32], dtype::FLOAT32);
        let pos_z = mlxcel_core::concatenate(&padding, z, 1);
        // One segment for the whole sequence: the segment mask is all zeros,
        // so it is not passed (see `transformer::segment_mask`).
        let pos_out = self.conditional.forward(
            Some(self.llm),
            &pos_z,
            &t_arr,
            &TokenInputs {
                position_ids: &layout.position_ids,
                indicator: &layout.indicator,
                mask: None,
            },
        )?;
        let pos_v =
            mlxcel_core::slice(&pos_out, &[0, t_len, 0], &[1, t_len + n, LATENT_DIM as i32]);
        let neg_v = self.unconditional.forward(
            None,
            z,
            &t_arr,
            &TokenInputs {
                position_ids: &layout.image_position_ids,
                indicator: &layout.image_indicator,
                mask: None,
            },
        )?;
        Ok((pos_v, neg_v))
    }

    /// Upstream's denoising loop from `noise` (`[1, N, 128]` f32).
    pub fn denoise(
        &self,
        mut z: UniquePtr<MlxArray>,
        plan: &SamplingPlan,
        width: u32,
        height: u32,
    ) -> Result<UniquePtr<MlxArray>> {
        let schedule = get_schedule_for_resolution(height, width, plan.mu, plan.std);
        let intervals = make_step_intervals(plan.steps)?;
        for i in (0..plan.steps as usize).rev() {
            let t_val = schedule.at(intervals[i + 1]);
            let s_val = schedule.at(intervals[i]);
            let (pos_v, neg_v) = self.velocities(&z, t_val)?;
            let g = plan.guidance_at(i);
            let velocity = mlxcel_core::add(
                &mlxcel_core::multiply(&f32_scalar(g), &pos_v),
                &mlxcel_core::multiply(&f32_scalar(1.0 - g), &neg_v),
            );
            z = mlxcel_core::add(
                &z,
                &mlxcel_core::multiply(&velocity, &f32_scalar(s_val - t_val)),
            );
            // `try_eval`: an MLX failure (allocation, kernel) must surface as an
            // error, not a C++ exception that aborts the process.
            mlxcel_core::try_eval(&z)
                .with_context(|| format!("Ideogram 4 denoising step {i} failed"))?;
        }
        Ok(z)
    }
}

/// Upstream `_decode`: latent norm, unpatchify `[1, gh, gw, 2, 2, C] ->
/// [1, C, 2gh, 2gw]`, FLUX.2 VAE decode, then `round((clip(x, -1, 1) + 1) *
/// 127.5)` as HWC `u8`.
pub fn decode_to_pixels(vae: &Flux2Vae, z: &MlxArray, grid_h: i32, grid_w: i32) -> Result<Vec<u8>> {
    let z = denormalize(z);
    let p = PATCH_SIZE as i32;
    let channels = LATENT_DIM as i32 / (p * p);
    let latents = mlxcel_core::reshape(&z, &[1, grid_h, grid_w, p, p, channels]);
    let latents = mlxcel_core::transpose_axes(&latents, &[0, 5, 1, 3, 2, 4]);
    let latents = mlxcel_core::reshape(&latents, &[1, channels, grid_h * p, grid_w * p]);
    // Upstream decodes in its bf16 precision; the VAE here runs in its own
    // load dtype, so cast to that rather than mixing dtypes.
    let vae_dtype = Flux2Vae::default_dtype();
    let decoded = vae.decode(&mlxcel_core::astype(&latents, vae_dtype))?;
    let decoded = mlxcel_core::astype(&decoded, dtype::FLOAT32);
    let clipped = mlxcel_core::clip(&decoded, &f32_scalar(-1.0), &f32_scalar(1.0));
    let scaled = mlxcel_core::multiply(
        &mlxcel_core::add(&clipped, &f32_scalar(1.0)),
        &f32_scalar(127.5),
    );
    let bytes = mlxcel_core::astype(&mlxcel_core::round(&scaled), dtype::UINT8);
    let nhwc = mlxcel_core::transpose_axes(&bytes, &[0, 2, 3, 1]);
    mlxcel_core::try_array_to_raw_bytes(&nhwc).context("Ideogram 4 VAE decode failed")
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
