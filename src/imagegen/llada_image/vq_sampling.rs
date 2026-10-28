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

//! LLaDA's block-diffusion sampler for the discrete image prior (VQ mode).
//!
//! A port of `vq_prompt_ids` and `generate_vq_tokens` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llada_image/vq_sampling.py>.
//! The LLaDA-2 backbone (with [`Llada2Numerics::LladaImage`]) denoises one
//! 32-token block at a time over the whole window so far, with no KV cache:
//! block-causal attention, 8 greedy steps per block with classifier-free
//! guidance 2 (the unconditional row left-pads its prompt with mask tokens
//! and shifts its positions), revealing every prediction above confidence
//! 0.95 or else the top `count` of the step's schedule
//! ([`transfer_mask_top_count`]). The generated ids, minus the image-token
//! offset, are SigVQ codebook ids.
//!
//! [`Llada2Numerics::LladaImage`]: crate::models::llada2_moe::Llada2Numerics::LladaImage

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::conditioning::{Dense, bool_array};
use super::config::MAX_TEXT_TOKENS;
use crate::models::Llada2MoeModel;
use crate::models::llada2_moe::transfer_mask_top_count;
use crate::tokenizer::MlxcelTokenizer;

/// First image token id in the LLaDA-2 vocabulary.
pub const IMAGE_TOKEN_OFFSET: i32 = 157184;
/// The diffusion mask token.
pub const MASK_TOKEN_ID: i32 = 156895;
/// End of image.
pub const END_OF_IMAGE_ID: i32 = 156902;
/// Block length of the sampler.
pub const BLOCK_LENGTH: usize = 32;
/// Denoising steps per block.
pub const STEPS: usize = 8;
/// Classifier-free guidance of the prior.
pub const CFG_SCALE: f32 = 2.0;
/// Confidence above which a prediction is revealed.
pub const THRESHOLD: f32 = 0.95;

/// Longest conditional prior prompt (system template, prompt and image
/// header), in tokens. Upstream has no bound; the sampler re-runs the whole
/// window without a KV cache and builds a `[2, 1, end, end]` mask per block,
/// so an unbounded prompt is unbounded host and device memory.
pub const MAX_VQ_PROMPT_TOKENS: usize = MAX_TEXT_TOKENS;

const SYSTEM_PROMPT: &str =
    "<role>SYSTEM</role> You are a text-to-image generation assistant. <role>HUMAN</role>";

/// The VQ token grid `(rows, columns)` of a `width x height` image: the
/// longer side is scaled to at most 512, then each side is divided by 16.
pub fn vq_grid(height: u32, width: u32) -> Result<(u32, u32)> {
    let scale = (f64::from(height.max(width)) / 512.0).max(1.0);
    let rows = (f64::from(height) / scale) as u32 / 16;
    let columns = (f64::from(width) / scale) as u32 / 16;
    if rows.min(columns) < 1 {
        bail!("Image aspect ratio leaves no VQ tokens on the shorter side");
    }
    Ok((rows, columns))
}

/// Reject a conditional prior prompt longer than [`MAX_VQ_PROMPT_TOKENS`].
pub fn check_vq_prompt_length(tokens: usize) -> Result<()> {
    if tokens > MAX_VQ_PROMPT_TOKENS {
        bail!(
            "LLaDA-Image VQ prompt is {tokens} tokens with the prior's template; at most {MAX_VQ_PROMPT_TOKENS} are allowed"
        );
    }
    Ok(())
}

/// Upstream `vq_prompt_ids`: the conditional and unconditional prompt ids
/// and the number of VQ tokens to generate. A conditional prompt longer than
/// [`MAX_VQ_PROMPT_TOKENS`] is an error (mlxcel bound; upstream has none).
pub fn vq_prompt_ids(
    tokenizer: &MlxcelTokenizer,
    prompt: &str,
    height: u32,
    width: u32,
) -> Result<(Vec<i32>, Vec<i32>, usize)> {
    let (rows, columns) = vq_grid(height, width)?;
    let encode = |text: &str| -> Result<Vec<i32>> {
        Ok(tokenizer
            .encode(text, true)?
            .into_iter()
            .map(|id| id as i32)
            .collect())
    };
    let mut image = encode(&format!(
        "<|image|><|reserved_token_{rows}|><|reserved_token_{columns}|><boi><|/image|>"
    ))?;
    image.pop();
    let mut conditional = encode(&format!("{SYSTEM_PROMPT}{prompt}<role>ASSISTANT</role>"))?;
    conditional.extend(&image);
    check_vq_prompt_length(conditional.len())?;
    let mut unconditional = encode(&format!(
        "{SYSTEM_PROMPT}<uncondition><role>ASSISTANT</role>"
    ))?;
    unconditional.extend(&image);
    Ok((conditional, unconditional, (rows * columns) as usize))
}

/// Tokens revealed at each of the `steps` steps of one block.
#[must_use]
pub fn transfer_schedule(block_length: usize, steps: usize) -> Vec<usize> {
    (0..steps)
        .map(|i| block_length / steps + usize::from(i < block_length % steps))
        .collect()
}

fn to_i32(array: &MlxArray) -> Result<Vec<i32>> {
    let a = mlxcel_core::astype(array, dtype::INT32);
    let bytes = mlxcel_core::try_array_to_raw_bytes(&a).context("LLaDA-Image VQ prior failed")?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// The prior's raw output: every token of the window, prompt included.
pub struct VqTokens {
    /// `prompt + blocks`, padded to whole blocks.
    pub tokens: Vec<i32>,
    pub prompt_length: usize,
    pub gen_length: usize,
}

impl VqTokens {
    /// `tokens[P : P + gen_length] - offset`, checked against the codebook.
    pub fn codebook_ids(&self, codebook_size: i32) -> Result<Vec<i32>> {
        let ids: Vec<i32> = self.tokens[self.prompt_length..self.prompt_length + self.gen_length]
            .iter()
            .map(|t| t - IMAGE_TOKEN_OFFSET)
            .collect();
        if ids.iter().any(|&id| id < 0 || id >= codebook_size) {
            bail!("The MLLM generated tokens outside the SigVQ codebook");
        }
        Ok(ids)
    }
}

/// The mask (`[2, 1, end, end]`) and positions (`[2, end]`) of a window
/// ending at `end`: `attention[:, :, :end, :end]` (block-causal) with the
/// unconditional row also hiding its `pad_length` pad keys, and that row's
/// positions shifted down by the pad (clamped at 0).
pub(crate) fn window_inputs(
    end: usize,
    pad_length: usize,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let mut mask = Vec::with_capacity(2 * end * end);
    for row in 0..2 {
        for q in 0..end {
            for k in 0..end {
                let causal = q / BLOCK_LENGTH >= k / BLOCK_LENGTH;
                mask.push(causal && (row == 0 || k >= pad_length));
            }
        }
    }
    let e = end as i32;
    let positions: Vec<i32> = (0..e)
        .chain((0..end).map(|p| p.saturating_sub(pad_length) as i32))
        .collect();
    (
        bool_array(&mask, &[2, 1, e, e]),
        mlxcel_core::from_slice_i32(&positions, &[2, e]),
    )
}

/// Upstream `generate_vq_tokens` with its default block length 32, 8 steps,
/// guidance 2 and threshold 0.95, returning the whole window.
pub fn generate_vq_tokens(
    text_encoder: &Llada2MoeModel,
    lm_head: &Dense,
    input_ids: &[i32],
    uncond_ids: &[i32],
    gen_length: usize,
) -> Result<VqTokens> {
    let (block_length, cfg) = (BLOCK_LENGTH, CFG_SCALE);
    if gen_length < 1 {
        bail!("Invalid VQ token count or diffusion step count");
    }
    let steps = STEPS.min(gen_length);
    let prompt_length = input_ids.len();
    let Some(pad_length) = prompt_length.checked_sub(uncond_ids.len()) else {
        bail!("The unconditional VQ prompt is longer than the conditional prompt");
    };
    let blocks = (prompt_length + gen_length).div_ceil(block_length);
    let length = blocks * block_length;
    let mut tokens = vec![MASK_TOKEN_ID; length];
    tokens[..prompt_length].copy_from_slice(input_ids);
    let mut uncond_prefix = vec![MASK_TOKEN_ID; pad_length];
    uncond_prefix.extend_from_slice(uncond_ids);
    let transfers = transfer_schedule(block_length, steps);

    for block in prompt_length / block_length..blocks {
        let end = (block + 1) * block_length;
        let (mask, positions) = window_inputs(end, pad_length);
        let e = end as i32;
        for &count in &transfers {
            let window = &tokens[end - block_length..end];
            let active: Vec<bool> = window.iter().map(|&t| t == MASK_TOKEN_ID).collect();
            let remaining = active.iter().filter(|&&a| a).count();
            if remaining == 0 {
                break;
            }
            let mut ids = tokens[..end].to_vec();
            ids.extend_from_slice(&uncond_prefix);
            ids.extend_from_slice(&tokens[prompt_length..end]);
            let ids = mlxcel_core::from_slice_i32(&ids, &[2, e]);
            let hidden = text_encoder
                .forward_embeds(&text_encoder.embed(&ids), Some(&mask), &positions)
                .map_err(anyhow::Error::msg)?;
            let d = mlxcel_core::array_shape(&hidden)[2];
            let tail = mlxcel_core::slice(&hidden, &[0, e - block_length as i32, 0], &[2, e, d]);
            let logits = lm_head.forward(&tail);
            let vocab = mlxcel_core::array_shape(&logits)[2];
            let b = block_length as i32;
            let cond = mlxcel_core::slice(&logits, &[0, 0, 0], &[1, b, vocab]);
            let uncond = mlxcel_core::slice(&logits, &[1, 0, 0], &[2, b, vocab]);
            let ldt = mlxcel_core::array_dtype(&logits);
            // `logits[1:] + cfg_scale * (logits[:1] - logits[1:])`.
            let guided = mlxcel_core::add(
                &uncond,
                &mlxcel_core::multiply(
                    &mlxcel_core::full_f32(&[], cfg, ldt),
                    &mlxcel_core::subtract(&cond, &uncond),
                ),
            );
            let predicted = mlxcel_core::argmax(&guided, -1, false);
            let confidence = mlxcel_core::take_along_axis(
                &mlxcel_core::softmax(&guided, -1),
                &mlxcel_core::expand_dims(&predicted, -1),
                -1,
            );
            let confidence = mlxcel_core::reshape(&confidence, &[1, b]);
            let active_arr = bool_array(&active, &[1, b]);
            let neg_inf = mlxcel_core::full_f32(
                &[],
                f32::NEG_INFINITY,
                mlxcel_core::array_dtype(&confidence),
            );
            let scores = mlxcel_core::where_cond(&active_arr, &confidence, &neg_inf);
            mlxcel_core::try_eval(&scores).context("LLaDA-Image VQ prior step failed")?;
            let selected = transfer_mask_top_count(&scores, THRESHOLD, count, remaining);
            let predicted = to_i32(&predicted)?;
            for (offset, (&chosen, &token)) in selected.iter().zip(&predicted).enumerate() {
                if chosen {
                    tokens[end - block_length + offset] = token;
                }
            }
            let generated = &tokens[prompt_length..end];
            if let Some(stop) = generated.iter().position(|&t| t == END_OF_IMAGE_ID)
                && !generated[..stop].contains(&MASK_TOKEN_ID)
            {
                if stop < gen_length {
                    bail!("The MLLM ended after {stop} VQ tokens; expected {gen_length}");
                }
                break;
            }
        }
    }
    Ok(VqTokens {
        tokens,
        prompt_length,
        gen_length,
    })
}

#[cfg(test)]
#[path = "vq_sampling_tests.rs"]
mod tests;
