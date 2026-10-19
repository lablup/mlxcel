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

//! Image-prompted arm of `logit_trace` (#2388).
//!
//! A vision-tower change moves every image embedding, so the only text it can
//! move is text conditioned on an image. With `MLXCEL_TRACE_IMAGE=PATH` the
//! harness renders the checkpoint's chat template around one image and the
//! prompt in `MLXCEL_TRACE_IMAGE_PROMPT` (default `Describe the image.`),
//! prepares the prompt embeddings through the same
//! `vlm_runtime::prepare_vlm_embeddings` path `mlxcel generate` uses, and then
//! teacher-forces the first `CHUNK_TOKENS` tokens of `TEXT_FILE` as the
//! assistant's continuation.
//!
//! The prompt is one forward with embeddings (its last row predicts the first
//! continuation token) and the continuation is one forward over the remaining
//! tokens against the same caches, the shape a generation prefill followed by
//! a multi-token append runs at. Every continuation token is scored, and the
//! rows are written as chunk 0 in the same TSV the text arm writes, so
//! `scripts/compare_logit_traces.py` reads a pair unmodified.

use anyhow::{Context, Result};
use mlxcel::LanguageModel;
use mlxcel::server::chat_front::load_model_chat_template;
use mlxcel::tokenizer::MlxcelTokenizer;
use mlxcel_core::{MlxArray, UniquePtr, eval, from_slice_i32};

/// Default user prompt when `MLXCEL_TRACE_IMAGE_PROMPT` is unset.
const DEFAULT_PROMPT: &str = "Describe the image.";

/// The image arm's settings, read from the environment.
pub struct ImageArm {
    pub image_path: String,
    pub prompt: String,
}

impl ImageArm {
    /// `Some` when `MLXCEL_TRACE_IMAGE` names an image.
    pub fn from_env() -> Option<Self> {
        let image_path = std::env::var("MLXCEL_TRACE_IMAGE").ok()?;
        let prompt = std::env::var("MLXCEL_TRACE_IMAGE_PROMPT")
            .unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
        Some(Self { image_path, prompt })
    }
}

/// Render the user turn with one image placeholder, as `mlxcel generate`
/// does for a single `--image`.
fn render_prompt(model_dir: &str, tokenizer: &MlxcelTokenizer, user_prompt: &str) -> String {
    let Some(processor) = load_model_chat_template(std::path::Path::new(model_dir), tokenizer)
    else {
        return user_prompt.to_string();
    };
    let messages = serde_json::json!([{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ],
    }]);
    processor
        .apply_raw(&messages, None)
        .unwrap_or_else(|_| user_prompt.to_string())
}

fn encode(tokenizer: &MlxcelTokenizer, text: &str, add_special: bool) -> Vec<i32> {
    tokenizer
        .encode(text, add_special)
        .unwrap_or_default()
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

/// The traced logits of one image-prompted continuation.
pub struct ImageTrace {
    /// Row `j` predicts `targets[j]`; shape `[1, targets.len(), vocab]`.
    pub logits: UniquePtr<MlxArray>,
    pub targets: Vec<i32>,
    /// Prompt length after image-token expansion.
    pub prompt_tokens: usize,
}

/// Run the image-prompted arm over the first `max_tokens` tokens of `text`.
pub fn trace(
    arm: &ImageArm,
    model: &mlxcel::LoadedModel,
    tokenizer: &MlxcelTokenizer,
    model_dir: &str,
    text: &str,
    max_tokens: usize,
) -> Result<ImageTrace> {
    let image = image::open(&arm.image_path)
        .with_context(|| format!("loading image {}", arm.image_path))?;
    let prompt = render_prompt(model_dir, tokenizer, &arm.prompt);
    let add_special = !tokenizer.prompt_carries_bos(&prompt);
    let mut prompt_ids = encode(tokenizer, &prompt, add_special);
    let prepared = mlxcel::vlm_runtime::prepare_vlm_embeddings(
        model,
        &mut prompt_ids,
        &prompt,
        std::slice::from_ref(&image),
        |text, add_special| encode(tokenizer, text, add_special),
    )?
    .context("the model prepared no image embeddings; is it a VLM?")?;
    let (embeds, mask) = mlxcel::vlm_runtime::prepared_embedding_refs(&prepared.embeddings)?;

    let mut targets = encode(tokenizer, text, false);
    targets.truncate(max_tokens);
    anyhow::ensure!(targets.len() >= 2, "continuation needs at least two tokens");

    model.reset_runtime_state();
    let mut caches = model.make_caches();
    let prompt_arr = from_slice_i32(&prompt_ids, &[1, prompt_ids.len() as i32]);
    let prompt_logits = model.forward_with_embeddings(&prompt_arr, Some(embeds), &mut caches, mask);
    let shape = mlxcel_core::array_shape(&prompt_logits);
    anyhow::ensure!(
        shape.len() == 3,
        "forward_with_embeddings returned {shape:?}"
    );
    let vocab = shape[2];
    // The last prompt row predicts the first continuation token.
    let first = mlxcel_core::slice(&prompt_logits, &[0, shape[1] - 1, 0], &[1, shape[1], vocab]);
    eval(&first);

    let cont = &targets[..targets.len() - 1];
    let cont_arr = from_slice_i32(cont, &[1, cont.len() as i32]);
    let rest = model.forward(&cont_arr, &mut caches, None);
    let rest_shape = mlxcel_core::array_shape(&rest);
    anyhow::ensure!(
        rest_shape.len() == 3 && rest_shape[1] == cont.len() as i32,
        "model.forward returned {rest_shape:?}, not per-position logits"
    );
    let first = mlxcel_core::astype(&first, mlxcel_core::dtype::FLOAT32);
    let rest = mlxcel_core::astype(&rest, mlxcel_core::dtype::FLOAT32);
    let logits = mlxcel_core::concatenate(&first, &rest, 1);
    eval(&logits);
    Ok(ImageTrace {
        logits,
        targets,
        prompt_tokens: prompt_ids.len(),
    })
}
