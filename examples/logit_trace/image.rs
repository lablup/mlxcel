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
use mlxcel::server::chat_template::ChatMessage;
use mlxcel::tokenizer::MlxcelTokenizer;
use mlxcel_core::cache::SequenceId;
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
/// does for a single `--image`: a typed `[image, text]` content list when the
/// template has an image branch, otherwise the question alone through the
/// text-only render (the image then reaches the prompt as tokens). Handing a
/// template without an image branch the list made it fail and fall back to
/// the bare question with no chat turn at all, which is how the InternVL trace
/// lost its whole template (issue #2547).
fn render_prompt(model_dir: &str, tokenizer: &MlxcelTokenizer, user_prompt: &str) -> String {
    let Some(processor) = load_model_chat_template(std::path::Path::new(model_dir), tokenizer)
    else {
        return user_prompt.to_string();
    };
    let text_only = || {
        let messages = [ChatMessage {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        }];
        processor
            .apply(&messages, None)
            .unwrap_or_else(|_| user_prompt.to_string())
    };
    if !processor.supports_image_content() {
        return text_only();
    }
    let messages = serde_json::json!([{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ],
    }]);
    processor
        .apply_raw(&messages, None)
        .unwrap_or_else(|_| text_only())
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
    /// The prompt after image-token expansion, as the model reads it.
    pub prompt_ids: Vec<i32>,
}

/// Run the image-prompted arm over the first `max_tokens` tokens of `text`.
pub fn trace(
    arm: &ImageArm,
    model: &mlxcel::LoadedModel,
    tokenizer: &MlxcelTokenizer,
    model_dir: &str,
    text: &str,
    max_tokens: usize,
    batch_rows: Option<usize>,
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

    let batched = batch_rows.is_some();
    let batch_rows = batch_rows.unwrap_or(1);
    let seq_ids: Vec<_> = (0..batch_rows)
        .map(|row| SequenceId::from_raw(0x2562_1000 + row as u64))
        .collect();
    model.reset_runtime_state();
    model.prepare_sequence_state(seq_ids[0]);
    model.bind_qwen_vl_mrope_state_to_sequence(seq_ids[0]);
    let mut caches = model.make_caches();
    let prompt_arr = from_slice_i32(&prompt_ids, &[1, prompt_ids.len() as i32]);
    let prompt_logits = model.forward_with_embeddings_and_sequence_id(
        &prompt_arr,
        Some(embeds),
        Some(seq_ids[0]),
        &mut caches,
        mask,
    );
    let shape = mlxcel_core::array_shape(&prompt_logits);
    anyhow::ensure!(
        shape.len() == 3,
        "forward_with_embeddings returned {shape:?}"
    );
    let vocab = shape[2];
    // The last prompt row predicts the first continuation token.
    let first = mlxcel_core::slice(&prompt_logits, &[0, shape[1] - 1, 0], &[1, shape[1], vocab]);
    eval(&first);

    let logits = if !batched {
        let cont = &targets[..targets.len() - 1];
        let cont_arr = from_slice_i32(cont, &[1, cont.len() as i32]);
        let rest = model.forward_with_sequence_id(&cont_arr, Some(seq_ids[0]), &mut caches, None);
        let rest_shape = mlxcel_core::array_shape(&rest);
        anyhow::ensure!(
            rest_shape.len() == 3 && rest_shape[1] == cont.len() as i32,
            "model.forward returned {rest_shape:?}, not per-position logits"
        );
        let first = mlxcel_core::astype(&first, mlxcel_core::dtype::FLOAT32);
        let rest = mlxcel_core::astype(&rest, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::concatenate(&first, &rest, 1)
    } else {
        trace_batched_continuation(model, &targets, first.as_ref().unwrap(), caches, &seq_ids)?
    };
    eval(&logits);
    for seq_id in seq_ids {
        model.release_sequence_state_by_id(seq_id);
    }
    Ok(ImageTrace {
        logits,
        targets,
        prompt_ids,
    })
}

fn trace_batched_continuation(
    model: &mlxcel::LoadedModel,
    targets: &[i32],
    first: &MlxArray,
    image_caches: Vec<mlxcel_core::layers::KVCache>,
    seq_ids: &[SequenceId],
) -> Result<UniquePtr<MlxArray>> {
    let mut caches = vec![image_caches];
    // Text-only companions have distinct prompt lengths and explicit fresh
    // identities. Binding after the image row drains no image state: that was
    // already transferred to `seq_ids[0]` before its prefill.
    for row in 1..seq_ids.len() {
        model.prepare_sequence_state(seq_ids[row]);
        model.bind_qwen_vl_mrope_state_to_sequence(seq_ids[row]);
        let prefix_len = row + 1;
        let prefix: Vec<i32> = (0..prefix_len)
            .map(|position| targets[(position + row) % targets.len()])
            .collect();
        let input = from_slice_i32(&prefix, &[1, prefix_len as i32]);
        let mut row_caches = model.make_caches();
        let warm =
            model.forward_with_sequence_id(&input, Some(seq_ids[row]), &mut row_caches, None);
        eval(&warm);
        caches.push(row_caches);
    }

    let mut traced = vec![mlxcel_core::astype(first, mlxcel_core::dtype::FLOAT32)];
    for position in 0..targets.len() - 1 {
        let tokens: Vec<i32> = (0..seq_ids.len())
            .map(|row| targets[(position + row) % targets.len()])
            .collect();
        let input = from_slice_i32(&tokens, &[seq_ids.len() as i32, 1]);
        let logits = {
            let mut rows: Vec<&mut [mlxcel_core::layers::KVCache]> =
                caches.iter_mut().map(Vec::as_mut_slice).collect();
            model.forward_batched_with_ids(&input, Some(seq_ids), &mut rows, None)
        };
        eval(&logits);
        let shape = mlxcel_core::array_shape(&logits);
        anyhow::ensure!(
            shape.len() == 3 && shape[0] == seq_ids.len() as i32 && shape[1] == 1,
            "image batched decode returned {shape:?}, expected [{}, 1, vocab]",
            seq_ids.len()
        );
        traced.push(mlxcel_core::astype(
            &mlxcel_core::slice(&logits, &[0, 0, 0], &[1, 1, shape[2]]),
            mlxcel_core::dtype::FLOAT32,
        ));
    }
    let refs: Vec<&MlxArray> = traced.iter().map(|row| &**row).collect();
    Ok(mlxcel_core::concatenate_many(&refs, 1))
}
