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

//! The Qwen3-VL multimodal reranker: the same yes/no read as the text Qwen3
//! reranker, over a prompt whose query and documents may carry images.
//!
//! `Qwen/Qwen3-VL-Reranker-2B` is an ordinary
//! `Qwen3VLForConditionalGeneration` export with two extra side files:
//!
//! - `additional_chat_templates/reranker.jinja`, which owns the prompt. It
//!   reads `role: query` and `role: document` messages (not `user`), supplies
//!   its own default instruction when no `system` message is present, and
//!   renders each image content item as
//!   `<|vision_start|><|image_pad|><|vision_end|>`. Rendering is delegated to
//!   it verbatim so a re-exported checkpoint with a different prompt stays
//!   correct.
//! - `1_LogitScore/config.json`, which names the two answer tokens
//!   (`true_token_id` 9693 = `yes`, `false_token_id` 2152 = `no`). Its
//!   `modules.json` entry is a `LogitScore` module rather than a `Pooling`
//!   one, which is exactly why detection refuses to treat this checkpoint as
//!   an embedding export.
//!
//! Image rows run one at a time. Qwen3-VL's M-RoPE index and its DeepStack
//! injection are computed for a single sequence (`compute_rope_index` reads
//! row 0 and the DeepStack state is per request), the same constraint the
//! merged Qwen3-VL-Embedding family documents. Text-only rows in the same
//! request are still batched and left-padded, so a mixed request pays for the
//! image rows only.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::utils::{create_causal_padding_mask, slice_axis};
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use crate::embeddings::limits::{read_json, resolve_pad_token_id};
use crate::embeddings::model::ImageInput;
use crate::embeddings::tokenize::{EncodedBatch, EncodedRow, PaddingSide};
use crate::models::qwen3_vl_embedding::{apply_pixel_bounds, expand_image_placeholders};
use crate::server::chat_template::ChatTemplateProcessor;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::qwen3_generative::resolve_max_length;
use super::{RerankItem, RerankScores, Reranker, RerankerKind, sigmoid_to_vec};

/// Relative path of the reranker prompt template inside the checkpoint.
pub const RERANKER_TEMPLATE_PATH: &str = "additional_chat_templates/reranker.jinja";

/// Relative path of the module that names the two answer tokens.
pub const LOGIT_SCORE_CONFIG_PATH: &str = "1_LogitScore/config.json";

/// The Qwen3-VL yes/no reranker.
pub struct Qwen3VlReranker {
    vlm: crate::vision::Qwen3VLModel,
    template: ChatTemplateProcessor,
    tokenizer: MlxcelTokenizer,
    true_id: u32,
    false_id: u32,
    pad_token_id: u32,
    max_length: usize,
    batch_size: usize,
    /// Tokens the rendered scaffold costs with empty query and document text,
    /// measured once at load so per-row truncation has a real budget.
    scaffold_tokens: usize,
}

/// Build the message list the reranker template consumes.
///
/// The `system` message is emitted only when the caller supplies an
/// instruction: the template substitutes its own default
/// ("Given a search query, retrieve relevant candidates that answer the
/// query.") when no `system` message is present, and reproducing that default
/// here would silently pin it.
///
/// Free rather than a method so the rendering can be exercised against a
/// checkpoint's `reranker.jinja` without loading its weights.
#[must_use]
pub fn rerank_messages(
    instruction: Option<&str>,
    query: &str,
    query_has_image: bool,
    document: &str,
    document_has_image: bool,
) -> Value {
    let side = |text: &str, with_image: bool| -> Vec<Value> {
        let mut content = Vec::with_capacity(2);
        if with_image {
            content.push(json!({"type": "image"}));
        }
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
        content
    };
    let mut messages: Vec<Value> = Vec::with_capacity(3);
    if let Some(instruction) = instruction.map(str::trim).filter(|task| !task.is_empty()) {
        messages.push(json!({
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        }));
    }
    messages.push(json!({"role": "query", "content": side(query, query_has_image)}));
    messages.push(json!({
        "role": "document",
        "content": side(document, document_has_image),
    }));
    Value::Array(messages)
}

/// Keep `query` and `document` inside `budget` tokens, dropping from whichever
/// side is longer.
///
/// Same fixed point as the tokenizer's `longest_first` strategy: the shorter
/// side survives untouched whenever it fits in its half of the budget, and
/// only the longer one pays.
pub fn longest_first_keep(
    query_tokens: usize,
    document_tokens: usize,
    budget: usize,
) -> (usize, usize) {
    if query_tokens + document_tokens <= budget {
        return (query_tokens, document_tokens);
    }
    let half = budget / 2;
    if query_tokens <= half {
        (query_tokens, budget - query_tokens)
    } else if document_tokens <= budget - half {
        (budget - document_tokens, document_tokens)
    } else {
        (half, budget - half)
    }
}

/// Read `true_token_id` / `false_token_id` from `1_LogitScore/config.json`.
fn logit_score_ids(model_dir: &Path) -> Option<(u32, u32)> {
    let config = read_json(&model_dir.join("1_LogitScore").join("config.json"))?;
    let id = |key: &str| config.get(key)?.as_u64().map(|v| v as u32);
    Some((id("true_token_id")?, id("false_token_id")?))
}

impl Qwen3VlReranker {
    /// Load a Qwen3-VL reranker checkpoint from its directory.
    pub fn load(
        model_dir: &Path,
        batch_size: usize,
        max_length_override: Option<usize>,
    ) -> Result<Self> {
        let loaded = crate::loading::load_qwen3_vl(model_dir)
            .with_context(|| format!("failed to load {} as Qwen3-VL", model_dir.display()))?;
        let crate::LoadedModel::Qwen3VL(mut vlm) = loaded else {
            bail!(
                "{} did not load as a Qwen3-VL stack; the multimodal reranker needs the dense \
                 Qwen3-VL layout",
                model_dir.display()
            );
        };
        apply_pixel_bounds(&mut vlm.processor, model_dir);

        let template_path = model_dir.join(RERANKER_TEMPLATE_PATH);
        let template_source = std::fs::read_to_string(&template_path).with_context(|| {
            format!(
                "{} ships no {RERANKER_TEMPLATE_PATH}; the Qwen3-VL reranker prompt comes from \
                 that template",
                model_dir.display()
            )
        })?;
        let template = ChatTemplateProcessor::with_template(template_source);

        let tokenizer = crate::embeddings::tokenize::strip_padding_and_truncation(
            load_tokenizer(model_dir).with_context(|| {
                format!("failed to load the tokenizer in {}", model_dir.display())
            })?,
        );
        let (true_id, false_id) = match logit_score_ids(model_dir) {
            Some(ids) => ids,
            None => bail!(
                "{} ships no readable {LOGIT_SCORE_CONFIG_PATH}; the Qwen3-VL reranker reads its \
                 yes/no token ids from that module",
                model_dir.display()
            ),
        };
        let pad_token_id = resolve_pad_token_id(model_dir, &tokenizer);
        let max_length = resolve_max_length(model_dir, max_length_override);

        let scaffold = template
            .apply_raw(&rerank_messages(None, "", false, "", false), None)
            .context("failed to render the Qwen3-VL reranker prompt scaffold")?;
        let scaffold_tokens = tokenizer
            .encode(&scaffold, false)
            .context("failed to encode the Qwen3-VL reranker prompt scaffold")?
            .len();
        if scaffold_tokens >= max_length {
            bail!(
                "the Qwen3-VL reranker prompt scaffold is {scaffold_tokens} tokens, which leaves \
                 no room inside the {max_length}-token limit"
            );
        }

        tracing::info!(
            target: "mlxcel::rerank",
            true_id,
            false_id,
            scaffold_tokens,
            max_length,
            pad_token_id,
            batch_size,
            "Qwen3-VL multimodal reranker loaded"
        );
        Ok(Self {
            vlm,
            template,
            tokenizer,
            true_id,
            false_id,
            pad_token_id,
            max_length,
            batch_size: batch_size.max(1),
            scaffold_tokens,
        })
    }

    /// Visual tokens each image contributes, in prompt order.
    fn visual_token_counts(&self, images: &[ImageInput]) -> Vec<usize> {
        let decoded: Vec<image::DynamicImage> =
            images.iter().map(|input| input.image.clone()).collect();
        let merge = self.vlm.spatial_merge_size.max(1) as i32;
        self.vlm
            .processor
            .compute_grid_thw(&decoded)
            .into_iter()
            .map(|(t, h, w)| (t * (h / merge) * (w / merge)).max(0) as usize)
            .collect()
    }

    /// Shorten the query and document text so the rendered row fits.
    fn truncate_texts(
        &self,
        query: &str,
        document: &str,
        image_tokens: usize,
    ) -> Result<(String, String)> {
        let budget = self
            .max_length
            .saturating_sub(self.scaffold_tokens + image_tokens);
        let query_ids = self.tokenizer.encode(query, false)?;
        let document_ids = self.tokenizer.encode(document, false)?;
        let (keep_query, keep_document) =
            longest_first_keep(query_ids.len(), document_ids.len(), budget);
        if keep_query == query_ids.len() && keep_document == document_ids.len() {
            return Ok((query.to_string(), document.to_string()));
        }
        let decode = |ids: &[u32], keep: usize| -> Result<String> {
            if keep >= ids.len() {
                return self.tokenizer.decode(ids, false);
            }
            self.tokenizer.decode(&ids[..keep], false)
        };
        Ok((
            decode(&query_ids, keep_query)?,
            decode(&document_ids, keep_document)?,
        ))
    }

    /// Render and encode one prompt row, expanding its image placeholders.
    fn encode_row(
        &self,
        instruction: Option<&str>,
        query: &RerankItem,
        document: &RerankItem,
    ) -> Result<(EncodedRow, Vec<ImageInput>)> {
        let images: Vec<ImageInput> = [query.image.as_ref(), document.image.as_ref()]
            .into_iter()
            .flatten()
            .cloned()
            .collect();
        let counts = if images.is_empty() {
            Vec::new()
        } else {
            self.visual_token_counts(&images)
        };
        let (query_text, document_text) = self.truncate_texts(
            query.text_or_empty(),
            document.text_or_empty(),
            counts.iter().sum(),
        )?;
        let prompt = self
            .template
            .apply_raw(
                &rerank_messages(
                    instruction,
                    &query_text,
                    query.has_image(),
                    &document_text,
                    document.has_image(),
                ),
                None,
            )
            .context("failed to render the Qwen3-VL reranker prompt")?;
        let ids = self
            .tokenizer
            .encode(&prompt, false)
            .context("failed to encode the Qwen3-VL reranker prompt")?;
        let ids = if counts.is_empty() {
            ids
        } else {
            let signed: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            let ones = vec![1i32; signed.len()];
            let (expanded, _) =
                expand_image_placeholders(&signed, &ones, self.vlm.image_token_id, &counts)?;
            expanded.into_iter().map(|id| id as u32).collect()
        };
        Ok((
            EncodedRow {
                ids,
                type_ids: None,
            },
            images,
        ))
    }

    /// `[B, 1, vocab]` logits at the last column of a padded batch.
    fn last_position_logits(
        &self,
        batch: &EncodedBatch,
        images: &[ImageInput],
    ) -> Result<UniquePtr<MlxArray>> {
        let input_ids = batch.input_ids_array();
        let attention_mask = batch.attention_mask_array();
        let text_model = &self.vlm.text_model;
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();

        let merged = if images.is_empty() {
            None
        } else {
            if batch.batch != 1 {
                bail!(
                    "the Qwen3-VL reranker scores image rows one at a time (M-RoPE and DeepStack \
                     state are per sequence), got a batch of {}",
                    batch.batch
                );
            }
            let decoded: Vec<image::DynamicImage> =
                images.iter().map(|input| input.image.clone()).collect();
            let (pixel_values, grid_thw) = self.vlm.processor.preprocess_with_grid(&decoded);
            Some(
                self.vlm
                    .get_input_embeddings(&input_ids, &pixel_values, &grid_thw),
            )
        };

        let mask = create_causal_padding_mask(&attention_mask, 0);
        let mut caches = text_model.make_caches();
        let hidden = text_model.forward_hidden(
            &input_ids,
            merged.as_ref().map(|m| m.inputs_embeds.as_ref().unwrap()),
            &mut caches,
            Some(&mask),
        );
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();

        let width = batch.width as i32;
        let last = slice_axis(&hidden, 1, width - 1, width);
        Ok(text_model.lm_head_forward(&last))
    }

    /// Score one micro-batch of already-encoded rows.
    fn score_rows(&self, rows: &[EncodedRow], images: &[ImageInput]) -> Result<(Vec<f32>, usize)> {
        let batch =
            EncodedBatch::from_rows_with_padding(rows, self.pad_token_id, None, PaddingSide::Left);
        let logits = self.last_position_logits(&batch, images)?;
        let indices =
            mlxcel_core::from_slice_i32(&[self.true_id as i32, self.false_id as i32], &[2]);
        let picked = mlxcel_core::take(&logits, &indices, 2);
        let yes = slice_axis(&picked, 2, 0, 1);
        let no = slice_axis(&picked, 2, 1, 2);
        let scores = sigmoid_to_vec(&mlxcel_core::subtract(&yes, &no))?;
        if scores.len() != rows.len() {
            bail!(
                "the Qwen3-VL reranker produced {} scores for {} documents",
                scores.len(),
                rows.len()
            );
        }
        Ok((scores, batch.total_tokens()))
    }
}

impl Reranker for Qwen3VlReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeVl
    }

    fn score(
        &self,
        query: &RerankItem,
        documents: &[RerankItem],
        instruction: Option<&str>,
    ) -> Result<RerankScores> {
        let mut encoded = Vec::with_capacity(documents.len());
        for document in documents {
            encoded.push(self.encode_row(instruction, query, document)?);
        }

        let mut scores: Vec<Option<f32>> = (0..encoded.len()).map(|_| None).collect();
        let mut prompt_tokens = 0usize;

        // Image rows go through one at a time; text-only rows are batched and
        // left-padded so a text-only request keeps the batched fast path.
        let mut text_indices: Vec<usize> = Vec::new();
        for (index, (row, images)) in encoded.iter().enumerate() {
            if images.is_empty() {
                text_indices.push(index);
                continue;
            }
            let (row_scores, tokens) = self.score_rows(std::slice::from_ref(row), images)?;
            prompt_tokens += tokens;
            scores[index] = row_scores.first().copied();
        }
        for chunk in text_indices.chunks(self.batch_size) {
            let rows: Vec<EncodedRow> = chunk.iter().map(|&i| encoded[i].0.clone()).collect();
            let (chunk_scores, tokens) = self.score_rows(&rows, &[])?;
            prompt_tokens += tokens;
            for (&index, score) in chunk.iter().zip(chunk_scores) {
                scores[index] = Some(score);
            }
        }

        let scores = scores
            .into_iter()
            .map(|score| score.context("a reranked document produced no score"))
            .collect::<Result<Vec<_>>>()?;
        Ok(RerankScores {
            scores,
            prompt_tokens,
        })
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn max_length(&self) -> usize {
        self.max_length
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
#[path = "qwen3_vl_generative_tests.rs"]
mod qwen3_vl_generative_tests;
