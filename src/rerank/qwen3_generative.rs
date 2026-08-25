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

//! The Qwen3 generative reranker: an ordinary `Qwen3ForCausalLM` checkpoint
//! asked a yes/no question and read at one position.
//!
//! `Qwen/Qwen3-Reranker-*` ships no classification head. The reference recipe
//! wraps the pair in a fixed chat prompt that ends on the assistant header,
//! runs one prefill, and turns the two candidate answers into a probability:
//!
//! ```text
//! score = sigmoid(logits[L - 1, id("yes")] - logits[L - 1, id("no")])
//! ```
//!
//! Three details make a batch of those reads correct:
//!
//! - the prompt is assembled from token ids, not from one formatted string, so
//!   truncation can shorten the pair without ever touching the prefix or the
//!   assistant header the score is read from;
//! - rows are **left**-padded, which is what puts every row's last real token
//!   at column `L - 1`. Right padding would read the score off a pad token for
//!   every row but the longest;
//! - the mask is [`create_causal_padding_mask`], so the pad prefix is blocked
//!   as a key everywhere and the leading padding rows keep a finite softmax.
//!
//! Absolute positions still run `0..L` over the padded row, exactly as the
//! reference `Qwen3ForCausalLM.forward(input_ids, attention_mask)` does when it
//! derives `position_ids` from the cache position: the pad prefix shifts every
//! real token by the same amount, and RoPE only sees position differences among
//! the keys that are not masked out.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::utils::{create_causal_padding_mask, slice_axis};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::embeddings::limits::{derive_max_length, resolve_pad_token_id};
use crate::embeddings::tokenize::{EncodedBatch, EncodedRow, PaddingSide, truncate_token_ids};
use crate::models::qwen3::{ModelArgs, Qwen3Model};
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::{
    RERANK_MAX_LENGTH_CAP, RerankItem, RerankScores, Reranker, RerankerKind, sigmoid_to_vec,
};

/// System turn and the opening of the user turn. Byte-identical to the
/// reference recipe published on the `Qwen/Qwen3-Reranker-*` model cards.
pub const PROMPT_PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";

/// Close of the user turn plus the assistant header with an empty think block.
/// The last token of this run is the position the score is read from.
pub const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Task description used when the request carries no `instruction`.
pub const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Render the variable middle of the prompt.
#[must_use]
pub fn prompt_content(instruction: &str, query: &str, document: &str) -> String {
    format!("<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}")
}

/// The task description a request actually renders with: its own
/// `instruction` when it carries a non-blank one, otherwise
/// [`DEFAULT_INSTRUCTION`].
#[must_use]
pub fn resolve_instruction(instruction: Option<&str>) -> &str {
    instruction
        .map(str::trim)
        .filter(|task| !task.is_empty())
        .unwrap_or(DEFAULT_INSTRUCTION)
}

/// The Qwen3 yes/no reranker.
pub struct Qwen3Reranker {
    model: Qwen3Model,
    tokenizer: MlxcelTokenizer,
    yes_id: u32,
    no_id: u32,
    prefix_ids: Vec<u32>,
    suffix_ids: Vec<u32>,
    pad_token_id: u32,
    max_length: usize,
    batch_size: usize,
}

/// Resolve the single token id a yes/no answer encodes to.
///
/// The score is a difference of two logits, so each answer has to be exactly
/// one token; a tokenizer that splits either word would make the read
/// meaningless rather than merely inaccurate.
fn single_token_id(tokenizer: &MlxcelTokenizer, word: &str) -> Result<u32> {
    let ids = tokenizer
        .encode(word, false)
        .with_context(|| format!("failed to encode the reranker answer token `{word}`"))?;
    match ids.as_slice() {
        [id] => Ok(*id),
        other => bail!(
            "the Qwen3 reranker needs `{word}` to be a single token, this tokenizer encodes it \
             as {other:?}"
        ),
    }
}

impl Qwen3Reranker {
    /// Load a Qwen3 reranker checkpoint from its directory.
    pub fn load(
        model_dir: &Path,
        batch_size: usize,
        max_length_override: Option<usize>,
    ) -> Result<Self> {
        let (model, args) = Qwen3Model::load(model_dir)
            .map_err(|e| anyhow::anyhow!("failed to load {} as Qwen3: {e}", model_dir.display()))?;
        let tokenizer = crate::embeddings::tokenize::strip_padding_and_truncation(
            load_tokenizer(model_dir).with_context(|| {
                format!("failed to load the tokenizer in {}", model_dir.display())
            })?,
        );
        let pad_token_id = resolve_pad_token_id(model_dir, &tokenizer);
        Self::from_parts(
            model,
            &args,
            tokenizer,
            pad_token_id,
            resolve_max_length(model_dir, max_length_override),
            batch_size,
        )
    }

    /// Assemble a reranker from an already-loaded backbone and tokenizer.
    ///
    /// Split from [`Self::load`] so the prompt assembly and the yes/no
    /// resolution can be exercised on a synthetic backbone.
    pub(crate) fn from_parts(
        model: Qwen3Model,
        args: &ModelArgs,
        tokenizer: MlxcelTokenizer,
        pad_token_id: u32,
        max_length: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let yes_id = single_token_id(&tokenizer, "yes")?;
        let no_id = single_token_id(&tokenizer, "no")?;
        for (word, id) in [("yes", yes_id), ("no", no_id)] {
            if args.vocab_size > 0 && id as usize >= args.vocab_size {
                bail!(
                    "the reranker answer token `{word}` is id {id}, outside the model's \
                     vocab_size {}",
                    args.vocab_size
                );
            }
        }
        let prefix_ids = tokenizer
            .encode(PROMPT_PREFIX, false)
            .context("failed to encode the reranker prompt prefix")?;
        let suffix_ids = tokenizer
            .encode(PROMPT_SUFFIX, false)
            .context("failed to encode the reranker prompt suffix")?;
        if prefix_ids.len() + suffix_ids.len() >= max_length {
            bail!(
                "the reranker prompt scaffold is {} tokens, which leaves no room inside the \
                 {max_length}-token limit",
                prefix_ids.len() + suffix_ids.len()
            );
        }
        tracing::info!(
            target: "mlxcel::rerank",
            yes_id,
            no_id,
            prefix_tokens = prefix_ids.len(),
            suffix_tokens = suffix_ids.len(),
            max_length,
            pad_token_id,
            batch_size,
            "Qwen3 generative reranker loaded"
        );
        Ok(Self {
            model,
            tokenizer,
            yes_id,
            no_id,
            prefix_ids,
            suffix_ids,
            pad_token_id,
            max_length,
            batch_size: batch_size.max(1),
        })
    }

    /// Tokens the pair itself may occupy once the scaffold is reserved.
    fn content_budget(&self) -> usize {
        self.max_length
            .saturating_sub(self.prefix_ids.len() + self.suffix_ids.len())
    }

    /// Build one prompt row: prefix, truncated content, suffix.
    pub(crate) fn encode_row(
        &self,
        instruction: &str,
        query: &str,
        document: &str,
    ) -> Result<Vec<u32>> {
        let content = prompt_content(instruction, query, document);
        let content_ids = self
            .tokenizer
            .encode(&content, false)
            .context("failed to encode a reranker query/document pair")?;
        let content_ids = truncate_token_ids(&content_ids, self.content_budget());
        let mut ids =
            Vec::with_capacity(self.prefix_ids.len() + content_ids.len() + self.suffix_ids.len());
        ids.extend_from_slice(&self.prefix_ids);
        ids.extend_from_slice(&content_ids);
        ids.extend_from_slice(&self.suffix_ids);
        Ok(ids)
    }

    /// `[B, vocab]` logits at the last column of a left-padded batch.
    fn last_position_logits(&self, batch: &EncodedBatch) -> UniquePtr<MlxArray> {
        let input_ids = batch.input_ids_array();
        let attention_mask = batch.attention_mask_array();
        let mask = create_causal_padding_mask(&attention_mask, 0);
        let mut caches = self.model.make_caches();
        let hidden = self
            .model
            .forward_hidden(&input_ids, None, &mut caches, Some(&mask));
        // Apply the head to one column instead of the whole sequence: the
        // `[B, L, vocab]` tensor would be hundreds of megabytes for a long
        // batch and every row but the last is discarded.
        let width = batch.width as i32;
        let last = slice_axis(&hidden, 1, width - 1, width);
        match &self.model.lm_head {
            Some(head) => head.forward(&last),
            None => self.model.embed_tokens.as_linear(&last),
        }
    }

    /// Score one micro-batch of prompt rows.
    fn score_rows(&self, rows: &[EncodedRow]) -> Result<(Vec<f32>, usize)> {
        let batch =
            EncodedBatch::from_rows_with_padding(rows, self.pad_token_id, None, PaddingSide::Left);
        let logits = self.last_position_logits(&batch);
        let indices = mlxcel_core::from_slice_i32(&[self.yes_id as i32, self.no_id as i32], &[2]);
        // `[B, 1, vocab]` gathered on the vocabulary axis -> `[B, 1, 2]`.
        let picked = mlxcel_core::take(&logits, &indices, 2);
        let yes = slice_axis(&picked, 2, 0, 1);
        let no = slice_axis(&picked, 2, 1, 2);
        let scores = sigmoid_to_vec(&mlxcel_core::subtract(&yes, &no))?;
        if scores.len() != rows.len() {
            bail!(
                "the Qwen3 reranker produced {} scores for {} documents",
                scores.len(),
                rows.len()
            );
        }
        Ok((scores, batch.total_tokens()))
    }
}

/// `max_length` for a generative reranker: the shared 8192 ceiling, lowered by
/// the checkpoint's declared limits and by `--rerank-max-length`.
///
/// The position table is RoPE, so `max_position_embeddings` (40960 on the 0.6B
/// checkpoint, 262144 on the VL one) is not a cap worth reading; the ceiling is
/// what keeps one pair's prefill bounded.
pub(crate) fn resolve_max_length(model_dir: &Path, override_value: Option<usize>) -> usize {
    derive_max_length(model_dir, false, override_value).min(RERANK_MAX_LENGTH_CAP)
}

impl Reranker for Qwen3Reranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn score(
        &self,
        query: &RerankItem,
        documents: &[RerankItem],
        instruction: Option<&str>,
    ) -> Result<RerankScores> {
        if query.has_image() || documents.iter().any(RerankItem::has_image) {
            bail!("the Qwen3 reranker is text-only; use a Qwen3-VL reranker for image documents");
        }
        let instruction = resolve_instruction(instruction);
        let query_text = query.text_or_empty();
        let rows = documents
            .iter()
            .map(|document| {
                Ok(EncodedRow {
                    ids: self.encode_row(instruction, query_text, document.text_or_empty())?,
                    type_ids: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut scores = Vec::with_capacity(rows.len());
        let mut prompt_tokens = 0usize;
        for chunk in rows.chunks(self.batch_size) {
            let (chunk_scores, tokens) = self.score_rows(chunk)?;
            prompt_tokens += tokens;
            scores.extend(chunk_scores);
        }
        Ok(RerankScores {
            scores,
            prompt_tokens,
        })
    }

    fn max_length(&self) -> usize {
        self.max_length
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
#[path = "qwen3_generative_tests.rs"]
mod qwen3_generative_tests;
