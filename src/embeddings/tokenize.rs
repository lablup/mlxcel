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

//! Batch tokenization with right padding for the embedding engine.
//!
//! Rows are encoded one at a time (so per-text truncation can keep a
//! trailing special token), then packed into a right-padded `[B, L]` block
//! with an attention mask, optional segment ids and the per-row real-token
//! counts the `usage` field reports.

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::tokenizer::MlxcelTokenizer;

/// One encoded input before padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedRow {
    /// Token ids, special tokens included, already truncated.
    pub ids: Vec<u32>,
    /// Segment ids (`token_type_ids`) when requested; same length as `ids`.
    pub type_ids: Option<Vec<u32>>,
}

/// Per-row encoding options.
#[derive(Debug, Clone, Copy)]
pub struct EncodeOptions {
    /// Let the tokenizer's post-processor add its special tokens.
    pub add_special_tokens: bool,
    /// Right-truncate rows longer than this, keeping trailing special tokens.
    pub max_length: usize,
    /// Collect `token_type_ids` (BERT pairs).
    pub with_token_type_ids: bool,
}

/// Which end of a row the padding is written to.
///
/// Encoders read the whole row through a bidirectional mask, so the padding
/// end does not matter and [`PaddingSide::Right`] keeps the real tokens at
/// index `0..n`. A causal decoder scored at one position is different: the
/// generative rerankers read the logits at `L - 1`, which is only the last
/// real token of every row when the padding sits in front of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingSide {
    /// Real tokens first, padding after them (the embedding engine).
    #[default]
    Right,
    /// Padding first, real tokens flush against the end of the row.
    Left,
}

/// A padded `[B, L]` block ready to become MLX arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedBatch {
    /// Row-major `[B, L]` token ids.
    pub input_ids: Vec<i32>,
    /// Row-major `[B, L]`, `1` = real token, `0` = padding.
    pub attention_mask: Vec<i32>,
    /// Row-major `[B, L]` segment ids when any row carried them.
    pub token_type_ids: Option<Vec<i32>>,
    /// Real-token count per row (special tokens included).
    pub token_counts: Vec<usize>,
    /// `B`.
    pub batch: usize,
    /// `L`.
    pub width: usize,
}

impl EncodedBatch {
    /// Pack rows into a right-padded block, padded to the longest row (or to
    /// `pad_to` when the family requires a fixed width) with `pad_id`.
    pub fn from_rows(rows: &[EncodedRow], pad_id: u32, pad_to: Option<usize>) -> Self {
        Self::from_rows_with_padding(rows, pad_id, pad_to, PaddingSide::Right)
    }

    /// Pack rows into a block padded on `side`.
    ///
    /// `token_counts` reports the real tokens of each row either way; only the
    /// columns a row occupies change.
    pub fn from_rows_with_padding(
        rows: &[EncodedRow],
        pad_id: u32,
        pad_to: Option<usize>,
        side: PaddingSide,
    ) -> Self {
        let longest = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let width = pad_to.map_or(longest, |w| w.max(longest)).max(1);
        let batch = rows.len();
        let any_type_ids = rows.iter().any(|r| r.type_ids.is_some());

        let mut input_ids = Vec::with_capacity(batch * width);
        let mut attention_mask = Vec::with_capacity(batch * width);
        let mut token_type_ids = any_type_ids.then(|| Vec::with_capacity(batch * width));
        let mut token_counts = Vec::with_capacity(batch);

        for row in rows {
            let n = row.ids.len();
            let (before, after) = match side {
                PaddingSide::Left => (width - n, 0),
                PaddingSide::Right => (0, width - n),
            };
            token_counts.push(n);
            input_ids.extend(std::iter::repeat_n(pad_id as i32, before));
            input_ids.extend(row.ids.iter().map(|&id| id as i32));
            input_ids.extend(std::iter::repeat_n(pad_id as i32, after));
            attention_mask.extend(std::iter::repeat_n(0, before));
            attention_mask.extend(std::iter::repeat_n(1, n));
            attention_mask.extend(std::iter::repeat_n(0, after));
            if let Some(types) = token_type_ids.as_mut() {
                types.extend(std::iter::repeat_n(0, before));
                match &row.type_ids {
                    Some(ids) => types.extend(ids.iter().map(|&t| t as i32)),
                    None => types.extend(std::iter::repeat_n(0, n)),
                }
                types.extend(std::iter::repeat_n(0, after));
            }
        }

        Self {
            input_ids,
            attention_mask,
            token_type_ids,
            token_counts,
            batch,
            width,
        }
    }

    fn shape(&self) -> [i32; 2] {
        [self.batch as i32, self.width as i32]
    }

    /// `[B, L]` int32 token ids.
    pub fn input_ids_array(&self) -> UniquePtr<MlxArray> {
        mlxcel_core::from_slice_i32(&self.input_ids, &self.shape())
    }

    /// `[B, L]` int32 attention mask.
    pub fn attention_mask_array(&self) -> UniquePtr<MlxArray> {
        mlxcel_core::from_slice_i32(&self.attention_mask, &self.shape())
    }

    /// `[B, L]` int32 segment ids when present.
    pub fn token_type_ids_array(&self) -> Option<UniquePtr<MlxArray>> {
        self.token_type_ids
            .as_ref()
            .map(|ids| mlxcel_core::from_slice_i32(ids, &self.shape()))
    }

    /// Total real tokens across the batch.
    pub fn total_tokens(&self) -> usize {
        self.token_counts.iter().sum()
    }
}

/// Number of trailing entries flagged as special in `special_mask`.
fn trailing_special_count(special_mask: &[u32]) -> usize {
    special_mask
        .iter()
        .rev()
        .take_while(|&&flag| flag != 0)
        .count()
}

/// Right-truncate `ids` to `max_length`, keeping the trailing special tokens
/// the tokenizer appended (so a Qwen3-Embedding input keeps its
/// `<|endoftext|>` and a BERT input its `[SEP]`).
pub fn truncate_keeping_trailing_special(
    ids: Vec<u32>,
    type_ids: Option<Vec<u32>>,
    special_mask: &[u32],
    max_length: usize,
) -> EncodedRow {
    let len = ids.len();
    if len <= max_length || max_length == 0 {
        return EncodedRow { ids, type_ids };
    }
    let trailing = trailing_special_count(special_mask).min(max_length);
    let keep = max_length - trailing;
    let cut = |v: &[u32]| -> Vec<u32> {
        let mut out = Vec::with_capacity(max_length);
        out.extend_from_slice(&v[..keep]);
        out.extend_from_slice(&v[len - trailing..]);
        out
    };
    EncodedRow {
        ids: cut(&ids),
        type_ids: type_ids.as_deref().map(cut),
    }
}

/// Plain right truncation for verbatim token-id inputs (no special-token
/// bookkeeping: the caller supplied the ids as they are).
pub fn truncate_token_ids(ids: &[u32], max_length: usize) -> Vec<u32> {
    if max_length == 0 || ids.len() <= max_length {
        ids.to_vec()
    } else {
        ids[..max_length].to_vec()
    }
}

/// Special-token mask for tokenizers without a HuggingFace encoding: the
/// tokens that appear in the special-token run but not in the plain run
/// are the post-processor's additions.
fn infer_special_mask(with: &[u32], without: &[u32]) -> Vec<u32> {
    let mut mask = vec![1u32; with.len()];
    if without.is_empty() {
        return mask;
    }
    if let Some(pos) = with
        .windows(without.len())
        .position(|window| window == without)
    {
        for flag in &mut mask[pos..pos + without.len()] {
            *flag = 0;
        }
    } else {
        mask.fill(0);
    }
    mask
}

/// Encode one text into a truncated row.
pub fn encode_row(
    tokenizer: &MlxcelTokenizer,
    text: &str,
    opts: EncodeOptions,
) -> Result<EncodedRow> {
    if let Some(hf) = tokenizer.hf_tokenizer() {
        let encoding = hf
            .encode(text, opts.add_special_tokens)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let ids = encoding.get_ids().to_vec();
        let type_ids = opts
            .with_token_type_ids
            .then(|| encoding.get_type_ids().to_vec());
        return Ok(truncate_keeping_trailing_special(
            ids,
            type_ids,
            encoding.get_special_tokens_mask(),
            opts.max_length,
        ));
    }

    let ids = tokenizer.encode(text, opts.add_special_tokens)?;
    let special_mask = if opts.add_special_tokens {
        let without = tokenizer.encode(text, false)?;
        infer_special_mask(&ids, &without)
    } else {
        vec![0; ids.len()]
    };
    let type_ids = opts.with_token_type_ids.then(|| vec![0; ids.len()]);
    Ok(truncate_keeping_trailing_special(
        ids,
        type_ids,
        &special_mask,
        opts.max_length,
    ))
}

/// Encode a `(text_a, text_b)` pair through the tokenizer's pair template
/// (`[CLS] a [SEP] b [SEP]` for BERT). Requires a HuggingFace tokenizer.
///
/// Used by: the `/v1/rerank` cross-encoder path.
pub fn encode_pair_row(
    tokenizer: &MlxcelTokenizer,
    text_a: &str,
    text_b: &str,
    opts: EncodeOptions,
) -> Result<EncodedRow> {
    let Some(hf) = tokenizer.hf_tokenizer() else {
        bail!("pair encoding requires a tokenizer.json (HuggingFace) tokenizer");
    };
    let encoding = hf
        .encode((text_a, text_b), opts.add_special_tokens)
        .map_err(|e| anyhow::anyhow!("pair tokenization failed: {e}"))?;
    let ids = encoding.get_ids().to_vec();
    let type_ids = opts
        .with_token_type_ids
        .then(|| encoding.get_type_ids().to_vec());
    Ok(truncate_keeping_trailing_special(
        ids,
        type_ids,
        encoding.get_special_tokens_mask(),
        opts.max_length,
    ))
}

/// Encode `texts` into one right-padded batch.
pub fn encode_batch(
    tokenizer: &MlxcelTokenizer,
    texts: &[&str],
    opts: EncodeOptions,
    pad_id: u32,
    pad_to: Option<usize>,
) -> Result<EncodedBatch> {
    let rows = texts
        .iter()
        .map(|text| encode_row(tokenizer, text, opts))
        .collect::<Result<Vec<_>>>()?;
    Ok(EncodedBatch::from_rows(&rows, pad_id, pad_to))
}

/// Encode text pairs into one right-padded batch.
pub fn encode_pairs(
    tokenizer: &MlxcelTokenizer,
    pairs: &[(&str, &str)],
    opts: EncodeOptions,
    pad_id: u32,
    pad_to: Option<usize>,
) -> Result<EncodedBatch> {
    let rows = pairs
        .iter()
        .map(|(a, b)| encode_pair_row(tokenizer, a, b, opts))
        .collect::<Result<Vec<_>>>()?;
    Ok(EncodedBatch::from_rows(&rows, pad_id, pad_to))
}

/// Return a tokenizer with any `tokenizer.json` built-in padding and
/// truncation removed.
///
/// sentence-transformers exports (`all-MiniLM-L6-v2` pads to a fixed 128 and
/// truncates at 128) bake those into `tokenizer.json`; the embedding engine
/// pads per micro-batch and truncates per checkpoint limit, so the built-in
/// settings would only waste tokens and hide the real length.
pub fn strip_padding_and_truncation(tokenizer: MlxcelTokenizer) -> MlxcelTokenizer {
    match tokenizer {
        MlxcelTokenizer::HuggingFace(mut hf) => {
            hf.with_padding(None);
            // `with_truncation(None)` cannot fail: only a `Some` config is
            // validated, and clearing it always succeeds.
            let _ = hf.with_truncation(None);
            MlxcelTokenizer::HuggingFace(hf)
        }
        other => other,
    }
}
