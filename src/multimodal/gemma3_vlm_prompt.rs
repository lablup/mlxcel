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

//! Gemma 3 VLM image-block newline merging (issue #2547).
//!
//! transformers' `Gemma3Processor` rewrites the rendered *text*: each
//! `<start_of_image>` becomes `\n\n<start_of_image>` + soft tokens +
//! `<end_of_image>\n\n`, and only then is the string tokenized. Gemma's
//! vocabulary has one token per run of newlines, so the wrapper's `\n\n` fuses
//! with a newline the template put next to the marker: the chat template's
//! `user\n<start_of_image>` becomes `user` + `\n\n\n`, one token.
//!
//! mlxcel expands the block on the token stream
//! ([`crate::vlm_prompt::apply_image_token_blocks`]), which leaves `\n` and
//! `\n\n` as two tokens and made every image prompt one token longer than the
//! reference (273 against 272 for one image). [`merge_image_block_newline_runs`]
//! fuses a wrapper newline run with an adjacent template newline run into the
//! single token the reference tokenizer produces.
//!
//! [`expand_image_blocks_with_newline_fusion`] is the one entry point for both
//! backends: the MLX path (`vlm_runtime`, the newline-run table cached per
//! model on `VisionModule`) and the OpenXLA host preprocessor
//! (`host_preprocessor::gemma3_iree_prompt_tokens`, the table built at load),
//! so the two cannot drift apart again (issue #2761).

use std::sync::OnceLock;

use crate::vlm_prompt::{
    ImageTokenBlockError, ImageTokenBlockInfo, ImageTokenBlockStats, apply_image_token_blocks,
};

/// Longest newline run probed for a single-token encoding. Gemma 3's
/// vocabulary covers runs up to 31.
const MAX_NEWLINE_RUN: usize = 64;

/// Token ids of newline runs: entry `k - 1` is the single token that encodes
/// `k` consecutive `\n`. Probing stops at the first run that does not encode to
/// one token, so the table only covers contiguous run lengths.
pub fn newline_run_token_ids<E>(encode: &mut E) -> Vec<i32>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let mut ids = Vec::new();
    for len in 1..=MAX_NEWLINE_RUN {
        match encode(&"\n".repeat(len), false).as_slice() {
            [id] => ids.push(*id),
            _ => break,
        }
    }
    ids
}

/// The model's newline-run table, probed with `encode` on first use and
/// reused after. Probing costs one encode per run length (32 on Gemma 3), so
/// the table is built once per model rather than once per image request.
pub fn cached_newline_runs<'a, E>(cell: &'a OnceLock<Vec<i32>>, encode: &mut E) -> &'a [i32]
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    cell.get_or_init(|| newline_run_token_ids(encode))
}

/// Whether the layout wraps each image block in tokens of its own (Gemma 3's
/// `\n\n`), the only case [`merge_image_block_newline_runs`] can change.
pub fn has_image_block_wrapper(info: &ImageTokenBlockInfo) -> bool {
    !info.block_prefix_tokens.is_empty() || !info.block_suffix_tokens.is_empty()
}

/// Expand the image blocks in `tokens` and, when the layout wraps each block,
/// fuse the wrapper newlines with the template's as the reference tokenizer
/// does. Returns what [`apply_image_token_blocks`] returns; a layout without a
/// wrapper is expanded and nothing else, so `newline_runs` may be empty there.
pub fn expand_image_blocks_with_newline_fusion(
    tokens: &mut Vec<i32>,
    info: &ImageTokenBlockInfo,
    image_count: usize,
    newline_runs: &[i32],
) -> Result<Option<ImageTokenBlockStats>, ImageTokenBlockError> {
    let stats = apply_image_token_blocks(tokens, info.clone(), image_count)?;
    if stats.is_some() && has_image_block_wrapper(info) {
        merge_image_block_newline_runs(tokens, info, newline_runs);
    }
    Ok(stats)
}

fn run_len(newline_runs: &[i32], token: i32) -> Option<usize> {
    newline_runs
        .iter()
        .position(|&id| id == token)
        .map(|index| index + 1)
}

/// Tokens for `total` consecutive newlines, longest runs first, which is how
/// the reference tokenizer splits a run longer than its longest newline token
/// (33 newlines encode as the 31-run then the 2-run).
fn encode_newline_run(newline_runs: &[i32], mut total: usize) -> Vec<i32> {
    let longest = newline_runs.len();
    let mut out = Vec::with_capacity(total.div_ceil(longest));
    while total > longest {
        out.push(newline_runs[longest - 1]);
        total -= longest;
    }
    if total > 0 {
        out.push(newline_runs[total - 1]);
    }
    out
}

/// Fuse each image block's newline wrapper with the newline runs around it.
///
/// The wrapper tokens are `info.block_prefix_tokens` right before every
/// `<start_of_image>` and `info.block_suffix_tokens` right after every
/// `<end_of_image>`. Every maximal stretch of newline-run tokens (see
/// [`newline_run_token_ids`]) that contains a wrapper token is re-encoded as
/// the reference tokenizer encodes that many newlines in a row, so a template
/// newline next to the marker, or the space between two images, becomes one
/// token again. Stretches without a wrapper token came from the tokenizer and
/// are left as they are. Returns the number of stretches re-encoded.
pub fn merge_image_block_newline_runs(
    prompt_tokens: &mut Vec<i32>,
    info: &ImageTokenBlockInfo,
    newline_runs: &[i32],
) -> usize {
    if !info.use_boi_eoi || newline_runs.is_empty() {
        return 0;
    }
    let prefix = info.block_prefix_tokens.as_slice();
    let suffix = info.block_suffix_tokens.as_slice();
    let is_run = |token: i32| run_len(newline_runs, token).is_some();
    if !prefix.iter().chain(suffix).any(|&token| is_run(token)) {
        return 0;
    }

    let n = prompt_tokens.len();
    let mut wrapper = vec![false; n];
    for (index, &token) in prompt_tokens.iter().enumerate() {
        if token == info.boi_token_id
            && !prefix.is_empty()
            && index >= prefix.len()
            && prompt_tokens[index - prefix.len()..index] == *prefix
        {
            wrapper[index - prefix.len()..index].fill(true);
        }
        if token == info.eoi_token_id
            && !suffix.is_empty()
            && prompt_tokens[index + 1..].starts_with(suffix)
        {
            wrapper[index + 1..index + 1 + suffix.len()].fill(true);
        }
    }

    let mut merges = 0;
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if !is_run(prompt_tokens[i]) {
            out.push(prompt_tokens[i]);
            i += 1;
            continue;
        }
        let start = i;
        while i < n && is_run(prompt_tokens[i]) {
            i += 1;
        }
        let stretch = &prompt_tokens[start..i];
        if !wrapper[start..i].contains(&true) || stretch.len() == 1 {
            out.extend_from_slice(stretch);
            continue;
        }
        let total = stretch
            .iter()
            .filter_map(|&token| run_len(newline_runs, token))
            .sum();
        let fused = encode_newline_run(newline_runs, total);
        if fused.as_slice() != stretch {
            merges += 1;
        }
        out.extend(fused);
    }
    *prompt_tokens = out;
    merges
}

#[cfg(test)]
#[path = "gemma3_vlm_prompt_tests.rs"]
mod tests;
