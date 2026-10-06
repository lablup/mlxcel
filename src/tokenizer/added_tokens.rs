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

//! `tokenizer_config.json` added tokens that `tokenizer.json` does not carry.
//!
//! Some conversions declare an added token only in `tokenizer_config.json`'s
//! `added_tokens_decoder`. Granite Vision 3.2 is one: its image placeholder
//! `<image>` (id 49155) is absent from `tokenizer.json`'s `added_tokens`, so the
//! `tokenizers` crate alone splits it into `<`, `image`, `>` and the VLM prompt
//! never carries the placeholder the image block expands from (issue #1683).
//! `transformers`' `PreTrainedTokenizerFast.__init__` registers every
//! `added_tokens_decoder` entry the fast tokenizer is missing, which is how
//! mlx-vlm encodes `<image>` as 49155 on the same checkpoint;
//! [`reconcile_config_added_tokens`] does the same here.

use std::path::Path;

use anyhow::Result;
use tokenizers::{AddedToken, Model};

/// Parse an `added_tokens_decoder` object into `(id, AddedToken)` pairs sorted
/// by id, keeping each entry's `special` / `single_word` / `lstrip` / `rstrip`
/// / `normalized` flags. `special` defaults to `false`, matching HuggingFace's
/// `AddedToken` default.
pub(super) fn parse_added_tokens_decoder(
    decoder: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<(u32, AddedToken)>> {
    let mut out = Vec::with_capacity(decoder.len());
    for (id, entry) in decoder {
        let id: u32 = id
            .parse()
            .map_err(|e| anyhow::anyhow!("added_tokens_decoder key {id:?}: {e}"))?;
        let content = entry
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("added_tokens_decoder[{id}] has no content"))?;
        let flag = |name: &str, default: bool| {
            entry.get(name).and_then(|v| v.as_bool()).unwrap_or(default)
        };
        let token = AddedToken::from(content.to_string(), flag("special", false))
            .single_word(flag("single_word", false))
            .lstrip(flag("lstrip", false))
            .rstrip(flag("rstrip", false))
            .normalized(flag("normalized", false));
        out.push((id, token));
    }
    out.sort_by_key(|(id, _)| *id);
    Ok(out)
}

/// The id the `tokenizers` crate gives the next added token whose content is
/// not already in the vocabulary. Mirrors `AddedVocabulary::add_tokens`
/// (tokenizers 0.22): one past the highest added id when that id is at or
/// beyond the model vocab, otherwise the model vocab size.
fn next_added_token_id(tokenizer: &tokenizers::Tokenizer) -> u32 {
    let model_size = tokenizer.get_model().get_vocab_size() as u32;
    match tokenizer.get_added_tokens_decoder().keys().copied().max() {
        Some(max) if max >= model_size || model_size == 0 => max + 1,
        _ => model_size,
    }
}

/// Register the `added_tokens_decoder` entries of `model_path`'s
/// `tokenizer_config.json` that `tokenizer` does not already know.
///
/// An entry is skipped when its content already resolves to an id or its id is
/// already taken: `tokenizer.json` is the more specific source, and a
/// disagreement between the two files is left as the checkpoint ships it.
/// Because the crate hands out added-token ids sequentially and offers no way
/// to choose one, an entry is added only when its declared id is exactly the
/// id the crate would assign next. The first entry that is not (a gap in the
/// declared ids) stops the reconciliation with a warning rather than an error:
/// such a checkpoint loaded before this repair and keeps loading, with the
/// remaining tokens split by the base model as they were.
///
/// Returns the number of tokens registered.
pub(super) fn reconcile_config_added_tokens(
    tokenizer: &mut tokenizers::Tokenizer,
    model_path: &Path,
) -> usize {
    let Some(config) = std::fs::read_to_string(model_path.join("tokenizer_config.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
    else {
        return 0;
    };
    let Some(decoder) = config
        .get("added_tokens_decoder")
        .and_then(|value| value.as_object())
    else {
        return 0;
    };
    let entries = match parse_added_tokens_decoder(decoder) {
        Ok(entries) => entries,
        Err(err) => {
            tracing::warn!(
                model = %model_path.display(),
                error = %err,
                "ignoring malformed added_tokens_decoder in tokenizer_config.json"
            );
            return 0;
        }
    };

    // Plan first, then register: every `add_tokens` call rebuilds the added
    // vocabulary's matchers, so per-token calls cost ~160 ms per 1000 tokens
    // (see `build_qwen2_bpe_tokenizer`), and ERNIE 4.5 conversions declare
    // about 1000 config-only tokens.
    let mut planned: Vec<(u32, AddedToken)> = Vec::new();
    let mut planned_contents = std::collections::HashSet::new();
    let mut next_id = next_added_token_id(tokenizer);
    for (id, token) in entries {
        if tokenizer.token_to_id(&token.content).is_some() || tokenizer.id_to_token(id).is_some() {
            continue;
        }
        if id != next_id || !planned_contents.insert(token.content.clone()) {
            tracing::warn!(
                model = %model_path.display(),
                token = %token.content,
                declared_id = id,
                next_id,
                "tokenizer_config.json declares an added token that tokenizer.json lacks at an \
                 id the tokenizer cannot assign; leaving it and later entries unregistered"
            );
            break;
        }
        planned.push((id, token));
        next_id += 1;
    }

    let mut added = 0usize;
    // Runs of the same `special` flag keep the declared order, so ids are
    // handed out exactly as planned.
    for batch in planned.chunk_by(|left, right| left.1.special == right.1.special) {
        let tokens: Vec<AddedToken> = batch.iter().map(|(_, token)| token.clone()).collect();
        if batch[0].1.special {
            tokenizer.add_special_tokens(&tokens);
        } else {
            tokenizer.add_tokens(&tokens);
        }
        for (id, token) in batch {
            let assigned = tokenizer.token_to_id(&token.content);
            if assigned != Some(*id) {
                // Unreachable while `next_added_token_id` mirrors the crate; kept
                // so a future `tokenizers` change surfaces instead of silently
                // shifting ids.
                tracing::warn!(
                    model = %model_path.display(),
                    token = %token.content,
                    declared_id = id,
                    assigned_id = ?assigned,
                    "added token landed at an unexpected id"
                );
                return added;
            }
            added += 1;
        }
    }
    added
}

#[cfg(test)]
#[path = "added_tokens_tests.rs"]
mod tests;
