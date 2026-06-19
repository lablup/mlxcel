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

//! Greedy decoding for the Whisper-style decoder, including token suppression
//! (`suppress_blank`, non-speech tokens, the timestamp range) and first-step
//! language detection.
//!
//! Beam search, word-level timestamps, and temperature fallback are out of
//! scope for this first port.

use anyhow::{Result, anyhow};

use mlxcel_core::{MlxArray, UniquePtr};

use super::decoder::TextDecoder;
use super::layers::KvCache;
use super::tokenizer::WhisperTokenizer;

/// Additive `[n_vocab]` mask (`0` keep / `-inf` suppress) applied at every step:
/// the non-speech / task-token suppression set plus the whole timestamp range.
pub(crate) fn build_suppression_mask(tokenizer: &WhisperTokenizer, n_vocab: i32) -> Vec<f32> {
    let n = n_vocab as usize;
    let mut mask = vec![0.0f32; n];
    for &id in &tokenizer.suppress {
        if (0..n_vocab).contains(&id) {
            mask[id as usize] = f32::NEG_INFINITY;
        }
    }
    // Suppress every timestamp token so the no-timestamps prompt is enforced.
    for id in tokenizer.timestamp_begin..n_vocab {
        mask[id as usize] = f32::NEG_INFINITY;
    }
    mask
}

/// Additive `[n_vocab]` mask applied only on the first generated step to forbid
/// a leading blank/space or an immediate end-of-transcript.
pub(crate) fn build_first_step_mask(tokenizer: &WhisperTokenizer, n_vocab: i32) -> Vec<f32> {
    let mut mask = vec![0.0f32; n_vocab as usize];
    if let Some(blank) = tokenizer.blank
        && (0..n_vocab).contains(&blank)
    {
        mask[blank as usize] = f32::NEG_INFINITY;
    }
    if (0..n_vocab).contains(&tokenizer.eot) {
        mask[tokenizer.eot as usize] = f32::NEG_INFINITY;
    }
    mask
}

/// Additive `[n_vocab]` mask that keeps only language tokens (`0`) and masks
/// everything else (`-inf`), used for first-step language detection.
pub(crate) fn build_language_mask(tokenizer: &WhisperTokenizer, n_vocab: i32) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; n_vocab as usize];
    for &(_, id) in &tokenizer.language_ids {
        if (0..n_vocab).contains(&id) {
            mask[id as usize] = 0.0;
        }
    }
    mask
}

fn empty_caches(n: usize) -> Vec<Option<KvCache>> {
    (0..n).map(|_| None).collect()
}

fn mask_array(data: &[f32], n_vocab: i32) -> UniquePtr<MlxArray> {
    // Built in f32 to keep the masked logits numerically stable.
    mlxcel_core::from_slice_f32(data, &[1, n_vocab])
}

/// Take the last-position logits, apply additive masks, and return the argmax
/// token id.
///
/// This is the single point in the decode loop where the lazily-built MLX graph
/// (encoder, decoder, masks, argmax) is forced. The evaluation is routed through
/// the fallible [`mlxcel_core::try_eval`] wrapper so an MLX failure surfaces as
/// an `Err` rather than an uncaught C++ exception that would abort the process.
fn argmax_with_masks(
    logits: &MlxArray,
    pos: i32,
    n_vocab: i32,
    always_mask: &MlxArray,
    first_mask: Option<&MlxArray>,
) -> Result<i32> {
    let sliced = mlxcel_core::slice(logits, &[0, pos, 0], &[1, pos + 1, n_vocab]);
    let sliced = mlxcel_core::reshape(&sliced, &[1, n_vocab]);
    let mut l = mlxcel_core::astype(&sliced, mlxcel_core::dtype::FLOAT32);
    l = mlxcel_core::add(&l, always_mask);
    if let Some(first) = first_mask {
        l = mlxcel_core::add(&l, first);
    }
    let idx = mlxcel_core::argmax(&l, -1, false);
    mlxcel_core::try_eval(&idx).map_err(|e| anyhow!("Whisper logits evaluation failed: {e}"))?;
    Ok(mlxcel_core::item_i32(&idx))
}

/// Detect the spoken language from a single `<|startoftranscript|>` decode step.
/// Returns `None` for English-only checkpoints (caller treats that as English).
pub(crate) fn detect_language<'a>(
    decoder: &TextDecoder,
    audio_features: &MlxArray,
    tokenizer: &'a WhisperTokenizer,
    n_vocab: i32,
) -> Result<Option<&'a str>> {
    if !tokenizer.multilingual {
        return Ok(None);
    }
    let mut self_caches = empty_caches(decoder.num_layers());
    let mut cross_caches = empty_caches(decoder.num_layers());
    let sot = mlxcel_core::from_slice_i32(&[tokenizer.sot], &[1, 1]);
    let logits = decoder.forward(&sot, audio_features, 0, &mut self_caches, &mut cross_caches);

    let lang_mask = mask_array(&build_language_mask(tokenizer, n_vocab), n_vocab);
    let picked = argmax_with_masks(&logits, 0, n_vocab, &lang_mask, None)?;
    Ok(tokenizer
        .language_ids
        .iter()
        .find(|(_, id)| *id == picked)
        .map(|(code, _)| *code))
}

/// Greedily decode one 30 s segment of `audio_features` into token ids and the
/// language code that was used (hint, detected, or `en`).
pub(crate) fn transcribe_segment(
    decoder: &TextDecoder,
    audio_features: &MlxArray,
    tokenizer: &WhisperTokenizer,
    n_vocab: i32,
    n_text_ctx: i32,
    language: Option<&str>,
    translate: bool,
) -> Result<(Vec<i32>, Option<String>)> {
    // Resolve the language once (hint takes priority over detection).
    let resolved: Option<String> = match language {
        Some(code) => Some(code.to_string()),
        None => detect_language(decoder, audio_features, tokenizer, n_vocab)?.map(str::to_string),
    };

    let initial = tokenizer.initial_tokens(resolved.as_deref(), translate);
    let always = mask_array(&build_suppression_mask(tokenizer, n_vocab), n_vocab);
    let first = mask_array(&build_first_step_mask(tokenizer, n_vocab), n_vocab);

    let mut self_caches = empty_caches(decoder.num_layers());
    let mut cross_caches = empty_caches(decoder.num_layers());

    // Prefill the start-of-transcript prompt.
    let init_arr = mlxcel_core::from_slice_i32(&initial, &[1, initial.len() as i32]);
    let logits = decoder.forward(
        &init_arr,
        audio_features,
        0,
        &mut self_caches,
        &mut cross_caches,
    );

    let mut generated: Vec<i32> = Vec::new();
    let mut next = argmax_with_masks(
        &logits,
        initial.len() as i32 - 1,
        n_vocab,
        &always,
        Some(&first),
    )?;
    let mut offset = initial.len() as i32;

    // `sample_len` mirrors the reference cap of n_text_ctx / 2.
    let max_new = n_text_ctx / 2;
    let mut step = 0;
    while next != tokenizer.eot && step < max_new && offset < n_text_ctx {
        generated.push(next);
        let tok = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        let logits = decoder.forward(
            &tok,
            audio_features,
            offset,
            &mut self_caches,
            &mut cross_caches,
        );
        offset += 1;
        step += 1;
        next = argmax_with_masks(&logits, 0, n_vocab, &always, None)?;
    }

    Ok((generated, resolved))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    fn synthetic() -> WhisperTokenizer {
        let json = r#"{
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [
                {"id": 4, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 5, "content": "<|startoftranscript|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 6, "content": "<|en|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 7, "content": "<|ko|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 8, "content": "<|transcribe|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 9, "content": "<|translate|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 10, "content": "<|notimestamps|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 11, "content": "<|0.00|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "vocab": {"a": 0, " ": 1, "b": 2}, "merges": []}
        }"#;
        WhisperTokenizer::from_tokenizer(Tokenizer::from_bytes(json.as_bytes()).unwrap()).unwrap()
    }

    #[test]
    fn suppression_mask_blocks_specials_and_timestamps_but_not_eot() {
        let tok = synthetic();
        let n_vocab = 13;
        let mask = build_suppression_mask(&tok, n_vocab);
        assert_eq!(mask.len(), n_vocab as usize);
        // Task / start tokens suppressed.
        assert_eq!(mask[tok.sot as usize], f32::NEG_INFINITY);
        assert_eq!(mask[tok.transcribe as usize], f32::NEG_INFINITY);
        // Timestamp range suppressed (>= timestamp_begin = 11).
        assert_eq!(mask[11], f32::NEG_INFINITY);
        assert_eq!(mask[12], f32::NEG_INFINITY);
        // End-of-transcript must remain selectable to stop decoding.
        assert_eq!(mask[tok.eot as usize], 0.0);
        // Plain text tokens stay open.
        assert_eq!(mask[0], 0.0);
    }

    #[test]
    fn first_step_mask_blocks_blank_and_eot() {
        let tok = synthetic();
        let n_vocab = 13;
        let mask = build_first_step_mask(&tok, n_vocab);
        assert_eq!(mask[tok.eot as usize], f32::NEG_INFINITY);
        if let Some(blank) = tok.blank {
            assert_eq!(mask[blank as usize], f32::NEG_INFINITY);
        }
        // A normal text token is left untouched.
        assert_eq!(mask[0], 0.0);
    }

    #[test]
    fn language_mask_keeps_only_language_tokens() {
        let tok = synthetic();
        let n_vocab = 13;
        let en = tok.language_token("en").unwrap();
        let ko = tok.language_token("ko").unwrap();
        let mask = build_language_mask(&tok, n_vocab);
        // Language tags are open (0.0); everything else is masked (-inf).
        assert_eq!(mask[en as usize], 0.0);
        assert_eq!(mask[ko as usize], 0.0);
        assert_eq!(mask[0], f32::NEG_INFINITY); // "a", not a language tag
        assert_eq!(mask[tok.sot as usize], f32::NEG_INFINITY);
    }
}
