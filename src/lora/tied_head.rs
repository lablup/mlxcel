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

//! The one verdict on a LoRA pair that addresses either end of a tied
//! input/output projection (issues #1577 and #2332).
//!
//! The single-process loaders validate an adapter against the raw safetensors
//! map, where a tied checkpoint has no `lm_head.weight`, so an `lm_head` pair
//! failed with "no base weight". The pipeline stage loader validates against
//! `models::load_text_weights`, whose `sanitize_tied_embeddings` has already
//! synthesized `lm_head.weight` from the embeddings, so the same pair fused,
//! and then the Llama stage built its output projection from
//! `model.embed_tokens` and never read the fused copy. Same adapter, same
//! checkpoint, a refusal on one path and a silent no-op on the other.
//!
//! The answer chosen is to refuse on every path, and to take the verdict from
//! the checkpoint itself (its `config.json` and its stored tensor names)
//! rather than from either path's weight map, so both reach it identically and
//! before any weight is read. Applying the pair instead would mean writing a
//! delta into a copy of the embedding matrix, which silently unties the input
//! and output projections the checkpoint was trained with. A tied mlx-lm model
//! has no `lm_head` module for `mlx_lm.lora` to wrap, so no adapter it trains
//! carries such a pair, which is what makes refusing cheap.
//!
//! An `embed_tokens` pair on the same checkpoint is the other end of the tie
//! and gets the same verdict (#2332). mlx-lm's `LoRAEmbedding` stores it as
//! `lora_a = [vocab, rank]`, `lora_b = [rank, hidden]` and adds the delta to
//! the embedding lookup and to the tied head alike (its `as_linear` and its
//! `fuse` both use the adapted matrix). The load paths here read every pair in
//! the `Linear` orientation instead, so they disagreed with mlx-lm and with
//! each other: fusing folded a `Linear`-shaped pair into the shared matrix (both
//! ends), the runtime path handed it to whichever constructor claimed the
//! `model.embed_tokens` prefix (the tied head on the Llama family, nothing on
//! Qwen3, whose head reads the embedding through `as_linear`), and an mlx-lm
//! embedding pair failed a shape check on every path with a message about
//! transposed weights. Reproducing mlx-lm would need an embedding-orientation
//! pair, a runtime lookup term in `UnifiedEmbedding`, and a transposed runtime
//! head term on every family that builds its head from the embedding, so the
//! pair is refused here, before any path reads a weight, with `mlx_lm.fuse`
//! named as the way to serve such an adapter.
//!
//! Issue #2421 added that support for the Llama decoder (`llama`, `qwen2`)
//! and Qwen3: the loaders read an embedding pair in the `LoRAEmbedding` layout
//! (`super::embedding`), the runtime path stages it for
//! `UnifiedEmbedding::from_weights` and attaches its transposed form to a head
//! built from the embedding, and the fused and stage paths fold it into the one
//! matrix both ends read. For those families, and only when their constructors
//! read the head from the embedding (see `applies_tied_embedding_pairs`), the
//! `model.embed_tokens` pair is no longer refused. Every other family, every
//! other embedding stem and every output-head pair keeps the verdict and its
//! wording.
//!
//! Used by: `loading::load_model_with_adapter`,
//! `loading::load_model_with_adapter_specs`, the pipeline Llama stage loader

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;

/// Output-projection weight stems an adapter pair can address, for a text
/// checkpoint and for a VLM that nests its language model.
const OUTPUT_HEADS: [&str; 2] = ["lm_head", "language_model.lm_head"];

/// Input-embedding stems an adapter pair can address, each with the output
/// head whose stored weight decides whether the two are tied: a text
/// checkpoint, a VLM that nests the language model's `model`, and the
/// transformers 4.52+ VLM layout that nests it inside `model` and keeps the
/// head at the top level.
const INPUT_EMBEDDINGS: [(&str, &str); 3] = [
    ("model.embed_tokens", "lm_head"),
    (
        "language_model.model.embed_tokens",
        "language_model.lm_head",
    ),
    ("model.language_model.embed_tokens", "lm_head"),
];

/// One end of a tied projection an adapter pair addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TiedTarget {
    /// An output head, by its stem.
    Head(&'static str),
    /// An input embedding, by its stem, with the head it is tied to.
    Embedding(&'static str, &'static str),
}

impl TiedTarget {
    /// The adapter stem this target was matched from.
    fn stem(self) -> &'static str {
        match self {
            Self::Head(stem) | Self::Embedding(stem, _) => stem,
        }
    }

    /// The output head whose stored weight decides the tie.
    fn head(self) -> &'static str {
        match self {
            Self::Head(head) | Self::Embedding(_, head) => head,
        }
    }
}

/// Largest safetensors header this probe reads. An adapter header is a few
/// kilobytes; anything past this is a malformed file the real loader reports.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Refuse an adapter that carries a LoRA pair for the output head or the
/// input embedding of a checkpoint whose output head is tied to its input
/// embeddings.
///
/// A checkpoint counts as tied for a pair when its `config.json` sets
/// `tie_word_embeddings: true` (top level or `text_config`), or when it stores
/// no `<head>.weight` tensor of its own for the head that pair's end belongs
/// to, which is the case the loaders fill by copying the embeddings. Only the
/// adapter's safetensors header and the checkpoint's headers are read, never
/// tensor data.
///
/// An adapter whose file cannot be found or parsed is passed through: the
/// adapter loader that runs next reports that with its own wording, and this
/// probe should not become a second, differently worded source of it.
///
/// # Errors
///
/// Returns an error naming every targeted head and embedding when the
/// checkpoint is tied, or when the checkpoint's `config.json` or headers
/// cannot be read.
pub fn reject_tied_output_head_adapter(model_dir: &Path, adapter_path: &Path) -> Result<()> {
    let Some(targets) = targeted_tied_ends(adapter_path) else {
        return Ok(());
    };
    if targets.is_empty() {
        return Ok(());
    }

    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|err| anyhow::anyhow!("failed to read {}: {err}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&crate::models::sanitize_config_json(&config_str))
            .map_err(|err| anyhow::anyhow!("failed to parse {}: {err}", config_path.display()))?;

    let (refused, reason) = if declares_tied_embeddings(&config) {
        (
            targets.into_iter().collect::<Vec<_>>(),
            "its config.json sets tie_word_embeddings".to_string(),
        )
    } else {
        let heads: BTreeSet<&str> = targets.iter().map(|target| target.head()).collect();
        let mut missing = BTreeSet::new();
        for head in heads {
            let stored = format!("{head}.weight");
            let present =
                mlxcel_core::weights::dir_has_tensor_name(model_dir, |name| name == stored)
                    .map_err(anyhow::Error::msg)?;
            if !present {
                missing.insert(head);
            }
        }
        let refused: Vec<TiedTarget> = targets
            .into_iter()
            .filter(|target| missing.contains(target.head()))
            .collect();
        if refused.is_empty() {
            return Ok(());
        }
        let stored: Vec<String> = missing
            .iter()
            .map(|head| format!("{head}.weight"))
            .collect();
        (
            refused,
            format!(
                "it stores no {} of its own and reuses the embedding matrix",
                stored.join(" / ")
            ),
        )
    };

    // The families that apply a `model.embed_tokens` pair at both ends of the
    // tie, the way mlx-lm does, take it (#2421); every other refusal stands
    // exactly as worded below.
    let refused: Vec<TiedTarget> = if refused.contains(&APPLIED_TIED_EMBEDDING)
        && applies_tied_embedding_pairs(model_dir, &config)
    {
        refused
            .into_iter()
            .filter(|target| *target != APPLIED_TIED_EMBEDDING)
            .collect()
    } else {
        refused
    };
    if refused.is_empty() {
        return Ok(());
    }

    let stems: Vec<&str> = refused.iter().map(|target| target.stem()).collect();
    let heads: Vec<&str> = refused
        .iter()
        .filter(|target| matches!(target, TiedTarget::Head(_)))
        .map(|target| target.stem())
        .collect();
    let embeddings: Vec<&str> = refused
        .iter()
        .filter(|target| matches!(target, TiedTarget::Embedding(..)))
        .map(|target| target.stem())
        .collect();

    let mut message = format!(
        "Adapter at {} carries a LoRA pair for {}, but this checkpoint ties its output projection \
         to its input embeddings ({reason}).",
        adapter_path.display(),
        stems.join(", "),
    );
    if !heads.is_empty() {
        message.push_str(&format!(
            " An output-head LoRA on a tied checkpoint is refused on every load path: the model \
             reads its logits through the embedding matrix, so the pair would either never reach \
             the forward pass or silently untie the two projections. Remove the {} pair from the \
             adapter (mlx-lm's LoRA training never targets it).",
            heads.join(" / "),
        ));
    }
    if !embeddings.is_empty() {
        message.push_str(&format!(
            " An input-embedding LoRA on a tied checkpoint is refused on every load path: mlx-lm \
             applies such a pair to the embedding lookup and to the tied output projection alike, \
             and the load paths here read every pair as a linear projection, so they cannot apply \
             it to both ends the same way. If mlx_lm.lora trained the adapter, merge it into the \
             checkpoint with mlx_lm.fuse and load the merged model; otherwise remove the {} pair \
             from the adapter.",
            embeddings.join(" / "),
        ));
    }
    anyhow::bail!("{message}");
}

/// The embedding end a supported family applies on a tied checkpoint.
const APPLIED_TIED_EMBEDDING: TiedTarget = TiedTarget::Embedding("model.embed_tokens", "lm_head");

/// Whether this checkpoint's family applies a `model.embed_tokens` pair to the
/// lookup and to the tied head alike, on every load path (#2421).
///
/// Only the Llama decoder (`llama`, its `mistral` alias, and `qwen2`, which
/// loads through the same constructors) and Qwen3 qualify, and only when the
/// constructors actually read the head from the embedding: Llama builds its
/// head from `model.embed_tokens` only when `config.json` sets
/// `tie_word_embeddings: true` (its default is `false`), and Qwen3 reads it
/// through `embed_tokens.as_linear` unless the config sets it `false`. A Llama
/// checkpoint that is tied only because it stores no `lm_head.weight` builds
/// its head from the copy `sanitize_tied_embeddings` makes, which the fused
/// path takes after fusing and the runtime and stage paths before, so the
/// paths would disagree and it keeps the refusal.
fn applies_tied_embedding_pairs(model_dir: &Path, config: &serde_json::Value) -> bool {
    use crate::models::ModelType;

    let Ok(model_type) = crate::models::get_model_type(model_dir) else {
        return false;
    };
    let tie = config
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool);
    match model_type {
        ModelType::Llama | ModelType::Qwen2 => tie == Some(true),
        ModelType::Qwen3 => tie != Some(false),
        _ => false,
    }
}

/// Whether the config declares the output projection tied, at the top level or
/// in a VLM's `text_config`. Only an explicit `true` counts: an absent key is
/// decided by whether the checkpoint stores the head, because family defaults
/// for the flag differ.
fn declares_tied_embeddings(config: &serde_json::Value) -> bool {
    let explicit = |value: &serde_json::Value| {
        value
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
    };
    explicit(config)
        .or_else(|| config.get("text_config").and_then(explicit))
        .unwrap_or(false)
}

/// The ends of a tied projection the adapter's tensors address, or `None` when
/// its weights file cannot be located or its header cannot be read.
fn targeted_tied_ends(adapter_path: &Path) -> Option<BTreeSet<TiedTarget>> {
    let weights_path =
        crate::distributed::pipeline::resolve_adapter_weights_path(adapter_path).ok()?;
    let names = safetensors_tensor_names(&weights_path)?;
    Some(names.iter().filter_map(|name| tied_end_of(name)).collect())
}

/// The layer stem of an adapter tensor name.
///
/// Accepts the mlx-lm `.lora_a` / `.lora_b` spelling, a PEFT `.base_layer`
/// wrapper, and a stem that names the `.weight` itself, the same stems the
/// base-weight resolver accepts.
fn adapter_stem(tensor_name: &str) -> Option<&str> {
    let stem = tensor_name
        .strip_suffix(".lora_a")
        .or_else(|| tensor_name.strip_suffix(".lora_b"))?;
    let stem = stem.strip_suffix(".base_layer").unwrap_or(stem);
    // The resolver also accepts a stem that already ends in `.weight`
    // (`lm_head.weight.lora_a` resolves to `lm_head.weight`), so the verdict
    // has to read it the same way or such an adapter would slip past it.
    Some(stem.strip_suffix(".weight").unwrap_or(stem))
}

/// The output head an adapter tensor name addresses, if any.
fn output_head_of(tensor_name: &str) -> Option<&'static str> {
    let stem = adapter_stem(tensor_name)?;
    OUTPUT_HEADS.into_iter().find(|head| *head == stem)
}

/// The end of a tied projection an adapter tensor name addresses, if any.
fn tied_end_of(tensor_name: &str) -> Option<TiedTarget> {
    if let Some(head) = output_head_of(tensor_name) {
        return Some(TiedTarget::Head(head));
    }
    let stem = adapter_stem(tensor_name)?;
    INPUT_EMBEDDINGS
        .into_iter()
        .find(|(embedding, _)| *embedding == stem)
        .map(|(embedding, head)| TiedTarget::Embedding(embedding, head))
}

/// Tensor names from a safetensors header, without reading tensor data.
fn safetensors_tensor_names(path: &Path) -> Option<Vec<String>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).ok()?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return None;
    }
    let mut header = vec![0u8; usize::try_from(header_len).ok()?];
    file.read_exact(&mut header).ok()?;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&header).ok()?;
    Some(
        header
            .into_iter()
            .map(|(name, _)| name)
            .filter(|name| name != "__metadata__")
            .collect(),
    )
}

#[cfg(test)]
#[path = "tied_head_tests.rs"]
mod tests;
