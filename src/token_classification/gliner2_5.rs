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

//! GLiNER2.5 (`model_type: gliner2` / `gliner2_5`, Fastino): a DeBERTa-v3
//! encoder with a boundary head that extracts spans for labels chosen per
//! request and classifies text against request-time label sets, without
//! generation. Port of `mlx_vlm/models/gliner2_5` (`BoundaryExtractor` and
//! `Model`); the encoder is [`crate::models::deberta_v2`], the head lives in
//! [`super::gliner2_5_head`] and the request preparation in
//! [`super::gliner2_5_prepare`].
//!
//! Detection: `model_type` `gliner2` or `gliner2_5`, or `architectures`
//! naming `Gliner2ForSchemaExtraction` or `BoundaryExtractor`, with
//! `architecture: boundary` (absent only with `BoundaryExtractor`). The live
//! `fastino/gliner2.5-small-v1` config declares `gliner2` and
//! `Gliner2ForSchemaExtraction`, which upstream's loader does not map; it
//! keys on `BoundaryExtractor` and reads a sidecar `encoder_config/config.json`
//! when the block is not inline, which [`resolve_encoder_config`] mirrors.
//!
//! Relation and record extraction are out of scope (their tensors are
//! dropped at load, as upstream does). Only `boundary` heads with a shared
//! candidate pool and no candidate / query attention layers load.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::embeddings::loader::{
    QuantizationParams, load_embedding_weights, quantization_params, read_embedding_config,
};
use crate::models::deberta_v2::{DebertaV2Args, DebertaV2Encoder};
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::extract::{ClassifyTask, EntityItem, EntityResult, ExtractOptions, SpanExtractorModel};
use super::gliner2_5_head::BoundaryHead;
use super::gliner2_5_layers::{BoundaryHeadSettings, linear};
use super::gliner2_5_prepare::{
    PreparedInput, SPECIAL_TOKENS, ScoredSpan, WordSplitter, char_slice, prepare,
    resolve_flat_overlaps, schema_tokens,
};

/// `model_type` spellings of the family.
pub const MODEL_TYPES: &[&str] = &["gliner2", "gliner2_5"];
/// `architectures` entries of the family.
pub const ARCHITECTURES: &[&str] = &["Gliner2ForSchemaExtraction", "BoundaryExtractor"];
/// Upstream's `ModelConfig.max_len` default.
pub const DEFAULT_MAX_LEN: usize = 4096;
/// Largest `max_len` accepted: every layer holds `[heads, L, L]` scores, so a
/// config past this would ask for tens of gigabytes per sequence.
pub const MAX_SUPPORTED_MAX_LEN: usize = 32_768;
/// A label whose abstention probability exceeds this returns no span.
const ABSTENTION_THRESHOLD: f32 = 0.5;

/// Tensor prefixes of the head variants this port does not run.
const UNSUPPORTED_PREFIXES: &[&str] = &[
    "boundary_head.boundary_proposer.",
    "boundary_head.pair_scorer.",
    "boundary_head.candidate_encoder.",
    "record_decoder.",
    "relation_scorer.",
];

fn architectures(config: &Value) -> Vec<&str> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .map(|archs| archs.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

/// Whether `config` is a GLiNER2 checkpoint at all, before the head check.
pub fn is_gliner2_config(config: &Value) -> bool {
    let by_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .is_some_and(|t| MODEL_TYPES.contains(&t.to_ascii_lowercase().as_str()));
    by_type
        || architectures(config)
            .iter()
            .any(|a| ARCHITECTURES.contains(a))
}

/// Check the head architecture of a GLiNER2 config: `boundary`, or absent
/// on a `BoundaryExtractor` export.
pub fn check_boundary_architecture(config: &Value) -> Result<()> {
    match config.get("architecture").and_then(Value::as_str) {
        Some("boundary") => Ok(()),
        None if architectures(config).contains(&"BoundaryExtractor") => Ok(()),
        other => bail!(
            "GLiNER2 architecture '{}' is not supported; only the boundary extractor is",
            other.unwrap_or("<missing>")
        ),
    }
}

/// The encoder config: inline `encoder_config`, else
/// `<model_dir>/encoder_config/config.json`.
pub fn resolve_encoder_config(model_dir: &Path, config: &Value) -> Result<Value> {
    if let Some(inline) = config.get("encoder_config").filter(|v| v.is_object()) {
        return Ok(inline.clone());
    }
    let path = model_dir.join("encoder_config").join("config.json");
    if !path.is_file() {
        bail!("GLiNER2.5 encoder config not found: {}", path.display());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

/// Split the checkpoint into the encoder map (Hugging Face `DebertaV2Model`
/// keys, the `encoder.` prefix stripped) and the head map, dropping the
/// tensors of the unsupported head variants.
///
/// An export written by mlx-vlm (`mlx_vlm.convert`, quantized or not) carries
/// the names its `sanitize` produced (`encoder.layers.`, `.self_attn.`,
/// `.layer_norm.`); those are mapped back to the Hugging Face names the
/// encoder reads. The head keeps its own `layer_norm` names, so the reverse
/// renames apply to encoder tensors only.
pub fn sanitize(weights: WeightMap) -> (WeightMap, WeightMap) {
    let mut encoder = WeightMap::new();
    let mut head = WeightMap::new();
    for (key, value) in weights {
        if UNSUPPORTED_PREFIXES.iter().any(|p| key.starts_with(p)) {
            continue;
        }
        match key.strip_prefix("encoder.") {
            Some(rest) => {
                let hf = rest
                    .replace("encoder.layers.", "encoder.layer.")
                    .replace(".attention.self_attn.", ".attention.self.")
                    .replace(".layer_norm.", ".LayerNorm.");
                encoder.insert(hf, value)
            }
            None => head.insert(key, value),
        };
    }
    (encoder, head)
}

/// The range check behind the config `max_len` and the CLI `--max-len`
/// override: one error text for both.
pub fn check_max_len(max_len: usize) -> Result<()> {
    if max_len == 0 || max_len > MAX_SUPPORTED_MAX_LEN {
        bail!("GLiNER2.5 max_len must be an integer in 1..={MAX_SUPPORTED_MAX_LEN}, got {max_len}");
    }
    Ok(())
}

/// GLiNER2.5 as a [`SpanExtractorModel`].
pub struct Gliner2_5Model {
    encoder: DebertaV2Encoder,
    classifier_hidden: UnifiedLinear,
    classifier_output: UnifiedLinear,
    head: BoundaryHead,
    tokenizer: Tokenizer,
    max_len: usize,
    unk_id: u32,
    word_splitter: WordSplitter,
}

impl Gliner2_5Model {
    /// Load the checkpoint in `model_dir` whose `config.json` is `config`.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        check_boundary_architecture(config)?;
        let settings =
            BoundaryHeadSettings::from_config(config.get("boundary_head").unwrap_or(&Value::Null))?;
        let max_len = match config.get("max_len") {
            None | Some(Value::Null) => DEFAULT_MAX_LEN,
            Some(value) => value
                .as_u64()
                .filter(|&v| v > 0 && v <= MAX_SUPPORTED_MAX_LEN as u64)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GLiNER2.5 max_len must be an integer in 1..={MAX_SUPPORTED_MAX_LEN}, got {value}"
                    )
                })?,
        };
        let encoder_args = DebertaV2Args::from_config(&resolve_encoder_config(model_dir, config)?)?;
        let quant = quantization_params(config).unwrap_or(QuantizationParams {
            group_size: 64,
            bits: 4,
        });
        let (encoder_weights, head_weights) = sanitize(load_embedding_weights(model_dir, config)?);
        let q = (quant.group_size, quant.bits);
        let hidden_size = encoder_args.hidden_size;
        let encoder = DebertaV2Encoder::from_weights(&encoder_weights, encoder_args, q.0, q.1)?;
        let head = BoundaryHead::from_weights(&head_weights, &settings, hidden_size, q)?;
        let classifier_hidden = linear(&head_weights, "classifier.0", q.0, q.1)?;
        let classifier_output = linear(&head_weights, "classifier.3", q.0, q.1)?;

        let tokenizer = load_tokenizer(model_dir)
            .with_context(|| format!("failed to load the tokenizer of {}", model_dir.display()))?;
        let MlxcelTokenizer::HuggingFace(mut tokenizer) = tokenizer else {
            bail!("GLiNER2.5 requires a tokenizer.json tokenizer");
        };
        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);
        // Each special token must be one added token, as upstream registers
        // them: a vocabulary that merely spells `[E]` in pieces would pass a
        // presence check and then shift every marker position.
        for token in SPECIAL_TOKENS {
            let whole = tokenizer.token_to_id(token).is_some_and(|id| {
                tokenizer
                    .encode(*token, false)
                    .is_ok_and(|encoding| encoding.get_ids() == [id])
            });
            if !whole {
                bail!("checkpoint tokenizer is missing GLiNER2.5 special tokens");
            }
        }
        let Some(unk_id) = tokenizer.token_to_id("[UNK]") else {
            bail!("checkpoint tokenizer defines no [UNK] token");
        };
        Ok(Self {
            encoder,
            classifier_hidden,
            classifier_output,
            head,
            tokenizer,
            max_len,
            unk_id,
            word_splitter: WordSplitter::default(),
        })
    }

    /// Load from `model_dir`, reading its `config.json`.
    pub fn load_dir(model_dir: &Path) -> Result<Self> {
        let config = read_embedding_config(model_dir)?;
        Self::load(model_dir, &config)
    }

    /// How the text is split into words (`whitespace` unless changed).
    pub fn word_splitter(&self) -> WordSplitter {
        self.word_splitter
    }

    pub fn set_word_splitter(&mut self, splitter: WordSplitter) {
        self.word_splitter = splitter;
    }

    /// Override `max_len` after load (the CLI `--max-len`); the same range
    /// and error text as the config value.
    pub fn set_max_len(&mut self, max_len: usize) -> Result<()> {
        check_max_len(max_len)?;
        self.max_len = max_len;
        Ok(())
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Build the sequence for `schema` over `text`.
    pub fn prepare(&self, text: &str, schema: &[String]) -> Result<PreparedInput> {
        prepare(
            &self.tokenizer,
            text,
            schema,
            self.max_len,
            self.word_splitter,
            self.unk_id,
        )
    }

    /// `[L, D]` encoder states of one prepared sequence.
    fn encode(&self, prepared: &PreparedInput) -> Result<UniquePtr<MlxArray>> {
        let vocab = self.encoder.vocab_rows();
        if let Some(&id) = prepared.input_ids.iter().find(|&&id| id as usize >= vocab) {
            bail!("token id {id} is outside the {vocab}-row word-embedding table");
        }
        let ids: Vec<i32> = prepared.input_ids.iter().map(|&id| id as i32).collect();
        let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);
        let encoded = self.encoder.encode(&input_ids, None)?;
        Ok(mlxcel_core::squeeze_axis(&encoded, 0))
    }

    /// Rows of `states` at `positions`.
    fn rows(states: &MlxArray, positions: &[usize]) -> UniquePtr<MlxArray> {
        let index: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        mlxcel_core::take(
            states,
            &mlxcel_core::from_slice_i32(&index, &[index.len() as i32]),
            0,
        )
    }

    fn read(array: &MlxArray) -> Result<Vec<f32>> {
        mlxcel_core::try_eval(array)
            .map_err(|e| anyhow::anyhow!("GLiNER2.5 evaluation failed: {e}"))?;
        Ok(array_to_vec_f32(array))
    }

    /// Sigmoid probabilities of the `[Q]` classifier logits of `choices`.
    fn classify(&self, choices: &MlxArray) -> Result<Vec<f32>> {
        let hidden = mlxcel_core::relu(&self.classifier_hidden.forward(choices));
        let logits = mlxcel_core::squeeze_axis(&self.classifier_output.forward(&hidden), -1);
        Self::read(&mlxcel_core::sigmoid(&mlxcel_core::astype(
            &logits,
            dtype::FLOAT32,
        )))
    }
}

impl SpanExtractorModel for Gliner2_5Model {
    fn extract_entities(
        &mut self,
        text: &str,
        labels: &[(String, Option<String>)],
        threshold: f32,
        options: ExtractOptions,
    ) -> Result<EntityResult> {
        if labels.is_empty() {
            return Ok(EntityResult::default());
        }
        let names: Vec<String> = labels.iter().map(|(label, _)| label.clone()).collect();
        let description = |label: &str| {
            labels
                .iter()
                .find(|(name, _)| name == label)
                .and_then(|(_, description)| description.as_deref())
        };
        let schema = schema_tokens("entities", &names, "[E]", None, description);
        let prepared = self.prepare(text, &schema)?;
        let encoded = self.encode(&prepared)?;
        let text_states = Self::rows(&encoded, &prepared.word_positions);
        let query_states = Self::rows(&encoded, &prepared.marker_positions[1..]);
        let output = self.head.forward(&text_states, &query_states)?;
        let probabilities = Self::read(&mlxcel_core::sigmoid(&mlxcel_core::astype(
            &output.logits,
            dtype::FLOAT32,
        )))?;
        let abstention = Self::read(&mlxcel_core::sigmoid(&mlxcel_core::astype(
            &output.null_logits,
            dtype::FLOAT32,
        )))?;
        let word_count = prepared.offsets.len();
        let queries = names.len();

        let mut entities = Vec::with_capacity(queries);
        for (query, label) in names.iter().enumerate() {
            let mut candidates: Vec<ScoredSpan> = Vec::new();
            if abstention[query] <= ABSTENTION_THRESHOLD {
                for (candidate, &(start, end)) in output
                    .pooled
                    .pairs
                    .iter()
                    .enumerate()
                    .take(output.pooled.count)
                {
                    let score = probabilities[candidate * queries + query];
                    if score < threshold || start >= end || end > word_count {
                        continue;
                    }
                    candidates.push((score, start, end));
                }
            }
            let items = resolve_flat_overlaps(&candidates)
                .into_iter()
                .map(|(score, start, end)| {
                    let char_start = prepared.offsets[start].0;
                    let char_end = prepared.offsets[end - 1].1;
                    EntityItem::new(
                        char_slice(&prepared.text, char_start, char_end),
                        score,
                        char_start,
                        char_end,
                        options,
                    )
                })
                .collect();
            entities.push((label.clone(), items));
        }
        Ok(EntityResult { entities })
    }

    fn classify_text(
        &mut self,
        text: &str,
        tasks: &[(String, ClassifyTask)],
        threshold: f32,
    ) -> Result<Value> {
        let mut results = Map::new();
        for (task, spec) in tasks {
            let labels = spec.labels().to_vec();
            if labels.is_empty() {
                bail!("task '{task}' has no labels");
            }
            let schema = schema_tokens(task, &labels, "[L]", spec.prompt(), |label| {
                spec.label_description(label)
            });
            let prepared = self.prepare(text, &schema)?;
            let encoded = self.encode(&prepared)?;
            let choices = Self::rows(&encoded, &prepared.marker_positions[1..]);
            let scores = self.classify(&choices)?;
            let scores = &scores[..labels.len().min(scores.len())];
            let value = if spec.multi_label() {
                let task_threshold = spec.threshold(threshold);
                Value::Array(
                    labels
                        .iter()
                        .zip(scores)
                        .filter(|(_, score)| **score >= task_threshold)
                        .map(|(label, _)| Value::String(label.clone()))
                        .collect(),
                )
            } else {
                let best = scores
                    .iter()
                    .enumerate()
                    .fold(
                        0usize,
                        |best, (index, &score)| if score > scores[best] { index } else { best },
                    );
                Value::String(labels[best].clone())
            };
            results.insert(task.clone(), value);
        }
        Ok(Value::Object(results))
    }
}

#[cfg(test)]
#[path = "gliner2_5_tests.rs"]
mod gliner2_5_tests;

#[cfg(test)]
#[path = "gliner2_5_checkpoint_tests.rs"]
mod gliner2_5_checkpoint_tests;
