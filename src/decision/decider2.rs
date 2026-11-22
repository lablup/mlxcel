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

//! Decider-2b (`model_type: decider2`, `nativ-community/decider-2b`).
//!
//! A Qwen3.5-2B finetune that answers `choice`, `score` and `bool` questions
//! by reading label-token logits at an open answer slot, with per-type
//! temperature calibration and no generation. Port of upstream's
//! `mlx_vlm/models/decider2/decider2.py` in its default independent mode:
//! every question (and every level of an isolated score) is its own prompt,
//! prompts are batched while `rows * longest <= 8192`, and each row's
//! post-norm hidden at its last real token is multiplied by the label
//! tokens' embedding rows (the tied head restricted to the labels).
//!
//! Upstream's packed mode (`independent=False`) and the per-call `isolated`
//! override are Python keyword arguments only and are not ported.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use serde_json::Value;

use super::decider2_prompt::{
    Decider2Prompter, LABEL_COUNT, RenderedQuestion, render_question, render_state,
};
use super::decider2_rows::{
    Row, Temperatures, answer, calibrated_softmax, group_rows, input_tokens, padded_length,
    plan_rows,
};
use super::{
    DecisionError, DecisionMedia, DecisionModel, DecisionResult, DecisionUsage, MediaKind,
    Question, QuestionKind,
};
use crate::models::qwen3_5::Qwen35Model;

/// `config.json` `model_type` the loader detects.
pub(crate) const DECIDER2_MODEL_TYPE: &str = "decider2";

const DECIDER2_TYPES: &[QuestionKind] = &[
    QuestionKind::Choice,
    QuestionKind::Score,
    QuestionKind::Bool,
];

fn default_max_state_tokens() -> usize {
    32768
}

/// Most prompt tokens one request may score, summed over its rows.
///
/// `max_state_tokens` bounds the state in each row, but not how many rows a
/// request asks for: every question (and every isolated score level) repeats
/// the state, so a 2 MB body of short questions over a 32k-token state would
/// otherwise queue billions of prefill tokens on the one decision worker.
/// 2^20 still scores 32 questions over a full 32768-token state.
pub(crate) const MAX_REQUEST_TOKENS: usize = 1 << 20;
const _: () = assert!(MAX_REQUEST_TOKENS >= 32 * 32768);

/// Row length cap when `config.json` has no `max_position_embeddings`.
const DEFAULT_MAX_ROW_TOKENS: usize = 262_144;

/// Encode every row's prompt, refusing a request whose rows exceed
/// `max_row_tokens` each or `max_request_tokens` together.
///
/// The state is encoded once and shared. Encoding stops at the first row
/// past either limit, so a refused request never holds more than the budget
/// in prompt ids. Both refusals are [`DecisionError::InvalidInput`] (a 400):
/// the instruction and option texts are not truncated, so only the request
/// can make a row too long.
pub(crate) fn build_prompts(
    prompter: &Decider2Prompter,
    context: &str,
    rows: &[Row],
    max_row_tokens: usize,
    max_request_tokens: usize,
) -> Result<Vec<Vec<u32>>, DecisionError> {
    let context_ids = prompter.context_ids(context)?;
    let mut prompts = Vec::with_capacity(rows.len().min(max_request_tokens));
    let mut total = 0usize;
    for row in rows {
        let ids = prompter.prompt(&context_ids, &row.instruction, &row.options)?;
        if ids.len() > max_row_tokens {
            return Err(DecisionError::InvalidInput(format!(
                "A question's prompt is {} tokens; this model reads at most {max_row_tokens}",
                ids.len()
            )));
        }
        total = total.saturating_add(ids.len());
        if total > max_request_tokens {
            return Err(DecisionError::InvalidInput(format!(
                "The questions need more than {max_request_tokens} prompt tokens in total; \
                 send fewer questions or a shorter state"
            )));
        }
        prompts.push(ids);
    }
    Ok(prompts)
}

/// `config.json` `decision_config`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct DecisionSettings {
    /// Temperature for a rendered type without its own entry.
    pub(crate) temperature: f32,
    /// Per rendered type (`choice`, `score`, `noul`).
    #[serde(default)]
    pub(crate) temperature_by_type: HashMap<String, f32>,
    /// Score each level of a `score` question as its own yes/no row.
    #[serde(default)]
    pub(crate) isolated_levels: bool,
    /// State tokens kept from the start of the context.
    #[serde(default = "default_max_state_tokens")]
    pub(crate) max_state_tokens: usize,
}

impl DecisionSettings {
    /// Read `decision_config` from a parsed `config.json`.
    pub(crate) fn from_config(config: &Value) -> Result<Self> {
        let section = config
            .get("decision_config")
            .ok_or_else(|| anyhow!("decider2 config.json has no decision_config"))?;
        let settings: Self =
            serde_json::from_value(section.clone()).context("parsing decider2 decision_config")?;
        let positive = |value: f32| value.is_finite() && value > 0.0;
        if !positive(settings.temperature)
            || !settings.temperature_by_type.values().all(|t| positive(*t))
        {
            bail!("decider2 decision_config temperatures must be finite and positive");
        }
        Ok(settings)
    }

    fn temperatures(&self) -> Temperatures {
        Temperatures {
            default: self.temperature,
            by_type: self.temperature_by_type.clone(),
        }
    }
}

/// The loaded Decider-2b scorer.
pub(crate) struct Decider2Model {
    prompter: Decider2Prompter,
    backbone: Qwen35Model,
    /// Embedding rows of the label tokens, transposed to `[hidden, 255]`.
    label_weights_t: UniquePtr<MlxArray>,
    settings: DecisionSettings,
    pad_token_id: i32,
    /// Longest prompt one row may be: the backbone's
    /// `max_position_embeddings`.
    max_row_tokens: usize,
}

// SAFETY: the decision worker and `mlxcel decide` both build the scorer and
// call `predict` on one thread (the worker runs its loader closure on its own
// thread), so the MLX handles inside never move between threads while in
// use. `DecisionModel` requires `Send` only so the boxed scorer can be
// returned from that loader.
unsafe impl Send for Decider2Model {}

/// `tokenizer_config.json` `pad_token`, else `config.json` `pad_token_id`,
/// else `eos_token_id`. Right padding sits after every real token, so the
/// value never reaches a scored position.
fn pad_token_id(model_dir: &Path, config: &Value, prompter: &Decider2Prompter) -> i32 {
    let from_tokenizer = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|tokenizer| {
            tokenizer
                .get("pad_token")
                .and_then(Value::as_str)
                .and_then(|token| prompter.token_id(token))
        });
    from_tokenizer
        .map(i64::from)
        .or_else(|| config.get("pad_token_id").and_then(Value::as_i64))
        .or_else(|| config.get("eos_token_id").and_then(Value::as_i64))
        .and_then(|id| i32::try_from(id).ok())
        .unwrap_or(0)
}

impl Decider2Model {
    /// Load the checkpoint at `model_dir`.
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config: Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;
        let settings = DecisionSettings::from_config(&config)?;
        let prompter = Decider2Prompter::load(model_dir, settings.max_state_tokens)?;
        let pad_token_id = pad_token_id(model_dir, &config, &prompter);
        let max_row_tokens = config
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_MAX_ROW_TOKENS);
        let (backbone, _) =
            Qwen35Model::load(model_dir).map_err(|error| anyhow!("decider2 backbone: {error}"))?;
        let ids: Vec<i32> = prompter
            .label_ids
            .iter()
            .map(|&id| i32::try_from(id).context("label token id out of range"))
            .collect::<Result<_>>()?;
        let ids = mlxcel_core::from_slice_i32(&ids, &[LABEL_COUNT as i32]);
        let rows = backbone.embed_tokens.forward(&ids);
        let label_weights_t = mlxcel_core::transpose_axes(&rows, &[1, 0]);
        mlxcel_core::try_eval(&label_weights_t)
            .map_err(|error| anyhow!("decider2 label weights: {error}"))?;
        Ok(Self {
            prompter,
            backbone,
            label_weights_t,
            settings,
            pad_token_id,
            max_row_tokens,
        })
    }

    /// Calibrated probabilities for every row, in row order.
    pub(crate) fn score_rows(
        &self,
        prompts: &[Vec<u32>],
        rows: &[Row],
    ) -> Result<Vec<Vec<f64>>, DecisionError> {
        let lengths: Vec<usize> = prompts.iter().map(Vec::len).collect();
        let mut probabilities = Vec::with_capacity(rows.len());
        for group in group_rows(&lengths) {
            let members = &prompts[group.clone()];
            let length = padded_length(lengths[group.clone()].iter().copied().max().unwrap_or(1));
            let logits = self.group_logits(members, length)?;
            for (offset, row) in rows[group].iter().enumerate() {
                let start = offset * LABEL_COUNT;
                let row_logits = &logits[start..start + row.options.len()];
                probabilities.push(calibrated_softmax(row_logits, row.temperature));
            }
        }
        Ok(probabilities)
    }

    /// `[rows, 255]` label logits of one right-padded group, row-major.
    fn group_logits(&self, prompts: &[Vec<u32>], length: usize) -> Result<Vec<f32>, DecisionError> {
        let internal = |what: &str| DecisionError::Internal(format!("decider2 {what}"));
        let rows = prompts.len();
        let mut ids = vec![self.pad_token_id; rows * length];
        let mut last = Vec::with_capacity(rows);
        for (row, prompt) in prompts.iter().enumerate() {
            if prompt.is_empty() || prompt.len() > length {
                return Err(internal("prompt does not fit its group"));
            }
            for (slot, &id) in ids[row * length..].iter_mut().zip(prompt) {
                *slot = i32::try_from(id).map_err(|_| internal("token id out of range"))?;
            }
            let position = row * length + prompt.len() - 1;
            last.push(i32::try_from(position).map_err(|_| internal("group too large"))?);
        }
        let (rows_i32, length_i32) = (
            i32::try_from(rows).map_err(|_| internal("group too large"))?,
            i32::try_from(length).map_err(|_| internal("group too large"))?,
        );
        let input = mlxcel_core::from_slice_i32(&ids, &[rows_i32, length_i32]);
        let hidden = self.backbone.forward_prefill_hidden(&input);
        let width = *mlxcel_core::array_shape(&hidden)
            .last()
            .ok_or_else(|| internal("hidden has no shape"))?;
        let flat = mlxcel_core::reshape(&hidden, &[rows_i32 * length_i32, width]);
        let positions = mlxcel_core::from_slice_i32(&last, &[rows_i32]);
        let slots = mlxcel_core::take(&flat, &positions, 0);
        // bf16 @ bf16 like upstream; the f32 cast comes after the product.
        let logits = mlxcel_core::matmul(&slots, &self.label_weights_t);
        mlxcel_core::try_eval(&logits)
            .map_err(|error| DecisionError::Internal(format!("decider2 forward: {error}")))?;
        Ok(mlxcel_core::utils::array_to_vec_f32(&logits))
    }
}

impl DecisionModel for Decider2Model {
    fn decision_types(&self) -> &[QuestionKind] {
        DECIDER2_TYPES
    }

    fn decision_media(&self) -> &[MediaKind] {
        &[]
    }

    fn predict(
        &mut self,
        state: Option<&Value>,
        questions: &[(String, Question)],
        _media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionError> {
        let context = render_state(state);
        let rendered: Vec<RenderedQuestion> = questions
            .iter()
            .map(|(_, question)| render_question(question))
            .collect::<Result<_, _>>()?;
        let (rows, slots) = plan_rows(
            &rendered,
            &self.settings.temperatures(),
            self.settings.isolated_levels,
        );
        let prompts = build_prompts(
            &self.prompter,
            &context,
            &rows,
            self.max_row_tokens,
            MAX_REQUEST_TOKENS,
        )?;
        let probabilities = self.score_rows(&prompts, &rows)?;
        let answers = questions
            .iter()
            .zip(&rendered)
            .zip(slots)
            .map(|(((name, question), rendered), slot)| {
                (
                    name.clone(),
                    answer(question, rendered, slot, &probabilities),
                )
            })
            .collect();
        Ok(DecisionResult {
            answers,
            usage: DecisionUsage {
                input_tokens: input_tokens(&prompts),
            },
        })
    }
}
