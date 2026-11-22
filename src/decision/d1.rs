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

//! LiquidAI d1-3B decisions: an LFM2.5-VL-3B finetune read at the answer
//! slot, ported from mlx-vlm's `mlx_vlm/models/d1/d1.py`.
//!
//! The checkpoint keeps `model_type: lfm2_vl`, so generation loads it as an
//! ordinary LFM2-VL chat model; only the decision loader routes it here, by
//! its `auto_map.AutoModel` ending in `D1Model`. A decision is one forward
//! pass that stops at the answer slot, no tokens are generated:
//!
//! - One question runs as a single pass over the prefix plus its suffix.
//! - Several questions share the prefix (state and images) as a trunk read
//!   once into an [`Lfm2LayerCache`]; each question continues it as its own
//!   row. Rows run in batches of similar length, longest first, each against
//!   the trunk cache tiled to the batch ([`Lfm2LayerCache::tile`]).
//! - A trunk longer than [`PREFILL_STEP`] tokens (or than the token budget)
//!   is read in chunks of that many, which bounds the memory a long state
//!   takes.
//!
//! The prompt and the readout (option codes, yes/no forms, level digits) are
//! in [`super::d1_prompt`].

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Map, Value, json};

use crate::models::lfm2::{Lfm2LayerCache, Lfm2Model};
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};
use crate::vision::Lfm2VlModel;

use super::d1_prompt::{D1Question, as_question, cap_pixels, prompt_ids, readout, readout_ids};
use super::{
    Answer, DecisionError, DecisionMedia, DecisionModel, DecisionResult, DecisionUsage, MediaKind,
    Question, QuestionKind,
};

/// Trunk tokens read per prefill chunk.
pub(crate) const PREFILL_STEP: usize = 2048;
/// Padded tokens per batch of question rows, trunk included (upstream's
/// `token_budget` default; no request override, as upstream exposes it only
/// as a Python argument).
pub(crate) const TOKEN_BUDGET: usize = 65_536;

const D1_TYPES: &[QuestionKind] = &[
    QuestionKind::Choice,
    QuestionKind::Score,
    QuestionKind::Bool,
];
const D1_MEDIA: &[MediaKind] = &[MediaKind::Image];

/// Whether a checkpoint's `config.json` is a d1 decision checkpoint: an
/// LFM2-VL `model_type` whose `auto_map.AutoModel` ends in `D1Model`.
pub(crate) fn is_d1_config(config: &Value) -> bool {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let auto_model = config
        .get("auto_map")
        .and_then(|map| map.get("AutoModel"))
        .and_then(Value::as_str)
        .unwrap_or("");
    matches!(model_type.as_str(), "lfm2_vl" | "lfm2-vl") && auto_model.ends_with("D1Model")
}

fn internal(err: impl std::fmt::Display) -> DecisionError {
    DecisionError::Internal(err.to_string())
}

/// The d1 scorer.
pub(crate) struct D1DecisionModel {
    vlm: Lfm2VlModel,
    tokenizer: MlxcelTokenizer,
    /// `tokenizer_config.json` `bos_token`, written at the head of the prompt.
    bos: String,
    /// `text_config.max_position_embeddings`: the most prompt tokens (trunk
    /// plus every question) a request may run, so a long state or a flood
    /// of questions is a 400 rather than an unbounded KV cache and forward.
    max_prompt_tokens: usize,
}

// SAFETY: `DecisionModel` is `Send` so the boxed scorer can be handed to the
// thread that runs it, but it is never shared: the server's decision worker
// loads it on its own thread and only that thread calls it, and
// `mlxcel decide` loads and calls it on the main thread. The MLX handles and
// the LFM2 backbone's single-threaded sequence state it owns are therefore
// only ever touched by one thread.
unsafe impl Send for D1DecisionModel {}

fn read_bos_token(model_dir: &Path) -> String {
    let path = model_dir.join("tokenizer_config.json");
    let Ok(raw) = std::fs::read_to_string(path) else {
        return String::new();
    };
    let Ok(config) = serde_json::from_str::<Value>(&raw) else {
        return String::new();
    };
    match config.get("bos_token") {
        Some(Value::String(token)) => token.clone(),
        Some(Value::Object(token)) => token
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

/// Prompt cap when `config.json` names no context length: the 32k
/// `text_config.max_position_embeddings` of the published d1-3B checkpoints.
pub(crate) const DEFAULT_MAX_PROMPT_TOKENS: usize = 32_768;

fn read_max_prompt_tokens(model_dir: &Path) -> usize {
    std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .as_ref()
        .and_then(max_prompt_tokens_from_config)
        .unwrap_or(DEFAULT_MAX_PROMPT_TOKENS)
}

/// `text_config.max_position_embeddings` (or a top-level one), when positive.
pub(crate) fn max_prompt_tokens_from_config(config: &Value) -> Option<usize> {
    config
        .get("text_config")
        .and_then(|text| text.get("max_position_embeddings"))
        .or_else(|| config.get("max_position_embeddings"))
        .and_then(Value::as_u64)
        .filter(|&n| n > 0)
        .and_then(|n| usize::try_from(n).ok())
}

/// Refuse a request whose prompt (the trunk once plus every question, the
/// `usage.input_tokens` it would report) exceeds the model's context, before
/// any forward runs. That bounds both the longest row, so the KV cache and
/// one forward stay within what the model was trained for, and the total
/// work, which the number of questions would otherwise grow without limit.
pub(crate) fn check_prompt_length(
    trunk_len: usize,
    branches: &[Vec<i32>],
    max_tokens: usize,
) -> Result<(), DecisionError> {
    let total = branches
        .iter()
        .fold(trunk_len, |sum, branch| sum.saturating_add(branch.len()));
    if total > max_tokens {
        return Err(DecisionError::InvalidInput(format!(
            "the decision prompt (state and all questions) is {total} tokens, more than the model's {max_tokens}-token context"
        )));
    }
    Ok(())
}

impl D1DecisionModel {
    /// Load a d1 checkpoint through the LFM2-VL loader.
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
        let loaded = crate::loading::load_lfm2_vl(model_dir)
            .with_context(|| format!("failed to load {} as LFM2-VL", model_dir.display()))?;
        let crate::LoadedModel::Lfm2VL(vlm) = loaded else {
            bail!("{} did not load as an LFM2-VL stack", model_dir.display());
        };
        let tokenizer = crate::embeddings::tokenize::strip_padding_and_truncation(
            load_tokenizer(model_dir).with_context(|| {
                format!("failed to load the tokenizer in {}", model_dir.display())
            })?,
        );
        let bos = read_bos_token(model_dir);
        let max_prompt_tokens = read_max_prompt_tokens(model_dir);
        Ok(Self {
            vlm,
            tokenizer,
            bos,
            max_prompt_tokens,
        })
    }

    /// Trunk ids (images expanded), branch ids, and the trunk embedding.
    /// One question folds its suffix into the trunk and has no branches.
    fn decision_inputs(
        &self,
        state: Option<&Value>,
        questions: &[D1Question],
        images: &[image::DynamicImage],
    ) -> Result<(Vec<i32>, Vec<Vec<i32>>, UniquePtr<MlxArray>), DecisionError> {
        let (mut trunk, branches) =
            prompt_ids(&self.tokenizer, &self.bos, state, questions, images.len())?;
        if images.is_empty() {
            check_prompt_length(trunk.len(), &branches, self.max_prompt_tokens)?;
            let ids = ids_array(&[trunk.clone()]);
            let embeds = self.vlm.text_model.input_embeddings(&ids);
            return Ok((trunk, branches, embeds));
        }
        // The prompt writes one `<image>` per image; one more written by the
        // caller in the state or a single question's instructions would not
        // match the images sent, which is the caller's error, not the model's.
        let markers = trunk
            .iter()
            .filter(|&&id| id == self.vlm.image_token_id)
            .count();
        if markers != images.len() {
            return Err(DecisionError::InvalidInput(format!(
                "the prompt holds {markers} `<image>` markers for {} images; the state and \
                 instructions must not contain `<image>`",
                images.len()
            )));
        }
        let capped: Vec<image::DynamicImage> = images.iter().map(cap_pixels).collect();
        let (pixel_values, layouts) = self.vlm.processor.preprocess_with_grid(&capped);
        crate::multimodal::lfm2_vl_prompt::insert_lfm2_vl_image_tokens(
            &mut trunk,
            &layouts,
            self.vlm.downsample_factor,
            self.vlm.image_token_id,
            self.vlm.image_start_id,
            self.vlm.image_end_id,
            &self.vlm.img_row_col_ids,
            self.vlm.img_thumbnail_id,
            self.vlm.use_image_special_tokens,
        )
        .map_err(internal)?;
        check_prompt_length(trunk.len(), &branches, self.max_prompt_tokens)?;
        let ids = ids_array(&[trunk.clone()]);
        let embeds = self
            .vlm
            .get_input_embeddings(&ids, &pixel_values, &layouts)
            .inputs_embeds;
        Ok((trunk, branches, embeds))
    }
}

/// `[rows, width]` int32 ids; every row must have the same length.
fn ids_array(rows: &[Vec<i32>]) -> UniquePtr<MlxArray> {
    let width = rows.first().map_or(0, Vec::len);
    let flat: Vec<i32> = rows.iter().flatten().copied().collect();
    mlxcel_core::from_slice_i32(&flat, &[rows.len() as i32, width as i32])
}

fn eval_caches(caches: &[Lfm2LayerCache]) {
    for cache in caches {
        match cache {
            Lfm2LayerCache::Attention(kv) => kv.eval_state(),
            Lfm2LayerCache::Conv(Some(state)) => mlxcel_core::eval(state),
            Lfm2LayerCache::Conv(None) => {}
        }
    }
}

/// Log-probabilities over the vocabulary at each `(row, position)` of
/// `hidden` (`[rows, width, hidden]`), as one `Vec<f32>` per pair. Logits
/// are taken in float32, as upstream's `_logz` does.
fn answer_slot_logprobs(
    text: &Lfm2Model,
    hidden: &MlxArray,
    slots: &[(usize, usize)],
) -> Result<Vec<Vec<f32>>, DecisionError> {
    let shape = mlxcel_core::array_shape(hidden);
    let (rows, width, dim) = (shape[0], shape[1], shape[2]);
    let flat = mlxcel_core::reshape(hidden, &[rows * width, dim]);
    let index: Vec<i32> = slots
        .iter()
        .map(|&(row, pos)| row as i32 * width + pos as i32)
        .collect();
    let index = mlxcel_core::from_slice_i32(&index, &[index.len() as i32]);
    let picked = mlxcel_core::take(&flat, &index, 0);
    let logits = mlxcel_core::astype(&text.project_logits(&picked), mlxcel_core::dtype::FLOAT32);
    let logz = mlxcel_core::subtract(&logits, &mlxcel_core::logsumexp_axis(&logits, -1, true));
    let values = mlxcel_core::utils::array_to_vec_f32(&logz);
    let vocab = *mlxcel_core::array_shape(&logz).last().unwrap_or(&0) as usize;
    if vocab == 0 || values.len() != vocab * slots.len() {
        return Err(DecisionError::Internal(format!(
            "answer-slot logits have {} values for {} rows",
            values.len(),
            slots.len()
        )));
    }
    Ok(values.chunks(vocab).map(<[f32]>::to_vec).collect())
}

/// Log-probabilities at the answer slot: of the trunk alone when there are
/// no branches, else of every branch continuing the trunk, which is read
/// once. `trunk_embeds` is `[1, trunk_len, hidden]`.
///
/// Used by: [`D1DecisionModel::predict`] and the f32 equivalence tests on a
/// synthetic backbone.
pub(crate) fn answer_logprobs(
    text: &Lfm2Model,
    trunk_embeds: &MlxArray,
    branches: &[Vec<i32>],
    token_budget: usize,
) -> Result<Vec<Vec<f32>>, DecisionError> {
    if token_budget == 0 {
        return Err(DecisionError::InvalidInput(
            "token_budget must be at least 1, got 0".to_string(),
        ));
    }
    let trunk_len = mlxcel_core::array_shape(trunk_embeds)[1] as usize;
    if trunk_len == 0 {
        return Err(DecisionError::Internal("empty decision prompt".to_string()));
    }
    let step = PREFILL_STEP.min(token_budget);
    let mut caches = text.make_caches();
    if branches.is_empty() && trunk_len <= step {
        let hidden = text.forward_hidden_with_caches(trunk_embeds, &mut caches);
        return answer_slot_logprobs(text, &hidden, &[(0, trunk_len - 1)]);
    }
    let dim = mlxcel_core::array_shape(trunk_embeds)[2];
    let chunk = |start: usize, end: usize| {
        mlxcel_core::slice(trunk_embeds, &[0, start as i32, 0], &[1, end as i32, dim])
    };
    let mut start = 0;
    while trunk_len - start > step {
        let end = start + step;
        let _ = text.forward_hidden_with_caches(&chunk(start, end), &mut caches);
        eval_caches(&caches);
        start = end;
    }
    let hidden = text.forward_hidden_with_caches(&chunk(start, trunk_len), &mut caches);
    if branches.is_empty() {
        return answer_slot_logprobs(text, &hidden, &[(0, trunk_len - start - 1)]);
    }
    drop(hidden);
    eval_caches(&caches);

    let lengths: Vec<usize> = branches.iter().map(Vec::len).collect();
    if lengths.contains(&0) {
        return Err(DecisionError::Internal("empty question branch".to_string()));
    }
    let mut out: Vec<Option<Vec<f32>>> = vec![None; branches.len()];
    for chunk in super::d1_prompt::branch_chunks(&lengths, trunk_len, token_budget) {
        let width = chunk.iter().map(|&j| lengths[j]).max().unwrap_or(0);
        let rows: Vec<Vec<i32>> = chunk
            .iter()
            .map(|&j| {
                let mut row = branches[j].clone();
                row.resize(width, 0);
                row
            })
            .collect();
        let mut tiled = caches
            .iter()
            .map(|cache| cache.tile(rows.len() as i32))
            .collect::<Result<Vec<_>, _>>()
            .map_err(internal)?;
        // A batch wider than `step` (one long question runs alone however
        // long it is) is read `step` positions at a time against the tiled
        // cache, as the trunk is, so the activations of one forward stay
        // bounded. The rows are causal, so each row's answer slot is read
        // from the piece that holds it.
        let ids = ids_array(&rows);
        let batch = rows.len() as i32;
        let mut start = 0;
        while start < width {
            let end = (start + step).min(width);
            let piece = if start == 0 && end == width {
                text.input_embeddings(&ids)
            } else {
                let piece_ids = mlxcel_core::slice(&ids, &[0, start as i32], &[batch, end as i32]);
                text.input_embeddings(&piece_ids)
            };
            let hidden = text.forward_hidden_with_caches(&piece, &mut tiled);
            let (rows_here, slots): (Vec<usize>, Vec<(usize, usize)>) = chunk
                .iter()
                .enumerate()
                .filter(|&(_, &j)| (start..end).contains(&(lengths[j] - 1)))
                .map(|(row, &j)| (j, (row, lengths[j] - 1 - start)))
                .unzip();
            if !slots.is_empty() {
                for (j, logz) in rows_here
                    .into_iter()
                    .zip(answer_slot_logprobs(text, &hidden, &slots)?)
                {
                    out[j] = Some(logz);
                }
            }
            drop(hidden);
            if end < width {
                eval_caches(&tiled);
            }
            start = end;
        }
    }
    out.into_iter()
        .map(|row| row.ok_or_else(|| internal("a question branch was not scored")))
        .collect()
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold(0, |best, (i, v)| if *v > values[best] { i } else { best })
}

/// The answer for one question from its option probabilities, rounded to
/// four places as upstream's `_answer` does (the expected score level is
/// computed from the unrounded probabilities, then rounded).
pub(crate) fn d1_answer(question: &Question, p: &[f64]) -> Result<Answer, DecisionError> {
    match question.kind {
        QuestionKind::Bool => Ok(Answer::bool(round4(p[0]), Map::new())),
        QuestionKind::Choice => {
            let best = argmax(p);
            let mut metadata = Map::new();
            metadata.insert("confidence".to_string(), json!(round4(p[best])));
            let mut answer =
                Answer::choice(question, p.iter().copied().map(round4).collect(), metadata)?;
            if let Answer::Choice { value, .. } = &mut answer {
                value.clone_from(&question.labels()[best].name);
            }
            Ok(answer)
        }
        QuestionKind::Score => {
            let best = argmax(p);
            let mut metadata = Map::new();
            metadata.insert("confidence".to_string(), json!(round4(p[best])));
            let mut answer = Answer::score(question, p.to_vec(), metadata)?;
            if let Answer::Score {
                value,
                probabilities,
                ..
            } = &mut answer
            {
                *value = round4(*value);
                probabilities.iter_mut().for_each(|v| *v = round4(*v));
            }
            Ok(answer)
        }
        other => Err(DecisionError::Internal(format!(
            "d1 does not answer {}",
            other.as_str()
        ))),
    }
}

impl DecisionModel for D1DecisionModel {
    fn decision_types(&self) -> &[QuestionKind] {
        D1_TYPES
    }

    fn decision_media(&self) -> &[MediaKind] {
        D1_MEDIA
    }

    fn predict(
        &mut self,
        state: Option<&Value>,
        questions: &[(String, Question)],
        media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionError> {
        let normalized = questions
            .iter()
            .map(|(_, q)| as_question(q))
            .collect::<Result<Vec<_>, _>>()?;
        let groups = normalized
            .iter()
            .map(|q| readout_ids(&self.tokenizer, q))
            .collect::<Result<Vec<_>, _>>()?;
        let (trunk, branches, embeds) = self.decision_inputs(state, &normalized, &media.images)?;
        let logz = answer_logprobs(&self.vlm.text_model, &embeds, &branches, TOKEN_BUDGET)?;
        let mut answers = Vec::with_capacity(questions.len());
        for (((name, question), group), z) in questions.iter().zip(&groups).zip(&logz) {
            let p = readout(group, z)?;
            answers.push((name.clone(), d1_answer(question, &p)?));
        }
        Ok(DecisionResult {
            answers,
            usage: DecisionUsage {
                input_tokens: trunk.len() + branches.iter().map(Vec::len).sum::<usize>(),
            },
        })
    }
}

#[cfg(test)]
impl D1DecisionModel {
    /// The trunk and branch ids a request would run, for the prompt gates.
    pub(crate) fn prompt_ids(
        &self,
        state: Option<&Value>,
        questions: &[(String, Question)],
        images: &[image::DynamicImage],
    ) -> Result<(Vec<i32>, Vec<Vec<i32>>), DecisionError> {
        let normalized = questions
            .iter()
            .map(|(_, q)| as_question(q))
            .collect::<Result<Vec<_>, _>>()?;
        let (trunk, branches, _) = self.decision_inputs(state, &normalized, images)?;
        Ok((trunk, branches))
    }
}
