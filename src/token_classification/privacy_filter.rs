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

//! [`TokenClassifierModel`] over OpenAI Privacy Filter
//! ([`crate::models::openai_privacy_filter`]), matching mlx-vlm's
//! `PrivacyFilter` pipeline: raw token windows of `default_n_ctx` ids with no
//! special tokens, no word grouping, and a constrained BIOES Viterbi decode
//! by default with per-token argmax on request.
//!
//! The Viterbi transition biases come from the checkpoint's
//! `viterbi_calibration.json` (`operating_points.<name>.biases`). A missing
//! file means all zeros; an unknown operating point or a missing key fails at
//! load, before any weight is read.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::Value;

use crate::models::openai_privacy_filter::PrivacyFilterModel;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::loader::id2label_labels;
use super::viterbi::{TransitionBiases, ViterbiDecoder};
use super::{DecodeMode, TokenClassifierModel, argmax_path};

/// Viterbi by default, argmax on `--argmax`.
const PRIVACY_FILTER_DECODE_MODES: &[DecodeMode] = &[DecodeMode::Viterbi, DecodeMode::Argmax];

/// The operating point used when none is requested.
pub const DEFAULT_OPERATING_POINT: &str = "default";

/// Sidecar file carrying the calibrated transition biases.
pub const CALIBRATION_FILE: &str = "viterbi_calibration.json";

/// Read `operating_points.<operating_point>.biases` from the checkpoint's
/// `viterbi_calibration.json`. A checkpoint without the file decodes with
/// zero biases, as upstream does.
pub fn load_transition_biases(model_dir: &Path, operating_point: &str) -> Result<TransitionBiases> {
    let path = model_dir.join(CALIBRATION_FILE);
    if !path.is_file() {
        return Ok(TransitionBiases::default());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let payload: Value = serde_json::from_str(&raw)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    let Some(biases) = payload
        .get("operating_points")
        .and_then(|points| points.get(operating_point))
        .and_then(|point| point.get("biases"))
    else {
        let known: Vec<String> = payload
            .get("operating_points")
            .and_then(Value::as_object)
            .map(|points| points.keys().cloned().collect())
            .unwrap_or_default();
        bail!(
            "Unknown Privacy Filter operating point '{operating_point}' ({} defines {})",
            path.display(),
            if known.is_empty() {
                "no operating points".to_string()
            } else {
                known
                    .iter()
                    .map(|name| format!("'{name}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
    };
    TransitionBiases::from_json(biases)
        .with_context(|| format!("operating point '{operating_point}' of {}", path.display()))
}

/// The Privacy Filter encoder as a pipeline model.
pub struct PrivacyFilterTagger {
    model: PrivacyFilterModel,
    labels: Vec<String>,
    decoder: ViterbiDecoder,
    operating_point: String,
}

impl PrivacyFilterTagger {
    /// Load the encoder, its tag set, the calibration for `operating_point`
    /// and the tokenizer. The label set and the calibration are validated
    /// before the weights are read.
    pub fn load(
        model_dir: &Path,
        config: &Value,
        operating_point: &str,
    ) -> Result<(Self, MlxcelTokenizer)> {
        let labels = id2label_labels(config)?;
        let biases = load_transition_biases(model_dir, operating_point)?;
        let decoder = ViterbiDecoder::new(&labels, biases)?;
        let model = PrivacyFilterModel::load(model_dir).map_err(|e| {
            anyhow::anyhow!(
                "failed to load openai_privacy_filter {}: {e}",
                model_dir.display()
            )
        })?;
        if labels.len() != model.num_labels() {
            bail!(
                "id2label has {} entries but the classifier has {} outputs",
                labels.len(),
                model.num_labels()
            );
        }
        let tokenizer = load_tokenizer(model_dir)
            .with_context(|| format!("failed to load the tokenizer of {}", model_dir.display()))?;
        Ok((
            Self {
                model,
                labels,
                decoder,
                operating_point: operating_point.to_string(),
            },
            tokenizer,
        ))
    }

    /// Build from an already loaded encoder (tests and callers with their
    /// own weights).
    pub fn from_parts(
        model: PrivacyFilterModel,
        labels: Vec<String>,
        biases: TransitionBiases,
    ) -> Result<Self> {
        if labels.len() != model.num_labels() {
            bail!(
                "id2label has {} entries but the classifier has {} outputs",
                labels.len(),
                model.num_labels()
            );
        }
        let decoder = ViterbiDecoder::new(&labels, biases)?;
        Ok(Self {
            model,
            labels,
            decoder,
            operating_point: DEFAULT_OPERATING_POINT.to_string(),
        })
    }

    pub fn model(&self) -> &PrivacyFilterModel {
        &self.model
    }

    pub fn decoder(&self) -> &ViterbiDecoder {
        &self.decoder
    }

    /// The calibration operating point in use.
    pub fn operating_point(&self) -> &str {
        &self.operating_point
    }

    fn window_logits(&self, window_ids: &[u32]) -> Result<UniquePtr<MlxArray>> {
        let length = i32::try_from(window_ids.len()).context("token window is too long")?;
        let ids: Vec<i32> = window_ids
            .iter()
            .map(|&id| i32::try_from(id).context("token id does not fit in int32"))
            .collect::<Result<_>>()?;
        let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, length]);
        let logits = self.model.logits(&input_ids);
        Ok(mlxcel_core::astype(&logits, dtype::FLOAT32))
    }
}

impl TokenClassifierModel for PrivacyFilterTagger {
    fn labels(&self) -> &[String] {
        &self.labels
    }

    fn context_size(&self) -> usize {
        self.model.args().default_n_ctx
    }

    fn special_tokens(&self) -> (Vec<u32>, Vec<u32>) {
        (Vec::new(), Vec::new())
    }

    fn group_words(&self) -> bool {
        false
    }

    fn decode_modes(&self) -> &[DecodeMode] {
        PRIVACY_FILTER_DECODE_MODES
    }

    /// Raw f32 label logits for one window; the pipeline takes the
    /// log-softmax.
    fn log_probs(&mut self, window_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        if window_ids.is_empty() {
            return Ok(Vec::new());
        }
        let vocab = self.model.vocab_rows();
        if let Some(&id) = window_ids.iter().find(|&&id| id as usize >= vocab) {
            bail!("token id {id} is outside the {vocab}-row token-embedding table");
        }
        let logits = self.window_logits(window_ids)?;
        mlxcel_core::try_eval(&logits)
            .map_err(|e| anyhow::anyhow!("token classifier evaluation failed: {e}"))?;
        let flat = array_to_vec_f32(&logits);
        let width = self.labels.len();
        if flat.len() != window_ids.len() * width {
            bail!(
                "token classifier produced {} scores for {} tokens x {width} labels",
                flat.len(),
                window_ids.len()
            );
        }
        Ok(flat.chunks(width).map(<[f32]>::to_vec).collect())
    }

    fn decode_path(&self, scores: &[Vec<f32>], mode: DecodeMode) -> Vec<usize> {
        match mode {
            DecodeMode::Viterbi => self.decoder.decode(scores),
            DecodeMode::Argmax => argmax_path(scores),
        }
    }
}

#[cfg(test)]
#[path = "privacy_filter_tests.rs"]
mod privacy_filter_tests;

#[cfg(test)]
#[path = "privacy_filter_checkpoint_tests.rs"]
mod privacy_filter_checkpoint_tests;
