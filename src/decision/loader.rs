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

//! `load_decision_model`: the kind dispatcher every decision family
//! registers with.
//!
//! Detection reads `config.json` alone, so a checkpoint directory is enough.
//! Callers resolve HuggingFace repo ids to a local directory first, the way
//! `mlxcel rerank` and `--reranker-model` do.
//!
//! # Registering a family
//!
//! A model port adds exactly three things here:
//!
//! 1. a [`DecisionKind`] variant (and its [`DecisionKind::as_str`] arm),
//! 2. a detection arm in [`detect_decision_kind`],
//! 3. a constructor arm in `construct`, returning its
//!    [`DecisionModel`](super::DecisionModel) scorer.
//!
//! Families: LiquidAI d1 (`lfm2_vl` with a `D1Model` auto_map, #2367),
//! Decider-2b (`decider2`, #2359) and Clef (`clef`, #2365), plus the
//! `#[cfg(test)]` stub.
//!
//! A checkpoint no arm recognizes fails with "<path> is not a decision
//! checkpoint".

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::embeddings::loader::read_embedding_config;

use super::{DecisionModel, MediaKind, QuestionKind};

/// Which decision family a checkpoint belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecisionKind {
    /// LiquidAI d1 (`lfm2_vl` with an `auto_map.AutoModel` ending in
    /// `D1Model`), scored by `src/decision/d1.rs`.
    D1,
    /// Decider-2b (`model_type: decider2`), `src/decision/decider2.rs`.
    Decider2,
    /// Clef and Clef-Flash (`model_type: clef`), `src/decision/clef.rs`.
    Clef,
    /// The deterministic test model in `src/decision/stub.rs`.
    #[cfg(test)]
    Stub,
}

impl DecisionKind {
    /// Short name for logs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            DecisionKind::D1 => "d1",
            DecisionKind::Decider2 => "decider2",
            DecisionKind::Clef => "clef",
            #[cfg(test)]
            DecisionKind::Stub => "stub",
        }
    }
}

/// Everything the worker and the CLI need to serve one decision checkpoint.
pub struct LoadedDecisionModel {
    /// The family scorer.
    pub model: Box<dyn DecisionModel>,
    /// Which family was selected.
    pub kind: DecisionKind,
    /// `config.json` `model_type`, for logs.
    pub model_type: String,
}

impl std::fmt::Debug for LoadedDecisionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedDecisionModel")
            .field("kind", &self.kind)
            .field("model_type", &self.model_type)
            .finish_non_exhaustive()
    }
}

/// Detect the decision family from a checkpoint's `config.json`.
pub fn detect_decision_kind(config: &Value) -> Result<DecisionKind> {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("");
    if super::d1::is_d1_config(config) {
        return Ok(DecisionKind::D1);
    }
    if model_type == super::decider2::DECIDER2_MODEL_TYPE {
        return Ok(DecisionKind::Decider2);
    }
    if model_type == super::clef::CLEF_MODEL_TYPE {
        return Ok(DecisionKind::Clef);
    }
    #[cfg(test)]
    if model_type == super::stub::STUB_MODEL_TYPE {
        return Ok(DecisionKind::Stub);
    }
    bail!("model_type `{model_type}` has no decision scorer in this build")
}

/// Resolve a path that may point at a file inside the checkpoint directory.
fn resolve_model_dir(model_path: &Path) -> PathBuf {
    if model_path.is_file() {
        model_path.parent().unwrap_or(model_path).to_path_buf()
    } else {
        model_path.to_path_buf()
    }
}

/// Load a decision checkpoint: detect the family, then build its scorer.
pub fn load_decision_model(model_path: &Path) -> Result<LoadedDecisionModel> {
    let model_dir = resolve_model_dir(model_path);
    let not_decision = || format!("{} is not a decision checkpoint", model_dir.display());
    let config = read_embedding_config(&model_dir).with_context(not_decision)?;
    let kind = detect_decision_kind(&config).with_context(not_decision)?;
    let model = construct(kind, &model_dir)?;
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    tracing::info!(
        target: "mlxcel::decision",
        path = %model_dir.display(),
        kind = kind.as_str(),
        model_type = %model_type,
        types = ?model.decision_types().iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        media = ?model.decision_media().iter().map(|m| m.field()).collect::<Vec<_>>(),
        "decision model loaded"
    );
    Ok(LoadedDecisionModel {
        model,
        kind,
        model_type,
    })
}

/// Build the scorer for `kind` from the checkpoint at `model_dir`.
fn construct(kind: DecisionKind, model_dir: &Path) -> Result<Box<dyn DecisionModel>> {
    match kind {
        DecisionKind::D1 => Ok(Box::new(super::d1::D1DecisionModel::load(model_dir)?)),
        DecisionKind::Decider2 => Ok(Box::new(super::decider2::Decider2Model::load(model_dir)?)),
        DecisionKind::Clef => Ok(Box::new(super::clef::ClefModel::load(model_dir)?)),
        #[cfg(test)]
        DecisionKind::Stub => Ok(Box::new(super::stub::StubDecisionModel::load(model_dir))),
    }
}

/// The declarations of a loaded model, copied out for the worker info.
pub(crate) fn declarations(model: &dyn DecisionModel) -> (Vec<QuestionKind>, Vec<MediaKind>) {
    (
        model.decision_types().to_vec(),
        model.decision_media().to_vec(),
    )
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
