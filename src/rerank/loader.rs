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

//! `load_reranker`: the kind dispatcher every reranker family registers with.
//!
//! Detection happens on `config.json` alone ([`super::detect_reranker_kind`]),
//! so a checkpoint directory is enough; the shared batch-size and length
//! overrides are applied here and each family owns its own tokenizer and
//! weights.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::embeddings::loader::read_embedding_config;

use super::qwen3_generative::Qwen3Reranker;
use super::qwen3_vl_generative::Qwen3VlReranker;
use super::sequence_classifier::SequenceClassifierReranker;
use super::{Reranker, RerankerKind, detect_reranker_kind};

/// Operator overrides applied while loading a reranker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RerankLoadOptions {
    /// `--rerank-batch-size`: pairs per forward pass. `None` takes the kind's
    /// default (8 for text, 2 for the multimodal reranker).
    pub batch_size: Option<usize>,
    /// `--rerank-max-length`: lowers the derived token cap on one pair.
    pub max_length: Option<usize>,
}

/// Everything the worker and the CLI need to serve one reranker checkpoint.
pub struct LoadedReranker {
    /// The family scoring path.
    pub reranker: Box<dyn Reranker>,
    /// Which scoring recipe was selected.
    pub kind: RerankerKind,
    /// `config.json` `model_type`, for logs and `/v1/rerank` diagnostics.
    pub model_type: String,
}

/// Read a checkpoint's `config.json` the way [`load_reranker_with_options`]
/// does, for tests that only exercise detection.
#[cfg(test)]
pub(crate) fn read_reranker_config_for_tests(model_dir: &Path) -> Value {
    read_embedding_config(model_dir).expect("config.json reads")
}

/// Resolve a path that may point at a file inside the checkpoint directory.
fn resolve_model_dir(model_path: &Path) -> PathBuf {
    if model_path.is_file() {
        model_path.parent().unwrap_or(model_path).to_path_buf()
    } else {
        model_path.to_path_buf()
    }
}

/// Load a reranker checkpoint with default options.
pub fn load_reranker(model_path: &Path) -> Result<LoadedReranker> {
    load_reranker_with_options(model_path, RerankLoadOptions::default())
}

/// Load a reranker checkpoint: detect the kind, then build its scoring path.
pub fn load_reranker_with_options(
    model_path: &Path,
    options: RerankLoadOptions,
) -> Result<LoadedReranker> {
    let model_dir = resolve_model_dir(model_path);
    let config = read_embedding_config(&model_dir)?;
    let kind = detect_reranker_kind(&config)
        .with_context(|| format!("{} is not a reranker checkpoint", model_dir.display()))?;
    let batch_size = options
        .batch_size
        .filter(|&n| n > 0)
        .unwrap_or_else(|| kind.default_batch_size());

    let reranker: Box<dyn Reranker> = match kind {
        RerankerKind::SequenceClassifier => Box::new(SequenceClassifierReranker::load(
            &model_dir,
            batch_size,
            options.max_length,
        )?),
        RerankerKind::GenerativeText => Box::new(Qwen3Reranker::load(
            &model_dir,
            batch_size,
            options.max_length,
        )?),
        RerankerKind::GenerativeVl => Box::new(Qwen3VlReranker::load(
            &model_dir,
            batch_size,
            options.max_length,
        )?),
    };

    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    tracing::info!(
        target: "mlxcel::rerank",
        path = %model_dir.display(),
        kind = kind.as_str(),
        model_type = %model_type,
        max_length = reranker.max_length(),
        batch_size = reranker.batch_size(),
        supports_images = reranker.supports_images(),
        "reranker loaded"
    );
    Ok(LoadedReranker {
        reranker,
        kind,
        model_type,
    })
}
