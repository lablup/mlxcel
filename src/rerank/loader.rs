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
use crate::embeddings::{
    DEFAULT_EMBEDDING_BATCH_SIZE, EmbedOptions, EmbeddingEngine, EmbeddingLoadOptions,
    LoadedEmbeddingModel, load_embedding_model_with_options,
};
use crate::model_metadata::is_embedding_model_type;
use crate::models::get_model_type;

use super::late_interaction::LateInteractionReranker;
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
        // `detect_reranker_kind` never returns it: late interaction scores
        // with the served embedding model, not a reranker checkpoint.
        RerankerKind::LateInteraction => anyhow::bail!(
            "{} cannot be loaded as a late-interaction reranker; serve a multi-vector \
             embedder with -m instead",
            model_dir.display()
        ),
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

/// Load a reranker checkpoint, or, when it is none, a multi-vector embedder
/// scored by MaxSim (`mlxcel rerank`, #3300).
///
/// This is the CLI counterpart of the server's late-interaction fallback
/// (`server::rerank_maxsim`): `/v1/rerank` scores with a multi-vector `-m`
/// embedder when no reranker is configured, so `mlxcel rerank -m` accepts the
/// same checkpoints. A reranker checkpoint loads exactly as
/// [`load_reranker_with_options`] loads it. Anything that is neither a
/// reranker nor a multi-vector embedder (a chat checkpoint, a single-vector
/// embedder) fails with the reranker detection error, unchanged; a chat
/// checkpoint is refused from `config.json` alone, before any weight is read.
///
/// `options.batch_size` becomes the embedding batch width, defaulting to the
/// server's `--embedding-batch-size` default so both sides batch the texts the
/// same way; `options.max_length` caps every embedded input.
pub fn load_reranker_or_late_interaction(
    model_path: &Path,
    options: RerankLoadOptions,
) -> Result<LoadedReranker> {
    let model_dir = resolve_model_dir(model_path);
    let config = read_embedding_config(&model_dir)?;
    let detection = match detect_reranker_kind(&config) {
        Ok(_) => return load_reranker_with_options(&model_dir, options),
        Err(err) => err.context(format!(
            "{} is not a reranker checkpoint",
            model_dir.display()
        )),
    };
    let is_embedder = get_model_type(&model_dir).is_ok_and(is_embedding_model_type);
    if !is_embedder {
        return Err(detection);
    }
    late_interaction_from_embedder(&model_dir, &config, options, detection, |dir, load| {
        load_embedding_model_with_options(dir, load)
    })
}

/// Load the embedder through `load` and wrap it as a late-interaction
/// reranker, or return `detection` when it is single-vector.
///
/// Split out so the single-vector refusal is testable with the stub model.
pub(crate) fn late_interaction_from_embedder(
    model_dir: &Path,
    config: &Value,
    options: RerankLoadOptions,
    detection: anyhow::Error,
    load: impl FnOnce(&Path, EmbeddingLoadOptions) -> Result<LoadedEmbeddingModel>,
) -> Result<LoadedReranker> {
    let loaded = load(
        model_dir,
        EmbeddingLoadOptions {
            max_length: options.max_length,
        },
    )
    .with_context(|| {
        format!(
            "{} is not a reranker checkpoint, and loading it as a multi-vector embedder \
             failed",
            model_dir.display()
        )
    })?;
    if !loaded.limits.multi_vector {
        // MaxSim over one row is a plain dot product, which the server does
        // not serve on `/v1/rerank` either.
        return Err(detection);
    }
    let batch_size = options
        .batch_size
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_EMBEDDING_BATCH_SIZE);
    let engine = EmbeddingEngine::new(loaded, batch_size);
    let reranker = LateInteractionReranker::new(engine, EmbedOptions::default());
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    tracing::info!(
        target: "mlxcel::rerank",
        path = %model_dir.display(),
        kind = RerankerKind::LateInteraction.as_str(),
        model_type = %model_type,
        max_length = reranker.max_length(),
        batch_size = reranker.batch_size(),
        supports_images = reranker.supports_images(),
        "multi-vector embedder loaded as a late-interaction reranker"
    );
    Ok(LoadedReranker {
        reranker: Box::new(reranker),
        kind: RerankerKind::LateInteraction,
        model_type,
    })
}
