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

//! The two heads that sit on a [`ModernBertEncoder`]: the `/v1/embeddings`
//! embedder and the `ModernBertForSequenceClassification` reranker head.
//!
//! - [`ModernBertEmbeddingModel`] pools the encoder's `[B, L, D]` hidden state
//!   into `[B, D]`. The engine owns L2 normalization and `dimensions`
//!   truncation, so this side only pools. The family default is `mean`, which
//!   `1_Pooling/config.json` confirms for `nomic-ai/modernbert-embed-base`.
//! - [`ModernBertSequenceClassifier`] applies upstream's
//!   `classifier(norm(gelu(dense(pooled))))` and returns `[B, num_labels]`
//!   logits. `/v1/rerank` wiring is #1356; this type is the loadable unit that
//!   issue consumes, and it is reachable directly by directory because
//!   `get_model_type` deliberately refuses to route a `ForSequenceClassification`
//!   checkpoint to an embedding variant.

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::loader::{
    QuantizationParams, load_embedding_weights, quantization_params, read_embedding_config,
};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};

use super::gpt2::{dim_eq, layer_norm_from_weights};
use super::modernbert::{
    DEFAULT_BITS, DEFAULT_GROUP_SIZE, MODERNBERT_SEQUENCE_CLASSIFICATION_ARCH, ModernBertArgs,
    ModernBertEncoder, sanitize_modernbert_weights,
};

/// Row count of `classifier.weight`, which is the real output width of
/// [`ModernBertSequenceClassifier::logits`].
///
/// Quantization packs the input axis, never the output axis, so the row count
/// is `num_labels` on both paths while the column count only equals
/// `hidden_size` on the dense path. A `config.json` whose `num_labels` /
/// `id2label` disagrees with the tensor is logged and overruled rather than
/// silently trusted.
fn classifier_rows(weights: &WeightMap, args: &ModernBertArgs) -> Result<usize, String> {
    let weight = weights
        .get("classifier.weight")
        .ok_or_else(|| "Weight not found: classifier.weight".to_string())?;
    let shape = mlxcel_core::array_shape(weight);
    let [rows, cols] = shape.as_slice() else {
        return Err(format!(
            "classifier.weight must be a 2-D [num_labels, hidden_size] matrix, got shape {shape:?}"
        ));
    };
    let rows = usize::try_from(*rows).unwrap_or(0);
    if rows == 0 {
        return Err("classifier.weight has no rows, so the head has no labels".to_string());
    }
    if !weights.contains_key("classifier.scales") && !dim_eq(*cols, args.hidden_size) {
        return Err(format!(
            "classifier.weight has shape {shape:?}, expected [num_labels, {}]",
            args.hidden_size
        ));
    }
    let declared = args.num_labels();
    if declared != rows {
        tracing::warn!(
            target: "mlxcel::embeddings",
            declared,
            rows,
            "ModernBERT config and classifier.weight disagree on the label count; using the tensor"
        );
    }
    Ok(rows)
}

/// Load, sanitize and parse the pieces both heads need from a checkpoint
/// directory. `keep_head` retains `head.*` / `classifier.*`.
fn load_parts(
    model_dir: &Path,
    config: &Value,
    keep_head: bool,
) -> Result<(ModernBertArgs, WeightMap, Option<QuantizationParams>)> {
    let args = ModernBertArgs::from_config(config).map_err(|e| anyhow!(e))?;
    let weights = load_embedding_weights(model_dir, config)?;
    let weights = sanitize_modernbert_weights(weights, keep_head);
    Ok((args, weights, quantization_params(config)))
}

/// ModernBERT served through `/v1/embeddings` and `mlxcel embed`.
pub struct ModernBertEmbeddingModel {
    encoder: ModernBertEncoder,
    pooling: PoolingMode,
}

impl ModernBertEmbeddingModel {
    /// Load `ModernBertModel` (or `ModernBertForMaskedLM` with its MLM head
    /// dropped) from a checkpoint directory.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let (args, weights, quant) = load_parts(model_dir, config, false)?;
        let encoder =
            ModernBertEncoder::from_weights(&weights, &args, quant).map_err(|e| anyhow!(e))?;
        let pooling = resolve_pooling_mode(model_dir, PoolingMode::Mean)?;
        Ok(Self { encoder, pooling })
    }

    /// Borrow the encoder (the reranker head and the tests reuse it).
    pub fn encoder(&self) -> &ModernBertEncoder {
        &self.encoder
    }
}

impl EmbeddingModel for ModernBertEmbeddingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        if batch.images.is_some_and(|images| !images.is_empty()) {
            bail!("ModernBERT is a text encoder and does not accept image inputs");
        }
        let hidden = self
            .encoder
            .encode(batch.input_ids, batch.attention_mask)
            .map_err(|e| anyhow!(e))?;
        let embeddings = pool(&hidden, batch.attention_mask, self.pooling);
        Ok(EmbeddingOutput {
            embeddings,
            last_hidden_state: Some(hidden),
        })
    }

    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::Mean
    }

    fn pooling(&self) -> PoolingMode {
        self.pooling
    }

    fn embedding_dim(&self) -> usize {
        self.encoder.hidden_size()
    }
}

/// `ModernBertForSequenceClassification`: the encoder plus upstream's
/// prediction head and a `[num_labels, D]` classifier.
pub struct ModernBertSequenceClassifier {
    encoder: ModernBertEncoder,
    head_dense: UnifiedLinear,
    head_norm: LayerNorm,
    classifier: UnifiedLinear,
    pooling: PoolingMode,
    num_labels: usize,
}

impl ModernBertSequenceClassifier {
    /// Load a classifier checkpoint by directory.
    ///
    /// Detection routes `ForSequenceClassification` away from every embedding
    /// variant on purpose (a reranker is not an embedder), so this constructor
    /// reads and validates `config.json` itself rather than going through
    /// `get_model_type`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = read_embedding_config(model_dir)?;
        let (args, weights, quant) = load_parts(model_dir, &config, true)?;
        if !args.is_sequence_classifier() {
            bail!(
                "{} declares architectures {:?}, not {MODERNBERT_SEQUENCE_CLASSIFICATION_ARCH}; \
                 load it as an embedding model instead",
                model_dir.display(),
                args.architectures
            );
        }
        Self::from_weights(&weights, &args, quant).map_err(|e| anyhow!(e))
    }

    /// Build the head from an already-sanitized weight map (`keep_head`).
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModernBertArgs,
        quant: Option<QuantizationParams>,
    ) -> Result<Self, String> {
        let encoder = ModernBertEncoder::from_weights(weights, args, quant)?;
        let (group_size, bits) = quant
            .map(|q| (q.group_size, q.bits))
            .unwrap_or((DEFAULT_GROUP_SIZE, DEFAULT_BITS));
        let head_dense = UnifiedLinear::from_weights(weights, "head.dense", group_size, bits)?;
        let head_norm =
            layer_norm_from_weights(weights, "head.norm", args.hidden_size, args.norm_eps())?;
        let classifier = UnifiedLinear::from_weights(weights, "classifier", group_size, bits)?;

        // `num_labels` decides the advertised output width, so it must come
        // from the tensor that actually produces it rather than from
        // `config.json`. A config that disagrees would leave `num_labels()`
        // lying about the `logits` shape. Row counts are never packed by
        // quantization; the column count is, so it is only checked on the
        // dense path.
        let num_labels = classifier_rows(weights, args)?;
        let pooling = match args.classifier_pooling.as_str() {
            "cls" => PoolingMode::Cls,
            // `validate` already rejects anything else.
            _ => PoolingMode::Mean,
        };
        Ok(Self {
            encoder,
            head_dense,
            head_norm,
            classifier,
            pooling,
            num_labels,
        })
    }

    /// Head width, i.e. the second axis of [`Self::logits`].
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Pooling the head applies before the dense projection.
    pub fn classifier_pooling(&self) -> PoolingMode {
        self.pooling
    }

    /// Borrow the encoder.
    pub fn encoder(&self) -> &ModernBertEncoder {
        &self.encoder
    }

    /// `[B, num_labels]` logits for one right-padded micro-batch.
    ///
    /// `input_ids` and `attention_mask` are `[B, L]` int32. Mean pooling is
    /// mask-weighted, so padding columns never enter the score.
    pub fn logits(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        let hidden = self
            .encoder
            .encode(input_ids, attention_mask)
            .map_err(|e| anyhow!(e))?;
        let pooled = pool(&hidden, attention_mask, self.pooling);
        let head = self
            .head_norm
            .forward(&mlxcel_core::gelu(&self.head_dense.forward(&pooled)));
        Ok(self.classifier.forward(&head))
    }
}
