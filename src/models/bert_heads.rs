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

//! The two heads that sit on the [`BertEncoder`] trunk.
//!
//! - [`BertEmbeddingModel`]: the `/v1/embeddings` family model, pooled and
//!   handed to the shared engine for normalization and readback.
//! - [`BertSequenceClassifier`]: the `BertForSequenceClassification` /
//!   `XLMRobertaForSequenceClassification` head, loaded directly from a
//!   reranker checkpoint (which never detects as an embedding variant) and
//!   consumed by the `/v1/rerank` port.

use std::path::Path;

use anyhow::{Result, bail};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::loader::{
    QuantizationParams, load_embedding_weights, quantization_params, read_embedding_config,
};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};

use super::bert::{BertArgs, BertEncoder, BertVariant, sanitize};

/// `group_size` / `bits` handed to `UnifiedLinear` and `UnifiedEmbedding` for
/// a checkpoint without a `quantization` block. Both are ignored on the
/// non-quantized path (the loaders key off a `.scales` tensor), so the values
/// only need to be valid.
const DENSE_QUANTIZATION: QuantizationParams = QuantizationParams {
    group_size: 64,
    bits: 4,
};

fn quantization(config: &Value) -> QuantizationParams {
    quantization_params(config).unwrap_or(DENSE_QUANTIZATION)
}

/// Load the encoder trunk shared by both heads.
///
/// `keep_pooler` is `true` only for the BERT classification head, the sole
/// consumer of the `pooler.dense` tensors an embedding export also ships.
fn load_encoder(
    model_dir: &Path,
    config: &Value,
    variant: BertVariant,
    keep_pooler: bool,
) -> Result<(BertEncoder, WeightMap, QuantizationParams)> {
    let args = BertArgs::from_config(config, variant)?;
    let quant = quantization(config);
    let weights = sanitize(load_embedding_weights(model_dir, config)?, keep_pooler);
    let encoder = BertEncoder::from_weights(&weights, args, quant.group_size, quant.bits)?;
    Ok((encoder, weights, quant))
}

/// BERT / XLM-RoBERTa served through `/v1/embeddings` and `mlxcel embed`.
pub struct BertEmbeddingModel {
    encoder: BertEncoder,
    pooling: PoolingMode,
}

impl BertEmbeddingModel {
    /// Family default when the checkpoint ships no `1_Pooling/config.json`.
    /// Every sentence-transformers BERT export declares its own mode; mean is
    /// the sentence-transformers default for a bare `BertModel`.
    pub const DEFAULT_POOLING: PoolingMode = PoolingMode::Mean;

    /// Load an embedding checkpoint from its directory.
    pub fn load(model_dir: &Path, config: &Value, variant: BertVariant) -> Result<Self> {
        let (encoder, _, _) = load_encoder(model_dir, config, variant, false)?;
        Ok(Self {
            pooling: resolve_pooling_mode(model_dir, Self::DEFAULT_POOLING)?,
            encoder,
        })
    }

    /// Build from already-sanitized weights with an explicit pooling mode.
    pub fn from_weights(
        weights: &WeightMap,
        args: BertArgs,
        quant: QuantizationParams,
        pooling: PoolingMode,
    ) -> Result<Self> {
        Ok(Self {
            encoder: BertEncoder::from_weights(weights, args, quant.group_size, quant.bits)?,
            pooling,
        })
    }

    /// Resolved encoder geometry, for callers that need the width or the
    /// positional cap without running a forward pass.
    pub fn args(&self) -> &BertArgs {
        self.encoder.args()
    }
}

impl EmbeddingModel for BertEmbeddingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        if batch.images.is_some_and(|images| !images.is_empty()) {
            bail!("BERT / XLM-RoBERTa embedders do not accept image inputs");
        }
        let hidden =
            self.encoder
                .encode(batch.input_ids, batch.attention_mask, batch.token_type_ids)?;
        let embeddings = pool(&hidden, batch.attention_mask, self.pooling);
        Ok(EmbeddingOutput {
            embeddings,
            last_hidden_state: Some(hidden),
        })
    }

    fn default_pooling(&self) -> PoolingMode {
        Self::DEFAULT_POOLING
    }

    fn embedding_dim(&self) -> usize {
        self.encoder.args().hidden_size
    }

    fn needs_token_type_ids(&self) -> bool {
        self.encoder.args().variant == BertVariant::Bert
    }

    fn max_sequence_length(&self) -> Option<usize> {
        Some(self.encoder.args().max_sequence_length())
    }
}

/// `tanh(dense(h[:, 0, :]))` followed by the label projection. BERT keeps the
/// first linear in `pooler.dense` and the projection in `classifier`;
/// XLM-RoBERTa keeps both under `classifier.`.
struct ClassifierHead {
    dense: UnifiedLinear,
    projection: UnifiedLinear,
}

impl ClassifierHead {
    fn from_weights(
        weights: &WeightMap,
        variant: BertVariant,
        quant: QuantizationParams,
    ) -> Result<Self> {
        let (dense_prefix, projection_prefix) = match variant {
            BertVariant::Bert => ("pooler.dense", "classifier"),
            BertVariant::XlmRoberta => ("classifier.dense", "classifier.out_proj"),
        };
        let linear = |prefix: &str| -> Result<UnifiedLinear> {
            UnifiedLinear::from_weights(weights, prefix, quant.group_size, quant.bits)
                .map_err(|e| anyhow::anyhow!("bert classification head: {e}"))
        };
        Ok(Self {
            dense: linear(dense_prefix)?,
            projection: linear(projection_prefix)?,
        })
    }

    /// `[B, L, D]` hidden states to `[B, num_labels]` logits.
    fn forward(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden);
        let first = mlxcel_core::utils::slice_axis(hidden, 1, 0, 1);
        let first = mlxcel_core::reshape(&first, &[shape[0], shape[2]]);
        let pooled = mlxcel_core::tanh(&self.dense.forward(&first));
        self.projection.forward(&pooled)
    }
}

/// `BertForSequenceClassification` / `XLMRobertaForSequenceClassification`.
///
/// Loaded straight from a checkpoint directory: a `ForSequenceClassification`
/// export is a reranker, so it is deliberately not an embedding variant and
/// never reaches [`crate::embeddings::load_embedding_model`].
pub struct BertSequenceClassifier {
    encoder: BertEncoder,
    head: ClassifierHead,
}

impl BertSequenceClassifier {
    /// Load a sequence-classification checkpoint from its directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = read_embedding_config(model_dir)?;
        let Some(variant) = BertVariant::from_config(&config) else {
            bail!(
                "{} is not a BERT or XLM-RoBERTa checkpoint; its config.json declares \
                 model_type `{}`",
                model_dir.display(),
                config
                    .get("model_type")
                    .and_then(Value::as_str)
                    .unwrap_or("<missing>")
            );
        };
        let (encoder, weights, quant) = load_encoder(model_dir, &config, variant, true)?;
        Ok(Self {
            encoder,
            head: ClassifierHead::from_weights(&weights, variant, quant)?,
        })
    }

    /// Build from already-sanitized weights (the `pooler.` tensors kept).
    pub fn from_weights(
        weights: &WeightMap,
        args: BertArgs,
        quant: QuantizationParams,
    ) -> Result<Self> {
        let variant = args.variant;
        Ok(Self {
            encoder: BertEncoder::from_weights(weights, args, quant.group_size, quant.bits)?,
            head: ClassifierHead::from_weights(weights, variant, quant)?,
        })
    }

    /// Resolved encoder geometry, including `num_labels`.
    pub fn args(&self) -> &BertArgs {
        self.encoder.args()
    }

    /// Whether a batch for this checkpoint must carry segment ids. BERT
    /// cross-encoders encode `[CLS] query [SEP] document [SEP]` with segment
    /// `1` on the document half; XLM-RoBERTa has a single-row table.
    pub fn needs_token_type_ids(&self) -> bool {
        self.encoder.args().variant == BertVariant::Bert
    }

    /// Score one right-padded `[B, L]` batch, returning `[B, num_labels]`.
    pub fn logits(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
        token_type_ids: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let hidden = self
            .encoder
            .encode(input_ids, attention_mask, token_type_ids)?;
        Ok(self.head.forward(&hidden))
    }
}

#[cfg(test)]
#[path = "bert_heads_tests.rs"]
mod bert_heads_tests;

#[cfg(test)]
#[path = "bert_real_checkpoint_tests.rs"]
mod bert_real_checkpoint_tests;
