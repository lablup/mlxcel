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

//! SigLIP text tower (`model_type: siglip`, `architectures: ["SiglipModel"]`
//! or `["SiglipTextModel"]`) served through `/v1/embeddings`.
//!
//! Layout: learned token + learned absolute position embeddings, the same
//! pre-norm encoder block the vision tower uses
//! (`crate::vision::encoders::siglip::EncoderLayer`), a final LayerNorm and a
//! linear projection `head`. The pooled vector is the hidden state at the
//! last position, which is what makes the padding rules load-bearing:
//!
//! - every input is right-padded to exactly `max_position_embeddings` (64),
//!   which [`EmbeddingModel::pad_to_max_length`] asks the engine for;
//! - the pad token and the EOS token are the same id (`</s>`, 1), so position
//!   63 always holds `</s>`, the "sticky EOS" the reference pools;
//! - no attention mask is applied, so every position attends to all 64
//!   positions. That is the training-time recipe: the checkpoint's own
//!   `tokenizer_config.json` declares `model_input_names: ["input_ids"]`, so
//!   the reference processor never produces an attention mask either.
//!
//! Pooling is therefore fixed: `1_Pooling/config.json` is not consulted and
//! [`EmbeddingModel::default_pooling`] only reports the equivalent mode.
//!
//! Image embeddings through the SigLIP vision tower are out of scope here;
//! this module serves the text side only.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use serde_json::Value;

use crate::embeddings::loader::{load_embedding_weights, quantization_params};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use crate::embeddings::pooling::PoolingMode;
use crate::vision::config::VisionHiddenActivation;
use crate::vision::encoders::siglip::{
    EncoderBlockShape, EncoderLayer, VisionMlpActivation, load_layer_norm, select_mlp_activation,
};

/// Weight prefix of the text tower inside a `SiglipModel` checkpoint.
/// A `SiglipTextModel` export keeps the same prefix, because the exported
/// module owns a `text_model` attribute in both cases.
const TEXT_PREFIX: &str = "text_model";

/// Quantization parameters used when `config.json` declares no `quantization`
/// block. `UnifiedLinear` / `UnifiedEmbedding` only consult them for a tensor
/// that actually ships `.scales`, so a dense checkpoint ignores them.
const DEFAULT_GROUP_SIZE: i32 = 64;
const DEFAULT_BITS: i32 = 4;

/// `text_config` of a SigLIP checkpoint.
///
/// Every key is optional; the defaults are the reference
/// `SiglipTextConfig` defaults, which is what
/// `google/siglip-base-patch16-224` relies on (its `text_config` declares
/// only `hidden_size`, `intermediate_size`, `num_attention_heads`,
/// `vocab_size` and `model_type`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct SigLipTextArgs {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    /// Width of the projection `head`; defaults to `hidden_size`.
    pub projection_size: Option<usize>,
    /// `None` (missing or null) means the reference default
    /// `gelu_pytorch_tanh`, not the `VisionHiddenActivation` default, which
    /// is the exact-erf GELU kept for older vision checkpoints.
    pub hidden_act: Option<VisionHiddenActivation>,
}

impl Default for SigLipTextArgs {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            hidden_size: 768,
            intermediate_size: 3_072,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            max_position_embeddings: 64,
            layer_norm_eps: 1e-6,
            projection_size: None,
            hidden_act: None,
        }
    }
}

impl SigLipTextArgs {
    /// Read the `text_config` block, falling back to the top level for a
    /// checkpoint that declares the text fields there.
    pub fn from_config(config: &Value) -> Result<Self> {
        let block = config.get("text_config").unwrap_or(config);
        serde_json::from_value(block.clone())
            .context("failed to parse the SigLIP `text_config` block")
    }

    /// Width of one embedding vector.
    #[must_use]
    pub fn projection_size(&self) -> usize {
        self.projection_size.unwrap_or(self.hidden_size)
    }

    fn activation(&self) -> VisionMlpActivation {
        select_mlp_activation(
            self.hidden_act
                .unwrap_or(VisionHiddenActivation::GeluPytorchTanh),
            false,
        )
    }
}

/// Drop everything the text tower must not see: the vision tower, the
/// contrastive `logit_scale` / `logit_bias` scalars, and any `position_ids`
/// buffer an older export baked into its shard. Remaining keys are used as
/// they are.
///
/// Used by: [`load_siglip_text_model`].
#[must_use]
pub fn sanitize_siglip_text_weights(weights: WeightMap) -> WeightMap {
    weights
        .into_iter()
        .filter(|(key, _)| keep_siglip_text_key(key))
        .collect()
}

fn keep_siglip_text_key(key: &str) -> bool {
    !(key.starts_with("vision_model.") || key.starts_with("logit_") || key.contains("position_ids"))
}

/// The SigLIP text tower.
pub struct SigLipTextModel {
    token_embedding: UnifiedEmbedding,
    position_embedding: UnifiedEmbedding,
    layers: Vec<EncoderLayer>,
    final_layer_norm: LayerNorm,
    head: UnifiedLinear,
    max_positions: usize,
    embedding_dim: usize,
}

impl SigLipTextModel {
    /// Build the tower from an already-sanitized weight map.
    pub fn from_weights(
        weights: &WeightMap,
        args: &SigLipTextArgs,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let token_embedding = UnifiedEmbedding::from_weights(
            weights,
            &format!("{TEXT_PREFIX}.embeddings.token_embedding"),
            group_size,
            bits,
        )?;
        let position_embedding = UnifiedEmbedding::from_weights(
            weights,
            &format!("{TEXT_PREFIX}.embeddings.position_embedding"),
            group_size,
            bits,
        )?;

        let activation = args.activation();
        let shape = EncoderBlockShape {
            hidden_size: args.hidden_size,
            num_attention_heads: args.num_attention_heads,
            layer_norm_eps: args.layer_norm_eps,
        };
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for index in 0..args.num_hidden_layers {
            layers.push(EncoderLayer::from_weights_parts(
                weights,
                &format!("{TEXT_PREFIX}.encoder.layers.{index}"),
                shape,
                group_size,
                bits,
                activation,
            )?);
        }

        let final_layer_norm = load_layer_norm(
            weights,
            &format!("{TEXT_PREFIX}.final_layer_norm"),
            args.layer_norm_eps,
        )?;
        let head =
            UnifiedLinear::from_weights(weights, &format!("{TEXT_PREFIX}.head"), group_size, bits)?;

        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
            head,
            max_positions: args.max_position_embeddings,
            embedding_dim: args.projection_size(),
        })
    }

    /// Run the encoder over `[B, L]` int32 ids and return the final
    /// LayerNorm output `[B, L, hidden_size]`. `L` must not exceed
    /// `max_position_embeddings`; the engine pads every batch to exactly that
    /// width through [`EmbeddingModel::pad_to_max_length`].
    pub fn encode(&self, input_ids: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(input_ids);
        if shape.len() != 2 {
            return Err(format!(
                "SigLIP text expects [B, L] input ids, got shape {shape:?}"
            ));
        }
        let length = shape[1];
        if length as usize > self.max_positions {
            return Err(format!(
                "SigLIP text has {} learned positions, got {length} tokens",
                self.max_positions
            ));
        }

        let mut hidden = self.token_embedding.forward(input_ids);
        let position_ids =
            mlxcel_core::reshape(&mlxcel_core::arange_i32(0, length, 1), &[1, length]);
        let positions = self.position_embedding.forward(&position_ids);
        hidden = mlxcel_core::add(&hidden, &positions);

        for layer in &self.layers {
            hidden = layer.forward(&hidden, None);
        }
        Ok(self.final_layer_norm.forward(&hidden))
    }

    /// Project the hidden state at the last position: `head(h[:, L - 1, :])`.
    fn pool_last_position(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden);
        let (batch, length, width) = (shape[0], shape[1], shape[2]);
        let last = mlxcel_core::slice(hidden, &[0, length - 1, 0], &[batch, length, width]);
        let pooled = mlxcel_core::reshape(&last, &[batch, width]);
        self.head.forward(&pooled)
    }
}

impl EmbeddingModel for SigLipTextModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        if batch.images.is_some_and(|images| !images.is_empty()) {
            bail!(
                "the SigLIP text tower does not accept image inputs; serving the vision tower on /v1/embeddings is a follow-up"
            );
        }
        let hidden = self.encode(batch.input_ids).map_err(|e| anyhow!(e))?;
        let embeddings = self.pool_last_position(&hidden);
        Ok(EmbeddingOutput {
            embeddings,
            last_hidden_state: Some(hidden),
        })
    }

    /// Reported only. The tower always pools the final position, which is the
    /// padded `</s>` slot, so no `1_Pooling/config.json` is consulted and this
    /// value never selects the behavior.
    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::LastToken
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn pad_to_max_length(&self) -> Option<usize> {
        Some(self.max_positions)
    }
}

/// Load a SigLIP checkpoint's text tower for `/v1/embeddings`.
///
/// Used by: `crate::embeddings::loader::build_family_model`
/// (`ModelType::SiglipText`).
pub fn load_siglip_text_model(model_dir: &Path, config: &Value) -> Result<Box<dyn EmbeddingModel>> {
    let args = SigLipTextArgs::from_config(config)?;
    let (group_size, bits) = quantization_params(config)
        .map_or((DEFAULT_GROUP_SIZE, DEFAULT_BITS), |quant| {
            (quant.group_size, quant.bits)
        });
    let weights = sanitize_siglip_text_weights(load_embedding_weights(model_dir, config)?);
    let model = SigLipTextModel::from_weights(&weights, &args, group_size, bits)
        .map_err(|e| anyhow!("failed to load the SigLIP text tower: {e}"))?;
    Ok(Box::new(model))
}

#[cfg(test)]
#[path = "siglip_text_tests.rs"]
mod siglip_text_tests;
