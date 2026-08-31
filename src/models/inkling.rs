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

//! Inkling text decoder.
//!
//! Inkling replaces RoPE with learned banded relative-position logits, adds
//! four causal short-convolution states to every layer, and combines selected
//! and shared experts through one logsigmoid-normalized router. Vision, audio,
//! and MTP tensors belong to separate model wrappers and are ignored here.

mod attention;
mod mlp;
mod runtime;
mod sanitize;
mod speculative;
mod validation;
mod validation_shapes;

use mlxcel_core::inkling_layer::{InklingDecoderLayer, InklingLayerSpec};
use mlxcel_core::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use self::mlp::InklingMlp;
use self::runtime::InklingLayerCache;
use super::model_owned::ModelOwnedSequenceState;
use crate::audio::inkling_tower::{InklingAudioConfig, InklingAudioTower};
use crate::vision::merge::InputEmbeddings;

const DEFAULT_EOS_TOKEN_ID: i32 = 200_006;

fn d_model_type() -> String {
    "inkling_mm_model".into()
}
fn d_text_model_type() -> String {
    "inkling".into()
}
fn d_hidden() -> usize {
    6144
}
fn d_layers() -> usize {
    66
}
fn d_vocab() -> usize {
    201_024
}
fn d_eps() -> f32 {
    1e-6
}
fn d_true() -> bool {
    true
}
fn d_one() -> f32 {
    1.0
}
fn d_heads() -> usize {
    64
}
fn d_kv_heads() -> usize {
    8
}
fn d_swa_kv_heads() -> usize {
    16
}
fn d_head_dim() -> usize {
    128
}
fn d_window() -> usize {
    512
}
fn d_rel() -> usize {
    16
}
fn d_extent() -> usize {
    1024
}
fn d_log_alpha() -> f32 {
    0.1
}
fn d_kernel() -> usize {
    4
}
fn d_moe_width() -> usize {
    24_576
}
fn d_routed() -> usize {
    256
}
fn d_topk() -> usize {
    6
}
fn d_shared() -> usize {
    2
}
fn d_route_scale() -> f32 {
    8.0
}
fn d_image_token() -> i32 {
    200_054
}
fn d_audio_token() -> i32 {
    200_053
}

#[derive(Debug, Clone, Deserialize)]
pub struct InklingTextConfig {
    #[serde(default = "d_text_model_type")]
    pub model_type: String,
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default)]
    pub unpadded_vocab_size: Option<usize>,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "d_true")]
    pub use_embed_norm: bool,
    #[serde(default = "d_one")]
    pub logits_mup_width_multiplier: f32,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_head_dim")]
    pub head_dim: usize,
    #[serde(default = "d_heads")]
    pub swa_num_attention_heads: usize,
    #[serde(default = "d_swa_kv_heads")]
    pub swa_num_key_value_heads: usize,
    #[serde(default = "d_head_dim")]
    pub swa_head_dim: usize,
    #[serde(default = "d_window")]
    pub sliding_window_size: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub local_layer_ids: Option<Vec<usize>>,
    #[serde(default = "d_rel")]
    pub d_rel: usize,
    #[serde(default = "d_extent")]
    pub rel_extent: usize,
    #[serde(default)]
    pub log_scaling_n_floor: Option<usize>,
    #[serde(default = "d_log_alpha")]
    pub log_scaling_alpha: f32,
    #[serde(default = "d_kernel", alias = "conv_kernel_size")]
    pub sconv_kernel_size: usize,
    #[serde(default)]
    pub dense_mlp_idx: usize,
    #[serde(default)]
    pub mlp_layer_types: Option<Vec<String>>,
    #[serde(default = "d_moe_width")]
    pub intermediate_size: usize,
    #[serde(default)]
    pub dense_intermediate_size: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default = "d_routed")]
    pub n_routed_experts: usize,
    #[serde(default = "d_topk")]
    pub num_experts_per_tok: usize,
    #[serde(default = "d_shared")]
    pub n_shared_experts: usize,
    #[serde(default = "d_route_scale")]
    pub route_scale: f32,
}

impl InklingTextConfig {
    pub fn layer_is_sliding(&self, i: usize) -> bool {
        if let Some(types) = &self.layer_types {
            return types.get(i).is_some_and(|kind| kind == "hybrid_sliding");
        }
        if let Some(ids) = &self.local_layer_ids {
            return ids.contains(&i);
        }
        !(i + 1).is_multiple_of(6)
    }

    pub fn layer_is_dense(&self, i: usize) -> bool {
        self.mlp_layer_types
            .as_ref()
            .map_or(i < self.dense_mlp_idx, |types| {
                types.get(i).is_some_and(|kind| kind == "dense")
            })
    }

    pub fn widths(&self) -> Result<(usize, usize), String> {
        if let Some(moe) = self.moe_intermediate_size {
            Ok((self.intermediate_size, moe))
        } else {
            self.dense_intermediate_size
                .map(|dense| (dense, self.intermediate_size))
                .ok_or_else(|| "Inkling native config requires dense_intermediate_size".into())
        }
    }

    fn layer_spec(&self, config: &InklingConfig, index: usize) -> Result<InklingLayerSpec, String> {
        let is_sliding = self.layer_is_sliding(index);
        let (num_attention_heads, num_key_value_heads, head_dim) = if is_sliding {
            (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            )
        } else {
            (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            )
        };
        let (dense_intermediate_size, _) = self.widths()?;
        let (quantization_group_size, quantization_bits, _) = config.quantization();
        Ok(InklingLayerSpec {
            hidden_size: self.hidden_size,
            rms_norm_eps: self.rms_norm_eps,
            is_sliding,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            sliding_window_size: self.sliding_window_size,
            d_rel: self.d_rel,
            rel_extent: self.rel_extent,
            log_scaling_n_floor: self.log_scaling_n_floor,
            log_scaling_alpha: self.log_scaling_alpha,
            sconv_kernel_size: self.sconv_kernel_size,
            dense_intermediate_size,
            quantization_group_size,
            quantization_bits,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct InklingConfig {
    #[serde(default = "d_model_type")]
    pub model_type: String,
    pub text_config: InklingTextConfig,
    #[serde(default)]
    pub audio_config: Option<InklingAudioConfig>,
    #[serde(default = "d_image_token")]
    pub image_token_id: i32,
    #[serde(default = "d_audio_token")]
    pub audio_token_id: i32,
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

impl InklingConfig {
    pub(crate) fn from_json_with_sidecar(path: &Path, raw: &str) -> Result<Self, String> {
        let raw = super::sanitize_config_json(raw);
        let mut value: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| format!("Failed to parse config.json: {e}"))?;
        sanitize::promote_nvfp4_config(path, &mut value)?;
        serde_json::from_value(value).map_err(|e| format!("Failed to parse Inkling config: {e}"))
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        let ids = super::parse_optional_eos_token_ids(&self.eos_token_id);
        if ids.is_empty() {
            vec![DEFAULT_EOS_TOKEN_ID]
        } else {
            ids
        }
    }

    pub(crate) fn quantization(&self) -> (i32, i32, &'static str) {
        let q = self
            .quantization
            .as_ref()
            .or(self.quantization_config.as_ref());
        let integer = |key: &str, default: i32| match q.and_then(|v| v.get(key)) {
            Some(value) => value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .unwrap_or(0),
            None => default,
        };
        let group = integer("group_size", 64);
        let bits = integer("bits", 4);
        let mode = q.and_then(|v| v.get("mode")).and_then(|v| v.as_str());
        (
            group,
            bits,
            if mode == Some("nvfp4") {
                "nvfp4"
            } else {
                "affine"
            },
        )
    }
}

pub struct InklingModel {
    config: InklingConfig,
    embed_tokens: UnifiedEmbedding,
    embed_norm: Option<RMSNorm>,
    layers: Vec<InklingDecoderLayer<InklingMlp>>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    audio_tower: Option<InklingAudioTower>,
    eos_token_ids: Vec<i32>,
    sequence_state: ModelOwnedSequenceState<InklingLayerCache>,
}

impl InklingModel {
    pub fn load(model_path: &str) -> Result<(Self, InklingConfig), String> {
        let path = Path::new(model_path);
        let raw = std::fs::read_to_string(path.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config = InklingConfig::from_json_with_sidecar(path, &raw)?;
        let model = Self::from_weights(config.clone(), super::load_text_weights(path, None)?)?;
        Ok((model, config))
    }

    pub fn from_weights(config: InklingConfig, weights: WeightMap) -> Result<Self, String> {
        validation::validate_config(&config)?;
        let weights = sanitize::sanitize_weights(weights)?;
        validation::validate_weight_shapes(&weights, &config)?;
        let (group, bits, _) = config.quantization();
        let text = &config.text_config;
        let embed_tokens =
            UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", group, bits)?;
        let embed_norm = text
            .use_embed_norm
            .then(|| weight(&weights, "model.embed_norm.weight"))
            .transpose()?
            .map(|w| RMSNorm::new(w, text.rms_norm_eps));
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for i in 0..text.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let spec = text.layer_spec(&config, i)?;
            let mlp = InklingMlp::from_weights(&weights, &config, i)?;
            layers.push(InklingDecoderLayer::from_weights(
                &weights, &prefix, &spec, mlp,
            )?);
        }
        let norm = RMSNorm::new(weight(&weights, "model.norm.weight")?, text.rms_norm_eps);
        let lm_head = if text.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                &weights, "lm_head", group, bits,
            )?)
        };
        let audio_tower = config
            .audio_config
            .as_ref()
            .map(|audio| InklingAudioTower::from_weights(&weights, audio, group, bits))
            .transpose()?;
        let state = (0..text.num_hidden_layers)
            .map(|_| InklingLayerCache::new())
            .collect();
        let eos_token_ids = config.eos_token_ids();
        Ok(Self {
            config,
            embed_tokens,
            embed_norm,
            layers,
            norm,
            lm_head,
            audio_tower,
            eos_token_ids,
            sequence_state: ModelOwnedSequenceState::new(state),
        })
    }

    pub fn input_embeddings(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input)
    }

    /// Text embeddings after Inkling's optional embedding RMSNorm.
    ///
    /// Image and audio soft tokens are already normalized by their towers, so
    /// multimodal wrappers merge them into this tensor and then enter the
    /// decoder without applying `embed_norm` a second time.
    /// Embed token IDs and apply Inkling's input RMS normalization.
    ///
    /// Multimodal towers scatter their already text-width features into this
    /// normalized matrix, so the decoder must subsequently use one of the
    /// `forward_prepared_*` entry points and must not normalize it again.
    pub fn normalized_input_embeddings(
        &self,
        input: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let embeddings = self.embed_tokens.forward(input);
        Ok(match &self.embed_norm {
            Some(norm) => norm.forward(&embeddings),
            None => embeddings,
        })
    }

    #[must_use]
    pub fn supports_audio(&self) -> bool {
        self.audio_tower.is_some()
    }

    #[must_use]
    pub fn audio_token_id(&self) -> i32 {
        self.config.audio_token_id
    }

    /// Merge valid dMel rows into already-normalized text/image embeddings.
    ///
    /// The caller owns prompt placeholder expansion and passes only valid
    /// `[frames, n_mel_bins]` rows in clip order. This makes the method usable
    /// by the image wrapper from #1327, where image rows are scattered first
    /// and audio rows must be scattered second.
    pub fn merge_audio_embeddings(
        &self,
        input_ids: &MlxArray,
        normalized_embeddings: &MlxArray,
        audio_input_ids: &MlxArray,
    ) -> Result<InputEmbeddings, String> {
        let input_shape = mlxcel_core::array_shape(input_ids);
        if mlxcel_core::array_dtype(input_ids) != mlxcel_core::dtype::INT32
            || input_shape.len() != 2
            || input_shape[0] != 1
            || input_shape[1] <= 0
        {
            return Err(format!(
                "Inkling audio merge requires int32 input_ids shaped [1, sequence], got {input_shape:?}"
            ));
        }
        let embedding_shape = mlxcel_core::array_shape(normalized_embeddings);
        if embedding_shape.len() != 3
            || embedding_shape[0] != 1
            || embedding_shape[1] != input_shape[1]
            || embedding_shape[2] != self.config.text_config.hidden_size as i32
        {
            return Err(format!(
                "Inkling audio merge requires normalized embeddings shaped [1, {}, {}], got {embedding_shape:?}",
                input_shape[1], self.config.text_config.hidden_size
            ));
        }
        let tower = self.audio_tower.as_ref().ok_or_else(|| {
            "This Inkling checkpoint was loaded without an audio tower".to_string()
        })?;
        let features = tower.forward(audio_input_ids)?;
        let placeholders = mlxcel_core::item_i32(&mlxcel_core::sum_all(&mlxcel_core::astype(
            &mlxcel_core::equal(
                input_ids,
                &mlxcel_core::from_slice_i32(&[self.config.audio_token_id], &[1]),
            ),
            mlxcel_core::dtype::INT32,
        )));
        let feature_rows = mlxcel_core::array_shape(&features)[0];
        if placeholders != feature_rows {
            return Err(format!(
                "Inkling prompt has {placeholders} audio placeholder rows but the tower produced {feature_rows} rows"
            ));
        }
        Ok(crate::vision::merge::merge_llava(
            self.config.audio_token_id,
            &features,
            normalized_embeddings,
            input_ids,
        ))
    }

    fn make_internal_caches(&self) -> Vec<InklingLayerCache> {
        (0..self.layers.len())
            .map(|_| InklingLayerCache::new())
            .collect()
    }

    fn pre_norm_hidden_embeddings_with_caches(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
    ) -> UniquePtr<MlxArray> {
        let normalized = self.embed_norm.as_ref().map_or_else(
            || mlxcel_core::copy(embeddings),
            |norm| norm.forward(embeddings),
        );
        self.hidden_prepared_embeddings_with_caches(&normalized, caches)
    }

    fn hidden_prepared_embeddings_with_caches(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
    ) -> UniquePtr<MlxArray> {
        let mut h = mlxcel_core::copy(embeddings);
        for (layer, cache) in self.layers.iter().zip(caches) {
            h = layer.forward(&h, cache);
        }
        h
    }

    fn hidden_embeddings_with_caches(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
    ) -> UniquePtr<MlxArray> {
        let hidden = self.pre_norm_hidden_embeddings_with_caches(embeddings, caches);
        self.norm.forward(&hidden)
    }

    fn project_hidden(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let h = mlxcel_core::divide_scalar(h, self.config.text_config.logits_mup_width_multiplier);
        let logits = self
            .lm_head
            .as_ref()
            .map_or_else(|| self.embed_tokens.as_linear(&h), |head| head.forward(&h));
        let limit = self
            .config
            .text_config
            .unpadded_vocab_size
            .unwrap_or(self.config.text_config.vocab_size) as i32;
        if mlxcel_core::array_shape(&logits)[2] > limit {
            mlxcel_core::utils::slice_axis(&logits, -1, 0, limit)
        } else {
            logits
        }
    }

    fn forward_embeddings_with_caches(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
    ) -> UniquePtr<MlxArray> {
        let hidden = self.hidden_embeddings_with_caches(embeddings, caches);
        self.project_hidden(&hidden)
    }

    fn forward_last_embeddings_with_caches(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.hidden_embeddings_with_caches(embeddings, caches);
        let shape = mlxcel_core::array_shape(&hidden);
        let row = mlxcel_core::slice(
            &hidden,
            &[0, last_pos as i32, 0],
            &[shape[0], last_pos as i32 + 1, shape[2]],
        );
        self.project_hidden(&row)
    }

    fn forward_prepared_embeddings_with_caches(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
    ) -> UniquePtr<MlxArray> {
        let hidden = self.hidden_prepared_embeddings_with_caches(embeddings, caches);
        self.project_hidden(&hidden)
    }

    fn forward_last_prepared_embeddings_with_caches(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.hidden_prepared_embeddings_with_caches(embeddings, caches);
        let shape = mlxcel_core::array_shape(&hidden);
        let row = mlxcel_core::slice(
            &hidden,
            &[0, last_pos as i32, 0],
            &[shape[0], last_pos as i32 + 1, shape[2]],
        );
        self.project_hidden(&row)
    }

    pub fn forward_prepared_embeddings(&self, embeddings: &MlxArray) -> UniquePtr<MlxArray> {
        self.sequence_state.with_sequence_state(None, |state| {
            self.forward_prepared_embeddings_with_caches(embeddings, state)
        })
    }

    pub fn forward_prepared_embeddings_with_sequence_id(
        &self,
        embeddings: &MlxArray,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| self.forward_prepared_embeddings_with_caches(embeddings, state),
        )
    }

    pub fn forward_last_prepared_embeddings_with_sequence_id(
        &self,
        embeddings: &MlxArray,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| self.forward_last_prepared_embeddings_with_caches(embeddings, state, last_pos),
        )
    }

    fn forward_with_caches(
        &self,
        input: &MlxArray,
        caches: &mut [InklingLayerCache],
    ) -> UniquePtr<MlxArray> {
        let embeddings = self.embed_tokens.forward(input);
        self.forward_embeddings_with_caches(&embeddings, caches)
    }
}

fn weight(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

#[cfg(test)]
#[path = "inkling_tests.rs"]
mod tests;
#[cfg(test)]
mod tiny_tests;
#[cfg(test)]
pub(crate) use tiny_tests::tiny_model;
