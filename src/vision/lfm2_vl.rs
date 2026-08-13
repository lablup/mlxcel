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

//! LFM2-VL (`lfm2_vl` / `lfm2-vl`) Vision-Language Model.
//!
//! Port of LiquidAI LFM2-VL. Composition:
//! - `vision_tower` ([`Lfm2VlVisionTower`]): packed-patch SigLIP2-style ViT at
//!   each image's native patch count.
//! - `connector` ([`Lfm2VlConnector`]): pixel unshuffle (space-to-depth) + a
//!   `LayerNorm -> Linear -> GELU -> Linear` projector into the text hidden size.
//! - `language_model` ([`Lfm2Model`]): the LFM2 hybrid short-conv/attention
//!   backbone (dense or MoE), whose per-layer mixed state is model-owned.
//!
//! Each image runs through the tower + connector independently (variable patch
//! count per image, no padding), and the resulting `T_i` feature rows replace
//! the `<image>` (396) placeholder rows in the prompt embedding via
//! [`crate::vision::merge::merge_llava`]. The wrapper delegates the entire
//! `LanguageModel` surface (including the model-owned sequence-state hooks) to
//! the backbone so the conv tail and KV state stay consistent across the
//! embedding-injected prefill and token-only decode steps.
//!
//! Used by: `loading::load_lfm2_vl`, `multimodal::vlm_runtime`.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::lfm2::Lfm2Model;
use crate::vision::connectors::lfm2_vl::Lfm2VlConnector;
use crate::vision::encoders::lfm2_vl::Lfm2VlVisionTower;
use crate::vision::feature_cache::{CacheKey, SingleArrayFeatures, VisionFeatureCache};
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::lfm2_vl::Lfm2VlProcessor;

/// Top-level LFM2-VL runtime.
pub struct Lfm2VlModel {
    pub text_model: Lfm2Model,
    pub vision_tower: Lfm2VlVisionTower,
    pub connector: Lfm2VlConnector,
    pub processor: Lfm2VlProcessor,
    /// `<image>` placeholder id (`image_token_index`, 396).
    pub image_token_id: i32,
    /// `<|image_start|>` (498) / `<|image_end|>` (499); `0` when absent.
    pub image_start_id: i32,
    pub image_end_id: i32,
    pub use_image_special_tokens: bool,
    pub downsample_factor: i32,
    pub patch_dim: i32,
    pub eos_token_ids: Vec<i32>,
}

impl Lfm2VlModel {
    /// Merge image features into the text embedding stream.
    ///
    /// `pixel_values`: `[1, sum_i(h_i*w_i), patch_dim]` (all images' packed
    /// patches concatenated); `grids`: per-image `(h_i, w_i)`. Each image runs the
    /// tower + connector at its own grid, and the `sum_i T_i` feature rows replace
    /// the `<image>` positions.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grids: &[(i32, i32)],
    ) -> InputEmbeddings {
        self.get_input_embeddings_with_cache(input_ids, pixel_values, grids, None, None)
    }

    /// Get input embeddings with optional vision feature caching.
    ///
    /// LFM2-VL runs every image through the tower + connector independently
    /// (variable patch count per image), and the resulting per-image feature
    /// rows are concatenated before the LLaVA-style merge. The cache stores
    /// the concatenated post-connector tensor keyed by the request-supplied
    /// image bytes (SHA-256 + soft-token budget), so multi-turn conversations
    /// that revisit the same image skip the vision tower + connector on
    /// subsequent turns.
    ///
    /// Note: the connector output shape depends on the image's native patch
    /// count, which is folded into the key by the soft-token budget; two
    /// different resolutions of the same image cannot collide.
    pub fn get_input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grids: &[(i32, i32)],
        cache_key: Option<&CacheKey>,
        vision_cache: Option<&std::sync::Mutex<VisionFeatureCache<SingleArrayFeatures>>>,
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.input_embeddings(input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);

        // Cache lookup: reuse the concatenated post-connector features when
        // the same image bytes have already been encoded.
        let cached_features = match (cache_key, vision_cache) {
            (Some(key), Some(cache)) => cache
                .lock()
                .ok()
                .and_then(|mut guard| guard.get(key))
                .map(|c| c.features),
            _ => None,
        };

        let image_features: UniquePtr<MlxArray> = if let Some(cached) = cached_features {
            cached
        } else {
            let pv = mlxcel_core::astype(pixel_values, embed_dtype);
            let mut features: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(grids.len());
            let mut offset = 0i32;
            for &(h, w) in grids {
                let n = h * w;
                let patches =
                    mlxcel_core::slice(&pv, &[0, offset, 0], &[1, offset + n, self.patch_dim]);
                offset += n;
                let vision_out = self.vision_tower.forward(&patches, (h, w));
                features.push(self.connector.forward(&vision_out, (h, w)));
            }

            let combined = match features.len() {
                0 => mlxcel_core::astype(&inputs_embeds, embed_dtype),
                1 => features.into_iter().next().unwrap(),
                _ => {
                    let mut iter = features.into_iter();
                    let first = iter.next().unwrap();
                    iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 0))
                }
            };

            // Populate the cache on miss so subsequent turns can skip the
            // vision tower + connector. Materialize before copying so we
            // cache a concrete array rather than a deferred graph node.
            if let (Some(key), Some(cache)) = (cache_key, vision_cache) {
                mlxcel_core::eval(&combined);
                let snapshot = SingleArrayFeatures::new(mlxcel_core::copy(combined.as_ref().unwrap()));
                if let Ok(mut guard) = cache.lock() {
                    guard.put(key.clone(), &snapshot);
                }
            }

            combined
        };

        merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
    }
}

// LanguageModel: delegate the full surface (including model-owned sequence-state
// hooks) to the LFM2 backbone; override only the EOS ids.
impl LanguageModel for Lfm2VlModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_sequence_id(input_ids, seq_id, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
        )
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched_with_context(input_ids, batch_caches, mask, context)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        self.text_model.embed_tokens(input_ids)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(&self.text_model)
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn supports_batching(&self) -> bool {
        self.text_model.supports_batching()
    }

    fn supports_padded_prefill(&self) -> bool {
        self.text_model.supports_padded_prefill()
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.text_model.prepare_sequence_state(seq_id);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.text_model.release_sequence_state_by_id(seq_id);
    }

    fn supports_snapshot_reuse(&self) -> bool {
        self.text_model.supports_snapshot_reuse()
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        self.text_model.snapshot_sequence_state(seq_id, token_len)
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        self.text_model.restore_sequence_state(seq_id, snapshot)
    }

    fn trim_internal_caches(&self, excess: i32) {
        self.text_model.trim_internal_caches(excess);
    }
}
