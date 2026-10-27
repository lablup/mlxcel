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

use mlxcel_core::cache::{CachePool, KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::feature_cache::{CacheKey, SingleArrayFeatures, VisionFeatureCache};
use crate::models::lfm2::Lfm2Model;
use crate::vision::connectors::lfm2_vl::Lfm2VlConnector;
use crate::vision::encoders::lfm2_vl::Lfm2VlVisionTower;
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::lfm2_vl::{Lfm2VlImageLayout, Lfm2VlProcessor};

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
    pub img_row_col_ids: [[i32; 10]; 10],
    pub img_thumbnail_id: i32,
    pub use_image_special_tokens: bool,
    pub downsample_factor: i32,
    pub patch_dim: i32,
    pub eos_token_ids: Vec<i32>,
}

impl Lfm2VlModel {
    /// Expected number of post-connector feature rows emitted for one logical
    /// image layout (`sum_v ceil(h_v / f) * ceil(w_v / f)`).
    fn expected_layout_feature_rows(&self, layout: &Lfm2VlImageLayout) -> i32 {
        let f = self.downsample_factor.max(1);
        layout
            .views
            .iter()
            .map(|&(h, w)| ((h + f - 1) / f) * ((w + f - 1) / f))
            .sum()
    }

    /// Merge image features into the text embedding stream.
    ///
    /// `pixel_values`: `[1, sum_i(h_i*w_i), patch_dim]` (all packed views
    /// concatenated); `layouts`: per-image view layouts. Each view runs the
    /// tower and connector at its own grid, and the `sum_i T_i` feature rows
    /// replace the `<image>` positions.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        layouts: &[Lfm2VlImageLayout],
    ) -> InputEmbeddings {
        self.get_input_embeddings_with_cache(input_ids, pixel_values, layouts, None, None)
    }

    /// Merge image features into the text embedding stream, consulting an
    /// optional per-image [`VisionFeatureCache`].
    ///
    /// Each entry in `layouts` corresponds to one logical input image (which
    /// may comprise a single view or multiple tiles plus an optional
    /// thumbnail). When `image_keys[i]` is `Some(key)` and `vision_cache` hits
    /// with a feature tensor whose leading dimension matches the layout's
    /// expected post-connector token count, all views of image `i` skip the
    /// vision tower and connector. On a miss (or row-count mismatch), image
    /// `i`'s views are encoded, concatenated into `[T_i, text_hidden]`,
    /// materialized, and stored in `vision_cache` before merging.
    pub fn get_input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        layouts: &[Lfm2VlImageLayout],
        image_keys: Option<&[Option<CacheKey>]>,
        vision_cache: Option<&std::sync::Mutex<VisionFeatureCache<SingleArrayFeatures>>>,
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.input_embeddings(input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let mut pv_cast: Option<UniquePtr<MlxArray>> = None;

        let mut per_image_features: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(layouts.len());
        let mut offset = 0i32;

        for (img_idx, layout) in layouts.iter().enumerate() {
            let img_patch_count: i32 = layout.views.iter().map(|&(h, w)| h * w).sum();
            let expected_rows = self.expected_layout_feature_rows(layout);
            let cache_key = image_keys
                .and_then(|keys| keys.get(img_idx))
                .and_then(|k| k.as_ref());

            if let (Some(key), Some(cache)) = (cache_key, vision_cache)
                && let Ok(mut guard) = cache.lock()
                && let Some(cached) = guard.get(key)
                && mlxcel_core::array_shape(&cached.features).first().copied()
                    == Some(expected_rows)
            {
                offset += img_patch_count;
                let feat = if mlxcel_core::array_dtype(&cached.features) == embed_dtype {
                    cached.features
                } else {
                    mlxcel_core::astype(&cached.features, embed_dtype)
                };
                per_image_features.push(feat);
                continue;
            }

            let pv = pv_cast.get_or_insert_with(|| mlxcel_core::astype(pixel_values, embed_dtype));
            let mut view_features: Vec<UniquePtr<MlxArray>> =
                Vec::with_capacity(layout.views.len());
            for &(h, w) in &layout.views {
                let n = h * w;
                // patches_i: [1, n, patch_dim]
                let patches =
                    mlxcel_core::slice(pv, &[0, offset, 0], &[1, offset + n, self.patch_dim]);
                offset += n;
                let vision_out = self.vision_tower.forward(&patches, (h, w));
                view_features.push(self.connector.forward(&vision_out, (h, w)));
            }

            let img_features = match view_features.len() {
                0 => continue,
                1 => view_features.into_iter().next().unwrap(),
                _ => {
                    let mut iter = view_features.into_iter();
                    let first = iter.next().unwrap();
                    iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 0))
                }
            };

            if let (Some(key), Some(cache)) = (cache_key, vision_cache) {
                mlxcel_core::eval(&img_features);
                let snapshot = SingleArrayFeatures::new(mlxcel_core::copy(&img_features));
                if let Ok(mut guard) = cache.lock() {
                    guard.put(key.clone(), &snapshot);
                }
            }

            per_image_features.push(img_features);
        }

        let image_features = match per_image_features.len() {
            0 => mlxcel_core::astype(&inputs_embeds, embed_dtype), // unused; no images
            1 => per_image_features.into_iter().next().unwrap(),
            _ => {
                let mut iter = per_image_features.into_iter();
                let first = iter.next().unwrap();
                iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 0))
            }
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

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched(input_ids, batch_caches, mask)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched_with_ids(input_ids, seq_ids, batch_caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        self.text_model.embed_tokens(input_ids)
    }

    fn embed_tokens_module(&self) -> Option<mlxcel_core::layers::UnifiedEmbedding> {
        self.text_model.embed_tokens_module()
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(&self.text_model)
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.text_model.set_kv_cache_layer_modes(modes)
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.text_model.kv_cache_layer_modes()
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

    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        self.text_model.batched_decode_kind()
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        self.text_model.sequence_state_layout()
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        self.text_model.sync_sequence_storage(seq_id, cache_pool)
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

    fn trim_state(
        &self,
        seq: Option<mlxcel_core::cache::SequenceId>,
        excess: i32,
    ) -> Result<(), String> {
        self.text_model.trim_state(seq, excess)
    }
}

/// DFlash target adapter for text-only requests against an LFM2-VL
/// checkpoint (issue #1339): every hook delegates to the LFM2 text
/// backbone, which owns the caches the burst allocates through
/// `Lfm2Model::make_speculative_caches`. Image-bearing requests never reach
/// the speculative path (the burst gate declines them to classic decode).
impl mlxcel_core::drafter::dflash::SpeculativeTarget for Lfm2VlModel {
    type Cache = crate::models::lfm2::Lfm2LayerCache;
    type VerifyOut = crate::models::lfm2::VerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        mlxcel_core::drafter::dflash::SpeculativeTarget::capture_layer_ids(&self.text_model)
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.text_model
            .forward_speculative(verify_input, caches, &[], true)
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        self.text_model
            .forward_speculative(verify_input, caches, capture_layer_ids, true)
    }

    /// Same prefill policy as the text backbone: no short-conv snapshots on a
    /// forward that is never rolled back.
    fn prefill_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        self.text_model
            .forward_speculative(verify_input, caches, capture_layer_ids, false)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.text_model.rollback_speculative_cache(
            caches,
            &verify_out.conv_states,
            accepted,
            block_size,
        );
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        mlxcel_core::drafter::dflash::SpeculativeTarget::concat_hidden_for_drafter(
            &self.text_model,
            verify_out,
        )
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        mlxcel_core::drafter::dflash::SpeculativeTarget::verify_logits(&self.text_model, verify_out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use image::{DynamicImage, Rgb, RgbImage};
    use mlxcel_core::weights::WeightMap;
    use serde_json::json;

    use super::*;
    use crate::LoadedModel;
    use crate::models::lfm2::ModelArgs;
    use crate::multimodal::vlm_runtime::prepare_vlm_embeddings_with_cache;
    use crate::vision::encoders::lfm2_vl::Lfm2VlVisionConfig;
    use crate::vision::feature_cache::ModelVisionCaches;
    use crate::vision::processors::lfm2_vl::Lfm2VlTilingPolicy;

    fn build_synthetic_lfm2_vl() -> Lfm2VlModel {
        let vocab_size = 512usize;
        let text_hidden = 8usize;
        let vision_hidden = 4usize;
        let patch_size = 2usize;
        let patch_dim = patch_size * patch_size * 3; // 12
        let downsample_factor = 2i32;
        let unshuffled_dim = vision_hidden * (downsample_factor as usize).pow(2); // 16

        let text_args: ModelArgs = serde_json::from_value(json!({
            "model_type": "lfm2",
            "vocab_size": vocab_size,
            "hidden_size": text_hidden,
            "num_hidden_layers": 0,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "norm_eps": 1e-5,
            "conv_bias": false,
            "conv_L_cache": 3
        }))
        .expect("parse synthetic LFM2 text ModelArgs");

        let mut text_weights = WeightMap::new();
        let embed_vals: Vec<f32> = (0..vocab_size * text_hidden)
            .map(|i| (i as f32) * 0.001)
            .collect();
        text_weights.insert(
            "model.embed_tokens.weight".to_string(),
            mlxcel_core::from_slice_f32(&embed_vals, &[vocab_size as i32, text_hidden as i32]),
        );
        text_weights.insert(
            "model.embedding_norm.weight".to_string(),
            mlxcel_core::ones(&[text_hidden as i32], mlxcel_core::dtype::FLOAT32),
        );
        let text_model =
            Lfm2Model::from_weights(text_args, text_weights).expect("build synthetic Lfm2Model");

        let vision_cfg = Lfm2VlVisionConfig {
            hidden_size: vision_hidden,
            intermediate_size: 8,
            num_hidden_layers: 0,
            num_attention_heads: 2,
            patch_size,
            num_patches: 4, // 2x2 base position grid
            layer_norm_eps: 1e-6,
            vision_feature_layer: -1,
        };

        let mut vlm_weights = WeightMap::new();
        let patch_w: Vec<f32> = (0..vision_hidden * patch_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        vlm_weights.insert(
            "vision_tower.embeddings.patch_embedding.weight".to_string(),
            mlxcel_core::from_slice_f32(&patch_w, &[vision_hidden as i32, patch_dim as i32]),
        );
        vlm_weights.insert(
            "vision_tower.embeddings.position_embedding.weight".to_string(),
            mlxcel_core::zeros(&[4, vision_hidden as i32], mlxcel_core::dtype::FLOAT32),
        );
        vlm_weights.insert(
            "vision_tower.post_layernorm.weight".to_string(),
            mlxcel_core::ones(&[vision_hidden as i32], mlxcel_core::dtype::FLOAT32),
        );

        let lin1_w: Vec<f32> = (0..text_hidden * unshuffled_dim)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.04)
            .collect();
        vlm_weights.insert(
            "multi_modal_projector.linear_1.weight".to_string(),
            mlxcel_core::from_slice_f32(&lin1_w, &[text_hidden as i32, unshuffled_dim as i32]),
        );
        let lin2_w: Vec<f32> = (0..text_hidden * text_hidden)
            .map(|i| if i % (text_hidden + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        vlm_weights.insert(
            "multi_modal_projector.linear_2.weight".to_string(),
            mlxcel_core::from_slice_f32(&lin2_w, &[text_hidden as i32, text_hidden as i32]),
        );

        let vision_tower =
            Lfm2VlVisionTower::from_weights(&vlm_weights, "vision_tower", &vision_cfg, 64, 4)
                .expect("build synthetic Lfm2VlVisionTower");
        let connector = Lfm2VlConnector::from_weights(
            &vlm_weights,
            "multi_modal_projector",
            downsample_factor,
            false,
            64,
            4,
        )
        .expect("build synthetic Lfm2VlConnector");

        let processor = Lfm2VlProcessor::new(
            patch_size,
            downsample_factor as usize,
            1,
            16,
            Lfm2VlTilingPolicy {
                do_image_splitting: false,
                ..Lfm2VlTilingPolicy::default()
            },
            image::imageops::FilterType::Triangle,
        );

        Lfm2VlModel {
            text_model,
            vision_tower,
            connector,
            processor,
            image_token_id: 396,
            image_start_id: 498,
            image_end_id: 499,
            img_row_col_ids: [[0; 10]; 10],
            img_thumbnail_id: 0,
            use_image_special_tokens: false,
            downsample_factor,
            patch_dim: patch_dim as i32,
            eos_token_ids: vec![2],
        }
    }

    fn extract_f32(arr: &MlxArray) -> Vec<f32> {
        let f = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&f);
        mlxcel_core::array_to_raw_bytes(&f)
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn lfm2_vl_delegates_native_batching_contract_to_text_tower() {
        let _guard = crate::models::embedding_test_support::mlx_test_guard();
        let model = build_synthetic_lfm2_vl();
        let reference = build_synthetic_lfm2_vl();
        assert!(model.supports_batching());
        assert_eq!(
            model.batched_decode_kind(),
            model.text_model.batched_decode_kind()
        );
        assert_eq!(
            model.sequence_state_layout(),
            model.text_model.sequence_state_layout()
        );

        let ids = [SequenceId::from_raw(257_301), SequenceId::from_raw(257_302)];
        for id in ids {
            model.prepare_sequence_state(id);
            reference.prepare_sequence_state(id);
        }
        let input = mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]);
        let mut first = model.make_caches();
        let mut second = model.make_caches();
        let mut reference_first = reference.make_caches();
        let mut reference_second = reference.make_caches();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                model.forward_batched(&input, &mut [&mut first, &mut second], None)
            }))
            .is_err(),
            "LFM2-VL non-ID batching must retain the text tower's fail-closed contract"
        );
        let reference_logits = reference.forward_batched_with_ids(
            &input,
            Some(&ids),
            &mut [&mut reference_first, &mut reference_second],
            None,
        );
        let logits = model.forward_batched_with_ids(
            &input,
            Some(&ids),
            &mut [&mut first, &mut second],
            None,
        );
        mlxcel_core::eval(&logits);
        assert_eq!(mlxcel_core::array_shape(&logits), vec![2, 1, 512]);
        assert!(
            mlxcel_core::item_bool(&mlxcel_core::allclose(&logits, &reference_logits, 0.0, 0.0,)),
            "rejected LFM2-VL non-ID batching must not mutate text-tower state"
        );
        for id in ids {
            model.release_sequence_state_by_id(id);
            reference.release_sequence_state_by_id(id);
        }
    }

    #[test]
    fn cache_miss_populates_cache_and_matches_uncached_embeddings() {
        let model = build_synthetic_lfm2_vl();
        // One 2x2 patch grid -> ceil(2/2)*ceil(2/2) = 1 compressed image token.
        let layouts = vec![Lfm2VlImageLayout {
            views: vec![(2, 2)],
            rows: 1,
            cols: 1,
        }];
        let patch_vals: Vec<f32> = (0..4 * 12).map(|i| (i as f32) * 0.02).collect();
        let pixel_values = mlxcel_core::from_slice_f32(&patch_vals, &[1, 4, 12]);
        let input_ids = mlxcel_core::from_slice_i32(&[1, 396, 2], &[1, 3]);

        let uncached = model.get_input_embeddings(&input_ids, &pixel_values, &layouts);
        let cache = Mutex::new(VisionFeatureCache::new(4));
        let keys = vec![Some(CacheKey::from_path("img0.png"))];

        let miss_out = model.get_input_embeddings_with_cache(
            &input_ids,
            &pixel_values,
            &layouts,
            Some(&keys),
            Some(&cache),
        );
        assert_eq!(cache.lock().unwrap().len(), 1);
        assert_eq!(
            extract_f32(&uncached.inputs_embeds),
            extract_f32(&miss_out.inputs_embeds)
        );

        // Second call hits the cache and produces bit-identical output even if
        // pixel_values is replaced with zeros.
        let zero_pixels = mlxcel_core::zeros(&[1, 4, 12], mlxcel_core::dtype::FLOAT32);
        let hit_out = model.get_input_embeddings_with_cache(
            &input_ids,
            &zero_pixels,
            &layouts,
            Some(&keys),
            Some(&cache),
        );
        assert_eq!(
            extract_f32(&uncached.inputs_embeds),
            extract_f32(&hit_out.inputs_embeds)
        );
    }

    #[test]
    fn partial_multi_image_cache_hit_reuses_first_image_and_encodes_second() {
        let model = build_synthetic_lfm2_vl();
        // Two images, each a 2x2 patch grid (1 compressed token each).
        let layouts = vec![
            Lfm2VlImageLayout {
                views: vec![(2, 2)],
                rows: 1,
                cols: 1,
            },
            Lfm2VlImageLayout {
                views: vec![(2, 2)],
                rows: 1,
                cols: 1,
            },
        ];
        let patch_vals: Vec<f32> = (0..8 * 12).map(|i| (i as f32) * 0.015).collect();
        let pixel_values = mlxcel_core::from_slice_f32(&patch_vals, &[1, 8, 12]);
        let input_ids = mlxcel_core::from_slice_i32(&[1, 396, 10, 396, 2], &[1, 5]);

        let cache = Mutex::new(VisionFeatureCache::new(4));
        let key0 = CacheKey::from_path("first.png");
        let key1 = CacheKey::from_path("second.png");

        // Pre-populate key0 with a sentinel [1, 8] feature vector of 42.0 while
        // leaving key1 uncached.
        let sentinel = mlxcel_core::from_slice_f32(&[42.0f32; 8], &[1, 8]);
        cache
            .lock()
            .unwrap()
            .put(key0.clone(), &SingleArrayFeatures::new(sentinel));

        let keys = vec![Some(key0), Some(key1.clone())];
        let out = model.get_input_embeddings_with_cache(
            &input_ids,
            &pixel_values,
            &layouts,
            Some(&keys),
            Some(&cache),
        );

        // Both images must now reside in the cache.
        assert_eq!(cache.lock().unwrap().len(), 2);

        // Verify token position 1 (first <image>) received the cached sentinel
        // (42.0), while token position 3 (second <image>) received the freshly
        // encoded second image's features (matching an uncached run of image 1).
        let out_vals = extract_f32(&out.inputs_embeds);
        assert_eq!(&out_vals[8..16], &[42.0f32; 8]);

        let second_patches = mlxcel_core::slice(&pixel_values, &[0, 4, 0], &[1, 8, 12]);
        let single_ids = mlxcel_core::from_slice_i32(&[1, 396, 2], &[1, 3]);
        let second_only = model.get_input_embeddings(&single_ids, &second_patches, &layouts[1..2]);
        let second_vals = extract_f32(&second_only.inputs_embeds);
        assert_eq!(&out_vals[24..32], &second_vals[8..16]);
    }

    #[test]
    fn multi_view_tiled_layout_caches_concatenated_views_and_advances_offset() {
        let model = build_synthetic_lfm2_vl();
        // Image 0 has two 2x2 views (2 compressed tokens total); image 1 has one
        // 2x2 view (1 compressed token).
        let layouts = vec![
            Lfm2VlImageLayout {
                views: vec![(2, 2), (2, 2)],
                rows: 1,
                cols: 2,
            },
            Lfm2VlImageLayout {
                views: vec![(2, 2)],
                rows: 1,
                cols: 1,
            },
        ];
        let patch_vals: Vec<f32> = (0..12 * 12).map(|i| (i as f32) * 0.01).collect();
        let pixel_values = mlxcel_core::from_slice_f32(&patch_vals, &[1, 12, 12]);
        let input_ids = mlxcel_core::from_slice_i32(&[1, 396, 396, 10, 396, 2], &[1, 6]);

        let uncached = model.get_input_embeddings(&input_ids, &pixel_values, &layouts);

        // First warm only the tiled image (image 0) in the cache.
        let cache = Mutex::new(VisionFeatureCache::new(4));
        let key0 = CacheKey::from_path("tiled.png");
        let key1 = CacheKey::from_path("single.png");
        let tiled_ids = mlxcel_core::from_slice_i32(&[1, 396, 396, 2], &[1, 4]);
        let tiled_patches = mlxcel_core::slice(&pixel_values, &[0, 0, 0], &[1, 8, 12]);
        model.get_input_embeddings_with_cache(
            &tiled_ids,
            &tiled_patches,
            &layouts[0..1],
            Some(&[Some(key0.clone())]),
            Some(&cache),
        );

        // Now run both images together: image 0 hits the cache (skipping 8
        // patches across 2 views) and image 1 must slice starting at offset 8.
        let combined = model.get_input_embeddings_with_cache(
            &input_ids,
            &pixel_values,
            &layouts,
            Some(&[Some(key0), Some(key1)]),
            Some(&cache),
        );
        assert_eq!(
            extract_f32(&uncached.inputs_embeds),
            extract_f32(&combined.inputs_embeds)
        );
    }

    #[test]
    fn mismatched_cached_row_count_is_ignored_and_overwritten() {
        let model = build_synthetic_lfm2_vl();
        let layouts = vec![Lfm2VlImageLayout {
            views: vec![(2, 2)],
            rows: 1,
            cols: 1,
        }];
        let patch_vals: Vec<f32> = (0..4 * 12).map(|i| (i as f32) * 0.03).collect();
        let pixel_values = mlxcel_core::from_slice_f32(&patch_vals, &[1, 4, 12]);
        let input_ids = mlxcel_core::from_slice_i32(&[1, 396, 2], &[1, 3]);

        let expected = model.get_input_embeddings(&input_ids, &pixel_values, &layouts);

        // Seed cache with a stale 3-row tensor under the same key (layout expects 1 row).
        let cache = Mutex::new(VisionFeatureCache::new(4));
        let key = CacheKey::from_path("resized_in_place.png");
        let stale = mlxcel_core::from_slice_f32(&[99.0f32; 24], &[3, 8]);
        cache
            .lock()
            .unwrap()
            .put(key.clone(), &SingleArrayFeatures::new(stale));

        let out = model.get_input_embeddings_with_cache(
            &input_ids,
            &pixel_values,
            &layouts,
            Some(&[Some(key.clone())]),
            Some(&cache),
        );
        assert_eq!(
            extract_f32(&expected.inputs_embeds),
            extract_f32(&out.inputs_embeds)
        );
        let refreshed = cache.lock().unwrap().get(&key).expect("refreshed entry");
        assert_eq!(mlxcel_core::array_shape(&refreshed.features), vec![1, 8]);
    }

    #[test]
    fn vlm_runtime_derives_distinct_per_image_hash_keys_for_lfm2_vl() {
        let loaded = LoadedModel::Lfm2VL(build_synthetic_lfm2_vl());
        let caches = ModelVisionCaches::new(8);

        let img_a = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([20, 40, 60])));
        let img_b = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([200, 100, 50])));

        let mut tokens = vec![1, 396, 396, 2];
        let first = prepare_vlm_embeddings_with_cache(
            &loaded,
            &mut tokens,
            "<image><image>",
            &[img_a.clone(), img_b.clone()],
            None,
            Some(&caches),
            None,
            |_, _| vec![],
        )
        .expect("prepare first multi-image turn")
        .expect("embeddings present");

        // Two distinct images with no explicit keys should populate two distinct
        // hash entries in the single-array vision cache.
        assert_eq!(caches.single.lock().unwrap().len(), 2);

        let mut tokens_repeat = vec![1, 396, 396, 2];
        let second = prepare_vlm_embeddings_with_cache(
            &loaded,
            &mut tokens_repeat,
            "<image><image>",
            &[img_a, img_b],
            None,
            Some(&caches),
            None,
            |_, _| vec![],
        )
        .expect("prepare second multi-image turn")
        .expect("embeddings present");

        assert_eq!(
            extract_f32(&first.embeddings.inputs_embeds),
            extract_f32(&second.embeddings.inputs_embeds)
        );
    }
}
