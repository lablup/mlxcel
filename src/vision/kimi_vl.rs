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

//! Kimi-VL / Kimi-VL 2.5 Vision-Language Model.
//!
//! Faithful port of the image path of upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl/kimi_vl.py
//! (and the `kimi_k25` variant).
//!
//! Composition:
//! - `vision_model` — [`KimiVLVisionModel`] (MoonViT native-resolution encoder,
//!   `vision::encoders::kimi_vl`). Produces `spatial_merge`-grouped patch
//!   features `[total_merged, kh*kw, vision_hidden]`.
//! - `projector` — [`KimiVLMultiModalProjector`]: `LayerNorm(vision_hidden) →
//!   Linear → GELU → Linear`, projecting merged patches into the text hidden
//!   size. Kimi-VL stores it under `multi_modal_projector.{pre_norm,linear_1,
//!   linear_2}`; Kimi-VL 2.5 under `mm_projector.{pre_norm,proj.0,proj.2}`.
//!   The two are numerically identical (per-token norm + MLP), so one
//!   implementation serves both — only the weight keys differ.
//! - `text_model` — [`DeepSeekV3Model`] (DeepSeek-V3-style MoE backbone). Kimi's
//!   text config is `deepseek_v3`, so the existing backbone is reused verbatim,
//!   extended only with an embeddings-injection forward for the image path.
//!
//! Vision/text fusion: the projected image features replace the
//! `media_placeholder_token_id` positions in the text embedding stream via
//! [`crate::vision::merge::merge_llava`] — exactly the upstream
//! `Model.get_input_embeddings` scatter.
//!
//! Scope: image and video. Kimi-VL 2.5's 3D MoonViT video path is handled by
//! the same tower via [`KimiMediaGrid`] media descriptors (issue #551).

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::deepseek_v3::DeepSeekV3Model;
use crate::vision::encoders::kimi_vl::{KimiMediaGrid, KimiVLVisionModel};
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::kimi_vl::KimiVLProcessor;

/// The vision-to-language projector.
///
/// `pre_norm` normalises each merged patch over the vision hidden dim; the
/// merged patch is then flattened to `vision_hidden * kh * kw` and projected
/// through `Linear → GELU → Linear` into the text hidden size.
pub struct KimiVLMultiModalProjector {
    pre_norm: LayerNorm,
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
    /// `vision_hidden * merge_h * merge_w` — the flattened merged-patch width.
    merged_hidden: i32,
}

impl KimiVLMultiModalProjector {
    /// Build from explicit weight-key prefixes so both the `kimi_vl`
    /// (`multi_modal_projector.{pre_norm,linear_1,linear_2}`) and `kimi_k25`
    /// (`mm_projector.{pre_norm,proj.0,proj.2}`) layouts are supported.
    pub fn from_weights(
        weights: &WeightMap,
        pre_norm_prefix: &str,
        linear_1_prefix: &str,
        linear_2_prefix: &str,
        merged_hidden: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight = weights
            .get(&format!("{pre_norm_prefix}.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {pre_norm_prefix}.weight"))?;
        let bias = weights
            .get(&format!("{pre_norm_prefix}.bias"))
            .map(|b| mlxcel_core::copy(b));
        // Upstream fixes the projector LayerNorm eps at 1e-5.
        let pre_norm = LayerNorm::new(weight, bias, 1e-5);

        let linear_1 = UnifiedLinear::from_weights(weights, linear_1_prefix, group_size, bits)?;
        let linear_2 = UnifiedLinear::from_weights(weights, linear_2_prefix, group_size, bits)?;

        Ok(Self {
            pre_norm,
            linear_1,
            linear_2,
            merged_hidden,
        })
    }

    /// `image_features`: `[total_merged, kh*kw, vision_hidden]`. Returns
    /// `[total_merged, text_hidden]`.
    pub fn forward(&self, image_features: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.pre_norm.forward(image_features);
        let h = mlxcel_core::reshape(&h, &[-1, self.merged_hidden]);
        let h = self.linear_1.forward(&h);
        let h = mlxcel_core::gelu(&h);
        self.linear_2.forward(&h)
    }
}

/// Top-level Kimi-VL / Kimi-VL 2.5 VLM runtime.
pub struct KimiVLModel {
    pub text_model: DeepSeekV3Model,
    pub vision_model: KimiVLVisionModel,
    pub projector: KimiVLMultiModalProjector,
    /// Native-resolution image processor (patchify + per-image grid).
    pub processor: KimiVLProcessor,
    /// Placeholder token id whose positions receive image features
    /// (`media_placeholder_token_id`, 163606 by default upstream).
    pub media_placeholder_token_id: i32,
    /// `spatial_merge_size` from the vision config; used by the runtime to
    /// expand each media placeholder into `(h/merge)*(w/merge)` tokens.
    pub spatial_merge_size: i32,
    /// EOS/stop token ids resolved from the config at load time.
    pub eos_token_ids: Vec<i32>,
}

impl KimiVLModel {
    /// Compute merged input embeddings for a request that carries pixel values.
    /// Mirrors `Model.get_input_embeddings` in upstream `kimi_vl.py`.
    ///
    /// `pixel_values`: channels-first `[total_patches, C, p, p]` (the processor's
    /// native layout), packed in media order (frame-major within each video).
    /// MoonViT is channels-last, so we transpose once here, matching the
    /// reference `pixel_values.transpose(0, 2, 3, 1)`. `media_grids` carries one
    /// [`KimiMediaGrid`] per item (image `(h, w)` or video `(t, h, w)`), in the
    /// same order the patches are concatenated.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        media_grids: &[KimiMediaGrid],
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);

        let vision_features = self.vision_model.forward_with_grid(&pv, media_grids);
        let image_features = self.projector.forward(&vision_features);

        merge::merge_llava(
            self.media_placeholder_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
    }
}

// LanguageModel — text-only forward paths delegate to the DeepSeek-V3 backbone.
// The `forward_with_embeddings` paths route through the backbone's
// embeddings-injection forward so the VLM image path is honoured (the backbone's
// own trait default would silently re-embed `input_ids`, dropping the merged
// vision features). EOS ids come from the VLM config, not the backbone default.
impl LanguageModel for KimiVLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_impl(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_impl_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_impl(input_ids, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_impl_with_embeddings(input_ids, input_embeddings, caches, mask)
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

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_batched_with_context_and_ids(
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches_impl()
    }

    fn num_layers(&self) -> usize {
        self.text_model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::KimiVLMultiModalProjector;
    use mlxcel_core::weights::WeightMap;

    fn insert(wm: &mut WeightMap, key: &str, data: &[f32], shape: &[i32]) {
        wm.insert(key.to_string(), mlxcel_core::from_slice_f32(data, shape));
    }

    #[test]
    fn projector_projects_merged_patches_to_text_hidden() {
        // vision_hidden d=4, merge 2x2 -> merged_hidden = 4*4 = 16; text_hidden = 3.
        let d = 4i32;
        let merged_hidden = 16i32;
        let text_hidden = 3i32;

        let mut wm = WeightMap::new();
        insert(&mut wm, "pn.weight", &[1.0; 4], &[d]);
        insert(&mut wm, "pn.bias", &[0.0; 4], &[d]);
        insert(
            &mut wm,
            "l1.weight",
            &[0.05; 256],
            &[merged_hidden, merged_hidden],
        );
        insert(&mut wm, "l1.bias", &[0.0; 16], &[merged_hidden]);
        insert(
            &mut wm,
            "l2.weight",
            &[0.05; 48],
            &[text_hidden, merged_hidden],
        );
        insert(&mut wm, "l2.bias", &[0.0; 3], &[text_hidden]);

        let proj =
            KimiVLMultiModalProjector::from_weights(&wm, "pn", "l1", "l2", merged_hidden, 64, 4)
                .expect("build projector");

        // image_features [total_merged=1, kh*kw=4, d=4].
        let feats =
            mlxcel_core::from_slice_f32(&(0..16).map(|i| i as f32).collect::<Vec<_>>(), &[1, 4, 4]);
        let out = proj.forward(&feats);
        mlxcel_core::eval(&out);
        assert_eq!(mlxcel_core::array_shape(&out), vec![1, text_hidden]);

        let mx = mlxcel_core::max_all(&out);
        mlxcel_core::eval(&mx);
        assert!(
            mlxcel_core::item_f32(&mx).is_finite(),
            "projector output must be finite"
        );
    }
}
