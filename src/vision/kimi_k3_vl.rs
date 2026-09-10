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

//! Kimi K3 vision-language model (`KimiK3ForConditionalGeneration`).
//!
//! Reference: https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_k3.py
//! (issue #1342).
//!
//! Composition:
//! - `vision`: [`MoonViT3DVisionModel`] (`vision::encoders::moonvit3d`),
//!   emitting `sd2_tpool`-merged tokens `[M, 4, 1024]`.
//! - `projector`: [`KimiK3Projector`] (`patchmergerv2`): flatten to
//!   `[M, 4096]` -> `Linear(4096, 4096)` (no bias) -> GELU (erf) ->
//!   `Linear(4096, 7168)` (no bias) -> `RMSNorm(7168, eps 1e-5)`.
//! - `text`: [`KimiK3Model`] (`models::kimi_k3`), the KDA / MLA / latent MoE
//!   backbone, reused verbatim with an embeddings-injection forward.
//!
//! The projected rows replace, in order, the positions where
//! `input_ids == media_placeholder_token_id` (`<|media_pad|>`, 163605); the
//! count must equal the number of rows or the request is refused, because the
//! LLaVA-style scatter would otherwise silently drop or misalign features.
//!
//! Used by: `loading::vlm_kimi_k3`, `multimodal::kimi_k3_prompt`,
//! `multimodal::vlm_runtime` (`VlmRuntimeRef::KimiK3`).

use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::KimiK3Model;
use crate::vision::encoders::moonvit3d::{MoonViT3DConfig, MoonViT3DGrid, MoonViT3DVisionModel};
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::kimi_k3::{KimiK3ImageProcessor, KimiK3PreparedImage};

/// The ids of the four media control tokens the image prompt is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KimiK3MediaTokenIds {
    /// `<|media_begin|>`
    pub begin: i32,
    /// `<|media_content|>`
    pub content: i32,
    /// `<|media_pad|>`; equals `media_placeholder_token_id`.
    pub pad: i32,
    /// `<|media_end|>`
    pub end: i32,
}

/// The `patchmergerv2` projector.
pub struct KimiK3Projector {
    proj_0: UnifiedLinear,
    proj_2: UnifiedLinear,
    post_norm: RMSNorm,
    merged_hidden: i32,
}

impl KimiK3Projector {
    /// Build from `{prefix}.proj.0`, `{prefix}.proj.2`, `{prefix}.post_norm`.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        merged_hidden: usize,
        post_norm_eps: f32,
    ) -> Result<Self, String> {
        let proj_0 = UnifiedLinear::from_weights(weights, &format!("{prefix}.proj.0"), 64, 4)?;
        let proj_2 = UnifiedLinear::from_weights(weights, &format!("{prefix}.proj.2"), 64, 4)?;
        let key = format!("{prefix}.post_norm.weight");
        let norm_weight = weights
            .get(&key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {key}"))?;
        Ok(Self {
            proj_0,
            proj_2,
            post_norm: RMSNorm::new(norm_weight, post_norm_eps),
            merged_hidden: merged_hidden as i32,
        })
    }

    /// `merged`: `[M, kh*kw, vision_hidden]`. Returns `[M, text_hidden]`.
    pub fn forward(&self, merged: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(merged);
        // The rank check comes first: `shape[0]` on a rank-0 array panics,
        // and a caller that hands the projector the wrong array deserves an
        // error rather than an abort.
        let width: i32 = shape.iter().skip(1).product();
        if shape.len() < 2 || width != self.merged_hidden {
            return Err(format!(
                "kimi_k3 projector: expected [M, ..] flattening to {} per row, got {shape:?}",
                self.merged_hidden
            ));
        }
        let rows = shape[0];
        let x = mlxcel_core::reshape(merged, &[rows, self.merged_hidden]);
        let x = self.proj_0.forward(&x);
        let x = mlxcel_core::gelu(&x);
        let x = self.proj_2.forward(&x);
        Ok(self.post_norm.forward(&x))
    }
}

/// Replace the `placeholder_id` positions of `inputs_embeds` with the rows of
/// `features` (`[M, text_hidden]`), in order.
///
/// Refuses a count mismatch instead of scattering: `merge_llava` pairs the
/// `k`-th placeholder with the `k`-th row, so fewer placeholders than rows
/// would drop image content and more would leave placeholder embeddings in
/// the stream, both silently.
pub fn merge_media_features(
    placeholder_id: i32,
    features: &MlxArray,
    inputs_embeds: &MlxArray,
    input_ids: &MlxArray,
) -> Result<InputEmbeddings, String> {
    let rows = mlxcel_core::array_shape(features)[0];
    let placeholders = count_placeholders(input_ids, placeholder_id);
    if rows != placeholders {
        return Err(format!(
            "kimi_k3: the prompt carries {placeholders} <|media_pad|> ({placeholder_id}) \
             positions but the vision tower produced {rows} feature rows; every image needs \
             exactly grid_h * grid_w / 4 placeholders"
        ));
    }
    Ok(merge::merge_llava(
        placeholder_id,
        features,
        inputs_embeds,
        input_ids,
    ))
}

fn count_placeholders(input_ids: &MlxArray, placeholder_id: i32) -> i32 {
    let ids = mlxcel_core::astype(input_ids, mlxcel_core::dtype::INT32);
    let target = mlxcel_core::full_f32(&[1], placeholder_id as f32, mlxcel_core::dtype::INT32);
    let hits = mlxcel_core::equal(&ids, &target);
    let hits = mlxcel_core::astype(&hits, mlxcel_core::dtype::INT32);
    let total = mlxcel_core::sum_all(&hits);
    mlxcel_core::eval(&total);
    mlxcel_core::item_i32(&total)
}

/// Bring the raw `vision_tower.*` / `mm_projector.*` keys into the layout the
/// tower reads: the PyTorch `[out, in, kh, kw]` patch-embed kernel becomes
/// MLX's channel-last `[out, kh, kw, in]`. Every other key is kept as-is.
/// Idempotent.
pub fn sanitize_kimi_k3_vision_weights(weights: WeightMap) -> WeightMap {
    let mut out = WeightMap::with_capacity(weights.len());
    for (key, value) in weights {
        if key.ends_with("patch_embed.proj.weight") {
            let shape = mlxcel_core::array_shape(&value);
            if shape.len() == 4 && !crate::loading::conv2d_weight_is_channel_last(&shape) {
                let transposed = mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1]);
                out.insert(key, mlxcel_core::copy(&transposed));
                continue;
            }
        }
        out.insert(key, value);
    }
    out
}

/// Top-level Kimi K3 VLM runtime.
pub struct KimiK3VLModel {
    pub text: KimiK3Model,
    pub vision: MoonViT3DVisionModel,
    pub projector: KimiK3Projector,
    pub processor: KimiK3ImageProcessor,
    /// `media_placeholder_token_id` (`<|media_pad|>`, 163605).
    pub media_placeholder_token_id: i32,
    /// The four media control ids the CLI splices an image block from.
    pub media_token_ids: KimiK3MediaTokenIds,
}

impl KimiK3VLModel {
    /// Build the projector for `config` under `mm_projector`.
    pub fn projector_from_weights(
        weights: &WeightMap,
        config: &MoonViT3DConfig,
    ) -> Result<KimiK3Projector, String> {
        KimiK3Projector::from_weights(
            weights,
            "mm_projector",
            config.merged_hidden(),
            config.projector_ln_eps,
        )
    }

    /// Tower + merge + projector: `pixel_values` `[N, 3, p, p]` (the
    /// processor's channels-first layout) to `[M, text_hidden]`.
    ///
    /// The whole-batch form. The request path uses
    /// [`Self::project_image_stream`], which never holds more than one
    /// image's pixels and activations at a time.
    pub fn project_images(
        &self,
        pixel_values: &MlxArray,
        grids: &[MoonViT3DGrid],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let dtype = self.text.activation_dtype();
        let pv = mlxcel_core::astype(pixel_values, dtype);
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);
        let merged = self.vision.forward_merged(&pv, grids)?;
        self.projector.forward(&merged)
    }

    /// Preprocess, run the tower and project one image at a time, returning
    /// the `[sum(M_i), text_hidden]` rows in image order.
    ///
    /// Attention is per image, so the batch tensor a whole-request path would
    /// build is sliced back apart inside the tower anyway; building it costs
    /// one host f32 buffer plus its device copy for every image at once, and
    /// with 16 images at the navit ceiling that is gigabytes before a single
    /// block runs. Each image is evaluated before the next one starts so its
    /// tower activations are released rather than accumulating in one lazy
    /// graph.
    ///
    /// `planned` must be [`KimiK3ImageProcessor::plan_images`] over the same
    /// images, in the same order: the caller has already sized the prompt
    /// from it, so a geometry that disagrees here is refused rather than
    /// projected into placeholders that no longer match.
    pub fn project_image_stream(
        &self,
        images: &[image::DynamicImage],
        planned: &[KimiK3PreparedImage],
    ) -> Result<UniquePtr<MlxArray>, String> {
        if images.len() != planned.len() {
            return Err(format!(
                "kimi_k3: {} image(s) against {} planned geometries",
                images.len(),
                planned.len()
            ));
        }
        let dtype = self.text.activation_dtype();
        let merge = self.vision.merge_kernel();
        let mut projected = Vec::with_capacity(images.len());
        for (index, (image, plan)) in images.iter().zip(planned).enumerate() {
            let (pixels, item) = self.processor.prepare_image_array(image, dtype)?;
            if item.grid != plan.grid {
                return Err(format!(
                    "kimi_k3: image {index} preprocessed to grid {:?} but the prompt was sized \
                     from {:?}",
                    item.grid, plan.grid
                ));
            }
            let features = self.vision.forward(&pixels, &[item.grid])?;
            let merged =
                crate::vision::encoders::moonvit3d::tpool_merge(&features[0], item.grid, merge)?;
            let rows = self.projector.forward(&merged)?;
            mlxcel_core::eval(&rows);
            projected.push(rows);
        }
        let refs: Vec<&MlxArray> = projected.iter().map(|r| r.as_ref().unwrap()).collect();
        if refs.len() == 1 {
            return Ok(mlxcel_core::copy(refs[0]));
        }
        Ok(mlxcel_core::concatenate_many(&refs, 0))
    }

    /// Merged input embeddings for a request that carries images: the text
    /// embeddings with the projected rows scattered over the
    /// `media_placeholder_token_id` positions.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grids: &[MoonViT3DGrid],
    ) -> Result<InputEmbeddings, String> {
        let inputs_embeds = self.text.embed_tokens.forward(input_ids);
        let features = self.project_images(pixel_values, grids)?;
        merge_media_features(
            self.media_placeholder_token_id,
            &features,
            &inputs_embeds,
            input_ids,
        )
    }
}

impl LanguageModel for KimiK3VLModel {
    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(&self.text)
    }

    fn supports_padded_prefill(&self) -> bool {
        LanguageModel::supports_padded_prefill(&self.text)
    }

    fn supports_batching(&self) -> bool {
        LanguageModel::supports_batching(&self.text)
    }

    fn supports_chunked_prefill(&self) -> bool {
        LanguageModel::supports_chunked_prefill(&self.text)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        LanguageModel::eos_token_ids(&self.text)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(&self.text)
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward(&self.text, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text
            .forward_for_sequence_with_embeddings(input_ids, input_embeddings, None)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_sequence_id(&self.text, input_ids, seq_id, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text
            .forward_for_sequence_with_embeddings(input_ids, input_embeddings, seq_id)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text.embed_tokens.forward(input_ids))
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        LanguageModel::sequence_state_layout(&self.text)
    }

    fn reset_runtime_state(&self) {
        LanguageModel::reset_runtime_state(&self.text);
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        LanguageModel::prepare_sequence_state(&self.text, seq_id);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        LanguageModel::release_sequence_state_by_id(&self.text, seq_id);
    }
}

#[cfg(test)]
#[path = "kimi_k3_vl_tests.rs"]
mod tests;
