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

//! Youtu-VL Vision-Language Model.
//!
//! Faithful port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/youtu_vl/youtu_vl.py.
//!
//! Composition:
//! - `vision_tower` — `YoutuVLVisionEncoder` (SigLIP2-style with windowed
//!   attention, 2D vision RoPE, and a built-in patch merger that already
//!   projects into the language model hidden size).
//! - `processor`  — `YoutuVLProcessor` (flattened-patch layout +
//!   `spatial_shapes`).
//! - `language_model` — `YoutuLanguageModel` (DeepSeek-V3-style MLA backbone).
//!
//! Vision/text fusion: image features replace `image_token_id` (or, when no
//! image token appears, `video_token_id`) positions in the text embedding
//! stream — exactly the upstream `merge_input_ids_with_image_features`
//! semantics. We reuse [`crate::vision::merge::merge_llava`] for the
//! placeholder-replacement step because the merger has already projected
//! image features into the language hidden size, so no additional connector
//! is needed.
//!
//! Used by: `loading::load_youtu_vl_vlm`, `multimodal::vlm_runtime`.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::youtu_vl_lm::YoutuLanguageModel;
use crate::vision::encoders::youtu_vl::YoutuVLVisionEncoder;
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::youtu_vl::YoutuVLProcessor;

/// Top-level Youtu-VL VLM runtime.
pub struct YoutuVLModel {
    pub text_model: YoutuLanguageModel,
    pub vision_encoder: YoutuVLVisionEncoder,
    pub processor: YoutuVLProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub vision_end_token_id: i32,
    pub spatial_merge_size: usize,
}

impl YoutuVLModel {
    /// Compute merged input embeddings for a request that carries pixel
    /// values. Mirrors `Model.get_input_embeddings` /
    /// `Model.merge_input_ids_with_image_features` in upstream `youtu_vl.py`.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        spatial_shapes: &[(i32, i32)],
    ) -> Result<InputEmbeddings, String> {
        // Text embeddings.
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        // Match the dtype of the patch embedding's weight when feeding pixel
        // values into the vision tower (mirrors upstream's
        // `pixel_values.astype(dtype)` step).
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);

        let vision_output = self
            .vision_encoder
            .forward_with_spatial(&pv, spatial_shapes);

        // Replace image tokens with vision features. Upstream falls back to
        // the video token when there are no image tokens; we do the same by
        // first checking which token id is present in the prompt.
        let (target_token_id, placeholders) = self.choose_target_token_id(input_ids);

        // The scatter below fills placeholders with features in order and
        // silently drops whatever does not fit. Upstream raises when the two
        // counts differ; do the same so a prompt that was not expanded to one
        // token per merged feature (#1618) fails instead of describing only
        // the image's first patch.
        let features = vision_output.hidden_states.as_ref().unwrap();
        let feature_rows = mlxcel_core::array_shape(features)[0];
        if placeholders != feature_rows {
            return Err(format!(
                "Youtu-VL image features and image tokens do not match: {placeholders} placeholder token(s) for {feature_rows} feature(s)"
            ));
        }

        Ok(merge::merge_llava(
            target_token_id,
            features,
            &inputs_embeds,
            input_ids,
        ))
    }

    /// Pick the placeholder token id to merge against. Returns
    /// `image_token_id` when any image token appears in the prompt; otherwise
    /// falls back to `video_token_id`, together with how many positions carry
    /// it. This mirrors the upstream behaviour where the merge silently
    /// switches placeholder kinds when the caller used a video token instead
    /// of an image token.
    fn choose_target_token_id(&self, input_ids: &MlxArray) -> (i32, i32) {
        let images = Self::count_token(input_ids, self.image_token_id);
        if images > 0 {
            (self.image_token_id, images)
        } else {
            (
                self.video_token_id,
                Self::count_token(input_ids, self.video_token_id),
            )
        }
    }

    /// Number of positions in `input_ids` equal to `token_id`, read back to
    /// the host so dispatch stays a plain Rust branch.
    fn count_token(input_ids: &MlxArray, token_id: i32) -> i32 {
        let target =
            mlxcel_core::full_f32(&[1], token_id as f32, mlxcel_core::array_dtype(input_ids));
        let cmp = mlxcel_core::equal(input_ids, &target);
        // `equal` returns a bool array; cast to int32 so `sum_axis` accumulates
        // a numeric count (mlx does not currently sum bool tensors directly).
        let cmp_int = mlxcel_core::astype(&cmp, mlxcel_core::dtype::INT32);
        let flat = mlxcel_core::flatten(&cmp_int);
        let count = mlxcel_core::sum_axis(&flat, 0, false);
        mlxcel_core::eval(&count);
        mlxcel_core::item_i32(&count)
    }
}

// LanguageModel trait — text-only forward paths delegate straight to the
// underlying `YoutuLanguageModel`, including the `forward_with_embeddings`
// path used by the VLM runtime to inject pre-merged inputs.
impl LanguageModel for YoutuVLModel {
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
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
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
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
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
        self.text_model.embed_tokens(input_ids)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.text_model.eos_token_ids()
    }
}
