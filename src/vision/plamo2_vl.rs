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

//! PLaMo 2.1 VL (`model_type: "plamo2vl"`) vision-language model (issue #1380).
//!
//! Composition (`pfnet/plamo-2.1-2b-vl`, `pfnet/plamo-2.1-8b-vl`):
//! - `vision_model.vision_encoder.model`: a plain SigLIP-so400m tower (384px,
//!   patch 14, 27 layers, tanh GELU) run once over every 384x384 tile; the
//!   post-layernorm output `[tiles, 729, 1152]` is used as is (no CLS, no
//!   attention-pooling head).
//! - `image_proj`: the RMSNorm / bias / Linear / GELU / Linear adapter
//!   ([`Plamo2VlImageProjector`]) into the text hidden size.
//! - The PLaMo 2 Mamba + attention decoder ([`Plamo2Model`]) with
//!   per-layer RoPE bases from the VL text config.
//!
//! The projected rows are scattered, in order, onto the `<plamo:image>`
//! (100002) positions of the prompt via [`merge::merge_llava`], as upstream
//! `_merge_input_ids_with_image_features` does. Every other `LanguageModel`
//! hook delegates to the text model, which owns the mixed Mamba / KV state.
//!
//! Reference: mlx-vlm `mlx_vlm/models/plamo2vl/plamo2vl.py` and the
//! checkpoint's `modeling_plamo2_vl.py`.
//!
//! Used by: `loading::load_plamo2_vl`, `multimodal::vlm_runtime`.

use mlxcel_core::cache::{CachePool, KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::Plamo2Model;
use crate::vision::connectors::plamo2_vl::Plamo2VlImageProjector;
use crate::vision::encoders::VisionEncoder;
use crate::vision::encoders::siglip::SigLipVisionModel;
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::plamo2_vl::Plamo2VlProcessor;

/// `<siglip2:img>`, the image-block opener.
pub const PLAMO2_VL_IMG_START_TOKEN_ID: i32 = 100_000;
/// `</siglip2:img>`, the image-block closer.
pub const PLAMO2_VL_IMG_END_TOKEN_ID: i32 = 100_001;
/// `<plamo:image>`, the per-feature placeholder.
pub const PLAMO2_VL_IMAGE_TOKEN_ID: i32 = 100_002;
/// `<|plamo:bos|>`: prepended when the checkpoint's tokenizer config sets
/// `add_bos_token` (the HF export does; `PlamoTokenizer` defaults it to false,
/// and the mlx-vlm conversions drop the key), and also the checkpoint's EOS.
pub const PLAMO2_VL_BOS_TOKEN_ID: i32 = 1;

pub struct Plamo2VlModel {
    pub text_model: Plamo2Model,
    pub vision_model: SigLipVisionModel,
    pub projector: Plamo2VlImageProjector,
    pub processor: Plamo2VlProcessor,
    /// Dtype the tower's weights were loaded in; pixels are cast to it.
    pub vision_dtype: i32,
    pub image_token_id: i32,
    pub img_start_token_id: i32,
    pub img_end_token_id: i32,
    /// `Some(<|plamo:bos|>)` when the tokenizer config sets `add_bos_token`.
    pub bos_token_id: Option<i32>,
    /// Feature rows per tile (729).
    pub num_image_token: usize,
    pub eos_token_ids: Vec<i32>,
}

impl Plamo2VlModel {
    /// Merged prompt embeddings for `input_ids` `[1, T]` and the processor's
    /// `pixel_values` `[tiles, 3, 384, 384]`. The caller has already checked
    /// that the prompt carries `tiles * 729` `<plamo:image>` tokens.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> Result<InputEmbeddings, String> {
        let inputs_embeds = LanguageModel::embed_tokens(&self.text_model, input_ids)
            .ok_or_else(|| "PLaMo 2 text model returned no token embeddings".to_string())?;

        // NCHW -> NHWC for the conv patch embedding, in the tower's dtype.
        let pixels = mlxcel_core::astype(pixel_values, self.vision_dtype);
        let pixels = mlxcel_core::transpose_axes(&pixels, &[0, 2, 3, 1]);
        let hidden = self.vision_model.forward(&pixels).hidden_states;

        // [tiles, 729, 1152] -> [tiles * 729, 1152] -> [tiles * 729, hidden].
        let shape = mlxcel_core::array_shape(&hidden);
        let width = *shape.last().ok_or("empty SigLIP output shape")?;
        let flat = mlxcel_core::reshape(&hidden, &[-1, width]);
        let features = self.projector.forward(&flat);

        Ok(merge::merge_llava(
            self.image_token_id,
            &features,
            &inputs_embeds,
            input_ids,
        ))
    }
}

impl LanguageModel for Plamo2VlModel {
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

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        // Both single-sequence forwards delegate to the text model unchanged.
        Some(mlxcel_core::generate::last_row_logits_from(
            &self.text_model,
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
            last_pos,
        ))
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

    /// The image framing ids exist only to align the prompt; sampling one
    /// would corrupt the text stream (issue #350).
    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        vec![
            self.img_start_token_id,
            self.img_end_token_id,
            self.image_token_id,
        ]
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

    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        self.text_model.kv_cache_storage_modes()
    }

    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(&self.text_model)
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

    fn trim_state(&self, seq: Option<SequenceId>, excess: i32) -> Result<(), String> {
        self.text_model.trim_state(seq, excess)
    }
}
