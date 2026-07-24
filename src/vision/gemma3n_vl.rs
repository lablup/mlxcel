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

//! Gemma 3n Vision-Language Model
//!
//! MobileNetV5 vision encoder + Gemma3n language model

use super::{
    encoders, gemma3n_per_layer_inputs_state::Gemma3nPerLayerInputsState, merge, processors,
};
use crate::LanguageModel;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Gemma 3n VLM: MobileNetV5 vision encoder + Gemma3n language model
///
/// Unlike ViT-based VLMs, Gemma 3n uses a convolutional MobileNetV5 encoder
/// with Multi-Scale Fusion Adapter. The language model has a unique
/// `per_layer_inputs` mechanism that requires special handling: the
/// projected tensor is produced during `get_input_embeddings` and consumed
/// later by the prefill `forward_with_embeddings_and_sequence_id` call.
///
/// Issue #85: the `per_layer_inputs` tensor used to live in a single
/// `RefCell<Option<UniquePtr<MlxArray>>>` field on this struct. Under
/// concurrent VLM requests on `mlxcel-server`, two prepares could overwrite
/// the slot before either prefill consumed it, leaking one row's tensor
/// into another row's prefill (or panicking on `Option::unwrap()` when the
/// slot was already drained). The state is now held in
/// [`Gemma3nPerLayerInputsState`], a per-`SequenceId` map with a fallback
/// slot for legacy single-instance callers, mirroring the Gemma 4
/// [`Gemma4PerLayerInputsState`](crate::vision::gemma4_per_layer_inputs_state::Gemma4PerLayerInputsState)
/// container.
pub struct Gemma3nVLModel {
    pub text_model: crate::models::Gemma3nModel,
    pub vision_tower: encoders::gemma3n::Gemma3nVisionModel,
    pub embed_vision: crate::models::gemma3n::Gemma3nMultimodalEmbedder,
    pub processor: processors::siglip::SigLipProcessor, // 224x224
    pub image_token_id: i32,                            // 262_145 (<image_soft_token>)
    pub boi_token_id: i32,                              // 255_999 (<start_of_image>)
    pub eoi_token_id: i32,                              // 262_144 (<end_of_image>)
    pub vision_hidden_size: usize,                      // 2048
    pub audio_tower: Option<crate::audio::gemma3n::Gemma3nAudioEncoder>,
    pub embed_audio: Option<crate::models::gemma3n::Gemma3nAudioEmbedder>,
    pub audio_token_id: i32,
    pub boa_token_id: i32,
    pub eoa_token_id: i32,
    pub audio_soft_tokens_per_clip: usize,
    pub audio_preprocess_policy: crate::audio::AudioFamilyPolicy,
    /// Per-sequence storage for the projected `per_layer_inputs` tensor
    /// that is produced during embedding prep and consumed during the
    /// prefill `forward_with_embeddings_and_sequence_id` call. See module
    /// [`crate::vision::gemma3n_per_layer_inputs_state`] for the lifecycle.
    per_layer_inputs_state: Gemma3nPerLayerInputsState,
}

impl Gemma3nVLModel {
    pub fn new(
        text_model: crate::models::Gemma3nModel,
        vision_tower: encoders::gemma3n::Gemma3nVisionModel,
        embed_vision: crate::models::gemma3n::Gemma3nMultimodalEmbedder,
        processor: processors::siglip::SigLipProcessor,
        image_token_id: i32,
        boi_token_id: i32,
        eoi_token_id: i32,
        vision_hidden_size: usize,
    ) -> Self {
        Self {
            text_model,
            vision_tower,
            embed_vision,
            processor,
            image_token_id,
            boi_token_id,
            eoi_token_id,
            vision_hidden_size,
            audio_tower: None,
            embed_audio: None,
            audio_token_id: 262_273,
            boa_token_id: 256_000,
            eoa_token_id: 262_272,
            audio_soft_tokens_per_clip: crate::audio::gemma3n::GEMMA3N_AUDIO_SOFT_TOKENS,
            audio_preprocess_policy: crate::audio::AudioFamilyPolicy::gemma3n(),
            per_layer_inputs_state: Gemma3nPerLayerInputsState::new(),
        }
    }

    pub fn set_audio(
        &mut self,
        audio_tower: crate::audio::gemma3n::Gemma3nAudioEncoder,
        embed_audio: crate::models::gemma3n::Gemma3nAudioEmbedder,
        audio_token_id: i32,
        boa_token_id: i32,
        eoa_token_id: i32,
        audio_soft_tokens_per_clip: usize,
        audio_preprocess_policy: crate::audio::AudioFamilyPolicy,
    ) {
        self.audio_tower = Some(audio_tower);
        self.embed_audio = Some(embed_audio);
        self.audio_token_id = audio_token_id;
        self.boa_token_id = boa_token_id;
        self.eoa_token_id = eoa_token_id;
        self.audio_soft_tokens_per_clip = audio_soft_tokens_per_clip;
        self.audio_preprocess_policy = audio_preprocess_policy;
    }

    pub fn supports_audio(&self) -> bool {
        self.audio_tower.is_some() && self.embed_audio.is_some()
    }

    /// Get input embeddings with vision features merged in.
    ///
    /// Side effect: writes the freshly projected `per_layer_inputs` tensor
    /// into the per-layer-inputs state container's fallback slot. The
    /// scheduler binds it to a `SequenceId` right after
    /// `prepare_request_vlm_embeddings` returns (see issue #85). Legacy
    /// single-row callers (CLI `mlxcel generate`, `mlxcel-bench-decode`,
    /// single-row tests) consume it via `take_fallback` on the next
    /// prefill.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> merge::InputEmbeddings {
        self.get_input_embeddings_with_media(input_ids, Some(pixel_values), None, None)
            .expect("Gemma3n image embedding preparation failed")
    }

    /// Prepare text embeddings and merge any supplied image and/or audio
    /// features. Audio masks use `true = invalid`, matching the encoder.
    pub fn get_input_embeddings_with_media(
        &self,
        input_ids: &MlxArray,
        pixel_values: Option<&MlxArray>,
        audio_features: Option<&MlxArray>,
        invalid_audio_mask: Option<&MlxArray>,
    ) -> Result<merge::InputEmbeddings, String> {
        // 1. Text embeddings
        let mut inputs_embeds = self.text_model.language_model.get_embed_tokens(input_ids);
        if let Some(embed_audio) = &self.embed_audio {
            inputs_embeds = embed_audio.merge_hard_tokens(input_ids, &inputs_embeds);
        }

        // 2. Per-layer inputs (image_token_id >= vocab_size_per_layer, auto-zeroed)
        let per_layer_inputs = self
            .text_model
            .language_model
            .get_per_layer_inputs(input_ids);

        let mut merged = merge::InputEmbeddings {
            inputs_embeds,
            attention_mask_4d: None,
        };

        // 3. Optional vision path.
        if let Some(pixel_values) = pixel_values {
            let embed_dtype = mlxcel_core::array_dtype(&merged.inputs_embeds);
            let pv = mlxcel_core::astype(pixel_values, embed_dtype);
            let vision_out = self.vision_tower.forward(&pv);
            let vo = mlxcel_core::transpose_axes(&vision_out, &[0, 3, 1, 2]);
            let vo_shape = mlxcel_core::array_shape(&vo);
            let (batch, channels) = (vo_shape[0], vo_shape[1]);
            let patches = vo_shape[2] * vo_shape[3];
            let vo = mlxcel_core::reshape(&vo, &[batch, channels, patches]);
            let vo = mlxcel_core::transpose_axes(&vo, &[0, 2, 1]);
            let vo = mlxcel_core::multiply_scalar(&vo, (self.vision_hidden_size as f32).sqrt());
            let image_features = self.embed_vision.forward_soft(&vo);
            merged = merge::merge_llava(
                self.image_token_id,
                &image_features,
                &merged.inputs_embeds,
                input_ids,
            );
        }

        // 4. Optional audio tower, projection, fixed-length padding and merge.
        match (audio_features, invalid_audio_mask) {
            (Some(audio_features), Some(invalid_audio_mask)) => {
                let audio_tower = self
                    .audio_tower
                    .as_ref()
                    .ok_or_else(|| "Gemma3n checkpoint has no audio tower".to_string())?;
                let embed_audio = self
                    .embed_audio
                    .as_ref()
                    .ok_or_else(|| "Gemma3n checkpoint has no audio embedder".to_string())?;
                let padding_embedding = embed_audio.padding_embedding();
                let input_shape = mlxcel_core::array_shape(audio_features);
                let mask_shape = mlxcel_core::array_shape(invalid_audio_mask);
                if input_shape.len() != 3 || input_shape[2] != 128 || mask_shape != input_shape[..2]
                {
                    return Err(format!(
                        "Gemma3n audio input shapes must be [B,T,128] and [B,T], got {input_shape:?} and {mask_shape:?}"
                    ));
                }
                let batch_size = input_shape[0];
                let projected = if input_shape[1] == 0 {
                    // The pinned processor returns zero mel frames for clips
                    // shorter than its 513-sample unfold. The reference model
                    // consequently emits only the fixed hard-padding rows.
                    let hidden = mlxcel_core::array_shape(&padding_embedding)[2];
                    mlxcel_core::broadcast_to(
                        &padding_embedding,
                        &[batch_size, self.audio_soft_tokens_per_clip as i32, hidden],
                    )
                } else {
                    let (encoded, encoded_invalid_mask) =
                        audio_tower.forward(audio_features, invalid_audio_mask)?;
                    let mut projected = embed_audio.forward_soft(&encoded);
                    let shape = mlxcel_core::array_shape(&projected);
                    if shape[1] > self.audio_soft_tokens_per_clip as i32 {
                        return Err(format!(
                            "Gemma3n audio encoder emitted {} tokens, exceeding the fixed {}-token contract",
                            shape[1], self.audio_soft_tokens_per_clip
                        ));
                    }
                    let invalid =
                        mlxcel_core::reshape(&encoded_invalid_mask, &[shape[0], shape[1], 1]);
                    projected = mlxcel_core::where_cond(&invalid, &padding_embedding, &projected);
                    let missing = self.audio_soft_tokens_per_clip as i32 - shape[1];
                    if missing > 0 {
                        let padding = mlxcel_core::broadcast_to(
                            &padding_embedding,
                            &[shape[0], missing, shape[2]],
                        );
                        projected = mlxcel_core::concatenate(&projected, &padding, 1);
                    }
                    projected
                };

                let expected_tokens = batch_size as usize * self.audio_soft_tokens_per_clip;
                let actual_tokens = count_i32_token(input_ids, self.audio_token_id)?;
                if actual_tokens != expected_tokens {
                    return Err(format!(
                        "Gemma3n audio features and placeholders do not match: {expected_tokens} features, {actual_tokens} tokens"
                    ));
                }
                merged = merge::merge_llava(
                    self.audio_token_id,
                    &projected,
                    &merged.inputs_embeds,
                    input_ids,
                );
            }
            (None, None) => {}
            _ => {
                return Err(
                    "Gemma3n audio features and invalid mask must be provided together".into(),
                );
            }
        }

        // The official multimodal forward projects PLE after hard/soft media
        // embeddings have replaced their placeholder rows. Projecting from
        // the initial text-table rows gives image/audio positions the wrong
        // router input.
        let per_layer_inputs = self
            .text_model
            .language_model
            .project_per_layer_inputs(&merged.inputs_embeds, &per_layer_inputs);

        // 6. Park the projected per_layer_inputs in the state container's
        //    fallback slot. The scheduler's
        //    `bind_gemma3n_per_layer_inputs_to_sequence` immediately
        //    transfers it into the per-sequence map under the request's
        //    `SequenceId`. Single-row callers (bench / CLI) consume it
        //    via `take_fallback` on the next prefill.
        self.per_layer_inputs_state
            .set_fallback(Some(per_layer_inputs));

        Ok(merged)
    }

    // -- Per-sequence per_layer_inputs (issue #85) --------------------
    //
    // Thin wrappers around `Gemma3nPerLayerInputsState`. The scheduler
    // calls them via `LoadedModel::*` capability helpers (see
    // `loaded_model_capabilities.rs`) right after
    // `prepare_request_vlm_embeddings`, mirroring the Gemma 4 binding
    // flow in `gemma4_per_layer_inputs_state`.

    /// Drain the container's fallback slot into the per-`SequenceId` map
    /// under `seq_id`. No-op when the slot is empty (text-only request
    /// after a Gemma 3n VLM model load).
    pub fn bind_per_layer_inputs_to_sequence(&self, seq_id: SequenceId) {
        self.per_layer_inputs_state
            .bind_fallback_to_sequence(seq_id);
    }

    /// Drop a sequence's stored `per_layer_inputs`. Called from
    /// [`Self::release_sequence_state_by_id`] so the map drains in step
    /// with the scheduler's per-sequence cache cleanup.
    pub fn release_per_layer_inputs(&self, seq_id: SequenceId) {
        self.per_layer_inputs_state.release_sequence(seq_id);
    }

    /// Take a sequence's tensor out of the map without dropping the
    /// `UniquePtr`. Used by the scheduler's preemption path so the
    /// tensor can be carried across the eviction (which releases the old
    /// sequence id) and reinstalled under the freshly allocated id with
    /// [`Self::install_per_layer_inputs_for_sequence`].
    pub fn take_per_layer_inputs_for_sequence(
        &self,
        seq_id: SequenceId,
    ) -> Option<UniquePtr<MlxArray>> {
        self.per_layer_inputs_state.take_for_sequence(seq_id)
    }

    /// Re-install a previously taken tensor under `seq_id`. No-op when
    /// the snapshot is `None`.
    pub fn install_per_layer_inputs_for_sequence(
        &self,
        seq_id: SequenceId,
        snapshot: Option<UniquePtr<MlxArray>>,
    ) {
        if let Some(value) = snapshot {
            self.per_layer_inputs_state.bind_for_sequence(seq_id, value);
        }
    }

    /// Resolve the active per-layer tensor. A server sequence must never
    /// consume the unbound fallback slot: doing so can reuse another
    /// concurrent request's PLE state.
    fn take_per_layer_inputs(&self, seq_id: Option<SequenceId>) -> Option<UniquePtr<MlxArray>> {
        match seq_id {
            Some(id) => self.per_layer_inputs_state.take_for_sequence(id),
            None => self.per_layer_inputs_state.take_fallback(),
        }
    }

    fn align_per_layer_inputs_to_embeddings(
        per_layer_inputs: &MlxArray,
        input_embeddings: &MlxArray,
    ) -> Option<UniquePtr<MlxArray>> {
        let pli_shape = mlxcel_core::array_shape(per_layer_inputs);
        let embed_shape = mlxcel_core::array_shape(input_embeddings);
        let current_seq = pli_shape[1];
        let target_seq = embed_shape[1];

        if current_seq == target_seq {
            return None;
        }

        if current_seq > target_seq {
            return Some(mlxcel_core::slice(
                per_layer_inputs,
                &[0, 0, 0, 0],
                &[pli_shape[0], target_seq, pli_shape[2], pli_shape[3]],
            ));
        }

        let pad_rows = target_seq - current_seq;
        let dtype = mlxcel_core::array_dtype(per_layer_inputs);
        let padding =
            mlxcel_core::zeros(&[pli_shape[0], pad_rows, pli_shape[2], pli_shape[3]], dtype);
        Some(mlxcel_core::concatenate(per_layer_inputs, &padding, 1))
    }
}

fn count_i32_token(input_ids: &MlxArray, token_id: i32) -> Result<usize, String> {
    if mlxcel_core::array_dtype(input_ids) != mlxcel_core::dtype::INT32 {
        return Err("Gemma3n input ids must use int32 dtype".into());
    }
    mlxcel_core::eval(input_ids);
    let bytes = mlxcel_core::array_to_raw_bytes(input_ids);
    Ok(bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .filter(|bytes| i32::from_ne_bytes((*bytes).try_into().unwrap()) == token_id)
        .count())
}

impl LanguageModel for Gemma3nVLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.language_model.forward(input_ids, caches)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // No `seq_id` plumbed through (legacy CLI / bench / direct call).
        // Route to the per-sequence dispatch with `seq_id == None`, which
        // resolves the per_layer_inputs from the fallback slot.
        self.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            None,
            caches,
            mask,
        )
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if let Some(embeds) = input_embeddings {
            // VLM prefill: resolve the request's projected per_layer_inputs
            // from the per-`SequenceId` map (server flow) or the fallback
            // slot (CLI / bench). An absent tensor at this point is an
            // internal invariant violation: `get_input_embeddings` must
            // have populated the fallback slot before this consumer ran.
            let pli = self
                .take_per_layer_inputs(seq_id)
                .expect("gemma3n VLM prefill: per_layer_inputs missing for this sequence");
            // M5 tile-aligned prefill pads the token stream and
            // merged embeddings to an NA tile length. The projected
            // per-layer tensor is produced before that generic padding step,
            // so align it here before Gemma3n's per-layer blend.
            let aligned_pli = Self::align_per_layer_inputs_to_embeddings(&pli, embeds);
            let pli_ref = aligned_pli
                .as_ref()
                .map_or_else(|| pli.as_ref().unwrap(), |array| array.as_ref().unwrap());
            self.text_model
                .language_model
                .forward_with_inputs(embeds, pli_ref, caches)
        } else {
            self.text_model.language_model.forward(input_ids, caches)
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.language_model.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        mlxcel_core::generate::LanguageModel::eos_token_ids(&self.text_model)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        // Issue #85: drop the per-sequence `per_layer_inputs` alongside
        // the text model's per-sequence cache release so the map cannot
        // grow without bound across long-running server sessions.
        self.per_layer_inputs_state.release_sequence(seq_id);
        // The text model (Gemma3nModel) uses the trait's default no-op
        // for release_sequence_state_by_id today; the call below is kept
        // for forward-compat when the text backbone gains per-seq state.
        mlxcel_core::generate::LanguageModel::release_sequence_state_by_id(
            &self.text_model,
            seq_id,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::Gemma3nVLModel;

    #[test]
    fn align_per_layer_inputs_pads_to_embedding_sequence_length() {
        let per_layer_inputs = mlxcel_core::zeros(&[1, 273, 2, 256], mlxcel_core::dtype::FLOAT32);
        let embeddings = mlxcel_core::zeros(&[1, 288, 128], mlxcel_core::dtype::FLOAT32);

        let aligned =
            Gemma3nVLModel::align_per_layer_inputs_to_embeddings(&per_layer_inputs, &embeddings)
                .expect("expected padding for shorter per_layer_inputs");

        assert_eq!(mlxcel_core::array_shape(&aligned), vec![1, 288, 2, 256]);
    }

    #[test]
    fn align_per_layer_inputs_slices_to_embedding_sequence_length() {
        let per_layer_inputs = mlxcel_core::zeros(&[1, 288, 2, 256], mlxcel_core::dtype::FLOAT32);
        let embeddings = mlxcel_core::zeros(&[1, 273, 128], mlxcel_core::dtype::FLOAT32);

        let aligned =
            Gemma3nVLModel::align_per_layer_inputs_to_embeddings(&per_layer_inputs, &embeddings)
                .expect("expected slicing for longer per_layer_inputs");

        assert_eq!(mlxcel_core::array_shape(&aligned), vec![1, 273, 2, 256]);
    }

    #[test]
    fn align_per_layer_inputs_leaves_matching_sequence_length_untouched() {
        let per_layer_inputs = mlxcel_core::zeros(&[1, 273, 2, 256], mlxcel_core::dtype::FLOAT32);
        let embeddings = mlxcel_core::zeros(&[1, 273, 128], mlxcel_core::dtype::FLOAT32);

        let aligned =
            Gemma3nVLModel::align_per_layer_inputs_to_embeddings(&per_layer_inputs, &embeddings);

        assert!(aligned.is_none());
    }
}
