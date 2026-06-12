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

//! Nemotron H Nano Omni vision-language model wrapper.
//!
//! Faithful Rust port of the multimodal path in
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/nemotron_h_nano_omni.py.
//! Composes:
//! - the existing Nemotron-H text backbone
//!   ([`crate::models::NemotronHModel`])
//! - the RADIO vision tower
//!   ([`crate::vision::encoders::nemotron_h_nano_omni::NemotronHNanoOmniVisionModel`])
//! - the multimodal projector (`mlp1`): `RMSNorm -> Linear -> ReLU² ->
//!   Linear` with the upstream "pixel shuffle" downsample applied to
//!   the patch grid before projection
//! - **** an optional audio path:
//!   `crate::audio::nemotron_h_nano_omni::{NemotronOmniSoundEncoder,
//!   NemotronOmniSoundProjection, NemotronOmniFeatureExtractor}`,
//!   wired only when `sound_config` is present in the released
//!   checkpoint's `config.json`.
//!
//! Used by: Nemotron H Nano Omni VLM

use crate::LanguageModel;
use crate::audio::nemotron_h_nano_omni::{
    NemotronOmniAudioConfig, NemotronOmniFeatureExtractor, NemotronOmniSoundEncoder,
    NemotronOmniSoundProjection,
};
use crate::models::NemotronHModel;
use crate::vision::encoders::nemotron_h_nano_omni::NemotronHNanoOmniVisionModel;
use crate::vision::merge::InputEmbeddings;
use crate::vision::processors::nemotron_h_nano_omni::{
    NemotronHNanoOmniImageInput, NemotronHNanoOmniImageProcessor,
};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Multimodal projector configuration.
///
/// Mirrors the upstream `ModelConfig` fields that drive `VisionProjection`,
/// the `pixel_shuffle` downsample, and the optional audio path. The
/// defaults match the released 30B-A3B Nano Omni checkpoint.
#[derive(Debug, Clone)]
pub struct NemotronHNanoOmniVlConfig {
    pub vit_hidden_size: usize,
    pub projector_hidden_size: usize,
    pub text_hidden_size: usize,
    pub downsample_ratio: f32,
    pub ps_version: String,
    /// Image placeholder token ID expanded into the prompt before
    /// `get_input_embeddings`. Mirrors upstream `img_context_token_id`.
    pub img_context_token_id: i32,
    /// Image-start / image-end framing token IDs. Both default to `0`
    /// when the upstream config does not surface them — `0` means "no
    /// framing tokens emitted in the prompt expansion".
    pub image_start_token_id: i32,
    pub image_end_token_id: i32,
    /// Audio placeholder token ID. `None` when the checkpoint does not
    /// surface `sound_context_token_id` (audio path remains disabled).
    pub sound_context_token_id: Option<i32>,
    /// Optional audio-start / audio-end framing token IDs (mirrors the
    /// upstream image_start/image_end behaviour applied to the audio
    /// modality). Both default to `0` when not provided.
    pub sound_start_token_id: i32,
    pub sound_end_token_id: i32,
    /// EOS token IDs from the released checkpoint's `generation_config.json`.
    pub eos_token_ids: Vec<i32>,
}

/// Inverse of `downsample_ratio` rounded to the nearest integer.
fn downsample_factor(ratio: f32) -> i32 {
    let raw = (1.0 / ratio.max(f32::EPSILON)).round() as i32;
    raw.max(1)
}

/// Multimodal projector (`mlp1`).
///
/// Faithful port of upstream `VisionProjection`:
/// `RMSNorm(in_features) -> Linear(in_features, proj_hidden) ->
/// SquaredReLU -> Linear(proj_hidden, text_hidden)`.
/// The upstream `sanitize` rewrites `mlp1.0.* -> mlp1.layers.0.*` etc.,
/// so the loader applies the same rewrite and we read from
/// `mlp1.layers.{0,1,3}`.
pub struct NemotronHNanoOmniProjector {
    norm: RMSNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl NemotronHNanoOmniProjector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let norm_weight = weights
            .get(&format!("{prefix}.layers.0.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| {
                format!("Missing multimodal projector norm weight ({prefix}.layers.0.weight)")
            })?;
        let norm = RMSNorm::new(norm_weight, 1e-5);
        let fc1 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.layers.1"), group_size, bits)?;
        let fc2 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.layers.3"), group_size, bits)?;
        Ok(Self { norm, fc1, fc2 })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.norm.forward(x);
        let h = self.fc1.forward(&h);
        let h = mlxcel_core::utils::relu_squared(&h);
        self.fc2.forward(&h)
    }
}

/// Top-level Nemotron H Nano Omni VLM.
///
/// introduced this struct with vision-only scope;
/// added the optional audio path (`audio` field set to `Some` when the
/// checkpoint ships a `sound_config`).
pub struct NemotronHNanoOmniVlModel {
    pub text_model: NemotronHModel,
    pub vision_tower: NemotronHNanoOmniVisionModel,
    pub projector: NemotronHNanoOmniProjector,
    pub processor: NemotronHNanoOmniImageProcessor,
    pub config: NemotronHNanoOmniVlConfig,
    /// Optional audio bundle. Present when the checkpoint shipped a
    /// `sound_config` block and the loader successfully built the
    /// audio encoder/projector.
    pub audio: Option<NemotronOmniAudioBundle>,
}

/// Bundles every audio-side artifact loaded from the checkpoint so the
/// model surface stays cohesive (one `Option`, not three).
pub struct NemotronOmniAudioBundle {
    pub config: NemotronOmniAudioConfig,
    pub feature_extractor: NemotronOmniFeatureExtractor,
    pub encoder: NemotronOmniSoundEncoder,
    pub projection: NemotronOmniSoundProjection,
}

impl NemotronHNanoOmniVlModel {
    pub fn new(
        text_model: NemotronHModel,
        vision_tower: NemotronHNanoOmniVisionModel,
        projector: NemotronHNanoOmniProjector,
        processor: NemotronHNanoOmniImageProcessor,
        config: NemotronHNanoOmniVlConfig,
    ) -> Self {
        Self {
            text_model,
            vision_tower,
            projector,
            processor,
            config,
            audio: None,
        }
    }

    /// Builder-style attach for the audio bundle, used by the loader
    /// when `sound_config` is present in `config.json`.
    pub fn with_audio(mut self, audio: NemotronOmniAudioBundle) -> Self {
        self.audio = Some(audio);
        self
    }

    /// `true` when this VLM was loaded with audio support enabled.
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Audio bundle accessor for the runtime layer.
    pub fn audio(&self) -> Option<&NemotronOmniAudioBundle> {
        self.audio.as_ref()
    }

    /// Run the vision tower + projector for a single preprocessed image.
    /// Returns `[1, num_tokens, text_hidden]` ready to scatter into the
    /// text embedding stream.
    fn extract_image_features_single(
        &self,
        image: &NemotronHNanoOmniImageInput,
    ) -> UniquePtr<MlxArray> {
        let pixel_values = image.pixel_values.as_ref().unwrap();
        let radio = self.vision_tower.forward(pixel_values, false);
        let features = radio.features;

        let in_shape = mlxcel_core::array_shape(pixel_values);
        let height = in_shape[2];
        let width = in_shape[3];
        let patch_size = self.vision_tower.patch_size() as i32;
        let patch_h = height / patch_size;
        let patch_w = width / patch_size;

        let feat_shape = mlxcel_core::array_shape(&features);
        let batch = feat_shape[0];
        let channels = feat_shape[2];
        let reshaped = mlxcel_core::reshape(&features, &[batch, patch_h, patch_w, channels]);
        let shuffled = self.pixel_shuffle(&reshaped);

        let post_shape = mlxcel_core::array_shape(&shuffled);
        let post_channels = post_shape[3];
        let flattened = mlxcel_core::reshape(&shuffled, &[batch, -1, post_channels]);
        self.projector.forward(&flattened)
    }

    /// Run the vision pipeline across all images and concatenate the
    /// per-image features along the token axis. Mirrors upstream
    /// `extract_feature` behaviour for a list of pixel tensors.
    pub fn extract_image_features(
        &self,
        images: &[NemotronHNanoOmniImageInput],
    ) -> Option<UniquePtr<MlxArray>> {
        if images.is_empty() {
            return None;
        }
        let mut iter = images.iter();
        let first = self.extract_image_features_single(iter.next()?);
        let mut acc = first;
        for image in iter {
            let features = self.extract_image_features_single(image);
            acc = mlxcel_core::concatenate(&acc, &features, 0);
        }
        Some(acc)
    }

    /// Compose vision features with text embeddings at image-token slots.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        images: &[NemotronHNanoOmniImageInput],
    ) -> InputEmbeddings {
        self.get_input_embeddings_full(input_ids, images, None)
    }

    /// Generalised entry-point that mirrors upstream
    /// `get_input_embeddings(input_ids, pixel_values, sound_clips=...)`.
    ///
    /// `audio_features` is the precomputed
    /// `[total_audio_tokens, hidden_size]` embedding produced by
    /// [`extract_audio_features`]; the runtime layer is responsible for
    /// running the audio encoder/projection so the heavy lifting can
    /// share the same image-token-merge helper as the vision path.
    pub fn get_input_embeddings_full(
        &self,
        input_ids: &MlxArray,
        images: &[NemotronHNanoOmniImageInput],
        audio_features: Option<&MlxArray>,
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.input_embeddings(input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);

        // Apply image features first (mirrors upstream ordering). When
        // the prompt has no image placeholders or no images were
        // supplied, this returns the unchanged embedding stream.
        let after_images = if let Some(features) = self.extract_image_features(images) {
            let features = mlxcel_core::astype(&features, embed_dtype);
            crate::vision::merge::merge_llava(
                self.config.img_context_token_id,
                &features,
                &inputs_embeds,
                input_ids,
            )
        } else {
            InputEmbeddings {
                inputs_embeds,
                attention_mask_4d: None,
            }
        };

        // Apply audio features over the image-merged stream. Without an
        // audio config or audio token id, the merge is a no-op.
        match (audio_features, self.config.sound_context_token_id) {
            (Some(audio), Some(token_id)) => {
                let audio = mlxcel_core::astype(audio, embed_dtype);
                crate::vision::merge::merge_llava(
                    token_id,
                    &audio,
                    &after_images.inputs_embeds,
                    input_ids,
                )
            }
            _ => after_images,
        }
    }

    /// Extract audio embeddings from a precomputed mel feature batch.
    ///
    /// `input_features: [B, T, num_mel_bins]`
    /// `attention_mask: [B, T]` (`1` = valid frame, `0` = padding)
    /// `feature_lengths: [B]` total frame count per clip (matches
    /// upstream `full_lengths`; falls back to `attention_mask.sum(-1)`
    /// when not provided).
    ///
    /// Returns `[total_audio_tokens, text_hidden_size]` — flattened
    /// across the batch, with each clip trimmed to its post-subsampling
    /// valid length. This mirrors upstream `_extract_sound_features`'s
    /// final concatenation that the merge step expects.
    pub fn extract_audio_features(
        &self,
        input_features: &MlxArray,
        attention_mask: Option<&MlxArray>,
        feature_lengths: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let bundle = self
            .audio
            .as_ref()
            .ok_or_else(|| "Audio bundle is not configured for this VLM".to_string())?;

        // Cast to the lm-head's dtype the same way upstream does
        // (`compute_dtype = lm_head.scales.dtype or lm_head.weight.dtype`).
        // We read the dtype off the projector's RMSNorm scale tensor,
        // which is loaded from the same checkpoint at the same compute
        // precision as the rest of the model. This is zero-cost
        // (just a metadata read) — strictly avoid running a real
        // embedding lookup or allocating a 1×1 input_ids tensor purely
        // to inspect the resulting dtype.
        let embed_dtype = bundle.projection.compute_dtype();
        let input_features = mlxcel_core::astype(input_features, embed_dtype);

        let encoded = bundle.encoder.forward(&input_features, attention_mask);
        let projected = bundle.projection.forward(&encoded);

        // Trim to post-subsampling valid lengths and concatenate across
        // the batch axis. Feature lengths default to attention_mask sum.
        let lengths_arr = match feature_lengths {
            Some(l) => mlxcel_core::astype(l, mlxcel_core::dtype::INT32),
            None => match attention_mask {
                Some(m) => {
                    let s = mlxcel_core::sum_axis(m, -1, false);
                    mlxcel_core::astype(&s, mlxcel_core::dtype::INT32)
                }
                None => {
                    // No mask either: assume the entire encoder output
                    // is valid for every clip.
                    let pre_shape = mlxcel_core::array_shape(&projected);
                    return Ok(mlxcel_core::reshape(
                        &projected,
                        &[pre_shape[0] * pre_shape[1], pre_shape[2]],
                    ));
                }
            },
        };

        // Compute per-batch output lengths via subsampling formula. We
        // need scalar lengths, so we sync each one once. `B` is small
        // (typically 1) per request.
        let kernel = bundle.config.subsampling_conv_kernel_size as i32;
        let stride = bundle.config.subsampling_conv_stride as i32;
        let stages = bundle.config.num_subsampling_layers();
        let output_lengths = subsampling_output_lengths(&lengths_arr, kernel, stride, stages);

        let pre_shape = mlxcel_core::array_shape(&projected);
        let batch = pre_shape[0] as usize;
        let hidden = pre_shape[2];
        if batch == 0 {
            return Ok(mlxcel_core::reshape(&projected, &[0, hidden]));
        }

        let lengths_cpu = read_int32_vec(&output_lengths);
        let mut pieces: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(batch);
        for (b, length) in lengths_cpu.into_iter().enumerate() {
            if length <= 0 {
                continue;
            }
            let length = length.min(pre_shape[1]);
            let starts = vec![b as i32, 0, 0];
            let ends = vec![b as i32 + 1, length, hidden];
            let slice = mlxcel_core::slice(&projected, &starts, &ends);
            // Drop the leading batch dim so we end up with [length, H].
            let slice = mlxcel_core::reshape(&slice, &[length, hidden]);
            pieces.push(slice);
        }

        if pieces.is_empty() {
            return Ok(mlxcel_core::reshape(&projected, &[0, hidden]));
        }

        let mut acc = pieces.remove(0);
        for piece in pieces {
            acc = mlxcel_core::concatenate(&acc, &piece, 0);
        }
        Ok(acc)
    }

    fn pixel_shuffle(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let scale = self.config.downsample_ratio;
        if scale == 1.0 {
            return mlxcel_core::copy(x);
        }
        let factor = downsample_factor(scale);

        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let width = shape[1];
        let height = shape[2];
        let channels = shape[3];

        // Step 1: reshape (B, W, H, C) -> (B, W, H/factor, C*factor)
        let new_h = height / factor;
        let new_c1 = channels * factor;
        let h1 = mlxcel_core::reshape(x, &[batch, width, new_h, new_c1]);

        // Step 2: transpose (0, 2, 1, 3) -> (B, H/factor, W, C*factor)
        let h2 = mlxcel_core::transpose_axes(&h1, &[0, 2, 1, 3]);

        // Step 3: reshape -> (B, H/factor, W/factor, C*factor*factor)
        let new_w = width / factor;
        let new_c2 = new_c1 * factor;
        let h3 = mlxcel_core::reshape(&h2, &[batch, new_h, new_w, new_c2]);

        // Step 4: optional final transpose for `ps_version != "v1"`.
        if self.config.ps_version != "v1" {
            mlxcel_core::transpose_axes(&h3, &[0, 2, 1, 3])
        } else {
            h3
        }
    }

    /// Hidden size advertised to the runtime — same as the text backbone.
    pub fn hidden_size(&self) -> usize {
        self.text_model.hidden_size()
    }
}

impl LanguageModel for NemotronHNanoOmniVlModel {
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
        if let Some(embeds) = input_embeddings {
            self.text_model.forward_with_inputs_embeds(embeds)
        } else {
            self.text_model.forward(input_ids, caches, mask)
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.input_embeddings(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // NemotronH owns its mixed-cache state internally and exposes
        // empty `KVCache` placeholders to satisfy the trait — the VLM
        // wrapper passes those through unchanged via the trait method.
        <NemotronHModel as LanguageModel>::make_caches(&self.text_model)
    }

    fn num_layers(&self) -> usize {
        <NemotronHModel as LanguageModel>::num_layers(&self.text_model)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        if !self.config.eos_token_ids.is_empty() {
            self.config.eos_token_ids.clone()
        } else {
            <NemotronHModel as LanguageModel>::eos_token_ids(&self.text_model)
        }
    }

    fn supports_padded_prefill(&self) -> bool {
        // Nemotron-H is a hybrid Mamba+Attention model: padding tokens
        // corrupt Mamba recurrent state. The text path declares this
        // explicitly; the VLM wrapper inherits the same constraint.
        false
    }

    fn supports_batching(&self) -> bool {
        // Internal cache state of the text backbone is not per-sequence
        // isolated, so multiple sequences cannot share one model
        // instance.
        false
    }

    fn trim_internal_caches(&self, excess: i32) {
        self.text_model.trim_internal_caches(excess);
    }
}

/// Apply N stages of upstream `_get_output_length` (same kernel/stride
/// per stage) on an int32 length tensor. Mirrors
/// `_get_subsampling_output_length` from
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/audio.py.
fn subsampling_output_lengths(
    lengths: &MlxArray,
    kernel_size: i32,
    stride: i32,
    num_stages: usize,
) -> UniquePtr<MlxArray> {
    let mut current = mlxcel_core::astype(lengths, mlxcel_core::dtype::INT32);
    let padding = (kernel_size - 1) / 2;
    let add_pad = (2 * padding - kernel_size) as f32;
    let stride_f = stride as f32;
    for _ in 0..num_stages {
        let f32_lengths = mlxcel_core::astype(&current, mlxcel_core::dtype::FLOAT32);
        let pad_arr = mlxcel_core::full_f32(&[1], add_pad, mlxcel_core::dtype::FLOAT32);
        let added = mlxcel_core::add(&f32_lengths, &pad_arr);
        let stride_arr = mlxcel_core::full_f32(&[1], stride_f, mlxcel_core::dtype::FLOAT32);
        let divided = mlxcel_core::divide(&added, &stride_arr);
        let one_arr = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
        let plus_one = mlxcel_core::add(&divided, &one_arr);
        let floored = mlxcel_core::floor(&plus_one);
        current = mlxcel_core::astype(&floored, mlxcel_core::dtype::INT32);
    }
    current
}

/// Materialize an int32 array as a host-side `Vec<i32>`.
fn read_int32_vec(arr: &MlxArray) -> Vec<i32> {
    mlxcel_core::eval(arr);
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    bytes
        .chunks_exact(4)
        .map(|b| {
            i32::from_ne_bytes(
                b.try_into()
                    .expect("chunks_exact(4) always yields a 4-byte slice"),
            )
        })
        .collect()
}

#[cfg(test)]
#[path = "nemotron_h_nano_omni_vl_tests.rs"]
mod tests;
