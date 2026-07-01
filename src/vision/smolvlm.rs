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

//! SmolVLM / SmolVLM2 (`smolvlm`) Vision-Language Model.
//!
//! Faithful port of
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/smolvlm/smolvlm.py
//! and its Idefics3 base
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/idefics3/idefics3.py.
//!
//! Composition (for `SmolVLM-Instruct` / `SmolVLM2`):
//! - `vision_model`: a SigLIP encoder ([`SigLipVisionModel`]). SmolVLM reuses
//!   the Idefics3 vision tower, which is the same SigLIP architecture mlxcel
//!   already ships (Conv2d patch embed + learned position embedding, N encoder
//!   layers, `post_layernorm`, no CLS token). For a full-resolution square tile
//!   with an all-ones patch attention mask, the Idefics3 position-id logic
//!   reduces to `arange(num_patches)`, which is exactly the SigLIP encoder's
//!   default path, so no bespoke encoder is needed.
//! - `connector` ([`SmolVLMConnector`]): `pixel_shuffle(scale_factor)` token
//!   compression followed by a single bias-free `modality_projection.proj`
//!   Linear that projects `vision_hidden * scale_factor^2` into the text hidden
//!   size.
//! - `language_model`: a SmolLM2 backbone. SmolLM2 is a plain Llama
//!   architecture (RoPE on every layer, SwiGLU MLP, RMSNorm), so we reuse
//!   mlxcel's [`crate::models::Llama3Model`] exactly as the InternVL runtime
//!   reuses it for its Qwen2 backbone. This gives the full VLM runtime surface
//!   (`forward_with_embeddings`, batched decode, sequence-state) for free. Note
//!   `smollm3.rs` implements the *SmolLM3* NoPE variant, which is a different
//!   model than SmolLM2 and would disable RoPE on some layers; Llama3Model is
//!   both the faithful and fully-integrated choice here.
//!
//! Vision/text fusion: the connector produces `num_image_token` feature vectors
//! per image tile in the text hidden size. Those replace the `<image>`
//! placeholder positions in the prompt embedding stream via
//! [`crate::vision::merge::merge_llava`], the LLaVA-style token-replacement
//! semantics that match the Idefics3 `masked_scatter` merge.
//!
//! Standard 1D RoPE is used (no MRoPE), so this runtime is structurally a
//! LLaVA-style VLM with a tiled image preprocessor and a pixel-shuffle
//! connector.
//!
//! Used by: `loading::load_smolvlm_vlm`, `multimodal::vlm_runtime`.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::Llama3Model;
use crate::vision::encoders::VisionEncoder;
use crate::vision::encoders::siglip::SigLipVisionModel;
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::smolvlm::SmolVLMProcessor;

/// The SmolVLM vision-to-language connector (`connector` in the checkpoint).
///
/// `pixel_shuffle(scale_factor)` collapses each `scale_factor x scale_factor`
/// block of the square patch grid into the channel dimension, reducing the
/// token count by `scale_factor^2` and growing the channel dim by the same
/// factor. A single bias-free Linear (`modality_projection.proj`) then projects
/// `vision_hidden * scale_factor^2` into the text hidden size.
pub struct SmolVLMConnector {
    proj: UnifiedLinear,
    scale_factor: i32,
}

impl SmolVLMConnector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        scale_factor: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        // `modality_projection` is a small MLP whose only parameter is a
        // bias-free Linear named `proj`.
        let proj = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.modality_projection.proj"),
            group_size,
            bits,
        )?;
        Ok(Self { proj, scale_factor })
    }

    /// `vision_features`: `[N, seq, C]` -> `[N, seq / s^2, hidden]` projected
    /// image tokens, where `s = scale_factor`.
    pub fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        let x = pixel_shuffle(vision_features, self.scale_factor);
        self.proj.forward(&x)
    }
}

/// Pixel shuffle (spatial -> channel) port of the Idefics3/SmolVLM connector
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/idefics3/idefics3.py#L53-L67.
///
/// Input `x`: `[B, seq, C]` where `seq = p*p` is a square patch grid.
/// Output: `[B, seq / s^2, C * s^2]` for `s = scale_factor`.
///
/// The reshape/transpose sequence mirrors the Python reference operation for
/// operation (it is mathematically identical to the InternVL
/// `pixel_shuffle(shuffle_ratio = 1 / scale_factor)` used elsewhere in the
/// tree, but is expressed in the integer `scale_factor` form the SmolVLM config
/// carries).
fn pixel_shuffle(x: &MlxArray, scale_factor: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let b = shape[0];
    let seq = shape[1];
    let embed_dim = shape[2];
    // height == width for a square patch grid.
    let side = (seq as f64).sqrt().round() as i32;

    // x = x.reshape(B, side, side, embed_dim)
    let x = mlxcel_core::reshape(x, &[b, side, side, embed_dim]);
    // x = x.reshape(B, side, side / s, embed_dim * s)
    let x = mlxcel_core::reshape(
        &x,
        &[b, side, side / scale_factor, embed_dim * scale_factor],
    );
    // x = x.transpose(0, 2, 1, 3)
    let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
    // x = x.reshape(B, side / s, side / s, embed_dim * s^2)
    let x = mlxcel_core::reshape(
        &x,
        &[
            b,
            side / scale_factor,
            side / scale_factor,
            embed_dim * scale_factor * scale_factor,
        ],
    );
    // x = x.transpose(0, 2, 1, 3)
    let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
    // x = x.reshape(B, seq / s^2, embed_dim * s^2)
    mlxcel_core::reshape(
        &x,
        &[
            b,
            seq / (scale_factor * scale_factor),
            embed_dim * scale_factor * scale_factor,
        ],
    )
}

/// Top-level SmolVLM (`smolvlm`) VLM runtime.
pub struct SmolVLMModel {
    pub text_model: Llama3Model,
    pub vision_model: SigLipVisionModel,
    pub connector: SmolVLMConnector,
    pub processor: SmolVLMProcessor,
    /// Token id of the `<image>` placeholder (`image_token_id` in config.json;
    /// 49153 for the released SmolVLM checkpoints). Image features are merged at
    /// these positions.
    pub image_token_id: i32,
    /// Token id of `<fake_token_around_image>` (frames each image block). `0`
    /// when the tokenizer does not expose it.
    pub fake_image_token_id: i32,
    /// Token id of `<global-img>` (opens the global-image block). `0` when the
    /// tokenizer does not expose it.
    pub global_image_token_id: i32,
    /// Number of image feature tokens emitted per tile after pixel-shuffle
    /// compression (`(image_size / patch_size / scale_factor)^2`).
    pub num_image_token: usize,
    /// EOS/stop token ids consumed by the server stop path.
    pub eos_token_ids: Vec<i32>,
}

impl SmolVLMModel {
    /// Compute merged input embeddings for a request that carries pixel values.
    /// Mirrors `Model.get_input_embeddings` in upstream `idefics3.py`.
    ///
    /// `pixel_values`: channels-first `[num_tiles, C, H, W]` (the processor's
    /// native layout). The SigLIP tower is channels-last, so we transpose once
    /// here, matching the reference `pixel_values.transpose(0, 2, 3, 1)`.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> InputEmbeddings {
        // Text embeddings (SmolLM2 token embedding table).
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        // Match the dtype of the text embeddings when feeding pixels into the
        // vision tower (mirrors upstream `pixel_values.astype(dtype)`).
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);

        // [num_tiles, C, H, W] -> [num_tiles, H, W, C] for the conv patch embed.
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);

        // SigLIP tower -> [num_tiles, num_patches, hidden_vit] (post_layernorm,
        // no CLS token to strip).
        let vision_output = self.vision_model.forward(&pv);

        // pixel_shuffle + modality_projection -> [num_tiles, num_image_token,
        // hidden_lm].
        let image_features = self.connector.forward(&vision_output.hidden_states);

        // Replace <image> positions with the projected image features.
        merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
    }
}

// LanguageModel: text-only forward paths delegate straight to the underlying
// SmolLM2 (Llama) backbone, including the `forward_with_embeddings` path used by
// the VLM runtime to inject pre-merged inputs. EOS ids are overridden so the
// server stop path uses SmolVLM's configured stop tokens rather than the Llama
// defaults.
impl LanguageModel for SmolVLMModel {
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
        self.eos_token_ids.clone()
    }

    fn supports_batched_prefill(&self) -> bool {
        self.text_model.supports_batched_prefill()
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        self.text_model.supports_maskless_padded_prefill()
    }

    fn supports_paged_decode_backend(&self) -> bool {
        self.text_model.supports_paged_decode_backend()
    }
}

#[cfg(test)]
mod tests {
    use super::pixel_shuffle;

    /// Reference pixel-shuffle computed by directly simulating the
    /// Idefics3/SmolVLM Python reshape/transpose chain over row-major
    /// `[B, seq, C]` data (batch 1), independent of the MLX implementation.
    /// This is the parity oracle for the net-new connector math.
    ///
    /// The chain, all in row-major so each `reshape` is a no-op on the flat
    /// buffer and only the two transposes move data:
    ///   s0 = [side, side, c]                       (== [seq, c])
    ///   s1 = reshape(s0, [side, side/s, c*s])
    ///   s2 = transpose(s1, (1, 0, 2))              -> [side/s, side, c*s]
    ///   s3 = reshape(s2, [side/s, side/s, c*s*s])
    ///   s4 = transpose(s3, (1, 0, 2))              -> [side/s, side/s, c*s*s]
    ///   out = reshape(s4, [out_seq, c*s*s])
    fn reference_pixel_shuffle(
        input: &[f32],
        seq: usize,
        c: usize,
        scale: usize,
    ) -> (Vec<f32>, usize, usize) {
        let side = (seq as f64).sqrt().round() as usize;
        assert_eq!(side * side, seq, "seq must be a perfect square");
        assert_eq!(side % scale, 0, "grid side must be divisible by scale");

        let ss = side / scale;
        let c1 = c * scale; // channels after the first reshape
        let c2 = c * scale * scale; // channels after the second reshape

        // s1_flat == input (row-major reshape). s2[j0, j1, j2] = s1[j1, j0, j2]
        // where s1 is viewed as [side, ss, c1].
        let mut s2 = vec![0f32; ss * side * c1];
        for j0 in 0..ss {
            for j1 in 0..side {
                for j2 in 0..c1 {
                    s2[((j0 * side) + j1) * c1 + j2] = input[((j1 * ss) + j0) * c1 + j2];
                }
            }
        }

        // s3_flat == s2_flat (row-major reshape into [ss, ss, c2]).
        // s4[m0, m1, m2] = s3[m1, m0, m2].
        let mut s4 = vec![0f32; ss * ss * c2];
        for m0 in 0..ss {
            for m1 in 0..ss {
                for m2 in 0..c2 {
                    s4[((m0 * ss) + m1) * c2 + m2] = s2[((m1 * ss) + m0) * c2 + m2];
                }
            }
        }

        (s4, ss * ss, c2)
    }

    #[test]
    fn pixel_shuffle_matches_reference_shape_and_values() {
        // 4x4 patch grid (seq=16), channel dim 3, scale_factor 2.
        let seq = 16usize;
        let c = 3usize;
        let scale = 2usize;
        let data: Vec<f32> = (0..(seq * c)).map(|i| i as f32).collect();

        let x = mlxcel_core::from_slice_f32(&data, &[1, seq as i32, c as i32]);
        let out = pixel_shuffle(&x, scale as i32);
        mlxcel_core::eval(&out);

        let shape = mlxcel_core::array_shape(&out);
        let (expected, out_seq, out_c) = reference_pixel_shuffle(&data, seq, c, scale);
        assert_eq!(shape, vec![1, out_seq as i32, out_c as i32]);

        // Element-for-element parity against the reference oracle.
        let flat = mlxcel_core::reshape(&out, &[(out_seq * out_c) as i32]);
        mlxcel_core::eval(&flat);
        for (i, &want) in expected.iter().enumerate() {
            let cell = mlxcel_core::slice(&flat, &[i as i32], &[i as i32 + 1]);
            mlxcel_core::eval(&cell);
            let got = mlxcel_core::item_f32(&cell);
            assert!(
                (got - want).abs() < 1e-6,
                "pixel_shuffle[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn pixel_shuffle_preserves_element_count_and_origin() {
        // 6x6 grid, channel dim 4, scale 2 -> 9 tokens of dim 16.
        let seq = 36i32;
        let c = 4i32;
        let data: Vec<f32> = (0..(seq * c)).map(|i| i as f32).collect();
        let x = mlxcel_core::from_slice_f32(&data, &[1, seq, c]);

        let out = pixel_shuffle(&x, 2);
        mlxcel_core::eval(&out);
        let shape = mlxcel_core::array_shape(&out);
        assert_eq!(shape, vec![1, 9, 16]);

        // Pure reshape/transpose: element count is preserved.
        let total: i32 = shape.iter().product();
        assert_eq!(total, seq * c);

        // The top-left patch, channel 0 stays at the origin of the output.
        let first = mlxcel_core::slice(&out, &[0, 0, 0], &[1, 1, 1]);
        mlxcel_core::eval(&first);
        assert_eq!(mlxcel_core::item_f32(&first), 0.0);
    }
}
