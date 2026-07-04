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

//! DeepSeek-OCR 2 Vision-Language Model wrapper.
//!
//! Runs the SAM tower ([`super::encoders::deepseekocr_sam`], with the 896-channel
//! compressor) on each view, flattens its grid, resamples it through the Qwen2
//! query resampler ([`super::encoders::deepseekocr_qwen2`]), and projects the
//! query outputs to the decoder width. Unlike DeepSeek-OCR 1 there is no CLIP
//! tower, no channel-concat fusion, and no `image_newline` mosaic: features are
//! flat runs assembled per image as `[tiles, global, view_separator]` and
//! scattered into the `<image>` placeholder positions of the DeepSeek MoE
//! decoder's embedded prompt.
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseekocr_2/` (deepencoderv2).

use super::merge::{InputEmbeddings, merge_llava};
use super::processors::deepseekocr::{DeepSeekOcrPreprocessed, DeepSeekOcrProcessor};
use crate::models::deepseek::DeepSeekModel;
use crate::vision::encoders::deepseekocr_qwen2::Qwen2Resampler;
use crate::vision::encoders::deepseekocr_sam::SamEncoder;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::{MlxArray, UniquePtr};

pub struct DeepSeekOcr2VlModel {
    pub text_model: DeepSeekModel,
    pub sam: SamEncoder,
    pub resampler: Qwen2Resampler,
    pub projector: UnifiedLinear,
    pub view_separator: UniquePtr<MlxArray>, // (n_embed,)
    pub processor: DeepSeekOcrProcessor,
    pub image_token_id: i32,
    pub eos_token_id: i32,
    pub n_embed: i32,
}

impl DeepSeekOcr2VlModel {
    /// One view batch `(n, side, side, 3)` -> flat projected features
    /// `(n, g*g, n_embed)` where `g = side/64` is the SAM grid side.
    fn view_features(&self, pixels: &MlxArray) -> UniquePtr<MlxArray> {
        let sam = self.sam.forward(pixels); // (n, g, g, 896)
        let cs = mlxcel_core::array_shape(&sam);
        let (n, g, c) = (cs[0], cs[1], cs[3]);
        let flat = mlxcel_core::reshape(&sam, &[n, g * g, c]); // (n, g*g, 896)
        let resampled = self.resampler.forward(&flat); // (n, g*g, 896)
        self.projector.forward(&resampled) // (n, g*g, n_embed)
    }

    /// Build all images' flat feature rows in prompt order and scatter into the
    /// `<image>` positions of the embedded prompt.
    pub fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pre: &DeepSeekOcrPreprocessed,
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.embed_tokens_forward(input_ids);
        let g_feat = self.view_features(&pre.global); // (n_img, g*g, n_embed)
        let gtok = mlxcel_core::array_shape(&g_feat)[1];
        let sep = mlxcel_core::reshape(&self.view_separator, &[1, self.n_embed]);

        let mut per_image: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(pre.crops.len());
        let mut tile_off = 0i32;
        for (i, &n_tiles) in pre.tiles_per_image.iter().enumerate() {
            let global = mlxcel_core::slice(
                &g_feat,
                &[i as i32, 0, 0],
                &[i as i32 + 1, gtok, self.n_embed],
            );
            let global = mlxcel_core::reshape(&global, &[gtok, self.n_embed]);

            let img_feat = if n_tiles > 0 {
                let tiles = pre.tiles.as_ref().expect("tiles present for tiled image");
                let ts = mlxcel_core::array_shape(tiles);
                let tile_batch = mlxcel_core::slice(
                    tiles,
                    &[tile_off, 0, 0, 0],
                    &[tile_off + n_tiles, ts[1], ts[2], ts[3]],
                );
                tile_off += n_tiles;
                let t_feat = self.view_features(&tile_batch); // (n_tiles, gt*gt, n_embed)
                let ttok = mlxcel_core::array_shape(&t_feat)[1];
                // Flat, row-major tile order: (n_tiles, gt*gt, C) -> (n_tiles*gt*gt, C).
                let tf = mlxcel_core::reshape(&t_feat, &[n_tiles * ttok, self.n_embed]);
                let tg = mlxcel_core::concatenate(&tf, &global, 0);
                mlxcel_core::concatenate(&tg, &sep, 0)
            } else {
                mlxcel_core::concatenate(&global, &sep, 0)
            };
            per_image.push(img_feat);
        }

        let all = match per_image.len() {
            1 => per_image.into_iter().next().unwrap(),
            _ => {
                let mut it = per_image.into_iter();
                let first = it.next().unwrap();
                it.fold(first, |acc, f| mlxcel_core::concatenate(&acc, &f, 0))
            }
        };
        merge_llava(self.image_token_id, &all, &inputs_embeds, input_ids)
    }
}

impl LanguageModel for DeepSeekOcr2VlModel {
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

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![self.eos_token_id]
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        // The image placeholder id must never be sampled during decode.
        vec![self.image_token_id]
    }
}
