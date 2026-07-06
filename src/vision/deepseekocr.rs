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

//! DeepSeek-OCR Vision-Language Model wrapper.
//!
//! Runs the SAM + CLIP towers ([`super::encoders::deepseekocr_sam`],
//! [`super::encoders::deepseekocr_clip`]) on each image's global view and tiles,
//! channel-fuses `[CLIP(drop CLS), SAM(flattened)]`, projects to the decoder
//! width, assembles the 2D tile mosaic with `image_newline` columns and a
//! trailing `view_separator`, and scatters the result into the `<image>`
//! placeholder positions of the DeepSeek MoE decoder's embedded prompt.
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseekocr/deepseekocr.py` (Model).
//! Feature order per image is `[tiles, global, view_separator]`; the placeholder
//! tokens are all id 128815, so the scatter fills them in that feature order.

use super::merge::{InputEmbeddings, merge_llava};
use super::processors::deepseekocr::{DeepSeekOcrPreprocessed, DeepSeekOcrProcessor};
use crate::models::deepseek::DeepSeekModel;
use crate::vision::encoders::deepseekocr_clip::ClipEncoder;
use crate::vision::encoders::deepseekocr_sam::SamEncoder;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::{MlxArray, UniquePtr};

pub struct DeepSeekOcrVlModel {
    pub text_model: DeepSeekModel,
    pub sam: SamEncoder,
    pub clip: ClipEncoder,
    pub projector: UnifiedLinear,
    pub image_newline: UniquePtr<MlxArray>,  // (n_embed,)
    pub view_separator: UniquePtr<MlxArray>, // (n_embed,)
    pub processor: DeepSeekOcrProcessor,
    pub image_token_id: i32,
    pub eos_token_id: i32,
    pub n_embed: i32,
}

impl DeepSeekOcrVlModel {
    /// One view batch `(n, side, side, 3)` -> projected features `(n, g*g, n_embed)`
    /// where `g = side/64` is the SAM grid side.
    fn view_features(&self, pixels: &MlxArray) -> (UniquePtr<MlxArray>, i32) {
        let sam = self.sam.forward(pixels); // (n, g, g, 1024)
        let clip = self.clip.forward(&sam); // (n, g*g+1, 1024)
        let cs = mlxcel_core::array_shape(&sam);
        let (n, g, c_sam) = (cs[0], cs[1], cs[3]);
        let tokens = g * g;
        let sam_flat = mlxcel_core::reshape(&sam, &[n, tokens, c_sam]);
        let ccl = mlxcel_core::array_shape(&clip);
        let c_clip = ccl[2];
        let clip_no_cls = mlxcel_core::slice(&clip, &[0, 1, 0], &[n, tokens + 1, c_clip]);
        let fused = mlxcel_core::concatenate(&clip_no_cls, &sam_flat, 2); // (n, tokens, 2048)
        (self.projector.forward(&fused), g)
    }

    /// Global mosaic `(g*(g+1), n_embed)`: newline column appended per row.
    fn global_mosaic(&self, feat_i: &MlxArray, g: i32) -> UniquePtr<MlxArray> {
        let grid = mlxcel_core::reshape(feat_i, &[g, g, self.n_embed]);
        let nl = mlxcel_core::reshape(&self.image_newline, &[1, 1, self.n_embed]);
        let nl = mlxcel_core::broadcast_to(&nl, &[g, 1, self.n_embed]);
        let grid = mlxcel_core::concatenate(&grid, &nl, 1); // (g, g+1, n_embed)
        mlxcel_core::reshape(&grid, &[g * (g + 1), self.n_embed])
    }

    /// Build all images' feature rows in prompt order and scatter into the
    /// `<image>` positions of the embedded prompt.
    pub fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pre: &DeepSeekOcrPreprocessed,
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.embed_tokens_forward(input_ids);
        let (g_feat, gg) = self.view_features(&pre.global); // (n_img, gg*gg, n_embed)
        let sep = mlxcel_core::reshape(&self.view_separator, &[1, self.n_embed]);

        let mut per_image: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(pre.crops.len());
        let mut tile_off = 0i32;
        for (i, &(w_crop, h_crop)) in pre.crops.iter().enumerate() {
            let gi = mlxcel_core::slice(
                &g_feat,
                &[i as i32, 0, 0],
                &[i as i32 + 1, gg * gg, self.n_embed],
            );
            let global = self.global_mosaic(&gi, gg);

            let img_feat = if w_crop > 1 || h_crop > 1 {
                let n_tiles = w_crop * h_crop;
                let tiles = pre.tiles.as_ref().expect("tiles present for cropped image");
                let ts = mlxcel_core::array_shape(tiles);
                let tile_batch = mlxcel_core::slice(
                    tiles,
                    &[tile_off, 0, 0, 0],
                    &[tile_off + n_tiles, ts[1], ts[2], ts[3]],
                );
                tile_off += n_tiles;
                let (t_feat, gt) = self.view_features(&tile_batch); // (n_tiles, gt*gt, n_embed)
                // (h, w, gt, gt, C) -> (h*gt, w*gt, C) then newline column.
                let t = mlxcel_core::reshape(&t_feat, &[h_crop, w_crop, gt, gt, self.n_embed]);
                let t = mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3, 4]);
                let t = mlxcel_core::reshape(&t, &[h_crop * gt, w_crop * gt, self.n_embed]);
                let nl = mlxcel_core::reshape(&self.image_newline, &[1, 1, self.n_embed]);
                let nl = mlxcel_core::broadcast_to(&nl, &[h_crop * gt, 1, self.n_embed]);
                let t = mlxcel_core::concatenate(&t, &nl, 1); // (h*gt, w*gt+1, C)
                let t =
                    mlxcel_core::reshape(&t, &[(h_crop * gt) * (w_crop * gt + 1), self.n_embed]);
                let tg = mlxcel_core::concatenate(&t, &global, 0);
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

impl LanguageModel for DeepSeekOcrVlModel {
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
