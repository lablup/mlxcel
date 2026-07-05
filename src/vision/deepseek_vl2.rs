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

//! DeepSeek-VL2 Vision-Language Model wrapper.
//!
//! Runs the SigLIP tower ([`super::encoders::deepseek_vl2`]) on each image's
//! global view and local tiles, projects with the `downsample_mlp_gelu`
//! connector ([`super::connectors::deepseek_vl2`]), assembles the 2D tile mosaic
//! with per-row `image_newline` columns and a `view_separator`, and scatters the
//! result into the `<image>` placeholder positions of the DeepSeek-MoE decoder's
//! embedded prompt.
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseek_vl_v2/deepseek_vl_v2.py` (Model).
//! Feature order per image is `[global, view_separator, local]`
//! (`global_view_pos == "head"`); the placeholder tokens are all the image id,
//! so the scatter fills them in that feature order.

use super::connectors::MultiModalConnector;
use super::connectors::deepseek_vl2::DownsampleMlpGelu;
use super::encoders::deepseek_vl2::DeepSeekVl2VisionEncoder;
use super::merge::{InputEmbeddings, merge_llava};
use super::processors::deepseek_vl2::{DeepSeekVl2Preprocessed, DeepSeekVl2Processor};
use crate::models::deepseek_v2::DeepSeekV2Model;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct DeepSeekVl2VlModel {
    pub text_model: DeepSeekV2Model,
    pub encoder: DeepSeekVl2VisionEncoder,
    pub projector: DownsampleMlpGelu,
    pub image_newline: UniquePtr<MlxArray>,  // (n_embed,)
    pub view_separator: UniquePtr<MlxArray>, // (n_embed,)
    pub processor: DeepSeekVl2Processor,
    pub image_token_id: i32,
    pub eos_token_id: i32,
    pub n_embed: i32,
}

impl DeepSeekVl2VlModel {
    /// One view batch `(n, img, img, 3)` -> projected features `(n, N', n_embed)`
    /// plus the per-axis query count `h'` (`N' = h' * h'`).
    fn view_features(&self, pixels: &MlxArray) -> (UniquePtr<MlxArray>, i32) {
        let enc = self.encoder.forward(pixels); // (n, N, width)
        let feat = self.projector.forward(&enc); // (n, N', n_embed)
        let s = mlxcel_core::array_shape(&feat);
        let hp = (s[1] as f64).sqrt().round() as i32;
        (feat, hp)
    }

    /// Global mosaic `(h'*(h'+1), n_embed)`: a newline column appended per row.
    fn global_mosaic(&self, feat_i: &MlxArray, hp: i32) -> UniquePtr<MlxArray> {
        let d = self.n_embed;
        let grid = mlxcel_core::reshape(feat_i, &[hp, hp, d]);
        let nl = mlxcel_core::reshape(&self.image_newline, &[1, 1, d]);
        let nl = mlxcel_core::broadcast_to(&nl, &[hp, 1, d]);
        let grid = mlxcel_core::concatenate(&grid, &nl, 1); // (h', h'+1, D)
        mlxcel_core::reshape(&grid, &[hp * (hp + 1), d])
    }

    /// Build all images' feature rows in prompt order and scatter them into the
    /// `<image>` positions of the embedded prompt.
    pub fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pre: &DeepSeekVl2Preprocessed,
    ) -> InputEmbeddings {
        let d = self.n_embed;
        let inputs_embeds = self.text_model.embed_tokens_forward(input_ids);
        let (g_feat, hg) = self.view_features(&pre.global); // (n_img, hg*hg, D)
        let sep = mlxcel_core::reshape(&self.view_separator, &[1, d]);
        let ts = mlxcel_core::array_shape(&pre.tiles);

        let mut per_image: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(pre.crops.len());
        let mut tile_off = 0i32;
        for (i, &(tw, th)) in pre.crops.iter().enumerate() {
            let gi = mlxcel_core::slice(&g_feat, &[i as i32, 0, 0], &[i as i32 + 1, hg * hg, d]);
            let global = self.global_mosaic(&gi, hg);

            // Local tiles are always present (>= 1 per image).
            let n_tiles = tw * th;
            let tile_batch = mlxcel_core::slice(
                &pre.tiles,
                &[tile_off, 0, 0, 0],
                &[tile_off + n_tiles, ts[1], ts[2], ts[3]],
            );
            tile_off += n_tiles;
            let (t_feat, gt) = self.view_features(&tile_batch); // (n_tiles, gt*gt, D)

            // (th, tw, gt, gt, D) -> (th, gt, tw, gt, D) -> (th*gt, tw*gt, D),
            // then a newline column per row.
            let t = mlxcel_core::reshape(&t_feat, &[th, tw, gt, gt, d]);
            let t = mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3, 4]);
            let t = mlxcel_core::reshape(&t, &[th * gt, tw * gt, d]);
            let nl = mlxcel_core::reshape(&self.image_newline, &[1, 1, d]);
            let nl = mlxcel_core::broadcast_to(&nl, &[th * gt, 1, d]);
            let t = mlxcel_core::concatenate(&t, &nl, 1); // (th*gt, tw*gt+1, D)
            let t = mlxcel_core::reshape(&t, &[(th * gt) * (tw * gt + 1), d]);

            // global_view_pos == "head": [global, view_separator, local].
            let gs = mlxcel_core::concatenate(&global, &sep, 0);
            per_image.push(mlxcel_core::concatenate(&gs, &t, 0));
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

impl LanguageModel for DeepSeekVl2VlModel {
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
        match input_embeddings {
            Some(embeds) => self.text_model.forward_from_embeds(embeds, caches, mask),
            None => self.text_model.forward(input_ids, caches, mask),
        }
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
