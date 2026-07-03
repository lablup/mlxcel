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

//! Granite Vision (`granite_vision` / `llava_next`+granite) Vision-Language Model.
//!
//! IBM's document VLM: a SigLIP tower with multi-layer feature taps, a 2-layer
//! GELU projector, a learned `image_newline` embedding, LLaVA-Next AnyRes
//! multi-tile preprocessing, and a dense Granite text backbone (the four Granite
//! scalar multipliers live in [`crate::models::granite::GraniteModel`]).
//!
//! Per image the tower runs over `1 + n_tiles_h * n_tiles_w` tiles (a base tile
//! plus the AnyRes grid). Four hidden-state taps are concatenated on the channel
//! axis, projected to the text hidden size, and the grid tiles are re-tiled into
//! a `(H, W)` feature grid, unpadded in token space, and given a trailing
//! `image_newline` column before being flattened and prepended with the base
//! tile's features. The resulting rows replace the `<image>` (49155) placeholder
//! positions via [`crate::vision::merge::merge_llava`]. The Granite
//! `embedding_multiplier` is applied by the backbone to the merged stream, so
//! image features must not be pre-scaled.

use super::merge::{self, InputEmbeddings};
use super::processors::anyres::{AnyResProcessor, AnyResTileInfo, unpadded_token_hw};
use crate::models::granite::GraniteModel;
use crate::vision::connectors::MultiModalConnector;
use crate::vision::connectors::mlp::MLPProjector;
use crate::vision::encoders::siglip::SigLipVisionModel;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Granite Vision VLM.
pub struct GraniteVisionVLModel {
    pub text_model: GraniteModel,
    pub vision_tower: SigLipVisionModel,
    pub projector: MLPProjector,
    /// Learned `image_newline` embedding `(text_hidden,)`.
    pub image_newline: UniquePtr<MlxArray>,
    pub processor: AnyResProcessor,
    pub image_token_index: i32,
    /// Hidden-state tap indices (e.g. `[-24, -20, -12, -1]`).
    pub vision_feature_layers: Vec<i32>,
    /// Patch grid side per tile (`image_size / patch_size`, 27).
    pub feature_side: i32,
    /// Tokens per tile (`feature_side^2`, 729).
    pub base_tokens: i32,
    pub eos_token_id: i32,
}

impl GraniteVisionVLModel {
    /// Merge tiled image features into the text embedding stream.
    ///
    /// `pixel_values`: `[sum_i num_tiles_i, 384, 384, 3]`; `infos`: per-image
    /// tile layout. Returns the merged (un-scaled) embeddings; the backbone
    /// applies `embedding_multiplier`.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        infos: &[AnyResTileInfo],
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.input_embeddings(input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);

        // Multi-tap tower over every tile, concatenated on the channel axis.
        let tap_outs = self
            .vision_tower
            .forward_collect_layers(&pv, &self.vision_feature_layers);
        let mut concat = mlxcel_core::copy(tap_outs[0].as_ref().unwrap());
        for t in &tap_outs[1..] {
            concat = mlxcel_core::concatenate(&concat, t, 2);
        }

        // Projector: [total_tiles, 729, vision_taps] -> [total_tiles, 729, D].
        let projected = self.projector.forward(&concat);
        let d = *mlxcel_core::array_shape(&projected).last().unwrap();

        // Pack each image independently, then concatenate all rows.
        let mut features: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(infos.len());
        let mut offset = 0i32;
        for info in infos {
            let n = info.num_tiles;
            let tiles = mlxcel_core::slice(
                &projected,
                &[offset, 0, 0],
                &[offset + n, self.base_tokens, d],
            );
            offset += n;
            features.push(self.pack_image_features(&tiles, info, d));
        }

        let image_features = match features.len() {
            1 => features.into_iter().next().unwrap(),
            _ => {
                let mut iter = features.into_iter();
                let first = iter.next().unwrap();
                iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 0))
            }
        };

        merge::merge_llava(
            self.image_token_index,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
    }

    /// Pack one image's per-tile features `[num_tiles, base_tokens, D]` into
    /// `[base_tokens + H*(W+1), D]` (base tile rows, then the unpadded grid with
    /// a trailing `image_newline` column per row).
    fn pack_image_features(
        &self,
        tiles: &MlxArray,
        info: &AnyResTileInfo,
        d: i32,
    ) -> UniquePtr<MlxArray> {
        let side = self.feature_side;
        let base_tokens = self.base_tokens;
        let (nth, ntw) = (info.n_tiles_h, info.n_tiles_w);
        let n_grid = nth * ntw;
        // Match the newline dtype to the projected features so concatenation
        // never mixes dtypes across platforms/quantization.
        let newline = mlxcel_core::astype(&self.image_newline, mlxcel_core::array_dtype(tiles));

        // Base tile features (tile 0): [base_tokens, D].
        let base = mlxcel_core::slice(tiles, &[0, 0, 0], &[1, base_tokens, d]);
        let base = mlxcel_core::reshape(&base, &[base_tokens, d]);

        if n_grid == 0 {
            // Degenerate no-grid case: base + a single image_newline row.
            let nl = mlxcel_core::reshape(&newline, &[1, d]);
            return mlxcel_core::concatenate(&base, &nl, 0);
        }

        // Grid tiles [n_grid, base_tokens, D] -> [nth, ntw, side, side, D]
        // -> transpose (4,0,2,1,3) -> [D, nth, side, ntw, side]
        // -> reshape [D, gh, gw].
        let grid = mlxcel_core::slice(tiles, &[1, 0, 0], &[1 + n_grid, base_tokens, d]);
        let grid = mlxcel_core::reshape(&grid, &[nth, ntw, side, side, d]);
        let grid = mlxcel_core::transpose_axes(&grid, &[4, 0, 2, 1, 3]);
        let gh = side * nth;
        let gw = side * ntw;
        let grid = mlxcel_core::reshape(&grid, &[d, gh, gw]);

        // Unpad in token space (same branch/values as the token counter).
        let (h, w) = unpadded_token_hw(info.orig_h, info.orig_w, nth, ntw, side);
        let unpad_rows = (info.orig_w as f64) / (info.orig_h as f64) > gw as f64 / gh as f64;
        let grid = if unpad_rows {
            let pad = (gh - h) / 2;
            mlxcel_core::slice(&grid, &[0, pad, 0], &[d, pad + h, gw])
        } else {
            let pad = (gw - w) / 2;
            mlxcel_core::slice(&grid, &[0, 0, pad], &[d, gh, pad + w])
        };

        // Append the newline column: [D] -> [D, H, 1], concat on axis 2.
        let nl = mlxcel_core::reshape(&newline, &[d, 1, 1]);
        let nl = mlxcel_core::broadcast_to(&nl, &[d, h, 1]);
        let grid = mlxcel_core::concatenate(&grid, &nl, 2); // [D, H, W+1]

        // [D, H, W+1] -> [D, H*(W+1)] -> transpose -> [H*(W+1), D].
        let grid = mlxcel_core::reshape(&grid, &[d, h * (w + 1)]);
        let grid = mlxcel_core::transpose_axes(&grid, &[1, 0]);

        mlxcel_core::concatenate(&base, &grid, 0)
    }
}

impl LanguageModel for GraniteVisionVLModel {
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
            Some(embeds) => self.text_model.forward_embeds(embeds, caches, mask),
            None => self.text_model.forward(input_ids, caches, mask),
        }
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
        match input_embeddings {
            Some(embeds) => self.text_model.forward_embeds(embeds, caches, mask),
            None => self.text_model.forward(input_ids, caches, mask),
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.input_embeddings(input_ids))
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
}
