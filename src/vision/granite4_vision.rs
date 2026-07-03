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

//! Granite 4 Vision (`granite4_vision`) Vision-Language Model.
//!
//! IBM's document VLM with multi-depth visual injection. A SigLIP tower feeds
//! eight window-QFormer projectors ([`super::connectors::granite4_vision`])
//! whose packed outputs are added into the residual stream at eight different
//! depths of a Granite 4 hybrid text backbone
//! ([`crate::models::granitemoehybrid::GraniteMoeHybridModel`]) during prefill,
//! rather than merged once into the input embeddings.
//!
//! Prefill flow: embed tokens, zero the `<image>` (100352) slots, run the tower
//! taps + 8 projectors + AnyRes packing to build 8 streams, then hand the
//! streams + position mask to the backbone's injection entry point (which owns
//! the recurrent conv/SSM + KV state). Decode steps carry no image input.

use std::cell::RefCell;

use super::merge::InputEmbeddings;
use super::processors::anyres::{AnyResProcessor, AnyResTileInfo, unpadded_token_hw};
use crate::models::granitemoehybrid::{GraniteMoeHybridModel, HybridInjection};
use crate::vision::connectors::granite4_vision::WindowQFormerProjector;
use crate::vision::encoders::siglip::SigLipVisionModel;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Stashed injection produced by `get_input_embeddings` and consumed by the
/// injecting `forward_with_embeddings*` on the very next prefill call.
struct StashedInjection {
    mask: UniquePtr<MlxArray>,
    targets: Vec<(usize, UniquePtr<MlxArray>)>,
}

/// Granite 4 Vision VLM.
pub struct Granite4VisionVLModel {
    pub text_model: GraniteMoeHybridModel,
    pub vision_tower: SigLipVisionModel,
    /// 8 projectors: `[0..4]` deepstack (layerwise), `[4..8]` spatial.
    pub projectors: Vec<WindowQFormerProjector>,
    pub image_newline: UniquePtr<MlxArray>,
    pub processor: AnyResProcessor,
    pub image_token_index: i32,
    /// The 4 SigLIP taps to collect (deepstack `L[9],L[15],L[21],L[27]`).
    pub deepstack_taps: Vec<i32>,
    /// Per stream: `(projector_index, tap_index, decoder_layer)`.
    pub stream_specs: Vec<(usize, usize, usize)>,
    pub feature_side: i32,
    pub base_tokens: i32,
    pub eos_token_id: i32,
    injection: RefCell<Option<StashedInjection>>,
}

impl Granite4VisionVLModel {
    /// Construct from loaded parts (the injection stash starts empty).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        text_model: GraniteMoeHybridModel,
        vision_tower: SigLipVisionModel,
        projectors: Vec<WindowQFormerProjector>,
        image_newline: UniquePtr<MlxArray>,
        processor: AnyResProcessor,
        image_token_index: i32,
        deepstack_taps: Vec<i32>,
        stream_specs: Vec<(usize, usize, usize)>,
        feature_side: i32,
        base_tokens: i32,
        eos_token_id: i32,
    ) -> Self {
        Self {
            text_model,
            vision_tower,
            projectors,
            image_newline,
            processor,
            image_token_index,
            deepstack_taps,
            stream_specs,
            feature_side,
            base_tokens,
            eos_token_id,
            injection: RefCell::new(None),
        }
    }

    /// Build the zeroed input embeddings and stash the 8 injection streams.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        infos: &[AnyResTileInfo],
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.input_embeddings(input_ids);

        // Collect the 4 deepstack taps once; spatial streams reuse the last tap.
        let taps = self
            .vision_tower
            .forward_collect_layers(pixel_values, &self.deepstack_taps);

        // Build the 8 packed streams.
        let mut targets: Vec<(usize, UniquePtr<MlxArray>)> =
            Vec::with_capacity(self.stream_specs.len());
        for &(proj_idx, tap_idx, layer) in &self.stream_specs {
            let proj_out = self.projectors[proj_idx].forward(taps[tap_idx].as_ref().unwrap());
            let packed = self.pack_streams(&proj_out, infos);
            mlxcel_core::eval(&packed);
            targets.push((layer, packed));
        }

        // Host-read the image-token positions to build the zero-multiplier and
        // the injection position mask.
        mlxcel_core::eval(input_ids);
        let seq_len = mlxcel_core::array_shape(input_ids)[1];
        let mut keep = vec![1.0f32; seq_len as usize];
        let mut mask = vec![0i32; seq_len as usize];
        for i in 0..seq_len as usize {
            let tok = mlxcel_core::slice(input_ids, &[0, i as i32], &[1, i as i32 + 1]);
            mlxcel_core::eval(&tok);
            if mlxcel_core::item_i32(&tok) == self.image_token_index {
                keep[i] = 0.0;
                mask[i] = 1;
            }
        }
        let keep_arr = mlxcel_core::from_slice_f32(&keep, &[1, seq_len, 1]);
        let keep_arr = mlxcel_core::astype(&keep_arr, mlxcel_core::array_dtype(&inputs_embeds));
        let zeroed = mlxcel_core::multiply(&inputs_embeds, &keep_arr);
        mlxcel_core::eval(&zeroed);

        let mask_arr = mlxcel_core::from_slice_i32(&mask, &[1, seq_len]);
        mlxcel_core::eval(&mask_arr);

        *self.injection.borrow_mut() = Some(StashedInjection {
            mask: mask_arr,
            targets,
        });

        InputEmbeddings {
            inputs_embeds: zeroed,
            attention_mask_4d: None,
        }
    }

    /// Pack one stream's per-tile features `(total_tiles, base_tokens, D)` into
    /// `(N_img_total, D)` (base tile rows + unpadded grid with a trailing
    /// `image_newline` column, per image). Same math as Granite Vision at grid
    /// side 12.
    fn pack_streams(&self, proj_out: &MlxArray, infos: &[AnyResTileInfo]) -> UniquePtr<MlxArray> {
        let d = *mlxcel_core::array_shape(proj_out).last().unwrap();
        let mut features: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(infos.len());
        let mut offset = 0i32;
        for info in infos {
            let n = info.num_tiles;
            let tiles = mlxcel_core::slice(
                proj_out,
                &[offset, 0, 0],
                &[offset + n, self.base_tokens, d],
            );
            offset += n;
            features.push(self.pack_one_image(&tiles, info, d));
        }
        match features.len() {
            1 => features.into_iter().next().unwrap(),
            _ => {
                let mut iter = features.into_iter();
                let first = iter.next().unwrap();
                iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 0))
            }
        }
    }

    fn pack_one_image(
        &self,
        tiles: &MlxArray,
        info: &AnyResTileInfo,
        d: i32,
    ) -> UniquePtr<MlxArray> {
        let side = self.feature_side;
        let base_tokens = self.base_tokens;
        let (nth, ntw) = (info.n_tiles_h, info.n_tiles_w);
        let n_grid = nth * ntw;
        let newline = mlxcel_core::astype(&self.image_newline, mlxcel_core::array_dtype(tiles));

        let base = mlxcel_core::slice(tiles, &[0, 0, 0], &[1, base_tokens, d]);
        let base = mlxcel_core::reshape(&base, &[base_tokens, d]);
        if n_grid == 0 {
            let nl = mlxcel_core::reshape(&newline, &[1, d]);
            return mlxcel_core::concatenate(&base, &nl, 0);
        }

        let grid = mlxcel_core::slice(tiles, &[1, 0, 0], &[1 + n_grid, base_tokens, d]);
        let grid = mlxcel_core::reshape(&grid, &[nth, ntw, side, side, d]);
        let grid = mlxcel_core::transpose_axes(&grid, &[4, 0, 2, 1, 3]);
        let gh = side * nth;
        let gw = side * ntw;
        let grid = mlxcel_core::reshape(&grid, &[d, gh, gw]);

        let (h, w) = unpadded_token_hw(info.orig_h, info.orig_w, nth, ntw, side);
        let unpad_rows = (info.orig_w as f64) / (info.orig_h as f64) > gw as f64 / gh as f64;
        let grid = if unpad_rows {
            let pad = (gh - h) / 2;
            mlxcel_core::slice(&grid, &[0, pad, 0], &[d, pad + h, gw])
        } else {
            let pad = (gw - w) / 2;
            mlxcel_core::slice(&grid, &[0, 0, pad], &[d, gh, pad + w])
        };

        let nl = mlxcel_core::reshape(&newline, &[d, 1, 1]);
        let nl = mlxcel_core::broadcast_to(&nl, &[d, h, 1]);
        let grid = mlxcel_core::concatenate(&grid, &nl, 2);
        let grid = mlxcel_core::reshape(&grid, &[d, h * (w + 1)]);
        let grid = mlxcel_core::transpose_axes(&grid, &[1, 0]);
        mlxcel_core::concatenate(&base, &grid, 0)
    }

    /// Prefill through the backbone with the stashed injection (or plain when
    /// there is nothing stashed / on decode).
    fn inject_forward(
        &self,
        input_embeddings: Option<&MlxArray>,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let Some(embeds) = input_embeddings else {
            return match seq_id {
                Some(id) => {
                    self.text_model
                        .forward_with_sequence_id(input_ids, Some(id), caches, mask)
                }
                None => self.text_model.forward(input_ids, caches, mask),
            };
        };
        match self.injection.borrow_mut().take() {
            Some(stash) => {
                let injection = HybridInjection {
                    visual_pos_mask: &stash.mask,
                    targets: stash.targets,
                };
                self.text_model
                    .forward_embeds_with_injection(embeds, Some(&injection), seq_id)
            }
            None => self
                .text_model
                .forward_embeds_with_injection(embeds, None, seq_id),
        }
    }
}

impl LanguageModel for Granite4VisionVLModel {
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
        self.inject_forward(input_embeddings, input_ids, None, caches, mask)
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
        self.inject_forward(input_embeddings, input_ids, seq_id, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.input_embeddings(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(&self.text_model)
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![self.eos_token_id]
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        false
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
    ) -> Option<mlxcel_core::generate::ModelStateSnapshot> {
        self.text_model.snapshot_sequence_state(seq_id, token_len)
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
    ) -> Result<(), String> {
        self.text_model.restore_sequence_state(seq_id, snapshot)
    }

    fn trim_internal_caches(&self, excess: i32) {
        self.text_model.trim_internal_caches(excess);
    }
}
