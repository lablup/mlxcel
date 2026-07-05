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

//! Hunyuan-VL Vision-Language Model.
//!
//! Composes the Hunyuan ViT (with its `perceive` merger, whose output rows
//! already include the newline column and the begin / end framing), the
//! smart-resize processor, and the Hunyuan XD-RoPE text decoder. The prompt
//! carries `mh * (mw + 1) + 2` `<|image_token|>` placeholders per image and the
//! merger rows scatter onto them 1:1 (LLaVA merge). Prefill position ids are
//! 4D `[P, T, H, W]`: `P` stays sequential everywhere; inside an image run
//! (starting one past the first placeholder, spanning `mh * (mw + 1)` tokens)
//! `W` cycles `0..=mw` per row (the newline slot takes `w = mw`), `H` repeats
//! per row, and `T` is the image index. Decode uses sequential positions on
//! all axes (`rope_deltas = 0`).
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/hunyuan_vl/hunyuan_vl.py>.

use super::{encoders, merge, processors};
use crate::LanguageModel;
use crate::models::hunyuan_vl::HunyuanVlTextModel;
use crate::multimodal::batched_dispatch::forward_batched_with_seq_ids_dispatch;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::DecodeBatchContext;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct HunyuanVlModel {
    pub text_model: HunyuanVlTextModel,
    pub vision_encoder: encoders::hunyuan_vl::HunyuanVlVisionEncoder,
    pub processor: processors::hunyuan_vl::HunyuanVlProcessor,
    pub image_token_id: i32,
    pub image_start_token_id: i32,
    pub image_end_token_id: i32,
    pub image_newline_token_id: i32,
    pub spatial_merge_size: usize,
}

impl HunyuanVlModel {
    /// Encode images, merge merger rows at placeholder positions, and store
    /// the 4D XD-RoPE prefill position ids on the text model (fallback slot).
    pub fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);
        let image_features = self.vision_encoder.forward_with_grid(&pv, grid_thw);

        let merged = merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        );

        let position_ids = self.compute_xdrope_positions(input_ids, grid_thw);
        // Decode continues at the sequential cache offset on all axes.
        self.text_model.set_mrope_state(position_ids, 0);
        merged
    }

    /// 4D `[P, T, H, W]` prefill positions, `[4, 1, seq]`.
    fn compute_xdrope_positions(
        &self,
        input_ids: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::eval(input_ids);
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1] as usize;

        let mut tokens = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let tok = mlxcel_core::slice(input_ids, &[0, i as i32], &[1, i as i32 + 1]);
            mlxcel_core::eval(&tok);
            tokens.push(mlxcel_core::item_i32(&tok));
        }

        let merge_size = self.spatial_merge_size as i32;
        let mut p_idx: Vec<i32> = (0..seq_len as i32).collect();
        let mut t_idx = p_idx.clone();
        let mut h_idx = p_idx.clone();
        let mut w_idx = p_idx.clone();
        let _ = &mut p_idx; // P stays sequential everywhere.

        let mut image_index = 0i32;
        let mut i = 0usize;
        while i < seq_len {
            if tokens[i] == self.image_token_id {
                // One image run: begin row at `i`, then mh * (mw + 1) grid rows,
                // then the end row. The grid pattern starts one past the begin.
                if let Some(&(_t, gh, gw)) = grid_thw.get(image_index as usize) {
                    let mh = gh / merge_size;
                    let mw = gw / merge_size;
                    let token_num = (mh * (mw + 1)) as usize;
                    let start = i + 1;
                    for k in 0..token_num.min(seq_len.saturating_sub(start)) {
                        let row = k as i32 / (mw + 1);
                        let col = k as i32 % (mw + 1);
                        h_idx[start + k] = row;
                        w_idx[start + k] = col;
                        t_idx[start + k] = image_index;
                    }
                }
                // Skip the whole placeholder run of this image.
                while i < seq_len && tokens[i] == self.image_token_id {
                    i += 1;
                }
                image_index += 1;
                continue;
            }
            i += 1;
        }

        let total = seq_len as i32;
        let mk = |v: &[i32]| mlxcel_core::from_slice_i32(v, &[1, 1, total]);
        let p = mk(&p_idx);
        let t = mk(&t_idx);
        let h = mk(&h_idx);
        let w = mk(&w_idx);
        let pt = mlxcel_core::concatenate(p.as_ref().unwrap(), t.as_ref().unwrap(), 0);
        let pth = mlxcel_core::concatenate(pt.as_ref().unwrap(), h.as_ref().unwrap(), 0);
        mlxcel_core::concatenate(pth.as_ref().unwrap(), w.as_ref().unwrap(), 0)
    }

    pub fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.text_model.bind_mrope_state_to_sequence(seq_id);
    }
}

impl LanguageModel for HunyuanVlModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_impl(input_ids, None, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_impl(input_ids, input_embeddings, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_for_sequence(input_ids, None, caches, mask, seq_id)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_for_sequence(input_ids, input_embeddings, caches, mask, seq_id)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.text_model.release_mrope_sequence(seq_id);
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        forward_batched_with_seq_ids_dispatch(
            &self.text_model,
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.get_embed_tokens(input_ids))
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

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        // Image placeholder / framing ids are input-alignment only.
        vec![
            self.image_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.image_newline_token_id,
        ]
    }
}
