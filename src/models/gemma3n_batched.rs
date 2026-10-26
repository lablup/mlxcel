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

//! Native single-token batching for Gemma 3n. Non-attention work remains
//! batched while each sequence retains its own concrete owner caches.

use mlxcel_core::batched_decode::{SplitAttention, SplitAttentionShape};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{
    DecoderLayer, Gemma3nAttention, Gemma3nLanguageModel, Gemma3nModel, apply_softcap,
    compute_magnitude, mean_arrays, normalize_magnitudes, normalize_magnitudes_from_idx,
    slice_altup_plane, slice_layer_input, split_altup_after_per_layer_update,
    use_fused_decode_path,
};

impl SplitAttention for Gemma3nAttention {
    fn split_shape(&self) -> SplitAttentionShape {
        SplitAttentionShape {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            scale: self.scale,
        }
    }

    fn qk_norm_heads(
        &self,
        q: UniquePtr<MlxArray>,
        k: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        (self.q_norm.forward(&q), self.k_norm.forward(&k))
    }

    fn rope_batched(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offsets: &[i32],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        (
            mlxcel_core::fast_rope_batched(q, self.head_dim, false, self.rope_theta, 1.0, offsets),
            mlxcel_core::fast_rope_batched(k, self.head_dim, false, self.rope_theta, 1.0, offsets),
        )
    }

    fn custom_row_attend(&self) -> bool {
        self.is_sliding
    }

    fn attend_row(
        &self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (keys, values) = cache.update_and_fetch(k, v);
        self.attend_cached_row(q, &keys, &values, mask)
    }
}

impl Gemma3nAttention {
    fn attend_cached_row(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let key_shape = mlxcel_core::array_shape(keys);
        let key_len = key_shape[2];
        let mask_key_len = mask
            .map(|value| *mlxcel_core::array_shape(value).last().unwrap_or(&key_len))
            .unwrap_or(key_len);
        let (window_keys, window_values) = if self.window_size > 0 && key_len > mask_key_len {
            let value_shape = mlxcel_core::array_shape(values);
            let start = key_len - mask_key_len;
            (
                Some(mlxcel_core::slice(
                    keys,
                    &[0, 0, start, 0],
                    &[key_shape[0], key_shape[1], key_len, key_shape[3]],
                )),
                Some(mlxcel_core::slice(
                    values,
                    &[0, 0, start, 0],
                    &[value_shape[0], value_shape[1], key_len, value_shape[3]],
                )),
            )
        } else {
            (None, None)
        };
        let keys = window_keys.as_deref().unwrap_or(keys);
        let values = window_values.as_deref().unwrap_or(values);
        let mask_ptr = mask.map_or(std::ptr::null(), |value| value as *const _);
        // SAFETY: `mask_ptr` is null or borrows `mask` for this call only;
        // `attention_from_ptr` does not retain the pointer.
        unsafe {
            mlxcel_core::layers::attention_from_ptr(
                queries,
                keys,
                values,
                self.scale,
                mask_ptr,
                0.0,
                self.window_size,
            )
        }
    }

    fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
        owner_offsets: &[i32],
        mask: Option<&MlxArray>,
        layer_idx: usize,
    ) -> UniquePtr<MlxArray> {
        if self.is_kv_shared_layer {
            return self.forward_shared_kv_batched(x, caches, owner_offsets, mask, layer_idx);
        }

        let shape = mlxcel_core::array_shape(x);
        let queries = self.q_proj.forward(x);
        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);
        let values = mlxcel_core::reshape(
            &values,
            &[shape[0], shape[1], self.num_kv_heads, self.head_dim],
        );
        let values = self.v_norm.forward(&values);
        let values = mlxcel_core::reshape(
            &values,
            &[shape[0], shape[1], self.num_kv_heads * self.head_dim],
        );
        let metadata = crate::models::batched_decode_cache::uniform_layer_metadata(
            caches,
            shape[1],
            self.window_size,
            "Gemma3n",
            layer_idx,
        );
        let attention = mlxcel_core::batched_decode::forward_split_attention(
            self, &queries, &keys, &values, caches, &metadata, mask,
        );
        self.o_proj.forward(&attention)
    }

    // Used by: every Gemma 3n KV-shared decoder layer. Owner K/V is read-only.
    fn forward_shared_kv_batched(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
        owner_offsets: &[i32],
        mask: Option<&MlxArray>,
        layer_idx: usize,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let queries = self.q_proj.forward(x);
        let queries = mlxcel_core::reshape(
            &queries,
            &[shape[0], shape[1], self.num_heads, self.head_dim],
        );
        let queries = self.q_norm.forward(&queries);
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let queries = mlxcel_core::fast_rope_batched(
            &queries,
            self.head_dim,
            false,
            self.rope_theta,
            1.0,
            owner_offsets,
        );

        let mut outputs = Vec::with_capacity(caches.len());
        for (row_idx, cache) in caches.iter_mut().enumerate() {
            let live_len = cache.live_len();
            let keys = cache.keys.as_ref().unwrap_or_else(|| {
                panic!(
                    "Gemma3n batched decode missing shared K at row {row_idx}, layer {layer_idx}"
                )
            });
            let values = cache.values.as_ref().unwrap_or_else(|| {
                panic!(
                    "Gemma3n batched decode missing shared V at row {row_idx}, layer {layer_idx}"
                )
            });
            let key_shape = mlxcel_core::array_shape(keys);
            let value_shape = mlxcel_core::array_shape(values);
            let keys = mlxcel_core::slice(
                keys,
                &[0, 0, 0, 0],
                &[key_shape[0], key_shape[1], live_len, key_shape[3]],
            );
            let values = mlxcel_core::slice(
                values,
                &[0, 0, 0, 0],
                &[value_shape[0], value_shape[1], live_len, value_shape[3]],
            );
            let query = slice_batch_row(&queries, row_idx);
            let row_mask = mask.map(|value| slice_mask_row(value, row_idx, shape[1]));
            outputs.push(self.attend_cached_row(&query, &keys, &values, row_mask.as_deref()));
        }
        let attention = concatenate_rows(outputs);
        let attention = mlxcel_core::transpose_axes(&attention, &[0, 2, 1, 3]);
        let attention = mlxcel_core::reshape(
            &attention,
            &[shape[0], shape[1], self.num_heads * self.head_dim],
        );
        self.o_proj.forward(&attention)
    }
}

impl DecoderLayer {
    fn forward_batched_decode(
        &self,
        x: &[UniquePtr<MlxArray>],
        caches: &mut [&mut KVCache],
        owner_offsets: &[i32],
        mask: Option<&MlxArray>,
        per_layer_input: &MlxArray,
        layer_idx: usize,
    ) -> Vec<UniquePtr<MlxArray>> {
        if use_fused_decode_path() {
            self.forward_stacked_batched(x, caches, owner_offsets, mask, per_layer_input, layer_idx)
        } else {
            self.forward_split_batched(x, caches, owner_offsets, mask, per_layer_input, layer_idx)
        }
    }

    fn forward_stacked_batched(
        &self,
        x: &[UniquePtr<MlxArray>],
        caches: &mut [&mut KVCache],
        owner_offsets: &[i32],
        mask: Option<&MlxArray>,
        per_layer_input: &MlxArray,
        layer_idx: usize,
    ) -> Vec<UniquePtr<MlxArray>> {
        let predictions = self.altup.predict_stacked(x);
        let active_prediction = slice_altup_plane(&predictions, self.altup_active_idx);
        let active_normed = self.input_layernorm.forward(&active_prediction);
        let laurel_output = self.laurel.forward(&active_normed);
        let attention = self.self_attn.forward_batched_decode(
            &active_normed,
            caches,
            owner_offsets,
            mask,
            layer_idx,
        );
        let attention = self.post_attention_layernorm.forward(&attention);
        let attention = mlxcel_core::add(&active_prediction, &attention);
        let attention = mlxcel_core::add(&attention, &laurel_output);
        let attention = mlxcel_core::multiply_scalar(&attention, std::f32::consts::FRAC_1_SQRT_2);
        let feed_forward = self.pre_feedforward_layernorm.forward(&attention);
        let feed_forward = self.mlp.forward(&feed_forward);
        let feed_forward = self.post_feedforward_layernorm.forward(&feed_forward);
        let activated = mlxcel_core::add(&attention, &feed_forward);
        let activated = mlxcel_core::astype(&activated, mlxcel_core::dtype::BFLOAT16);
        let corrected = self
            .altup
            .correct_stacked(&predictions, &active_prediction, &activated);
        let first = slice_altup_plane(&corrected, self.altup_active_idx);
        let first = if self.altup_correct_scale {
            mlxcel_core::multiply(&first, &self.altup.correct_output_scale)
        } else {
            first
        };
        let gate = self.per_layer_input_gate.forward(&first);
        let gated = mlxcel_core::compiled_geglu_approx_activation(&gate, per_layer_input);
        let projected = self.per_layer_projection.forward(&gated);
        let prediction = self.post_per_layer_input_norm.forward(&projected);
        split_altup_after_per_layer_update(&corrected, &prediction, self.altup.altup_num_inputs)
    }

    fn forward_split_batched(
        &self,
        x: &[UniquePtr<MlxArray>],
        caches: &mut [&mut KVCache],
        owner_offsets: &[i32],
        mask: Option<&MlxArray>,
        per_layer_input: &MlxArray,
        layer_idx: usize,
    ) -> Vec<UniquePtr<MlxArray>> {
        let predictions = self.altup.predict(x);
        let active_prediction = &predictions[self.altup_active_idx];
        let active_normed = self.input_layernorm.forward(active_prediction);
        let laurel_output = self.laurel.forward(&active_normed);
        let attention = self.self_attn.forward_batched_decode(
            &active_normed,
            caches,
            owner_offsets,
            mask,
            layer_idx,
        );
        let attention = self.post_attention_layernorm.forward(&attention);
        let attention = mlxcel_core::add(active_prediction, &attention);
        let attention = mlxcel_core::add(&attention, &laurel_output);
        let attention = mlxcel_core::multiply_scalar(&attention, std::f32::consts::FRAC_1_SQRT_2);
        let feed_forward = self.pre_feedforward_layernorm.forward(&attention);
        let feed_forward = self.mlp.forward(&feed_forward);
        let feed_forward = self.post_feedforward_layernorm.forward(&feed_forward);
        let activated = mlxcel_core::add(&attention, &feed_forward);
        let activated = mlxcel_core::astype(&activated, mlxcel_core::dtype::BFLOAT16);
        let corrected = self.altup.correct(&predictions, &activated);
        let first = &corrected[self.altup_active_idx];
        let first = if self.altup_correct_scale {
            mlxcel_core::multiply(first, &self.altup.correct_output_scale)
        } else {
            mlxcel_core::copy(first)
        };
        let gate = self.per_layer_input_gate.forward(&first);
        let gated = mlxcel_core::compiled_geglu_approx_activation(&gate, per_layer_input);
        let projected = self.per_layer_projection.forward(&gated);
        let prediction = self.post_per_layer_input_norm.forward(&projected);
        let mut result = Vec::with_capacity(corrected.len());
        let mut corrected = corrected.into_iter();
        if let Some(first) = corrected.next() {
            result.push(first);
        }
        result.extend(corrected.map(|plane| mlxcel_core::add(&plane, &prediction)));
        result
    }
}

impl Gemma3nLanguageModel {
    fn forward_batched_decode(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let input_shape = mlxcel_core::array_shape(input_ids);
        assert_eq!(
            input_shape,
            vec![batch_caches.len() as i32, 1],
            "Gemma3n native batched decode expects [B, 1] ids"
        );
        let owner_count = self.config.first_kv_shared_layer_idx();
        let owner_offsets: Vec<i32> = batch_caches
            .iter()
            .enumerate()
            .map(|(row_idx, caches)| {
                assert_eq!(
                    caches.len(),
                    owner_count,
                    "Gemma3n batched decode cache mismatch at row {row_idx}: expected {owner_count} owner layers, got {}",
                    caches.len()
                );
                let offset = caches[0].offset;
                for (cache_idx, cache) in caches.iter().enumerate().skip(1) {
                    assert_eq!(
                        cache.offset, offset,
                        "Gemma3n batched decode cache offset mismatch at row {row_idx}, layer {cache_idx}"
                    );
                }
                offset
            })
            .collect();

        let embeddings = self.embed_tokens.forward(input_ids);
        let hidden =
            mlxcel_core::multiply_scalar(&embeddings, (self.config.hidden_size as f32).sqrt());
        let token_inputs = self.get_per_layer_inputs(input_ids);
        let per_layer_inputs = self.project_per_layer_inputs(&hidden, &token_inputs);
        let target_magnitude = compute_magnitude(&hidden);
        let mut planes = vec![mlxcel_core::copy(&hidden)];
        planes.extend(
            self.altup_projections
                .iter()
                .map(|projection| projection.forward(&hidden)),
        );
        normalize_magnitudes(&mut planes, &target_magnitude);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let cache_idx = self.layer_idx_to_cache_idx[layer_idx];
            let mut layer_caches = collect_mapped_layer_caches(batch_caches, cache_idx, layer_idx);
            let layer_input = slice_layer_input(
                &per_layer_inputs,
                layer_idx as i32,
                input_shape[0],
                1,
                self.config.hidden_size_per_layer_input as i32,
            );
            planes = layer.forward_batched_decode(
                &planes,
                &mut layer_caches,
                &owner_offsets,
                mask,
                &layer_input,
                layer_idx,
            );
        }

        let target_magnitude = compute_magnitude(&planes[0]);
        for (plane, projection) in planes
            .iter_mut()
            .skip(1)
            .zip(&self.altup_unembed_projections)
        {
            *plane = projection.forward(plane);
        }
        normalize_magnitudes_from_idx(&mut planes, 1, &target_magnitude);
        let hidden = mean_arrays(&planes);
        let hidden = self.norm.forward(&hidden);
        let hidden = if self.embed_tokens.is_quantized() {
            hidden
        } else {
            mlxcel_core::astype(
                &hidden,
                mlxcel_core::array_dtype(self.embed_tokens.weight()),
            )
        };
        let mut logits = self.lm_head(&hidden);
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = apply_softcap(&logits, cap);
        }
        logits
    }
}

impl Gemma3nModel {
    pub(super) fn forward_batched_impl(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.language_model
            .forward_batched_decode(input_ids, batch_caches, mask)
    }
}

fn collect_mapped_layer_caches<'a>(
    batch_caches: &'a mut [&mut [KVCache]],
    cache_idx: usize,
    layer_idx: usize,
) -> Vec<&'a mut KVCache> {
    batch_caches
        .iter_mut()
        .enumerate()
        .map(|(row_idx, caches)| {
            let available = caches.len();
            assert!(
                cache_idx < available,
                "Gemma3n batched decode cache mismatch at row {row_idx}, layer {layer_idx}: owner cache {cache_idx} is missing from {available} caches"
            );
            &mut caches[cache_idx]
        })
        .collect()
}

fn slice_batch_row(array: &MlxArray, row: usize) -> UniquePtr<MlxArray> {
    mlxcel_core::slice(
        array,
        &[row as i32, 0, 0, 0],
        &[row as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
    )
}

fn slice_mask_row(mask: &MlxArray, row: usize, query_len: i32) -> UniquePtr<MlxArray> {
    let row = mlxcel_core::slice(
        mask,
        &[row as i32, 0, 0],
        &[row as i32 + 1, query_len, i32::MAX],
    );
    mlxcel_core::squeeze_axis(&row, 0)
}

fn concatenate_rows(rows: Vec<UniquePtr<MlxArray>>) -> UniquePtr<MlxArray> {
    let rows: Vec<&MlxArray> = rows.iter().map(|row| &**row).collect();
    mlxcel_core::concatenate_many(&rows, 0)
}
