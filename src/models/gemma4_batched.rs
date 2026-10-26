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

//! Native single-token batching for Gemma 4.
//!
//! The decoder keeps embeddings, projections, norms, dense/MoE feed-forward
//! work and the tied LM head batched. Attention alone is split by row because
//! every scheduler sequence owns a distinct mixed full/sliding cache stack.

use std::collections::HashMap;

use mlxcel_core::{MlxArray, UniquePtr};

use super::{
    Attention, AttentionProjection, Cache, DecoderLayer, Gemma4Model, Gemma4TextModel, SubopTimer,
    slice_layer_input,
};

struct SharedKvRow {
    keys: UniquePtr<MlxArray>,
    values: UniquePtr<MlxArray>,
    offset: i32,
}

struct BatchedSharedKv {
    rows: Vec<SharedKvRow>,
}

impl Attention {
    fn rotate_batched_decode(&self, x: &MlxArray, offsets: &[i32]) -> UniquePtr<MlxArray> {
        if let Some(freqs) = self.proportional_rope_freqs.as_ref() {
            mlxcel_core::rope_proportional::apply_proportional_rope_batched(
                x,
                self.head_dim,
                self.proportional_partial_rotary_factor,
                offsets,
                Some(freqs),
            )
        } else {
            mlxcel_core::fast_rope_batched(x, self.rope_dims, false, self.rope_theta, 1.0, offsets)
        }
    }

    fn heads_batched_decode(
        &self,
        projected: &MlxArray,
        norm: &mlxcel_core::layers::RMSNorm,
        heads: i32,
        offsets: &[i32],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(projected);
        let projected =
            mlxcel_core::reshape(projected, &[shape[0], shape[1], heads, self.head_dim]);
        let projected = norm.forward(&projected);
        let projected = mlxcel_core::transpose_axes(&projected, &[0, 2, 1, 3]);
        self.rotate_batched_decode(&projected, offsets)
    }

    /// One batched Q/K/V projection and one output projection, with the cache
    /// append and attention reduction kept row-local.
    fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut Cache],
        shared_kv: Option<&BatchedSharedKv>,
    ) -> (UniquePtr<MlxArray>, Option<BatchedSharedKv>) {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0] as usize;
        debug_assert_eq!(shape[1], 1);
        debug_assert_eq!(batch, caches.len());

        let offsets: Vec<i32> = match shared_kv {
            Some(shared) => shared.rows.iter().map(|row| row.offset).collect(),
            None => caches.iter().map(|cache| cache.offset()).collect(),
        };
        let (raw_queries, raw_keys, raw_values) = match &self.projection {
            AttentionProjection::Fused(projection) => {
                let (q, k, v) = projection.forward(x);
                (q, Some(k), Some(v))
            }
            AttentionProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => {
                let q = q_proj.forward(x);
                let k = k_proj.forward(x);
                let v = v_proj.as_ref().map(|projection| projection.forward(x));
                (q, Some(k), v)
            }
            AttentionProjection::KvShared { q_proj } => (q_proj.forward(x), None, None),
        };
        let queries = self.heads_batched_decode(&raw_queries, &self.q_norm, self.n_heads, &offsets);

        let mut outputs = Vec::with_capacity(batch);
        let mut stored_rows = self.store_full_length_kv.then(|| Vec::with_capacity(batch));
        if let Some(shared) = shared_kv {
            debug_assert_eq!(shared.rows.len(), batch);
            for (row_idx, (cache, row)) in caches.iter_mut().zip(&shared.rows).enumerate() {
                cache.as_interface().set_offset(row.offset);
                let query = slice_batch_row(&queries, row_idx);
                outputs.push(self.attend(&query, &row.keys, &row.values, None, false, None));
            }
        } else {
            let raw_keys = raw_keys.expect("non-shared Gemma 4 attention needs K");
            let k_norm = self
                .k_norm
                .as_ref()
                .expect("non-shared Gemma 4 attention needs k_norm");
            let keys = self.heads_batched_decode(&raw_keys, k_norm, self.n_kv_heads, &offsets);
            let raw_values = raw_values.as_deref().unwrap_or(&raw_keys);
            let values = mlxcel_core::reshape(
                raw_values,
                &[shape[0], shape[1], self.n_kv_heads, self.head_dim],
            );
            let values = self.v_norm.forward(&values);
            let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

            for (row_idx, cache) in caches.iter_mut().enumerate() {
                let query = slice_batch_row(&queries, row_idx);
                let new_keys = slice_batch_row(&keys, row_idx);
                let new_values = slice_batch_row(&values, row_idx);
                let offset = offsets[row_idx];
                let (keys, values) = cache.as_interface().update_and_fetch(new_keys, new_values);
                outputs.push(self.attend(&query, &keys, &values, None, false, None));
                if let Some(rows) = stored_rows.as_mut() {
                    rows.push(SharedKvRow {
                        keys,
                        values,
                        offset,
                    });
                }
            }
        }

        let attention = concatenate_rows(outputs);
        (
            self.project_output(&attention, shape[0], shape[1]),
            stored_rows.map(|rows| BatchedSharedKv { rows }),
        )
    }
}

impl DecoderLayer {
    fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut Cache],
        per_layer_input: Option<&MlxArray>,
        shared_kv: Option<&BatchedSharedKv>,
        layer_idx: usize,
        profile_subops: bool,
    ) -> (UniquePtr<MlxArray>, Option<BatchedSharedKv>) {
        let mut timer = SubopTimer::new(profile_subops, layer_idx);
        let attention_input = self.input_layernorm.forward(x);
        timer.tick("input_layernorm", &attention_input);
        let (attention, stored_kv) =
            self.self_attn
                .forward_batched_decode(&attention_input, caches, shared_kv);
        timer.tick("self_attn", &attention);
        let hidden =
            self.finish_layer(x, &attention, per_layer_input, &mut timer, false, layer_idx);
        (hidden, stored_kv)
    }
}

impl Gemma4TextModel {
    pub(super) fn forward_batched_decode_with_caches(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [Vec<Cache>],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let batch = shape[0];
        debug_assert_eq!(shape[1], 1);
        debug_assert_eq!(batch as usize, batch_caches.len());

        let embeddings = self.embed_tokens.forward(input_ids);
        let mut hidden =
            mlxcel_core::multiply_scalar(&embeddings, (self.config.hidden_size as f32).sqrt());
        let per_layer_inputs = if self.config.hidden_size_per_layer_input > 0 {
            let token_inputs = self.get_per_layer_inputs(input_ids);
            Some(self.project_per_layer_inputs(&hidden, Some(&token_inputs)))
        } else {
            None
        };

        let profile_per_layer = std::env::var_os("MLXCEL_PROFILE_PER_LAYER").is_some();
        let profile_subops = std::env::var_os("MLXCEL_PROFILE_LAYER_SUBOPS").is_some();
        let mut shared_kv_store: HashMap<usize, BatchedSharedKv> = HashMap::new();
        let layer_count = self.layers.len();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let shared_kv = layer
                .self_attn
                .kv_shared_layer_index
                .and_then(|source| shared_kv_store.get(&source));
            let layer_input = per_layer_inputs.as_ref().map(|inputs| {
                slice_layer_input(
                    inputs,
                    layer_idx as i32,
                    batch,
                    1,
                    self.config.hidden_size_per_layer_input as i32,
                )
            });
            let mut layer_caches: Vec<&mut Cache> = batch_caches
                .iter_mut()
                .map(|caches| &mut caches[layer_idx])
                .collect();
            let started = profile_per_layer.then(std::time::Instant::now);
            let (next_hidden, stored_kv) = layer.forward_batched_decode(
                &hidden,
                &mut layer_caches,
                layer_input.as_deref(),
                shared_kv,
                layer_idx,
                profile_subops,
            );
            hidden = next_hidden;
            if let Some(started) = started {
                mlxcel_core::eval(&hidden);
                eprintln!(
                    "[LAYER {layer_idx:02}] type={} seq_len=1 ms={:.3}",
                    layer.layer_type,
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
            if let Some(stored_kv) = stored_kv {
                shared_kv_store.insert(layer_idx, stored_kv);
            }
            mlxcel_core::utils::pipeline_hint(&hidden, layer_idx, layer_count);
        }
        self.norm.forward(&hidden)
    }
}

impl Gemma4Model {
    pub(super) fn forward_batched_decode_with_caches(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [Vec<Cache>],
    ) -> UniquePtr<MlxArray> {
        let hidden = self
            .text_model
            .forward_batched_decode_with_caches(input_ids, batch_caches);
        let mut logits = self.text_model.embed_tokens.as_linear(&hidden);
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = mlxcel_core::compiled_softcap(&logits, cap);
        }
        logits
    }
}

fn slice_batch_row(array: &MlxArray, row: usize) -> UniquePtr<MlxArray> {
    mlxcel_core::slice(
        array,
        &[row as i32, 0, 0, 0],
        &[row as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
    )
}

fn concatenate_rows(rows: Vec<UniquePtr<MlxArray>>) -> UniquePtr<MlxArray> {
    let rows: Vec<&MlxArray> = rows
        .iter()
        .map(|row| {
            row.as_ref()
                .expect("Gemma 4 batched decode row is non-null")
        })
        .collect();
    mlxcel_core::concatenate_many(&rows, 0)
}
