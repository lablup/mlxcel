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

//! Z1T (Extropic Z1T-0, `model_type: z1t`): an attention-free decoder with an
//! AFT-conv token mixer, Dynamic Tanh norms and fixed fan-in sparse
//! tanh-linears.
//!
//! Reference: `mlx_vlm/models/z1t` (Blaizzy/mlx-vlm#2166). There are no K/V
//! tensors: each layer keeps running pool sums plus a `ksize - 1` conv window,
//! owned per sequence through [`ModelOwnedSequenceState`] like RWKV-7, so
//! decode is O(1) per token and exact against the flat forward.

use std::path::Path;

use mlxcel_core::cache::{CachePool, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, route_batched_decode_with_ids};
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::model_owned::ModelOwnedSequenceState;

#[path = "z1t_batch.rs"]
mod batch;
#[path = "z1t_config.rs"]
mod config;
#[path = "z1t_layers.rs"]
mod layers;

pub use config::{Z1tConfig, Z1tQuantization};

pub use layers::Z1tLayerState;
use layers::{Block, DyT};

/// The first prefix in `candidates` whose `{prefix}.{probe}` tensor exists.
/// Accepts the raw JAX names the checkpoint ships and the names upstream
/// `z1t.py::sanitize` remaps them to.
fn resolve_prefix(weights: &WeightMap, candidates: &[String], probe: &str) -> Option<String> {
    candidates
        .iter()
        .find(|prefix| weights.contains_key(&format!("{prefix}.{probe}")))
        .cloned()
}

fn candidates(raw: &str, mapped: &str) -> Vec<String> {
    vec![
        raw.to_string(),
        mapped.to_string(),
        format!("language_model.{mapped}"),
    ]
}

/// Upstream `_build_pe`: interleaved `(sin, cos)` pairs, used only when a
/// checkpoint omits its `pe.pe` table.
fn sinusoidal_pe(dim: usize, max_seq: usize) -> UniquePtr<MlxArray> {
    let half = dim / 2;
    let mut values = vec![0.0f32; max_seq * dim];
    for pos in 0..max_seq {
        for i in 0..half {
            let freq = (-(1e4f64).ln() * i as f64 / half as f64).exp();
            let angle = pos as f64 * freq;
            values[pos * dim + 2 * i] = angle.sin() as f32;
            values[pos * dim + 2 * i + 1] = angle.cos() as f32;
        }
    }
    mlxcel_core::from_slice_f32(&values, &[max_seq as i32, dim as i32])
}

struct Z1tModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<Block>,
    norm: DyT,
    /// The positional table with one zero row appended, so positions past
    /// `max_position_embeddings` clamp onto zeros (upstream zero-extends).
    pe_ext: UniquePtr<MlxArray>,
    max_positions: usize,
}

impl Z1tModel {
    fn from_weights(weights: &WeightMap, config: &Z1tConfig) -> Result<Self, String> {
        let embed_prefix = resolve_prefix(
            weights,
            &candidates("embedding", "model.embed_tokens"),
            "weight",
        )
        .ok_or("Z1T embedding weight not found")?;
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &embed_prefix,
            config.group_size(),
            config.bits(),
        )?;
        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                let prefix = resolve_prefix(
                    weights,
                    &candidates(&format!("blocks.{i}"), &format!("model.layers.{i}")),
                    "norm1.weight",
                )
                .ok_or_else(|| format!("Z1T block {i} weights not found"))?;
                Block::from_weights(weights, &prefix, config)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let norm_prefix = resolve_prefix(weights, &candidates("norm", "model.norm"), "alpha")
            .ok_or("Z1T final norm weights not found")?;
        let norm = DyT::from_weights(weights, &norm_prefix, config.hidden_size)?;

        let (dim, max_positions) = (config.hidden_size, config.max_position_embeddings);
        let pe = ["pe.pe", "model.pe", "language_model.model.pe"]
            .iter()
            .find_map(|key| weights.get(*key).map(|w| mlxcel_core::copy(w)))
            .unwrap_or_else(|| sinusoidal_pe(dim, max_positions));
        let pe_shape = mlxcel_core::array_shape(&pe);
        if pe_shape != [max_positions as i32, dim as i32] {
            return Err(format!(
                "Z1T positional table has shape {pe_shape:?}, expected [{max_positions}, {dim}]"
            ));
        }
        let zero_row = mlxcel_core::zeros(&[1, dim as i32], mlxcel_core::array_dtype(&pe));
        let pe_ext = mlxcel_core::concatenate(&pe, &zero_row, 0);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            pe_ext,
            max_positions,
        })
    }

    /// Positional rows for `offsets.len()` rows of `len` tokens each.
    fn positions(&self, offsets: &[usize], len: usize) -> UniquePtr<MlxArray> {
        let index: Vec<i32> = offsets
            .iter()
            .flat_map(|&offset| (0..len).map(move |t| (offset + t).min(self.max_positions) as i32))
            .collect();
        let index = mlxcel_core::from_slice_i32(&index, &[offsets.len() as i32, len as i32]);
        mlxcel_core::take(&self.pe_ext, &index, 0)
    }

    /// Final-normed hidden states for every position (`row == None`) or only
    /// position `row`. `offsets` holds each row's consumed-token count; every
    /// layer still runs over all positions so the state advances fully.
    fn forward_rows(
        &self,
        input_ids: &MlxArray,
        states: &mut [Z1tLayerState],
        offsets: &[usize],
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        assert_eq!(
            states.len(),
            self.layers.len(),
            "Z1T state cardinality must match layer count"
        );
        let shape = mlxcel_core::array_shape(input_ids);
        assert_eq!(offsets.len(), shape[0] as usize, "Z1T offset/row mismatch");
        let len = shape[1] as usize;
        let x = self.embed_tokens.forward(input_ids);
        let pos = mlxcel_core::astype(&self.positions(offsets, len), mlxcel_core::array_dtype(&x));
        let mut h = mlxcel_core::add(&x, &pos);
        for (layer, state) in self.layers.iter().zip(states.iter_mut()) {
            h = layer.forward(&h, state);
            state.offset += len;
        }
        let h = mlxcel_core::generate::select_row(h, row);
        self.norm.forward(&h)
    }
}

/// Z1T language model with model-owned per-sequence streaming state.
pub struct Z1t {
    config: Z1tConfig,
    model: Z1tModel,
    lm_head: UnifiedLinear,
    sequence_state: ModelOwnedSequenceState<Z1tLayerState>,
    eos_token_ids: Vec<i32>,
}

impl Z1t {
    pub fn from_weights(weights: &WeightMap, config: Z1tConfig) -> Result<Self, String> {
        config.validate()?;
        let model = Z1tModel::from_weights(weights, &config)?;
        let head_prefix = resolve_prefix(weights, &candidates("clf", "lm_head"), "weight")
            .ok_or("Z1T head (clf) weight not found")?;
        let lm_head =
            UnifiedLinear::from_weights(weights, &head_prefix, config.group_size(), config.bits())?;
        let eos_token_ids = super::parse_optional_eos_token_ids(&config.eos_token_id);
        let internal = make_states(config.num_hidden_layers);
        Ok(Self {
            config,
            model,
            lm_head,
            sequence_state: ModelOwnedSequenceState::new(internal),
            eos_token_ids,
        })
    }

    pub fn load(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))?;
        let config: Z1tConfig = serde_json::from_str(&config_str)?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let mut model = Self::from_weights(&weights, config)?;
        model.set_eos_token_ids(crate::loading::read_eos_token_ids(model_dir));
        Ok(model)
    }

    pub(crate) fn set_eos_token_ids(&mut self, eos_token_ids: Vec<i32>) {
        if !eos_token_ids.is_empty() {
            self.eos_token_ids = eos_token_ids;
        }
    }

    fn fresh_states(&self) -> Vec<Z1tLayerState> {
        make_states(self.config.num_hidden_layers)
    }

    fn run(
        &self,
        input_ids: &MlxArray,
        states: &mut [Z1tLayerState],
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let offset = states.first().map_or(0, |state| state.offset);
        let h = self.model.forward_rows(input_ids, states, &[offset], row);
        self.lm_head.forward(&h)
    }

    /// Logits for every position (`row == None`) or only position `row`.
    fn forward_rows_for_sequence(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(input_ids)[1];
        if seq_id.is_none() && seq_len > 1 {
            // A multi-token call without an id starts an independent prompt.
            self.sequence_state.replace_internal(self.fresh_states());
        }
        if let Some(seq_id) = seq_id {
            self.sequence_state
                .with_existing_sequence_state(seq_id, |states| self.run(input_ids, states, row))
                .unwrap_or_else(|err| panic!("Z1T {err}"))
        } else {
            self.sequence_state
                .with_sequence_state(None, |states| self.run(input_ids, states, row))
        }
    }
}

fn make_states(num_layers: usize) -> Vec<Z1tLayerState> {
    (0..num_layers).map(|_| Z1tLayerState::default()).collect()
}

impl LanguageModel for Z1t {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // The KVCache slice is a scheduler placeholder; state is model-owned.
        self.forward_rows_for_sequence(input_ids, None, None)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.sequence_state.replace_internal(self.fresh_states());
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    /// No mode table: every layer is AFT-conv state, no KV (issue #2790).
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<mlxcel_core::cache::KVCacheMode>>> {
        crate::models::model_owned::table_less_storage_modes(self.config.num_hidden_layers, |_| {
            crate::models::model_owned::LayerKv::None
        })
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Pad tokens would enter the running pool sums.
    }

    fn supports_chunked_prefill(&self) -> bool {
        // Same no-id contract as RWKV-7: a multi-token call without a sequence
        // id resets the internal state, so the prompt stays in one call.
        false
    }

    fn supports_batching(&self) -> bool {
        true // Streaming state is model-owned and isolated by sequence id.
    }

    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.config.num_hidden_layers)
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        self.sequence_state
            .with_existing_sequence_state(seq_id, |state| {
                cache_pool.sync_paged_state_with_lengths(seq_id, &vec![0; state.len()])
            })?
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_with_ids(input_ids, None, batch_caches, mask)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        assert_eq!(shape.len(), 2, "Z1T batched input must have rank two");
        assert_eq!(
            shape[0] as usize,
            batch_caches.len(),
            "Z1T input/cache row mismatch"
        );
        if let Some(ids) = seq_ids {
            assert_eq!(
                ids.len(),
                batch_caches.len(),
                "Z1T input/sequence-id row mismatch"
            );
        }
        route_batched_decode_with_ids(self, input_ids, seq_ids, batch_caches, |batch_caches| {
            let ids = seq_ids.expect("Z1T model-owned batched decode requires sequence ids");
            self.sequence_state
                .validate_sequence_states(ids)
                .unwrap_or_else(|err| panic!("Z1T batched decode: {err}"));
            let initialized: Vec<Vec<bool>> = ids
                .iter()
                .map(|id| {
                    self.sequence_state
                        .with_sequence_state_ref(*id, |state| {
                            state.iter().map(Z1tLayerState::is_initialized).collect()
                        })
                        .expect("validated Z1T sequence state")
                })
                .collect();
            let mixed = initialized.windows(2).any(|rows| rows[0] != rows[1]);
            if shape[1] != 1 || mask.is_some() || mixed {
                let outputs: Vec<_> = batch_caches
                    .iter_mut()
                    .enumerate()
                    .map(|(row, caches)| {
                        let input = mlxcel_core::slice(
                            input_ids,
                            &[row as i32, 0],
                            &[row as i32 + 1, shape[1]],
                        );
                        self.forward_with_sequence_id(&input, Some(ids[row]), caches, mask)
                    })
                    .collect();
                return mlxcel_core::concatenate_many(
                    &outputs.iter().map(|row| &**row).collect::<Vec<_>>(),
                    0,
                );
            }
            self.sequence_state
                .with_batched_sequence_states(ids, |rows| {
                    assert!(self.batch_states_uniform(rows));
                    self.forward_batched_decode(input_ids, rows)
                })
                .unwrap_or_else(|err| panic!("Z1T batched decode: {err}"))
        })
    }

    fn reset_runtime_state(&self) {
        self.sequence_state.replace_internal(self.fresh_states());
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.fresh_states());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows_for_sequence(input_ids, seq_id, None)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        // Text-only model: decline the embeddings path.
        if input_embeddings.is_some() {
            return None;
        }
        Some(self.forward_rows_for_sequence(input_ids, seq_id, Some(last_pos)))
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "z1t_tests.rs"]
mod tests;
