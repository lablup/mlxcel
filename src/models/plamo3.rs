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

//! PLaMo 3 (`plamo3`, `Plamo3ForCausalLM`, Preferred Networks with NICT).
//!
//! Structure from mlx-lm's `mlx_lm/models/plamo3.py` (ml-explore/mlx-lm#1234);
//! RoPE from the checkpoint's own `modeling_plamo.py`, which mlx-lm does not
//! follow:
//!
//! - A dense decoder interleaving sliding and full attention: layer `i` is full
//!   iff `(i + 1) % sliding_window_pattern == 0`. Sliding layers hold a
//!   `RotatingKVCache(window_size)` and full layers a `KVCache`, carried in
//!   model-owned state through the gemma3 `Cache` enum (the AFMoE / Kolibri-1
//!   arrangement, shared through [`crate::models::sliding_hybrid_lm`]).
//! - The sliding prefill mask spans `window_size + 1` keys, as mlx-lm builds it
//!   (`create_attention_mask(..., window_size=self.window_size + 1)`), while the
//!   ring and its key accounting stay at `window_size`.
//! - Offset RMSNorms (`rms_norm(x, weight + offset)`) with PLaMo 2's offsets,
//!   folded into the stored weights, plus per-head `q_norm` / `k_norm` (offset
//!   1) that run before RoPE.
//! - Sliding layers rotate with plain `rope_local_theta`. Full layers rotate
//!   with `rope_theta` and, when `rope_scaling_factor != 1`, the reference's
//!   YaRN (`factor`, `original = initial_context_length`, `beta_fast` 32,
//!   `beta_slow` 1, `truncate: false`, concentration `0.1 ln(factor) + 1`).
//!   `pfnet/plamo-3-610m-fin-instruct` ships factor 256 over 4096. The table
//!   goes through [`RopeScalingKind::resolve`], so `--rope-scaling none` drops
//!   it (which is how the mlx-lm comparison, which ignores YaRN, is run).

use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask_with_window_full, prefill_causal_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

use crate::models::gemma3::Cache;
use crate::models::gpt2::validate_embedding_table;
use crate::models::model_owned::{KvCacheLayerModes, ModelOwnedSequenceState};
use crate::models::plamo2::{FINAL_NORM_OFFSET, offset_rms_norm};
pub use crate::models::plamo3_config::{ModelArgs, is_full_attention};
use crate::models::plamo3_layers::{DecoderLayer, LayerRope};
use crate::models::rope_utils::RopeScalingKind;

/// Preferred Networks PLaMo 3.
pub struct Plamo3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    /// Ring size of the sliding caches (`window_size`).
    pub(crate) sliding_window: i32,
    full_attention_index: Option<usize>,
    sliding_index: Option<usize>,
    pub(crate) eos_token_ids: Vec<i32>,
    pub(crate) sequence_state: ModelOwnedSequenceState<Cache>,
    pub(crate) kv_cache_layer_modes: KvCacheLayerModes,
}

/// The sliding-layer prefill mask: a `window + 1` band over the keys a
/// `RotatingKVCache(window)` returns (at most `window - 1` retained prior keys
/// plus every new one). `None` for a single-token step.
pub(crate) fn sliding_prefill_mask(
    seq_len: i32,
    live_len: i32,
    window: i32,
) -> Option<UniquePtr<MlxArray>> {
    if seq_len <= 1 {
        return None;
    }
    let kept_prior = if live_len <= 0 {
        0
    } else {
        live_len.min((window - 1).max(0))
    };
    Some(create_causal_mask_with_window_full(
        seq_len,
        kept_prior,
        Some(window.saturating_add(1)),
    ))
}

impl Plamo3Model {
    fn project(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.norm.forward(h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }

    pub(crate) fn forward_batched_decode_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Vec<Cache>],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            let mut rows: Vec<_> = caches.iter_mut().map(|state| &mut state[i]).collect();
            h = layer.forward_batched_decode(&h, &mut rows);
        }
        self.project(&h)
    }

    fn forward_rows(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];
        let live = |idx: Option<usize>, caches: &mut [Cache]| {
            idx.map_or(0, |i| caches[i].as_interface().live_len())
        };
        let full_mask = if let (true, Some(full_idx)) = (seq_len > 1, self.full_attention_index) {
            let attention = self.layers[full_idx].mixer.prefill_attention();
            let live = live(self.full_attention_index, caches);
            prefill_causal_mask(
                seq_len,
                live,
                shape[0],
                attention,
                mlxcel_core::array_dtype(&h),
            )
        } else {
            None
        };
        // No sliding layer (fewer layers than the pattern): no windowed mask.
        let sliding_mask = self.sliding_index.and_then(|_| {
            sliding_prefill_mask(
                seq_len,
                live(self.sliding_index, caches),
                self.sliding_window,
            )
        });
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.is_full {
                full_mask.as_deref()
            } else {
                sliding_mask.as_deref()
            };
            h = layer.forward(&h, caches[i].as_interface(), mask);
        }
        let h = mlxcel_core::generate::select_row(h, row);
        self.project(&h)
    }

    pub(crate) fn make_internal_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| {
                if layer.is_full {
                    let mode = self.kv_cache_layer_modes.mode_for_layer(layer_idx);
                    Cache::Standard(KVCache::new_with_mode(mode))
                } else {
                    let mode = self
                        .kv_cache_layer_modes
                        .rotating_mode_for_layer(layer_idx, self.sliding_window);
                    Cache::Rotating(RotatingKVCache::new_with_mode(self.sliding_window, mode))
                }
            })
            .collect()
    }

    /// The single-sequence forward against the model-owned caches of
    /// `seq_id`. An id-less multi-token forward starts an independent prompt.
    pub(crate) fn forward_for_sequence(
        &self,
        input: &MlxArray,
        seq_id: Option<SequenceId>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(input)[1];
        if seq_id.is_none() && seq_len > 1 {
            self.sequence_state
                .replace_internal(self.make_internal_caches());
        }
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| self.forward_rows(input, caches, row),
        )
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        args.validate()?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args))
    }

    /// The full-attention rotation: the reference's YaRN block (or none)
    /// resolved through the shared path, so `--rope-scaling` and
    /// `--rope-freq-base` reach it.
    pub(crate) fn full_attention_rope(args: &ModelArgs) -> Result<RopeScalingKind, String> {
        let spec = args.full_attention_rope_spec();
        let base = args.full_rope_theta();
        let kind = RopeScalingKind::resolve(
            spec.as_ref(),
            args.head_dim,
            base,
            Some(args.max_position_embeddings as f32),
            "PLaMo 3",
        );
        if spec.is_some()
            && crate::models::rope_overrides::installed().is_none()
            && !matches!(kind, RopeScalingKind::Yarn { .. })
        {
            return Err(
                "PLaMo 3 rope_scaling_factor declares YaRN on the full-attention layers, but \
                 the YaRN table could not be built"
                    .to_string(),
            );
        }
        Ok(kind)
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;
        let (gs, bits) = (args.group_size(), args.bits());
        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;
        validate_embedding_table(
            &embed_tokens,
            "model.embed_tokens",
            args.vocab_size,
            "vocab_size",
            args.hidden_size,
            "hidden_size",
        )?;
        let sliding_window = crate::models::model_owned::cache_window(
            "PLaMo 3",
            args.sliding_window(),
            args.sliding_index().is_some(),
        )?;
        let full_rope = Self::full_attention_rope(args)?;
        let full_base = crate::models::rope_overrides::resolve_base(args.full_rope_theta());
        let layers = (0..args.num_hidden_layers)
            .map(|i| {
                let rope = if args.is_full_layer(i) {
                    LayerRope::from_kind(full_rope.duplicate(), full_base, args.head_dim)?
                } else {
                    // The reference's sliding layers are always plain RoPE.
                    LayerRope::plain(args.rope_local_theta)
                };
                DecoderLayer::from_weights(weights, args, i, rope)
            })
            .collect::<Result<Vec<_>, _>>()?;
        crate::models::afmoe::validate_norm(weights, "model.norm.weight", args.hidden_size)?;
        let norm = offset_rms_norm(
            weights,
            "model.norm.weight",
            args.rms_norm_eps,
            FINAL_NORM_OFFSET,
        )?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?)
        };
        let internal_caches = layers
            .iter()
            .map(|layer| {
                if layer.is_full {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(sliding_window))
                }
            })
            .collect();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window,
            full_attention_index: args.full_attention_index(),
            sliding_index: args.sliding_index(),
            eos_token_ids: args.eos_token_ids(),
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
        })
    }
}

#[cfg(test)]
#[path = "plamo3_tests.rs"]
mod tests;
