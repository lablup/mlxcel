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

//! Unlimited-OCR Vision-Language Model wrapper.
//!
//! Unlimited-OCR (`baidu/Unlimited-OCR`) recombines the DeepSeek-OCR stack: the
//! SAM ViT-B + CLIP-L/14-224 encoders, the linear projector, and the 12-layer
//! DeepSeek MoE decoder are identical, so the whole vision path (encoders,
//! processor, `image_newline` mosaic, `<image>` scatter) is reused verbatim from
//! [`DeepSeekOcrVlModel`]. The one genuinely new piece is the decode-time cache:
//! attention runs against a per-layer [`RingSlidingKVCache`] that keeps the full
//! prefill KV permanently and rotates only the most recent
//! `sliding_window_size` decode tokens (see the cache module for the exact
//! prefill / warmup / steady-state contract).
//!
//! The ring caches are model-owned (kept in [`ModelOwnedSequenceState`]) rather
//! than in the external `KVCache` slice, so the standard `KVCache`-based padded
//! prefill, prompt-cache, and distributed-handoff paths are declined for this
//! family: [`LanguageModel::supports_padded_prefill`] and
//! [`LanguageModel::supports_batching`] both return `false`, mirroring how the
//! internal-state (NemotronH-style) models are handled today.

use super::deepseekocr::DeepSeekOcrVlModel;
use crate::models::deepseek::{Attention, TransformerBlock};
use crate::models::model_owned::ModelOwnedSequenceState;
use mlxcel_core::cache::{RingSlidingKVCache, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Unlimited-OCR VLM: the DeepSeek-OCR vision + text stack driven by ring
/// sliding decode caches.
pub struct UnlimitedOcrVlModel {
    /// Reused DeepSeek-OCR runtime (SAM + CLIP + projector + DeepSeek MoE
    /// decoder + processor). Its vision path and decoder weights are consumed
    /// directly; only the decode cache differs.
    pub inner: DeepSeekOcrVlModel,
    /// Sliding window size `W` (`sliding_window_size`, 128 for the reference
    /// checkpoint).
    pub window: i32,
    /// Per-layer ring caches, kept model-owned so the external `KVCache` slice
    /// stays unused.
    sequence_state: ModelOwnedSequenceState<RingSlidingKVCache>,
}

impl UnlimitedOcrVlModel {
    /// Wrap a loaded DeepSeek-OCR runtime with the ring decode cache.
    #[must_use]
    pub fn new(inner: DeepSeekOcrVlModel, window: i32) -> Self {
        let num_layers = inner.text_model.layers.len();
        let internal = (0..num_layers)
            .map(|_| RingSlidingKVCache::new(window))
            .collect();
        Self {
            inner,
            window,
            sequence_state: ModelOwnedSequenceState::new(internal),
        }
    }

    fn make_ring_caches(&self) -> Vec<RingSlidingKVCache> {
        (0..self.inner.text_model.layers.len())
            .map(|_| RingSlidingKVCache::new(self.window))
            .collect()
    }

    /// Full decoder forward with ring caches. `embeds` (when `Some`) is the
    /// merged image+text prefill embedding; otherwise `input_ids` is embedded.
    fn forward_ring(
        &self,
        input_ids: &MlxArray,
        embeds: Option<&MlxArray>,
        caches: &mut [RingSlidingKVCache],
    ) -> UniquePtr<MlxArray> {
        let tm = &self.inner.text_model;
        let mut h = match embeds {
            Some(e) => mlxcel_core::copy(e),
            None => tm.embed_tokens_forward(input_ids),
        };
        for (i, layer) in tm.layers.iter().enumerate() {
            h = Self::ring_block_forward(layer, &h, &mut caches[i]);
        }
        let h = tm.norm.forward(&h);
        tm.lm_head.forward(&h)
    }

    /// One transformer block with ring attention (pre-norm attention + MoE/MLP).
    fn ring_block_forward(
        layer: &TransformerBlock,
        x: &MlxArray,
        cache: &mut RingSlidingKVCache,
    ) -> UniquePtr<MlxArray> {
        let normed = layer.input_layernorm.forward(x);
        let attn_out = Self::ring_attention(&layer.self_attn, &normed, cache);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = layer.post_attention_layernorm.forward(&h);
        let ff_out = layer.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    /// DeepSeek MHA attention against the ring cache.
    ///
    /// Mirrors [`Attention::forward`] but sources absolute RoPE positions and
    /// the attention K/V window from [`RingSlidingKVCache`]: Q and the fresh K
    /// are rotated at the cache's absolute offset, the fresh K/V is written into
    /// the ring (append during prefill/warmup, in-place overwrite once the
    /// window is full), and attention then runs over the full physical window
    /// (causal during prefill, maskless during single-token decode).
    fn ring_attention(
        attn: &Attention,
        x: &MlxArray,
        cache: &mut RingSlidingKVCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = attn.q_proj.forward(x);
        let k = attn.k_proj.forward(x);
        let v = attn.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, attn.num_heads, attn.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, attn.num_kv_heads, attn.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, attn.num_kv_heads, attn.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Absolute RoPE position for this step (kept increasing past the ring
        // window). Read before the write.
        let offset = cache.offset();
        let q = mlxcel_core::fast_rope(&q, attn.rope_dims, false, attn.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, attn.rope_dims, false, attn.rope_base, 1.0, offset);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        let attn_out = if cache.prefill_causal(l) {
            // Prefill: standard causal masking over the retained prompt.
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, attn.scale, 0.0, 0)
        } else {
            // Single-token decode: attend over the whole physical window, no
            // mask (every retained entry is attendable).
            // SAFETY: null mask pointer selects the maskless attention path.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &cache_k,
                    &cache_v,
                    attn.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            }
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, attn.num_heads * attn.head_dim]);
        attn.o_proj.forward(&attn_out)
    }
}

impl LanguageModel for UnlimitedOcrVlModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state
            .with_sequence_state(None, |ring| self.forward_ring(input_ids, None, ring))
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_sequence_state(None, |ring| {
            self.forward_ring(input_ids, input_embeddings, ring)
        })
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_ring_caches(),
            |ring| self.forward_ring(input_ids, None, ring),
        )
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_ring_caches(),
            |ring| self.forward_ring(input_ids, input_embeddings, ring),
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.inner.text_model.embed_tokens_forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Ring caches live model-owned; reset the fallback internal set for a
        // fresh generation and hand back empty external caches for trait
        // compatibility (they are never written).
        self.sequence_state
            .replace_internal(self.make_ring_caches());
        (0..self.num_layers()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.inner.text_model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![self.inner.eos_token_id]
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        // The image placeholder id must never be sampled during decode.
        vec![self.inner.image_token_id]
    }

    fn reset_runtime_state(&self) {
        self.sequence_state
            .replace_internal(self.make_ring_caches());
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_ring_caches());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn supports_padded_prefill(&self) -> bool {
        // Ring caches record the prefill boundary from the physical length;
        // appending padding tokens would corrupt that boundary, so opt out of
        // tile-aligned padded prefill (single-pass prefill only).
        false
    }

    fn supports_chunked_prefill(&self) -> bool {
        // The ring cache infers the prefill -> decode boundary from the first
        // single-token forward. A chunked prefill whose final chunk is one
        // token would misclassify that prompt token as the first decode token
        // and let the ring evict it, so force single-pass prefill. The OCR
        // image path already prefills in a single embeddings pass.
        false
    }

    fn supports_batching(&self) -> bool {
        // Internal ring caches are not compatible with per-sequence KV cache
        // isolation used by continuous batching.
        false
    }
}

#[cfg(test)]
#[path = "unlimited_ocr_tests.rs"]
mod tests;
