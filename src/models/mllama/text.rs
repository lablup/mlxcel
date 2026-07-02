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

//! Llama 3.2 Vision text backbone with interleaved gated cross-attention.
//!
//! Faithful port of
//! `references/mlx-vlm/mlx_vlm/models/mllama/language.py`.
//!
//! The backbone is a standard Llama-3 decoder in which the layers listed in
//! `cross_attention_layers` are replaced by [`MllamaCrossAttentionDecoderLayer`]
//! adapters that attend to the vision tower's features. Self-attention layers
//! are the ordinary Llama-3 block, so they reuse
//! [`crate::models::llama3::TransformerBlock`] verbatim (fused QKV, plain RoPE
//! with `base = rope_theta`). The cross-attention adapters add per-head
//! `q_norm`/`k_norm` (RMSNorm over `head_dim`) and two learned `tanh` gates on
//! the attention and MLP residual branches.

use std::cell::RefCell;

use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear, attention_from_ptr};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::MllamaTextConfig;
use crate::models::llama3::{MLP, ModelArgs, TransformerBlock};

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

fn load_rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    Ok(RMSNorm::new(
        get_weight_copy(weights, &format!("{prefix}.weight"))?,
        eps,
    ))
}

/// Gated cross-attention: queries come from the text stream, keys/values from
/// the projected vision features (`cross_attention_states`). Mirrors
/// `MllamaTextCrossAttention` in the reference.
pub struct MllamaTextCrossAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    /// Fill-once cache of the post-norm, reshaped/transposed cross-attention
    /// key and value derived from the fixed image `cross_states`. The image
    /// features do not change during a generation, so their projected key/value
    /// are constant and are computed once (on the first forward after the state
    /// is set) then reused on every subsequent decode step, instead of being
    /// recomputed per token. Invalidated by [`Self::invalidate_kv_cache`] when
    /// the owning [`crate::vision::MllamaVLModel`] sets or clears its
    /// `cross_attention_states`. Mirrors the fill-once cross-attention KV cache
    /// in the mlx-vlm reference (`cross_attention_states` fill path followed by
    /// `cache.fetch()` on later steps).
    cross_kv: RefCell<Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>>,
}

impl MllamaTextCrossAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaTextConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim() as i32;
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.q_proj"),
                group_size,
                bits,
            )?,
            k_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.k_proj"),
                group_size,
                bits,
            )?,
            v_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.v_proj"),
                group_size,
                bits,
            )?,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.o_proj"),
                group_size,
                bits,
            )?,
            q_norm: load_rms_norm(weights, &format!("{prefix}.q_norm"), config.rms_norm_eps)?,
            k_norm: load_rms_norm(weights, &format!("{prefix}.k_norm"), config.rms_norm_eps)?,
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            cross_kv: RefCell::new(None),
        })
    }

    /// Project the fixed image `cross_states` into the per-head cross-attention
    /// key and value: `k_proj`/`v_proj` then reshape to
    /// `[b, kv_len, num_kv_heads, head_dim]`, transpose to
    /// `[b, num_kv_heads, kv_len, head_dim]`, and apply `k_norm` to the key.
    ///
    /// `b` is the text batch size (`hidden_states.shape[0]`), which equals
    /// `cross_states.shape[0]`. The result depends only on the (fixed) image
    /// features and this layer's weights, never on the per-step text query, so
    /// [`Self::forward`] computes it once and caches it in `cross_kv`.
    fn compute_kv(
        &self,
        cross_states: &MlxArray,
        b: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let kv_len = mlxcel_core::array_shape(cross_states)[1];
        let key = self.k_proj.forward(cross_states);
        let key = mlxcel_core::reshape(&key, &[b, kv_len, self.num_kv_heads, self.head_dim]);
        let key = mlxcel_core::transpose_axes(&key, &[0, 2, 1, 3]);
        let key = self.k_norm.forward(&key);

        let value = self.v_proj.forward(cross_states);
        let value = mlxcel_core::reshape(&value, &[b, kv_len, self.num_kv_heads, self.head_dim]);
        let value = mlxcel_core::transpose_axes(&value, &[0, 2, 1, 3]);
        (key, value)
    }

    /// Drop any cached cross-attention key/value so the next [`Self::forward`]
    /// recomputes them from the current `cross_states`.
    fn invalidate_kv_cache(&self) {
        *self.cross_kv.borrow_mut() = None;
    }

    /// `hidden_states`: `[B, q_len, hidden]`.
    /// `cross_states`: `[B, kv_len, hidden]` projected vision features.
    /// `mask`: optional `[B, 1, q_len, kv_len]` additive cross-attention mask.
    fn forward(
        &self,
        hidden_states: &MlxArray,
        cross_states: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let b = shape[0];
        let q_len = shape[1];

        // Query from the text stream, per-head RMSNorm over head_dim.
        let query = self.q_proj.forward(hidden_states);
        let query = mlxcel_core::reshape(&query, &[b, q_len, self.num_heads, self.head_dim]);
        let query = mlxcel_core::transpose_axes(&query, &[0, 2, 1, 3]);
        let query = self.q_norm.forward(&query);

        // Key/Value from the vision features are identical on every decode step
        // (the image is fixed), so compute them once and reuse. The cache is
        // invalidated whenever the owning MllamaVLModel changes or clears its
        // cross_states (see MllamaTextModel::invalidate_cross_attention_cache).
        // Read the borrow into a bool first so the immutable borrow is released
        // before the mutable borrow below (avoids a RefCell double-borrow).
        let need_fill = self.cross_kv.borrow().is_none();
        if need_fill {
            let kv = self.compute_kv(cross_states, b);
            *self.cross_kv.borrow_mut() = Some(kv);
        }
        let cross_kv = self.cross_kv.borrow();
        let (key, value) = cross_kv
            .as_ref()
            .expect("cross-attention KV cache filled above");
        // Deref-coerce the cached UniquePtr handles to &MlxArray for the kernel.
        let key: &MlxArray = key;
        let value: &MlxArray = value;

        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        // GQA is handled inside the shared attention kernel (num_heads vs
        // num_kv_heads). Cross-attention is never causal.
        let attn = unsafe { attention_from_ptr(&query, key, value, self.scale, mask_ptr, 0.0, 0) };

        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, q_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attn)
    }
}

/// A gated cross-attention decoder layer (`MllamaCrossAttentionDecoderLayer`).
pub struct MllamaCrossAttentionDecoderLayer {
    input_layernorm: RMSNorm,
    cross_attn: MllamaTextCrossAttention,
    post_attention_layernorm: RMSNorm,
    mlp: MLP,
    attn_gate: UniquePtr<MlxArray>,
    mlp_gate: UniquePtr<MlxArray>,
}

impl MllamaCrossAttentionDecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaTextConfig,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let group_size = args.group_size();
        let bits = args.bits();
        Ok(Self {
            input_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.input_layernorm"),
                config.rms_norm_eps,
            )?,
            cross_attn: MllamaTextCrossAttention::from_weights(
                weights,
                config,
                &format!("{prefix}.cross_attn"),
                group_size,
                bits,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.post_attention_layernorm"),
                config.rms_norm_eps,
            )?,
            mlp: MLP::from_weights(weights, args, &format!("{prefix}.mlp"))?,
            attn_gate: get_weight_copy(weights, &format!("{prefix}.cross_attn_attn_gate"))?,
            mlp_gate: get_weight_copy(weights, &format!("{prefix}.cross_attn_mlp_gate"))?,
        })
    }

    /// Drop this layer's cached cross-attention key/value (see
    /// [`MllamaTextCrossAttention::invalidate_kv_cache`]).
    fn invalidate_kv_cache(&self) {
        self.cross_attn.invalidate_kv_cache();
    }

    /// Forward with vision cross-attention state.
    ///
    /// When `cross_states` is `None` (a text-only request with no image), the
    /// layer is a pass-through: with no image features to attend to there is
    /// nothing for cross-attention to contribute. This matches HuggingFace
    /// `MllamaForConditionalGeneration`, which skips the cross-attention block
    /// when `cross_attention_states is None` and the cache is empty.
    fn forward(
        &self,
        hidden_states: &MlxArray,
        cross_states: Option<&MlxArray>,
        cross_mask: Option<&MlxArray>,
        full_text_row_masked_out_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let Some(cross_states) = cross_states else {
            return mlxcel_core::copy(hidden_states);
        };

        // Gated cross-attention branch: h = h + tanh(attn_gate) * attn(norm(h)).
        let normed = self.input_layernorm.forward(hidden_states);
        let attn = self.cross_attn.forward(&normed, cross_states, cross_mask);
        let attn_gate = mlxcel_core::tanh(&self.attn_gate);
        let gated_attn = mlxcel_core::multiply(&attn_gate, &attn);
        let hidden_states = mlxcel_core::add(hidden_states, &gated_attn);

        // Gated MLP branch, optionally zeroing rows with no visible image.
        let normed = self.post_attention_layernorm.forward(&hidden_states);
        let mut mlp_out = self.mlp.forward(&normed);
        if let Some(row_mask) = full_text_row_masked_out_mask {
            // row_mask: [B, 1, q_len, 1] -> [B, q_len, 1], broadcasts over hidden.
            let row_mask = mlxcel_core::squeeze_axis(row_mask, 1);
            mlp_out = mlxcel_core::multiply(&row_mask, &mlp_out);
        }
        let mlp_gate = mlxcel_core::tanh(&self.mlp_gate);
        let gated_mlp = mlxcel_core::multiply(&mlp_gate, &mlp_out);
        mlxcel_core::add(&hidden_states, &gated_mlp)
    }
}

/// One decoder layer: either a standard Llama-3 self-attention block or a gated
/// cross-attention adapter.
enum TextLayer {
    SelfAttn(Box<TransformerBlock>),
    Cross(Box<MllamaCrossAttentionDecoderLayer>),
}

/// The interleaved self/cross-attention text model (`MllamaTextModel` +
/// `LanguageModel` head from the reference, fused into one struct).
pub struct MllamaTextModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<TextLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    num_layers: usize,
}

impl MllamaTextModel {
    pub fn from_weights(weights: &WeightMap, config: &MllamaTextConfig) -> Result<Self, String> {
        let args = config.to_llama3_args();
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            if config.is_cross_attention_layer(idx) {
                layers.push(TextLayer::Cross(Box::new(
                    MllamaCrossAttentionDecoderLayer::from_weights(weights, config, &args, idx)?,
                )));
            } else {
                layers.push(TextLayer::SelfAttn(Box::new(
                    TransformerBlock::from_weights(weights, &args, idx)?,
                )));
            }
        }

        let norm = load_rms_norm(weights, "model.norm", config.rms_norm_eps)?;
        let lm_head = if config.tie_word_embeddings {
            UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            num_layers: config.num_hidden_layers,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.num_layers).map(|_| KVCache::new()).collect()
    }

    /// Invalidate the fill-once image key/value cache in every cross-attention
    /// layer.
    ///
    /// [`crate::vision::MllamaVLModel`] calls this whenever it sets or clears
    /// its `cross_attention_states`, tying the KV-cache lifecycle to the
    /// cross_states lifecycle: a new image (or a reset to text-only) forces the
    /// next forward to recompute the projected key/value from the current
    /// features rather than reusing a stale cache.
    pub fn invalidate_cross_attention_cache(&self) {
        for layer in &self.layers {
            if let TextLayer::Cross(layer) = layer {
                layer.invalidate_kv_cache();
            }
        }
    }

    pub fn embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Full forward with optional vision cross-attention state.
    ///
    /// - `input_embeds` overrides `input_ids` when present (VLM inject path).
    /// - Self-attention layers consume `caches[i]` and `mask`.
    /// - Cross-attention layers consume `cross_states` / `cross_mask` /
    ///   `full_text_row_masked_out_mask`; their `caches[i]` slot is left unused.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: Option<&MlxArray>,
        input_embeds: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        cross_states: Option<&MlxArray>,
        cross_mask: Option<&MlxArray>,
        full_text_row_masked_out_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match (input_embeds, input_ids) {
            (Some(embeds), _) => mlxcel_core::copy(embeds),
            (None, Some(ids)) => self.embed_tokens.forward(ids),
            (None, None) => panic!("MllamaTextModel::forward requires input_ids or input_embeds"),
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = match layer {
                TextLayer::SelfAttn(block) => block.forward(&h, &mut caches[i], mask),
                TextLayer::Cross(layer) => {
                    layer.forward(&h, cross_states, cross_mask, full_text_row_masked_out_mask)
                }
            };
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::{MllamaTextConfig, MllamaTextCrossAttention};
    use mlxcel_core::weights::WeightMap;
    use mlxcel_core::{MlxArray, UniquePtr};

    // Tiny cross-attention layer dimensions (2 heads, 1 KV head, head_dim 2).
    const HIDDEN: i32 = 4;
    const HEADS: i32 = 2;
    const KV_HEADS: i32 = 1;
    const HEAD_DIM: i32 = HIDDEN / HEADS; // 2
    const B: i32 = 1;
    const Q_LEN: i32 = 3;
    const KV_LEN: i32 = 5; // vision cross-attention key/value length

    fn tiny_config() -> MllamaTextConfig {
        serde_json::from_str(
            r#"{
                "model_type": "mllama",
                "hidden_size": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-5
            }"#,
        )
        .expect("tiny mllama text config")
    }

    /// Deterministic pseudo-random fill in roughly `[-0.5, 0.5]`.
    fn fill(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 131 + seed * 977 + 7) % 251) as f32 / 251.0 - 0.5)
            .collect()
    }

    fn put(map: &mut WeightMap, key: &str, shape: &[i32], seed: usize) {
        let n: i32 = shape.iter().product();
        map.insert(
            key.to_string(),
            mlxcel_core::from_slice_f32(&fill(n as usize, seed), shape),
        );
    }

    fn put_const(map: &mut WeightMap, key: &str, shape: &[i32], value: f32) {
        let n: i32 = shape.iter().product();
        map.insert(
            key.to_string(),
            mlxcel_core::from_slice_f32(&vec![value; n as usize], shape),
        );
    }

    /// Dense (unquantized) weights for a single cross-attention layer under the
    /// `cross_attn` prefix.
    fn cross_attn_weights() -> WeightMap {
        let mut w = WeightMap::new();
        let q = HEADS * HEAD_DIM; // 4
        let kv = KV_HEADS * HEAD_DIM; // 2
        put(&mut w, "cross_attn.q_proj.weight", &[q, HIDDEN], 1);
        put(&mut w, "cross_attn.k_proj.weight", &[kv, HIDDEN], 2);
        put(&mut w, "cross_attn.v_proj.weight", &[kv, HIDDEN], 3);
        put(&mut w, "cross_attn.o_proj.weight", &[HIDDEN, q], 4);
        put_const(&mut w, "cross_attn.q_norm.weight", &[HEAD_DIM], 1.0);
        put_const(&mut w, "cross_attn.k_norm.weight", &[HEAD_DIM], 1.0);
        w
    }

    fn build_layer(config: &MllamaTextConfig) -> MllamaTextCrossAttention {
        let args = config.to_llama3_args();
        MllamaTextCrossAttention::from_weights(
            &cross_attn_weights(),
            config,
            "cross_attn",
            args.group_size(),
            args.bits(),
        )
        .expect("build tiny cross-attention layer")
    }

    fn hidden_states() -> UniquePtr<MlxArray> {
        mlxcel_core::from_slice_f32(
            &fill((B * Q_LEN * HIDDEN) as usize, 20),
            &[B, Q_LEN, HIDDEN],
        )
    }

    fn cross_states(seed: usize) -> UniquePtr<MlxArray> {
        mlxcel_core::from_slice_f32(
            &fill((B * KV_LEN * HIDDEN) as usize, seed),
            &[B, KV_LEN, HIDDEN],
        )
    }

    /// Max absolute elementwise difference between two arrays.
    fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
        let diff = mlxcel_core::subtract(a, b);
        let m = mlxcel_core::max_all(&mlxcel_core::abs(&diff));
        mlxcel_core::eval(&m);
        mlxcel_core::item_f32(&m)
    }

    /// Pure memoization: the key/value the forward pass caches must be
    /// byte-identical to a fresh recompute from the same `cross_states`. This is
    /// the correctness guarantee for reusing the cache across decode steps
    /// (greedy output cannot change, because cached K/V == recomputed K/V).
    #[test]
    fn cached_cross_kv_equals_fresh_recompute() {
        let config = tiny_config();
        let layer = build_layer(&config);
        let h = hidden_states();
        let cross = cross_states(42);

        // The cache is empty until the first forward fills it.
        assert!(layer.cross_kv.borrow().is_none());
        let _ = layer.forward(&h, &cross, None);
        assert!(layer.cross_kv.borrow().is_some());

        // A fresh recompute from the same features must match exactly.
        let (fresh_key, fresh_value) = layer.compute_kv(&cross, B);
        let cache = layer.cross_kv.borrow();
        let (cached_key, cached_value) = cache.as_ref().expect("cache filled above");
        assert_eq!(
            max_abs_diff(cached_key, &fresh_key),
            0.0,
            "cached key must be byte-identical to a fresh recompute"
        );
        assert_eq!(
            max_abs_diff(cached_value, &fresh_value),
            0.0,
            "cached value must be byte-identical to a fresh recompute"
        );
    }

    /// A `cross_states` change (a new image) invalidates the cache, and the next
    /// forward rebuilds the key/value from the new features instead of reusing
    /// the stale ones.
    #[test]
    fn cross_states_change_invalidates_cache() {
        let config = tiny_config();
        let layer = build_layer(&config);
        let h = hidden_states();
        let cross_a = cross_states(42);
        let cross_b = cross_states(99);

        // Fill from image A and confirm the cache matches A.
        let _ = layer.forward(&h, &cross_a, None);
        let (a_key, _) = layer.compute_kv(&cross_a, B);
        {
            let cache = layer.cross_kv.borrow();
            let (cached_key, _) = cache.as_ref().expect("cache filled from A");
            assert_eq!(max_abs_diff(cached_key, &a_key), 0.0);
        }

        // Invalidation mirrors MllamaVLModel setting/clearing cross_states.
        layer.invalidate_kv_cache();
        assert!(layer.cross_kv.borrow().is_none());

        // Forward with image B rebuilds from the new features.
        let _ = layer.forward(&h, &cross_b, None);
        let (b_key, _) = layer.compute_kv(&cross_b, B);
        let cache = layer.cross_kv.borrow();
        let (cached_key, _) = cache.as_ref().expect("cache filled from B");
        assert_eq!(
            max_abs_diff(cached_key, &b_key),
            0.0,
            "the rebuilt cache must match image B"
        );
        assert!(
            max_abs_diff(cached_key, &a_key) > 1e-3,
            "a new image must produce a different cached key, not a stale one"
        );
    }

    // --- Real-tile cross-states equivalence (issue #527 perf follow-up). ---
    //
    // The reference masks every padding-tile position out of the text
    // cross-attention with an additive -1e9 (`cross_attention_mask` derived
    // from the per-image `num_tiles`). `exp(logit - 1e9)` underflows to
    // exactly 0.0, so those positions add exact zeros to the softmax numerator
    // and denominator: attending over the REAL-tile rows alone is the same
    // computation. These tests pin that equivalence at the byte level, which
    // is what licenses `MllamaVLModel` to drop the padding-tile rows from
    // `cross_attention_states` instead of threading a reference-style mask.

    /// `[1, rows, HIDDEN]` cross-states with identifiable per-row content.
    fn cross_rows(rows: i32, seed: usize) -> UniquePtr<MlxArray> {
        mlxcel_core::from_slice_f32(&fill((rows * HIDDEN) as usize, seed), &[1, rows, HIDDEN])
    }

    /// Additive `[1, 1, Q_LEN, kv]` mask: 0 everywhere, -1e9 on `masked` keys
    /// (the reference's padding-tile columns).
    fn additive_kv_mask(kv: i32, masked: &[i32]) -> UniquePtr<MlxArray> {
        let mut vals = vec![0.0f32; (Q_LEN * kv) as usize];
        for q in 0..Q_LEN {
            for &k in masked {
                vals[(q * kv + k) as usize] = -1e9;
            }
        }
        mlxcel_core::from_slice_f32(&vals, &[1, 1, Q_LEN, kv])
    }

    fn rows_slice(x: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray> {
        mlxcel_core::slice(x, &[0, start, 0], &[1, end, HIDDEN])
    }

    /// Single image, 1 real tile of 4 (2 patch rows per tile): attention over
    /// the real rows only is byte-identical to reference-masked attention over
    /// all rows, garbage padding-lane features included.
    #[test]
    fn real_tile_rows_match_reference_masked_full_rows() {
        let config = tiny_config();
        let layer = build_layer(&config);
        let h = hidden_states();

        // 4 tiles x 2 patches = 8 kv rows; rows 2..8 are padding-lane features
        // (arbitrary non-zero values, as the tower genuinely produces).
        let full = cross_rows(8, 42);
        let mask = additive_kv_mask(8, &[2, 3, 4, 5, 6, 7]);
        let masked_out = layer.forward(&h, &full, Some(&mask));

        // The fill-once KV cache belongs to the previous cross-states.
        layer.invalidate_kv_cache();
        let real_only = rows_slice(&full, 0, 2);
        let sliced_out = layer.forward(&h, &real_only, None);

        assert_eq!(
            max_abs_diff(&masked_out, &sliced_out),
            0.0,
            "dropping the -1e9-masked padding-tile rows must be byte-identical"
        );
    }

    /// Ragged multi-image (media 0: 1 real tile of 2, media 1: 2 of 2):
    /// media-major concatenation of each image's real rows is byte-identical
    /// to reference-masked attention over the full row set.
    #[test]
    fn ragged_real_tile_rows_match_reference_masked_full_rows() {
        let config = tiny_config();
        let layer = build_layer(&config);
        let h = hidden_states();

        // [media0 t0, media0 t1(pad), media1 t0, media1 t1] x 2 patches.
        let full = cross_rows(8, 77);
        let mask = additive_kv_mask(8, &[2, 3]);
        let masked_out = layer.forward(&h, &full, Some(&mask));

        layer.invalidate_kv_cache();
        let media0 = rows_slice(&full, 0, 2);
        let media1 = rows_slice(&full, 4, 8);
        let sliced = mlxcel_core::concatenate(&media0, &media1, 1);
        let sliced_out = layer.forward(&h, &sliced, None);

        assert_eq!(
            max_abs_diff(&masked_out, &sliced_out),
            0.0,
            "ragged per-image selection must be byte-identical to the \
             reference-masked full attention"
        );
    }

    /// All tiles real: the all-zero reference mask is the no-mask computation;
    /// there is nothing to drop and no behavior change.
    #[test]
    fn full_tile_mask_is_the_unmasked_computation() {
        let config = tiny_config();
        let layer = build_layer(&config);
        let h = hidden_states();

        let full = cross_rows(8, 99);
        let mask = additive_kv_mask(8, &[]);
        let masked_out = layer.forward(&h, &full, Some(&mask));

        layer.invalidate_kv_cache();
        let unmasked_out = layer.forward(&h, &full, None);

        assert_eq!(
            max_abs_diff(&masked_out, &unmasked_out),
            0.0,
            "an all-zero mask must not perturb the attention output"
        );
    }
}
