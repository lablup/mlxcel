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

//! Baichuan model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - Fused W_pack QKV projection (Q, K, V concatenated in one weight)
//! - Sliding window attention with separate head configs for SWA layers
//! - RMSNorm + SiLU (standard Llama style)

use mlxcel_core::cache::{DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::Path;

use crate::models::model_owned::ModelOwnedSequenceState;

// Config.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
pub struct BaichuanConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub sliding_window_layers: Vec<usize>,
    pub conv_window: usize,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub num_swa_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_swa_key_value_heads: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

mlxcel_core::impl_rope_lifting_deserialize!(BaichuanConfig);

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

impl BaichuanConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }
}

// Attention.
pub struct BaichuanAttention {
    pub w_pack: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    pub is_swa: bool,
    pub sliding_window: Option<i32>,
    // Custom convolution weights for K and V
    pub conv_k: UniquePtr<MlxArray>,
    pub conv_v: UniquePtr<MlxArray>,
    pub conv_window: i32,
}

impl BaichuanAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BaichuanConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let is_swa = cfg.sliding_window_layers.contains(&layer_idx);
        let group_size = cfg.group_size();
        let bits = cfg.bits();

        let n_heads = if is_swa {
            cfg.num_swa_attention_heads
                .unwrap_or(cfg.num_attention_heads)
        } else {
            cfg.num_attention_heads
        } as i32;

        let n_kv_heads = if is_swa {
            cfg.num_swa_key_value_heads
                .unwrap_or(cfg.num_key_value_heads)
        } else {
            cfg.num_key_value_heads
        } as i32;

        // head_dim is computed based on the number of heads for this layer type
        // SWA layers may have different head counts, so head_dim differs
        let head_dim = (cfg.hidden_size / n_heads as usize) as i32;

        let sliding_window = if is_swa {
            Some(cfg.sliding_window as i32)
        } else {
            None
        };

        // Load convolution weights for K and V
        let conv_k = get_weight_copy(weights, &format!("{}.conv_k", prefix))?;
        let conv_v = get_weight_copy(weights, &format!("{}.conv_v", prefix))?;

        Ok(Self {
            w_pack: UnifiedLinear::from_weights(
                weights,
                &format!("{}.W_pack", prefix),
                group_size,
                bits,
            )?,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.o_proj", prefix),
                group_size,
                bits,
            )?,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_base: cfg.rope_theta,
            is_swa,
            sliding_window,
            conv_k,
            conv_v,
            conv_window: cfg.conv_window as i32,
        })
    }

    /// Custom 1D convolution for K/V preprocessing
    /// Input u has shape [B, H, L, D], weights shape [1, 1, H, 1, 2]
    /// Returns u_prev * w0 + u * w1 where w0, w1 are the two conv weights
    fn custom_convolution(
        &self,
        u: &MlxArray,
        weights: &MlxArray,
        state: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(u);
        let b = shape[0];
        let h = shape[1];
        let l = shape[2];
        let d = shape[3];

        // Reshape weights from [1, 1, H, 1, 2] to [1, H, 2, 1, 1]
        // Then slice to get w0 and w1
        let weights_reshaped = mlxcel_core::reshape(weights, &[1, h, self.conv_window, 1, 1]);
        let w0 = mlxcel_core::utils::slice_axis(&weights_reshaped, 2, 0, 1); // [1, H, 1, 1, 1]
        let w1 = mlxcel_core::utils::slice_axis(&weights_reshaped, 2, 1, 2); // [1, H, 1, 1, 1]

        // Squeeze axis 2 to get [1, H, 1, 1]
        let w0 = mlxcel_core::squeeze_axis(&w0, 2);
        let w1 = mlxcel_core::squeeze_axis(&w1, 2);

        // Create u_prev: concatenate state with u[:, :, :-1]
        let u_dtype = mlxcel_core::array_dtype(u);
        let u_prev = if l > 1 {
            // For prefill: concat zeros (or state) with u[:-1]
            let zero_state = if let Some(s) = state {
                mlxcel_core::copy(s)
            } else {
                mlxcel_core::zeros(&[b, h, 1, d], u_dtype)
            };
            let u_slice = mlxcel_core::utils::slice_axis(u, 2, 0, l - 1);
            mlxcel_core::concatenate(&zero_state, &u_slice, 2)
        } else {
            // For single token decode: use state directly
            if let Some(s) = state {
                mlxcel_core::copy(s)
            } else {
                mlxcel_core::zeros(&[b, h, 1, d], u_dtype)
            }
        };

        // Compute u_prev * w0 + u * w1
        let term1 = mlxcel_core::multiply(&u_prev, &w0);
        let term2 = mlxcel_core::multiply(u, &w1);
        mlxcel_core::add(&term1, &term2)
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        conv_state: &mut ConvLayerState,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let dim = shape[2];

        // Fused QKV projection
        let qkv = self.w_pack.forward(x);

        // Split into Q, K, V
        // W_pack = [Q (dim) | K (n_kv_heads * head_dim) | V (n_kv_heads * head_dim)]
        let kv_size = self.n_kv_heads * self.head_dim;

        let q = mlxcel_core::slice_last_dim(&qkv, 0, dim);
        let k = mlxcel_core::slice_last_dim(&qkv, dim, dim + kv_size);
        let v = mlxcel_core::slice_last_dim(&qkv, dim + kv_size, dim + 2 * kv_size);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Store k_init, v_init for conv state update (last token before conv)
        let k_last = mlxcel_core::utils::slice_axis(&k, 2, l - 1, l);
        let v_last = mlxcel_core::utils::slice_axis(&v, 2, l - 1, l);

        // Apply custom convolution to K and V with the previous token's
        // pre-convolution K/V (none at the start of a sequence).
        let (last_k_state, last_v_state) = conv_state.last();
        let k = self.custom_convolution(&k, &self.conv_k, last_k_state);
        let v = self.custom_convolution(&v, &self.conv_v, last_v_state);

        // Record this forward's last pre-convolution k/v as the next step's
        // previous token.
        conv_state.push(k_last, v_last);

        let offset = cache.offset;

        // Apply RoPE using fast_rope directly (after convolution)
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            mlxcel_core::causal_attention(
                &q,
                &cache_k,
                &cache_v,
                self.scale,
                0.0,
                self.sliding_window.unwrap_or(0),
            )
        } else {
            // Single token or explicit mask
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.sliding_window.unwrap_or(0),
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.n_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }
}

// MLP.
pub struct BaichuanMLP {
    pub gate_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
}

impl BaichuanMLP {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BaichuanConfig,
    ) -> Result<Self, String> {
        let group_size = cfg.group_size();
        let bits = cfg.bits();

        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
            )?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // Use compiled SwiGLU activation for kernel fusion
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }
}

// Transformer Block.
pub struct BaichuanDecoderLayer {
    pub self_attn: BaichuanAttention,
    pub mlp: BaichuanMLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl BaichuanDecoderLayer {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BaichuanConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let self_attn = BaichuanAttention::from_weights(
            weights,
            &format!("{}.self_attn", prefix),
            cfg,
            layer_idx,
        )?;
        let mlp = BaichuanMLP::from_weights(weights, &format!("{}.mlp", prefix), cfg)?;

        // Load RMSNorm weights manually
        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, cfg.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_norm_weight, cfg.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        conv_state: &mut ConvLayerState,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Attention block with residual
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, conv_state, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // MLP block with residual
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// Model.
/// Convolution state for Baichuan attention (stores last K/V before conv)
pub type ConvState = Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>;

/// One layer's K/V convolution state for one sequence: the last
/// pre-convolution K/V token, plus the previous entries the server's decode
/// lookahead may rewind to.
///
/// Every forward pushes one entry (a single decode token or a prefill
/// chunk, whose last token is what the next step convolves with). The
/// scheduler tears a pipelined step down with
/// `LanguageModel::rewind_decode_appends`, which pops the speculative
/// forwards again; without that the finishing tick's synchronous re-run
/// convolved its input with the speculative token (#2567). The history is
/// bounded to [`DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS`] previous entries,
/// the most the scheduler ever unwinds at once.
pub struct ConvLayerState {
    /// `back()` is the current state; older entries precede it.
    history: VecDeque<ConvState>,
}

impl ConvLayerState {
    const CAPACITY: usize = DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS + 1;

    /// A sequence start: no previous token.
    pub fn new() -> Self {
        let mut history = VecDeque::with_capacity(Self::CAPACITY);
        history.push_back(None);
        Self { history }
    }

    /// The previous token's pre-convolution (K, V), `None` at a sequence start.
    fn last(&self) -> (Option<&MlxArray>, Option<&MlxArray>) {
        match self.history.back() {
            Some(Some((k, v))) => (k.as_ref(), v.as_ref()),
            _ => (None, None),
        }
    }

    /// Record a forward's last pre-convolution K/V as the new current state.
    fn push(&mut self, k_last: UniquePtr<MlxArray>, v_last: UniquePtr<MlxArray>) {
        if self.history.len() == Self::CAPACITY {
            self.history.pop_front();
        }
        self.history.push_back(Some((k_last, v_last)));
    }

    /// Drop the `n` most recent forwards, restoring the state before them.
    fn rewind(&mut self, n: i32) -> Result<(), String> {
        let n = usize::try_from(n).map_err(|_| format!("negative rewind {n}"))?;
        if n >= self.history.len() {
            return Err(format!(
                "cannot rewind {n} conv-state appends: only {} previous entries are retained",
                self.history.len() - 1
            ));
        }
        for _ in 0..n {
            self.history.pop_back();
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn depth(&self) -> usize {
        self.history.len()
    }
}

impl Default for ConvLayerState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BaichuanModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<BaichuanDecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    /// Per-layer K/V convolution state, keyed by the scheduler's `SequenceId`
    /// (#2567). Each attention layer convolves the new K/V with the last
    /// pre-convolution K/V token of the *same* sequence, so the state is
    /// recurrent and belongs to one request. Server rows resolve their own
    /// entry through `forward_with_sequence_id`; the fallback slot serves the
    /// CLI (`seq_id == None`) and is reset by [`Self::make_caches`]. The KV
    /// caches themselves stay in the scheduler's dense slice.
    conv_states: ModelOwnedSequenceState<ConvLayerState>,
}

impl BaichuanModel {
    fn fresh_conv_states(num_layers: usize) -> Vec<ConvLayerState> {
        (0..num_layers).map(|_| ConvLayerState::new()).collect()
    }

    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, caches, mask, None)
    }

    /// Forward over the conv state that belongs to `seq_id` (the fallback slot
    /// when `None`). A sequence the scheduler never prepared gets a fresh
    /// all-`None` state rather than another request's.
    pub(crate) fn forward_for_sequence(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers
        let num_layers = self.layers.len();
        self.conv_states.with_or_create_sequence_state(
            seq_id,
            || Self::fresh_conv_states(num_layers),
            |conv_states| {
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(&h, &mut caches[i], &mut conv_states[i], mask);
                }
            },
        );

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    /// Fresh KV caches for a new generation. Also resets the fallback conv
    /// state, so a CLI generation never convolves its first token with the
    /// previous generation's last one.
    pub fn make_caches(&self) -> Vec<KVCache> {
        self.conv_states
            .replace_internal(Self::fresh_conv_states(self.layers.len()));
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, BaichuanConfig), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: BaichuanConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &config)?;

        Ok((model, config))
    }

    pub fn from_weights(weights: &WeightMap, config: &BaichuanConfig) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        // Load embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            layers.push(BaichuanDecoderLayer::from_weights(
                weights, &prefix, config, i,
            )?);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

        // Load lm_head (if not tied)
        let lm_head = if !config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        // Fallback conv state (one slot per layer); server sequences get
        // their own vector through `prepare_sequence_state`.
        let conv_states =
            ModelOwnedSequenceState::new(Self::fresh_conv_states(config.num_hidden_layers));

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            conv_states,
        })
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for BaichuanModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        BaichuanModel::forward(self, input_ids, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, caches, mask, seq_id)
    }

    /// Per-row dispatch with each row's own `SequenceId`, so every row
    /// convolves with its own last token. The trait default would drop the ids
    /// and run every row over the shared fallback state.
    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        crate::multimodal::batched_dispatch::forward_batched_with_seq_ids_dispatch(
            self,
            input_ids,
            seq_ids,
            batch_caches,
            mask,
        )
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.conv_states
            .prepare_sequence_state(seq_id, Self::fresh_conv_states(self.layers.len()));
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.conv_states.release_sequence_state(seq_id);
    }

    fn reset_runtime_state(&self) {
        self.conv_states
            .replace_internal(Self::fresh_conv_states(self.layers.len()));
    }

    /// The conv state is per-token state the scheduler's KV slice does not
    /// hold: the engine must rewind it on a lookahead teardown, and the
    /// prompt cache and KV handoff cannot restore it (#2567).
    fn kv_cache_captures_sequence_state(&self) -> bool {
        false
    }

    fn supports_decode_lookahead_rewind(&self) -> bool {
        true
    }

    /// Pop the `n` speculative forwards from every layer's conv history
    /// (`Engine::unwind_appends`, which has already trimmed the KV caches).
    fn rewind_decode_appends(&self, seq_id: SequenceId, n: i32) -> Result<(), String> {
        self.conv_states
            .with_existing_sequence_state(seq_id, |layers| {
                layers.iter_mut().enumerate().try_for_each(|(layer_idx, state)| {
                    state.rewind(n).map_err(|err| {
                        format!("Baichuan decode rewind, sequence {seq_id}, layer {layer_idx}: {err}")
                    })
                })
            })?
    }

    /// Padding tokens would land in the conv state as the "last token" of the
    /// prompt, and the scheduler's KV trim does not reach that state.
    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn make_caches(&self) -> Vec<KVCache> {
        BaichuanModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Baichuan EOS token
        vec![2] // </s>
    }
}

#[cfg(test)]
#[path = "baichuan_tests.rs"]
pub(crate) mod baichuan_tests;
