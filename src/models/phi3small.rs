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

//! Phi3Small model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - Fused query_key_value projection
//! - GeGELU activation (gelu * (linear + 1))
//! - mup scaling (attention, embedding, width multipliers)
//! - Blocksparse attention pattern (local block window + per-head vertical
//!   stride) on non-dense layers; periodic fully-dense layers stay causal
//! - LayerNorm (not RMSNorm)
//! - bias=True for all projections
//! - Tied word embeddings

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub dense_attention_every_n_layers: usize,
    pub ff_intermediate_size: usize,
    pub gegelu_limit: f32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_epsilon: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,

    #[serde(default = "default_mup_attn")]
    pub mup_attn_multiplier: f32,

    #[serde(default = "default_mup_scaling")]
    pub mup_use_scaling: bool,

    #[serde(default = "default_mup_emb")]
    pub mup_embedding_multiplier: f32,

    #[serde(default = "default_mup_width")]
    pub mup_width_multiplier: f32,

    #[serde(default = "default_rope_base")]
    pub rope_embedding_base: f32,

    #[serde(default = "default_rope_scale")]
    pub rope_position_scale: f32,

    #[serde(default = "default_block_size")]
    pub blocksparse_block_size: usize,

    #[serde(default = "default_local_blocks")]
    pub blocksparse_num_local_blocks: usize,

    #[serde(default = "default_vert_stride")]
    pub blocksparse_vert_stride: usize,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_mup_attn() -> f32 {
    1.0
}
fn default_mup_scaling() -> bool {
    true
}
fn default_mup_emb() -> f32 {
    10.0
}
fn default_mup_width() -> f32 {
    8.0
}
fn default_rope_base() -> f32 {
    1000000.0
}
fn default_rope_scale() -> f32 {
    1.0
}
fn default_block_size() -> usize {
    64
}
fn default_local_blocks() -> usize {
    16
}
fn default_vert_stride() -> usize {
    8
}

impl ModelArgs {
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

// Attention with Fused QKV and mup scaling.
pub struct Attention {
    pub query_key_value: UnifiedLinear, // Fused Q, K, V projection
    pub dense: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub n_q_per_kv: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub block_sparse: bool, // Whether this layer uses block sparse attention
    pub blocksparse_block_size: i32,
    pub blocksparse_num_local_blocks: i32,
    pub blocksparse_vert_stride: i32,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Fused QKV projection
        let qkv = self.query_key_value.forward(x);

        // Reshape to [B, L, n_kv_heads, n_q_per_kv + 2, head_dim]
        let qkv = mlxcel_core::reshape(
            &qkv,
            &[b, l, self.num_kv_heads, self.n_q_per_kv + 2, self.head_dim],
        );

        // Split into Q, K, V using slicing
        // queries = qkv[..., :-2, :] -> qkv[..., :n_q_per_kv, :]
        // keys = qkv[..., -2, :]
        // values = qkv[..., -1, :]

        // Q: slice [0:n_q_per_kv] on axis 3, then reshape
        let q = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, 0, 0],
            &[b, l, self.num_kv_heads, self.n_q_per_kv, self.head_dim],
        );
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);

        // K: slice [n_q_per_kv:n_q_per_kv+1] on axis 3, then squeeze
        let k = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, self.n_q_per_kv, 0],
            &[b, l, self.num_kv_heads, self.n_q_per_kv + 1, self.head_dim],
        );
        let k = mlxcel_core::squeeze_axis(&k, 3);

        // V: slice [n_q_per_kv+1:n_q_per_kv+2] on axis 3, then squeeze
        let v = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, self.n_q_per_kv + 1, 0],
            &[b, l, self.num_kv_heads, self.n_q_per_kv + 2, self.head_dim],
        );
        let v = mlxcel_core::squeeze_axis(&v, 3);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE
        q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention.
        //
        // Non-dense layers use the Phi-3-small blocksparse pattern (local window
        // of blocks plus a per-head vertical stride); the periodic dense layers
        // (`block_sparse == false`) keep the plain causal path. See
        // `blocksparse_attention` for the mask/selection details.
        let attn_out = if self.block_sparse {
            self.blocksparse_attention(&q, &cache_k, &cache_v, l, offset, mask)
        } else {
            self.dense_attention(&q, &cache_k, &cache_v, l, mask)
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.dense.forward(&attn_out)
    }

    /// Plain causal attention, used by the periodic dense layers
    /// (`block_sparse == false`) and by short-context blocksparse layers where
    /// the pattern degenerates to causal. Byte-for-byte the original fallback.
    fn dense_attention(
        &self,
        q: &MlxArray,
        cache_k: &MlxArray,
        cache_v: &MlxArray,
        l: i32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(q, cache_k, cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            // SAFETY: `mask_ptr` is either null or a live `MlxArray` for this call.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    q, cache_k, cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        }
    }

    /// Phi-3-small blocksparse attention for a non-dense layer.
    ///
    /// A query token at block `qb` attends a key block `kb` (with `kb <= qb`)
    /// when the key block is inside the local window
    /// (`qb - kb < blocksparse_num_local_blocks`) OR the key block is selected
    /// by the per-head vertical stride (`(kb + head + 1) % blocksparse_vert_stride
    /// == 0`). Token-level causality is applied on top so intra-block ordering
    /// matches the reference. This mirrors mlx-lm's
    /// `mlx_lm/models/phi3small.py` (`Attention._block_sparse_attention` +
    /// `_block_sparse_mask`).
    ///
    /// Short-context / in-window decode: when the full key length fits inside
    /// the local window (`kv_len <= num_local_blocks * block_size`) the pattern
    /// is exactly causal, so we reuse the dense path. This keeps short-context
    /// output identical to the dense fallback and, for the common `L == 1`
    /// decode inside the window, avoids materializing any per-step mask (the
    /// maskless single-query fast path selects the whole cache directly). Beyond
    /// the window the union of attended blocks across all heads spans the full
    /// causal range for Phi-3-small's head count (`vert_stride` divides the head
    /// set), so a per-head gather cannot shrink the fused-SDPA work; we build a
    /// single additive mask (`[1, n_heads, L, kv_len]`, one row at decode) and
    /// let the fused kernel consume it.
    fn blocksparse_attention(
        &self,
        q: &MlxArray,
        cache_k: &MlxArray,
        cache_v: &MlxArray,
        l: i32,
        offset: i32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let kv_len = mlxcel_core::array_shape(cache_k)[2];
        let local_span = self.blocksparse_num_local_blocks * self.blocksparse_block_size;

        // In-window: blocksparse degenerates to plain causal. Reuse the dense
        // path so short-context output matches the dense fallback exactly and
        // no per-step mask is allocated for in-window decode.
        if kv_len <= local_span {
            return self.dense_attention(q, cache_k, cache_v, l, mask);
        }

        // Long context: build the additive blocksparse mask and run fused SDPA.
        let bs_mask = build_blocksparse_mask(
            self.num_heads,
            l,
            kv_len,
            offset,
            self.blocksparse_block_size,
            self.blocksparse_num_local_blocks,
            self.blocksparse_vert_stride,
        );
        // Mirror the reference `if mask is not None: scores += mask`: fold any
        // externally supplied mask (e.g. padding) into the blocksparse mask. The
        // blocksparse mask already carries token-level causality, so this is
        // additive and cannot create an all-`-inf` row (every query keeps its
        // own diagonal key).
        let bs_mask = match mask {
            Some(m) => mlxcel_core::add(&bs_mask, m),
            None => bs_mask,
        };
        // SAFETY: `bs_mask` is a live `MlxArray` for the duration of this call.
        unsafe {
            mlxcel_core::layers::attention_from_ptr(
                q,
                cache_k,
                cache_v,
                self.scale,
                &*bs_mask as *const _,
                0.0,
                0,
            )
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let query_key_value = UnifiedLinear::from_weights(
            weights,
            &format!("{}.query_key_value", prefix),
            group_size,
            bits,
        )?;
        let dense =
            UnifiedLinear::from_weights(weights, &format!("{}.dense", prefix), group_size, bits)?;

        let head_dim = args.head_dim() as i32;
        let n_q_per_kv = (args.num_attention_heads / args.num_key_value_heads) as i32;

        // mup scaling for attention
        let scale = if args.mup_use_scaling {
            1.0 / (head_dim as f32 / args.mup_attn_multiplier)
        } else {
            1.0 / (head_dim as f32).sqrt()
        };

        // Block sparse for non-dense layers
        let block_sparse = !(layer_idx + 1).is_multiple_of(args.dense_attention_every_n_layers);

        Ok(Self {
            query_key_value,
            dense,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            n_q_per_kv,
            head_dim,
            scale,
            rope_dims: head_dim,
            rope_base: args.rope_embedding_base,
            block_sparse,
            blocksparse_block_size: args.blocksparse_block_size as i32,
            blocksparse_num_local_blocks: args.blocksparse_num_local_blocks as i32,
            blocksparse_vert_stride: args.blocksparse_vert_stride as i32,
        })
    }
}

/// Build the additive Phi-3-small blocksparse attention mask.
///
/// Returns a `[1, n_heads, q_len, kv_len]` additive mask (`0.0` where a query
/// may attend a key, `-inf` otherwise) that combines, per head, the blocksparse
/// block pattern with token-level causality. The head dimension broadcasts
/// against the fused-SDPA score tensor `[B, n_heads, q_len, kv_len]`.
///
/// This is a faithful port of `mlx_lm/models/phi3small.py`
/// (`Attention._block_sparse_mask`), including its right-aligned query-block
/// assignment: query blocks occupy the last `ceil(q_len / block_size)` block
/// rows of a `ceil(kv_len / block_size)`-block grid, and the token-to-block
/// mapping reproduces the reference `repeat(...)[..., -q_len:, :kv_len]` slice.
///
/// Block selection for head `h`, query block `qb`, key block `kb`:
/// `(qb >= kb) && ((qb - kb < local_blocks) || ((kb + h + 1) % vert_stride == 0))`.
///
/// The whole mask is built on-device with broadcasting ops; nothing of size
/// `q_len * kv_len` is materialized on the host.
fn build_blocksparse_mask(
    n_heads: i32,
    q_len: i32,
    kv_len: i32,
    offset: i32,
    block_size: i32,
    local_blocks: i32,
    vert_stride: i32,
) -> UniquePtr<MlxArray> {
    use mlxcel_core::dtype;

    // Block-grid geometry, matching the reference exactly.
    let kv_blocks = (kv_len + block_size - 1) / block_size;
    let q_blocks = (q_len + block_size - 1) / block_size;
    let front_pad = q_blocks * block_size - q_len; // padding dropped by [-q_len:]
    let offset_blocks = kv_blocks - q_blocks; // absolute index of the first query block

    // Reusable i32 scalar operands (shape [1]) for broadcasting.
    let block_size_arr = mlxcel_core::from_slice_i32(&[block_size], &[1]);
    let vert_stride_arr = mlxcel_core::from_slice_i32(&[vert_stride], &[1]);
    let local_blocks_arr = mlxcel_core::from_slice_i32(&[local_blocks], &[1]);
    let offset_blocks_arr = mlxcel_core::from_slice_i32(&[offset_blocks], &[1]);
    let one_arr = mlxcel_core::from_slice_i32(&[1], &[1]);
    let zero_arr = mlxcel_core::from_slice_i32(&[0], &[1]);

    // Absolute query-block index per query row -> shape [1, q_len, 1].
    // full_row = arange(front_pad, front_pad + q_len); qb = offset_blocks + full_row // block_size.
    let full_row = mlxcel_core::arange_i32(front_pad, front_pad + q_len, 1);
    let qi = mlxcel_core::floor_divide(&full_row, &block_size_arr);
    let q_block = mlxcel_core::add(&qi, &offset_blocks_arr);
    let q_block = mlxcel_core::reshape(&q_block, &[1, q_len, 1]);

    // Absolute key-block index per key column -> shape [1, 1, kv_len].
    let k_positions = mlxcel_core::arange_i32(0, kv_len, 1);
    let k_block = mlxcel_core::floor_divide(&k_positions, &block_size_arr);
    let k_block = mlxcel_core::reshape(&k_block, &[1, 1, kv_len]);

    // Per-head vertical stride selection -> shape [n_heads, 1, kv_len].
    // vert[h, k] = ((k_block[k] + h + 1) % vert_stride) == 0.
    let heads = mlxcel_core::arange_i32(0, n_heads, 1);
    let heads = mlxcel_core::reshape(&heads, &[n_heads, 1, 1]);
    let vert_sum = mlxcel_core::add(&k_block, &heads); // [n_heads, 1, kv_len]
    let vert_sum = mlxcel_core::add(&vert_sum, &one_arr);
    let vert_mod = mlxcel_core::remainder(&vert_sum, &vert_stride_arr);
    let vert = mlxcel_core::equal(&vert_mod, &zero_arr); // bool [n_heads, 1, kv_len]

    // Block-level causal + local window -> shape [1, q_len, kv_len].
    let causal_block = mlxcel_core::greater_equal(&q_block, &k_block);
    let block_dist = mlxcel_core::subtract(&q_block, &k_block);
    let local = mlxcel_core::less(&block_dist, &local_blocks_arr);
    let local_or_vert = mlxcel_core::logical_or(&local, &vert); // [n_heads, q_len, kv_len]
    let block_attend = mlxcel_core::logical_and(&causal_block, &local_or_vert);

    // Token-level causality -> shape [1, q_len, kv_len].
    // Query token j sits at absolute position offset + j.
    let q_tok = mlxcel_core::arange_i32(offset, offset + q_len, 1);
    let q_tok = mlxcel_core::reshape(&q_tok, &[1, q_len, 1]);
    let k_tok = mlxcel_core::reshape(&k_positions, &[1, 1, kv_len]);
    let token_causal = mlxcel_core::greater_equal(&q_tok, &k_tok);

    // Combined boolean attend mask -> [n_heads, q_len, kv_len].
    let attend = mlxcel_core::logical_and(&block_attend, &token_causal);

    // Convert to an additive 0 / -inf mask and add the batch axis.
    let zero_f = mlxcel_core::zeros(&[1, 1, 1], dtype::FLOAT32);
    let neg_inf_f = mlxcel_core::full_f32(&[1, 1, 1], f32::NEG_INFINITY, dtype::FLOAT32);
    let additive = mlxcel_core::where_cond(&attend, &zero_f, &neg_inf_f);
    mlxcel_core::reshape(&additive, &[1, n_heads, q_len, kv_len])
}

// MLP with GeGELU activation.
pub struct MLP {
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
    pub gegelu_limit: f32,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.up_proj.forward(x);
        let h = mlxcel_core::utils::gegelu(&h, self.gegelu_limit);
        self.down_proj.forward(&h)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            up_proj,
            down_proj,
            gegelu_limit: args.gegelu_limit,
        })
    }
}

// Transformer Block.
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm FFN
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn =
            Attention::from_weights(weights, args, &format!("{}.self_attn", prefix), layer_idx)?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        // LayerNorm with bias
        let input_ln_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let input_ln_bias = weights
            .get(&format!("{}.input_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let input_layernorm =
            LayerNorm::new(input_ln_weight, input_ln_bias, args.layer_norm_epsilon);

        let post_ln_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let post_ln_bias = weights
            .get(&format!("{}.post_attention_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let post_attention_layernorm =
            LayerNorm::new(post_ln_weight, post_ln_bias, args.layer_norm_epsilon);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// Phi3Small Model.
pub struct Phi3SmallModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub final_layernorm: LayerNorm,
    pub mup_embedding_multiplier: f32,
    pub mup_width_multiplier: f32,
}

impl Phi3SmallModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens with mup scaling
        let mut h = self.embed_tokens.forward(input_ids);
        if self.mup_embedding_multiplier != 1.0 {
            h = mlxcel_core::multiply_scalar(&h, self.mup_embedding_multiplier);
        }

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.final_layernorm.forward(&h);

        // LM head (tied embeddings) with mup width scaling
        let mut logits = self.embed_tokens.as_linear(&h);
        if self.mup_width_multiplier != 1.0 {
            logits = mlxcel_core::divide_scalar(&logits, self.mup_width_multiplier);
        }
        logits
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final layernorm with bias
        let norm_weight = get_weight_copy(weights, "model.final_layernorm.weight")?;
        let norm_bias = weights
            .get("model.final_layernorm.bias")
            .map(|w| mlxcel_core::copy(w));
        let final_layernorm = LayerNorm::new(norm_weight, norm_bias, args.layer_norm_epsilon);

        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            mup_embedding_multiplier: args.mup_embedding_multiplier,
            mup_width_multiplier: args.mup_width_multiplier,
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
impl LanguageModel for Phi3SmallModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Phi3SmallModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Phi3SmallModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![32000] // Phi3Small EOS token
    }
}

#[cfg(test)]
#[path = "phi3small_tests.rs"]
mod tests;
