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

//! Apple OpenELM text model implementation using mlxcel-core.
//!
//! Ported from
//! <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/openelm.py>.
//!
//! OpenELM's defining feature is **layer-wise scaling**: nothing about a block's
//! width is a single global number. `num_query_heads`, `num_kv_heads` and
//! `ffn_multipliers` are per-layer lists, so every block is built with its own
//! head counts and its own FFN width. On `OpenELM-1_1B-Instruct` the query head
//! count climbs 16 -> 20 -> 24 -> 28 -> 32 across the 28 layers while the KV head
//! count climbs 4 -> 8, so the GQA grouping, the fused QKV output width and the
//! `out_proj` *input* width all differ layer to layer. A loader that reads head
//! counts once and reuses them builds layer 0's geometry for all 28 blocks.
//!
//! FFN width is not read from the config either. It is computed per layer as
//! `make_divisible(ffn_multipliers[i] * model_dim, ffn_dim_divisor)`, a rounding
//! helper carried over from the original TensorFlow MobileNet code.
//!
//! The rest is familiar: QK-RMSNorm on the head dimension (the qwen3 pattern),
//! GQA, SwiGLU through a single fused `proj_1` that is split in half, and shared
//! input/output embeddings.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.

/// Same shape as the per-family enums in [`crate::models::gpt2`] and
/// [`crate::models::helium`]; serde fails the whole config when one field does
/// not match its declared type, so the list form has to be accepted even though
/// published OpenELM checkpoints write a single int.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TokenIdField {
    Single(i32),
    Multiple(Vec<i32>),
}

impl TokenIdField {
    fn ids(&self) -> Vec<i32> {
        match self {
            Self::Single(id) => vec![*id],
            Self::Multiple(ids) => ids.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "openelm".to_string()
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_freq_constant() -> f32 {
    10000.0
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub model_dim: usize,
    pub head_dim: usize,
    pub num_transformer_layers: usize,
    pub ffn_dim_divisor: usize,

    /// Per-layer query head count. Not a scalar: on `OpenELM-1_1B-Instruct` this
    /// runs 16 through 32 across 28 layers.
    pub num_query_heads: Vec<usize>,

    /// Per-layer KV head count, paired index-for-index with
    /// [`Self::num_query_heads`].
    pub num_kv_heads: Vec<usize>,

    /// Per-layer FFN width multiplier, fed through [`make_divisible`] rather
    /// than used directly.
    pub ffn_multipliers: Vec<f32>,

    #[serde(default = "default_true")]
    pub ffn_with_glu: bool,

    #[serde(default = "default_true")]
    pub normalize_qk_projections: bool,

    #[serde(default = "default_true")]
    pub share_input_output_layers: bool,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_freq_constant")]
    pub rope_freq_constant: f32,

    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

impl ModelArgs {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdField::ids)
            .unwrap_or_default()
    }

    /// FFN width for one layer, matching upstream's
    /// `make_divisible(ffn_multipliers[i] * model_dim, ffn_dim_divisor)`.
    pub fn intermediate_size(&self, layer: usize) -> usize {
        make_divisible(
            self.ffn_multipliers[layer] * self.model_dim as f32,
            self.ffn_dim_divisor,
        )
    }

    /// Reject a config whose per-layer lists cannot describe the declared stack
    /// before anything indexes them. An MLX C++ exception crossing the cxx
    /// bridge is an uncatchable `std::terminate` at the first forward pass
    /// rather than a Rust error, so shape problems have to be caught here.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_transformer_layers == 0 {
            return Err("num_transformer_layers must be non-zero".to_string());
        }
        for (name, len) in [
            ("num_query_heads", self.num_query_heads.len()),
            ("num_kv_heads", self.num_kv_heads.len()),
            ("ffn_multipliers", self.ffn_multipliers.len()),
        ] {
            if len < self.num_transformer_layers {
                return Err(format!(
                    "{name} has {len} entries but num_transformer_layers is {}; \
                     OpenELM reads one entry per layer by index",
                    self.num_transformer_layers
                ));
            }
        }
        if self.head_dim == 0 || self.model_dim == 0 || self.vocab_size == 0 {
            return Err("head_dim, model_dim and vocab_size must be non-zero".to_string());
        }
        if self.ffn_dim_divisor == 0 {
            return Err("ffn_dim_divisor must be non-zero".to_string());
        }
        for layer in 0..self.num_transformer_layers {
            if self.num_query_heads[layer] == 0 || self.num_kv_heads[layer] == 0 {
                return Err(format!("layer {layer} declares zero query or KV heads"));
            }
            if !self.num_query_heads[layer].is_multiple_of(self.num_kv_heads[layer]) {
                return Err(format!(
                    "layer {layer} declares {} query heads against {} KV heads, \
                     which is not a whole GQA grouping",
                    self.num_query_heads[layer], self.num_kv_heads[layer]
                ));
            }
            if !self.ffn_multipliers[layer].is_finite() || self.ffn_multipliers[layer] <= 0.0 {
                return Err(format!(
                    "layer {layer} declares a non-positive or non-finite ffn multiplier"
                ));
            }
        }
        Ok(())
    }
}

/// Round a scaled width up to a multiple of `divisor`.
///
/// Carried over verbatim from upstream, which took it from the TensorFlow
/// MobileNet source. Two details are load-bearing and easy to lose in
/// translation: the floor is `divisor` itself (upstream's `min_value` defaults
/// to `divisor`), and the "do not round down by more than 10%" correction adds
/// one more `divisor` when the rounded value falls below `0.9 * v`.
pub fn make_divisible(v: f32, divisor: usize) -> usize {
    let min_value = divisor;
    // Python's `int()` truncates toward zero, which for the positive widths this
    // is called with is the same as an `as usize` cast.
    let truncated = (v + divisor as f32 / 2.0) as usize;
    let new_v = std::cmp::max(min_value, truncated / divisor * divisor);
    if (new_v as f32) < 0.9 * v {
        new_v + divisor
    } else {
        new_v
    }
}

// Attention.

/// One OpenELM attention block, built with **this layer's** head counts.
///
/// `qkv_proj` outputs `(n_heads + 2 * n_kv_heads) * head_dim` columns and
/// `out_proj` takes `n_heads * head_dim`, so both widths move with the layer.
pub struct Attention {
    pub qkv_proj: UnifiedLinear,
    pub out_proj: UnifiedLinear,
    pub q_norm: Option<RMSNorm>,
    pub k_norm: Option<RMSNorm>,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
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

        let qkv = self.qkv_proj.forward(x);

        // Upstream reshapes to [B, L, n_q + 2 * n_kv, head_dim], transposes, then
        // splits on the head axis. Reshaping groups consecutive channels into
        // heads, so slicing the channel axis at the same boundaries first is the
        // identical partition with one fewer transpose of a wider tensor.
        let q_end = self.num_heads * self.head_dim;
        let k_end = q_end + self.num_kv_heads * self.head_dim;
        let v_end = k_end + self.num_kv_heads * self.head_dim;

        let q = mlxcel_core::slice_last_dim(&qkv, 0, q_end);
        let k = mlxcel_core::slice_last_dim(&qkv, q_end, k_end);
        let v = mlxcel_core::slice_last_dim(&qkv, k_end, v_end);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // QK-RMSNorm normalizes over head_dim, which is the last axis both here
        // and after the transpose, so applying it before the transpose (the
        // qwen3 convention) matches upstream applying it after.
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k),
            None => k,
        };

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Generation calls `forward` with `mask == None`, so a multi-token
        // prefill has to build its own causal mask here or the prefill runs
        // bidirectionally (issues #991 / #999).
        let attn_out = if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.out_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let qkv_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.qkv_proj"), group_size, bits)?;
        let out_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.out_proj"), group_size, bits)?;

        let (q_norm, k_norm) = if args.normalize_qk_projections {
            (
                Some(RMSNorm::new(
                    get_weight_copy(weights, &format!("{prefix}.q_norm.weight"))?,
                    args.rms_norm_eps,
                )),
                Some(RMSNorm::new(
                    get_weight_copy(weights, &format!("{prefix}.k_norm.weight"))?,
                    args.rms_norm_eps,
                )),
            )
        } else {
            (None, None)
        };

        let head_dim = args.head_dim as i32;

        Ok(Self {
            qkv_proj,
            out_proj,
            q_norm,
            k_norm,
            num_heads: args.num_query_heads[layer] as i32,
            num_kv_heads: args.num_kv_heads[layer] as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_base: args.rope_freq_constant,
        })
    }
}

// Feed-forward.

/// SwiGLU through a single fused `proj_1` whose output is split in half. The
/// first half is the gate, matching upstream's `mx.split(x, 2, axis=-1)`
/// followed by `swiglu(gate, x)`.
pub struct MLP {
    pub proj_1: UnifiedLinear,
    pub proj_2: UnifiedLinear,
    pub intermediate_size: i32,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let fused = self.proj_1.forward(x);

        let gate = mlxcel_core::slice_last_dim(&fused, 0, self.intermediate_size);
        let up =
            mlxcel_core::slice_last_dim(&fused, self.intermediate_size, 2 * self.intermediate_size);

        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.proj_2.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let proj_1 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.proj_1"), group_size, bits)?;
        let proj_2 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.proj_2"), group_size, bits)?;

        Ok(Self {
            proj_1,
            proj_2,
            intermediate_size: args.intermediate_size(layer) as i32,
        })
    }
}

// Transformer block.

pub struct TransformerBlock {
    pub attn: Attention,
    pub ffn: MLP,
    pub attn_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.attn_norm.forward(x);
        let attn_out = self.attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.ffn_norm.forward(&h);
        let ffn_out = self.ffn.forward(&normed);
        mlxcel_core::add(&h, &ffn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer: usize,
    ) -> Result<Self, String> {
        let prefix = format!("transformer.layers.{layer}");

        let attn = Attention::from_weights(weights, args, layer, &format!("{prefix}.attn"))?;
        let ffn = MLP::from_weights(weights, args, layer, &format!("{prefix}.ffn"))?;

        let attn_norm = RMSNorm::new(
            get_weight_copy(weights, &format!("{prefix}.attn_norm.weight"))?,
            args.rms_norm_eps,
        );
        let ffn_norm = RMSNorm::new(
            get_weight_copy(weights, &format!("{prefix}.ffn_norm.weight"))?,
            args.rms_norm_eps,
        );

        Ok(Self {
            attn,
            ffn,
            attn_norm,
            ffn_norm,
        })
    }
}

// OpenELM model.

pub struct OpenElmModel {
    pub token_embeddings: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub eos_token_ids: Vec<i32>,
}

impl OpenElmModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.token_embeddings.forward(input_ids);

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.norm.forward(&h);

        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.token_embeddings.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;

        let group_size = args.group_size();
        let bits = args.bits();

        let token_embeddings = UnifiedEmbedding::from_weights(
            weights,
            "transformer.token_embeddings",
            group_size,
            bits,
        )?;

        let mut layers = Vec::with_capacity(args.num_transformer_layers);
        for i in 0..args.num_transformer_layers {
            layers.push(TransformerBlock::from_weights(weights, args, i)?);
        }

        let norm = RMSNorm::new(
            get_weight_copy(weights, "transformer.norm.weight")?,
            args.rms_norm_eps,
        );

        // `share_input_output_layers` is the OpenELM spelling of tied
        // embeddings; published checkpoints set it and ship no `lm_head`.
        let lm_head = if args.share_input_output_layers {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        Ok(Self {
            token_embeddings,
            layers,
            norm,
            lm_head,
            eos_token_ids: args.eos_token_ids(),
        })
    }
}

// Helper functions.

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.

impl LanguageModel for OpenElmModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        OpenElmModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        OpenElmModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "openelm_tests.rs"]
mod openelm_tests;
