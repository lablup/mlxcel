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

//! Databricks DBRX MoE model implementation using mlxcel-core.
//!
//! Ported from
//! <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/dbrx.py>.
//!
//! What separates DBRX from the other sparse-MoE decoders already in this tree:
//! - A fused `Wqkv` projection whose output is **clipped** to
//!   `[-clip_qkv, +clip_qkv]` before the uneven Q/K/V split.
//! - A "norm-attn-norm" block: one norm feeds attention, the residual is added,
//!   and a *second* norm of that residual feeds the FFN. The FFN therefore reads
//!   a different tensor than the one it adds back to.
//! - `nn.LayerNorm(bias=False)` rather than RMSNorm.
//! - Expert projections named `w1` (gate), `v1` (up), `w2` (down), living under
//!   `transformer.blocks.{i}.ffn.experts.{e}`.
//!
//! Config geometry is nested under `attn_config` / `ffn_config` instead of the
//! flat Llama-style fields, and the layer stack is `transformer.blocks` rather
//! than `model.layers`.

use crate::models::switch_layers::SwitchGLU;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

/// `<|endoftext|>` in the DBRX tiktoken vocabulary. Published DBRX configs
/// leave `eos_token_id` null and carry the id only in `tokenizer_config.json`,
/// so the model-side fallback has to name it explicitly.
const DBRX_EOS_TOKEN_ID: i32 = 100257;

/// MLX's `nn.LayerNorm` default. DBRX configs carry no epsilon field of their
/// own, so the upstream default is the only source for it.
const DBRX_LAYER_NORM_EPS: f32 = 1e-5;

// Configuration.

/// Same shape as the per-family enums in [`crate::models::gpt2`] and
/// [`crate::models::helium`]; serde fails the whole config when one field does
/// not match its declared type, so the list form has to be accepted even though
/// published DBRX checkpoints leave the field null.
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

/// Attention geometry, nested under `attn_config` in the checkpoint config.
///
/// Published DBRX configs write a full `PretrainedConfig` dump into this block
/// (dozens of generation-time fields such as `top_k` and `temperature` that
/// have nothing to do with attention), so only the four fields that matter are
/// declared here and serde drops the rest.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct AttnConfig {
    #[serde(default)]
    pub kv_n_heads: Option<usize>,

    /// Symmetric clamp applied to the fused QKV projection output. `null` in a
    /// config means "no clipping", which is why this is an `Option` rather than
    /// a defaulted float: 0.0 would clamp everything to zero.
    #[serde(default)]
    pub clip_qkv: Option<f32>,

    #[serde(default)]
    pub rope_theta: Option<f32>,
}

/// Feed-forward / MoE geometry, nested under `ffn_config`. Carries the same
/// `PretrainedConfig` noise as [`AttnConfig`].
#[derive(Debug, Clone, Deserialize, Default)]
pub struct FfnConfig {
    #[serde(default)]
    pub ffn_hidden_size: Option<usize>,

    #[serde(default)]
    pub moe_num_experts: Option<usize>,

    #[serde(default)]
    pub moe_top_k: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,

    #[serde(default)]
    pub attn_config: AttnConfig,

    #[serde(default)]
    pub ffn_config: FfnConfig,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_model_type() -> String {
    "dbrx".to_string()
}

impl ModelArgs {
    pub fn num_kv_heads(&self) -> usize {
        self.attn_config.kv_n_heads.unwrap_or(self.n_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn rope_theta(&self) -> f32 {
        self.attn_config.rope_theta.unwrap_or(10000.0)
    }

    pub fn clip_qkv(&self) -> Option<f32> {
        self.attn_config.clip_qkv
    }

    pub fn ffn_hidden_size(&self) -> usize {
        self.ffn_config.ffn_hidden_size.unwrap_or(0)
    }

    pub fn moe_num_experts(&self) -> usize {
        self.ffn_config.moe_num_experts.unwrap_or(0)
    }

    pub fn moe_top_k(&self) -> usize {
        self.ffn_config.moe_top_k.unwrap_or(1)
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

    pub fn eos_token_ids(&self) -> Vec<i32> {
        match self.eos_token_id.as_ref().map(TokenIdField::ids) {
            Some(ids) if !ids.is_empty() => ids,
            _ => vec![DBRX_EOS_TOKEN_ID],
        }
    }
}

// Attention.

/// DBRX attention: one fused `Wqkv`, a symmetric clamp, then an **uneven**
/// three-way split. Query occupies `d_model` columns while key and value take
/// `head_dim * kv_n_heads` each, so the split offsets are not thirds.
pub struct Attention {
    pub wqkv: UnifiedLinear,
    pub out_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    pub clip_qkv: Option<f32>,
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

        let qkv = self.wqkv.forward(x);

        // Clamp before the split, matching upstream. Applying it after would
        // change nothing numerically, but keeping the order identical keeps the
        // graph comparable against the reference when diffing activations.
        let qkv = match self.clip_qkv {
            Some(limit) => {
                let dtype = mlxcel_core::array_dtype(&qkv);
                let lo = mlxcel_core::full_f32(&[1], -limit, dtype);
                let hi = mlxcel_core::full_f32(&[1], limit, dtype);
                mlxcel_core::clip(&qkv, &lo, &hi)
            }
            None => qkv,
        };

        let q_end = self.num_heads * self.head_dim;
        let k_end = q_end + self.num_kv_heads * self.head_dim;
        let v_end = k_end + self.num_kv_heads * self.head_dim;

        let q = mlxcel_core::slice(&qkv, &[0, 0, 0], &[b, l, q_end]);
        let k = mlxcel_core::slice(&qkv, &[0, 0, q_end], &[b, l, k_end]);
        let v = mlxcel_core::slice(&qkv, &[0, 0, k_end], &[b, l, v_end]);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // DBRX uses non-traditional (split-half) RoPE over the full head_dim.
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
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let wqkv =
            UnifiedLinear::from_weights(weights, &format!("{}.Wqkv", prefix), group_size, bits)?;
        let out_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.out_proj", prefix),
            group_size,
            bits,
        )?;

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            wqkv,
            out_proj,
            num_heads: args.n_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_base: args.rope_theta(),
            clip_qkv: args.clip_qkv(),
        })
    }
}

// Sparse MoE block.

/// DBRX routed experts. `w1` is the gate projection, `v1` the up projection and
/// `w2` the down projection, which is why the shared [`SwitchGLU`] loader is
/// given those leaf names instead of the `gate_proj`/`up_proj`/`down_proj`
/// default.
pub struct SparseMoeBlock {
    pub router: UnifiedLinear,
    pub experts: SwitchGLU,
    pub num_experts_per_tok: usize,
}

impl SparseMoeBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        let logits = self.router.forward(&x_flat);

        let k = self.num_experts_per_tok as i32;
        let n_experts = mlxcel_core::array_shape(&logits)[1];
        let kth = n_experts - k;

        let indices = mlxcel_core::argpartition(&logits, kth, -1);
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices =
            mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

        // Upstream softmaxes over *all* experts, selects top-k, then divides by
        // the L1 norm of the selected scores (`moe_normalize_expert_weights: 1`).
        // Softmax is monotonic, so the selection is unchanged, and
        // `softmax_all(z)_i / sum_{j in topk} softmax_all(z)_j` reduces exactly
        // to a softmax over the top-k logits. Taking the shorter route keeps the
        // graph smaller without changing the result.
        let topk_logits = mlxcel_core::take_along_axis(&logits, &topk_indices, -1);
        let scores = mlxcel_core::softmax(&topk_logits, -1);

        let result = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1
                && crate::models::switch_layers::fused_moe_enabled()
            {
                self.experts
                    .forward_fused_kernel(&x_flat, &topk_indices, &scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden_dim]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.experts.forward(&x_flat, &topk_indices);
                    crate::models::switch_layers::moe_weighted_sum(
                        &expert_out,
                        &scores,
                        mlxcel_core::array_dtype(&x_flat),
                    )
                }
            }
        };

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.router.layer", prefix),
            args.group_size(),
            args.bits(),
        )?;

        // `{prefix}.switch_mlp` is the virtual prefix the shared loader keys off
        // to find the per-expert layout `{prefix}.experts.{idx}.{leaf}`, which is
        // exactly how mlx-community DBRX conversions ship: the upstream
        // stacked-tensor split already ran before quantization, so nothing here
        // has to slice a packed quantized tensor (which would be unsound for
        // `gather_qmm`).
        let experts = SwitchGLU::from_weights_with_proj_names(
            weights,
            &format!("{}.switch_mlp", prefix),
            args.group_size(),
            args.bits(),
            ["w1", "v1", "w2"], // gate=w1, up=v1, down=w2
        )?;

        Ok(Self {
            router,
            experts,
            num_experts_per_tok: args.moe_top_k(),
        })
    }
}

// Transformer block.

/// The DBRX "norm-attn-norm" block.
///
/// Unlike the standard pre-norm layout, the second norm does not feed a
/// residual of its own: `norm_2` normalizes the *post-attention residual*, the
/// FFN consumes that, and the FFN output is added back to the un-normalized
/// residual. Mirroring this faithfully matters, since the tensor the FFN reads
/// and the tensor it adds to are different.
pub struct DecoderLayer {
    pub norm_1: LayerNorm,
    pub attn: Attention,
    pub norm_2: LayerNorm,
    pub ffn: SparseMoeBlock,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.norm_1.forward(x);
        let attn_out = self.attn.forward(&normed, cache, mask);
        let residual = mlxcel_core::add(x, &attn_out);

        let normed = self.norm_2.forward(&residual);
        let ffn_out = self.ffn.forward(&normed);
        mlxcel_core::add(&residual, &ffn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("transformer.blocks.{}", layer_idx);

        let attn =
            Attention::from_weights(weights, args, &format!("{}.norm_attn_norm.attn", prefix))?;
        let ffn = SparseMoeBlock::from_weights(weights, args, &format!("{}.ffn", prefix))?;

        // `nn.LayerNorm(..., bias=False)` upstream, so no bias tensor exists.
        let norm_1 = LayerNorm::new(
            get_weight_copy(weights, &format!("{}.norm_attn_norm.norm_1.weight", prefix))?,
            None,
            DBRX_LAYER_NORM_EPS,
        );
        let norm_2 = LayerNorm::new(
            get_weight_copy(weights, &format!("{}.norm_attn_norm.norm_2.weight", prefix))?,
            None,
            DBRX_LAYER_NORM_EPS,
        );

        Ok(Self {
            norm_1,
            attn,
            norm_2,
            ffn,
        })
    }
}

// DBRX model.

pub struct DbrxModel {
    pub wte: UnifiedEmbedding,
    pub blocks: Vec<DecoderLayer>,
    pub norm_f: LayerNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub eos_token_ids: Vec<i32>,
}

impl DbrxModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.wte.forward(input_ids);

        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &mut caches[i], mask);
        }

        let h = self.norm_f.forward(&h);

        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.wte.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.blocks.len()).map(|_| KVCache::new()).collect()
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
        let group_size = args.group_size();
        let bits = args.bits();

        // Published DBRX conversions leave `transformer.wte` unquantized even
        // when every projection is 4-bit, so the shared loader's quantized/plain
        // detection is what picks the right path here.
        let wte = UnifiedEmbedding::from_weights(weights, "transformer.wte", group_size, bits)?;

        let mut blocks = Vec::with_capacity(args.n_layers);
        for i in 0..args.n_layers {
            blocks.push(DecoderLayer::from_weights(weights, args, i)?);
        }

        let norm_f = LayerNorm::new(
            get_weight_copy(weights, "transformer.norm_f.weight")?,
            None,
            DBRX_LAYER_NORM_EPS,
        );

        let lm_head = if !args.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            wte,
            blocks,
            norm_f,
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

impl LanguageModel for DbrxModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        DbrxModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        DbrxModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "dbrx_tests.rs"]
mod dbrx_tests;
