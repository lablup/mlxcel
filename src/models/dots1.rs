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

//! dots.llm1 (`dots1`) model implementation using mlxcel-core
//!
//! Upstream reference:
//! <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/dots1.py>
//!
//! rednote's dots.llm1 is structurally DeepSeek-V3 *without* MLA:
//! - Standard multi-head attention (no GQA in the released checkpoint) with
//!   per-head Q/K RMSNorm applied after the reshape-to-heads and before
//!   transpose + RoPE (exactly like Qwen3).
//! - `first_k_dense_replace` leading layers are a plain SwiGLU MLP; the rest are
//!   a DeepSeek-V3-style MoE: sigmoid router with `e_score_correction_bias`,
//!   routed experts through `SwitchGLU`, plus a single fused shared MLP that is
//!   always added.
//! - Separate `q_proj`/`k_proj`/`v_proj`/`o_proj` projections (not fused). The
//!   mlx-community export is mixed 4/6-bit (`v_proj` and the `down_proj`s are
//!   6-bit), so the projections must stay separate: the unified loaders infer
//!   the real bit width per tensor from the packed shape.
//!
//! The router/expert/shared machinery mirrors `src/models/glm4_moe.rs` and
//! `src/models/deepseek_v3.rs`; the experts reuse the shared
//! `src/models/switch_layers.rs` `SwitchGLU`.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{slice_axis, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use super::switch_layers::{SwitchGLU, group_mask_scores, moe_weighted_sum};

// Configuration.

/// `eos_token_id` may be a single id or a list of ids depending on the export.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(i64),
    Multiple(Vec<i64>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,

    // MoE parameters.
    pub first_k_dense_replace: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    // Grouped routing. Null in the released checkpoint (treated as 1), in which
    // case the group-limited selection step degenerates to a no-op and is
    // skipped (mirrors the reference `k = n_group - topk_group == 0` guard).
    #[serde(default)]
    pub n_group: Option<usize>,
    #[serde(default)]
    pub topk_group: Option<usize>,

    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_max_position_embeddings() -> usize {
    32768
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_scoring_func() -> String {
    "sigmoid".to_string()
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

    /// Per-head dimension: explicit `head_dim` or `hidden_size / num_heads`.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Effective number of expert groups (defaults to 1 when absent).
    pub fn n_group(&self) -> usize {
        self.n_group.unwrap_or(1)
    }

    /// Effective number of selected groups (defaults to 1 when absent).
    pub fn topk_group(&self) -> usize {
        self.topk_group.unwrap_or(1)
    }

    /// `true` for MoE layers (layer index past the dense prefix).
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_k_dense_replace
    }

    /// Resolve the stop-token set for this checkpoint.
    ///
    /// The dots.llm1-inst export lists `<|endofresponse|>` (151649) in
    /// `config.json` and additionally `<|endoftext|>` (151643) in
    /// `generation_config.json`; both are merged so chat turns stop reliably.
    pub fn resolved_eos(&self) -> Vec<i32> {
        let mut eos: Vec<i32> = match &self.eos_token_id {
            Some(EosTokenId::Single(v)) => vec![*v as i32],
            Some(EosTokenId::Multiple(vs)) => vs.iter().map(|&v| v as i32).collect(),
            None => Vec::new(),
        };
        for fallback in [151643, 151649] {
            if !eos.contains(&fallback) {
                eos.push(fallback);
            }
        }
        eos
    }
}

// Attention: standard MHA with per-head Q/K RMSNorm.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
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

        // Separate Q/K/V projections (mixed 4/6-bit weights stay independent).
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq, heads, head_dim].
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Per-head Q/K RMSNorm BEFORE transpose (qwen3-style).
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // Transpose to [batch, heads, seq, head_dim].
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Non-traditional RoPE over the full head_dim, base = rope_theta.
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Update KV cache and fetch sliced views.
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention.
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

        // Transpose back and merge heads.
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_weight, args.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, args.rms_norm_eps),
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim,
            rope_base: args.rope_theta,
        })
    }
}

// Dense SwiGLU MLP. Used for the dense prefix layers and the shared expert.
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let gate_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?;
        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

// MoE block: DeepSeek-V3 sigmoid router + routed experts + shared expert.
pub struct Dots1MoE {
    pub router_weight: UniquePtr<MlxArray>,
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub experts: SwitchGLU,
    pub shared_experts: MLP,
    pub num_experts_per_tok: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
}

impl Dots1MoE {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        // Flatten [B, L, hidden] -> [n_tokens, hidden].
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Router logits: x @ router_weight.T.
        let router_t = mlxcel_core::transpose_axes(&self.router_weight, &[1, 0]);
        let logits = mlxcel_core::matmul(&x_flat, &router_t);

        // Sigmoid scoring; selection uses biased scores, the combine uses the
        // unbiased scores gathered at the selected experts.
        let scores = mlxcel_core::sigmoid(&logits);
        let orig_scores = mlxcel_core::copy(&scores);

        // `e_score_correction_bias` participates only in expert selection.
        let scores = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Group-limited selection. Skipped when n_group <= 1 or n_group ==
        // topk_group (reference `k = n_group - topk_group == 0`).
        let scores = if self.n_group > 1 && self.n_group != self.topk_group {
            group_mask_scores(&scores, self.n_group as i32, self.topk_group as i32)
        } else {
            scores
        };

        // Top-k experts by the biased scores.
        let k = self.num_experts_per_tok as i32;
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, k - 1, -1);
        let topk_indices = slice_axis(&indices, -1, 0, k);

        // Gather the UNBIASED sigmoid scores at the selected experts.
        let mut topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);

        if self.num_experts_per_tok > 1 && self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&topk_scores, -1, true);
            topk_scores = mlxcel_core::divide(&topk_scores, &sum);
        }

        let scale = mlxcel_core::from_slice_f32(&[self.routed_scaling_factor], &[1]);
        let topk_scores = mlxcel_core::multiply(&topk_scores, &scale);

        // Routed experts: [n_tokens, top_k, hidden] -> weighted sum [n, hidden].
        // Fused single-token decode kernel (#268) on by default; MLXCEL_FUSED_MOE=0
        // disables (dots.llm1 experts are gate/up 4-bit, down 6-bit); otherwise
        // the SwitchGLU + moe_weighted_sum path, also the kernel's fallback.
        let mut result = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1
                && crate::models::switch_layers::fused_moe_enabled()
            {
                self.experts
                    .forward_fused_kernel(&x_flat, &topk_indices, &topk_scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden_dim]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.experts.forward(&x_flat, &topk_indices);
                    moe_weighted_sum(&expert_out, &topk_scores, mlxcel_core::array_dtype(&x_flat))
                }
            }
        };

        // Shared expert is always added.
        let shared_out = self.shared_experts.forward(&x_flat);
        result = mlxcel_core::add(&result, &shared_out);

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
        let group_size = args.group_size();
        let bits = args.bits();

        // Router gate is a plain (unquantized) linear weight.
        let router_weight = get_weight_copy(weights, &format!("{}.gate.weight", prefix))?;
        let e_score_correction_bias =
            get_weight_copy(weights, &format!("{}.gate.e_score_correction_bias", prefix))?;

        let experts =
            SwitchGLU::from_weights(weights, &format!("{}.experts", prefix), group_size, bits)?;

        let shared_experts =
            MLP::from_weights(weights, args, &format!("{}.shared_experts", prefix))?;

        Ok(Self {
            router_weight,
            e_score_correction_bias,
            experts,
            shared_experts,
            num_experts_per_tok: args.num_experts_per_tok,
            n_group: args.n_group(),
            topk_group: args.topk_group(),
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

// Per-layer FFN: dense MLP (prefix layers) or MoE.
pub enum FFN {
    Dense(MLP),
    Moe(Dots1MoE),
}

impl FFN {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            FFN::Dense(mlp) => mlp.forward(x),
            FFN::Moe(moe) => moe.forward(x),
        }
    }
}

// Decoder layer.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: FFN,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;

        let mlp = if args.is_moe_layer(layer_idx) {
            FFN::Moe(Dots1MoE::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            FFN::Dense(MLP::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        };

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: RMSNorm::new(input_norm_weight, args.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_attn_norm_weight, args.rms_norm_eps),
        })
    }
}

// dots.llm1 model.
pub struct Dots1Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
    pub eos_token_ids: Vec<i32>,
}

impl Dots1Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
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
        let weights = Self::sanitize_weights(weights, &args);

        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args))
    }

    /// Stack per-expert MoE tensors and drop rotary inverse-frequency buffers.
    ///
    /// The mlx-community export already ships pre-stacked experts
    /// (`mlp.experts.{gate,up,down}_proj.*`); in that case the stacking step is a
    /// no-op (it only fires when per-expert `experts.0.*` tensors are present).
    /// Whole per-expert tensors are stacked with `stack_arrays`; quantized
    /// tensors are never sliced (that is unsound for packed weights).
    fn sanitize_weights(mut weights: WeightMap, args: &ModelArgs) -> WeightMap {
        for l in 0..args.num_hidden_layers {
            if !args.is_moe_layer(l) {
                continue;
            }
            let prefix = format!("model.layers.{}.mlp", l);

            for m in ["gate_proj", "up_proj", "down_proj"] {
                // Probe for per-expert tensors; if absent the layer is already
                // stacked and nothing is done.
                let probe = format!("{}.experts.0.{}.weight", prefix, m);
                if !weights.contains_key(&probe) {
                    continue;
                }

                for k in ["weight", "scales", "biases"] {
                    let first = format!("{}.experts.0.{}.{}", prefix, m, k);
                    if !weights.contains_key(&first) {
                        continue;
                    }

                    let mut expert_arrays = Vec::with_capacity(args.n_routed_experts);
                    for e in 0..args.n_routed_experts {
                        let key = format!("{}.experts.{}.{}.{}", prefix, e, m, k);
                        if let Some(w) = weights.get(&key) {
                            expert_arrays.push(mlxcel_core::copy(w));
                        }
                    }

                    if !expert_arrays.is_empty() {
                        let stacked = stack_arrays(&expert_arrays, 0);
                        weights.insert(format!("{}.experts.{}.{}", prefix, m, k), stacked);

                        for e in 0..args.n_routed_experts {
                            weights.remove(&format!("{}.experts.{}.{}.{}", prefix, e, m, k));
                        }
                    }
                }
            }
        }

        // Drop rotary inverse-frequency buffers (recomputed at runtime).
        let inv_freq_keys: Vec<String> = weights
            .keys()
            .filter(|k| k.contains("rotary_emb.inv_freq"))
            .cloned()
            .collect();
        for key in inv_freq_keys {
            weights.remove(&key);
        }

        weights
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // dots.llm1 has no multi-token-prediction layer: every hidden layer is
        // a real decoder layer (unlike DeepSeek-V3, which drops the last one).
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(weights, args, i)?);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Untied vocabulary head (`tie_word_embeddings == false`).
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            eos_token_ids: args.resolved_eos(),
        })
    }
}

// Helper.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for Dots1Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Dots1Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Dots1Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}
