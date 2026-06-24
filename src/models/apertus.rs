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

//! Apertus (Swiss AI) dense model implementation using mlxcel-core.
//!
//! Apertus is a Llama-style decoder (mirrored from
//! https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/apertus.py)
//! with three deltas:
//!
//! 1. **xIELU activation MLP (no gate).** `ApertusMLP` is
//!    `down_proj(xielu(up_proj(x)))` with two linears (no `gate_proj`, no
//!    SwiGLU). The xIELU activation carries per-layer learnable scalars
//!    `alpha_p` / `alpha_n` (stored pre-softplus) plus fixed `beta` / `eps`,
//!    mirroring
//!    https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/activations.py (XieLU).
//! 2. **QK-norm.** `RMSNorm(head_dim)` applied to Q and K after the
//!    reshape-to-heads and before transpose + RoPE (like
//!    `src/models/qwen3.rs`), gated on the `qk_norm` config flag.
//! 3. **llama3 RoPE scaling + non-standard norm names.** RoPE uses the llama3
//!    scaled frequencies (same helper shape as `src/models/exaone4.rs`). The
//!    per-layer norms are `attention_layernorm` / `feedforward_layernorm`, with
//!    standard residuals. `post_norm` is parsed but unused (false for
//!    Apertus-8B). Embeddings are untied (separate `lm_head`).

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::pipeline_hint;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

// Configuration.
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

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    /// QK-norm is true for Apertus-8B; gate the q_norm/k_norm on it.
    #[serde(default)]
    pub qk_norm: bool,

    /// Parsed for completeness but unused in the reference forward (false for
    /// Apertus-8B). See the module doc.
    #[serde(default)]
    pub post_norm: bool,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub mlp_bias: bool,

    /// False for Apertus-8B (untied embeddings, separate `lm_head`).
    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
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

/// Numerically stable softplus for the scalar activation parameters:
/// `max(x, 0) + ln(1 + exp(-|x|))`.
pub(crate) fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// xIELU activation, mirroring mlx-lm's `XieLU`
/// (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/activations.py).
///
/// `alpha_p` / `alpha_n` are the per-layer scalars AFTER softplus
/// (`alpha_p = softplus(p_raw)`, `alpha_n = beta + softplus(n_raw)`); `beta`
/// and `eps` are the fixed scalars. Element-wise:
/// - `x > 0`:  `alpha_p * x^2 + beta * x`
/// - `x <= 0`: `(expm1(min(x, eps)) - x) * alpha_n + beta * x`
fn apertus_xielu(
    x: &MlxArray,
    alpha_p: f32,
    alpha_n: f32,
    beta: f32,
    eps: f32,
) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(x);

    // Positive-branch core: alpha_p * x^2.
    let x_sq = mlxcel_core::square(x);
    let pos_core = mlxcel_core::multiply_scalar(&x_sq, alpha_p);

    // Negative-branch core: (expm1(min(x, eps)) - x) * alpha_n.
    let eps_arr = mlxcel_core::full_f32(&[1], eps, dtype);
    let clamped = mlxcel_core::minimum(x, &eps_arr);
    let neg_core = mlxcel_core::subtract(&mlxcel_core::expm1(&clamped), x);
    let neg_core = mlxcel_core::multiply_scalar(&neg_core, alpha_n);

    // Select the per-element branch on x > 0, then add the shared `beta * x`
    // once, outside the select. Both branches end in `+ beta * x`, so factoring
    // that term out drops one elementwise add (and its kernel launch) per layer
    // per token. The arithmetic is unchanged: the select only chooses which
    // value `beta * x` is added to, so greedy temp-0 decode stays bit-identical.
    let zero_arr = mlxcel_core::full_f32(&[1], 0.0, dtype);
    let cond = mlxcel_core::greater(x, &zero_arr);
    let selected = mlxcel_core::where_cond(&cond, &pos_core, &neg_core);
    mlxcel_core::add(&selected, &mlxcel_core::multiply_scalar(x, beta))
}

// Decode profiling (env-gated, default-off).
//
// Mirrors the lightweight hooks in `src/models/nemotron_h.rs` and
// `src/models/qwen3_moe.rs` for the dense apertus path. Both flags are read
// once from the environment and cached, so the steady-state decode path pays
// only an atomic load per token when profiling is disabled (the default).
//
// - `MLXCEL_PROFILE_BLOCKS`: attribute single-token decode wall-clock between
//   the attention and MLP (xIELU) sub-blocks. Active profiling `eval()`s the
//   sub-block outputs, which serializes the lazy graph; use only for analysis.
// - `MLXCEL_PROFILE_FORWARD`: split single-token decode into graph-build versus
//   GPU-eval time for the whole forward.

fn profile_blocks_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MLXCEL_PROFILE_BLOCKS").is_some())
}

fn profile_forward_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MLXCEL_PROFILE_FORWARD").is_some())
}

/// Route the MLP activation through the fused single-launch Metal xIELU kernel
/// (`mlxcel_core::fused_xielu`) instead of the ~11-op elementwise `apertus_xielu`
/// graph. Read once from `MLXCEL_FUSED_XIELU` and cached.
///
/// Default-on as of the M5 Max validation: the fused kernel keeps every
/// intermediate in the input dtype (bf16 for native Apertus) and reproduces
/// MLX's `expm1f`, so greedy temp-0 decode is byte-identical to the elementwise
/// path on Apple Silicon (see #409), and it measured a decode speedup on both
/// M1 Ultra (+2.7%) and M5 Max (+1.9%) with no regression. Set
/// `MLXCEL_FUSED_XIELU=0` (also `false`/`off`/`no`, case-insensitive) to force
/// the elementwise path; any other value, or leaving it unset, keeps the kernel
/// on. On non-Metal back-ends the FFI falls back to an equivalent elementwise
/// graph, so the kernel is safe everywhere. Mirrors the `MLXCEL_FUSED_MOE`
/// default-on opt-out convention.
fn fused_xielu_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        fused_xielu_enabled_from(std::env::var("MLXCEL_FUSED_XIELU").ok().as_deref())
    })
}

/// Pure decision behind [`fused_xielu_enabled`], split out so it can be
/// unit-tested without mutating process-global environment state. `None` (unset)
/// is on; an explicit `0`/`false`/`off`/`no` (case-insensitive, trimmed) is off;
/// any other value is on.
fn fused_xielu_enabled_from(value: Option<&str>) -> bool {
    match value {
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        None => true,
    }
}

/// Per-block decode timing accumulator. Constructed only when
/// `MLXCEL_PROFILE_BLOCKS` is set for a single-token decode step.
#[derive(Default)]
struct BlockTiming {
    attn_ns: u128,
    mlp_ns: u128,
}

impl BlockTiming {
    fn report(&self) {
        let total = (self.attn_ns + self.mlp_ns).max(1);
        eprintln!(
            "[BLOCKS] attn:{:.3}ms({:.0}%) mlp:{:.3}ms({:.0}%) total:{:.3}ms",
            self.attn_ns as f64 / 1e6,
            self.attn_ns as f64 * 100.0 / total as f64,
            self.mlp_ns as f64 / 1e6,
            self.mlp_ns as f64 * 100.0 / total as f64,
            total as f64 / 1e6,
        );
    }
}

// MLP (xIELU, no gate).
pub struct MLP {
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
    /// Post-softplus positive coefficient.
    pub alpha_p: f32,
    /// Post-softplus negative coefficient (already offset by `beta`).
    pub alpha_n: f32,
    pub beta: f32,
    pub eps: f32,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let up = self.up_proj.forward(x);
        let activated = if fused_xielu_enabled() {
            mlxcel_core::fused_xielu(&up, self.alpha_p, self.alpha_n, self.beta, self.eps)
        } else {
            apertus_xielu(&up, self.alpha_p, self.alpha_n, self.beta, self.eps)
        };
        self.down_proj.forward(&activated)
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

        // `beta` / `eps` ship in the checkpoint but are fixed constants; fall
        // back to the reference defaults when absent. Read `beta` before
        // `alpha_n` since `alpha_n = beta + softplus(alpha_n_raw)`.
        let beta = read_scalar(weights, &format!("{}.act_fn.beta", prefix)).unwrap_or(0.5);
        let eps = read_scalar(weights, &format!("{}.act_fn.eps", prefix)).unwrap_or(-1e-6);

        let alpha_p_raw = read_scalar(weights, &format!("{}.act_fn.alpha_p", prefix))
            .ok_or_else(|| format!("Weight not found: {}.act_fn.alpha_p", prefix))?;
        let alpha_n_raw = read_scalar(weights, &format!("{}.act_fn.alpha_n", prefix))
            .ok_or_else(|| format!("Weight not found: {}.act_fn.alpha_n", prefix))?;

        Ok(Self {
            up_proj,
            down_proj,
            alpha_p: softplus(alpha_p_raw),
            alpha_n: beta + softplus(alpha_n_raw),
            beta,
            eps,
        })
    }
}

// Attention (QK-norm + llama3-scaled RoPE).
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    /// `RMSNorm(head_dim)` for Q/K, present only when `qk_norm` is set.
    pub q_norm: Option<RMSNorm>,
    pub k_norm: Option<RMSNorm>,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    /// Pre-computed llama3-scaled frequencies (None = plain `rope_base` theta).
    pub rope_freqs: Option<UniquePtr<MlxArray>>,
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

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim].
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // QK-norm BEFORE transpose (per-head RMSNorm over head_dim).
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k),
            None => k,
        };

        // Transpose to [batch, n_heads, seq_len, head_dim].
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // RoPE AFTER normalization. Use the precomputed llama3-scaled freqs
        // when present, otherwise plain base theta.
        let (q, k) = if let Some(ref freqs) = self.rope_freqs {
            let q =
                mlxcel_core::fast_rope_with_freqs(&q, self.rope_dims, false, 1.0, offset, freqs);
            let k =
                mlxcel_core::fast_rope_with_freqs(&k, self.rope_dims, false, 1.0, offset, freqs);
            (q, k)
        } else {
            let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
            let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);
            (q, k)
        };

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

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

        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        rope_freqs: Option<UniquePtr<MlxArray>>,
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

        let head_dim = args.head_dim() as i32;

        let (q_norm, k_norm) = if args.qk_norm {
            let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
            let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;
            (
                Some(RMSNorm::new(q_norm_weight, args.rms_norm_eps)),
                Some(RMSNorm::new(k_norm_weight, args.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim,
            rope_base: args.rope_theta,
            rope_freqs,
        })
    }
}

// Transformer block (attention_layernorm / feedforward_layernorm naming).
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub attention_layernorm: RMSNorm,
    pub feedforward_layernorm: RMSNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_timed(x, cache, mask, None)
    }

    /// Block forward with optional per-sub-block timing. The production path
    /// passes `None`, which keeps the body identical to a plain forward (no
    /// extra `eval()` or `Instant`). When `timing` is `Some`, the attention and
    /// MLP outputs are `eval()`d to attribute wall-clock time to each.
    fn forward_timed(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        mut timing: Option<&mut BlockTiming>,
    ) -> UniquePtr<MlxArray> {
        // h = x + self_attn(attention_layernorm(x))
        let attn_start = timing.as_ref().map(|_| Instant::now());
        let normed = self.attention_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);
        if let Some(t) = timing.as_mut() {
            mlxcel_core::eval(&h);
            t.attn_ns += attn_start.unwrap().elapsed().as_nanos();
        }

        // out = h + mlp(feedforward_layernorm(h))
        let mlp_start = timing.as_ref().map(|_| Instant::now());
        let normed = self.feedforward_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        let out = mlxcel_core::add(&h, &ff_out);
        if let Some(t) = timing.as_mut() {
            mlxcel_core::eval(&out);
            t.mlp_ns += mlp_start.unwrap().elapsed().as_nanos();
        }
        out
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
        rope_freqs: Option<UniquePtr<MlxArray>>,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn =
            Attention::from_weights(weights, args, &format!("{}.self_attn", prefix), rope_freqs)?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let attention_norm_weight =
            get_weight_copy(weights, &format!("{}.attention_layernorm.weight", prefix))?;
        let feedforward_norm_weight =
            get_weight_copy(weights, &format!("{}.feedforward_layernorm.weight", prefix))?;

        Ok(Self {
            self_attn,
            mlp,
            attention_layernorm: RMSNorm::new(attention_norm_weight, args.rms_norm_eps),
            feedforward_layernorm: RMSNorm::new(feedforward_norm_weight, args.rms_norm_eps),
        })
    }
}

// Apertus model.
pub struct ApertusModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    /// Separate head when embeddings are untied (Apertus-8B); `None` falls back
    /// to `embed_tokens.as_linear`.
    pub lm_head: Option<UnifiedLinear>,
}

impl ApertusModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Decode profiling is off by default: the cached flags resolve to two
        // atomic loads and `array_shape` is only queried when a flag is set, so
        // the steady-state decode path is unchanged.
        let profiling = profile_blocks_enabled() || profile_forward_enabled();
        let is_decode = profiling && mlxcel_core::array_shape(input_ids)[1] == 1;
        let profile_blocks = is_decode && profile_blocks_enabled();
        let profile_forward = is_decode && !profile_blocks;
        let build_start = profile_forward.then(Instant::now);

        let mut h = self.embed_tokens.forward(input_ids);
        let mut timing = profile_blocks.then(BlockTiming::default);

        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_timed(&h, &mut caches[i], mask, timing.as_mut());
            pipeline_hint(&h, i, n);
        }

        let h = self.norm.forward(&h);

        // No logit multiplier for Apertus.
        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        };

        if let Some(t) = timing {
            t.report();
        }
        if let Some(start) = build_start {
            let build_ms = start.elapsed().as_nanos() as f64 / 1e6;
            let eval_start = Instant::now();
            mlxcel_core::eval(&logits);
            let eval_ms = eval_start.elapsed().as_nanos() as f64 / 1e6;
            eprintln!(
                "[FORWARD] build:{:.3}ms eval:{:.3}ms total:{:.3}ms",
                build_ms,
                eval_ms,
                build_ms + eval_ms,
            );
        }

        logits
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
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // llama3-scaled RoPE frequencies are computed once and shared (copied)
        // into every layer's attention.
        let rope_freqs = compute_rope_freqs(args);

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let freqs_i = rope_freqs.as_ref().map(|f| mlxcel_core::copy(f));
            layers.push(TransformerBlock::from_weights(weights, args, i, freqs_i)?);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Untied (Apertus-8B): load `lm_head`. Tied: reuse the embedding.
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
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

/// Read a checkpoint scalar (e.g. an `act_fn` parameter) as `f32`. The
/// reference `sanitize` squeezes `alpha_p` / `alpha_n` (they may ship as
/// `[1]` or `[1, 1]`); squeezing first makes the `item` read shape-agnostic.
fn read_scalar(weights: &WeightMap, name: &str) -> Option<f32> {
    weights.get(name).map(|w| {
        let squeezed = mlxcel_core::squeeze(w);
        // The `act_fn` scalars ship as bf16; cast to f32 before the item read so
        // the value (not the raw bytes) is recovered. Without this the xIELU
        // coefficients collapse to ~zero and the MLP branch vanishes.
        let as_f32 = mlxcel_core::astype(&squeezed, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&as_f32);
        mlxcel_core::item_f32(&as_f32)
    })
}

/// Compute llama3 RoPE frequencies from `rope_scaling`. Returns `None` for
/// default RoPE (absent scaling or an unsupported `rope_type`), mirroring
/// mlx-lm's `Llama3RoPE`
/// (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py).
fn compute_rope_freqs(args: &ModelArgs) -> Option<UniquePtr<MlxArray>> {
    let scaling = args.rope_scaling.as_ref()?;
    let rope_type = scaling
        .get("rope_type")
        .or_else(|| scaling.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    if rope_type != "llama3" {
        return None;
    }

    let factor = scaling
        .get("factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let low_freq_factor = scaling
        .get("low_freq_factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let high_freq_factor = scaling
        .get("high_freq_factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(4.0) as f32;
    let old_context_len = scaling
        .get("original_max_position_embeddings")
        .and_then(|v| v.as_f64())
        .unwrap_or(8192.0) as f32;

    let dims = args.head_dim();
    let base = args.rope_theta;

    let low_freq_wavelen = old_context_len / low_freq_factor;
    let high_freq_wavelen = old_context_len / high_freq_factor;

    // freqs = base^(arange(0, dims, 2) / dims), adjusted per the llama3 bands.
    let half_dims = dims / 2;
    let mut freq_vals = Vec::with_capacity(half_dims);
    for i in 0..half_dims {
        let exp = (2 * i) as f32 / dims as f32;
        let freq = base.powf(exp);
        let wavelen = 2.0 * std::f32::consts::PI * freq;

        let adjusted = if wavelen > low_freq_wavelen {
            // Low frequency (long wavelength): scale by factor.
            freq * factor
        } else if wavelen > high_freq_wavelen {
            // Medium frequency: smooth interpolation.
            let smooth = (old_context_len / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            freq / ((1.0 - smooth) / factor + smooth)
        } else {
            // High frequency (short wavelength): unchanged.
            freq
        };
        freq_vals.push(adjusted);
    }

    Some(mlxcel_core::from_slice_f32(&freq_vals, &[half_dims as i32]))
}

// LanguageModel trait implementation.
impl LanguageModel for ApertusModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        ApertusModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        ApertusModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Apertus-8B-Instruct generation_config: </s>, <|assistant_end|>,
        // <|tools_suffix|>.
        vec![2, 68, 72]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // bf16 dtype id for `astype` (see mlx_cxx_internal.h to_dtype()).
    const BF16: i32 = 12;

    /// The fused xIELU Metal kernel must be bit-for-bit identical to the
    /// elementwise `apertus_xielu` reference on the native bf16 path, so the
    /// `MLXCEL_FUSED_XIELU` flag never perturbs greedy temp-0 decode. The kernel
    /// keeps every intermediate in bf16 and reproduces MLX's expm1f, so each
    /// sub-expression rounds exactly where the reference rounds. The test spans
    /// the full activation domain: large/small positive, the `x > 0` boundary,
    /// near-zero negatives that straddle `eps`, and large negatives that
    /// saturate `expm1`. On a non-Metal back-end `fused_xielu` runs the
    /// equivalent elementwise fallback, which is identical by construction.
    #[test]
    fn fused_xielu_matches_elementwise_bit_for_bit() {
        // Per-layer scalars in the range the Apertus-8B checkpoint produces:
        // alpha_p/alpha_n are post-softplus (positive), beta/eps are the fixed
        // reference defaults.
        let alpha_p = 0.8731f32;
        let alpha_n = 0.6042f32;
        let beta = 0.5f32;
        let eps = -1e-6f32;

        let mut vals: Vec<f32> = vec![
            0.0, -0.0, 1e-7, -1e-7, 5e-7, -5e-7, 1e-6, -1e-6, 2e-6, -2e-6, 1e-4, -1e-4, 0.001,
            -0.001, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5, 8.0, -8.0, 42.0, -42.0, 90.0, -90.0,
        ];
        // A deterministic spread so the test exercises a dense range, not just
        // the hand-picked edge cases above.
        for k in 0..2048i32 {
            vals.push((k as f32 - 1024.0) * 0.05);
        }
        let n = vals.len() as i32;

        let x_f32 = mlxcel_core::from_slice_f32(&vals, &[1, 1, n]);
        let x = mlxcel_core::astype(&x_f32, BF16);

        let reference = apertus_xielu(&x, alpha_p, alpha_n, beta, eps);
        let fused = mlxcel_core::fused_xielu(&x, alpha_p, alpha_n, beta, eps);

        let equal = mlxcel_core::array_equal(&reference, &fused, true);
        mlxcel_core::eval(&equal);
        let all_equal = mlxcel_core::item_f32(&equal) != 0.0;

        if !all_equal {
            // Report the worst element-wise gap to make a regression debuggable.
            let diff = mlxcel_core::abs(&mlxcel_core::subtract(&reference, &fused));
            let max_diff = mlxcel_core::max_all(&diff);
            mlxcel_core::eval(&max_diff);
            panic!(
                "fused xIELU diverged from the elementwise reference (max abs diff {})",
                mlxcel_core::item_f32(&max_diff)
            );
        }
    }

    /// The `MLXCEL_FUSED_XIELU` gate is default-on with an opt-out, mirroring
    /// `MLXCEL_FUSED_MOE` (#282). Unset, or any unrecognized value, keeps the
    /// fused kernel on; an explicit `0`/`false`/`off`/`no` (case-insensitive,
    /// trimmed) forces the elementwise reference path. Greedy temp-0 decode is
    /// byte-identical either way (see `fused_xielu_matches_elementwise_bit_for_bit`),
    /// so flipping the default is a pure throughput change.
    #[test]
    fn fused_xielu_gate_defaults_on_with_opt_out() {
        // Default-on: unset and unrecognized values enable the kernel.
        assert!(fused_xielu_enabled_from(None));
        assert!(fused_xielu_enabled_from(Some("1")));
        assert!(fused_xielu_enabled_from(Some("on")));
        assert!(fused_xielu_enabled_from(Some("anything")));
        // Opt-out: explicit falsy values force the elementwise path.
        assert!(!fused_xielu_enabled_from(Some("0")));
        assert!(!fused_xielu_enabled_from(Some("false")));
        assert!(!fused_xielu_enabled_from(Some("off")));
        assert!(!fused_xielu_enabled_from(Some("no")));
        // Trimmed and case-insensitive.
        assert!(!fused_xielu_enabled_from(Some("  OFF  ")));
    }
}
