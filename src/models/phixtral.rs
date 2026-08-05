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

//! Phixtral (`phi-msft` with `num_local_experts`), a Mixtral-style sparse MoE
//! on the Phi-2 backbone.
//!
//! Ported from mlx-lm's
//! [`phixtral.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/phixtral.py).
//!
//! # There is no `phixtral` model_type, and the arm that would catch it exists
//!
//! No published phixtral checkpoint declares `model_type: "phixtral"`.
//! `mlabonne/phixtral-4x2_8` declares **`phi-msft`**, and upstream reaches this
//! implementation through a rename table rather than the config value:
//!
//! ```python
//! # mlx_lm/utils.py
//! MODEL_REMAPPING = { ..., "phi-msft": "phixtral", ... }
//! ```
//!
//! This tree already had an arm for that string, pointing at the **dense** Phi
//! decoder, so before this port a phixtral checkpoint routed to
//! [`crate::models::phi`] and failed on the first missing `self_attn.q_proj`.
//! Adding a `"phixtral"` arm would have fixed nothing. The arm has to be
//! *discriminated* instead, which
//! [`crate::models::detection::detect_phi_model_type`] does on
//! `num_local_experts`, the same signal upstream's config carries.
//!
//! # The checkpoint is not Llama-named, and it is not `phi`-named either
//!
//! [`crate::models::phi`] loads the modern `model.layers.{i}.self_attn.q_proj`
//! spelling. Phixtral ships the original Microsoft layout throughout:
//! `transformer.embd.wte`, `transformer.h.{i}.ln`,
//! `transformer.h.{i}.mixer.Wqkv`, `transformer.h.{i}.mixer.out_proj`,
//! `transformer.h.{i}.moe.gate`, and an output head split into
//! `lm_head.ln` + `lm_head.linear`. None of it overlaps the dense Phi loader,
//! which is why this is a separate module rather than a flag on that one.
//!
//! # Upstream reads four config fields it cannot find
//!
//! `phixtral.py`'s `ModelArgs` names `num_vocab`, `model_dim`, `num_heads` and
//! `num_layers`, while the checkpoint's `config.json` writes `vocab_size`,
//! `n_embd`, `n_head` and `n_layer`. `BaseModelArgs.from_dict` keeps only keys
//! that match a field name, so **upstream silently falls back to its defaults**
//! for all four. It works on `mlabonne/phixtral-4x2_8` only because that model
//! is built on Phi-2, whose dimensions are exactly those defaults (51200 /
//! 2560 / 32 / 32). A phixtral of any other size would load upstream at the
//! wrong shape.
//!
//! This loader reads the spellings the checkpoint actually uses, and accepts
//! upstream's names as aliases, so it is correct for both. See [`ModelArgs`].
//!
//! # The block is parallel-residual, and the norm is shared
//!
//! One LayerNorm per block feeds attention **and** the MoE, and their outputs
//! are summed into the residual: `x + mixer(ln(x)) + moe(ln(x))`. There is no
//! post-attention norm. Feeding the MoE the attention output instead is the
//! obvious mistake and produces fluent text.
//!
//! # Routing softmaxes the top-k logits, not the full row
//!
//! `MOE.__call__` selects on the raw gate logits, gathers those logits at the
//! selected indices, and only then softmaxes **over the k gathered values**.
//! Softmaxing the full expert row first and gathering afterwards is a different
//! normalization (it divides by the sum over all `num_local_experts` instead of
//! over the selected `k`), and it leaves every output finite. See
//! [`PhixtralMoe::forward`].
//!
//! # The published checkpoint's router is degenerate, so the tie-break decides
//!
//! Every gate row of `mlabonne/phixtral-4x2_8` is **identical across all four
//! experts, in every one of its 32 layers** (verified by dequantizing the
//! router). Each token's four logits are therefore exactly tied: the routing
//! weights are always a uniform `1/k`, and *which* experts run is decided
//! entirely by how `argpartition` breaks the tie.
//!
//! That makes an orientation that would otherwise be cosmetic load-bearing.
//! `argpartition(-gates, kth = k - 1)` then the first `k`, which is upstream's
//! form, selects experts 0 and 1 here; the equivalent-looking
//! `argpartition(gates, kth = n - k)` then the last `k` selects 2 and 3. The
//! experts are distinct, so that is a different output, not a reordering of the
//! same one, and it moves every layer's MoE result by ~0.5%. This port mirrors
//! upstream's form; see [`PhixtralMoe::forward`].
//!
//! # Attention runs in float32, and that is not a stray cast
//!
//! Upstream casts the queries to `float32` immediately before the SDPA call and
//! back to the values' dtype immediately after, which promotes the whole score
//! computation. It is required: Phi-2 carries large outlier activations, this
//! checkpoint ships float16, and the `q @ k^T` products reach the 65504 ceiling
//! in the deep layers. Running the scores in f16 produces NaN from layer 30 of
//! 32 on a four-token prompt, with layers 0 through 29 tracking the reference to
//! three decimal places first, so the only symptom is a late, total one. See
//! [`Attention::forward`].
//!
//! # The experts carry biases
//!
//! Phixtral's `SwitchMLP` is built with `bias=True`, so each expert projection
//! has a `[num_local_experts, out_features]` bias plane on top of the stacked
//! weight. [`crate::models::switch_layers::SwitchLinear`] implements no bias, so
//! the per-expert row is gathered and added here; see
//! [`PhixtralSwitchMlp::forward`]. `SwitchGLU` cannot be reused at all
//! regardless: these experts are a two-projection `fc2(gelu(fc1(x)))` MLP, not
//! a gated SwiGLU triple.
//!
//! # Untrusted config
//!
//! Same contract as the other ports in this tree: `config.json` arrives from a
//! third-party HuggingFace repo in the ordinary `mlxcel generate -m <org>/<repo>`
//! flow, so [`ModelArgs::validate`] rejects every scalar that could size an
//! allocation, divide, or violate an undocumented MLX C++ precondition, and
//! [`validate_weights`] rejects every tensor whose real shape disagrees with the
//! config. An MLX C++ exception crossing the cxx bridge is an uncatchable
//! `std::terminate` at the first forward pass, not a Rust error.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use crate::models::gpt2::{dim_eq, validate_embedding_table};
use crate::models::switch_layers::{SwitchLinear, gather_sort, moe_weighted_sum, scatter_unsort};

// Configuration.

/// Phixtral `config.json`.
///
/// Every dimension is read under the spelling the checkpoint uses, with
/// upstream's `ModelArgs` field name accepted as a serde alias. Upstream itself
/// reads only the alias and therefore falls back to a default for all four; see
/// the module docs.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    /// `n_embd` on `mlabonne/phixtral-4x2_8` (2560).
    #[serde(default = "default_model_dim", alias = "model_dim")]
    pub n_embd: usize,

    /// `n_head` (32). Phixtral is multi-head: there is no KV-head count, and
    /// the fused `Wqkv` is an even three-way split.
    #[serde(default = "default_num_heads", alias = "num_heads")]
    pub n_head: usize,

    /// `n_layer` (32).
    #[serde(default = "default_num_layers", alias = "num_layers")]
    pub n_layer: usize,

    /// `vocab_size` (51200).
    #[serde(default = "default_num_vocab", alias = "num_vocab")]
    pub vocab_size: usize,

    /// Channels per head that RoPE rotates, 32 of the 80-wide head on
    /// `phixtral-4x2_8`. The remaining 48 pass through unrotated.
    #[serde(default = "default_rotary_dim")]
    pub rotary_dim: usize,

    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: usize,

    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    /// Not a field of upstream's `ModelArgs`: upstream builds a plain
    /// `nn.LayerNorm(dims)` and takes MLX's default epsilon, which happens to be
    /// the `1e-5` this checkpoint declares. Read here so a checkpoint that
    /// declares something else is honoured rather than silently overridden.
    #[serde(default = "default_layer_norm_epsilon", alias = "layer_norm_eps")]
    pub layer_norm_epsilon: f32,

    /// Explicit FFN width. Upstream hardcodes `mlp_dims = model_dim * 4`, and
    /// `phixtral-4x2_8` writes `null` here, which is that same `4 * 2560`. A
    /// checkpoint that declares a width is honoured.
    #[serde(default)]
    pub n_inner: Option<usize>,

    /// Rotary base. Absent from `phixtral-4x2_8`; upstream builds `nn.RoPE`
    /// without a base and takes MLX's `10000` default.
    #[serde(default = "default_rope_theta", alias = "rotary_base")]
    pub rope_theta: f32,

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

fn default_model_type() -> String {
    "phi-msft".to_string()
}
fn default_model_dim() -> usize {
    2560
}
fn default_num_heads() -> usize {
    32
}
fn default_num_layers() -> usize {
    32
}
fn default_num_vocab() -> usize {
    51200
}
fn default_rotary_dim() -> usize {
    32
}
fn default_num_local_experts() -> usize {
    4
}
fn default_num_experts_per_tok() -> usize {
    2
}
fn default_layer_norm_epsilon() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10_000.0
}

/// Upper bounds on the architecture scalars a phixtral `config.json` may
/// declare. Same rationale as the other ports: `config.json` is untrusted input
/// on the `mlxcel generate -m <org>/<repo>` path, and these keep a hostile value
/// from sizing an allocation, dividing, or truncating through an `as i32` cast.
/// Each sits orders of magnitude above `phixtral-4x2_8` (32 layers, 2560 wide,
/// 4 experts, 51200 vocab).
const MAX_NUM_LAYERS: usize = 1024;
/// See [`MAX_NUM_LAYERS`].
const MAX_MODEL_DIM: usize = 65_536;
/// See [`MAX_NUM_LAYERS`].
const MAX_NUM_HEADS: usize = 4096;
/// See [`MAX_NUM_LAYERS`].
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
/// See [`MAX_NUM_LAYERS`].
const MAX_VOCAB_SIZE: usize = 1 << 24;
/// See [`MAX_NUM_LAYERS`]. Also bounds the per-expert probe in
/// [`validate_experts`].
const MAX_NUM_EXPERTS: usize = 4096;

impl ModelArgs {
    /// Head width, `n_embd / n_head` (80 on `phixtral-4x2_8`).
    ///
    /// Only valid after [`ModelArgs::validate`], which rejects `n_head == 0`.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// FFN width per expert: `n_inner` when declared, `4 * n_embd` otherwise.
    pub fn intermediate_size(&self) -> usize {
        self.n_inner.unwrap_or(4 * self.n_embd)
    }

    /// Output width of the fused `Wqkv`: `3 * n_embd`.
    ///
    /// Phixtral is multi-head, so unlike the GQA families in this tree the split
    /// really is even and `mx.split(qkv, 3, axis=-1)` is faithful.
    pub fn qkv_out_features(&self) -> usize {
        3 * self.n_embd
    }

    /// Channels per head that RoPE rotates. Saturates rather than wrapping so an
    /// absurd `rotary_dim` cannot reach `fast_rope` as a negative `dims`.
    pub fn rope_dims(&self) -> i32 {
        i32::try_from(self.rotary_dim).unwrap_or(i32::MAX)
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

    /// Reject a `config.json` that cannot describe a real phixtral, before any
    /// of its fields sizes an allocation, divides, or reaches an MLX kernel.
    pub fn validate(&self) -> Result<(), String> {
        // The zero checks precede the divisibility check: `0.is_multiple_of(0)`
        // is true in Rust, so a config with `n_embd == n_head == 0` would pass
        // it and then divide by zero in `head_dim()`.
        if self.n_head == 0 || self.n_head > MAX_NUM_HEADS {
            return Err(format!(
                "Phixtral n_head ({}) must be between 1 and {MAX_NUM_HEADS}",
                self.n_head
            ));
        }
        if self.n_embd == 0 || self.n_embd > MAX_MODEL_DIM {
            return Err(format!(
                "Phixtral n_embd ({}) must be between 1 and {MAX_MODEL_DIM}",
                self.n_embd
            ));
        }
        if !self.n_embd.is_multiple_of(self.n_head) {
            return Err(format!(
                "Phixtral n_embd ({}) must be divisible by n_head ({}); the head width is \
                 n_embd // n_head and the fused Wqkv is reshaped with both",
                self.n_embd, self.n_head
            ));
        }
        if self.n_layer == 0 || self.n_layer > MAX_NUM_LAYERS {
            return Err(format!(
                "Phixtral n_layer ({}) must be between 1 and {MAX_NUM_LAYERS}",
                self.n_layer
            ));
        }
        if self.vocab_size == 0 || self.vocab_size > MAX_VOCAB_SIZE {
            return Err(format!(
                "Phixtral vocab_size ({}) must be between 1 and {MAX_VOCAB_SIZE}",
                self.vocab_size
            ));
        }
        let intermediate = self.intermediate_size();
        if intermediate == 0 || intermediate > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Phixtral expert FFN width ({intermediate}) must be between 1 and \
                 {MAX_INTERMEDIATE_SIZE}"
            ));
        }

        self.validate_routing()?;
        self.validate_rope()?;
        self.validate_norm_eps()?;
        self.validate_quantization()
    }

    /// Reject routing parameters that would index out of range inside MLX.
    ///
    /// Upstream calls `argpartition(-gates, kth=k - 1, axis=-1)`, which MLX
    /// refuses when `k - 1` is outside the expert row. MLX signals that by
    /// throwing, and an MLX C++ exception crossing the cxx bridge is an
    /// uncatchable `std::terminate` at the first forward pass rather than a load
    /// error.
    fn validate_routing(&self) -> Result<(), String> {
        if self.num_local_experts == 0 || self.num_local_experts > MAX_NUM_EXPERTS {
            return Err(format!(
                "Phixtral num_local_experts ({}) must be between 1 and {MAX_NUM_EXPERTS}",
                self.num_local_experts
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_local_experts {
            return Err(format!(
                "Phixtral num_experts_per_tok ({}) must be between 1 and num_local_experts ({}); \
                 the router selects that many indices out of a row of num_local_experts scores",
                self.num_experts_per_tok, self.num_local_experts
            ));
        }
        Ok(())
    }

    /// Reject a rotary width MLX would throw on.
    ///
    /// `mlx::core::fast::rope` requires `dims` positive, even, and no larger
    /// than the input's last axis. `fast_rope` crosses the cxx bridge as
    /// `UniquePtr<MlxArray>` rather than a `Result`, so a violation aborts the
    /// process at the first forward pass rather than failing the load.
    fn validate_rope(&self) -> Result<(), String> {
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!(
                "Phixtral rope_theta ({}) must be a finite positive number; RoPE exponentiates it \
                 per channel, so a zero, negative or non-finite base makes every rotated channel \
                 NaN and that NaN reaches the logits without anything throwing",
                self.rope_theta
            ));
        }
        let head_dim = self.head_dim();
        let dims = self.rotary_dim;
        if dims == 0 || dims > head_dim {
            return Err(format!(
                "Phixtral rotary_dim ({dims}) must be between 2 and the head width ({head_dim}). \
                 MLX throws on a rope `dims` outside that range, and an MLX C++ exception crossing \
                 the cxx bridge is an uncatchable abort at the first forward pass rather than a \
                 load error."
            ));
        }
        if !dims.is_multiple_of(2) {
            return Err(format!(
                "Phixtral rotary_dim ({dims}) must be even; RoPE rotates channel pairs and MLX \
                 throws on an odd `dims`."
            ));
        }
        Ok(())
    }

    fn validate_norm_eps(&self) -> Result<(), String> {
        if !self.layer_norm_epsilon.is_finite() || self.layer_norm_epsilon <= 0.0 {
            return Err(format!(
                "Phixtral layer_norm_epsilon ({}) must be a finite positive number; it is added to \
                 the variance under an rsqrt, so a non-finite, negative or zero value makes every \
                 normalized hidden state NaN and that NaN reaches the logits without anything \
                 throwing",
                self.layer_norm_epsilon
            ));
        }
        Ok(())
    }

    fn validate_quantization(&self) -> Result<(), String> {
        let Some(quantization) = self.quantization.as_ref() else {
            return Ok(());
        };
        mlxcel_core::layers::validate_quantization_params(
            quantization.group_size,
            quantization.bits,
        )
        .map_err(|e| format!("Phixtral config.json: {e}"))
    }
}

// Attention.

/// Phi-2 `RoPEAttention`: fused `Wqkv`, multi-head, partial RoPE, `out_proj`.
///
/// Every projection carries a bias, which the Phi family uses and the Llama
/// family does not.
pub struct Attention {
    pub wqkv: UnifiedLinear,
    /// Output projection. Phixtral names it `out_proj`; the dense Phi loader in
    /// this tree calls the same tensor `dense`.
    pub out_proj: UnifiedLinear,
    pub num_heads: i32,
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
        let width = self.num_heads * self.head_dim;

        // An even three-way split, unlike the GQA families in this tree: there
        // is no separate KV-head count, so all three blocks are `n_embd` wide.
        let qkv = self.wqkv.forward(x);
        let q = mlxcel_core::slice_last_dim(&qkv, 0, width);
        let k = mlxcel_core::slice_last_dim(&qkv, width, 2 * width);
        let v = mlxcel_core::slice_last_dim(&qkv, 2 * width, 3 * width);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;
        // Partial RoPE: only the first `rope_dims` channels of each head rotate.
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // **The attention runs in float32, and that is load-bearing.**
        //
        // Upstream writes `queries.astype(mx.float32)` immediately before the
        // SDPA call and `.astype(values.dtype)` immediately after, which
        // promotes the whole score computation to f32 because MLX promotes on
        // the wider operand. It reads like a stray cast and it is not: Phi-2
        // carries famously large outlier activations, and this checkpoint ships
        // float16, whose 65504 ceiling the `q @ k^T` products reach in the deep
        // layers. Running the scores in f16 produces NaN from layer 30 of 32 on
        // a four-token prompt (layers 0 through 29 track the reference to three
        // decimal places first, so the overflow is the only symptom and it
        // arrives late). The cache itself stays f16; only the arithmetic widens.
        let dtype = mlxcel_core::array_dtype(&cache_v);
        let f32_dtype = mlxcel_core::dtype::FLOAT32;
        let q = mlxcel_core::astype(&q, f32_dtype);
        let cache_k = mlxcel_core::astype(&cache_k, f32_dtype);
        let cache_v = mlxcel_core::astype(&cache_v, f32_dtype);

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

        let attn_out = mlxcel_core::astype(&attn_out, dtype);
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, width]);
        self.out_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let head_dim = args.head_dim() as i32;

        Ok(Self {
            wqkv: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.Wqkv"),
                group_size,
                bits,
            )?,
            out_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.out_proj"),
                group_size,
                bits,
            )?,
            num_heads: args.n_head as i32,
            head_dim,
            // Upstream writes `math.sqrt(1 / queries.shape[-1])`, which is the
            // same `head_dim ** -0.5` every other port in this tree uses.
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: args.rope_dims(),
            rope_base: args.rope_theta,
        })
    }
}

// Experts.

/// Upstream's `SwitchMLP`: `fc2(gelu(fc1(x)))` over the selected experts.
///
/// Two differences from [`crate::models::switch_layers::SwitchGLU`] make it a
/// separate type rather than a configuration of that one. It is a
/// two-projection MLP rather than a gated SwiGLU triple, and its projections
/// carry per-expert biases, which `SwitchLinear` does not implement.
pub struct PhixtralSwitchMlp {
    pub fc1: SwitchLinear,
    pub fc2: SwitchLinear,
    /// `[num_local_experts, intermediate_size]`.
    pub fc1_bias: UniquePtr<MlxArray>,
    /// `[num_local_experts, n_embd]`.
    pub fc2_bias: UniquePtr<MlxArray>,
}

/// Add the per-expert bias row selected by `indices`.
///
/// `y` is the `gather_mm` / `gather_qmm` output, whose trailing axes are
/// `[..., 1, out_features]`. `bias` is `[num_experts, out_features]`, so
/// gathering along axis 0 with `indices` yields `[..., out_features]` and one
/// `expand_dims` lines it up with the singleton axis. This is exactly upstream's
/// `x + mx.expand_dims(self["bias"][indices], -2)`.
fn add_expert_bias(y: &MlxArray, bias: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
    let rows = mlxcel_core::take(bias, indices, 0);
    mlxcel_core::add(y, &mlxcel_core::expand_dims(&rows, -2))
}

impl PhixtralSwitchMlp {
    /// `x` is `[n_tokens, n_embd]` and `indices` is `[n_tokens, top_k]`; the
    /// result is `[n_tokens, top_k, n_embd]`.
    ///
    /// The sort above 64 routed slots mirrors `SwitchGLU`: it makes each
    /// expert's rows contiguous for the gather, and is a pure permutation, so it
    /// cannot change the result.
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let do_sort = indices_shape[0] * indices_shape[1] >= 64;

        let x_exp = mlxcel_core::expand_dims(x, -2);
        let x_exp = mlxcel_core::expand_dims(&x_exp, -3);

        if do_sort {
            let (sorted_x, sorted_idx, inv_order) = gather_sort(&x_exp, indices);
            let h = self.fc1.forward(&sorted_x, &sorted_idx, true);
            let h = add_expert_bias(&h, &self.fc1_bias, &sorted_idx);
            let h = mlxcel_core::utils::gelu_approx(&h);
            let out = self.fc2.forward(&h, &sorted_idx, true);
            let out = add_expert_bias(&out, &self.fc2_bias, &sorted_idx);
            scatter_unsort(&out, &inv_order, &indices_shape)
        } else {
            let h = self.fc1.forward(&x_exp, indices, false);
            let h = add_expert_bias(&h, &self.fc1_bias, indices);
            let h = mlxcel_core::utils::gelu_approx(&h);
            let out = self.fc2.forward(&h, indices, false);
            let out = add_expert_bias(&out, &self.fc2_bias, indices);
            mlxcel_core::squeeze_axis(&out, -2)
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let fc1_bias_key = format!("{prefix}.fc1.bias");
        let fc2_bias_key = format!("{prefix}.fc2.bias");
        Ok(Self {
            fc1: SwitchLinear::from_weights(weights, &format!("{prefix}.fc1"), group_size, bits)?,
            fc2: SwitchLinear::from_weights(weights, &format!("{prefix}.fc2"), group_size, bits)?,
            fc1_bias: weights
                .get(&fc1_bias_key)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Weight not found: {fc1_bias_key}"))?,
            fc2_bias: weights
                .get(&fc2_bias_key)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Weight not found: {fc2_bias_key}"))?,
        })
    }
}

/// Upstream's `MOE`: an `nn.Linear` router over a [`PhixtralSwitchMlp`].
pub struct PhixtralMoe {
    pub gate: UnifiedLinear,
    pub switch_mlp: PhixtralSwitchMlp,
    pub top_k: i32,
}

impl PhixtralMoe {
    /// **The softmax runs over the k gathered logits, not the full expert row.**
    ///
    /// Upstream selects on the raw gate logits, gathers those logits at the
    /// selected indices, and softmaxes the `k` gathered values. Softmaxing the
    /// whole row first and gathering afterwards normalizes by the sum over all
    /// `num_local_experts` rather than over the selected `k`, which changes
    /// every routed weight while leaving the output finite and the text fluent.
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = *orig_shape.last().unwrap_or(&0);
        let x_flat = if orig_shape.len() > 2 {
            let tokens: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[tokens, hidden])
        } else {
            mlxcel_core::copy(x)
        };

        let gates = self.gate.forward(&x_flat);

        // Top-k over the raw logits, as `argpartition(-gates, kth = k - 1)`
        // followed by the FIRST k, which is upstream's orientation rather than
        // the equivalent-looking `argpartition(gates, n - k)` plus the last k.
        //
        // **On this family the orientation is not cosmetic.** Every gate row of
        // `mlabonne/phixtral-4x2_8` is identical across all four experts, in
        // every one of its 32 layers (verified by dequantizing the router), so
        // each token's four logits are exactly tied: the routing weights are
        // always a uniform `1/k`, and *which* experts run is decided entirely by
        // how `argpartition` breaks the tie. The two orientations break it
        // differently (this checkpoint selects experts 0 and 1 upstream, and
        // would select 2 and 3 under the mirrored form), and since the experts
        // are distinct that is a different output, not a reordering of the same
        // one. Mirroring upstream is what makes a checkpoint decoded here match
        // the same checkpoint decoded under the reference.
        //
        // The expert count comes from the real score row rather than from the
        // config, so the partition pivot is in range even if the two disagree.
        let gates_shape = mlxcel_core::array_shape(&gates);
        let n_experts = *gates_shape.last().unwrap_or(&0);
        let kth = (self.top_k - 1).clamp(0, (n_experts - 1).max(0));
        let order = mlxcel_core::argpartition(&mlxcel_core::negative(&gates), kth, -1);
        let indices = slice_axis(&order, -1, 0, self.top_k);

        let selected = mlxcel_core::take_along_axis(&gates, &indices, -1);
        // `precise=True` upstream: the softmax runs in float32 and is cast back
        // by `moe_weighted_sum` when the routed outputs are combined.
        let scores = mlxcel_core::softmax(
            &mlxcel_core::astype(&selected, mlxcel_core::dtype::FLOAT32),
            -1,
        );

        let expert_out = self.switch_mlp.forward(&x_flat, &indices);
        let routed = moe_weighted_sum(&expert_out, &scores, mlxcel_core::array_dtype(&x_flat));

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&routed, &orig_shape)
        } else {
            routed
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            gate: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate"),
                args.group_size(),
                args.bits(),
            )?,
            switch_mlp: PhixtralSwitchMlp::from_weights(
                weights,
                args,
                &format!("{prefix}.switch_mlp"),
            )?,
            top_k: args.num_experts_per_tok as i32,
        })
    }
}

// Decoder block and model.

/// Upstream's `ParallelBlock`.
///
/// One LayerNorm feeds both branches and their outputs are summed into the
/// residual. There is no post-attention norm and the MoE never sees the
/// attention output.
pub struct ParallelBlock {
    pub ln: LayerNorm,
    pub mixer: Attention,
    pub moe: PhixtralMoe,
}

impl ParallelBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = self.ln.forward(x);
        let attn_out = self.mixer.forward(&h, cache, mask);
        let moe_out = self.moe.forward(&h);
        mlxcel_core::add(&mlxcel_core::add(&attn_out, &moe_out), x)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("transformer.h.{layer_idx}");
        Ok(Self {
            ln: layer_norm_from_weights(weights, &format!("{prefix}.ln"), args.layer_norm_epsilon)?,
            mixer: Attention::from_weights(weights, args, &format!("{prefix}.mixer"))?,
            moe: PhixtralMoe::from_weights(weights, args, &format!("{prefix}.moe"))?,
        })
    }
}

/// Load a LayerNorm whose bias is required, as every Phi-family norm has one.
fn layer_norm_from_weights(
    weights: &WeightMap,
    prefix: &str,
    eps: f32,
) -> Result<LayerNorm, String> {
    let weight_key = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {weight_key}"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Phixtral: a Mixtral-style sparse MoE on the Phi-2 backbone.
pub struct PhixtralModel {
    /// Token table. Phixtral names it `transformer.embd.wte`.
    pub wte: UnifiedEmbedding,
    pub layers: Vec<ParallelBlock>,
    /// Upstream's `OutputHead` is a LayerNorm followed by a Linear, so the final
    /// norm lives under `lm_head.ln` rather than at the top of the transformer.
    pub lm_head_ln: LayerNorm,
    pub lm_head: Option<UnifiedLinear>,
    eos_token_ids: Vec<i32>,
}

impl PhixtralModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.wte.forward(input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }
        let h = self.lm_head_ln.forward(&h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.wte.as_linear(&h),
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        // Reject an impossible config before reading the weights, not after.
        args.validate()?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        // `config.json` is untrusted, and `from_weights` is also reachable
        // directly from the registration table, so revalidate here rather than
        // trusting that `load` ran first.
        args.validate()?;
        validate_weights(weights, args)?;

        let group_size = args.group_size();
        let bits = args.bits();

        let wte =
            UnifiedEmbedding::from_weights(weights, "transformer.embd.wte", group_size, bits)?;
        // Token ids are bounded by `vocab_size`, a config field, and an
        // embedding gather wraps a negative index but does not range-check a
        // positive one, so a config that overstates the table turns an ordinary
        // prompt into an out-of-bounds read whose result reaches the logits.
        validate_embedding_table(
            &wte,
            "transformer.embd.wte",
            args.vocab_size,
            "vocab_size",
            args.n_embd,
            "n_embd",
        )?;

        let mut layers = Vec::with_capacity(args.n_layer);
        for i in 0..args.n_layer {
            layers.push(ParallelBlock::from_weights(weights, args, i)?);
        }

        let lm_head_ln = layer_norm_from_weights(weights, "lm_head.ln", args.layer_norm_epsilon)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights,
                "lm_head.linear",
                group_size,
                bits,
            )?)
        };

        Ok(Self {
            wte,
            layers,
            lm_head_ln,
            lm_head,
            // `phixtral-4x2_8` inherits the GPT-2 BPE vocabulary, whose
            // `<|endoftext|>` is 50256 and is what its `generation_config`
            // stops on. Upstream's `Model` declares no eos at all and leaves it
            // to the tokenizer.
            eos_token_ids: vec![50256],
        })
    }
}

// Weight-shape validation.

fn quant_mode(weights: &WeightMap, prefix: &str, group_size: i32, bits: i32) -> &'static str {
    let has_biases = weights.contains_key(&format!("{prefix}.biases"));
    mlxcel_core::layers::infer_quantization_mode(has_biases, group_size, bits)
}

fn validate_quantized_packing(
    weights: &WeightMap,
    prefix: &str,
    in_features: usize,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    let Some(scales) = weights.get(&format!("{prefix}.scales")) else {
        return Ok(());
    };
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight (but {prefix}.scales exists)"))?;
    let w_shape = mlxcel_core::array_shape(weight);
    let s_shape = mlxcel_core::array_shape(scales);
    let bias_shape = weights
        .get(&format!("{prefix}.biases"))
        .map(|biases| mlxcel_core::array_shape(biases));

    mlxcel_core::layers::validate_quantized_packing(
        prefix,
        &mlxcel_core::layers::QuantizedTensorShapes {
            weight: &w_shape,
            scales: &s_shape,
            biases: bias_shape.as_deref(),
        },
        in_features,
        group_size,
        bits,
        quant_mode(weights, prefix, group_size, bits),
    )
}

/// Check one `[out_features, in_features]` projection against the config.
fn validate_projection(
    weights: &WeightMap,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    let weight_name = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let shape = mlxcel_core::array_shape(weight);
    if shape.len() != 2 {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected a 2-D [out, in] projection"
        ));
    }
    let quantized = weights.contains_key(&format!("{prefix}.scales"));
    if !dim_eq(shape[0], out_features) {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected {out_features} rows"
        ));
    }
    if !quantized && !dim_eq(shape[1], in_features) {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected [{out_features}, {in_features}]"
        ));
    }
    validate_quantized_packing(weights, prefix, in_features, group_size, bits)?;

    if let Some(bias) = weights.get(&format!("{prefix}.bias")) {
        let bias_shape = mlxcel_core::array_shape(bias);
        if bias_shape.len() != 1 || !dim_eq(bias_shape[0], out_features) {
            return Err(format!(
                "unexpected {prefix}.bias shape {bias_shape:?}: expected [{out_features}]"
            ));
        }
    }
    Ok(())
}

/// Check a LayerNorm's weight and its bias.
fn validate_layer_norm(weights: &WeightMap, prefix: &str, dim: usize) -> Result<(), String> {
    for leaf in ["weight", "bias"] {
        let key = format!("{prefix}.{leaf}");
        let Some(tensor) = weights.get(&key) else {
            if leaf == "bias" {
                continue;
            }
            return Err(format!("Weight not found: {key}"));
        };
        let shape = mlxcel_core::array_shape(tensor);
        if shape.len() != 1 || !dim_eq(shape[0], dim) {
            return Err(format!(
                "unexpected {key} shape {shape:?}: expected [{dim}]"
            ));
        }
    }
    Ok(())
}

/// Check one stacked `[num_experts, out_features, in_features]` expert tensor
/// and its `[num_experts, out_features]` bias plane.
///
/// The leading axis is the gather axis of `gather_mm` / `gather_qmm`, and the
/// router can emit any index below `num_local_experts`. MLX's gather adds the
/// axis size to a negative index but performs no range check on a positive one,
/// so a stacked tensor with fewer planes than the config claims turns an
/// ordinary token into an out-of-bounds read whose result reaches the logits.
/// More planes than claimed is accepted: the router can never reach them.
fn validate_expert_projection(
    weights: &WeightMap,
    prefix: &str,
    num_experts: usize,
    out_features: usize,
    in_features: usize,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    let weight_name = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let shape = mlxcel_core::array_shape(weight);
    if shape.len() != 3 {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected a 3-D [num_experts, out, in] \
             stacked expert tensor"
        ));
    }
    let planes = usize::try_from(shape[0]).unwrap_or(0);
    if planes < num_experts {
        return Err(format!(
            "config num_local_experts ({num_experts}) exceeds the {planes} expert planes present \
             in {weight_name}. The router selects indices below num_local_experts and the gather \
             behind gather_mm / gather_qmm does not range-check a positive index, so the missing \
             planes would be read out of bounds and the result would reach the logits."
        ));
    }
    if !dim_eq(shape[1], out_features) {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected {out_features} rows per expert"
        ));
    }
    let quantized = weights.contains_key(&format!("{prefix}.scales"));
    if !quantized && !dim_eq(shape[2], in_features) {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected [{planes}, {out_features}, \
             {in_features}]"
        ));
    }
    validate_quantized_packing(weights, prefix, in_features, group_size, bits)?;

    // The bias plane is not optional here: `SwitchMLP` is built with
    // `bias=True`, so a checkpoint without it is not a phixtral this loader can
    // reproduce, and silently dropping it would shift every expert's output.
    let bias_name = format!("{prefix}.bias");
    let bias = weights
        .get(&bias_name)
        .ok_or_else(|| format!("Weight not found: {bias_name} (phixtral experts carry biases)"))?;
    let bias_shape = mlxcel_core::array_shape(bias);
    if bias_shape.len() != 2
        || usize::try_from(bias_shape[0]).unwrap_or(0) < num_experts
        || !dim_eq(bias_shape[1], out_features)
    {
        return Err(format!(
            "unexpected {bias_name} shape {bias_shape:?}: expected [{num_experts}, {out_features}]"
        ));
    }
    Ok(())
}

fn validate_experts(weights: &WeightMap, args: &ModelArgs, moe_prefix: &str) -> Result<(), String> {
    let hidden = args.n_embd;
    let intermediate = args.intermediate_size();
    let experts = args.num_local_experts;
    let group_size = args.group_size();
    let bits = args.bits();

    validate_expert_projection(
        weights,
        &format!("{moe_prefix}.switch_mlp.fc1"),
        experts,
        intermediate,
        hidden,
        group_size,
        bits,
    )?;
    validate_expert_projection(
        weights,
        &format!("{moe_prefix}.switch_mlp.fc2"),
        experts,
        hidden,
        intermediate,
        group_size,
        bits,
    )
}

/// Reject a checkpoint whose real tensor shapes disagree with `config.json`,
/// before any of them reaches MLX.
///
/// This has to run before the model is built, not after: the fused `Wqkv`
/// output is split at config-derived offsets and reshaped with config-derived
/// head counts, and MLX's `slice` clamps an out-of-range stop rather than
/// throwing, so a too-narrow projection silently yields a short V block and the
/// reshape aborts the process instead of returning an error.
pub fn validate_weights(weights: &WeightMap, args: &ModelArgs) -> Result<(), String> {
    let hidden = args.n_embd;
    let group_size = args.group_size();
    let bits = args.bits();

    validate_quantized_packing(weights, "transformer.embd.wte", hidden, group_size, bits)?;
    validate_layer_norm(weights, "lm_head.ln", hidden)?;
    if !args.tie_word_embeddings {
        validate_projection(
            weights,
            "lm_head.linear",
            args.vocab_size,
            hidden,
            group_size,
            bits,
        )?;
    }

    for layer in 0..args.n_layer {
        let prefix = format!("transformer.h.{layer}");
        validate_layer_norm(weights, &format!("{prefix}.ln"), hidden)?;
        validate_projection(
            weights,
            &format!("{prefix}.mixer.Wqkv"),
            args.qkv_out_features(),
            hidden,
            group_size,
            bits,
        )?;
        validate_projection(
            weights,
            &format!("{prefix}.mixer.out_proj"),
            hidden,
            hidden,
            group_size,
            bits,
        )?;
        let moe = format!("{prefix}.moe");
        validate_projection(
            weights,
            &format!("{moe}.gate"),
            args.num_local_experts,
            hidden,
            group_size,
            bits,
        )?;
        validate_experts(weights, args, &moe)?;
    }
    Ok(())
}

// LanguageModel trait implementation.

impl LanguageModel for PhixtralModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        PhixtralModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        PhixtralModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "phixtral_tests.rs"]
mod tests;
