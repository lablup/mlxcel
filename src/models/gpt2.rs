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

//! GPT-2 (`gpt2`) text model implementation using mlxcel-core.
//!
//! Reference: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gpt2.py>
//!
//! Deltas versus the Llama-style dense decoders in this tree:
//!
//! - Learned absolute position embeddings (`wpe`) added to the token
//!   embeddings (`wte`) at the input boundary. There is no RoPE anywhere, so
//!   the KV-cache offset has to be threaded into the position lookup instead
//!   of into a rotation.
//! - `LayerNorm` with bias (`layer_norm_epsilon`), not RMSNorm.
//! - One fused `c_attn` projection that is split three ways on the last axis
//!   into Q, K and V. Multi-head attention: `n_kv_heads == n_head`.
//! - Simple MLP `c_proj(gelu_approx(c_fc(x)))` with an intermediate width of
//!   `4 * n_embd`. `mlxcel_core::utils::gelu_approx` is the erf-based exact
//!   GELU in this tree despite its name (see its own doc comment), not the
//!   tanh approximation that upstream's `nn.gelu_approx` evaluates for the
//!   checkpoint's `gelu_new` activation; the erf/tanh difference is under
//!   1e-3 and produces identical generated tokens.
//! - Tied output head: logits come from `wte.as_linear`, there is no separate
//!   `lm_head` tensor in the checkpoint.
//!
//! Two checkpoint-layout traps are handled at load time, see [`Gpt2Layout`]
//! and [`strip_causal_mask_buffers`].

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, Linear, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.

/// GPT-2 `config.json`.
///
/// GPT-2 predates the `hidden_size` / `num_attention_heads` naming that the
/// rest of the tree uses, so the field names are the original OpenAI ones.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_n_embd")]
    pub n_embd: usize,

    #[serde(default = "default_n_head")]
    pub n_head: usize,

    #[serde(default = "default_n_layer")]
    pub n_layer: usize,

    /// Number of rows in the learned position-embedding table `wpe`.
    #[serde(default = "default_n_positions")]
    pub n_positions: usize,

    #[serde(default = "default_n_ctx")]
    pub n_ctx: usize,

    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

/// `eos_token_id` may be a single int or a list of ints (same shape as
/// `src/models/cohere2.rs`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(i32),
    Multiple(Vec<i32>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "gpt2".to_string()
}
fn default_n_embd() -> usize {
    768
}
fn default_n_head() -> usize {
    12
}
fn default_n_layer() -> usize {
    12
}
fn default_n_positions() -> usize {
    1024
}
fn default_n_ctx() -> usize {
    1024
}
fn default_layer_norm_epsilon() -> f32 {
    1e-5
}
fn default_vocab_size() -> usize {
    50257
}

/// GPT-2's own EOS/BOS token (`<|endoftext|>`), used when the config omits it.
pub const GPT2_EOS_TOKEN_ID: i32 = 50256;

/// Upper bounds on the architecture scalars a GPT-2 `config.json` may declare.
///
/// `config.json` is untrusted input. The common `mlxcel generate -m <org>/<repo>`
/// flow downloads a third-party HuggingFace repo and loads it in the same
/// command; the download layer validates repo ids, filenames and transport but
/// never parses the file, so these fields arrive exactly as the checkpoint
/// author wrote them.
///
/// Each ceiling sits orders of magnitude above the largest real GPT-2 (XL: 48
/// layers, `n_embd` 1600, 1024 positions) and above every dense decoder in this
/// tree, so no genuine checkpoint is affected. They exist so `n_layer` cannot
/// size the `Vec::with_capacity` in [`Gpt2Model::from_weights`], and so the
/// `as i32` casts these values feed (`n_embd`, `3 * n_embd`, `n_positions - 1`)
/// stay inside `i32` instead of truncating to a negative number.
const MAX_N_LAYER: usize = 1024;
/// See [`MAX_N_LAYER`].
const MAX_N_EMBD: usize = 65_536;
/// See [`MAX_N_LAYER`].
const MAX_N_POSITIONS: usize = 1 << 22;

impl ModelArgs {
    /// Reject a `config.json` that cannot describe a real GPT-2, before any of
    /// its fields sizes an allocation, indexes a table, or divides.
    ///
    /// Rejecting once at load rather than guarding ad hoc downstream follows the
    /// `SparseAttentionConfig::validate` precedent in
    /// `src/models/minimax_m3_config.rs`. See [`MAX_N_LAYER`] for why the
    /// magnitude ceilings exist at all.
    pub fn validate(&self) -> Result<(), String> {
        // The zero check has to come first: `0.is_multiple_of(0)` is true, so a
        // `n_embd == n_head == 0` config would otherwise pass the divisibility
        // check below and reach `head_dim()`, which divides by `n_head`.
        if self.n_head == 0 {
            return Err(
                "GPT-2 n_head must be non-zero (zero divides by zero in head_dim)".to_string(),
            );
        }
        if self.n_embd == 0 || self.n_embd > MAX_N_EMBD {
            return Err(format!(
                "GPT-2 n_embd ({}) must be between 1 and {MAX_N_EMBD}",
                self.n_embd
            ));
        }
        if !self.n_embd.is_multiple_of(self.n_head) {
            return Err(format!(
                "GPT-2 n_embd ({}) must be divisible by n_head ({})",
                self.n_embd, self.n_head
            ));
        }
        if self.n_layer == 0 || self.n_layer > MAX_N_LAYER {
            return Err(format!(
                "GPT-2 n_layer ({}) must be between 1 and {MAX_N_LAYER}",
                self.n_layer
            ));
        }
        if self.n_positions == 0 || self.n_positions > MAX_N_POSITIONS {
            return Err(format!(
                "GPT-2 n_positions ({}) must be between 1 and {MAX_N_POSITIONS}",
                self.n_positions
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// GPT-2's MLP is always four times the model width.
    pub fn intermediate_size(&self) -> usize {
        4 * self.n_embd
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
        match &self.eos_token_id {
            Some(EosTokenId::Single(id)) => vec![*id],
            Some(EosTokenId::Multiple(ids)) if !ids.is_empty() => ids.clone(),
            _ => vec![GPT2_EOS_TOKEN_ID],
        }
    }
}

// Checkpoint layout.

/// Key layout of the GPT-2 checkpoint being loaded.
///
/// Two shapes reach this loader:
///
/// - A raw HuggingFace export (`mlx-community` does not have to be involved):
///   top-level `wte.weight` / `h.0.attn.c_attn.weight` / `ln_f.weight` keys,
///   with the attention and MLP projections still stored in the HuggingFace
///   `Conv1D` layout `[in, out]`. `mlxcel_core`'s `Linear` wants `[out, in]`,
///   so those weights need a transpose. The bias vectors are 1-D and must
///   never be transposed.
/// - An MLX conversion, which is produced from an already-sanitized module
///   tree and therefore carries a `model.` prefix and `[out, in]` weights.
///
/// The transpose decision is taken from the shape of `h.0.attn.c_attn.weight`
/// rather than from the key prefix: that tensor is `[n_embd, 3 * n_embd]` in
/// Conv1D layout and `[3 * n_embd, n_embd]` once transposed, which is
/// unambiguous for any real `n_embd`. Deciding by shape also keeps
/// `h.N.attn.c_proj.weight` correct, which is square (`[n_embd, n_embd]`) and
/// therefore carries no layout signal of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gpt2Layout {
    /// Prefix in front of `wte` / `wpe` / `h.N` / `ln_f`.
    pub prefix: String,
    /// Whether the `c_attn` / `c_proj` / `c_fc` weights still need the
    /// Conv1D `[in, out]` -> `[out, in]` transpose.
    pub conv1d: bool,
}

/// Key prefixes a GPT-2 checkpoint can use, most specific last so that the
/// bare (raw HuggingFace) form is probed first.
const GPT2_PREFIXES: [&str; 4] = ["", "transformer.", "model.", "model.transformer."];

impl Gpt2Layout {
    /// Detect the prefix and the Conv1D layout of `weights`.
    pub fn detect(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let prefix = GPT2_PREFIXES
            .iter()
            .find(|p| weights.contains_key(&format!("{p}wte.weight")))
            .ok_or_else(|| {
                "GPT-2 token embedding not found: expected one of \
                 wte.weight / transformer.wte.weight / model.wte.weight / \
                 model.transformer.wte.weight"
                    .to_string()
            })?;

        let c_attn = format!("{prefix}h.0.attn.c_attn");
        // A quantized checkpoint can only come from an MLX conversion, which
        // has already been sanitized; its packed weight shape carries no
        // layout signal, so short-circuit before the shape probe.
        let conv1d = if weights.contains_key(&format!("{c_attn}.scales")) {
            false
        } else {
            let weight_name = format!("{c_attn}.weight");
            let weight = weights
                .get(&weight_name)
                .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
            let shape = mlxcel_core::array_shape(weight);
            let n_embd = args.n_embd as i32;
            match shape.as_slice() {
                [rows, cols] if *rows == n_embd && *cols == 3 * n_embd => true,
                [rows, cols] if *rows == 3 * n_embd && *cols == n_embd => false,
                other => {
                    return Err(format!(
                        "unexpected {weight_name} shape {other:?}: expected \
                         [{n_embd}, {}] (HuggingFace Conv1D) or [{}, {n_embd}] \
                         (already transposed) for n_embd={n_embd}",
                        3 * n_embd,
                        3 * n_embd
                    ));
                }
            }
        };

        Ok(Self {
            prefix: (*prefix).to_string(),
            conv1d,
        })
    }

    fn key(&self, suffix: &str) -> String {
        format!("{}{}", self.prefix, suffix)
    }

    /// Load a projection that HuggingFace stores as a `Conv1D`.
    ///
    /// The transpose applies to the weight only. `c_attn.bias`
    /// (`[3 * n_embd]`), `c_proj.bias` and `c_fc.bias` are 1-D and are copied
    /// through untouched.
    fn conv1d_linear(
        &self,
        weights: &WeightMap,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<UnifiedLinear, String> {
        // A quantized projection is packed, so its stored shape matches neither
        // float layout, and the MLX conversion it can only have come from is
        // already sanitized. `UnifiedLinear` reconciles the quantization layout
        // itself, so hand it over unchecked.
        if weights.contains_key(&format!("{prefix}.scales")) {
            return UnifiedLinear::from_weights(weights, prefix, group_size, bits);
        }

        let weight_name = format!("{prefix}.weight");
        let weight = weights
            .get(&weight_name)
            .ok_or_else(|| format!("Weight not found: {weight_name}"))?;

        // The layout was decided once, from `h.0.attn.c_attn.weight`. Check
        // every projection against the shape that decision implies instead of
        // trusting the rest of the checkpoint to agree with layer 0. A
        // checkpoint whose layers disagree, or one with a degenerate rank, would
        // otherwise be transposed (or not) into a graph that only fails at
        // `matmul`, and an MLX C++ exception crossing the cxx bridge is
        // `std::terminate`: the process dies with no diagnostic instead of
        // returning a load error naming the tensor.
        let (expected, layout_note) = if self.conv1d {
            ([in_dim, out_dim], "HuggingFace Conv1D [in, out]")
        } else {
            ([out_dim, in_dim], "already-transposed [out, in]")
        };
        let shape = mlxcel_core::array_shape(weight);
        let shape_ok =
            shape.len() == 2 && dim_eq(shape[0], expected[0]) && dim_eq(shape[1], expected[1]);
        if !shape_ok {
            return Err(format!(
                "unexpected {weight_name} shape {shape:?}: expected {expected:?} \
                 ({layout_note}), the layout detected from h.0.attn.c_attn.weight"
            ));
        }

        if let Some(bias) = weights.get(&format!("{prefix}.bias")) {
            let bias_shape = mlxcel_core::array_shape(bias);
            if bias_shape.len() != 1 || !dim_eq(bias_shape[0], out_dim) {
                return Err(format!(
                    "unexpected {prefix}.bias shape {bias_shape:?}: expected [{out_dim}]"
                ));
            }
        }

        if !self.conv1d {
            return UnifiedLinear::from_weights(weights, prefix, group_size, bits);
        }

        let transposed = mlxcel_core::transpose(weight);
        let bias = weights
            .get(&format!("{prefix}.bias"))
            .map(|b| mlxcel_core::copy(b));

        Ok(UnifiedLinear::Regular(Linear::new(transposed, bias)))
    }
}

/// Remove the per-layer `h.N.attn.bias` causal-mask buffers.
///
/// HuggingFace GPT-2 registers a `[1, 1, n_ctx, n_ctx]` lower-triangular mask
/// buffer under `h.N.attn.bias`. The name collides with the attention
/// projection bias namespace, and the tensor is neither a bias nor anything
/// this runtime needs: `mlxcel` builds its causal mask from the cache offset.
/// The model graph never reads the key (it only ever asks for
/// `h.N.attn.c_attn` and `h.N.attn.c_proj`), but dropping it before
/// construction releases ~4 MB per layer for a 1024-token context instead of
/// holding it for the lifetime of the weight map.
///
/// Returns the number of buffers removed.
pub fn strip_causal_mask_buffers(weights: &mut WeightMap) -> usize {
    let before = weights.len();
    weights.retain(|key, _| !is_causal_mask_buffer_key(key));
    before - weights.len()
}

/// Whether `key` names a `<prefix>h.<N>.attn.bias` causal-mask buffer.
///
/// Matching the key shape in one pass over the weight map, rather than probing
/// `<prefix>h.<layer>.attn.bias` for every prefix and every `layer` in
/// `0..n_layer`, keeps the strip proportional to the checkpoint instead of to a
/// number read out of `config.json`. `n_layer` is attacker-controlled and is not
/// checked against anything before this runs on the `load()` path, and the
/// probing form has no early exit: a config declaring `n_layer` in the billions
/// spins here for hours before the first weight lookup could reject it.
///
/// The real projection biases are `.c_attn.bias`, `.c_proj.bias` and
/// `.c_fc.bias`; none of them ends in `.attn.bias`, so none is ever matched.
fn is_causal_mask_buffer_key(key: &str) -> bool {
    let Some(rest) = key.strip_suffix(".attn.bias") else {
        return false;
    };
    let Some((head, index)) = rest.rsplit_once("h.") else {
        return false;
    };
    // `h.` has to start a path segment, so that a key such as `blah.0.attn.bias`
    // (which also contains the substring `h.`) is not read as layer 0.
    if !(head.is_empty() || head.ends_with('.')) {
        return false;
    }
    !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit())
}

/// Position ids for a forward pass that starts at KV-cache offset `offset`.
///
/// GPT-2 has no RoPE, so this is the only place the cache offset enters the
/// graph: prefill runs at `offset = 0` and every decode step after it must
/// continue from the number of tokens already cached, otherwise every
/// generated token is embedded as if it were at position 0.
///
/// The learned table has exactly `n_positions` rows, so positions past the end
/// are clamped to the last row rather than indexing out of bounds. GPT-2's
/// hard 1024-token context makes that reachable from the CLI. Clamping keeps an
/// in-flight request alive, but the tokens it clamps are all embedded at the
/// same position, so [`exceeds_position_table`] lets the caller say so instead
/// of degrading silently.
pub fn position_ids(offset: i32, seq_len: i32, n_positions: usize) -> Vec<i32> {
    let last = last_position_row(n_positions);
    (0..seq_len.max(0))
        .map(|i| offset.saturating_add(i).clamp(0, last))
        .collect()
}

/// Index of the last row of an `n_positions`-row table, saturated into `i32`.
///
/// A plain `as i32` truncates: `1usize << 32` becomes `0` and `(1usize << 40) - 1`
/// becomes `-1`. A negative upper bound makes the `clamp` in [`position_ids`]
/// panic (`clamp` panics when `min > max`), and a panic on a server worker
/// thread is a process abort, so the conversion saturates instead.
/// [`ModelArgs::validate`] already holds `n_positions` far below `i32::MAX`, but
/// both helpers are public and must not depend on their caller for that.
fn last_position_row(n_positions: usize) -> i32 {
    n_positions.saturating_sub(1).min(i32::MAX as usize) as i32
}

/// Whether a forward pass at KV-cache offset `offset` covering `seq_len` tokens
/// reaches past the last row of the learned `wpe` table.
///
/// The default CLI and server budget already keeps generation inside the
/// window: `context_window_from_config` reads GPT-2's `n_positions`, so the
/// `-n -1` sentinel resolves to `n_positions - prompt_len`. This predicate
/// covers the paths that bypass that budget, chiefly an explicit `-n N` and a
/// prompt that is itself longer than the context.
pub fn exceeds_position_table(offset: i32, seq_len: i32, n_positions: usize) -> bool {
    if seq_len <= 0 {
        return false;
    }
    let last = last_position_row(n_positions);
    offset.saturating_add(seq_len - 1) > last
}

// Attention (fused c_attn QKV, no RoPE).

pub struct Attention {
    pub c_attn: UnifiedLinear,
    pub c_proj: UnifiedLinear,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
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
        let d = self.num_heads * self.head_dim;

        // Fused QKV projection, split three ways on the last axis.
        let qkv = self.c_attn.forward(x);
        let q = mlxcel_core::slice_last_dim(&qkv, 0, d);
        let k = mlxcel_core::slice_last_dim(&qkv, d, 2 * d);
        let v = mlxcel_core::slice_last_dim(&qkv, 2 * d, 3 * d);

        // [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

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
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, d]);

        self.c_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layout: &Gpt2Layout,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // `{prefix}.bias` is the registered causal-mask buffer, never a linear
        // bias: only the `.c_attn` / `.c_proj` sub-prefixes are loaded here.
        let n_embd = args.n_embd;
        let c_attn = layout.conv1d_linear(
            weights,
            &format!("{prefix}.c_attn"),
            n_embd,
            3 * n_embd,
            group_size,
            bits,
        )?;
        let c_proj = layout.conv1d_linear(
            weights,
            &format!("{prefix}.c_proj"),
            n_embd,
            n_embd,
            group_size,
            bits,
        )?;

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            c_attn,
            c_proj,
            num_heads: args.n_head as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }
}

// MLP (erf-based exact GELU via `gelu_approx`, no gate/up pattern).

pub struct MLP {
    pub c_fc: UnifiedLinear,
    pub c_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.c_fc.forward(x);
        let h = mlxcel_core::utils::gelu_approx(&h);
        self.c_proj.forward(&h)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layout: &Gpt2Layout,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let n_embd = args.n_embd;
        let intermediate = args.intermediate_size();
        let c_fc = layout.conv1d_linear(
            weights,
            &format!("{prefix}.c_fc"),
            n_embd,
            intermediate,
            group_size,
            bits,
        )?;
        let c_proj = layout.conv1d_linear(
            weights,
            &format!("{prefix}.c_proj"),
            intermediate,
            n_embd,
            group_size,
            bits,
        )?;

        Ok(Self { c_fc, c_proj })
    }
}

// Transformer block (pre-norm, sequential attention then MLP).

pub struct TransformerBlock {
    pub attn: Attention,
    pub mlp: MLP,
    pub ln_1: LayerNorm,
    pub ln_2: LayerNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.ln_1.forward(x);
        let attn_out = self.attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.ln_2.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layout: &Gpt2Layout,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = layout.key(&format!("h.{layer_idx}"));

        let attn = Attention::from_weights(weights, args, layout, &format!("{prefix}.attn"))?;
        let mlp = MLP::from_weights(weights, args, layout, &format!("{prefix}.mlp"))?;
        let ln_1 = layer_norm_from_weights(
            weights,
            &format!("{prefix}.ln_1"),
            args.n_embd,
            args.layer_norm_epsilon,
        )?;
        let ln_2 = layer_norm_from_weights(
            weights,
            &format!("{prefix}.ln_2"),
            args.n_embd,
            args.layer_norm_epsilon,
        )?;

        Ok(Self {
            attn,
            mlp,
            ln_1,
            ln_2,
        })
    }
}

// GPT-2 model.

pub struct Gpt2Model {
    /// Token embedding, also the tied output head.
    pub wte: UnifiedEmbedding,
    /// Learned absolute position embedding.
    pub wpe: UnifiedEmbedding,
    pub h: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
    /// Clamp bound for the learned position lookup.
    ///
    /// Private, and never larger than the number of rows actually present in
    /// `wpe.weight` (enforced by [`validate_position_table`] in
    /// [`Gpt2Model::from_weights`], the only constructor). That invariant is the
    /// whole bounds check on the lookup: MLX's gather wraps a negative index but
    /// does not range-check a positive one, so a value larger than the table
    /// reads past the end of the buffer instead of faulting.
    n_positions: usize,
    eos_token_ids: Vec<i32>,
}

impl Gpt2Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let input_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = *input_shape.last().unwrap_or(&0);

        let mut h = self.wte.forward(input_ids);

        // Learned absolute positions. Read the offset before any layer runs:
        // `update_and_fetch` advances `caches[0].offset` during the loop below.
        let offset = caches.first().map(|c| c.offset).unwrap_or(0);
        let positions = position_ids(offset, seq_len, self.n_positions);
        if exceeds_position_table(offset, seq_len, self.n_positions) {
            // Once per process: the condition holds for every remaining step of
            // an over-long generation, and it is a prompt/budget problem, not a
            // per-step event worth repeating.
            static CLAMP_WARNED: std::sync::Once = std::sync::Once::new();
            CLAMP_WARNED.call_once(|| {
                // `eprintln!` rather than `tracing::warn!`: only `mlxcel-server`
                // installs a tracing subscriber, and the overrun this reports is
                // reachable from the CLI, where a `tracing` event is a no-op.
                eprintln!(
                    "GPT-2 reached position {} but the learned wpe table has only {} rows; \
                     every token from position {} on is embedded at the same row and output \
                     quality degrades. Keep prompt plus generated tokens within the model's \
                     {}-token context.",
                    offset.saturating_add(seq_len.max(1) - 1),
                    self.n_positions,
                    self.n_positions.saturating_sub(1),
                    self.n_positions
                );
            });
        }
        let position_index = mlxcel_core::from_slice_i32(&positions, &[positions.len() as i32]);
        let position_embeds = self.wpe.forward(&position_index);
        h = mlxcel_core::add(&h, &position_embeds);

        for (i, layer) in self.h.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.ln_f.forward(&h);

        // Tied output head.
        self.wte.as_linear(&h)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.h.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        // Reject an impossible config before reading the checkpoint, not after.
        // `from_weights` validates again for the owned-weights route.
        args.validate()?;

        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        strip_causal_mask_buffers(&mut weights);

        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        // `config.json` is untrusted: reject impossible scalars before any of
        // them sizes an allocation, indexes a table, or divides.
        args.validate()?;

        let layout = Gpt2Layout::detect(weights, args)?;
        let group_size = args.group_size();
        let bits = args.bits();

        let wte = UnifiedEmbedding::from_weights(weights, &layout.key("wte"), group_size, bits)?;
        let wpe_key = layout.key("wpe");
        let wpe = UnifiedEmbedding::from_weights(weights, &wpe_key, group_size, bits)?;
        validate_embedding_table(
            &wpe,
            &wpe_key,
            args.n_positions,
            "n_positions",
            args.n_embd,
            "n_embd",
        )?;

        let mut h = Vec::with_capacity(args.n_layer);
        for i in 0..args.n_layer {
            h.push(TransformerBlock::from_weights(weights, args, &layout, i)?);
        }

        let ln_f = layer_norm_from_weights(
            weights,
            &layout.key("ln_f"),
            args.n_embd,
            args.layer_norm_epsilon,
        )?;

        Ok(Self {
            wte,
            wpe,
            h,
            ln_f,
            n_positions: args.n_positions,
            eos_token_ids: args.eos_token_ids(),
        })
    }
}

// Helper functions.

/// Whether an MLX dimension equals an expected extent.
///
/// `array_shape` returns `i32`; a negative or oversized value can never match a
/// real extent, so the conversion failing is itself a mismatch.
///
/// Used by: BailingMoe, Gpt2, GptBigCode, GptNeoX
pub(crate) fn dim_eq(dim: i32, expected: usize) -> bool {
    usize::try_from(dim).is_ok_and(|d| d == expected)
}

/// Check a loaded embedding table against the config field that bounds lookups
/// into it, and return its row count.
///
/// Lookups are bounded by a number read out of `config.json`, not by the table:
/// `position_ids` clamps to `n_positions - 1`, and token ids are bounded by
/// `vocab_size`. Nothing else bounds the gather. MLX adds the axis size to a
/// negative index but performs no range check on a positive one, so an index
/// past the last row reads whatever follows the table in the buffer rather than
/// faulting or clamping. A checkpoint whose config overstates the table
/// therefore turns an ordinary prompt into an out-of-bounds read whose result
/// reaches the logits, and the row count is the only place that can be caught.
///
/// `claimed_rows_field` names the config field that bounds the lookup in the
/// rejection message (`n_positions` for `wpe`, `vocab_size` for `wte`), and
/// `width_field` names the config field that gives the model width (`n_embd` in
/// the GPT-2 lineage, `hidden_size` in the families that use the modern naming).
/// Both are message-only; naming the field the reader will actually find in
/// their `config.json` is the whole point of passing them.
///
/// A quantized table is packed along the last axis only, so its row count is
/// still dimension 0; the width check is skipped for it because the packed width
/// is a function of the bit depth rather than the model width.
///
/// Used by: BailingMoe, Gpt2, GptBigCode, GptNeoX
pub(crate) fn validate_embedding_table(
    table: &UnifiedEmbedding,
    key: &str,
    claimed_rows: usize,
    claimed_rows_field: &str,
    expected_cols: usize,
    width_field: &str,
) -> Result<usize, String> {
    let shape = mlxcel_core::array_shape(table.weight());
    let [rows, cols] = shape.as_slice() else {
        return Err(format!(
            "{key}.weight must be a 2-D [rows, n_embd] table, got shape {shape:?}"
        ));
    };
    let rows = usize::try_from(*rows).unwrap_or(0);
    if rows == 0 {
        return Err(format!("{key}.weight has no rows"));
    }
    if claimed_rows > rows {
        return Err(format!(
            "config {claimed_rows_field} ({claimed_rows}) exceeds the {rows} rows present in \
             {key}.weight. Lookups into this table are bounded by {claimed_rows_field}, so a \
             config that overstates it indexes past the end of the table, and the gather \
             behind an embedding lookup does not range-check a positive index."
        ));
    }
    if !table.is_quantized() && !dim_eq(*cols, expected_cols) {
        return Err(format!(
            "{key}.weight is [{rows}, {cols}] but {width_field} is {expected_cols}; every \
             embedding table must be the model width"
        ));
    }
    Ok(rows)
}

/// Load a `LayerNorm` with an optional bias (GPT-2 always ships the bias).
///
/// `dim` is the model width the norm must have. A norm weight of the wrong
/// width would otherwise broadcast-fail inside MLX, and an MLX C++ exception
/// crossing the cxx bridge is an uncatchable `std::terminate` rather than a
/// load error.
///
/// Used by: Gpt2, GptBigCode, GptNeoX
pub(crate) fn layer_norm_from_weights(
    weights: &WeightMap,
    prefix: &str,
    dim: usize,
    eps: f32,
) -> Result<LayerNorm, String> {
    let weight_name = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let weight_shape = mlxcel_core::array_shape(weight);
    if weight_shape.len() != 1 || !dim_eq(weight_shape[0], dim) {
        return Err(format!(
            "unexpected {weight_name} shape {weight_shape:?}: expected [{dim}]"
        ));
    }
    let weight = mlxcel_core::copy(weight);

    let bias = match weights.get(&format!("{prefix}.bias")) {
        Some(bias) => {
            let bias_shape = mlxcel_core::array_shape(bias);
            if bias_shape.len() != 1 || !dim_eq(bias_shape[0], dim) {
                return Err(format!(
                    "unexpected {prefix}.bias shape {bias_shape:?}: expected [{dim}]"
                ));
            }
            Some(mlxcel_core::copy(bias))
        }
        None => None,
    };

    Ok(LayerNorm::new(weight, bias, eps))
}

// LanguageModel trait implementation.

impl LanguageModel for Gpt2Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Gpt2Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Gpt2Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.h.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "gpt2_tests.rs"]
mod tests;
