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

//! Qwen3 MoE model implementation using mlxcel-core
//!
//! Implements Qwen3 MoE architecture with:
//! - Q/K normalization (RMSNorm after projection, before RoPE)
//! - Sparse MoE with top-k expert selection per token
//! - norm_topk_prob: normalized top-k scores after softmax
//! - decoder_sparse_step: MoE layer interval (dense MLP otherwise)
//! - mlp_only_layers: explicit list of dense layers
//! - Standard RoPE positional embeddings
//! - RMSNorm normalization

use crate::models::rope_utils::{RopeScalingKind, RopeScalingSpec};
use crate::models::switch_layers::validate_expert_quantization_params;
use mlxcel_core::cache::BatchedAttentionMetadata;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{FusedQKVLinear, KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    /// Checkpoint name used only to key and label RoPE fallback diagnostics.
    #[serde(skip)]
    pub checkpoint_label: Option<String>,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub norm_topk_prob: bool,

    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_tie_word_embeddings() -> bool {
    false
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

    pub fn set_checkpoint_label(&mut self, model_dir: &Path) {
        self.checkpoint_label = model_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|name| !name.is_empty());
    }

    pub fn model_label(&self) -> &str {
        self.checkpoint_label.as_deref().unwrap_or(&self.model_type)
    }

    /// Resolve `rope_scaling` for Qwen3-MoE attention.
    ///
    /// Unsupported or malformed schemes warn and keep the unscaled table,
    /// matching the shared Qwen3/Llama policy for args types that can be fed by
    /// broader VLM loader config shapes.
    pub fn rope_scaling_kind(&self) -> RopeScalingKind {
        let spec = self
            .rope_scaling
            .as_ref()
            .map(|block| RopeScalingSpec::from_lookup(|key| block.get(key)));
        RopeScalingKind::resolve(
            spec.as_ref(),
            self.head_dim,
            self.rope_theta,
            self.max_position_embeddings.map(|n| n as f32),
            self.model_label(),
        )
    }
}

// SwitchLinear: Stacked expert weights for MoE.
/// Stacked linear layers for MoE experts
/// Weights shape: [num_experts, output_dim, input_dim_packed]
pub enum SwitchLinear {
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
        num_experts: usize,
    },
    Regular {
        weight: UniquePtr<MlxArray>,
    },
}

impl SwitchLinear {
    /// Forward pass using gather_qmm for quantized or gather_mm for regular
    pub fn forward(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
        sorted_indices: bool,
    ) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                ..
            } => unsafe {
                mlxcel_core::gather_qmm(
                    x,
                    weight,
                    scales,
                    biases
                        .as_ref()
                        .map(|b| b as *const _)
                        .unwrap_or(std::ptr::null()),
                    std::ptr::null(), // lhs_indices
                    indices as *const _,
                    true, // transpose
                    *group_size,
                    *bits,
                    sorted_indices,
                    "affine",
                )
            },
            Self::Regular { weight } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                unsafe {
                    mlxcel_core::gather_mm(
                        x,
                        &wt,
                        std::ptr::null(),
                        indices as *const _,
                        sorted_indices,
                    )
                }
            }
        }
    }

    /// Borrowed quantized parts (weight, scales, biases, group_size, bits) for
    /// the fused MoE kernel; None for the Regular (non-quantized) variant.
    fn quantized_parts(&self) -> Option<(&MlxArray, &MlxArray, &MlxArray, i32, i32)> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                ..
            } => Some((
                weight.as_ref().unwrap(),
                scales.as_ref().unwrap(),
                biases.as_ref().unwrap(),
                *group_size,
                *bits,
            )),
            Self::Regular { .. } => None,
        }
    }
}

// SwitchGLU: SwiGLU with stacked expert weights.
/// SwitchGLU: SwiGLU activation with stacked expert weights for MoE
pub struct SwitchGLU {
    pub gate_proj: SwitchLinear,
    pub up_proj: SwitchLinear,
    pub down_proj: SwitchLinear,
}

impl SwitchGLU {
    /// Forward pass with kernel-fused SwiGLU activation
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];

        // Check if we should use sorted_indices optimization (>= 64 tokens)
        let total_elements = n_tokens * top_k;
        let do_sort = total_elements >= 64;

        // Expand x for broadcasting: [n_tokens, hidden] -> [n_tokens, 1, 1, hidden]
        let x_expanded = mlxcel_core::expand_dims(x, -2);
        let x_expanded = mlxcel_core::expand_dims(&x_expanded, -3);

        if do_sort {
            // Sort tokens by expert for better memory access
            let (sorted_x, sorted_idx, inv_order) = self.gather_sort(&x_expanded, indices);

            // Apply projections with sorted_indices=true
            let x_gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            let x_up = self.up_proj.forward(&sorted_x, &sorted_idx, true);

            // Kernel-fused SwiGLU: silu(gate) * up
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);

            // Down projection
            let output = self.down_proj.forward(&activated, &sorted_idx, true);

            // Restore original order
            self.scatter_unsort(&output, &inv_order, &indices_shape)
        } else {
            // Direct path without sorting
            let x_gate = self.gate_proj.forward(&x_expanded, indices, false);
            let x_up = self.up_proj.forward(&x_expanded, indices, false);

            // Kernel-fused SwiGLU: silu(gate) * up
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);

            // Down projection
            let output = self.down_proj.forward(&activated, indices, false);

            // Squeeze: [n_tokens, top_k, 1, hidden] -> [n_tokens, top_k, hidden]
            mlxcel_core::squeeze_axis(&output, -2)
        }
    }

    /// Single-token decode via the fused MoE expert Metal kernel (#268).
    /// gate/up are 4/8-bit, down also handles 6-bit. Returns None (caller falls
    /// back to `forward` + `moe_weighted_sum`) for any unsupported config:
    /// gate/up not 4/8-bit or down not 4/6/8-bit, gate/up bits mismatch,
    /// group_size mismatch, the Regular variant, or a non-single token `x`.
    pub fn forward_fused_kernel(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
        scores: &MlxArray,
    ) -> Option<UniquePtr<MlxArray>> {
        let (gw, gs, gb, ggs, gbits) = self.gate_proj.quantized_parts()?;
        let (uw, us, ub, ugs, ubits) = self.up_proj.quantized_parts()?;
        let (dw, ds, db, dgs, dbits) = self.down_proj.quantized_parts()?;
        // gate/up power-of-2 (kernel A); down also handles 6-bit (kernel B).
        if gbits != 4 && gbits != 8 {
            return None;
        }
        if dbits != 4 && dbits != 8 && dbits != 6 {
            return None;
        }
        if gbits != ubits || ggs != ugs || ggs != dgs {
            return None;
        }
        let gw_shape = mlxcel_core::array_shape(gw);
        if gw_shape.len() != 3 {
            return None;
        }
        let dff = gw_shape[1];
        let din = gw_shape[2] * (32 / gbits);
        if dbits == 6 && dff % 16 != 0 {
            return None;
        }
        let k = *mlxcel_core::array_shape(indices).last()?;
        let x_elems: i32 = mlxcel_core::array_shape(x).iter().product();
        if x_elems != din {
            return None;
        }
        let x_flat = mlxcel_core::reshape(x, &[din]);
        let idx_flat = mlxcel_core::reshape(indices, &[k]);
        let sc_flat = mlxcel_core::reshape(scores, &[k]);
        Some(mlxcel_core::fused_moe_expert_kernel(
            &x_flat, &idx_flat, gw, gs, gb, uw, us, ub, dw, ds, db, &sc_flat, din, dff, k, gbits,
            dbits, ggs,
        ))
    }

    /// Sort tokens by expert index for better memory access
    fn gather_sort(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let indices_shape = mlxcel_core::array_shape(indices);
        let top_k = indices_shape[indices_shape.len() - 1];

        // Flatten indices: [n_tokens, top_k] -> [n_tokens * top_k]
        let flat_indices = mlxcel_core::reshape(indices, &[-1]);

        // Sort indices by expert
        let order = mlxcel_core::argsort(&flat_indices, -1);
        let inv_order = mlxcel_core::argsort(&order, -1);

        // x is [n_tokens, 1, 1, hidden]
        // Flatten: [n_tokens, 1, hidden]
        let x_shape = mlxcel_core::array_shape(x);
        let x_flat = mlxcel_core::reshape(x, &[x_shape[0], 1, x_shape[3]]);

        // Divide order by top_k to get token indices
        let top_k_arr = mlxcel_core::from_slice_i32(&[top_k], &[1]);
        let token_indices = mlxcel_core::divide(&order, &top_k_arr);
        let token_indices = mlxcel_core::astype(&token_indices, mlxcel_core::dtype::INT32);

        // Take x rows in sorted order
        let sorted_x = mlxcel_core::take(&x_flat, &token_indices, 0);

        // Get sorted expert indices
        let sorted_indices = mlxcel_core::take(&flat_indices, &order, 0);

        (sorted_x, sorted_indices, inv_order)
    }

    /// Restore original order after sorted expert computation
    fn scatter_unsort(
        &self,
        x: &MlxArray,
        inv_order: &MlxArray,
        orig_shape: &[i32],
    ) -> UniquePtr<MlxArray> {
        // x has shape [n_sorted, 1, hidden]
        // Reorder by inv_order
        let unsorted = mlxcel_core::take(x, inv_order, 0);

        // Unflatten and squeeze
        let x_shape = mlxcel_core::array_shape(&unsorted);
        let n_tokens = orig_shape[0];
        let top_k = orig_shape[1];

        let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
        mlxcel_core::squeeze_axis(&reshaped, 2)
    }
}

/// Phase timings of one [`SparseMoeBlock::forward_profiled`] call.
///
/// `path` names the expert path the production dispatch takes on the same
/// input (`"fused"` or `"gather_qmm"`) and `tokens` the number of routed rows,
/// so a `[B, 1]` batched step and a single-token step are distinguishable in
/// the trace (#1616).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeProfile {
    pub gate_ms: f64,
    pub expert_ms: f64,
    pub combine_ms: f64,
    pub path: &'static str,
    pub tokens: i32,
}

impl MoeProfile {
    /// The `[QWEN3_MOE_SPARSE]` trace line shared by the single-sequence and
    /// batched decode paths.
    fn print(&self) {
        eprintln!(
            "[QWEN3_MOE_SPARSE] path={} tokens={} gate={:.2}ms expert={:.2}ms combine={:.2}ms",
            self.path, self.tokens, self.gate_ms, self.expert_ms, self.combine_ms
        );
    }
}

// Sparse MoE Block.
/// Qwen3 sparse mixture of experts layer
pub struct SparseMoeBlock {
    pub router: UnifiedLinear,
    pub experts: SwitchGLU,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if std::env::var("MLXCEL_PROFILE_QWEN3_MOE_DETAIL").is_ok() {
            let (out, _) = self.forward_profiled(x);
            return out;
        }

        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        // Flatten to [n_tokens, hidden]
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Get router logits
        let logits = self.router.forward(&x_flat);

        // Apply softmax to get routing probabilities
        let gates = mlxcel_core::softmax(&logits, -1);

        // Top-k selection using argpartition
        let k = self.num_experts_per_tok as i32;
        let n_experts = mlxcel_core::array_shape(&logits)[1];
        let kth = n_experts - k;

        let indices = mlxcel_core::argpartition(&logits, kth, -1);

        // Slice to get top-k: indices[..., kth:]
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices =
            mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

        // Get scores for top-k experts
        let mut scores = mlxcel_core::take_along_axis(&gates, &topk_indices, -1);

        // Normalize scores if enabled
        if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&scores, -1, true);
            scores = mlxcel_core::divide(&scores, &sum);
        }

        // Apply experts and weighted-sum. Fused single-token decode kernel
        // (#268) on by default; MLXCEL_FUSED_MOE=0 forces the proven SwitchGLU +
        // moe_weighted_sum path (also the automatic fallback when the kernel does
        // not support the config).
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

        // Reshape back to original shape
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    /// Device-synchronizing phase split of one MoE call, for
    /// `MLXCEL_PROFILE_QWEN3_MOE_DETAIL=1`.
    ///
    /// Takes the same expert path [`Self::forward`] would take on the same
    /// input: the fused kernel at one routed token, `gather_qmm` plus
    /// [`crate::models::switch_layers::moe_weighted_sum`] otherwise. Before
    /// #1616 it always ran the gather path, so a single-token trace attributed
    /// a kernel the production step never launched. The eval after each phase
    /// is what makes the split readable, and it also makes every number here a
    /// synchronization-inflated attribution figure rather than throughput;
    /// compare phases within one run and re-measure unsynchronized afterwards.
    pub fn forward_profiled(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, MoeProfile) {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };
        let tokens = mlxcel_core::array_shape(&x_flat)[0];

        let gate_start = std::time::Instant::now();
        let logits = self.router.forward(&x_flat);
        let gates = mlxcel_core::softmax(&logits, -1);
        let k = self.num_experts_per_tok as i32;
        let n_experts = mlxcel_core::array_shape(&logits)[1];
        let kth = n_experts - k;
        let indices = mlxcel_core::argpartition(&logits, kth, -1);
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices =
            mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);
        let mut scores = mlxcel_core::take_along_axis(&gates, &topk_indices, -1);
        if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&scores, -1, true);
            scores = mlxcel_core::divide(&scores, &sum);
        }
        mlxcel_core::eval(&scores);
        let gate_ms = gate_start.elapsed().as_secs_f64() * 1000.0;

        // Same dispatch as `forward`: the fused kernel only ever sees one
        // routed token, and it declines any config it does not support.
        let fused = if tokens == 1 && crate::models::switch_layers::fused_moe_enabled() {
            self.experts
                .forward_fused_kernel(&x_flat, &topk_indices, &scores)
                .map(|out| mlxcel_core::reshape(&out, &[1, hidden_dim]))
        } else {
            None
        };

        let expert_start = std::time::Instant::now();
        let (result, path, expert_ms, combine_ms) = match fused {
            Some(out) => {
                // The kernel pair folds the score-weighted K-sum into its
                // second launch, so there is no separate combine phase.
                mlxcel_core::eval(&out);
                let expert_ms = expert_start.elapsed().as_secs_f64() * 1000.0;
                (out, "fused", expert_ms, 0.0)
            }
            None => {
                let expert_out = self.experts.forward(&x_flat, &topk_indices);
                mlxcel_core::eval(&expert_out);
                let expert_ms = expert_start.elapsed().as_secs_f64() * 1000.0;

                let combine_start = std::time::Instant::now();
                let result = crate::models::switch_layers::moe_weighted_sum(
                    &expert_out,
                    &scores,
                    mlxcel_core::array_dtype(&x_flat),
                );
                mlxcel_core::eval(&result);
                let combine_ms = combine_start.elapsed().as_secs_f64() * 1000.0;
                (result, "gather_qmm", expert_ms, combine_ms)
            }
        };
        let result = if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        };

        (
            result,
            MoeProfile {
                gate_ms,
                expert_ms,
                combine_ms,
                path,
                tokens,
            },
        )
    }
}

// Dense MLP.
/// Dense MLP layer (used for mlp_only_layers)
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

// Attention with Q/K Normalization.
pub struct Attention {
    /// Fused QKV projection: Q, K, V weights concatenated along output dim.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm, // Q normalization
    pub k_norm: RMSNorm, // K normalization
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub rope_freqs: Option<UniquePtr<MlxArray>>,
    /// YaRN attention-magnitude multiplier applied to Q and K before the
    /// rotation (#1472). `1.0` (a skipped multiply) for every other scheme.
    pub rope_mscale: f32,
}

impl Attention {
    /// The Qwen3-MoE fused Q/K-norm launcher accepts a linear position scale,
    /// but it cannot accept a precomputed frequency table.
    fn fused_qk_norm_launcher_usable(&self) -> bool {
        self.rope_freqs.is_none()
    }

    fn apply_rope(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offset: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self.rope_freqs.as_ref() {
            Some(freqs) => {
                // YaRN magnitude correction (#1472): Q and K scale by
                // `rope_mscale` before the rotation so scores carry mscale^2.
                // At 1.0 the multiply is skipped and the graph is unchanged.
                let (q_scaled, k_scaled);
                let (q, k): (&MlxArray, &MlxArray) = if self.rope_mscale != 1.0 {
                    q_scaled = mlxcel_core::multiply_scalar(q, self.rope_mscale);
                    k_scaled = mlxcel_core::multiply_scalar(k, self.rope_mscale);
                    (&q_scaled, &k_scaled)
                } else {
                    (q, k)
                };
                (
                    mlxcel_core::fast_rope_with_freqs(
                        q,
                        self.rope_dims,
                        false,
                        self.rope_scale,
                        offset,
                        freqs,
                    ),
                    mlxcel_core::fast_rope_with_freqs(
                        k,
                        self.rope_dims,
                        false,
                        self.rope_scale,
                        offset,
                        freqs,
                    ),
                )
            }
            None => (
                mlxcel_core::fast_rope(
                    q,
                    self.rope_dims,
                    false,
                    self.rope_base,
                    self.rope_scale,
                    offset,
                ),
                mlxcel_core::fast_rope(
                    k,
                    self.rope_dims,
                    false,
                    self.rope_base,
                    self.rope_scale,
                    offset,
                ),
            ),
        }
    }

    /// Batched-decode RoPE: the same rotation as [`Self::apply_rope`] with one
    /// cache offset per batch row instead of a single offset for the whole
    /// tensor. Frequency-table schemes go through the table launcher and the
    /// YaRN magnitude multiply is applied exactly as on the single-sequence
    /// path, so a row decoded here matches the same row decoded alone.
    ///
    /// Used by: Qwen3MoE batched decode (Attention::forward_split_attention)
    fn apply_rope_batched(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offsets: &[i32],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self.rope_freqs.as_ref() {
            Some(freqs) => {
                let (q_scaled, k_scaled);
                let (q, k): (&MlxArray, &MlxArray) = if self.rope_mscale != 1.0 {
                    q_scaled = mlxcel_core::multiply_scalar(q, self.rope_mscale);
                    k_scaled = mlxcel_core::multiply_scalar(k, self.rope_mscale);
                    (&q_scaled, &k_scaled)
                } else {
                    (q, k)
                };
                (
                    mlxcel_core::fast_rope_batched_with_freqs(
                        q,
                        self.rope_dims,
                        false,
                        self.rope_scale,
                        offsets,
                        freqs,
                    ),
                    mlxcel_core::fast_rope_batched_with_freqs(
                        k,
                        self.rope_dims,
                        false,
                        self.rope_scale,
                        offsets,
                        freqs,
                    ),
                )
            }
            None => (
                mlxcel_core::fast_rope_batched(
                    q,
                    self.rope_dims,
                    false,
                    self.rope_base,
                    self.rope_scale,
                    offsets,
                ),
                mlxcel_core::fast_rope_batched(
                    k,
                    self.rope_dims,
                    false,
                    self.rope_base,
                    self.rope_scale,
                    offsets,
                ),
            ),
        }
    }

    /// Split-attention forward for batched decode (#1616).
    ///
    /// Receives the fused QKV projection of the whole batch as three
    /// `[B, T, proj_dim]` tensors, applies Q/K RMSNorm and per-row RoPE while
    /// still batched, then runs the KV-cache update and SDPA per sequence
    /// (each row owns its cache) and concatenates back to
    /// `[B, T, num_heads * head_dim]`. This is the dense-cache loop of the
    /// Qwen3 dense port; the paged-pool fast paths are not reproduced because
    /// this family does not opt into `supports_paged_decode_backend()`.
    ///
    /// Used by: Qwen3MoE batched decode (DecoderLayer::forward_batched)
    pub fn forward_split_attention(
        &self,
        q_batched: &MlxArray,
        k_batched: &MlxArray,
        v_batched: &MlxArray,
        caches: &mut [&mut KVCache],
        metadata: &BatchedAttentionMetadata,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let b = caches.len();
        let seq_len = mlxcel_core::array_shape(q_batched)[1];
        debug_assert_eq!(metadata.len(), b);

        let q_batched = mlxcel_core::reshape(
            q_batched,
            &[b as i32, seq_len, self.num_heads, self.head_dim],
        );
        let k_batched = mlxcel_core::reshape(
            k_batched,
            &[b as i32, seq_len, self.num_kv_heads, self.head_dim],
        );
        let v_batched = mlxcel_core::reshape(
            v_batched,
            &[b as i32, seq_len, self.num_kv_heads, self.head_dim],
        );

        // Q/K norm before the head transpose, then RoPE, as in `forward`.
        let q_batched = self.q_norm.forward(&q_batched);
        let k_batched = self.k_norm.forward(&k_batched);

        let q_batched = mlxcel_core::transpose_axes(&q_batched, &[0, 2, 1, 3]);
        let k_batched = mlxcel_core::transpose_axes(&k_batched, &[0, 2, 1, 3]);
        let v_batched = mlxcel_core::transpose_axes(&v_batched, &[0, 2, 1, 3]);

        let (q_batched, k_batched) =
            self.apply_rope_batched(&q_batched, &k_batched, &metadata.rope_offsets);

        let mut attn_outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(b);
        for (i, cache) in caches.iter_mut().enumerate() {
            let q_i = mlxcel_core::slice(
                &q_batched,
                &[i as i32, 0, 0, 0],
                &[i as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );
            let k_i = mlxcel_core::slice(
                &k_batched,
                &[i as i32, 0, 0, 0],
                &[i as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );
            let v_i = mlxcel_core::slice(
                &v_batched,
                &[i as i32, 0, 0, 0],
                &[i as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );

            let (cache_k, cache_v) = cache.update_and_fetch(k_i, v_i);

            let mask_i = mask.map(|m| {
                let sliced =
                    mlxcel_core::slice(m, &[i as i32, 0, 0], &[i as i32 + 1, seq_len, i32::MAX]);
                mlxcel_core::squeeze_axis(&sliced, 0)
            });

            let attn_out = if seq_len > 1 && mask_i.is_none() {
                mlxcel_core::causal_attention(&q_i, &cache_k, &cache_v, self.scale, 0.0, 0)
            } else {
                let mask_ptr = mask_i
                    .as_ref()
                    .map(|m| m.as_ref().unwrap() as *const _)
                    .unwrap_or(std::ptr::null());
                unsafe {
                    mlxcel_core::layers::attention_from_ptr(
                        &q_i, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                    )
                }
            };

            let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
            let attn_out =
                mlxcel_core::reshape(&attn_out, &[1, seq_len, self.num_heads * self.head_dim]);
            attn_outputs.push(attn_out);
        }

        let mut result = attn_outputs.remove(0);
        for attn_out in attn_outputs {
            result = mlxcel_core::concatenate(&result, &attn_out, 0);
        }
        result
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let offset = cache.offset;

        // On decode (l == 1) collapse the QKV projection, split, Q/K RMSNorm and
        // RoPE into one fused C++ kernel to cut per-token op count (#326). The
        // norm reduces over head_dim, which the head transpose leaves untouched,
        // so the fused result matches the graph path below. Prefill (l > 1),
        // non-quantized weights (the kernel returns None), and
        // MLXCEL_FUSED_QK_NORM=0 all take the graph path. Frequency-table
        // rope_scaling schemes also take the graph path because this launcher
        // can consume a linear scale but not a table.
        let fused = if l == 1
            && mlxcel_core::layers::fused_qk_norm_enabled()
            && self.fused_qk_norm_launcher_usable()
        {
            self.qkv_proj.forward_split_norm_rope_quantized(
                x,
                &self.q_norm,
                &self.k_norm,
                self.rope_dims,
                self.rope_base,
                self.rope_scale,
                offset,
            )
        } else {
            None
        };

        let (q, k, v) = if let Some((q, k, v)) = fused {
            (q, k, v)
        } else {
            // Fused QKV projection: single matmul → split into Q, K, V
            let (q, k, v) = self.qkv_proj.forward(x);

            // Reshape to [batch, seq_len, n_heads, head_dim]
            let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
            let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
            let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

            // Apply Q/K normalization BEFORE transpose
            let q = self.q_norm.forward(&q);
            let k = self.k_norm.forward(&k);

            // Transpose to [batch, n_heads, seq_len, head_dim]
            let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
            let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
            let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

            // Apply RoPE AFTER normalization
            let (q, k) = self.apply_rope(&q, &k, offset);
            (q, k, v)
        };

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
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

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        Self::from_weights_with_rope(weights, args, prefix, &args.rope_scaling_kind())
    }

    pub fn from_weights_with_rope(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        rope: &RopeScalingKind,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        // Load Q/K normalization weights
        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let head_dim = args.head_dim as i32;
        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_key_value_heads as i32;

        let rope = rope.duplicate();
        let rope_scale = rope.scale();
        let rope_mscale = rope.attn_scale();
        let rope_freqs = match rope {
            RopeScalingKind::Llama3 { freqs } | RopeScalingKind::Yarn { freqs, .. } => Some(freqs),
            _ => None,
        };

        if let Some(freqs) = rope_freqs.as_ref() {
            let shape = mlxcel_core::array_shape(freqs);
            let expected = head_dim / 2;
            if shape.len() != 1 || shape[0] != expected {
                return Err(format!(
                    "{prefix}: rope_scaling frequency table has shape {shape:?}, but this block rotates {head_dim} dims and needs [{expected}]"
                ));
            }
        }

        // Fused QKV: concatenate q/k/v weights into one projection at load time
        let qkv_proj = FusedQKVLinear::from_weights_separate(
            weights,
            prefix,
            group_size,
            bits,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_weight, args.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, args.rms_norm_eps),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim,
            rope_base: crate::models::rope_overrides::resolve_base(args.rope_theta),
            rope_scale,
            rope_freqs,
            rope_mscale,
        })
    }
}

// MLP Type Enum.
/// MLP type selection based on decoder_sparse_step and mlp_only_layers
pub enum MLPType {
    Dense(MLP),
    MoE(SparseMoeBlock),
}

impl MLPType {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MLPType::Dense(mlp) => mlp.forward(x),
            MLPType::MoE(moe) => moe.forward(x),
        }
    }
}

// Transformer Block.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: MLPType,
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
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm MLP/MoE
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn forward_profiled(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, f64, f64) {
        mlxcel_core::eval(x);

        let t0 = std::time::Instant::now();
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);
        mlxcel_core::eval(&h);
        let attn_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let normed = self.post_attention_layernorm.forward(&h);
        let mut moe_detail = None;
        let mlp_out = match &self.mlp {
            MLPType::Dense(mlp) => mlp.forward(&normed),
            MLPType::MoE(moe) => {
                if std::env::var("MLXCEL_PROFILE_QWEN3_MOE_DETAIL").is_ok() {
                    let (out, profile) = moe.forward_profiled(&normed);
                    moe_detail = Some(profile);
                    out
                } else {
                    moe.forward(&normed)
                }
            }
        };
        let out = mlxcel_core::add(&h, &mlp_out);
        mlxcel_core::eval(&out);
        let mlp_ms = t1.elapsed().as_secs_f64() * 1000.0;

        if let Some(p) = moe_detail {
            p.print();
        }

        (out, attn_ms, mlp_ms)
    }

    /// Batched decode step (#1616): the norms, the fused QKV projection, the
    /// output projection and the MoE / MLP block run once over
    /// `[B, T, hidden]`, while attention runs per sequence against each row's
    /// own cache.
    ///
    /// The MoE block flattens the input to `[B, hidden]`, so from B=2 its
    /// single-token fused-kernel gate declines and the experts run through
    /// `gather_qmm` over `B * top_k` slots in one launch chain. Before this
    /// method existed the `LanguageModel` default ran `forward` once per
    /// sequence, so a B=4 tick was four serialized single-token graphs; the
    /// op-level profile in
    /// `docs/benchmark_results/moe-batched-decode-m1ultra-2026-09-04.md` is
    /// what justified replacing it.
    ///
    /// Used by: Qwen3MoeModel::forward_batched_impl
    pub fn forward_batched(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_timed(x, caches, mask, false).0
    }

    /// [`Self::forward_batched`] with the `MLXCEL_PROFILE_BLOCKS` phase split.
    /// With `profile` set the graph is evaluated at the attention residual and
    /// at the block output so the two wall-clock spans can be attributed, which
    /// also makes them synchronization-inflated attribution numbers rather
    /// than throughput. With `profile` unset no eval is issued and both spans
    /// are reported as zero.
    pub fn forward_batched_timed(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
        mask: Option<&MlxArray>,
        profile: bool,
    ) -> (UniquePtr<MlxArray>, f64, f64) {
        if profile {
            mlxcel_core::eval(x);
        }
        let t0 = std::time::Instant::now();
        let normed = self.input_layernorm.forward(x);
        let (q, k, v) = self.self_attn.qkv_proj.forward(&normed);
        let seq_len = mlxcel_core::array_shape(&q)[1];
        let metadata = BatchedAttentionMetadata::uniform_kv_caches(caches, seq_len, 0)
            .expect("valid qwen3_moe batched attention metadata");
        let attn_concat = self
            .self_attn
            .forward_split_attention(&q, &k, &v, caches, &metadata, mask);
        let attn_out = self.self_attn.o_proj.forward(&attn_concat);
        let h = mlxcel_core::add(x, &attn_out);
        let attn_ms = if profile {
            mlxcel_core::eval(&h);
            t0.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };

        let t1 = std::time::Instant::now();
        let normed = self.post_attention_layernorm.forward(&h);
        // Same `MLXCEL_PROFILE_QWEN3_MOE_DETAIL` phase line as the
        // single-sequence path, so a batched tick reports the expert path it
        // took (`gather_qmm` at `tokens=B`) instead of staying silent.
        let mlp_out = match &self.mlp {
            MLPType::MoE(moe)
                if profile && std::env::var_os("MLXCEL_PROFILE_QWEN3_MOE_DETAIL").is_some() =>
            {
                let (out, p) = moe.forward_profiled(&normed);
                p.print();
                out
            }
            _ => self.mlp.forward(&normed),
        };
        let out = mlxcel_core::add(&h, &mlp_out);
        let mlp_ms = if profile {
            mlxcel_core::eval(&out);
            t1.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        (out, attn_ms, mlp_ms)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        Self::from_weights_with_rope(weights, args, layer_idx, &args.rope_scaling_kind())
    }

    pub fn from_weights_with_rope(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
        rope: &RopeScalingKind,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights_with_rope(
            weights,
            args,
            &format!("{}.self_attn", prefix),
            rope,
        )?;

        // Determine if this layer should be MoE or dense
        let is_moe = !args.mlp_only_layers.contains(&layer_idx)
            && args.num_experts > 0
            && (layer_idx + 1).is_multiple_of(args.decoder_sparse_step);

        let mlp = if is_moe {
            MLPType::MoE(SparseMoeBlock::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            MLPType::Dense(MLP::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        };

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// Qwen3 MoE Model.
pub struct Qwen3MoeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub tie_word_embeddings: bool,
}

impl Qwen3MoeModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let profile_blocks = std::env::var("MLXCEL_PROFILE_BLOCKS").is_ok()
            && mlxcel_core::array_shape(input_ids)[1] == 1;

        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers
        let mut attn_ms_total = 0.0f64;
        let mut moe_ms_total = 0.0f64;
        for (i, layer) in self.layers.iter().enumerate() {
            if profile_blocks {
                let is_moe = matches!(layer.mlp, MLPType::MoE(_));
                let (next_h, attn_ms, mlp_ms) = layer.forward_profiled(&h, &mut caches[i], mask);
                h = next_h;
                attn_ms_total += attn_ms;
                moe_ms_total += mlp_ms;
                let mlp_tag = if is_moe { "E" } else { "M" };
                eprintln!(
                    "[QWEN3_MOE_BLOCK {}] attn={:.2}ms mlp{}={:.2}ms",
                    i, attn_ms, mlp_tag, mlp_ms
                );
            } else {
                h = layer.forward(&h, &mut caches[i], mask);
            }
        }

        if profile_blocks {
            let total = (attn_ms_total + moe_ms_total).max(1e-9);
            eprintln!(
                "[QWEN3_MOE_BLOCKS] A:{:.1}ms({:.0}%) M:{:.1}ms({:.0}%) T:{:.1}ms",
                attn_ms_total,
                attn_ms_total * 100.0 / total,
                moe_ms_total,
                moe_ms_total * 100.0 / total,
                total
            );
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    /// Batched decode (#1616): `[B, 1]` token ids against `B` per-sequence
    /// cache slices, returning `[B, 1, vocab]` logits. The embedding, every
    /// layer's projections and MoE block, the final norm and the LM head run
    /// once over the batch; attention runs per sequence inside each layer.
    /// `MLXCEL_PROFILE_BLOCKS=1` prints the same per-layer attention / MLP
    /// split as the single-sequence path, tagged with the batch size.
    ///
    /// Used by: LanguageModel::forward_batched for Qwen3MoeModel
    pub fn forward_batched_impl(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let b = batch_caches.len();
        let profile_blocks = std::env::var_os("MLXCEL_PROFILE_BLOCKS").is_some();

        let mut h = self.embed_tokens.forward(input_ids);

        let mut attn_ms_total = 0.0f64;
        let mut mlp_ms_total = 0.0f64;
        for layer_idx in 0..self.layers.len() {
            let mut layer_caches: Vec<&mut KVCache> = batch_caches
                .iter_mut()
                .map(|caches| &mut caches[layer_idx])
                .collect();
            let (next_h, attn_ms, mlp_ms) = self.layers[layer_idx].forward_batched_timed(
                &h,
                &mut layer_caches,
                mask,
                profile_blocks,
            );
            h = next_h;
            if profile_blocks {
                attn_ms_total += attn_ms;
                mlp_ms_total += mlp_ms;
                eprintln!(
                    "[QWEN3_MOE_BLOCK {layer_idx}] b={b} attn={attn_ms:.2}ms mlp={mlp_ms:.2}ms"
                );
            }
        }
        if profile_blocks {
            let total = (attn_ms_total + mlp_ms_total).max(1e-9);
            eprintln!(
                "[QWEN3_MOE_BLOCKS] b={b} A:{:.1}ms({:.0}%) M:{:.1}ms({:.0}%) T:{:.1}ms",
                attn_ms_total,
                attn_ms_total * 100.0 / total,
                mlp_ms_total,
                mlp_ms_total * 100.0 / total,
                total
            );
        }

        let h = self.norm.forward(&h);
        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        };
        debug_assert_eq!(mlxcel_core::array_shape(&logits)[0], b as i32);
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
        let mut args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        args.set_checkpoint_label(model_dir);

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let rope = args.rope_scaling_kind();

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = DecoderLayer::from_weights_with_rope(weights, args, i, &rope)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head (or use tied embeddings)
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
            tie_word_embeddings: args.tie_word_embeddings,
        })
    }
}

// MoE Implementation Details.
impl SparseMoeBlock {
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate", prefix),
            args.group_size(),
            args.bits(),
        )?;

        let experts = SwitchGLU::from_weights(weights, args, &format!("{}.switch_mlp", prefix))?;

        Ok(Self {
            router,
            experts,
            num_experts_per_tok: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

impl SwitchGLU {
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(weights, args, &format!("{}.gate_proj", prefix))?,
            up_proj: SwitchLinear::from_weights(weights, args, &format!("{}.up_proj", prefix))?,
            down_proj: SwitchLinear::from_weights(weights, args, &format!("{}.down_proj", prefix))?,
        })
    }
}

impl SwitchLinear {
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
        let scales_key = format!("{}.scales", prefix);
        if weights.contains_key(&scales_key) {
            let (group_size, bits) = (args.group_size(), args.bits());
            // This type never reaches `reconcile_quantization_layout`, so the
            // declared pair is bounded here, where it is stored (issue #958).
            validate_expert_quantization_params(prefix, group_size, bits)?;
            let scales = mlxcel_core::copy(weights.get(&scales_key).unwrap());
            let biases = get_weight_copy(weights, &format!("{}.biases", prefix))?;
            let shape = mlxcel_core::array_shape(&weight);
            let num_experts = shape[0] as usize;
            Ok(Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                num_experts,
            })
        } else {
            Ok(Self::Regular { weight })
        }
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
impl LanguageModel for Qwen3MoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Qwen3MoeModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Qwen3MoeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Qwen3 MoE EOS token (same as Qwen3)
        vec![151645]
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        true
    }

    /// One row keeps the exact single-sequence graph (fused MoE kernel
    /// included, mask dropped as the trait default did); two or more rows take
    /// the batched forward (#1616). Zero rows return the trait default's empty
    /// logits so the scheduler's guard step stays a no-op.
    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        match batch_caches.len() {
            0 => mlxcel_core::zeros(&[0, 1, 1], mlxcel_core::dtype::FLOAT32),
            1 => Qwen3MoeModel::forward(self, input_ids, batch_caches[0], None),
            _ => self.forward_batched_impl(input_ids, batch_caches, mask),
        }
    }

    /// The context only carries paged-decode state, which this family does
    /// not opt into (`supports_paged_decode_backend` stays false), so it is
    /// ignored and the dense batched forward runs.
    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let _ = context;
        LanguageModel::forward_batched(self, input_ids, batch_caches, mask)
    }
}

#[cfg(test)]
#[path = "qwen3_moe_tests.rs"]
mod tests;
