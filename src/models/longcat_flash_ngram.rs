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

//! LongcatFlash and LongcatFlashNgram model implementation
//!
//! Key Features:
//! - MLA (Multi-head Latent Attention) with Q/KV LoRA compression
//! - Dual sub-layer decoder architecture (2x attn + 2x MLP per layer)
//! - MoE with identity zero experts and SwitchGLU
//! - NgramEmbedding with n-gram hash lookups (ngram variant)
//! - CacheList per layer (2 KVCaches) + ArraysCache for ngram context
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/longcat_flash.py, longcat_flash_ngram.py

use crate::models::gpt2::validate_embedding_table;
use crate::models::mla_rope::MlaRope;
use crate::models::model_owned::KvCacheLayerModes;
use crate::models::switch_layers::SwitchGLU;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::dtype;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, MultiLinear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{slice_axis, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
pub struct LongcatFlashNgramConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub moe_topk: usize,
    pub expert_ffn_hidden_size: usize,
    pub n_routed_experts: usize,
    pub zero_expert_num: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default)]
    pub mla_scale_q_lora: bool,
    #[serde(default)]
    pub mla_scale_kv_lora: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_zero_expert_type")]
    pub zero_expert_type: String,
    #[serde(default = "default_ngram_vocab_size_ratio")]
    pub ngram_vocab_size_ratio: usize,
    #[serde(default = "default_emb_neighbor_num")]
    pub emb_neighbor_num: usize,
    #[serde(default = "default_emb_split_num")]
    pub emb_split_num: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub router_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

mlxcel_core::impl_rope_lifting_deserialize!(LongcatFlashNgramConfig);

fn default_zero_expert_type() -> String {
    "identity".to_string()
}
fn default_ngram_vocab_size_ratio() -> usize {
    78
}
fn default_emb_neighbor_num() -> usize {
    4
}
fn default_emb_split_num() -> usize {
    4
}

impl LongcatFlashNgramConfig {
    fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub(super) fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub(super) fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    fn get_attention_scale(&self) -> f32 {
        let qk_head_dim = self.qk_head_dim() as f32;
        let mut scale = qk_head_dim.powf(-0.5);

        if let Some(ref rope_scaling) = self.rope_scaling
            && let (Some(mscale_all_dim), Some(factor)) = (
                rope_scaling.get("mscale_all_dim").and_then(|v| v.as_f64()),
                rope_scaling.get("factor").and_then(|v| v.as_f64()),
            )
            && mscale_all_dim > 0.0
            && factor > 1.0
        {
            let s = (0.1 * mscale_all_dim * factor.ln() + 1.0) as f32;
            scale = scale * s * s;
        }

        scale
    }

    fn is_ngram_model(&self) -> bool {
        self.model_type == "longcat_flash_ngram"
    }
}

// Helper: get weight with copy.
pub(super) fn get_weight_copy(
    weights: &WeightMap,
    name: &str,
) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Missing weight: {name}"))
}

// LongcatFlashMLA - Multi-head Latent Attention.
// Used by: LongcatFlash, LongcatFlashNgram, LongcatFlashSparse (which runs
// `project` and `attend_window` around its own RoPE and LSA key selection).
pub(super) struct LongcatFlashMLA {
    // Q projection with LoRA
    q_a_proj: Option<UnifiedLinear>,
    q_a_layernorm: Option<RMSNorm>,
    q_b_proj: Option<UnifiedLinear>,
    // Q projection without LoRA
    q_proj: Option<UnifiedLinear>,

    // KV projection with compression
    kv_a_proj_with_mqa: UnifiedLinear,
    kv_a_layernorm: RMSNorm,

    // MLA: embed_q and unembed_out replace kv_b_proj
    embed_q: MultiLinear,
    unembed_out: MultiLinear,

    o_proj: UnifiedLinear,

    pub(super) num_heads: i32,
    qk_head_dim: i32,
    pub(super) qk_rope_head_dim: i32,
    pub(super) qk_nope_head_dim: i32,
    pub(super) v_head_dim: i32,
    pub(super) kv_lora_rank: i32,
    scale: f32,
    /// The `q_pe` / `k_pe` rotation: the checkpoint's YaRN table
    /// (`longcat-flash-lite` ships `factor` 10 over 32768) or plain
    /// `fast_rope` at `rope_theta` (#3295).
    pub(super) rope: MlaRope,
    mla_scale_q_lora: Option<f32>,
    mla_scale_kv_lora: Option<f32>,
}

/// One step's MLA projections before RoPE.
pub(super) struct MlaProjections {
    /// The LoRA query hidden `q_a_layernorm(q_a_proj(x))`, the input of a
    /// LongCat-Flash-Lite-Sparse indexer; `None` without a query LoRA.
    pub(super) qr: Option<UniquePtr<MlxArray>>,
    /// `[B, H, L, qk_nope_head_dim]`.
    pub(super) q_nope: UniquePtr<MlxArray>,
    /// `[B, H, L, qk_rope_head_dim]`, not yet rotated.
    pub(super) q_pe: UniquePtr<MlxArray>,
    /// `[B, 1, L, kv_lora_rank]`, normed and scaled.
    pub(super) kv_latent: UniquePtr<MlxArray>,
    /// `[B, 1, L, qk_rope_head_dim]`, not yet rotated.
    pub(super) k_pe: UniquePtr<MlxArray>,
}

impl LongcatFlashMLA {
    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let MlaProjections {
            q_nope,
            q_pe,
            kv_latent,
            k_pe,
            ..
        } = self.project(x);

        // Apply RoPE
        let offset = cache.offset;
        let q_pe = self.rope.apply(&q_pe, true, offset);
        let k_pe = self.rope.apply(&k_pe, true, offset);

        // Cache stores (kv_latent, k_pe). An `int8` cache stores the latent as
        // the affine group-64 triple and hands back a quantized window; the
        // fresh chunk is quantized before this step reads it, so a prefill
        // reads what later decode steps read (#3161).
        let window = crate::models::mla_batched::store_latent_step(
            cache,
            self.geometry(),
            kv_latent,
            k_pe,
            "longcat_flash",
            true,
        );
        debug_assert!(
            l == 1 || mask.is_some() || window.seq_len() == cache.live_len(),
            "causal mode needs the live window to end at this step"
        );
        let absorbed = self.absorbed(l, cache);
        self.attend_window(window, &q_nope, &q_pe, absorbed, mask, b, l)
    }

    /// The layer's MLA shape, for the latent cache view.
    pub(super) fn geometry(&self) -> mlxcel_core::mla::MlaGeometry {
        crate::models::mla_batched::mla_geometry(
            self.num_heads,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
    }

    /// Cost-model gate (issue #2408): one query is always absorbed, and a
    /// multi-token step behind a long enough cache can be cheaper absorbed
    /// than up-projecting the whole window. A cold prefill stays on the
    /// materializing side. `cache` is the latent cache after this step's write.
    pub(super) fn absorbed(&self, l: i32, cache: &KVCache) -> bool {
        mlxcel_core::mla::use_absorbed_mla(
            l as usize,
            cache.seq_len() as usize,
            self.kv_lora_rank as usize,
            self.qk_nope_head_dim as usize,
            self.v_head_dim as usize,
        )
    }

    /// The query, the latent and the rope keys of one step, before RoPE.
    pub(super) fn project(&self, x: &MlxArray) -> MlaProjections {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Compute Q
        let (qr, q) = if let (Some(q_a_proj), Some(q_a_norm), Some(q_b_proj)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_a = q_a_proj.forward(x);
            let q_a_norm = q_a_norm.forward(&q_a);
            let q = q_b_proj.forward(&q_a_norm);
            (Some(q_a_norm), q)
        } else {
            (None, self.q_proj.as_ref().unwrap().forward(x))
        };

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.qk_head_dim]);
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Apply Q LoRA scaling
        if let Some(scale) = self.mla_scale_q_lora {
            let s = mlxcel_core::full_f32(&[1], scale, mlxcel_core::array_dtype(&q));
            q = mlxcel_core::multiply(&q, &s);
        }

        // Split Q into nope and pe
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim, -1);

        // Compute compressed KV
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);
        let compressed = slice_axis(&compressed_kv, -1, 0, self.kv_lora_rank);
        let k_pe = slice_axis(&compressed_kv, -1, self.kv_lora_rank, -1);

        // Reshape k_pe to [B, 1, L, rope_dim]
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        // KV latent with layernorm and optional scaling
        let mut kv_latent = self.kv_a_layernorm.forward(&compressed);
        if let Some(scale) = self.mla_scale_kv_lora {
            let s = mlxcel_core::full_f32(&[1], scale, mlxcel_core::array_dtype(&kv_latent));
            kv_latent = mlxcel_core::multiply(&kv_latent, &s);
        }

        // Expand kv_latent to [B, 1, L, kv_lora_rank] for cache
        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);

        MlaProjections {
            qr,
            q_nope,
            q_pe,
            kv_latent,
            k_pe,
        }
    }

    /// Attend `q_nope` / `q_pe` (rotated) over a latent window and project
    /// the result to `[B, L, hidden]`.
    ///
    /// `mask` is additive. A multi-token step with no mask runs MLX's causal
    /// mode over the window, which must then end at this step; a mask is
    /// applied as given (LongCat-Flash-Lite-Sparse folds its top-k selection
    /// into one). A single query attends every row of the window it is handed,
    /// which is how the sparse variant's gathered rows reach it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attend_window(
        &self,
        window: mlxcel_core::mla::LatentWindow,
        q_nope: &MlxArray,
        q_pe: &MlxArray,
        absorbed: bool,
        mask: Option<&MlxArray>,
        b: i32,
        l: i32,
    ) -> UniquePtr<MlxArray> {
        // A quantized window's one absorbed query goes to
        // `quantized_absorbed_attention` (the fused kernel's quantized variant
        // where it applies, #3156, the #3155 composed decode otherwise), which
        // records its own path. Every other step over a quantized window
        // dequantizes it once and runs the fp16 code below.
        let (kv_latent, k_pe) = match window {
            mlxcel_core::mla::LatentWindow::Quantized {
                ckv_q,
                scales,
                biases,
                kpe,
                bits,
                group_size,
            } if absorbed && l == 1 => {
                let q_projected = self.embed_q.forward(q_nope);
                let output = mlxcel_core::mla::quantized_absorbed_attention(
                    &q_projected,
                    q_pe,
                    &ckv_q,
                    &scales,
                    &biases,
                    &kpe,
                    bits,
                    group_size,
                    self.scale,
                    mask,
                );
                return self.project_attention_output(&self.unembed_out.forward(&output), b, l);
            }
            // A multi-token step reads the window dequantized, in the
            // queries' dtype so a bf16 model stays bf16 (#3161).
            window => {
                crate::models::mla_batched::dense_latent_in(window, mlxcel_core::array_dtype(q_pe))
            }
        };
        mlxcel_core::mla::stats::record(if absorbed {
            if l == 1 {
                mlxcel_core::mla::MlaDecodePath::AbsorbedComposed
            } else {
                mlxcel_core::mla::MlaDecodePath::AbsorbedPrefill
            }
        } else {
            mlxcel_core::mla::MlaDecodePath::Decompressed
        });

        // MLA attention: absorbed or materializing, chosen by the cost model
        // above rather than by the query count alone.
        let output = if l > 1 {
            // Multi-token step (issue #3099): fold the rope half into the SDPA
            // head dim, `[q_nope | q_pe] . [k | k_pe]`, instead of building the
            // dense `[B, H, L, S]` rope-score tensor and handing it to SDPA as
            // an additive mask. A maskless step is causal over the live window
            // `update_and_fetch` returned, through MLX's bottom-right aligned
            // causal mode; a caller mask is applied as given.
            let folded_mask = match mask {
                Some(m) => mlxcel_core::mla::FoldedMask::Array(m),
                None => mlxcel_core::mla::FoldedMask::Causal,
            };
            if absorbed {
                // Absorbed: Q in latent space, the latent is K and V.
                let q_projected = self.embed_q.forward(q_nope);
                let output = mlxcel_core::mla::rope_folded_attention(
                    &q_projected,
                    q_pe,
                    &kv_latent,
                    &k_pe,
                    &kv_latent,
                    self.scale,
                    folded_mask,
                );
                self.unembed_out.forward(&output)
            } else {
                // Materialize: project kv_latent to per-head K and V.
                let k = self.embed_q.forward_no_transpose(&kv_latent);
                let v = self.unembed_out.forward(&kv_latent);
                mlxcel_core::mla::rope_folded_attention(
                    q_nope,
                    q_pe,
                    &k,
                    &k_pe,
                    &v,
                    self.scale,
                    folded_mask,
                )
            }
        } else {
            // Single-token decode keeps the rope term as an additive mask: its
            // score row is one `[B, H, 1, S]` slice.
            // PE scores: (q_pe * scale) @ k_pe^T
            let scale_arr = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(q_pe));
            let scaled_q_pe = mlxcel_core::multiply(q_pe, &scale_arr);
            let k_pe_t = mlxcel_core::transpose_axes(&k_pe, &[0, 1, 3, 2]);
            let mut pe_scores = mlxcel_core::matmul(&scaled_q_pe, &k_pe_t);
            // Additive caller mask: 0 for attend, -inf for don't attend.
            if let Some(m) = mask {
                pe_scores = mlxcel_core::add(&pe_scores, m);
            }
            let pe_mask_ptr = &*pe_scores as *const MlxArray;
            if absorbed {
                // Absorbed: project Q into latent space, use kv_latent as K=V
                let q_projected = self.embed_q.forward(q_nope);
                // SAFETY: `pe_scores` outlives this call.
                let output = unsafe {
                    mlxcel_core::layers::attention_from_ptr(
                        &q_projected,
                        &kv_latent,
                        &kv_latent,
                        self.scale,
                        pe_mask_ptr,
                        0.0,
                        0,
                    )
                };
                // Project output from latent space to v_head_dim
                self.unembed_out.forward(&output)
            } else {
                // Materialize: project kv_latent to K and V
                let k = self.embed_q.forward_no_transpose(&kv_latent);
                let v = self.unembed_out.forward(&kv_latent);
                // SAFETY: `pe_scores` outlives this call.
                unsafe {
                    mlxcel_core::layers::attention_from_ptr(
                        q_nope,
                        &k,
                        &v,
                        self.scale,
                        pe_mask_ptr,
                        0.0,
                        0,
                    )
                }
            }
        };

        self.project_attention_output(&output, b, l)
    }

    /// `[B, H, L, v_head_dim]` attention output to `[B, L, hidden]`.
    pub(super) fn project_attention_output(
        &self,
        output: &MlxArray,
        b: i32,
        l: i32,
    ) -> UniquePtr<MlxArray> {
        let attn_out = mlxcel_core::transpose_axes(output, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.v_head_dim]);
        self.o_proj.forward(&attn_out)
    }

    /// Resolve this config's RoPE once, for every attention to duplicate.
    ///
    /// No training-context fallback, matching upstream `longcat_flash`
    /// (`initialize_rope` without `max_position_embeddings`, so `YarnRoPE`
    /// defaults a missing original context to 4096).
    pub(super) fn resolve_rope(args: &LongcatFlashNgramConfig) -> MlaRope {
        MlaRope::from_block(
            args.rope_scaling.as_ref(),
            args.qk_rope_head_dim,
            args.rope_theta,
            None,
            &args.model_type,
        )
    }

    pub(super) fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        prefix: &str,
        rope: MlaRope,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        // Q projection: with or without LoRA
        let has_q_lora = weights.contains_key(&format!("{prefix}.q_a_proj.weight"))
            || weights.contains_key(&format!("{prefix}.q_a_proj.scales"));

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if has_q_lora {
            let q_a =
                UnifiedLinear::from_weights(weights, &format!("{prefix}.q_a_proj"), gs, bits)?;
            let q_a_norm_w = get_weight_copy(weights, &format!("{prefix}.q_a_layernorm.weight"))?;
            let q_a_norm = RMSNorm::new(q_a_norm_w, args.rms_norm_eps);
            let q_b =
                UnifiedLinear::from_weights(weights, &format!("{prefix}.q_b_proj"), gs, bits)?;
            (Some(q_a), Some(q_a_norm), Some(q_b), None)
        } else {
            let q = UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), gs, bits)?;
            (None, None, None, Some(q))
        };

        let kv_a_proj_with_mqa = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.kv_a_proj_with_mqa"),
            gs,
            bits,
        )?;
        let kv_a_norm_w = get_weight_copy(weights, &format!("{prefix}.kv_a_layernorm.weight"))?;
        let kv_a_layernorm = RMSNorm::new(kv_a_norm_w, args.rms_norm_eps);

        // MLA: embed_q and unembed_out (decomposed from kv_b_proj by sanitize_weights)
        let embed_q = MultiLinear::from_weights(weights, &format!("{prefix}.embed_q"), gs, bits)?;
        let unembed_out =
            MultiLinear::from_weights(weights, &format!("{prefix}.unembed_out"), gs, bits)?;

        let o_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), gs, bits)?;

        let mla_scale_q_lora = if args.mla_scale_q_lora {
            Some((args.hidden_size as f32 / args.q_lora_rank as f32).sqrt())
        } else {
            None
        };
        let mla_scale_kv_lora = if args.mla_scale_kv_lora {
            Some((args.hidden_size as f32 / args.kv_lora_rank as f32).sqrt())
        } else {
            None
        };

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            embed_q,
            unembed_out,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            qk_head_dim: args.qk_head_dim() as i32,
            qk_rope_head_dim: args.qk_rope_head_dim as i32,
            qk_nope_head_dim: args.qk_nope_head_dim as i32,
            v_head_dim: args.v_head_dim as i32,
            kv_lora_rank: args.kv_lora_rank as i32,
            scale: args.get_attention_scale(),
            rope,
            mla_scale_q_lora,
            mla_scale_kv_lora,
        })
    }
}

// LongcatFlashMLP - Dense MLP (used for per-sublayer MLPs).
pub(super) struct LongcatFlashMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl LongcatFlashMLP {
    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let _ = hidden_size; // Used by Python for is_expert branch; we just load from weights
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.up_proj"), gs, bits)?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                gs,
                bits,
            )?,
        })
    }
}

// LongcatFlashTopkRouter.
struct LongcatFlashTopkRouter {
    classifier: UnifiedLinear,
    e_score_correction_bias: UniquePtr<MlxArray>,
    top_k: usize,
    _n_routed_experts: usize, // includes zero_expert_num
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
}

impl LongcatFlashTopkRouter {
    /// Returns (topk_indices, topk_weights)
    fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let logits = self.classifier.forward(x);
        let scores = mlxcel_core::softmax(&logits, -1);

        // Add correction bias
        let corrected = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Top-k selection via argpartition
        let neg_corrected = mlxcel_core::negative(&corrected);
        let indices = mlxcel_core::argpartition(&neg_corrected, self.top_k as i32 - 1, -1);

        // `[..., :top_k]` on the expert axis. The router input is
        // `[B, L, hidden]`, so the scores are rank 3: a two-axis `slice`
        // threw from MLX (an uncatchable abort across the bridge) on every
        // forward, and on a rank-2 input it would have cut the token axis.
        let topk_indices = slice_axis(&indices, -1, 0, self.top_k as i32);

        // Gather original scores (not corrected) for topk
        let topk_weights = mlxcel_core::take_along_axis(&scores, &topk_indices, -1);

        // Normalize if needed
        let topk_weights = if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&topk_weights, -1, true);
            let w_dtype = mlxcel_core::array_dtype(&topk_weights);
            let eps = mlxcel_core::full_f32(&[1], 1e-20, w_dtype);
            let denom = mlxcel_core::add(&sum, &eps);
            mlxcel_core::divide(&topk_weights, &denom)
        } else {
            topk_weights
        };

        // Scale
        let scale_arr = mlxcel_core::full_f32(
            &[1],
            self.routed_scaling_factor,
            mlxcel_core::array_dtype(&topk_weights),
        );
        let topk_weights = mlxcel_core::multiply(&topk_weights, &scale_arr);

        // Cast back to input dtype
        let input_dtype = mlxcel_core::array_dtype(x);
        let topk_weights = mlxcel_core::astype(&topk_weights, input_dtype);

        (topk_indices, topk_weights)
    }

    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        let total_experts = args.n_routed_experts + args.zero_expert_num;

        // classifier with special quantization (8-bit, group_size 64)
        let classifier = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.classifier"),
            64, // quant_predicate: group_size=64, bits=8 for classifier
            8,
        )?;

        let bias_key = format!("{prefix}.e_score_correction_bias");
        let e_score_correction_bias = if let Some(w) = weights.get(&bias_key) {
            mlxcel_core::copy(w)
        } else {
            mlxcel_core::zeros(&[total_experts as i32], dtype::FLOAT32)
        };

        let _ = (gs, bits); // router uses special quantization

        Ok(Self {
            classifier,
            e_score_correction_bias,
            top_k: args.moe_topk,
            _n_routed_experts: total_experts,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

// LongcatFlashMoE - MoE with identity zero experts.
pub(super) struct LongcatFlashMoE {
    router: LongcatFlashTopkRouter,
    switch_mlp: SwitchGLU,
    n_routed_experts: usize, // number of actual routed experts (not including zero)
}

impl LongcatFlashMoE {
    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (topk_indices, topk_weights) = self.router.forward(x);

        // Mask indices that point to zero (identity) experts
        let n_routed = mlxcel_core::from_slice_i32(&[self.n_routed_experts as i32], &[1]);
        let mask = mlxcel_core::greater_equal(&topk_indices, &n_routed);

        // Replace identity expert indices with 0 (they won't contribute)
        let zero = mlxcel_core::zeros_like(&topk_indices);
        let clamped_indices = mlxcel_core::where_cond(&mask, &zero, &topk_indices);

        // Zero out weights for identity experts in regular path
        let zero_f = mlxcel_core::zeros_like(&topk_weights);
        let regular_weights = mlxcel_core::where_cond(&mask, &zero_f, &topk_weights);

        // Run SwitchGLU for regular experts. It takes `[tokens, hidden]` and
        // `[tokens, k]`: handed `[B, L, ..]` it read `B` as the token count and
        // `L` as `k`, which broadcast by accident below 64 positions and threw
        // in the sorted path's reshape at 64 or more (#3161).
        let x_shape = mlxcel_core::array_shape(x);
        let hidden = x_shape[x_shape.len() - 1];
        // The router slices its indices to `[..., :top_k]`, so `top_k` is the
        // last axis; reading the configured value leaves no panic path.
        let k = self.router.top_k as i32;
        let flat = self.switch_mlp.forward(
            &mlxcel_core::reshape(x, &[-1, hidden]),
            &mlxcel_core::reshape(&clamped_indices, &[-1, k]),
        );
        let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
        out_shape.extend([k, hidden]);
        let regular_outputs = mlxcel_core::reshape(&flat, &out_shape);

        // Weight and sum expert outputs
        let weights_exp = mlxcel_core::expand_dims(&regular_weights, -1);
        let weighted = mlxcel_core::multiply(&regular_outputs, &weights_exp);
        let mut output = mlxcel_core::sum_axis(&weighted, -2, false);

        // Add identity expert contribution: x * sum(identity_weights)
        let identity_weights = mlxcel_core::where_cond(&mask, &topk_weights, &zero_f);
        let identity_sum = mlxcel_core::sum_axis(&identity_weights, -1, true);
        let identity_contrib = mlxcel_core::multiply(x, &identity_sum);
        output = mlxcel_core::add(&output, &identity_contrib);

        output
    }

    pub(super) fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        let router =
            LongcatFlashTopkRouter::from_weights(weights, args, &format!("{prefix}.router"))?;
        let switch_mlp =
            SwitchGLU::from_weights(weights, &format!("{prefix}.switch_mlp"), gs, bits)?;

        Ok(Self {
            router,
            switch_mlp,
            n_routed_experts: args.n_routed_experts,
        })
    }
}

// LongcatFlashDecoderLayer - dual sub-layer architecture.
// Each layer has 2 attention + 2 MLP sub-layers plus a shared MoE MLP.
struct LongcatFlashDecoderLayer {
    self_attn: [LongcatFlashMLA; 2],
    mlps: [LongcatFlashMLP; 2],
    input_layernorm: [RMSNorm; 2],
    post_attention_layernorm: [RMSNorm; 2],
    moe_mlp: LongcatFlashMoE,
}

#[allow(clippy::needless_range_loop)]
impl LongcatFlashDecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        caches: &mut [&mut KVCache; 2],
    ) -> UniquePtr<MlxArray> {
        let mut hidden = mlxcel_core::copy(x);
        let mut shortcut_mlp_output: Option<UniquePtr<MlxArray>> = None;

        for i in 0..2 {
            let residual = mlxcel_core::copy(&hidden);

            // Self-attention
            let normed = self.input_layernorm[i].forward(&hidden);
            let attn_out = self.self_attn[i].forward(&normed, caches[i], mask);
            hidden = mlxcel_core::add(&residual, &attn_out);

            // MLP
            let residual = mlxcel_core::copy(&hidden);
            let normed = self.post_attention_layernorm[i].forward(&hidden);

            // Compute MoE on first sub-layer's post-attn norm
            if i == 0 {
                shortcut_mlp_output = Some(self.moe_mlp.forward(&normed));
            }

            let mlp_out = self.mlps[i].forward(&normed);
            hidden = mlxcel_core::add(&residual, &mlp_out);

            // Add shortcut MoE output after second sub-layer
            if i == 1
                && let Some(ref shortcut) = shortcut_mlp_output
            {
                hidden = mlxcel_core::add(&hidden, shortcut);
            }
        }

        hidden
    }

    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        rope: &MlaRope,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        let self_attn_0 = LongcatFlashMLA::from_weights(
            weights,
            args,
            &format!("{prefix}.self_attn.0"),
            rope.duplicate(),
        )?;
        let self_attn_1 = LongcatFlashMLA::from_weights(
            weights,
            args,
            &format!("{prefix}.self_attn.1"),
            rope.duplicate(),
        )?;

        let mlp_0 = LongcatFlashMLP::from_weights(
            weights,
            &format!("{prefix}.mlps.0"),
            args.hidden_size,
            gs,
            bits,
        )?;
        let mlp_1 = LongcatFlashMLP::from_weights(
            weights,
            &format!("{prefix}.mlps.1"),
            args.hidden_size,
            gs,
            bits,
        )?;

        let in_norm_0_w = get_weight_copy(weights, &format!("{prefix}.input_layernorm.0.weight"))?;
        let in_norm_1_w = get_weight_copy(weights, &format!("{prefix}.input_layernorm.1.weight"))?;
        let post_norm_0_w = get_weight_copy(
            weights,
            &format!("{prefix}.post_attention_layernorm.0.weight"),
        )?;
        let post_norm_1_w = get_weight_copy(
            weights,
            &format!("{prefix}.post_attention_layernorm.1.weight"),
        )?;

        let moe_mlp = LongcatFlashMoE::from_weights(weights, args, &format!("{prefix}.mlp"))?;

        Ok(Self {
            self_attn: [self_attn_0, self_attn_1],
            mlps: [mlp_0, mlp_1],
            input_layernorm: [
                RMSNorm::new(in_norm_0_w, args.rms_norm_eps),
                RMSNorm::new(in_norm_1_w, args.rms_norm_eps),
            ],
            post_attention_layernorm: [
                RMSNorm::new(post_norm_0_w, args.rms_norm_eps),
                RMSNorm::new(post_norm_1_w, args.rms_norm_eps),
            ],
            moe_mlp,
        })
    }
}

// NgramEmbedding - n-gram hash embeddings.
struct NgramEmbedder {
    embedding: UnifiedEmbedding,
    post_proj: UnifiedLinear,
    _vocab_dim: usize,
}

pub(super) struct NgramEmbedding {
    word_embeddings: UnifiedEmbedding,
    embedders: Vec<NgramEmbedder>,
    vocab_mods: HashMap<(usize, usize), Vec<i64>>,
    _vocab_size: usize,
    m: usize, // ngram_vocab_size_ratio * vocab_size
    k: usize, // emb_split_num
    n: usize, // emb_neighbor_num
}

impl NgramEmbedding {
    /// Check the config fields the n-gram arithmetic divides by, multiplies
    /// and casts to INT32, returning `(m, num_embedders)`. A checkpoint config
    /// is untrusted, so every case that would otherwise underflow, divide by
    /// zero, overflow or truncate is an `Err` here, before any of it runs.
    pub(super) fn validate_config(
        args: &LongcatFlashNgramConfig,
    ) -> Result<(usize, usize), String> {
        let k = args.emb_split_num;
        let n = args.emb_neighbor_num;
        if n < 2 {
            return Err(format!(
                "config emb_neighbor_num ({n}) must be at least 2 for an n-gram model"
            ));
        }
        if k == 0 {
            return Err("config emb_split_num must be at least 1".to_string());
        }
        let num_embedders = k.checked_mul(n - 1).ok_or_else(|| {
            format!("config emb_split_num ({k}) * (emb_neighbor_num ({n}) - 1) overflows")
        })?;
        if args.hidden_size == 0 || !args.hidden_size.is_multiple_of(num_embedders) {
            return Err(format!(
                "config hidden_size ({}) must be a positive multiple of \
                 emb_split_num * (emb_neighbor_num - 1) ({num_embedders})",
                args.hidden_size
            ));
        }
        if i32::try_from(args.vocab_size).is_err() {
            return Err(format!(
                "config vocab_size ({}) exceeds i32::MAX",
                args.vocab_size
            ));
        }
        let m = args
            .ngram_vocab_size_ratio
            .checked_mul(args.vocab_size)
            .ok_or_else(|| {
                format!(
                    "config ngram_vocab_size_ratio ({}) * vocab_size ({}) overflows",
                    args.ngram_vocab_size_ratio, args.vocab_size
                )
            })?;
        // The largest embedder vocab is cast to INT32 in `forward`.
        let largest = (num_embedders - 1)
            .checked_mul(2)
            .and_then(|extra| m.checked_add(extra))
            .and_then(|v| v.checked_add(1));
        if largest.is_none_or(|v| i32::try_from(v).is_err()) {
            return Err(format!(
                "config ngram_vocab_size_ratio * vocab_size ({m}) plus the n-gram embedder \
                 offsets exceeds i32::MAX"
            ));
        }
        Ok((m, num_embedders))
    }

    pub(super) fn compute_vocab_mods(
        n: usize,
        k: usize,
        m: usize,
        vocab_size: usize,
    ) -> HashMap<(usize, usize), Vec<i64>> {
        let mut vocab_mods = HashMap::new();
        for i in 2..=n {
            for j in 0..k {
                let index = (i - 2) * k + j;
                let emb_vocab_dim = (m + index * 2 + 1) as i64;
                let mut mods = Vec::new();
                let mut power_mod: i64 = 1;
                for _ in 0..i - 1 {
                    power_mod = (power_mod * vocab_size as i64) % emb_vocab_dim;
                    mods.push(power_mod);
                }
                vocab_mods.insert((i, j), mods);
            }
        }
        vocab_mods
    }

    pub(super) fn shift_right(x: &MlxArray, n: usize) -> UniquePtr<MlxArray> {
        if n == 0 {
            return mlxcel_core::copy(x);
        }
        let shape = mlxcel_core::array_shape(x);
        let batch_size = shape[0];
        let seq_len = shape[1];
        if seq_len <= n as i32 {
            return mlxcel_core::zeros_like(x);
        }
        let zeros = mlxcel_core::zeros(&[batch_size, n as i32], dtype::INT64);
        let kept = mlxcel_core::slice(x, &[0, 0], &[batch_size, seq_len - n as i32]);
        mlxcel_core::concatenate(&zeros, &kept, -1)
    }

    /// The lookup ids of one n-gram embedder for this call's `seq_len`
    /// tokens: `(context + sum_k shifted_k * vocab_mods[k - 2]) % emb_vocab_dim`
    /// over the `ngram - 1` preceding tokens, the last `seq_len` positions,
    /// as INT32.
    ///
    /// Used by: LongcatFlashNgram (`NgramEmbedding::forward`) and
    /// LongcatFlashSparse (`longcat_flash_sparse::SparseNgramEmbedding`).
    pub(super) fn hashed_ids(
        cur_context: &MlxArray,
        shifted_ids: &HashMap<usize, UniquePtr<MlxArray>>,
        vocab_mods: &[i64],
        ngram: usize,
        emb_vocab_dim: i64,
        seq_len: i32,
    ) -> UniquePtr<MlxArray> {
        // Compute ngram_ids = context + sum(shifted * power_mod)
        let mut ngram_ids = mlxcel_core::copy(cur_context);
        for k_idx in 2..=ngram {
            let shifted = &shifted_ids[&k_idx];
            let mod_val = vocab_mods[k_idx - 2];
            let mod_arr = mlxcel_core::from_slice_i64(&[mod_val], &[1]);
            let term = mlxcel_core::multiply(shifted, &mod_arr);
            ngram_ids = mlxcel_core::add(&ngram_ids, &term);
        }

        // new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
        let dim_arr = mlxcel_core::from_slice_i64(&[emb_vocab_dim], &[1]);
        let modded = mlxcel_core::remainder(&ngram_ids, &dim_arr);
        let modded_shape = mlxcel_core::array_shape(&modded);
        let modded_len = modded_shape[modded_shape.len() - 1];
        let new_ids = if modded_len > seq_len {
            slice_axis(&modded, 1, modded_len - seq_len, modded_len)
        } else {
            modded
        };

        // Cast to int32 for embedding lookup
        mlxcel_core::astype(&new_ids, dtype::INT32)
    }

    /// The n-gram context this call hashes over (the carried tail joined with
    /// the new tokens, as INT64), saving its last `n - 1` tokens back into
    /// `context` for the next call.
    ///
    /// Used by: LongcatFlashNgram and LongcatFlashSparse, as
    /// [`Self::hashed_ids`].
    pub(super) fn advance_context(
        input_ids: &MlxArray,
        context: &mut Option<UniquePtr<MlxArray>>,
        n: usize,
    ) -> UniquePtr<MlxArray> {
        let input_i64 = mlxcel_core::astype(input_ids, dtype::INT64);

        // The hashes of this call's tokens read up to `n - 1` tokens back, so
        // they run over the carried tail joined with the new tokens. Only the
        // saved tail is cut to `n - 1`: cutting the joined context too, as
        // this once did, dropped the oldest neighbour of every decode step.
        let cur_context = match context.as_ref() {
            Some(prev) => mlxcel_core::concatenate(prev, &input_i64, -1),
            None => mlxcel_core::copy(&input_i64),
        };

        // Save context for next call
        let ctx_shape = mlxcel_core::array_shape(&cur_context);
        let ctx_len = ctx_shape[ctx_shape.len() - 1];
        let keep = (n as i32 - 1).max(0);
        *context = Some(if ctx_len > keep {
            let start = ctx_len - keep;
            slice_axis(&cur_context, 1, start, ctx_len)
        } else {
            mlxcel_core::copy(&cur_context)
        });
        cur_context
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        context: &mut Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape[shape.len() - 1];

        let cur_context = Self::advance_context(input_ids, context, self.n);

        // Word embeddings
        let mut x = self.word_embeddings.forward(input_ids);

        // Compute shifted IDs
        let mut shifted_ids: HashMap<usize, UniquePtr<MlxArray>> = HashMap::new();
        for i in 2..=self.n {
            shifted_ids.insert(i, Self::shift_right(&cur_context, i - 1));
        }

        // Add n-gram embeddings
        let mut emb_idx = 0;
        for i in 2..=self.n {
            for j in 0..self.k {
                let emb_vocab_dim = (self.m + emb_idx * 2 + 1) as i64;
                let new_ids_i32 = Self::hashed_ids(
                    &cur_context,
                    &shifted_ids,
                    &self.vocab_mods[&(i, j)],
                    i,
                    emb_vocab_dim,
                    seq_len,
                );

                // Lookup embedding and project
                let x_ngram = self.embedders[emb_idx].embedding.forward(&new_ids_i32);
                let x_proj = self.embedders[emb_idx].post_proj.forward(&x_ngram);
                x = mlxcel_core::add(&x, &x_proj);

                emb_idx += 1;
            }
        }

        // Normalize by (1 + k * (n-1))
        let divisor = (1 + self.k * (self.n - 1)) as f32;
        let div_arr = mlxcel_core::full_f32(&[1], divisor, mlxcel_core::array_dtype(&x));
        mlxcel_core::divide(&x, &div_arr)
    }

    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        let (m, num_embedders) = Self::validate_config(args)?;
        let k = args.emb_split_num;
        let n = args.emb_neighbor_num;
        let emb_dim = args.hidden_size / num_embedders;

        // `from_weights` detects a quantized table by its `.scales` key: the
        // MLX 4-bit checkpoints quantize the word and n-gram tables, whose
        // packed U32 rows a plain `Embedding` would hand back as 1/8-wide
        // integers (#3161).
        let word_embeddings = UnifiedEmbedding::from_weights(
            weights,
            &format!("{prefix}.word_embeddings"),
            gs,
            bits,
        )?;
        // The gathers do not range-check a positive index, so a table shorter
        // or narrower than the config claims must fail here, not on a prompt.
        validate_embedding_table(
            &word_embeddings,
            &format!("{prefix}.word_embeddings"),
            args.vocab_size,
            "vocab_size",
            args.hidden_size,
            "hidden_size",
        )?;

        let mut embedders = Vec::with_capacity(num_embedders);
        for i in 0..num_embedders {
            let emb_vocab_size = m + i * 2 + 1;
            let embedding = UnifiedEmbedding::from_weights(
                weights,
                &format!("{prefix}.embedders.{i}"),
                gs,
                bits,
            )?;
            validate_embedding_table(
                &embedding,
                &format!("{prefix}.embedders.{i}"),
                emb_vocab_size,
                "ngram_vocab_size_ratio * vocab_size + 2 * i + 1",
                emb_dim,
                "hidden_size / (emb_split_num * (emb_neighbor_num - 1))",
            )?;
            let post_proj = UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.post_projs.{i}"),
                gs,
                bits,
            )?;
            embedders.push(NgramEmbedder {
                embedding,
                post_proj,
                _vocab_dim: emb_vocab_size,
            });
        }

        let vocab_mods = Self::compute_vocab_mods(n, k, m, args.vocab_size);

        Ok(Self {
            word_embeddings,
            embedders,
            vocab_mods,
            _vocab_size: args.vocab_size,
            m,
            k,
            n,
        })
    }
}

/// Per-sequence state the model owns (#2544): the dual sub-layer KV caches
/// and the n-gram token tail. The family cannot batch, so the engine hands
/// `forward` an empty cache slice; `make_caches` resets this instead.
struct LongcatState {
    caches: Vec<[KVCache; 2]>,
    ngram_context: Option<UniquePtr<MlxArray>>,
}

impl LongcatState {
    /// Fresh state whose layer `i` pair is built in `mode_for_layer(i)`; both
    /// sub-layer attentions of a layer share its mode.
    fn new(num_layers: usize, mode_for_layer: impl Fn(usize) -> KVCacheMode) -> Self {
        Self {
            caches: (0..num_layers)
                .map(|i| {
                    let mode = mode_for_layer(i);
                    [KVCache::new_with_mode(mode), KVCache::new_with_mode(mode)]
                })
                .collect(),
            ngram_context: None,
        }
    }
}

// LongcatFlashNgramModel - top-level model.
pub struct LongcatFlashNgramModel {
    // For ngram model
    ngram_embeddings: Option<NgramEmbedding>,
    // For plain longcat_flash model
    embed_tokens: Option<UnifiedEmbedding>,

    layers: Vec<LongcatFlashDecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    num_layers: usize,
    _is_ngram: bool,
    state: RefCell<LongcatState>,
    /// The per-layer KV cache modes the state's caches are built in: the
    /// injected table with every mode the latent pair cannot hold replaced by
    /// FP16 (#3161). Unset until a table is injected, which builds FP16.
    kv_cache_layer_modes: KvCacheLayerModes,
    /// Set once the refused-mode warning has been logged, so a client that
    /// injects modes per sequence (`DirectEngine`) does not repeat it.
    kv_mode_warned: Cell<bool>,
    /// Whether an injected `int8` layer keeps the affine int8 latent. Read
    /// from `mla::stores_quantized_latent_pair` at load, so it follows
    /// `mla::QUANTIZED_LATENT_FAMILIES`, which lists both LongCat types since
    /// LongCat-Flash-Lite passed the #3161 quality gate, and the MLA latent
    /// option in force (`--mla-latent-quant fp16` turns it off, issue #3179).
    /// Off, `int8` resolves to FP16 like any other refused mode.
    pub(crate) quantized_latent: Cell<bool>,
}

impl LongcatFlashNgramModel {
    /// The `config.json` `model_type` this instance was built from, the key
    /// the latent-pair mode rules match on.
    fn model_type(&self) -> &'static str {
        if self._is_ngram {
            "longcat_flash_ngram"
        } else {
            "longcat_flash"
        }
    }

    pub fn load(model_path: &str) -> Result<(Self, usize), String> {
        let path = Path::new(model_path);
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config: {e}"))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: LongcatFlashNgramConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config: {e}"))?;

        let mut weights = crate::models::load_text_weights(path, None)?;
        weights = sanitize_weights(weights, &args)?;

        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args.vocab_size))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        // Embeddings
        let (ngram_embeddings, embed_tokens) = if args.is_ngram_model() {
            let ngram = NgramEmbedding::from_weights(weights, args, "model.ngram_embeddings")?;
            (Some(ngram), None)
        } else {
            let embed = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;
            validate_embedding_table(
                &embed,
                "model.embed_tokens",
                args.vocab_size,
                "vocab_size",
                args.hidden_size,
                "hidden_size",
            )?;
            (None, Some(embed))
        };

        // Layers
        // One YaRN table for the model, duplicated into every attention.
        let rope = LongcatFlashMLA::resolve_rope(args);
        let mut layers = Vec::with_capacity(args.num_layers);
        for i in 0..args.num_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(LongcatFlashDecoderLayer::from_weights(
                weights, args, &rope, &prefix,
            )?);
        }

        // Final norm
        let norm_w = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_w, args.rms_norm_eps);

        // LM head
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?;

        Ok(Self {
            ngram_embeddings,
            embed_tokens,
            layers,
            norm,
            lm_head,
            num_layers: args.num_layers,
            _is_ngram: args.is_ngram_model(),
            state: RefCell::new(LongcatState::new(args.num_layers, |_| KVCacheMode::Fp16)),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
            kv_mode_warned: Cell::new(false),
            quantized_latent: Cell::new(mlxcel_core::mla::stores_quantized_latent_pair(
                if args.is_ngram_model() {
                    "longcat_flash_ngram"
                } else {
                    "longcat_flash"
                },
            )),
        })
    }
}

impl LongcatFlashNgramModel {
    /// Internal forward pass with proper cache management.
    ///
    /// `row == None` projects every position; `Some(row)` projects only that
    /// position (`[B, 1, vocab]`, the prefill's sampled row, #3070). The
    /// caches and the n-gram tail advance over every position either way.
    fn internal_forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [[KVCache; 2]],
        ngram_context: &mut Option<UniquePtr<MlxArray>>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        // Compute embeddings
        let h = if let Some(ref ngram) = self.ngram_embeddings {
            ngram.forward(input_ids, ngram_context)
        } else {
            self.embed_tokens.as_ref().unwrap().forward(input_ids)
        };

        // No mask array: a multi-token step runs MLX's causal mode inside the
        // rope-folded attention, bottom-right aligned over the live window
        // `update_and_fetch` returns, which masks the same keys as
        // the `[seq_len, live_len + seq_len]` additive mask did (#421, #3099).
        let mut hidden = h;
        for (i, layer) in self.layers.iter().enumerate() {
            let [ref mut c0, ref mut c1] = caches[i];
            hidden = layer.forward(&hidden, None, &mut [c0, c1]);
        }

        let hidden = mlxcel_core::generate::select_row(hidden, row);

        // Final norm and LM head
        let normed = self.norm.forward(&hidden);
        self.lm_head.forward(&normed)
    }
}

impl LanguageModel for LongcatFlashNgramModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // The caller's slice is ignored (empty on the engine path). Building
        // fresh caches and an empty n-gram context here, as this once did,
        // made every decode step attend to its own token alone at RoPE
        // position 0 and lost every earlier prefill chunk.
        let mut state = self.state.borrow_mut();
        let LongcatState {
            caches,
            ngram_context,
        } = &mut *state;
        self.internal_forward(input_ids, caches, ngram_context, None)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<mlxcel_core::cache::SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        // No embeddings path: decline so the default keeps its behavior.
        if input_embeddings.is_some() {
            return None;
        }
        let mut state = self.state.borrow_mut();
        let LongcatState {
            caches,
            ngram_context,
        } = &mut *state;
        Some(self.internal_forward(input_ids, caches, ngram_context, Some(last_pos)))
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn supports_batching(&self) -> bool {
        // One model-owned state slot (#2544); batching would need it keyed
        // per sequence.
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        // Pad positions would land in the model-owned caches and n-gram tail,
        // which no caller trim reaches (#1755).
        false
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset the model-owned state: the engine (`DirectEngine` opening a
        // sequence) and the scheduler (at prefill) call this per sequence.
        // The returned vector is never read.
        self.reset_runtime_state();
        (0..self.num_layers).map(|_| KVCache::new()).collect()
    }

    fn reset_runtime_state(&self) {
        *self.state.borrow_mut() = LongcatState::new(self.num_layers, |i| {
            self.kv_cache_layer_modes.mode_for_layer(i)
        });
    }

    /// Record the per-layer table the latent pair can hold and rebuild the
    /// model-owned caches in it.
    ///
    /// Each layer's MLA attention stores the `(kv_latent, k_pe)` pair through
    /// `mla_batched::store_latent_step`, so an `int8` layer keeps the affine
    /// group-64 latent with fp16 rope keys when the model opted in
    /// (`quantized_latent`, #3161). A mode
    /// `mlxcel_core::mla::latent_pair_supports_mode` refuses (every Turbo
    /// mode, whose codebook calibrated from one slot would be applied to the
    /// other, and `int8` while LongCat is not a quantized latent family) is
    /// replaced by FP16 with one warning; the mode resolver already maps such
    /// requests to FP16 on that rule, so this is the backstop for a caller
    /// that injects a table directly.
    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        let model_type = self.model_type();
        let opted_in = self.quantized_latent.get();
        let kept: Vec<KVCacheMode> = modes
            .into_iter()
            .map(|mode| {
                let refusal = if opted_in && mode.latent_affine_bits().is_some() {
                    None
                } else {
                    mlxcel_core::mla::latent_pair_supports_mode(model_type, mode).err()
                };
                match refusal {
                    None => mode,
                    Some(reason) => {
                        if !self.kv_mode_warned.replace(true) {
                            tracing::warn!(
                                %mode,
                                "{model_type}: {reason}; layers requesting this KV cache mode \
                                 use fp16"
                            );
                        }
                        KVCacheMode::Fp16
                    }
                }
            })
            .collect();
        self.kv_cache_layer_modes.set(kept);
        self.reset_runtime_state();
    }

    /// The modes the model-owned caches are built in, FP16 for every layer
    /// until a table is injected.
    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        Some(
            (0..self.num_layers)
                .map(|i| self.kv_cache_layer_modes.mode_for_layer(i))
                .collect(),
        )
    }
}

// Weight sanitization.
pub fn sanitize_weights(
    mut weights: WeightMap,
    args: &LongcatFlashNgramConfig,
) -> Result<WeightMap, String> {
    // Stack MoE expert weights into SwitchGLU format
    for l in 0..args.num_layers {
        let prefix = format!("model.layers.{l}");
        for (old_name, new_name) in [
            ("gate_proj", "gate_proj"),
            ("down_proj", "down_proj"),
            ("up_proj", "up_proj"),
        ] {
            for weight_kind in ["weight", "scales", "biases"] {
                let first_key = format!("{prefix}.mlp.experts.0.{old_name}.{weight_kind}");
                if weights.contains_key(&first_key) {
                    let mut to_join = Vec::new();
                    for e in 0..args.n_routed_experts {
                        let key = format!("{prefix}.mlp.experts.{e}.{old_name}.{weight_kind}");
                        if let Some(w) = weights.remove(&key) {
                            to_join.push(w);
                        }
                    }
                    if !to_join.is_empty() {
                        let stacked = stack_arrays(&to_join, 0);
                        let new_key = format!("{prefix}.mlp.switch_mlp.{new_name}.{weight_kind}");
                        weights.insert(new_key, stacked);
                    }
                }
            }
        }
    }

    // Decompose kv_b_proj into embed_q and unembed_out for MLA
    let num_heads = args.num_attention_heads as i32;
    let head_dim = (args.qk_nope_head_dim + args.v_head_dim) as i32;
    let qk_nope_head_dim = args.qk_nope_head_dim as i32;

    for l in 0..args.num_layers {
        for i in 0..2 {
            let prefix = format!("model.layers.{l}.self_attn.{i}");
            let kv_b_key = format!("{prefix}.kv_b_proj.weight");
            let embed_q_key = format!("{prefix}.embed_q.weight");

            // Skip if already decomposed or no kv_b_proj
            if weights.contains_key(&embed_q_key) || !weights.contains_key(&kv_b_key) {
                continue;
            }

            // Check if quantized
            let scales_key = format!("{prefix}.kv_b_proj.scales");
            let is_quantized = weights.contains_key(&scales_key);

            let w = weights.remove(&kv_b_key).unwrap();

            let w_full = if is_quantized {
                let s = weights
                    .remove(&format!("{prefix}.kv_b_proj.scales"))
                    .unwrap();
                // `is_quantized` gates on `.scales` alone, and the block-float
                // modes (mxfp4 / nvfp4 / mxfp8) ship scales with no zero
                // points, so a block-float export satisfies that gate and
                // arrives here carrying no `.biases` plane. The `.unwrap()`
                // this replaces turned that into a panic during sanitization,
                // which in the server takes the process down rather than
                // rejecting one model load (issue #1026). `dequantize` below is
                // hardcoded `"affine"` and so could not decompose such a plane
                // in any case; what this buys is a load error naming the key
                // that is missing.
                let b_key = format!("{prefix}.kv_b_proj.biases");
                let b = weights.remove(&b_key).ok_or_else(|| {
                    format!(
                        "layer {l}: kv_b_proj has scales but no biases at key `{b_key}`; \
                         the checkpoint may be corrupted or only partially converted"
                    )
                })?;
                // Solve the packed pair from the shapes and bound it before it
                // reaches `dequantize`. The shared helper also checks each
                // divisor before dividing: `kv_lora_rank` is a config field and
                // the scales axis is checkpoint data, so the naive form panics
                // on a zero divisor and overflows i32 on a large packed axis,
                // both before the bound could fire. This is why the function is
                // fallible (issue #958).
                let (inferred_gs, inferred_bits) =
                    mlxcel_core::layers::infer_mla_quantization_params(
                        &mlxcel_core::array_shape(&w),
                        &mlxcel_core::array_shape(&s),
                        args.kv_lora_rank as i32,
                        &format!("{prefix}.kv_b_proj"),
                    )?;

                unsafe {
                    mlxcel_core::dequantize(
                        &w,
                        &s,
                        &*b as *const _,
                        inferred_gs,
                        inferred_bits,
                        "affine",
                    )
                }
            } else {
                mlxcel_core::copy(&w)
            };

            // Reshape: [num_heads * head_dim, kv_lora_rank] → [num_heads, head_dim, kv_lora_rank]
            let w_3d = mlxcel_core::reshape(&w_full, &[num_heads, head_dim, -1]);

            // Split: wk = [:, :qk_nope_head_dim, :], wv = [:, qk_nope_head_dim:, :]
            let wk = slice_axis(&w_3d, 1, 0, qk_nope_head_dim);
            let wv = slice_axis(&w_3d, 1, qk_nope_head_dim, -1);

            // embed_q: wk.swapaxes(-1, -2) = [num_heads, kv_lora_rank, qk_nope_head_dim]
            let wk = mlxcel_core::transpose_axes(&wk, &[0, 2, 1]);
            let wk = mlxcel_core::copy(&wk);
            let wv = mlxcel_core::copy(&wv);

            weights.insert(format!("{prefix}.embed_q.weight"), wk);
            weights.insert(format!("{prefix}.unembed_out.weight"), wv);
        }
    }

    // Remove MTP (multi-token prediction) weights
    weights.retain(|k, _| !k.starts_with("model.mtp"));

    // For ngram model: rename embed_tokens → ngram_embeddings.word_embeddings
    if args.is_ngram_model() {
        for part in ["weight", "scales", "biases"] {
            if let Some(w) = weights.remove(&format!("model.embed_tokens.{part}")) {
                weights.insert(format!("model.ngram_embeddings.word_embeddings.{part}"), w);
            }
        }
    }

    Ok(weights)
}

#[cfg(test)]
#[path = "longcat_flash_ngram_engine_tests.rs"]
mod engine_tests;

#[cfg(test)]
#[path = "longcat_flash_ngram_latent_tests.rs"]
mod latent_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal config that exercises only the `kv_b_proj` MLA decomposition:
    /// `n_routed_experts` is set but never triggers the MoE stacking branch,
    /// which is gated on the presence of `mlp.experts.0.*` keys that this
    /// fixture never inserts.
    fn mla_config(kv_lora_rank: usize) -> LongcatFlashNgramConfig {
        let json = format!(
            r#"{{
            "model_type": "longcat_flash",
            "hidden_size": 8,
            "ffn_hidden_size": 8,
            "moe_topk": 1,
            "expert_ffn_hidden_size": 8,
            "n_routed_experts": 1,
            "zero_expert_num": 0,
            "num_layers": 1,
            "vocab_size": 16,
            "max_position_embeddings": 64,
            "num_attention_heads": 2,
            "kv_lora_rank": {kv_lora_rank},
            "q_lora_rank": 8,
            "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 4,
            "v_head_dim": 4,
            "routed_scaling_factor": 1.0,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0
        }}"#
        );
        serde_json::from_str(&json).expect("parse tiny longcat_flash config")
    }

    /// An affine 4-bit `kv_b_proj` on sub-attention 0 of every layer in `args`.
    ///
    /// Honest geometry at `mla_config(16)`: packed_in * 32 == bits *
    /// num_groups * group_size (2 * 32 == 4 * 1 * 16), with
    /// num_attention_heads 2 and qk_nope_head_dim = v_head_dim = 4, so rows =
    /// 2 * 8 = 16.
    ///
    /// The packed plane is UINT32, not a float dtype: `dequantize` rejects any
    /// other packed dtype by throwing, and that throw is the uncatchable abort
    /// these guards exist to keep out of the forward path. A float fixture
    /// here takes the test binary down on the positive control.
    fn affine_kv_b_proj_weights(args: &LongcatFlashNgramConfig) -> WeightMap {
        let heads = args.num_attention_heads as i32;
        let head_dim = (args.qk_nope_head_dim + args.v_head_dim) as i32;
        let rows = heads * head_dim;
        let mut weights = WeightMap::new();
        for l in 0..args.num_layers {
            let prefix = format!("model.layers.{l}.self_attn.0.kv_b_proj");
            weights.insert(
                format!("{prefix}.weight"),
                mlxcel_core::zeros(&[rows, 2], mlxcel_core::dtype::UINT32),
            );
            weights.insert(
                format!("{prefix}.scales"),
                mlxcel_core::zeros(&[rows, 1], mlxcel_core::dtype::FLOAT32),
            );
            weights.insert(
                format!("{prefix}.biases"),
                mlxcel_core::zeros(&[rows, 1], mlxcel_core::dtype::FLOAT32),
            );
        }
        weights
    }

    /// The same plane an mxfp4 / nvfp4 / mxfp8 export ships: `.scales`
    /// present, `.biases` absent, because the block-float modes carry no zero
    /// points.
    fn block_float_kv_b_proj_weights(args: &LongcatFlashNgramConfig) -> WeightMap {
        let mut weights = affine_kv_b_proj_weights(args);
        for l in 0..args.num_layers {
            weights.remove(&format!("model.layers.{l}.self_attn.0.kv_b_proj.biases"));
        }
        weights
    }

    /// The MLA `kv_b_proj` pair `sanitize_weights` decomposes is solved from
    /// `kv_lora_rank` and two tensor axes rather than declared, and every
    /// input is untrusted: `kv_lora_rank` comes from `config.json` and the
    /// axes from the checkpoint. Before issue #958 the naive arithmetic
    /// divided by both without checking them, so a `kv_lora_rank` of 0
    /// panicked on integer division and a solved pair outside anything MLX
    /// can describe reached `dequantize`, which crosses the cxx bridge as
    /// `UniquePtr<MlxArray>` rather than `Result` and therefore aborts during
    /// weight sanitization rather than failing the load.
    ///
    /// `sanitize_weights` decomposes both dual sub-attentions (`self_attn.0`
    /// and `self_attn.1`) per layer; the fixtures above only build
    /// `self_attn.0`, and `self_attn.1` is skipped by the loader's own "no
    /// `kv_b_proj` present" check, which is not what this test targets.
    #[test]
    fn sanitize_rejects_a_kv_lora_rank_no_packing_can_describe() {
        // Positive control first, so a guard that rejected every quantized
        // kv_b_proj could not pass this test.
        let honest = mla_config(16);
        let sanitized = sanitize_weights(affine_kv_b_proj_weights(&honest), &honest)
            .expect("an honest quantized kv_b_proj must still sanitize");
        assert!(sanitized.contains_key("model.layers.0.self_attn.0.embed_q.weight"));

        // A zero `kv_lora_rank` is Rust integer division by zero on the very
        // first solve, so it has to be refused before the division rather
        // than after.
        let zero_rank = mla_config(0);
        let err = sanitize_weights(affine_kv_b_proj_weights(&zero_rank), &zero_rank)
            .err()
            .unwrap_or_else(|| panic!("kv_lora_rank 0 must be refused, not divided by"));
        assert!(err.contains("kv_lora_rank"), "unhelpful error: {err}");

        // A large `kv_lora_rank` truncates the solved bit width to 0, which
        // is the divisor MLX would then divide by.
        let wide_rank = mla_config(4096);
        let err = sanitize_weights(affine_kv_b_proj_weights(&wide_rank), &wide_rank)
            .err()
            .unwrap_or_else(|| panic!("a solved bit width of 0 must be refused"));
        assert!(err.contains("bits"), "unhelpful error: {err}");

        // A non-quantized kv_b_proj carries no packing, so the pair is never
        // solved and a `kv_lora_rank` that no packing could describe must not gate
        // it. The tensor is built at `wide_rank`'s own width so it still satisfies
        // the separate shape cross-check below the solve, which would otherwise
        // reject this for an unrelated reason.
        let mut float_only = WeightMap::new();
        let rows = wide_rank.num_attention_heads as i32
            * (wide_rank.qk_nope_head_dim + wide_rank.v_head_dim) as i32;
        float_only.insert(
            "model.layers.0.self_attn.0.kv_b_proj.weight".to_string(),
            mlxcel_core::zeros(
                &[rows, wide_rank.kv_lora_rank as i32],
                mlxcel_core::dtype::FLOAT32,
            ),
        );
        sanitize_weights(float_only, &wide_rank)
            .expect("a float kv_b_proj must not be gated on quantization params");
    }

    /// Issue #1026: `sanitize_weights` decides `kv_b_proj` is quantized on
    /// `.scales` alone, then used to take `.biases` with `.unwrap()`.
    ///
    /// Affine stores zero points; the block-float modes (mxfp4 / nvfp4 /
    /// mxfp8) do not, which is exactly what `infer_quantization_mode` keys on.
    /// A block-float `kv_b_proj` therefore satisfies the `.scales` gate and
    /// arrives at the `.biases` removal with nothing to take, and the
    /// `.unwrap()` made that a panic during weight sanitization. In the server
    /// that takes the process down rather than rejecting one model load, which
    /// is strictly worse than the load error it should be.
    ///
    /// This family is the one of the four whose prefix carries a sub-attention
    /// index (`self_attn.0` / `self_attn.1`) rather than ending at
    /// `self_attn`, so the key the error names is worth pinning here
    /// specifically: it is the site where a copy of the fix could most easily
    /// blame the wrong tensor.
    ///
    /// The decomposition still dequantizes as `"affine"`, so a genuine
    /// block-float `kv_b_proj` is not supported either way: what this pins is
    /// that it is refused by name instead of unwinding. The assertion is on
    /// the returned `Result` and no forward pass runs, so a regression fails
    /// cleanly here.
    #[test]
    fn sanitize_rejects_scales_with_no_biases() {
        let args = mla_config(16);

        // Positive control first, so a check that rejected every quantized
        // kv_b_proj could not pass this test.
        let sanitized = sanitize_weights(affine_kv_b_proj_weights(&args), &args)
            .expect("a kv_b_proj carrying both planes must still sanitize");
        assert!(sanitized.contains_key("model.layers.0.self_attn.0.embed_q.weight"));

        let err = sanitize_weights(block_float_kv_b_proj_weights(&args), &args)
            .err()
            .expect("scales with no biases must be refused at load, not unwrapped");
        assert!(
            err.contains("model.layers.0.self_attn.0.kv_b_proj.biases"),
            "the error must name the key that is missing, got: {err}"
        );
        assert!(
            err.contains("scales but no biases"),
            "the wording must match the other four MLA sanitizers, got: {err}"
        );
    }
}
