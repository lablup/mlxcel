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

//! DeepSeek-V3 model implementation using mlxcel-core
//!
//! Key features:
//! - MLA (Multi-head Latent Attention) with Q/KV LoRA compression
//! - Complex MoE with sigmoid scoring and group-based expert selection
//! - Shared experts + routed experts
//! - First k layers are dense (no MoE)

use crate::models::mla_rope::MlaRope;
use crate::models::speculative_exactness_diag as diag;
use crate::models::switch_layers::validate_expert_quantization_params;
use crate::models::switch_layers::{
    EXPERT_PLANE_LEAVES, refuse_mismatched_expert_parts, stack_expert_leaf,
};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, route_batched_decode};
use mlxcel_core::layers::{KVCache, MultiLinear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
pub struct DeepSeekV3Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    #[serde(default)]
    pub n_shared_experts: Option<usize>,

    #[serde(default)]
    pub n_routed_experts: Option<usize>,

    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    pub kv_lora_rank: usize,

    // `q_lora_rank` is null for backbones that project the query directly
    // (e.g. the Moonlight-16B text model used by Kimi-VL); a numeric value
    // selects the LoRA-style q_a_proj -> q_a_layernorm -> q_b_proj chain
    // (genuine DeepSeek-V3). Accept both null and an absent key.
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,

    #[serde(default = "default_topk_method")]
    pub topk_method: String,

    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,

    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    #[serde(default = "default_n_group")]
    pub n_group: usize,

    #[serde(default = "default_topk_group")]
    pub topk_group: usize,

    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,

    #[serde(default)]
    pub first_k_dense_replace: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

mlxcel_core::impl_rope_lifting_deserialize!(DeepSeekV3Config);

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "deepseek_v3".to_string()
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_topk_method() -> String {
    "noaux_tc".to_string()
}
fn default_scoring_func() -> String {
    "sigmoid".to_string()
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_n_group() -> usize {
    1
}
fn default_topk_group() -> usize {
    1
}
fn default_num_experts_per_tok() -> usize {
    1
}
fn default_moe_layer_freq() -> usize {
    1
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl DeepSeekV3Config {
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
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

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.n_routed_experts.is_some()
            && layer_idx >= self.first_k_dense_replace
            && layer_idx.is_multiple_of(self.moe_layer_freq)
    }

    pub fn get_attention_scale(&self) -> f32 {
        let q_head_dim = self.q_head_dim() as f32;
        let mut scale = q_head_dim.powf(-0.5);

        // Adjust scale for mscale_all_dim
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
}

// DeepSeek-V3 MLA (Multi-head Latent Attention).
pub struct DeepSeekV3Attention {
    // Q projection: either a direct `q_proj` (when `q_lora_rank` is null) or the
    // LoRA-style chain q_a_proj -> q_a_layernorm -> q_b_proj. Exactly one form is
    // populated by `from_weights`.
    pub q_proj: Option<UnifiedLinear>,
    pub q_a_proj: Option<UnifiedLinear>,
    pub q_a_layernorm: Option<RMSNorm>,
    pub q_b_proj: Option<UnifiedLinear>,

    // KV projection with compression
    pub kv_a_proj_with_mqa: UnifiedLinear,
    pub kv_a_layernorm: RMSNorm,

    // MLA: embed_q and unembed_out replace kv_b_proj
    // embed_q: projects Q_nope into latent space (decode) or latent to K (prefill)
    // unembed_out: projects latent to V (prefill) or attention output back (decode)
    pub embed_q: MultiLinear,
    pub unembed_out: MultiLinear,

    pub o_proj: UnifiedLinear,

    pub num_heads: i32,
    pub q_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub qk_nope_head_dim: i32,
    pub v_head_dim: i32,
    pub kv_lora_rank: i32,
    pub scale: f32,
    /// Rotate interleaved (adjacent) pairs rather than half-split pairs.
    ///
    /// True for every DeepSeek MLA backbone, which is why `from_weights` sets
    /// it unconditionally. Youtu exposes the choice as a config key
    /// (`rope_interleave`), so its decoder overrides this through
    /// [`DeepSeekV3Attention::with_rope_traditional`] (issue #1371).
    pub rope_traditional: bool,
    /// The `q_pe` / `k_pe` rotation: the checkpoint's `rope_scaling` table
    /// (YaRN on every shipped DeepSeek-V3 / Kimi-K2 checkpoint) with its
    /// magnitude multiplier, or plain `fast_rope` at `rope_theta` when the
    /// config has no block (#3295). Youtu passes a plain one through
    /// [`DeepSeekV3Attention::from_weights_with_rope`].
    pub(crate) rope: MlaRope,
}

impl DeepSeekV3Attention {
    /// Replace the rotation after construction.
    ///
    /// Used by: tests (production callers pass the rope to
    /// [`Self::from_weights_with_rope`]).
    #[cfg(test)]
    pub(crate) fn with_rope(mut self, rope: MlaRope) -> Self {
        self.rope = rope;
        self
    }

    /// Resolve this config's RoPE once, for every layer to duplicate.
    ///
    /// No training-context fallback, matching upstream `deepseek_v3.py`:
    /// `YarnRoPE` defaults a missing `original_max_position_embeddings` to
    /// 4096, not to `max_position_embeddings`.
    pub(crate) fn resolve_rope(args: &DeepSeekV3Config) -> MlaRope {
        MlaRope::from_block(
            args.rope_scaling.as_ref(),
            args.qk_rope_head_dim,
            args.rope_theta,
            None,
            &args.model_type,
        )
    }

    /// RoPE for the single-sequence path.
    fn rope(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        self.rope.apply(x, self.rope_traditional, offset)
    }

    /// RoPE for the batched decode path, one offset per row.
    fn rope_batched(&self, x: &MlxArray, offsets: &[i32]) -> UniquePtr<MlxArray> {
        self.rope.apply_batched(x, self.rope_traditional, offsets)
    }

    /// Override the RoPE pair layout after construction.
    ///
    /// Used by: `models::youtu_vl_lm::YoutuDecoderLayer`.
    pub fn with_rope_traditional(mut self, traditional: bool) -> Self {
        self.rope_traditional = traditional;
        self
    }

    /// The layer's MLA shape, for the latent cache view.
    fn geometry(&self) -> mlxcel_core::mla::MlaGeometry {
        crate::models::mla_batched::mla_geometry(
            self.num_heads,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
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

        // Compute Q: direct projection when `q_lora_rank` is null, otherwise the
        // LoRA-style chain q_a_proj -> q_a_layernorm -> q_b_proj. `from_weights`
        // guarantees exactly one form is populated.
        // The projections go through `row_linear`, which is the layer's own
        // forward unless the model entered a row-invariant scope (#3418).
        let q = if let Some(ref q_proj) = self.q_proj {
            row_linear(q_proj, x)
        } else {
            let q_a = row_linear(self.q_a_proj.as_ref().unwrap(), x);
            let q_a_norm = self.q_a_layernorm.as_ref().unwrap().forward(&q_a);
            row_linear(self.q_b_proj.as_ref().unwrap(), &q_a_norm)
        };

        // Reshape Q to [batch, seq, heads, head_dim] then transpose
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.q_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Split Q into nope and pe parts
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim, -1);

        // Compute KV with compression
        let compressed_kv = row_linear(&self.kv_a_proj_with_mqa, x);

        // Split into compressed and k_pe
        let compressed = slice_axis(&compressed_kv, -1, 0, self.kv_lora_rank);
        let k_pe = slice_axis(&compressed_kv, -1, self.kv_lora_rank, -1);

        // Reshape k_pe to [batch, 1, seq, rope_dim]
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        // kv_latent = layernorm(compressed)
        let kv_latent = self.kv_a_layernorm.forward(&compressed);

        // Apply RoPE to pe parts. Capture the pre-update cache state: `offset`
        // is the monotonic rope position, `live_before` the number of K rows
        // already in the cache (mask columns for the causal fallback below).
        let offset = cache.offset;
        let live_before = cache.live_len();
        let q_pe = self.rope(&q_pe, offset);
        let k_pe = self.rope(&k_pe, offset);

        // Expand kv_latent for caching: [B, L, kv_lora_rank] → [B, 1, L, kv_lora_rank]
        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);

        // Cache stores (kv_latent, k_pe) for memory efficiency. An `int8`
        // cache stores the latent as the affine group-64 triple and hands back
        // a quantized window; the fresh chunk is quantized before this step
        // reads it, so a prefill reads what later decode steps read (#3160).
        let window = crate::models::mla_batched::store_latent_step(
            cache,
            self.geometry(),
            kv_latent,
            k_pe,
            "DeepSeekV3Attention",
            true,
        );

        // One query is its own path: the fused kernel or the composed decode,
        // always absorbed, over a dense or a quantized window.
        if l == 1 {
            let output = self.single_query_attention(&q_nope, &q_pe, window, mask);
            return self.project_attention_output(&output, b, l);
        }

        // A verify block in the row-invariant scope (issue #3418) runs every
        // row as the single query its decode step would run, over the keys
        // that step would see. The block
        // forms below pick absorbed or materialized SDPA by a cost model, and
        // neither equals the fused single-query kernel bit for bit, so a
        // token-only verify round (prompt lookup, a full-model drafter) read
        // different logits than plain decode from layer 0 on. The latent of
        // the whole block is already stored, so row `r` reads a prefix of it.
        // The rows count as one MLA call in the path stats (#3421).
        if b == 1 && mask.is_none() && mlxcel_core::row_invariant::in_verify_block() {
            let total = window.seq_len();
            let rows: Vec<UniquePtr<MlxArray>> = mlxcel_core::mla::stats::as_one_call(|| {
                (0..l)
                    .map(|r| {
                        self.single_query_attention(
                            &slice_axis(&q_nope, 2, r, r + 1),
                            &slice_axis(&q_pe, 2, r, r + 1),
                            super::mla_batched::latent_window_prefix(&window, total - l + r + 1),
                            None,
                        )
                    })
                    .collect()
            });
            let refs: Vec<&MlxArray> = rows.iter().map(|row| &**row).collect();
            let output = mlxcel_core::concatenate_many(&refs, 2);
            return self.project_attention_output(&output, b, l);
        }

        // Cost-model gate (issue #2408): a multi-token step behind a long
        // enough cache can be cheaper absorbed than up-projecting the whole
        // window. A cold prefill stays on the materializing side.
        let absorbed = mlxcel_core::mla::use_absorbed_mla(
            l as usize,
            cache.seq_len() as usize,
            self.kv_lora_rank as usize,
            self.qk_nope_head_dim as usize,
            self.v_head_dim as usize,
        );
        mlxcel_core::mla::stats::record(if absorbed {
            mlxcel_core::mla::MlaDecodePath::AbsorbedPrefill
        } else {
            mlxcel_core::mla::MlaDecodePath::Decompressed
        });

        // A multi-token step over a quantized window dequantizes it once. The
        // dequantized window is FP16; cast it to the query dtype so a bf16
        // checkpoint's step is not promoted to float32 (#3177).
        let (kv_latent, k_pe) =
            super::mla_batched::dense_latent_in(window, mlxcel_core::array_dtype(&q_pe));

        // MLA attention: absorbed or materializing, chosen by the cost model
        // above rather than by the query count alone.
        let output = {
            // Multi-token step (issue #3071): fold the rope half into the SDPA
            // head dim, `[q_nope | q_pe] . [k | k_pe]`, instead of building the
            // dense `[B, H, L, S]` rope-score tensor and handing it to SDPA as
            // an additive mask. That tensor grew with the query length times
            // the whole cache in every layer and existed before SDPA ran, so
            // the attention chunk gate could not bound it.
            //
            // Causality is still the model's job. The standard generation
            // paths (CLI text prefill and the VLM embeddings prefill) call the
            // model with `mask == None`, expecting the model to apply its own
            // causal mask, exactly like the reference
            // `create_attention_mask(h, cache[0])` in mlx-lm deepseek_v3.py /
            // mlx-vlm kimi_vl/language.py (issue #525 round 3). A maskless
            // step therefore runs MLX's bottom-right aligned causal mode over
            // the `live_before + l` live keys; a caller mask (padded prefill)
            // is applied as given.
            debug_assert_eq!(
                mlxcel_core::array_shape(&kv_latent)[2],
                live_before + l,
                "causal mode needs the live window to end at this step"
            );
            let folded_mask = match mask {
                Some(m) => mlxcel_core::mla::FoldedMask::Array(m),
                None => mlxcel_core::mla::FoldedMask::Causal,
            };
            if absorbed {
                // Absorbed: Q in latent space, the latent is K and V.
                let q_projected = self.embed_q.forward(&q_nope);
                let output = mlxcel_core::mla::rope_folded_attention(
                    &q_projected,
                    &q_pe,
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
                    &q_nope,
                    &q_pe,
                    &k,
                    &k_pe,
                    &v,
                    self.scale,
                    folded_mask,
                )
            }
        };

        self.project_attention_output(&output, b, l)
    }

    /// One query `[B, H, 1, ..]` against a latent `window`, absorbed: the
    /// output `[B, H, 1, v_head_dim]` after `unembed_out`.
    ///
    /// A quantized window goes to `quantized_absorbed_attention`, which runs
    /// the fused kernel's quantized variant where it applies (Metal, unmasked,
    /// up to `FUSED_DECODE_QUANTIZED_MAX_KEYS` keys, #3156; records
    /// `AbsorbedFused`) and the #3155 composed decode on the packed rows
    /// otherwise (records `AbsorbedComposed`). A dense window takes the fused
    /// kernel when unmasked (#3121) and the composed form otherwise.
    ///
    /// Used by: [`Self::forward`], for a decode step and for every row of a
    /// verify block.
    fn single_query_attention(
        &self,
        q_nope: &MlxArray,
        q_pe: &MlxArray,
        window: mlxcel_core::mla::LatentWindow,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (kv_latent, k_pe) = match window {
            mlxcel_core::mla::LatentWindow::Quantized {
                ckv_q,
                scales,
                biases,
                kpe,
                bits,
                group_size,
            } => {
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
                return self.unembed_out.forward(&output);
            }
            window => super::mla_batched::dense_latent_in(window, mlxcel_core::array_dtype(q_pe)),
        };

        // Unmasked single-token decode on the fused kernel (#3121): both
        // score halves, the softmax and the latent sum in one pass plus a
        // merge. `None` keeps the composed form below.
        if mask.is_none() {
            let q_projected = self.embed_q.forward(q_nope);
            if let Some(output) = mlxcel_core::mla::fused_absorbed_decode(
                &q_projected,
                q_pe,
                &kv_latent,
                &k_pe,
                self.scale,
            ) {
                return self.unembed_out.forward(&output);
            }
        }
        mlxcel_core::mla::stats::record(mlxcel_core::mla::MlaDecodePath::AbsorbedComposed);
        // The composed form keeps the rope term as an additive mask: its
        // score row is one `[B, H, 1, S]` slice.
        // pe_scores = (q_pe * scale) @ k_pe.T
        let scale_scalar = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(q_pe));
        let q_pe_scaled = mlxcel_core::multiply(q_pe, &scale_scalar);
        let k_pe_t = mlxcel_core::transpose_axes(&k_pe, &[0, 1, 3, 2]);
        let pe_scores = mlxcel_core::matmul(&q_pe_scaled, &k_pe_t);
        let pe_scores = match mask {
            // mask has 0 for attend, -inf for blocked
            Some(m) => mlxcel_core::add(&pe_scores, m),
            None => pe_scores,
        };
        let pe_mask_ptr = &*pe_scores as *const MlxArray;
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
        self.unembed_out.forward(&output)
    }

    /// `[B, H, L, v_head_dim]` attention output to `[B, L, hidden]`.
    fn project_attention_output(&self, output: &MlxArray, b: i32, l: i32) -> UniquePtr<MlxArray> {
        let output = mlxcel_core::transpose_axes(output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, self.num_heads * self.v_head_dim]);
        row_linear(&self.o_proj, &output)
    }

    /// Native multi-sequence single-token decode. Projections stay batched;
    /// only each sequence-owned latent cache and ragged softmax run per row.
    pub fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        assert_eq!(
            shape[1], 1,
            "DeepSeekV3 MLA batched decode requires one token"
        );
        assert_eq!(
            batch as usize,
            caches.len(),
            "DeepSeekV3 MLA cache batch mismatch"
        );

        let q = if let Some(ref q_proj) = self.q_proj {
            q_proj.forward(x)
        } else {
            let q_a = self.q_a_proj.as_ref().unwrap().forward(x);
            let q_a = self.q_a_layernorm.as_ref().unwrap().forward(&q_a);
            self.q_b_proj.as_ref().unwrap().forward(&q_a)
        };
        let q = mlxcel_core::reshape(&q, &[batch, 1, self.num_heads, self.q_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim, -1);

        let compressed = self.kv_a_proj_with_mqa.forward(x);
        let kv_latent = slice_axis(&compressed, -1, 0, self.kv_lora_rank);
        let k_pe = slice_axis(&compressed, -1, self.kv_lora_rank, -1);
        let kv_latent = self.kv_a_layernorm.forward(&kv_latent);
        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);
        let k_pe = mlxcel_core::reshape(&k_pe, &[batch, 1, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);
        let offsets: Vec<i32> = caches.iter().map(|cache| cache.offset).collect();
        let q_pe = self.rope_batched(&q_pe, &offsets);
        let k_pe = self.rope_batched(&k_pe, &offsets);
        let q_latent = self.embed_q.forward(&q_nope);
        let output = crate::models::mla_batched::mla_latent_decode_rows(
            "DeepSeekV3Attention",
            true,
            self.geometry(),
            &q_latent,
            &q_pe,
            &kv_latent,
            &k_pe,
            caches,
            self.scale,
            None,
        );
        let output = self.unembed_out.forward(&output);
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[batch, 1, self.num_heads * self.v_head_dim]);
        self.o_proj.forward(&output)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        Self::from_weights_with_rope(weights, args, prefix, Self::resolve_rope(args))
    }

    /// [`Self::from_weights`] with a rope the caller resolved (once per
    /// model, see [`Self::resolve_rope`]).
    pub(crate) fn from_weights_with_rope(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
        rope: MlaRope,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let q_head_dim = args.q_head_dim() as i32;

        // Q projection: a null `q_lora_rank` selects the direct `q_proj` weight
        // (Moonlight-16B / Kimi-VL backbone); a numeric value selects the
        // LoRA-style q_a_proj -> q_a_layernorm -> q_b_proj chain (DeepSeek-V3).
        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = if args.q_lora_rank.is_none() {
            (
                Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.q_proj", prefix),
                    group_size,
                    bits,
                )?),
                None,
                None,
                None,
            )
        } else {
            (
                None,
                Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.q_a_proj", prefix),
                    group_size,
                    bits,
                )?),
                Some(RMSNorm::new(
                    get_weight_copy(weights, &format!("{}.q_a_layernorm.weight", prefix))?,
                    1e-6,
                )),
                Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.q_b_proj", prefix),
                    group_size,
                    bits,
                )?),
            )
        };
        let kv_a_proj_with_mqa = UnifiedLinear::from_weights(
            weights,
            &format!("{}.kv_a_proj_with_mqa", prefix),
            group_size,
            bits,
        )?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        // Load embed_q and unembed_out (decomposed from kv_b_proj by sanitize_weights)
        let embed_q =
            MultiLinear::from_weights(weights, &format!("{}.embed_q", prefix), group_size, bits)?;
        let unembed_out = MultiLinear::from_weights(
            weights,
            &format!("{}.unembed_out", prefix),
            group_size,
            bits,
        )?;

        let kv_a_norm_weight =
            get_weight_copy(weights, &format!("{}.kv_a_layernorm.weight", prefix))?;
        let kv_a_layernorm = RMSNorm::new(kv_a_norm_weight, 1e-6);

        Ok(Self {
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            embed_q,
            unembed_out,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            q_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim as i32,
            qk_nope_head_dim: args.qk_nope_head_dim as i32,
            v_head_dim: args.v_head_dim as i32,
            kv_lora_rank: args.kv_lora_rank as i32,
            scale: args.get_attention_scale(),
            // Every DeepSeek MLA backbone rotates interleaved pairs. A backbone
            // that exposes the choice overrides this with
            // `with_rope_traditional` after construction.
            rope_traditional: true,
            rope,
        })
    }
}

// Dense MLP.
pub struct DenseMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl DenseMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = row_linear(&self.gate_proj, x);
        let up = row_linear(&self.up_proj, x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        row_linear(&self.down_proj, &activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
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

/// A projection of the decoder: the layer's own forward, or the
/// row-invariant dense product inside the model's scope (issue #3418).
///
/// Used by: this decoder and `models::xing4_0`, which runs the same layers.
pub(crate) fn row_linear(layer: &UnifiedLinear, x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::row_invariant::linear_whole_block(layer, x)
}

// MoE Gate.
pub struct MoEGate {
    pub weight: UniquePtr<MlxArray>,
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub top_k: i32,
    pub n_routed_experts: i32,
    pub routed_scaling_factor: f32,
    pub n_group: i32,
    pub topk_group: i32,
    pub norm_topk_prob: bool,
}

impl MoEGate {
    /// Forward pass returns (expert_indices, expert_scores)
    pub fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // gates = x @ weight.T (row-invariant inside the model's scope, #3418)
        let gates = mlxcel_core::row_invariant::dense(x, &self.weight);

        // Sigmoid scoring in float32, mirroring the reference
        // `mx.sigmoid(gates.astype(mx.float32))` (mlx-lm deepseek_v3.py and
        // mlx-vlm kimi_vl/language.py `group_expert_select`). Routing on f16
        // scores loses ties/ordering precision after the correction-bias add
        // and skews the normalized expert weights on f16 checkpoints.
        let gates = mlxcel_core::astype(&gates, mlxcel_core::dtype::FLOAT32);

        // Sigmoid scoring
        let scores = mlxcel_core::sigmoid(&gates);
        let orig_scores = mlxcel_core::copy(&scores);

        // Add correction bias
        let scores = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Group-based expert masking (zero out non-selected groups).
        // `group_mask_scores` takes `[tokens, num_experts]`; the decoder hands
        // this gate `[B, L, hidden]`, so the scores are flattened around the
        // call, as DeepSeek V3.2's gate does. Unflattened, rank-3 scores were
        // skipped with a warning and every token routed as if the model
        // declared no groups (DeepSeek-V3 / R1: `n_group` 8, `topk_group` 4).
        let scores = if self.n_group > 1 {
            let shape = mlxcel_core::array_shape(&scores);
            let n_experts = shape.last().copied().unwrap_or(1);
            let flat = mlxcel_core::reshape(&scores, &[-1, n_experts]);
            let masked =
                super::switch_layers::group_mask_scores(&flat, self.n_group, self.topk_group);
            mlxcel_core::reshape(&masked, &shape)
        } else {
            scores
        };

        // Get top-k expert indices using argpartition
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, self.top_k - 1, -1);
        let topk_indices = slice_axis(&indices, -1, 0, self.top_k);

        // Get scores from orig_scores
        let topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);

        // Normalize if needed
        let topk_scores = if self.top_k > 1 && self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&topk_scores, -1, true);
            mlxcel_core::divide(&topk_scores, &sum)
        } else {
            topk_scores
        };

        // Scale scores
        let scale = mlxcel_core::full_f32(
            &[1],
            self.routed_scaling_factor,
            mlxcel_core::array_dtype(&topk_scores),
        );
        let topk_scores = mlxcel_core::multiply(&topk_scores, &scale);

        (topk_indices, topk_scores)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let n_routed = args.n_routed_experts.unwrap_or(1) as i32;

        let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
        let e_score_correction_bias =
            get_weight_copy(weights, &format!("{}.e_score_correction_bias", prefix))?;

        Ok(Self {
            weight,
            e_score_correction_bias,
            top_k: args.num_experts_per_tok as i32,
            n_routed_experts: n_routed,
            routed_scaling_factor: args.routed_scaling_factor,
            n_group: args.n_group as i32,
            topk_group: args.topk_group as i32,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

/// Clipped SwiGLU activation for the routed experts:
/// `clip(silu(gate), -100, 100) * up`.
///
/// Mirrors upstream `clipped_silu` in mlx-vlm kimi_vl/language.py, which the
/// reference passes as the SwitchGLU activation "to prevent fp16 from
/// overflowing": Kimi-VL / Moonlight expert activations spike far beyond the
/// f16 range on real checkpoints stored in float16, and an unclipped
/// `silu(gate) * up` overflows to inf, turning downstream norms into NaN and
/// the logits argmax into a constant token 0 (issue #525 round 3). The clip is
/// the identity for `|silu(gate)| <= 100`, so bf16/f32 checkpoints operating
/// in the normal activation range are numerically unchanged.
pub(crate) fn clipped_swiglu(gate: &MlxArray, up: &MlxArray) -> UniquePtr<MlxArray> {
    let silu_gate = mlxcel_core::compiled_silu(gate);
    let dt = mlxcel_core::array_dtype(&silu_gate);
    let lo = mlxcel_core::full_f32(&[1], -100.0, dt);
    let hi = mlxcel_core::full_f32(&[1], 100.0, dt);
    let clipped = mlxcel_core::clip(&silu_gate, &lo, &hi);
    mlxcel_core::multiply(&clipped, up)
}

// Quantized routed experts (three affine `gather_qmm` calls).
pub struct QuantizedSwitchGLU {
    // Expert weights: [num_experts, output_dim, input_dim]
    pub gate_weight: UniquePtr<MlxArray>,
    pub gate_scales: UniquePtr<MlxArray>,
    pub gate_biases: UniquePtr<MlxArray>,
    pub up_weight: UniquePtr<MlxArray>,
    pub up_scales: UniquePtr<MlxArray>,
    pub up_biases: UniquePtr<MlxArray>,
    pub down_weight: UniquePtr<MlxArray>,
    pub down_scales: UniquePtr<MlxArray>,
    pub down_biases: UniquePtr<MlxArray>,
    pub group_size: i32,
    /// Width of the `gate_proj` plane, and of every plane unless a per-path
    /// quantization override stores `up_proj` / `down_proj` differently.
    pub bits: i32,
    /// Width of the `up_proj` plane (equal to `bits` on uniform checkpoints).
    pub up_bits: i32,
    /// Width of the `down_proj` plane (equal to `bits` on uniform checkpoints).
    pub down_bits: i32,
}

impl QuantizedSwitchGLU {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        // `x` is `[..., hidden]` with one or more leading token axes: a flat
        // `[n_tokens, hidden]` stream in the text-only path, or a 3D
        // `[batch, seq, hidden]` tensor in the VLM embeddings path. The two
        // `expand_dims` below insert the gather (`top_k`) and matmul axes for
        // `gather_qmm` regardless of how many leading axes `x` carries.
        let x_exp = mlxcel_core::expand_dims(x, -2);
        let x_exp = mlxcel_core::expand_dims(&x_exp, -3);

        // Get bias pointers (biases field is UniquePtr, need to get raw pointer)
        let gate_bias_ptr = self
            .gate_biases
            .as_ref()
            .map(|b| b as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let up_bias_ptr = self
            .up_biases
            .as_ref()
            .map(|b| b as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let down_bias_ptr = self
            .down_biases
            .as_ref()
            .map(|b| b as *const MlxArray)
            .unwrap_or(std::ptr::null());

        // Gate projection with gather_qmm
        let x_gate = unsafe {
            mlxcel_core::gather_qmm(
                &x_exp,
                &self.gate_weight,
                &self.gate_scales,
                gate_bias_ptr,
                std::ptr::null(), // lhs_indices
                indices as *const _,
                true, // transpose
                self.group_size,
                self.bits,
                false, // sorted_indices
                "affine",
                std::ptr::null(),
            )
        };

        // Up projection
        let x_up = unsafe {
            mlxcel_core::gather_qmm(
                &x_exp,
                &self.up_weight,
                &self.up_scales,
                up_bias_ptr,
                std::ptr::null(),
                indices as *const _,
                true,
                self.group_size,
                self.up_bits,
                false,
                "affine",
                std::ptr::null(),
            )
        };

        // Clipped SwiGLU activation (see `clipped_swiglu`): mirrors the
        // reference kimi_vl SwitchGLU `activation=clipped_silu` fp16 guard.
        let activated = clipped_swiglu(&x_gate, &x_up);

        // Down projection
        // activated is [n_tokens, top_k, 1, intermediate_size]
        let output = unsafe {
            mlxcel_core::gather_qmm(
                &activated,
                &self.down_weight,
                &self.down_scales,
                down_bias_ptr,
                std::ptr::null(),
                indices as *const _,
                true,
                self.group_size,
                self.down_bits,
                false,
                "affine",
                std::ptr::null(),
            )
        };

        // Drop the singleton matmul axis, preserving every leading token axis:
        // `[..., top_k, 1, hidden] -> [..., top_k, hidden]`. A previous
        // `reshape(&[n_tokens, top_k, -1])` here assumed a 2D `[n_tokens, hidden]`
        // input; for a 3D `[batch, seq, hidden]` input it misread `n_tokens` and
        // `top_k` from the leading axes and folded the real `top_k` and `hidden`
        // axes into one (`[batch, seq, top_k * hidden]`). That corrupted tensor
        // then failed to broadcast against `scores[..., None]` in
        // `moe_weighted_sum` (`[batch, seq, top_k*hidden]` vs `[batch, seq, top_k, 1]`).
        mlxcel_core::squeeze_axis(&output, -2)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // This variant is unconditionally quantized (it requires scales and
        // biases for all three projections below) and hands the stored pair to
        // three `gather_qmm` calls without ever reaching
        // `reconcile_quantization_layout`, so the bound lives here (issue #958).
        validate_expert_quantization_params(prefix, group_size, bits)?;

        // This expert path runs affine `gather_qmm` only. A ModelOpt NVFP4
        // plane (no zero points, a stacked `inv_global_scale` multiplier) is
        // refused by name rather than as a missing `.biases` key (issue #2351).
        for proj in SWITCH_GLU_PLANES {
            let plane = format!("{prefix}.{proj}");
            if weights.contains_key(&format!("{plane}.inv_global_scale"))
                || (weights.contains_key(&format!("{plane}.scales"))
                    && !weights.contains_key(&format!("{plane}.biases")))
            {
                return Err(format!(
                    "{plane}: DeepSeek-V3 routed experts load affine-quantized planes only; this \
                     plane is block-float (no zero-point biases, as in a ModelOpt NVFP4 export), \
                     which this family's expert path does not run"
                ));
            }
        }

        // Load stacked expert weights
        let gate_weight = get_weight_copy(weights, &format!("{}.gate_proj.weight", prefix))?;
        let gate_scales = get_weight_copy(weights, &format!("{}.gate_proj.scales", prefix))?;
        let gate_biases = get_weight_copy(weights, &format!("{}.gate_proj.biases", prefix))?;

        let up_weight = get_weight_copy(weights, &format!("{}.up_proj.weight", prefix))?;
        let up_scales = get_weight_copy(weights, &format!("{}.up_proj.scales", prefix))?;
        let up_biases = get_weight_copy(weights, &format!("{}.up_proj.biases", prefix))?;

        let down_weight = get_weight_copy(weights, &format!("{}.down_proj.weight", prefix))?;
        let down_scales = get_weight_copy(weights, &format!("{}.down_proj.scales", prefix))?;
        let down_biases = get_weight_copy(weights, &format!("{}.down_proj.biases", prefix))?;

        // A per-path quantization override (mlx-lm's `class_predicate`, e.g.
        // the OptiQ Xing4.0 export with some expert planes at 8 bits under a
        // 4-bit default, issue #2356) stores a plane at a width the top-level
        // block does not declare. Each plane's width is solved from its shapes
        // at the shared group size; a plane that matches keeps the declared
        // width, so uniform checkpoints load exactly as before.
        let plane_bits = |proj: &str, w: &MlxArray, s: &MlxArray| -> Result<i32, String> {
            let layout = mlxcel_core::layers::reconcile_quantization_layout(
                &mlxcel_core::array_shape(w),
                &mlxcel_core::array_shape(s),
                group_size,
                bits,
                "affine",
            )
            .map_err(|e| format!("{prefix}.{proj}: {e}"))?;
            if layout.group_size != group_size {
                return Err(format!(
                    "{prefix}.{proj}: expert plane group size {} differs from the declared {group_size}",
                    layout.group_size
                ));
            }
            // A plane whose shapes solve to 1 bit would reach `gather_qmm`,
            // which has no 1-bit kernel and aborts at the first routed forward
            // pass; `SwitchLinear` refuses the same layout (issue #1370).
            if layout.bits == 1 {
                return Err(format!(
                    "{prefix}.{proj}: 1-bit quantization is supported for dense linears and \
                     embeddings only; 1-bit MoE expert stacks (gather_qmm) are not supported"
                ));
            }
            validate_expert_quantization_params(
                &format!("{prefix}.{proj}"),
                group_size,
                layout.bits,
            )?;
            Ok(layout.bits)
        };
        let gate_bits = plane_bits("gate_proj", &gate_weight, &gate_scales)?;
        let up_bits = plane_bits("up_proj", &up_weight, &up_scales)?;
        let down_bits = plane_bits("down_proj", &down_weight, &down_scales)?;

        Ok(Self {
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            down_weight,
            down_scales,
            down_biases,
            group_size,
            bits: gate_bits,
            up_bits,
            down_bits,
        })
    }
}

/// Routed experts of one DeepSeek-V3 MoE block (`mlp.switch_mlp`).
///
/// A quantized export stores every expert plane with affine `.scales` and
/// `.biases` and runs three `gather_qmm` calls ([`QuantizedSwitchGLU`]). An
/// unquantized (bf16 / f16 / f32) export stores bare `[E, out, in]` planes,
/// which run `gather_mm` through the shared `SwitchLinear` ([`FloatSwitchGLU`],
/// issue #3319). Both apply [`clipped_swiglu`], which the shared
/// `switch_layers::SwitchGLU` does not offer, so this block is kept local.
/// The variant is decided per block from the presence of `.scales`.
pub enum SwitchGLU {
    Quantized(QuantizedSwitchGLU),
    Float(FloatSwitchGLU),
}

const SWITCH_GLU_PLANES: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

impl SwitchGLU {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized(q) => q.forward(x, indices),
            Self::Float(f) => f.forward(x, indices),
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let (quantized, float): (Vec<&str>, Vec<&str>) = SWITCH_GLU_PLANES
            .into_iter()
            .partition(|proj| weights.contains_key(&format!("{prefix}.{proj}.scales")));
        if quantized.is_empty() {
            return FloatSwitchGLU::from_weights(weights, args, prefix).map(Self::Float);
        }
        if !float.is_empty() {
            return Err(format!(
                "{prefix}: DeepSeek-V3 routed experts mix quantized and float planes ({} \
                 quantized, {} float)",
                quantized.join("/"),
                float.join("/")
            ));
        }
        QuantizedSwitchGLU::from_weights(weights, args, prefix).map(Self::Quantized)
    }
}

/// Unquantized routed experts: three `Regular` `SwitchLinear` planes
/// (`gate_proj` / `up_proj` `[E, inter, hidden]`, `down_proj`
/// `[E, hidden, inter]`) driven through `gather_mm` with the clipped SwiGLU
/// activation (issue #3319).
pub struct FloatSwitchGLU {
    pub gate_proj: crate::models::switch_layers::SwitchLinear,
    pub up_proj: crate::models::switch_layers::SwitchLinear,
    pub down_proj: crate::models::switch_layers::SwitchLinear,
}

impl FloatSwitchGLU {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        // Same gather layout as the quantized path: `[..., hidden]` ->
        // `[..., 1, 1, hidden]` for any number of leading token axes. The
        // gather indices are built once and shared by the three planes.
        let x_exp = mlxcel_core::expand_dims_multi(x, &[-3, -2]);
        let gi = crate::models::switch_layers::prepare_gather_indices(&x_exp, indices);
        let lhs = gi.lhs.as_deref();
        let x_gate = self.gate_proj.forward_indexed(&x_exp, lhs, &gi.rhs, false);
        let x_up = self.up_proj.forward_indexed(&x_exp, lhs, &gi.rhs, false);
        let activated = clipped_swiglu(&x_gate, &x_up);
        // `activated` is `[..., top_k, 1, inter]`, whose batch axes no longer
        // match the lhs built for `x_exp`, so MLX builds its own default.
        let output = self
            .down_proj
            .forward_indexed(&activated, None, &gi.rhs, false);
        mlxcel_core::squeeze_axis(&output, -2)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let load = |proj: &str| -> Result<crate::models::switch_layers::SwitchLinear, String> {
            let plane = format!("{prefix}.{proj}");
            // Zero-point biases with no scales are what a quantized export
            // looks like after its `.scales` were lost; its packed integer
            // weight would otherwise load here and abort in `gather_mm`.
            if weights.contains_key(&format!("{plane}.biases")) {
                return Err(format!(
                    "{plane}: expert plane has quantization `.biases` but no `.scales`; a float \
                     plane carries neither"
                ));
            }
            let linear = crate::models::switch_layers::SwitchLinear::from_weights(
                weights,
                &plane,
                args.group_size(),
                args.bits(),
            )?;
            match &linear {
                crate::models::switch_layers::SwitchLinear::Regular { weight } => {
                    let dt = mlxcel_core::array_dtype(weight);
                    if ![
                        mlxcel_core::dtype::FLOAT16,
                        mlxcel_core::dtype::BFLOAT16,
                        mlxcel_core::dtype::FLOAT32,
                    ]
                    .contains(&dt)
                    {
                        return Err(format!(
                            "{plane}: float expert plane has dtype {dt}; an unquantized plane \
                             must be f16, bf16 or f32 (a packed quantized plane without \
                             `.scales`?)"
                        ));
                    }
                    Ok(linear)
                }
                // Reachable only through `SwitchLinear`'s per-expert stacking
                // fallback (unstacked `experts.{e}.*.scales`), which sanitize
                // normally stacks first.
                _ => Err(format!(
                    "{plane}: expected a float expert plane (no `.scales`), but it loaded as \
                     quantized"
                )),
            }
        };
        Ok(Self {
            gate_proj: load("gate_proj")?,
            up_proj: load("up_proj")?,
            down_proj: load("down_proj")?,
        })
    }
}

// MoE Block.
pub struct MoEBlock {
    pub gate: MoEGate,
    pub switch_mlp: SwitchGLU,
    pub shared_experts: Option<DenseMLP>,
}

impl MoEBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (indices, scores) = self.gate.forward(x);

        // Expert computation. Float experts multiply through `gather_mm`,
        // which, like `matmul`, takes one kernel for a single token and
        // another for several, so a verify block runs them one position at a
        // time (issue #3418). The quantized gather is left whole.
        let shape = mlxcel_core::array_shape(x);
        let per_position = matches!(self.switch_mlp, SwitchGLU::Float(_))
            && shape.len() == 3
            && shape[1] > 1
            && mlxcel_core::row_invariant::in_verify_block();
        let y = if per_position {
            let parts: Vec<UniquePtr<MlxArray>> = (0..shape[1])
                .map(|i| {
                    self.switch_mlp.forward(
                        &slice_axis(x, 1, i, i + 1),
                        &slice_axis(&indices, 1, i, i + 1),
                    )
                })
                .collect();
            let refs: Vec<&MlxArray> = parts.iter().map(|part| &**part).collect();
            mlxcel_core::concatenate_many(&refs, 1)
        } else {
            self.switch_mlp.forward(x, &indices)
        };

        let mut result = crate::models::switch_layers::moe_weighted_sum(
            &y,
            &scores,
            mlxcel_core::array_dtype(x),
        );

        // Add shared experts if present
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(x);
            result = mlxcel_core::add(&result, &shared_out);
        }

        result
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let gate = MoEGate::from_weights(weights, args, &format!("{}.gate", prefix))?;
        let switch_mlp = SwitchGLU::from_weights(weights, args, &format!("{}.switch_mlp", prefix))?;

        let shared_experts = if args.n_shared_experts.is_some() {
            Some(DenseMLP::from_weights(
                weights,
                args,
                &format!("{}.shared_experts", prefix),
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            switch_mlp,
            shared_experts,
        })
    }
}

// MLP Type (Dense or MoE).
pub enum MLPType {
    Dense(DenseMLP),
    MoE(MoEBlock),
}

// Decoder Layer.
pub struct DecoderLayer {
    pub self_attn: DeepSeekV3Attention,
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
        // Attention with residual
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // MLP with residual
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = match &self.mlp {
            MLPType::Dense(mlp) => mlp.forward(&normed),
            MLPType::MoE(moe) => moe.forward(&normed),
        };
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn = self.self_attn.forward_batched_decode(&normed, caches);
        let h = mlxcel_core::add(x, &attn);
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp = match &self.mlp {
            MLPType::Dense(mlp) => mlp.forward(&normed),
            MLPType::MoE(moe) => moe.forward(&normed),
        };
        mlxcel_core::add(&h, &mlp)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let rope = DeepSeekV3Attention::resolve_rope(args);
        Self::from_weights_with_rope(weights, args, layer_idx, &rope)
    }

    /// [`Self::from_weights`] with the model's resolved rope, duplicated into
    /// this layer's attention.
    pub(crate) fn from_weights_with_rope(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        layer_idx: usize,
        rope: &MlaRope,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = DeepSeekV3Attention::from_weights_with_rope(
            weights,
            args,
            &format!("{}.self_attn", prefix),
            rope.duplicate(),
        )?;

        let mlp = if args.is_moe_layer(layer_idx) {
            MLPType::MoE(MoEBlock::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            MLPType::Dense(DenseMLP::from_weights(
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

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_attn_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// DeepSeek-V3 Model.
pub struct DeepSeekV3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
    pub config: DeepSeekV3Config,
}

impl DeepSeekV3Model {
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows(input_ids, None, caches, mask, None)
    }

    /// The single-sequence forward, from token ids or from precomputed
    /// embeddings, projecting every position (`row == None`) or only position
    /// `row` (`[B, 1, vocab]`, the prefill's sampled row, #3070).
    ///
    /// Used by: `forward_impl`, `forward_impl_with_embeddings`,
    /// `last_row_logits_impl`
    fn forward_rows(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match input_embeddings {
            Some(embeds) => mlxcel_core::copy(embeds),
            None => self.embed_tokens.forward(input_ids),
        };

        // Preserve any caller-provided mask (for padded prefill) but let the
        // attention stack use its implicit causal path for standard prefill.
        let mask = mask.map(mlxcel_core::copy);

        // Row-invariant products for this forward when
        // `MLXCEL_MLA_ROW_INVARIANT` is on (issue #3418): a verify block of
        // the engine's token-only round then equals the decode steps it
        // stands for. A masked block is a padded prefill, never a verify.
        let _row_invariant =
            super::mla_batched::row_invariant_scope(&mlxcel_core::array_shape(&h), mask.is_some());

        // Pass through layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask.as_deref());
            // Recorded only under the block-versus-chain diagnostic (#3409).
            diag::capture(i, "mla", "layer output", &h);
        }
        let h = mlxcel_core::generate::select_row(h, row);

        // Final norm and lm_head
        let h = self.norm.forward(&h);
        row_linear(&self.lm_head, &h)
    }

    /// Logits `[B, 1, vocab]` for position `last_pos` only, from token ids or
    /// from precomputed embeddings: the last-row counterpart of
    /// [`Self::forward_impl_with_embeddings`].
    ///
    /// Used by: `LanguageModel::last_row_logits` for `DeepSeekV3Model` and
    /// `vision::kimi_vl::KimiVLModel`.
    pub fn last_row_logits_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows(input_ids, input_embeddings, caches, mask, Some(last_pos))
    }

    /// Text-embedding lookup, exposed so a VLM wrapper can build merged
    /// text/vision embeddings before calling [`Self::forward_impl_with_embeddings`].
    ///
    /// Used by: `vision::kimi_vl::KimiVLModel`.
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward pass that optionally consumes precomputed input embeddings
    /// (the VLM image path) instead of embedding `input_ids`. When
    /// `input_embeddings` is `None` this is identical to [`Self::forward_impl`].
    ///
    /// Used by: `vision::kimi_vl::KimiVLModel`.
    pub fn forward_impl_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows(input_ids, input_embeddings, caches, mask, None)
    }

    pub fn make_caches_impl(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn forward_batched_impl(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
    ) -> UniquePtr<MlxArray> {
        crate::models::mla_batched::validate_mla_decode_batch(
            "DeepSeekV3Model",
            input_ids,
            batch_caches,
            self.layers.len(),
        );
        let mut h = self.embed_tokens.forward(input_ids);
        for layer_idx in 0..self.layers.len() {
            let mut layer_caches: Vec<&mut KVCache> = batch_caches
                .iter_mut()
                .map(|row| &mut row[layer_idx])
                .collect();
            h = self.layers[layer_idx].forward_batched_decode(&h, &mut layer_caches);
        }
        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, DeepSeekV3Config), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: DeepSeekV3Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Sanitize weights (stack expert weights if needed)
        let weights = Self::sanitize_weights(weights, &config)?;

        // Create model
        let model = Self::from_weights(&weights, &config)?;

        Ok((model, config))
    }

    /// Public wrapper over [`Self::sanitize_weights`] for callers outside of
    /// `src/models/deepseek_v3.rs` (e.g. the pipeline-parallel stage executor
    /// that needs to sanitise the full weight map before partial-loading
    /// filters out layers from other stages).
    ///
    /// Used by: DeepSeek V3 pipeline stage executor
    pub fn sanitize_weights_with_args(
        weights: WeightMap,
        config: &DeepSeekV3Config,
    ) -> Result<WeightMap, String> {
        Self::sanitize_weights(weights, config)
    }

    fn sanitize_weights(
        mut weights: WeightMap,
        config: &DeepSeekV3Config,
    ) -> Result<WeightMap, String> {
        // Remap pre-stacked weight names that omit the `.mlp.` segment.
        //
        // Some HuggingFace MLX shards for DeepSeek-V3 store pre-stacked MoE expert
        // weights under keys like `model.layers.{l}.switch_mlp.*` rather than
        // `model.layers.{l}.mlp.switch_mlp.*`. Similarly, the router gate weight may
        // appear as `model.layers.{l}.mlp.gate_weight` instead of the expected
        // `model.layers.{l}.mlp.gate.weight` / `model.layers.{l}.mlp.gate.e_score_correction_bias`.
        //
        // This remapping step normalises these alternative naming conventions before
        // the rest of sanitize_weights runs, so all subsequent logic can assume the
        // canonical key structure.
        {
            let num_layers = config.num_hidden_layers;

            // Collect renames to avoid borrowing `weights` mutably while iterating it.
            let mut renames: Vec<(String, String)> = Vec::new();

            for l in 0..num_layers {
                let prefix = format!("model.layers.{l}");

                // Alias `{prefix}.switch_mlp.*` → `{prefix}.mlp.switch_mlp.*`
                for m in ["gate_proj", "down_proj", "up_proj"] {
                    for k in ["weight", "scales", "biases"] {
                        let src = format!("{prefix}.switch_mlp.{m}.{k}");
                        let dst = format!("{prefix}.mlp.switch_mlp.{m}.{k}");
                        if weights.contains_key(&src) && !weights.contains_key(&dst) {
                            renames.push((src, dst));
                        }
                    }
                }

                // Alias `{prefix}.mlp.gate_weight` → `{prefix}.mlp.gate.weight`
                let gate_weight_src = format!("{prefix}.mlp.gate_weight");
                let gate_weight_dst = format!("{prefix}.mlp.gate.weight");
                if weights.contains_key(&gate_weight_src) && !weights.contains_key(&gate_weight_dst)
                {
                    renames.push((gate_weight_src, gate_weight_dst));
                }

                // Alias `{prefix}.mlp.gate.e_score_correction_bias` if stored at
                // `{prefix}.mlp.e_score_correction_bias` (some checkpoints omit the sub-key)
                let bias_src = format!("{prefix}.mlp.e_score_correction_bias");
                let bias_dst = format!("{prefix}.mlp.gate.e_score_correction_bias");
                if weights.contains_key(&bias_src) && !weights.contains_key(&bias_dst) {
                    renames.push((bias_src, bias_dst));
                }
            }

            for (src, dst) in renames {
                if let Some(v) = weights.remove(&src) {
                    weights.insert(dst, v);
                }
            }
        }

        // Check if weights need stacking (individual experts.N format) or are already stacked (switch_mlp format)
        if let Some(n_routed) = config.n_routed_experts {
            let num_layers = config.num_hidden_layers;

            for l in 0..num_layers {
                let prefix = format!("model.layers.{}", l);

                // Check if this layer has individual expert weights that need stacking
                let first_key = format!("{}.mlp.experts.0.gate_proj.weight", prefix);
                if weights.contains_key(&first_key) {
                    // Stack expert weights from individual experts to [num_experts, ...] tensors.
                    // The shared leaf list includes the ModelOpt NVFP4 scale
                    // sidecar, which lands as `inv_global_scale` (issue #2351).
                    for m in ["gate_proj", "down_proj", "up_proj"] {
                        for k in EXPERT_PLANE_LEAVES {
                            let check_key = format!("{}.mlp.experts.0.{}.{}", prefix, m, k);
                            if weights.contains_key(&check_key) {
                                let expert_key =
                                    |e: usize| format!("{}.mlp.experts.{}.{}.{}", prefix, e, m, k);
                                let parts: Vec<(usize, &MlxArray)> = (0..n_routed)
                                    .filter_map(|e| {
                                        weights.get(&expert_key(e)).map(|w| {
                                            (
                                                e,
                                                w.as_ref()
                                                    .expect("weight map holds no null handles"),
                                            )
                                        })
                                    })
                                    .collect();

                                if !parts.is_empty() {
                                    // `stack` aborts on a shape mismatch, so
                                    // refuse one before it runs (issue #3320).
                                    // The sidecar is concatenated, not stacked.
                                    if k != "global_scale" {
                                        refuse_mismatched_expert_parts(&parts, expert_key)?;
                                    }
                                    let expert_ptrs: Vec<*const MlxArray> =
                                        parts.iter().map(|(_, w)| *w as *const MlxArray).collect();
                                    // Stacked before any expert key is removed,
                                    // and the stack holds its own handles.
                                    let (leaf, stacked) = stack_expert_leaf(k, &expert_ptrs);
                                    let new_key =
                                        format!("{}.mlp.switch_mlp.{}.{}", prefix, m, leaf);
                                    weights.insert(new_key, stacked);

                                    for e in 0..n_routed {
                                        let key =
                                            format!("{}.mlp.experts.{}.{}.{}", prefix, e, m, k);
                                        weights.remove(&key);
                                    }
                                }
                            }
                        }
                    }
                }
                // If switch_mlp weights already exist, they're already stacked - no action needed
            }
        }

        // Decompose kv_b_proj into embed_q and unembed_out (if not already decomposed)
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads as i32;
        let head_dim = (config.qk_nope_head_dim + config.v_head_dim) as i32;
        let qk_nope_head_dim = config.qk_nope_head_dim as i32;

        for l in 0..num_layers {
            let prefix = format!("model.layers.{}.self_attn", l);
            let kv_b_key = format!("{}.kv_b_proj.weight", prefix);
            let embed_q_key = format!("{}.embed_q.weight", prefix);

            // Skip if already decomposed
            if weights.contains_key(&embed_q_key) || !weights.contains_key(&kv_b_key) {
                continue;
            }

            // Check if quantized
            let scales_key = format!("{}.kv_b_proj.scales", prefix);
            let is_quantized = weights.contains_key(&scales_key);

            // Get the kv_b_proj weight
            let w = weights.remove(&kv_b_key).unwrap();

            let w_full = if is_quantized {
                // Dequantize first
                let s = weights
                    .remove(&format!("{}.kv_b_proj.scales", prefix))
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
                let b_key = format!("{}.kv_b_proj.biases", prefix);
                let b = weights.remove(&b_key).ok_or_else(|| {
                    format!(
                        "layer {l}: kv_b_proj has scales but no biases at key `{b_key}`; \
                         the checkpoint may be corrupted or only partially converted"
                    )
                })?;

                // Solve the packed pair from the shapes and bound it before it
                // reaches `dequantize`. The shared helper checks each divisor
                // before dividing: `kv_lora_rank` is a config field and the
                // scales axis is checkpoint data, so the naive form panics on a
                // zero divisor and overflows i32 on a large packed axis, both
                // before the bound could fire (issue #958).
                let (inferred_gs, inferred_bits) =
                    mlxcel_core::layers::infer_mla_quantization_params(
                        &mlxcel_core::array_shape(&w),
                        &mlxcel_core::array_shape(&s),
                        config.kv_lora_rank as i32,
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

            // embed_q weight: wk.swapaxes(-1, -2) = [num_heads, kv_lora_rank, qk_nope_head_dim]
            let wk = mlxcel_core::transpose_axes(&wk, &[0, 2, 1]);
            // Make contiguous
            let wk = mlxcel_core::copy(&wk);
            let wv = mlxcel_core::copy(&wv);

            // Store as non-quantized (dequantized) MultiLinear weights
            weights.insert(format!("{}.embed_q.weight", prefix), wk);
            weights.insert(format!("{}.unembed_out.weight", prefix), wv);
        }

        // Remove the multi-token prediction trailer layer and rotary freqs.
        //
        // Checkpoints that ship the MTP head store it at layer index
        // `num_hidden_layers` (e.g. `model.layers.61` for genuine DeepSeek-V3
        // with num_hidden_layers = 61), one PAST the decoder stack; mlx-lm's
        // sanitize strips exactly that index. Layers `0..num_hidden_layers`
        // are all real decoder layers and must survive: a previous
        // `num_hidden_layers - 1` here silently deleted the final decoder
        // layer of every checkpoint without an MTP trailer (Kimi-VL /
        // Moonlight, issue #525 round 3).
        let mtp_prefix = format!("model.layers.{}.", config.num_hidden_layers);
        let keys_to_remove: Vec<String> = weights
            .keys()
            .filter(|k| k.starts_with(&mtp_prefix) || k.contains("rotary_emb.inv_freq"))
            .cloned()
            .collect();

        for key in keys_to_remove {
            weights.remove(&key);
        }

        Ok(weights)
    }

    pub fn from_weights(weights: &WeightMap, args: &DeepSeekV3Config) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load ALL `num_hidden_layers` decoder layers (mirrors the reference
        // `range(config.num_hidden_layers)`). The multi-token prediction
        // trailer, when present, lives at index `num_hidden_layers` and was
        // already stripped by `sanitize_weights`.
        let num_layers = args.num_hidden_layers;
        let rope = DeepSeekV3Attention::resolve_rope(args);
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = DecoderLayer::from_weights_with_rope(weights, args, i, &rope)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load lm_head
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: args.clone(),
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
impl LanguageModel for DeepSeekV3Model {
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, caches, mask)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<mlxcel_core::cache::SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        // The trait-level embeddings forward is the default, which ignores
        // embeddings: decline so that behavior stays the default's.
        if input_embeddings.is_some() {
            return None;
        }
        Some(self.last_row_logits_impl(input_ids, None, caches, mask, last_pos))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.make_caches_impl()
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        route_batched_decode(self, input_ids, batch_caches, |caches| {
            self.forward_batched_impl(input_ids, caches)
        })
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![1] // DeepSeek-V3 EOS token
    }
}

#[cfg(test)]
#[path = "deepseek_v3_latent_tests.rs"]
mod latent_tests;

#[cfg(test)]
#[path = "deepseek_v3_verify_block_tests.rs"]
mod verify_block_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a minimal DeepSeekV3Config for testing weight sanitization.
    fn test_config_moe() -> DeepSeekV3Config {
        DeepSeekV3Config {
            model_type: "deepseek_v3".to_string(),
            vocab_size: 1,
            hidden_size: 1,
            intermediate_size: 1,
            moe_intermediate_size: 1,
            num_hidden_layers: 3, // 3 real decoder layers (an MTP trailer would sit at index 3)
            num_attention_heads: 16,
            num_key_value_heads: 1,
            n_shared_experts: None,
            n_routed_experts: Some(4),
            routed_scaling_factor: 1.0,
            kv_lora_rank: 32,
            q_lora_rank: Some(1),
            qk_rope_head_dim: 1,
            v_head_dim: 64,
            qk_nope_head_dim: 64,
            topk_method: "greedy".to_string(),
            scoring_func: "sigmoid".to_string(),
            norm_topk_prob: false,
            n_group: 1,
            topk_group: 1,
            num_experts_per_tok: 1,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            max_position_embeddings: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            quantization: None,
        }
    }

    // ---- MoE 3D-input regression (issue #525 round 2) ---------------------
    //
    // The Kimi-VL text backbone is this DeepSeek-V3 model, driven through
    // `forward_impl_with_embeddings` on 3D `[batch, seq, hidden]` hidden
    // states. `SwitchGLU::forward` used to end with
    // `reshape(&output, &[n_tokens, top_k, -1])`, where `n_tokens`/`top_k`
    // were read from `indices.shape[0]`/`[1]`. That is only correct for a 2D
    // `[n_tokens, hidden]` input; for a 3D input it read the batch/seq axes as
    // `n_tokens`/`top_k` and folded the real `top_k` and `hidden` axes into a
    // single `top_k * hidden` axis. The following `moe_weighted_sum` then tried
    // to broadcast `[batch, seq, top_k*hidden]` against `[batch, seq, top_k, 1]`
    // and aborted (the reported `(1,91,12288)` vs `(1,91,6,1)` crash).

    /// Deterministic, varied, finite fill so quantization scales stay non-degenerate.
    pub(crate) fn ramp(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 37 + 11) % 97) as f32 - 48.0) * 0.01)
            .collect()
    }

    fn quantized_triple(
        data: &[f32],
        shape: &[i32],
        group_size: i32,
        bits: i32,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let w = mlxcel_core::from_slice_f32(data, shape);
        (
            mlxcel_core::quantize_weights_w(&w, group_size, bits),
            mlxcel_core::quantize_weights_scales(&w, group_size, bits),
            mlxcel_core::quantize_weights_biases(&w, group_size, bits),
        )
    }

    /// Tiny quantized SwitchGLU: `gate`/`up` are `[E, inter, hidden]`,
    /// `down` is `[E, hidden, inter]` (group_size 64, 4-bit).
    fn tiny_switch_glu(e: i32, hidden: i32, inter: i32) -> SwitchGLU {
        let (gs, bits) = (64i32, 4i32);
        let (gate_weight, gate_scales, gate_biases) = quantized_triple(
            &ramp((e * inter * hidden) as usize),
            &[e, inter, hidden],
            gs,
            bits,
        );
        let (up_weight, up_scales, up_biases) = quantized_triple(
            &ramp((e * inter * hidden) as usize),
            &[e, inter, hidden],
            gs,
            bits,
        );
        let (down_weight, down_scales, down_biases) = quantized_triple(
            &ramp((e * hidden * inter) as usize),
            &[e, hidden, inter],
            gs,
            bits,
        );
        SwitchGLU::Quantized(QuantizedSwitchGLU {
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            down_weight,
            down_scales,
            down_biases,
            group_size: gs,
            bits,
            up_bits: bits,
            down_bits: bits,
        })
    }

    fn tiny_gate(e: i32, hidden: i32, top_k: i32) -> MoEGate {
        MoEGate {
            weight: mlxcel_core::from_slice_f32(&ramp((e * hidden) as usize), &[e, hidden]),
            e_score_correction_bias: mlxcel_core::from_slice_f32(&vec![0.0f32; e as usize], &[e]),
            top_k,
            n_routed_experts: e,
            routed_scaling_factor: 1.0,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
        }
    }

    /// Grouped routing on the `[B, L, hidden]` input the decoder passes. Two
    /// groups of two experts with one kept: router logits 2.0, 0.0, -2.0, 1.5
    /// give group `{0, 1}` the larger top-two sum, so both picks come from it,
    /// where ungrouped top-2 would take `{0, 3}`. A rank-2 input must route the
    /// same way.
    #[test]
    fn moe_gate_applies_grouped_routing_to_rank_3_scores() {
        let _runtime = crate::initialize_runtime();
        let (e, hidden) = (4i32, 6i32);
        let weight: Vec<f32> = [2.0f32, 0.0, -2.0, 1.5]
            .iter()
            .flat_map(|logit| std::iter::repeat_n(logit / hidden as f32, hidden as usize))
            .collect();
        let gate = MoEGate {
            weight: mlxcel_core::from_slice_f32(&weight, &[e, hidden]),
            n_group: 2,
            topk_group: 1,
            ..tiny_gate(e, hidden, 2)
        };
        let picked_pairs = |x: &MlxArray| -> Vec<[i32; 2]> {
            let (indices, _) = gate.forward(x);
            let picked = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
                &indices,
                mlxcel_core::dtype::FLOAT32,
            ));
            picked
                .chunks(2)
                .map(|t| {
                    let mut pair = [t[0] as i32, t[1] as i32];
                    pair.sort_unstable();
                    pair
                })
                .collect()
        };
        let x3 = mlxcel_core::ones(&[1, 3, hidden], mlxcel_core::dtype::FLOAT32);
        assert_eq!(picked_pairs(&x3), vec![[0, 1]; 3], "rank-3 input");
        let x2 = mlxcel_core::ones(&[3, hidden], mlxcel_core::dtype::FLOAT32);
        assert_eq!(picked_pairs(&x2), vec![[0, 1]; 3], "rank-2 input");
    }

    #[test]
    fn switch_glu_forward_preserves_rank_for_2d_and_3d() {
        let (e, hidden, inter, top_k) = (4i32, 64i32, 64i32, 2i32);
        let glu = tiny_switch_glu(e, hidden, inter);
        let gate = tiny_gate(e, hidden, top_k);

        // 2D flat token stream: [n_tokens, hidden] -> [n_tokens, top_k, hidden].
        let x2 = mlxcel_core::from_slice_f32(&ramp((3 * hidden) as usize), &[3, hidden]);
        let (idx2, _) = gate.forward(&x2);
        let y2 = glu.forward(&x2, &idx2);
        mlxcel_core::eval(&y2);
        assert_eq!(mlxcel_core::array_shape(&y2), vec![3, top_k, hidden]);

        // 3D VLM embeddings path: [batch, seq, hidden] -> [batch, seq, top_k, hidden].
        // The pre-fix reshape collapsed this to [batch, seq, top_k*hidden].
        let x3 = mlxcel_core::from_slice_f32(&ramp((3 * hidden) as usize), &[1, 3, hidden]);
        let (idx3, _) = gate.forward(&x3);
        let y3 = glu.forward(&x3, &idx3);
        mlxcel_core::eval(&y3);
        assert_eq!(mlxcel_core::array_shape(&y3), vec![1, 3, top_k, hidden]);
    }

    // ---- Unquantized routed experts (issue #3319) -------------------------
    //
    // A bf16 / f16 / f32 export stacks `mlp.experts.{e}.*` into bare
    // `switch_mlp.{gate,up,down}_proj.weight` planes with no `.scales`. The
    // expert block used to require `.scales` on every plane and failed the
    // load with `Weight not found: ...switch_mlp.gate_proj.scales`.

    /// Stacked float plane `[e, out, inp]` filled with a deterministic ramp,
    /// returned both as the weight-map tensor and as the host values.
    fn float_plane(
        e: i32,
        out: i32,
        inp: i32,
        scale: f32,
        salt: usize,
    ) -> (Vec<f32>, UniquePtr<MlxArray>) {
        let data: Vec<f32> = (0..(e * out * inp) as usize)
            .map(|i| (((i * 29 + salt * 13 + 7) % 53) as f32 - 26.0) * 0.02 * scale)
            .collect();
        let arr = mlxcel_core::from_slice_f32(&data, &[e, out, inp]);
        (data, arr)
    }

    #[test]
    fn float_switch_glu_matches_per_expert_clipped_swiglu_reference() {
        let _runtime = crate::initialize_runtime();
        let (e, hidden, inter) = (3i32, 4i32, 5i32);
        let (h, n) = (hidden as usize, inter as usize);
        // Expert 2's gate plane is scaled so `silu(gate)` passes 100 for some
        // rows and the clip is exercised, not just the identity range.
        let gate: Vec<f32> = float_plane(e, inter, hidden, 1.0, 1)
            .0
            .iter()
            .enumerate()
            .map(|(i, v)| if i / (n * h) == 2 { v * 400.0 } else { *v })
            .collect();
        let gate_arr = mlxcel_core::from_slice_f32(&gate, &[e, inter, hidden]);
        let (up, up_arr) = float_plane(e, inter, hidden, 1.0, 2);
        let (down, down_arr) = float_plane(e, hidden, inter, 1.0, 3);

        let mut weights = WeightMap::new();
        weights.insert("p.gate_proj.weight".to_string(), gate_arr);
        weights.insert("p.up_proj.weight".to_string(), up_arr);
        weights.insert("p.down_proj.weight".to_string(), down_arr);
        let glu = SwitchGLU::from_weights(&weights, &test_config_moe(), "p")
            .expect("a float expert block loads without `.scales`");
        assert!(matches!(glu, SwitchGLU::Float(_)));

        let (n_tokens, top_k) = (2usize, 2usize);
        let x: Vec<f32> = (0..n_tokens * h).map(|i| (i as f32 - 3.0) * 0.25).collect();
        let picks = [2i32, 0, 1, 2];
        let x_arr = mlxcel_core::from_slice_f32(&x, &[n_tokens as i32, hidden]);
        let idx = mlxcel_core::from_slice_i32(&picks, &[n_tokens as i32, top_k as i32]);
        let y = glu.forward(&x_arr, &idx);
        mlxcel_core::eval(&y);
        assert_eq!(
            mlxcel_core::array_shape(&y),
            vec![n_tokens as i32, top_k as i32, hidden]
        );
        let got = mlxcel_core::utils::array_to_vec_f32(&y);

        let mut clipped_any = false;
        for t in 0..n_tokens {
            for k in 0..top_k {
                let ex = picks[t * top_k + k] as usize;
                let xt = &x[t * h..(t + 1) * h];
                let act: Vec<f64> = (0..n)
                    .map(|j| {
                        let row = |w: &[f32]| -> f64 {
                            (0..h)
                                .map(|c| w[(ex * n + j) * h + c] as f64 * xt[c] as f64)
                                .sum()
                        };
                        let g = row(&gate);
                        let silu = g / (1.0 + (-g).exp());
                        clipped_any |= silu > 100.0;
                        silu.clamp(-100.0, 100.0) * row(&up)
                    })
                    .collect();
                for o in 0..h {
                    let want: f64 = (0..n)
                        .map(|j| down[(ex * h + o) * n + j] as f64 * act[j])
                        .sum();
                    let have = got[(t * top_k + k) * h + o] as f64;
                    let tol = 1e-4 * want.abs().max(1.0);
                    assert!(
                        (have - want).abs() <= tol,
                        "token {t} pick {k} (expert {ex}) out {o}: {have} vs {want}"
                    );
                }
            }
        }
        assert!(clipped_any, "the reference must exercise the clip");

        // The VLM embeddings path feeds `[batch, seq, hidden]`; the rank is kept.
        let x3 = mlxcel_core::from_slice_f32(&x, &[1, n_tokens as i32, hidden]);
        let idx3 = mlxcel_core::from_slice_i32(&picks, &[1, n_tokens as i32, top_k as i32]);
        let y3 = glu.forward(&x3, &idx3);
        mlxcel_core::eval(&y3);
        assert_eq!(
            mlxcel_core::array_shape(&y3),
            vec![1, n_tokens as i32, top_k as i32, hidden]
        );
        let got3 = mlxcel_core::utils::array_to_vec_f32(&y3);
        for (a, b) in got.iter().zip(&got3) {
            assert!((a - b).abs() <= 1e-5 * a.abs().max(1.0), "{a} vs {b}");
        }
    }

    /// A quantized export that lost every `.scales` key must not load its
    /// packed `uint32` planes as float weights (that would abort in
    /// `gather_mm` at the first routed forward pass).
    #[test]
    fn float_switch_glu_refuses_planes_that_lost_their_scales() {
        let _runtime = crate::initialize_runtime();
        let (e, hidden, inter) = (2i32, 64i32, 64i32);
        let plane =
            |shape: &[i32]| quantized_triple(&ramp((e * inter * hidden) as usize), shape, 64, 4);
        let mut with_biases = WeightMap::new();
        let mut packed_only = WeightMap::new();
        for (proj, shape) in [
            ("gate_proj", [e, inter, hidden]),
            ("up_proj", [e, inter, hidden]),
            ("down_proj", [e, hidden, inter]),
        ] {
            let (w, _s, b) = plane(&shape);
            packed_only.insert(format!("p.{proj}.weight"), mlxcel_core::copy(&w));
            with_biases.insert(format!("p.{proj}.weight"), w);
            with_biases.insert(format!("p.{proj}.biases"), b);
        }
        let err = match SwitchGLU::from_weights(&with_biases, &test_config_moe(), "p") {
            Ok(_) => panic!("orphan `.biases` must be refused"),
            Err(err) => err,
        };
        assert!(
            err.contains("p.gate_proj") && err.contains("no `.scales`"),
            "{err}"
        );
        let err = match SwitchGLU::from_weights(&packed_only, &test_config_moe(), "p") {
            Ok(_) => panic!("a packed integer plane must be refused"),
            Err(err) => err,
        };
        assert!(
            err.contains("p.gate_proj") && err.contains("f16, bf16 or f32"),
            "{err}"
        );
    }

    #[test]
    fn switch_glu_refuses_mixed_quantized_and_float_planes() {
        let _runtime = crate::initialize_runtime();
        let (e, hidden, inter) = (2i32, 64i32, 64i32);
        let mut weights = WeightMap::new();
        let (w, s, b) = quantized_triple(
            &ramp((e * inter * hidden) as usize),
            &[e, inter, hidden],
            64,
            4,
        );
        weights.insert("p.gate_proj.weight".to_string(), w);
        weights.insert("p.gate_proj.scales".to_string(), s);
        weights.insert("p.gate_proj.biases".to_string(), b);
        weights.insert(
            "p.up_proj.weight".to_string(),
            float_plane(e, inter, hidden, 1.0, 2).1,
        );
        weights.insert(
            "p.down_proj.weight".to_string(),
            float_plane(e, hidden, inter, 1.0, 3).1,
        );
        let err = match SwitchGLU::from_weights(&weights, &quant_config(64, 4), "p") {
            Ok(_) => panic!("a block mixing quantized and float planes must be refused"),
            Err(err) => err,
        };
        assert_eq!(
            err,
            "p: DeepSeek-V3 routed experts mix quantized and float planes (gate_proj quantized, \
             up_proj/down_proj float)"
        );
    }

    #[test]
    fn moe_block_3d_forward_matches_reference_shape() {
        // Full MoE block on a 3D `[batch, seq, hidden]` input exercises the exact
        // crash path: gate -> SwitchGLU -> moe_weighted_sum. Before the fix this
        // aborted on the broadcast mismatch; it must now return `[batch, seq, hidden]`.
        let (e, hidden, inter, top_k) = (4i32, 64i32, 64i32, 2i32);
        let moe = MoEBlock {
            gate: tiny_gate(e, hidden, top_k),
            switch_mlp: tiny_switch_glu(e, hidden, inter),
            shared_experts: None,
        };
        let x = mlxcel_core::from_slice_f32(&ramp((3 * hidden) as usize), &[1, 3, hidden]);
        let out = moe.forward(&x);
        mlxcel_core::eval(&out);
        assert_eq!(mlxcel_core::array_shape(&out), vec![1, 3, hidden]);

        let mx = mlxcel_core::max_all(&out);
        mlxcel_core::eval(&mx);
        assert!(
            mlxcel_core::item_f32(&mx).is_finite(),
            "MoE output must be finite"
        );
    }

    /// Regression guard for the real Kimi-VL / Moonlight text config, whose
    /// `q_lora_rank` is JSON `null` (the backbone projects the query directly).
    /// Parsing this shape used to fail with "invalid type: null, expected
    /// usize"; the Kimi-VL loader reuses this exact `DeepSeekV3Config`.
    #[test]
    fn parses_null_q_lora_rank_config() {
        // Field values mirror the real kimi-vl-a3b-thinking config.json
        // `text_config` block.
        let json = r#"{
            "model_type": "deepseek_v3",
            "vocab_size": 163840,
            "hidden_size": 2048,
            "intermediate_size": 11264,
            "moe_intermediate_size": 1408,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "kv_lora_rank": 512,
            "q_lora_rank": null,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "qk_nope_head_dim": 128,
            "first_k_dense_replace": 1,
            "rope_theta": 800000.0
        }"#;
        let cfg: DeepSeekV3Config =
            serde_json::from_str(json).expect("null q_lora_rank must parse");
        assert_eq!(cfg.q_lora_rank, None);
        assert_eq!(cfg.kv_lora_rank, 512);
        // q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64.
        assert_eq!(cfg.q_head_dim(), 192);
    }

    /// A numeric `q_lora_rank` (genuine DeepSeek-V3) still parses to `Some`, and
    /// an absent key defaults to `None` via `#[serde(default)]`.
    #[test]
    fn parses_numeric_and_absent_q_lora_rank() {
        let base = r#"{
            "vocab_size": 1,
            "hidden_size": 1,
            "intermediate_size": 1,
            "moe_intermediate_size": 1,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "kv_lora_rank": 1,
            "qk_rope_head_dim": 1,
            "v_head_dim": 1,
            "qk_nope_head_dim": 1
        }"#;
        // Absent key -> None.
        let absent: DeepSeekV3Config =
            serde_json::from_str(base).expect("absent q_lora_rank must parse");
        assert_eq!(absent.q_lora_rank, None);

        // Numeric value -> Some.
        let with_rank = base.replace(
            "\"kv_lora_rank\": 1,",
            "\"kv_lora_rank\": 1, \"q_lora_rank\": 1536,",
        );
        let numeric: DeepSeekV3Config =
            serde_json::from_str(&with_rank).expect("numeric q_lora_rank must parse");
        assert_eq!(numeric.q_lora_rank, Some(1536));
    }

    /// Insert a dummy (empty) weight entry — sufficient for key-presence checks.
    fn insert_dummy(weights: &mut WeightMap, key: &str) {
        weights.insert(
            key.to_string(),
            mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
        );
    }

    /// One expert whose plane differs in shape from expert 0's is a load
    /// error naming its key and both shapes, not an MLX `[stack]` abort
    /// (issue #3320).
    #[test]
    fn sanitize_weights_refuses_a_mismatched_expert_plane() {
        let config = test_config_moe();
        let mut weights = WeightMap::new();
        for (e, cols) in [(0usize, 8i32), (1, 6)] {
            weights.insert(
                format!("model.layers.0.mlp.experts.{e}.gate_proj.weight"),
                mlxcel_core::from_slice_f32(&vec![0.0; 4 * cols as usize], &[4, cols]),
            );
        }
        let err = DeepSeekV3Model::sanitize_weights(weights, &config)
            .err()
            .expect("a mismatched expert must be refused");
        assert!(
            err.contains("model.layers.0.mlp.experts.1.gate_proj.weight")
                && err.contains("[4, 6]")
                && err.contains("[4, 8]"),
            "{err}"
        );
    }

    #[test]
    fn sanitize_weights_remaps_switch_mlp_without_mlp_segment() {
        let config = test_config_moe();
        let mut weights = WeightMap::new();

        // Simulate pre-stacked weights stored WITHOUT the `.mlp.` segment
        for l in 0..2usize {
            for m in ["gate_proj", "down_proj", "up_proj"] {
                for k in ["weight", "scales", "biases"] {
                    insert_dummy(
                        &mut weights,
                        &format!("model.layers.{l}.switch_mlp.{m}.{k}"),
                    );
                }
            }
        }

        let out =
            DeepSeekV3Model::sanitize_weights(weights, &config).expect("sanitize must succeed");

        // After sanitization, keys must have the `.mlp.` segment
        for l in 0..2usize {
            for m in ["gate_proj", "down_proj", "up_proj"] {
                let canonical_key = format!("model.layers.{l}.mlp.switch_mlp.{m}.weight");
                assert!(
                    out.contains_key(&canonical_key),
                    "expected key '{canonical_key}' not found after remapping"
                );
                // Old key must be gone
                let old_key = format!("model.layers.{l}.switch_mlp.{m}.weight");
                assert!(
                    !out.contains_key(&old_key),
                    "old key '{old_key}' should have been removed"
                );
            }
        }
    }

    #[test]
    fn sanitize_weights_does_not_overwrite_existing_canonical_keys() {
        let config = test_config_moe();
        let mut weights = WeightMap::new();

        // Both old and canonical keys present — canonical should survive
        for l in 0..2usize {
            for m in ["gate_proj"] {
                {
                    let k = "weight";
                    insert_dummy(
                        &mut weights,
                        &format!("model.layers.{l}.switch_mlp.{m}.{k}"),
                    );
                    insert_dummy(
                        &mut weights,
                        &format!("model.layers.{l}.mlp.switch_mlp.{m}.{k}"),
                    );
                }
            }
        }

        let out =
            DeepSeekV3Model::sanitize_weights(weights, &config).expect("sanitize must succeed");

        // Canonical key must still be present
        for l in 0..2usize {
            assert!(
                out.contains_key(&format!("model.layers.{l}.mlp.switch_mlp.gate_proj.weight")),
                "canonical key should survive when both old and canonical were present"
            );
        }
    }

    // ---- Issue #525 round 3: backbone correctness regressions --------------
    //
    // The real kimi-vl-a3b-thinking checkpoint has 27 REAL decoder layers
    // (`model.layers.0..=26`) and NO multi-token-prediction trailer; genuine
    // DeepSeek-V3 stores its MTP head at the out-of-range index
    // `model.layers.{num_hidden_layers}` (61), which mlx-lm's sanitize strips.
    // The port previously treated index `num_hidden_layers - 1` as the MTP
    // layer: sanitize deleted the last real layer's weights, skipped its
    // kv_b_proj decomposition, and `from_weights` built one layer too few.

    /// Minimal config for a 2-layer dense backbone with a direct q projection
    /// (`q_lora_rank: None`), the exact shape of the Kimi-VL / Moonlight text
    /// config.
    pub(crate) fn test_config_dense_direct_q() -> DeepSeekV3Config {
        DeepSeekV3Config {
            model_type: "deepseek_v3".to_string(),
            vocab_size: 11,
            hidden_size: 8,
            intermediate_size: 8,
            moe_intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            n_shared_experts: None,
            n_routed_experts: None,
            routed_scaling_factor: 1.0,
            kv_lora_rank: 8,
            q_lora_rank: None,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            qk_nope_head_dim: 4,
            topk_method: "noaux_tc".to_string(),
            scoring_func: "sigmoid".to_string(),
            norm_topk_prob: true,
            n_group: 1,
            topk_group: 1,
            num_experts_per_tok: 1,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            quantization: None,
        }
    }

    fn insert_ramp(weights: &mut WeightMap, key: &str, shape: &[i32]) {
        let n: usize = shape.iter().product::<i32>() as usize;
        weights.insert(
            key.to_string(),
            mlxcel_core::from_slice_f32(&ramp(n), shape),
        );
    }

    /// Insert the full unquantized weight set for one direct-q decoder layer.
    pub(crate) fn insert_dense_layer_weights(
        weights: &mut WeightMap,
        cfg: &DeepSeekV3Config,
        l: usize,
    ) {
        let p = format!("model.layers.{l}");
        let hidden = cfg.hidden_size as i32;
        let heads = cfg.num_attention_heads as i32;
        let q_head_dim = cfg.q_head_dim() as i32;
        let kv_lora = cfg.kv_lora_rank as i32;
        let rope = cfg.qk_rope_head_dim as i32;
        let nope = cfg.qk_nope_head_dim as i32;
        let v_dim = cfg.v_head_dim as i32;
        let inter = cfg.intermediate_size as i32;

        insert_ramp(
            weights,
            &format!("{p}.self_attn.q_proj.weight"),
            &[heads * q_head_dim, hidden],
        );
        insert_ramp(
            weights,
            &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
            &[kv_lora + rope, hidden],
        );
        insert_ramp(
            weights,
            &format!("{p}.self_attn.kv_a_layernorm.weight"),
            &[kv_lora],
        );
        insert_ramp(
            weights,
            &format!("{p}.self_attn.kv_b_proj.weight"),
            &[heads * (nope + v_dim), kv_lora],
        );
        insert_ramp(
            weights,
            &format!("{p}.self_attn.o_proj.weight"),
            &[hidden, heads * v_dim],
        );
        insert_ramp(
            weights,
            &format!("{p}.mlp.gate_proj.weight"),
            &[inter, hidden],
        );
        insert_ramp(
            weights,
            &format!("{p}.mlp.up_proj.weight"),
            &[inter, hidden],
        );
        insert_ramp(
            weights,
            &format!("{p}.mlp.down_proj.weight"),
            &[hidden, inter],
        );
        insert_ramp(weights, &format!("{p}.input_layernorm.weight"), &[hidden]);
        insert_ramp(
            weights,
            &format!("{p}.post_attention_layernorm.weight"),
            &[hidden],
        );
    }

    pub(crate) fn tiny_dense_model_weights(cfg: &DeepSeekV3Config) -> WeightMap {
        let mut weights = WeightMap::new();
        let hidden = cfg.hidden_size as i32;
        let vocab = cfg.vocab_size as i32;
        insert_ramp(&mut weights, "model.embed_tokens.weight", &[vocab, hidden]);
        insert_ramp(&mut weights, "model.norm.weight", &[hidden]);
        insert_ramp(&mut weights, "lm_head.weight", &[vocab, hidden]);
        for l in 0..cfg.num_hidden_layers {
            insert_dense_layer_weights(&mut weights, cfg, l);
        }
        weights
    }

    /// The last in-range decoder layer is REAL and must survive sanitize (its
    /// kv_b_proj decomposed like every other layer); the MTP trailer at index
    /// `num_hidden_layers` must be stripped.
    #[test]
    fn sanitize_keeps_last_decoder_layer_and_strips_mtp_trailer() {
        let cfg = test_config_dense_direct_q();
        let mut weights = tiny_dense_model_weights(&cfg);
        // MTP trailer at the out-of-range index `num_hidden_layers` (= 2).
        insert_dense_layer_weights(&mut weights, &cfg, cfg.num_hidden_layers);

        let out = DeepSeekV3Model::sanitize_weights(weights, &cfg).expect("sanitize must succeed");

        for l in 0..cfg.num_hidden_layers {
            let embed_q = format!("model.layers.{l}.self_attn.embed_q.weight");
            assert!(
                out.contains_key(&embed_q),
                "layer {l} kv_b_proj must be decomposed into {embed_q}"
            );
        }
        assert!(
            !out.keys()
                .any(|k| k.starts_with(&format!("model.layers.{}.", cfg.num_hidden_layers))),
            "MTP trailer at index {} must be stripped",
            cfg.num_hidden_layers
        );
    }

    /// `from_weights` must build ALL `num_hidden_layers` decoder layers
    /// (reference: `range(config.num_hidden_layers)` in mlx-lm deepseek_v3.py
    /// and mlx-vlm kimi_vl/language.py) and produce finite logits end to end.
    #[test]
    fn from_weights_builds_all_num_hidden_layers() {
        let cfg = test_config_dense_direct_q();
        let weights = tiny_dense_model_weights(&cfg);
        let weights =
            DeepSeekV3Model::sanitize_weights(weights, &cfg).expect("sanitize must succeed");
        let model = DeepSeekV3Model::from_weights(&weights, &cfg)
            .expect("tiny dense direct-q model must load");
        assert_eq!(
            model.layers.len(),
            cfg.num_hidden_layers,
            "must build every real decoder layer"
        );

        let input = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
        let mut caches = model.make_caches_impl();
        let logits = model.forward_impl(&input, &mut caches, None);
        mlxcel_core::eval(&logits);
        assert_eq!(
            mlxcel_core::array_shape(&logits),
            vec![1, 3, cfg.vocab_size as i32]
        );
        let mx = mlxcel_core::max_all(&mlxcel_core::abs(&logits));
        mlxcel_core::eval(&mx);
        assert!(
            mlxcel_core::item_f32(&mx).is_finite(),
            "logits must be finite"
        );
    }

    #[test]
    fn latent_mla_batched_decode_harness_covers_ragged_single_and_empty() {
        let _runtime = crate::initialize_runtime();
        let cfg = test_config_dense_direct_q();
        let weights = tiny_dense_model_weights(&cfg);
        let weights =
            DeepSeekV3Model::sanitize_weights(weights, &cfg).expect("sanitize must succeed");
        let model = DeepSeekV3Model::from_weights(&weights, &cfg).expect("tiny model must load");

        crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
            &model,
            &[&[1], &[2, 3], &[4, 5, 6]],
            &[7, 8, 9],
            1e-5,
        );
        crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
            &model,
            &[1, 2, 3],
            4,
        );
        crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
    }

    /// A multi-token prefill with `mask == None` must be CAUSAL: position 0 of
    /// a 2-token prefill must produce the same attention output as a 1-token
    /// forward of that first token alone. The standard generation paths (CLI
    /// text prefill and the VLM embeddings prefill) pass `mask == None`, so
    /// before the round-3 fix prefill was fully bidirectional and every layer
    /// above the first cached future-contaminated K/V.
    #[test]
    fn prefill_is_causal_without_caller_mask() {
        let cfg = test_config_dense_direct_q();
        let mut weights = WeightMap::new();
        insert_dense_layer_weights(&mut weights, &cfg, 0);
        let weights =
            DeepSeekV3Model::sanitize_weights(weights, &cfg).expect("sanitize must succeed");
        let attn = DeepSeekV3Attention::from_weights(&weights, &cfg, "model.layers.0.self_attn")
            .expect("tiny attention must load");

        let hidden = cfg.hidden_size as i32;
        let x_data = ramp(2 * cfg.hidden_size);
        let x_two = mlxcel_core::from_slice_f32(&x_data, &[1, 2, hidden]);
        let x_first = mlxcel_core::from_slice_f32(&x_data[..cfg.hidden_size], &[1, 1, hidden]);

        let mut cache_prefill = KVCache::new();
        let out_two = attn.forward(&x_two, &mut cache_prefill, None);
        let pos0_of_two = mlxcel_core::slice(&out_two, &[0, 0, 0], &[1, 1, hidden]);

        let mut cache_single = KVCache::new();
        let out_one = attn.forward(&x_first, &mut cache_single, None);

        let diff = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(
            &pos0_of_two,
            &out_one,
        )));
        mlxcel_core::eval(&diff);
        let diff = mlxcel_core::item_f32(&diff);
        assert!(
            diff < 1e-4,
            "position 0 must not attend to position 1 during prefill (max diff {diff})"
        );
    }

    /// `clipped_swiglu` mirrors upstream kimi_vl `clipped_silu`: identity in
    /// the normal activation range, clipped where an f16 `silu(gate) * up`
    /// would overflow to inf (the issue #525 round-3 "!" spam source).
    #[test]
    fn clipped_swiglu_prevents_f16_overflow() {
        // Overflow case: silu(300) * 300 = 90000 > 65504 (f16 max) -> inf
        // unclipped; clip(silu(300)) * 300 = 100 * 300 = 30000 stays finite.
        let big = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&[300.0], &[1]),
            mlxcel_core::dtype::FLOAT16,
        );
        let clipped = clipped_swiglu(&big, &big);
        mlxcel_core::eval(&clipped);
        let val =
            mlxcel_core::item_f32(&mlxcel_core::astype(&clipped, mlxcel_core::dtype::FLOAT32));
        assert!(
            val.is_finite(),
            "clipped activation must stay finite in f16"
        );
        assert!(
            (val - 30_000.0).abs() < 64.0,
            "expected clip(silu(300)) * 300 = 30000, got {val}"
        );

        let unclipped = mlxcel_core::compiled_swiglu_activation(&big, &big);
        mlxcel_core::eval(&unclipped);
        let raw = mlxcel_core::item_f32(&mlxcel_core::astype(
            &unclipped,
            mlxcel_core::dtype::FLOAT32,
        ));
        assert!(
            raw.is_infinite(),
            "sanity: the unclipped f16 activation must overflow for this input"
        );

        // Identity case: |silu(gate)| <= 100 must be numerically unchanged.
        let gate = mlxcel_core::from_slice_f32(&[2.0], &[1]);
        let up = mlxcel_core::from_slice_f32(&[3.0], &[1]);
        let ours = clipped_swiglu(&gate, &up);
        let reference = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        let diff =
            mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(&ours, &reference)));
        mlxcel_core::eval(&diff);
        assert!(
            mlxcel_core::item_f32(&diff) < 1e-6,
            "clip must be the identity in the normal range"
        );
    }

    /// Routing scores must be computed in float32 regardless of the activation
    /// dtype (reference: `mx.sigmoid(gates.astype(mx.float32))` in mlx-lm
    /// deepseek_v3.py / mlx-vlm kimi_vl language.py `group_expert_select`).
    #[test]
    fn moe_gate_routes_in_f32_for_f16_activations() {
        let (e, hidden, top_k) = (4i32, 64i32, 2i32);
        let mut gate = tiny_gate(e, hidden, top_k);
        gate.weight = mlxcel_core::astype(&gate.weight, mlxcel_core::dtype::FLOAT16);

        let x_f16 = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&ramp((3 * hidden) as usize), &[3, hidden]),
            mlxcel_core::dtype::FLOAT16,
        );
        let (indices, scores) = gate.forward(&x_f16);
        mlxcel_core::eval(&scores);
        assert_eq!(
            mlxcel_core::array_dtype(&scores),
            mlxcel_core::dtype::FLOAT32,
            "expert scores must be computed in float32"
        );
        assert_eq!(mlxcel_core::array_shape(&indices), vec![3, top_k]);
        let mx = mlxcel_core::max_all(&mlxcel_core::abs(&scores));
        mlxcel_core::eval(&mx);
        assert!(mlxcel_core::item_f32(&mx).is_finite());
    }

    /// The DeepSeek-V3 expert stacking keeps the ModelOpt NVFP4 per-expert
    /// `weight_scale_2` sidecar as the `[num_experts]` output multiplier
    /// (issue #2351). This family's own expert path is affine-only, so the
    /// stacked block-float plane is then refused by name at load instead of
    /// as a missing `.biases` key.
    #[test]
    fn modelopt_expert_sidecars_stack_and_are_refused_by_the_affine_expert_path() {
        use crate::models::switch_layers::{
            insert_modelopt_expert_plane, stacked_inv_global_scale,
        };
        let mut config = quant_config(16, 4);
        config.n_routed_experts = Some(2);
        let mut weights = WeightMap::new();
        for e in 0..2usize {
            for (j, (proj, out, input)) in [
                ("gate_proj", 32, 64),
                ("up_proj", 32, 64),
                ("down_proj", 64, 32),
            ]
            .into_iter()
            .enumerate()
            {
                insert_modelopt_expert_plane(
                    &mut weights,
                    &format!("model.layers.0.mlp.experts.{e}.{proj}"),
                    out,
                    input,
                    (1 + e + j) as f32,
                    (e * 3 + j) as u32,
                );
            }
        }
        let weights = DeepSeekV3Model::sanitize_weights(weights, &config).expect("sanitize");
        let prefix = "model.layers.0.mlp.switch_mlp";
        for (j, proj) in ["gate_proj", "up_proj", "down_proj"]
            .into_iter()
            .enumerate()
        {
            let want: Vec<f32> = (0..2).map(|e| (1 + e + j) as f32).collect();
            assert_eq!(
                stacked_inv_global_scale(&weights, &format!("{prefix}.{proj}")),
                want
            );
        }
        let err = match SwitchGLU::from_weights(&weights, &config, prefix) {
            Ok(_) => panic!("the affine-only expert path must refuse a ModelOpt NVFP4 plane"),
            Err(err) => err,
        };
        assert!(err.contains("affine-quantized planes only"), "{err}");
    }

    /// Smallest DeepSeek-V3 config that parses, varying only the declared
    /// quantization pair. Built through serde rather than a struct literal so
    /// the `#[serde(default)]` fields fill themselves in.
    fn quant_config(group_size: i32, bits: i32) -> DeepSeekV3Config {
        serde_json::from_value(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 8,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "v_head_dim": 4,
            "qk_nope_head_dim": 2,
            "quantization": { "group_size": group_size, "bits": bits },
        }))
        .expect("test config must parse")
    }

    /// This `SwitchGLU` is unconditionally quantized: it stores the declared
    /// pair and hands it to three `gather_qmm` calls, which cross the cxx
    /// bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++ throw there
    /// is an uncatchable `std::terminate`,
    /// so losing the bound turns a rejected load into an uncatchable abort at
    /// the first routed forward pass in production. This test asserts on the
    /// load result rather than running a forward pass, so a regression fails
    /// cleanly here instead of aborting the test binary.
    #[test]
    fn deepseek_v3_switch_glu_rejects_quantization_params_that_would_abort_gather_qmm() {
        // Honest 4-bit expert geometry: `packed_in * 32 == bits * num_groups *
        // group_size` (8 * 32 == 4 * 1 * 64), so the positive control below is
        // a plane MLX can actually describe.
        const EXPERTS: i32 = 3;
        const OUT: i32 = 4;
        const PACKED_IN: i32 = 8;
        const NUM_GROUPS: i32 = 1;
        const GROUP_SIZE: i32 = 64;
        const BITS: i32 = 4;
        const PREFIX: &str = "model.layers.0.mlp.switch_mlp";

        let mut weights = WeightMap::new();
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            crate::models::switch_layers::insert_stacked_quantized_expert_plane(
                &mut weights,
                &format!("{PREFIX}.{proj}"),
                EXPERTS,
                OUT,
                PACKED_IN,
                NUM_GROUPS,
            );
        }

        // Positive control first, so a guard that rejected every quantized
        // plane could not pass this test.
        SwitchGLU::from_weights(&weights, &quant_config(GROUP_SIZE, BITS), PREFIX)
            .expect("honest 4-bit expert planes must load");

        for (group_size, bits, field) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
            let err =
                match SwitchGLU::from_weights(&weights, &quant_config(group_size, bits), PREFIX) {
                    Ok(_) => panic!(
                        "(group_size {group_size}, bits {bits}) must be refused at load, \
                         not stored for gather_qmm"
                    ),
                    Err(e) => e,
                };
            assert!(
                err.contains(field),
                "(group_size {group_size}, bits {bits}) must be blamed on {field}, got: {err}"
            );
        }
    }

    /// An affine 4-bit `kv_b_proj` for every layer in `cfg`.
    ///
    /// Honest geometry at `kv_lora_rank` 16: packed_in * 32 == bits *
    /// num_groups * group_size (2 * 32 == 4 * 1 * 16). `test_config_dense_direct_q`
    /// gives num_attention_heads 2, qk_nope_head_dim 4, v_head_dim 4, so
    /// rows = 2 * 8 = 16.
    ///
    /// The packed plane is UINT32, not a float ramp: `dequantize` rejects any
    /// other packed dtype by throwing, and that throw is the uncatchable abort
    /// these guards exist to keep out of the forward path.
    fn affine_kv_b_proj_weights(cfg: &DeepSeekV3Config) -> WeightMap {
        let heads = cfg.num_attention_heads as i32;
        let head_dim = (cfg.qk_nope_head_dim + cfg.v_head_dim) as i32;
        let rows = heads * head_dim;
        let mut weights = WeightMap::new();
        for l in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{l}.self_attn.kv_b_proj");
            weights.insert(
                format!("{prefix}.weight"),
                mlxcel_core::zeros(&[rows, 2], mlxcel_core::dtype::UINT32),
            );
            insert_ramp(&mut weights, &format!("{prefix}.scales"), &[rows, 1]);
            insert_ramp(&mut weights, &format!("{prefix}.biases"), &[rows, 1]);
        }
        weights
    }

    /// The same plane an mxfp4 / nvfp4 / mxfp8 export ships: `.scales`
    /// present, `.biases` absent, because the block-float modes carry no zero
    /// points.
    fn block_float_kv_b_proj_weights(cfg: &DeepSeekV3Config) -> WeightMap {
        let mut weights = affine_kv_b_proj_weights(cfg);
        for l in 0..cfg.num_hidden_layers {
            weights.remove(&format!("model.layers.{l}.self_attn.kv_b_proj.biases"));
        }
        weights
    }

    /// The MLA `kv_b_proj` pair `DeepSeekV3Model::sanitize_weights` decomposes
    /// is solved from `kv_lora_rank` and two tensor axes rather than declared,
    /// and every input is untrusted: `kv_lora_rank` comes from `config.json`
    /// and the axes from the checkpoint. Before issue #958 the naive
    /// arithmetic divided by both without checking them, so a `kv_lora_rank`
    /// of 0 panicked on integer division and a solved pair outside anything
    /// MLX can describe reached `dequantize`, which crosses the cxx bridge as
    /// `UniquePtr<MlxArray>` rather than `Result` and therefore aborts during
    /// weight sanitization rather than failing the load.
    ///
    /// This drives the real `DeepSeekV3Model::sanitize_weights` rather than
    /// the shared helper directly, so the guard is exercised where a
    /// checkpoint reaches it. The assertions are on the returned `Result`, so
    /// a regression fails cleanly here instead of aborting the test binary.
    #[test]
    fn quantized_kv_b_proj_rejects_a_kv_lora_rank_no_packing_can_describe() {
        // Positive control first, so a guard that rejected every quantized
        // kv_b_proj could not pass this test.
        let mut honest = test_config_dense_direct_q();
        honest.kv_lora_rank = 16;
        let sanitized =
            DeepSeekV3Model::sanitize_weights(affine_kv_b_proj_weights(&honest), &honest)
                .expect("an honest quantized kv_b_proj must still sanitize");
        assert!(sanitized.contains_key("model.layers.0.self_attn.embed_q.weight"));

        // A zero `kv_lora_rank` is Rust integer division by zero on the very
        // first solve, so it has to be refused before the division rather
        // than after.
        let mut zero_rank = honest.clone();
        zero_rank.kv_lora_rank = 0;
        let err =
            DeepSeekV3Model::sanitize_weights(affine_kv_b_proj_weights(&zero_rank), &zero_rank)
                .err()
                .unwrap_or_else(|| panic!("kv_lora_rank 0 must be refused, not divided by"));
        assert!(err.contains("kv_lora_rank"), "unhelpful error: {err}");

        // A large `kv_lora_rank` truncates the solved bit width to 0, which
        // is the divisor MLX would then divide by.
        let mut wide_rank = honest.clone();
        wide_rank.kv_lora_rank = 4096;
        let err =
            DeepSeekV3Model::sanitize_weights(affine_kv_b_proj_weights(&wide_rank), &wide_rank)
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
        insert_ramp(
            &mut float_only,
            "model.layers.0.self_attn.kv_b_proj.weight",
            &[rows, wide_rank.kv_lora_rank as i32],
        );
        DeepSeekV3Model::sanitize_weights(float_only, &wide_rank)
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
    /// The decomposition still dequantizes as `"affine"`, so a genuine
    /// block-float `kv_b_proj` is not supported either way: what this pins is
    /// that it is refused by name instead of unwinding. The assertion is on
    /// the returned `Result` and no forward pass runs, so a regression fails
    /// cleanly here.
    #[test]
    fn quantized_kv_b_proj_rejects_scales_with_no_biases() {
        let mut cfg = test_config_dense_direct_q();
        cfg.kv_lora_rank = 16;

        // Positive control first, so a check that rejected every quantized
        // kv_b_proj could not pass this test.
        let sanitized = DeepSeekV3Model::sanitize_weights(affine_kv_b_proj_weights(&cfg), &cfg)
            .expect("a kv_b_proj carrying both planes must still sanitize");
        assert!(sanitized.contains_key("model.layers.0.self_attn.embed_q.weight"));

        let err = DeepSeekV3Model::sanitize_weights(block_float_kv_b_proj_weights(&cfg), &cfg)
            .err()
            .expect("scales with no biases must be refused at load, not unwrapped");
        assert!(
            err.contains("model.layers.0.self_attn.kv_b_proj.biases"),
            "the error must name the key that is missing, got: {err}"
        );
        assert!(
            err.contains("scales but no biases"),
            "the wording must match the other four MLA sanitizers, got: {err}"
        );
    }
    /// One dense direct-q MLA layer over an 11-token vocabulary.
    ///
    /// Used by: the last-row logits cases here and the Kimi-VL wrapper case in
    /// `vision::encoders::kimi_vl_tests`
    pub(crate) fn tiny_dense_model() -> DeepSeekV3Model {
        let cfg = test_config_dense_direct_q();
        let weights = DeepSeekV3Model::sanitize_weights(tiny_dense_model_weights(&cfg), &cfg)
            .expect("sanitize must succeed");
        DeepSeekV3Model::from_weights(&weights, &cfg).expect("tiny model must load")
    }

    #[test]
    fn deepseek_v3_last_row_logits_match_the_full_forward() {
        let _guard = crate::models::embedding_test_support::mlx_test_guard();
        crate::models::last_row_logits_harness::assert_last_row_logits(
            tiny_dense_model,
            crate::models::last_row_logits_harness::TOKENS,
        );
    }
}
