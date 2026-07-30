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

//! Llama 3.1 model implementation using mlxcel-core
//!
//! This implements the standard Llama architecture for dense models
//! like Llama 3.1 8B Instruct.

use mlxcel_core::cache::{BatchedAttentionMetadata, PagedDecodeMetadata};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{
    FusedQKVLinear, KVCache, KVCacheMode, RMSNorm, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::pipeline_hint;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub mlp_bias: bool,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    #[serde(default)]
    pub quantization: Option<Quantization>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Rotate interleaved channel pairs `(2i, 2i+1)` instead of the split-half
    /// pairs `(i, i + dims/2)`.
    ///
    /// Deserialized from `config.json`, defaulting to `false`. This mirrors the
    /// reference implementation in
    /// [`mlx_lm/models/llama.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/llama.py),
    /// whose `ModelArgs` declares `rope_traditional: bool = False` and passes it
    /// straight into `initialize_rope`, so a checkpoint that declares the key
    /// now decodes here the way it decodes upstream instead of being silently
    /// rotated with the other convention.
    ///
    /// The default carries every existing checkpoint unchanged: no
    /// `mlx-community` Llama, Qwen2 or Qwen2.5 checkpoint declares the key, so
    /// they all keep the split-half rotation they load with today.
    ///
    /// A loader may still set the flag programmatically for a family whose
    /// upstream definition fixes the convention in code rather than in the
    /// config. That is what [`crate::models::helium`] does (upstream builds
    /// `nn.RoPE(head_dim, traditional=True, base=rope_theta)` and its
    /// `config.json` carries no `rope_traditional` key at all), which is why
    /// deserializing the key does not make `helium::ModelArgs::to_llama3_args`
    /// redundant.
    ///
    /// An explicit `null` parses as `false` rather than as an error. Until
    /// #931 this field was `#[serde(skip)]`, which ignored whatever the key
    /// held, so a config carrying `"rope_traditional": null` loaded fine;
    /// tolerating it keeps such a config loading instead of turning a
    /// previously-ignored key into a load failure. It also matches the
    /// reference, where a `None` is falsy and selects the split-half rotation.
    ///
    /// See [`Attention::forward`] for why setting this also disables the two
    /// fused RoPE fast paths.
    #[serde(default, deserialize_with = "deserialize_rope_traditional")]
    pub rope_traditional: bool,
}

/// Parse `rope_traditional`, mapping an explicit JSON `null` to `false`.
///
/// See [`ModelArgs::rope_traditional`] for why this one field tolerates `null`
/// where a plain `bool` would reject it.
fn deserialize_rope_traditional<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<bool>::deserialize(deserializer)?.unwrap_or(false))
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type", default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
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

/// Opts a quantized prefill into `fused_causal_prefill_attention`. Read by the
/// gate in [`Attention::forward`].
pub(crate) const FUSED_CAUSAL_PREFILL_ENV: &str = "MLXCEL_ENABLE_FUSED_CAUSAL_PREFILL_ATTENTION";

/// Opts a quantized projection into `fused_qkv_project_split_rope`. Read inside
/// [`FusedQKVLinear::forward_split_rope`], not here.
pub(crate) const FUSED_QKV_SPLIT_ROPE_ENV: &str = "MLXCEL_ENABLE_FUSED_QKV_SPLIT_ROPE";

/// Both environment variables that opt a quantized checkpoint into a fused RoPE
/// launcher.
///
/// Both launchers hardcode the split-half rotation inside C++ and take no flag,
/// so a traditional-RoPE checkpoint is routed around them. Listing them once
/// keeps the gate in [`Attention::forward`], the notice in
/// [`Attention::from_weights`] and the tests reading the same set.
pub(crate) const FUSED_ROPE_ENV_VARS: [&str; 2] =
    [FUSED_CAUSAL_PREFILL_ENV, FUSED_QKV_SPLIT_ROPE_ENV];

/// Print, at most once per process, that a traditional-RoPE checkpoint has been
/// routed around a fused RoPE launcher that an environment variable asked for.
///
/// Correctness does not depend on this: [`Attention::forward`] bypasses the
/// launchers regardless. What it buys is that the fallback is reported rather
/// than silent, which is the one real cost the bypass carries. The notice fires
/// only when both conditions hold, so a user who never sets the variables never
/// sees it.
///
/// `eprintln!` rather than `tracing::warn!` on purpose: only the server installs
/// a `tracing` subscriber, so a `warn!` here is a no-op in the `mlxcel` CLI,
/// which is where this is most likely to be read.
fn report_fused_rope_bypass_once() {
    static NOTICE: std::sync::Once = std::sync::Once::new();

    let requested: Vec<&str> = FUSED_ROPE_ENV_VARS
        .iter()
        .copied()
        .filter(|name| std::env::var(name).is_ok())
        .collect();
    if requested.is_empty() {
        return;
    }

    NOTICE.call_once(|| {
        eprintln!(
            "note: this checkpoint sets rope_traditional, so the fused RoPE fast path(s) \
             requested by {} are bypassed. Those launchers apply the split-half rotation inside \
             C++ and cannot express the interleaved one, so using them would silently mis-rotate \
             attention. The graph path is used instead; it applies the correct rotation and costs \
             a handful of extra FFI calls per layer, not GPU work.",
            requested.join(" and ")
        );
    });
}

// Attention.
pub struct Attention {
    /// Fused QKV projection: Q, K, V weights concatenated along output dim.
    /// Replaces separate q_proj, k_proj, v_proj for better NA utilization.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    /// See [`ModelArgs::rope_traditional`]. `false` unless the checkpoint's
    /// `config.json` declares the key or a loader sets it programmatically, so
    /// every family that used this attention before Helium keeps its graph.
    pub rope_traditional: bool,
}

impl Attention {
    /// Dense attention with RoPE.
    ///
    /// Used by: Llama (`llama` / `mistral` checkpoints), Qwen2 / Qwen2.5 (which
    /// re-export this attention), Helium (which reuses it with
    /// `rope_traditional` set), the Llama-3.2-Vision (`mllama`) text decoder's
    /// self-attention layers, every VLM whose text backbone is `Llama3Model` or
    /// `Qwen2Model` (Pixtral, LLaVA, SmolVLM / Idefics3, Idefics2, InternVL,
    /// FastVLM, dots.ocr), the `llama` and `mistral` pipeline stage executors,
    /// and the tensor-parallel Llama runtime.
    ///
    /// # Traditional RoPE and the fused fast paths
    ///
    /// Three code paths below can rotate Q and K, and all three must agree on
    /// the rotation convention or the model decodes a different graph depending
    /// on an environment variable and on whether the checkpoint is quantized:
    ///
    /// 1. `fused_causal_prefill_attention` (quantized prefill, opt-in through
    ///    `MLXCEL_ENABLE_FUSED_CAUSAL_PREFILL_ATTENTION`),
    /// 2. [`FusedQKVLinear::forward_split_rope`] (quantized projection, opt-in
    ///    through `MLXCEL_ENABLE_FUSED_QKV_SPLIT_ROPE`),
    /// 3. the graph fallback, which calls `fast_rope` directly.
    ///
    /// Only the third can express the convention. The first two apply RoPE
    /// inside a C++ launcher that hardcodes `traditional = false`
    /// (`mlx_cxx_bridge.cpp`, `fused_causal_prefill_attention` and
    /// `fused_qkv_project_split_rope`), and neither takes a flag, so there is no
    /// way to ask them for the interleaved rotation. Routing a traditional-RoPE
    /// model through either one applies the wrong rotation to correctly shaped
    /// tensors: nothing throws, the KV cache is the right shape, and the model
    /// emits fluent text out of a mis-rotated attention.
    ///
    /// Both are therefore gated on `!self.rope_traditional`, so a
    /// traditional-RoPE model always takes the graph fallback, which receives
    /// the flag.
    ///
    /// ## Why the bypass stays, now that the flag is config-driven (#931)
    ///
    /// #930 introduced the bypass for a single family whose flag was fixed at
    /// load time and called extending the cxx bridge the better long-term fix.
    /// Deserializing the key (#931) makes the flag reachable from any Llama,
    /// Qwen2 or Qwen2.5 `config.json`, which is the point at which that call had
    /// to be re-made rather than inherited. The bypass stays, for reasons that
    /// are about what the launchers actually are:
    ///
    /// - Neither launcher is a fused kernel. Read the two C++ bodies: each is a
    ///   `quantized_matmul`, three `slice`s, `reshape`, `transpose` and
    ///   `fast::rope`, which is the same MLX graph the Rust fallback builds op
    ///   for op. What they save is roughly eleven cxx crossings per layer per
    ///   forward, not GPU work. So the "silent performance cliff" a bypass would
    ///   open is FFI call overhead, not throughput, and it is invisible next to
    ///   the quantized matmuls on either side of it.
    /// - Both are opt-in behind an environment variable and off by default, so
    ///   no configuration shipped today loses anything.
    /// - The flag would have to be threaded through a signature shared with
    ///   `fused_qkv_project_and_rope` and `fused_qkv_project_split_norm_rope`,
    ///   which hardcode the same constant and serve other families. Changing two
    ///   of the four leaves an inconsistent surface; changing all four adds a
    ///   parameter that is `false` at every call site in the tree and puts
    ///   families that are not part of this fix into the blast radius.
    /// - It could not be validated where it matters. No checkpoint in existence
    ///   pairs `rope_traditional` with the quantized Llama path, so an extended
    ///   bridge would ship on synthetic evidence, while the bypass is provable
    ///   from the launcher's own behavior (see
    ///   `the_fused_qkv_rope_launcher_cannot_express_traditional_rope` in
    ///   `helium_tests.rs`, which pins exactly that).
    ///
    /// What the bypass was missing was observability, and that is fixed rather
    /// than argued away: [`Attention::from_weights`] prints a one-time notice on
    /// stderr when a traditional-RoPE checkpoint is loaded while one of the two
    /// environment variables asked for a fused path. The fallback is then a
    /// reported decision instead of a silent one. Revisit this if either
    /// launcher is ever promoted to default-on, or if either grows a genuine
    /// kernel behind it; both are conditions the notice makes visible.
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

        if l > 1
            && mask.is_none()
            && cache.is_empty()
            && !self.rope_traditional
            && std::env::var(FUSED_CAUSAL_PREFILL_ENV).is_ok()
            && let (Some(qkv_weight), Some(o_weight)) = (
                self.qkv_proj.qkv_proj.as_quantized_weight(),
                self.o_proj.as_quantized_weight(),
            )
        {
            let mut output = UniquePtr::null();
            let mut k = UniquePtr::null();
            let mut v = UniquePtr::null();
            unsafe {
                mlxcel_core::fused_causal_prefill_attention(
                    x,
                    &qkv_weight.weight,
                    &qkv_weight.scales,
                    qkv_weight.biases_ptr(),
                    &o_weight.weight,
                    &o_weight.scales,
                    o_weight.biases_ptr(),
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.rope_dims,
                    self.rope_base,
                    self.scale,
                    qkv_weight.group_size,
                    qkv_weight.bits,
                    &qkv_weight.mode,
                    &mut output,
                    &mut k,
                    &mut v,
                );
            }
            cache.update(k, v);
            return output;
        }

        let fused_split_rope = if self.rope_traditional {
            // The fused launcher hardcodes `traditional = false`; see the method
            // doc comment. Skipping it is what keeps the rotation correct.
            None
        } else {
            self.qkv_proj
                .forward_split_rope(x, self.rope_dims, self.rope_base, offset)
        };

        // Fused q/k RoPE + KV-append-layout kernel (#905). Unlike
        // `forward_split_rope` above, this one takes `traditional` as a real
        // parameter, so traditional-RoPE checkpoints reach it too. It emits
        // K/V in the dense `KVCache` slab layout, which is what
        // `update_and_fetch` splices below; the paged-pool layout the kernel
        // also supports belongs to the batched paged decode path (#899) and is
        // deliberately not wired here. `MLXCEL_FUSED_ROPE_APPEND=0` falls back
        // to the reshape / transpose / `fast_rope` graph below.
        let fused_rope_append = if fused_split_rope.is_some() {
            None
        } else {
            self.qkv_proj.forward_fused_rope_append(
                x,
                self.rope_dims,
                self.rope_base,
                1.0,
                self.rope_traditional,
                offset,
                mlxcel_core::layers::FusedRopeDestLayout::DenseSlab,
            )
        };

        let (q, k, v) = if let Some((q, k, v)) = fused_split_rope {
            (q, k, v)
        } else if let Some((q, k, v)) = fused_rope_append {
            (q, k, v)
        } else {
            // Fallback for non-quantized weights: preserve the existing Rust path.
            let (q, k, v) = self.qkv_proj.forward(x);

            // Reshape to [batch, seq_len, n_heads, head_dim]
            let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
            let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
            let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

            // Transpose to [batch, n_heads, seq_len, head_dim]
            let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
            let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
            let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

            let q = mlxcel_core::fast_rope(
                &q,
                self.rope_dims,
                self.rope_traditional,
                self.rope_base,
                1.0,
                offset,
            );
            let k = mlxcel_core::fast_rope(
                &k,
                self.rope_dims,
                self.rope_traditional,
                self.rope_base,
                1.0,
                offset,
            );
            (q, k, v)
        };

        // Decode-case (l == 1) attention dispatch for the Turbo quantized cache
        // modes. Prefill (l > 1) builds the cache from scratch and falls through
        // to the standard masked/causal paths below. This module's `Attention`
        // is also re-exported as Qwen2 / Qwen2.5's attention, so the routing here
        // covers those families too.
        //
        // Turbo4Asym (FP16 K + 4-bit V) decodes via dequant-first native SDPA by
        // default: V is dequantized to FP16 transiently and fed with the FP16 K
        // to native SDPA. This is exact and ~3-6x faster than the lossy sparse-V
        // weighted-sum path, which stays reachable behind
        // `MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=0` for A/B and fallback. Symmetric
        // Turbo4 and Turbo4Delegated mirror mlx-swift-lm's dequant-first policy.
        // Each gate is parsed once and cached in a `OnceLock<bool>`.
        let use_turbo4_asym_dequant_sdpa =
            mlxcel_core::cache::turbo::sparse_v::turbo4_asym_dequant_sdpa_enabled();
        let use_delegated_compressed =
            mlxcel_core::cache::turbo::sparse_v::turbo4_delegated_compressed_attention_enabled();
        let use_turbo4_dequant_sdpa =
            mlxcel_core::cache::turbo::sparse_v::turbo4_dequant_sdpa_enabled();
        let attn_out = if l == 1
            && use_turbo4_asym_dequant_sdpa
            && cache.turbo4_asym_dequant_sdpa_available()
        {
            // Default Turbo4Asym decode: dequantize the 4-bit V to FP16 and run
            // native SDPA with the FP16 K. Exact full-dequant attention.
            cache.update_and_turbo4_asym_dequant_sdpa_attention(&q, k, v, self.scale, mask)
        } else if l == 1 && cache.sparse_v_available() {
            // Sparse-V fallback: only taken when the dequant-SDPA gate above is
            // disabled (`MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=0`).
            cache
                .update_and_sparse_v_attention(&q, k, v, self.scale, mask)
                .expect("update_and_sparse_v_attention returned None despite sparse_v_available")
        } else if l == 1 && use_turbo4_dequant_sdpa && cache.turbo4_dequant_sdpa_available() {
            cache.update_and_turbo4_dequant_sdpa_attention(&q, k, v, self.scale, mask)
        } else if l == 1 && use_delegated_compressed && cache.turbo4_delegated_available() {
            // The helper always produces an attention output: it routes
            // through the fused Metal kernel when available and falls
            // through to the graph-only reference path otherwise.
            cache.update_and_turbo4_delegated_attention(&q, k, v, self.scale, mask)
        } else if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            let (cache_k, cache_v) = cache.update_and_fetch(k, v);
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            // Single token or explicit mask
            let (cache_k, cache_v) = cache.update_and_fetch(k, v);
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

    /// Split-attention forward for batched decode.
    ///
    /// Receives pre-projected Q/K/V tensors of shape `[B, T, proj_dim]`,
    /// applies RoPE using batched positional metadata, then runs per-sequence
    /// cache updates and attention before concatenating the results back into
    /// `[B, T, hidden_dim]`.
    ///
    /// Used by: the batched decode and full-sequence batched prefill of every
    /// family that reaches [`Attention::forward`]; see that method's `Used by`
    /// list, which this path shares.
    pub fn forward_split_attention(
        &self,
        q_batched: &MlxArray,
        k_batched: &MlxArray,
        v_batched: &MlxArray,
        caches: &mut [&mut KVCache],
        metadata: &BatchedAttentionMetadata,
        mask: Option<&MlxArray>,
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let b = caches.len();
        let seq_len = mlxcel_core::array_shape(q_batched)[1];
        debug_assert_eq!(metadata.len(), b);
        let mut attn_outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(b);

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

        let q_batched = mlxcel_core::transpose_axes(&q_batched, &[0, 2, 1, 3]);
        let k_batched = mlxcel_core::transpose_axes(&k_batched, &[0, 2, 1, 3]);
        let v_batched = mlxcel_core::transpose_axes(&v_batched, &[0, 2, 1, 3]);

        // Batched decode / batched prefill share the same rotation convention as
        // the single-sequence path above; dropping the flag on one route while
        // honoring it on the other would make a sequence decode differently
        // depending on whether it was scheduled alone or in a batch.
        let q_batched = mlxcel_core::fast_rope_batched(
            &q_batched,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            1.0,
            &metadata.rope_offsets,
        );
        let k_batched = mlxcel_core::fast_rope_batched(
            &k_batched,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            1.0,
            &metadata.rope_offsets,
        );

        let paged_decode = decode_context.and_then(|context| {
            if seq_len != 1 || mask.is_some() || !context.is_paged_decode() {
                return None;
            }
            if caches.iter().any(|cache| cache.mode != KVCacheMode::Fp16) {
                return None;
            }
            // Pool-backed caches (scheduler paged decode, #121) keep no dense
            // `keys`/`values` buffers for the native paged kernel to read.
            // Route them through the per-sequence `update_and_fetch` loop below,
            // whose transparent pool intercept writes new K/V into the shared
            // `PagedBlockPool` (`write_prefill`) and gathers the visible window
            // back (`gather_visible`) — the #152-validated single-stream path.
            if caches.iter().any(|cache| cache.is_paged_backed()) {
                return None;
            }
            let metadata =
                PagedDecodeMetadata::from_attention_metadata(metadata, context.paged_block_size)
                    .ok()?;
            Some((context.use_native_paged_kernel, metadata))
        });

        if let Some((use_native_kernel, paged_metadata)) = paged_decode {
            tracing::debug!(
                batch_size = b,
                block_size = paged_metadata.block_size,
                native_kernel = use_native_kernel,
                "Llama3 paged decode attention dispatch"
            );
            let mut cache_keys: Vec<*const MlxArray> = Vec::with_capacity(b);
            let mut cache_values: Vec<*const MlxArray> = Vec::with_capacity(b);

            for (i, cache) in caches.iter_mut().enumerate() {
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
                cache.update(k_i, v_i);
                cache_keys.push(cache.keys.as_ref().unwrap().as_ref().unwrap() as *const MlxArray);
                cache_values
                    .push(cache.values.as_ref().unwrap().as_ref().unwrap() as *const MlxArray);
            }

            let attn_out = if use_native_kernel {
                mlxcel_core::layers::paged_decode_attention_dense_compat(
                    &q_batched,
                    &cache_keys,
                    &cache_values,
                    &paged_metadata,
                    self.scale,
                )
            } else {
                mlxcel_core::layers::paged_decode_attention_dense_fallback(
                    &q_batched,
                    &cache_keys,
                    &cache_values,
                    &paged_metadata,
                    self.scale,
                )
            }
            .expect("valid llama3 paged decode attention inputs");

            let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
            return mlxcel_core::reshape(
                &attn_out,
                &[b as i32, seq_len, self.num_heads * self.head_dim],
            );
        }

        for (i, cache) in caches.iter_mut().enumerate() {
            // Slice [B, heads, T, dim] -> [1, heads, T, dim] for sequence i.
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

            // Update KV cache
            let (cache_k, cache_v) = cache.update_and_fetch(k_i, v_i);

            let mask_i = mask.map(|m| {
                let sliced =
                    mlxcel_core::slice(m, &[i as i32, 0, 0], &[i as i32 + 1, seq_len, i32::MAX]);
                mlxcel_core::squeeze_axis(&sliced, 0)
            });

            // Causal prefill without explicit masks can use the shared causal
            // helper; masked/padded prefill and decode keep using the unified
            // attention dispatcher.
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

            // Transpose back: [1, n_heads, T, head_dim] -> [1, T, n_heads * head_dim]
            let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
            let attn_out =
                mlxcel_core::reshape(&attn_out, &[1, seq_len, self.num_heads * self.head_dim]);

            attn_outputs.push(attn_out);
        }

        // Concatenate along batch dim: B * [1, T, hidden] -> [B, T, hidden]
        let mut result = attn_outputs.remove(0);
        for attn_out in attn_outputs {
            result = mlxcel_core::concatenate(&result, &attn_out, 0);
        }
        result
    }

    /// Build the dense attention block from a checkpoint.
    ///
    /// `rope_traditional` is carried over from [`ModelArgs`], where it is
    /// deserialized from `config.json` with a default of `false` and may also be
    /// set programmatically by a loader whose family fixes the convention in
    /// code (Helium). This is also where the one-time fused-path notice is
    /// emitted, because it is the only place that sees the flag off the hot
    /// path; [`Attention::forward`] explains what it reports and why.
    ///
    /// Used by: Llama (`llama` / `mistral` checkpoints), Qwen2 / Qwen2.5 (which
    /// re-export this attention), Helium (which reuses it with
    /// `rope_traditional` set), the Llama-3.2-Vision (`mllama`) text decoder's
    /// self-attention layers, every VLM whose text backbone is `Llama3Model` or
    /// `Qwen2Model` (Pixtral, LLaVA, SmolVLM / Idefics3, Idefics2, InternVL,
    /// FastVLM, dots.ocr), the `llama` and `mistral` pipeline stage executors,
    /// and the tensor-parallel Llama runtime.
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        if args.rope_traditional {
            report_fused_rope_bypass_once();
        }

        let head_dim = args.head_dim() as i32;
        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_kv_heads() as i32;

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
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim,
            rope_base: args.rope_theta,
            rope_traditional: args.rope_traditional,
        })
    }
}

// MLP (SwiGLU).
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        // Non-quantized path: fused compiled FP MLP (single compiled graph)
        if let Some(result) = mlxcel_core::layers::compiled_swiglu_mlp_fp16(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ) {
            return result;
        }

        // Quantized path: separate projections + compiled SwiGLU activation
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

// Transformer Block.
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
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

        // Residual join + pre-FFN norm in one dispatch (#905). `h` is the
        // residual stream carried to the second join; `normed` feeds the MLP.
        // `MLXCEL_FUSED_ADD_RMSNORM=0` restores the `add` + `fast_rms_norm`
        // pair this replaced.
        let (normed, h) =
            mlxcel_core::layers::fused_add_rms_norm(&self.post_attention_layernorm, &attn_out, x);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    /// Batched decode forward: batch norms + projections + FFN, per-sequence attention.
    ///
    /// `x` has shape `[B, T, hidden_dim]`, `caches[i]` is the KVCache for
    /// the i-th sequence. Returns `[B, T, hidden_dim]`.
    ///
    /// Used by: Llama3Model::forward_batched
    pub fn forward_batched(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
        mask: Option<&MlxArray>,
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        // Batched pre-attention norm
        let normed = self.input_layernorm.forward(x);

        // Batched Q/K/V projection (fused single matmul)
        let (q, k, v) = self.self_attn.qkv_proj.forward(&normed);
        let seq_len = mlxcel_core::array_shape(&q)[1];
        let metadata = BatchedAttentionMetadata::uniform_kv_caches(caches, seq_len, 0)
            .expect("valid llama3 batched attention metadata");

        // Per-sequence attention still owns cache mutation, but positional
        // metadata and RoPE now stay on a batched path.
        let attn_concat = self.self_attn.forward_split_attention(
            &q,
            &k,
            &v,
            caches,
            &metadata,
            mask,
            decode_context,
        );

        // Batched output projection
        let attn_out = self.self_attn.o_proj.forward(&attn_concat);

        // Residual join + batched post-attention norm in one dispatch (#905).
        let (normed, h) =
            mlxcel_core::layers::fused_add_rms_norm(&self.post_attention_layernorm, &attn_out, x);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

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

// Llama Model.
pub struct Llama3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
}

impl Llama3Model {
    /// Forward pass through the entire model
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers
        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
            pipeline_hint(&h, i, n);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        self.lm_head.forward(&h)
    }

    /// Forward pass with optional pre-computed embeddings (for VLM support)
    pub fn forward_with_embeddings_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Use provided embeddings or compute from input_ids
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        // Pass through transformer layers
        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
            pipeline_hint(&h, i, n);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        self.lm_head.forward(&h)
    }

    /// Batched forward pass: batch compute-bound layers, per-sequence attention.
    ///
    /// `input_ids` has shape `[B, T]`. `batch_caches[i]` is the per-layer
    /// KV cache slice for the i-th sequence. Returns `[B, T, vocab_size]`.
    ///
    /// This is the explicit batched implementation that amortizes weight-loading
    /// bandwidth for embedding, normalization, linear projections, and FFN/MLP
    /// across all B sequences, while running attention per-sequence to handle
    /// different KV cache lengths.
    ///
    /// Used by: LanguageModel::forward_batched (overrides the loop-based default)
    pub fn forward_batched_impl(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let b = batch_caches.len();

        // Batched embedding lookup: [B, 1] -> [B, 1, hidden_dim]
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers with split-attention
        for layer_idx in 0..self.layers.len() {
            // Collect per-sequence caches for this layer
            let mut layer_caches: Vec<&mut KVCache> = batch_caches
                .iter_mut()
                .map(|caches| &mut caches[layer_idx])
                .collect();

            h = self.layers[layer_idx].forward_batched(&h, &mut layer_caches, mask, decode_context);
        }

        // Batched final norm: [B, 1, hidden_dim]
        let h = self.norm.forward(&h);

        // Batched lm_head: [B, 1, vocab_size]
        let logits = self.lm_head.forward(&h);

        // Sanity check in debug builds
        debug_assert_eq!(mlxcel_core::array_shape(&logits)[0], b as i32);

        logits
    }

    /// Get token embeddings (no sqrt scaling, unlike Gemma3)
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Create KV caches for all layers
    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Load model from directory
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

    /// Create model from loaded weights
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

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head
        let lm_head = if args.tie_word_embeddings {
            // Use embedding weights for lm_head
            UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?
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

// LanguageModel trait implementation.
impl LanguageModel for Llama3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Llama3Model::forward(self, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Llama3Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Llama 3.1 EOS tokens: <|end_of_text|>, <|eot_id|>
        vec![128001, 128009]
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_impl(input_ids, batch_caches, mask, None)
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_impl(input_ids, batch_caches, mask, context)
    }

    fn supports_batched_prefill(&self) -> bool {
        true
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        true
    }

    fn supports_paged_decode_backend(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "llama3_tests.rs"]
mod tests;
