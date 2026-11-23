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

//! Gemma 3 model implementation using mlxcel-core
//!
//! Key features:
//! - sliding_window_pattern (int): layer i is global if (i+1) % pattern == 0
//! - Local: RotatingKVCache + sliding window mask + local RoPE base
//! - Global: KVCache + full mask + global RoPE base
//! - Q/K norm with (1+weight) GemmaRMSNorm
//! - 4 RMSNorm per layer
//! - GELU activation
//! - clip_residual_f16 for float16 safety
//! - Embedding scaling: h *= sqrt(hidden_size)

use crate::distributed::pipeline::LayerFilter;
use crate::distributed::pipeline::StageExecutionOutput;
use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models::kv_ring_checkpoint;
use crate::models::kv_snapshot::{self, KvSnapshotNames};
use crate::models::model_owned::KvCacheLayerModes;
use crate::models::rope_utils::{RopeScalingSpec, printable_label};
use mlxcel_core::cache::{
    DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS, KVCacheMode, KvAttention, LayerState,
    LayerStateEncoding, LayerStateForward, LayerStateKind, LayerStateSnapshotSpec, SequenceId,
    attend_batched_rows,
};
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::{
    FusedQKVLinear, GemmaRMSNorm, KVCache, RotatingKVCache, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::{create_sliding_window_prefill_mask, pipeline_hint};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::window_sdpa::{self, WindowSdpaExtras, WindowSdpaShape};
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::cell::RefCell;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
#[serde(default)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    /// Global RoPE base frequency (used for non-sliding-window layers)
    #[serde(alias = "rope_global_base_freq")]
    pub rope_theta: f32,
    pub rope_local_base_freq: f32,
    pub query_pre_attn_scalar: f32,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,

    pub rope_scaling: Option<std::collections::HashMap<String, serde_json::Value>>,

    pub quantization: Option<Quantization>,
}

mlxcel_core::impl_rope_lifting_deserialize!(ModelArgs);

impl Default for ModelArgs {
    fn default() -> Self {
        // Defaults match Python mlx-vlm TextConfig for Gemma3
        Self {
            model_type: "gemma3_text".to_string(),
            hidden_size: 2048,
            num_hidden_layers: 26,
            intermediate_size: 16384,
            num_attention_heads: 8,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            vocab_size: 262208,
            num_key_value_heads: 4,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            query_pre_attn_scalar: 256.0,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            max_position_embeddings: 4096,
            rope_scaling: None,
            quantization: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

impl ModelArgs {
    /// The sliding window as the `i32` the rotating caches take, checked at
    /// load (issue #2790): a `sliding_window_pattern` of 0 would divide by
    /// zero in every layer-kind predicate, and a window of 0 (or above
    /// `i32::MAX`) cannot back a `RotatingKVCache`. A config whose pattern
    /// leaves no sliding layer keeps any window, unused.
    pub(crate) fn validated_sliding_window(&self) -> Result<i32, String> {
        if self.sliding_window_pattern == 0 {
            return Err("Gemma 3: sliding_window_pattern must be positive".to_string());
        }
        let has_sliding_layer = self.sliding_window_pattern > 1 && self.num_hidden_layers > 0;
        crate::models::model_owned::cache_window("Gemma 3", self.sliding_window, has_sliding_layer)
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

    /// Give the config the layout of an NVFP4 export's repacked linears:
    /// 4-bit codes in the 16-element blocks the transcode keeps.
    ///
    /// An export from NVIDIA ModelOpt or compressed-tensors has no MLX
    /// `quantization` block, and without one [`Self::group_size`] is the
    /// affine default of 64, which every repacked linear then has to be
    /// reconciled away from. A config that already has the block is left
    /// alone.
    ///
    /// Used by: `load_gemma3_vlm` (`src/loading/vlm_gemma.rs`)
    pub(crate) fn declare_modelopt_nvfp4_layout(&mut self) {
        self.quantization.get_or_insert(Quantization {
            group_size: crate::models::sanitize::NVFP4_SOURCE_GROUP_SIZE as i32,
            bits: 4,
        });
    }

    /// The RoPE position scale the global-attention layers rotate at.
    ///
    /// Gemma 3 applies `rope_scaling` to the global layers only. mlx-lm reaches
    /// `initialize_rope` with a `scaling_config` on the non-sliding branch alone
    /// ([`mlx_lm/models/gemma3_text.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3_text.py)),
    /// and mlx-vlm spells the same rule as `scaling_config=None if
    /// self.is_sliding else config.rope_scaling`
    /// ([`mlx_vlm/models/gemma3/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma3/language.py)).
    /// The sliding layers keep `rope_local_base_freq` at an unscaled position,
    /// which is why this is a *global* scale and [`layer_rope_params`] hands
    /// `1.0` to the sliding half.
    ///
    /// `1.0 / factor` for `linear` mirrors upstream's `scale = 1 /
    /// scaling_config["factor"]`, because `mx.fast.rope` multiplies the position
    /// by `scale`. Every published Gemma 3 checkpoint from 4B up declares
    /// `{"rope_type": "linear", "factor": 8.0}`, so `0.125` is the value the
    /// family actually decodes with; 1B declares no block and stays at `1.0`.
    ///
    /// The block is read through the shared [`RopeScalingSpec`] map lookup
    /// rather than a derived `#[serde(rename = "type", alias = "rope_type")]`
    /// field. A derived alias turns a config that carries *both* spellings into
    /// a hard `duplicate field` parse error, and several checkpoints in the
    /// local model set spell both (#1355).
    ///
    /// # Why an unimplemented scheme is an error here, and a warning on the shared Llama path
    ///
    /// [`crate::models::rope_utils::RopeScalingKind::resolve`] warns and falls
    /// back to the unscaled table, because the args type it serves is what
    /// several VLM loaders parse an arbitrary `text_config` into, and one
    /// shipped checkpoint (`models/internvl3-1b`) declares a scheme that path
    /// does not implement. A load error there would take a working checkpoint
    /// offline.
    ///
    /// Nothing comparable reaches `gemma3::ModelArgs`. It is parsed only from a
    /// Gemma 3 config or a Gemma 3 VLM `text_config`; every published Gemma 3
    /// checkpoint that declares the block declares `linear`; and Gemma 3n and
    /// Gemma 4 parse their own args types, so neither passes through here. The
    /// only failure available is the one a load error prevents, a `yarn` or
    /// `llama3` block silently decoding at `1.0`, which reads as fluent text.
    pub fn global_rope_scale(&self) -> Result<f32, String> {
        let declared = self
            .rope_scaling
            .as_ref()
            .map(|block| RopeScalingSpec::from_lookup(|key| block.get(key)));
        // An operator `--rope-scaling` / `--rope-freq-scale` request replaces
        // the checkpoint's block before it is read, the same way it does on the
        // shared `rope_utils` path. Gemma 3's global layers are the only ones a
        // position scale reaches; the sliding layers keep unit scale (see
        // [`layer_rope_params`]), which is also what llama.cpp does with its
        // separate SWA rope parameters.
        let Some(spec) = crate::models::rope_overrides::resolve_spec(declared.as_ref()) else {
            return Ok(1.0);
        };

        match spec.rope_type() {
            "default" => Ok(1.0),
            "linear" => {
                // Upstream indexes `scaling_config["factor"]` with no default,
                // so a `linear` block without one is malformed. Defaulting it to
                // 1.0 would reintroduce this issue's silent no-op one level
                // down, where nothing would ever report it.
                let factor = spec.factor.ok_or_else(|| {
                    "Gemma 3 config declares rope_scaling type \"linear\" but names no numeric \
                     \"factor\", so the position scale it selects is undefined"
                        .to_string()
                })?;
                if crate::models::rope_utils::is_usable_scalar(factor) {
                    Ok(1.0 / factor)
                } else {
                    Err(format!(
                        "Gemma 3 config declares rope_scaling type \"linear\" with factor \
                         {factor}, which is not a positive finite number"
                    ))
                }
            }
            other => Err(format!(
                "Gemma 3 config declares rope_scaling type \"{}\", which the Gemma 3 RoPE path \
                 does not implement; only \"linear\" and \"default\" are supported. Decoding it \
                 as unscaled would change every global-attention layer's rotation without \
                 reporting it.",
                printable_label(other)
            )),
        }
    }
}

/// The `(is_sliding, rope_base, rope_scale)` triple layer `layer_idx` rotates
/// with.
///
/// Split out of [`Attention::from_weights`] so the per-layer rule is reachable
/// without a weight map: it is pure config arithmetic, and the sliding/global
/// split is the half of this that a fix is most likely to get backwards.
/// Handing the scale to every layer is as wrong as handing it to none, and both
/// mistakes still load and still read fluently.
///
/// The scale is resolved before the branch, not inside it, so an unsupported
/// scheme is a load error at layer 0 rather than at whichever index first
/// happens to be global. That also covers a config whose
/// `sliding_window_pattern` exceeds `num_hidden_layers` and therefore has no
/// global layer to trip over.
fn layer_rope_params(args: &ModelArgs, layer_idx: usize) -> Result<(bool, f32, f32), String> {
    let is_sliding = !(layer_idx + 1).is_multiple_of(args.sliding_window_pattern);
    let global_rope_scale = args.global_rope_scale()?;

    Ok(if is_sliding {
        // The sliding layers keep the checkpoint's local base: llama.cpp holds
        // Gemma 3's SWA rotation in its own `rope_freq_base_train_swa`, which
        // `--rope-freq-base` does not reach either.
        (true, args.rope_local_base_freq, 1.0)
    } else {
        (
            false,
            crate::models::rope_overrides::resolve_base(args.rope_theta),
            global_rope_scale,
        )
    })
}

// Attention.
pub struct Attention {
    /// Fused QKV projection: Q, K, V weights concatenated along output dim.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub is_sliding: bool,
    pub rope_base: f32,
    /// Position scale handed to every RoPE call in this block: `1.0` on a
    /// sliding layer, `ModelArgs::global_rope_scale()` on a global one. See
    /// [`layer_rope_params`].
    pub rope_scale: f32,
    pub q_norm: GemmaRMSNorm,
    pub k_norm: GemmaRMSNorm,
}

impl Attention {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let offset = cache.offset();

        let (q, k, v) = if let Some((q, k, v)) = self.qkv_proj.forward_split_norm_rope_quantized(
            x,
            &self.q_norm,
            &self.k_norm,
            self.head_dim,
            self.rope_base,
            self.rope_scale,
            offset,
        ) {
            (q, k, v)
        } else {
            // Fused QKV projection: single matmul → split into Q, K, V
            let (q, k, v) = self.qkv_proj.forward(x);

            // Reshape and transpose to [batch, n_heads, seq_len, head_dim]
            let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
            let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
            let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

            let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
            let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
            let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

            // Apply Q/K normalization AFTER transpose (matches Python mlx-lm)
            let q = self.q_norm.forward(&q);
            let k = self.k_norm.forward(&k);

            // Apply RoPE
            let q = mlxcel_core::fast_rope(
                &q,
                self.head_dim,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            );
            let k = mlxcel_core::fast_rope(
                &k,
                self.head_dim,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            );
            (q, k, v)
        };

        // The cache appends K/V and runs the attention for its storage
        // (ADR 0008, #2172): a global `KVCache` takes the dense route of
        // `KVCache::attend`, a sliding `RotatingKVCache` takes
        // `RotatingKVCache::attend`, whose write path is `update_and_fetch`
        // so the decode undo log (#2182) sees every single-token write. A
        // multi-token unmasked forward whose window fits is a prefill whose
        // mask would be plain causal, and the entries take the causal helper
        // for it, as this block used to.
        let attn_out = cache.attend(&q, k, v, self.scale, mask);

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    pub(crate) fn forward_batched_decode<C: SlidingLayerCache>(
        &self,
        x: &MlxArray,
        caches: &mut [&mut C],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0] as usize;
        let seq_len = shape[1];

        let (q, k, v) = self.qkv_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[shape[0], seq_len, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[shape[0], seq_len, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[shape[0], seq_len, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        let offsets: Vec<i32> = caches.iter().map(|cache| cache.layer_offset()).collect();
        let q = mlxcel_core::fast_rope_batched(
            &q,
            self.head_dim,
            false,
            self.rope_base,
            self.rope_scale,
            &offsets,
        );
        let k = mlxcel_core::fast_rope_batched(
            &k,
            self.head_dim,
            false,
            self.rope_base,
            self.rope_scale,
            &offsets,
        );

        // One attention per row through the row's own cache
        // (`Cache::attend`; ADR 0008, #2172). Gemma 3's state is never
        // pool-backed, so there is no whole-batch launch to take, and the
        // dense-pointer paged kernels this path used to select from the
        // scheduler's `DecodeBatchContext` were per-row loops themselves.
        let attn_out = attend_batched_rows(&q, &k, &v, caches, self.scale, None);
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(
            &attn_out,
            &[batch as i32, seq_len, self.num_heads * self.head_dim],
        );
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let head_dim = args.head_dim as i32;
        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_key_value_heads as i32;
        let scale = 1.0 / args.query_pre_attn_scalar.sqrt();

        // Layer type, RoPE base and RoPE position scale. Gemma 3 alternates
        // sliding and global attention, and `rope_scaling` applies to the
        // global layers only (#1340).
        let (is_sliding, rope_base, rope_scale) = layer_rope_params(args, layer_idx)?;

        // Load Q/K normalization
        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let q_norm = GemmaRMSNorm::new(q_norm_weight, args.rms_norm_eps);
        let k_norm = GemmaRMSNorm::new(k_norm_weight, args.rms_norm_eps);

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
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            is_sliding,
            rope_base,
            rope_scale,
            q_norm,
            k_norm,
        })
    }
}

// MLP (GELU activation).
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // GeGLU: gelu(gate_proj(x)) * up_proj(x), then down_proj
        // Quantized path: fused compiled quantized MLP, when the three
        // projections share one quantization layout.
        if let Some((gate_qw, up_qw, down_qw)) = mlxcel_core::layers::fused_mlp_quantized_weights(
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ) {
            return unsafe {
                mlxcel_core::compiled_gelu_approx_mlp_forward(
                    x,
                    &gate_qw.weight,
                    &gate_qw.scales,
                    gate_qw.biases_ptr(),
                    &up_qw.weight,
                    &up_qw.scales,
                    up_qw.biases_ptr(),
                    &down_qw.weight,
                    &down_qw.scales,
                    down_qw.biases_ptr(),
                    gate_qw.group_size,
                    gate_qw.bits,
                    &gate_qw.mode,
                )
            };
        }

        // Fallback: separate operations with compiled activation
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
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

// Transformer Block (4 RMSNorm per layer).
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: GemmaRMSNorm,
    pub post_attention_layernorm: GemmaRMSNorm,
    pub pre_feedforward_layernorm: GemmaRMSNorm,
    pub post_feedforward_layernorm: GemmaRMSNorm,
}

impl TransformerBlock {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let post_attn_normed = self.post_attention_layernorm.forward(&attn_out);
        let h = mlxcel_core::compiled_clip_residual(x, &post_attn_normed);

        // Pre-norm FFN
        let normed = self.pre_feedforward_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        let post_ff_normed = self.post_feedforward_layernorm.forward(&ff_out);
        mlxcel_core::compiled_clip_residual(&h, &post_ff_normed)
    }

    fn forward_batched_decode<C: SlidingLayerCache>(
        &self,
        x: &MlxArray,
        caches: &mut [&mut C],
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward_batched_decode(&normed, caches);
        let post_attn_normed = self.post_attention_layernorm.forward(&attn_out);
        let h = mlxcel_core::compiled_clip_residual(x, &post_attn_normed);

        let normed = self.pre_feedforward_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        let post_ff_normed = self.post_feedforward_layernorm.forward(&ff_out);
        mlxcel_core::compiled_clip_residual(&h, &post_ff_normed)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        Self::from_weights_with_prefix(
            weights,
            args,
            &format!("model.layers.{layer_idx}"),
            layer_idx,
        )
    }

    /// Load layer `layer_idx` from `{prefix}.*` instead of the text model's
    /// `model.layers.{layer_idx}`.
    ///
    /// Used by: [`crate::models::gemma3_backbone::Gemma3Backbone`], which
    /// hosts Gemma 3 layers under another model's key tree (the Nemotron
    /// VoiceChat EAR-TTS backbone).
    pub fn from_weights_with_prefix(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let self_attn =
            Attention::from_weights(weights, args, &format!("{}.self_attn", prefix), layer_idx)?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let pre_ff_norm_weight = get_weight_copy(
            weights,
            &format!("{}.pre_feedforward_layernorm.weight", prefix),
        )?;
        let post_ff_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_feedforward_layernorm.weight", prefix),
        )?;

        let input_layernorm = GemmaRMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = GemmaRMSNorm::new(post_attn_norm_weight, args.rms_norm_eps);
        let pre_feedforward_layernorm = GemmaRMSNorm::new(pre_ff_norm_weight, args.rms_norm_eps);
        let post_feedforward_layernorm = GemmaRMSNorm::new(post_ff_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }
}

// Cache Interface.
pub(crate) trait CacheInterface {
    fn offset(&self) -> i32;
    /// Number of live keys the cache will actually return on the next
    /// `update_and_fetch` (`offset - live_start` for a trimmed `KVCache`,
    /// the live window for a `RotatingKVCache`). Used to size prefill masks
    /// so the mask key axis matches the returned K/V; `offset()` stays for
    /// RoPE. See issue #430 (mirrors #419/#420, #421/#422).
    fn live_len(&self) -> i32;
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>);
    /// Append `k`/`v` and run attention through the entry of the storage
    /// behind this cache (`KVCache::attend`, `RotatingKVCache::attend`;
    /// ADR 0008, #2172). The single-row forward's only attention call.
    fn attend(
        &mut self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray>;
}

impl CacheInterface for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn live_len(&self) -> i32 {
        KVCache::live_len(self)
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }

    fn attend(
        &mut self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        KVCache::attend(self, q, k, v, scale, mask)
    }
}

impl CacheInterface for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn live_len(&self) -> i32 {
        // `RotatingKVCache` has no `live_start`, but `seq_len()` is the
        // PHYSICAL buffer length, and that is not what the next append
        // returns. `update_in_place` grows the buffer by `step` (256) blocks
        // during decode, so a cache that has generated even one token holds
        // `physical > offset`; `update_concat` then concatenates only
        // `visible_len() = min(physical, offset)` prior keys onto the new
        // ones. Sizing the mask from the physical length made it wider than
        // the returned K/V and tripped `broadcast_shapes` on the first
        // multi-token append after a decode, which is exactly the shape of a
        // prompt-cache snapshot restore followed by the appended-token
        // prefill (issue #1335). Live prefill-only sequences never diverged
        // because concat leaves `physical == offset`.
        self.visible_len()
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }

    fn attend(
        &mut self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        RotatingKVCache::attend(self, q, k, v, scale, mask)
    }
}

/// The `gemma3` family tag and the tensor names Gemma 3's layers are stored
/// under, as the scheduler's generic snapshot code reads them (issue #3254).
///
/// AFMoE and Kolibri-1 reuse the `Cache` enum below but are other families,
/// so each passes its own [`KvSnapshotNames`]: the snapshot's family tag keeps
/// them apart, and the label keeps their error messages from claiming to come
/// from Gemma 3.
pub(crate) const GEMMA3_LAYER_STATE_SNAPSHOT: LayerStateSnapshotSpec = LayerStateSnapshotSpec {
    family: "gemma3",
    label: "Gemma 3",
    encoding: LayerStateEncoding::Windowed {
        standard: "standard",
        rotating: "rotating",
    },
};

pub(crate) enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    /// Copy this layer's state into `snapshot` under `prefix` (issue #1335).
    ///
    /// Used by: Gemma 3, AFMoE, Kolibri-1.
    pub(crate) fn snapshot_into(
        &self,
        snapshot: &mut ModelStateSnapshot,
        prefix: &str,
        names: KvSnapshotNames,
    ) -> Result<(), String> {
        match self {
            Cache::Standard(cache) => {
                kv_snapshot::snapshot_standard(cache, snapshot, prefix, names)
            }
            Cache::Rotating(cache) => {
                kv_snapshot::snapshot_rotating(cache, snapshot, prefix, names)
            }
        }
    }

    /// Restore this layer's state from `snapshot` under `prefix` (issue #1335).
    ///
    /// Used by: Gemma 3, AFMoE, Kolibri-1.
    pub(crate) fn restore_from(
        &mut self,
        snapshot: &ModelStateSnapshot,
        prefix: &str,
        names: KvSnapshotNames,
    ) -> Result<(), String> {
        match self {
            Cache::Standard(cache) => kv_snapshot::restore_standard(cache, snapshot, prefix, names),
            Cache::Rotating(cache) => kv_snapshot::restore_rotating(cache, snapshot, prefix, names),
        }
    }

    /// Restore this layer from a ring checkpoint and its donor (issue #3255):
    /// a sliding layer from the checkpoint's ring, a global layer from the
    /// donor cut back to `target` tokens.
    ///
    /// Used by: Gemma 3, AFMoE.
    pub(crate) fn restore_from_ring_checkpoint(
        &mut self,
        checkpoint: &ModelStateSnapshot,
        donor: &ModelStateSnapshot,
        prefix: &str,
        target: i32,
        names: KvSnapshotNames,
    ) -> Result<(), String> {
        match self {
            Cache::Standard(cache) => {
                kv_ring_checkpoint::restore_standard_from_donor(cache, donor, prefix, target, names)
            }
            Cache::Rotating(cache) => kv_ring_checkpoint::restore_rotating_from_checkpoint(
                cache, checkpoint, donor, prefix, target, names,
            ),
        }
    }

    /// Whether the layer state stored under `prefix` can be restored covering
    /// only its first `target_len` tokens (issue #1335).
    ///
    /// Answers from the snapshot's own scalars rather than from a live cache,
    /// because the prompt-cache store has to decide before it allocates a
    /// sequence. Global layers hold every token at its own slot and truncate
    /// unconditionally; sliding layers only while their ring is unwrapped.
    ///
    /// Used by: Gemma 3, AFMoE, Kolibri-1.
    pub(crate) fn snapshot_truncatable_to(
        snapshot: &ModelStateSnapshot,
        prefix: &str,
        target_len: i32,
        names: KvSnapshotNames,
    ) -> bool {
        // Each helper is vacuously true when its own tensors are absent, so
        // the conjunction reduces to whichever arm this layer stored.
        kv_snapshot::rotating_truncatable_to(snapshot, prefix, target_len, names)
            && kv_snapshot::standard_truncatable_to(snapshot, prefix, target_len, names)
    }

    /// Drop everything past `target_len` tokens, leaving the cache in the state
    /// it would have had if only the first `target_len` tokens had ever been
    /// processed (issue #1335).
    ///
    /// Used by: Gemma 3, AFMoE, Kolibri-1.
    pub(crate) fn truncate_to(
        &mut self,
        target_len: i32,
        names: KvSnapshotNames,
    ) -> Result<(), String> {
        match self {
            Cache::Standard(cache) => kv_snapshot::truncate_standard(cache, target_len, names),
            Cache::Rotating(cache) => kv_snapshot::truncate_rotating(cache, target_len, names),
        }
    }

    pub(crate) fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            Cache::Standard(c) => c,
            Cache::Rotating(c) => c,
        }
    }

    pub(crate) fn offset(&self) -> i32 {
        match self {
            Cache::Standard(c) => c.offset,
            Cache::Rotating(c) => c.offset,
        }
    }

    /// Keys the next decode step attends: the live length of a global layer,
    /// the visible window of a sliding one. Used by: `sync_sequence_storage`
    /// (the pool's per-sequence length mirror).
    pub(crate) fn visible_len(&self) -> usize {
        match self {
            Cache::Standard(c) => c.seq_len().max(0) as usize,
            Cache::Rotating(c) => c.visible_len().max(0) as usize,
        }
    }
}

/// The batched decode entry for one Gemma 3 layer row (ADR 0008, #2172): a
/// global layer attends through `KVCache::attend`, a sliding layer through
/// `RotatingKVCache::attend`. Both are model-owned dense storage, never
/// pool-backed, so the kernel is the dense one for the layer's mode, and the
/// sliding write still goes through `RotatingKVCache::update_and_fetch`, the
/// path the decode undo log (#2182) hooks.
impl KvAttention for Cache {
    fn attend(
        &mut self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        match self {
            Cache::Standard(c) => KVCache::attend(c, q, k, v, scale, mask),
            Cache::Rotating(c) => RotatingKVCache::attend(c, q, k, v, scale, mask),
        }
    }
}

/// The per-layer cache a Gemma 3 forward runs over: the model-owned [`Cache`]
/// of the families that still own their state (AFMoE, Kolibri-1, the pipeline
/// stage and the backbone), and the scheduler-owned [`LayerState`] of Gemma 3
/// itself (issue #3254). Both hold a `KVCache` on a global layer and a
/// `RotatingKVCache` on a sliding one, so the forward is written once.
pub(crate) trait SlidingLayerCache: KvAttention {
    /// The single-row cache surface of this layer.
    fn as_interface(&mut self) -> &mut dyn CacheInterface;
    /// Tokens this layer has consumed.
    fn layer_offset(&self) -> i32;
    /// The sliding ring, `None` on a global layer.
    fn rotating(&self) -> Option<&RotatingKVCache>;
}

impl SlidingLayerCache for Cache {
    fn as_interface(&mut self) -> &mut dyn CacheInterface {
        Cache::as_interface(self)
    }

    fn layer_offset(&self) -> i32 {
        self.offset()
    }

    fn rotating(&self) -> Option<&RotatingKVCache> {
        match self {
            Cache::Rotating(c) => Some(c),
            Cache::Standard(_) => None,
        }
    }
}

impl SlidingLayerCache for LayerState {
    /// Gemma 3's table holds only `Full` and `Sliding` layers, so the other
    /// storage variants never reach its forward.
    fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            LayerState::Kv(c) => c,
            LayerState::Rotating(c) => c,
            other => panic!(
                "Gemma 3 forward over a {} layer: the layer state does not match layer_state_kinds()",
                other.storage_name()
            ),
        }
    }

    fn layer_offset(&self) -> i32 {
        self.offset()
    }

    fn rotating(&self) -> Option<&RotatingKVCache> {
        self.as_rotating()
    }
}

/// Whether a sliding-layer prefill of `seq_len` rows after `sliding_live_len`
/// live keys plans the native window route (issue #2621). When it does, the
/// forward builds no sliding mask: `RotatingKVCache::attend` carries the
/// window, and `causal_attention`'s over-window branch applies the band
/// through MLX's native `window_size` entry, or, should the per-layer
/// decision decline after all, through the same full-width mask this model
/// built before, so a disagreement costs a mask build, never semantics.
///
/// `attn` is any sliding layer's attention (the head counts and head dim are
/// shared), `h` the hidden state whose dtype the projections keep, and
/// `caches` the forward's caches. The plan reads the K/V dtype the sliding
/// caches return ([`sliding_caches_kv_dtype`]), so a `Turbo4Asym` ring under
/// bf16 activations plans the legacy route here, with one mask per forward,
/// instead of planning native and then declining at every sliding layer.
/// Sliding layers whose caches return different dtypes (Boundary-V keeps the
/// outer layers at FP16 under a Turbo mode) also plan legacy, since no one
/// shape describes them all.
///
/// Used by: [`text_prefill_sliding_mask`].
fn sliding_prefill_plans_native<C: SlidingLayerCache>(
    attn: &Attention,
    h: &MlxArray,
    caches: &[C],
    seq_len: i32,
    sliding_live_len: i32,
    window: i32,
) -> bool {
    let activation_dtype = mlxcel_core::array_dtype(h);
    let Some(kv_dtype) = sliding_caches_kv_dtype(caches, activation_dtype) else {
        return false;
    };
    let shape = WindowSdpaShape::prefill(
        mlxcel_core::array_shape(h)[0],
        attn.num_heads,
        attn.num_kv_heads,
        seq_len,
        window_sdpa::rotating_prefill_key_len(seq_len, sliding_live_len, window),
        attn.head_dim,
        activation_dtype,
    )
    .with_kv_dtype(kv_dtype);
    window_sdpa::plan(&shape, window, WindowSdpaExtras::default()).is_native()
}

/// The sliding-layer mask of a text-only prefill of `seq_len` rows after
/// `sliding_live_len` live keys, or `None` when the attention kernels derive
/// it themselves. Global layers never take a mask on this path: a plain
/// bottom-right causal mask goes through MLX's maskless causal SDPA (mlx-lm
/// passes `"causal"`). The sliding mask is also left `None` while every live
/// key is inside the window, and past the window when the native window route
/// is planned ([`sliding_prefill_plans_native`]); otherwise it is the dense
/// band.
///
/// Every Gemma 3 text forward builds its prefill masks here so they all run
/// the same attention kernels. A pipeline stage that built explicit masks
/// instead took the masked SDPA path while the single-process model took the
/// maskless one, and its logits drifted from the dense ones (issue #2654).
///
/// `sliding_attn` is any sliding layer's attention and `sliding_live_len` the
/// live length of a sliding cache.
///
/// Used by: `Gemma3Model::forward_with_caches_and_embeddings`,
/// `Gemma3StageModel::execute_hidden`, `Gemma3Backbone::forward_embeds`.
pub(crate) fn text_prefill_sliding_mask<C: SlidingLayerCache>(
    sliding_attn: &Attention,
    h: &MlxArray,
    caches: &[C],
    seq_len: i32,
    sliding_live_len: i32,
    window: i32,
) -> Option<UniquePtr<MlxArray>> {
    if sliding_live_len + seq_len <= window
        || sliding_prefill_plans_native(sliding_attn, h, caches, seq_len, sliding_live_len, window)
    {
        None
    } else {
        Some(create_sliding_window_prefill_mask(
            seq_len,
            sliding_live_len,
            window,
        ))
    }
}

/// The sliding-layer form of an external (VLM) prefill mask: `external` with
/// every key `window` or more positions behind its query masked, the band
/// `create_sliding_window_prefill_mask` applies (`q - k < window`), or `None`
/// when the key axis fits the window and the band would mask nothing.
///
/// transformers composes Gemma 3's image-prompt masks the same way: the
/// full-attention layers use `causal OR same_image_block`, the sliding layers
/// `sliding_window AND (causal OR same_image_block)`. Queries are aligned to
/// the end of the key axis, so a mask over prior keys keeps its positions. A
/// boolean mask stays boolean; an additive one gets `f32::MIN` outside the
/// band.
///
/// Used by: `Gemma3Model::hidden_with_caches_and_embeddings`.
pub(crate) fn external_mask_with_window_band(
    external: &MlxArray,
    window: i32,
) -> Option<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(external);
    let (q_len, k_len) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    if k_len <= window {
        return None;
    }
    let query_pos = mlxcel_core::reshape(
        &mlxcel_core::arange_i32(k_len - q_len, k_len, 1),
        &[q_len, 1],
    );
    let key_pos = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, k_len, 1), &[1, k_len]);
    let window_arr = mlxcel_core::from_slice_i32(&[window], &[1]);
    let in_band = mlxcel_core::less(&mlxcel_core::subtract(&query_pos, &key_pos), &window_arr);
    Some(
        if mlxcel_core::array_dtype(external) == mlxcel_core::dtype::BOOL {
            mlxcel_core::logical_and(external, &in_band)
        } else {
            let masked = mlxcel_core::from_slice_f32(&[f32::MIN], &[1]);
            mlxcel_core::where_cond(&in_band, external, &masked)
        },
    )
}

/// [`mlxcel_core::window_sdpa::sliding_caches_kv_dtype`] over this model's
/// sliding (`Rotating`) caches. Used by: `sliding_prefill_plans_native`
/// (issue #2621).
pub(crate) fn sliding_caches_kv_dtype<C: SlidingLayerCache>(
    caches: &[C],
    activation_dtype: i32,
) -> Option<i32> {
    mlxcel_core::window_sdpa::sliding_caches_kv_dtype(
        caches.iter().filter_map(SlidingLayerCache::rotating),
        activation_dtype,
    )
}

// Gemma3 Model.
pub struct Gemma3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: GemmaRMSNorm,
    pub lm_head: UnifiedLinear,
    /// Validated at load ([`ModelArgs::validated_sliding_window`]).
    pub sliding_window: i32,
    pub sliding_window_pattern: usize,
    pub hidden_size: usize,
}

impl Gemma3Model {
    /// Get token embeddings scaled by sqrt(hidden_size)
    /// Used by: VisionModule for merging vision and text embeddings
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        let scale = (self.hidden_size as f32).sqrt();
        mlxcel_core::multiply_scalar(&h, scale)
    }

    /// Forward pass with pre-computed embeddings (for VLM)
    /// If input_embeddings is Some, skip embed_tokens and use provided embeddings
    pub(crate) fn forward_with_caches_and_embeddings<C: SlidingLayerCache>(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [C],
        external_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_with_caches_and_embeddings(
            input_ids,
            input_embeddings,
            caches,
            external_mask,
        );
        self.lm_head.forward(&h)
    }

    /// Logits `[B, 1, vocab]` for position `last_pos` only: the final-normed
    /// hidden state is sliced before the LM head, which acts per position, so
    /// a prefill does not project every prompt row through a 262k vocabulary.
    pub(crate) fn last_logits_with_caches_and_embeddings<C: SlidingLayerCache>(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [C],
        external_mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_with_caches_and_embeddings(
            input_ids,
            input_embeddings,
            caches,
            external_mask,
        );
        let row = mlxcel_core::generate::logits_at_position(&h, last_pos);
        self.lm_head.forward(&row)
    }

    /// Embeddings (scaled by sqrt(hidden_size)), every layer and the final
    /// norm.
    fn hidden_with_caches_and_embeddings<C: SlidingLayerCache>(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [C],
        external_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape[1];

        // Use pre-computed embeddings if provided, otherwise embed tokens.
        // In both cases, apply Gemma's sqrt(hidden_size) normalizer.
        // Python: h *= mx.array(self.args.hidden_size**0.5, mx.bfloat16)
        let mut h = if let Some(embeddings) = input_embeddings {
            mlxcel_core::copy(embeddings)
        } else {
            self.embed_tokens.forward(input_ids)
        };
        // Gemma scaling: h *= sqrt(hidden_size)
        let scale = (self.hidden_size as f32).sqrt();
        h = mlxcel_core::multiply_scalar(&h, scale);

        let n = self.layers.len();
        if let Some(ext_mask) = external_mask {
            // A VLM prefill hands in its own 4D mask (for Gemma 3 image
            // prompts: causal plus bidirectional image blocks, built by
            // `vision::merge`). As in transformers, the global layers take it
            // as is and the sliding layers take it intersected with their
            // window band; the band is built once per forward, and only when
            // the key axis is wider than the window.
            let sliding_ext_mask = if self.sliding_window_pattern > 1 {
                external_mask_with_window_band(ext_mask, self.sliding_window)
            } else {
                None
            };
            for (i, layer) in self.layers.iter().enumerate() {
                let is_global =
                    (i % self.sliding_window_pattern) == (self.sliding_window_pattern - 1);
                let mask = if is_global {
                    ext_mask
                } else {
                    sliding_ext_mask.as_deref().unwrap_or(ext_mask)
                };
                h = layer.forward(&h, caches[i].as_interface(), Some(mask));
                pipeline_hint(&h, i, n);
            }
        } else if seq_len == 1 {
            // Decode path (seq_len=1): no mask needed — matches Python mlx-lm
            // which returns None from create_attention_mask when N=1.
            // The fused SDPA handles single-token attention without explicit masks.
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward(&h, caches[i].as_interface(), None);
                pipeline_hint(&h, i, n);
            }
        } else {
            // Prefill path (seq_len > 1): create causal masks.
            //
            // Size both prefill masks from the cache's live window
            // (`live_len() = offset - live_start`), not the monotonic
            // `offset`. Under `--max-kv-size`, `trim_front` slices the global
            // (full-attention) `KVCache` to its live window and advances
            // `live_start` while `offset` keeps growing to preserve the RoPE
            // relative positions, so `update_and_fetch` returns only
            // `live_len` keys. A mask sized from `offset` would be wider than
            // the returned K/V and trip `broadcast_shapes`. With no trim
            // (`live_start == 0`), `live_len == offset`, so this is
            // byte-identical to the untrimmed path. RoPE keeps reading each
            // layer's own monotonic `offset` inside the attention layer. See
            // issue #430 (mirrors #419/#420, #421/#422).
            //
            // A mask that is plain bottom-right causal is left `None`, and the
            // attention layer takes MLX's maskless causal SDPA instead of
            // reading an `[l, l + offset]` array (as mlx-lm passes `"causal"`
            // for `N <= window`). That is always the case for the global
            // layers, and for the sliding layers until the live keys outgrow
            // the window.
            let global_mask: Option<UniquePtr<MlxArray>> = None;

            let sliding_mask = if self.sliding_window_pattern > 1 {
                let sliding_live_len = caches[0].as_interface().live_len();
                text_prefill_sliding_mask(
                    &self.layers[0].self_attn,
                    &h,
                    caches,
                    seq_len,
                    sliding_live_len,
                    self.sliding_window,
                )
            } else {
                None
            };

            for (i, layer) in self.layers.iter().enumerate() {
                let is_global =
                    (i % self.sliding_window_pattern) == (self.sliding_window_pattern - 1);
                let mask = if is_global {
                    global_mask.as_ref().map(|m| m.as_ref().unwrap())
                } else {
                    sliding_mask.as_ref().map(|m| m.as_ref().unwrap())
                };
                h = layer.forward(&h, caches[i].as_interface(), mask);
                pipeline_hint(&h, i, n);
            }
        }

        // Final norm
        self.norm.forward(&h)
    }

    /// Forward pass through the entire model
    #[cfg(test)]
    pub(crate) fn forward_with_caches<C: SlidingLayerCache>(
        &self,
        input_ids: &MlxArray,
        caches: &mut [C],
    ) -> UniquePtr<MlxArray> {
        self.forward_with_caches_and_embeddings(input_ids, None, caches, None)
    }

    /// Create KV caches for all layers
    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        self.make_caches_with_modes(None)
    }

    pub(crate) fn make_caches_with_modes(&self, modes: Option<&KvCacheLayerModes>) -> Vec<Cache> {
        (0..self.layers.len())
            .map(|i| {
                let is_global =
                    (i % self.sliding_window_pattern) == (self.sliding_window_pattern - 1);
                if is_global {
                    let mode = modes
                        .map(|modes| modes.mode_for_layer(i))
                        .unwrap_or(KVCacheMode::Fp16);
                    Cache::Standard(KVCache::new_with_mode(mode))
                } else {
                    let window = self.sliding_window;
                    let mode = modes.map_or(KVCacheMode::Fp16, |modes| {
                        modes.rotating_mode_for_layer(i, window)
                    });
                    Cache::Rotating(RotatingKVCache::new_with_mode(window, mode))
                }
            })
            .collect()
    }

    fn forward_batched_decode_with_caches<C: SlidingLayerCache, R: AsMut<[C]>>(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [R],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        let scale = (self.hidden_size as f32).sqrt();
        h = mlxcel_core::multiply_scalar(&h, scale);

        let n = self.layers.len();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut layer_caches: Vec<&mut C> = batch_caches
                .iter_mut()
                .map(|caches| &mut caches.as_mut()[layer_idx])
                .collect();
            h = layer.forward_batched_decode(&h, &mut layer_caches);
            pipeline_hint(&h, layer_idx, n);
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
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

        // Load weights (with tied-embedding sanitization)
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let sliding_window = args.validated_sliding_window()?;
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
        let norm = GemmaRMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window,
            sliding_window_pattern: args.sliding_window_pattern,
            hidden_size: args.hidden_size,
        })
    }
}

pub(crate) struct Gemma3StageModel {
    filter: LayerFilter,
    embed_tokens: Option<UnifiedEmbedding>,
    layers: Vec<TransformerBlock>,
    norm: Option<GemmaRMSNorm>,
    lm_head: Option<UnifiedLinear>,
    sliding_window: i32,
    sliding_window_pattern: usize,
    hidden_size: usize,
}

impl Gemma3StageModel {
    pub(crate) fn load(
        model_dir: &Path,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read {}: {}", config_path.display(), e))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse {}: {}", config_path.display(), e))?;

        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        filter_weight_map(&mut weights, filter);
        Self::from_filtered_weights(&weights, &args, filter, stage_index)
    }

    pub(crate) fn from_filtered_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let sliding_window = args.validated_sliding_window()?;
        let group_size = args.group_size();
        let bits = args.bits();
        let embed_tokens = if filter.has_embedding {
            Some(UnifiedEmbedding::from_weights(
                weights,
                "model.embed_tokens",
                group_size,
                bits,
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            layers.push(TransformerBlock::from_weights(weights, args, layer_idx)?);
        }

        if layers.is_empty() {
            return Err(format!(
                "stage {} did not load any layers from range {}..{}",
                stage_index, filter.layer_range.start, filter.layer_range.end
            ));
        }

        let (norm, lm_head) = if filter.has_lm_head {
            let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
            let norm = GemmaRMSNorm::new(norm_weight, args.rms_norm_eps);
            let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;
            (Some(norm), Some(lm_head))
        } else {
            (None, None)
        };

        Ok(Self {
            filter: filter.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window,
            sliding_window_pattern: args.sliding_window_pattern,
            hidden_size: args.hidden_size,
        })
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        (0..self.layers.len())
            .map(|layer_idx| {
                let global_idx = self.global_layer_index(layer_idx);
                let is_global =
                    (global_idx % self.sliding_window_pattern) == (self.sliding_window_pattern - 1);
                if is_global {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window))
                }
            })
            .collect()
    }

    pub(crate) fn execute_from_token_ids(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
        last_pos: Option<usize>,
    ) -> Result<StageExecutionOutput, String> {
        let mut hidden = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| {
                "stage does not host embeddings; hidden-state input required".to_string()
            })?
            .forward(input_ids);
        hidden = mlxcel_core::multiply_scalar(&hidden, (self.hidden_size as f32).sqrt());
        self.execute_hidden(hidden, caches, last_pos)
    }

    pub(crate) fn execute_from_hidden_states(
        &self,
        hidden: UniquePtr<MlxArray>,
        caches: &mut [Cache],
        last_pos: Option<usize>,
    ) -> Result<StageExecutionOutput, String> {
        if self.filter.has_embedding {
            return Err("entry stage expects token IDs, not hidden states".to_string());
        }
        self.execute_hidden(hidden, caches, last_pos)
    }

    /// `last_pos` asks the final stage to project only that hidden row through
    /// the norm and LM head (#3081); a non-final stage ignores it.
    fn execute_hidden(
        &self,
        mut hidden: UniquePtr<MlxArray>,
        caches: &mut [Cache],
        last_pos: Option<usize>,
    ) -> Result<StageExecutionOutput, String> {
        if caches.len() != self.layers.len() {
            return Err(format!(
                "stage cache count mismatch: expected {}, got {}",
                self.layers.len(),
                caches.len()
            ));
        }

        let seq_len = mlxcel_core::array_shape(hidden.as_ref().unwrap())[1];
        let n = self.layers.len();
        if seq_len > 1 {
            // The same masks the single-process forward builds, from the
            // stage's own first sliding layer and cache: global layers take
            // the maskless causal SDPA, and sliding layers a dense band only
            // past the window when the native route is not planned. Building
            // explicit masks here instead ran different attention kernels
            // than the dense model and broke remote parity (issue #2654).
            let sliding_mask = self.first_sliding_cache_index().and_then(|sliding_idx| {
                let sliding_live_len = caches[sliding_idx].as_interface().live_len();
                text_prefill_sliding_mask(
                    &self.layers[sliding_idx].self_attn,
                    hidden.as_ref().unwrap(),
                    caches,
                    seq_len,
                    sliding_live_len,
                    self.sliding_window,
                )
            });

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mask = if self.is_global_layer(layer_idx) {
                    None
                } else {
                    sliding_mask.as_deref()
                };
                hidden = layer.forward(
                    hidden.as_ref().unwrap(),
                    caches[layer_idx].as_interface(),
                    mask,
                );
                pipeline_hint(&hidden, layer_idx, n);
            }
        } else {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward(
                    hidden.as_ref().unwrap(),
                    caches[layer_idx].as_interface(),
                    None,
                );
                pipeline_hint(&hidden, layer_idx, n);
            }
        }

        match (&self.norm, &self.lm_head) {
            (Some(norm), Some(lm_head)) => {
                let hidden = match last_pos {
                    Some(pos) => {
                        mlxcel_core::generate::logits_at_position(hidden.as_ref().unwrap(), pos)
                    }
                    None => hidden,
                };
                let hidden = norm.forward(hidden.as_ref().unwrap());
                Ok(StageExecutionOutput::Logits(lm_head.forward(&hidden)))
            }
            _ => Ok(StageExecutionOutput::HiddenStates(hidden)),
        }
    }

    fn global_layer_index(&self, local_idx: usize) -> usize {
        self.filter.layer_range.start + local_idx
    }

    fn is_global_layer(&self, local_idx: usize) -> bool {
        let global_idx = self.global_layer_index(local_idx);
        (global_idx % self.sliding_window_pattern) == (self.sliding_window_pattern - 1)
    }

    fn first_sliding_cache_index(&self) -> Option<usize> {
        (0..self.layers.len()).find(|&idx| !self.is_global_layer(idx))
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

/// Wrapper for Gemma3Model that implements LanguageModel trait.
///
/// Gemma 3 is scheduler-owned (issue #3254, ADR 0010): a sequence's sliding
/// and global caches are the [`LayerState`] vector its pool entry holds, built
/// by [`LanguageModel::make_layer_state`] and handed to
/// [`LanguageModel::forward_layer_state`]. The wrapper keeps only the id-less
/// fallback slot a bare `forward` runs on.
pub struct Gemma3Wrapper {
    model: Gemma3Model,
    /// State of the id-less single-sequence run (a bare `forward`, the CLI
    /// drafter and benchmarks), which no pool entry backs.
    fallback: RefCell<Vec<LayerState>>,
    kv_cache_layer_modes: KvCacheLayerModes,
}

impl Gemma3Wrapper {
    pub fn new(model: Gemma3Model) -> Self {
        let kv_cache_layer_modes = KvCacheLayerModes::new();
        let fallback = RefCell::new(Self::build_layer_state(&model, &kv_cache_layer_modes));
        Self {
            model,
            fallback,
            kv_cache_layer_modes,
        }
    }

    /// What layer `layer_idx` keeps: a sliding ring or a full cache.
    fn layer_kind(model: &Gemma3Model, layer_idx: usize) -> LayerStateKind {
        let pattern = model.sliding_window_pattern;
        if (layer_idx % pattern) != (pattern - 1) {
            LayerStateKind::sliding(model.sliding_window)
        } else {
            LayerStateKind::Full
        }
    }

    /// Fresh state for every layer under the injected modes, with no decode
    /// undo log. A misaligned `Turbo4Asym` window falls back to FP16 with one
    /// warning per table (`KvCacheLayerModes::rotating_mode_for_layer`).
    fn build_layer_state(model: &Gemma3Model, modes: &KvCacheLayerModes) -> Vec<LayerState> {
        (0..model.layers.len())
            .map(|layer_idx| {
                let kind = Self::layer_kind(model, layer_idx);
                let mode = match kind {
                    LayerStateKind::Sliding { window } => {
                        modes.rotating_mode_for_layer(layer_idx, window)
                    }
                    _ => modes.mode_for_layer(layer_idx),
                };
                LayerState::new(kind, mode)
            })
            .collect()
    }

    pub fn reset_caches(&self) {
        *self.fallback.borrow_mut() =
            Self::build_layer_state(&self.model, &self.kv_cache_layer_modes);
    }

    /// Every sliding layer stores plain FP16 K/V, the only storage the
    /// rotating undo log covers (INT8 and Turbo4Asym keep sidecars).
    fn supports_lookahead_rewind(&self) -> bool {
        (0..self.model.layers.len()).all(|layer_idx| {
            !self.is_sliding_layer(layer_idx)
                || self.kv_cache_layer_modes.mode_for_layer(layer_idx) == KVCacheMode::Fp16
        })
    }

    /// Whether layer `layer_idx` is a sliding-window layer.
    fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        matches!(
            Self::layer_kind(&self.model, layer_idx),
            LayerStateKind::Sliding { .. }
        )
    }

    /// One sequence's forward over its layer state, then the undo-row copies
    /// the forward recorded. A text forward ignores the caller's mask and
    /// builds its own causal and sliding-window masks; an embedding (VLM)
    /// forward passes the mask on.
    fn forward_row(
        &self,
        input_ids: &MlxArray,
        request: &LayerStateForward<'_>,
        state: &mut [LayerState],
    ) -> UniquePtr<MlxArray> {
        let mask = request.input_embeddings.and(request.mask);
        let logits = match request.last_pos {
            Some(last_pos) => self.model.last_logits_with_caches_and_embeddings(
                input_ids,
                request.input_embeddings,
                state,
                mask,
                last_pos,
            ),
            None => self.model.forward_with_caches_and_embeddings(
                input_ids,
                request.input_embeddings,
                state,
                mask,
            ),
        };
        LayerState::flush_decode_undo(state.iter_mut());
        logits
    }

    /// The id-less forward over the fallback slot.
    fn forward_fallback(&self, request: &LayerStateForward<'_>) -> UniquePtr<MlxArray> {
        let mut state = self.fallback.borrow_mut();
        self.forward_row(request.input_ids, request, &mut state)
    }

    /// A `KVCache`-slice entry reached with a scheduler sequence id. The
    /// engine forwards a scheduler-owned sequence through
    /// `forward_layer_state`, so this is a caller that bypassed the pool.
    fn refuse_sequence_id(seq_id: SequenceId) -> ! {
        panic!(
            "Gemma 3 state is scheduler-owned: sequence {seq_id} must be forwarded through \
             forward_layer_state with its pool layer state"
        )
    }
}

impl mlxcel_core::generate::LanguageModel for Gemma3Wrapper {
    fn batched_decode_kind(&self) -> mlxcel_core::generate::BatchedDecodeKind {
        mlxcel_core::generate::BatchedDecodeKind::Native
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_sequence_id(input_ids, None, caches, None)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            None,
            caches,
            mask,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        // Return RAW embeddings (without sqrt(hidden_size) scaling).
        // Python mlx-vlm calls embed_tokens() to get unscaled embeddings for VLM merge,
        // then the language model forward applies sqrt(hidden_size) to ALL embeddings
        // (both text and merged image features). Returning scaled embeddings here would
        // cause double-scaling for text tokens while leaving image features unscaled.
        Some(self.model.embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        Vec::new()
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.kv_cache_layer_modes.set(modes);
        self.reset_caches();
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.kv_cache_layer_modes.clone_modes()
    }

    /// Sliding layers hold a `RotatingKVCache`, which stores FP16 for every
    /// mode but an aligned `Turbo4Asym` (issue #2783). Derived from the same
    /// table the pool allocates from.
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        self.kv_cache_layer_modes
            .storage_modes(|layer_idx| Self::layer_kind(&self.model, layer_idx))
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// Full attention on every `sliding_window_pattern`-th layer, a sliding
    /// window on the rest (issue #3254). The default
    /// `sequence_state_layout()` reads this and answers scheduler-owned.
    fn layer_state_kinds(&self) -> Option<Vec<LayerStateKind>> {
        Some(
            (0..self.model.layers.len())
                .map(|layer_idx| Self::layer_kind(&self.model, layer_idx))
                .collect(),
        )
    }

    /// State for one scheduler sequence. When the decode lookahead can be
    /// rewound, every sliding layer logs its newest single-token writes deep
    /// enough for any lookahead teardown (issue #2159).
    fn make_layer_state(&self) -> Option<Vec<LayerState>> {
        let mut state = Self::build_layer_state(&self.model, &self.kv_cache_layer_modes);
        if self.supports_lookahead_rewind() {
            for layer in &mut state {
                if let LayerState::Rotating(rotating) = layer {
                    rotating.set_decode_undo_depth(DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS);
                }
            }
        }
        Some(state)
    }

    fn layer_state_snapshot_spec(&self) -> Option<LayerStateSnapshotSpec> {
        Some(GEMMA3_LAYER_STATE_SNAPSHOT)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    /// Gemma 3 takes NA tile-aligned padded prefill (issue #1755).
    ///
    /// Its K/V is the sequence's layer state, not a `KVCache` slice, so the
    /// engine drops the pad positions from every layer with
    /// [`LayerState::rewind_padded_prefill`] (the CLI fallback slot through
    /// [`Self::trim_state`] with `None`), which leaves every layer as an
    /// unpadded prefill would, `offset` included.
    ///
    /// The text forwards ignore the caller's padding mask and build their own
    /// causal and sliding-window masks over the padded chunk. Pad positions
    /// trail the real ones, so causal attention already keeps them out of every
    /// real row, and the sliding layers keep their window.
    ///
    /// A sliding layer stored as Turbo4Asym cannot be rewound (its V side is
    /// packed into sidecars), so that configuration declines padding instead.
    fn supports_padded_prefill(&self) -> bool {
        !(0..self.model.layers.len()).any(|layer_idx| {
            self.is_sliding_layer(layer_idx)
                && self.kv_cache_layer_modes.mode_for_layer(layer_idx) == KVCacheMode::Turbo4Asym
        })
    }

    /// `None` rewinds the fallback slot. A scheduler sequence's layer state
    /// is rewound by the engine, which holds it, so `Some` is refused.
    fn trim_state(&self, seq: Option<SequenceId>, excess: i32) -> Result<(), String> {
        if let Some(seq_id) = seq {
            return Err(format!(
                "Gemma 3 pad trim: sequence {seq_id} is scheduler-owned and is trimmed through \
                 its layer state"
            ));
        }
        if excess <= 0 {
            return Ok(());
        }
        for (layer_idx, layer) in self.fallback.borrow_mut().iter_mut().enumerate() {
            layer
                .rewind_padded_prefill(excess)
                .map_err(|err| format!("Gemma 3 pad trim, layer {layer_idx}: {err}"))?;
        }
        Ok(())
    }

    /// Gemma 3 decodes on the scheduler's lookahead pipeline (issue #2159).
    /// Its global layers are plain `KVCache`s whose trim is exact, and its
    /// sliding layers log the rows each decode write overwrites, so a
    /// teardown restores a wrapped window exactly. Declined unless every
    /// sliding layer is FP16, the only storage the log covers. The engine
    /// runs the rewind over the sequence's layer state.
    fn supports_decode_lookahead_rewind(&self) -> bool {
        self.supports_lookahead_rewind()
    }

    /// Retained after the 2026-10-11 GB10 paired/null storage sweep in
    /// `docs/benchmark_results/versions/v0.8.0.md#model-owned-storage-gb10-2026-10-11`.
    /// Across 512/4096-token prompts at concurrency 1/4/8, neither backend
    /// cleared the paged-repeat null spread in any Gemma 3 cell.
    fn supports_paged_decode_backend(&self) -> bool {
        true
    }

    fn reset_runtime_state(&self) {
        // Used by: bare single-row runs outside an engine sequence. Reset the
        // fallback slot before each fresh CLI / benchmark generation so a
        // second VLM prefill does not reuse offsets from a previous run with
        // a stale 4D attention mask.
        self.reset_caches();
    }

    /// Snapshots are taken, restored, truncated and checkpointed by the
    /// prompt cache's generic code over the sequence's layer state
    /// (`server::prompt_cache::state_snapshot`), under
    /// [`Self::layer_state_snapshot_spec`].
    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn forward_layer_state(
        &self,
        request: &LayerStateForward<'_>,
        rows: &mut [&mut [LayerState]],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(request.input_ids);
        if shape.len() != 2 || shape[0] as usize != rows.len() {
            return Err(format!(
                "Gemma 3 forward: input shape {shape:?} does not match {} state rows",
                rows.len()
            ));
        }
        let n_layers = self.model.layers.len();
        if let Some(row) = rows.iter().position(|row| row.len() != n_layers) {
            return Err(format!(
                "Gemma 3 forward: row {row} holds {} layers of state, the model has {n_layers}",
                rows[row].len()
            ));
        }
        if let [state] = rows {
            return Ok(self.forward_row(request.input_ids, request, state));
        }
        if shape[1] != 1 || request.mask.is_some() {
            // A multi-token batch runs row by row, each over its own state.
            if request.input_embeddings.is_some() || request.last_pos.is_some() {
                return Err(
                    "Gemma 3 forward: a multi-row batch takes token ids and returns every position"
                        .to_string(),
                );
            }
            let mut result: Option<UniquePtr<MlxArray>> = None;
            for (row_idx, state) in rows.iter_mut().enumerate() {
                let input = mlxcel_core::slice(
                    request.input_ids,
                    &[row_idx as i32, 0],
                    &[row_idx as i32 + 1, shape[1]],
                );
                let logits = self.forward_row(&input, request, state);
                result = Some(match result {
                    Some(acc) => mlxcel_core::concatenate(&acc, &logits, 0),
                    None => logits,
                });
            }
            return result.ok_or_else(|| "Gemma 3 forward: empty batch".to_string());
        }

        let logits = self
            .model
            .forward_batched_decode_with_caches(request.input_ids, rows);
        LayerState::flush_decode_undo(rows.iter_mut().flat_map(|row| row.iter_mut()));
        Ok(logits)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if let Some(seq_id) = seq_id {
            Self::refuse_sequence_id(seq_id);
        }
        self.forward_fallback(&LayerStateForward::tokens(input_ids))
    }

    // The three last-logits entry points mirror `forward`,
    // `forward_with_sequence_id` and `forward_with_embeddings_and_sequence_id`
    // (same fallback state, same mask handling) and slice before the LM head.
    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.forward_last_logits_with_sequence_id(input_ids, None, caches, None, last_pos)
    }

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        if let Some(seq_id) = seq_id {
            Self::refuse_sequence_id(seq_id);
        }
        self.forward_fallback(&LayerStateForward::tokens(input_ids).with_last_pos(Some(last_pos)))
    }

    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        if let Some(seq_id) = seq_id {
            Self::refuse_sequence_id(seq_id);
        }
        // This entry passes the mask on even without embeddings, as it did
        // while the state lived in the wrapper.
        let mut state = self.fallback.borrow_mut();
        let logits = self.model.last_logits_with_caches_and_embeddings(
            input_ids,
            input_embeddings,
            &mut state,
            mask,
            last_pos,
        );
        LayerState::flush_decode_undo(state.iter_mut());
        logits
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if let Some(seq_id) = seq_id {
            Self::refuse_sequence_id(seq_id);
        }
        let mut state = self.fallback.borrow_mut();
        let logits = self.model.forward_with_caches_and_embeddings(
            input_ids,
            input_embeddings,
            &mut state,
            mask,
        );
        LayerState::flush_decode_undo(state.iter_mut());
        logits
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0, 1, 106] // Gemma3: <pad> (0), <eos> (1), <end_of_turn> (106)
    }
}

#[cfg(test)]
mod gemma3_mask_tests {
    // These tests pin the shared `mlxcel_core::utils::create_sliding_window_prefill_mask`
    // helper (hoisted in #410) from the Gemma 3 caller's perspective: Gemma 3
    // forwards the mask straight to attention with no `trim_mask_to_keys` step,
    // so it relies on both the full over-window mask (fresh prefill) and the
    // window-clamped shape (rolled-over cache) the helper produces.
    use super::*;
    use mlxcel_core::utils::create_causal_mask;

    fn mask_at(mask: &MlxArray, q: i32, k: i32) -> f32 {
        let scalar = mlxcel_core::slice(mask, &[q, k], &[q + 1, k + 1]);
        mlxcel_core::item_f32(&scalar)
    }

    /// Issue #401: a fresh single-pass prefill (`sliding_offset == 0`) longer
    /// than the sliding window keeps every prefill key, so the mask must span
    /// the full `[seq_len, seq_len]` key axis and have no fully-masked row.
    #[test]
    fn sliding_prefill_mask_fresh_over_window_spans_full_key_axis() {
        let mask = create_sliding_window_prefill_mask(4, 0, 2);
        mlxcel_core::eval(&mask);
        assert_eq!(
            mlxcel_core::array_shape(&mask),
            vec![4, 4],
            "fresh over-window prefill mask must span the full key axis"
        );
        // Every query row must attend to at least itself (diagonal == 0.0),
        // proving no all-masked row that would softmax to NaN.
        for q in 0..4 {
            assert_eq!(mask_at(&mask, q, q), 0.0, "row {q} must attend to itself");
        }
    }

    /// Issue #678: a non-fresh multi-token append (`sliding_offset > 0`)
    /// exposes `min(sliding_offset, window - 1)` retained prior keys plus ALL
    /// new keys (the rotating cache no longer clamps its return to `window`),
    /// and Gemma 3 forwards the mask directly with no `trim_mask_to_keys`
    /// step. The mask must therefore span that exact key axis; the previous
    /// `[seq_len, window]` clamp starved the earliest rows of a
    /// `seq_len >= window` continuation chunk (all-masked rows -> NaN ->
    /// `<pad>`) and no longer broadcasts against the cache return.
    #[test]
    fn sliding_prefill_mask_nonfresh_over_window_matches_cache_return() {
        // window 2, prior 3 -> kept prior = 1; 4 new keys -> axis 5.
        let mask = create_sliding_window_prefill_mask(4, 3, 2);
        mlxcel_core::eval(&mask);
        assert_eq!(
            mlxcel_core::array_shape(&mask),
            vec![4, 5],
            "non-fresh over-window prefill mask must span kept prior keys plus all new keys"
        );
        // No row may be fully masked (the pre-#678 failure mode).
        for q in 0..4 {
            assert_eq!(
                mask_at(&mask, q, q + 1),
                0.0,
                "row {q} must attend to its own position"
            );
        }
    }

    // [H, n, D] K/V tensor for an `[1, H, n, D]` cache entry; only the key
    // axis (n) matters for these shape invariants.
    fn make_kv(h: i32, n: i32, d: i32, base: f32) -> UniquePtr<MlxArray> {
        let count = (h * n * d) as usize;
        let data: Vec<f32> = (0..count).map(|i| base + i as f32).collect();
        mlxcel_core::from_slice_f32(&data, &[1, h, n, d])
    }

    /// Regression for #430: the global (full-attention) prefill mask must be
    /// sized from the `Cache::Standard` live window via `CacheInterface::
    /// live_len()`, not the monotonic `offset()`. After a `--max-kv-size`
    /// `trim_front` (`live_start > 0`), `live_len < offset`, and a continuation
    /// chunk's `update_and_fetch` returns only `live_len + m` keys; a mask
    /// sized from `live_len` matches that axis while an `offset`-sized mask is
    /// strictly wider (the crash).
    #[test]
    fn global_prefill_mask_sizes_from_live_len_not_offset_after_trim() {
        const H: i32 = 2;
        const D: i32 = 4;
        let (n1, trimmed, m) = (8, 3, 4);

        let mut cache = Cache::Standard(KVCache::new());
        let _ = cache
            .as_interface()
            .update_and_fetch(make_kv(H, n1, D, 0.0), make_kv(H, n1, D, 100.0));
        let Cache::Standard(ref mut std_cache) = cache else {
            unreachable!("constructed as Standard")
        };
        assert_eq!(std_cache.trim_front(trimmed), trimmed);

        // RoPE invariant: `offset()` stays monotonic; the live window shrinks.
        let offset = cache.as_interface().offset();
        let live_len = cache.as_interface().live_len();
        assert_eq!(offset, n1, "offset stays monotonic after trim");
        assert_eq!(live_len, n1 - trimmed, "live_len == offset - live_start");
        assert!(live_len < offset, "trim must make live_len < offset");

        // What the continuation chunk's `update_and_fetch` actually returns.
        let (cont_k, _) = cache
            .as_interface()
            .update_and_fetch(make_kv(H, m, D, 200.0), make_kv(H, m, D, 300.0));
        let returned_klen = mlxcel_core::array_shape(&cont_k)[2];
        assert_eq!(
            returned_klen,
            live_len + m,
            "cache returns live_len + m keys"
        );

        // live_len-sized mask matches the returned K/V; offset-sized is wider.
        let live_mask = create_causal_mask(m, live_len);
        mlxcel_core::eval(&live_mask);
        assert_eq!(
            *mlxcel_core::array_shape(&live_mask).last().unwrap(),
            returned_klen,
            "create_causal_mask(m, live_len) key axis must match the returned K/V"
        );
        let offset_mask = create_causal_mask(m, offset);
        mlxcel_core::eval(&offset_mask);
        assert!(
            *mlxcel_core::array_shape(&offset_mask).last().unwrap() > returned_klen,
            "offset-sized mask must be wider than the returned K/V (the bug)"
        );
    }

    /// The sliding (`Cache::Rotating`) prefill mask must size from the same
    /// `CacheInterface::live_len()` accessor, which for a `RotatingKVCache`
    /// returns `visible_len()`, the count of prior keys `update_concat`
    /// actually concatenates. On a freshly concatenated cache that equals
    /// `seq_len()`, so this case pins the common value; the decode-grown case
    /// where the two diverge is covered by
    /// `rotating_live_len::live_len_matches_the_keys_a_multi_token_append_returns`.
    #[test]
    fn sliding_cache_live_len_matches_returned_keys() {
        const H: i32 = 2;
        const D: i32 = 4;
        let window = 6;
        let m = 4;

        let mut cache = Cache::Rotating(RotatingKVCache::new(window));
        let (k, _) = cache
            .as_interface()
            .update_and_fetch(make_kv(H, m, D, 0.0), make_kv(H, m, D, 100.0));
        let returned_klen = mlxcel_core::array_shape(&k)[2];
        assert_eq!(
            cache.as_interface().live_len(),
            returned_klen,
            "RotatingKVCache live_len() (== visible_len) must equal the returned key axis"
        );
    }
}

#[cfg(test)]
#[path = "gemma3_tests.rs"]
pub(super) mod gemma3_tests;

#[cfg(test)]
#[path = "gemma3_stage_tests.rs"]
pub(crate) mod gemma3_stage_tests;
