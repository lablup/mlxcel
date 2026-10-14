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

//! Reusable Gemma 3 decoder stack driven by caller-supplied embeddings.
//!
//! Ports the pieces of `mlx_vlm.models.gemma3.language.Gemma3Model` that a
//! host model uses when it builds the backbone with
//! `scale_inputs_embeds=False`, deletes `embed_tokens`, and calls it as
//! `backbone(None, inputs_embeds=..., cache=...)`: the transformer layers,
//! the final `(1 + weight)` RMSNorm, the sliding/global cache split, and the
//! causal / sliding-window mask choice. Unlike
//! [`crate::models::gemma3::Gemma3Model`], this stack does **not** multiply
//! the injected embeddings by `sqrt(hidden_size)`, has no token embedding and
//! no LM head, and reads its layers from an arbitrary key prefix.
//!
//! Used by: the Nemotron VoiceChat EAR-TTS decoder
//! (`crate::models::nemotron_voicechat::tts`), whose backbone lives under
//! `tts_model.tts_model.backbone`.
//!
//! The layers reuse [`TransformerBlock`]'s weights and attention unchanged,
//! so the per-layer RoPE base/scale rule, fused QKV projection, and SDPA
//! dispatch are the same as the Gemma 3 text model's. The MLP activation is
//! the one exception: it runs [`gelu_approx`], an op-for-op port of
//! `mlx.nn.gelu_approx`, instead of the text model's fused GeGLU kernel,
//! because the two round differently and a host that samples from the
//! backbone state (the EAR-TTS RVQ `argmin`) sees the difference. One
//! consequence of the reuse worth knowing: the global
//! layers resolve their RoPE base through `rope_overrides`, so an operator
//! `--rope-freq-base` override installed for a process also reaches a
//! backbone loaded in that process.

use crate::models::gemma3::{Cache, CacheInterface, ModelArgs, TransformerBlock};
use mlxcel_core::dtype;
use mlxcel_core::layers::{GemmaRMSNorm, KVCache, RotatingKVCache};
use mlxcel_core::utils::create_sliding_window_prefill_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// `mlx.nn.gelu_approx`, op for op:
/// `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))`, with every
/// literal in `x`'s dtype (Python's weak scalars) and the cube taken with
/// `power`.
///
/// The crate's fused GeGLU (`compiled_geglu_approx_activation`) cubes with
/// two multiplies, groups the halving differently, and for bf16 rounds only
/// once; it disagrees with the reference on about half of all bf16 elements
/// and on the last bits of `f32` ones.
pub fn gelu_approx(x: &MlxArray) -> UniquePtr<MlxArray> {
    let d = mlxcel_core::array_dtype(x);
    let lit = |v: f64| mlxcel_core::full_f32(&[], v as f32, d);
    let cube = mlxcel_core::power(x, &lit(3.0));
    let inner = mlxcel_core::add(x, &mlxcel_core::multiply(&lit(0.044715), &cube));
    let inner = mlxcel_core::multiply(&lit((2.0 / std::f64::consts::PI).sqrt()), &inner);
    let cdf = mlxcel_core::add(&lit(1.0), &mlxcel_core::tanh(&inner));
    mlxcel_core::multiply(&mlxcel_core::multiply(&lit(0.5), x), &cdf)
}

/// One Gemma 3 decoder layer with the reference MLP activation.
fn layer_forward(
    layer: &TransformerBlock,
    x: &MlxArray,
    cache: &mut dyn CacheInterface,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let attn = layer
        .self_attn
        .forward(&layer.input_layernorm.forward(x), cache, mask);
    let h = mlxcel_core::compiled_clip_residual(x, &layer.post_attention_layernorm.forward(&attn));
    let normed = layer.pre_feedforward_layernorm.forward(&h);
    let mlp = &layer.mlp;
    let gated = mlxcel_core::multiply(
        &gelu_approx(&mlp.gate_proj.forward(&normed)),
        &mlp.up_proj.forward(&normed),
    );
    let ff = mlp.down_proj.forward(&gated);
    mlxcel_core::compiled_clip_residual(&h, &layer.post_feedforward_layernorm.forward(&ff))
}

/// Rebuild `norm` so its `1 + w` is the stored-dtype `1 + w` cast to f32.
///
/// [`GemmaRMSNorm::new`] adds one in the weight's dtype, so it is handed
/// `a - 1` in f32, where `a` is the widened `1 + w`. Both steps are exact:
/// for half-precision `w`, `a` is zero or a multiple of `2^-11` at least
/// (`1 + w` near zero needs `w` near `-1`, where the half ulp is `2^-8` for
/// bf16 and `2^-11` for f16), so for `|a| < 2^24` (far beyond any norm
/// scale) `a - 1` fits the f32 significand and `1 + (a - 1)` gives back `a`. The rebuilt norm's `weight` field holds
/// `a - 1`; every forward path reads only the adjusted weight.
pub(crate) fn widen_norm_to_f32(norm: &mut GemmaRMSNorm) {
    let adjusted = norm.adjusted_weight();
    let stored = mlxcel_core::array_dtype(adjusted);
    if stored != dtype::BFLOAT16 && stored != dtype::FLOAT16 {
        return;
    }
    let wide = mlxcel_core::astype(adjusted, dtype::FLOAT32);
    let offset = mlxcel_core::subtract(&wide, &mlxcel_core::full_f32(&[], 1.0, dtype::FLOAT32));
    let rebuilt = GemmaRMSNorm::new(offset, norm.eps);
    let ptrs = [
        &*rebuilt.weight as *const MlxArray,
        rebuilt.adjusted_weight() as *const MlxArray,
    ];
    // SAFETY: both pointers refer to arrays owned by `rebuilt`, which
    // outlives the call.
    unsafe { mlxcel_core::eval_all(&ptrs) };
    *norm = rebuilt;
}

/// Per-layer attention caches for a [`Gemma3Backbone`].
///
/// Global layers (`(i + 1) % sliding_window_pattern == 0`) hold a full
/// [`KVCache`]; sliding layers hold a [`RotatingKVCache`] bounded by
/// `sliding_window` with `keep = 0`, matching mlx-vlm's `make_cache`.
pub struct Gemma3BackboneCaches {
    caches: Vec<Cache>,
}

impl Gemma3BackboneCaches {
    /// Number of per-layer caches.
    pub fn len(&self) -> usize {
        self.caches.len()
    }

    /// Whether the stack has no layers.
    pub fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }

    /// Tokens processed so far (the first layer's RoPE offset).
    pub fn offset(&self) -> i32 {
        self.caches.first().map_or(0, Cache::offset)
    }

    /// Whether layer `idx` holds a full (global-attention) cache.
    pub fn is_global(&self, idx: usize) -> bool {
        matches!(self.caches.get(idx), Some(Cache::Standard(_)))
    }
}

/// Gemma 3 transformer layers plus final norm, fed with embeddings.
pub struct Gemma3Backbone {
    layers: Vec<TransformerBlock>,
    norm: GemmaRMSNorm,
    sliding_window: usize,
    sliding_window_pattern: usize,
}

impl Gemma3Backbone {
    /// Load `{prefix}.layers.{i}.*` for `i < args.num_hidden_layers` and
    /// `{prefix}.norm.weight`.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        args: &ModelArgs,
    ) -> Result<Self, String> {
        if args.sliding_window_pattern == 0 {
            return Err(format!(
                "Gemma 3 backbone at {prefix}: sliding_window_pattern must be positive"
            ));
        }
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for idx in 0..args.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{idx}");
            layers.push(TransformerBlock::from_weights_with_prefix(
                weights,
                args,
                &layer_prefix,
                idx,
            )?);
        }
        let norm_key = format!("{prefix}.norm.weight");
        let norm_weight = weights
            .get(&norm_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {norm_key}"))?;
        Ok(Self {
            layers,
            norm: GemmaRMSNorm::new(norm_weight, args.rms_norm_eps),
            sliding_window: args.sliding_window,
            sliding_window_pattern: args.sliding_window_pattern,
        })
    }

    /// Hold every norm's `1 + w` as f32: the `1 + w` built in the stored
    /// dtype, widened exactly. For a backbone that runs an f32 stream against
    /// half-precision weights (the Nemotron VoiceChat EAR-TTS, issue #2109).
    ///
    /// The reference builds `1 + w` in the weight's dtype and `fast::rms_norm`
    /// promotes it against the f32 input. CUDA builds resolve bf16 with f32
    /// to bf16 instead, which would demote the stream at every norm, so the
    /// widened weight is precomputed here. The values the kernel reads are
    /// unchanged under upstream promotion. Norms whose `1 + w` is already f32
    /// are left alone; [`GemmaRMSNorm`] itself is shared and unchanged.
    pub fn widen_norms_to_f32(&mut self) {
        for layer in &mut self.layers {
            for norm in [
                &mut layer.input_layernorm,
                &mut layer.post_attention_layernorm,
                &mut layer.pre_feedforward_layernorm,
                &mut layer.post_feedforward_layernorm,
                &mut layer.self_attn.q_norm,
                &mut layer.self_attn.k_norm,
            ] {
                widen_norm_to_f32(norm);
            }
        }
        widen_norm_to_f32(&mut self.norm);
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn is_global(&self, idx: usize) -> bool {
        idx % self.sliding_window_pattern == self.sliding_window_pattern - 1
    }

    /// Fresh caches: full `KVCache` for global layers, `RotatingKVCache`
    /// (`max_size = sliding_window`, `keep = 0`) for sliding layers.
    pub fn make_caches(&self) -> Gemma3BackboneCaches {
        let caches = (0..self.layers.len())
            .map(|idx| {
                if self.is_global(idx) {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window as i32))
                }
            })
            .collect();
        Gemma3BackboneCaches { caches }
    }

    /// Run `inputs` (`[B, L, hidden]`, used as-is with no embedding scale)
    /// through every layer and the final norm, appending to `caches`.
    ///
    /// Masks follow mlx-vlm: a single-token step attends without a mask; a
    /// multi-token call is plain causal on every global layer and on the
    /// sliding layers until the live keys outgrow the window, after which the
    /// sliding layers get an explicit sliding-window mask.
    pub fn forward_embeds(
        &self,
        inputs: &MlxArray,
        caches: &mut Gemma3BackboneCaches,
    ) -> Result<UniquePtr<MlxArray>, String> {
        if caches.caches.len() != self.layers.len() {
            return Err(format!(
                "Gemma 3 backbone has {} layers but {} caches were supplied",
                self.layers.len(),
                caches.caches.len()
            ));
        }
        let shape = mlxcel_core::array_shape(inputs);
        if shape.len() != 3 {
            return Err(format!(
                "Gemma 3 backbone expects [batch, length, hidden] inputs, got {shape:?}"
            ));
        }
        let seq_len = shape[1];

        let sliding_mask = if seq_len > 1 && self.sliding_window_pattern > 1 {
            let live_len = caches.caches[0].as_interface().live_len();
            let window = self.sliding_window as i32;
            (live_len + seq_len > window)
                .then(|| create_sliding_window_prefill_mask(seq_len, live_len, window))
        } else {
            None
        };

        let mut h = mlxcel_core::copy(inputs);
        for (idx, layer) in self.layers.iter().enumerate() {
            let mask = if self.is_global(idx) {
                None
            } else {
                sliding_mask.as_deref()
            };
            h = layer_forward(layer, &h, caches.caches[idx].as_interface(), mask);
        }
        Ok(self.norm.forward(&h))
    }
}

#[cfg(test)]
#[path = "gemma3_backbone_tests.rs"]
mod gemma3_backbone_tests;
