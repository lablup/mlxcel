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

//! nanochat (`nanochat`) text model implementation using mlxcel-core.
//!
//! Reference: the `nanochat` speedrun models (d20 / d32) and the HuggingFace
//! `transformers` `NanoChatForCausalLM` port.
//!
//! A GPT-2-shaped decoder with four non-standard choices. Skipping any one of
//! them yields fluent but wrong text, so each is called out where it happens:
//!
//! - RMSNorm without a learnable weight everywhere: on the embedding output,
//!   before attention, before the MLP, on the final hidden state, and on q and
//!   k per head. The checkpoint has no norm tensors at all.
//! - QK-norm runs AFTER RoPE (rotate first, then normalize).
//! - Mirrored RoPE: the rotation angle is `-p / base^(i / half)`, i.e. the
//!   opposite direction of the usual one, with the half-split pairing
//!   (`traditional = false`).
//! - `relu(x)^2` MLP with `c_fc` / `c_proj` names.
//! - Output logits are soft-capped: `cap * tanh(logits / cap)`, `cap = 15`.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{pipeline_hint, relu_squared};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

#[path = "nanochat_config.rs"]
mod config;
pub use config::{EosTokenId, ModelArgs, NANOCHAT_EOS_TOKEN_ID, Quantization};

// Weightless RMSNorm.

/// `x * rsqrt(mean(x^2, -1) + eps)` with no learnable weight.
#[inline]
fn norm(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::fast_rms_norm_no_weight(x, eps)
}

/// Mirrored RoPE frequency table: `-(base ** (arange(half) / half))`.
///
/// `fast_rope_with_freqs` divides the position by each entry, so negating the
/// table negates the rotation angle. Computed in f64 and stored as f32.
pub(crate) fn mirrored_rope_freqs(head_dim: usize, base: f32) -> UniquePtr<MlxArray> {
    let half = head_dim / 2;
    let table: Vec<f32> = (0..half)
        .map(|i| -(f64::from(base).powf(i as f64 / half as f64)) as f32)
        .collect();
    mlxcel_core::from_slice_f32(&table, &[half as i32])
}

// Attention.

pub struct Attention {
    pub c_q: UnifiedLinear,
    pub c_k: UnifiedLinear,
    pub c_v: UnifiedLinear,
    pub c_proj: UnifiedLinear,
    pub freqs: UniquePtr<MlxArray>,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub eps: f32,
}

impl Attention {
    /// Projects, rotates, then QK-norms (RoPE first: the order matters).
    pub(crate) fn rotate_and_norm(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offset: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let q =
            mlxcel_core::fast_rope_with_freqs(q, self.head_dim, false, 1.0, offset, &self.freqs);
        let k =
            mlxcel_core::fast_rope_with_freqs(k, self.head_dim, false, 1.0, offset, &self.freqs);
        (norm(&q, self.eps), norm(&k, self.eps))
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let offset = cache.offset;

        let split = |t: UniquePtr<MlxArray>| {
            let t = mlxcel_core::reshape(&t, &[b, l, self.num_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3])
        };
        let q = split(self.c_q.forward(x));
        let k = split(self.c_k.forward(x));
        let v = split(self.c_v.forward(x));

        let (q, k) = self.rotate_and_norm(&q, &k, offset);
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // A prefill with no caller-supplied mask must take the causal path:
        // every generation path passes `mask == None` and expects the model to
        // build its own causal mask (see the gemma2 note on issue #999).
        let attn = if let Some(m) = mask {
            let mask_ptr = m as *const _;
            // SAFETY: `mask_ptr` derives from a live `&MlxArray` that outlives
            // this call.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        } else {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        };

        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, l, self.num_heads * self.head_dim]);
        self.c_proj.forward(&attn)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let (gs, bits) = (args.group_size(), args.bits());
        let lin = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), gs, bits)
        };
        let head_dim = args.head_dim();
        Ok(Self {
            c_q: lin("c_q")?,
            c_k: lin("c_k")?,
            c_v: lin("c_v")?,
            c_proj: lin("c_proj")?,
            freqs: mirrored_rope_freqs(head_dim, args.rope_theta),
            num_heads: args.num_attention_heads as i32,
            head_dim: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
            eps: args.rms_norm_eps,
        })
    }
}

// MLP (relu squared).

pub struct MLP {
    pub c_fc: UnifiedLinear,
    pub c_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.c_proj.forward(&relu_squared(&self.c_fc.forward(x)))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let (gs, bits) = (args.group_size(), args.bits());
        Ok(Self {
            c_fc: UnifiedLinear::from_weights(weights, &format!("{prefix}.c_fc"), gs, bits)?,
            c_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.c_proj"), gs, bits)?,
        })
    }
}

// Transformer block (no norm weights).

pub struct TransformerBlock {
    pub attn: Attention,
    pub mlp: MLP,
    pub eps: f32,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let a = self.attn.forward(&norm(x, self.eps), cache, mask);
        let h = mlxcel_core::add(x, &a);
        let m = self.mlp.forward(&norm(&h, self.eps));
        mlxcel_core::add(&h, &m)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("transformer.h.{layer_idx}");
        Ok(Self {
            attn: Attention::from_weights(weights, args, &format!("{prefix}.attn"))?,
            mlp: MLP::from_weights(weights, args, &format!("{prefix}.mlp"))?,
            eps: args.rms_norm_eps,
        })
    }
}

// Model.

pub struct NanoChatModel {
    pub wte: UnifiedEmbedding,
    pub h: Vec<TransformerBlock>,
    pub lm_head: Option<UnifiedLinear>,
    pub softcap: Option<f32>,
    pub eps: f32,
    eos_token_ids: Vec<i32>,
}

/// Apply `cap * tanh(x / cap)`; pass through when `cap` is `None`.
pub(crate) fn apply_softcap(logits: UniquePtr<MlxArray>, cap: Option<f32>) -> UniquePtr<MlxArray> {
    match cap {
        Some(cap) => mlxcel_core::compiled_softcap(&logits, cap),
        None => logits,
    }
}

impl NanoChatModel {
    fn hidden_states(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // The embedding output is itself normed.
        let mut h = norm(&self.wte.forward(input_ids), self.eps);
        let n = self.h.len();
        for (i, layer) in self.h.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
            pipeline_hint(&h, i, n);
        }
        norm(&h, self.eps)
    }

    fn logits_from_hidden(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let logits = match &self.lm_head {
            Some(head) => head.forward(h),
            None => self.wte.as_linear(h),
        };
        apply_softcap(logits, self.softcap)
    }

    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_states(input_ids, caches, mask);
        self.logits_from_hidden(&h)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.h.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        args.validate()?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;
        // Both checkpoint layouts load: a transformers-format export is mapped
        // onto the `transformer.*` layout first. Idempotent on the MLX layout.
        let sanitized;
        let weights = if needs_sanitize(weights) {
            sanitized = sanitize_weights(weights);
            &sanitized
        } else {
            weights
        };

        let (gs, bits) = (args.group_size(), args.bits());
        let wte = UnifiedEmbedding::from_weights(weights, "transformer.wte", gs, bits)?;
        let mut h = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            h.push(TransformerBlock::from_weights(weights, args, i)?);
        }
        let lm_head = if args.tie_word_embeddings && !weights.contains_key("lm_head.weight") {
            None
        } else {
            Some(UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?)
        };

        Ok(Self {
            wte,
            h,
            lm_head,
            softcap: args.soft_cap(),
            eps: args.rms_norm_eps,
            eos_token_ids: args.eos_token_ids(),
        })
    }
}

// Weight sanitation.

/// Map a transformers-format key onto the MLX `transformer.*` layout.
/// Returns the key unchanged when it is not a transformers-layout key.
pub(crate) fn remap_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.embed_tokens.") {
        return format!("transformer.wte.{rest}");
    }
    let Some(rest) = key.strip_prefix("model.layers.") else {
        return key.to_string();
    };
    let Some((idx, tail)) = rest.split_once('.') else {
        return key.to_string();
    };
    for (from, to) in [
        ("self_attn.q_proj.", "attn.c_q."),
        ("self_attn.k_proj.", "attn.c_k."),
        ("self_attn.v_proj.", "attn.c_v."),
        ("self_attn.o_proj.", "attn.c_proj."),
        ("mlp.fc1.", "mlp.c_fc."),
        ("mlp.fc2.", "mlp.c_proj."),
    ] {
        if let Some(suffix) = tail.strip_prefix(from) {
            return format!("transformer.h.{idx}.{to}{suffix}");
        }
    }
    key.to_string()
}

fn needs_sanitize(weights: &WeightMap) -> bool {
    weights
        .keys()
        .any(|k| k.contains("rotary_emb.inv_freq") || remap_key(k) != *k)
}

/// Rename transformers-layout keys and drop `rotary_emb.inv_freq` buffers.
/// Idempotent: keys already in the MLX layout pass through unchanged.
pub fn sanitize_weights(weights: &WeightMap) -> WeightMap {
    weights
        .iter()
        .filter(|(k, _)| !k.contains("rotary_emb.inv_freq"))
        .map(|(k, v)| (remap_key(k), mlxcel_core::copy(v)))
        .collect()
}

// LanguageModel trait implementation.

impl LanguageModel for NanoChatModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        NanoChatModel::forward(self, input_ids, caches, mask)
    }

    /// Slice the hidden state before the LM head so a prefill does not project
    /// every prompt row through the 65k vocabulary. The head and softcap are
    /// per position, so this equals slicing the full logits.
    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_states(input_ids, caches, mask);
        let row = mlxcel_core::generate::logits_at_position(&h, last_pos);
        self.logits_from_hidden(&row)
    }

    /// The engine's prefill entry (CLI and server alike) passes a sequence id;
    /// nanochat keeps dense caches, so the id changes nothing and the prefill
    /// still projects only the last row.
    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        _seq_id: Option<mlxcel_core::cache::SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_last_logits(self, input_ids, caches, mask, last_pos)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        NanoChatModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.h.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "nanochat_tests.rs"]
mod tests;
