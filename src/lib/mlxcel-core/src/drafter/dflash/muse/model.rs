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

//! The Muse Glimmer assistant drafter model (issue #1343).
//!
//! Five sliding-attention decoder layers fed with five of the target's
//! residual streams through `encoder.fc` (`5 * hidden -> hidden`) and
//! `encoder.output_norm_enc`, predicting a whole block in one non-causal
//! forward. The checkpoint ships no `embed_tokens` and no `lm_head`: both
//! are bound from the target at [`MuseAssistantModel::bind_target`], the RAW
//! embedding table (not the target's `embed_norm`-wrapped lookup) and the
//! untied head (not followed by the target's `output_multiplier` or
//! `final_logit_softcapping`).

use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::attention::MuseAssistantAttention;
use super::cache::MuseAssistantContextCache;
use super::config::MuseAssistantConfig;
use crate::drafter::dflash::mlp::DFlashMlp;

/// One drafter transformer block: pre-norm attention, pre-norm SwiGLU MLP,
/// both under residual connections. Plain `RMSNorm` with weight `w`, not
/// the target's centered `1 + w` form.
pub struct MuseAssistantDecoderLayer {
    pub self_attn: MuseAssistantAttention,
    pub mlp: DFlashMlp,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl MuseAssistantDecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        context: &MlxArray,
        cache: &mut MuseAssistantContextCache,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn = self.self_attn.forward(&normed, context, cache);
        let h = ffi::add(x, &attn);
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp = self.mlp.forward(&normed);
        ffi::add(&h, &mlp)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &MuseAssistantConfig,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: MuseAssistantAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
            )?,
            mlp: DFlashMlp::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                config.group_size(),
                config.bits(),
            )?,
            input_layernorm: load_norm(weights, &format!("{prefix}.input_layernorm"), config)?,
            post_attention_layernorm: load_norm(
                weights,
                &format!("{prefix}.post_attention_layernorm"),
                config,
            )?,
        })
    }
}

fn load_norm(
    weights: &WeightMap,
    prefix: &str,
    config: &MuseAssistantConfig,
) -> Result<RMSNorm, String> {
    let key = format!("{prefix}.weight");
    let weight = weights
        .get(&key)
        .map(|w| ffi::copy(w))
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    Ok(RMSNorm::new(weight, config.rms_norm_eps))
}

/// The assembled drafter.
pub struct MuseAssistantModel {
    pub config: MuseAssistantConfig,
    /// Target-bound raw embedding table; `None` until
    /// [`Self::bind_target`] unless the checkpoint shipped its own.
    embed_tokens: Option<UnifiedEmbedding>,
    /// Target-bound untied head; `None` until [`Self::bind_target`] unless
    /// the checkpoint shipped its own.
    lm_head: Option<UnifiedLinear>,
    /// `encoder.fc`: `[hidden, len(target_layer_ids) * hidden]`.
    pub fc: UnifiedLinear,
    /// `encoder.output_norm_enc`.
    pub output_norm_enc: RMSNorm,
    pub layers: Vec<MuseAssistantDecoderLayer>,
    pub norm: RMSNorm,
}

impl MuseAssistantModel {
    /// Strip a leading `model.` from every key. The published checkpoint
    /// carries none; a re-export through a HuggingFace wrapper would.
    pub fn sanitize(weights: &mut WeightMap) {
        let prefixed: Vec<String> = weights
            .keys()
            .filter(|k| k.starts_with("model."))
            .cloned()
            .collect();
        for key in prefixed {
            if let Some(tensor) = weights.remove(&key) {
                weights.insert(key["model.".len()..].to_string(), tensor);
            }
        }
    }

    /// Build the drafter from sanitized weights. `embed_tokens` and
    /// `lm_head` are loaded only when the checkpoint ships them.
    pub fn from_weights(weights: &WeightMap, config: MuseAssistantConfig) -> Result<Self, String> {
        config.validate()?;
        let group_size = config.group_size();
        let bits = config.bits();

        let embed_tokens = if weights.contains_key("embed_tokens.weight")
            || weights.contains_key("embed_tokens.scales")
        {
            Some(UnifiedEmbedding::from_weights(
                weights,
                "embed_tokens",
                group_size,
                bits,
            )?)
        } else {
            None
        };
        let lm_head =
            if weights.contains_key("lm_head.weight") || weights.contains_key("lm_head.scales") {
                Some(UnifiedLinear::from_weights(
                    weights, "lm_head", group_size, bits,
                )?)
            } else {
                None
            };

        let fc = UnifiedLinear::from_weights(weights, "encoder.fc", group_size, bits)?;
        if let UnifiedLinear::Regular(linear) = &fc {
            let shape = ffi::array_shape(&linear.weight);
            let expected_in = (config.target_layer_ids.len() * config.hidden_size) as i32;
            if shape.len() != 2 || shape[0] != config.hidden_size as i32 || shape[1] != expected_in
            {
                return Err(format!(
                    "Muse Glimmer assistant encoder.fc.weight has shape {shape:?}; expected \
                     [{}, {expected_in}] (hidden_size x len(target_layer_ids) * hidden_size)",
                    config.hidden_size
                ));
            }
        }
        let output_norm_enc = load_norm(weights, "encoder.output_norm_enc", &config)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MuseAssistantDecoderLayer::from_weights(
                weights,
                &format!("layers.{i}"),
                &config,
            )?);
        }
        let norm = load_norm(weights, "norm", &config)?;

        Ok(Self {
            config,
            embed_tokens,
            lm_head,
            fc,
            output_norm_enc,
            layers,
            norm,
        })
    }

    /// Whether the checkpoint shipped no embedding table of its own.
    pub fn needs_embed_binding(&self) -> bool {
        self.embed_tokens.is_none()
    }

    /// Whether the checkpoint shipped no head of its own.
    pub fn needs_lm_head_binding(&self) -> bool {
        self.lm_head.is_none()
    }

    /// Install the target's raw embedding table and untied head. A
    /// checkpoint that shipped either keeps its own.
    pub fn bind_target(&mut self, embed: UnifiedEmbedding, lm_head: UnifiedLinear) {
        if self.embed_tokens.is_none() {
            self.embed_tokens = Some(embed);
        }
        if self.lm_head.is_none() {
            self.lm_head = Some(lm_head);
        }
    }

    /// Input width of `encoder.fc` when it can be read off a dense weight;
    /// `None` for a quantized export, whose packed width says nothing.
    pub fn fc_in_features(&self) -> Option<usize> {
        match &self.fc {
            UnifiedLinear::Regular(linear) => {
                ffi::array_shape(&linear.weight).last().map(|&n| n as usize)
            }
            UnifiedLinear::Quantized { .. } => None,
        }
    }

    /// The dtype the drafter's dense weights carry (f16 after load on Apple
    /// Silicon), or `None` for a quantized export. The forward casts its two
    /// target-sourced inputs to it, so the drafter runs in one dtype whatever
    /// the target's residual streams and embedding table are stored as.
    fn dense_dtype(&self) -> Option<i32> {
        match &self.fc {
            UnifiedLinear::Regular(linear) => Some(ffi::array_dtype(&linear.weight)),
            UnifiedLinear::Quantized { .. } => None,
        }
    }

    /// One fresh context window per drafter layer.
    pub fn make_caches(&self) -> Vec<MuseAssistantContextCache> {
        (0..self.layers.len())
            .map(|_| MuseAssistantContextCache::new(self.config.sliding_window as i32))
            .collect()
    }

    fn embed(&self) -> &UnifiedEmbedding {
        self.embed_tokens.as_ref().expect(
            "MuseAssistantModel::forward called before bind_target installed the target's \
             embedding table",
        )
    }

    fn head(&self) -> &UnifiedLinear {
        self.lm_head.as_ref().expect(
            "MuseAssistantModel::forward called before bind_target installed the target's lm_head",
        )
    }

    /// One draft forward: `block` is `[1, bs]` (`[bonus, mask, ...]`),
    /// `target_hidden` is `[1, S, len(target_layer_ids) * hidden]`; returns
    /// `[1, bs, vocab]` raw head logits (no softcap).
    ///
    /// A prompt longer than the window keeps only its last `window` rows
    /// of context and the caches account for the dropped rows, so the
    /// absolute positions the mask reads stay right.
    pub fn forward(
        &self,
        block: &MlxArray,
        target_hidden: &MlxArray,
        caches: &mut [MuseAssistantContextCache],
    ) -> UniquePtr<MlxArray> {
        debug_assert_eq!(
            caches.len(),
            self.layers.len(),
            "MuseAssistantModel::forward: {} caches for {} layers",
            caches.len(),
            self.layers.len()
        );
        // Drop the rows past the window BEFORE the encoder runs. `fc` and
        // `output_norm_enc` are both per-row, so this is the same context
        // the post-projection trim produced, at the cost of the rows that
        // survive rather than the rows that arrived. It is the whole prompt
        // that arrives on the first round: at `-c 16384` the incoming
        // `[1, S, 5 * 6656]` slab alone is 1.1 GB in f16, and projecting all
        // of it to throw away seven eighths cost that again in the cast and
        // in `fc`'s output.
        let window = self.config.sliding_window as i32;
        let incoming = ffi::array_shape(target_hidden);
        let target_hidden = if incoming.len() == 3 && incoming[1] > window {
            for cache in caches.iter_mut() {
                cache.skip(incoming[1] - window);
            }
            ffi::slice(
                target_hidden,
                &[0, incoming[1] - window, 0],
                &[incoming[0], incoming[1], incoming[2]],
            )
        } else {
            ffi::copy(target_hidden)
        };

        // Both inputs come from the target: the residual streams and the
        // embedding lookup. Cast them to the drafter's own dtype so the
        // attention runs uniformly rather than promoting to f32 on a
        // bf16 / f16 mix.
        let (h, target_hidden) = match self.dense_dtype() {
            Some(dtype) => (
                ffi::astype(&self.embed().forward(block), dtype),
                ffi::astype(&target_hidden, dtype),
            ),
            None => (self.embed().forward(block), target_hidden),
        };
        let mut h = h;
        let projected = self.fc.forward(&target_hidden);
        let context = self.output_norm_enc.forward(&projected);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, &context, cache);
        }
        let h = self.norm.forward(&h);
        self.head().forward(&h)
    }

    /// Greedy draft: `block_size - 1` argmax proposals for the masked
    /// slots after `last_bonus`.
    pub fn draft_block(
        &self,
        last_bonus: i32,
        target_hidden: &MlxArray,
        caches: &mut [MuseAssistantContextCache],
        block_size: usize,
    ) -> Vec<i32> {
        assert!(
            block_size >= 2,
            "Muse assistant draft_block requires block_size >= 2 (got {block_size})"
        );
        let logits = self.forward(
            &block_input(last_bonus, self.config.mask_token_id, block_size),
            target_hidden,
            caches,
        );
        let shape = ffi::array_shape(&logits);
        let proposals = ffi::slice(
            &logits,
            &[0, 1, 0],
            &[shape[0], block_size as i32, shape[2]],
        );
        let argmax = ffi::argmax(&proposals, 2, false);
        crate::drafter::dflash::materialize_argmax_i32_vec(&argmax, block_size - 1)
    }
}

/// `[1, block_size]` draft input: the bonus token followed by
/// `block_size - 1` mask placeholders.
///
/// Used by: [`MuseAssistantModel::draft_block`], `MuseAssistantDrafter`.
pub(crate) fn block_input(
    last_bonus: i32,
    mask_token_id: i32,
    block_size: usize,
) -> UniquePtr<MlxArray> {
    let mut block: Vec<i32> = Vec::with_capacity(block_size);
    block.push(last_bonus);
    block.resize(block_size, mask_token_id);
    ffi::from_slice_i32(&block, &[1, block_size as i32])
}
