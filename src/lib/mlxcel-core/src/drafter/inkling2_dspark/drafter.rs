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

//! [`Inkling2DsparkDraftModel`]: the bundled drafter behind the
//! [`Drafter`] trait, with upstream `Model.draft_block` and `Model.bind`.

use std::path::Path;

use cxx::UniquePtr;

use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::RMSNorm;
use crate::weights::WeightMap;

use super::Inkling2DsparkBinding;
use super::config::{Inkling2DsparkConfig, MTP_WEIGHTS_FILE};
use super::markov::MarkovHead;
use super::model::{DraftContextCache, DraftLayer, FuseWeights, f32};

pub struct Inkling2DsparkDraftModel {
    config: Inkling2DsparkConfig,
    layers: Vec<DraftLayer>,
    norm: RMSNorm,
    markov_head: MarkovHead,
    fuse: FuseWeights,
    mask_embed: UniquePtr<MlxArray>,
    caches: Vec<DraftContextCache>,
    /// The target parts bound last, and the address of the target they came
    /// from: the fold runs once per target, as upstream's identity check.
    binding: Option<(usize, Inkling2DsparkBinding)>,
}

/// `(group_size, bits)` of an mlx-vlm split that quantized the layers;
/// the native sidecar is BF16 and never reads it.
fn quantization(config_json: &serde_json::Value) -> (i32, i32) {
    let block = config_json.get("quantization");
    let read = |key: &str, default: i64| {
        block
            .and_then(|value| value.get(key))
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(default) as i32
    };
    (read("group_size", 64), read("bits", 4))
}

fn load_failed(reason: String) -> DrafterError {
    DrafterError::LoadFailed { reason }
}

impl Inkling2DsparkDraftModel {
    /// Load `dir/mtp.safetensors` (unprefixed names, kept in its BF16) with
    /// the `mtp_config` of `dir/config.json`.
    pub fn from_path(dir: &Path) -> Result<Self, DrafterError> {
        let config = Inkling2DsparkConfig::from_dir(dir).map_err(DrafterError::Config)?;
        let raw = std::fs::read(dir.join("config.json")).map_err(|e| DrafterError::ConfigIo {
            path: dir.join("config.json").display().to_string(),
            source: e,
        })?;
        let json: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| DrafterError::ConfigParse {
                path: dir.join("config.json").display().to_string(),
                source: e,
            })?;
        let weights = crate::weights::load_safetensors(dir.join(MTP_WEIGHTS_FILE))
            .map_err(|reason| DrafterError::WeightLoad { reason })?;
        Self::from_weights(config, &weights, quantization(&json))
    }

    pub fn from_weights(
        config: Inkling2DsparkConfig,
        weights: &WeightMap,
        quant: (i32, i32),
    ) -> Result<Self, DrafterError> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| DraftLayer::from_weights(weights, index, &config, quant))
            .collect::<Result<Vec<_>, _>>()
            .map_err(load_failed)?;
        let norm = super::model::norm_weight(weights, "norm.weight", config.rms_norm_eps)
            .map_err(load_failed)?;
        let markov_head = MarkovHead::from_weights(
            weights,
            config.hidden_size,
            config.markov_rank,
            config.rms_norm_eps,
        )
        .map_err(load_failed)?;
        let mask_embed = weights
            .get("mask_embed")
            .map(|value| ffi::copy(value))
            .ok_or_else(|| load_failed("Inkling 2 DSpark weight not found: mask_embed".into()))?;
        let fuse_weight = weights
            .get("fuse_weight")
            .ok_or_else(|| load_failed("Inkling 2 DSpark weight not found: fuse_weight".into()))?;
        // The hidden rows the target hands over are in its activation dtype,
        // which is the drafter's weight dtype on every release.
        let fuse = FuseWeights::new(fuse_weight, ffi::array_dtype(&mask_embed), &config)
            .map_err(load_failed)?;
        let mut model = Self {
            config,
            layers,
            norm,
            markov_head,
            fuse,
            mask_embed,
            caches: Vec::new(),
            binding: None,
        };
        model.caches = model.make_context_caches();
        Ok(model)
    }

    pub fn config(&self) -> &Inkling2DsparkConfig {
        &self.config
    }

    fn make_context_caches(&self) -> Vec<DraftContextCache> {
        self.layers
            .iter()
            .map(|layer| DraftContextCache::new(layer.self_attn.window))
            .collect()
    }

    /// The target's binding, refused with both values when its shape is not
    /// the one the drafter was trained against.
    fn checked_binding(
        &self,
        target: &dyn LanguageModel,
    ) -> Result<Inkling2DsparkBinding, DrafterError> {
        let binding = target
            .inkling2_dspark_binding()
            .ok_or_else(|| DrafterError::BindFailed {
                reason: "the Inkling 2 DSpark drafter requires its matching Inkling 2 target; \
                         this target does not expose the Inkling 2 embedding, embed_norm and LM \
                         head"
                    .to_string(),
            })?;
        if let Some((name, drafter, target)) = self.config.target.first_mismatch(&binding.shape) {
            return Err(DrafterError::BindFailed {
                reason: format!(
                    "the Inkling 2 DSpark drafter does not match this target: drafter {name} = \
                     {drafter}, target {name} = {target}. Pair a checkpoint's mtp.safetensors \
                     with that same checkpoint."
                ),
            });
        }
        Ok(binding)
    }

    /// Upstream `Model.draft_block`, greedy: `[1, block_size - 1]` proposals
    /// for the anchor `last_bonus` and the newly committed target hidden rows
    /// `hidden` `[1, L, taps * hidden]`, kept on device.
    pub fn draft_block_tokens(
        &mut self,
        last_bonus: i32,
        hidden: &MlxArray,
        block_size: usize,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        if let Some(reason) = self.config.verify_width_error(block_size) {
            return Err(DrafterError::DraftFailed { reason });
        }
        let Some((_, binding)) = self.binding.as_ref() else {
            return Err(DrafterError::DraftFailed {
                reason: "bind the Inkling 2 target before drafting".to_string(),
            });
        };
        let shape = ffi::array_shape(hidden);
        let width = (self.config.target_layer_ids.len() * self.config.hidden_size) as i32;
        if shape.len() != 3 || shape[0] != 1 || shape[1] < 1 || shape[2] != width {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Inkling 2 DSpark needs captured hidden rows [1, L >= 1, {width}], got {shape:?}"
                ),
            });
        }
        let count = block_size as i32 - 1;
        let trained = self.config.block_size as i32;
        let hidden_size = self.config.hidden_size as i32;

        let anchor = ffi::from_slice_i32(&[last_bonus], &[1]);
        let embed = binding.embed_norm.forward(
            &binding
                .embed_tokens
                .forward(&ffi::reshape(&anchor, &[1, 1])),
        );
        let embed_dtype = ffi::array_dtype(&embed);
        let masks = ffi::broadcast_to(
            &ffi::astype(&self.mask_embed, embed_dtype),
            &[1, trained - 1, hidden_size],
        );
        // Bidirectional attention needs the full trained block even when
        // fewer proposals are asked for near the end of the budget.
        let mut x = crate::ops::concatenate(&embed, &masks, 1);
        let mut residual: Option<UniquePtr<MlxArray>> = None;
        for (index, (layer, cache)) in self.layers.iter().zip(&mut self.caches).enumerate() {
            let context = self.fuse.mix(hidden, index, &layer.dflare_norm);
            let (out, next) = layer.forward(&x, residual.as_deref(), &context, cache);
            x = out;
            residual = Some(next);
        }
        let residual = residual.expect("at least one drafter layer");
        let x = ffi::astype(
            &self.norm.forward(&ffi::add(&f32(&x), &f32(&residual))),
            embed_dtype,
        );
        let x = ffi::slice(&x, &[0, 1, 0], &[1, count + 1, hidden_size]);
        // Draft logits are BF16, unlike the target's f32 vocabulary product.
        let base = ffi::astype(&binding.lm_head.forward(&x), ffi::array_dtype(&x));
        let vocab = ffi::array_shape(&base)[2];
        let last = ffi::slice(hidden, &[0, shape[1] - 1, 0], &[1, shape[1], width]);
        let f_rnn = self
            .fuse
            .mix(&last, self.fuse.last_row(), &self.markov_head.dflare_norm);
        let context = self.markov_head.context(&x, &f_rnn);
        let rank = self.markov_head.rank;
        let mut state = ffi::zeros(&[1, rank], ffi::array_dtype(&x));
        let mut previous = anchor;
        let cap = self.config.final_logit_softcapping;
        let unpadded = (self.config.unpadded_vocab_size as i32).min(vocab);
        let mut tokens = Vec::with_capacity(count as usize);
        for i in 0..count {
            let row =
                ffi::squeeze_axis(&ffi::slice(&context, &[0, i, 0], &[1, i + 1, 3 * rank]), 1);
            let (next_state, bias) = self.markov_head.step(&state, &previous, &row);
            state = next_state;
            let base_row = ffi::squeeze_axis(&ffi::slice(&base, &[0, i, 0], &[1, i + 1, vocab]), 1);
            let mut logits = ffi::add(&base_row, &bias);
            if cap > 0.0 {
                let capped = crate::ops::multiply_scalar(
                    &ffi::tanh(&crate::ops::divide_scalar(&f32(&logits), cap)),
                    cap,
                );
                logits = ffi::astype(&capped, ffi::array_dtype(&x));
            }
            let logits = ffi::slice(&logits, &[0, 0], &[1, unpadded]);
            previous = ffi::astype(&ffi::argmax(&logits, -1, false), crate::dtype::INT32);
            tokens.push(ffi::reshape(&previous, &[1, 1]));
        }
        let refs: Vec<&MlxArray> = tokens.iter().map(|t| t.as_ref().expect("token")).collect();
        Ok(crate::ops::concatenate_many(&refs, 1))
    }
}

impl Drafter for Inkling2DsparkDraftModel {
    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.checked_binding(target).map(|_| ())
    }

    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let address = target as *const dyn LanguageModel as *const () as usize;
        if self
            .binding
            .as_ref()
            .is_some_and(|(bound, _)| *bound == address)
            && self.markov_head.is_folded()
        {
            return Ok(());
        }
        let binding = self.checked_binding(target)?;
        self.markov_head.fold(&binding.lm_head);
        self.binding = Some((address, binding));
        Ok(())
    }

    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.bind(target)?;
        self.caches = self.make_context_caches();
        Ok(())
    }

    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        Some(&self.config.target_layer_ids)
    }

    fn configured_block_size(&self) -> Option<usize> {
        Some(self.config.default_verify_width())
    }

    fn prefer_requested_block_size(&self) -> bool {
        true
    }

    fn greedy_only(&self) -> bool {
        true
    }

    fn is_inkling2_dspark(&self) -> bool {
        true
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        let tokens = self.draft_block_array(last_bonus, hidden, block_size, sampler)?;
        ffi::eval(&tokens);
        let count = ffi::array_shape(&tokens).iter().product::<i32>() as usize;
        Ok((0..count)
            .map(|i| ffi::item_i32(&ffi::slice(&tokens, &[0, i as i32], &[1, i as i32 + 1])))
            .collect())
    }

    fn draft_block_array(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        if !crate::drafter::dflash::drafter::sampler_is_greedy(sampler) {
            return Err(DrafterError::GreedyOnly {
                kind: self.kind(),
                temperature: sampler.temperature,
                top_k: sampler.top_k,
            });
        }
        let hidden = hidden.ok_or_else(|| DrafterError::DraftFailed {
            reason: "Inkling 2 DSpark requires the captured target hidden rows".to_string(),
        })?;
        self.draft_block_tokens(last_bonus, hidden, block_size)
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
}
