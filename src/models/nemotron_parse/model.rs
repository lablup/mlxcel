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

//! The assembled Nemotron-Parse model: C-RADIO tower plus neck as the
//! encoder, the pre-norm mBART decoder, and the greedy seq2seq loop.
//!
//! Per request the page is encoded once into `[1, 3329, 1024]` states; the
//! decoder is then seeded with the tokenized task prompt in a single prefill
//! call and extended one token per step against the cached cross-attention
//! K/V, exactly like `transformers` `generate` with `num_beams=1`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow};
use serde_json::Value;

use mlxcel_core::layers::{UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::florence2::layers::Florence2LayerCache;

use super::checkpoint::{
    TEXT_PREFIX, VISION_PREFIX, canonicalize_keys, reject_unsupported_quantized_tensors,
};
use super::config::NemotronParseConfig;
use super::decoder::NemotronParseDecoder;
use super::encoder::RadioVisionTower;
use super::neck::NemotronParseNeck;

/// Decode state for one sequence: per-layer self + cross K/V and the number
/// of decoder positions consumed.
pub struct NemotronParseSeqCache {
    layers: Vec<Florence2LayerCache>,
    offset: i32,
}

impl NemotronParseSeqCache {
    fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| Florence2LayerCache::default())
                .collect(),
            offset: 0,
        }
    }

    /// Decoder positions already consumed (seed plus generated tokens).
    pub fn offset(&self) -> i32 {
        self.offset
    }
}

/// Result of one greedy decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronParseGeneration {
    /// Generated ids, seed and the terminating EOS excluded.
    pub tokens: Vec<i32>,
    /// Whether decoding stopped on EOS (as opposed to a budget, the position
    /// bound, or cancellation).
    pub hit_eos: bool,
}

/// HF `RepetitionPenaltyLogitsProcessor` over `history` (every id in the
/// decoder sequence, seed included): a positive logit is divided by
/// `penalty`, a negative one multiplied. Each id is penalized once.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[i32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = std::collections::HashSet::with_capacity(history.len());
    for &id in history {
        let Ok(idx) = usize::try_from(id) else {
            continue;
        };
        if idx >= logits.len() || !seen.insert(idx) {
            continue;
        }
        let v = logits[idx];
        logits[idx] = if v < 0.0 { v * penalty } else { v / penalty };
    }
}

/// Index of the largest value; the first one wins a tie, like `torch.argmax`.
pub(crate) fn argmax_f32(values: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in values.iter().enumerate() {
        match best {
            Some((_, b)) if v <= b => {}
            _ if v.is_nan() => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i)
}

/// The loaded model. Holds MLX weight handles, so the owning provider
/// serializes access.
pub struct NemotronParseModel {
    config: NemotronParseConfig,
    vision: RadioVisionTower,
    neck: NemotronParseNeck,
    shared: UnifiedEmbedding,
    decoder: NemotronParseDecoder,
    lm_head: Option<UnifiedLinear>,
    /// Activation dtype of the decoder: always f32 (see `from_weights`).
    text_dtype: i32,
}

impl NemotronParseModel {
    /// Load from a checkpoint directory (`config.json` + safetensors), in
    /// either the Hub or the MLX-converted key layout.
    pub fn load(model_path: &Path) -> Result<Self> {
        let config_path = model_path.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("Failed to read {config_path:?}: {e}"))?;
        let raw = crate::models::sanitize_config_json(&raw);
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("Failed to parse Nemotron-Parse config: {e}"))?;
        let config = NemotronParseConfig::from_model_config(&value)?;

        let weights = mlxcel_core::weights::load_weights_from_dir(model_path)
            .map_err(|e| anyhow!("Failed to load Nemotron-Parse weights: {e}"))?;
        let mut weights = canonicalize_keys(weights);
        reject_unsupported_quantized_tensors(&weights).map_err(|e| anyhow!("{e}"))?;
        // Apple Silicon precision policy (bf16 -> f16) for the dense tensors.
        // The quantized decoder's scales and biases keep their stored width:
        // they are dequantization operands, and rounding them perturbs every
        // reconstructed weight.
        let quantized = crate::models::Florence2Quantization::config_is_quantized(&value);
        let _ = crate::models::convert_bf16_weights_with_keep(&mut weights, |key| {
            quantized && key.starts_with(TEXT_PREFIX)
        });
        Self::from_weights(&weights, config).map_err(|e| anyhow!("{e}"))
    }

    /// Build from a canonical-layout weight map.
    pub fn from_weights(weights: &WeightMap, config: NemotronParseConfig) -> Result<Self, String> {
        let q = config.quantization;
        let vision = RadioVisionTower::from_weights(
            weights,
            VISION_PREFIX,
            &config.vision,
            q.group_size,
            q.bits,
        )?;
        let neck = NemotronParseNeck::from_weights(
            weights,
            &format!("{VISION_PREFIX}.neck"),
            &config.vision,
            q.group_size,
            q.bits,
        )?;

        let shared_key = format!("{TEXT_PREFIX}.model.shared");
        let shared = UnifiedEmbedding::from_weights(weights, &shared_key, q.group_size, q.bits)
            .map_err(|e| format!("Nemotron-Parse {e}"))?;
        crate::models::gpt2::validate_embedding_table(
            &shared,
            &shared_key,
            config.text.vocab_size as usize,
            "vocab_size",
            config.text.d_model as usize,
            "d_model",
        )
        .map_err(|e| format!("Nemotron-Parse {e}"))?;
        // The decoder runs its activations in f32 whatever the weights are
        // stored in. It is small (10 layers of width 1024 over a short
        // sequence), and greedy OCR decoding hits near-ties: on the parity
        // page the reference's top-2 margin is 0.04-0.07 at two steps, and
        // bf16 activations against the 8-bit export flip both, while f32
        // activations reproduce the fp32 reference token for token. The
        // quantized projections keep their stored bf16 scales; only the
        // activations are widened.
        let text_dtype = mlxcel_core::dtype::FLOAT32;

        let decoder = NemotronParseDecoder::from_weights(
            weights,
            &format!("{TEXT_PREFIX}.model.decoder"),
            &config.text,
            q,
        )?;

        // v2.0 ties the head to `shared` (the MLX exports still store it
        // explicitly); v1.x ships a real untied head. Either way the head's
        // row count has to equal the vocabulary, because the argmax over its
        // output is fed back into the `shared` gather, which does not
        // range-check a positive index.
        let head_key = format!("{TEXT_PREFIX}.lm_head");
        let lm_head = match weights.get(&format!("{head_key}.weight")) {
            Some(w) => {
                let rows = mlxcel_core::array_shape(w)[0];
                if rows != config.text.vocab_size {
                    return Err(format!(
                        "Nemotron-Parse {head_key}.weight has {rows} rows but vocab_size is {}",
                        config.text.vocab_size
                    ));
                }
                Some(
                    UnifiedLinear::from_weights(weights, &head_key, q.group_size, q.bits)
                        .map_err(|e| format!("Nemotron-Parse {e}"))?,
                )
            }
            None if config.tie_word_embeddings => None,
            None => {
                return Err(format!(
                    "Nemotron-Parse config sets tie_word_embeddings=false but {head_key}.weight \
                     is missing"
                ));
            }
        };

        Ok(Self {
            config,
            vision,
            neck,
            shared,
            decoder,
            lm_head,
            text_dtype,
        })
    }

    pub fn config(&self) -> &NemotronParseConfig {
        &self.config
    }

    /// Activation dtype of the decoder.
    pub fn text_dtype(&self) -> i32 {
        self.text_dtype
    }

    /// Encode normalized `[1, 3, H, W]` pixels into decoder memory
    /// `[1, hp * wp / 4 + 1, d_model]` in the decoder's activation dtype.
    pub fn encode_image(&self, pixel_values: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let out = self.vision.forward(pixel_values).map_err(|e| anyhow!(e))?;
        let hidden = self.neck.forward(&out.features, &out.summary, out.grid);
        // The tower runs in its stored dtype (f16 after the bf16 conversion);
        // the decoder runs in f32, see `from_weights`.
        let hidden = mlxcel_core::astype(&hidden, self.text_dtype);
        mlxcel_core::try_eval(&hidden)
            .map_err(|e| anyhow!("Nemotron-Parse encoder evaluation failed: {e}"))?;
        Ok(hidden)
    }

    /// Fresh decode cache sized to the decoder depth.
    pub fn make_cache(&self) -> NemotronParseSeqCache {
        NemotronParseSeqCache::new(self.decoder.num_layers())
    }

    /// Scaled token embeddings for `[B, T]` ids.
    fn embed(&self, ids: &MlxArray) -> UniquePtr<MlxArray> {
        let e = self.shared.forward(ids);
        let e = mlxcel_core::astype(&e, self.text_dtype);
        if self.config.text.scale_embedding {
            mlxcel_core::multiply_scalar(&e, (self.config.text.d_model as f32).sqrt())
        } else {
            e
        }
    }

    /// Run the decoder over `[B, T]` ids against `encoder_hidden`, advancing
    /// `cache` by `T`. Returns logits `[B, T, vocab]`.
    pub fn decode(
        &self,
        ids: &MlxArray,
        encoder_hidden: &MlxArray,
        cache: &mut NemotronParseSeqCache,
    ) -> UniquePtr<MlxArray> {
        let h = self.decoder.forward(
            &self.embed(ids),
            encoder_hidden,
            cache.offset,
            &mut cache.layers,
        );
        cache.offset += mlxcel_core::array_shape(ids)[1];
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.shared.as_linear(&h),
        }
    }

    /// Greedy next token from the last position of `[1, T, vocab]` logits,
    /// with the repetition penalty applied over `history` when it is not 1.
    fn next_token(&self, logits: &MlxArray, history: &[i32], penalty: f32) -> Result<i32> {
        let shape = mlxcel_core::array_shape(logits);
        let last = mlxcel_core::slice(logits, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
        let last = mlxcel_core::astype(&last, mlxcel_core::dtype::FLOAT32);
        if penalty == 1.0 {
            let idx = mlxcel_core::argmax(&last, -1, false);
            mlxcel_core::try_eval(&idx)
                .map_err(|e| anyhow!("Nemotron-Parse logits evaluation failed: {e}"))?;
            let reshaped = mlxcel_core::reshape(&idx, &[1]);
            return Ok(mlxcel_core::item_i32(&reshaped));
        }
        mlxcel_core::try_eval(&last)
            .map_err(|e| anyhow!("Nemotron-Parse logits evaluation failed: {e}"))?;
        let mut values = mlxcel_core::utils::array_to_vec_f32(&last);
        apply_repetition_penalty(&mut values, history, penalty);
        argmax_f32(&values)
            .map(|i| i as i32)
            .ok_or_else(|| anyhow!("Nemotron-Parse logits are all NaN"))
    }

    /// Encode `pixel_values`, prefill the decoder with `seed`, and decode
    /// greedily until EOS, `max_new_tokens`, `max_sequence_length` decoder
    /// positions, or `cancel`.
    ///
    /// `repetition_penalty` follows `transformers`: it covers every id in the
    /// decoder sequence, the seed included, and 1.0 disables it.
    pub fn generate(
        &self,
        pixel_values: &MlxArray,
        seed: &[i32],
        max_new_tokens: usize,
        repetition_penalty: f32,
        cancel: Option<&AtomicBool>,
    ) -> Result<NemotronParseGeneration> {
        if seed.is_empty() {
            return Err(anyhow!("Nemotron-Parse decoder seed is empty"));
        }
        let bound = self.config.max_sequence_length as usize;
        if seed.len() >= bound {
            return Err(anyhow!(
                "Nemotron-Parse task prompt is {} tokens; the decoder holds at most {bound} \
                 positions",
                seed.len()
            ));
        }
        let vocab = self.config.text.vocab_size;
        if let Some(bad) = seed.iter().find(|id| !(0..vocab).contains(*id)) {
            return Err(anyhow!(
                "Nemotron-Parse seed id {bad} is outside the vocabulary (0..{vocab})"
            ));
        }
        if !(repetition_penalty.is_finite() && repetition_penalty > 0.0) {
            return Err(anyhow!(
                "Nemotron-Parse repetition penalty must be a positive number, got \
                 {repetition_penalty}"
            ));
        }

        let encoder_hidden = self.encode_image(pixel_values)?;
        let mut cache = self.make_cache();
        let mut history: Vec<i32> = seed.to_vec();
        let mut tokens = Vec::new();
        let mut input = mlxcel_core::from_slice_i32(seed, &[1, seed.len() as i32]);
        while tokens.len() < max_new_tokens {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                break;
            }
            let logits = self.decode(&input, &encoder_hidden, &mut cache);
            let next = self.next_token(&logits, &history, repetition_penalty)?;
            if next == self.config.eos_token_id {
                return Ok(NemotronParseGeneration {
                    tokens,
                    hit_eos: true,
                });
            }
            tokens.push(next);
            history.push(next);
            if cache.offset() as usize + 1 >= bound {
                break;
            }
            input = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        }
        Ok(NemotronParseGeneration {
            tokens,
            hit_eos: false,
        })
    }
}
