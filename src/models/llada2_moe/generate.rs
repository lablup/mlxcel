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

//! Block-unmasking generation engine for LLaDA-2 MoE (issue #546).
//!
//! Semi-autoregressive masked diffusion: the sequence lives on a fixed-length
//! int32 canvas whose generation region starts as `<|mask|>` tokens. The
//! prompt is committed into per-layer KV caches block by block; each
//! generation block is then denoised by repeatedly forwarding the block
//! against the frozen prefix and unmasking positions whose chosen-token
//! probability clears a linearly decaying threshold (revealing at least one
//! position per iteration to guarantee progress). A committed block is
//! appended to the caches and its new tokens streamed.
//!
//! The schedule and transfer-mask selection are pure host-side functions
//! ([`block_threshold`], [`transfer_mask`], [`block_num_blocks`],
//! [`truncate_at_eos`]) so they are unit-testable without a model, mirroring
//! `src/models/diffusion_gemma/generate.rs`.

use super::Llada2MoeModel;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::{MlxArray, dtype};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Default generation-block length `B`.
pub const DEFAULT_BLOCK_LENGTH: usize = 32;
/// Default denoising steps per block `S`.
pub const DEFAULT_STEPS: usize = 32;
/// Default confidence threshold `T`.
pub const DEFAULT_THRESHOLD: f32 = 0.95;

/// Caller-facing options for one LLaDA-2 block-unmasking generation.
#[derive(Debug, Clone)]
pub struct Llada2GenerateOptions {
    /// Maximum number of NEW tokens to generate.
    pub max_new_tokens: usize,
    /// Generation-block length `B`.
    pub block_length: usize,
    /// Denoising steps per block `S` (clamped to `min(S, gen_length)`, >= 1).
    pub steps: usize,
    /// Confidence threshold `T` at the first denoising step.
    pub threshold: f32,
    /// Threshold floor at the last denoising step (`<= threshold`).
    pub min_threshold: f32,
    /// Sampling temperature; `<= 0` selects greedy argmax.
    pub temperature: f32,
    /// Top-k cutoff (`0` disables), applied only when `temperature > 0`.
    pub top_k: i32,
    /// Top-p (nucleus) cutoff (`1.0` disables), applied when `temperature > 0`.
    pub top_p: f32,
    /// Min-p cutoff (`0.0` disables), applied when `temperature > 0`.
    pub min_p: f32,
    /// Extra stop ids unioned with the model's EOS set.
    pub extra_eos_token_ids: Vec<i32>,
    /// Cooperative cancellation flag, polled once per denoising step so a
    /// cancelled request aborts within one step instead of finishing a block.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for Llada2GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            block_length: DEFAULT_BLOCK_LENGTH,
            steps: DEFAULT_STEPS,
            threshold: DEFAULT_THRESHOLD,
            min_threshold: DEFAULT_THRESHOLD,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            extra_eos_token_ids: Vec::new(),
            cancel: None,
        }
    }
}

/// Why a LLaDA-2 generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Llada2FinishReason {
    /// Hit `max_new_tokens`.
    Length,
    /// Hit an EOS / stop token.
    Stop,
    /// The streaming callback or cancel flag asked to stop.
    Aborted,
}

/// Timing and work counters for one LLaDA-2 generation call.
#[derive(Debug, Clone)]
pub struct Llada2GenerationStats {
    pub prompt_tokens: usize,
    pub prompt_time_s: f64,
    pub prompt_tps: f64,
    pub generated_tokens: usize,
    pub generation_time_s: f64,
    pub generation_tps: f64,
    /// Total denoising forward passes across all blocks.
    pub denoising_steps: usize,
    pub blocks: usize,
    pub finish_reason: Llada2FinishReason,
}

// ---------------------------------------------------------------------------
// Pure host-side helpers (unit-tested without a model)
// ---------------------------------------------------------------------------

/// Number of blocks covering the prompt plus the requested generation length:
/// `ceil((prompt_len + gen_length) / block_length)`, at least one.
pub fn block_num_blocks(prompt_len: usize, gen_length: usize, block_length: usize) -> usize {
    let block_length = block_length.max(1);
    let total = prompt_len + gen_length.max(1);
    total.div_ceil(block_length).max(1)
}

/// Linearly decaying per-step confidence threshold:
/// `thr = T - (T - min_threshold) * (t - 1) / max(1, S - 1)`.
///
/// `t` counts UP from 1 to `S`, so the first step uses `T` and the last uses
/// `min_threshold`. A single-step block (`S == 1`) always uses `T`.
pub fn block_threshold(t: usize, s: usize, threshold: f32, min_threshold: f32) -> f32 {
    let denom = (s.max(1) - 1).max(1) as f32;
    let progress = (t.saturating_sub(1)) as f32 / denom;
    threshold - (threshold - min_threshold) * progress
}

/// Transfer (unmask) selection for one denoising step.
///
/// Reveals every still-masked (`active`) position whose confidence clears
/// `threshold`. When none clears it but active positions remain, reveals
/// exactly the single highest-confidence active position (lowest block index
/// on an exact tie, matching `argmax`). Guarantees at least one reveal per
/// step whenever any position is active, so a block of `B` masked positions
/// fully unmasks within `B` iterations.
pub fn transfer_mask(active: &[bool], conf: &[f32], threshold: f32) -> Vec<bool> {
    debug_assert_eq!(active.len(), conf.len());
    let mut mask: Vec<bool> = active
        .iter()
        .zip(conf)
        .map(|(&a, &c)| a && c > threshold)
        .collect();

    let any_active = active.iter().any(|&a| a);
    if any_active && !mask.iter().any(|&m| m) {
        let mut best_index: Option<usize> = None;
        let mut best_conf = f32::NEG_INFINITY;
        for (i, (&a, &c)) in active.iter().zip(conf).enumerate() {
            if a && c > best_conf {
                best_conf = c;
                best_index = Some(i);
            }
        }
        if let Some(i) = best_index {
            mask[i] = true;
        }
    }
    mask
}

/// Truncate a generated-token window at the first EOS id (exclusive), matching
/// the output-window rule `x[P..P+gen_length]` truncated at the first EOS.
pub fn truncate_at_eos(tokens: &[i32], eos: &[i32]) -> Vec<i32> {
    match tokens.iter().position(|id| eos.contains(id)) {
        Some(pos) => tokens[..pos].to_vec(),
        None => tokens.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Device <-> host helpers
// ---------------------------------------------------------------------------

fn to_vec_f32(array: &MlxArray) -> Vec<f32> {
    let f32_array = if mlxcel_core::array_dtype(array) == dtype::FLOAT32 {
        None
    } else {
        Some(mlxcel_core::astype(array, dtype::FLOAT32))
    };
    let source = f32_array
        .as_ref()
        .map(|a| a.as_ref().expect("non-null astype output"))
        .unwrap_or(array);
    mlxcel_core::eval(source);
    mlxcel_core::array_to_raw_bytes(source)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().expect("4-byte f32 chunk")))
        .collect()
}

fn to_vec_i32(array: &MlxArray) -> Vec<i32> {
    let i32_array = if mlxcel_core::array_dtype(array) == dtype::INT32 {
        None
    } else {
        Some(mlxcel_core::astype(array, dtype::INT32))
    };
    let source = i32_array
        .as_ref()
        .map(|a| a.as_ref().expect("non-null astype output"))
        .unwrap_or(array);
    mlxcel_core::eval(source);
    mlxcel_core::array_to_raw_bytes(source)
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes(b.try_into().expect("4-byte i32 chunk")))
        .collect()
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

impl Llada2MoeModel {
    /// Sample one token per block position from `logits` `[1, B, vocab]`, and
    /// return `(chosen ids, confidence)` as host vectors of length `B`.
    ///
    /// Confidence is always the softmax probability of the chosen token under
    /// the RAW logits (`softmax_f32(logits)[x0]`), independent of the sampler.
    /// Greedy (`temperature <= 0`) uses a single device argmax; stochastic
    /// sampling reuses the validated per-token `sample_token_optimized` for
    /// each block position (handling temperature / top-k / top-p / min-p).
    fn sample_block(
        &self,
        logits: &MlxArray,
        options: &Llada2GenerateOptions,
    ) -> (Vec<i32>, Vec<f32>) {
        let shape = mlxcel_core::array_shape(logits);
        let block = shape[1];
        let vocab = shape[2];
        let probs = mlxcel_core::softmax_precise(logits, -1);

        let x0 = if options.temperature <= 0.0 {
            mlxcel_core::astype(&mlxcel_core::argmax(logits, -1, false), dtype::INT32)
        } else {
            let cfg = SamplingConfig {
                temperature: options.temperature,
                top_k: options.top_k,
                top_p: options.top_p,
                min_p: options.min_p,
                seed: None,
                ..SamplingConfig::default()
            };
            let mut ids = Vec::with_capacity(block as usize);
            for j in 0..block {
                let row = mlxcel_core::slice(logits, &[0, j, 0], &[1, j + 1, vocab]);
                let (token, _logprobs) =
                    mlxcel_core::sampling::sample_token_optimized(&row, &cfg, &[]);
                mlxcel_core::eval(&token);
                ids.push(mlxcel_core::item_i32(&token));
            }
            mlxcel_core::from_slice_i32(&ids, &[1, block])
        };

        let conf = mlxcel_core::squeeze_axis(
            &mlxcel_core::take_along_axis(&probs, &mlxcel_core::expand_dims(&x0, -1), -1),
            -1,
        );
        (to_vec_i32(&x0), to_vec_f32(&conf))
    }

    /// Stream one LLaDA-2 block-unmasking generation.
    ///
    /// `on_token` receives each committed generated token id and returns
    /// `false` to abort. EOS ids stop generation WITHOUT being emitted,
    /// matching the diffusion worker convention.
    pub fn generate_llada2_streaming<F: FnMut(i32) -> bool>(
        &self,
        prompt_tokens: &[i32],
        options: &Llada2GenerateOptions,
        mut on_token: F,
    ) -> Result<Llada2GenerationStats, String> {
        if prompt_tokens.is_empty() {
            return Err("LLaDA-2 MoE: prompt must contain at least one token".to_string());
        }
        if options.block_length == 0 {
            return Err("LLaDA-2 block length must be a positive integer".to_string());
        }
        if !(0.0..=1.0).contains(&options.threshold)
            || !(0.0..=1.0).contains(&options.min_threshold)
        {
            return Err("LLaDA-2 thresholds must be between 0 and 1".to_string());
        }
        // The mask token seeds the generation canvas and is embedded every step;
        // an out-of-range id (e.g. a misconfigured default against a small
        // vocab) would fault the embedding lookup, so reject it up front.
        if self.mask_token_id < 0 || self.mask_token_id >= self.vocab_size {
            return Err(format!(
                "LLaDA-2 mask_token_id {} is out of range for vocab_size {}",
                self.mask_token_id, self.vocab_size
            ));
        }

        let prompt_len = prompt_tokens.len();
        let gen_length = options.max_new_tokens.max(1);
        let block_length = options.block_length.max(1);
        // Clamp steps to the generation length (at least one).
        let steps = options.steps.max(1).min(gen_length).max(1);
        let num_blocks = block_num_blocks(prompt_len, gen_length, block_length);
        let total_len = num_blocks * block_length;
        let output_end = prompt_len + gen_length;

        let mut eos_ids = self.eos_token_ids.clone();
        for &id in &options.extra_eos_token_ids {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }

        // Canvas: prompt ids followed by mask tokens.
        let mut canvas = vec![self.mask_token_id; total_len];
        canvas[..prompt_len].copy_from_slice(prompt_tokens);

        let mut caches = self.make_diffusion_caches();

        // 1. Prefill: commit full prompt blocks into the prefix KV caches.
        let prefill_start = Instant::now();
        let prefill_blocks = prompt_len / block_length;
        for pb in 0..prefill_blocks {
            let s = pb * block_length;
            let e = s + block_length;
            let ids = mlxcel_core::from_slice_i32(&canvas[s..e], &[1, block_length as i32]);
            let _ = self.forward_append(&ids, &mut caches, s as i32);
            for cache in &caches {
                cache.eval_state();
            }
            mlxcel_core::clear_memory_cache();
        }
        let prompt_time_s = prefill_start.elapsed().as_secs_f64().max(1e-9);
        let generation_start = Instant::now();

        let mut generated_tokens = 0usize;
        let mut denoising_steps = 0usize;
        let mut blocks = 0usize;
        let mut finish_reason = Llada2FinishReason::Length;

        // 2. Block loop over the generation region.
        'blocks: for nb in prefill_blocks..num_blocks {
            let s = nb * block_length;
            let e = s + block_length;

            for t in 1..=steps {
                if options
                    .cancel
                    .as_ref()
                    .is_some_and(|c| c.load(Ordering::Relaxed))
                {
                    finish_reason = Llada2FinishReason::Aborted;
                    break 'blocks;
                }

                let active: Vec<bool> = (0..block_length)
                    .map(|j| canvas[s + j] == self.mask_token_id)
                    .collect();
                if !active.iter().any(|&a| a) {
                    break; // block fully unmasked
                }

                let ids = mlxcel_core::from_slice_i32(&canvas[s..e], &[1, block_length as i32]);
                let logits = self.forward_readonly_logits(&ids, &caches, s as i32);
                let logits = mlxcel_core::astype(&logits, dtype::FLOAT32);
                let (x0, conf) = self.sample_block(&logits, options);
                denoising_steps += 1;

                let thr = block_threshold(t, steps, options.threshold, options.min_threshold);
                let transfer = transfer_mask(&active, &conf, thr);
                for (j, &reveal) in transfer.iter().enumerate() {
                    if reveal {
                        canvas[s + j] = x0[j];
                    }
                }
                mlxcel_core::clear_memory_cache();
            }

            // 3. Commit: append the block K/V to the prefix caches.
            let ids = mlxcel_core::from_slice_i32(&canvas[s..e], &[1, block_length as i32]);
            let _ = self.forward_append(&ids, &mut caches, s as i32);
            for cache in &caches {
                cache.eval_state();
            }
            blocks += 1;

            // Stream the newly generated positions in [max(s, P), min(e, P+gen)).
            let stream_start = s.max(prompt_len);
            let stream_end = e.min(output_end);
            for &token in &canvas[stream_start..stream_end] {
                if eos_ids.contains(&token) {
                    finish_reason = Llada2FinishReason::Stop;
                    break 'blocks;
                }
                generated_tokens += 1;
                if !on_token(token) {
                    finish_reason = Llada2FinishReason::Aborted;
                    break 'blocks;
                }
                if generated_tokens >= gen_length {
                    finish_reason = Llada2FinishReason::Length;
                    break 'blocks;
                }
            }
            mlxcel_core::clear_memory_cache();
        }

        let generation_time_s = generation_start.elapsed().as_secs_f64().max(1e-9);
        Ok(Llada2GenerationStats {
            prompt_tokens: prompt_len,
            prompt_time_s,
            prompt_tps: prompt_len as f64 / prompt_time_s,
            generated_tokens,
            generation_time_s,
            generation_tps: generated_tokens as f64 / generation_time_s,
            denoising_steps,
            blocks,
            finish_reason,
        })
    }
}
