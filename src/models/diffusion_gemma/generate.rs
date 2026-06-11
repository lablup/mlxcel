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

//! Block-diffusion generation engine for DiffusionGemma (issue #217).
//!
//! Mirrors `stream_diffusion_generate` in
//! `references/mlx-vlm/mlx_vlm/generate/diffusion.py` for the batch-1,
//! no-padding, dynamic-cache CLI case: per block, initialize a random
//! canvas, denoise it for up to `max_denoising_steps` iterations under a
//! linear temperature schedule with self-conditioning, accept positions via
//! the entropy-bound (default) or confidence-threshold sampler, early-stop
//! on a stable-and-confident canvas, then commit the block to the KV-cached
//! prefix and stream its tokens.

use super::{DiffusionGemmaModel, DiffusionStoppingConfig};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use std::collections::VecDeque;
use std::time::Instant;

/// Default smallest canvas allocated for the tail of a generation.
pub const DEFAULT_MIN_CANVAS_LENGTH: usize = 64;

/// Default prompt prefill chunk size. Matches the chunked-prefill granularity
/// the memory-estimation activation model assumes for bounded prefill.
pub const DEFAULT_PREFILL_CHUNK_SIZE: usize = 512;

/// Which per-step acceptance rule the denoising loop uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionSamplerKind {
    /// Accept the largest low-entropy prefix whose summed entropy stays
    /// within the checkpoint's `entropy_bound` (default).
    EntropyBound,
    /// Reveal positions whose denoiser-token probability clears a fixed
    /// threshold; revealed positions keep their draft tokens.
    ConfidenceThreshold,
}

/// Caller-facing options for one diffusion generation call.
#[derive(Debug, Clone)]
pub struct DiffusionGenerateOptions {
    /// Maximum number of NEW tokens to generate (CLI `-n/--max-tokens`).
    pub max_new_tokens: usize,
    /// Denoiser sampling temperature (CLI `--temp`); `<= 0` means argmax.
    pub temperature: f32,
    pub sampler: DiffusionSamplerKind,
    /// Confidence threshold for [`DiffusionSamplerKind::ConfidenceThreshold`].
    pub confidence_threshold: f32,
    /// Override for the checkpoint's `max_denoising_steps`.
    pub max_denoising_steps: Option<usize>,
    /// Smallest canvas allocated for the generation tail.
    pub min_canvas_length: usize,
    /// Optional cap on the per-block canvas length (clamped to the model's
    /// `canvas_length`).
    pub max_canvas_length: Option<usize>,
    /// Always allocate the model's full `canvas_length` per block.
    pub full_canvas: bool,
    /// Extra stop ids (tokenizer / template / CLI stop ids) unioned with the
    /// checkpoint EOS set.
    pub extra_eos_token_ids: Vec<i32>,
    /// Prompt prefill chunk size; prompts longer than this are prefilled in
    /// chunks with the cache evaluated between chunks.
    pub prefill_chunk_size: usize,
}

impl Default for DiffusionGenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.0,
            sampler: DiffusionSamplerKind::EntropyBound,
            confidence_threshold: 0.9,
            max_denoising_steps: None,
            min_canvas_length: DEFAULT_MIN_CANVAS_LENGTH,
            max_canvas_length: None,
            full_canvas: false,
            extra_eos_token_ids: Vec::new(),
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
        }
    }
}

/// Why a diffusion generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionFinishReason {
    /// Hit `max_new_tokens`.
    Length,
    /// Hit an EOS / stop token.
    Stop,
    /// The streaming callback asked to stop.
    Aborted,
}

/// Timing and work counters for one diffusion generation call.
#[derive(Debug, Clone)]
pub struct DiffusionGenerationStats {
    pub prompt_tokens: usize,
    pub prompt_time_s: f64,
    pub prompt_tps: f64,
    pub generated_tokens: usize,
    pub generation_time_s: f64,
    pub generation_tps: f64,
    /// Total canvas positions denoised (sum of block canvas lengths).
    pub canvas_tokens: usize,
    /// Total denoising forward passes across all blocks.
    pub denoising_steps: usize,
    /// Total work = sum over blocks of `canvas_length * steps`.
    pub work_tokens: usize,
    pub canvas_tps: f64,
    pub work_tps: f64,
    pub blocks: usize,
    pub finish_reason: DiffusionFinishReason,
}

// ---------------------------------------------------------------------------
// Pure host-side helpers (unit-tested without a model)
// ---------------------------------------------------------------------------

/// Linear denoising temperature schedule:
/// `tau = t_min + (t_max - t_min) * (cur_step / max_steps)`.
/// `cur_step` counts DOWN from `max_steps` to 1, so the first iteration runs
/// near `t_max` and the last near `t_min`.
pub(crate) fn linear_schedule_temperature(
    cur_step: usize,
    max_steps: usize,
    t_min: f32,
    t_max: f32,
) -> f32 {
    t_min + (t_max - t_min) * (cur_step as f32 / max_steps as f32)
}

/// Entropy-bound acceptance count over ASCENDING-sorted entropies.
///
/// The reference rule `(cumsum - cummax) <= bound` over an ascending
/// non-negative sequence reduces to "the sum of entropies strictly before
/// rank `i` is `<= bound`", which is monotone, so acceptance is a prefix.
/// Returns `max(1, count)` (at least one position is always accepted),
/// or 0 for an empty input.
pub(crate) fn entropy_bound_accept_count(sorted_entropies: &[f32], bound: f32) -> usize {
    if sorted_entropies.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    let mut prefix = 0.0f32;
    for &entropy in sorted_entropies {
        if prefix <= bound {
            count += 1;
            prefix += entropy;
        } else {
            break;
        }
    }
    count.max(1)
}

/// Build the boolean acceptance mask for the entropy-bound sampler: accept
/// the `k` lowest-entropy positions, with `k` from
/// [`entropy_bound_accept_count`]. Stable ascending order (ties keep the
/// lower position index, matching `mx.argsort`).
pub(crate) fn entropy_bound_acceptance_mask(entropies: &[f32], bound: f32) -> Vec<bool> {
    let mut order: Vec<usize> = (0..entropies.len()).collect();
    // `total_cmp` is a total order, so the stable sort never sees an
    // inconsistent comparator (a NaN entropy would make `partial_cmp` return
    // `None`); NaN sorts last and is therefore never accepted, which is the
    // desired behavior for a maximally uncertain position.
    order.sort_by(|&a, &b| entropies[a].total_cmp(&entropies[b]));
    let sorted: Vec<f32> = order.iter().map(|&i| entropies[i]).collect();
    let accept = entropy_bound_accept_count(&sorted, bound);
    let mut mask = vec![false; entropies.len()];
    for &position in order.iter().take(accept) {
        mask[position] = true;
    }
    mask
}

/// Confidence-threshold transfer mask
/// (`_diffusion_confidence_transfer_mask` in the reference).
///
/// Accepts unrevealed positions whose confidence clears `threshold`; when
/// none clears it but unrevealed positions remain, forces the single
/// highest-confidence unrevealed position (first index on exact ties,
/// matching `mx.argmax`). `force_all` accepts every unrevealed position.
pub(crate) fn confidence_transfer_mask(
    confidence: &[f32],
    unrevealed: &[bool],
    threshold: f32,
    force_all: bool,
) -> Vec<bool> {
    debug_assert_eq!(confidence.len(), unrevealed.len());
    if force_all {
        return unrevealed.to_vec();
    }
    let mut mask: Vec<bool> = unrevealed
        .iter()
        .zip(confidence)
        .map(|(&u, &c)| u && c >= threshold)
        .collect();
    let has_unrevealed = unrevealed.iter().any(|&u| u);
    if has_unrevealed && !mask.iter().any(|&m| m) {
        let mut best_index = None;
        let mut best_confidence = f32::NEG_INFINITY;
        for (i, (&u, &c)) in unrevealed.iter().zip(confidence).enumerate() {
            if u && c > best_confidence {
                best_confidence = c;
                best_index = Some(i);
            }
        }
        if let Some(i) = best_index {
            mask[i] = true;
        }
    }
    mask
}

/// Per-block canvas length rule:
/// `full_canvas` always allocates the model canvas; otherwise
/// `min(max_canvas, max(remaining, min_canvas))`.
pub(crate) fn canvas_length_for(
    remaining_tokens: usize,
    min_canvas: usize,
    max_canvas: usize,
    model_canvas: usize,
    full_canvas: bool,
) -> usize {
    if full_canvas {
        model_canvas
    } else {
        max_canvas.min(remaining_tokens.max(min_canvas))
    }
}

/// Deterministic debug canvas pattern (env `MLXCEL_DIFFUSION_DEBUG_CANVAS=1`):
/// `token[i] = ((i + 1) * 7919 + k * 104729) % vocab_size`, where `i` is the
/// 0-based canvas position and `k` the 0-based global randomization-call
/// counter within one generate invocation.
pub(crate) fn debug_canvas_pattern(length: usize, vocab_size: i64, k: i64) -> Vec<i32> {
    (0..length as i64)
        .map(|i| (((i + 1) * 7919 + k * 104_729) % vocab_size) as i32)
        .collect()
}

/// Whether the deterministic debug canvas mode is active.
pub fn diffusion_debug_canvas_enabled() -> bool {
    std::env::var("MLXCEL_DIFFUSION_DEBUG_CANVAS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Stability tracker mirroring `_diffusion_stable_and_confident`'s history
/// semantics: the stability verdict is computed against the EXISTING history
/// (only when it is already full), and the current canvas is appended
/// afterward (evicting the oldest entry).
pub(crate) struct StabilityTracker {
    history: VecDeque<Vec<i32>>,
    threshold: usize,
}

impl StabilityTracker {
    pub(crate) fn new(threshold: usize) -> Self {
        Self {
            history: VecDeque::new(),
            threshold,
        }
    }

    /// Record one argmax canvas; returns true when the canvas equals every
    /// entry of a FULL history (checked BEFORE this canvas is pushed).
    pub(crate) fn observe(&mut self, canvas: &[i32]) -> bool {
        let stable = self.history.len() == self.threshold
            && self.history.iter().all(|previous| previous == canvas);
        self.history.push_back(canvas.to_vec());
        if self.history.len() > self.threshold {
            self.history.pop_front();
        }
        stable
    }
}

// ---------------------------------------------------------------------------
// Device <-> host helpers
// ---------------------------------------------------------------------------

fn to_vec_f32(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(array);
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

fn bool_mask_array(mask: &[bool]) -> UniquePtr<MlxArray> {
    let ints: Vec<i32> = mask.iter().map(|&m| i32::from(m)).collect();
    let arr = mlxcel_core::from_slice_i32(&ints, &[1, ints.len() as i32]);
    mlxcel_core::astype(&arr, dtype::BOOL)
}

/// Canvas randomization source. In normal mode draws uniform random ids in
/// `[0, vocab)` from MLX's global RNG (seedable via `--seed`); in debug mode
/// (`MLXCEL_DIFFUSION_DEBUG_CANVAS=1`) every call returns the deterministic
/// pattern from [`debug_canvas_pattern`] with a per-invocation call counter.
struct CanvasRng {
    vocab_size: i32,
    debug: bool,
    calls: u64,
}

impl CanvasRng {
    fn new(vocab_size: i32) -> Self {
        Self {
            vocab_size,
            debug: diffusion_debug_canvas_enabled(),
            calls: 0,
        }
    }

    fn canvas(&mut self, length: usize) -> UniquePtr<MlxArray> {
        let k = self.calls;
        self.calls += 1;
        if self.debug {
            let ids = debug_canvas_pattern(length, self.vocab_size as i64, k as i64);
            mlxcel_core::from_slice_i32(&ids, &[1, length as i32])
        } else {
            // SAFETY: null key selects MLX's global RNG state.
            unsafe {
                mlxcel_core::random_randint(
                    0,
                    self.vocab_size,
                    &[1, length as i32],
                    dtype::INT32,
                    std::ptr::null(),
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

struct DenoiseOutcome {
    /// Committed canvas ids for this block.
    commit: UniquePtr<MlxArray>,
    steps: usize,
}

impl DiffusionGemmaModel {
    /// Stream one block-diffusion generation.
    ///
    /// `on_token` receives each emitted token id and returns `false` to
    /// abort. EOS ids stop the generation WITHOUT being emitted. Mirrors
    /// `stream_diffusion_generate` (batch-1, no padding, dynamic cache).
    pub fn generate_diffusion_streaming<F: FnMut(i32) -> bool>(
        &self,
        prompt_tokens: &[i32],
        options: &DiffusionGenerateOptions,
        mut on_token: F,
    ) -> Result<DiffusionGenerationStats, String> {
        if prompt_tokens.is_empty() {
            return Err("DiffusionGemma: prompt must contain at least one token".to_string());
        }
        if options.min_canvas_length == 0 {
            return Err("diffusion min canvas length must be a positive integer".to_string());
        }
        if options.max_canvas_length == Some(0) {
            return Err("diffusion max canvas length must be a positive integer".to_string());
        }
        if !(0.0..=1.0).contains(&options.confidence_threshold) {
            return Err("diffusion threshold must be between 0 and 1".to_string());
        }
        if options.prefill_chunk_size == 0 {
            return Err("prefill chunk size must be a positive integer".to_string());
        }

        let generation_config = &self.generation_config;
        let max_denoising_steps = options
            .max_denoising_steps
            .unwrap_or(generation_config.max_denoising_steps)
            .max(1);
        let model_canvas = self.canvas_length.max(1);
        let max_canvas = if options.full_canvas {
            model_canvas
        } else {
            model_canvas.min(options.max_canvas_length.unwrap_or(model_canvas))
        };
        let min_canvas = max_canvas.min(options.min_canvas_length);
        let max_new_tokens = options.max_new_tokens.max(1);
        let vocab_size = self.text.config.vocab_size as i32;

        let mut eos_ids = self.eos_token_ids.clone();
        for &id in &options.extra_eos_token_ids {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }

        let mut caches: Vec<KVCache> = self.make_diffusion_caches();
        let mut rng = CanvasRng::new(vocab_size);
        let debug_mode = rng.debug;

        // Prompt prefill (chunked for long prompts; batch-1 / no padding
        // always holds on this path).
        let prefill_start = Instant::now();
        if prompt_tokens.len() > options.prefill_chunk_size {
            for chunk in prompt_tokens.chunks(options.prefill_chunk_size) {
                let ids = mlxcel_core::from_slice_i32(chunk, &[1, chunk.len() as i32]);
                let _ = self.forward_encoder(&ids, &mut caches, None);
                for cache in &caches {
                    cache.eval_state();
                }
                mlxcel_core::clear_memory_cache();
            }
        } else {
            let ids = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
            let _ = self.forward_encoder(&ids, &mut caches, None);
            for cache in &caches {
                cache.eval_state();
            }
        }
        let prompt_time_s = prefill_start.elapsed().as_secs_f64();
        let generation_start = Instant::now();

        // Float view of the embedding table for self-conditioning soft
        // embeddings, dequantized ONCE per generate call (the reference
        // measured quantized_matmul(transpose=false) several times slower).
        let soft_embedding_table = self.text.embed_tokens.dequantized_weight();
        let table_dtype = mlxcel_core::array_dtype(&soft_embedding_table);

        let mut generated_tokens = 0usize;
        let mut canvas_tokens = 0usize;
        let mut denoising_steps = 0usize;
        let mut work_tokens = 0usize;
        let mut blocks = 0usize;
        let mut stopped = false;
        let mut finish_reason = DiffusionFinishReason::Length;
        let mut pending_commit: Option<UniquePtr<MlxArray>> = None;

        while generated_tokens < max_new_tokens && !stopped {
            // Append the previous block's FULL committed canvas to the
            // KV-cached prefix before denoising the next block.
            if let Some(previous) = pending_commit.take() {
                let _ = self.forward_encoder(&previous, &mut caches, None);
            }

            let remaining = max_new_tokens - generated_tokens;
            let canvas_len = canvas_length_for(
                remaining,
                min_canvas,
                max_canvas,
                model_canvas,
                options.full_canvas,
            );

            let outcome = self.denoise_block(
                &caches,
                canvas_len,
                max_denoising_steps,
                options,
                &soft_embedding_table,
                table_dtype,
                &mut rng,
            )?;
            canvas_tokens += canvas_len;
            denoising_steps += outcome.steps;
            work_tokens += canvas_len * outcome.steps;

            mlxcel_core::eval(&outcome.commit);
            let commit_ids = to_vec_i32(&outcome.commit);
            if debug_mode {
                let rendered: Vec<String> = commit_ids.iter().map(|id| id.to_string()).collect();
                eprintln!(
                    "DIFFUSION_COMMIT block={} ids={}",
                    blocks,
                    rendered.join(",")
                );
            }
            blocks += 1;

            for &token_id in &commit_ids {
                generated_tokens += 1;
                if eos_ids.contains(&token_id) {
                    stopped = true;
                    finish_reason = DiffusionFinishReason::Stop;
                    break;
                }
                if !on_token(token_id) {
                    stopped = true;
                    finish_reason = DiffusionFinishReason::Aborted;
                    break;
                }
                if generated_tokens >= max_new_tokens {
                    stopped = true;
                    finish_reason = DiffusionFinishReason::Length;
                    break;
                }
            }

            if !stopped {
                pending_commit = Some(outcome.commit);
                mlxcel_core::clear_memory_cache();
            }
        }

        let generation_time_s = generation_start.elapsed().as_secs_f64().max(1e-9);
        Ok(DiffusionGenerationStats {
            prompt_tokens: prompt_tokens.len(),
            prompt_time_s,
            prompt_tps: prompt_tokens.len() as f64 / prompt_time_s.max(1e-9),
            generated_tokens,
            generation_time_s,
            generation_tps: generated_tokens as f64 / generation_time_s,
            canvas_tokens,
            denoising_steps,
            work_tokens,
            canvas_tps: canvas_tokens as f64 / generation_time_s,
            work_tps: work_tokens as f64 / generation_time_s,
            blocks,
            finish_reason,
        })
    }

    /// Denoise one canvas block. Returns the committed canvas ids and the
    /// number of denoising forward passes executed.
    #[allow(clippy::too_many_arguments)]
    fn denoise_block(
        &self,
        caches: &[KVCache],
        canvas_len: usize,
        max_denoising_steps: usize,
        options: &DiffusionGenerateOptions,
        soft_embedding_table: &MlxArray,
        table_dtype: i32,
        rng: &mut CanvasRng,
    ) -> Result<DenoiseOutcome, String> {
        let generation_config = &self.generation_config;
        let entropy_bound = generation_config.entropy_bound;
        let stopping: Option<DiffusionStoppingConfig> = generation_config.stopping;
        let mut tracker = stopping.map(|s| StabilityTracker::new(s.stability_threshold));

        let mut current_canvas = rng.canvas(canvas_len);
        let mut self_conditioning: Option<UniquePtr<MlxArray>> = None;
        let mut steps = 0usize;

        // Confidence-threshold sampler state.
        let mut reveal_mask = vec![false; canvas_len];
        let mut draft_canvas = mlxcel_core::copy(&current_canvas);

        // Committed canvas: the argmax canvas of the LAST executed step in
        // EVERY exit path, matching the reference (its post-loop
        // `current_canvas = argmax_canvas` overwrites the confidence
        // sampler's draft assignment, which is dead code there).
        let mut commit = mlxcel_core::copy(&current_canvas);

        for cur_step in (1..=max_denoising_steps).rev() {
            steps += 1;
            let self_conditioning_ref = self_conditioning
                .as_ref()
                .map(|sc| sc.as_ref().expect("non-null self-conditioning embeddings"));
            let logits = self.forward_canvas(&current_canvas, caches, self_conditioning_ref);
            let logits = mlxcel_core::astype(&logits, dtype::FLOAT32);
            let tau = linear_schedule_temperature(
                cur_step,
                max_denoising_steps,
                generation_config.t_min,
                generation_config.t_max,
            );
            let logits = mlxcel_core::multiply_scalar(&logits, 1.0 / tau);

            let argmax_canvas =
                mlxcel_core::astype(&mlxcel_core::argmax(&logits, -1, false), dtype::INT32);
            commit = mlxcel_core::copy(&argmax_canvas);
            if cur_step == 1 {
                break;
            }

            let denoiser_canvas = if options.temperature <= 0.0 {
                mlxcel_core::copy(&argmax_canvas)
            } else {
                let sampling_logits = if options.temperature != 1.0 {
                    mlxcel_core::multiply_scalar(&logits, 1.0 / options.temperature)
                } else {
                    mlxcel_core::copy(&logits)
                };
                mlxcel_core::astype(
                    &mlxcel_core::random_categorical(&sampling_logits, -1),
                    dtype::INT32,
                )
            };

            match options.sampler {
                DiffusionSamplerKind::EntropyBound => {
                    // probs / entropy / soft-embedding chain over the
                    // schedule-divided f32 logits.
                    let log_probs = mlxcel_core::subtract(
                        &logits,
                        &mlxcel_core::logsumexp_axis(&logits, -1, true),
                    );
                    let probs = mlxcel_core::exp(&log_probs);
                    let entropy = mlxcel_core::negative(&mlxcel_core::sum_axis(
                        &mlxcel_core::multiply(&probs, &log_probs),
                        -1,
                        false,
                    ));
                    let soft_embeddings = mlxcel_core::multiply_scalar(
                        &mlxcel_core::matmul(
                            &mlxcel_core::astype(&probs, table_dtype),
                            soft_embedding_table,
                        ),
                        self.embed_scale,
                    );

                    let entropy_host = to_vec_f32(&entropy);
                    let acceptance = entropy_bound_acceptance_mask(&entropy_host, entropy_bound);
                    let condition = bool_mask_array(&acceptance);
                    let accepted_canvas =
                        mlxcel_core::where_cond(&condition, &denoiser_canvas, &current_canvas);
                    let fresh = rng.canvas(canvas_len);
                    current_canvas = mlxcel_core::where_cond(&condition, &accepted_canvas, &fresh);

                    // Early stop: stable argmax history AND low mean entropy.
                    if let (Some(tracker), Some(stop)) = (tracker.as_mut(), stopping.as_ref()) {
                        let argmax_host = to_vec_i32(&argmax_canvas);
                        if tracker.observe(&argmax_host) {
                            let mean_entropy = entropy_host.iter().map(|&e| e as f64).sum::<f64>()
                                / entropy_host.len().max(1) as f64;
                            if mean_entropy < f64::from(stop.confidence_threshold) {
                                break;
                            }
                        }
                    }

                    self_conditioning = Some(soft_embeddings);
                }
                DiffusionSamplerKind::ConfidenceThreshold => {
                    // confidence = probability of the denoiser token.
                    let token_logits = mlxcel_core::squeeze_axis(
                        &mlxcel_core::take_along_axis(
                            &logits,
                            &mlxcel_core::expand_dims(&denoiser_canvas, -1),
                            -1,
                        ),
                        -1,
                    );
                    let confidence = mlxcel_core::exp(&mlxcel_core::subtract(
                        &token_logits,
                        &mlxcel_core::logsumexp_axis(&logits, -1, false),
                    ));
                    let confidence_host = to_vec_f32(&confidence);
                    let unrevealed: Vec<bool> = reveal_mask.iter().map(|&r| !r).collect();
                    let transfer = confidence_transfer_mask(
                        &confidence_host,
                        &unrevealed,
                        options.confidence_threshold,
                        false,
                    );

                    let transfer_condition = bool_mask_array(&transfer);
                    let accepted_canvas = mlxcel_core::where_cond(
                        &transfer_condition,
                        &denoiser_canvas,
                        &draft_canvas,
                    );
                    let keep: Vec<bool> = reveal_mask
                        .iter()
                        .zip(&transfer)
                        .map(|(&r, &t)| r || t)
                        .collect();
                    let keep_condition = bool_mask_array(&keep);
                    let fresh = rng.canvas(canvas_len);
                    current_canvas =
                        mlxcel_core::where_cond(&keep_condition, &accepted_canvas, &fresh);
                    draft_canvas = mlxcel_core::where_cond(
                        &transfer_condition,
                        &accepted_canvas,
                        &draft_canvas,
                    );
                    for (revealed, &t) in reveal_mask.iter_mut().zip(&transfer) {
                        *revealed = *revealed || t;
                    }

                    if reveal_mask.iter().all(|&r| r) {
                        // All positions revealed: stop denoising. The commit
                        // stays this step's argmax canvas, like the reference.
                        break;
                    }

                    if let (Some(tracker), Some(stop)) = (tracker.as_mut(), stopping.as_ref()) {
                        let argmax_host = to_vec_i32(&argmax_canvas);
                        if tracker.observe(&argmax_host) {
                            let log_probs = mlxcel_core::subtract(
                                &logits,
                                &mlxcel_core::logsumexp_axis(&logits, -1, true),
                            );
                            let probs = mlxcel_core::exp(&log_probs);
                            let entropy = mlxcel_core::negative(&mlxcel_core::sum_axis(
                                &mlxcel_core::multiply(&probs, &log_probs),
                                -1,
                                false,
                            ));
                            let mean_entropy =
                                mlxcel_core::item_f32(&mlxcel_core::mean_all(&entropy));
                            if f64::from(mean_entropy) < f64::from(stop.confidence_threshold) {
                                break;
                            }
                        }
                    }

                    // Self-conditioning for the next step (the confidence
                    // branch computes it from the same schedule-divided
                    // logits via precise softmax).
                    let probs = mlxcel_core::softmax_precise(&logits, -1);
                    self_conditioning = Some(mlxcel_core::multiply_scalar(
                        &mlxcel_core::matmul(
                            &mlxcel_core::astype(&probs, table_dtype),
                            soft_embedding_table,
                        ),
                        self.embed_scale,
                    ));
                }
            }
        }

        Ok(DenoiseOutcome { commit, steps })
    }
}
