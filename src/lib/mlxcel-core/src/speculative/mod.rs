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

//! Speculative decoding for accelerated inference
//!
//! Uses a small "draft" model to generate candidate tokens, then verifies
//! them in batch with the main model. Accepted tokens skip individual
//! forward passes, improving throughput when the draft model's predictions
//! match the main model's.
//!
//! Algorithm:
//! 1. Prefill prompt through both models
//! 2. Sample first token from main model
//! 3. Loop:
//!    a. Draft: generate `num_draft` tokens with draft model
//!    b. Verify: forward [current + draft tokens] through main model
//!    c. Accept matching prefix, rewind caches for rejected tokens
//!    d. Continue from the divergence point
//!
//! ## Sibling modules
//!
//! - [`mtp`] — Multi-Token Prediction (MTP) round-loop generator for the
//!   Gemma 4 assistant drafter family. Peer code path to
//!   [`SpeculativeGenerator`] with fundamentally different semantics
//!   (drafter has no own KV cache; verify is a single forward over the
//!   whole draft block; rollback uses per-row tail-zero rather than
//!   `trim_caches`). See [`mtp::MtpGenerator`].

pub mod mtp;

use crate::cache::can_trim_prompt_cache;
use crate::ffi;
use crate::ffi::MlxThreadLocalStream;
use crate::generate::{GenerationStats, LanguageModel, SamplingConfig};
use crate::generation_policy::{initial_token_history, merged_eos_token_ids};
use crate::hardware;
use crate::layers::KVCache;
use crate::sampling::{sample_token_optimized, TokenBiasMap};
use crate::streams::{install_thread_local_default_stream, new_thread_local_generation_stream};
use crate::utils::{align_to_na_tile, create_padded_prefill_mask};
use cxx::UniquePtr;
use std::borrow::Cow;
use std::time::Instant;

/// Default chunk size for chunked prefill in speculative decoding.
///
/// Mirrors the `prefill_step_size` default used by upstream mlx-lm
/// `speculative_generate_step` (512 tokens). Processing the prompt in
/// chunks reduces peak memory pressure for long prompts and ensures the
/// loop can correctly reserve the last token for the first speculation step.
const PREFILL_STEP_SIZE: usize = 512;

/// Returns true when the current hardware is M5+ with a Neural Accelerator
/// and tile-aligned verification batching should be applied.
#[inline]
fn should_align_verification() -> bool {
    let hw = hardware::get_hardware();
    hw.has_neural_accelerator && hw.macos_supports_na
}

/// Speculative decoding generator
///
/// Uses a draft model to propose candidate tokens and a main model to verify them.
/// When the draft model's predictions match, multiple tokens are accepted per
/// main model forward pass, improving throughput.
pub struct SpeculativeGenerator {
    main_caches: Vec<KVCache>,
    draft_caches: Vec<KVCache>,
    generated_tokens: Vec<i32>,
    /// Thread-local generation stream — see `mlxcel_core::streams`
    /// (issue #556). Resolves to a per-thread `MlxStream` on the
    /// worker thread that calls `generate`, so dispatch and
    /// synchronization stay paired even when the generator is moved
    /// across threads after construction.
    generation_stream: Option<UniquePtr<MlxThreadLocalStream>>,
    /// Cached per-generator `TokenBiasMap` resolved from a `LangBiasConfig`.
    ///
    /// **Axis B invariant**: the bias is applied **only** to the target
    /// (main) model's sampler. The draft model must keep seeing the
    /// unmodified policy so its candidate distribution stays aligned with
    /// its own weights; otherwise the accept/reject comparison becomes
    /// biased on two different policies and speculative acceptance rate
    /// collapses. See [`Self::compose_target_sampling`] and
    /// [`Self::draft_sampling`] — only the former injects the cached bias.
    token_bias: TokenBiasMap,
}

impl SpeculativeGenerator {
    /// Create a new speculative generator
    pub fn new(main_num_layers: usize, draft_num_layers: usize) -> Self {
        Self {
            main_caches: (0..main_num_layers).map(|_| KVCache::new()).collect(),
            draft_caches: (0..draft_num_layers).map(|_| KVCache::new()).collect(),
            generated_tokens: Vec::new(),
            generation_stream: new_thread_local_generation_stream(),
            token_bias: TokenBiasMap::default(),
        }
    }

    /// Attach a pre-resolved `TokenBiasMap` to this speculative generator.
    ///
    /// The bias is cached for the generator's lifetime and applied **only** to
    /// the target model's sampling during verification (and the first-token
    /// prefill). The draft model's sampling is left untouched to preserve
    /// speculative acceptance behavior.
    pub fn with_token_bias(mut self, bias: TokenBiasMap) -> Self {
        self.token_bias = bias;
        self
    }

    /// Returns a reference to the cached target-only token-bias map.
    ///
    /// Used by tests to assert that the bias was wired in correctly and that
    /// the draft model never observes it.
    pub fn token_bias(&self) -> &TokenBiasMap {
        &self.token_bias
    }

    /// Compose the effective **target-model** sampling config from the cached
    /// `token_bias` and the caller's [`SamplingConfig`].
    ///
    /// Empty cached bias => borrowed unchanged (bit-exact baseline). Non-empty
    /// bias but caller already set `sampling.token_bias` => caller wins.
    /// Otherwise the caller's config is cloned and the cached bias is injected.
    fn compose_target_sampling<'a>(&self, sampling: &'a SamplingConfig) -> Cow<'a, SamplingConfig> {
        if self.token_bias.is_empty() || !sampling.token_bias.is_empty() {
            Cow::Borrowed(sampling)
        } else {
            let mut cloned = sampling.clone();
            cloned.token_bias = self.token_bias.clone();
            Cow::Owned(cloned)
        }
    }

    /// Returns the sampling config used by the **draft** model.
    ///
    /// **Axis B**: by design this ignores the generator's cached
    /// `token_bias`. Biasing the draft sampler would skew candidate
    /// distribution away from the draft model's trained distribution and
    /// collapse speculative acceptance rates (the target's accept/reject
    /// comparison already reflects the bias on the verification side).
    #[inline]
    fn draft_sampling<'a>(&self, sampling: &'a SamplingConfig) -> &'a SamplingConfig {
        sampling
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        for cache in &mut self.main_caches {
            *cache = KVCache::new();
        }
        for cache in &mut self.draft_caches {
            *cache = KVCache::new();
        }
        self.generated_tokens.clear();
    }

    /// Get the generated tokens
    pub fn tokens(&self) -> &[i32] {
        &self.generated_tokens
    }

    /// Generate tokens using speculative decoding
    ///
    /// # Arguments
    /// * `main_model` - The main (target) model for verification
    /// * `draft_model` - The smaller draft model for candidate generation
    /// * `prompt_tokens` - Input prompt token IDs
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `num_draft` - Number of draft tokens to generate per speculation step
    /// * `sampling` - Sampling configuration
    ///
    /// # Panics
    ///
    /// Panics if any KV cache in the main model's caches is not trimmable,
    /// since speculative decoding requires cache rewind on draft rejection.
    /// All current `KVCache` variants are trimmable; this guard future-proofs
    /// the code against non-trimmable cache types (mirrors upstream mlx-lm
    /// `can_trim_prompt_cache` validation added in PR #1109 / commit `f56d997`).
    pub fn generate<M: LanguageModel, D: LanguageModel>(
        &mut self,
        main_model: &M,
        draft_model: &D,
        prompt_tokens: &[i32],
        max_tokens: usize,
        num_draft: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        self.reset();

        // Validate that all KV cache entries support trimming before we begin.
        // Speculative decoding rewrites the cache on every rejected draft token,
        // so a non-trimmable cache type would silently corrupt the state.
        // Mirrors upstream mlx-lm `can_trim_prompt_cache` check added in
        // PR #1109 / commit f56d997. All current KVCache mode variants are
        // trimmable; this assertion fires only when a new non-trimmable type
        // is introduced (fail fast rather than silent corruption).
        assert!(
            can_trim_prompt_cache(&self.main_caches),
            "speculative decoding requires a trimmable prompt cache (main model). \
             At least one KV cache entry does not support trimming. \
             Use a standard (non-speculative) generation path or switch to a \
             trimmable cache type."
        );
        assert!(
            can_trim_prompt_cache(&self.draft_caches),
            "speculative decoding requires a trimmable prompt cache (draft model). \
             At least one KV cache entry does not support trimming."
        );

        // Guard against empty prompts: speculative decoding requires at least
        // one token so the final forward pass (which produces the first
        // generated token's logits) can run. An empty prompt would cause a
        // `[1, 0]` tensor forward with undefined behaviour.
        assert!(
            !prompt_tokens.is_empty(),
            "speculative generate requires at least one prompt token"
        );

        // Axis B: compose target-only sampling once; draft sampling stays raw.
        // `target_cow` owns the merged config when a bias is active, otherwise
        // it borrows the caller's. `draft_sampling` always returns `sampling`
        // unchanged — biasing the draft would collapse acceptance rate.
        let target_cow = self.compose_target_sampling(sampling);
        let target_sampling: &SamplingConfig = target_cow.as_ref();
        let draft_sampling: &SamplingConfig = self.draft_sampling(sampling);

        // Set generation stream
        install_thread_local_default_stream(self.generation_stream.as_ref());

        // History + EOS handling inherit the caller's policy; history-based
        // penalties apply to both models so we read flags from the caller's
        // raw config (same shape as `target_sampling` except for `token_bias`).
        let eos_tokens = merged_eos_token_ids(main_model.eos_token_ids(), &sampling.stop_token_ids);
        let needs_history = sampling.needs_token_history();
        let mut token_history = initial_token_history(prompt_tokens, needs_history);

        // PREFILL PHASE.
        //
        // Process the prompt in chunks of `PREFILL_STEP_SIZE`, always reserving
        // the last token for the first speculation step. This mirrors the
        // upstream mlx-lm fix in PR #1109 / commit f56d997:
        //
        //   while y.size > 1:
        //       n_to_process = min(prefill_step_size, y.size - 1)
        //
        // The old single-shot prefill (`forward` over all tokens at once) worked
        // for short prompts but could process every token, leaving none for the
        // speculation bootstrap — causing output corruption when prompt length
        // was an exact multiple of the step size.
        let prefill_start = Instant::now();

        // Chunked prefill: process all but the last token in step-sized blocks,
        // evaluating and clearing the memory cache between steps to bound peak
        // memory usage for long prompts.
        let n = prompt_tokens.len();
        let mut consumed = 0usize;
        while n - consumed > 1 {
            let step = PREFILL_STEP_SIZE.min(n - consumed - 1);
            let chunk = &prompt_tokens[consumed..consumed + step];
            let chunk_input = ffi::from_slice_i32(chunk, &[1, step as i32]);
            // Prefill both models with the chunk (logits discarded — we only
            // need the KV cache state for the continuation).
            let _main_chunk_logits = main_model.forward(&chunk_input, &mut self.main_caches, None);
            let _draft_chunk_logits =
                draft_model.forward(&chunk_input, &mut self.draft_caches, None);
            // Evaluate only the KV cache state so it is materialised before
            // the next chunk. This mirrors the upstream mlx-lm pattern:
            //   mx.eval([c.state for c in cache])
            // Evaluating the full logit tensors (_main_chunk_logits /
            // _draft_chunk_logits) would force the LM-head matmul and a
            // peak allocation of ~3.5 GB per chunk for Llama-3 class models,
            // defeating the memory-bounding goal of chunked prefill.
            for cache in &self.main_caches {
                cache.eval_state();
            }
            for cache in &self.draft_caches {
                cache.eval_state();
            }
            consumed += step;
            ffi::clear_memory_cache();
        }

        // Forward the final token through both models to get the logits used
        // for sampling the first generated token. By construction `consumed < n`
        // so there is always at least one remaining token here.
        let last_chunk = &prompt_tokens[consumed..];
        let last_input = ffi::from_slice_i32(last_chunk, &[1, last_chunk.len() as i32]);
        let main_logits = main_model.forward(&last_input, &mut self.main_caches, None);
        let _draft_logits = draft_model.forward(&last_input, &mut self.draft_caches, None);

        // Sample first token from main model (target: bias applied).
        let (first_token_arr, _) =
            sample_token_optimized(&main_logits, target_sampling, &token_history);
        ffi::eval(&first_token_arr);
        let first_token = ffi::item_i32(&first_token_arr);
        let prefill_time = prefill_start.elapsed();

        if eos_tokens.contains(&first_token) || max_tokens == 0 {
            let stats = Self::build_stats(
                prompt_tokens.len(),
                self.generated_tokens.len(),
                prefill_time,
                std::time::Duration::ZERO,
            );
            return (self.generated_tokens.clone(), stats);
        }

        self.generated_tokens.push(first_token);
        if needs_history {
            token_history.push(first_token);
        }

        if max_tokens <= 1 {
            let stats = Self::build_stats(
                prompt_tokens.len(),
                self.generated_tokens.len(),
                prefill_time,
                std::time::Duration::ZERO,
            );
            return (self.generated_tokens.clone(), stats);
        }

        // DECODE PHASE.
        let decode_start = Instant::now();
        let mut current_token = first_token;
        let mut done = false;

        while self.generated_tokens.len() < max_tokens && !done {
            // Step 1: Generate draft tokens
            let mut draft_tokens = Vec::with_capacity(num_draft);
            let mut draft_token = current_token;

            for _ in 0..num_draft {
                let draft_input = ffi::from_slice_i32(&[draft_token], &[1, 1]);
                let draft_logits = draft_model.forward(&draft_input, &mut self.draft_caches, None);
                // Axis B: draft sampler MUST NOT see the bias. See
                // `draft_sampling` for the rationale.
                let (tok_arr, _) =
                    sample_token_optimized(&draft_logits, draft_sampling, &token_history);
                ffi::eval(&tok_arr);
                draft_token = ffi::item_i32(&tok_arr);
                draft_tokens.push(draft_token);

                if eos_tokens.contains(&draft_token) {
                    break;
                }
            }

            if draft_tokens.is_empty() {
                break;
            }

            // Step 2: Verify draft tokens with main model in a single batched forward pass.
            // Input: [current_token, draft_token_0, ..., draft_token_n-1] shape [1, N+1]
            // Output: logits shape [1, N+1, vocab_size]
            //
            // This is structurally identical to a prefill pass, converting N memory-bound
            // GEMV decode operations into one compute-bound GEMM. On M5+ Neural Accelerator
            // hardware, this yields 3-4x speedup via tile-aligned GEMM dispatch.
            let mut verify_tokens = vec![current_token];
            verify_tokens.extend_from_slice(&draft_tokens);
            let actual_verify_len = verify_tokens.len();

            // On M5+ hardware with Neural Accelerator, pad the verification sequence
            // to a 32-token tile boundary for optimal GEMM throughput. On other
            // hardware, no padding is needed (batching is still beneficial but
            // tile alignment does not apply).
            let main_logits = if should_align_verification() && main_model.supports_padded_prefill()
            {
                let padded_len = align_to_na_tile(actual_verify_len);
                // Capture the current KV cache offset before the verification pass
                // so the attention mask correctly spans [offset, offset + padded_len).
                let kv_offset = self.main_caches.first().map(|c| c.offset).unwrap_or(0);

                if padded_len > actual_verify_len {
                    // Pad with zeros up to the tile boundary
                    let mut padded_tokens = verify_tokens.clone();
                    padded_tokens.resize(padded_len, 0);
                    let verify_input = ffi::from_slice_i32(&padded_tokens, &[1, padded_len as i32]);
                    // Create attention mask so padding positions cannot attend to
                    // anything and real tokens cannot attend to padding keys.
                    let mask = create_padded_prefill_mask(
                        actual_verify_len as i32,
                        padded_len as i32,
                        kv_offset,
                    );
                    let raw_logits = main_model.forward(
                        &verify_input,
                        &mut self.main_caches,
                        Some(mask.as_ref().unwrap()),
                    );
                    // Trim padding positions from KV caches so subsequent decode
                    // steps see the correct cache offset (actual_verify_len tokens,
                    // not padded_len tokens).
                    let excess = (padded_len - actual_verify_len) as i32;
                    for cache in self.main_caches.iter_mut() {
                        cache.trim(excess);
                    }
                    main_model.trim_internal_caches(excess);
                    // Return only the logits for the actual (non-padded) positions,
                    // sliced to shape [1, actual_verify_len, vocab].
                    let vocab = ffi::array_shape(&raw_logits)[2];
                    ffi::slice(
                        &raw_logits,
                        &[0, 0, 0],
                        &[1, actual_verify_len as i32, vocab],
                    )
                } else {
                    // Sequence already aligns to a tile boundary; no padding needed.
                    let verify_input =
                        ffi::from_slice_i32(&verify_tokens, &[1, actual_verify_len as i32]);
                    main_model.forward(&verify_input, &mut self.main_caches, None)
                }
            } else {
                // Non-NA hardware: plain batched forward pass, no tile alignment.
                let verify_input =
                    ffi::from_slice_i32(&verify_tokens, &[1, actual_verify_len as i32]);
                main_model.forward(&verify_input, &mut self.main_caches, None)
            };

            // The main model returns logits for each position:
            // - Position 0 (current_token): logits that would produce draft_tokens[0]
            // - Position i: logits that would produce draft_tokens[i]
            // - Last position: logits for the token after all draft tokens

            // Step 3: Compare draft tokens with main model's choices
            let main_shape = ffi::array_shape(&main_logits);
            let seq_len = main_shape[1]; // Number of logit positions (actual, not padded)
            let mut accepted = 0;

            for (i, draft_token) in draft_tokens.iter().copied().enumerate() {
                if (i as i32) >= seq_len {
                    break;
                }

                // Get main model's logits at position i
                let pos_logits = ffi::slice(
                    &main_logits,
                    &[0, i as i32, 0],
                    &[1, (i as i32) + 1, main_shape[2]],
                );
                // Reshape to [1, 1, vocab] for sample_token_optimized
                let pos_logits = ffi::reshape(&pos_logits, &[1, 1, main_shape[2]]);

                // Axis B: target verification uses the bias-augmented sampling.
                let (main_tok_arr, _) =
                    sample_token_optimized(&pos_logits, target_sampling, &token_history);
                ffi::eval(&main_tok_arr);
                let main_token = ffi::item_i32(&main_tok_arr);

                if main_token == draft_token {
                    // Accept draft token
                    accepted += 1;

                    if eos_tokens.contains(&draft_token) {
                        done = true;
                        break;
                    }

                    self.generated_tokens.push(draft_token);
                    if needs_history {
                        token_history.push(draft_token);
                    }

                    if self.generated_tokens.len() >= max_tokens {
                        done = true;
                        break;
                    }
                } else {
                    // Reject: use main model's token instead
                    if eos_tokens.contains(&main_token) {
                        done = true;
                    } else {
                        self.generated_tokens.push(main_token);
                        if needs_history {
                            token_history.push(main_token);
                        }
                    }
                    break;
                }
            }

            // If all draft tokens were accepted and we're not done,
            // sample one more token from the main model's last logit position
            if accepted == draft_tokens.len() && !done && self.generated_tokens.len() < max_tokens {
                let last_pos = seq_len - 1;
                let last_logits = ffi::slice(
                    &main_logits,
                    &[0, last_pos, 0],
                    &[1, last_pos + 1, main_shape[2]],
                );
                let last_logits = ffi::reshape(&last_logits, &[1, 1, main_shape[2]]);
                // Axis B: bonus token comes from the main model → target bias.
                let (bonus_tok_arr, _) =
                    sample_token_optimized(&last_logits, target_sampling, &token_history);
                ffi::eval(&bonus_tok_arr);
                let bonus_token = ffi::item_i32(&bonus_tok_arr);

                if eos_tokens.contains(&bonus_token) {
                    done = true;
                } else {
                    self.generated_tokens.push(bonus_token);
                    if needs_history {
                        token_history.push(bonus_token);
                    }
                }

                current_token = bonus_token;
            } else if !done {
                current_token = *self.generated_tokens.last().unwrap();
            }

            // Step 4: Rewind caches for rejected tokens
            let rejected = draft_tokens.len() - accepted;
            if rejected > 0 {
                // Rewind main model caches: we forwarded all verify_tokens but
                // only accepted `accepted` draft tokens + 1 (the divergence token from main)
                // Main model cache has current_token + all draft tokens = verify_tokens.len() positions
                // We need to keep: accepted + 1 (for the token we're continuing from)
                // So trim: draft_tokens.len() - accepted
                let main_trim = rejected as i32;
                trim_caches(&mut self.main_caches, main_trim);

                // Rewind draft model caches: drafted all draft_tokens
                // Need to trim all rejected + 1 (because draft went past accepted)
                let draft_trim = (rejected + 1) as i32;
                trim_caches(
                    &mut self.draft_caches,
                    draft_trim.min(draft_tokens.len() as i32),
                );
            }

            // Periodic cache clearing
            if self.generated_tokens.len().is_multiple_of(256) {
                ffi::clear_memory_cache();
            }
        }

        let decode_time = decode_start.elapsed();

        let stats = Self::build_stats(
            prompt_tokens.len(),
            self.generated_tokens.len(),
            prefill_time,
            decode_time,
        );

        (self.generated_tokens.clone(), stats)
    }

    fn build_stats(
        prompt_count: usize,
        gen_count: usize,
        prefill_time: std::time::Duration,
        decode_time: std::time::Duration,
    ) -> GenerationStats {
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;

        GenerationStats {
            prompt_tokens: prompt_count,
            generated_tokens: gen_count,
            prefill_time_ms: prefill_ms,
            decode_time_ms: decode_ms,
            prefill_tok_per_sec: if prefill_ms > 0.0 {
                prompt_count as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            },
            decode_tok_per_sec: if decode_ms > 0.0 {
                gen_count as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            },
        }
    }
}

/// Trim the last `n` entries from all caches in the slice
/// Returns the number of entries actually trimmed (from the first cache)
fn trim_caches(caches: &mut [KVCache], n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut trimmed = 0;
    for cache in caches.iter_mut() {
        trimmed = cache.trim(n);
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;

    #[test]
    fn test_kv_cache_trim_basic() {
        let mut cache = KVCache::new();

        // Add some data: [batch=1, heads=2, seq_len=5, head_dim=4]
        let keys = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
        cache.update(keys, values);
        assert_eq!(cache.offset, 5);

        // Trim 2
        let trimmed = cache.trim(2);
        assert_eq!(trimmed, 2);
        assert_eq!(cache.offset, 3);

        // Verify shapes
        let k_shape = ffi::array_shape(cache.keys.as_ref().unwrap());
        assert_eq!(k_shape, vec![1, 2, 3, 4]);
        let v_shape = ffi::array_shape(cache.values.as_ref().unwrap());
        assert_eq!(v_shape, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_kv_cache_trim_all() {
        let mut cache = KVCache::new();
        let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        cache.update(keys, values);

        // Trim all
        let trimmed = cache.trim(3);
        assert_eq!(trimmed, 3);
        assert_eq!(cache.offset, 0);
        assert!(cache.keys.is_none());
        assert!(cache.values.is_none());
    }

    #[test]
    fn test_kv_cache_trim_zero() {
        let mut cache = KVCache::new();
        let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        cache.update(keys, values);

        let trimmed = cache.trim(0);
        assert_eq!(trimmed, 0);
        assert_eq!(cache.offset, 3);
    }

    #[test]
    fn test_kv_cache_trim_more_than_available() {
        let mut cache = KVCache::new();
        let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        cache.update(keys, values);

        // Trim more than available - should trim only what's available
        let trimmed = cache.trim(10);
        assert_eq!(trimmed, 3);
        assert_eq!(cache.offset, 0);
        assert!(cache.keys.is_none());
    }

    #[test]
    fn test_trim_caches_helper() {
        let mut caches = vec![KVCache::new(), KVCache::new()];
        for cache in caches.iter_mut() {
            let keys = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
            let values = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
            cache.update(keys, values);
        }

        let trimmed = trim_caches(&mut caches, 2);
        assert_eq!(trimmed, 2);
        for cache in &caches {
            assert_eq!(cache.offset, 3);
        }
    }

    // ------------------------------------------------------------------
    // B8 — token-bias wiring (target-only)
    // ------------------------------------------------------------------

    fn make_bias(entries: &[(i32, f32)]) -> TokenBiasMap {
        let mut m = TokenBiasMap::new();
        for &(id, b) in entries {
            m.insert(id, b);
        }
        m
    }

    /// Default construction yields an empty token-bias cache.
    #[test]
    fn speculative_generator_default_bias_is_empty() {
        let g = SpeculativeGenerator::new(4, 2);
        assert!(g.token_bias().is_empty());
    }

    /// `with_token_bias` caches the supplied map and exposes it via the
    /// inspector — the target path sees this map, the draft path never does.
    #[test]
    fn speculative_generator_passes_bias_to_target_only() {
        let bias = make_bias(&[(7, f32::NEG_INFINITY), (11, 2.0)]);
        let g = SpeculativeGenerator::new(4, 2).with_token_bias(bias.clone());

        // Target-side composition must inject the cached bias into a caller
        // config that lacks one.
        let caller = SamplingConfig::default();
        let target = g.compose_target_sampling(&caller);
        assert_eq!(
            target.token_bias.len(),
            2,
            "target sampler must carry the cached bias"
        );
        assert!(
            target.token_bias.contains(7),
            "target bias must contain id=7"
        );

        // Draft-side composition MUST remain unbiased regardless of the cached
        // map — this is the core speculative-acceptance invariant.
        let draft = g.draft_sampling(&caller);
        assert!(
            draft.token_bias.is_empty(),
            "draft sampler must NEVER carry the cached bias (got {} entries): \
             speculative acceptance is computed by comparing draft candidates \
             against target sampling, and biasing the draft collapses the \
             accept ratio",
            draft.token_bias.len()
        );
    }

    /// Caller-supplied bias wins over the generator-cached bias (explicit
    /// per-call override).
    #[test]
    fn speculative_generator_caller_bias_wins() {
        let cached = make_bias(&[(1, 1.0)]);
        let caller_bias = make_bias(&[(42, f32::NEG_INFINITY)]);
        let g = SpeculativeGenerator::new(2, 1).with_token_bias(cached);

        let mut caller = SamplingConfig::default();
        caller.token_bias = caller_bias;
        let target = g.compose_target_sampling(&caller);

        assert_eq!(
            target.token_bias.len(),
            1,
            "caller's explicit token_bias wins"
        );
        assert!(target.token_bias.contains(42));
    }

    /// Empty cached bias + empty caller bias yields the caller config
    /// unchanged (bit-exact baseline — `Cow::Borrowed`).
    #[test]
    fn speculative_generator_empty_bias_is_bit_exact() {
        let g = SpeculativeGenerator::new(2, 1);
        let caller = SamplingConfig::default();
        let target = g.compose_target_sampling(&caller);
        assert!(matches!(target, Cow::Borrowed(_)));
        assert!(target.token_bias.is_empty());
    }

    // ------------------------------------------------------------------
    // Issue #589 — trimmable cache validation and last-token reservation
    // ------------------------------------------------------------------

    /// All freshly-constructed KVCache entries must report `is_trimmable()`.
    /// This is the per-entry predicate consumed by `can_trim_prompt_cache`.
    #[test]
    fn kv_cache_is_trimmable_always_true() {
        // Empty cache
        let c = KVCache::new();
        assert!(c.is_trimmable());

        // Cache with accumulated state
        let mut c = KVCache::new();
        let k = ffi::ones(&[1, 2, 4, 4], dtype::FLOAT32);
        let v = ffi::ones(&[1, 2, 4, 4], dtype::FLOAT32);
        c.update(k, v);
        assert!(c.is_trimmable());
    }

    /// `can_trim_prompt_cache` returns `true` for a slice of standard KVCaches.
    #[test]
    fn can_trim_prompt_cache_all_standard() {
        use crate::cache::can_trim_prompt_cache;

        let caches: Vec<KVCache> = (0..4).map(|_| KVCache::new()).collect();
        assert!(can_trim_prompt_cache(&caches));
    }

    /// `can_trim_prompt_cache` returns `true` even for an empty slice
    /// (vacuously: all members of the empty set satisfy the predicate).
    #[test]
    fn can_trim_prompt_cache_empty_slice() {
        use crate::cache::can_trim_prompt_cache;

        let caches: Vec<KVCache> = Vec::new();
        assert!(can_trim_prompt_cache(&caches));
    }

    /// Verify that `PREFILL_STEP_SIZE` matches the upstream mlx-lm default.
    /// If this constant is changed, the test must be updated deliberately so
    /// reviewers are aware of the deviation from upstream behavior.
    #[test]
    fn prefill_step_size_matches_upstream_default() {
        assert_eq!(
            PREFILL_STEP_SIZE, 512,
            "PREFILL_STEP_SIZE must match upstream mlx-lm default (512). \
             Update this test if you intentionally deviate."
        );
    }

    struct FixedLogitModel {
        preferred_token: usize,
        eos_tokens: Vec<i32>,
    }

    impl LanguageModel for FixedLogitModel {
        fn forward(
            &self,
            input_ids: &crate::ffi::MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&crate::ffi::MlxArray>,
        ) -> UniquePtr<crate::ffi::MlxArray> {
            let shape = ffi::array_shape(input_ids);
            let batch = shape[0] as usize;
            let seq_len = shape[1] as usize;
            let vocab = 4usize;
            let mut logits = vec![-10.0f32; batch * seq_len * vocab];
            for b in 0..batch {
                for s in 0..seq_len {
                    logits[(b * seq_len + s) * vocab + self.preferred_token] = 10.0;
                }
            }
            ffi::from_slice_f32(&logits, &[shape[0], shape[1], vocab as i32])
        }

        fn make_caches(&self) -> Vec<KVCache> {
            vec![KVCache::new()]
        }

        fn num_layers(&self) -> usize {
            1
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            self.eos_tokens.clone()
        }
    }

    #[test]
    fn speculative_generate_max_tokens_one_emits_first_non_eos_token() {
        let main = FixedLogitModel {
            preferred_token: 2,
            eos_tokens: vec![3],
        };
        let draft = FixedLogitModel {
            preferred_token: 2,
            eos_tokens: vec![3],
        };
        let mut generator = SpeculativeGenerator::new(main.num_layers(), draft.num_layers());

        let (tokens, stats) =
            generator.generate(&main, &draft, &[42], 1, 1, &SamplingConfig::greedy());

        assert_eq!(
            tokens,
            vec![2],
            "max_tokens=1 must still return the first sampled non-EOS token"
        );
        assert_eq!(stats.generated_tokens, 1);
    }
}
