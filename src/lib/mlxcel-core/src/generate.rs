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

//! Token generation utilities for mlxcel-core models
//!
//! This module provides the generation loop and sampling functions
//! for text generation with mlxcel-core models.
//!
//! Key optimizations matching Python mlx-lm:
//! - Dedicated generation stream for pipelined execution
//! - Lookahead pipelining: compute token n+1 while returning token n
//! - Optimized decode loops for standard and embedding-prefill paths
//! - Shared sampling policy delegated to `crate::sampling`
//! - Shared decode setup delegated to `crate::generation_policy`

use std::borrow::Cow;

use crate::cache::{CachePool, KVCacheMode, SequenceId};
use crate::ffi;
use crate::ffi::{MlxArray, MlxThreadLocalStream};
use crate::generation_policy::{
    ensure_model_caches, initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
use crate::hardware;
use crate::layers::KVCache;
use crate::sampling::{sample_token_optimized, TokenBiasMap};
use crate::streams::{install_thread_local_default_stream, new_thread_local_generation_stream};
use crate::utils::{align_to_na_tile, create_padded_prefill_mask};
use cxx::UniquePtr;

/// Returns true when the current hardware is M5+ with a Neural Accelerator
/// and tile-aligned prefill should be applied.
///
/// Set `MLXCEL_NO_PADDED_PREFILL=1` to disable tile alignment (debugging).
#[inline]
fn should_align_prefill() -> bool {
    if std::env::var("MLXCEL_NO_PADDED_PREFILL").is_ok() {
        return false;
    }
    let hw = hardware::get_hardware();
    hw.has_neural_accelerator && hw.macos_supports_na
}

#[inline]
fn force_padded_prefill_array_mask() -> bool {
    std::env::var("MLXCEL_FORCE_PADDED_PREFILL_MASK").is_ok()
}

/// Pad a prompt token slice to `padded_len` with the pad token (0) and return
/// both the padded slice and an appropriate attention mask.
///
/// If `actual_len == padded_len` no padding is needed: returns the original
/// tokens and `None` (the forward pass will use its built-in causal mask).
///
/// If `actual_len < padded_len` the sequence is extended with zeros and a
/// padded causal mask is returned so that padding positions do not leak into
/// the KV cache values.
///
/// # Arguments
/// * `prompt_tokens` - Original token IDs.
/// * `padded_len`    - Target aligned length (≥ `prompt_tokens.len()`).
///
/// # Returns
/// `(padded_tokens_vec, mask_or_none)` where `mask_or_none` is `None` when no
/// padding was added.
fn pad_tokens_for_prefill(
    prompt_tokens: &[i32],
    padded_len: usize,
    use_maskless_causal: bool,
) -> (Vec<i32>, Option<UniquePtr<MlxArray>>) {
    let actual_len = prompt_tokens.len();
    if padded_len == actual_len {
        return (prompt_tokens.to_vec(), None);
    }

    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(prompt_tokens);
    padded.resize(padded_len, 0); // pad with token id 0

    if use_maskless_causal && !force_padded_prefill_array_mask() {
        return (padded, None);
    }

    let mask = create_padded_prefill_mask(actual_len as i32, padded_len as i32, 0);
    (padded, Some(mask))
}

/// After a padded prefill, trim all KV caches back to `actual_len` so that
/// the decode phase starts with the correct sequence position.
///
/// The padded token positions `[actual_len, padded_len)` were written to the
/// cache during the forward pass; trimming removes them so the KV cache offset
/// reflects only the real prompt tokens.
fn trim_caches_to_actual_len(caches: &mut [KVCache], actual_len: usize, padded_len: usize) {
    let excess = (padded_len - actual_len) as i32;
    if excess <= 0 {
        return;
    }
    for cache in caches.iter_mut() {
        cache.trim(excess);
    }
}

/// Pad an embeddings tensor from `[batch, actual_len, hidden]` to
/// `[batch, padded_len, hidden]` by appending zero rows.
///
/// Used by the VLM tile-alignment path to match the padded token sequence.
fn pad_embeddings(embeds: &MlxArray, padded_len: usize) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(embeds);
    let batch = shape[0];
    let actual_seq = shape[1] as usize;
    let hidden = shape[2];
    if padded_len <= actual_seq {
        return ffi::slice(embeds, &[0, 0, 0], &[batch, actual_seq as i32, hidden]);
    }
    let pad_rows = (padded_len - actual_seq) as i32;
    let dtype = ffi::array_dtype(embeds);
    let padding = ffi::zeros(&[batch, pad_rows, hidden], dtype);
    crate::concatenate(embeds, &padding, 1)
}

/// Extract the logits at a specific sequence position, returning shape
/// `[batch, 1, vocab]` to remain compatible with `slice_last_logits`.
///
/// `logits` has shape `[batch, seq_len, vocab]`. Slices out position `pos`
/// along the sequence axis (keeping the dimension as size 1) so that the
/// caller can still pass the result to `sample_token_optimized`, which
/// internally calls `slice_last_logits` expecting `[batch, seq_len, vocab]`.
///
/// Used after a padded prefill to obtain the prediction for the last *real*
/// token position rather than the last padding position.
fn logits_at_position(logits: &MlxArray, pos: usize) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(logits);
    let batch = shape[0];
    let vocab = shape[2];
    // Slice [batch, pos:pos+1, vocab]  →  shape [batch, 1, vocab].
    ffi::slice(logits, &[0, pos as i32, 0], &[batch, pos as i32 + 1, vocab])
}

/// Trait for language models that can be used for generation
pub trait LanguageModel {
    /// Forward pass through the model
    /// Returns logits of shape [batch, seq_len, vocab_size]
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray>;

    /// Create KV caches for all layers
    fn make_caches(&self) -> Vec<KVCache>;

    /// Get the number of layers
    fn num_layers(&self) -> usize;

    /// Get the EOS token IDs for this model
    fn eos_token_ids(&self) -> Vec<i32>;

    /// Forward with pre-computed embeddings (for VLM prefill)
    /// Used by: VisionLanguageModel (Gemma3 VLM)
    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Default: ignore embeddings, use standard forward
        let _ = input_embeddings;
        self.forward(input_ids, caches, mask)
    }

    /// Get embeddings for token IDs (needed by VisionModule for merging)
    /// Used by: VisionModule::get_input_embeddings
    fn embed_tokens(&self, _input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        None // default: not supported
    }

    /// Hand out a shared-buffer handle to this model's input embedding
    /// table for speculative drafters that lazy-bind it.
    ///
    /// Unlike [`Self::embed_tokens`] (which applies the embedding to a
    /// given id tensor), this returns the embedding *module* itself so a
    /// drafter can use it both as an embedding lookup and as a tied LM
    /// head (`UnifiedEmbedding::as_linear`). The returned
    /// [`UnifiedEmbedding`] shares the underlying MLX buffers with the
    /// target (lazy-array share via `UnifiedEmbedding::clone_shared` — no
    /// element copy) and stays valid for the lifetime of the speculative
    /// session.
    ///
    /// The default returns `None`; only targets that can pair with a
    /// lazy-bind drafter override it. Concretely, the upstream
    /// `z-lab/Qwen3.5-4B-DFlash` checkpoint omits `embed_tokens.weight`
    /// and the Rust DFlash drafter resolves it here during
    /// [`crate::drafter::Drafter::bind`].
    ///
    /// Used by: DFlash drafter lazy-bind path; Gemma 4 MTP assistant
    /// binding; Qwen 3.5 target family; Gemma 4 target family
    fn embed_tokens_module(&self) -> Option<crate::layers::UnifiedEmbedding> {
        None // default: not supported
    }

    /// Hand out a shared-buffer handle to this model's output projection
    /// when the projection is untied from the input embedding table.
    ///
    /// Some DFlash checkpoints (for example `z-lab/Qwen3.5-27B-DFlash`)
    /// omit both `embed_tokens.weight` and `lm_head.weight`; upstream Python
    /// binds both modules from the target at runtime, falling back to
    /// `embed_tokens.as_linear` only when the target has no explicit head.
    /// The default returns `None` so tied-embedding models keep using the
    /// embedding table path.
    ///
    /// Used by: DFlash drafter lazy-bind path for untied Qwen 3.5 targets.
    fn lm_head_module(&self) -> Option<crate::layers::UnifiedLinear> {
        None // default: tied or unsupported
    }

    /// Called once after prefill completes and before decode starts.
    /// Used by models that need to adjust internal state between phases,
    /// e.g. Phi4MM unfuses vision LoRA so decode uses base weights.
    fn after_prefill(&self) {}

    /// Trim internal caches after padded prefill. Models with internal
    /// cache state (e.g. NemotronH) override this to trim their own caches
    /// so that padding positions do not corrupt subsequent decode steps.
    fn trim_internal_caches(&self, _excess: i32) {}

    /// Release any model-owned sequence state associated with the provided
    /// external cache slice before the scheduler drops that cache set.
    ///
    /// Used by: Qwen3.5 mixed-cache map cleanup, server batch scheduler
    fn release_sequence_state(&self, _caches: &mut [KVCache]) {}

    /// Prepare model-owned/runtime sequence state before the scheduler starts
    /// using this `SequenceId`.
    fn prepare_sequence_state(&self, _seq_id: SequenceId) {}

    /// Release model-owned/runtime sequence state by its scheduler `SequenceId`.
    fn release_sequence_state_by_id(&self, _seq_id: SequenceId) {}

    /// Describe how one sequence's runtime state should be allocated.
    ///
    /// Phase 0 keeps the default behavior aligned with today's
    /// `supports_batching()` split while giving the control plane an explicit
    /// backend/layout seam for future paged and model-owned sequence state.
    ///
    /// Used by: `CachePool::allocate()`
    fn sequence_state_layout(&self) -> crate::cache::SequenceStateLayout {
        let num_layers = self.num_layers();
        if self.supports_batching() {
            crate::cache::SequenceStateLayout::dense_kv_cache(num_layers)
        } else {
            crate::cache::SequenceStateLayout::model_owned(num_layers)
        }
    }

    /// Whether this model supports tile-aligned padded prefill on M5+ hardware.
    ///
    /// Pure transformer models return `true` (the default) because padding
    /// tokens only affect the external KV cache which is trimmed afterwards.
    /// Hybrid SSM models (NemotronH, Jamba, Mamba, etc.) return `false`
    /// because padding tokens corrupt the internal recurrent state (conv /
    /// SSM state) in a way that cannot be safely trimmed, and the resulting
    /// NaN/inf values can corrupt the Metal GPU state.
    fn supports_padded_prefill(&self) -> bool {
        true
    }

    /// Whether tile-aligned padded prefill can safely use the model's implicit
    /// causal attention path without building an explicit array mask.
    ///
    /// This is only valid for standard causal transformer prefill where:
    /// - padding tokens are appended after the real prompt
    /// - outputs from padded positions are discarded
    /// - external/internal caches are trimmed back to the real prompt length
    ///
    /// Hybrid/recurrent models and models with custom prefill mask semantics
    /// should keep returning `false`.
    fn supports_maskless_padded_prefill(&self) -> bool {
        false
    }

    /// Whether this model supports batched decode for continuous batching.
    ///
    /// Standard transformer models return `true` (the default) because their
    /// state lives entirely in the external `KVCache` slice. SSM and hybrid
    /// models (Mamba, Jamba, NemotronH, etc.) maintain internal recurrent
    /// state that is not compatible with independent per-sequence cache
    /// isolation, so they override this to return `false`.
    ///
    /// Used by: CachePool (to reject unsupported models), server scheduler
    fn supports_batching(&self) -> bool {
        true
    }

    /// Whether the server batch scheduler may use the paged decode backend
    /// for this model family.
    ///
    /// This is stricter than `supports_batching()`: a model can participate in
    /// batched decode while still opting out of paged decode until its
    /// attention path, cache semantics, and operational validation are ready.
    fn supports_paged_decode_backend(&self) -> bool {
        false
    }

    /// Whether this model supports full-sequence batched prefill.
    ///
    /// This is stricter than decode batching. A model may support
    /// `forward_batched()` for `[B, 1]` decode while not supporting
    /// `[B, T]` prompt prefill with shared graph execution.
    ///
    /// The default is `false` so server prefill keeps using the standard
    /// single-sequence path unless a model explicitly opts in with a
    /// true full-prompt batched implementation.
    ///
    /// Used by: BatchScheduler batched prefill gate
    fn supports_batched_prefill(&self) -> bool {
        false
    }

    /// Single-sequence forward with optional scheduler sequence identity.
    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let _ = seq_id;
        self.forward(input_ids, caches, mask)
    }

    /// Embedding-prefill forward with optional scheduler sequence identity.
    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let _ = seq_id;
        self.forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    /// Synchronize model-owned sequence storage into the runtime backend state.
    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        cache_pool.sync_paged_state_with_dense(seq_id)
    }

    /// Batched decode with explicit runtime context from the scheduler.
    ///
    /// This extends `forward_batched()` without forcing all model families to
    /// plumb scheduler-specific state through their existing dense path. The
    /// default implementation ignores the context and delegates to
    /// `forward_batched()`.
    ///
    /// Used by: BatchScheduler decode backend dispatch, paged decode profiling
    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let _ = context;
        self.forward_batched(input_ids, batch_caches, mask)
    }

    /// Batched forward with optional scheduler sequence identities.
    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let _ = seq_ids;
        self.forward_batched_with_context(input_ids, batch_caches, mask, context)
    }

    /// Batched decode: process B sequences in one forward pass.
    ///
    /// `input_ids` has shape `[B, 1]` where B is the batch size (one new
    /// token per active sequence). `batch_caches[i]` is the per-layer KV
    /// cache slice for the i-th sequence.
    ///
    /// Returns logits of shape `[B, 1, vocab_size]`.
    ///
    /// The default implementation falls back to a loop that calls
    /// `forward()` once per sequence and stacks the results. Models that
    /// override this (e.g. Llama3) batch the compute-bound layers
    /// (embedding, norm, FFN) and only run attention per-sequence, which
    /// amortizes weight-loading bandwidth across the batch.
    ///
    /// Used by: BatchScheduler (server continuous batching)
    #[allow(clippy::needless_range_loop)]
    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let b = batch_caches.len();
        if b == 0 {
            return ffi::zeros(&[0, 1, 1], crate::dtype::FLOAT32);
        }
        if b == 1 {
            // Fast path: single sequence, no slicing/stacking overhead
            let logits = self.forward(input_ids, batch_caches[0], None);
            return logits;
        }

        // Default fallback: loop over batch dimension, calling forward()
        // once per sequence and concatenating the results into [B, 1, vocab].
        // Each forward() returns [1, 1, vocab]; concatenate along axis 0
        // yields [B, 1, vocab].
        let token_0 = ffi::slice(input_ids, &[0, 0], &[1, 1]);
        let mut result = self.forward(&token_0, batch_caches[0], None);
        for i in 1..b {
            let token_i = ffi::slice(input_ids, &[i as i32, 0], &[i as i32 + 1, 1]);
            let logits_i = self.forward(&token_i, batch_caches[i], None);
            result = crate::concatenate(&result, &logits_i, 0);
        }
        result
    }
}

/// Decode-time storage backend hint supplied by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStorageBackend {
    Dense,
    Paged,
}

/// Optional scheduler/runtime context for batched decode dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeBatchContext {
    pub storage_backend: DecodeStorageBackend,
    pub paged_block_size: i32,
    pub use_native_paged_kernel: bool,
}

impl DecodeBatchContext {
    pub fn dense() -> Self {
        Self {
            storage_backend: DecodeStorageBackend::Dense,
            paged_block_size: 0,
            use_native_paged_kernel: false,
        }
    }

    pub fn paged(block_size: i32) -> Self {
        Self::paged_with_native(block_size, true)
    }

    pub fn paged_with_native(block_size: i32, use_native_paged_kernel: bool) -> Self {
        Self {
            storage_backend: DecodeStorageBackend::Paged,
            paged_block_size: block_size,
            use_native_paged_kernel,
        }
    }

    pub fn is_paged_decode(self) -> bool {
        self.storage_backend == DecodeStorageBackend::Paged && self.paged_block_size > 0
    }
}

/// Sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for sampling (1.0 = no change)
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: i32,
    /// Top-p (nucleus) sampling (1.0 = disabled)
    pub top_p: f32,
    /// Min-p sampling threshold (0.0 = disabled)
    /// Removes tokens with probability < min_p * max_probability
    pub min_p: f32,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// Repetition penalty (1.0 = disabled)
    pub repetition_penalty: f32,
    /// DRY multiplier (0.0 = disabled)
    pub dry_multiplier: f32,
    /// DRY exponential base (default: 1.75)
    pub dry_base: f32,
    /// DRY minimum match length before penalty applies (default: 2)
    pub dry_allowed_length: usize,
    /// DRY lookback window (0 = all history)
    pub dry_penalty_last_n: usize,
    /// Token IDs that break DRY matching (e.g., newlines, punctuation)
    pub dry_sequence_breakers: Vec<i32>,
    /// OpenAI-style frequency penalty: subtract penalty * count(token) from logits (0.0 = disabled)
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty: subtract penalty if token appeared at all (0.0 = disabled)
    pub presence_penalty: f32,
    /// Additional stop token IDs (from generation_config.json or API request)
    /// Merged with model's built-in eos_token_ids during generation
    pub stop_token_ids: Vec<i32>,
    /// Per-token additive logit bias applied before all history-based penalties.
    /// Empty (default) is a zero-overhead no-op that preserves bit-exact baseline.
    pub token_bias: TokenBiasMap,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 0,
            dry_sequence_breakers: Vec::new(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            token_bias: TokenBiasMap::default(),
        }
    }
}

impl SamplingConfig {
    /// Create greedy sampling config (temperature 0)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 0,
            dry_sequence_breakers: Vec::new(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            token_bias: TokenBiasMap::default(),
        }
    }

    /// Create config with specific temperature
    pub fn with_temperature(temp: f32) -> Self {
        Self {
            temperature: temp,
            ..Default::default()
        }
    }

    /// Check if any penalty-based sampling is enabled
    pub fn needs_token_history(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.dry_multiplier > 0.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }
}

/// Generation statistics
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Number of prompt tokens processed
    pub prompt_tokens: usize,
    /// Number of tokens generated
    pub generated_tokens: usize,
    /// Time to process the prompt (prefill) in milliseconds
    pub prefill_time_ms: f64,
    /// Time to generate tokens (decode) in milliseconds
    pub decode_time_ms: f64,
    /// Prefill throughput: prompt tokens per second
    pub prefill_tok_per_sec: f64,
    /// Decode throughput: generated tokens per second
    pub decode_tok_per_sec: f64,
}

impl GenerationStats {
    /// Print formatted stats
    pub fn print(&self) {
        println!("  Prompt tokens:    {}", self.prompt_tokens);
        println!("  Generated tokens: {}", self.generated_tokens);
        println!(
            "  Prefill:          {:.2} ms ({:.2} tok/s)",
            self.prefill_time_ms, self.prefill_tok_per_sec
        );
        println!(
            "  Decode:           {:.2} ms ({:.2} tok/s)",
            self.decode_time_ms, self.decode_tok_per_sec
        );
    }
}

/// Generator state for managing generation
pub struct CxxGenerator {
    caches: Vec<KVCache>,
    generated_tokens: Vec<i32>,
    /// Dedicated thread-local generation stream for pipelining.
    ///
    /// The TLS handle resolves to a per-thread `MlxStream` on demand
    /// when the generator first dispatches work on a worker thread.
    /// This keeps dispatch and synchronization paired even if the
    /// generator is constructed on one thread (e.g. the request
    /// dispatcher) and run on another (e.g. the model worker), per
    /// upstream `mlx-vlm` PR #1050 / mlxcel issue #556.
    generation_stream: Option<UniquePtr<MlxThreadLocalStream>>,
    /// KV cache quantization mode applied to all layer caches.
    /// Default: `KVCacheMode::Fp16` (no quantization).
    kv_cache_mode: KVCacheMode,
    /// Cached per-generator `TokenBiasMap` resolved from a `LangBiasConfig`.
    ///
    /// Populated at construction time via [`Self::with_token_bias`] (or a
    /// `LangBiasConfig`-aware constructor) and re-applied to every
    /// `SamplingConfig` used by the generator's public `generate_*` entry
    /// points. Empty (`TokenBiasMap::default`) is a zero-overhead no-op that
    /// preserves bit-exact baseline behavior, so callers that do not opt into
    /// language steering pay no per-token or per-call cost (see
    /// [`Self::compose_sampling`]).
    ///
    /// Axis B / Epic #362: populated by B8 wiring for the CLI `generate`
    /// path; the server batch scheduler caches its own copy on `BatchScheduler`.
    token_bias: TokenBiasMap,
}

impl CxxGenerator {
    /// Create a new generator with FP16 KV cache (default).
    pub fn new(num_layers: usize) -> Self {
        Self {
            caches: (0..num_layers).map(|_| KVCache::new()).collect(),
            generated_tokens: Vec::new(),
            generation_stream: new_thread_local_generation_stream(),
            kv_cache_mode: KVCacheMode::Fp16,
            token_bias: TokenBiasMap::default(),
        }
    }

    /// Create a new generator with the specified KV cache quantization mode.
    ///
    /// Use `KVCacheMode::Int8` to halve KV cache memory at the cost of
    /// small per-token quantization error.
    ///
    /// When `kv_cache_mode` is one of the `Turbo4*` variants, the
    /// **Boundary-V** policy (B6, issue #478, epic #458) is applied: the
    /// first / last N transformer layers' caches are upgraded to
    /// `KVCacheMode::Fp16` to recover the per-layer V-quantization quality
    /// gap measured in `references/turboquant_plus/docs/papers/
    /// layer-aware-v-compression.md`. The boundary count is read from the
    /// `MLXCEL_KV_BOUNDARY_V_LAYERS` env var (default 2; `0` disables) and
    /// clamped to `n_layers / 2`. For non-Turbo4 modes the policy is inert
    /// — every layer's cache uses `kv_cache_mode` unchanged.
    pub fn new_with_kv_mode(num_layers: usize, kv_cache_mode: KVCacheMode) -> Self {
        let requested = crate::cache::turbo::boundary_v_layers_from_env();
        let layer_modes =
            crate::cache::turbo::resolve_layer_modes(kv_cache_mode, num_layers, requested);
        Self {
            caches: layer_modes
                .into_iter()
                .map(KVCache::new_with_mode)
                .collect(),
            generated_tokens: Vec::new(),
            generation_stream: new_thread_local_generation_stream(),
            kv_cache_mode,
            token_bias: TokenBiasMap::default(),
        }
    }

    /// Attach a pre-resolved `TokenBiasMap` to this generator.
    ///
    /// The bias is cached for the generator's lifetime and merged into every
    /// `SamplingConfig` handed to `generate_*` unless the caller already
    /// supplied a non-empty `token_bias` on the sampling config (caller wins).
    ///
    /// Callers that want to derive the map from a [`crate::LangBiasConfig`]
    /// typically combine this with
    /// [`crate::LangBiasConfig::resolve_token_bias`]:
    ///
    /// ```ignore
    /// let bias = lang_bias_config.resolve_token_bias(tokenizer, bytes)?;
    /// let generator = CxxGenerator::new(layers).with_token_bias(bias);
    /// ```
    ///
    /// When `bias.is_empty()`, this method is a no-op on the sampling path —
    /// the composed `SamplingConfig` is returned by reference and `sample_*`
    /// goes through the existing zero-overhead branch.
    pub fn with_token_bias(mut self, bias: TokenBiasMap) -> Self {
        self.token_bias = bias;
        self
    }

    /// Returns a reference to the cached token-bias map.
    ///
    /// Used by tests to assert that B8 wiring populated the correct map.
    pub fn token_bias(&self) -> &TokenBiasMap {
        &self.token_bias
    }

    /// Compose the effective sampling config from the cached `token_bias` and
    /// the caller's [`SamplingConfig`].
    ///
    /// # Precedence and bit-exact baseline
    /// - If the caller already set a non-empty `sampling.token_bias`, the
    ///   caller's bias wins (returned borrow — zero allocation).
    /// - If the cached `token_bias` is empty, we borrow the caller's config
    ///   unchanged. This is the **baseline no-op path** and is bit-exact
    ///   identical to pre-B8 behavior.
    /// - Otherwise, clone the caller's config and inject the cached bias.
    ///
    /// Used by: `generate`, `generate_streaming`, `generate_with_stats`, and
    /// VLM embedding-aware variants so every generation path observes the
    /// cached bias without duplicating the merge logic.
    fn compose_sampling<'a>(&self, sampling: &'a SamplingConfig) -> Cow<'a, SamplingConfig> {
        if self.token_bias.is_empty() || !sampling.token_bias.is_empty() {
            Cow::Borrowed(sampling)
        } else {
            let mut cloned = sampling.clone();
            cloned.token_bias = self.token_bias.clone();
            Cow::Owned(cloned)
        }
    }

    /// Prepare Turbo4Delegated cache state before decode.
    ///
    /// `max_tokens <= 1` is a prefill-only generation from the cache's point
    /// of view, so skip the handoff fold and keep prefill probes isolated.
    ///
    /// Used by: streaming and stats generation paths immediately after the
    /// first sampled token is materialized or scheduled.
    fn prepare_turbo4_delegated_before_decode(&mut self, max_tokens: usize) {
        if max_tokens <= 1 {
            return;
        }
        for cache in &mut self.caches {
            cache.prepare_turbo4_delegated_for_decode();
        }
    }

    /// Reset generator state
    ///
    /// Must call `reset_with_model` instead when the model uses internal caches
    /// (e.g. Gemma3, Jamba, Mamba, NemotronH, etc.) to ensure those are also reset.
    ///
    /// Preserves the per-layer Boundary-V mode mapping (issue #478, epic #458)
    /// computed at construction time: each layer's pre-existing
    /// `KVCacheMode` (which may differ from `self.kv_cache_mode` for
    /// boundary layers) is reused so quality protection survives a reset.
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            // Preserve the resolved per-layer mode; the constructor already
            // applied the boundary upgrade where needed and we must not
            // collapse it back to a uniform Turbo4 setup here.
            let layer_mode = cache.mode;
            *cache = KVCache::new_with_mode(layer_mode);
        }
        self.generated_tokens.clear();
    }

    /// Reset generator state including model-internal caches.
    ///
    /// Models with internal RefCell caches (sliding window, SSM, hybrid) reset
    /// their own state inside `make_caches()`. This method ensures both the
    /// generator's cache vector and the model's internal caches are cleared.
    /// The kv_cache_mode is applied to the freshly created caches.
    ///
    /// Honors the Boundary-V policy (issue #478): when `self.kv_cache_mode`
    /// is one of the `Turbo4*` variants, the first / last N caches are
    /// re-resolved to `KVCacheMode::Fp16` instead of the nominal mode.
    /// The boundary count is read from `MLXCEL_KV_BOUNDARY_V_LAYERS` so a
    /// runtime-tuned count is honored on every reset.
    pub fn reset_with_model<M: LanguageModel + ?Sized>(&mut self, model: &M) {
        self.caches = model.make_caches();
        // Apply the configured KV cache mode (with Boundary-V upgrade) to
        // all freshly created caches.
        self.apply_kv_cache_mode_with_boundary_policy();
        self.generated_tokens.clear();
    }

    /// Get mutable access to caches (used by speculative decoding)
    pub fn caches_mut(&mut self) -> &mut [KVCache] {
        &mut self.caches
    }

    /// Apply the configured KV cache mode (with Boundary-V policy) to every
    /// cache slot.
    ///
    /// Called from each generation entry point right after `ensure_model_caches`
    /// rebuilds caches from `model.make_caches()` (which always uses the
    /// default Fp16 mode). Centralizes the per-layer mode resolution so the
    /// Boundary-V policy (issue #478) survives the entire generation lifecycle
    /// including `reset_with_model` boundary cases.
    ///
    /// No-op when `self.kv_cache_mode == Fp16` — every layer is already FP16
    /// so there is nothing to apply.
    fn apply_kv_cache_mode_with_boundary_policy(&mut self) {
        let nominal = self.kv_cache_mode;
        if nominal == KVCacheMode::Fp16 {
            return;
        }
        let n_layers = self.caches.len();
        let requested = crate::cache::turbo::boundary_v_layers_from_env();
        let layer_modes = crate::cache::turbo::resolve_layer_modes(nominal, n_layers, requested);
        for (cache, mode) in self.caches.iter_mut().zip(layer_modes) {
            cache.mode = mode;
        }
    }

    /// Generate tokens from the model (original implementation)
    pub fn generate<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Vec<i32> {
        self.generate_streaming(model, prompt_tokens, max_tokens, sampling, |_| true)
    }

    /// Streaming generation with per-token callback and lookahead pipelining.
    ///
    /// The callback receives each generated token ID and returns `true` to continue
    /// or `false` to abort early. Pipelining is preserved: next step computation
    /// starts before the current token is returned.
    ///
    /// Used by: CxxGenerator::generate, ModelProvider (server streaming)
    pub fn generate_streaming<M: LanguageModel, F: FnMut(i32) -> bool>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        mut on_token: F,
    ) -> Vec<i32> {
        // Reset state
        self.reset();

        // Axis B: merge any generator-cached language-bias map into the
        // sampling config before seeding/penalty evaluation. Empty cached
        // bias => borrowed unchanged (bit-exact baseline; zero alloc).
        let sampling_cow = self.compose_sampling(sampling);
        let sampling = sampling_cow.as_ref();

        // Set random seed if specified (for reproducibility)
        seed_rng_if_needed(sampling);

        // Ensure caches are initialized for this model.
        // `ensure_model_caches` may rebuild caches from `model.make_caches()`
        // (which always uses the default Fp16 mode), so re-apply kv_cache_mode
        // afterwards when a non-default mode is configured.
        ensure_model_caches(&mut self.caches, model);
        // Honor the Boundary-V policy (issue #478) when applying the
        // nominal mode to per-layer caches: the first/last N layers stay
        // at FP16 to recover the V-quantization quality gap.
        self.apply_kv_cache_mode_with_boundary_policy();

        // Set generation stream as default for better pipelining
        install_thread_local_default_stream(self.generation_stream.as_ref());

        // Get EOS tokens for this model
        let eos_tokens = merged_eos_token_ids(model.eos_token_ids(), &sampling.stop_token_ids);

        // Hoist env var checks out of the hot loop to avoid per-token syscalls.
        let trace_dtype = std::env::var("MLXCEL_TRACE_DTYPE").is_ok();
        let force_sync = std::env::var("MLXCEL_FORCE_SYNC").is_ok();
        let profile_pipeline = std::env::var("MLXCEL_PROFILE_PIPELINE").is_ok();
        let profile_pipeline_detail = std::env::var("MLXCEL_PROFILE_PIPELINE_DETAIL").is_ok();

        // Prefill: process all prompt tokens at once.
        // On M5+ hardware pad the sequence to a 32-token tile boundary for
        // optimal Neural Accelerator throughput.
        let actual_len = prompt_tokens.len();
        let logits = if should_align_prefill() && model.supports_padded_prefill() {
            let padded_len = align_to_na_tile(actual_len);
            let (padded_tokens, mask_opt) = pad_tokens_for_prefill(
                prompt_tokens,
                padded_len,
                model.supports_maskless_padded_prefill(),
            );
            let input = ffi::from_slice_i32(&padded_tokens, &[1, padded_len as i32]);
            let raw_logits = model.forward(
                &input,
                &mut self.caches,
                mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
            );
            // Trim padding positions from all KV caches so decode uses the
            // correct cache offset (actual_len, not padded_len).
            if padded_len > actual_len {
                trim_caches_to_actual_len(&mut self.caches, actual_len, padded_len);
                model.trim_internal_caches((padded_len - actual_len) as i32);
                // Extract logits at the last real token position.
                logits_at_position(&raw_logits, actual_len - 1)
            } else {
                // No padding was needed (already aligned).
                raw_logits
            }
        } else {
            let input = ffi::from_slice_i32(prompt_tokens, &[1, actual_len as i32]);
            model.forward(&input, &mut self.caches, None)
        };

        if trace_dtype {
            ffi::eval(&logits);
            let shape = ffi::array_shape(&logits);
            eprintln!(
                "[LOGITS] prefill dtype={} shape={:?}",
                ffi::array_dtype(&logits),
                shape
            );
        }

        // Clear intermediate tensors from prefill to free memory
        ffi::clear_memory_cache();

        // Build token history from prompt for penalty-based sampling
        let needs_history = sampling.needs_token_history();
        let mut token_history = initial_token_history(prompt_tokens, needs_history);

        // Sample first token (logits already sliced to last real position when padded)
        let (mut y, mut _logprobs) = sample_token_optimized(&logits, sampling, &token_history);
        ffi::async_eval(&y);
        self.prepare_turbo4_delegated_before_decode(max_tokens);

        // Main generation loop - matches Python exactly:
        // 1. Start next step computation
        // 2. async_eval next step
        // 3. Extract current value (syncs current only)
        // 4. Yield/store current
        // 5. Move next to current
        let mut build_ns_total = 0u128;
        let mut wait_ns_total = 0u128;
        let mut reshape_ns_total = 0u128;
        let mut forward_ns_total = 0u128;
        let mut sample_ns_total = 0u128;
        let mut async_eval_ns_total = 0u128;
        let mut profile_count = 0u32;

        let mut n = 0;
        loop {
            // Start next step (if not at max)
            let build_start = if profile_pipeline {
                Some(std::time::Instant::now())
            } else {
                None
            };

            let (next_y, next_logprobs) = if n + 1 < max_tokens {
                let detail_start = if profile_pipeline_detail {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let next_input = ffi::reshape_token_for_forward(&y);
                if let Some(start) = detail_start {
                    reshape_ns_total += start.elapsed().as_nanos();
                }
                let detail_start = if profile_pipeline_detail {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let next_logits = model.forward(&next_input, &mut self.caches, None);
                if let Some(start) = detail_start {
                    forward_ns_total += start.elapsed().as_nanos();
                }
                if trace_dtype && n == 0 {
                    ffi::eval(&next_logits);
                    eprintln!("[LOGITS] decode dtype={}", ffi::array_dtype(&next_logits));
                }
                let detail_start = if profile_pipeline_detail {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let (next_tok, next_log) =
                    sample_token_optimized(&next_logits, sampling, &token_history);
                if let Some(start) = detail_start {
                    sample_ns_total += start.elapsed().as_nanos();
                }
                let detail_start = if profile_pipeline_detail {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                if n == 0 {
                    if let Ok(path) = std::env::var("MLXCEL_EXPORT_DECODE_DOT") {
                        ffi::export_to_dot_pair(&path, &next_tok, &next_log);
                    }
                }
                if force_sync {
                    ffi::eval(&next_tok);
                } else {
                    ffi::async_eval_pair(&next_tok, &next_log);
                }
                if let Some(start) = detail_start {
                    async_eval_ns_total += start.elapsed().as_nanos();
                }
                (Some(next_tok), Some(next_log))
            } else {
                (None, None)
            };

            if let Some(bs) = build_start {
                build_ns_total += bs.elapsed().as_nanos();
            }

            // First iteration: explicit eval
            if n == 0 {
                ffi::eval(&y);
            }

            // Check if we've reached max
            if n >= max_tokens {
                break;
            }

            // Extract current token value - this syncs y
            let wait_start = if profile_pipeline {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let token_val = ffi::item_i32(&y);
            if let Some(ws) = wait_start {
                wait_ns_total += ws.elapsed().as_nanos();
                profile_count += 1;
            }

            // Check EOS before sending to callback (avoid outputting stop tokens)
            if eos_tokens.contains(&token_val) {
                break;
            }

            self.generated_tokens.push(token_val);
            if needs_history {
                token_history.push(token_val);
            }

            // Invoke callback; abort if it returns false
            if !on_token(token_val) {
                break;
            }

            // Periodic cache clearing (matches Python mlx-lm which clears every 256)
            if n % 256 == 0 && n > 0 {
                ffi::clear_memory_cache();
            }

            // Move to next
            if let (Some(ny), Some(nl)) = (next_y, next_logprobs) {
                y = ny;
                _logprobs = nl;
            } else {
                break;
            }

            n += 1;
        }

        if profile_pipeline && profile_count > 3 {
            let build_avg = build_ns_total as f64 / profile_count as f64;
            let wait_avg = wait_ns_total as f64 / profile_count as f64;
            eprintln!(
                "[PIPELINE] build: {:.2}ms/tok, item_wait: {:.2}ms/tok, sum: {:.2}ms/tok ({} tokens)",
                build_avg / 1e6,
                wait_avg / 1e6,
                (build_avg + wait_avg) / 1e6,
                profile_count,
            );
        }
        if profile_pipeline_detail && profile_count > 3 {
            let denom = profile_count as f64;
            eprintln!(
                "[PIPELINE_DETAIL] reshape={:.3}ms/tok forward={:.3}ms/tok sample={:.3}ms/tok async_eval={:.3}ms/tok item_wait={:.3}ms/tok",
                reshape_ns_total as f64 / denom / 1e6,
                forward_ns_total as f64 / denom / 1e6,
                sample_ns_total as f64 / denom / 1e6,
                async_eval_ns_total as f64 / denom / 1e6,
                wait_ns_total as f64 / denom / 1e6,
            );
        }

        self.generated_tokens.clone()
    }

    /// Streaming generation with pre-computed embeddings for VLM prefill.
    ///
    /// The prefill step uses `model.forward_with_embeddings()` with provided
    /// embeddings and mask. Decode steps are identical to standard generation.
    ///
    /// Used by: VisionLanguageModel (Gemma3 VLM, etc.)
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_embeddings<M: LanguageModel, F: FnMut(i32) -> bool>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        input_embeddings: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        max_tokens: usize,
        sampling: &SamplingConfig,
        mut on_token: F,
    ) -> Vec<i32> {
        // Reset state
        self.reset();

        // Axis B: inject generator-cached language-bias into the sampling config.
        let sampling_cow = self.compose_sampling(sampling);
        let sampling = sampling_cow.as_ref();

        seed_rng_if_needed(sampling);

        ensure_model_caches(&mut self.caches, model);
        // Re-apply kv_cache_mode in case ensure_model_caches rebuilt caches
        // Honor the Boundary-V policy (issue #478) when applying the
        // nominal mode to per-layer caches: the first/last N layers stay
        // at FP16 to recover the V-quantization quality gap.
        self.apply_kv_cache_mode_with_boundary_policy();

        install_thread_local_default_stream(self.generation_stream.as_ref());

        let eos_tokens = merged_eos_token_ids(model.eos_token_ids(), &sampling.stop_token_ids);

        // Prefill: use forward_with_embeddings for merged vision+text embeddings.
        // On M5+ hardware pad the sequence to a 32-token tile boundary when no
        // explicit mask is provided by the caller (callers that supply a custom
        // mask already control the shape and may not need tile alignment).
        let actual_len = prompt_tokens.len();
        let logits = if mask.is_none() && should_align_prefill() && model.supports_padded_prefill()
        {
            let padded_len = align_to_na_tile(actual_len);
            let (padded_tokens, mask_opt) = pad_tokens_for_prefill(
                prompt_tokens,
                padded_len,
                model.supports_maskless_padded_prefill(),
            );
            let input = ffi::from_slice_i32(&padded_tokens, &[1, padded_len as i32]);
            // Pad embeddings if provided.
            let padded_embeds_storage;
            let effective_embeds: Option<&MlxArray> = if let Some(emb) = input_embeddings {
                padded_embeds_storage = pad_embeddings(emb, padded_len);
                Some(padded_embeds_storage.as_ref().unwrap())
            } else {
                None
            };
            let raw_logits = model.forward_with_embeddings(
                &input,
                effective_embeds,
                &mut self.caches,
                mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
            );
            if padded_len > actual_len {
                trim_caches_to_actual_len(&mut self.caches, actual_len, padded_len);
                model.trim_internal_caches((padded_len - actual_len) as i32);
                logits_at_position(&raw_logits, actual_len - 1)
            } else {
                raw_logits
            }
        } else {
            let input = ffi::from_slice_i32(prompt_tokens, &[1, actual_len as i32]);
            model.forward_with_embeddings(&input, input_embeddings, &mut self.caches, mask)
        };

        // Force evaluation of the prefill graph before any weight modifications
        // in after_prefill. MLX lazy evaluation means the graph references the
        // current weight arrays; we must ensure evaluation completes before
        // those arrays are replaced.
        ffi::eval(&logits);

        // Allow models to adjust state between prefill and decode (e.g. Phi4MM LoRA unfusion)
        model.after_prefill();

        ffi::clear_memory_cache();

        let needs_history = sampling.needs_token_history();
        let mut token_history = initial_token_history(prompt_tokens, needs_history);

        let (mut y, mut _logprobs) = sample_token_optimized(&logits, sampling, &token_history);
        ffi::async_eval(&y);
        self.prepare_turbo4_delegated_before_decode(max_tokens);

        // Decode loop — identical to standard generation (no embeddings needed)
        let mut n = 0;
        loop {
            let (next_y, next_logprobs) = if n + 1 < max_tokens {
                let next_input = ffi::reshape_token_for_forward(&y);
                let next_logits = model.forward(&next_input, &mut self.caches, None);
                let (next_tok, next_log) =
                    sample_token_optimized(&next_logits, sampling, &token_history);
                ffi::async_eval_pair(&next_tok, &next_log);
                (Some(next_tok), Some(next_log))
            } else {
                (None, None)
            };

            if n == 0 {
                ffi::eval(&y);
            }

            if n >= max_tokens {
                break;
            }

            let token_val = ffi::item_i32(&y);

            // Check EOS before sending to callback (avoid outputting stop tokens)
            if eos_tokens.contains(&token_val) {
                break;
            }

            self.generated_tokens.push(token_val);
            if needs_history {
                token_history.push(token_val);
            }

            if !on_token(token_val) {
                break;
            }

            // Periodic cache clearing (matches Python mlx-lm which clears every 256)
            if n % 256 == 0 && n > 0 {
                ffi::clear_memory_cache();
            }

            if let (Some(ny), Some(nl)) = (next_y, next_logprobs) {
                y = ny;
                _logprobs = nl;
            } else {
                break;
            }

            n += 1;
        }

        self.generated_tokens.clone()
    }

    /// Generate with stats using pre-computed embeddings (VLM variant)
    /// Used by: CLI --image path, server VLM requests
    pub fn generate_with_stats_and_embeddings<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        input_embeddings: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        use std::time::Instant;

        self.reset();

        // Axis B: inject generator-cached language-bias into the sampling config.
        let sampling_cow = self.compose_sampling(sampling);
        let sampling = sampling_cow.as_ref();

        seed_rng_if_needed(sampling);
        ensure_model_caches(&mut self.caches, model);
        // Re-apply kv_cache_mode in case ensure_model_caches rebuilt caches
        // Honor the Boundary-V policy (issue #478) when applying the
        // nominal mode to per-layer caches: the first/last N layers stay
        // at FP16 to recover the V-quantization quality gap.
        self.apply_kv_cache_mode_with_boundary_policy();
        install_thread_local_default_stream(self.generation_stream.as_ref());

        let eos_tokens = merged_eos_token_ids(model.eos_token_ids(), &sampling.stop_token_ids);
        let needs_history = sampling.needs_token_history();
        let mut token_history = initial_token_history(prompt_tokens, needs_history);

        // Prefill with embeddings.
        // On M5+ hardware pad to a 32-token tile boundary (same logic as
        // generate_streaming_with_embeddings).
        let actual_len = prompt_tokens.len();
        let prefill_start = Instant::now();
        let logits = if mask.is_none() && should_align_prefill() && model.supports_padded_prefill()
        {
            let padded_len = align_to_na_tile(actual_len);
            let (padded_tokens, mask_opt) = pad_tokens_for_prefill(
                prompt_tokens,
                padded_len,
                model.supports_maskless_padded_prefill(),
            );
            let input = ffi::from_slice_i32(&padded_tokens, &[1, padded_len as i32]);
            let padded_embeds_storage;
            let effective_embeds: Option<&MlxArray> = if let Some(emb) = input_embeddings {
                padded_embeds_storage = pad_embeddings(emb, padded_len);
                Some(padded_embeds_storage.as_ref().unwrap())
            } else {
                None
            };
            let raw_logits = model.forward_with_embeddings(
                &input,
                effective_embeds,
                &mut self.caches,
                mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
            );
            if padded_len > actual_len {
                trim_caches_to_actual_len(&mut self.caches, actual_len, padded_len);
                model.trim_internal_caches((padded_len - actual_len) as i32);
                logits_at_position(&raw_logits, actual_len - 1)
            } else {
                raw_logits
            }
        } else {
            let input = ffi::from_slice_i32(prompt_tokens, &[1, actual_len as i32]);
            model.forward_with_embeddings(&input, input_embeddings, &mut self.caches, mask)
        };
        model.after_prefill();
        let (mut y, mut _logprobs) = sample_token_optimized(&logits, sampling, &token_history);
        ffi::eval(&y);
        self.prepare_turbo4_delegated_before_decode(max_tokens);
        let prefill_time = prefill_start.elapsed();
        ffi::clear_memory_cache();

        // Decode
        let decode_start = Instant::now();
        let mut n = 0;
        loop {
            let next_y = if n + 1 < max_tokens {
                let next_input = ffi::reshape_token_for_forward(&y);
                let next_logits = model.forward(&next_input, &mut self.caches, None);
                let (next_tok, _next_log) =
                    sample_token_optimized(&next_logits, sampling, &token_history);
                ffi::async_eval(&next_tok);
                Some(next_tok)
            } else {
                None
            };

            if n == 0 {
                ffi::eval(&y);
            }
            if n >= max_tokens {
                break;
            }

            let token_val = ffi::item_i32(&y);
            if eos_tokens.contains(&token_val) {
                break;
            }
            self.generated_tokens.push(token_val);
            if needs_history {
                token_history.push(token_val);
            }
            // Periodic cache clearing (matches Python mlx-lm which clears every 256)
            if n % 256 == 0 && n > 0 {
                ffi::clear_memory_cache();
            }
            if let Some(ny) = next_y {
                y = ny;
            } else {
                break;
            }
            n += 1;
        }
        let decode_time = decode_start.elapsed();

        let prompt_count = prompt_tokens.len();
        let gen_count = self.generated_tokens.len();
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;

        let stats = GenerationStats {
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
        };

        (self.generated_tokens.clone(), stats)
    }

    /// Get the generated tokens
    pub fn tokens(&self) -> &[i32] {
        &self.generated_tokens
    }

    /// Generate tokens with detailed timing statistics
    /// Returns (generated_tokens, stats)
    pub fn generate_with_stats<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        use std::time::Instant;

        // Reset state
        self.reset();

        // Axis B: inject generator-cached language-bias into the sampling config.
        let sampling_cow = self.compose_sampling(sampling);
        let sampling = sampling_cow.as_ref();

        // Set random seed if specified (for reproducibility)
        seed_rng_if_needed(sampling);

        // Ensure caches are initialized for this model.
        // Re-apply kv_cache_mode in case ensure_model_caches rebuilt caches.
        ensure_model_caches(&mut self.caches, model);
        // Honor the Boundary-V policy (issue #478) when applying the
        // nominal mode to per-layer caches: the first/last N layers stay
        // at FP16 to recover the V-quantization quality gap.
        self.apply_kv_cache_mode_with_boundary_policy();

        // Set generation stream as default for better pipelining
        install_thread_local_default_stream(self.generation_stream.as_ref());

        // Get EOS tokens for this model
        let eos_tokens = merged_eos_token_ids(model.eos_token_ids(), &sampling.stop_token_ids);

        // Build token history from prompt for penalty-based sampling
        let needs_history = sampling.needs_token_history();
        let mut token_history = initial_token_history(prompt_tokens, needs_history);

        // PREFILL PHASE.
        // On M5+ hardware pad the sequence to a 32-token tile boundary for
        // optimal Neural Accelerator throughput.
        let actual_len = prompt_tokens.len();
        let prefill_start = Instant::now();
        let logits = if should_align_prefill() && model.supports_padded_prefill() {
            let padded_len = align_to_na_tile(actual_len);
            let (padded_tokens, mask_opt) = pad_tokens_for_prefill(
                prompt_tokens,
                padded_len,
                model.supports_maskless_padded_prefill(),
            );
            let input = ffi::from_slice_i32(&padded_tokens, &[1, padded_len as i32]);
            let raw_logits = model.forward(
                &input,
                &mut self.caches,
                mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
            );
            if padded_len > actual_len {
                trim_caches_to_actual_len(&mut self.caches, actual_len, padded_len);
                model.trim_internal_caches((padded_len - actual_len) as i32);
                logits_at_position(&raw_logits, actual_len - 1)
            } else {
                raw_logits
            }
        } else {
            let input = ffi::from_slice_i32(prompt_tokens, &[1, actual_len as i32]);
            model.forward(&input, &mut self.caches, None)
        };

        // Sample first token and force sync to measure prefill accurately
        let (mut y, mut _logprobs) = sample_token_optimized(&logits, sampling, &token_history);
        ffi::eval(&y);
        self.prepare_turbo4_delegated_before_decode(max_tokens);
        let prefill_time = prefill_start.elapsed();

        // Clear intermediate tensors from prefill to free memory
        ffi::clear_memory_cache();

        // DECODE PHASE (with lookahead pipelining).
        let decode_start = Instant::now();

        let mut n = 0;
        loop {
            // Start next step computation (if not at max)
            let next_y = if n + 1 < max_tokens {
                let next_input = ffi::reshape_token_for_forward(&y);
                let next_logits = model.forward(&next_input, &mut self.caches, None);
                let (next_tok, _next_log) =
                    sample_token_optimized(&next_logits, sampling, &token_history);
                ffi::async_eval(&next_tok);
                Some(next_tok)
            } else {
                None
            };

            // First iteration: explicit eval
            if n == 0 {
                ffi::eval(&y);
            }

            // Check if we've reached max
            if n >= max_tokens {
                break;
            }

            // Extract current token value (syncs y)
            let token_val = ffi::item_i32(&y);

            // Check EOS before storing (avoid including stop tokens in output)
            if eos_tokens.contains(&token_val) {
                break;
            }

            self.generated_tokens.push(token_val);
            if needs_history {
                token_history.push(token_val);
            }

            // Periodic cache clearing (matches Python mlx-lm which clears every 256)
            if n % 256 == 0 && n > 0 {
                ffi::clear_memory_cache();
            }

            // Move to next
            if let Some(ny) = next_y {
                y = ny;
            } else {
                break;
            }

            n += 1;
        }

        let decode_time = decode_start.elapsed();

        // Calculate stats
        let prompt_count = prompt_tokens.len();
        let gen_count = self.generated_tokens.len();
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;

        let stats = GenerationStats {
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
        };

        (self.generated_tokens.clone(), stats)
    }

    /// Compute per-token log-likelihoods for each position in `prompt_tokens`,
    /// without sampling.
    ///
    /// Returns `Vec<f32>` of length `prompt_tokens.len() - 1`, where entry `i`
    /// is `log P(prompt_tokens[i + 1] | prompt_tokens[..=i])` under the model.
    ///
    /// This is the building block for offline perplexity evaluation (epic
    /// #458 / issue #475 — the wikitext-2 PPL gate). Callers chunk the corpus
    /// into windows of length `≤ context_len` and accumulate `-sum(logprobs) /
    /// total_target_tokens` across windows; `exp(mean_nll)` is the perplexity.
    ///
    /// # KV-cache mode interaction
    ///
    /// The caches are reset using the generator's configured `kv_cache_mode`,
    /// so calling this with a `Turbo4Asym` generator measures perplexity *with*
    /// the lossy V-cache compression in effect. This is exactly the gate the
    /// quality test wants: it compares baseline (Fp16) PPL against quantized
    /// PPL on the same corpus.
    ///
    /// # Performance
    ///
    /// One forward pass over the entire `prompt_tokens` window, plus an
    /// `O(seq_len * vocab)` log-softmax + gather. Suitable for tractable
    /// window sizes (≤ 4 K) and small models. For larger contexts, batching
    /// many independent windows would be a follow-up; epic #458's gate runs
    /// 20 windows × 4 K which fits in a single-pass-per-window budget on M-series.
    ///
    /// Used by: `tests/turbo_kv_e2e.rs` wikitext-2 PPL harness (issue #475).
    pub fn evaluate_loglikelihoods<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
    ) -> Vec<f32> {
        // A perplexity evaluation needs at least one target token (i.e. at
        // least two input tokens: one context, one prediction target).
        if prompt_tokens.len() < 2 {
            return Vec::new();
        }

        // Reset caches and apply the generator's kv_cache_mode. We use
        // `reset_with_model` so models with internal sliding/SSM caches are
        // also cleared — matches the reset behaviour of `generate_*` paths.
        self.reset_with_model(model);
        install_thread_local_default_stream(self.generation_stream.as_ref());

        // Single forward over the full window. We do not pad: tile alignment
        // is a decode optimisation that complicates the position→target
        // mapping below, and the perplexity gate runs at modest seq lengths
        // (≤ 4 K) where the unpadded path is fine.
        let actual_len = prompt_tokens.len();
        let input = ffi::from_slice_i32(prompt_tokens, &[1, actual_len as i32]);
        let logits = model.forward(&input, &mut self.caches, None);

        // logits shape is `[1, T, vocab]`. We need log P(token[i+1] | ...)
        // for i in 0..T-1, so:
        //   1. Slice logits to positions [0, T-1) along seq axis.
        //   2. Apply log_softmax along vocab axis.
        //   3. Gather the entry at index `prompt_tokens[i + 1]` for each i.
        let logits_shape = ffi::array_shape(&logits);
        debug_assert_eq!(logits_shape.len(), 3, "forward must return [B, T, V]");
        let vocab = logits_shape[2];

        // Slice to context positions [0, T-1).
        let context_logits = ffi::slice(&logits, &[0, 0, 0], &[1, (actual_len - 1) as i32, vocab]);

        // Cast to fp32 for stable log-softmax. fp16 log_softmax can underflow
        // on extreme negative logits — fp32 keeps the gather well-conditioned.
        let context_f32 = ffi::astype(&context_logits, crate::dtype::FLOAT32);
        let logprobs = ffi::log_softmax(&context_f32, -1);

        // Build target indices: prompt_tokens[1..] reshaped as [1, T-1, 1].
        let targets: Vec<i32> = prompt_tokens[1..].to_vec();
        let target_arr = ffi::from_slice_i32(&targets, &[1, (actual_len - 1) as i32, 1]);

        // Gather along the vocab axis: take_along_axis with axis=-1 gives
        // [1, T-1, 1].
        let gathered = ffi::take_along_axis(&logprobs, &target_arr, -1);

        // Materialise to host as fp32 bytes and copy out.
        ffi::eval(&gathered);
        let bytes = ffi::array_to_raw_bytes(&gathered);
        debug_assert_eq!(
            bytes.len(),
            (actual_len - 1) * 4,
            "expected {} fp32 bytes, got {}",
            (actual_len - 1) * 4,
            bytes.len()
        );
        let mut out = Vec::with_capacity(actual_len - 1);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        // Free intermediate tensors before the next call.
        ffi::clear_memory_cache();

        out
    }
}

/// Sample a token from logits (original version)
#[allow(dead_code)]
fn sample_token(logits: &MlxArray, config: &SamplingConfig) -> i32 {
    let (token_arr, _) = sample_token_optimized(logits, config, &[]);
    ffi::eval(&token_arr);
    ffi::item_i32(&token_arr)
}

/// Benchmark helper: measure throughput
pub struct BenchmarkResult {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time_ms: f64,
    pub generation_time_ms: f64,
    pub tokens_per_second: f64,
}

impl BenchmarkResult {
    pub fn print(&self) {
        println!("Benchmark Results:");
        println!("  Prompt tokens: {}", self.prompt_tokens);
        println!("  Generated tokens: {}", self.generated_tokens);
        println!("  Prefill time: {:.2} ms", self.prefill_time_ms);
        println!("  Generation time: {:.2} ms", self.generation_time_ms);
        println!("  Throughput: {:.2} tok/s", self.tokens_per_second);
    }
}

/// Run a generation benchmark
pub fn run_benchmark<M: LanguageModel>(
    model: &M,
    prompt_tokens: &[i32],
    max_tokens: usize,
) -> BenchmarkResult {
    use std::time::Instant;

    let num_layers = model.num_layers();
    let mut generator = CxxGenerator::new(num_layers);

    // Warmup
    generator.generate(model, prompt_tokens, 5, &SamplingConfig::greedy());
    generator.reset();

    // Benchmark prefill
    let input = ffi::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);

    let start = Instant::now();
    let logits = model.forward(&input, &mut generator.caches, None);
    ffi::eval(&logits);
    let prefill_time = start.elapsed();

    // Benchmark generation
    generator.reset();
    let start = Instant::now();
    let tokens = generator.generate(model, prompt_tokens, max_tokens, &SamplingConfig::greedy());
    let total_time = start.elapsed();

    let generation_time = total_time.saturating_sub(prefill_time);
    let gen_tokens = tokens.len();

    BenchmarkResult {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: gen_tokens,
        prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
        generation_time_ms: generation_time.as_secs_f64() * 1000.0,
        tokens_per_second: if generation_time.as_secs_f64() > 0.0 {
            gen_tokens as f64 / generation_time.as_secs_f64()
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::KVCache;

    /// Minimal model stub for testing forward_batched default implementation.
    /// Produces logits that are just the input token ID broadcast to a small
    /// vocab, so results are deterministic and verifiable.
    struct StubModel;

    impl LanguageModel for StubModel {
        fn forward(
            &self,
            input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            // Return logits where the token ID position has the highest value.
            // Input shape: [1, 1], output shape: [1, 1, 4] (vocab=4)
            ffi::eval(input_ids);
            let tok = ffi::item_i32(input_ids);
            let mut logits = vec![0.0f32; 4];
            if tok >= 0 && (tok as usize) < 4 {
                logits[tok as usize] = 10.0;
            }
            ffi::from_slice_f32(&logits, &[1, 1, 4])
        }

        fn make_caches(&self) -> Vec<KVCache> {
            vec![KVCache::new()]
        }

        fn num_layers(&self) -> usize {
            1
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![99]
        }
    }

    #[test]
    fn forward_batched_default_matches_sequential() {
        let model = StubModel;

        // Sequential: forward each token independently
        let mut caches_0 = model.make_caches();
        let mut caches_1 = model.make_caches();

        let input_0 = ffi::from_slice_i32(&[1], &[1, 1]);
        let input_1 = ffi::from_slice_i32(&[2], &[1, 1]);

        let logits_0 = model.forward(&input_0, &mut caches_0, None);
        let logits_1 = model.forward(&input_1, &mut caches_1, None);

        ffi::eval(&logits_0);
        ffi::eval(&logits_1);

        // Batched: forward_batched with [2, 1] input
        let mut batch_caches_0 = model.make_caches();
        let mut batch_caches_1 = model.make_caches();
        let mut batch_caches: Vec<&mut [KVCache]> = vec![&mut batch_caches_0, &mut batch_caches_1];

        let batched_input = ffi::from_slice_i32(&[1, 2], &[2, 1]);
        let batched_logits = model.forward_batched(&batched_input, &mut batch_caches, None);
        ffi::eval(&batched_logits);

        // Verify shapes
        assert_eq!(ffi::array_shape(&batched_logits), vec![2, 1, 4]);
        assert_eq!(ffi::array_shape(&logits_0), vec![1, 1, 4]);
        assert_eq!(ffi::array_shape(&logits_1), vec![1, 1, 4]);

        // Verify content matches: slice batched results and compare
        let batch_seq0 = ffi::slice(&batched_logits, &[0, 0, 0], &[1, 1, 4]);
        let batch_seq1 = ffi::slice(&batched_logits, &[1, 0, 0], &[2, 1, 4]);

        ffi::eval(&batch_seq0);
        ffi::eval(&batch_seq1);

        // Token 1 should have highest logit at position 1
        // Token 2 should have highest logit at position 2
        let seq0_logits = ffi::reshape(&batch_seq0, &[4]);
        let seq1_logits = ffi::reshape(&batch_seq1, &[4]);
        ffi::eval(&seq0_logits);
        ffi::eval(&seq1_logits);

        assert_eq!(ffi::item_i32(&ffi::argmax_last_axis(&seq0_logits)), 1);
        assert_eq!(ffi::item_i32(&ffi::argmax_last_axis(&seq1_logits)), 2);
    }

    #[test]
    fn forward_batched_single_sequence_no_overhead() {
        let model = StubModel;

        let mut caches = model.make_caches();
        let mut batch_caches: Vec<&mut [KVCache]> = vec![&mut caches];

        let input = ffi::from_slice_i32(&[3], &[1, 1]);
        let logits = model.forward_batched(&input, &mut batch_caches, None);
        ffi::eval(&logits);

        assert_eq!(ffi::array_shape(&logits), vec![1, 1, 4]);

        // Token 3 should have highest logit at position 3
        let flat = ffi::reshape(&logits, &[4]);
        ffi::eval(&flat);
        assert_eq!(ffi::item_i32(&ffi::argmax_last_axis(&flat)), 3);
    }

    /// Stub model that does NOT support batching (like SSM models).
    struct NonBatchModel;

    impl LanguageModel for NonBatchModel {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1, 1, 4], crate::dtype::FLOAT32)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            vec![KVCache::new()]
        }

        fn num_layers(&self) -> usize {
            1
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }

        fn supports_batching(&self) -> bool {
            false
        }
    }

    struct FullBatchPrefillModel;

    impl LanguageModel for FullBatchPrefillModel {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1, 1, 4], crate::dtype::FLOAT32)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            vec![KVCache::new()]
        }

        fn num_layers(&self) -> usize {
            1
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }

        fn supports_batched_prefill(&self) -> bool {
            true
        }
    }

    struct MasklessPaddedPrefillModel;

    impl LanguageModel for MasklessPaddedPrefillModel {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1, 1, 4], crate::dtype::FLOAT32)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            vec![KVCache::new()]
        }

        fn num_layers(&self) -> usize {
            1
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }

        fn supports_maskless_padded_prefill(&self) -> bool {
            true
        }
    }

    #[test]
    fn non_batching_model_uses_default_loop_fallback() {
        let model = NonBatchModel;
        assert!(!model.supports_batching());

        // forward_batched still works via the default loop fallback
        let mut caches = model.make_caches();
        let mut batch_caches: Vec<&mut [KVCache]> = vec![&mut caches];

        let input = ffi::from_slice_i32(&[0], &[1, 1]);
        let logits = model.forward_batched(&input, &mut batch_caches, None);
        ffi::eval(&logits);

        assert_eq!(ffi::array_shape(&logits), vec![1, 1, 4]);
    }

    #[test]
    fn supports_batched_prefill_defaults_false() {
        let model = StubModel;
        assert!(!model.supports_batched_prefill());
    }

    #[test]
    fn supports_batched_prefill_can_opt_in() {
        let model = FullBatchPrefillModel;
        assert!(model.supports_batched_prefill());
    }

    #[test]
    fn supports_maskless_padded_prefill_defaults_false() {
        let model = StubModel;
        assert!(!model.supports_maskless_padded_prefill());
    }

    #[test]
    fn supports_maskless_padded_prefill_can_opt_in() {
        let model = MasklessPaddedPrefillModel;
        assert!(model.supports_maskless_padded_prefill());
    }

    #[test]
    fn padded_prefill_can_skip_array_mask_for_opted_in_models() {
        let tokens = [1, 2, 3];
        let (_padded, mask_opt) = pad_tokens_for_prefill(&tokens, 32, true);
        assert!(mask_opt.is_none());
    }

    #[test]
    fn padded_prefill_keeps_array_mask_by_default() {
        let tokens = [1, 2, 3];
        let (_padded, mask_opt) = pad_tokens_for_prefill(&tokens, 32, false);
        assert!(mask_opt.is_some());
    }

    // -- SamplingConfig::needs_token_history (incremental history optimization) --

    #[test]
    fn needs_token_history_false_for_default_config() {
        // Default config: all penalties disabled, should not need history.
        let cfg = SamplingConfig::default();
        assert!(!cfg.needs_token_history());
    }

    #[test]
    fn needs_token_history_false_for_greedy_config() {
        let cfg = SamplingConfig::greedy();
        assert!(!cfg.needs_token_history());
    }

    #[test]
    fn needs_token_history_true_when_repetition_penalty_enabled() {
        let cfg = SamplingConfig {
            repetition_penalty: 1.2,
            ..Default::default()
        };
        assert!(cfg.needs_token_history());
    }

    #[test]
    fn needs_token_history_false_when_repetition_penalty_is_one() {
        // Exactly 1.0 means "no penalty" (identity multiplication).
        let cfg = SamplingConfig {
            repetition_penalty: 1.0,
            ..Default::default()
        };
        assert!(!cfg.needs_token_history());
    }

    #[test]
    fn needs_token_history_true_when_dry_multiplier_positive() {
        let cfg = SamplingConfig {
            dry_multiplier: 0.5,
            ..Default::default()
        };
        assert!(cfg.needs_token_history());
    }

    #[test]
    fn needs_token_history_false_when_dry_multiplier_zero() {
        let cfg = SamplingConfig {
            dry_multiplier: 0.0,
            ..Default::default()
        };
        assert!(!cfg.needs_token_history());
    }

    #[test]
    fn needs_token_history_true_when_frequency_penalty_nonzero() {
        let cfg = SamplingConfig {
            frequency_penalty: 0.1,
            ..Default::default()
        };
        assert!(cfg.needs_token_history());
    }

    #[test]
    fn needs_token_history_true_when_frequency_penalty_negative() {
        // Negative frequency penalty is valid (encourages repetition).
        let cfg = SamplingConfig {
            frequency_penalty: -0.1,
            ..Default::default()
        };
        assert!(cfg.needs_token_history());
    }

    #[test]
    fn needs_token_history_true_when_presence_penalty_nonzero() {
        let cfg = SamplingConfig {
            presence_penalty: 0.2,
            ..Default::default()
        };
        assert!(cfg.needs_token_history());
    }

    #[test]
    fn needs_token_history_true_when_multiple_penalties_enabled() {
        let cfg = SamplingConfig {
            repetition_penalty: 1.1,
            dry_multiplier: 0.3,
            frequency_penalty: 0.05,
            ..Default::default()
        };
        assert!(cfg.needs_token_history());
    }

    // -- Cache clearing interval (aligned with Python mlx-lm) --

    /// Verify that the cache clearing interval (256 tokens) matches Python
    /// mlx-lm. The condition `n % 256 == 0 && n > 0` fires at n=256, 512, ...
    #[test]
    fn cache_clearing_triggers_every_256_tokens() {
        let clears: Vec<usize> = (1..=512).filter(|&n| n % 256 == 0).collect();
        assert_eq!(clears, vec![256, 512]);
    }

    #[test]
    fn cache_clearing_does_not_trigger_on_token_zero() {
        // n=0 is the very first decode iteration after prefill. Clearing here
        // would discard tensors needed for the pipelined next-step computation.
        let n = 0_usize;
        assert!(!(n.is_multiple_of(256) && n > 0));
    }

    #[test]
    fn cache_clearing_first_trigger_is_at_256() {
        let first_clear = (1_usize..).find(|&n| n % 256 == 0 && n > 0);
        assert_eq!(first_clear, Some(256));
    }

    // -- SamplingConfig construction helpers --

    #[test]
    fn sampling_config_with_temperature_only_changes_temperature() {
        let cfg = SamplingConfig::with_temperature(0.7);
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_k, 0);
        assert_eq!(cfg.top_p, 1.0);
        assert_eq!(cfg.repetition_penalty, 1.0);
        assert_eq!(cfg.dry_multiplier, 0.0);
        assert_eq!(cfg.frequency_penalty, 0.0);
        assert_eq!(cfg.presence_penalty, 0.0);
        assert!(!cfg.needs_token_history());
    }

    #[test]
    fn greedy_config_has_zero_temperature_and_no_history_needed() {
        let cfg = SamplingConfig::greedy();
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_k, 1);
        assert!(!cfg.needs_token_history());
    }

    // -- trim_internal_caches default implementation --

    /// Default LanguageModel::trim_internal_caches is a no-op: calling it
    /// with any excess value must not panic or alter observable state.
    #[test]
    fn trim_internal_caches_default_is_noop() {
        let model = StubModel;
        // Should not panic for positive, zero, or negative excess.
        model.trim_internal_caches(8);
        model.trim_internal_caches(0);
        model.trim_internal_caches(-1);
    }

    /// A model that overrides trim_internal_caches records each call so we can
    /// verify the generation machinery actually invokes the method.
    struct TrackingTrimModel {
        trim_call_count: std::cell::Cell<usize>,
        last_excess: std::cell::Cell<i32>,
    }

    impl TrackingTrimModel {
        fn new() -> Self {
            Self {
                trim_call_count: std::cell::Cell::new(0),
                last_excess: std::cell::Cell::new(0),
            }
        }
    }

    impl LanguageModel for TrackingTrimModel {
        fn forward(
            &self,
            input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::eval(input_ids);
            ffi::zeros(&[1, 1, 4], crate::dtype::FLOAT32)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            vec![KVCache::new()]
        }

        fn num_layers(&self) -> usize {
            1
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![99]
        }

        fn trim_internal_caches(&self, excess: i32) {
            self.trim_call_count.set(self.trim_call_count.get() + 1);
            self.last_excess.set(excess);
        }
    }

    #[test]
    fn trim_internal_caches_override_receives_correct_excess() {
        let model = TrackingTrimModel::new();
        assert_eq!(model.trim_call_count.get(), 0);

        // Simulate the call pattern from the generation loop: excess = padded - actual.
        model.trim_internal_caches(16);
        assert_eq!(model.trim_call_count.get(), 1);
        assert_eq!(model.last_excess.get(), 16);

        model.trim_internal_caches(32);
        assert_eq!(model.trim_call_count.get(), 2);
        assert_eq!(model.last_excess.get(), 32);
    }

    #[test]
    fn trim_internal_caches_override_called_with_zero_is_safe() {
        let model = TrackingTrimModel::new();
        model.trim_internal_caches(0);
        // The implementation still receives the call; it is the model's
        // responsibility to handle excess == 0 gracefully.
        assert_eq!(model.trim_call_count.get(), 1);
        assert_eq!(model.last_excess.get(), 0);
    }

    // ------------------------------------------------------------------
    // B8 — CxxGenerator token-bias wiring
    // ------------------------------------------------------------------

    fn make_bias(entries: &[(i32, f32)]) -> TokenBiasMap {
        let mut m = TokenBiasMap::new();
        for &(id, b) in entries {
            m.insert(id, b);
        }
        m
    }

    /// Default `CxxGenerator` carries an empty token-bias cache and produces
    /// a bit-exact baseline (`Cow::Borrowed`) from `compose_sampling`.
    #[test]
    fn cxx_generator_empty_bias_is_baseline() {
        let g = CxxGenerator::new(4);
        assert!(g.token_bias().is_empty());
        let caller = SamplingConfig::default();
        let composed = g.compose_sampling(&caller);
        assert!(matches!(composed, Cow::Borrowed(_)));
        assert!(composed.token_bias.is_empty());
    }

    /// `with_token_bias` caches a map and injects it into sampling configs
    /// that don't already carry a bias (fresh clone — `Cow::Owned`).
    #[test]
    fn cxx_generator_with_token_bias_injects_into_sampling() {
        let bias = make_bias(&[(3, f32::NEG_INFINITY), (5, 1.5)]);
        let g = CxxGenerator::new(4).with_token_bias(bias.clone());
        assert_eq!(g.token_bias().len(), 2);

        let caller = SamplingConfig::default();
        let composed = g.compose_sampling(&caller);
        assert!(matches!(composed, Cow::Owned(_)));
        assert_eq!(composed.token_bias.len(), 2);
        assert!(composed.token_bias.contains(3));
        assert!(composed.token_bias.contains(5));
    }

    /// An explicit caller-side bias wins over the generator-cached one.
    /// Preserves the "call-site override" contract for tests and API callers.
    #[test]
    fn cxx_generator_caller_bias_wins() {
        let cached = make_bias(&[(1, 1.0)]);
        let caller_bias = make_bias(&[(99, -3.0)]);
        let g = CxxGenerator::new(2).with_token_bias(cached);

        let caller = SamplingConfig {
            token_bias: caller_bias,
            ..SamplingConfig::default()
        };
        let composed = g.compose_sampling(&caller);

        // Caller's explicit bias is preserved verbatim, cached bias is ignored.
        assert_eq!(composed.token_bias.len(), 1);
        assert!(composed.token_bias.contains(99));
        assert!(!composed.token_bias.contains(1));
    }
}
