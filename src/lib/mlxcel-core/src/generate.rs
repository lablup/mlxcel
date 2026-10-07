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

use crate::cache::{CachePool, KVCacheMode, SequenceId};
use crate::ffi;
use crate::ffi::MlxArray;
use crate::hardware;
use crate::layers::KVCache;
use crate::loop_detection::LoopDetectionConfig;
use crate::sampling::TokenBiasMap;
use cxx::UniquePtr;

/// One named tensor captured from a model-owned recurrent sequence state.
///
/// The prompt-cache snapshot path stores these tensors outside the model and
/// later asks the same model family to restore them into a fresh sequence id.
/// The names are intentionally model-defined: the core runtime only provides a
/// small typed container and byte accounting, while each model validates the
/// fields it understands during restore.
pub struct ModelStateTensor {
    name: String,
    array: UniquePtr<MlxArray>,
}

impl ModelStateTensor {
    /// Capture `array` under `name` through `ffi::copy`. The copy is lazy
    /// and, once evaluated, shares `array`'s buffer rather than duplicating
    /// it. Rows inside the captured window stay unchanged because a later
    /// `slice_update` of `array` copies instead of writing in place while
    /// this copy still references `array` or its buffer (#2052); the in-place
    /// decode write (`inplace_slice_write`) only fills rows past the window.
    pub fn new(name: impl Into<String>, array: &MlxArray) -> Self {
        Self {
            name: name.into(),
            array: ffi::copy(array),
        }
    }

    /// Field name chosen by the model snapshot implementation.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the captured tensor.
    pub fn array(&self) -> &MlxArray {
        self.array
            .as_ref()
            .expect("model-state snapshot tensor must not be null")
    }

    /// Byte footprint of this captured tensor.
    pub fn nbytes(&self) -> usize {
        ffi::array_nbytes(self.array())
    }
}

/// Inert copy of a model-owned recurrent/cache state at an exact token prefix.
///
/// This is deliberately separate from detached KV-cache entries: recurrent
/// SSM / linear-attention families cannot safely share arbitrary KV blocks, so
/// the server parks a full-state snapshot and restores copies only on an exact
/// stored-prefix hit.
pub struct ModelStateSnapshot {
    family: String,
    token_len: usize,
    tensors: Vec<ModelStateTensor>,
}

impl ModelStateSnapshot {
    /// Build an empty snapshot for `family` at `token_len` tokens.
    pub fn new(family: impl Into<String>, token_len: usize) -> Self {
        Self {
            family: family.into(),
            token_len,
            tensors: Vec::new(),
        }
    }

    /// Model-family tag used to reject accidental cross-family restores.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Number of tokens represented by this exact-prefix state.
    pub fn token_len(&self) -> usize {
        self.token_len
    }

    /// Append a named tensor copy.
    pub fn push_tensor(&mut self, name: impl Into<String>, array: &MlxArray) {
        self.tensors.push(ModelStateTensor::new(name, array));
    }

    /// Borrow the named tensor if present.
    pub fn tensor(&self, name: &str) -> Option<&MlxArray> {
        self.tensors
            .iter()
            .find(|t| t.name() == name)
            .map(ModelStateTensor::array)
    }

    /// Whether no tensor payload was captured.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Sum of all captured tensor byte footprints.
    pub fn nbytes(&self) -> usize {
        self.tensors.iter().map(ModelStateTensor::nbytes).sum()
    }
}

/// Whether prefill should be padded to the 32-token Neural Accelerator tile.
///
/// True on M5+ hardware with a Neural Accelerator driven by the running macOS.
/// Two debugging overrides apply, checked in this order:
///
/// - `MLXCEL_NO_PADDED_PREFILL` (presence) disables tile alignment.
/// - `MLXCEL_FORCE_PADDED_PREFILL` (presence) enables it on any hardware, so
///   the padded prefill and its cache trim can be exercised on hosts without a
///   Neural Accelerator (issue #1755). Padding is pure overhead off M5.
///
/// Only the prefill padding decision follows these overrides; speculative
/// verification alignment and the M5 numerical workarounds keep reading the
/// hardware directly.
///
/// Used by: the CLI prefill paths in this module and the server batch
/// scheduler's prefill sites.
#[inline]
pub fn prefill_tile_alignment_enabled() -> bool {
    if std::env::var_os("MLXCEL_NO_PADDED_PREFILL").is_some() {
        return false;
    }
    if std::env::var_os("MLXCEL_FORCE_PADDED_PREFILL").is_some() {
        return true;
    }
    let hw = hardware::get_hardware();
    hw.has_neural_accelerator && hw.macos_supports_na
}

/// Pad an embeddings tensor from `[batch, actual_len, hidden]` to
/// `[batch, padded_len, hidden]` by appending zero rows.
///
/// Used by the VLM tile-alignment path to match the padded token sequence.
pub(crate) fn pad_embeddings(embeds: &MlxArray, padded_len: usize) -> UniquePtr<MlxArray> {
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
/// token position rather than the last padding position. Works on any
/// `[batch, seq_len, width]` tensor, so a model's `forward_last_logits`
/// override also uses it to slice the hidden state before its LM head.
///
/// Used by: the `forward_last_logits*` defaults, Llama3Model::last_logits,
/// Gemma4Model::logits_at
pub fn logits_at_position(logits: &MlxArray, pos: usize) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(logits);
    let batch = shape[0];
    let vocab = shape[2];
    // Slice [batch, pos:pos+1, vocab]  →  shape [batch, 1, vocab].
    ffi::slice(logits, &[0, pos as i32, 0], &[batch, pos as i32 + 1, vocab])
}

pub use crate::prefill_plan::{DEFAULT_PREFILL_CHUNK, prefill_chunk_len};

/// Per-layer KV cache modes for `n_layers` caches under the nominal `mode`,
/// with the Boundary-V upgrade (`MLXCEL_KV_BOUNDARY_V_LAYERS`) applied.
///
/// Used by: the engine's raw-completion client (`engine::DirectEngine`)
pub(crate) fn resolve_kv_cache_layer_modes(mode: KVCacheMode, n_layers: usize) -> Vec<KVCacheMode> {
    let requested = crate::cache::turbo::boundary_v_layers_from_env();
    crate::cache::turbo::resolve_layer_modes(mode, n_layers, requested)
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

    /// Inject a resolved per-layer KV cache mode table for model-owned caches.
    ///
    /// Most dense models keep caches in the generator-provided slice and use
    /// the default no-op. Hybrid / sliding / VLM families that own attention
    /// caches internally override this and build those caches from the same
    /// resolved table used for dense external caches.
    fn set_kv_cache_layer_modes(&self, _modes: Vec<KVCacheMode>) {}

    /// Return the currently injected model-owned KV cache mode table, if any.
    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        None
    }

    /// Get the number of layers
    fn num_layers(&self) -> usize;

    /// Get the EOS token IDs for this model
    fn eos_token_ids(&self) -> Vec<i32>;

    /// Token ids this model must never emit as generated text output.
    ///
    /// Multimodal models reserve placeholder ids (audio / image / video span
    /// markers, e.g. Gemma 4 Unified's `audio_token_id`, `image_token_id`,
    /// `boi_token_id`, ...) purely for INPUT alignment. If one becomes the
    /// argmax at a near-tie decode step it corrupts the text stream
    /// (issue #350). Generation paths mask these to `f32::NEG_INFINITY` in
    /// the per-step [`crate::sampling::TokenBiasMap`] so they can never be
    /// sampled.
    ///
    /// The default is empty: non-multimodal models have nothing to suppress
    /// and pay zero cost (the bias map stays empty and `apply_token_bias`
    /// short-circuits). Only families with reserved output-illegal ids
    /// override this, and they must return ONLY those placeholder ids, never
    /// real EOS or normal text ids.
    ///
    /// Used by: CLI `generate` (`run_generation_mode`) and the server batch
    /// scheduler, which merge the returned ids into the effective
    /// `TokenBiasMap` (via [`crate::sampling::TokenBiasMap::suppress_tokens`])
    /// at generator / scheduler construction.
    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    /// Whether this model supports a cache-level chunked prefill: feeding the
    /// prompt through several consecutive multi-token `forward` calls that
    /// continue from the KV caches, instead of one single-pass call.
    ///
    /// Defaults to true: continuing a multi-token forward from cache state is
    /// the same contract the decode loop, multi-token verify, and the server
    /// scheduler's `prefill_chunk_size` path already rely on. Override to
    /// false for models that stash one-shot prompt state on the model which
    /// only the FIRST forward consumes (e.g. multimodal prefills that `take()`
    /// per-layer inputs or a prompt-shaped attention-mask captured at
    /// `prepare_prompt` time), mirroring mlx-vlm's `chunked_prefill_policy`
    /// opt-out (issue #674).
    fn supports_chunked_prefill(&self) -> bool {
        true
    }

    /// Forward pass for a single-sequence prefill whose caller only needs the
    /// logits of one position (`last_pos`, 0-based within this call's
    /// sequence). Returns `[batch, 1, vocab]`.
    ///
    /// The default computes the full `[batch, seq_len, vocab]` logits via
    /// [`Self::forward`] and slices out `last_pos`, which is
    /// behavior-identical to what the prefill call sites previously did
    /// inline. Models with a large vocabulary should override this to project
    /// only the `last_pos` hidden row through the LM head: for a 262k-vocab
    /// gemma-4 at a 32k-token prefill, the full logits tensor is ~17 GiB in
    /// f16 plus a same-size `final_logit_softcapping` copy, none of which is
    /// needed to sample the first generated token (issue #672).
    ///
    /// Used by: the single-sequence prefill in `generate_streaming` and
    /// `generate_with_stats`. Verify/speculative/logprobs paths keep calling
    /// [`Self::forward`] for full per-position logits.
    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let logits = self.forward(input_ids, caches, mask);
        logits_at_position(&logits, last_pos)
    }

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

    /// Hand out the target's final RMSNorm to a native MTP drafter.
    fn final_norm_module(&self) -> Option<crate::layers::RMSNorm> {
        None
    }

    /// Called once after prefill completes and before decode starts.
    /// Used by models that need to adjust internal state between phases.
    fn after_prefill(&self) {}

    /// Drop the trailing `excess` pad positions a padded prefill piece wrote
    /// into this model's own state (issues #1755, #2170).
    ///
    /// `seq` selects the state: `None` is the fallback single-sequence slot
    /// the CLI generate paths and the speculative verify use, `Some(seq_id)`
    /// is the per-sequence state of a scheduler sequence. The batch scheduler
    /// calls this after every padded prefill piece whenever the model's own
    /// [`Self::sequence_state_layout`] is
    /// [`crate::cache::SequenceStateBackend::ModelOwned`], because the
    /// `CachePool` entry of such a sequence holds no per-layer `KVCache` for
    /// its own trim to reach. On return the addressed state must be what an
    /// unpadded prefill of the same tokens would have left: every layer's
    /// `offset` equal to the real token count, and no pad K/V left anywhere a
    /// later step can read. Recurrent state that absorbed the pad positions
    /// cannot be rewound; the families that keep such state decline padding
    /// through [`Self::supports_padded_prefill`] and reset it here.
    ///
    /// The default keeps no model-owned state: the fallback slot has nothing
    /// to trim, a non-batching model's internal state is the one sequence the
    /// scheduler runs (nothing to trim either), and a batching model keeps one
    /// state per `SequenceId` the default cannot address, so `Some` returns
    /// `Err` there; a model-owned family that keeps
    /// [`Self::supports_padded_prefill`] `true` must override this. The
    /// scheduler aborts the request on `Err` rather than decode from a
    /// desynchronized offset; the CLI logs it.
    ///
    /// Not a decode rewind: [`Self::rewind_decode_appends`] unwinds the
    /// scheduler's speculative single-token appends exactly and stays separate.
    ///
    /// Used by: `Engine::prefill` / `Engine::score` (a padded plan piece's
    /// `trim_excess`), the speculative verify padding, and the server
    /// scheduler's `trim_padded_prefill`.
    fn trim_state(&self, seq: Option<SequenceId>, excess: i32) -> Result<(), String> {
        match seq {
            Some(seq_id) if self.supports_batching() => Err(format!(
                "model-owned sequence {seq_id} cannot drop {excess} pad positions: the model \
                 does not implement trim_state"
            )),
            _ => Ok(()),
        }
    }

    /// Whether [`Self::rewind_decode_appends`] can unwind the speculative
    /// single-token appends of the batch scheduler's decode lookahead from
    /// this model's own per-sequence state (issue #2159).
    ///
    /// The scheduler pipelines decode for a family whose
    /// [`Self::sequence_state_layout`] is model-owned only when this is
    /// `true`; every other model-owned family stays on synchronous decode.
    /// `true` promises that a rewind of up to
    /// [`crate::cache::DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS`] appends is
    /// exact for every layer, including a sliding window that has wrapped.
    ///
    /// Used by: server batch scheduler `lookahead_params`.
    fn supports_decode_lookahead_rewind(&self) -> bool {
        false
    }

    /// Unwind the last `n` single-token decode appends from the model-owned
    /// state of scheduler sequence `seq_id`, leaving every layer as it was
    /// before them (issue #2159).
    ///
    /// Called on every decode lookahead teardown for a model that reports
    /// [`Self::supports_decode_lookahead_rewind`]. `Err` means the state may
    /// no longer match the sequence's tokens; the scheduler fails the request
    /// rather than decode or donate from it. The default refuses.
    ///
    /// Used by: server batch scheduler `apply_lookahead_trim`.
    fn rewind_decode_appends(&self, seq_id: SequenceId, n: i32) -> Result<(), String> {
        Err(format!(
            "model-owned sequence {seq_id} cannot rewind {n} decode appends: the model does \
             not implement rewind_decode_appends"
        ))
    }

    /// Reset model-owned fallback runtime state before a fresh single-row
    /// run that carries no `SequenceId`.
    ///
    /// Most models store all request-local state in the `KVCache` slice the
    /// engine's pool owns, so the default is a no-op. Models that keep a
    /// fallback cache slot on `&self` (for a bare `forward` without a
    /// `SequenceId`, as some tests and benchmarks run) override this to clear
    /// that slot without touching the per-sequence maps the engine resets
    /// through [`Self::prepare_sequence_state`] and
    /// [`Self::release_sequence_state_by_id`] on open and close (#2176).
    fn reset_runtime_state(&self) {}

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

    /// Which RoPE frequency table a prompt of `total_prompt_len` tokens will be
    /// rotated with, when this model chooses one from the sequence length.
    ///
    /// `None`, the default, means every position uses the same table, so any
    /// cached prefix of this model is reusable by any request. A model that
    /// switches tables returns a small opaque tag: two requests that get the
    /// same tag agree on every position's rotation, and two that do not must
    /// never share KV.
    ///
    /// Phi-3 / Phi-4 LongRoPE is the case this exists for. It rotates with
    /// `short_factor` while the sequence fits in
    /// `original_max_position_embeddings` and with `long_factor` above it, so a
    /// prefix that one request encoded under the short table is not valid input
    /// for a longer request that will read it under the long table, and the
    /// reverse holds too. Restoring across that boundary leaves one KV cache
    /// holding keys built from two tables, which reads as fluent-looking
    /// repetition rather than as an error.
    ///
    /// The tag is an identity, not an ordering: callers may only compare tags
    /// for equality. It is derived from the length of the whole prompt, which is
    /// what selects the table (see [`crate::prefill_span`]), so a caller must
    /// pass the request's full prompt length and not the length of the prefix it
    /// is trying to reuse.
    ///
    /// Used by: the server prompt cache, which folds the tag into its bucket
    /// identity so a lookup only ever matches entries stored under the same tag.
    fn rope_table_regime(&self, _total_prompt_len: usize) -> Option<u8> {
        None
    }

    /// Whether this model can donate and restore exact-prefix model-owned
    /// state snapshots for cross-request prompt-cache reuse.
    fn supports_snapshot_reuse(&self) -> bool {
        false
    }

    /// Capture an exact-prefix snapshot for a scheduler-owned sequence.
    ///
    /// Models that return `true` from [`Self::supports_snapshot_reuse`] should
    /// override this and return a full copy of the recurrent/model-owned state
    /// for `seq_id`. The default keeps all existing families on the legacy
    /// dense/paged KV-cache donation path.
    fn snapshot_sequence_state(
        &self,
        _seq_id: SequenceId,
        _token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        None
    }

    /// Restore a previously captured exact-prefix snapshot into `seq_id`.
    fn restore_sequence_state(
        &self,
        _seq_id: SequenceId,
        _snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        Err("model does not support exact-prefix state snapshots".to_string())
    }

    /// Whether `snapshot` can be restored covering only its first
    /// `target_len` tokens instead of all of them (issue #1145).
    ///
    /// This is the model's own answer, not something the prompt-cache store
    /// may infer from a family name: only the model knows how its per-layer
    /// state is laid out and whether dropping a tail is mechanically sound.
    /// The default is `false`, so a family opts in explicitly. That default
    /// is the safe one for the recurrent families (GatedDeltaNet, Mamba and
    /// the SSM hybrids), whose state is a fixed-size summary of every token
    /// consumed and cannot be rewound to an earlier boundary at all.
    ///
    /// Implementors must answer for the state actually stored in `snapshot`,
    /// per layer, at this specific `target_len`. A blanket `true` would let
    /// the caller install a corrupt cache.
    fn snapshot_truncatable_to(&self, _snapshot: &ModelStateSnapshot, _target_len: usize) -> bool {
        false
    }

    /// Restore `snapshot` into `seq_id` covering only its first `target_len`
    /// tokens.
    ///
    /// Callers must have had [`Self::snapshot_truncatable_to`] agree for the
    /// same snapshot and length first. Implementors should still re-check
    /// rather than trust the caller, and return `Err` instead of installing a
    /// partially truncated state.
    fn restore_sequence_state_truncated(
        &self,
        _seq_id: SequenceId,
        _snapshot: &ModelStateSnapshot,
        _target_len: usize,
    ) -> Result<(), String> {
        Err("model does not support truncated state snapshot restore".to_string())
    }

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
    /// Pure transformer models whose K/V lives in the external `KVCache` slice
    /// return `true` (the default) because padding tokens only reach that
    /// slice, which the caller trims afterwards.
    ///
    /// That reasoning does not hold for a family whose
    /// [`Self::sequence_state_layout`] is model-owned: the scheduler's
    /// `CachePool` entry for such a sequence holds no `KVCache`, so its trim
    /// reaches nothing and the pad positions stay in the model's own caches.
    /// A model-owned family may answer `true` only if it also implements
    /// [`Self::trim_state`] (server with `Some`, CLI fallback with `None`)
    /// (CLI) so that both rewind its own state; otherwise it must answer
    /// `false` (issue #1755).
    ///
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

    /// Sequence-aware counterpart of [`Self::forward_last_logits`].
    ///
    /// Server prefills need model-owned state keyed by `seq_id`, but only the
    /// last real row is sampled. The default preserves the prior behavior by
    /// forwarding the whole sequence and slicing; large-vocabulary recurrent
    /// models can override it to slice hidden states before the LM head.
    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let logits = self.forward_with_sequence_id(input_ids, seq_id, caches, mask);
        logits_at_position(&logits, last_pos)
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

    /// Embedding-prefill counterpart of
    /// [`Self::forward_last_logits_with_sequence_id`].
    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let logits = self.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
        );
        logits_at_position(&logits, last_pos)
    }

    /// Synchronize model-owned sequence storage into the runtime backend state.
    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        cache_pool.sync_paged_state_with_dense(seq_id)
    }

    /// Batched forward with the scheduler's sequence identities.
    ///
    /// `seq_ids[i]` names the sequence whose per-model state row `i` uses; a
    /// family whose per-sequence state lives inside the model (Gemma 3,
    /// Llama 4, Qwen3.5, Muse Glimmer, the VLM wrappers) overrides this so
    /// each row resolves its own state. The default ignores the ids and runs
    /// `forward_batched`. Storage is a property of each row's cache
    /// (`KVCache::attend`, ADR 0008), so there is no per-step storage hint:
    /// the `DecodeBatchContext` argument this entry used to take was removed
    /// in #2172 once no family read it.
    ///
    /// Used by: `Engine::step` and `Engine::prefill_cohort` for every batch of
    /// more than one row.
    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let _ = seq_ids;
        self.forward_batched(input_ids, batch_caches, mask)
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

/// The `dry_penalty_last_n` sentinel meaning "scan the entire token
/// history". b10621 reconciliation (#1436): `0` now DISABLES the DRY stage
/// (upstream's sentinel), so the old mlxcel "0 = full history" form moved to
/// this explicit value. The CLI and OpenAI-shaped surfaces spell it `-1`.
pub const DRY_FULL_HISTORY: usize = usize::MAX;

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
    /// DRY lookback window. b10621 semantics (#1436): `0` DISABLES the DRY
    /// stage entirely, [`DRY_FULL_HISTORY`] scans the whole history (the
    /// explicit successor of the pre-#1436 `0`), and any other value is the
    /// number of recent tokens scanned.
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
    /// N-gram tail repetition / loop detection. Disabled by default (all zero),
    /// a zero-overhead no-op that preserves the bit-exact baseline for every
    /// model that does not opt in (mirrors `token_bias` / `stop_token_ids`).
    /// When enabled, the decode loops end generation early once the raw
    /// generated stream collapses into a short repeated pattern.
    pub loop_detection: LoopDetectionConfig,
    /// XTC (Exclude Top Choices) per-step probability (0.0 = disabled, the
    /// default). With this probability, each sampling step applies the XTC
    /// filter (see [`crate::sampling::apply_xtc_filter`]); otherwise the step
    /// samples normally. `0.0` is a zero-overhead no-op that preserves the
    /// bit-exact baseline: the gate is skipped entirely rather than drawing
    /// (and thereby consuming) a random sample.
    pub xtc_probability: f32,
    /// XTC probability threshold: among the tokens whose probability exceeds
    /// this value, the filter removes all but the single least-probable one.
    /// Valid range `0.0..=0.5` (enforced at the request layer). Unused while
    /// `xtc_probability == 0.0`.
    pub xtc_threshold: f32,
    /// Token ids the XTC filter must never remove, even when selected by its
    /// removal rule: the tokenizer's newline token id(s) plus every id in the
    /// merged end-of-sequence set. Resolved once per request at enqueue time
    /// (empty by default, and unused while `xtc_probability == 0.0`).
    pub xtc_special_token_ids: Vec<i32>,
    /// Top-n-sigma logit filter (`0.0` = disabled, the default). Keeps only
    /// the tokens whose raw logit lies within `top_n_sigma` standard
    /// deviations of the row maximum (`logit >= max - n * std`, statistics
    /// over the finite entries of the vocabulary row) and masks the rest to
    /// `-inf`. Thresholds logits rather than probabilities, so it is
    /// invariant to temperature and to any per-row additive shift. `0.0` is a
    /// zero-overhead no-op that preserves the bit-exact baseline; the greedy
    /// path (`temperature == 0.0 || top_k == 1`) skips the filter entirely.
    /// See [`crate::sampling::apply_row_filters`].
    pub top_n_sigma: f32,
    /// Hyperparameter-free p-less truncation (`false` = disabled, the
    /// default). Keeps every token whose probability is at least the
    /// collision probability `L = sum_v p(v)^2` of the temperature-scaled
    /// distribution and masks the rest to `-inf`. The argmax always
    /// survives (`L <= p_max`). Runs after `top_n_sigma` and before
    /// `typical_p`; the greedy path skips it. See
    /// [`crate::sampling::apply_row_filters`].
    pub p_less: bool,
    /// Locally typical sampling (`1.0` = disabled, the default). Keeps the
    /// tokens whose surprisal `-log p` is closest to the row entropy,
    /// accumulating probability mass in that typicality order until it
    /// reaches `typical_p`, and masks the rest to `-inf`. Unlike top-p the
    /// argmax can be dropped, which is why the greedy path
    /// (`temperature == 0.0 || top_k == 1`) skips the filter entirely.
    /// Valid enabled range `(0.0, 1.0)`; `1.0` is a zero-overhead no-op that
    /// preserves the bit-exact baseline. See
    /// [`crate::sampling::apply_row_filters`].
    pub typical_p: f32,
    /// Repetition / frequency / presence penalty lookback window, b10621's
    /// `repeat_last_n` (#1436). `-1` (the default) scans the entire token
    /// history, preserving mlxcel's pre-#1436 behavior; `0` DISABLES the
    /// three history penalties entirely (they become inert whatever their
    /// values, and the config stays fused-batch eligible); `N > 0` penalizes
    /// over only the last `N` generated tokens, matching b10621's windowed
    /// penalty ring buffer. DRY has its own window (`dry_penalty_last_n`).
    pub penalty_last_n: i32,
    /// Mirostat mode, b10621's `--mirostat` (#1485): `0` disables (the
    /// default), `1` selects Mirostat, `2` selects Mirostat 2.0. A non-zero
    /// value REPLACES the whole sampler chain, exactly as b10621's
    /// `common_sampler_init` does: penalties, DRY, and every truncation
    /// filter (top-k / top-p / min-p / typical / top-n-sigma / XTC) are
    /// skipped, and the step becomes token-bias -> temperature ->
    /// mirostat-select. Requires per-sequence state (the surprise target
    /// `mu`), so a mirostat config never joins the fused batch path.
    pub mirostat: i32,
    /// Mirostat target entropy tau (b10621 default `5.0`). Unused while
    /// `mirostat == 0`.
    pub mirostat_tau: f32,
    /// Mirostat learning rate eta (b10621 default `0.1`). Unused while
    /// `mirostat == 0`.
    pub mirostat_eta: f32,
    /// Dynamic-temperature half range, b10621's `--dynatemp-range` (#1485):
    /// `0.0` disables (the default). When positive, the temperature stage
    /// scales by an entropy-dependent temperature in
    /// `[max(0, temperature - range), temperature + range]` instead of the
    /// fixed `temperature`, computed per step over the candidates that
    /// survive the preceding filters (b10621's `llama_sampler_temp_ext`).
    /// Data-dependent, so the row runs the Rust extended chain rather than
    /// the fused C++ chain. NOTE: a positive range makes even a
    /// `temperature == 0.0` config stochastic, exactly as it does upstream;
    /// see [`SamplingConfig::is_greedy_path`].
    pub dynatemp_range: f32,
    /// Dynamic-temperature exponent (b10621 default `1.0`): how the
    /// normalized candidate entropy maps into the range. Unused while
    /// `dynatemp_range <= 0.0`.
    pub dynatemp_exponent: f32,
    /// Adaptive-p target probability, b10621's `--adaptive-target` (#1485):
    /// negative disables (the default `-1.0`). Enabled ONLY when the request
    /// also names the `adaptive_p` stage in its sampler list (b10621
    /// activates the sampler solely through `params.samplers`), which the
    /// resolution layer expresses by leaving this field negative otherwise.
    /// When enabled the final draw is replaced by b10621's adaptive-p
    /// transform over the post-chain distribution, with a per-sequence EMA
    /// state, so such a config never joins the fused batch path.
    pub adaptive_target: f32,
    /// Adaptive-p EMA decay (b10621 default `0.90`, valid `0.0..=0.99`);
    /// history approximates `1 / (1 - decay)` tokens. Unused while
    /// `adaptive_target < 0.0`.
    pub adaptive_decay: f32,
    /// b10621's `min_keep` (#1485): when `>= 2`, the top-p / min-p /
    /// typical-p truncations each keep at least this many candidates (in
    /// their own keep order), and XTC skips its removal entirely when it
    /// would leave fewer survivors. `0` and `1` are no-ops (every filter
    /// already keeps at least one token). A `min_keep >= 2` row runs the
    /// Rust extended chain, because the C++ fused filters have no floor
    /// parameter.
    pub min_keep: usize,
    /// DRY restart-sequence heads derived from b10621's breaker STRINGS
    /// (#1485): maps a head token id to the list of tail token sequences
    /// that, together with the head, spell out a breaker string
    /// (`get_overlapping_token_sequences` in upstream
    /// `src/llama-sampler.cpp`). A head with an empty tail (the common case:
    /// the token's decoded text contains the breaker string outright) breaks
    /// DRY matching at its position on its own; a head with a non-empty tail
    /// breaks only when the following window tokens spell the tail. Derived
    /// once per breaker set (server startup for the CLI default set, enqueue
    /// time for per-request sets) because the derivation needs the
    /// vocabulary. The legacy `dry_sequence_breakers` id list keeps its
    /// exact-token semantics alongside for the mlxcel-native surfaces.
    pub dry_breaker_heads: std::sync::Arc<std::collections::HashMap<i32, Vec<Vec<i32>>>>,
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
            dry_penalty_last_n: DRY_FULL_HISTORY,
            dry_sequence_breakers: Vec::new(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            token_bias: TokenBiasMap::default(),
            loop_detection: LoopDetectionConfig::default(),
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            xtc_special_token_ids: Vec::new(),
            top_n_sigma: 0.0,
            p_less: false,
            typical_p: 1.0,
            penalty_last_n: -1,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            dynatemp_range: 0.0,
            dynatemp_exponent: 1.0,
            adaptive_target: -1.0,
            adaptive_decay: 0.9,
            min_keep: 0,
            dry_breaker_heads: std::sync::Arc::new(std::collections::HashMap::new()),
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
            dry_penalty_last_n: DRY_FULL_HISTORY,
            dry_sequence_breakers: Vec::new(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            token_bias: TokenBiasMap::default(),
            loop_detection: LoopDetectionConfig::default(),
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            xtc_special_token_ids: Vec::new(),
            top_n_sigma: 0.0,
            p_less: false,
            typical_p: 1.0,
            penalty_last_n: -1,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            dynatemp_range: 0.0,
            dynatemp_exponent: 1.0,
            adaptive_target: -1.0,
            adaptive_decay: 0.9,
            min_keep: 0,
            dry_breaker_heads: std::sync::Arc::new(std::collections::HashMap::new()),
        }
    }

    /// Create config with specific temperature
    pub fn with_temperature(temp: f32) -> Self {
        Self {
            temperature: temp,
            ..Default::default()
        }
    }

    /// The top-n-sigma value the sampler will actually apply: `0.0`
    /// (disabled) when the config is greedy (`temperature == 0.0 ||
    /// top_k == 1`), or when the raw field is non-positive or non-finite
    /// (which covers the b10621 `-1.0` "disabled" sentinel). Batch-uniformity
    /// gates ([`crate::sampling::FusedSampleParams::from_config`], the
    /// speculative-window `sampling_config_eq`) compare THIS value rather
    /// than the raw field, so two rows whose sampled outputs are necessarily
    /// identical (e.g. greedy rows differing only in an inert `top_n_sigma`)
    /// are not needlessly split off the fused batch, pipelined-lookahead, or
    /// batched-speculative paths.
    pub fn effective_top_n_sigma(&self) -> f32 {
        if self.is_greedy_path() || self.effective_mirostat() != 0 {
            return 0.0;
        }
        if self.top_n_sigma > 0.0 && self.top_n_sigma.is_finite() {
            self.top_n_sigma
        } else {
            0.0
        }
    }

    /// Whether p-less truncation will actually run: `false` when the config
    /// is greedy or mirostat replaces the chain. Batch-uniformity gates
    /// compare THIS value so inert differences do not split a batch.
    pub fn effective_p_less(&self) -> bool {
        self.p_less && !self.is_greedy_path() && self.effective_mirostat() == 0
    }

    /// The typical-p value the sampler will actually apply: `1.0` (disabled)
    /// when the config is greedy (`temperature == 0.0 || top_k == 1`), or
    /// when the raw field is outside the enabled range `(0.0, 1.0)` or
    /// non-finite. Batch-uniformity gates compare THIS value, mirroring
    /// [`SamplingConfig::effective_top_n_sigma`], so rows whose sampled
    /// outputs are necessarily identical stay on the shared fused paths.
    pub fn effective_typical_p(&self) -> f32 {
        if self.is_greedy_path() || self.effective_mirostat() != 0 {
            return 1.0;
        }
        if self.typical_p.is_finite() && self.typical_p > 0.0 && self.typical_p < 1.0 {
            self.typical_p
        } else {
            1.0
        }
    }

    /// Whether the sampler resolves to the deterministic argmax path.
    ///
    /// This is the pre-#1485 `temperature == 0.0 || top_k == 1` check, with
    /// one refinement from b10621's `llama_sampler_temp_ext`: a positive
    /// `dynatemp_range` re-widens a `temperature == 0.0` config into a
    /// stochastic one (the dynamic temperature lands in
    /// `[max(0, temp - range), temp + range]`), so only `top_k == 1` stays
    /// unconditionally greedy. Every greedy fold (`effective_*`, fused-batch
    /// gates, filter skips) routes through this so the definition cannot
    /// drift between call sites.
    pub fn is_greedy_path(&self) -> bool {
        self.top_k == 1
            || (self.temperature == 0.0
                && !(self.dynatemp_range > 0.0 && self.dynatemp_range.is_finite()))
    }

    /// The mirostat mode the sampler will actually run: `1` or `2` when the
    /// raw field selects a version, else `0` (disabled). Values outside
    /// `{0, 1, 2}` are refused at the request/CLI layers (where b10621 would
    /// abort the process in `common_sampler_init`); the fold here is the
    /// defense-in-depth mirror so an unvalidated caller degrades to the
    /// disabled chain instead of an undefined mode.
    pub fn effective_mirostat(&self) -> i32 {
        if self.mirostat == 1 || self.mirostat == 2 {
            self.mirostat
        } else {
            0
        }
    }

    /// The dynamic-temperature half range the sampler will actually apply:
    /// `0.0` (disabled) unless the raw field is positive and finite. A
    /// `top_k == 1` config folds to disabled (one candidate leaves no
    /// entropy to map), but `temperature == 0.0` does NOT: b10621's
    /// `temp_ext` applies the dynamic range regardless of the base
    /// temperature. Mirostat replaces the whole chain, so it folds this
    /// stage off too.
    pub fn effective_dynatemp_range(&self) -> f32 {
        if self.top_k == 1 || self.effective_mirostat() != 0 {
            return 0.0;
        }
        if self.dynatemp_range > 0.0 && self.dynatemp_range.is_finite() {
            self.dynatemp_range
        } else {
            0.0
        }
    }

    /// The `min_keep` floor the sampler will actually apply: `0` (inert)
    /// unless the raw field is `>= 2` on a non-greedy, non-mirostat config.
    /// `0` and `1` are inert because every truncation filter already keeps
    /// at least one candidate, and the greedy path keeps exactly the argmax
    /// whatever the floor, so folding them keeps such rows on the fused
    /// batch path.
    pub fn effective_min_keep(&self) -> usize {
        if self.is_greedy_path() || self.effective_mirostat() != 0 || self.min_keep < 2 {
            0
        } else {
            self.min_keep
        }
    }

    /// The adaptive-p target the sampler will actually use: `-1.0`
    /// (disabled) unless the raw field is a finite non-negative value on a
    /// non-greedy, non-mirostat config (b10621 clamps the enabled value into
    /// `[0, 1]` at apply time; the resolution layers enforce the domain).
    /// Greedy folds to disabled because a single-candidate draw is the same
    /// token with or without the transform, which keeps such rows on the
    /// fused batch path.
    pub fn effective_adaptive_target(&self) -> f32 {
        if self.is_greedy_path() || self.effective_mirostat() != 0 {
            return -1.0;
        }
        if self.adaptive_target.is_finite() && self.adaptive_target >= 0.0 {
            self.adaptive_target.min(1.0)
        } else {
            -1.0
        }
    }

    /// Whether this config runs the Rust extended sampler chain
    /// ([`crate::sampling`]'s `apply_extended_chain`) instead of the fused
    /// C++ chain: any of dynamic temperature, a `min_keep` floor, or
    /// adaptive-p requires arithmetic the C++ chain has no parameters for.
    /// Mirostat is NOT part of this: it bypasses the chain entirely on its
    /// own path.
    pub fn needs_extended_chain(&self) -> bool {
        self.effective_mirostat() == 0
            && (self.effective_dynatemp_range() > 0.0
                || self.effective_min_keep() >= 2
                || self.effective_adaptive_target() >= 0.0)
    }

    /// Whether the sampler carries mutable per-sequence feedback state
    /// across steps: mirostat's surprise target `mu`, or adaptive-p's EMA.
    /// Such a config must be sampled through
    /// [`crate::sampling::sample_token_optimized_with_state`] so the state
    /// persists, and can never join the fused batch path.
    pub fn needs_sampler_feedback_state(&self) -> bool {
        self.effective_mirostat() != 0 || self.effective_adaptive_target() >= 0.0
    }

    /// Check if any penalty-based sampling is enabled.
    ///
    /// A `penalty_last_n` of `0` makes the repetition / frequency / presence
    /// stage inert regardless of the penalty values, and a
    /// `dry_penalty_last_n` of `0` disables DRY the same way (b10621's
    /// sentinels, #1436), so neither counts as needing history then. This is
    /// what lets a request that zeroes its windows stay on the fused batch
    /// path.
    pub fn needs_token_history(&self) -> bool {
        // Mirostat replaces the whole chain (b10621's common_sampler_init):
        // penalties and DRY never run, so their configuration cannot make
        // the row need history.
        if self.effective_mirostat() != 0 {
            return false;
        }
        let penalties_active = self.penalty_last_n != 0
            && (self.repetition_penalty != 1.0
                || self.frequency_penalty != 0.0
                || self.presence_penalty != 0.0);
        let dry_active = self.dry_multiplier > 0.0 && self.dry_penalty_last_n != 0;
        penalties_active || dry_active
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

/// Named phases of the pre-first-token path, in nanoseconds.
///
/// The decode-loop profiling covers only the decode loop, so before issue
/// #1545 the entire pre-first-token phase was one opaque number: the
/// `Prefill:` line of `--profile`. On a CUDA host that number is not mostly
/// prefill arithmetic. It also carries the lazy materialization of every
/// weight (MLX loads safetensors as unevaluated `Load` arrays, so the host
/// read and the host-to-device copy are charged to the first `eval`), the
/// first load of every JIT module, and the first instantiation of every CUDA
/// graph the forward pass needs.
///
/// The split these fields draw is between work MLX does on the host while the
/// device is idle and the single blocking `eval` that does everything else.
/// `build` is pure graph construction: MLX is lazy, so `forward_last_logits`
/// returns without touching the device. `eval` is where the weights, the
/// modules, the graphs and the kernels all land. A large `build` points at
/// host-side op-recording overhead; a large `eval` points at the device, the
/// PCIe link, or the CUDA graph machinery, and separating those two is what
/// `docs/benchmark_results/volta-ttft-fixed-cost-2026-09-01.md` needed.
///
/// `setup_ns` sits *outside* the reported prefill time, because the generator
/// resets its caches before the prefill clock starts. Every other field sits
/// inside it, and [`TtftPhases::format_line`] prints the residual so a reader
/// can see that they account for it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TtftPhases {
    /// Generator reset, KV-cache construction, KV-mode application and stream
    /// install. Measured outside the reported prefill time.
    pub setup_ns: u128,
    /// Lazy construction of the prefill forward graph. Host only: no kernel
    /// runs and no weight is materialized here.
    pub build_ns: u128,
    /// Lazy construction of the sampler graph for the first token.
    pub sample_ns: u128,
    /// The one blocking `eval` that materializes the first token, and with it
    /// every weight, JIT module and CUDA graph the forward pass reaches.
    pub eval_ns: u128,
    /// Fixed post-eval work before the decode loop starts.
    pub post_ns: u128,
}

impl TtftPhases {
    /// Sum of the phases that sit inside the reported prefill time.
    ///
    /// `setup_ns` is excluded: it is measured before the prefill clock starts.
    #[must_use]
    pub fn in_prefill_ns(&self) -> u128 {
        self.build_ns + self.sample_ns + self.eval_ns + self.post_ns
    }

    /// Sum of every phase, including the setup that precedes prefill.
    #[must_use]
    pub fn total_ns(&self) -> u128 {
        self.setup_ns + self.in_prefill_ns()
    }

    /// One `[TTFT]` diagnostic line.
    ///
    /// `prefill_ns` is the wall clock `--profile` reports as `Prefill:`. The
    /// trailing `residual` is that number minus [`Self::in_prefill_ns`], so a
    /// reader can confirm the named phases account for the total instead of
    /// taking it on trust. It is a saturating subtraction: the phases are
    /// nested inside the prefill clock, so a negative residual would mean the
    /// timers disagree, and reporting `0.00` there is preferable to wrapping.
    #[must_use]
    pub fn format_line(&self, prompt_tokens: usize, prefill_ns: u128) -> String {
        let ms = |ns: u128| ns as f64 / 1e6;
        format!(
            "[TTFT] prompt={prompt_tokens} tok setup={:.2}ms build={:.2}ms sample={:.2}ms eval={:.2}ms post={:.2}ms | prefill={:.2}ms residual={:.2}ms",
            ms(self.setup_ns),
            ms(self.build_ns),
            ms(self.sample_ns),
            ms(self.eval_ns),
            ms(self.post_ns),
            ms(prefill_ns),
            ms(prefill_ns.saturating_sub(self.in_prefill_ns())),
        )
    }
}

/// Whether `MLXCEL_PROFILE_TTFT` asked for the pre-first-token breakdown.
///
/// Read once per process: the phases are timed on the first-token path, which
/// runs once per generation, but a server calls it per request and an
/// environment read per request is pointless.
#[must_use]
pub fn ttft_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("MLXCEL_PROFILE_TTFT").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::KVCache;
    use crate::prefill_plan::{PrefillCaps, PrefillPlan};

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

    /// Multi-position stub: logits value encodes the sequence position, so a
    /// slicing bug in `forward_last_logits` is directly visible.
    struct SeqStubModel;

    impl LanguageModel for SeqStubModel {
        fn forward(
            &self,
            input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            // Input [1, L] -> logits [1, L, 4] with row i filled with i as f32.
            let seq_len = ffi::array_shape(input_ids)[1] as usize;
            let mut logits = Vec::with_capacity(seq_len * 4);
            for i in 0..seq_len {
                logits.extend_from_slice(&[i as f32; 4]);
            }
            ffi::from_slice_f32(&logits, &[1, seq_len as i32, 4])
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

    /// Stateful stub: accumulates every token it has seen (its "KV cache")
    /// and returns logits whose value is the running total, so a chunked
    /// prefill that dropped or re-fed tokens would produce a different final
    /// value than a single pass.
    /// A model that records what [`crate::prefill_span`] reported inside each
    /// forward, so a driver's announcement can be observed from where a real
    /// model would read it.
    /// Every chunk of a chunked prefill sees the whole prompt's length, and the
    /// announcement is gone once the prefill returns.
    ///
    /// A model that selects a RoPE frequency table from the prompt length reads
    /// exactly this (Phi-3 / Phi-4 LongRoPE, #1358). Per chunk it would rotate
    /// the head of a threshold-crossing prompt with one table and the tail with
    /// another, into one KV cache.
    /// The trait default opts in; an overriding model opts out.
    #[test]
    fn supports_chunked_prefill_default_and_override() {
        struct OptOutModel;
        impl LanguageModel for OptOutModel {
            fn forward(
                &self,
                _input_ids: &MlxArray,
                _caches: &mut [KVCache],
                _mask: Option<&MlxArray>,
            ) -> UniquePtr<MlxArray> {
                ffi::from_slice_f32(&[0.0; 4], &[1, 1, 4])
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
            fn supports_chunked_prefill(&self) -> bool {
                false
            }
        }

        assert!(StubModel.supports_chunked_prefill());
        assert!(!OptOutModel.supports_chunked_prefill());
    }

    /// Chunked prefill must feed every prompt token exactly once, in order,
    /// and return the same final-position logits as a single pass, for every
    /// partition the plan can produce (chunks, a history boundary, both).
    /// The default `forward_last_logits` must equal forward + slice at the
    /// requested position, with shape `[batch, 1, vocab]`.
    #[test]
    fn forward_last_logits_default_matches_forward_slice() {
        let model = SeqStubModel;
        let input = ffi::from_slice_i32(&[5, 6, 7, 8], &[1, 4]);

        for pos in [0usize, 2, 3] {
            let mut caches = model.make_caches();
            let last = model.forward_last_logits(&input, &mut caches, None, pos);
            assert_eq!(ffi::array_shape(&last).as_slice(), &[1, 1, 4]);
            let first = ffi::slice(&last, &[0, 0, 0], &[1, 1, 1]);
            ffi::eval(&first);
            assert_eq!(ffi::item_f32(&first), pos as f32, "wrong row at pos {pos}");
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

    /// A padded piece's input for a model that opted into maskless padded
    /// prefill carries no mask array; the default keeps it.
    fn padded_piece_mask<M: LanguageModel>(model: &M) -> bool {
        let tokens = [1, 2, 3];
        let caps = PrefillCaps {
            align_prefill: true,
            ..PrefillCaps::for_model(model, true)
        };
        let plan = PrefillPlan::new(tokens.len(), 0, caps);
        let piece = &plan.pieces()[0];
        assert!(piece.is_padded(), "the aligned plan pads a 3-token piece");
        let (_input, mask) = crate::engine::piece_input(&plan, piece, &tokens, 0);
        mask.is_some()
    }

    #[test]
    fn padded_prefill_can_skip_array_mask_for_opted_in_models() {
        assert!(!padded_piece_mask(&MasklessPaddedPrefillModel));
    }

    #[test]
    fn padded_prefill_keeps_array_mask_by_default() {
        assert!(padded_piece_mask(&StubModel));
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

    // -- Cache clearing cadence (backend-aware, #627) --

    /// The decode loops gate the periodic clear through
    /// `memory::should_clear_cache_at`. Verify the contract the loops rely on:
    /// it fires on multiples of the interval, never on token 0, and honors a
    /// longer cadence.
    #[test]
    fn periodic_clear_gate_respects_interval() {
        use crate::memory::should_clear_cache_at;
        assert!(should_clear_cache_at(256, 256));
        assert!(should_clear_cache_at(512, 256));
        assert!(!should_clear_cache_at(0, 256));
        assert!(!should_clear_cache_at(255, 256));
        assert!(should_clear_cache_at(4096, 4096));
        assert!(!should_clear_cache_at(256, 4096));
    }

    /// Interval 0 is the CUDA default: the periodic clear never fires, so the
    /// buffer cache stays resident for CUDA-graph reuse.
    #[test]
    fn periodic_clear_disabled_when_interval_zero() {
        use crate::memory::should_clear_cache_at;
        for n in [0_usize, 1, 256, 512, 4096, 100_000] {
            assert!(!should_clear_cache_at(n, 0));
        }
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

    // -- trim_state default implementation --

    /// The default `LanguageModel::trim_state` keeps no state: the fallback
    /// slot accepts any excess as a no-op, and a scheduler sequence of a
    /// batching model is refused because the default cannot address it.
    #[test]
    fn trim_state_default_is_noop_for_fallback_and_refuses_batched_sequences() {
        let model = StubModel;
        // Should not fail for positive, zero, or negative excess.
        assert_eq!(model.trim_state(None, 8), Ok(()));
        assert_eq!(model.trim_state(None, 0), Ok(()));
        assert_eq!(model.trim_state(None, -1), Ok(()));
        assert!(model.supports_batching());
        let err = model
            .trim_state(Some(SequenceId::from_raw(7)), 8)
            .expect_err("a batching model's per-sequence state is unaddressed by default");
        assert!(err.contains("seq-7"), "{err}");

        // A non-batching model's internal state is the one scheduler
        // sequence, so `Some` is accepted there.
        assert_eq!(
            NonBatchModel.trim_state(Some(SequenceId::from_raw(7)), 8),
            Ok(())
        );
    }

    /// A model that overrides trim_state records each call so we can verify
    /// the generation machinery actually invokes the method.
    struct TrackingTrimModel {
        trim_call_count: std::cell::Cell<usize>,
        last_excess: std::cell::Cell<i32>,
        last_seq: std::cell::Cell<Option<SequenceId>>,
    }

    impl TrackingTrimModel {
        fn new() -> Self {
            Self {
                trim_call_count: std::cell::Cell::new(0),
                last_excess: std::cell::Cell::new(0),
                last_seq: std::cell::Cell::new(None),
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

        fn trim_state(&self, seq: Option<SequenceId>, excess: i32) -> Result<(), String> {
            self.trim_call_count.set(self.trim_call_count.get() + 1);
            self.last_excess.set(excess);
            self.last_seq.set(seq);
            Ok(())
        }
    }

    #[test]
    fn trim_state_override_receives_the_target_and_excess() {
        let model = TrackingTrimModel::new();
        assert_eq!(model.trim_call_count.get(), 0);

        // The CLI's call pattern: fallback slot, excess = padded - actual.
        assert_eq!(model.trim_state(None, 16), Ok(()));
        assert_eq!(model.trim_call_count.get(), 1);
        assert_eq!(model.last_excess.get(), 16);
        assert_eq!(model.last_seq.get(), None);

        // The scheduler's: one sequence.
        let seq = SequenceId::from_raw(3);
        assert_eq!(model.trim_state(Some(seq), 32), Ok(()));
        assert_eq!(model.trim_call_count.get(), 2);
        assert_eq!(model.last_excess.get(), 32);
        assert_eq!(model.last_seq.get(), Some(seq));
    }
}

#[cfg(test)]
#[path = "generate_history_tests.rs"]
mod history_tests;

#[cfg(test)]
#[path = "generate_finish_tests.rs"]
mod finish_tests;
