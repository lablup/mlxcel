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

//! MTP target abstraction.
//!
//! The [`MtpTarget`] trait captures the **superset** of the standard
//! [`crate::generate::LanguageModel`] surface needed by [`super::MtpGenerator`].
//! Specifically, in addition to the LM-trait's `forward` and `embed_tokens`,
//! the round-loop driver needs:
//!
//! 1. A sink-aware verify forward that returns logits AND captures the
//!    target's last-layer hidden + last full/SWA shared K/V slabs — this
//!    is the Gemma 4 `forward_with_speculative_sinks` hook.
//! 2. A per-sequence cache rollback that trims by `block_size - accepted - 1`
//!    with per-row tail-zeroing for batched paths — this is the Gemma 4
//!    `rollback_speculative_cache` hook.
//!
//! Both hooks are Gemma-4-specific. Other model families (Llama, Qwen, …)
//! do not implement them today. Surfacing them through a trait keeps
//! `mlxcel-core` free of a `gemma4`-shaped concrete type while still
//! letting `mlxcel-core` host the round-loop driver and its unit tests.
//!
//! ## Two-phase verify
//!
//! The verify pass is split into two trait methods:
//!
//! - [`MtpTarget::verify_forward`] runs the sink-aware forward and returns
//!   the target tokens plus a captured state opaque to the round-loop
//!   ([`VerifyCaptured`]). The cache is left in its post-forward state at
//!   this point.
//! - [`MtpTarget::verify_finalize`] takes the walk's `accepted` count
//!   plus the captured state, performs the per-row tail-zero rollback
//!   on the cache, slices the captured shared K/V to the post-rollback
//!   length, and returns the next-round inputs as [`MtpVerifyOutput`].
//!
//! This mirrors upstream Python's natural shape: forward → walk →
//! rollback → re-slice → rebind.
//!
//! Tests build a tiny mock implementation that satisfies [`MtpTarget`]
//! without pulling the FFI / real model weights — see `tests.rs`.

use crate::ffi::MlxArray;
use crate::generate::SamplingConfig;
use crate::sampling::{LogprobsConfig, TokenLogprobData};
use cxx::UniquePtr;

use crate::drafter::DrafterError;

/// Captured target state from [`MtpTarget::verify_forward`].
///
/// Opaque to the round-loop: the round-loop only forwards this back to
/// [`MtpTarget::verify_finalize`] alongside the walk's accepted count.
/// The trait implementation is responsible for stashing whatever it needs
/// to perform the rollback (e.g., the captured `hidden_full` and the
/// pre-slice shared K/V slabs, plus any per-sequence cache handle).
///
/// Backed by `Vec<UniquePtr<MlxArray>>` so the concrete Gemma 4 impl can
/// keep MLX-array handles without pulling in `Box<dyn Any>` indirection.
/// The mock target in tests fills the same slot with dummy tensors.
pub struct VerifyCaptured {
    /// Implementation-defined storage. The Gemma 4 impl uses index 0 for
    /// `hidden_full`, indices 1..N for shared K/V slabs in `[k_full,
    /// v_full, k_swa, v_swa]` order.
    pub tensors: Vec<UniquePtr<MlxArray>>,
    /// Implementation-defined scalar state (e.g., pre-rollback
    /// `prompt_cache[0].offset`). The Gemma 4 impl uses `[kv_offset_pre,
    /// bonus_position_pre]`.
    pub scalars: Vec<i32>,
}

// Manual `Debug`: tensor bodies are GPU-resident and not safe to read
// off the dispatch thread.
impl std::fmt::Debug for VerifyCaptured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyCaptured")
            .field("num_tensors", &self.tensors.len())
            .field("scalars", &self.scalars)
            .finish()
    }
}

/// Output of one [`MtpTarget::verify_forward`] call.
///
/// Pairs the target tokens (used for the speculative walk) with the
/// captured state (forwarded to [`MtpTarget::verify_finalize`]).
pub struct VerifyForwardOutput {
    /// Greedy choice (argmax) at each verify position. Length equals
    /// `verify_input.len()` (the verify input was `[bonus, draft_0, …,
    /// draft_{K-2}]` and the forward emits one logit per position).
    /// At temperature 0 these are pure argmax; at temperature > 0 the
    /// trait impl samples from the per-position logits.
    pub target_tokens: Vec<i32>,
    /// Per-position log-probability data, aligned 1:1 with
    /// `target_tokens`. `None` when the caller's [`LogprobsConfig`] is
    /// disabled (zero-overhead path); `Some(vec)` with one entry per
    /// verify position when logprobs are requested. The
    /// round loop forwards the entries for *accepted* positions on to
    /// the burst's `finalize_burst_success` so speculative responses
    /// carry the same `TokenWithLogprobs` payload as the classic path.
    pub target_logprobs: Option<Vec<TokenLogprobData>>,
    /// Opaque captured state for the finalize call.
    pub captured: VerifyCaptured,
}

impl std::fmt::Debug for VerifyForwardOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyForwardOutput")
            .field("target_tokens", &self.target_tokens)
            .field("has_target_logprobs", &self.target_logprobs.is_some())
            .field("captured", &self.captured)
            .finish()
    }
}

/// Output of one [`MtpTarget::verify_finalize`] call (and of
/// [`MtpTarget::seed_verify`]).
///
/// Mirrors the next-round inputs the round-loop hands back to the drafter:
///
/// - `next_hidden`: target's last decoder layer pre-norm hidden at the
///   **accepted** position. The round-loop uses this to seed the
///   drafter's `h_prev` for the next round. Shape `[1, 1, backbone]`
///   in the model's native dtype.
/// - `next_shared_kv`: re-sliced shared K/V slabs for the next round.
///   Layout matches [`crate::drafter::SharedKv`]'s convention (2 or 4
///   tensors: `[k_full, v_full]` or `[k_full, v_full, k_swa, v_swa]`).
///   The trait implementation is responsible for cropping the K/V along
///   the sequence axis by `rejected = block_size - accepted - 1` so
///   that the drafter's next call sees the post-rollback shared K/V.
/// - `kv_offset`: absolute position offset of the post-rollback shared
///   K/V slice (= `prompt_cache[0].offset` after `rollback_speculative_cache`).
///   The drafter uses this to RoPE-rotate its cross-attention queries.
/// - `bonus_position`: position id of the bonus token whose prediction
///   the drafter is extending. For B=1 this equals the position of the
///   last accepted token in the target's KV cache (post-rollback).
pub struct MtpVerifyOutput {
    /// Last-layer hidden state at the accepted position. Used by the
    /// round-loop to seed the drafter's `h_prev` for the next round.
    pub next_hidden: UniquePtr<MlxArray>,
    /// Re-sliced shared K/V slabs in the [`crate::drafter::SharedKv`]
    /// tensor order. The round-loop reborrows this Vec into a
    /// `SharedKv` for the next `Drafter::set_shared_kv` call.
    pub next_shared_kv: Vec<UniquePtr<MlxArray>>,
    /// Absolute position offset of the post-rollback shared K/V slice.
    pub kv_offset: usize,
    /// Position id of the bonus token.
    pub bonus_position: usize,
}

impl std::fmt::Debug for MtpVerifyOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpVerifyOutput")
            .field("next_shared_kv_len", &self.next_shared_kv.len())
            .field("kv_offset", &self.kv_offset)
            .field("bonus_position", &self.bonus_position)
            .finish()
    }
}

impl MtpVerifyOutput {
    /// Materialise a borrow vector suitable for constructing a
    /// [`crate::drafter::SharedKv`].
    ///
    /// The returned `Vec<&MlxArray>` borrows from `self.next_shared_kv`;
    /// the caller is expected to construct `SharedKv::new(&refs)` on
    /// its own stack frame so the outer slice has a name and a stable
    /// lifetime. We deliberately do NOT return `SharedKv<'_>` directly:
    /// `SharedKv` needs a `&'a [&'a MlxArray]` which requires the outer
    /// slice to also be borrowed from a caller-owned local; lifting
    /// that into a method return type would force a `Box::leak` or
    /// unstable GAT lifetimes. See
    /// [`super::generator::MtpGenerator::set_shared_kv_from_verify`]
    /// for the canonical call shape.
    pub fn shared_kv_refs(&self) -> Vec<&MlxArray> {
        self.next_shared_kv
            .iter()
            .map(|ptr| {
                ptr.as_ref()
                    .expect("MtpVerifyOutput: non-null shared_kv ptr")
            })
            .collect()
    }
}

/// Trait for MTP-compatible target models.
///
/// Implemented by `mlxcel::models::gemma4::Gemma4Wrapper` (and any future
/// MTP-capable wrapper). The trait is intentionally **not** a supertrait
/// of [`crate::generate::LanguageModel`] — the standard LM contract is a
/// strict subset of what MTP needs, and most LanguageModel implementations
/// (Llama, Qwen, etc.) are not MTP-capable.
///
/// ## Object safety
///
/// The trait is object-safe so the round-loop driver can hold a
/// `&dyn MtpTarget` for diagnostic / dispatch purposes. The generator
/// itself is generic over `T: MtpTarget` to avoid one v-table hop per
/// call.
pub trait MtpTarget {
    /// Combined prefill + seed capture for the MTP round-loop.
    ///
    /// Runs the prompt through the target's sink-aware forward,
    /// samples the first bonus token from the last-position logits
    /// using `sampler`, and returns `(first_bonus, seed)` where `seed`
    /// carries the captured last-token hidden and the last full/SWA
    /// shared K/V slabs.
    ///
    /// `token_history` is the history-dependent-penalty context for the
    /// first-bonus sample (repetition / frequency / presence / DRY). The
    /// server burst path passes `initial_token_history(&prompt, ..)` so
    /// the first bonus is byte-identical to the classic decode path's
    /// first token; callers with no penalty configured pass `&[]`.
    ///
    /// `logprobs_config` controls first-bonus log-probability capture.
    /// When [`LogprobsConfig::enabled`] is false the returned
    /// `Option<TokenLogprobData>` is `None` (zero-overhead path); when
    /// true it carries the first bonus's log-probability data computed
    /// from the same penalty-adjusted logits the bonus was sampled from
    /// — byte-identical to the classic decode path's first-token
    /// logprobs.
    ///
    /// **Cache state on return**: the target's per-sequence cache has
    /// `prompt.len()` entries (the prompt prefill). The bonus token is
    /// NOT yet in the cache — it will be forwarded as the first slot of
    /// the first round's verify input.
    ///
    /// This single-call shape mirrors upstream Python's
    /// `out = lm(prompt_ids, cache=prompt_cache, return_hidden=True,
    /// return_shared_kv=True)` followed by
    /// `bonus = sampler(out.logits[:, -1:, :])`. The combined call
    /// avoids an extra forward of the bonus token between prefill and
    /// the first round (which would double-mutate the bonus position
    /// in the target's KV cache).
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &LogprobsConfig,
    ) -> (i32, MtpVerifyOutput, Option<TokenLogprobData>);

    /// Embed a token id through the target's embedding table. Equivalent
    /// to `LanguageModel::embed_tokens(...)` on the trait, but lifted
    /// here so the round-loop and unit tests don't need a
    /// `&dyn LanguageModel` reference floating around.
    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray>;

    /// First-phase verify: forward the `[bonus, draft_0, …,
    /// draft_{K-2}]` sequence through the target with sink-aware
    /// semantics and produce:
    ///
    /// 1. `target_tokens`: argmax (or sampled at temp>0) per position.
    /// 2. `target_logprobs`: per-position log-probability data when
    ///    `logprobs_config.enabled`, else `None`.
    /// 3. `captured`: the captured `hidden_full` and pre-slice shared
    ///    K/V slabs, opaque to the round-loop.
    ///
    /// **Cache state on return**: the target's per-sequence cache has
    /// `block_size` new entries appended (one per verify position).
    /// [`Self::verify_finalize`] will trim this back based on the walk's
    /// `accepted` count.
    ///
    /// `sampler` controls how target tokens are produced from the
    /// verify logits. At temperature 0 the impl MUST use argmax to
    /// preserve the greedy-parity invariant.
    ///
    /// `logprobs_config` controls per-position log-probability capture.
    /// When disabled the impl MUST leave `VerifyForwardOutput::target_logprobs`
    /// as `None` (zero-overhead path); when enabled it populates one
    /// [`TokenLogprobData`] per verify position, aligned 1:1 with
    /// `target_tokens`.
    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &LogprobsConfig,
    ) -> VerifyForwardOutput;

    /// Second-phase verify: apply the per-row tail-zero rollback to the
    /// target's KV cache based on the walk's `accepted` count, slice the
    /// captured shared K/V to the post-rollback length, and return the
    /// next-round inputs.
    ///
    /// Mirrors the upstream Python sequence:
    /// ```python
    /// if accepted < bs - 1:
    ///     lm.rollback_speculative_cache(prompt_cache, None, accepted, bs)
    /// rejected = bs - (accepted + 1)
    /// for k, kv in verify_out.shared_kv_states.items():
    ///     K, V = kv
    ///     next_shared_kv[k] = (K[..., :K.shape[-2] - rejected, :], ...)
    /// ```
    ///
    /// `captured` is the state returned by [`Self::verify_forward`].
    /// Consuming it by value reflects the one-way data flow: the
    /// captured state is for finalize only and the round-loop must
    /// not reuse it.
    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput;

    /// Number of decoder layers in the target. Used by the round-loop
    /// only for diagnostic logging.
    fn num_layers(&self) -> usize;

    /// EOS token ids of the target. The round-loop checks every emitted
    /// token against this set to honor the standard stop condition.
    fn eos_token_ids(&self) -> Vec<i32>;

    // ------------------------------------------------------------
    // Batched MTP path (B > 1).
    // ------------------------------------------------------------
    //
    // The trait carries default impls that error out so existing single-batch
    // targets keep compiling. Real batched targets (Gemma 4 wrapper, future
    // VLM wrappers) override these alongside their per-row rollback hook.

    /// Combined prefill + seed capture for B > 1 rows.
    ///
    /// `prompt_tokens_per_row[r]` is row `r`'s prompt token sequence; rows
    /// may have **different prompt lengths**, so the implementation is
    /// responsible for left-padding into a `[B, max_prompt_len]` tensor and
    /// populating the per-row prefill caches accordingly.
    ///
    /// Returns `(first_bonus_per_row, seed)` where:
    /// - `first_bonus_per_row[r]` is row `r`'s sampled first bonus.
    /// - `seed` is a [`MtpBatchedVerifyOutput`] carrying the per-row hidden
    ///   plus the batched shared K/V slabs and per-row metadata (`kv_offset`,
    ///   `bonus_position`, `kv_valid_len`, `left_padding`).
    ///
    /// Default impl returns
    /// [`DrafterError::DraftFailed`] so non-batched targets (mock or
    /// single-batch concrete) fail fast when accidentally driven into the
    /// batched path.
    #[allow(unused_variables)]
    fn prefill_and_seed_batched(
        &self,
        prompt_tokens_per_row: &[Vec<i32>],
        sampler: &SamplingConfig,
    ) -> Result<(Vec<i32>, MtpBatchedVerifyOutput), DrafterError> {
        Err(DrafterError::DraftFailed {
            reason: "MtpTarget::prefill_and_seed_batched not implemented; target must \
                     override for B > 1"
                .to_string(),
        })
    }

    /// First-phase batched verify: forward `verify_input_per_row` through the
    /// target with sink-aware semantics and produce per-row target-tokens +
    /// captured state for finalize.
    ///
    /// `verify_input_per_row[r]` has length `block_size` for every row (the
    /// caller flattens rows into a `[B, block_size]` tensor before
    /// forwarding through the target). Implementations that prefer raw
    /// `MlxArray` access can flatten themselves.
    ///
    /// Returns target argmax (per row, per position) plus a
    /// [`VerifyCaptured`] opaque to the round-loop.
    #[allow(unused_variables)]
    fn verify_forward_batched(
        &self,
        verify_input_per_row: &[Vec<i32>],
        sampler: &SamplingConfig,
    ) -> Result<MtpBatchedVerifyForwardOutput, DrafterError> {
        Err(DrafterError::DraftFailed {
            reason: "MtpTarget::verify_forward_batched not implemented; target must \
                     override for B > 1"
                .to_string(),
        })
    }

    /// Second-phase batched verify: per-row tail-zero rollback +
    /// re-slice + next-round inputs.
    ///
    /// `accepted_per_row[r]` is row `r`'s accept count from the batched walk.
    /// The implementation MUST trim the caches by `block_size - max(accepted) - 1`
    /// (the global trim amount) and per-row zero the tail for rows whose
    /// accept count is below `max(accepted)`. This mirrors the Gemma 4
    /// `Cache::zero_partial_accept_tail` hook and the
    /// `rollback_speculative_cache(..., accepted: &[i32], block_size)` API.
    #[allow(unused_variables)]
    fn verify_finalize_batched(
        &self,
        accepted_per_row: &[usize],
        block_size: usize,
        captured: VerifyCaptured,
    ) -> Result<MtpBatchedVerifyOutput, DrafterError> {
        Err(DrafterError::DraftFailed {
            reason: "MtpTarget::verify_finalize_batched not implemented; target must \
                     override for B > 1"
                .to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Batched MTP types
// ---------------------------------------------------------------------------

/// Output of [`MtpTarget::verify_forward_batched`] — per-row target argmax
/// plus captured state for finalize.
///
/// Per-row argmax is materialized as `Vec<Vec<i32>>` so the batched walk
/// can iterate cell-by-cell without re-entering MLX. The captured state is
/// opaque to the round-loop and forwarded into
/// [`MtpTarget::verify_finalize_batched`] verbatim.
pub struct MtpBatchedVerifyForwardOutput {
    /// `target_tokens_per_row[r][s]` = argmax of the target's logits at
    /// position `s` of row `r`. Outer length `B`, inner length `block_size`.
    pub target_tokens_per_row: Vec<Vec<i32>>,
    /// Opaque captured state forwarded to finalize.
    pub captured: VerifyCaptured,
}

impl std::fmt::Debug for MtpBatchedVerifyForwardOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpBatchedVerifyForwardOutput")
            .field("batch_size", &self.target_tokens_per_row.len())
            .field("captured", &self.captured)
            .finish()
    }
}

/// Output of [`MtpTarget::prefill_and_seed_batched`] and
/// [`MtpTarget::verify_finalize_batched`].
///
/// Mirrors [`MtpVerifyOutput`] but carries per-row scalar metadata so the
/// drafter can be rebound with per-row positions and left-padding extents.
///
/// - `next_hidden`: target's last-layer pre-norm hidden at each accepted
///   position. Shape `[B, 1, backbone]` in the model's native dtype.
/// - `next_shared_kv`: batched shared K/V slabs for the next round. Layout
///   matches [`crate::drafter::SharedKv`]'s convention (2 or 4 tensors).
///   Each tensor's leading dim is `B`; the seq-len axis covers the
///   per-row valid prefix plus any left-padding slack.
/// - `kv_offset_per_row`: absolute KV cache offset per row (post-rollback,
///   equal to `prompt_cache[0].offset[r]`).
/// - `bonus_position_per_row`: per-row bonus token absolute position.
/// - `kv_valid_len_per_row`: per-row valid prefix length in the shared K/V
///   seq-len axis. Used to build the `kv_valid_len` BatchScalar handed to
///   [`crate::drafter::masks::normalize_batched_shared_kv_states`].
/// - `left_padding_per_row`: per-row left-padding extent in the shared K/V
///   seq-len axis. Each row's valid prefix begins at this offset.
pub struct MtpBatchedVerifyOutput {
    pub next_hidden: UniquePtr<MlxArray>,
    pub next_shared_kv: Vec<UniquePtr<MlxArray>>,
    pub kv_offset_per_row: Vec<usize>,
    pub bonus_position_per_row: Vec<usize>,
    pub kv_valid_len_per_row: Vec<usize>,
    pub left_padding_per_row: Vec<usize>,
}

impl std::fmt::Debug for MtpBatchedVerifyOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpBatchedVerifyOutput")
            .field("batch_size", &self.kv_offset_per_row.len())
            .field("next_shared_kv_len", &self.next_shared_kv.len())
            .field("kv_offset_per_row", &self.kv_offset_per_row)
            .field("bonus_position_per_row", &self.bonus_position_per_row)
            .field("kv_valid_len_per_row", &self.kv_valid_len_per_row)
            .field("left_padding_per_row", &self.left_padding_per_row)
            .finish()
    }
}

impl MtpBatchedVerifyOutput {
    /// Materialise a borrow vector suitable for constructing a
    /// [`crate::drafter::SharedKv`].
    ///
    /// Same idiom as [`MtpVerifyOutput::shared_kv_refs`] — see that doc
    /// comment for the lifetime rationale.
    pub fn shared_kv_refs(&self) -> Vec<&MlxArray> {
        self.next_shared_kv
            .iter()
            .map(|ptr| {
                ptr.as_ref()
                    .expect("MtpBatchedVerifyOutput: non-null shared_kv ptr")
            })
            .collect()
    }

    /// Batch size derived from the per-row metadata. Tests and the
    /// round-loop driver use this to gate batched-only invariants.
    pub fn batch_size(&self) -> usize {
        self.kv_offset_per_row.len()
    }
}
