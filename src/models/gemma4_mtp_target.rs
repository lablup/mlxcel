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

//! MTP target adapter for Gemma 4.
//!
//! Glue layer that wires the binary-side
//! [`crate::models::gemma4::Gemma4Wrapper`] (and its VLM variant
//! [`crate::vision::Gemma4VLModel`]) to the
//! [`mlxcel_core::speculative::mtp::target::MtpTarget`] trait defined in
//! `mlxcel-core`. The trait sits between the round-loop
//! driver [`mlxcel_core::speculative::mtp::MtpGenerator`] and the concrete
//! Gemma 4 wrapper so the driver can stay in the core crate without
//! pulling in `mlxcel`-binary types.
//!
//! ## Why a separate adapter struct?
//!
//! The trait methods need to address a **specific per-sequence cache
//! slot** on the wrapper (the scheduler allocates a [`SequenceId`] per
//! request and routes through
//! [`Gemma4Wrapper::sequence_state`](crate::models::gemma4::Gemma4Wrapper)).
//! The trait itself takes `&self` only — it has no `seq_id` parameter —
//! so we pair the wrapper reference with the sequence id in an adapter
//! struct ([`Gemma4MtpTargetAdapter`]). The round-loop driver holds the
//! adapter by value and the adapter delegates each call to the matching
//! `forward_with_speculative_sinks` /
//! `rollback_speculative_cache` hook with the captured `seq_id`.
//!
//! ## Scope
//!
//! This adapter implements the B = 1 trait methods (`prefill_and_seed`,
//! `verify_forward`, `verify_finalize`, `embed_token`, `num_layers`,
//! `eos_token_ids`). [`Gemma4MtpBatchedTargetAdapter`] implements the
//! B > 1 batched surface for continuous batching.

use std::cell::RefCell;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::drafter::DrafterError;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::sample_token_optimized;
use mlxcel_core::speculative::mtp::target::{
    MtpBatchedVerifyForwardOutput, MtpBatchedVerifyOutput, MtpTarget, MtpVerifyOutput,
    VerifyCaptured, VerifyForwardOutput,
};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gemma4::{Cache, Gemma4SpeculativeSinks, Gemma4Wrapper, first_cache_offset};

/// Materialize an integer argmax tensor into host token ids with one
/// contiguous copy.
///
/// Used by: Gemma 4 MTP B=1/B>1 verify argmax extraction. Mirrors the
/// DFlash bulk materializer but stays local to the Gemma 4 adapter to
/// avoid exporting a speculative-internal helper as public API.
fn materialize_argmax_i32_vec(argmax: &MlxArray, expected_len: usize) -> Vec<i32> {
    let itemsize = mlxcel_core::array_itemsize(argmax);
    let bytes = mlxcel_core::array_to_raw_bytes(argmax);
    match itemsize {
        4 => bytes
            .chunks_exact(4)
            .take(expected_len)
            .map(|chunk| i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        8 => bytes
            .chunks_exact(8)
            .take(expected_len)
            .map(|chunk| {
                i64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as i32
            })
            .collect(),
        _ => {
            let flat = mlxcel_core::reshape(argmax, &[expected_len as i32]);
            let mut out = Vec::with_capacity(expected_len);
            for i in 0..expected_len {
                let cell = mlxcel_core::slice(&flat, &[i as i32], &[(i + 1) as i32]);
                let scalar = mlxcel_core::reshape(&cell, &[]);
                out.push(mlxcel_core::item_i32(&scalar));
            }
            out
        }
    }
}

/// Latest upstream Gemma 4 MTP anchors the drafter's frozen query/RoPE
/// position to the last valid target-cache token, while passing the full
/// valid cache length separately for masks.
fn mtp_draft_position(kv_valid_len: usize) -> usize {
    kv_valid_len.saturating_sub(1)
}

/// Derive the per-row `(kv_offset, bonus_position)` anchors for a batched MTP
/// seed/finalize from each row's logical `kv_valid_len` and constant
/// `left_padding`, using the shifted-frame formula that keeps both the
/// equal-length and ragged (left-padded) paths byte-identical to standalone
/// B = 1 runs:
///
/// - `kv_offset[r] = left_padding[r] + kv_valid_len[r]` — the physical/padded
///   absolute offset, i.e. the drafter's RoPE frame for row `r`. The shared
///   K/V is baked at padded positions `[left_padding[r], kv_offset[r])`, so the
///   drafter's query must rotate in the same `+left_padding[r]` shifted frame.
/// - `bonus_position[r] = kv_offset[r] - 1` — the row's frozen bonus anchor
///   (the last valid token's padded position).
///
/// For equal-length rows (`left_padding[r] == 0`) this reduces to
/// `kv_offset == kv_valid_len` and `bonus_position == kv_valid_len - 1`,
/// exactly the legacy equal-length metadata.
fn seed_anchors_from_valid_len(
    kv_valid_len: &[usize],
    left_padding: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let kv_offset_per_row: Vec<usize> = kv_valid_len
        .iter()
        .zip(left_padding)
        .map(|(&valid, &lp)| lp + valid)
        .collect();
    let bonus_position_per_row: Vec<usize> = kv_offset_per_row
        .iter()
        .map(|&offset| mtp_draft_position(offset))
        .collect();
    (kv_offset_per_row, bonus_position_per_row)
}

/// Output of [`Gemma4MtpBatchedTargetAdapter::left_padded_input`].
///
/// Carries the left-padded `[B, max_len]` prefill tensor plus the per-row
/// padding bookkeeping the ragged prefill seed needs:
/// - `max_len`: padded prompt width (= `max(prompt_len)` across the window).
/// - `left_padding[r]`: leading padding columns for row `r` (`max_len - L_r`).
/// - `valid_len[r]`: row `r`'s real prompt length `L_r`.
struct LeftPaddedPrefill {
    arr: UniquePtr<MlxArray>,
    max_len: usize,
    left_padding: Vec<usize>,
    valid_len: Vec<usize>,
}

/// Upstream mlx-vlm buffers Gemma 4 MTP rotating target caches by
/// `max(32, min(128, max(configured, requested) * 8))` tokens. The Rust
/// adapter receives the effective requested block size from the server-side
/// dispatch; the 32-token floor covers the current configured K=4 assistants.
pub(crate) fn mtp_rotating_buffer_size(requested_block_size: usize) -> i32 {
    let requested = requested_block_size.max(1);
    (requested * 8).clamp(32, 128) as i32
}

/// MTP target adapter binding a [`Gemma4Wrapper`] to a specific
/// per-sequence cache slot.
///
/// Constructed by the server-scheduler dispatch hook at the start of a
/// speculative request (after prefill) and consumed by
/// [`mlxcel_core::speculative::mtp::MtpGenerator::generate`] over the
/// course of the round loop. The adapter does NOT own the wrapper — it
/// only borrows it for the lifetime of the round-loop driver, which is
/// strictly within a single decode tick on the worker thread.
pub struct Gemma4MtpTargetAdapter<'a> {
    /// Borrowed target wrapper. The wrapper owns the per-sequence cache
    /// slot identified by `seq_id`; this adapter resolves into it on every
    /// trait call without copying state.
    wrapper: &'a Gemma4Wrapper,
    /// Per-sequence cache slot identifier. `None` selects the wrapper's
    /// internal fallback slot (used by the CLI / single-row tests). The
    /// server scheduler always passes `Some(seq_id)`.
    seq_id: Option<SequenceId>,
    /// Buffered rotating-cache slack for MTP verify append + rollback.
    rotating_buffer_size: i32,
}

impl<'a> Gemma4MtpTargetAdapter<'a> {
    /// Construct an adapter that routes every trait call through the
    /// per-sequence cache slot at `seq_id`.
    ///
    /// The scheduler calls this once per speculative request and hands
    /// the adapter to
    /// [`mlxcel_core::speculative::mtp::MtpGenerator::new`].
    pub fn new(wrapper: &'a Gemma4Wrapper, seq_id: Option<SequenceId>) -> Self {
        Self::new_with_block_size(wrapper, seq_id, 4)
    }

    /// Construct an adapter with the effective MTP block size requested by
    /// the dispatch path. Used to size the upstream-style rotating-cache
    /// rollback buffer.
    pub fn new_with_block_size(
        wrapper: &'a Gemma4Wrapper,
        seq_id: Option<SequenceId>,
        block_size: usize,
    ) -> Self {
        Self {
            wrapper,
            seq_id,
            rotating_buffer_size: mtp_rotating_buffer_size(block_size),
        }
    }

    /// Slice a `[B, T, H]` hidden tensor down to one position,
    /// preserving the singleton sequence axis (`[B, 1, H]`).
    ///
    /// MTP drafters consume the target hidden state aligned to the last
    /// emitted token, not the full prompt / verify block. During prefill
    /// this is the final prompt position; during verify it is the
    /// `accepted` position selected by the speculative walk.
    pub(crate) fn hidden_at_position(
        hidden_full: &MlxArray,
        position: usize,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_full);
        debug_assert_eq!(shape.len(), 3, "hidden must be 3-D [B, T, H]");
        let seq_len = shape[1].max(1);
        let pos = (position as i32).min(seq_len - 1);
        mlxcel_core::slice(hidden_full, &[0, pos, 0], &[shape[0], pos + 1, shape[2]])
    }

    /// Slice the final position of a `[B, T, H]` hidden tensor to
    /// `[B, 1, H]`.
    pub(crate) fn last_position_hidden(hidden_full: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_full);
        debug_assert_eq!(shape.len(), 3, "hidden must be 3-D [B, T, H]");
        let last = shape[1].saturating_sub(1) as usize;
        Self::hidden_at_position(hidden_full, last)
    }

    /// Slice one hidden position and apply the Gemma 4 final norm before
    /// handing it to the assistant drafter.
    ///
    /// Mirrors upstream `speculative_draft_hidden()`; the hidden captured by
    /// `Gemma4SpeculativeSinks` is pre-final-norm, while the MTP assistant
    /// consumes the normalized target hidden stream.
    fn draft_hidden_at_position(
        &self,
        hidden_full: &MlxArray,
        position: usize,
    ) -> UniquePtr<MlxArray> {
        let hidden = Self::hidden_at_position(hidden_full, position);
        self.wrapper
            .speculative_draft_hidden(hidden.as_ref().unwrap())
    }

    /// Final-position variant of [`Self::draft_hidden_at_position`].
    fn last_position_draft_hidden(&self, hidden_full: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = Self::last_position_hidden(hidden_full);
        self.wrapper
            .speculative_draft_hidden(hidden.as_ref().unwrap())
    }

    /// Greedy target-token extraction from pre-norm hidden states without a
    /// Rust-side per-position FFI loop.
    ///
    /// This keeps the upstream-style `skip_final_norm=True` verify path but
    /// projects the whole `[B=1, K, H]` hidden block through
    /// `speculative_logits_from_hidden()` in one MLX graph, then materializes
    /// the `[K]` argmax tensor with one host copy. It deliberately does not
    /// early-stop on the first mismatch: for the small Gemma 4 MTP block sizes
    /// we use today, avoiding `K` separate cxx/MLX calls is more important
    /// than skipping the tail projection on low-accept rounds.
    fn argmax_from_hidden_positions(&self, hidden_full: &MlxArray) -> Vec<i32> {
        let shape = mlxcel_core::array_shape(hidden_full);
        debug_assert_eq!(shape.len(), 3, "hidden must be 3-D [B, T, H]");
        let expected_len = shape[1].max(0) as usize;
        let logits = self.wrapper.speculative_logits_from_hidden(hidden_full);
        let argmax = mlxcel_core::argmax_last_axis(logits.as_ref().unwrap());
        mlxcel_core::eval(&argmax);
        materialize_argmax_i32_vec(&argmax, expected_len)
    }

    /// Slice the captured shared K/V tensors along the seq-len axis by
    /// `rejected = block_size - accepted - 1` so the post-rollback K/V
    /// matches the trimmed cache.
    ///
    /// Mirrors the upstream Python sequence:
    ///
    /// ```python
    /// rejected = block_size - (accepted + 1)
    /// for k, kv in verify_out.shared_kv_states.items():
    ///     K, V = kv
    ///     next_shared_kv[k] = (K[..., :K.shape[-2] - rejected, :], ...)
    /// ```
    ///
    /// `tensors` is the slab vector captured by [`Self::verify_forward`]
    /// (`[k_full, v_full, k_swa, v_swa]` ordering). `rejected == 0`
    /// short-circuits and returns the unchanged tensors so the
    /// full-accept case pays zero overhead.
    pub(crate) fn slice_shared_kv(
        tensors: Vec<UniquePtr<MlxArray>>,
        rejected: usize,
    ) -> Vec<UniquePtr<MlxArray>> {
        if rejected == 0 {
            return tensors;
        }
        tensors
            .into_iter()
            .map(|ptr| {
                let array = ptr.as_ref().expect("shared K/V slab must be non-null");
                let shape = mlxcel_core::array_shape(array);
                // Shape is `[B, num_kv_heads, kv_len, head_dim]`. We
                // crop along axis 2 (kv_len) by `rejected` tokens.
                debug_assert!(
                    shape.len() == 4,
                    "shared K/V slab must be 4-D, got shape {:?}",
                    shape
                );
                let kv_len = shape[2];
                let new_kv_len = kv_len - rejected as i32;
                debug_assert!(
                    new_kv_len >= 0,
                    "slice_shared_kv: rejected ({rejected}) exceeds kv_len ({kv_len})",
                );
                // mlxcel_core::slice takes &[start_0, start_1, ...] and
                // &[stop_0, stop_1, ...] for every axis.
                let starts: Vec<i32> = vec![0, 0, 0, 0];
                let stops: Vec<i32> = vec![shape[0], shape[1], new_kv_len, shape[3]];
                mlxcel_core::slice(array, &starts, &stops)
            })
            .collect()
    }

    /// Compute the argmax along the last axis for each row of a
    /// `[1, block_size, vocab]` logits tensor and return the
    /// per-position token ids.
    ///
    /// At temperature > 0 the caller is expected to override this with
    /// a real sampler; this helper handles only the greedy-parity path
    /// (`temperature == 0`). The MTP greedy-parity invariant
    /// (referenced in https://github.com/Blaizzy/mlx-vlm) requires the
    /// target tokens to match the target's own argmax extension, so for
    /// `temperature == 0` this is the load-bearing choice.
    ///
    /// Mirrors `argmax_logits_to_vec` in
    /// [`mlxcel_core::drafter::dflash::round_loop`] — duplicating the
    /// helper rather than re-exporting because it is a single-call
    /// utility and the DFlash module's version is private. Future
    /// refactor: lift this into `mlxcel_core::utils` once a second
    /// adapter (Qwen 3.5 MTP variant) lands and needs the same shape.
    pub(crate) fn argmax_per_position(logits: &MlxArray) -> Vec<i32> {
        let shape = mlxcel_core::array_shape(logits);
        debug_assert!(shape.len() == 3, "expected [1, block_size, vocab] logits");
        let block_size = shape[1];

        // `argmax_last_axis` reduces over the trailing axis, producing
        // `[1, block_size]`.
        let argmax = mlxcel_core::argmax_last_axis(logits);
        mlxcel_core::eval(&argmax);

        // Materialize all positions with one host copy. Re-entering MLX
        // once per scalar made the real-model MTP path sync on every
        // verify position, which dominated the small K=4 verify loop.
        materialize_argmax_i32_vec(&argmax, block_size as usize)
    }

    /// Compute per-position [`mlxcel_core::sampling::TokenLogprobData`]
    /// for a `[1, block_size, vocab]` verify-logits tensor, one entry
    /// per position aligned 1:1 with `target_tokens`.
    ///
    /// The verify forward is greedy-only today (`temperature == 0`,
    /// pure argmax — see [`Self::argmax_per_position`]), so the
    /// per-position logits ARE the penalty-adjusted logits the classic
    /// decode path would feed `compute_logprobs`: no temperature
    /// scaling, no history-dependent penalty applied during verify.
    /// That makes the resulting logprobs byte-identical to the classic
    /// path's decode-token logprobs for greedy sampling.
    ///
    /// Returns `None` when `logprobs_config.enabled` is false (the
    /// zero-overhead path — no slicing, no log-softmax).
    fn per_position_logprobs(
        logits: &MlxArray,
        target_tokens: &[i32],
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> Option<Vec<mlxcel_core::sampling::TokenLogprobData>> {
        if !logprobs_config.enabled {
            return None;
        }
        let shape = mlxcel_core::array_shape(logits);
        debug_assert!(shape.len() == 3, "expected [1, block_size, vocab] logits");
        let vocab = shape[2];
        let mut out: Vec<mlxcel_core::sampling::TokenLogprobData> =
            Vec::with_capacity(target_tokens.len());
        for (pos, &tok) in target_tokens.iter().enumerate() {
            // Slice position `pos` to a `[1, vocab]` tensor — the shape
            // `compute_logprobs` expects.
            let pos_i32 = pos as i32;
            let pos_logits_3d =
                mlxcel_core::slice(logits, &[0, pos_i32, 0], &[1, pos_i32 + 1, vocab]);
            let pos_logits = mlxcel_core::reshape(&pos_logits_3d, &[1, vocab]);
            // `compute_logprobs` returns `Some` here because
            // `logprobs_config.enabled` is true (checked above); the
            // `unwrap_or` is a defensive fallback that should never
            // fire.
            let lp = mlxcel_core::sampling::compute_logprobs(&pos_logits, tok, logprobs_config)
                .unwrap_or(mlxcel_core::sampling::TokenLogprobData {
                    token_id: tok,
                    logprob: 0.0,
                    top_alternatives: Vec::new(),
                });
            out.push(lp);
        }
        Some(out)
    }
}

impl<'a> MtpTarget for Gemma4MtpTargetAdapter<'a> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        let prompt_arr =
            mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);

        // Capture last-layer hidden + last full/SWA shared K/V slabs.
        // Gemma 4 owns its own caches via `ModelOwnedSequenceState`;
        // the wrapper resolves `seq_id` to the matching slot internally
        // and the trait method does not take an external cache slice.
        let mut sinks = Gemma4SpeculativeSinks::with_hidden_and_shared_kv();
        let logits = self.wrapper.forward_with_speculative_sinks(
            &prompt_arr,
            None,
            None,
            None,
            self.seq_id,
            None,
            Some(&mut sinks),
            None,
        );
        self.wrapper
            .enable_mtp_rotating_cache_buffer(self.seq_id, self.rotating_buffer_size);

        // Sample the first bonus from the last-position logits.
        // `token_history` carries the history-dependent-penalty context
        // (repetition / frequency / presence / DRY) so the first bonus
        // is byte-identical to the classic decode path's first token
        // `sample_token_optimized` returns
        // `(token_arr, adjusted_logits)`; `adjusted_logits` is the
        // penalty-adjusted `[1, vocab]` slice the bonus was sampled
        // from, and feeds `compute_logprobs` so the first-bonus logprob
        // is byte-identical to the classic path's first-token logprob
        let (token_arr, adjusted_logits) = sample_token_optimized(&logits, sampler, token_history);
        mlxcel_core::eval(&token_arr);
        let first_bonus = mlxcel_core::item_i32(&token_arr);
        let first_bonus_lp =
            mlxcel_core::sampling::compute_logprobs(&adjusted_logits, first_bonus, logprobs_config);

        // Build the seed MtpVerifyOutput from the captured sinks.
        // `hidden_full` is the last entry in the hidden-sink (since we
        // didn't pass `capture_layer_ids`, the sink has exactly one
        // entry: the last decoder layer's pre-norm hidden state for the
        // full prompt). The drafter consumes only the final prompt
        // position, so slice `[1, prompt_len, H]` to `[1, 1, H]`.
        let hidden_sink = sinks
            .hidden_sink
            .expect("with_hidden_and_shared_kv installs the hidden sink");
        let hidden_full = hidden_sink
            .into_iter()
            .next_back()
            .expect("hidden sink must carry at least one entry after a forward pass");
        let next_hidden = self.last_position_draft_hidden(hidden_full.as_ref().unwrap());

        // Materialize the shared K/V vector in the canonical
        // `[k_full, v_full, k_swa, v_swa]` order. The wrapper's
        // `shared_kv_sink` is a HashMap keyed by attention kind name.
        let shared_kv_map = sinks
            .shared_kv_sink
            .expect("with_hidden_and_shared_kv installs the shared K/V sink");
        let mut next_shared_kv: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(4);
        if let Some((k_full, v_full)) = shared_kv_map.get("full_attention") {
            next_shared_kv.push(mlxcel_core::copy(k_full.as_ref().unwrap()));
            next_shared_kv.push(mlxcel_core::copy(v_full.as_ref().unwrap()));
        }
        if let Some((k_swa, v_swa)) = shared_kv_map.get("sliding_attention") {
            next_shared_kv.push(mlxcel_core::copy(k_swa.as_ref().unwrap()));
            next_shared_kv.push(mlxcel_core::copy(v_swa.as_ref().unwrap()));
        }

        // After prefill the cache holds `prompt_tokens.len()` entries;
        // the bonus token is NOT yet in the cache (it will be the first
        // slot of the first round's verify input). `kv_offset` therefore
        // equals `prompt_tokens.len()`, while the drafter's RoPE/query
        // anchor is the last valid cache token (`kv_offset - 1`) per the
        // latest upstream reference.
        let kv_offset = prompt_tokens.len();
        let bonus_position = mtp_draft_position(kv_offset);

        let seed = MtpVerifyOutput {
            next_hidden,
            next_shared_kv,
            kv_offset,
            bonus_position,
        };
        (first_bonus, seed, first_bonus_lp)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        // Build a `[1, 1]` input_ids tensor and route through the
        // wrapper's `input_embeddings` accessor (which forwards to the
        // text model's `embed_tokens.forward(...)`).
        let input_ids = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        // `Gemma4Wrapper::input_embeddings` is `pub(crate)`, so we go
        // through the `LanguageModel::embed_tokens` trait method
        // (returns `Option<UniquePtr<MlxArray>>` for the LM contract;
        // Gemma 4 always returns `Some` because it owns its embedding
        // table).
        <Gemma4Wrapper as mlxcel_core::generate::LanguageModel>::embed_tokens(
            self.wrapper,
            &input_ids,
        )
        .expect("Gemma4Wrapper exposes its embed_tokens table")
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> VerifyForwardOutput {
        // Sink-aware forward over `[bonus, draft_0, …, draft_{K-2}]`.
        let verify_arr = mlxcel_core::from_slice_i32(verify_input, &[1, verify_input.len() as i32]);
        let mut sinks = Gemma4SpeculativeSinks::with_hidden_and_shared_kv();
        // Greedy/no-logprobs can use the latest upstream deferred path: run
        // the transformer once with `skip_final_norm=True`, capture pre-norm
        // hidden/shared K/V, then project hidden positions to logits only as
        // needed. Keep the full-logits path for non-greedy or logprob
        // requests so existing sampler/logprob semantics stay unchanged.
        // The upstream Python reference uses deferred greedy hidden→logits
        // projection by default. In Rust/MLX today that path projects one
        // position at a time across the cxx bridge and is slower than the
        // batched `[K, vocab]` LM-head projection for Gemma 4 31B on local
        // Apple Silicon runs. Keep it available for parity experiments, but
        // leave the faster full-logits verifier as the default until we have
        // a fused/graph-side deferred walk.
        let use_deferred_greedy = std::env::var("MLXCEL_ENABLE_MTP_DEFERRED").ok().as_deref()
            == Some("1")
            && sampler.temperature == 0.0
            && !logprobs_config.enabled;
        let logits = if use_deferred_greedy {
            let _ = self.wrapper.forward_hidden_with_speculative_sinks(
                &verify_arr,
                None,
                None,
                None,
                self.seq_id,
                None,
                Some(&mut sinks),
                true,
            );
            None
        } else {
            Some(self.wrapper.forward_with_speculative_sinks(
                &verify_arr,
                None,
                None,
                None,
                self.seq_id,
                None,
                Some(&mut sinks),
                None,
            ))
        };

        // Greedy-parity gate: pull the per-position argmax tokens from
        // the verify logits. At temperature == 0 this is byte-identical
        // to the drafter-less target's own argmax extension.
        //
        // At temperature > 0 a future enhancement plumbs the sampler
        // through per-position; the round-loop driver's perf-sensitive
        // path is greedy, so we keep argmax-only for now.
        let target_tokens = if let Some(logits) = logits.as_ref() {
            Self::argmax_per_position(logits)
        } else {
            // The hidden sink is still owned by `sinks`; pull it below but
            // compute after extraction so the hidden handle is available for
            // both token projection and `VerifyCaptured`.
            Vec::new()
        };

        // Per-position log-probability data, aligned 1:1 with
        // `target_tokens`. `None` (zero-overhead) when logprobs are
        // disabled; the round loop forwards the entries for accepted
        // positions on to `finalize_burst_success`.
        let target_logprobs = logits.as_ref().and_then(|logits| {
            Self::per_position_logprobs(logits, &target_tokens, logprobs_config)
        });

        // Capture the hidden + pre-slice shared K/V for the finalize step.
        let hidden_sink = sinks
            .hidden_sink
            .expect("with_hidden_and_shared_kv installs the hidden sink");
        let hidden_full = hidden_sink
            .into_iter()
            .next_back()
            .expect("hidden sink must carry at least one entry");
        let target_tokens = if target_tokens.is_empty() && use_deferred_greedy {
            self.argmax_from_hidden_positions(hidden_full.as_ref().unwrap())
        } else {
            target_tokens
        };

        let shared_kv_map = sinks
            .shared_kv_sink
            .expect("with_hidden_and_shared_kv installs the shared K/V sink");
        let mut captured_tensors: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(5);
        captured_tensors.push(hidden_full);
        if let Some((k_full, v_full)) = shared_kv_map.get("full_attention") {
            captured_tensors.push(mlxcel_core::copy(k_full.as_ref().unwrap()));
            captured_tensors.push(mlxcel_core::copy(v_full.as_ref().unwrap()));
        }
        if let Some((k_swa, v_swa)) = shared_kv_map.get("sliding_attention") {
            captured_tensors.push(mlxcel_core::copy(k_swa.as_ref().unwrap()));
            captured_tensors.push(mlxcel_core::copy(v_swa.as_ref().unwrap()));
        }

        VerifyForwardOutput {
            target_tokens,
            target_logprobs,
            captured: VerifyCaptured {
                tensors: captured_tensors,
                // No per-step scalar metadata used today; the per-row
                // batched path will populate these with kv_offset_pre /
                // bonus_position_pre.
                scalars: Vec::new(),
            },
        }
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        // Pop the hidden out of the captured tensors. The convention
        // documented on `VerifyCaptured` is index 0 = hidden_full,
        // indices 1.. = the shared K/V slabs in
        // `[k_full, v_full, k_swa, v_swa]` order.
        let mut tensors = captured.tensors;
        assert!(
            !tensors.is_empty(),
            "VerifyCaptured must carry at least the hidden tensor at index 0"
        );
        let hidden_full = tensors.remove(0);
        let next_hidden = self.draft_hidden_at_position(hidden_full.as_ref().unwrap(), accepted);

        // Per-row tail-zero rollback. For B = 1 the accept slice is a
        // single-element view; the rotating-cache zeroing inside
        // `rollback_speculative_cache` is a no-op when `accepted.len() == 1`.
        let accepted_i32 = accepted as i32;
        let block_size_i32 = block_size as i32;
        let _ =
            self.wrapper
                .rollback_speculative_cache(self.seq_id, &[accepted_i32], block_size_i32);

        // Slice the captured shared K/V to the post-rollback length.
        let rejected = block_size - accepted - 1;
        let next_shared_kv = Self::slice_shared_kv(tensors, rejected);

        // Upstream rebinds the drafter with `prompt_cache[0].offset`
        // *after* rollback, i.e. the absolute post-rollback target cache
        // offset. Returning `accepted + 1` here was only a per-round
        // delta; the generator forwards this value verbatim to
        // `set_shared_kv`, so the drafter's RoPE position and
        // bidirectional masks drifted back to tiny offsets after the
        // first verify round. Read the wrapper-owned cache directly so
        // the Rust path mirrors the Python reference and keeps
        // `kv_valid_len == kv_len` in the no-padding fast path. The drafter
        // query/RoPE anchor itself is `kv_offset - 1`, matching upstream
        // `_mtp_draft_position(kv_offset)`.
        let kv_offset = self
            .wrapper
            .speculative_cache_offset(self.seq_id, "full_attention")
            .max(0) as usize;
        let bonus_position = mtp_draft_position(kv_offset);

        MtpVerifyOutput {
            next_hidden,
            next_shared_kv,
            kv_offset,
            bonus_position,
        }
    }

    fn num_layers(&self) -> usize {
        self.wrapper.num_layers_value()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.wrapper.eos_token_ids_value()
    }
}

/// MTP target adapter for the Gemma 4 VLM wrapper.
///
/// Reuses [`Gemma4MtpTargetAdapter`] internally by delegating to the
/// VLM wrapper's inner text model. The VLM wrapper itself holds no
/// speculative state — vision features are fully prefilled before the
/// MTP round loop begins, so the round loop only interacts with the
/// text backbone (mirrors the [`mlxcel_core::drafter::dflash::SpeculativeTarget`]
/// VLM impl on `Qwen35VLModel`).
pub struct Gemma4VLMtpTargetAdapter<'a> {
    inner: Gemma4MtpTargetAdapter<'a>,
}

impl<'a> Gemma4VLMtpTargetAdapter<'a> {
    /// Construct an adapter that routes every trait call through the
    /// inner text model's per-sequence cache slot at `seq_id`.
    pub fn new(vlm: &'a crate::vision::Gemma4VLModel, seq_id: Option<SequenceId>) -> Self {
        Self::new_with_block_size(vlm, seq_id, 4)
    }

    /// Construct an adapter with the effective MTP block size requested by
    /// the dispatch path.
    pub fn new_with_block_size(
        vlm: &'a crate::vision::Gemma4VLModel,
        seq_id: Option<SequenceId>,
        block_size: usize,
    ) -> Self {
        Self {
            inner: Gemma4MtpTargetAdapter::new_with_block_size(&vlm.text_model, seq_id, block_size),
        }
    }
}

impl<'a> MtpTarget for Gemma4VLMtpTargetAdapter<'a> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        self.inner
            .prefill_and_seed(prompt_tokens, sampler, token_history, logprobs_config)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        self.inner.embed_token(token_id)
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> VerifyForwardOutput {
        self.inner
            .verify_forward(verify_input, sampler, logprobs_config)
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        self.inner.verify_finalize(accepted, block_size, captured)
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.inner.eos_token_ids()
    }
}

/// MTP target adapter for the Gemma 4 Unified wrapper.
///
/// Reuses [`Gemma4MtpTargetAdapter`] internally by delegating to the Unified
/// wrapper's inner text model. Identical rationale to
/// [`Gemma4VLMtpTargetAdapter`]: any multimodal (image / audio) features are
/// fully prefilled before the MTP round loop begins, so the round loop only
/// interacts with the text backbone — draft and verify operate on text tokens.
/// The encoder-free vision embedder and per-layer-inputs state of
/// [`crate::vision::Gemma4UnifiedModel`] hold no speculative state.
pub struct Gemma4UnifiedMtpTargetAdapter<'a> {
    inner: Gemma4MtpTargetAdapter<'a>,
}

impl<'a> Gemma4UnifiedMtpTargetAdapter<'a> {
    /// Construct an adapter that routes every trait call through the inner
    /// text model's per-sequence cache slot at `seq_id`.
    pub fn new(unified: &'a crate::vision::Gemma4UnifiedModel, seq_id: Option<SequenceId>) -> Self {
        Self::new_with_block_size(unified, seq_id, 4)
    }

    /// Construct an adapter with the effective MTP block size requested by the
    /// dispatch path.
    pub fn new_with_block_size(
        unified: &'a crate::vision::Gemma4UnifiedModel,
        seq_id: Option<SequenceId>,
        block_size: usize,
    ) -> Self {
        Self {
            inner: Gemma4MtpTargetAdapter::new_with_block_size(
                &unified.text_model,
                seq_id,
                block_size,
            ),
        }
    }
}

impl<'a> MtpTarget for Gemma4UnifiedMtpTargetAdapter<'a> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        self.inner
            .prefill_and_seed(prompt_tokens, sampler, token_history, logprobs_config)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        self.inner.embed_token(token_id)
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> VerifyForwardOutput {
        self.inner
            .verify_forward(verify_input, sampler, logprobs_config)
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        self.inner.verify_finalize(accepted, block_size, captured)
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.inner.eos_token_ids()
    }
}

// ===========================================================================
// Batched MTP target adapter (B > 1)
// ===========================================================================

/// Drafter-requested capture-layer ids for the batched verify path.
///
/// `None` means "capture the last decoder layer's pre-norm hidden state
/// only" — which is exactly the seed/verify contract the B = 1 adapter
/// uses (it passes `capture_layer_ids: None` to
/// `forward_with_speculative_sinks`). The batched MTP path reuses that
/// shape: `MtpBatchedVerifyOutput::next_hidden` is the last-layer hidden
/// at the accepted position, `[B, 1, backbone]`.
const BATCHED_CAPTURE_LAYER_IDS: Option<&[usize]> = None;

/// Batched MTP target adapter binding a [`Gemma4Wrapper`] to a
/// **caller-owned** `[B, ...]` cache vector.
///
/// ## Why this is a distinct struct from [`Gemma4MtpTargetAdapter`]
///
/// The B = 1 adapter routes every trait call through the wrapper's
/// per-[`SequenceId`] cache slot. That slot model is single-row by
/// construction — one `Vec<Cache>` per `SequenceId` — so it cannot
/// express a `[B, ...]` verify forward where all `B` rows advance
/// through one MLX dispatch. The batched adapter instead owns a single
/// `Vec<Cache>` (every per-layer cache carries a leading batch dim `B`)
/// and drives it through
/// [`Gemma4Wrapper::forward_with_speculative_sinks_explicit_cache`].
///
/// ## Interior mutability
///
/// [`MtpTarget`]'s trait methods take `&self` (the round-loop driver
/// holds the target by shared reference for the burst's lifetime), but
/// the batched verify forward MUST mutate the `[B, ...]` cache in place.
/// The cache is therefore held behind a [`RefCell`]. The scheduler is
/// single-threaded (every MLX dispatch goes through the same stream), so
/// a `RefCell` is sufficient — no `Mutex` is needed. Each trait method
/// borrows the cache mutably for the duration of one forward; the borrow
/// is released before the method returns, so no two borrows ever
/// overlap.
///
/// ## Scope: equal-length and variable-length prompts within a window
///
/// `prefill_and_seed_batched` forwards the `[B, max_prompt_len]` prompt
/// batch in one pass. When every row's prompt is the **same length**, the
/// 2-D causal masks (`create_causal_mask(L, 0)`) broadcast cleanly across
/// the batch and the result is byte-identical to running B separate B = 1
/// prefills.
///
/// **Variable-length (ragged)** prompts are handled by
/// [`Self::prefill_and_seed_batched_ragged`] (routed automatically when the
/// rows differ in length), gated upstream by the
/// `MLXCEL_ENABLE_MTP_BATCH_RAGGED` opt-in. Each row is left-padded to
/// `max_prompt_len` with a per-row left-padding causal mask; in the eligible
/// non-capped regime (`max_prompt_len <= sliding_window`) that single mask is
/// byte-identical to the windowed left-padding mask, so it is correct for both
/// the full-attention and sliding-window layers. Greedy parity is preserved by
/// the left-padding uniform per-row position shift (see that method's docs).
/// The verify rounds are always uniform width (`block_size` per row) so they
/// are unconditionally batched; the constant per-row `left_padding` stashed at
/// prefill keeps every round in each row's shifted RoPE frame.
pub struct Gemma4MtpBatchedTargetAdapter<'a> {
    /// Borrowed target wrapper. The wrapper owns its weights; this
    /// adapter owns the per-burst `[B, ...]` cache separately.
    wrapper: &'a Gemma4Wrapper,
    /// The caller-owned `[B, ...]` per-layer cache for this burst. Starts
    /// empty; the first `prefill_and_seed_batched` grows every per-layer
    /// cache with a leading batch dim `B`. Held behind a `RefCell` so the
    /// `&self` trait methods can mutate it in place — see the struct
    /// docstring.
    caches: RefCell<Vec<Cache>>,
    /// Batch size. Set at construction; every trait call's per-row
    /// vectors must match this length.
    batch_size: usize,
    /// Buffered rotating-cache slack for MTP verify append + rollback.
    rotating_buffer_size: i32,
    /// Per-row logical target cache lengths (= each row's `kv_valid_len`). The
    /// physical cache offset is the global max after rollback, but shorter rows
    /// have their tails zeroed and must pass their own valid length to the
    /// drafter masks and RoPE anchor. For the equal-length path every entry
    /// equals the shared prompt length; for the ragged (left-padded) path each
    /// entry is the row's unpadded prompt length plus the tokens accepted so
    /// far.
    positions: RefCell<Vec<usize>>,
    /// Per-row left-padding extent in the shared K/V seq-len axis. `0` for every
    /// row on the equal-length path; `max_prompt_len - prompt_len[r]` on the
    /// ragged path, constant across the whole burst (the prompt's leading
    /// padding is never trimmed). Threaded into every seed so the drafter rolls
    /// each row's K/V into the prefix-valid layout and rotates queries in the
    /// row's shifted frame.
    left_padding: RefCell<Vec<usize>>,
}

impl<'a> Gemma4MtpBatchedTargetAdapter<'a> {
    /// Construct a batched adapter for a `batch_size`-row burst.
    ///
    /// The adapter allocates a fresh `[B, ...]` cache vector via
    /// [`Gemma4Wrapper::make_speculative_caches`]; the cache is empty
    /// until the first `prefill_and_seed_batched` call.
    pub fn new(wrapper: &'a Gemma4Wrapper, batch_size: usize) -> Self {
        Self::new_with_block_size(wrapper, batch_size, 4)
    }

    /// Construct a batched adapter with the effective MTP block size requested
    /// by the dispatch path. Sliding layers get a small upstream-style
    /// rollback buffer from the first prefill onward.
    pub fn new_with_block_size(
        wrapper: &'a Gemma4Wrapper,
        batch_size: usize,
        block_size: usize,
    ) -> Self {
        assert!(
            batch_size >= 1,
            "Gemma4MtpBatchedTargetAdapter: batch_size must be >= 1",
        );
        let rotating_buffer_size = mtp_rotating_buffer_size(block_size);
        Self {
            wrapper,
            caches: RefCell::new(wrapper.make_speculative_caches()),
            batch_size,
            rotating_buffer_size,
            positions: RefCell::new(vec![0; batch_size]),
            left_padding: RefCell::new(vec![0; batch_size]),
        }
    }

    /// Batch size accessor (test / diagnostic).
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Build a `[B, width]` i32 tensor from per-row token slices that all
    /// have the same `width`. Returns an `Err` if any row's width
    /// differs from the first (the batched verify forward requires a
    /// rectangular input).
    fn rectangular_input(
        per_row: &[Vec<i32>],
        expected_batch: usize,
    ) -> Result<(UniquePtr<MlxArray>, i32), DrafterError> {
        if per_row.len() != expected_batch {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Gemma4 batched MTP target: expected {expected_batch} rows, got {}",
                    per_row.len()
                ),
            });
        }
        if per_row.is_empty() {
            return Err(DrafterError::DraftFailed {
                reason: "Gemma4 batched MTP target: empty batch".to_string(),
            });
        }
        let width = per_row[0].len();
        if width == 0 {
            return Err(DrafterError::DraftFailed {
                reason: "Gemma4 batched MTP target: rows must be non-empty".to_string(),
            });
        }
        for (r, row) in per_row.iter().enumerate() {
            if row.len() != width {
                return Err(DrafterError::DraftFailed {
                    reason: format!(
                        "Gemma4 batched MTP target: row {r} has width {} but row 0 has \
                         width {width}; the batched verify forward requires a \
                         rectangular [B, width] input (equal-length rows)",
                        row.len()
                    ),
                });
            }
        }
        let mut flat: Vec<i32> = Vec::with_capacity(expected_batch * width);
        for row in per_row {
            flat.extend_from_slice(row);
        }
        let arr = mlxcel_core::from_slice_i32(&flat, &[expected_batch as i32, width as i32]);
        Ok((arr, width as i32))
    }

    /// Build a **left-padded** `[B, max_prompt_len]` i32 tensor from per-row
    /// prompt slices of (possibly) different lengths, returning the padded
    /// tensor plus per-row `(max_len, left_padding, valid_len)` metadata.
    ///
    /// Row `r` (length `L_r`) is placed at padded indices
    /// `[max_len - L_r, max_len)`; the leading `left_padding[r] = max_len - L_r`
    /// columns hold a padding token (`0`). Left-padding is the parity-preserving
    /// layout: every real token in a row is shifted right by the same constant
    /// `left_padding[r]`, so the proportional-RoPE phase applied uniformly across
    /// the batch (scalar offset `0` → position == padded index) preserves every
    /// intra-row relative position. The leading padding columns are masked out by
    /// the caller-supplied left-padding attention mask, so they never affect any
    /// real query's output, and each row's last real token lands at the shared
    /// padded index `max_len - 1` (so the seed bonus is sampled from the uniform
    /// last position, exactly as the equal-length path does).
    ///
    /// Returns `Err` on batch-size mismatch or an empty / zero-length row.
    fn left_padded_input(
        per_row: &[Vec<i32>],
        expected_batch: usize,
    ) -> Result<LeftPaddedPrefill, DrafterError> {
        if per_row.len() != expected_batch {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Gemma4 batched MTP target: expected {expected_batch} rows, got {}",
                    per_row.len()
                ),
            });
        }
        if per_row.is_empty() {
            return Err(DrafterError::DraftFailed {
                reason: "Gemma4 batched MTP target: empty batch".to_string(),
            });
        }
        let mut max_len = 0usize;
        for (r, row) in per_row.iter().enumerate() {
            if row.is_empty() {
                return Err(DrafterError::DraftFailed {
                    reason: format!("Gemma4 batched MTP target: prompt row {r} must be non-empty"),
                });
            }
            max_len = max_len.max(row.len());
        }

        let mut flat: Vec<i32> = vec![0; expected_batch * max_len];
        let mut left_padding: Vec<usize> = Vec::with_capacity(expected_batch);
        let mut valid_len: Vec<usize> = Vec::with_capacity(expected_batch);
        for (r, row) in per_row.iter().enumerate() {
            let lp = max_len - row.len();
            let base = r * max_len + lp;
            flat[base..base + row.len()].copy_from_slice(row);
            left_padding.push(lp);
            valid_len.push(row.len());
        }
        let arr = mlxcel_core::from_slice_i32(&flat, &[expected_batch as i32, max_len as i32]);
        Ok(LeftPaddedPrefill {
            arr,
            max_len,
            left_padding,
            valid_len,
        })
    }

    /// Per-row argmax over a `[B, width, vocab]` logits tensor.
    ///
    /// Returns `out[r][s]` = argmax of `logits[r, s, :]`. Materialises a
    /// nested `Vec<Vec<i32>>` so the batched walk can iterate without
    /// re-entering MLX per cell. Mirrors `argmax_logits_per_row` in the
    /// DFlash batched round loop.
    fn argmax_per_row(logits: &MlxArray, batch_size: i32, width: i32) -> Vec<Vec<i32>> {
        let shape = mlxcel_core::array_shape(logits);
        debug_assert_eq!(shape.len(), 3, "expected [B, width, vocab] logits");
        let argmax = mlxcel_core::argmax_last_axis(logits);
        mlxcel_core::eval(&argmax);
        let flat = materialize_argmax_i32_vec(&argmax, (batch_size * width) as usize);
        let mut out: Vec<Vec<i32>> = Vec::with_capacity(batch_size as usize);
        for r in 0..batch_size {
            let start = (r * width) as usize;
            let end = start + width as usize;
            out.push(flat[start..end].to_vec());
        }
        out
    }

    /// Divergent-round analogue of [`Self::slice_shared_kv_batched`]
    /// (issue #203): compact each captured shared K/V slab the same way
    /// `rollback_speculative_cache_divergent` compacts the live caches, so
    /// the slabs handed to the drafter keep every row's valid K/V as a
    /// contiguous prefix (`normalize_batched_shared_kv_states` assumes the
    /// row's real entries occupy `[left_padding[r], left_padding[r] +
    /// kv_valid_len[r])`).
    ///
    /// Per row: move the accepted window entries from the shared physical
    /// write base `[o_pre, o_pre + accepted[r] + 1)` down to the row's
    /// logical valid end `ve_pre[r]`, zero the vacated region up to
    /// `o_post`, then crop every slab to `o_post`.
    fn compact_shared_kv_batched(
        tensors: Vec<UniquePtr<MlxArray>>,
        ve_pre: &[i32],
        accepted: &[i32],
        o_pre: i32,
        o_post: i32,
    ) -> Result<Vec<UniquePtr<MlxArray>>, DrafterError> {
        tensors
            .into_iter()
            .map(|ptr| {
                let array = ptr.as_ref().expect("shared K/V slab must be non-null");
                let shape = mlxcel_core::array_shape(array);
                if shape.len() != 4 || shape[0] != ve_pre.len() as i32 {
                    return Err(DrafterError::DraftFailed {
                        reason: format!(
                            "Gemma4 batched MTP target: shared K/V slab must be \
                             [B={}, H, kv, D], got shape {:?}",
                            ve_pre.len(),
                            shape
                        ),
                    });
                }
                // Validate every row against the slab axis BEFORE any
                // slice_update (mirrors compact_partial_accept_rows).
                let slab_len = shape[2];
                for (r, (&ve, &a)) in ve_pre.iter().zip(accepted).enumerate() {
                    if ve > o_pre || a < 0 || o_pre + a + 1 > slab_len || o_post > slab_len {
                        return Err(DrafterError::DraftFailed {
                            reason: format!(
                                "Gemma4 batched MTP target: slab compaction row {r} out \
                                 of bounds (ve_pre {ve}, accepted {a}, o_pre {o_pre}, \
                                 o_post {o_post}, slab_len {slab_len})"
                            ),
                        });
                    }
                }
                let dtype = mlxcel_core::array_dtype(array);
                let mut out = mlxcel_core::copy(array);
                for (r, (&ve, &a)) in ve_pre.iter().zip(accepted).enumerate() {
                    let n = a + 1;
                    let bi = r as i32;
                    if ve < o_pre {
                        // Materialize the source slice with an explicit copy
                        // BEFORE the update: a bare slice is a lazy view of
                        // the same buffer, and `slice_update` may donate that
                        // buffer to its output, so an overlapping move
                        // (`ve + n > o_pre`) would read already-overwritten
                        // rows without the copy.
                        let src = mlxcel_core::copy(&mlxcel_core::slice(
                            &out,
                            &[bi, 0, o_pre, 0],
                            &[bi + 1, shape[1], o_pre + n, shape[3]],
                        ));
                        out = mlxcel_core::slice_update(
                            &out,
                            &src,
                            &[bi, 0, ve, 0],
                            &[bi + 1, shape[1], ve + n, shape[3]],
                        );
                    }
                    let z_start = ve + n;
                    if z_start < o_post {
                        let span = o_post - z_start;
                        let zero = mlxcel_core::zeros(&[1, shape[1], span, shape[3]], dtype);
                        out = mlxcel_core::slice_update(
                            &out,
                            &zero,
                            &[bi, 0, z_start, 0],
                            &[bi + 1, shape[1], o_post, shape[3]],
                        );
                    }
                }
                Ok(mlxcel_core::slice(
                    &out,
                    &[0, 0, 0, 0],
                    &[shape[0], shape[1], o_post, shape[3]],
                ))
            })
            .collect()
    }

    /// Slice the captured shared K/V slabs to the post-rollback length.
    ///
    /// Batched analogue of [`Gemma4MtpTargetAdapter::slice_shared_kv`]:
    /// crops every slab along the `kv_len` axis (axis 2 of
    /// `[B, num_kv_heads, kv_len, head_dim]`) by `rejected`. `rejected ==
    /// 0` short-circuits so the full-accept case pays zero overhead.
    fn slice_shared_kv_batched(
        tensors: Vec<UniquePtr<MlxArray>>,
        rejected: usize,
    ) -> Vec<UniquePtr<MlxArray>> {
        if rejected == 0 {
            return tensors;
        }
        tensors
            .into_iter()
            .map(|ptr| {
                let array = ptr.as_ref().expect("shared K/V slab must be non-null");
                let shape = mlxcel_core::array_shape(array);
                debug_assert!(
                    shape.len() == 4,
                    "shared K/V slab must be 4-D, got shape {:?}",
                    shape
                );
                let kv_len = shape[2];
                let new_kv_len = kv_len - rejected as i32;
                debug_assert!(
                    new_kv_len >= 0,
                    "slice_shared_kv_batched: rejected ({rejected}) exceeds kv_len ({kv_len})",
                );
                let starts: Vec<i32> = vec![0, 0, 0, 0];
                let stops: Vec<i32> = vec![shape[0], shape[1], new_kv_len, shape[3]];
                mlxcel_core::slice(array, &starts, &stops)
            })
            .collect()
    }

    /// Run the sink-aware `[B, width]` forward and return the logits plus
    /// the captured hidden (last entry, `[B, width, hidden]`) and the
    /// canonical-order shared K/V vector (`[k_full, v_full, k_swa,
    /// v_swa]`).
    ///
    /// Borrows the adapter's `[B, ...]` cache mutably for the duration of
    /// the forward. The borrow is released before this returns.
    fn batched_sink_forward(
        &self,
        input_arr: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        Vec<UniquePtr<MlxArray>>,
    ) {
        // Verify rounds (and the equal-length prefill) take the `mask == None`
        // path, where the forward derives both attention-family masks from the
        // cache offsets. For a ragged burst the prompt's per-row leading
        // padding is still resident in the shared cache, so we thread the
        // stashed per-row `left_padding` so the verify forward masks each row's
        // `[0, left_padding[r])` padding keys — without it the verify query
        // attends the padding K/V and breaks greedy parity for the
        // most-left-padded row. For the equal-length path `left_padding` is
        // all-zero, so the forward takes its byte-identical plain-mask branch.
        let left_padding: Vec<i32> = self
            .left_padding
            .borrow()
            .iter()
            .map(|&p| p as i32)
            .collect();
        let lp_ref: Option<&[i32]> = if left_padding.iter().any(|&p| p > 0) {
            Some(&left_padding)
        } else {
            None
        };
        // Prefill (offset 0) has no stale tail, so `per_row_valid_end` is None;
        // the verify forward is the sole producer of `Some` (issue #163).
        self.batched_sink_forward_with_mask(input_arr, None, lp_ref, None)
    }

    /// Sink-aware batched forward with an optional explicit attention mask.
    ///
    /// `mask` is `None` for the equal-length path and for every verify round
    /// (the forward derives both the full-attention and sliding-window causal
    /// masks from the cache offsets). In the `mask == None` path, `left_padding`
    /// (per-row leading padding columns) is threaded so the verify forward masks
    /// each ragged row's resident `[0, left_padding[r])` padding keys; it is
    /// `None` / all-zero for the equal-length path (byte-identical plain masks).
    ///
    /// For the ragged **prefill** the caller instead passes a single explicit
    /// **left-padding** causal mask (with `left_padding == None`): in the
    /// eligible regime (`max_prompt_len <= sliding_window`, non-capped key axis)
    /// the windowed left-padding mask is byte-identical to the plain
    /// left-padding mask, so one `[B, 1, max_len, max_len]` mask is correct for
    /// *both* the full-attention and sliding-window layers (the forward copies
    /// the single mask into both slots). This equivalence is unit-tested in
    /// `mlxcel_core::utils` (`windowed_left_padding_mask_matches_plain_left_padding_when_uncapped`).
    ///
    /// `per_row_valid_end` (issue #163) is supplied ONLY by the verify forward:
    /// it carries each row's logical valid key end (`left_padding[r] +
    /// positions[r]`) so the offset-derived masks exclude the row's stale
    /// `[valid_end, offset)` rejected-draft / zeroed tail after divergent
    /// accepts. Both prefill paths pass `None` (offset 0 has no gap).
    fn batched_sink_forward_with_mask(
        &self,
        input_arr: &MlxArray,
        mask: Option<&MlxArray>,
        left_padding: Option<&[i32]>,
        per_row_valid_end: Option<&[i32]>,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        Vec<UniquePtr<MlxArray>>,
    ) {
        let mut sinks = Gemma4SpeculativeSinks::with_hidden_and_shared_kv();
        let logits = {
            let mut caches = self.caches.borrow_mut();
            self.wrapper.forward_with_speculative_sinks_explicit_cache(
                input_arr,
                None,
                None,
                mask,
                &mut caches,
                BATCHED_CAPTURE_LAYER_IDS,
                Some(&mut sinks),
                left_padding,
                per_row_valid_end,
            )
        };

        let hidden_sink = sinks
            .hidden_sink
            .expect("with_hidden_and_shared_kv installs the hidden sink");
        let hidden_full = hidden_sink
            .into_iter()
            .next_back()
            .expect("hidden sink must carry at least one entry after a forward pass");

        let shared_kv_map = sinks
            .shared_kv_sink
            .expect("with_hidden_and_shared_kv installs the shared K/V sink");
        let mut shared_kv: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(4);
        if let Some((k_full, v_full)) = shared_kv_map.get("full_attention") {
            shared_kv.push(mlxcel_core::copy(k_full.as_ref().unwrap()));
            shared_kv.push(mlxcel_core::copy(v_full.as_ref().unwrap()));
        }
        if let Some((k_swa, v_swa)) = shared_kv_map.get("sliding_attention") {
            shared_kv.push(mlxcel_core::copy(k_swa.as_ref().unwrap()));
            shared_kv.push(mlxcel_core::copy(v_swa.as_ref().unwrap()));
        }

        (logits, hidden_full, shared_kv)
    }

    fn enable_rotating_cache_buffer(&self) {
        let mut caches = self.caches.borrow_mut();
        for cache in caches.iter_mut() {
            if let Err(error) = cache.enable_mtp_rotating_buffer(self.rotating_buffer_size) {
                tracing::warn!(
                    error,
                    buffer_size = self.rotating_buffer_size,
                    "Gemma4 batched MTP could not enable rotating-cache rollback buffer"
                );
            }
        }
    }

    /// Slice the `[B, T, *]` hidden tensor down to its last position,
    /// yielding `[B, 1, *]`.
    fn last_position_hidden(hidden_full: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_full);
        debug_assert_eq!(shape.len(), 3, "hidden must be 3-D [B, T, hidden]");
        let last = shape[1] - 1;
        mlxcel_core::slice(hidden_full, &[0, last, 0], &[shape[0], last + 1, shape[2]])
    }

    fn last_position_draft_hidden(&self, hidden_full: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = Self::last_position_hidden(hidden_full);
        self.wrapper
            .speculative_draft_hidden(hidden.as_ref().unwrap())
    }

    /// Slice a `[B, T, H]` hidden tensor at a per-row accepted position,
    /// yielding `[B, 1, H]`.
    ///
    /// Each row's next drafter state must align with that row's last
    /// emitted token (`accepted_per_row[r]` in the verify block). A
    /// simple last-position slice is only correct when every row fully
    /// accepts the whole block.
    fn hidden_at_positions_batched(
        hidden_full: &MlxArray,
        positions: &[usize],
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let shape = mlxcel_core::array_shape(hidden_full);
        debug_assert_eq!(shape.len(), 3, "hidden must be 3-D [B, T, hidden]");
        let batch = shape[0] as usize;
        let seq_len = shape[1].max(1);
        if positions.len() != batch {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Gemma4 batched MTP target: hidden position count {} does not match \
                     batch size {batch}",
                    positions.len()
                ),
            });
        }

        let mut rows: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(batch);
        for (r, &position) in positions.iter().enumerate() {
            let pos = (position as i32).min(seq_len - 1);
            rows.push(mlxcel_core::slice(
                hidden_full,
                &[r as i32, pos, 0],
                &[(r as i32) + 1, pos + 1, shape[2]],
            ));
        }

        let mut iter = rows.into_iter();
        let mut out = iter
            .next()
            .expect("batch size is non-zero for hidden_at_positions_batched");
        for row in iter {
            out = mlxcel_core::concatenate(out.as_ref().unwrap(), row.as_ref().unwrap(), 0);
        }
        Ok(out)
    }

    fn draft_hidden_at_positions_batched(
        &self,
        hidden_full: &MlxArray,
        positions: &[usize],
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let hidden = Self::hidden_at_positions_batched(hidden_full, positions)?;
        Ok(self
            .wrapper
            .speculative_draft_hidden(hidden.as_ref().unwrap()))
    }

    /// Variable-length-prompt (ragged) prefill + seed.
    ///
    /// Left-pads every row to `max_prompt_len`, runs one sink-aware forward
    /// with a per-row left-padding causal mask, and emits per-row seed metadata
    /// so each row's later verify rounds and drafter cross-attention stay
    /// byte-identical to that row's standalone B = 1 run.
    ///
    /// ## Greedy-parity mechanism (left-padding uniform shift)
    ///
    /// Row `r` (length `L_r`) is placed at padded indices `[lp_r, max_len)`
    /// with `lp_r = max_len - L_r`. Proportional RoPE is applied with a single
    /// scalar offset (`0` at prefill), so a token at padded index `i` is rotated
    /// to absolute position `i`. Every real token of row `r` is therefore
    /// shifted right by the same constant `lp_r` versus a standalone run that
    /// would place it at `[0, L_r)`. Because attention depends only on the
    /// *relative* RoPE phase `q_pos - k_pos`, and both query and key of any
    /// in-row pair carry the same `+lp_r` shift, every in-row relative phase —
    /// and thus every attention score and every argmax — is preserved. The
    /// left-padding mask blocks all real queries from attending to the leading
    /// padding keys, so padding contributes nothing. In subsequent verify
    /// rounds the shared physical cache offset is `max_len` for every row, so
    /// each row's appended verify tokens land at absolute positions
    /// `max_len + j`, again a uniform `+lp_r` shift versus the standalone
    /// `L_r + j` — parity is preserved end-to-end.
    ///
    /// Consequently the seed's `kv_offset_per_row` and `bonus_position_per_row`
    /// are **uniform** (`max_len` / `max_len - 1`, the padded frame), while
    /// `left_padding_per_row[r] = lp_r` and `kv_valid_len_per_row[r] = L_r` are
    /// per-row. The drafter's `set_shared_kv_batched` consumes the per-row
    /// left-padding / valid-length to left-roll each row's shared K/V into the
    /// prefix-valid layout (`normalize_batched_shared_kv_states`) and mask the
    /// padded tail.
    ///
    /// ## Eligibility
    ///
    /// Restricted to `max_prompt_len <= sliding_window` (non-capped
    /// RotatingKVCache regime). In that regime the windowed left-padding mask is
    /// byte-identical to the plain left-padding mask, so the single mask passed
    /// to the forward is correct for both full-attention and sliding-window
    /// layers. Outside it, the function returns `Err(DrafterError::DraftFailed)`
    /// so the burst driver declines the window and the scheduler re-enqueues the
    /// rows for per-row B = 1 service.
    fn prefill_and_seed_batched_ragged(
        &self,
        prompt_tokens_per_row: &[Vec<i32>],
        sampler: &SamplingConfig,
    ) -> Result<(Vec<i32>, MtpBatchedVerifyOutput), DrafterError> {
        let LeftPaddedPrefill {
            arr: prompt_arr,
            max_len,
            left_padding,
            valid_len,
        } = Self::left_padded_input(prompt_tokens_per_row, self.batch_size)?;

        // Eligibility: only the non-capped sliding-window regime is supported.
        let sliding_window = self.wrapper.sliding_window_value();
        if sliding_window > 0 && max_len > sliding_window {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Gemma4 ragged batched MTP prefill declined: max_prompt_len {max_len} \
                     exceeds sliding_window {sliding_window} (capped RotatingKVCache regime is \
                     not supported by the windowed left-padding mask); falling back to per-row \
                     B=1 service"
                ),
            });
        }

        // Build the per-row left-padding causal mask `[B, 1, max_len, max_len]`.
        // In the eligible non-capped regime this is correct for both the
        // full-attention and sliding-window layers (proven equivalent to the
        // windowed left-padding mask in `mlxcel_core::utils`).
        let left_padding_i32: Vec<i32> = left_padding.iter().map(|&p| p as i32).collect();
        let mask = mlxcel_core::utils::create_causal_mask_with_left_padding(
            max_len as i32,
            0,
            &left_padding_i32,
        );

        let (logits, hidden_full, shared_kv) = self.batched_sink_forward_with_mask(
            &prompt_arr,
            Some(mask.as_ref().unwrap()),
            // Prefill passes an explicit left-padding mask above, so the
            // `mask == None` left-padding plumbing is not needed here.
            None,
            // Prefill is at offset 0: no stale tail, so no per-row valid end.
            None,
        );
        self.enable_rotating_cache_buffer();

        // Every row's last real token sits at the shared padded index
        // `max_len - 1` (left-padding right-aligns the prompts), so the seed
        // bonus is sampled from the uniform last position — the same op
        // sequence as the equal-length path. At temperature 0 each row's bonus
        // is byte-identical to that row's standalone B = 1 seed.
        let (token_arr, _) = sample_token_optimized(&logits, sampler, &[]);
        mlxcel_core::eval(&token_arr);
        let first_bonus_per_row = scalar_tokens_per_row(&token_arr, self.batch_size);

        let next_hidden = self.last_position_draft_hidden(&hidden_full);

        // Stash per-row logical valid lengths (= unpadded prompt lengths) and
        // the constant per-row left-padding for the verify-round bookkeeping in
        // `verify_finalize_batched`. The physical/padded anchors are uniform at
        // `max_len`; only `kv_valid_len` and `left_padding` are per-row.
        {
            let mut positions = self.positions.borrow_mut();
            *positions = valid_len.clone();
        }
        {
            let mut lp = self.left_padding.borrow_mut();
            *lp = left_padding.clone();
        }
        let seed = self.build_seed_metadata(next_hidden, shared_kv, &valid_len, &left_padding);
        Ok((first_bonus_per_row, seed))
    }

    /// Assemble the per-row seed/finalize metadata from each row's logical
    /// `kv_valid_len` and constant `left_padding`, using the uniform shifted-
    /// frame formula that keeps both the equal-length and ragged paths
    /// byte-identical to standalone B = 1 runs:
    ///
    /// - `kv_valid_len[r]` — count of real K/V entries (prompt + accepted).
    /// - `left_padding[r]` — leading padding columns (constant per burst).
    /// - `kv_offset[r] = left_padding[r] + kv_valid_len[r]` — physical/padded
    ///   absolute offset (the drafter's RoPE frame for row `r`).
    /// - `bonus_position[r] = kv_offset[r] - 1` — the row's frozen bonus anchor.
    ///
    /// For the equal-length path `left_padding[r] == 0`, so this reduces to
    /// `kv_offset == kv_valid_len == positions[r]` and `bonus_position ==
    /// positions[r] - 1` — exactly the legacy equal-length metadata.
    fn build_seed_metadata(
        &self,
        next_hidden: UniquePtr<MlxArray>,
        next_shared_kv: Vec<UniquePtr<MlxArray>>,
        kv_valid_len: &[usize],
        left_padding: &[usize],
    ) -> MtpBatchedVerifyOutput {
        let (kv_offset_per_row, bonus_position_per_row) =
            seed_anchors_from_valid_len(kv_valid_len, left_padding);
        MtpBatchedVerifyOutput {
            next_hidden,
            next_shared_kv,
            kv_offset_per_row,
            bonus_position_per_row,
            kv_valid_len_per_row: kv_valid_len.to_vec(),
            left_padding_per_row: left_padding.to_vec(),
        }
    }
}

impl<'a> MtpTarget for Gemma4MtpBatchedTargetAdapter<'a> {
    // The B = 1 surface is required by the trait but never driven on the
    // batched adapter — the batched round-loop driver only calls the
    // `*_batched` methods. We panic rather than silently mis-routing so a
    // wiring bug surfaces loudly in tests. (`token_history` is part of the B = 1 signature since and `logprobs_config` since the batched path's history-aware and logprobs-aware sampling is gated at the scheduler's window collector — see `prefill_and_seed_batched` below.)
    fn prefill_and_seed(
        &self,
        _prompt_tokens: &[i32],
        _sampler: &SamplingConfig,
        _token_history: &[i32],
        _logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        panic!(
            "Gemma4MtpBatchedTargetAdapter must be driven through the B > 1 \
             (`*_batched`) MtpTarget methods, not the B = 1 surface"
        );
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        // Embedding is row-independent; the batched drafter calls this
        // per row. Delegate to the same wrapper accessor the B = 1
        // adapter uses.
        let input_ids = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        <Gemma4Wrapper as mlxcel_core::generate::LanguageModel>::embed_tokens(
            self.wrapper,
            &input_ids,
        )
        .expect("Gemma4Wrapper exposes its embed_tokens table")
    }

    fn verify_forward(
        &self,
        _verify_input: &[i32],
        _sampler: &SamplingConfig,
        _logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> VerifyForwardOutput {
        panic!(
            "Gemma4MtpBatchedTargetAdapter must be driven through \
             verify_forward_batched, not the B = 1 verify_forward"
        );
    }

    fn verify_finalize(
        &self,
        _accepted: usize,
        _block_size: usize,
        _captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        panic!(
            "Gemma4MtpBatchedTargetAdapter must be driven through \
             verify_finalize_batched, not the B = 1 verify_finalize"
        );
    }

    fn num_layers(&self) -> usize {
        self.wrapper.num_layers_value()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.wrapper.eos_token_ids_value()
    }

    // --- Batched surface (the actual implementation) -----------------

    fn prefill_and_seed_batched(
        &self,
        prompt_tokens_per_row: &[Vec<i32>],
        sampler: &SamplingConfig,
    ) -> Result<(Vec<i32>, MtpBatchedVerifyOutput), DrafterError> {
        // Route by prompt-length uniformity. Equal-length rows take the
        // original rectangular [B, L] prefill (mask derived internally from
        // offset 0); variable-length rows take the left-padding path.
        let first_len = prompt_tokens_per_row.first().map(Vec::len).unwrap_or(0);
        let is_ragged = prompt_tokens_per_row
            .iter()
            .any(|row| row.len() != first_len);
        if is_ragged {
            return self.prefill_and_seed_batched_ragged(prompt_tokens_per_row, sampler);
        }

        // Build the rectangular [B, L] prompt batch.
        let (prompt_arr, prompt_len) =
            Self::rectangular_input(prompt_tokens_per_row, self.batch_size)?;

        // Sink-aware prefill. The adapter's [B, ...] cache now holds
        // `prompt_len` entries per row.
        let (logits, hidden_full, shared_kv) = self.batched_sink_forward(&prompt_arr);
        self.enable_rotating_cache_buffer();

        // Sample the per-row first bonus. `sample_token_optimized`
        // internally slices the last position of the [B, prompt_len,
        // vocab] logits down to [B, vocab] and returns one token per
        // row — the exact op sequence the B = 1 adapter
        // (`Gemma4MtpTargetAdapter::prefill_and_seed`) uses, so at
        // temperature 0 each row's bonus is byte-identical to the B = 1
        // seed sample.
        let (token_arr, _) = sample_token_optimized(&logits, sampler, &[]);
        mlxcel_core::eval(&token_arr);
        let first_bonus_per_row = scalar_tokens_per_row(&token_arr, self.batch_size);

        // Build the seed MtpBatchedVerifyOutput. After prefill every row
        // has `prompt_len` cache entries; the bonus token is NOT yet in
        // the cache, so each row's `kv_offset` / `kv_valid_len` is
        // `prompt_len`, while the drafter position is `prompt_len - 1`.
        // With equal-length prompts there is no left-padding.
        let next_hidden = self.last_position_draft_hidden(&hidden_full);
        let prompt_len_usize = prompt_len as usize;
        {
            let mut positions = self.positions.borrow_mut();
            *positions = vec![prompt_len_usize; self.batch_size];
        }
        {
            let mut lp = self.left_padding.borrow_mut();
            *lp = vec![0; self.batch_size];
        }
        let valid_len = self.positions.borrow().clone();
        let left_padding = self.left_padding.borrow().clone();
        let seed = self.build_seed_metadata(next_hidden, shared_kv, &valid_len, &left_padding);
        Ok((first_bonus_per_row, seed))
    }

    fn verify_forward_batched(
        &self,
        verify_input_per_row: &[Vec<i32>],
        _sampler: &SamplingConfig,
    ) -> Result<MtpBatchedVerifyForwardOutput, DrafterError> {
        // Build the rectangular [B, block_size] verify batch.
        let (verify_arr, width) = Self::rectangular_input(verify_input_per_row, self.batch_size)?;

        // Per-row valid-length tail exclusion (issue #163). Compute each row's
        // logical valid key end (`left_padding[r] + positions[r]`) BEFORE the
        // forward: `positions` here reflects the state after the previous
        // round's finalize, i.e. the pre-window logical end. After divergent
        // accepts a shorter row's valid end lags the physical cache offset, so
        // threading it lets the verify mask exclude that row's stale
        // `[valid_end, offset)` rejected-draft / zeroed tail (this is the sole
        // producer of `Some(per_row_valid_end)`). The first verify round after
        // prefill is uniform (`valid_end == offset` for every row), a
        // byte-identical no-op. `left_padding` is also threaded so a ragged
        // burst keeps masking each row's resident `[0, left_padding[r])` keys;
        // it is all-zero (→ None) for the equal-length path.
        let (left_padding, valid_ends): (Vec<i32>, Vec<i32>) = {
            let lp = self.left_padding.borrow();
            let pos = self.positions.borrow();
            let lp_i32: Vec<i32> = lp.iter().map(|&p| p as i32).collect();
            let ve: Vec<i32> = lp
                .iter()
                .zip(pos.iter())
                .map(|(&l, &p)| (l + p) as i32)
                .collect();
            (lp_i32, ve)
        };
        let lp_ref: Option<&[i32]> = if left_padding.iter().any(|&p| p > 0) {
            Some(&left_padding)
        } else {
            None
        };

        // Sink-aware verify forward. The [B, ...] cache grows by `width`
        // entries per row; `verify_finalize_batched` trims it back.
        let (logits, hidden_full, shared_kv) =
            self.batched_sink_forward_with_mask(&verify_arr, None, lp_ref, Some(&valid_ends));

        // Greedy-parity: per-row argmax over the [B, width, vocab]
        // logits. At temperature 0 this matches the drafter-less
        // target's own argmax extension, identically to the B = 1
        // adapter's `argmax_per_position`.
        let target_tokens_per_row = Self::argmax_per_row(&logits, self.batch_size as i32, width);

        // Stash the captured hidden + shared K/V for finalize. Index 0 =
        // hidden_full ([B, width, hidden]), indices 1.. = shared K/V
        // slabs in [k_full, v_full, k_swa, v_swa] order — same
        // convention as the B = 1 adapter's `VerifyCaptured`.
        let mut tensors: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(5);
        tensors.push(hidden_full);
        tensors.extend(shared_kv);

        Ok(MtpBatchedVerifyForwardOutput {
            target_tokens_per_row,
            captured: VerifyCaptured {
                tensors,
                scalars: Vec::new(),
            },
        })
    }

    fn verify_finalize_batched(
        &self,
        accepted_per_row: &[usize],
        block_size: usize,
        captured: VerifyCaptured,
    ) -> Result<MtpBatchedVerifyOutput, DrafterError> {
        if accepted_per_row.len() != self.batch_size {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Gemma4 batched MTP target: verify_finalize_batched expected \
                     {} accept counts, got {}",
                    self.batch_size,
                    accepted_per_row.len()
                ),
            });
        }

        // Pop the captured hidden out (index 0); indices 1.. are the
        // pre-slice shared K/V slabs.
        let mut tensors = captured.tensors;
        if tensors.is_empty() {
            return Err(DrafterError::DraftFailed {
                reason: "Gemma4 batched MTP target: VerifyCaptured must carry the \
                         hidden tensor at index 0"
                    .to_string(),
            });
        }
        let hidden_full = tensors.remove(0);
        let next_hidden = self
            .draft_hidden_at_positions_batched(hidden_full.as_ref().unwrap(), accepted_per_row)?;

        // Per-row rollback on the adapter's own [B, ...] cache.
        //
        // Uniform rounds (every row's logical valid end still equals the
        // shared physical write base, which is always true until the first divergent
        // accept) keep the legacy contract:
        // `rollback_speculative_cache_explicit` trims by `block_size -
        // max(accepted) - 1` and per-row zeros the tails for rows below max.
        //
        // Divergent rounds (issue #203) instead COMPACT: each row's accepted
        // window K/V moves down to the row's logical valid end so physical
        // slot == logical position is restored for every row, exactly like
        // the row's standalone B = 1 run. The captured drafter slabs get the
        // same per-row compaction so `normalize_batched_shared_kv_states`'s
        // contiguous-prefix assumption holds.
        let accepted_i32: Vec<i32> = accepted_per_row.iter().map(|&a| a as i32).collect();
        let width = block_size as i32;
        let ve_pre: Option<Vec<i32>> = {
            let lp = self.left_padding.borrow();
            let pos = self.positions.borrow();
            if lp.len() == self.batch_size && pos.len() == self.batch_size {
                Some(
                    lp.iter()
                        .zip(pos.iter())
                        .map(|(&l, &p)| (l + p) as i32)
                        .collect(),
                )
            } else {
                None
            }
        };
        let (post_rollback_kv_offset, next_shared_kv) = {
            let mut caches = self.caches.borrow_mut();
            let o_pre =
                crate::models::gemma4::first_present_cache_offset(caches.as_slice()) - width;
            let divergent = !crate::models::gemma4::mtp_divergent_fix_disabled()
                && ve_pre
                    .as_ref()
                    .map(|ve| ve.iter().any(|&v| v != o_pre))
                    .unwrap_or(false);
            if divergent {
                let ve_pre = ve_pre.as_ref().expect("divergent implies Some(ve_pre)");
                tracing::debug!(
                    o_pre,
                    ?ve_pre,
                    accepted = ?accepted_i32,
                    "gemma4 batched MTP divergent finalize (compacting rollback)"
                );
                let o_post = self
                    .wrapper
                    .rollback_speculative_cache_divergent(&mut caches, ve_pre, &accepted_i32, width)
                    .map_err(|e| DrafterError::DraftFailed {
                        reason: format!("Gemma4 batched MTP divergent rollback failed: {e}"),
                    })?;
                let next_shared_kv =
                    Self::compact_shared_kv_batched(tensors, ve_pre, &accepted_i32, o_pre, o_post)?;
                (o_post.max(0) as usize, next_shared_kv)
            } else {
                self.wrapper
                    .rollback_speculative_cache_explicit(&mut caches, &accepted_i32, width)
                    .map_err(|e| DrafterError::DraftFailed {
                        reason: format!("Gemma4 batched MTP rollback failed: {e}"),
                    })?;
                let offset =
                    first_cache_offset(caches.as_mut_slice(), "full_attention").max(0) as usize;
                // Slice the captured shared K/V to the post-rollback length.
                // The global trim amount is `block_size - max(accepted) - 1`;
                // this matches the cache trim above.
                let max_accepted = accepted_per_row.iter().copied().max().unwrap_or(0);
                let rejected = block_size.saturating_sub(max_accepted).saturating_sub(1);
                (offset, Self::slice_shared_kv_batched(tensors, rejected))
            }
        };

        // Post-rollback per-row logical metadata. The physical cache offset
        // remains the global post-rollback max (used above for slicing the
        // captured K/V slabs), but each row advanced by its own
        // accepted+bonus count. `self.positions` tracks each row's logical
        // valid length (`kv_valid_len`); advance it by `accepted + 1`. The
        // constant per-row `left_padding` (0 for equal-length, `max_prompt_len
        // - prompt_len[r]` for ragged) was stashed at prefill and persists
        // unchanged — the prompt's leading padding is never trimmed.
        {
            let mut positions = self.positions.borrow_mut();
            if positions.len() != self.batch_size {
                *positions = vec![post_rollback_kv_offset; self.batch_size];
            }
            for (row_pos, &accepted) in positions.iter_mut().zip(accepted_per_row) {
                *row_pos += accepted + 1;
            }
        }
        let kv_valid_len = self.positions.borrow().clone();
        let left_padding = self.left_padding.borrow().clone();
        // `build_seed_metadata` derives `kv_offset = left_padding + kv_valid_len`
        // and `bonus_position = kv_offset - 1`, threading the row's shifted RoPE
        // frame so the drafter masks zeroed tails and rotates queries at the
        // row's own bonus anchor. For equal-length rows (`left_padding == 0`)
        // this reduces to the legacy `kv_offset == kv_valid_len` metadata.
        let out =
            self.build_seed_metadata(next_hidden, next_shared_kv, &kv_valid_len, &left_padding);
        Ok(out)
    }
}

/// Batched MTP target adapter for the Gemma 4 VLM wrapper.
///
/// Reuses [`Gemma4MtpBatchedTargetAdapter`] internally by delegating to
/// the VLM wrapper's inner text model. Same rationale as the B = 1
/// [`Gemma4VLMtpTargetAdapter`]: vision features are fully prefilled
/// before the MTP round loop begins, so the round loop only interacts
/// with the text backbone.
pub struct Gemma4VLMtpBatchedTargetAdapter<'a> {
    inner: Gemma4MtpBatchedTargetAdapter<'a>,
}

impl<'a> Gemma4VLMtpBatchedTargetAdapter<'a> {
    /// Construct a batched adapter routing every trait call through the
    /// inner text model's own `[B, ...]` cache.
    pub fn new(vlm: &'a crate::vision::Gemma4VLModel, batch_size: usize) -> Self {
        Self::new_with_block_size(vlm, batch_size, 4)
    }

    /// Construct a batched VLM MTP adapter with the effective requested block
    /// size.
    pub fn new_with_block_size(
        vlm: &'a crate::vision::Gemma4VLModel,
        batch_size: usize,
        block_size: usize,
    ) -> Self {
        Self {
            inner: Gemma4MtpBatchedTargetAdapter::new_with_block_size(
                &vlm.text_model,
                batch_size,
                block_size,
            ),
        }
    }

    /// Batch size accessor (test / diagnostic).
    pub fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }
}

impl<'a> MtpTarget for Gemma4VLMtpBatchedTargetAdapter<'a> {
    // The B = 1 surface forwards to the inner batched adapter, whose B = 1
    // stubs panic — the batched VLM adapter is only ever driven through
    // the `*_batched` methods. `token_history` and
    // `logprobs_config` are forwarded verbatim so the
    // signature matches the trait even though the inner panics.
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        self.inner
            .prefill_and_seed(prompt_tokens, sampler, token_history, logprobs_config)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        self.inner.embed_token(token_id)
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> VerifyForwardOutput {
        self.inner
            .verify_forward(verify_input, sampler, logprobs_config)
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        self.inner.verify_finalize(accepted, block_size, captured)
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.inner.eos_token_ids()
    }

    fn prefill_and_seed_batched(
        &self,
        prompt_tokens_per_row: &[Vec<i32>],
        sampler: &SamplingConfig,
    ) -> Result<(Vec<i32>, MtpBatchedVerifyOutput), DrafterError> {
        self.inner
            .prefill_and_seed_batched(prompt_tokens_per_row, sampler)
    }

    fn verify_forward_batched(
        &self,
        verify_input_per_row: &[Vec<i32>],
        sampler: &SamplingConfig,
    ) -> Result<MtpBatchedVerifyForwardOutput, DrafterError> {
        self.inner
            .verify_forward_batched(verify_input_per_row, sampler)
    }

    fn verify_finalize_batched(
        &self,
        accepted_per_row: &[usize],
        block_size: usize,
        captured: VerifyCaptured,
    ) -> Result<MtpBatchedVerifyOutput, DrafterError> {
        self.inner
            .verify_finalize_batched(accepted_per_row, block_size, captured)
    }
}

/// Batched MTP target adapter for the Gemma 4 Unified wrapper.
///
/// Reuses [`Gemma4MtpBatchedTargetAdapter`] internally by delegating to the
/// Unified wrapper's inner text model. Same rationale as the B = 1
/// [`Gemma4UnifiedMtpTargetAdapter`]: any multimodal features are fully
/// prefilled before the MTP round loop begins, so the round loop only
/// interacts with the text backbone.
pub struct Gemma4UnifiedMtpBatchedTargetAdapter<'a> {
    inner: Gemma4MtpBatchedTargetAdapter<'a>,
}

impl<'a> Gemma4UnifiedMtpBatchedTargetAdapter<'a> {
    /// Construct a batched adapter routing every trait call through the inner
    /// text model's own `[B, ...]` cache.
    pub fn new(unified: &'a crate::vision::Gemma4UnifiedModel, batch_size: usize) -> Self {
        Self::new_with_block_size(unified, batch_size, 4)
    }

    /// Construct a batched Unified MTP adapter with the effective requested
    /// block size.
    pub fn new_with_block_size(
        unified: &'a crate::vision::Gemma4UnifiedModel,
        batch_size: usize,
        block_size: usize,
    ) -> Self {
        Self {
            inner: Gemma4MtpBatchedTargetAdapter::new_with_block_size(
                &unified.text_model,
                batch_size,
                block_size,
            ),
        }
    }

    /// Batch size accessor (test / diagnostic).
    pub fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }
}

impl<'a> MtpTarget for Gemma4UnifiedMtpBatchedTargetAdapter<'a> {
    // The B = 1 surface forwards to the inner batched adapter, whose B = 1
    // stubs panic — the batched Unified adapter is only ever driven through
    // the `*_batched` methods.
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        self.inner
            .prefill_and_seed(prompt_tokens, sampler, token_history, logprobs_config)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        self.inner.embed_token(token_id)
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> VerifyForwardOutput {
        self.inner
            .verify_forward(verify_input, sampler, logprobs_config)
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        self.inner.verify_finalize(accepted, block_size, captured)
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.inner.eos_token_ids()
    }

    fn prefill_and_seed_batched(
        &self,
        prompt_tokens_per_row: &[Vec<i32>],
        sampler: &SamplingConfig,
    ) -> Result<(Vec<i32>, MtpBatchedVerifyOutput), DrafterError> {
        self.inner
            .prefill_and_seed_batched(prompt_tokens_per_row, sampler)
    }

    fn verify_forward_batched(
        &self,
        verify_input_per_row: &[Vec<i32>],
        sampler: &SamplingConfig,
    ) -> Result<MtpBatchedVerifyForwardOutput, DrafterError> {
        self.inner
            .verify_forward_batched(verify_input_per_row, sampler)
    }

    fn verify_finalize_batched(
        &self,
        accepted_per_row: &[usize],
        block_size: usize,
        captured: VerifyCaptured,
    ) -> Result<MtpBatchedVerifyOutput, DrafterError> {
        self.inner
            .verify_finalize_batched(accepted_per_row, block_size, captured)
    }
}

/// Materialise a per-row `Vec<i32>` from a `[B]` / `[B, 1]` token tensor
/// produced by `sample_token_optimized`. The sampler returns one token
/// per batch row; we extract them cell-by-cell.
fn scalar_tokens_per_row(token_arr: &MlxArray, batch_size: usize) -> Vec<i32> {
    let shape = mlxcel_core::array_shape(token_arr);
    // `sample_token_optimized` may return `[B]`, `[B, 1]`, or `[B, 1, 1]`
    // depending on the input rank; reshape to a flat `[B]` so the
    // per-row extraction is uniform.
    let flat = mlxcel_core::reshape(token_arr, &[batch_size as i32]);
    debug_assert!(
        shape.iter().product::<i32>() == batch_size as i32,
        "sample output must carry exactly batch_size tokens, got shape {shape:?}"
    );
    let mut out: Vec<i32> = Vec::with_capacity(batch_size);
    for r in 0..batch_size as i32 {
        let cell = mlxcel_core::slice(&flat, &[r], &[r + 1]);
        let scalar = mlxcel_core::reshape(&cell, &[]);
        out.push(mlxcel_core::item_i32(&scalar));
    }
    out
}

#[cfg(test)]
#[path = "gemma4_mtp_target_tests.rs"]
mod tests;
