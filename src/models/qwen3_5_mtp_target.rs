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

//! MTP target adapter for the Qwen 3.5 family.
//!
//! Glue layer that wires [`crate::models::qwen3_5::Qwen35Model`] (and its VLM
//! wrapper [`crate::vision::Qwen35VLModel`]) to the
//! [`mlxcel_core::speculative::mtp::target::MtpTarget`] trait, the same role
//! [`crate::models::gemma4_mtp_target`] plays for Gemma 4.
//!
//! ## Cache ownership
//!
//! `MtpTarget`'s methods are `&self` with no cache parameter, while the
//! Qwen 3.5 speculative hooks (`forward_speculative`,
//! `rollback_speculative_cache`) take a caller-owned `&mut [Qwen3NextCache]`.
//! The adapter bridges the two through the model's **sequence-routed**
//! wrappers (`*_for_sequence`), which run a hook over one sequence's caches.
//! A scheduler sequence's caches are the layer state its pool entry holds
//! (issue #3254), which the adapter borrows for its lifetime; the offline CLI
//! passes none and runs on the model's fallback slot. The adapter owns no
//! tensor state either way, so it stays a cheaply reconstructible view, the
//! shape the tick-cooperative slice path depends on (it rebuilds the adapter
//! every scheduler tick).
//!
//! ## Hidden-state semantics
//!
//! The Qwen MTP drafter consumes the target's **post-final-norm** hidden:
//! upstream's `return_hidden` appends the model output (which ends in
//! `self.norm(h)`), and Qwen has no `speculative_draft_hidden` hook. The
//! captured per-layer hidden here is pre-norm, so every hidden handed out
//! (seed, per-round `next_hidden`, and the full-block
//! `verify_hidden_full` for the stateful drafter's history hooks) goes
//! through [`Qwen35Model::apply_final_norm`] first.
//!
//! ## Shared K/V
//!
//! Qwen 3.5 has no shared-K/V concept (the drafter owns its own per-layer KV
//! cache), so `next_shared_kv` is always empty and the drafter's
//! `set_shared_kv` consumes only the position metadata, mirroring upstream.
//!
//! ## Scope
//!
//! B = 1 only. The batched `MtpTarget` methods keep their erroring trait
//! defaults, so a B > 1 dispatch declines with the existing named error and
//! the burst driver falls back to classic decode.

use std::cell::RefCell;

use mlxcel_core::cache::{LayerState, SequenceId};
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::{LogprobsConfig, TokenLogprobData};
use mlxcel_core::speculative::mtp::target::{
    MtpTarget, MtpVerifyOutput, VerifyCaptured, VerifyForwardOutput,
};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gemma4_mtp_target::Gemma4MtpTargetAdapter;
use crate::models::qwen3_5::{GdnRollbackSnapshot, Qwen35Model};

/// Flatten the post-norm verify hidden plus the GDN rollback snapshots into
/// the opaque [`VerifyCaptured`] slots.
///
/// Encoding (pinned by `captured_round_trip_preserves_gdn_snapshots`):
///
/// - `scalars[0]` = number of snapshots `N`; then per snapshot `i`:
///   `scalars[1 + 2i]` = `layer_idx`, `scalars[2 + 2i]` = `has_init_state`
///   (0/1).
/// - `tensors[0]` = post-norm hidden `[1, block, H]`; then per snapshot, in
///   snapshot order: `q, k, v, a, b, conv_input`, plus `init_state` when the
///   flag is set. A mis-ordered flatten produces subtly wrong recurrent
///   state rather than a crash, so any change here must update the
///   round-trip test in lockstep.
fn flatten_captured(
    hidden_post_norm: UniquePtr<MlxArray>,
    gdn_states: Vec<GdnRollbackSnapshot>,
) -> VerifyCaptured {
    let mut scalars: Vec<i32> = Vec::with_capacity(1 + 2 * gdn_states.len());
    scalars.push(gdn_states.len() as i32);
    let mut tensors: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(1 + 7 * gdn_states.len());
    tensors.push(hidden_post_norm);
    for snap in gdn_states {
        scalars.push(snap.layer_idx as i32);
        scalars.push(snap.init_state.is_some() as i32);
        tensors.push(snap.q);
        tensors.push(snap.k);
        tensors.push(snap.v);
        tensors.push(snap.a);
        tensors.push(snap.b);
        tensors.push(snap.conv_input);
        if let Some(init_state) = snap.init_state {
            tensors.push(init_state);
        }
    }
    VerifyCaptured { tensors, scalars }
}

/// Inverse of [`flatten_captured`]. Panics on a malformed encoding — the
/// captured state never crosses a serialization boundary, so a mismatch is a
/// programming error in this module, not an input condition.
fn unflatten_captured(captured: VerifyCaptured) -> (UniquePtr<MlxArray>, Vec<GdnRollbackSnapshot>) {
    let VerifyCaptured { tensors, scalars } = captured;
    let mut tensors = tensors.into_iter();
    let hidden = tensors
        .next()
        .expect("VerifyCaptured must carry the hidden tensor at index 0");
    let n = *scalars
        .first()
        .expect("VerifyCaptured scalars must carry the snapshot count") as usize;
    assert_eq!(
        scalars.len(),
        1 + 2 * n,
        "VerifyCaptured scalars must carry (layer_idx, has_init) per snapshot"
    );
    let mut gdn_states = Vec::with_capacity(n);
    for i in 0..n {
        let layer_idx = scalars[1 + 2 * i] as usize;
        let has_init = scalars[2 + 2 * i] != 0;
        let q = tensors.next().expect("snapshot q");
        let k = tensors.next().expect("snapshot k");
        let v = tensors.next().expect("snapshot v");
        let a = tensors.next().expect("snapshot a");
        let b = tensors.next().expect("snapshot b");
        let conv_input = tensors.next().expect("snapshot conv_input");
        let init_state = has_init.then(|| tensors.next().expect("snapshot init_state"));
        gdn_states.push(GdnRollbackSnapshot {
            layer_idx,
            q,
            k,
            v,
            a,
            b,
            init_state,
            conv_input,
        });
    }
    assert!(
        tensors.next().is_none(),
        "VerifyCaptured tensor count disagrees with the scalar encoding"
    );
    (hidden, gdn_states)
}

/// MTP target adapter binding a [`Qwen35Model`] to a per-sequence cache slot.
///
/// Constructed by the server dispatch (or the offline CLI with
/// `seq_id = None`, which selects the model's internal fallback slot) and
/// consumed by [`mlxcel_core::speculative::mtp::MtpGenerator`] over the round
/// loop. The adapter does NOT own the model and holds no tensor state.
pub struct Qwen35MtpTargetAdapter<'a> {
    model: &'a Qwen35Model,
    /// Where the sequence's caches live (issue #3254).
    state: StateSource<'a>,
    /// Adopted prompt-cache prefix length (issue #518). When `> 0`,
    /// `prefill_and_seed` forwards only `prompt_tokens[prefill_start_offset..]`
    /// over the KV already restored into the sequence slot. The seed then
    /// advertises no full prompt hidden (`verify_hidden_full = None`), so the
    /// stateful drafter skips its prompt prefill and builds history from
    /// accepted tokens only — reduced draft context, identical correctness.
    prefill_start_offset: usize,
}

/// The caches an adapter runs over.
enum StateSource<'a> {
    /// The model's id-less fallback slot.
    Fallback,
    /// One scheduler sequence's layer state, borrowed from the pool for the
    /// life of the adapter. In a `RefCell` because `MtpTarget` is `&self`-only.
    Pool(RefCell<&'a mut [LayerState]>),
    /// A scheduler sequence the pool held no layer state for.
    Missing(SequenceId),
}

impl<'a> Qwen35MtpTargetAdapter<'a> {
    /// An adapter over the model's internal fallback slot: the offline CLI
    /// shape, which runs one sequence and holds no pool.
    pub fn fallback(model: &'a Qwen35Model) -> Self {
        Self {
            model,
            state: StateSource::Fallback,
            prefill_start_offset: 0,
        }
    }

    /// An adapter over scheduler sequence `seq_id`, whose layer state the
    /// caller borrows from the pool (`CachePool::layer_state_mut`).
    ///
    /// A sequence with no layer state is kept as such and refused at the
    /// first call. It never falls back to the shared slot: that would run
    /// one request on another's state.
    pub fn for_sequence(
        model: &'a Qwen35Model,
        seq_id: SequenceId,
        state: Option<&'a mut [LayerState]>,
    ) -> Self {
        Self {
            model,
            state: match state {
                Some(state) => StateSource::Pool(RefCell::new(state)),
                None => StateSource::Missing(seq_id),
            },
            prefill_start_offset: 0,
        }
    }

    /// Run `f` over the sequence's layer state, or `None` for the fallback
    /// slot.
    fn with_state<R>(&self, f: impl FnOnce(Option<&mut [LayerState]>) -> R) -> R {
        match &self.state {
            StateSource::Pool(state) => f(Some(&mut **state.borrow_mut())),
            StateSource::Fallback => f(None),
            StateSource::Missing(seq_id) => panic!(
                "Qwen 3.5 MTP target: sequence {seq_id} holds no scheduler-owned layer state"
            ),
        }
    }

    /// Set the adopted prompt-cache prefix length (issue #518). See the
    /// field doc for the drafter-history consequence.
    #[must_use]
    pub fn with_prefill_start_offset(mut self, prefill_start_offset: usize) -> Self {
        self.prefill_start_offset = prefill_start_offset;
        self
    }

    /// Capture-layer list for the verify pass: the last decoder layer only.
    fn capture_last_layer(&self) -> [usize; 1] {
        [self.model.num_layers().saturating_sub(1)]
    }
}

impl<'a> MtpTarget for Qwen35MtpTargetAdapter<'a> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &LogprobsConfig,
    ) -> (i32, MtpVerifyOutput, Option<TokenLogprobData>) {
        // Suffix-only prefill for an adopted prompt-cache prefix (issue
        // #518): the sequence slot already holds KV for `[..offset]`
        // (restored by the scheduler's APC snapshot adoption), so forward
        // only the suffix; RoPE continues from the cache's restored offset.
        let offset = self.prefill_start_offset.min(prompt_tokens.len());
        let forward_tokens = &prompt_tokens[offset..];
        let prompt_arr =
            mlxcel_core::from_slice_i32(forward_tokens, &[1, forward_tokens.len() as i32]);

        // Batched-causal prefill, byte-identical to the classic path (NOT
        // the per-position verify attention — see
        // `forward_prefill_with_last_hidden`'s doc for why that matters for
        // first-token parity and prefill cost).
        let (logits, hidden_pre) = self.with_state(|state| {
            self.model
                .forward_prefill_with_last_hidden_for_sequence(&prompt_arr, state)
        });

        // First bonus from the last-position logits, with the
        // history-dependent-penalty context and the sampler's token bias —
        // the same computation the classic decode path's first token uses.
        let logits_shape = mlxcel_core::array_shape(&logits);
        let last_pos = logits_shape[1] - 1;
        let vocab = logits_shape[2];
        let last_logits = mlxcel_core::slice(
            &logits,
            &[0, last_pos, 0],
            &[logits_shape[0], last_pos + 1, vocab],
        );
        let (token_arr, adjusted_logits) =
            mlxcel_core::sampling::sample_token_optimized(&last_logits, sampler, token_history);
        mlxcel_core::eval(&token_arr);
        let first_bonus = mlxcel_core::item_i32(&token_arr);
        let first_bonus_lp =
            mlxcel_core::sampling::compute_logprobs(&adjusted_logits, first_bonus, logprobs_config);

        // Post-final-norm hidden for the drafter: the seed `next_hidden` is
        // the last forwarded position; the full block feeds the stateful
        // drafter's prompt prefill, but ONLY when it covers the whole prompt
        // (an adopted-prefix suffix does not).
        let hidden_post = self.model.apply_final_norm(&hidden_pre);
        let next_hidden = Gemma4MtpTargetAdapter::last_position_hidden(&hidden_post);
        let verify_hidden_full = (offset == 0).then_some(hidden_post);

        // Absolute post-prefill cache offset (= prompt length when cold, or
        // restored offset + suffix length under adoption).
        let kv_offset = self
            .with_state(|state| self.model.speculative_cache_offset_for_sequence(state))
            .max(0) as usize;
        let bonus_position = kv_offset.saturating_sub(1);

        let seed = MtpVerifyOutput {
            next_hidden,
            // Qwen has no shared-K/V concept; the drafter ignores tensors.
            next_shared_kv: Vec::new(),
            kv_offset,
            bonus_position,
            verify_hidden_full,
        };
        (first_bonus, seed, first_bonus_lp)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        let input_ids = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        <Qwen35Model as mlxcel_core::generate::LanguageModel>::embed_tokens(self.model, &input_ids)
            .expect("Qwen35Model exposes its embed_tokens table")
    }

    /// Refuse a draft tree, naming the reason rather than inheriting the
    /// default.
    ///
    /// Qwen 3.5 is a hybrid: with `full_attention_interval` 4 over 64 layers,
    /// 48 are GatedDeltaNet and 16 are attention. The tree mask solves the
    /// attention layers and does nothing for the other 48, because a
    /// recurrence is a fold with no mask: flattening a tree through it carries
    /// state from a node into its sibling and produces a wrong answer with no
    /// signal. Verifying a tree here needs a tree-aware scan that forks state
    /// at branch points, which does not exist (issue #1204).
    ///
    /// Overridden rather than left to the trait default so the refusal states
    /// its cause at the place a reader looks for it.
    fn verify_forward_tree(
        &self,
        _tree: &mlxcel_core::speculative::mtp::tree::DraftTree,
        _sampler: &SamplingConfig,
        _logprobs_config: &LogprobsConfig,
    ) -> Result<VerifyForwardOutput, mlxcel_core::speculative::mtp::target::TreeVerifyUnsupported>
    {
        Err(
            mlxcel_core::speculative::mtp::target::TreeVerifyUnsupported {
                reason: "Qwen 3.5 is a GatedDeltaNet hybrid; a recurrence has no attention mask, \
                         so a flattened tree would fold state through siblings",
            },
        )
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &LogprobsConfig,
    ) -> VerifyForwardOutput {
        let verify_arr = mlxcel_core::from_slice_i32(verify_input, &[1, verify_input.len() as i32]);
        let capture = self.capture_last_layer();
        let out = self.with_state(|state| {
            self.model
                .forward_speculative_for_sequence(&verify_arr, state, &capture)
        });

        // issue #350: apply the model's output-suppression bias BEFORE the
        // per-position argmax and the logprob extraction, exactly as the
        // Gemma 4 adapter does. An empty map short-circuits to the raw
        // logits, preserving the bit-exact baseline.
        let logits = if sampler.token_bias.is_empty() {
            out.logits
        } else {
            mlxcel_core::sampling::apply_token_bias(&out.logits, &sampler.token_bias)
        };

        // Greedy-parity gate: per-position argmax. At temperature 0 this is
        // byte-identical to the drafter-less target's own argmax extension
        // (the verify pass runs the per-position target-verify attention).
        let target_tokens = Gemma4MtpTargetAdapter::argmax_per_position(&logits);
        let target_logprobs =
            Gemma4MtpTargetAdapter::per_position_logprobs(&logits, &target_tokens, logprobs_config);

        // Post-norm full-block hidden: consumed by finalize for
        // `next_hidden` / the drafter's accept hook.
        let mut hidden_states = out.hidden_states;
        let hidden_pre = hidden_states
            .pop()
            .expect("forward_speculative returns the requested last-layer capture");
        let hidden_post = self.model.apply_final_norm(&hidden_pre);

        VerifyForwardOutput {
            target_tokens,
            target_logprobs,
            captured: flatten_captured(hidden_post, out.gdn_states),
        }
    }

    /// Refuse a tree rollback, naming the reason rather than inheriting the
    /// default.
    ///
    /// Unreachable in practice, because [`Self::verify_forward_tree`] refuses
    /// first and no tree block ever reaches this cache. It is here because the
    /// two halves are one capability: a reader checking whether Qwen 3.5 can
    /// keep a tree path should find the answer at the rollback too, and a
    /// future forward that stops refusing must not silently inherit a rollback
    /// that cannot serve it.
    ///
    /// The cause is the same one. 48 of 64 layers are GatedDeltaNet, and their
    /// rollback replays the captured block over the accepted prefix. Replaying
    /// a gathered path instead of a prefix is an indexing change rather than a
    /// new mechanism, so this is reachable work; it is simply not done
    /// (issue #1204).
    fn verify_finalize_tree(
        &self,
        _path: &[usize],
        _block_size: usize,
        _captured: VerifyCaptured,
    ) -> Result<MtpVerifyOutput, mlxcel_core::speculative::mtp::target::TreeVerifyUnsupported> {
        Err(
            mlxcel_core::speculative::mtp::target::TreeVerifyUnsupported {
                reason: "Qwen 3.5 rolls GatedDeltaNet state back by replaying the accepted                          prefix, and replaying a scattered path is not implemented",
            },
        )
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        let (hidden_post, gdn_states) = unflatten_captured(captured);

        // Hidden at the accepted position seeds the drafter's next round.
        let next_hidden = Gemma4MtpTargetAdapter::hidden_at_position(&hidden_post, accepted);

        // Roll the KV + GDN caches back to the accepted prefix. Upstream
        // skips the call entirely on full accept (`accepted < bs - 1`
        // guard): the trim would be zero and the GDN replay would recompute
        // the state it already has.
        if accepted + 1 < block_size {
            let _ = self.with_state(|state| {
                self.model.rollback_speculative_cache_for_sequence(
                    state,
                    &gdn_states,
                    &[accepted as i32],
                    block_size as i32,
                )
            });
        }

        // Absolute post-rollback cache offset, read from the model-owned
        // cache (mirrors upstream's `prompt_cache[0].offset` rebind).
        let kv_offset = self
            .with_state(|state| self.model.speculative_cache_offset_for_sequence(state))
            .max(0) as usize;

        MtpVerifyOutput {
            next_hidden,
            next_shared_kv: Vec::new(),
            kv_offset,
            bonus_position: kv_offset.saturating_sub(1),
            // Full-block post-norm hidden for the stateful drafter's accept
            // hook (trim + extend + next-seed precompute).
            verify_hidden_full: Some(hidden_post),
        }
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        <Qwen35Model as mlxcel_core::generate::LanguageModel>::eos_token_ids(self.model)
    }
}

/// MTP target adapter for the Qwen 3.5 VLM wrapper.
///
/// Pure delegation to the inner text model, mirroring the DFlash VL adapter
/// (`src/vision/qwen3_5_vl.rs`) and the Gemma 4 VL MTP shell: vision
/// features are fully prefilled before the MTP round loop begins, so the
/// round loop only touches the text backbone. Multimodal requests never
/// reach here — the burst gate rejects them — and the speculative forward
/// follows the DFlash convention of standard (non-MRoPE) positions for
/// text-only requests on a VL target.
pub struct Qwen35VLMtpTargetAdapter<'a> {
    inner: Qwen35MtpTargetAdapter<'a>,
}

impl<'a> Qwen35VLMtpTargetAdapter<'a> {
    /// An adapter over the inner text model's fallback slot.
    pub fn fallback(vlm: &'a crate::vision::Qwen35VLModel) -> Self {
        Self {
            inner: Qwen35MtpTargetAdapter::fallback(&vlm.text_model),
        }
    }

    /// An adapter over scheduler sequence `seq_id`'s layer state; see
    /// [`Qwen35MtpTargetAdapter::for_sequence`].
    pub fn for_sequence(
        vlm: &'a crate::vision::Qwen35VLModel,
        seq_id: SequenceId,
        state: Option<&'a mut [LayerState]>,
    ) -> Self {
        Self {
            inner: Qwen35MtpTargetAdapter::for_sequence(&vlm.text_model, seq_id, state),
        }
    }

    /// Set the adopted prompt-cache prefix length (issue #518), delegating
    /// to the inner text-model adapter.
    #[must_use]
    pub fn with_prefill_start_offset(mut self, prefill_start_offset: usize) -> Self {
        self.inner = self.inner.with_prefill_start_offset(prefill_start_offset);
        self
    }
}

impl<'a> MtpTarget for Qwen35VLMtpTargetAdapter<'a> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &LogprobsConfig,
    ) -> (i32, MtpVerifyOutput, Option<TokenLogprobData>) {
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
        logprobs_config: &LogprobsConfig,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(vals: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
        mlxcel_core::from_slice_f32(vals, shape)
    }

    fn first_val(arr: &MlxArray) -> f32 {
        mlxcel_core::eval(arr);
        let bytes = mlxcel_core::array_to_raw_bytes(arr);
        f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    /// The flatten/unflatten pair must preserve snapshot order, per-snapshot
    /// tensor identity, `layer_idx`, and the optional `init_state` — a
    /// mis-ordered flatten produces subtly wrong recurrent state after
    /// rollback rather than a crash, which is exactly the failure mode this
    /// pin exists to catch.
    #[test]
    fn captured_round_trip_preserves_gdn_snapshots() {
        let hidden = tensor(&[42.0, 43.0], &[1, 2, 1]);
        // Two snapshots with distinguishable leading values; the second has
        // an init_state, the first does not.
        let snap_a = GdnRollbackSnapshot {
            layer_idx: 3,
            q: tensor(&[1.0], &[1, 1, 1, 1]),
            k: tensor(&[2.0], &[1, 1, 1, 1]),
            v: tensor(&[3.0], &[1, 1, 1, 1]),
            a: tensor(&[4.0], &[1, 1, 1]),
            b: tensor(&[5.0], &[1, 1, 1]),
            init_state: None,
            conv_input: tensor(&[6.0], &[1, 1, 1]),
        };
        let snap_b = GdnRollbackSnapshot {
            layer_idx: 7,
            q: tensor(&[10.0], &[1, 1, 1, 1]),
            k: tensor(&[20.0], &[1, 1, 1, 1]),
            v: tensor(&[30.0], &[1, 1, 1, 1]),
            a: tensor(&[40.0], &[1, 1, 1]),
            b: tensor(&[50.0], &[1, 1, 1]),
            init_state: Some(tensor(&[70.0], &[1, 1, 1, 1])),
            conv_input: tensor(&[60.0], &[1, 1, 1]),
        };

        let captured = flatten_captured(hidden, vec![snap_a, snap_b]);
        assert_eq!(captured.scalars, vec![2, 3, 0, 7, 1]);
        // 1 hidden + 6 (snapshot without init) + 7 (snapshot with init).
        assert_eq!(captured.tensors.len(), 14);

        let (hidden, gdn) = unflatten_captured(captured);
        assert_eq!(first_val(hidden.as_ref().unwrap()), 42.0);
        assert_eq!(gdn.len(), 2);

        assert_eq!(gdn[0].layer_idx, 3);
        assert!(gdn[0].init_state.is_none());
        assert_eq!(first_val(gdn[0].q.as_ref().unwrap()), 1.0);
        assert_eq!(first_val(gdn[0].b.as_ref().unwrap()), 5.0);
        assert_eq!(first_val(gdn[0].conv_input.as_ref().unwrap()), 6.0);

        assert_eq!(gdn[1].layer_idx, 7);
        assert_eq!(first_val(gdn[1].q.as_ref().unwrap()), 10.0);
        assert_eq!(first_val(gdn[1].conv_input.as_ref().unwrap()), 60.0);
        assert_eq!(
            first_val(gdn[1].init_state.as_ref().unwrap().as_ref().unwrap()),
            70.0
        );
    }

    /// Zero snapshots (a hypothetical all-attention layout) round-trips to
    /// an empty snapshot list without touching the hidden slot.
    #[test]
    fn captured_round_trip_handles_zero_snapshots() {
        let captured = flatten_captured(tensor(&[9.0], &[1, 1, 1]), Vec::new());
        assert_eq!(captured.scalars, vec![0]);
        assert_eq!(captured.tensors.len(), 1);
        let (hidden, gdn) = unflatten_captured(captured);
        assert_eq!(first_val(hidden.as_ref().unwrap()), 9.0);
        assert!(gdn.is_empty());
    }

    /// One prefill, one verify block and one rollback over `adapter`: the
    /// first bonus, the verify tokens and the cache offsets after each step.
    fn drive(adapter: &Qwen35MtpTargetAdapter<'_>) -> (i32, usize, Vec<i32>, usize, Vec<f32>) {
        use mlxcel_core::generate::SamplingConfig;
        use mlxcel_core::sampling::LogprobsConfig;

        let sampler = SamplingConfig::greedy();
        let logprobs = LogprobsConfig::default();
        let (bonus, seed, _) = adapter.prefill_and_seed(&[1, 2, 3, 4, 5], &sampler, &[], &logprobs);
        let verify = adapter.verify_forward(&[bonus, 3, 6, 2], &sampler, &logprobs);
        // Accept one of the four verify positions: the rollback trims the
        // attention rows and replays the GatedDeltaNet state.
        let out = adapter.verify_finalize(1, 4, verify.captured);
        let hidden = mlxcel_core::astype(&out.next_hidden, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&hidden);
        let hidden: Vec<f32> = mlxcel_core::array_to_raw_bytes(&hidden)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (
            bonus,
            seed.kv_offset,
            verify.target_tokens,
            out.kv_offset,
            hidden,
        )
    }

    /// The adapter over a pool sequence's layer state (issue #3254) runs the
    /// computation the fallback-slot adapter runs, and leaves the result in
    /// the pool: prefill, verify and rollback agree step for step, and the
    /// pool's attention layer ends at the rolled-back offset.
    #[test]
    fn a_pool_sequence_adapter_matches_the_fallback_adapter() {
        use mlxcel_core::cache::CachePool;
        use mlxcel_core::generate::LanguageModel;

        let _guard = crate::models::embedding_test_support::mlx_test_guard();
        let model = crate::models::qwen3_5_tests::two_layer_model();
        model.reset_runtime_state();
        let want = drive(&Qwen35MtpTargetAdapter::fallback(&model));

        let mut pool = CachePool::new(4);
        let seq = pool.allocate(&model).expect("pool sequence");
        let got = drive(&Qwen35MtpTargetAdapter::for_sequence(
            &model,
            seq,
            pool.layer_state_mut(seq),
        ));
        assert_eq!(got.0, want.0, "first bonus");
        assert_eq!(got.1, want.1, "offset after prefill");
        assert_eq!(got.2, want.2, "verify tokens");
        assert_eq!(got.3, want.3, "offset after rollback");
        assert_eq!(got.4, want.4, "hidden at the accepted position");
        assert_eq!(got.1, 5);
        assert_eq!(got.3, 7, "five prompt rows plus the two kept verify rows");

        // The state the adapter left is the pool's, at the rolled-back offset.
        let layers = pool.layer_state(seq).expect("layer state");
        let offsets: Vec<i32> = layers.iter().map(|layer| layer.offset()).collect();
        assert_eq!(offsets, vec![7, 7]);
        assert!(
            layers[0]
                .as_recurrent()
                .is_some_and(|r| r.get("state_cache").is_some())
        );
    }

    /// A scheduler sequence with no layer state is refused. It must not run
    /// on the shared fallback slot.
    #[test]
    #[should_panic(expected = "holds no scheduler-owned layer state")]
    fn a_sequence_without_pool_state_is_refused() {
        let _guard = crate::models::embedding_test_support::mlx_test_guard();
        let model = crate::models::qwen3_5_tests::two_layer_model();
        let adapter = Qwen35MtpTargetAdapter::for_sequence(
            &model,
            mlxcel_core::cache::SequenceId::from_raw(3_254_700),
            None,
        );
        let sampler = mlxcel_core::generate::SamplingConfig::greedy();
        let _ = adapter.prefill_and_seed(
            &[1, 2, 3],
            &sampler,
            &[],
            &mlxcel_core::sampling::LogprobsConfig::default(),
        );
    }
}
