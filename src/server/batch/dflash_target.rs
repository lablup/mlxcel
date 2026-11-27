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

//! Target contract of the server-side DFlash bursts, and the generic drivers
//! that run on it (issue #1339).
//!
//! `DFlashGenerator` already accepts any
//! [`mlxcel_core::drafter::dflash::SpeculativeTarget`], but the server side
//! of a burst also has to allocate the target's heterogeneous cache vector,
//! read logits and captured hidden states off the target's own verify-output
//! type, and decide how much of the prompt's captured hidden the first draft
//! sees. [`DFlashTargetModel`] holds those, so most of what a family needs
//! lives in one `impl` block here:
//!
//! - Qwen 3.5 (DFlash): fresh `Qwen3NextCache`s, the last prompt position's
//!   hidden as the first draft input.
//! - LFM2 / LFM2.5 (DSpark): fresh `Lfm2LayerCache`s, EVERY prompt row as the
//!   first draft input (the drafter's own context cache holds the whole
//!   prompt), the block-versus-chain exactness probe as a gate, a
//!   drafter-family requirement (DSpark only), and B = 1 only.
//! - Qwen3 and Llama (DSpark, issue #3261): fresh FP16 `KVCache`s, every
//!   prompt row as the first draft input, the classic prefill partition, the
//!   exactness probe as a gate, no required drafter family (as on Qwen 3.5)
//!   but a tap-depth and vocabulary check of their own, and B = 1 only.
//! - Muse Glimmer (assistant, issue #1343): fresh `MuseCache`s with the
//!   sliding ones armed with a speculative buffer, every prompt row as the
//!   first draft input, the exactness probe as a gate, a drafter-family
//!   requirement (the Muse assistant only), and B = 1 only.
//!
//! The `impl` is not the whole cost, and the count is worth stating plainly
//! before a fifth family is added. Each family also touches three match
//! arms in [`super::speculative_burst`], because the burst reaches the
//! target as a `LoadedModel` enum and only a match recovers the concrete
//! type the trait is implemented on: the exactness gate, the `drive!`
//! dispatch, and the batched variant gate. All three are uniform
//! one-liners that read their policy off this trait; the batched gate reads
//! [`DFlashTargetModel::supports_batched`], so no per-family policy lives
//! outside the trait any more. A fourth arm, `model_variant_label`, is the
//! pre-existing project-wide label table (#1613) and is not specific to
//! DFlash.
//!
//! The VLM wrappers of both families implement the trait by delegating to
//! their text backbone, which is what lets a text-only request against a
//! VLM checkpoint run the burst.
//!
//! What the trait deliberately does NOT carry: a per-round block-size
//! policy. That lives on the drafter, not the target: `DFlashGenerator`
//! reads `Drafter::configured_block_size` and
//! `Drafter::prefer_requested_block_size` and, for a drafter that declares
//! a depth below the requested width without insisting on it (the Muse
//! Glimmer assistant, issue #1343), runs the MTP loop's throughput
//! comparator between the two. The burst still hands in one `block_size`,
//! which is the ceiling.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use mlxcel_core::drafter::Drafter;
use mlxcel_core::drafter::DrafterKind;
use mlxcel_core::drafter::dflash::{DFlashBatchedGenerator, DFlashGenerator, SpeculativeTarget};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::sampling::TokenLogprobData;
use mlxcel_core::{MlxArray, UniquePtr};

use super::speculative_burst::{BurstOutcome, WorkerDrafterSlot};
use crate::models::MuseGlimmerTextWrapper;
use crate::models::dense_speculative::DenseSpeculativeTarget;
use crate::server::model_provider::SpeculativeStats;

/// Uniform access to what the burst reads off a target's verify output.
///
/// Each family's `VerifyOut` is its own struct (Qwen's carries GDN
/// snapshots, LFM2's carries short-conv snapshots); the burst only ever
/// needs the logits and the captured hidden slabs.
pub(crate) trait DFlashVerifyOutput {
    /// `[B, L, vocab]` logits for every verify row.
    fn logits(&self) -> &MlxArray;
    /// One `[B, L, hidden]` slab per captured layer, in capture order.
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>];
}

impl DFlashVerifyOutput for crate::models::qwen3_5::VerifyOutput {
    fn logits(&self) -> &MlxArray {
        self.logits
            .as_ref()
            .expect("verify logits must be non-null")
    }
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>] {
        &self.hidden_states
    }
}

impl DFlashVerifyOutput for crate::models::lfm2::VerifyOutput {
    fn logits(&self) -> &MlxArray {
        self.logits
            .as_ref()
            .expect("verify logits must be non-null")
    }
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>] {
        &self.hidden_states
    }
}

impl DFlashVerifyOutput for crate::models::dense_speculative::DenseVerifyOutput {
    fn logits(&self) -> &MlxArray {
        self.logits
            .as_ref()
            .expect("verify logits must be non-null")
    }
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>] {
        &self.hidden_states
    }
}

impl DFlashVerifyOutput for crate::models::laguna_speculative::LagunaVerifyOutput {
    fn logits(&self) -> &MlxArray {
        self.logits
            .as_ref()
            .expect("verify logits must be non-null")
    }
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>] {
        &self.hidden_states
    }
}

impl DFlashVerifyOutput for crate::models::inkling2::speculative::Inkling2VerifyOutput {
    fn logits(&self) -> &MlxArray {
        self.logits
            .as_ref()
            .expect("Inkling 2 verify output must carry logits")
    }

    fn hidden_states(&self) -> &[UniquePtr<MlxArray>] {
        &self.hidden_states
    }
}

impl DFlashVerifyOutput for crate::models::raon1_mini::speculative::Raon1MiniVerifyOutput {
    fn logits(&self) -> &MlxArray {
        &self.logits
    }
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>] {
        &self.hidden_states
    }
}

impl DFlashVerifyOutput for crate::models::muse_glimmer::VerifyOutput {
    fn logits(&self) -> &MlxArray {
        self.logits
            .as_ref()
            .expect("verify logits must be non-null")
    }
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>] {
        &self.hidden_states
    }
}

/// The DFlash-family drafters the round loop can be handed, as the target
/// gate tells them apart.
///
/// All three share [`DrafterKind::Dflash`], one loader and one round loop,
/// and differ in the target whose residual streams their `fc` projection was
/// trained on. The drafter cannot tell targets apart on its own (it sees a
/// [`LanguageModel`] with no architecture string), so the target names the
/// family it admits through [`DFlashTargetModel::required_drafter_family`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DFlashDrafterFamily {
    /// The Qwen 3.5 DFlash drafter.
    Dflash,
    /// A DSpark drafter on the DFlash backbone: the LFM2 / LFM2.5 ones (issue
    /// #1339) and the Red Hat Speculators and OpenBMB ones for dense Qwen3
    /// and Llama targets (issue #3261).
    DSpark,
    /// The Muse Glimmer assistant (issue #1343).
    MuseAssistant,
    /// The Poolside Laguna DFlash drafter (issue #1351).
    Laguna,
    /// The DSpark drafter bundled with an Inkling 2 checkpoint (issue #3374).
    Inkling2DSpark,
    /// The Raon-1 mini DFlash2 drafter (issue #3436).
    RaonDFlash2,
}

impl DFlashDrafterFamily {
    /// Which family a loaded drafter is, read off its two self-reports.
    fn of(drafter: &dyn Drafter) -> Self {
        if drafter.is_inkling2_dspark() {
            Self::Inkling2DSpark
        } else if drafter.is_raon_dflash2() {
            Self::RaonDFlash2
        } else if drafter.is_dspark() {
            Self::DSpark
        } else if drafter.is_muse_assistant() {
            Self::MuseAssistant
        } else if drafter.is_laguna_dflash() {
            Self::Laguna
        } else {
            Self::Dflash
        }
    }
}

/// How much of the prompt's captured hidden the first draft round consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FirstHiddenRows {
    /// Only the last prompt position (`[B, 1, dim]`). The Qwen 3.5 DFlash
    /// drafter starts its context cache from the bonus token onwards.
    LastPromptPosition,
    /// Every prompt row (`[B, S, dim]`). The LFM2 DSpark drafter appends the
    /// whole prompt to its context cache before its first proposal.
    EveryPromptRow,
}

/// Server-side target contract of the DFlash bursts.
///
/// `SpeculativeTarget` is what the round loop drives; this trait adds what
/// the burst around it needs. The associated-type bound lets the generic
/// drivers below read logits and hidden states without naming any family's
/// `VerifyOut`.
pub(crate) trait DFlashTargetModel:
    LanguageModel + SpeculativeTarget<VerifyOut: DFlashVerifyOutput>
{
    /// A fresh per-layer cache vector for one burst. Independent of the
    /// scheduler-owned per-sequence slots, which is why an adopted
    /// prompt-cache prefix declines to classic decode on this path.
    fn make_dflash_caches(&self) -> Vec<Self::Cache>;

    /// [`Self::make_dflash_caches`] for the burst of `seq_id`, for a target
    /// whose forward depends on per-sequence state the scheduler set at
    /// admission (Raon-1 mini's reasoning-effort alpha, issue #3436). The
    /// default ignores the id.
    fn make_dflash_caches_for(
        &self,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
    ) -> Vec<Self::Cache> {
        let _ = seq_id;
        self.make_dflash_caches()
    }

    /// Which prompt rows the first draft sees. Qwen 3.5 keeps the last
    /// position; DSpark wants every row.
    ///
    /// An associated function rather than a method: the policy is a
    /// property of the drafter family the target is paired with, not of a
    /// loaded instance, and taking no `self` is what lets the unit test
    /// below pin each family's answer without a checkpoint on disk.
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::LastPromptPosition
    }

    /// The burst's prompt prefill. `chunk` and `boundary` are the scheduler's
    /// classic partition of this prompt (`--prefill-chunk-size` and the
    /// prompt-cache snapshot splits), which a target whose drafter-on text
    /// must equal classic decode past one chunk mirrors (Inkling 2, issue
    /// #3410). The default prefills through
    /// `SpeculativeTarget::prefill_forward_with_capture_layers` unchanged.
    fn prefill_burst(
        &self,
        prompt: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
        chunk: usize,
        boundary: mlxcel_core::prefill_plan::PrefillSplits,
    ) -> Self::VerifyOut {
        let _ = (chunk, boundary);
        self.prefill_forward_with_capture_layers(prompt, caches, capture_layer_ids)
    }

    /// Hook for a target whose caches need arming for a multi-row verify
    /// (for example a rotating buffer sized to the block). No-op by
    /// default; runs on the fresh caches before the prefill.
    fn enable_speculative_buffers(&self, _caches: &mut [Self::Cache], _block_size: usize) {}

    /// Hook for a target that can capture only the newest `rows` prompt rows
    /// at prefill, for a drafter that keeps a bounded context window
    /// (`Drafter::context_window`, issue #3436). Runs on the fresh caches
    /// before the prefill; the burst then tells the drafter how many rows
    /// were left out. No-op by default (every row is captured).
    fn limit_prefill_capture(&self, _caches: &mut [Self::Cache], _rows: usize) {}

    /// Whether a `block_size`-row verify block is byte-identical to the
    /// single-token decode chain on this host, and the burst may therefore
    /// promise the temperature-0 contract. Every family with a server-side
    /// DFlash target now overrides this with the measured block-versus-chain
    /// probe; the permissive default remains only for a target that has not
    /// been measured yet, and issue #1935 is what a permissive default costs.
    fn exactness_allows(&self, _block_size: usize) -> bool {
        true
    }

    /// The one DFlash-family drafter this target can be driven by, or `None`
    /// when the target does not restrict the pairing.
    ///
    /// `Some(DSpark)` for LFM2 / LFM2.5 and `Some(MuseAssistant)` for Muse
    /// Glimmer: the only drafter published for each is its own, and any
    /// other DFlash-family drafter's `fc` projection reads residual streams
    /// of a width this target does not produce. `Drafter::validate_target_compat`
    /// cannot catch that pairing on its own, because a plain DFlash drafter
    /// sees the target as a `LanguageModel` with no architecture string and
    /// returns `Ok`; the target is the side that knows. An associated function
    /// for the same reason as `first_hidden_rows`: it is a property of the
    /// family, not of a loaded instance, so the test below can pin it without
    /// a checkpoint.
    ///
    /// Qwen 3.5 keeps the permissive default. Narrowing it there would change
    /// which DFlash pairings that family accepts, which is a separate call.
    fn required_drafter_family() -> Option<DFlashDrafterFamily> {
        None
    }

    /// Why `drafter` cannot drive this target, from what the target itself
    /// knows (its depth and vocabulary), or `None` when it has no objection.
    ///
    /// The family gate above is a yes/no per family, and the drafter's own
    /// `validate_target_compat` is a no-op for a plain DFlash drafter. A
    /// target that admits every DFlash-family drafter and whose residual
    /// streams have a common width (Qwen3 and Llama, issue #3261) checks the
    /// two numbers that still tell a mismatched drafter apart: every tap must
    /// be below the target depth, and the drafter's vocabulary must be the
    /// target's. The default has no objection, which keeps Qwen 3.5 and every
    /// single-family target exactly as they were.
    fn drafter_pairing_error(&self, _drafter: &dyn Drafter) -> Option<String> {
        None
    }

    /// Whether this family can be driven by the B > 1 batched burst.
    ///
    /// `true` for Qwen 3.5, whose target implements the per-row batched
    /// rollback and whose drafter drafts batched. `false` for LFM2 (its
    /// drafter has no batched draft and its short-conv state no per-row
    /// rollback) and for Muse Glimmer (its drafter is B = 1 and its rotating
    /// caches have no per-row rewind). A `false` here keeps the scheduler
    /// from collecting a window, so a queued head runs the B = 1 burst
    /// (#2442), and a window that reaches the batched variant gate anyway
    /// declines to classic decode. An associated function for the same
    /// reason as `first_hidden_rows`.
    fn supports_batched() -> bool {
        true
    }
}

impl DFlashTargetModel for crate::models::Qwen35Model {
    fn make_dflash_caches(&self) -> Vec<crate::models::qwen3_next::Qwen3NextCache> {
        self.make_speculative_caches()
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.dflash_exactness_allows(block_size)
    }
}

impl DFlashTargetModel for crate::vision::Qwen35VLModel {
    fn make_dflash_caches(&self) -> Vec<crate::models::qwen3_next::Qwen3NextCache> {
        self.text_model.make_speculative_caches()
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.text_model.dflash_exactness_allows(block_size)
    }
}

impl DFlashTargetModel for crate::models::Lfm2Model {
    fn make_dflash_caches(&self) -> Vec<crate::models::lfm2::Lfm2LayerCache> {
        self.make_speculative_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.dflash_exactness_allows(block_size)
    }
    fn required_drafter_family() -> Option<DFlashDrafterFamily> {
        Some(DFlashDrafterFamily::DSpark)
    }
    fn supports_batched() -> bool {
        false
    }
}

impl DFlashTargetModel for crate::vision::Lfm2VlModel {
    fn make_dflash_caches(&self) -> Vec<crate::models::lfm2::Lfm2LayerCache> {
        self.text_model.make_speculative_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.text_model.dflash_exactness_allows(block_size)
    }
    fn required_drafter_family() -> Option<DFlashDrafterFamily> {
        Some(DFlashDrafterFamily::DSpark)
    }
    fn supports_batched() -> bool {
        false
    }
}

/// Dense Qwen3 (issue #3261), driven by a DSpark drafter such as
/// `RedHatAI/Qwen3-4B-speculator.dspark`: fresh FP16 `KVCache`s, every prompt
/// row as the first draft input (the drafter keeps its own window of them),
/// the classic prefill partition, the measured block-versus-chain probe as
/// the gate, and B = 1 only. No required drafter family, as on Qwen 3.5; the
/// tap and vocabulary bounds are checked by `drafter_pairing_error` instead.
impl DFlashTargetModel for crate::models::Qwen3Model {
    fn make_dflash_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        self.make_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn prefill_burst(
        &self,
        prompt: &MlxArray,
        caches: &mut [mlxcel_core::layers::KVCache],
        capture_layer_ids: &[usize],
        chunk: usize,
        boundary: mlxcel_core::prefill_plan::PrefillSplits,
    ) -> Self::VerifyOut {
        crate::models::dense_speculative::prefill_planned(
            self,
            prompt,
            caches,
            capture_layer_ids,
            chunk,
            boundary,
        )
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        DenseSpeculativeTarget::dflash_exactness_allows(self, block_size)
    }
    fn drafter_pairing_error(&self, drafter: &dyn Drafter) -> Option<String> {
        DenseSpeculativeTarget::dflash_drafter_pairing_error(self, drafter)
    }
    fn supports_batched() -> bool {
        false
    }
}

/// Dense Llama (issue #3261), driven by a DSpark drafter such as
/// `openbmb/MiniCPM5-2B-DSpark` (MiniCPM5-2B is a `llama` checkpoint). Same
/// policy as the Qwen3 target above.
impl DFlashTargetModel for crate::models::Llama3Model {
    fn make_dflash_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        self.make_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn prefill_burst(
        &self,
        prompt: &MlxArray,
        caches: &mut [mlxcel_core::layers::KVCache],
        capture_layer_ids: &[usize],
        chunk: usize,
        boundary: mlxcel_core::prefill_plan::PrefillSplits,
    ) -> Self::VerifyOut {
        crate::models::dense_speculative::prefill_planned(
            self,
            prompt,
            caches,
            capture_layer_ids,
            chunk,
            boundary,
        )
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        DenseSpeculativeTarget::dflash_exactness_allows(self, block_size)
    }
    fn drafter_pairing_error(&self, drafter: &dyn Drafter) -> Option<String> {
        DenseSpeculativeTarget::dflash_drafter_pairing_error(self, drafter)
    }
    fn supports_batched() -> bool {
        false
    }
}

/// Muse Glimmer (issue #1343). The sliding caches are armed with
/// `clamp(block_size * 8, 32, 128)` rows of speculative slack before the
/// prefill so a verify block appended across the ring boundary can be
/// rewound, and the first draft sees every prompt row because the
/// assistant's context window holds the prompt.
impl DFlashTargetModel for MuseGlimmerTextWrapper {
    fn make_dflash_caches(&self) -> Vec<crate::models::muse_glimmer::MuseCache> {
        self.make_speculative_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn enable_speculative_buffers(
        &self,
        caches: &mut [crate::models::muse_glimmer::MuseCache],
        block_size: usize,
    ) {
        MuseGlimmerTextWrapper::enable_speculative_buffers(
            self,
            caches,
            crate::models::muse_glimmer::speculative_buffer_size(block_size),
        );
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.dflash_exactness_allows_every_width(block_size)
    }
    fn required_drafter_family() -> Option<DFlashDrafterFamily> {
        Some(DFlashDrafterFamily::MuseAssistant)
    }
    fn supports_batched() -> bool {
        false
    }
}

/// Laguna (issue #1351). The sliding-window layers' rotating caches are armed
/// with `block_size` rows of speculative slack before the prefill so a verify
/// block is appended inside the temporal region and rolled back by a tail
/// trim; the first draft sees every prompt row (vLLM's Laguna DFlash proposer
/// hands the drafter the whole prompt, and the drafter keeps the newest
/// window of it). The exactness probe is the measured block-versus-chain
/// gate shared with LFM2 and Muse Glimmer.
impl DFlashTargetModel for crate::models::LagunaWrapper {
    fn make_dflash_caches(&self) -> Vec<crate::models::laguna_layers::LagunaCache> {
        self.model.make_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn enable_speculative_buffers(
        &self,
        caches: &mut [crate::models::laguna_layers::LagunaCache],
        block_size: usize,
    ) {
        crate::models::LagunaModel::enable_speculative_buffers(&self.model, caches, block_size);
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.model.dflash_exactness_allows(block_size)
    }
    fn required_drafter_family() -> Option<DFlashDrafterFamily> {
        Some(DFlashDrafterFamily::Laguna)
    }
    fn supports_batched() -> bool {
        false
    }
}

/// Raon-1 mini (issue #3436), driven by its DFlash2 drafter: every prompt row
/// seeds the drafter context, the burst caches carry the sequence's
/// reasoning-effort alpha, the width is gated by the exactness probe, and the
/// burst is B = 1 only.
impl DFlashTargetModel for crate::models::Raon1MiniModel {
    fn make_dflash_caches(
        &self,
    ) -> Vec<crate::models::raon1_mini::speculative::Raon1MiniSpecCache> {
        self.make_speculative_caches(None)
    }
    fn make_dflash_caches_for(
        &self,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
    ) -> Vec<crate::models::raon1_mini::speculative::Raon1MiniSpecCache> {
        self.make_speculative_caches(seq_id)
    }
    fn limit_prefill_capture(
        &self,
        caches: &mut [crate::models::raon1_mini::speculative::Raon1MiniSpecCache],
        rows: usize,
    ) {
        crate::models::Raon1MiniModel::limit_prefill_capture(self, caches, rows);
    }
    /// The classic partition of the prompt: Raon-1 mini's prefill splits at
    /// the prompt-cache snapshot boundaries (issue #3458), and drafted text
    /// must equal the text of that split prefill.
    fn prefill_burst(
        &self,
        prompt: &MlxArray,
        caches: &mut [crate::models::raon1_mini::speculative::Raon1MiniSpecCache],
        capture_layer_ids: &[usize],
        chunk: usize,
        boundary: mlxcel_core::prefill_plan::PrefillSplits,
    ) -> Self::VerifyOut {
        self.prefill_speculative_planned(prompt, caches, capture_layer_ids, chunk, boundary)
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.dflash_exactness_allows(block_size)
    }
    fn required_drafter_family() -> Option<DFlashDrafterFamily> {
        Some(DFlashDrafterFamily::RaonDFlash2)
    }
    fn supports_batched() -> bool {
        false
    }
}

/// Inkling 2 (issue #3374), driven by the DSpark drafter bundled in its own
/// checkpoint: every prompt row seeds the drafter context, the block width is
/// gated by the exactness probe, and the burst is B = 1 only.
impl DFlashTargetModel for crate::vision::Inkling2VlModel {
    fn make_dflash_caches(&self) -> Vec<mlxcel_core::inkling_layer::InklingLayerCache> {
        self.text.make_speculative_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn prefill_burst(
        &self,
        prompt: &MlxArray,
        caches: &mut [mlxcel_core::inkling_layer::InklingLayerCache],
        capture_layer_ids: &[usize],
        chunk: usize,
        boundary: mlxcel_core::prefill_plan::PrefillSplits,
    ) -> Self::VerifyOut {
        let suppressed = crate::models::inkling2::speculative::suppression_map(
            &LanguageModel::output_suppressed_token_ids(self),
        );
        self.text.prefill_speculative_planned(
            prompt,
            caches,
            capture_layer_ids,
            &suppressed,
            chunk,
            boundary,
        )
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.text.dflash_exactness_allows(block_size)
    }
    fn required_drafter_family() -> Option<DFlashDrafterFamily> {
        Some(DFlashDrafterFamily::Inkling2DSpark)
    }
    fn supports_batched() -> bool {
        false
    }
}

impl DFlashTargetModel for crate::vision::MuseGlimmerVlmModel {
    fn make_dflash_caches(&self) -> Vec<crate::models::muse_glimmer::MuseCache> {
        self.text.make_speculative_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn enable_speculative_buffers(
        &self,
        caches: &mut [crate::models::muse_glimmer::MuseCache],
        block_size: usize,
    ) {
        self.text.enable_speculative_buffers(
            caches,
            crate::models::muse_glimmer::speculative_buffer_size(block_size),
        );
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.text.dflash_exactness_allows_every_width(block_size)
    }
    fn required_drafter_family() -> Option<DFlashDrafterFamily> {
        Some(DFlashDrafterFamily::MuseAssistant)
    }
    fn supports_batched() -> bool {
        false
    }
}

/// What one DFlash target run hands back to `run_dflash_burst`.
///
/// A struct rather than a tuple because the run grew a fourth member with
/// issue #1314 and a four-element tuple of two vectors, a float and an option
/// is read once and then guessed at forever.
pub(crate) struct DFlashTargetRun {
    /// Every token the run emitted, the first bonus included.
    pub(crate) tokens: Vec<i32>,
    /// Per-token logprob payloads, index-aligned with [`Self::tokens`]; empty
    /// when the request did not ask for logprobs.
    pub(crate) logprobs: Vec<Option<TokenLogprobData>>,
    /// Round-loop decode wall clock in milliseconds.
    pub(crate) decode_time_ms: f64,
    /// Client-facing drafter acceptance counters (issue #1314); `None` when
    /// the run executed no verify round.
    pub(crate) speculative: Option<SpeculativeStats>,
}

/// Concatenate the captured hidden slabs along the feature axis:
/// `[B, L, num_layers * hidden]`.
fn concat_captured_hidden(slabs: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let refs: Vec<&MlxArray> = slabs
        .iter()
        .map(|slab| slab.as_ref().expect("hidden state must be non-null"))
        .collect();
    mlxcel_core::concatenate_many(&refs, -1)
}

/// Whether a target that admits one drafter family got it, as the
/// operator-facing reason to decline.
///
/// `None` when the pairing is admissible (no requirement, or the drafter is
/// the required family). Takes the two decided facts rather than the target
/// and the drafter so both run arms and the unit test below share one
/// message per family.
///
/// Used by: [`run_dflash_on_target`], [`run_dflash_batched_on_target`].
fn required_family_pairing_error(
    required: Option<DFlashDrafterFamily>,
    actual: DFlashDrafterFamily,
) -> Option<String> {
    let required = required?;
    if required == actual {
        return None;
    }
    let got = match actual {
        DFlashDrafterFamily::Dflash => "a plain DFlash drafter",
        DFlashDrafterFamily::DSpark => "a DSpark drafter",
        DFlashDrafterFamily::MuseAssistant => "the Muse Glimmer assistant drafter",
        DFlashDrafterFamily::Laguna => "a Laguna DFlash drafter",
        DFlashDrafterFamily::Inkling2DSpark => "the DSpark drafter bundled with Inkling 2",
        DFlashDrafterFamily::RaonDFlash2 => "the Raon-1 mini DFlash2 drafter",
    };
    Some(match required {
        DFlashDrafterFamily::DSpark => format!(
            "the target is an LFM2 / LFM2.5 checkpoint and the drafter passed to --model-draft is \
             {got}, not a DSpark one. An LFM2 target can only be paired with the LiquidAI \
             DSpark drafter published for that exact checkpoint (LFM2.5-2.6B with \
             LFM2.5-2.6B-DSpark, LFM2.5-8B-A1B with LFM2.5-8B-A1B-DSpark, and so on): another \
             drafter's fc projection reads the residual streams it was trained on, at a width \
             this target does not produce, and its mask token id indexes a different \
             vocabulary. Point --model-draft at the LFM2 DSpark drafter for this checkpoint \
             (a DSpark drafter published for a Qwen3 or Llama target is refused by the \
             drafter's own pairing gate), or drop it to serve this model with classic decode."
        ),
        DFlashDrafterFamily::MuseAssistant => format!(
            "the target is a Muse Glimmer checkpoint and the drafter passed to --model-draft is \
             {got}, not the Muse Glimmer assistant. A Muse Glimmer target can only be paired \
             with meta-models/Muse-Glimmer-30B-assistant (model_type muse_glimmer_assistant): \
             another drafter's fc projection reads the residual streams it was trained on, at \
             a width this target does not produce, and its mask token id indexes a different \
             vocabulary. Point --model-draft at the Muse Glimmer assistant, or drop it to \
             serve this model with classic decode."
        ),
        DFlashDrafterFamily::Laguna => format!(
            "the target is a Laguna checkpoint and the drafter passed to --model-draft is \
             {got}, not a Laguna DFlash one. A Laguna target can only be paired with the \
             Poolside DFlash speculator published for its release (Laguna-XS-2.1 with \
             Laguna-XS-2.1-DFlash, and so on): another drafter's fc projection reads the \
             residual streams it was trained on, at a width this target does not produce, \
             and its mask token id indexes a different vocabulary. Point --model-draft at a \
             Laguna DFlash drafter, or drop it to serve this model with classic decode."
        ),
        DFlashDrafterFamily::Inkling2DSpark => format!(
            "the target is an Inkling 2 checkpoint and the drafter passed to --model-draft is \
             {got}, not its bundled DSpark drafter. An Inkling 2 target drafts with the \
             mtp.safetensors shipped in its own checkpoint: pass the same directory to \
             --model-draft, or drop it to serve this model with classic decode."
        ),
        DFlashDrafterFamily::RaonDFlash2 => format!(
            "the target is a Raon-1 mini checkpoint and the drafter passed to --model-draft is \
             {got}, not KRAFTON/Raon-1-mini-dflash2. A Raon-1 mini target drafts with that \
             speculator only: another drafter's fc projection reads residual streams of a width \
             this target does not produce. Point --model-draft at Raon-1-mini-dflash2, or drop \
             it to serve this model with classic decode."
        ),
        DFlashDrafterFamily::Dflash => format!(
            "the target admits only the Qwen 3.5 DFlash drafter and the drafter passed to \
             --model-draft is {got}. Point --model-draft at a DFlash drafter, or drop it to \
             serve this model with classic decode."
        ),
    })
}

/// The prompt rows the first draft round consumes, per
/// [`DFlashTargetModel::first_hidden_rows`].
///
/// Takes the policy rather than the target so the two branches are
/// testable without a loaded checkpoint; the call sites read it off
/// [`DFlashTargetModel::first_hidden_rows`].
///
/// Takes `concatenated` BY VALUE, because the every-row policy wants exactly
/// the array it was handed. `mlxcel_core::copy` is a real MLX `Copy`
/// primitive rather than a handle clone, so returning a copy here allocated a
/// second full `[1, S, len(target_layer_ids) * hidden]` slab per request (168
/// MB at an 8k prompt on LFM2.5-2.6B, 671 MB at 32k) only to satisfy a
/// borrowed signature. Moving ownership in also ends the caller's retention
/// of the original across the whole round loop. The last-position arm slices,
/// and a slice holds its own reference to the input, so dropping the owner
/// here is safe on both arms.
fn first_hidden_for(
    rows: FirstHiddenRows,
    concatenated: UniquePtr<MlxArray>,
    last_pos: i32,
) -> UniquePtr<MlxArray> {
    match rows {
        FirstHiddenRows::EveryPromptRow => concatenated,
        FirstHiddenRows::LastPromptPosition => {
            let shape = mlxcel_core::array_shape(&concatenated);
            debug_assert_eq!(shape.len(), 3, "concatenated hidden must be 3-D");
            mlxcel_core::slice(
                &concatenated,
                &[0, last_pos, 0],
                &[shape[0], last_pos + 1, shape[2]],
            )
        }
    }
}

/// Prompt rows before the first row of `first_hidden`: nonzero only under
/// the every-row policy when the target captured just the newest rows of
/// the prompt ([`DFlashTargetModel::limit_prefill_capture`], issue #3436).
fn first_context_start(rows: FirstHiddenRows, prompt_len: usize, first_hidden: &MlxArray) -> usize {
    match rows {
        FirstHiddenRows::EveryPromptRow => {
            let shape = mlxcel_core::array_shape(first_hidden);
            let captured = shape
                .get(shape.len().saturating_sub(2))
                .map_or(0, |&n| usize::try_from(n).unwrap_or(0));
            prompt_len.saturating_sub(captured)
        }
        FirstHiddenRows::LastPromptPosition => 0,
    }
}

/// DFlash B = 1 burst on any [`DFlashTargetModel`]: prefill through the
/// target's speculative verify hook, first-bonus and first-hidden
/// extraction, then `DFlashGenerator::run`.
///
/// `token_history` is the history-dependent-penalty context for the
/// first-bonus sample (repetition / frequency / presence / DRY); it is
/// forwarded to `sample_token_optimized` so a penalty-bearing request's first
/// bonus is byte-identical to the classic decode path. The round loop itself
/// runs greedy at temp=0, so the per-round target argmax is unaffected; only
/// the first bonus reads the history.
///
/// `cancel` is checked once per round so a disconnected client's burst stops
/// occupying the worker thread. `logprobs_config` controls per-token
/// log-probability capture: the first-bonus logprob is computed here from
/// the same penalty-adjusted logits the bonus was sampled from, and the
/// round-loop tokens' logprobs come back in `DFlashRunOutput::logprobs`.
///
/// **Drafter bind happens inside the round loop.** `DFlashGenerator::run`
/// binds and resets the drafter on every invocation; do not bind here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_dflash_on_target<T>(
    target: &T,
    prompt_tokens: &[i32],
    sampling: &SamplingConfig,
    token_history: &[i32],
    eos_token_ids: &[i32],
    owned_drafter: Box<dyn Drafter>,
    block_size: u32,
    max_tokens: usize,
    drafter_slot: &mut WorkerDrafterSlot,
    cancel: &AtomicBool,
    logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    prefill_partition: (usize, mlxcel_core::prefill_plan::PrefillSplits),
    seq_id: Option<mlxcel_core::cache::SequenceId>,
) -> Result<DFlashTargetRun, BurstOutcome>
where
    T: DFlashTargetModel,
{
    // An empty prompt would make `last_pos` -1 and hand MLX a negative slice
    // start, which throws inside C++ and crosses the cxx bridge as a process
    // abort rather than a failed request. The scheduler does not produce one
    // today; this is here so it stays a request error if it ever does.
    if prompt_tokens.is_empty() {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(
            "DFlash speculative burst got an empty prompt; there is no last position to \
             sample the first bonus from"
                .to_string(),
        ));
    }

    // Fresh caches for this request; the scheduler-owned `sequence_state`
    // map is never touched, which is why `run_dflash_burst` declines an
    // adopted prompt-cache prefix (`prefill_start_offset > 0`) to classic
    // decode before reaching here.
    // Compatibility gate BEFORE the round loop binds. `DFlashGenerator::run`
    // binds internally, and a bind that succeeds on a mismatched pairing
    // fails later as a shape error deep inside the drafter forward: a DSpark
    // drafter borrows the target's embedding table as both its input
    // embedding and its LM head, and its Markov head is a transition table
    // over the target's vocabulary. The default trait impl is a no-op, so the
    // Qwen 3.5 DFlash pairing is unaffected.
    if let Err(e) = owned_drafter.validate_target_compat(target as &dyn LanguageModel) {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(format!(
            "DFlash drafter incompatible with target: {e}"
        )));
    }
    // Family half of the same gate, which the drafter cannot decide on its
    // own: a plain DFlash drafter reads the target as a `LanguageModel`, sees
    // no architecture string, and returns `Ok` for an LFM2 or Muse target it
    // cannot run. Past this point the mismatch is an MLX shape throw inside
    // the drafter forward, and an MLX C++ exception crossing the cxx bridge
    // aborts the process rather than failing the request.
    if let Some(reason) = required_family_pairing_error(
        T::required_drafter_family(),
        DFlashDrafterFamily::of(owned_drafter.as_ref()),
    ) {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(format!(
            "DFlash drafter incompatible with target: {reason}"
        )));
    }
    // The target's own numbers (issue #3261): a tap at or past its depth, or
    // a drafter vocabulary that is not its vocabulary. A no-op for every
    // target that names a required family or keeps the default.
    if let Some(reason) = target.drafter_pairing_error(owned_drafter.as_ref()) {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(format!(
            "DFlash drafter incompatible with target: {reason}"
        )));
    }

    let mut caches: Vec<T::Cache> = target.make_dflash_caches_for(seq_id);
    target.enable_speculative_buffers(&mut caches, block_size as usize);
    if let Some(window) = owned_drafter.context_window() {
        target.limit_prefill_capture(&mut caches, window);
    }

    // Prefill through the speculative verify hook, capturing the layers the
    // drafter checkpoint asks for (`target_layer_ids`).
    let capture_layer_ids = owned_drafter
        .dflash_target_layer_ids()
        .filter(|ids| !ids.is_empty())
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| target.capture_layer_ids().to_vec());
    let prompt_arr = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let prefill_verify_start = Instant::now();
    let (chunk, boundary) = prefill_partition;
    let verify_out = target.prefill_burst(
        &prompt_arr,
        &mut caches,
        &capture_layer_ids,
        chunk,
        boundary,
    );
    let prefill_verify_ms = prefill_verify_start.elapsed().as_secs_f64() * 1000.0;

    // First bonus from the last-position logits, under the request's
    // sampler and penalty history so it matches the classic path's first
    // token byte for byte.
    let first_bonus_start = Instant::now();
    let last_pos = prompt_tokens.len() as i32 - 1;
    let logits = verify_out.logits();
    let logits_shape = mlxcel_core::array_shape(logits);
    let vocab = logits_shape[2];
    // The last logits row: every prompt row's for most targets, the last
    // row only for a target whose plain prefill computes just that one
    // (Raon-1 mini, issue #3436).
    let last_row = logits_shape[1] - 1;
    let last_logits = mlxcel_core::slice(
        logits,
        &[0, last_row, 0],
        &[logits_shape[0], last_row + 1, vocab],
    );
    let (first_bonus_arr, first_bonus_adjusted_logits) =
        mlxcel_core::sampling::sample_token_optimized(&last_logits, sampling, token_history);
    mlxcel_core::eval(&first_bonus_arr);
    let first_bonus = mlxcel_core::item_i32(&first_bonus_arr);
    let first_bonus_lp = mlxcel_core::sampling::compute_logprobs(
        &first_bonus_adjusted_logits,
        first_bonus,
        logprobs_config,
    );
    let first_bonus_ms = first_bonus_start.elapsed().as_secs_f64() * 1000.0;

    // First draft input: the captured hidden concatenated along the feature
    // axis, then either the last prompt position or every prompt row.
    let first_hidden_start = Instant::now();
    let hidden_states = verify_out.hidden_states();
    if hidden_states.is_empty() {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(
            "DFlash prefill returned no captured hidden layers".to_string(),
        ));
    }
    let first_hidden = first_hidden_for(
        T::first_hidden_rows(),
        concat_captured_hidden(hidden_states),
        last_pos,
    );
    let first_context_start =
        first_context_start(T::first_hidden_rows(), prompt_tokens.len(), &first_hidden);
    let first_hidden_ms = first_hidden_start.elapsed().as_secs_f64() * 1000.0;

    // Release the prefill's rollback snapshots before the round loop starts.
    // They are sized by the PROMPT, not by a verify block (LFM2 keeps one
    // `[1, S, hidden]` gated conv input per short-conv layer, Qwen 3.5 its GDN
    // states), and nothing past this point reads them: a prefill is never
    // rolled back. Holding `verify_out` to end of scope would pin that for the
    // whole generation. The captured hidden slabs survive regardless, because
    // the lazy `first_hidden` graph references them.
    drop(verify_out);

    let mut generator = DFlashGenerator::new(
        owned_drafter,
        sampling.clone(),
        block_size,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    // A sampled verify row reads the history classic decode would (#3436).
    generator.set_token_history(token_history.to_vec());
    generator.set_first_context_start(first_context_start);
    let result = generator.run(
        target,
        target as &dyn LanguageModel,
        &mut caches,
        first_bonus,
        first_hidden,
        eos_token_ids,
        max_tokens,
        cancel,
        logprobs_config,
    );

    // Whatever the round loop's outcome, recover the drafter so the slot is
    // consistent for the next request.
    let recovered = generator.into_drafter();
    drafter_slot.return_drafter(recovered, target as &dyn LanguageModel);

    match result {
        Ok(output) => {
            let diagnostics = output.diagnostics.clone();
            // b10621 spec_decode_* Prometheus counters (#1440).
            super::observability::spec_counters::record(
                diagnostics.proposed_tokens,
                diagnostics.accepted_tokens,
                diagnostics.rounds,
            );
            tracing::info!(
                block_size = diagnostics.block_size,
                rounds = diagnostics.rounds,
                proposed_tokens = diagnostics.proposed_tokens,
                accepted_tokens = diagnostics.accepted_tokens,
                acceptance_rate = diagnostics.acceptance_rate(),
                emitted_per_verify = diagnostics.emitted_per_verify(),
                zero_accept_rounds = diagnostics.zero_accept_rounds,
                partial_accept_rounds = diagnostics.partial_accept_rounds,
                full_accept_rounds = diagnostics.full_accept_rounds,
                classic_steps = diagnostics.classic_steps,
                prefill_verify_ms,
                first_bonus_ms,
                first_hidden_ms,
                bind_reset_ms = diagnostics.bind_reset_time_ms,
                draft_ms = diagnostics.draft_time_ms,
                verify_ms = diagnostics.verify_time_ms,
                target_argmax_sync_ms = diagnostics.target_argmax_time_ms,
                logprobs_ms = diagnostics.logprobs_time_ms,
                walk_ms = diagnostics.walk_time_ms,
                hidden_concat_ms = diagnostics.hidden_concat_time_ms,
                rollback_ms = diagnostics.rollback_time_ms,
                decode_ms = diagnostics.total_decode_time_ms,
                "DFlash diagnostics"
            );
            // The per-round accept lengths, at `debug` because they are one
            // line per burst and only a parity investigation reads them. They
            // are what lets an in-process replay reproduce a served burst's
            // exact cache history (rewind depth per round) rather than a
            // synthetic accept cycle, which is the gap issue #1935's residual
            // divergence is being chased through.
            tracing::debug!(accept_lens = ?output.accept_lens, "DFlash accept lengths");
            let mut tokens = Vec::with_capacity(output.tokens.len() + 1);
            tokens.push(first_bonus);
            tokens.extend(output.tokens);
            // Same assembly as `tokens`: the first-bonus logprob prepended to
            // the round loop's. Stays empty when logprobs are disabled so
            // `finalize_burst_success` emits plain `Token` events.
            let logprobs: Vec<Option<TokenLogprobData>> = if logprobs_config.enabled {
                let mut lp = Vec::with_capacity(output.logprobs.len() + 1);
                lp.push(first_bonus_lp);
                lp.extend(output.logprobs);
                lp
            } else {
                Vec::new()
            };
            Ok(DFlashTargetRun {
                tokens,
                logprobs,
                decode_time_ms: output.stats.decode_time_ms,
                // Client-facing acceptance counters (issue #1314) from the
                // same diagnostics the log line reports. The kind is `Dflash`
                // for DSpark too: it is the DFlash round loop and dispatch
                // variant that served the request.
                speculative: SpeculativeStats::from_counts(
                    DrafterKind::Dflash,
                    diagnostics.rounds,
                    diagnostics.proposed_tokens,
                    diagnostics.accepted_tokens,
                ),
            })
        }
        Err(e) => Err(BurstOutcome::Error(format!(
            "DFlash round loop failed: {e}"
        ))),
    }
}

/// Per-row emitted-token output of a batched burst dispatch arm.
pub(crate) type BatchedBurstTokens = Vec<Vec<i32>>;

/// DFlash batched burst (B > 1) on any [`DFlashTargetModel`]: `[B, L]`
/// prefill, per-row first-bonus and first-hidden extraction, then
/// `DFlashBatchedGenerator::run_batched`.
///
/// Mirrors [`run_dflash_on_target`] for the B > 1 path. The prompts are
/// equal length (window-collector contract) so the `[B, L]` prefill is
/// byte-identical to B separate `[1, L]` prefills. The drafter binds inside
/// `run_batched`, as on the B = 1 path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_dflash_batched_on_target<T>(
    target: &T,
    prompts: &[Vec<i32>],
    sampling: &SamplingConfig,
    eos_token_ids: &[i32],
    owned_drafter: Box<dyn Drafter>,
    block_size: u32,
    max_tokens: usize,
    drafter_slot: &mut WorkerDrafterSlot,
) -> Result<(BatchedBurstTokens, Instant), BurstOutcome>
where
    T: DFlashTargetModel,
{
    // Same guard as the B = 1 arm, plus the empty window `prompts[0]` would
    // panic on. Neither is reachable from the scheduler today.
    let Some(first_prompt) = prompts.first().filter(|p| !p.is_empty()) else {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(
            "DFlash batched speculative burst got an empty window or an empty prompt".to_string(),
        ));
    };
    let batch_size = prompts.len();
    let prompt_len = first_prompt.len();

    // Compatibility gate BEFORE the round loop binds. `DFlashGenerator::run`
    // binds internally, and a bind that succeeds on a mismatched pairing
    // fails later as a shape error deep inside the drafter forward: a DSpark
    // drafter borrows the target's embedding table as both its input
    // embedding and its LM head, and its Markov head is a transition table
    // over the target's vocabulary. The default trait impl is a no-op, so the
    // Qwen 3.5 DFlash pairing is unaffected.
    if let Err(e) = owned_drafter.validate_target_compat(target as &dyn LanguageModel) {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(format!(
            "DFlash drafter incompatible with target: {e}"
        )));
    }
    // Family half of the same gate, as on the B = 1 arm.
    if let Some(reason) = required_family_pairing_error(
        T::required_drafter_family(),
        DFlashDrafterFamily::of(owned_drafter.as_ref()),
    ) {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(format!(
            "DFlash drafter incompatible with target: {reason}"
        )));
    }
    // And the target's own numbers, as on the B = 1 arm.
    if let Some(reason) = target.drafter_pairing_error(owned_drafter.as_ref()) {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(format!(
            "DFlash drafter incompatible with target: {reason}"
        )));
    }

    let mut caches: Vec<T::Cache> = target.make_dflash_caches();
    target.enable_speculative_buffers(&mut caches, block_size as usize);

    let capture_layer_ids = owned_drafter
        .dflash_target_layer_ids()
        .filter(|ids| !ids.is_empty())
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| target.capture_layer_ids().to_vec());
    let mut flat_prompt: Vec<i32> = Vec::with_capacity(batch_size * prompt_len);
    for row in prompts {
        flat_prompt.extend_from_slice(row);
    }
    let prompt_arr =
        mlxcel_core::from_slice_i32(&flat_prompt, &[batch_size as i32, prompt_len as i32]);
    let verify_out =
        target.prefill_forward_with_capture_layers(&prompt_arr, &mut caches, &capture_layer_ids);

    // Per-row first bonus from the `[B, prompt_len, vocab]` last-position
    // logits.
    let logits = verify_out.logits();
    let logits_shape = mlxcel_core::array_shape(logits);
    let last_pos = prompt_len as i32 - 1;
    let vocab = logits_shape[2];
    let last_logits = mlxcel_core::slice(
        logits,
        &[0, last_pos, 0],
        &[logits_shape[0], last_pos + 1, vocab],
    );
    let (first_bonus_arr, _) =
        mlxcel_core::sampling::sample_token_optimized(&last_logits, sampling, &[]);
    mlxcel_core::eval(&first_bonus_arr);
    let first_bonus_per_row = scalar_tokens_from_array(&first_bonus_arr, batch_size);
    // Target prefill done, first bonus sampled: round 0 starts here.
    let prefill_end = Instant::now();

    let hidden_states = verify_out.hidden_states();
    if hidden_states.is_empty() {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(
            "DFlash batched prefill returned no captured hidden layers".to_string(),
        ));
    }
    let first_hidden = first_hidden_for(
        T::first_hidden_rows(),
        concat_captured_hidden(hidden_states),
        last_pos,
    );
    // Same reason as the B = 1 arm: the prefill's rollback snapshots are
    // prompt-sized and a prefill is never rolled back.
    drop(verify_out);

    let mut generator = DFlashBatchedGenerator::new(
        owned_drafter,
        sampling.clone(),
        block_size,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let run = generator.run_batched(
        target,
        target as &dyn LanguageModel,
        &mut caches,
        &first_bonus_per_row,
        first_hidden,
        eos_token_ids,
        max_tokens,
    );

    let recovered = generator.into_drafter();
    drafter_slot.return_drafter(recovered, target as &dyn LanguageModel);

    match run {
        Ok(output) => {
            debug_assert_eq!(output.tokens.len(), batch_size);
            tracing::debug!(
                batch_size,
                rounds = output.accept_lens.first().map_or(0, Vec::len),
                clamped_rounds = output.clamped_rounds,
                accept_lens = ?output.accept_lens,
                committed_accept_lens = ?output.committed_accept_lens,
                "DFlash batched burst accept lengths (walk vs committed under the active accept policy; \
                 clamped_rounds is non-zero only on the uniform-clamp arm)"
            );
            let mut rows: BatchedBurstTokens = Vec::with_capacity(batch_size);
            for (r, row_tokens) in output.tokens.into_iter().enumerate() {
                let mut full = Vec::with_capacity(row_tokens.len() + 1);
                full.push(first_bonus_per_row[r]);
                full.extend(row_tokens);
                rows.push(full);
            }
            Ok((rows, prefill_end))
        }
        Err(e) => Err(BurstOutcome::Error(format!(
            "DFlash batched round loop failed: {e}"
        ))),
    }
}

/// Materialise a per-row `Vec<i32>` from a `[B]` / `[B, 1]` token tensor
/// produced by `sample_token_optimized`. Mirrors the helper in
/// `gemma4_mtp_target.rs`; duplicated rather than re-exported because the
/// burst module is binary-side glue and the helper is a single-call
/// utility.
fn scalar_tokens_from_array(token_arr: &MlxArray, batch_size: usize) -> Vec<i32> {
    let flat = mlxcel_core::reshape(token_arr, &[batch_size as i32]);
    let mut out: Vec<i32> = Vec::with_capacity(batch_size);
    for r in 0..batch_size as i32 {
        let cell = mlxcel_core::slice(&flat, &[r], &[r + 1]);
        let scalar = mlxcel_core::reshape(&cell, &[]);
        out.push(mlxcel_core::item_i32(&scalar));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one policy split this trait exists to carry (issue #1339): a Qwen
    /// 3.5 DFlash drafter starts from the last prompt position, an LFM2
    /// DSpark drafter from the whole prompt. Both branches of
    /// [`first_hidden_for`] run here, because feeding a DSpark drafter one
    /// row instead of the prompt is not a shape error anywhere: it silently
    /// leaves the drafter's context cache holding one row and degrades
    /// acceptance.
    #[test]
    fn first_hidden_for_slices_only_under_the_last_position_policy() {
        let _runtime = crate::initialize_runtime();
        // `[1, 4, 6]`: four prompt rows of six features.
        let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let last_pos = 3;

        let whole = first_hidden_for(
            FirstHiddenRows::EveryPromptRow,
            mlxcel_core::from_slice_f32(&values, &[1, 4, 6]),
            last_pos,
        );
        assert_eq!(mlxcel_core::array_shape(&whole), vec![1, 4, 6]);
        assert_eq!(mlxcel_core::utils::array_to_vec_f32(&whole), values);

        let last = first_hidden_for(
            FirstHiddenRows::LastPromptPosition,
            mlxcel_core::from_slice_f32(&values, &[1, 4, 6]),
            last_pos,
        );
        assert_eq!(mlxcel_core::array_shape(&last), vec![1, 1, 6]);
        assert_eq!(
            mlxcel_core::utils::array_to_vec_f32(&last),
            vec![18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
        );
    }

    /// The drafter skips exactly the prompt rows a limited prefill did not
    /// capture, and nothing under the last-position policy (issue #3436).
    #[test]
    fn first_context_start_counts_the_uncaptured_prompt_rows() {
        let _runtime = crate::initialize_runtime();
        let tail = mlxcel_core::zeros(&[1, 3, 6], mlxcel_core::dtype::FLOAT32);
        assert_eq!(
            first_context_start(FirstHiddenRows::EveryPromptRow, 10, &tail),
            7
        );
        assert_eq!(
            first_context_start(FirstHiddenRows::EveryPromptRow, 3, &tail),
            0
        );
        let last = mlxcel_core::zeros(&[1, 1, 6], mlxcel_core::dtype::FLOAT32);
        assert_eq!(
            first_context_start(FirstHiddenRows::LastPromptPosition, 10, &last),
            0
        );
    }

    /// A target that admits one drafter family declines every other by
    /// name, before any forward (issues #1339 and #1343). Both facts the
    /// decision reads are pinned: the family policy each `DFlashTargetModel`
    /// impl declares, and the message the run arms return for an
    /// inadmissible combination. The drafter's own `validate_target_compat`
    /// returns `Ok` for a plain DFlash drafter on any target, so without this
    /// gate the mismatch reached MLX as a shape throw.
    #[test]
    fn a_restricted_target_declines_every_other_drafter_family() {
        use DFlashDrafterFamily::{DSpark, Dflash, Laguna, MuseAssistant};
        assert_eq!(
            <crate::models::Lfm2Model as DFlashTargetModel>::required_drafter_family(),
            Some(DSpark)
        );
        assert_eq!(
            <crate::vision::Lfm2VlModel as DFlashTargetModel>::required_drafter_family(),
            Some(DSpark)
        );
        assert_eq!(
            <crate::models::MuseGlimmerTextWrapper as DFlashTargetModel>::required_drafter_family(),
            Some(MuseAssistant)
        );
        assert_eq!(
            <crate::vision::MuseGlimmerVlmModel as DFlashTargetModel>::required_drafter_family(),
            Some(MuseAssistant)
        );
        assert_eq!(
            <crate::models::LagunaWrapper as DFlashTargetModel>::required_drafter_family(),
            Some(Laguna)
        );
        assert_eq!(
            <crate::models::Qwen35Model as DFlashTargetModel>::required_drafter_family(),
            None
        );
        assert_eq!(
            <crate::vision::Qwen35VLModel as DFlashTargetModel>::required_drafter_family(),
            None
        );

        let reason = required_family_pairing_error(Some(DSpark), Dflash)
            .expect("an LFM2 target with a plain DFlash drafter must decline");
        assert!(reason.contains("DSpark"), "{reason}");
        assert!(reason.contains("--model-draft"), "{reason}");

        let reason = required_family_pairing_error(Some(MuseAssistant), Dflash)
            .expect("a Muse target with a plain DFlash drafter must decline");
        assert!(reason.contains("Muse Glimmer assistant"), "{reason}");
        assert!(reason.contains("plain DFlash drafter"), "{reason}");
        let reason = required_family_pairing_error(Some(MuseAssistant), DSpark)
            .expect("a Muse target with a DSpark drafter must decline");
        assert!(reason.contains("DSpark drafter"), "{reason}");
        let reason = required_family_pairing_error(Some(DSpark), MuseAssistant)
            .expect("an LFM2 target with the Muse assistant must decline");
        assert!(reason.contains("Muse Glimmer assistant"), "{reason}");
        let reason = required_family_pairing_error(Some(Laguna), Dflash)
            .expect("a Laguna target with a plain DFlash drafter must decline");
        assert!(reason.contains("Laguna DFlash"), "{reason}");
        assert!(reason.contains("plain DFlash drafter"), "{reason}");
        let reason = required_family_pairing_error(Some(DSpark), Laguna)
            .expect("an LFM2 target with a Laguna drafter must decline");
        assert!(reason.contains("Laguna DFlash drafter"), "{reason}");

        for family in [Dflash, DSpark, MuseAssistant, Laguna] {
            assert!(
                required_family_pairing_error(Some(family), family).is_none(),
                "the published pairing must be admitted: {family:?}"
            );
            assert!(
                required_family_pairing_error(None, family).is_none(),
                "the Qwen 3.5 DFlash pairing must be untouched by this gate: {family:?}"
            );
        }
    }

    /// The batched variant gate reads its policy off the trait (issue
    /// #1343): Qwen 3.5 runs the B > 1 burst, LFM2 and Muse Glimmer decline
    /// the window to classic decode.
    #[test]
    fn supports_batched_is_a_qwen_only_policy() {
        assert!(<crate::models::Qwen35Model as DFlashTargetModel>::supports_batched());
        assert!(<crate::vision::Qwen35VLModel as DFlashTargetModel>::supports_batched());
        assert!(!<crate::models::Lfm2Model as DFlashTargetModel>::supports_batched());
        assert!(!<crate::vision::Lfm2VlModel as DFlashTargetModel>::supports_batched());
        assert!(!<crate::models::MuseGlimmerTextWrapper as DFlashTargetModel>::supports_batched());
        assert!(!<crate::vision::MuseGlimmerVlmModel as DFlashTargetModel>::supports_batched());
        assert!(!<crate::models::LagunaWrapper as DFlashTargetModel>::supports_batched());
    }

    /// The per-family policies, so a future `impl` that forgets
    /// `first_hidden_rows` is caught here rather than as a quiet acceptance
    /// regression on the LFM2 or Muse pairing.
    #[test]
    fn first_hidden_rows_defaults_to_the_qwen_policy() {
        assert_eq!(
            FirstHiddenRows::LastPromptPosition,
            <crate::models::Qwen35Model as DFlashTargetModel>::first_hidden_rows()
        );
        assert_eq!(
            FirstHiddenRows::EveryPromptRow,
            <crate::models::Lfm2Model as DFlashTargetModel>::first_hidden_rows()
        );
        assert_eq!(
            FirstHiddenRows::EveryPromptRow,
            <crate::models::MuseGlimmerTextWrapper as DFlashTargetModel>::first_hidden_rows()
        );
        assert_eq!(
            FirstHiddenRows::EveryPromptRow,
            <crate::vision::MuseGlimmerVlmModel as DFlashTargetModel>::first_hidden_rows()
        );
        assert_eq!(
            FirstHiddenRows::EveryPromptRow,
            <crate::models::LagunaWrapper as DFlashTargetModel>::first_hidden_rows()
        );
    }
}

/// The dense Qwen3 and Llama targets (issue #3261), kept in their own file so
/// this module stays readable.
#[cfg(test)]
#[path = "dflash_target_dense_tests.rs"]
mod dense_tests;
