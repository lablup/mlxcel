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

//! Adaptive MTP enable/decline policy (issue #333).
//!
//! ## What this replaces
//!
//! Before #333 the singleton (B=1) MTP burst was gated by a purely static
//! per-hardware rule ([`super::speculative_burst::mtp_b1_default`], issue
//! #165): non-batchable 12B targets default on everywhere, batch-capable 31B
//! targets default on only on M5+. Those static gates are correct for the
//! pairings they were measured on, but they leave performance on the table
//! when a new (target, drafter, hardware) pairing is favorable yet the static
//! rule declines it, and they keep running MTP on a pairing that turns out
//! unfavorable.
//!
//! ## What it does instead
//!
//! This module profiles the first few B=1 burst requests of a
//! (target, drafter, hardware) pairing and settles to a data-driven verdict:
//!
//! - **Profiling.** While profiling, [`MtpPolicy::should_attempt_b1`] forces
//!   MTP on so the burst path runs and reports its acceptance + latency split
//!   ([`MtpBurstProfile`]). This is the only way to discover a pairing the
//!   static gate would have declined. The cost is bounded to
//!   [`PROFILE_SAMPLE_TARGET`] qualifying requests and is paid once: the
//!   verdict is persisted, so a restart on the same pairing skips profiling.
//! - **Settling.** After [`PROFILE_SAMPLE_TARGET`] qualifying samples the
//!   accumulated profile yields an optimistic speedup estimate (see
//!   [`estimate_speedup`]). A clearly favorable estimate enables MTP, a clearly
//!   unfavorable one declines it (both overriding the static gate), and an
//!   ambiguous estimate falls back to the static per-hardware default (which
//!   already encodes the M1-Ultra-vs-M5 reality).
//! - **Manual override.** `MLXCEL_ENABLE_MTP_B1` still pins the decision in
//!   both directions; when it is set the policy never profiles. Setting
//!   `MLXCEL_MTP_ADAPTIVE=0` disables the adaptive path entirely and restores
//!   the pre-#333 pure-static gates.
//!
//! ## Exactness is untouched
//!
//! MTP speculative decode is mathematically exact: the drafter proposes, the
//! target verifies, and accepted tokens are exactly what the target would have
//! produced. At `temperature == 0` MTP output is byte-identical to classic
//! decode. This policy only decides *when* to run MTP; it never touches the
//! tokens the burst emits, so the exactness guarantee is preserved.
//!
//! ## What is persisted
//!
//! Only a coarse per-pairing hint: the enable/decline verdict, the coarse
//! measured acceptance rate, and the sample count. No prompt data, no token
//! ids, nothing request-identifying. Hints live one file per pairing under
//! `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/<key-hash>.json`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use mlxcel_core::speculative::mtp::MtpAcceptanceSummary;

/// Number of qualifying B=1 samples (each with at least one speculative round)
/// to accumulate before settling on a verdict. "A few requests" per the issue;
/// kept small so the profiling cost is bounded and paid once per pairing.
pub(crate) const PROFILE_SAMPLE_TARGET: usize = 4;

/// Classic-step probe rounds requested per profiled burst (issue #736). While
/// profiling, the burst's generator runs this many drafterless rounds whose
/// `[1, 1]` verify forward is shape-identical to a classic decode step; the
/// measured mean is the classic single-token step time the measured-cost
/// estimator divides by. Each probe emits one real greedy token (no wasted
/// compute), so the total profiling overhead is
/// `PROFILE_SAMPLE_TARGET * PROFILE_PROBE_ROUNDS_PER_BURST` classic-paced
/// tokens per pairing lifetime.
pub(crate) const PROFILE_PROBE_ROUNDS_PER_BURST: usize = 2;

/// Optimistic speedup at or above which MTP is auto-enabled (overriding a
/// static decline). [`estimate_speedup`] is an upper bound: it assumes the
/// K-wide verify forward costs the same as a single classic decode forward.
/// Real speedup is never higher, so requiring a comfortable margin above 1.0
/// keeps the enable decision robust to verify-forward inflation on
/// compute-bound GPUs.
pub(crate) const ENABLE_SPEEDUP_FLOOR: f64 = 1.5;

/// Optimistic speedup at or below which MTP is auto-declined (overriding a
/// static enable). At or below 1.0 the pairing cannot beat classic decode even
/// under the most generous assumption, so declining is safe.
pub(crate) const DECLINE_SPEEDUP_CEIL: f64 = 1.0;

/// On-disk hint format version. Bump when the schema changes OR when verdict
/// semantics change; older/newer versions are ignored on load so a stale file
/// just triggers a re-profile.
/// v2: added `block_size` to `PolicyKey` and `PolicyHint` (K affects acceptance
/// length and verify latency, so a verdict profiled at one K must not be reused
/// if K changes).
/// v3: verdicts are settled by the measured-cost estimator (issue #736) on top
/// of the multirow-qmv verify kernel (issue #725). v2 verdicts on CUDA were
/// settled against the pre-#725 per-row verify (and the sqrt(K) heuristic), so
/// they are systematically stale in both directions and must re-profile.
const HINT_VERSION: u32 = 3;

/// Subdirectory under the mlxcel cache root that holds per-pairing hints.
const HINT_SUBDIR: &str = "mtp-policy";

// ── Pure decision core ──────────────────────────────────────────────────────

/// Optimistic upper-bound speedup of MTP over classic decode for one
/// aggregated profile.
///
/// Per speculative round the MTP path pays `drafter_ms + verify_ms` and yields
/// `accepted_len + 1` tokens (the accepted draft prefix plus the target's
/// bonus/correction token). Classic decode pays one target forward per token;
/// the K-wide verify forward costs about one classic forward in the
/// memory-bandwidth-bound regime, so we approximate `classic_ms ≈ verify_ms`.
/// That gives:
///
/// ```text
/// speedup = (accepted_len + 1) / (1 + drafter_ms / verify_ms)
/// ```
///
/// This is an *upper* bound because a K-wide verify is never cheaper than a
/// 1-wide forward; on a compute-bound GPU it is strictly more expensive, so
/// the real speedup is lower. Returns `None` when there is no timing signal
/// (`verify_ms <= 0`), which the caller treats as "ambiguous".
// Retained as the closed-form (bandwidth-bound) reference exercised by the unit
// tests; production settles through `estimate_speedup_scaled` with a
// backend-specific multiple (issue #638).
#[allow(dead_code)]
#[must_use]
pub(crate) fn estimate_speedup(accepted_len: f64, verify_ms: f64, drafter_ms: f64) -> Option<f64> {
    // Bandwidth-bound (Apple Silicon) default: the K-wide verify amortizes to
    // about one classic decode forward, so the verify-cost multiple is 1.0.
    estimate_speedup_scaled(accepted_len, verify_ms, drafter_ms, 1.0)
}

/// Backend-scaled variant of [`estimate_speedup`] (issue #638).
///
/// `verify_cost_multiple` is how many classic decode forwards one K-wide verify
/// forward costs. On memory-bandwidth-bound hardware (Apple Silicon) a K-wide
/// verify reads the weights once, so it costs about one classic forward
/// (`multiple == 1.0`) and this reduces to the original estimate. On
/// compute-bound hardware (GB10 / Blackwell) the target GPU is already saturated
/// at B=1, so a K-wide verify does *not* run for free; the round cost grows with
/// K and the optimistic `multiple == 1.0` model over-predicts the speedup.
///
/// One round yields `accepted_len + 1` tokens and costs
/// `verify_cost_multiple` classic-forward-equivalents for the verify plus the
/// drafter's share (`verify_cost_multiple * drafter_ms / verify_ms`), giving:
///
/// ```text
/// speedup = (accepted_len + 1) / (verify_cost_multiple * (1 + drafter_ms / verify_ms))
/// ```
///
/// which recovers the original formula exactly at `verify_cost_multiple == 1.0`.
#[must_use]
pub(crate) fn estimate_speedup_scaled(
    accepted_len: f64,
    verify_ms: f64,
    drafter_ms: f64,
    verify_cost_multiple: f64,
) -> Option<f64> {
    if !verify_ms.is_finite()
        || verify_ms <= 0.0
        || drafter_ms < 0.0
        || accepted_len < 0.0
        || !verify_cost_multiple.is_finite()
        || verify_cost_multiple <= 0.0
    {
        return None;
    }
    Some((accepted_len + 1.0) / (verify_cost_multiple * (1.0 + drafter_ms / verify_ms)))
}

/// Measured-cost speedup estimate (issue #736).
///
/// One speculative round yields `accepted_len + 1` tokens and costs
/// `round_ms` (verify + drafter + walk/finalize/re-arm overhead). Classic
/// decode yields one token per `classic_step_ms` (measured from the burst's
/// classic-step probe rounds, whose `[1, 1]` verify forward is
/// shape-identical to a classic decode step). So:
///
/// ```text
/// speedup = (accepted_len + 1) * classic_step_ms / round_ms
/// ```
///
/// Unlike [`estimate_speedup_scaled`], this makes no assumption about how
/// the K-wide verify relates to a classic forward: on hardware where the
/// verify amortizes (Apple Silicon, post-#725 CUDA multirow qmv) the measured
/// ratio reflects it, and on hardware where it does not (pre-#725 per-row
/// qmv, unknown backends) the measured ratio reflects that too. The probe's
/// verify carries the shared-KV capture that classic decode does not pay, so
/// `classic_step_ms` is slightly over-measured, which mildly INFLATES the
/// estimate (a slower-looking classic baseline raises the ratio; measured
/// +1.7% on GB10, probe 76.3 ms vs forced-classic ~75 ms), well inside the
/// 1.5x enable margin. Returns `None` without a usable signal.
#[must_use]
pub(crate) fn estimate_speedup_measured(
    accepted_len: f64,
    round_ms: f64,
    classic_step_ms: f64,
) -> Option<f64> {
    if !round_ms.is_finite()
        || round_ms <= 0.0
        || !classic_step_ms.is_finite()
        || classic_step_ms <= 0.0
        || accepted_len < 0.0
    {
        return None;
    }
    Some((accepted_len + 1.0) * classic_step_ms / round_ms)
}

/// Which side of the decision an estimated speedup falls on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeedupZone {
    /// Clearly favorable: enable MTP, overriding a static decline.
    Enable,
    /// Clearly unfavorable: decline MTP, overriding a static enable.
    Decline,
    /// Neither clearly favorable nor unfavorable: defer to the static default.
    Ambiguous,
}

/// Classify an estimated speedup into a decision zone.
#[must_use]
pub(crate) fn classify_speedup(speedup: f64) -> SpeedupZone {
    if speedup <= DECLINE_SPEEDUP_CEIL {
        SpeedupZone::Decline
    } else if speedup >= ENABLE_SPEEDUP_FLOOR {
        SpeedupZone::Enable
    } else {
        SpeedupZone::Ambiguous
    }
}

/// Resolve a speedup zone (plus the static per-hardware default) into a
/// concrete run/decline boolean. The ambiguous zone and a missing estimate
/// both defer to `static_default`.
#[must_use]
pub(crate) fn resolve_verdict(zone: Option<SpeedupZone>, static_default: bool) -> bool {
    match zone {
        Some(SpeedupZone::Enable) => true,
        Some(SpeedupZone::Decline) => false,
        Some(SpeedupZone::Ambiguous) | None => static_default,
    }
}

// ── Per-request profile sample ────────────────────────────────────────────────

/// One B=1 MTP burst's coarse profile, handed back to the scheduler so it can
/// feed [`MtpPolicy::record_b1_sample`]. Built from the generator's
/// [`MtpAcceptanceSummary`] plus the batch size and prompt length the issue
/// asks the profiler to record. Carries no prompt data.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MtpBurstProfile {
    /// Concurrent batch size of the burst (1 for the singleton path).
    pub batch_size: usize,
    /// Prompt length in tokens (the prompt-shape dimension; eligibility is
    /// already enforced by `should_burst_for_sequence`).
    pub prompt_len: usize,
    /// Speculative rounds executed.
    pub rounds: usize,
    /// Total draft tokens proposed across all rounds.
    pub proposed_tokens: usize,
    /// Total draft tokens accepted across all rounds.
    pub accepted_draft_tokens: usize,
    /// Cumulative drafter latency (ms) across all rounds.
    pub draft_ms: f64,
    /// Cumulative verify-forward latency (ms) across all rounds.
    pub verify_forward_ms: f64,
    /// Cumulative non-forward round overhead (ms): speculative walk, verify
    /// finalize, and drafter shared-KV re-arm (issue #736).
    pub overhead_ms: f64,
    /// Classic-step probe rounds the burst executed while profiling
    /// (issue #736).
    pub probe_rounds: usize,
    /// Cumulative probe verify-forward latency (ms); `probe_ms /
    /// probe_rounds` is the measured classic single-token step time.
    pub probe_ms: f64,
}

impl MtpBurstProfile {
    /// Build a profile sample from a generator acceptance summary plus the
    /// batch size and prompt length.
    #[must_use]
    pub(crate) fn from_summary(
        summary: MtpAcceptanceSummary,
        batch_size: usize,
        prompt_len: usize,
    ) -> Self {
        Self {
            batch_size,
            prompt_len,
            rounds: summary.rounds,
            proposed_tokens: summary.proposed_tokens,
            accepted_draft_tokens: summary.accepted_draft_tokens,
            draft_ms: summary.draft_ms,
            verify_forward_ms: summary.verify_forward_ms,
            overhead_ms: summary.overhead_ms,
            probe_rounds: summary.probe_rounds,
            probe_ms: summary.probe_ms,
        }
    }
}

// ── Profile accumulator ───────────────────────────────────────────────────────

/// Running aggregate over the profiling window. Weights every round equally by
/// summing totals and dividing at verdict time, so a longer request
/// contributes proportionally more signal than a short one.
#[derive(Debug, Default, Clone)]
pub(crate) struct ProfileAccumulator {
    samples: usize,
    total_rounds: usize,
    total_proposed: usize,
    total_accepted: usize,
    total_draft_ms: f64,
    total_verify_ms: f64,
    /// Largest concurrent batch size seen (the batch-size dimension the issue
    /// asks the profiler to record). Always 1 for the singleton B=1 path.
    max_batch_size: usize,
    /// Total prompt length across samples, for the mean (prompt-shape
    /// dimension). Burst eligibility itself is enforced upstream by
    /// `should_burst_for_sequence`.
    total_prompt_len: usize,
    /// Largest prompt length seen.
    max_prompt_len: usize,
    /// Cumulative non-forward round overhead (ms) across samples
    /// (issue #736).
    total_overhead_ms: f64,
    /// Per-burst mean classic-step probe times (ms), one entry per sample
    /// that ran probes (issue #736). The estimator takes the MEDIAN of
    /// these: the first burst's probes absorb one-time CUDA kernel/graph
    /// compilation for the `[1, 1]` verify shape (measured ~1.6 s vs the
    /// ~74 ms steady state on GB10), and a median over per-burst means is
    /// robust to that one-off pollution where a plain mean is not.
    probe_burst_means: Vec<f64>,
}

impl ProfileAccumulator {
    /// Fold one sample in. Samples with zero speculative rounds carry no
    /// acceptance or per-round timing signal (the request hit EOS on the seed
    /// bonus or the budget went to probe rounds), so they neither count
    /// toward [`PROFILE_SAMPLE_TARGET`] nor skew the speculative aggregate.
    /// Their classic-step probe measurements are still folded in (issue
    /// #736): a probe's signal is independent of whether the same burst also
    /// completed a speculative round.
    pub(crate) fn add(&mut self, profile: &MtpBurstProfile) {
        // Cap the probe-mean history: the median only needs a handful of
        // bursts, and without a cap a pairing that never settles (an
        // all-tiny-completion workload where probes consume every round
        // budget) would grow this vec by one entry per request for the
        // process lifetime.
        const MAX_PROBE_BURST_MEANS: usize = 2 * PROFILE_SAMPLE_TARGET;
        if profile.probe_rounds > 0 && self.probe_burst_means.len() < MAX_PROBE_BURST_MEANS {
            let mean = profile.probe_ms / profile.probe_rounds as f64;
            if mean.is_finite() && mean > 0.0 {
                self.probe_burst_means.push(mean);
            }
        }
        if profile.rounds == 0 {
            return;
        }
        self.samples += 1;
        self.total_rounds += profile.rounds;
        self.total_proposed += profile.proposed_tokens;
        self.total_accepted += profile.accepted_draft_tokens;
        self.total_draft_ms += profile.draft_ms;
        self.total_verify_ms += profile.verify_forward_ms;
        self.total_overhead_ms += profile.overhead_ms;
        self.max_batch_size = self.max_batch_size.max(profile.batch_size);
        self.total_prompt_len += profile.prompt_len;
        self.max_prompt_len = self.max_prompt_len.max(profile.prompt_len);
    }

    /// Number of qualifying samples accumulated so far.
    #[must_use]
    pub(crate) fn samples(&self) -> usize {
        self.samples
    }

    /// Mean accepted draft tokens per round.
    #[must_use]
    pub(crate) fn accepted_len(&self) -> f64 {
        if self.total_rounds == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_rounds as f64
        }
    }

    /// Mean drafter latency (ms) per round.
    #[must_use]
    pub(crate) fn drafter_ms(&self) -> f64 {
        if self.total_rounds == 0 {
            0.0
        } else {
            self.total_draft_ms / self.total_rounds as f64
        }
    }

    /// Mean verify-forward latency (ms) per round.
    #[must_use]
    pub(crate) fn verify_ms(&self) -> f64 {
        if self.total_rounds == 0 {
            0.0
        } else {
            self.total_verify_ms / self.total_rounds as f64
        }
    }

    /// Mean non-forward overhead (ms) per round (walk + finalize + shared-KV
    /// re-arm; issue #736).
    #[must_use]
    pub(crate) fn overhead_ms(&self) -> f64 {
        if self.total_rounds == 0 {
            0.0
        } else {
            self.total_overhead_ms / self.total_rounds as f64
        }
    }

    /// Mean full round cost (ms): verify + drafter + overhead (issue #736).
    #[must_use]
    pub(crate) fn round_cost_ms(&self) -> f64 {
        self.verify_ms() + self.drafter_ms() + self.overhead_ms()
    }

    /// Measured classic single-token step time (ms): the MEDIAN of the
    /// per-burst probe means, or `None` when no probe ran (issue #736). The
    /// median discards the first burst's one-time CUDA kernel/graph
    /// compilation cost, which would inflate a plain mean by an order of
    /// magnitude and produce a nonsense speedup estimate.
    #[must_use]
    pub(crate) fn classic_step_ms(&self) -> Option<f64> {
        if self.probe_burst_means.is_empty() {
            return None;
        }
        let mut sorted = self.probe_burst_means.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        (median.is_finite() && median > 0.0).then_some(median)
    }

    /// Coarse acceptance rate (accepted / proposed) across the window.
    #[must_use]
    pub(crate) fn acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_proposed as f64
        }
    }

    /// Largest concurrent batch size observed during profiling.
    #[must_use]
    pub(crate) fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Largest prompt length observed during profiling.
    #[must_use]
    pub(crate) fn max_prompt_len(&self) -> usize {
        self.max_prompt_len
    }

    /// Mean prompt length across profiling samples (0 with no samples).
    #[must_use]
    pub(crate) fn mean_prompt_len(&self) -> usize {
        self.total_prompt_len.checked_div(self.samples).unwrap_or(0)
    }

    /// Optimistic speedup estimate from the aggregate, or `None` when there is
    /// no timing signal yet. Assumes bandwidth-bound verify amortization
    /// (`verify_cost_multiple == 1.0`); the policy uses
    /// [`Self::estimated_speedup_scaled`] with a backend-specific multiple.
    /// Retained for the no-signal unit test; production uses the scaled form.
    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn estimated_speedup(&self) -> Option<f64> {
        estimate_speedup(self.accepted_len(), self.verify_ms(), self.drafter_ms())
    }

    /// Speedup estimate scaled by a backend-specific verify-cost multiple
    /// (issue #638). `verify_cost_multiple == 1.0` reproduces
    /// [`Self::estimated_speedup`] exactly (Apple Silicon); a value `> 1.0`
    /// de-rates the estimate for hardware where a K-wide verify does not
    /// amortize to one classic forward. Since issue #736 this is the
    /// fallback used only when no classic-step probe measurement exists
    /// ([`Self::estimated_speedup_measured`] returns `None`).
    #[must_use]
    pub(crate) fn estimated_speedup_scaled(&self, verify_cost_multiple: f64) -> Option<f64> {
        estimate_speedup_scaled(
            self.accepted_len(),
            self.verify_ms(),
            self.drafter_ms(),
            verify_cost_multiple,
        )
    }

    /// Measured-cost speedup estimate (issue #736): tokens per round divided
    /// by the measured round-cost-to-classic-step ratio. `None` when no
    /// classic-step probe measurement or no speculative round exists yet;
    /// the caller then falls back to [`Self::estimated_speedup_scaled`].
    #[must_use]
    pub(crate) fn estimated_speedup_measured(&self) -> Option<f64> {
        let classic = self.classic_step_ms()?;
        estimate_speedup_measured(self.accepted_len(), self.round_cost_ms(), classic)
    }
}

// ── Pairing key + hardware label ──────────────────────────────────────────────

/// Identity of a (target, drafter, hardware, block_size) pairing. The persisted
/// hint is keyed on this; the file name is a hash so model basenames never have
/// to be filesystem-safe, and the readable fields are stored inside the file.
///
/// `block_size` (K) is included because acceptance length and verify latency
/// both depend on K: a verdict profiled at K=8 must not be reused at K=4 (or
/// vice versa). Changing `--num-draft-tokens` / `MLXCEL_DRAFT_BLOCK_SIZE` therefore
/// produces a different key and triggers a fresh profiling window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PolicyKey {
    target: String,
    drafter: String,
    hardware: String,
    block_size: u32,
}

impl PolicyKey {
    pub(crate) fn new(target: String, drafter: String, hardware: String, block_size: u32) -> Self {
        Self {
            target,
            drafter,
            hardware,
            block_size,
        }
    }

    /// Human-readable key (`target|drafter|hardware|K{block_size}`) for logs
    /// and the stored hint body.
    #[must_use]
    pub(crate) fn display(&self) -> String {
        format!(
            "{}|{}|{}|K{}",
            self.target, self.drafter, self.hardware, self.block_size
        )
    }

    /// Stable short hash used as the hint file stem.
    #[must_use]
    pub(crate) fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.display().as_bytes());
        let digest = hasher.finalize();
        // 16 hex chars (8 bytes) is ample to avoid collisions across the
        // handful of pairings a host ever profiles.
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Coarse hardware-class label, e.g. `"M5-16c"` / `"M1-20c"`. Apple-silicon
/// generation plus the GPU-core proxy distinguishes M1 Max from M1 Ultra (the
/// regression discriminator in #165) without recording anything
/// request-specific. Non-Apple hosts collapse to `"Unknown-0c"`.
#[must_use]
pub(crate) fn hardware_label() -> String {
    let hw = mlxcel_core::hardware::get_hardware();
    format!("{}-{}c", hw.silicon_gen, hw.gpu_core_count)
}

// ── Persisted hint ────────────────────────────────────────────────────────────

/// The enable/decline verdict as persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Verdict {
    Enable,
    Decline,
}

impl Verdict {
    #[must_use]
    pub(crate) fn from_run(run: bool) -> Self {
        if run {
            Verdict::Enable
        } else {
            Verdict::Decline
        }
    }

    #[must_use]
    pub(crate) fn runs(self) -> bool {
        matches!(self, Verdict::Enable)
    }
}

/// On-disk hint. Deliberately coarse: a verdict, the coarse acceptance rate,
/// and the sample count, plus the readable key fields for debuggability. No
/// prompt data, no token ids, nothing request-identifying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PolicyHint {
    pub version: u32,
    pub target: String,
    pub drafter: String,
    pub hardware: String,
    /// Draft block size (K) profiled under. A hint is only loaded when the
    /// current K matches; changing block size re-profiles.
    pub block_size: u32,
    pub verdict: Verdict,
    /// Coarse measured acceptance rate, rounded to two decimals.
    pub acceptance_rate: f64,
    /// Qualifying samples behind the verdict.
    pub samples: usize,
}

impl PolicyHint {
    fn new(key: &PolicyKey, verdict: Verdict, acceptance_rate: f64, samples: usize) -> Self {
        Self {
            version: HINT_VERSION,
            target: key.target.clone(),
            drafter: key.drafter.clone(),
            hardware: key.hardware.clone(),
            block_size: key.block_size,
            verdict,
            // Round to two decimals so the persisted value stays coarse.
            acceptance_rate: (acceptance_rate * 100.0).round() / 100.0,
            samples,
        }
    }
}

// ── Persistence store ─────────────────────────────────────────────────────────

/// Reads and writes per-pairing hints. `dir == None` (no resolvable cache
/// root) makes persistence a silent no-op; the policy still works in-memory
/// for the session.
#[derive(Debug, Clone)]
pub(crate) struct PolicyStore {
    dir: Option<PathBuf>,
}

impl PolicyStore {
    /// Resolve the store under the mlxcel cache root
    /// (`${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy`).
    #[must_use]
    pub(crate) fn from_cache_root() -> Self {
        Self {
            dir: mlxcel_core::cache_root().map(|root| root.join(HINT_SUBDIR)),
        }
    }

    /// Construct a store rooted at an explicit directory (test injection).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_dir(dir: Option<PathBuf>) -> Self {
        Self { dir }
    }

    fn hint_file(&self, key: &PolicyKey) -> Option<PathBuf> {
        self.dir
            .as_ref()
            .map(|dir| dir.join(format!("{}.json", key.hash())))
    }

    /// Load a stored hint for `key`, or `None` when there is no usable hint
    /// (missing file, unreadable, unparseable, wrong version, key mismatch from
    /// a hash collision, or a block_size mismatch meaning the hint was profiled
    /// under a different K and must not be reused).
    #[must_use]
    pub(crate) fn load(&self, key: &PolicyKey) -> Option<PolicyHint> {
        let path = self.hint_file(key)?;
        let raw = std::fs::read_to_string(&path).ok()?;
        let hint: PolicyHint = serde_json::from_str(&raw).ok()?;
        if hint.version != HINT_VERSION
            || hint.target != key.target
            || hint.drafter != key.drafter
            || hint.hardware != key.hardware
            || hint.block_size != key.block_size
        {
            return None;
        }
        Some(hint)
    }

    /// Persist a hint. Best-effort: creates the directory, writes to a
    /// temporary file, and renames it into place (atomic on the same volume).
    /// One file per pairing, so concurrent writers for different pairings
    /// never contend. Returns the IO error for the caller to log; never
    /// panics. On rename failure the temporary file is cleaned up so no
    /// orphaned `.tmp.<pid>` files accumulate.
    pub(crate) fn save(&self, hint: &PolicyHint) -> std::io::Result<()> {
        let Some(dir) = self.dir.clone() else {
            return Ok(());
        };
        let key = PolicyKey::new(
            hint.target.clone(),
            hint.drafter.clone(),
            hint.hardware.clone(),
            hint.block_size,
        );
        let Some(path) = self.hint_file(&key) else {
            return Ok(());
        };
        std::fs::create_dir_all(&dir)?;
        let body = serde_json::to_string_pretty(hint)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = dir.join(format!("{}.json.tmp.{}", key.hash(), std::process::id()));
        std::fs::write(&tmp, body.as_bytes())?;
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    }
}

// ── Environment overrides ─────────────────────────────────────────────────────

/// Values that read as "off" for the boolean env knobs, matching the existing
/// `mtp_b1_default` convention.
fn env_is_off(value: &str) -> bool {
    matches!(value, "0" | "false" | "FALSE" | "no" | "off")
}

/// Parse `MLXCEL_ENABLE_MTP_B1` into a manual force: `Some(true)` forces MTP
/// on, `Some(false)` forces it off, `None` leaves the decision adaptive.
#[must_use]
pub(crate) fn parse_force_override(value: Option<&str>) -> Option<bool> {
    value.map(|v| !env_is_off(v))
}

/// Whether the adaptive policy is enabled. Defaults on; `MLXCEL_MTP_ADAPTIVE`
/// set to an off value restores the pre-#333 pure-static gates.
#[must_use]
pub(crate) fn adaptive_enabled(value: Option<&str>) -> bool {
    value.map(|v| !env_is_off(v)).unwrap_or(true)
}

// ── Stateful policy ───────────────────────────────────────────────────────────

/// Internal policy state machine.
#[derive(Debug, Clone)]
enum PolicyState {
    /// `MLXCEL_ENABLE_MTP_B1` pinned the decision; never profiles.
    Forced(bool),
    /// Profiling: force MTP on and accumulate until the window closes.
    Profiling(ProfileAccumulator),
    /// A verdict is in effect (from profiling or a loaded hint).
    Settled(bool),
}

/// Adaptive MTP enable/decline policy for one worker's (target, drafter,
/// hardware) pairing. Single-threaded: the scheduler owns it on the worker
/// thread, so it needs no locking. Consult [`Self::should_attempt_b1`] before
/// dispatching a B=1 MTP burst and feed [`Self::record_b1_sample`] afterward.
#[derive(Debug, Clone)]
pub(crate) struct MtpPolicy {
    key: PolicyKey,
    target_supports_batching: bool,
    has_neural_accelerator: bool,
    /// True on compute-bound (non-Apple-Silicon, e.g. CUDA / GB10) hardware,
    /// where a K-wide verify forward does not amortize to one classic decode
    /// forward. Drives the backend-specific verify-cost multiple (issue #638).
    compute_bound: bool,
    state: PolicyState,
    store: PolicyStore,
}

impl MtpPolicy {
    /// Build the policy for the current process.
    ///
    /// Returns `None` when the adaptive path is disabled
    /// (`MLXCEL_MTP_ADAPTIVE=0`); the caller then keeps the pre-#333 static
    /// gate. Reads `MLXCEL_ENABLE_MTP_B1` and `MLXCEL_MTP_ADAPTIVE` exactly
    /// once here, mirroring the env-caching pattern, so the per-request gate
    /// touches no environment.
    ///
    /// `block_size` is the resolved draft block size (K) for this pairing. It
    /// is part of the key so a verdict profiled at one K is never reused when K
    /// changes (acceptance length and verify latency both depend on K).
    #[must_use]
    pub(crate) fn initialize(
        target_id: String,
        drafter_id: String,
        block_size: u32,
        target_supports_batching: bool,
    ) -> Option<Self> {
        if !adaptive_enabled(std::env::var("MLXCEL_MTP_ADAPTIVE").ok().as_deref()) {
            return None;
        }
        let key = PolicyKey::new(target_id, drafter_id, hardware_label(), block_size);
        let hw = mlxcel_core::hardware::get_hardware();
        let has_neural_accelerator = hw.has_neural_accelerator;
        // Compute-bound = non-Apple-Silicon (CUDA / GB10): the runtime hardware
        // probe reports `AppleSiliconGen::Unknown` off Apple GPUs. On such hosts
        // the K-wide verify does not amortize (issue #638), so the policy
        // de-rates its optimistic speedup estimate. Caveat: `parse_silicon_gen`
        // also maps Apple generations newer than the enumerated ones to
        // `Unknown`, so the "Apple byte-identical" guarantee is scoped to the
        // enumerated gens; extend the enum when a new Apple generation ships
        // (same staleness contract as `has_neural_accelerator`).
        let compute_bound = matches!(
            hw.silicon_gen,
            mlxcel_core::hardware::AppleSiliconGen::Unknown
        );
        let force = parse_force_override(std::env::var("MLXCEL_ENABLE_MTP_B1").ok().as_deref());
        let store = PolicyStore::from_cache_root();
        Some(Self::from_parts(
            key,
            target_supports_batching,
            has_neural_accelerator,
            compute_bound,
            force,
            store,
        ))
    }

    /// Assemble the policy from resolved parts (test seam: lets a test inject a
    /// store directory and pre-resolved env/hardware without touching the real
    /// cache or process environment).
    #[must_use]
    pub(crate) fn from_parts(
        key: PolicyKey,
        target_supports_batching: bool,
        has_neural_accelerator: bool,
        compute_bound: bool,
        force: Option<bool>,
        store: PolicyStore,
    ) -> Self {
        let state = if let Some(forced) = force {
            PolicyState::Forced(forced)
        } else if let Some(hint) = store.load(&key) {
            tracing::info!(
                "adaptive MTP policy: loaded persisted verdict {:?} (acceptance≈{:.2}) for {}",
                hint.verdict,
                hint.acceptance_rate,
                key.display(),
            );
            PolicyState::Settled(hint.verdict.runs())
        } else {
            PolicyState::Profiling(ProfileAccumulator::default())
        };
        Self {
            key,
            target_supports_batching,
            has_neural_accelerator,
            compute_bound,
            state,
            store,
        }
    }

    /// Backend-specific verify-cost multiple for the FALLBACK speedup
    /// estimate (issue #638), used only when the profiling window collected
    /// no classic-step probe measurement (issue #736). `1.0` on
    /// bandwidth-bound Apple Silicon (the K-wide verify amortizes to one
    /// classic forward). `sqrt(K)` on non-Apple hosts, a shape heuristic
    /// calibrated against the pre-#725 per-row-qmv verify; it is known to be
    /// wrong in both directions across kernel eras (it under-costed the
    /// pre-#725 linear-in-K verify and over-costs the post-#725 multirow
    /// verify), which is exactly why the measured estimator replaced it as
    /// the primary path.
    #[must_use]
    fn verify_cost_multiple(&self) -> f64 {
        if self.compute_bound {
            (self.key.block_size as f64).max(1.0).sqrt()
        } else {
            1.0
        }
    }

    /// Classic-step probe rounds the burst should run for the next request
    /// (issue #736): a few per burst while profiling, zero once forced or
    /// settled, so the steady state carries no probe cost.
    #[must_use]
    pub(crate) fn profile_probe_rounds(&self) -> usize {
        match &self.state {
            PolicyState::Profiling(_) => PROFILE_PROBE_ROUNDS_PER_BURST,
            PolicyState::Forced(_) | PolicyState::Settled(_) => 0,
        }
    }

    /// The static per-hardware default ([`super::speculative_burst::mtp_b1_default`]
    /// with no env override) used to resolve an ambiguous profile.
    #[must_use]
    fn static_default(&self) -> bool {
        super::speculative_burst::mtp_b1_default(
            None,
            self.target_supports_batching,
            self.has_neural_accelerator,
        )
    }

    /// Whether to run the B=1 MTP burst for the next request.
    ///
    /// - `Forced(b)` → `b`.
    /// - `Profiling` → `true` (force MTP on to collect a sample; this is what
    ///   lets the policy discover a pairing the static gate would decline).
    /// - `Settled(b)` → `b`.
    ///
    /// This is a pure read with no allocation or IO, so once the verdict has
    /// settled there is no per-request (and certainly no per-token) overhead
    /// beyond the match.
    #[must_use]
    pub(crate) fn should_attempt_b1(&self) -> bool {
        match &self.state {
            PolicyState::Forced(run) => *run,
            PolicyState::Profiling(_) => true,
            PolicyState::Settled(run) => *run,
        }
    }

    /// Whether the policy has finished profiling (forced or settled). Once true,
    /// [`Self::record_b1_sample`] is a no-op. Test/diagnostic accessor.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_settled(&self) -> bool {
        !matches!(self.state, PolicyState::Profiling(_))
    }

    /// Record one completed B=1 burst's profile. Folds the sample into the
    /// accumulator while profiling and, once [`PROFILE_SAMPLE_TARGET`]
    /// qualifying samples are in, settles the verdict and persists the coarse
    /// hint. A no-op in the forced/settled states, so this never runs once the
    /// decision is fixed.
    pub(crate) fn record_b1_sample(&mut self, profile: MtpBurstProfile) {
        // Resolve the backend-scaled verify-cost multiple of the FALLBACK
        // estimator before borrowing `self.state` mutably below (issue #638).
        // The primary estimator is the measured-cost one (issue #736); the
        // fallback only applies when the window collected no classic-step
        // probe measurement.
        let verify_cost_multiple = self.verify_cost_multiple();
        let PolicyState::Profiling(acc) = &mut self.state else {
            return;
        };
        acc.add(&profile);
        if acc.samples() < PROFILE_SAMPLE_TARGET {
            return;
        }
        // Read everything off the accumulator first so its borrow of
        // `self.state` ends before we touch other `self` fields or reassign
        // the state below.
        //
        // Estimator preference (issue #736): the measured-cost estimate
        // (tokens per round over the measured round-cost-to-classic-step
        // ratio) makes no assumption about verify amortization, so it is
        // correct across backends and kernel eras. The shape-heuristic
        // fallback only fires when no probe measurement exists.
        let measured = acc.estimated_speedup_measured();
        let speedup = measured.or_else(|| acc.estimated_speedup_scaled(verify_cost_multiple));
        let estimator = if measured.is_some() {
            "measured"
        } else {
            "heuristic-fallback"
        };
        let acceptance_rate = acc.acceptance_rate();
        let samples = acc.samples();
        let accepted_len = acc.accepted_len();
        let drafter_ms = acc.drafter_ms();
        let verify_ms = acc.verify_ms();
        let overhead_ms = acc.overhead_ms();
        let classic_step_ms = acc.classic_step_ms();
        let max_batch_size = acc.max_batch_size();
        let max_prompt_len = acc.max_prompt_len();
        let mean_prompt_len = acc.mean_prompt_len();
        let zone = speedup.map(classify_speedup);
        let static_default = self.static_default();
        let run = resolve_verdict(zone, static_default);
        tracing::info!(
            "adaptive MTP policy: settled verdict {} for {} after {} samples \
             (accepted_len={:.2}, drafter_ms={:.2}, verify_ms={:.2}, overhead_ms={:.2}, \
             classic_step_ms={}, acceptance≈{:.2}, est_speedup={} [{}], zone={:?}, \
             static_default={}, max_batch={}, prompt_len mean={}/max={})",
            if run { "ENABLE" } else { "DECLINE" },
            self.key.display(),
            samples,
            accepted_len,
            drafter_ms,
            verify_ms,
            overhead_ms,
            classic_step_ms
                .map(|c| format!("{c:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
            acceptance_rate,
            speedup
                .map(|s| format!("{s:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
            estimator,
            zone,
            static_default,
            max_batch_size,
            mean_prompt_len,
            max_prompt_len,
        );
        let hint = PolicyHint::new(&self.key, Verdict::from_run(run), acceptance_rate, samples);
        if let Err(e) = self.store.save(&hint) {
            tracing::warn!(
                "adaptive MTP policy: failed to persist verdict for {}: {e}; \
                 keeping it in memory for this session",
                self.key.display(),
            );
        }
        self.state = PolicyState::Settled(run);
    }
}

#[cfg(test)]
#[path = "mtp_policy_tests.rs"]
mod tests;
