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

//! N-gram repetition / loop detection over the raw generated token stream.
//!
//! Some checkpoints (e.g. the Gemma 4 family under tool or grammar-constrained
//! decoding) can collapse into a degenerate loop where a single token or a
//! short block repeats until the token budget is exhausted. Sampling penalties
//! (`repetition_penalty`, DRY) reshape the probability distribution, but once
//! the logits collapse the top-k candidates are themselves garbage, so
//! distribution shaping cannot recover. The countermeasure here is the one vLLM
//! exposes in `SamplingParams`: detect when a short pattern repeats at the tail
//! of the stream and end generation early.
//!
//! This module is intentionally model-family agnostic. The policy for when to
//! turn it on (per-request override, global operator override, family-specific
//! auto-enable) lives in the server control plane, not here. The default is a
//! zero-overhead no-op that preserves the bit-exact baseline for every model
//! that does not opt in.

/// Hard upper bound on the effective `max_pattern_size` actually scanned by
/// [`detect_repetition_loop`], regardless of the configured value.
///
/// The per-decode-step scan cost grows with the largest pattern size it sweeps:
/// for each `p` up to the maximum it checks the last `p * min_count` tokens, so
/// an unbounded maximum makes the per-step work grow with the generated length
/// (and the whole-generation cost grow super-linearly), letting an untrusted
/// per-request `max_pattern_size` or the global-env override pin the worker
/// thread. Both override surfaces are untrusted, so the value is capped here
/// rather than at each entry point. The cap still covers every loop length that
/// occurs in practice: real Gemma 4 collapses are 1-20 tokens, and vLLM-style
/// detection targets short patterns, so 64 leaves a wide margin while keeping
/// per-step work bounded.
pub const MAX_EFFECTIVE_PATTERN_SIZE: usize = 64;

/// Configuration for tail N-gram repetition detection.
///
/// Field names and semantics mirror vLLM's `SamplingParams` so the same JSON
/// request fields work against either engine:
///
/// - `max_pattern_size` (default `0` = disabled): largest N-gram size to scan.
/// - `min_pattern_size` (default `0`, treated as `1`): smallest N-gram size to
///   scan; clamped to `1..=max_pattern_size`.
/// - `min_count` (must be `>= 2`): how many consecutive repeats of a pattern at
///   the tail trigger an early stop.
///
/// The all-zero `Default` is disabled, matching vLLM's opt-in default and the
/// way [`crate::generate::SamplingConfig::token_bias`] /
/// [`crate::generate::SamplingConfig::stop_token_ids`] keep an empty no-op
/// baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoopDetectionConfig {
    /// Largest N-gram pattern size to scan. `0` disables detection entirely.
    pub max_pattern_size: usize,
    /// Smallest N-gram pattern size to scan. `0` is treated as `1`.
    pub min_pattern_size: usize,
    /// Minimum consecutive repeats of a pattern that triggers an early stop.
    /// Must be `>= 2`; any smaller value disables detection.
    pub min_count: usize,
}

impl LoopDetectionConfig {
    /// The disabled (zero-overhead, bit-exact baseline) configuration.
    pub const fn disabled() -> Self {
        Self {
            max_pattern_size: 0,
            min_pattern_size: 0,
            min_count: 0,
        }
    }

    /// Build a configuration from raw vLLM-style fields without normalization.
    /// `is_enabled` and [`detect_repetition_loop`] apply the clamping rules.
    pub const fn new(min_pattern_size: usize, max_pattern_size: usize, min_count: usize) -> Self {
        Self {
            max_pattern_size,
            min_pattern_size,
            min_count,
        }
    }

    /// Returns `true` when the configuration can ever trigger.
    ///
    /// Detection requires a non-zero `max_pattern_size` and a `min_count` of at
    /// least two repeats (a single occurrence is not a loop). When this returns
    /// `false`, [`detect_repetition_loop`] short-circuits before touching the
    /// token slice, preserving the zero-overhead baseline.
    pub const fn is_enabled(&self) -> bool {
        self.max_pattern_size > 0 && self.min_count >= 2
    }

    /// Smallest pattern size actually scanned: `0` becomes `1`, then clamped to
    /// `1..=effective_max_pattern_size()`. The ceiling is the capped maximum (not
    /// the raw `max_pattern_size`) so a huge configured maximum paired with a min
    /// above the cap still yields a well-formed scan range. Only meaningful when
    /// [`is_enabled`] is true, so `max_pattern_size >= 1` holds and the clamp
    /// range is non-empty.
    ///
    /// [`is_enabled`]: Self::is_enabled
    fn effective_min_pattern_size(&self) -> usize {
        let requested = if self.min_pattern_size == 0 {
            1
        } else {
            self.min_pattern_size
        };
        requested.clamp(1, self.effective_max_pattern_size())
    }

    /// Largest pattern size actually scanned: `max_pattern_size` capped at
    /// [`MAX_EFFECTIVE_PATTERN_SIZE`]. This bounds the per-decode-step cost no
    /// matter what a per-request or global override supplies.
    fn effective_max_pattern_size(&self) -> usize {
        self.max_pattern_size.min(MAX_EFFECTIVE_PATTERN_SIZE)
    }
}

/// Returns `true` when the tail of `generated` is a short block repeated at
/// least `cfg.min_count` times consecutively.
///
/// For each pattern size `p` in `min_pattern_size..=max_pattern_size`, the last
/// `p * min_count` tokens are checked: they trigger when they equal a single
/// `p`-length block `B` repeated `min_count` times (`tail == B·B·…·B`). The
/// first `p` that matches wins. This catches single-token collapses (`p = 1`,
/// e.g. `様様様様`) and short multi-token loops (`p = 2,3,…`, e.g.
/// `abcdabcd…`).
///
/// The scan only runs when [`LoopDetectionConfig::is_enabled`] is true, so a
/// model that does not opt in pays nothing beyond the cheap config check. The
/// caller passes the raw generated token stream so loops inside reasoning /
/// tool-call spans are caught, not just the final answer.
///
/// Used by: `CxxGenerator` decode loops (`generate.rs`), `BatchScheduler`
/// decode sites (`server/batch/scheduler.rs`).
pub fn detect_repetition_loop(generated: &[i32], cfg: &LoopDetectionConfig) -> bool {
    if !cfg.is_enabled() {
        return false;
    }

    let len = generated.len();
    let count = cfg.min_count;
    let min_p = cfg.effective_min_pattern_size();
    let max_p = cfg.effective_max_pattern_size();

    for p in min_p..=max_p {
        // `window` grows monotonically with `p`, so once it exceeds the stream
        // length no larger `p` can fit either: stop scanning.
        let window = match p.checked_mul(count) {
            Some(w) => w,
            None => break,
        };
        if len < window {
            break;
        }

        let tail = &generated[len - window..];
        let block = &tail[..p];
        if tail.chunks_exact(p).all(|chunk| chunk == block) {
            return true;
        }
    }

    false
}

#[cfg(test)]
#[path = "loop_detection_tests.rs"]
mod tests;
