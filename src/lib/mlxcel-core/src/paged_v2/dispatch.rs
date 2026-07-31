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

//! Production dispatch policy for paged decode v2 (issue #899).
//!
//! Issue #898 landed v2 as a library capability and measured it against the
//! production gather-then-SDPA path on an Apple M1 Ultra
//! (`docs/benchmark_results/paged-decode-v2-m1ultra-2026-07-31.md`, medians of
//! three repetitions, head_dim 128 / 32 q heads / 8 kv heads / block 32):
//!
//! | batch | ctx 1024 | ctx 4096 | ctx 16384 | ctx 32768 |
//! |---|---|---|---|---|
//! | 1 | **0.91x** | 1.08x | 1.29x | 1.47x |
//! | 4 | 1.41x | 2.04x | 2.78x | 3.08x |
//! | 8 | 1.47x | 2.00x | 2.27x | 2.32x |
//!
//! Exactly one cell loses: batch 1 at 1024 tokens, 0.91x median and below
//! parity in all three repetitions. The plan degenerates to two pages per chunk
//! there, and the merge pass plus the workspace round trip costs more than the
//! whole attention does at that size. So v2 must not be dispatched
//! unconditionally.
//!
//! ## The threshold
//!
//! Re-index that table by **total visible KV tokens in the launch**
//! (`batch * ctx`, which is what the kernel actually reads and therefore what
//! the fixed merge + workspace cost has to amortize against):
//!
//! | total tokens | cells | v2 / gather |
//! |---|---|---|
//! | 1024 | b1 / 1K | **0.91x** |
//! | 4096 | b1 / 4K, b4 / 1K | 1.08x, 1.41x |
//! | 8192 | b8 / 1K | 1.47x |
//! | 16384+ | everything else measured | 1.29x to 3.08x |
//!
//! One number separates the measured loss from every measured win:
//! [`MIN_TOTAL_KV_TOKENS`] = 4096. The only loss sits strictly below it, and
//! both cells sitting exactly on it win (the weaker of the two, batch 1 at
//! 4096, returned 1.02x to 1.21x across repetitions, so it is a win in every
//! repetition). Single-sequence decode therefore crosses over at 4096 tokens
//! and batch 4 crosses over at 1024 tokens each, which is the narrow gate the
//! measurements support.
//!
//! Total tokens is also the cheapest possible gate: [`PagedCsrView::seq_lens`]
//! already carries it, so the decision is made before the plan is built rather
//! than after.
//!
//! [`PagedCsrView::seq_lens`]: crate::cache::PagedCsrView::seq_lens

use std::sync::OnceLock;

/// Where a pooled paged decode batch runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedV2Dispatch {
    /// The fused v2 kernel: CSR page table, cross-CTA split-KV, merge.
    V2,
    /// Per-sequence `gather_visible` + dense SDPA, the pre-#899 production
    /// path.
    Gather,
}

/// Minimum total visible KV tokens in a launch for v2 to be dispatched.
///
/// Derived directly from the #898 measurement table in this module's docs: the
/// single measured loss (batch 1 at 1024 tokens) has 1024 total tokens, and the
/// weakest measured win (batch 1 at 4096 tokens, 1.08x) has 4096. Change this
/// constant to move the crossover; [`MIN_KV_TOKENS_ENV`] moves it without a
/// rebuild.
pub const MIN_TOTAL_KV_TOKENS: usize = 4096;

/// Environment override for [`MIN_TOTAL_KV_TOKENS`].
///
/// A non-negative integer. `0` dispatches v2 for every servable shape (useful
/// for re-measuring the crossover on new hardware); a very large value keeps
/// every batch on gather without disabling the code path the way
/// `MLXCEL_PAGED_ATTENTION_NATIVE=0` does.
pub const MIN_KV_TOKENS_ENV: &str = "MLXCEL_PAGED_V2_MIN_KV_TOKENS";

/// Pure parse of a [`MIN_KV_TOKENS_ENV`] value.
///
/// Anything that is not a non-negative integer falls back to the default, so a
/// typo degrades to the measured policy rather than to an unmeasured one.
#[must_use]
pub fn parse_min_total_kv_tokens(value: Option<&str>, default: usize) -> usize {
    match value.map(str::trim) {
        Some(raw) if !raw.is_empty() => raw.parse::<usize>().unwrap_or(default),
        _ => default,
    }
}

/// The active token floor, read once per process so the decode hot path never
/// touches the environment.
#[must_use]
pub fn min_total_kv_tokens() -> usize {
    static FLOOR: OnceLock<usize> = OnceLock::new();
    *FLOOR.get_or_init(|| {
        let raw = std::env::var(MIN_KV_TOKENS_ENV).ok();
        let resolved = parse_min_total_kv_tokens(raw.as_deref(), MIN_TOTAL_KV_TOKENS);
        if resolved != MIN_TOTAL_KV_TOKENS {
            tracing::info!(
                "paged decode v2 token floor overridden to {resolved} (default {MIN_TOTAL_KV_TOKENS})"
            );
        }
        resolved
    })
}

/// Pure regime selector: v2 once the launch reads at least `floor` visible KV
/// tokens in total, gather below that.
///
/// Allocation-free and MLX-free, so it is trivially unit-testable and cheap
/// enough to evaluate per layer without memoization.
#[must_use]
pub fn select_paged_v2_dispatch(total_visible_tokens: usize, floor: usize) -> PagedV2Dispatch {
    if total_visible_tokens >= floor {
        PagedV2Dispatch::V2
    } else {
        PagedV2Dispatch::Gather
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_only_measured_loss_is_below_the_floor() {
        // batch 1 / ctx 1024, the 0.91x cell.
        assert_eq!(
            select_paged_v2_dispatch(1024, MIN_TOTAL_KV_TOKENS),
            PagedV2Dispatch::Gather
        );
    }

    #[test]
    fn every_measured_win_is_at_or_above_the_floor() {
        // (batch, ctx) cells from the #898 table that beat gather.
        for (batch, ctx) in [
            (1, 4096),
            (1, 16384),
            (1, 32768),
            (4, 1024),
            (4, 4096),
            (4, 16384),
            (4, 32768),
            (8, 1024),
            (8, 4096),
            (8, 16384),
            (8, 32768),
        ] {
            assert_eq!(
                select_paged_v2_dispatch(batch * ctx, MIN_TOTAL_KV_TOKENS),
                PagedV2Dispatch::V2,
                "batch {batch} / ctx {ctx} should dispatch v2"
            );
        }
    }

    #[test]
    fn an_empty_batch_stays_on_gather() {
        assert_eq!(
            select_paged_v2_dispatch(0, MIN_TOTAL_KV_TOKENS),
            PagedV2Dispatch::Gather
        );
    }

    #[test]
    fn a_zero_floor_dispatches_everything() {
        assert_eq!(select_paged_v2_dispatch(1, 0), PagedV2Dispatch::V2);
        assert_eq!(select_paged_v2_dispatch(0, 0), PagedV2Dispatch::V2);
    }

    #[test]
    fn the_env_override_parses_only_non_negative_integers() {
        assert_eq!(parse_min_total_kv_tokens(None, 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some(""), 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some("  "), 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some("nope"), 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some("-1"), 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some("0"), 4096), 0);
        assert_eq!(parse_min_total_kv_tokens(Some(" 8192 "), 4096), 8192);
    }
}
