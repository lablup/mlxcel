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
//! ## Why the floor is two-regime, and not one number
//!
//! The first cut of this policy summed the launch's visible tokens and required
//! 4096 of them, on the reasoning that `batch * ctx` is what the kernel reads
//! and therefore what the fixed merge and workspace cost has to amortize
//! against. That fits the table (the loss has 1024 total tokens, every win has
//! 4096 or more) but it is the wrong *shape*, and the server benchmark for #899
//! showed it plainly: **the loss is a property of batch 1, not of total
//! tokens.**
//!
//! Read the ctx-1024 column: 0.91x at batch 1, 1.41x at batch 4, 1.47x at batch
//! 8. Same per-request context, same plan degeneracy, opposite outcome. What
//! changes is that at batch 4 the same chunk count is spread over four
//! requests, so the plan picks a larger chunk size, emits fewer chunks per
//! request, and amortizes the merge over four times the useful work. A
//! total-token floor separates those two cells only by accident, and with
//! almost no margin: a batched launch at 1024 tokens per request sits exactly
//! on 4096. The production benchmark's nominal "1K" scenario delivered 956
//! tokens per request, `4 * 956 = 3824`, and was declined despite being the
//! same shape as the measured 1.41x win.
//!
//! So the floor is now stated the way the measurements are:
//!
//! | launch | floor | evidence |
//! |---|---|---|
//! | one request | [`MIN_SINGLE_REQUEST_KV_TOKENS`] visible tokens | 1024 loses (0.91x), 4096 wins (1.08x) |
//! | more than one | [`MIN_BATCHED_KV_TOKENS_PER_REQUEST`] per request | 1024 per request wins at batch 4 (1.41x) and batch 8 (1.47x) |
//!
//! The batched floor sits at 512 rather than 1024 deliberately. 1024 is the
//! *lowest measured* batched context and it wins comfortably; putting a floor
//! on top of a measured point means a real workload landing just under it (a
//! tokenizer delivering 956 tokens for a nominal 1K prompt, exactly what
//! happened) is declined for no measured reason. 512 keeps margin below the
//! evidence while still refusing launches small enough that the gather path is
//! trivially cheap. Batch 2 and 3 are interpolated, not measured; the trend
//! across batch 1, 4, and 8 at fixed context is monotone and the mechanism
//! above explains why, so the interpolation is stated rather than hidden.
//!
//! ## Overrides
//!
//! Both floors have an environment override for re-measurement on new hardware.
//! To force the fused path for every servable shape, including the regime this
//! policy declines, use `MLXCEL_PAGED_ATTENTION_NATIVE=1`, which bypasses the
//! selector entirely; that is the supported way to benchmark the declined
//! corner, rather than driving a floor to zero.

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

/// Visible tokens a **single-request** launch needs before v2 is dispatched.
///
/// From the table in this module's docs: batch 1 at 1024 tokens is the only
/// measured loss (0.91x), and batch 1 at 4096 is the weakest measured win
/// (1.08x median, 1.02x to 1.21x across repetitions, so a win in every
/// repetition).
pub const MIN_SINGLE_REQUEST_KV_TOKENS: usize = 4096;

/// Visible tokens **per request** a multi-request launch needs.
///
/// Every measured multi-request cell wins, the smallest being batch 4 at 1024
/// tokens per request (1.41x). This sits below that with margin rather than on
/// top of it; see the module docs for why the margin matters.
pub const MIN_BATCHED_KV_TOKENS_PER_REQUEST: usize = 512;

/// Environment override for [`MIN_SINGLE_REQUEST_KV_TOKENS`].
pub const MIN_KV_TOKENS_ENV: &str = "MLXCEL_PAGED_V2_MIN_KV_TOKENS";

/// Environment override for [`MIN_BATCHED_KV_TOKENS_PER_REQUEST`].
pub const MIN_KV_TOKENS_PER_REQUEST_ENV: &str = "MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST";

/// Pure parse of a floor override.
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

/// The active single-request floor, read once per process.
#[must_use]
pub fn min_total_kv_tokens() -> usize {
    static FLOOR: OnceLock<usize> = OnceLock::new();
    *FLOOR.get_or_init(|| {
        let raw = std::env::var(MIN_KV_TOKENS_ENV).ok();
        parse_min_total_kv_tokens(raw.as_deref(), MIN_SINGLE_REQUEST_KV_TOKENS)
    })
}

/// The active per-request batched floor, read once per process.
#[must_use]
pub fn min_kv_tokens_per_request() -> usize {
    static FLOOR: OnceLock<usize> = OnceLock::new();
    *FLOOR.get_or_init(|| {
        let raw = std::env::var(MIN_KV_TOKENS_PER_REQUEST_ENV).ok();
        parse_min_total_kv_tokens(raw.as_deref(), MIN_BATCHED_KV_TOKENS_PER_REQUEST)
    })
}

/// Visible tokens this launch must reach, given how many requests it serves.
///
/// Pure, so the number that decided a dispatch can be logged next to the number
/// that was measured against it.
#[must_use]
pub fn required_visible_tokens(batch: usize, single: usize, per_request: usize) -> usize {
    if batch <= 1 {
        single
    } else {
        batch.saturating_mul(per_request)
    }
}

/// The floor this launch faces under the process's active configuration.
#[must_use]
pub fn active_required_visible_tokens(batch: usize) -> usize {
    required_visible_tokens(batch, min_total_kv_tokens(), min_kv_tokens_per_request())
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

    /// Apply the shipped policy to a `(batch, per-request context)` cell.
    fn decide(batch: usize, ctx: usize) -> PagedV2Dispatch {
        let floor = required_visible_tokens(
            batch,
            MIN_SINGLE_REQUEST_KV_TOKENS,
            MIN_BATCHED_KV_TOKENS_PER_REQUEST,
        );
        select_paged_v2_dispatch(batch * ctx, floor)
    }

    #[test]
    fn the_only_measured_loss_is_declined() {
        assert_eq!(decide(1, 1024), PagedV2Dispatch::Gather);
    }

    #[test]
    fn every_measured_win_is_dispatched() {
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
                decide(batch, ctx),
                PagedV2Dispatch::V2,
                "batch {batch} / ctx {ctx} should dispatch v2"
            );
        }
    }

    #[test]
    fn a_batched_launch_just_under_a_nominal_1k_prompt_still_dispatches() {
        // The regression that motivated the two-regime floor: the production
        // benchmark's nominal 1K scenario delivered 956 tokens per request, and
        // a flat 4096-total floor declined it even though batch 4 at ~1K is a
        // measured 1.41x win.
        assert_eq!(decide(4, 956), PagedV2Dispatch::V2);
        assert_eq!(decide(2, 956), PagedV2Dispatch::V2);
    }

    #[test]
    fn a_single_request_is_not_helped_by_the_batched_floor() {
        // The batched floor must never leak into batch 1, which is the only
        // regime with a measured loss.
        assert_eq!(decide(1, 512), PagedV2Dispatch::Gather);
        assert_eq!(decide(1, 2048), PagedV2Dispatch::Gather);
        assert_eq!(decide(1, 4095), PagedV2Dispatch::Gather);
        assert_eq!(decide(1, 4096), PagedV2Dispatch::V2);
    }

    #[test]
    fn a_tiny_batched_launch_is_still_declined() {
        // Not measured, and the gather path is trivially cheap at this size.
        assert_eq!(decide(4, 128), PagedV2Dispatch::Gather);
        assert_eq!(decide(2, 256), PagedV2Dispatch::Gather);
    }

    #[test]
    fn an_empty_batch_stays_on_gather() {
        assert_eq!(decide(0, 0), PagedV2Dispatch::Gather);
        assert_eq!(decide(4, 0), PagedV2Dispatch::Gather);
    }

    #[test]
    fn the_floor_scales_with_the_request_count() {
        assert_eq!(required_visible_tokens(1, 4096, 512), 4096);
        assert_eq!(required_visible_tokens(2, 4096, 512), 1024);
        assert_eq!(required_visible_tokens(4, 4096, 512), 2048);
        assert_eq!(required_visible_tokens(8, 4096, 512), 4096);
        // Saturating, so an absurd request count cannot wrap the floor down.
        assert_eq!(required_visible_tokens(usize::MAX, 4096, 512), usize::MAX);
    }

    #[test]
    fn the_env_overrides_parse_only_non_negative_integers() {
        assert_eq!(parse_min_total_kv_tokens(None, 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some(""), 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some("  "), 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some("nope"), 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some("-1"), 4096), 4096);
        assert_eq!(parse_min_total_kv_tokens(Some("0"), 4096), 0);
        assert_eq!(parse_min_total_kv_tokens(Some(" 8192 "), 4096), 8192);
    }
}
