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

//! What a production paged decode launch did, and why (issue #899).
//!
//! ## Why this is a value and not a log line
//!
//! The first cut of #899 reported declines with `tracing::debug!` from inside
//! the pool. The production benchmark then measured gather against gather for a
//! full sweep without anything saying so, because a real `mlxcel-server` run
//! emits no `DEBUG` from this crate: an operator has to opt into it, and
//! `RUST_LOG="info,mlxcel_core=debug"` did not produce a single line in the run
//! that was supposed to validate the change. A diagnostic that is invisible in
//! the one situation it exists for is not a diagnostic.
//!
//! So the decision is now a **return value**. The pool reports what it did, the
//! caller decides how loudly to say it, and the reason carries the numbers that
//! produced it rather than a bare "declined". Nothing on the hot path formats a
//! string unless a genuinely new outcome occurs.

/// The outcome of one production paged decode launch, for one layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedDecodeOutcome {
    /// The fused v2 kernel ran.
    Fused {
        /// Requests in the launch.
        batch: usize,
        /// Visible KV tokens the launch read.
        visible_tokens: usize,
        /// Chunks the plan emitted.
        chunks: usize,
        /// Whether the merge pass ran.
        merged: bool,
    },
    /// The layer's rows do not fit one contiguous pool buffer, so neither fused
    /// kernel can address them. This is the decline that silently disables the
    /// whole fused path when the slab is sized too small.
    MultiSlab {
        k_slabs: usize,
        v_slabs: usize,
        slab_blocks: usize,
    },
    /// The kernel cannot serve this head geometry.
    UnservableGeometry(String),
    /// No request in the launch has a visible token.
    NoVisibleTokens,
    /// The launch is smaller than the dispatch policy's floor.
    BelowFloor {
        batch: usize,
        visible_tokens: usize,
        floor: usize,
    },
    /// `MLXCEL_PAGED_ATTENTION_NATIVE` pinned the gather path.
    KillSwitch,
    /// The plan failed its structural check.
    PlanRejected(String),
    /// The page table could not be built.
    ViewFailed(String),
    /// The caller declined before any pool write: not all caches are
    /// pool-backed on one pool and layer, a soft-cap was requested, or the step
    /// carries more than one query token (batched prefill, speculative verify).
    NotServable(&'static str),
}

/// Number of distinct outcome kinds, for the caller's one-shot log table.
pub const PAGED_DECODE_OUTCOME_KINDS: usize = 9;

impl PagedDecodeOutcome {
    /// Whether the fused kernel actually ran.
    #[must_use]
    pub fn is_fused(&self) -> bool {
        matches!(self, Self::Fused { .. })
    }

    /// Stable index for this kind, so a caller can keep a fixed-size table of
    /// "already reported" flags without allocating or hashing.
    #[must_use]
    pub fn kind_index(&self) -> usize {
        match self {
            Self::Fused { .. } => 0,
            Self::MultiSlab { .. } => 1,
            Self::UnservableGeometry(_) => 2,
            Self::NoVisibleTokens => 3,
            Self::BelowFloor { .. } => 4,
            Self::KillSwitch => 5,
            Self::PlanRejected(_) => 6,
            Self::ViewFailed(_) => 7,
            Self::NotServable(_) => 8,
        }
    }

    /// One-line human summary carrying the numbers that produced the decision.
    ///
    /// Allocates, so callers format it only when they are about to emit it.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Fused {
                batch,
                visible_tokens,
                chunks,
                merged,
            } => format!(
                "fused v2 launch (batch {batch}, {visible_tokens} visible KV tokens, \
                 {chunks} chunks, merge {})",
                if *merged { "on" } else { "skipped" }
            ),
            Self::MultiSlab {
                k_slabs,
                v_slabs,
                slab_blocks,
            } => format!(
                "gather: the layer spans {k_slabs} K / {v_slabs} V pool slabs and the fused \
                 kernels read one buffer per side (slab_blocks={slab_blocks}); raise --ctx-size \
                 or MLXCEL_PAGED_SLAB_BLOCKS so a layer's rows fit one slab"
            ),
            Self::UnservableGeometry(reason) => {
                format!("gather: the kernel cannot serve this geometry ({reason})")
            }
            Self::NoVisibleTokens => "gather: no request has a visible token".to_string(),
            Self::BelowFloor {
                batch,
                visible_tokens,
                floor,
            } => format!(
                "gather: {visible_tokens} visible KV tokens across {batch} request(s) is below \
                 the {floor}-token dispatch floor"
            ),
            Self::KillSwitch => "gather: pinned by MLXCEL_PAGED_ATTENTION_NATIVE".to_string(),
            Self::PlanRejected(reason) => format!("gather: the chunk plan was rejected ({reason})"),
            Self::ViewFailed(reason) => {
                format!("gather: the page table failed to build ({reason})")
            }
            Self::NotServable(reason) => format!("gather: batch not servable ({reason})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_indices_are_distinct_and_in_range() {
        let all = [
            PagedDecodeOutcome::Fused {
                batch: 1,
                visible_tokens: 1,
                chunks: 1,
                merged: false,
            },
            PagedDecodeOutcome::MultiSlab {
                k_slabs: 2,
                v_slabs: 2,
                slab_blocks: 32,
            },
            PagedDecodeOutcome::UnservableGeometry("x".to_string()),
            PagedDecodeOutcome::NoVisibleTokens,
            PagedDecodeOutcome::BelowFloor {
                batch: 1,
                visible_tokens: 1,
                floor: 2,
            },
            PagedDecodeOutcome::KillSwitch,
            PagedDecodeOutcome::PlanRejected("x".to_string()),
            PagedDecodeOutcome::ViewFailed("x".to_string()),
            PagedDecodeOutcome::NotServable("x"),
        ];
        assert_eq!(all.len(), PAGED_DECODE_OUTCOME_KINDS);
        let mut seen = [false; PAGED_DECODE_OUTCOME_KINDS];
        for outcome in &all {
            let i = outcome.kind_index();
            assert!(i < PAGED_DECODE_OUTCOME_KINDS, "{outcome:?} index {i}");
            assert!(!seen[i], "duplicate index for {outcome:?}");
            seen[i] = true;
            assert!(!outcome.describe().is_empty());
        }
    }

    #[test]
    fn only_the_fused_variant_reports_a_fused_launch() {
        assert!(
            PagedDecodeOutcome::Fused {
                batch: 4,
                visible_tokens: 4096,
                chunks: 16,
                merged: true,
            }
            .is_fused()
        );
        assert!(!PagedDecodeOutcome::NoVisibleTokens.is_fused());
        assert!(!PagedDecodeOutcome::KillSwitch.is_fused());
    }

    #[test]
    fn the_multi_slab_message_names_the_knob_that_fixes_it() {
        let text = PagedDecodeOutcome::MultiSlab {
            k_slabs: 3,
            v_slabs: 3,
            slab_blocks: 32,
        }
        .describe();
        assert!(text.contains("MLXCEL_PAGED_SLAB_BLOCKS"), "{text}");
        assert!(text.contains("--ctx-size"), "{text}");
    }

    #[test]
    fn the_floor_message_carries_both_numbers() {
        let text = PagedDecodeOutcome::BelowFloor {
            batch: 4,
            visible_tokens: 3824,
            floor: 4096,
        }
        .describe();
        assert!(text.contains("3824"), "{text}");
        assert!(text.contains("4096"), "{text}");
    }
}
