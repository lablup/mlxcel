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

//! Paged-attention decode v2: CSR page table, cross-CTA split-KV, and a
//! variable-length merge (issue #898).
//!
//! ## What v1 cannot do
//!
//! The v1 fused kernel (`src/lib/mlx-cpp/turbo/paged_attention.cpp`, entered
//! through [`crate::cache::PagedBlockPool::paged_decode_fused`]) splits the KV
//! range inside a single threadgroup. One CTA serves one `(batch, query head)`
//! pair no matter how long the context is, so a small batch with a long context
//! leaves most of the GPU idle and adding context adds no parallelism. ADR 0001
//! measures what the gather-then-SDPA fallback costs instead: ~56% over the
//! contiguous-SDPA lower bound at 16384 tokens and ~67% at 32768.
//!
//! ## What v2 does
//!
//! Four composable pieces, following the flash-decoding lineage:
//!
//! 1. **CSR page table** ([`crate::cache::paged_csr`]): the whole batch's block
//!    tables flattened into `indices` / `indptr` / `last_page_len` (plus the
//!    mlxcel `first_page_offset` extension for sliding-window sequences). No
//!    gather pre-pass, one launch for the batch.
//! 2. **Cross-CTA split-KV** ([`plan`], kernel in `paged_attention_v2.cpp`):
//!    each request's page range is cut into chunks assigned to different CTAs,
//!    each producing an online-softmax partial. One CTA covers all query heads
//!    of one KV head group, so KV is read once per group.
//! 3. **Variable-length merge** (`paged_attention_v2_merge.cpp`): partial
//!    states merge through the closed-form `exp2`/`log2` rescale under an
//!    `o_indptr` grouping. Deliberately paging-agnostic so issue #903's cascade
//!    decomposition reuses it unchanged.
//! 4. **Host-side plan** ([`PagedDecodePlan`]): binary-searches the chunk size
//!    for an occupancy-derived CTA target and emits the flat index arrays as
//!    plain data that can be cached across decode steps.
//!
//! ## Default off
//!
//! This issue lands v2 as a library capability. `MLXCEL_PAGED_ATTENTION_V2=1`
//! selects it inside
//! [`crate::cache::PagedBlockPool::paged_decode_fused`]; with the variable
//! unset the entry point is one `OnceLock` read away from the exact v1 code it
//! ran before, no v2 array is built, and no v2 kernel is JIT-compiled. v1 is
//! left fully intact for comparison. Production wiring (scheduler, server
//! defaults, dispatch thresholds) is issue #899.
//!
//! ## Not covered
//!
//! Sliding-window and softcap variants keep their documented fallbacks, and
//! speculative multi-token verify stays on its current path: both are out of
//! scope per the issue. `first_page_offset` means the *page table* copes with a
//! trimmed window, but v2 applies no windowing mask of its own.

use std::sync::OnceLock;

pub mod launch;
pub mod plan;

pub use launch::{V2Context, geometry_from_shapes, resolve_plan, run_decode_v2};
pub use plan::{
    MAX_CHUNKS, PagedDecodeGeometry, PagedDecodePlan, TARGET_CTAS_ENV, chunks_for_batch,
    chunks_for_request, device_target_ctas, max_pages_per_chunk, min_pages_per_chunk,
    search_pages_per_chunk,
};

/// Environment variable selecting the v2 decode path. Default off.
pub const V2_ENV: &str = "MLXCEL_PAGED_ATTENTION_V2";

/// Pure parse of a [`V2_ENV`] value.
///
/// Accepts the tree's usual on spellings. Anything else, including an unset
/// variable, is off; a typo therefore degrades to v1 rather than silently
/// enabling a path the operator did not ask for.
#[must_use]
pub fn parse_v2_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

/// Whether the v2 decode path is selected, read once per process.
///
/// One `OnceLock` load on the decode hot path when off, which is what keeps the
/// default path byte-identical to pre-#898 behavior.
#[must_use]
pub fn v2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_v2_enabled(std::env::var(V2_ENV).ok().as_deref()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_is_off_unless_explicitly_enabled() {
        assert!(!parse_v2_enabled(None));
        assert!(!parse_v2_enabled(Some("")));
        assert!(!parse_v2_enabled(Some("0")));
        assert!(!parse_v2_enabled(Some("off")));
        assert!(!parse_v2_enabled(Some("2")));
        assert!(!parse_v2_enabled(Some("maybe")));
    }

    #[test]
    fn v2_accepts_the_usual_on_spellings() {
        for v in ["1", "true", "TRUE", "on", "ON", "yes", " yes "] {
            assert!(parse_v2_enabled(Some(v)), "{v:?} should enable v2");
        }
    }
}
