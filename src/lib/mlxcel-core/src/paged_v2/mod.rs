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
//! ## Two entry points, two gates
//!
//! **The v1 entry point** ([`crate::cache::PagedBlockPool::paged_decode_fused`],
//! reached only from the library-only [`crate::layers::paged_decode_attention_pooled`])
//! tries v2 first under `MLXCEL_PAGED_ATTENTION_V2=1`, default off. That is
//! issue #898's comparison gate and is unchanged.
//!
//! **The production entry point**
//! ([`crate::cache::PagedBlockPool::paged_decode_batched`], reached from the
//! server's pool-backed batched decode) is issue #899 and is **default on**. It
//! is governed by [`dispatch`] (a measured token floor below which the gather
//! path is kept) and by `MLXCEL_PAGED_ATTENTION_NATIVE`, whose force-off values
//! are its kill switch.
//!
//! ## Not covered
//!
//! Logit-softcap families and speculative / MTP multi-token verify steps stay
//! on their existing paths; see [`crate::cache::paged_batch_decode`] for where
//! each is declined. `first_page_offset` means the *page table* copes with a
//! trimmed sliding window, but v2 applies no windowing mask of its own, so the
//! visible range has to be expressed entirely in the CSR range.

use std::sync::OnceLock;

pub mod dispatch;
pub mod launch;
pub mod plan;
pub mod plan_cache;

pub use dispatch::{
    MIN_KV_TOKENS_ENV, MIN_TOTAL_KV_TOKENS, PagedV2Dispatch, min_total_kv_tokens,
    parse_min_total_kv_tokens, select_paged_v2_dispatch,
};
pub use launch::{V2Context, geometry_from_shapes, resolve_plan, run_decode_v2};
pub use plan::{
    MAX_CHUNKS, PagedDecodeGeometry, PagedDecodePlan, TARGET_CTAS_ENV, chunks_for_batch,
    chunks_for_request, device_target_ctas, max_pages_per_chunk, min_pages_per_chunk,
    search_pages_per_chunk,
};
pub use plan_cache::{PagedDecodeV2Cache, PlanCacheStats, RequestFingerprint};

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
