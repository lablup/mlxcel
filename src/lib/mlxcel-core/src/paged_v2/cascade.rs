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

//! Host-side plan for two-level cascade decode (issue #903).
//!
//! ## The duplication this removes
//!
//! mlxcel already deduplicates the *storage* of a shared prompt prefix: the
//! paged pool refcounts blocks and `CachePool::clone_detached_paged_prefix`
//! hands the same [`PagedBlockId`](crate::cache::PagedBlockId)s to every
//! sequence that hits the same APC entry. The *compute* is still duplicated,
//! though. In a batched decode every sequence reads the shared span
//! independently, so attention bandwidth over that span scales with the batch
//! even though every byte read is identical.
//!
//! Cascade attention splits the step into two levels and merges them with the
//! online-softmax algebra:
//!
//! * **Level 0**, the shared span, attended **once** for the whole subgroup.
//! * **Level 1**, each request's private suffix, attended per request.
//! * A merge of the two `(V, LSE)` states per request, which is exact because
//!   the two key ranges are disjoint and softmax states compose.
//!
//! ## Detection reads the page table, not the refcounts
//!
//! [`detect_shared_prefix`] compares the CSR view's `indices` positionally
//! across requests. That is not an approximation of the refcount check, it *is*
//! the refcount check, observed at the only place that matters: within one
//! [`PagedCsrView`] every live block id resolves to exactly one physical pool
//! row, and two live block ids never share a row. So two requests whose page
//! `i` is the same row hold the same block, which means `refcount > 1` and
//! means the bytes are identical. Reading it off the view keeps the whole
//! decision inside `mlxcel-core`, needs no scheduler plumbing, and is exact for
//! sharing that arose any way at all (APC adoption, a forked sequence, a future
//! block-hash dedup) rather than only for sharing the server knows it created.
//!
//! ## What the plan refuses, and why each refusal is load-bearing
//!
//! | condition | reason |
//! |---|---|
//! | `first_page_offset[r] != 0` | a sliding window has trimmed into the middle of the request's first page, so the emitted page list no longer starts on a block boundary and the level-0 span cannot be stated as whole pages |
//! | fewer than 2 private pages behind the shared span | the shared span must consist only of **full** pages. Every page of a request except its last is full, so capping the span at `pages - 1` is what makes `last_page_len = page_size` true for level 0 |
//! | shared span below [`DEFAULT_MIN_SHARED_PAGES`] | the two extra launches and the merge cost more than the duplicated reads they save |
//! | fewer than [`DEFAULT_MIN_MEMBERS`] members | with one member there is no duplication to remove |
//!
//! Sequences outside the chosen subgroup are not excluded from the step: they
//! keep their whole page range at level 1 and their merge group is a single
//! row, which the merge kernel resolves to the identity (`l = 1`, `out = v`).
//! That is what lets one launch serve a mixed batch.
//!
//! ## Thresholds are unmeasured
//!
//! The defaults below are derived from what the decomposition costs, not from a
//! benchmark: two extra launches plus a merge against the bytes saved. The
//! feature is therefore **default off** ([`DEFAULT_CASCADE_ENABLED`]) until
//! `docs/benchmark_results/` carries a number for it, and the whole gate is one
//! constant plus two environment overrides so re-measuring does not need a code
//! change.

use std::sync::OnceLock;

use crate::cache::paged_csr::PagedCsrView;

/// Kill switch and enable flag for cascade decode.
pub const CASCADE_ENV: &str = "MLXCEL_CASCADE_ATTENTION";

/// Whether cascade decode runs when the environment says nothing.
///
/// **Off.** The issue specifies `MLXCEL_CASCADE_ATTENTION=0` as the kill
/// switch, which reads as "default on"; that spelling still disables the
/// feature here, so the documented kill switch behaves as documented under
/// either default. Shipping default-on would mean routing production decode
/// through a path whose benefit has not been measured on this hardware, which
/// is exactly the mistake epic #909 has already paid for twice. Flip this one
/// constant once a shared-prefix benchmark exists.
pub const DEFAULT_CASCADE_ENABLED: bool = false;

/// Environment override for [`DEFAULT_MIN_SHARED_PAGES`].
pub const MIN_SHARED_PAGES_ENV: &str = "MLXCEL_CASCADE_MIN_SHARED_PAGES";

/// Environment override for [`DEFAULT_MIN_MEMBERS`].
pub const MIN_MEMBERS_ENV: &str = "MLXCEL_CASCADE_MIN_MEMBERS";

/// Shared pages a subgroup needs before the decomposition is worth its two
/// extra launches. 16 pages is 512 tokens at the default block size of 32.
pub const DEFAULT_MIN_SHARED_PAGES: usize = 16;

/// Requests that must share the span. Two is the smallest count at which any
/// duplication exists at all.
pub const DEFAULT_MIN_MEMBERS: usize = 2;

/// Pure parse of [`CASCADE_ENV`].
///
/// Explicit on and off spellings both win; anything else, including an unset or
/// misspelled value, takes [`DEFAULT_CASCADE_ENABLED`], so a typo degrades to
/// the shipped default rather than silently flipping the path.
#[must_use]
pub fn parse_cascade_enabled(value: Option<&str>) -> bool {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "on" | "yes") => true,
        Some("0" | "false" | "off" | "no") => false,
        _ => DEFAULT_CASCADE_ENABLED,
    }
}

/// Whether cascade decode is enabled, read once per process.
#[must_use]
pub fn cascade_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_cascade_enabled(std::env::var(CASCADE_ENV).ok().as_deref()))
}

/// Pure parse of a threshold override. A non-integer falls back to the default.
#[must_use]
pub fn parse_threshold(value: Option<&str>, default: usize) -> usize {
    match value.map(str::trim) {
        Some(raw) if !raw.is_empty() => raw.parse::<usize>().unwrap_or(default),
        _ => default,
    }
}

/// The active shared-span floor, read once per process.
#[must_use]
pub fn min_shared_pages() -> usize {
    static FLOOR: OnceLock<usize> = OnceLock::new();
    *FLOOR.get_or_init(|| {
        let raw = std::env::var(MIN_SHARED_PAGES_ENV).ok();
        parse_threshold(raw.as_deref(), DEFAULT_MIN_SHARED_PAGES)
    })
}

/// The active member-count floor, read once per process.
#[must_use]
pub fn min_members() -> usize {
    static FLOOR: OnceLock<usize> = OnceLock::new();
    *FLOOR.get_or_init(|| {
        let raw = std::env::var(MIN_MEMBERS_ENV).ok();
        parse_threshold(raw.as_deref(), DEFAULT_MIN_MEMBERS)
    })
}

/// A subgroup of the batch that shares a whole-page prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CascadeGroup {
    /// Pages of the page table every member holds in common, starting at its
    /// first emitted page.
    pub shared_pages: usize,
    /// Request indices of the members, ascending.
    pub members: Vec<usize>,
}

impl CascadeGroup {
    /// Page reads the decomposition removes: every member beyond the first
    /// stops reading the shared span.
    ///
    /// This is the quantity the group choice maximizes, and the only thing that
    /// makes one candidate subgroup better than another.
    #[must_use]
    pub fn saved_page_reads(&self) -> usize {
        self.shared_pages
            .saturating_mul(self.members.len().saturating_sub(1))
    }
}

/// Find the subgroup whose shared prefix saves the most page reads.
///
/// Pure, allocation-light and MLX-free, so the decision that produced a launch
/// can be reconstructed from the view alone in a test. Returns `None` when no
/// subgroup clears both thresholds, which is the common case and is not an
/// error.
///
/// Candidates are grouped by their first emitted page: two requests can only
/// share a prefix if their page `0` is the same block. Within a candidate every
/// member must share the entire span (a member that diverges early shortens the
/// span for all of them rather than being dropped). That is deliberately the
/// simple rule: the shape this exists for is N clients behind one system
/// prompt, where every member diverges at the same page, and a subgroup search
/// that also considers dropping members would cost more host time than the
/// cases it wins.
#[must_use]
pub fn detect_shared_prefix(
    view: &PagedCsrView,
    min_shared_pages: usize,
    min_members: usize,
) -> Option<CascadeGroup> {
    if min_shared_pages == 0 || min_members < 2 || view.page_size <= 0 {
        return None;
    }
    let batch = view.batch();
    if batch < min_members || view.validate().is_err() {
        return None;
    }

    // A request is eligible when its window starts on a block boundary (so the
    // level-0 span is expressible in whole pages) and it has at least one page
    // that cannot be shared, which is what keeps every shared page full.
    let eligible: Vec<usize> = (0..batch)
        .filter(|&r| {
            view.seq_lens[r] > 0 && view.first_page_offset[r] == 0 && view.pages_for(r) >= 2
        })
        .collect();
    if eligible.len() < min_members {
        return None;
    }

    let first_row = |r: usize| view.indices[view.indptr[r] as usize];
    let page_at = |r: usize, i: usize| view.indices[view.indptr[r] as usize + i];

    let mut best: Option<CascadeGroup> = None;
    let mut anchors: Vec<i32> = Vec::new();
    for &anchor in &eligible {
        let row = first_row(anchor);
        if anchors.contains(&row) {
            continue;
        }
        anchors.push(row);
        let members: Vec<usize> = eligible
            .iter()
            .copied()
            .filter(|&r| first_row(r) == row)
            .collect();
        if members.len() < min_members {
            continue;
        }
        // Cap the span so every member keeps a private page: the last page of a
        // request is the only one that may be partially filled, and level 0
        // declares its pages full.
        let cap = members
            .iter()
            .map(|&r| view.pages_for(r) - 1)
            .min()
            .unwrap_or(0);
        let mut shared = 0usize;
        while shared < cap {
            let want = page_at(members[0], shared);
            if members[1..].iter().any(|&r| page_at(r, shared) != want) {
                break;
            }
            shared += 1;
        }
        if shared < min_shared_pages {
            continue;
        }
        let candidate = CascadeGroup {
            shared_pages: shared,
            members,
        };
        let better = best
            .as_ref()
            .is_none_or(|current| candidate.saved_page_reads() > current.saved_page_reads());
        if better {
            best = Some(candidate);
        }
    }
    best
}

/// The two page tables and the merge grouping one cascade decode step needs.
///
/// Plain data, like [`PagedCsrView`] and
/// [`PagedDecodePlan`](crate::paged_v2::PagedDecodePlan): building it touches
/// no MLX array, so it can be validated, compared and cached independently of a
/// launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CascadePlan {
    /// The subgroup this plan serves.
    pub group: CascadeGroup,
    /// Level 0: a one-request view over the shared span. Its query is every
    /// member's query stacked onto the head axis, see
    /// [`crate::paged_v2::cascade_launch`].
    pub prefix_view: PagedCsrView,
    /// Level 1: the whole batch. Members carry only the pages behind the shared
    /// span; non-members carry their unchanged range.
    pub suffix_view: PagedCsrView,
    /// Rows of `concat(level-1 output, level-0 output)` in merge order, so each
    /// request's partials are contiguous.
    pub merge_order: Vec<i32>,
    /// `[batch + 1]` grouping over [`Self::merge_order`]: two rows for a
    /// member, one for everyone else.
    pub o_indptr: Vec<i32>,
    /// `[members]` request indices as i32, the gather that stacks the member
    /// queries for level 0.
    pub member_rows: Vec<i32>,
}

impl CascadePlan {
    /// Requests in the batch.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.suffix_view.batch()
    }

    /// Members of the shared subgroup.
    #[must_use]
    pub fn members(&self) -> usize {
        self.group.members.len()
    }

    /// Visible tokens the shared span holds.
    #[must_use]
    pub fn shared_tokens(&self) -> usize {
        self.group
            .shared_pages
            .saturating_mul(self.prefix_view.page_size.max(0) as usize)
    }

    /// Whether the member gather can be skipped because level 0's members are
    /// exactly the batch in order.
    #[must_use]
    pub fn members_are_whole_batch(&self) -> bool {
        self.group.members.len() == self.batch()
            && self.group.members.iter().enumerate().all(|(j, &r)| j == r)
    }

    /// Structural check, run on every build for the same reason
    /// [`PagedCsrView::validate`] is: a malformed grouping is an out-of-bounds
    /// read inside the merge kernel rather than an error.
    pub fn validate(&self) -> Result<(), String> {
        self.prefix_view
            .validate()
            .map_err(|e| format!("cascade: level-0 view is malformed ({e})"))?;
        self.suffix_view
            .validate()
            .map_err(|e| format!("cascade: level-1 view is malformed ({e})"))?;
        if self.prefix_view.batch() != 1 {
            return Err(format!(
                "cascade: level-0 view must hold exactly one request, got {}",
                self.prefix_view.batch()
            ));
        }
        let batch = self.batch();
        if self.o_indptr.len() != batch + 1 {
            return Err(format!(
                "cascade: o_indptr has {} entries, expected batch + 1 = {}",
                self.o_indptr.len(),
                batch + 1
            ));
        }
        if self.o_indptr.first() != Some(&0)
            || self.o_indptr.last().copied().unwrap_or(0) as usize != self.merge_order.len()
        {
            return Err("cascade: o_indptr does not span merge_order".to_string());
        }
        if self.merge_order.len() != batch + self.members() {
            return Err(format!(
                "cascade: merge_order has {} rows, expected batch {batch} + {} members",
                self.merge_order.len(),
                self.members()
            ));
        }
        if self.member_rows.len() != self.members() {
            return Err("cascade: member_rows disagrees with the group".to_string());
        }
        Ok(())
    }
}

/// Build the two page tables and the merge grouping for `group` over `view`.
///
/// `view` is the flat batch page table the non-cascade path would have
/// launched; this splits it rather than rebuilding it from the pool, so the two
/// levels are guaranteed to cover exactly the same pages the flat launch would
/// have read, in the same order.
pub fn build_cascade_plan(view: &PagedCsrView, group: CascadeGroup) -> Result<CascadePlan, String> {
    view.validate()
        .map_err(|e| format!("cascade: source view is malformed ({e})"))?;
    let batch = view.batch();
    let page_size = view.page_size;
    let shared = group.shared_pages;
    if shared == 0 {
        return Err("cascade: a shared span of zero pages has nothing to hoist".to_string());
    }
    if group.members.windows(2).any(|w| w[0] >= w[1]) {
        // `binary_search` below and the merge grouping both assume it.
        return Err("cascade: members must be strictly ascending request indices".to_string());
    }
    for &r in &group.members {
        if r >= batch {
            return Err(format!(
                "cascade: member {r} is outside the batch of {batch}"
            ));
        }
        if view.first_page_offset[r] != 0 {
            return Err(format!(
                "cascade: member {r} starts mid-page (first_page_offset {})",
                view.first_page_offset[r]
            ));
        }
        if view.pages_for(r) <= shared {
            return Err(format!(
                "cascade: member {r} holds {} pages, which does not exceed the {shared}-page shared span",
                view.pages_for(r)
            ));
        }
    }

    // Level 0: one request, the shared pages, all of them full.
    let anchor = *group
        .members
        .first()
        .ok_or_else(|| "cascade: an empty group has no shared span".to_string())?;
    let anchor_begin = view.indptr[anchor] as usize;
    let shared_tokens = i64::from(page_size) * shared as i64;
    let shared_tokens = i32::try_from(shared_tokens)
        .map_err(|_| format!("cascade: shared span of {shared_tokens} tokens overflows i32"))?;
    let prefix_view = PagedCsrView {
        page_size,
        indices: view.indices[anchor_begin..anchor_begin + shared].to_vec(),
        indptr: vec![0, shared as i32],
        last_page_len: vec![page_size],
        first_page_offset: vec![0],
        seq_lens: vec![shared_tokens],
        rope_offsets: vec![shared_tokens],
    };

    // Level 1: the whole batch, with members' shared pages removed.
    let mut suffix_view = PagedCsrView {
        page_size,
        indices: Vec::with_capacity(view.indices.len()),
        indptr: Vec::with_capacity(batch + 1),
        last_page_len: Vec::with_capacity(batch),
        first_page_offset: Vec::with_capacity(batch),
        seq_lens: Vec::with_capacity(batch),
        rope_offsets: view.rope_offsets.clone(),
    };
    suffix_view.indptr.push(0);
    for r in 0..batch {
        let begin = view.indptr[r] as usize;
        let end = view.indptr[r + 1] as usize;
        let is_member = group.members.binary_search(&r).is_ok();
        let skip = if is_member { shared } else { 0 };
        suffix_view
            .indices
            .extend_from_slice(&view.indices[begin + skip..end]);
        suffix_view.indptr.push(suffix_view.indices.len() as i32);
        suffix_view.last_page_len.push(view.last_page_len[r]);
        suffix_view.first_page_offset.push(if is_member {
            0
        } else {
            view.first_page_offset[r]
        });
        suffix_view.seq_lens.push(if is_member {
            view.seq_lens[r] - shared_tokens
        } else {
            view.seq_lens[r]
        });
    }

    // Merge grouping over `concat(level-1 rows, level-0 rows)`. A member's two
    // partials land next to each other; everyone else forms a group of one,
    // which the merge kernel resolves to the identity.
    let mut merge_order: Vec<i32> = Vec::with_capacity(batch + group.members.len());
    let mut o_indptr: Vec<i32> = Vec::with_capacity(batch + 1);
    o_indptr.push(0);
    for r in 0..batch {
        merge_order.push(r as i32);
        if let Ok(j) = group.members.binary_search(&r) {
            merge_order.push((batch + j) as i32);
        }
        o_indptr.push(merge_order.len() as i32);
    }
    let member_rows: Vec<i32> = group.members.iter().map(|&r| r as i32).collect();

    let plan = CascadePlan {
        group,
        prefix_view,
        suffix_view,
        merge_order,
        o_indptr,
        member_rows,
    };
    plan.validate()?;
    Ok(plan)
}

#[cfg(test)]
#[path = "cascade_tests.rs"]
mod cascade_tests;
