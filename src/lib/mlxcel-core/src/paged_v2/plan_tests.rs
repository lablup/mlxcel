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

//! Unit tests for the paged decode v2 plan (issue #898).
//!
//! The plan is pure arithmetic over page counts, so these tests need no GPU.
//! They pin the three properties the kernels depend on: the chunk arrays and
//! the merge `indptr` agree, the binary search really returns the largest
//! saturating chunk size, and the emitted chunk count always stays inside the
//! grid bound.

use super::*;

fn geometry() -> PagedDecodeGeometry {
    PagedDecodeGeometry {
        q_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        page_size: 32,
    }
}

fn plan(page_counts: &[usize], ppc: i32) -> PagedDecodePlan {
    PagedDecodePlan::with_chunk_size(geometry(), page_counts, ppc, 128, Source::Default)
}

// ---------------------------------------------------------------------------
// Chunk arithmetic
// ---------------------------------------------------------------------------

#[test]
fn every_request_owns_at_least_one_chunk() {
    // A request with no visible tokens still needs an output row, so it still
    // gets a chunk. Its partial comes back empty and merges to zeros.
    assert_eq!(chunks_for_request(0, 8), 1);
    assert_eq!(chunks_for_request(1, 8), 1);
    assert_eq!(chunks_for_request(8, 8), 1);
    assert_eq!(chunks_for_request(9, 8), 2);
    assert_eq!(chunks_for_request(17, 8), 3);
}

#[test]
fn chunk_count_is_non_increasing_in_chunk_size() {
    // The binary search relies on this monotonicity.
    let counts = vec![37usize, 4, 128, 0, 71];
    let mut previous = usize::MAX;
    for ppc in 1..=140 {
        let n = chunks_for_batch(&counts, ppc);
        assert!(
            n <= previous,
            "chunk count grew from {previous} to {n} at pages_per_chunk {ppc}"
        );
        previous = n;
    }
    assert_eq!(chunks_for_batch(&counts, 128), counts.len());
}

// ---------------------------------------------------------------------------
// Plan structure
// ---------------------------------------------------------------------------

#[test]
fn plan_arrays_agree_with_the_merge_indptr() {
    let counts = vec![10usize, 3, 0];
    let p = plan(&counts, 4);
    // 10 pages -> 3 chunks, 3 pages -> 1, 0 pages -> 1.
    assert_eq!(p.num_chunks, 5);
    assert_eq!(p.request_indices, vec![0, 0, 0, 1, 2]);
    assert_eq!(p.kv_tile_indices, vec![0, 1, 2, 0, 0]);
    assert_eq!(p.o_indptr, vec![0, 3, 4, 5]);
    assert!(p.needs_merge);
    p.validate().expect("plan is well formed");
}

#[test]
fn one_chunk_per_request_needs_no_merge() {
    let counts = vec![10usize, 3, 7];
    let p = plan(&counts, 16);
    assert_eq!(p.num_chunks, 3);
    assert!(!p.needs_merge);
    // Chunks are emitted in request order, which is what lets the run step
    // reshape the partial output straight into [B, Hq, 1, D].
    assert_eq!(p.request_indices, vec![0, 1, 2]);
    assert_eq!(p.kv_tile_indices, vec![0, 0, 0]);
    assert_eq!(p.o_indptr, vec![0, 1, 2, 3]);
    p.validate().expect("plan is well formed");
}

#[test]
fn total_ctas_counts_every_kv_head_and_q_group() {
    let counts = vec![64usize];
    let p = plan(&counts, 8);
    assert_eq!(p.num_chunks, 8);
    assert_eq!(p.total_ctas, 8 * geometry().ctas_per_chunk());
}

#[test]
fn workspace_size_matches_the_kernel_outputs() {
    let counts = vec![10usize, 10];
    let p = plan(&counts, 4);
    let g = geometry();
    assert_eq!(
        p.workspace_partial_v_elems(),
        p.num_chunks * g.q_heads as usize * g.head_dim as usize
    );
    assert_eq!(p.workspace_lse_elems(), p.num_chunks * g.q_heads as usize);
    assert_eq!(
        p.workspace_bytes(),
        (p.workspace_partial_v_elems() + p.workspace_lse_elems()) * 4
    );
}

#[test]
fn matches_tracks_page_growth_across_decode_steps() {
    // A decode step appends one token, which changes a page count only when it
    // crosses a page boundary. The plan stays valid until then.
    let counts = vec![8usize, 8];
    let p = plan(&counts, 4);
    assert!(p.matches(&geometry(), &[8, 8]));
    assert!(
        p.matches(&geometry(), &[8, 7]),
        "same chunk count, still valid"
    );
    assert!(
        !p.matches(&geometry(), &[9, 8]),
        "9 pages needs a third chunk"
    );
    assert!(!p.matches(&geometry(), &[8]), "batch composition changed");
    let mut other = geometry();
    other.head_dim = 64;
    assert!(!p.matches(&other, &[8, 8]), "geometry changed");
}

#[test]
fn validate_rejects_corrupted_plans() {
    let counts = vec![10usize, 3];
    let mut p = plan(&counts, 4);
    p.o_indptr.pop();
    assert!(p.validate().is_err());

    let mut p = plan(&counts, 4);
    p.request_indices.pop();
    assert!(p.validate().is_err());

    let mut p = plan(&counts, 16);
    assert!(!p.needs_merge);
    p.num_chunks += 1;
    p.request_indices.push(0);
    p.kv_tile_indices.push(1);
    assert!(
        p.validate().is_err(),
        "a merge-free plan with more chunks than requests must be rejected"
    );
}

#[test]
fn validate_rejects_a_geometry_the_kernel_cannot_serve() {
    let bad = PagedDecodeGeometry {
        q_heads: 12,
        kv_heads: 8, // 12 % 8 != 0
        head_dim: 128,
        page_size: 32,
    };
    let p = PagedDecodePlan::with_chunk_size(bad, &[4], 4, 128, Source::Default);
    assert!(p.validate().is_err());
}

// ---------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------

#[test]
fn search_returns_the_largest_chunk_size_that_reaches_the_target() {
    // 1 request, 64 pages, 8 CTAs per chunk, target 64 CTAs => 8 chunks needed
    // => 8 pages per chunk. 9 pages would give 8 chunks too (ceil(64/9) = 8),
    // and 10 would give 7, so the largest saturating value is 9.
    let counts = vec![64usize];
    let ppc = search_pages_per_chunk(&counts, 8, 64);
    assert_eq!(chunks_for_batch(&counts, ppc) * 8, 64);
    assert!(
        chunks_for_batch(&counts, ppc + 1) * 8 < 64,
        "pages_per_chunk {ppc} was not maximal"
    );
}

#[test]
fn search_is_exhaustively_maximal_over_a_sweep() {
    // Brute-force the same predicate the binary search encodes.
    for &counts in &[
        &[1usize, 1, 1, 1][..],
        &[64, 64, 64, 64][..],
        &[1024][..],
        &[0, 0, 512][..],
        &[7, 13, 29, 31][..],
    ] {
        for &per_chunk in &[1usize, 4, 8, 32] {
            for &target in &[1usize, 16, 64, 128, 512, 4096] {
                let got = search_pages_per_chunk(counts, per_chunk, target);
                let lo = min_pages_per_chunk(counts);
                let hi = max_pages_per_chunk(counts).max(lo);
                let reaches =
                    |p: i32| chunks_for_batch(counts, p).saturating_mul(per_chunk) >= target;
                let expected = if reaches(lo) {
                    (lo..=hi).filter(|&p| reaches(p)).max().unwrap_or(lo)
                } else {
                    lo
                };
                assert_eq!(
                    got, expected,
                    "counts {counts:?} per_chunk {per_chunk} target {target}"
                );
            }
        }
    }
}

#[test]
fn search_falls_back_to_the_finest_split_when_the_target_is_unreachable() {
    // 1 request of 2 pages against a target no split can reach.
    let counts = vec![2usize];
    let ppc = search_pages_per_chunk(&counts, 1, 10_000);
    assert_eq!(ppc, 1);
    assert_eq!(chunks_for_batch(&counts, ppc), 2);
}

#[test]
fn search_never_exceeds_the_grid_bound() {
    // A pathological batch: many requests, each long enough that a one-page
    // chunk would blow past MAX_CHUNKS. The floor keeps the plan launchable.
    let counts = vec![4096usize; 64];
    let ppc = search_pages_per_chunk(&counts, 1, usize::MAX);
    let p = plan(&counts, ppc);
    assert!(
        p.num_chunks <= MAX_CHUNKS,
        "{} chunks exceeds the grid bound",
        p.num_chunks
    );
    p.validate().expect("bounded plan is valid");
}

#[test]
fn min_pages_per_chunk_bounds_the_chunk_count() {
    for counts in [
        vec![1_000_000usize],
        vec![100_000usize; 8],
        vec![1usize; 1000],
        vec![0usize; 16],
    ] {
        let lo = min_pages_per_chunk(&counts);
        assert!(
            chunks_for_batch(&counts, lo) <= MAX_CHUNKS,
            "counts of len {} at pages_per_chunk {lo} produced {} chunks",
            counts.len(),
            chunks_for_batch(&counts, lo)
        );
    }
}

#[test]
fn heuristic_plan_records_its_target_and_source() {
    let counts = vec![32usize, 32];
    let p = PagedDecodePlan::heuristic(geometry(), &counts, 256);
    assert_eq!(p.target_ctas, 256);
    assert_eq!(p.chunk_source, Source::Default);
    p.validate().expect("heuristic plan is valid");
}

// ---------------------------------------------------------------------------
// Geometry helpers (these consult the C++ launcher, so they pin the contract
// the plan and the kernel share)
// ---------------------------------------------------------------------------

#[test]
fn q_heads_per_cta_always_divides_n_rep() {
    for &q_heads in &[1i32, 2, 4, 8, 16, 32, 64] {
        for &kv_heads in &[1i32, 2, 4, 8] {
            if q_heads % kv_heads != 0 {
                continue;
            }
            for &head_dim in &[64i32, 128, 256] {
                let g = PagedDecodeGeometry {
                    q_heads,
                    kv_heads,
                    head_dim,
                    page_size: 32,
                };
                let per_cta = g.q_heads_per_cta();
                assert!(per_cta >= 1);
                assert_eq!(
                    g.n_rep() % per_cta,
                    0,
                    "q_heads_per_cta {per_cta} does not divide n_rep {} (dim {head_dim})",
                    g.n_rep()
                );
                assert_eq!(g.q_groups() * per_cta, g.n_rep());
                g.check().expect("supported geometry");
            }
        }
    }
}

#[test]
fn num_warps_stays_inside_the_threadgroup_budget() {
    for &head_dim in &[64i32, 128, 256] {
        for &q_per_cta in &[1i32, 2, 4, 8] {
            let g = PagedDecodeGeometry {
                q_heads: q_per_cta,
                kv_heads: 1,
                head_dim,
                page_size: 32,
            };
            let warps = ffi::paged_attention_v2_num_warps(head_dim, q_per_cta);
            assert!((1..=8).contains(&warps), "warps {warps} out of range");
            assert!(
                (warps as u32).is_power_of_two(),
                "warps {warps} is not a power of two"
            );
            let bytes = warps as usize * q_per_cta as usize * head_dim as usize * 4;
            assert!(bytes <= 28672, "tg_acc would need {bytes} bytes");
            let _ = g;
        }
    }
}
