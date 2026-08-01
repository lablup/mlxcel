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

//! Unit tests for the CSR view / chunk plan cache (issue #899).
//!
//! MLX-free: every test drives synthetic [`PagedLayerState`]s and the pure
//! [`build_paged_csr_view`] builder, so the reuse predicate is exercised
//! without a GPU.

use super::*;
use crate::cache::{PagedBlockId, build_paged_csr_view};

const PAGE: usize = 32;

/// A layer state with `len` written tokens, `logical_start` trimmed from the
/// front, and enough blocks to cover `len`. Block ids are `base + page` so
/// distinct requests get distinct blocks.
fn layer(base: u64, len: usize, logical_start: usize) -> PagedLayerState {
    let blocks = len.div_ceil(PAGE);
    PagedLayerState {
        block_ids: (0..blocks)
            .map(|p| PagedBlockId::from_raw(base + p as u64))
            .collect(),
        len,
        logical_start,
    }
}

/// Row assignment mirroring a real pool closely enough for the builder: block
/// id `n` lives on row `n`.
fn build(layers: &[&PagedLayerState]) -> Result<PagedCsrView, String> {
    build_paged_csr_view(PAGE, layers, |id| Some(id.as_u64() as usize))
}

fn geometry() -> PagedDecodeGeometry {
    PagedDecodeGeometry {
        q_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        page_size: PAGE as i32,
    }
}

#[test]
fn a_step_inside_the_current_pages_reuses_the_page_list() {
    let mut cache = PagedDecodeV2Cache::new();
    let a = layer(0, 100, 0);
    let b = layer(100, 200, 0);
    let layers: Vec<&PagedLayerState> = vec![&a, &b];
    let first = cache
        .view_for(0, 7, PAGE, &layers, || build(&layers))
        .expect("build")
        .clone();
    assert_eq!(cache.stats().view_rebuilds, 1);

    // One decode token each: 100 -> 101 and 200 -> 201, neither crosses a page
    // boundary (100 % 32 = 4, 200 % 32 = 8).
    let a2 = layer(0, 101, 0);
    let b2 = layer(100, 201, 0);
    let layers2: Vec<&PagedLayerState> = vec![&a2, &b2];
    let reused = cache
        .view_for(0, 7, PAGE, &layers2, || panic!("must not rebuild"))
        .expect("reuse")
        .clone();

    assert_eq!(cache.stats().view_reuses, 1);
    assert_eq!(cache.stats().view_rebuilds, 1);
    assert_eq!(reused.indices, first.indices);
    assert_eq!(reused.indptr, first.indptr);
    assert_eq!(reused.seq_lens, vec![101, 201]);
    assert_eq!(reused.rope_offsets, vec![101, 201]);
    assert_eq!(reused.last_page_len, vec![101 - 96, 201 - 192]);
}

#[test]
fn a_reused_view_is_identical_to_a_fresh_build() {
    let mut cache = PagedDecodeV2Cache::new();
    let a = layer(0, 100, 0);
    let b = layer(100, 200, 5);
    let layers: Vec<&PagedLayerState> = vec![&a, &b];
    cache
        .view_for(0, 1, PAGE, &layers, || build(&layers))
        .expect("build");

    let a2 = layer(0, 101, 0);
    let b2 = layer(100, 201, 5);
    let layers2: Vec<&PagedLayerState> = vec![&a2, &b2];
    let reused = cache
        .view_for(0, 1, PAGE, &layers2, || panic!("must not rebuild"))
        .expect("reuse")
        .clone();
    let fresh = build(&layers2).expect("fresh build");
    assert_eq!(reused, fresh);
}

#[test]
fn crossing_a_page_boundary_rebuilds() {
    let mut cache = PagedDecodeV2Cache::new();
    // 96 = 3 full pages; the next token opens page 3.
    let a = layer(0, 96, 0);
    let layers: Vec<&PagedLayerState> = vec![&a];
    cache
        .view_for(0, 1, PAGE, &layers, || build(&layers))
        .expect("build");

    let a2 = layer(0, 97, 0);
    let layers2: Vec<&PagedLayerState> = vec![&a2];
    let rebuilt = cache
        .view_for(0, 1, PAGE, &layers2, || build(&layers2))
        .expect("rebuild")
        .clone();
    assert_eq!(cache.stats().view_rebuilds, 2);
    assert_eq!(cache.stats().view_reuses, 0);
    assert_eq!(rebuilt.indices.len(), 4);
}

#[test]
fn an_epoch_bump_rebuilds_even_when_the_fingerprints_match() {
    let mut cache = PagedDecodeV2Cache::new();
    let a = layer(0, 100, 0);
    let layers: Vec<&PagedLayerState> = vec![&a];
    cache
        .view_for(0, 1, PAGE, &layers, || build(&layers))
        .expect("build");
    cache
        .view_for(0, 2, PAGE, &layers, || build(&layers))
        .expect("rebuild after epoch bump");
    assert_eq!(cache.stats().view_rebuilds, 2);
    assert_eq!(cache.stats().view_reuses, 0);
}

#[test]
fn a_front_trim_that_retires_a_page_rebuilds() {
    let mut cache = PagedDecodeV2Cache::new();
    let a = layer(0, 200, 0);
    let layers: Vec<&PagedLayerState> = vec![&a];
    let before = cache
        .view_for(0, 1, PAGE, &layers, || build(&layers))
        .expect("build")
        .clone();
    assert_eq!(before.indices.len(), 7);

    // logical_start 64 retires pages 0 and 1: same epoch, different first_page.
    let trimmed = layer(0, 200, 64);
    let layers2: Vec<&PagedLayerState> = vec![&trimmed];
    let after = cache
        .view_for(0, 1, PAGE, &layers2, || build(&layers2))
        .expect("rebuild")
        .clone();
    assert_eq!(cache.stats().view_rebuilds, 2);
    assert_eq!(after.indices.len(), 5);
    assert_eq!(after.indices[0], 2);
    assert_eq!(after.first_page_offset, vec![0]);
    assert_eq!(after.seq_lens, vec![136]);
}

#[test]
fn a_front_trim_inside_the_same_page_reuses_and_moves_the_offset() {
    let mut cache = PagedDecodeV2Cache::new();
    let a = layer(0, 200, 3);
    let layers: Vec<&PagedLayerState> = vec![&a];
    cache
        .view_for(0, 1, PAGE, &layers, || build(&layers))
        .expect("build");

    // Still inside page 0, so the page list is unchanged.
    let a2 = layer(0, 200, 7);
    let layers2: Vec<&PagedLayerState> = vec![&a2];
    let reused = cache
        .view_for(0, 1, PAGE, &layers2, || panic!("must not rebuild"))
        .expect("reuse")
        .clone();
    assert_eq!(cache.stats().view_reuses, 1);
    assert_eq!(reused.first_page_offset, vec![7]);
    assert_eq!(reused.seq_lens, vec![193]);
    assert_eq!(reused, build(&layers2).expect("fresh"));
}

#[test]
fn a_batch_size_change_rebuilds() {
    let mut cache = PagedDecodeV2Cache::new();
    let a = layer(0, 100, 0);
    let b = layer(100, 100, 0);
    let two: Vec<&PagedLayerState> = vec![&a, &b];
    cache
        .view_for(0, 1, PAGE, &two, || build(&two))
        .expect("build");
    let one: Vec<&PagedLayerState> = vec![&a];
    cache
        .view_for(0, 1, PAGE, &one, || build(&one))
        .expect("rebuild");
    assert_eq!(cache.stats().view_rebuilds, 2);
}

#[test]
fn layers_are_cached_independently() {
    let mut cache = PagedDecodeV2Cache::new();
    let l0 = layer(0, 100, 0);
    let l1 = layer(500, 100, 0);
    let a: Vec<&PagedLayerState> = vec![&l0];
    let b: Vec<&PagedLayerState> = vec![&l1];
    cache.view_for(0, 1, PAGE, &a, || build(&a)).expect("l0");
    cache.view_for(5, 1, PAGE, &b, || build(&b)).expect("l5");
    assert_eq!(cache.stats().view_rebuilds, 2);

    let l0b = layer(0, 101, 0);
    let l1b = layer(500, 101, 0);
    let a2: Vec<&PagedLayerState> = vec![&l0b];
    let b2: Vec<&PagedLayerState> = vec![&l1b];
    let v0 = cache
        .view_for(0, 1, PAGE, &a2, || panic!("l0 must reuse"))
        .expect("l0 reuse")
        .clone();
    let v5 = cache
        .view_for(5, 1, PAGE, &b2, || panic!("l5 must reuse"))
        .expect("l5 reuse")
        .clone();
    assert_eq!(cache.stats().view_reuses, 2);
    assert_eq!(v0.indices, vec![0, 1, 2, 3]);
    assert_eq!(v5.indices, vec![500, 501, 502, 503]);
}

#[test]
fn clear_drops_every_layer() {
    let mut cache = PagedDecodeV2Cache::new();
    let a = layer(0, 100, 0);
    let layers: Vec<&PagedLayerState> = vec![&a];
    cache
        .view_for(0, 1, PAGE, &layers, || build(&layers))
        .expect("build");
    cache.clear();
    cache
        .view_for(0, 1, PAGE, &layers, || build(&layers))
        .expect("rebuild after clear");
    assert_eq!(cache.stats().view_rebuilds, 2);
    assert_eq!(cache.stats().view_reuses, 0);
}

#[test]
fn the_plan_is_reused_while_the_chunk_grouping_holds() {
    let mut cache = PagedDecodeV2Cache::new();
    let geo = geometry();
    let counts = vec![64usize, 64];
    let plan = cache
        .plan_for(&geo, &counts, || {
            PagedDecodePlan::heuristic(geo, &counts, 512)
        })
        .clone();
    assert_eq!(cache.stats().plan_rebuilds, 1);

    // Same page counts: the plan is bit-for-bit reusable.
    let again = cache
        .plan_for(&geo, &counts, || panic!("must not rebuild"))
        .clone();
    assert_eq!(cache.stats().plan_reuses, 1);
    assert_eq!(again, plan);
}

#[test]
fn the_plan_is_rebuilt_when_a_request_needs_another_chunk() {
    let mut cache = PagedDecodeV2Cache::new();
    let geo = geometry();
    let counts = vec![64usize, 64];
    let plan = cache
        .plan_for(&geo, &counts, || {
            PagedDecodePlan::heuristic(geo, &counts, 512)
        })
        .clone();

    // Grow one request by a whole chunk's worth of pages so `matches` fails.
    let grown = vec![64usize + plan.pages_per_chunk as usize, 64];
    let rebuilt = cache
        .plan_for(&geo, &grown, || {
            PagedDecodePlan::heuristic(geo, &grown, 512)
        })
        .clone();
    assert_eq!(cache.stats().plan_rebuilds, 2);
    assert!(rebuilt.num_chunks > plan.num_chunks);
}

#[test]
fn the_plan_is_rebuilt_when_the_geometry_changes() {
    let mut cache = PagedDecodeV2Cache::new();
    let geo = geometry();
    let counts = vec![64usize];
    cache.plan_for(&geo, &counts, || {
        PagedDecodePlan::heuristic(geo, &counts, 512)
    });
    let other = PagedDecodeGeometry {
        head_dim: 64,
        ..geo
    };
    cache.plan_for(&other, &counts, || {
        PagedDecodePlan::heuristic(other, &counts, 512)
    });
    assert_eq!(cache.stats().plan_rebuilds, 2);
    assert_eq!(cache.stats().plan_reuses, 0);
}

#[test]
fn fingerprints_capture_the_visible_page_range() {
    let empty = layer(0, 0, 0);
    let fp = RequestFingerprint::of(&empty, PAGE);
    assert!(!fp.visible);
    assert_eq!(fp.blocks, 0);

    let one = layer(0, 1, 0);
    let fp = RequestFingerprint::of(&one, PAGE);
    assert!(fp.visible);
    assert_eq!((fp.first_page, fp.last_page), (0, 0));

    let spanning = layer(0, 100, 40);
    let fp = RequestFingerprint::of(&spanning, PAGE);
    assert_eq!((fp.first_page, fp.last_page), (1, 3));

    // A fully trimmed layer has no visible tokens even though `len` is
    // non-zero, and must not compare equal to a visible one.
    let trimmed = layer(0, 100, 100);
    let fp_trimmed = RequestFingerprint::of(&trimmed, PAGE);
    assert!(!fp_trimmed.visible);
    assert_ne!(fp_trimmed, fp);
}
