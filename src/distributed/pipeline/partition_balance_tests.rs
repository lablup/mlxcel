// Copyright 2025-2026 Lablup Inc.
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

use super::*;

#[test]
fn balance_uniform_layers_two_stages() {
    let bytes = vec![100u64; 8];
    let budget = vec![1_000, 1_000];
    let (ranges, warnings) =
        balance_layers(&bytes, &budget, &[], &[], &["d0".into(), "d1".into()]).unwrap();
    assert_eq!(ranges, vec![0..4, 4..8]);
    assert!(warnings.is_empty());
}

#[test]
fn balance_variable_layers_skews_toward_heavy_stage() {
    // Front-loaded: layers 0..3 are heavy, 3..8 are cheap.
    let bytes = vec![400, 400, 400, 50, 50, 50, 50, 50];
    let budget = vec![5_000, 5_000];
    let (ranges, _) =
        balance_layers(&bytes, &budget, &[], &[], &["d0".into(), "d1".into()]).unwrap();
    // Optimal split puts heavy block on one stage, light block on the
    // other.
    assert!(ranges[0].end <= 3, "got ranges {ranges:?}");
}

#[test]
fn balance_respects_forbidden_boundaries() {
    let bytes = vec![100u64; 8];
    let budget = vec![1_000, 1_000];
    // Forbid splitting at 4 — the balancer must slide to 3 or 5.
    let (ranges, _) =
        balance_layers(&bytes, &budget, &[4], &[], &["d0".into(), "d1".into()]).unwrap();
    assert_ne!(ranges[0].end, 4);
}

#[test]
fn balance_respects_tight_budget() {
    // Total 800 bytes, one stage may hold at most 300.
    let bytes = vec![100u64; 8];
    let budget = vec![300, 1_000];
    let (ranges, _) =
        balance_layers(&bytes, &budget, &[], &[], &["d0".into(), "d1".into()]).unwrap();
    assert!(ranges[0].end <= 3);
}

#[test]
fn balance_errors_when_every_layer_exceeds_largest_budget() {
    // A single layer is 400 bytes but even the largest budget is
    // 100 bytes. No split can accommodate the heaviest single layer,
    // so the balancer rejects the request up front.
    let bytes = vec![400u64; 4];
    let budget = vec![100, 100];
    let err = balance_layers(&bytes, &budget, &[], &[], &["d0".into(), "d1".into()]).unwrap_err();
    assert!(err.to_string().contains("cannot balance"));
}

#[test]
fn balance_accommodates_one_tight_stage_with_room_on_another() {
    // Per-layer = 100 bytes, total = 800. Stage 0 has a tight 300-byte
    // budget; stage 1 has ample room at 1000 bytes. The balancer must
    // find a 3:5 or similar split that respects stage 0's cap.
    let bytes = vec![100u64; 8];
    let budget = vec![300, 1000];
    let (ranges, _) = balance_layers(&bytes, &budget, &[], &[], &["tight".into(), "roomy".into()])
        .expect("balancer must find a plan when one stage has headroom");
    assert!(ranges[0].end <= 3);
}

#[test]
fn balance_single_stage_returns_whole_model() {
    let bytes = vec![100u64; 5];
    let budget = vec![10_000];
    let (ranges, warnings) = balance_layers(&bytes, &budget, &[], &[], &["d0".into()]).unwrap();
    assert_eq!(ranges, vec![0..5]);
    assert!(warnings.is_empty());
}

#[test]
fn imbalance_warning_fires_on_forced_skew() {
    // Adjacency forces layers 0..7 to stay together; layer 7 is the
    // only option for the second stage. 1:6 split triggers the warning.
    let bytes = vec![10u64; 8];
    let budget = vec![10_000, 10_000];
    let forbidden: Vec<usize> = (1..7).collect();
    let adjacency = vec![LayerAdjacencyGroup {
        layers: 0..7,
        reason: "synthetic forced-join".into(),
    }];
    let (ranges, warnings) = balance_layers(
        &bytes,
        &budget,
        &forbidden,
        &adjacency,
        &["d0".into(), "d1".into()],
    )
    .unwrap();
    // Any split must be at boundary 7 because 1..7 is forbidden.
    assert_eq!(ranges[0].end, 7);
    assert!(!warnings.is_empty(), "expected imbalance warning");
}
