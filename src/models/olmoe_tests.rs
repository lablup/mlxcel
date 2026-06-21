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

//! Unit tests for the OLMoE router scoring (issue #318).
//!
//! These pin the softmax-then-gather order of `router_topk_scores` against
//! ml-explore/mlx-lm OlmoeSparseMoeBlock
//! (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/olmoe.py) so
//! the norm_topk_prob=false case (OLMoE-1B-7B-0125) keeps full-softmax
//! probabilities that sum to < 1, instead of a top-k-only softmax that always
//! sums to 1.

use super::router_topk_scores;

#[test]
fn fused_dispatch_gate_is_callable() {
    // Confirm that the fused_moe_enabled gate used in SparseMoeBlock::forward
    // compiles and returns a bool. The actual dispatch is exercised at runtime
    // with real model weights.
    let _enabled: bool = crate::models::switch_layers::fused_moe_enabled();
}

// Four-expert, top-2 router logits with a known full softmax.
// logits = [1, 3, 2, 0]; full softmax over all four experts is
//   p = [0.0871443, 0.6439142, 0.2368828, 0.0320586].
// The top-2 by logit are experts 1 and 2, so the gathered full-softmax scores
// sum to p1 + p2 = 0.880797 (< 1) when norm_topk_prob is false.
const LOGITS: [f32; 4] = [1.0, 3.0, 2.0, 0.0];
const FULL_SOFTMAX: [f32; 4] = [0.0871443, 0.6439142, 0.2368828, 0.0320586];

// Read the selected expert indices back as a sorted Vec so assertions are
// independent of argpartition's internal ordering within the top-k.
fn selected_index_set(topk_indices: &mlxcel_core::MlxArray) -> Vec<i32> {
    let idx = mlxcel_core::astype(topk_indices, mlxcel_core::dtype::INT32);
    let i0 = mlxcel_core::item_i32(&mlxcel_core::slice(&idx, &[0, 0], &[1, 1]));
    let i1 = mlxcel_core::item_i32(&mlxcel_core::slice(&idx, &[0, 1], &[1, 2]));
    let mut got = vec![i0, i1];
    got.sort();
    got
}

#[test]
#[ignore = "requires serial MLX execution"]
fn router_scores_are_full_softmax_gathered_when_norm_topk_prob_false() {
    let logits = mlxcel_core::from_slice_f32(&LOGITS, &[1, 4]);
    let (topk_indices, scores) = router_topk_scores(&logits, 2, false);

    // The selected experts are the top-2 by logit (1 and 2), unchanged by the
    // scoring fix.
    assert_eq!(selected_index_set(&topk_indices), vec![1, 2]);

    // Scores must equal the FULL softmax probabilities gathered at the selected
    // experts. Gather the hand-computed full softmax with the same indices so the
    // comparison is order-agnostic.
    let full = mlxcel_core::from_slice_f32(&FULL_SOFTMAX, &[1, 4]);
    let expected = mlxcel_core::take_along_axis(&full, &topk_indices, -1);
    let close = mlxcel_core::allclose(&scores, &expected, 1e-4, 1e-5);
    mlxcel_core::eval(&close);
    assert!(
        mlxcel_core::item_bool(&close),
        "scores must equal the full softmax gathered at the top-k experts"
    );

    // They sum to < 1 (0.880797), and crucially NOT to 1: the pre-#318 top-k-only
    // softmax would have summed to exactly 1, silently behaving as if
    // norm_topk_prob were true.
    let sum = mlxcel_core::sum_axis(&scores, -1, true);
    mlxcel_core::eval(&sum);
    let sum = mlxcel_core::item_f32(&sum);
    assert!((sum - 0.880797).abs() < 1e-3, "unexpected score sum: {sum}");
    assert!(
        sum < 0.99,
        "norm_topk_prob=false scores must sum to < 1, got {sum}"
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn router_scores_are_renormalized_when_norm_topk_prob_true() {
    let logits = mlxcel_core::from_slice_f32(&LOGITS, &[1, 4]);
    let (topk_indices, scores) = router_topk_scores(&logits, 2, true);

    // Same experts selected regardless of the norm flag.
    assert_eq!(selected_index_set(&topk_indices), vec![1, 2]);

    // Expected = full softmax gathered at the top-k, then renormalized to sum to
    // 1. (This renormalized value also equals a softmax over the top-k logits,
    // which is why the two implementations agree only when norm_topk_prob=true.)
    let full = mlxcel_core::from_slice_f32(&FULL_SOFTMAX, &[1, 4]);
    let gathered = mlxcel_core::take_along_axis(&full, &topk_indices, -1);
    let gathered_sum = mlxcel_core::sum_axis(&gathered, -1, true);
    let expected = mlxcel_core::divide(&gathered, &gathered_sum);
    let close = mlxcel_core::allclose(&scores, &expected, 1e-4, 1e-5);
    mlxcel_core::eval(&close);
    assert!(
        mlxcel_core::item_bool(&close),
        "scores must equal the renormalized top-k probabilities"
    );

    // Renormalized scores sum to 1.
    let sum = mlxcel_core::sum_axis(&scores, -1, true);
    mlxcel_core::eval(&sum);
    let sum = mlxcel_core::item_f32(&sum);
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "renormalized scores must sum to 1, got {sum}"
    );
}
