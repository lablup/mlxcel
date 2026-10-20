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

//! Issue #2462 A1: the sorted `gather_mm` branch of the CUDA overlay
//! (`gather_mm_rhs`, taken for `M == 1 && right_sorted_`) must compute every
//! gathered row, whatever strides the activations arrive with and whatever
//! ran before it in the process.
//!
//! The branch stacks consecutive gathered rows of `a = [rows, 1, K]` into one
//! grouped-GEMM problem per expert and used to take `a.strides(-2)` as the
//! leading dimension between those rows. That is the stride of a size-1 axis,
//! which MLX ignores when it flags an array row-contiguous, so it is whatever
//! the producer left there. The OpenAI Privacy Filter's per-expert bias add
//! (`out + expand_dims(take(bias, ids), -2)`) hands down a `[rows, 1, N]`
//! whose singleton axis has stride 1 whenever the add inherits the bias
//! operand's layout, which depends on which operand the allocator let it
//! donate. Every row after the first of an expert group then read shifted
//! activations: PR #2460's "allocation-history-dependent" drift.
//!
//! The layout tests in `switch_layers_sorted_gather_layout_tests.rs` construct
//! that layout directly and through the Privacy Filter's own op chain; both
//! failed on the unfixed overlay. The history tests here run the same sorted gather before and after predecessor
//! workloads (an 8-bit `gather_qmm` MoE, allocator churn, other sorted shapes)
//! and compare byte for byte; they pass before and after the fix and record
//! that the allocator history alone was never the trigger. Every result is
//! also held against an f64 host reference of the uploaded bytes.
//!
//! GPU-only; skipped on a CPU-only build.

use super::sorted_gather_test_support::{
    SortedCase, allocator_churn, eight_bit_workload, history_test,
};

#[test]
fn sorted_f32_x_bf16_gather_mm_is_independent_of_an_8bit_moe_predecessor() {
    history_test("8-bit MoE predecessor", |rng| {
        eight_bit_workload(rng, 1200, 3);
        eight_bit_workload(rng, 2048, 2);
    });
}

#[test]
fn sorted_f32_x_bf16_gather_mm_is_independent_of_allocator_churn() {
    history_test("allocator churn predecessor", |rng| allocator_churn(rng, 6));
}

#[test]
fn sorted_f32_x_bf16_gather_mm_is_independent_of_other_sorted_shapes() {
    history_test("other sorted shapes predecessor", |rng| {
        for tokens in [512, 7, 300, 1] {
            let other = SortedCase::new(rng, tokens);
            mlxcel_core::eval(&other.sorted());
            mlxcel_core::eval(&other.unsorted());
        }
    });
}
