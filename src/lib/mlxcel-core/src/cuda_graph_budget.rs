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

//! The CUDA graph capture budget mlxcel applies on GB10 (sm_121), issue #1798.
//!
//! MLX's CUDA backend captures every eval into a CUDA graph and commits the
//! graph when either of two budgets is exceeded (`needs_commit`,
//! `mlx/backend/cuda/device.cpp:458-460` at the pin in
//! `src/lib/mlx-cpp/CMakeLists.txt`): an op count, and a byte count. Both come
//! from a per-architecture table (`get_graph_limits`, `device.cpp:181-202`),
//! overridable through `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER`.
//! Compute capability 12.1 (DGX Spark / GB10) is given 20 ops and 25 "MB", the
//! tightest byte budget in the table; consumer Blackwell (12.0), H100 and B200
//! get 100 ops and 1000 "MB".
//!
//! The byte counter is not bytes. `set_input_array` (`device.cpp:228`) adds
//! `array::data_size()` for every input, and `data_size` is an element count
//! (`mlx/array.h:346`, "in units of `item_size` (not bytes)"). Any op whose
//! input has more than `25 << 20` (26.2M) elements therefore commits its own
//! graph regardless of how many ops precede it: every 256-expert NVFP4 expert
//! stack on Laguna (about 120 `gather_qmm` per token), and every 4-bit
//! `lm_head` or tied embedding over 26.2M packed words (Qwen 3.5 4B at 79.5M,
//! Llama 3.1 8B at 65.7M, Gemma 3 4B at 83.9M). On a model where that happens
//! per layer, capture at the default budgets costs more than it saves: the
//! #1799 controls measured Laguna classic decode faster with
//! `MLX_USE_CUDA_GRAPHS=0` than at the defaults, and faster again with both
//! budgets raised. The two knobs mask each other: while the byte cap forces a
//! commit the op cap is never reached, so the op budget alone reads as inert.
//!
//! The measurement behind the value here is
//! `docs/benchmark_results/cuda-graph-budget-gb10-2026-09-12.md`. This is a
//! default only: an operator-set value of either variable always wins, per
//! variable, the same contract as [`crate::hardware::apply_cuda_graph_cache_default`].

use std::env;

use crate::cuda_arch::cuda_compute_capability;

/// The two MLX CUDA graph budgets, in MLX's own units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaGraphBudget {
    /// `MLX_MAX_OPS_PER_BUFFER`: ops captured before a graph is committed.
    pub max_ops: u32,
    /// `MLX_MAX_MB_PER_BUFFER`: input elements (not bytes) `>> 20` captured
    /// before a graph is committed.
    pub max_mb: u32,
}

/// The budget applied on compute capability 12.1 (GB10): MLX's own row for
/// consumer Blackwell, H100 and B200.
pub const GB10_GRAPH_BUDGET: CudaGraphBudget = CudaGraphBudget {
    max_ops: 100,
    max_mb: 1000,
};

/// Environment variable MLX reads for the op budget.
pub const MAX_OPS_ENV: &str = "MLX_MAX_OPS_PER_BUFFER";
/// Environment variable MLX reads for the byte (element) budget.
pub const MAX_MB_ENV: &str = "MLX_MAX_MB_PER_BUFFER";

/// The graph budget to apply for a device's compute capability, or `None` to
/// leave MLX's table value alone.
///
/// Only 12.1 is raised. Every other capability, `None` included (Metal, CPU,
/// a CUDA build with no visible device), returns `None`: MLX already gives
/// 12.0, 9.0 and 10.0 the budget applied here, 8.0 has its own measured row,
/// and 7.0 was measured immaterial on Volta in #1545. Pure so it is testable
/// without a device.
#[must_use]
pub fn cuda_graph_budget_default(
    compute_capability: Option<(u32, u32)>,
) -> Option<CudaGraphBudget> {
    match compute_capability {
        Some((12, 1)) => Some(GB10_GRAPH_BUDGET),
        _ => None,
    }
}

/// Which of the two variables [`apply_cuda_graph_budget_default`] would set,
/// given which ones the operator already set. Pure, so the env-wins contract
/// is testable without touching the process environment.
#[must_use]
pub fn budget_vars_to_set(
    budget: Option<CudaGraphBudget>,
    ops_already_set: bool,
    mb_already_set: bool,
) -> Vec<(&'static str, u32)> {
    let Some(budget) = budget else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(2);
    if !ops_already_set {
        out.push((MAX_OPS_ENV, budget.max_ops));
    }
    if !mb_already_set {
        out.push((MAX_MB_ENV, budget.max_mb));
    }
    out
}

/// Apply the [`cuda_graph_budget_default`] for the running device to the
/// process environment, leaving any variable the operator already set alone.
///
/// Call this once, early in `main()`, before any MLX op runs and before
/// spawning threads: MLX latches both variables the first time a stream's
/// `CommandEncoder` is constructed (`get_graph_limits` through
/// `env::max_ops_per_buffer` / `env::max_mb_per_buffer`, whose values are
/// function-local statics), and setting an environment variable is only sound
/// while the process is effectively single-threaded. The compute-capability
/// probe this calls reads device properties only (`device_info`); it does not
/// construct a `CommandEncoder`, so the variables are still unread when they
/// are set. No-op on Metal, CPU-only and non-CUDA builds, and on every CUDA
/// capability other than 12.1.
pub fn apply_cuda_graph_budget_default() {
    let ops_set = env::var_os(MAX_OPS_ENV).is_some();
    let mb_set = env::var_os(MAX_MB_ENV).is_some();
    if ops_set && mb_set {
        return;
    }
    let budget = cuda_graph_budget_default(cuda_compute_capability());
    for (name, value) in budget_vars_to_set(budget, ops_set, mb_set) {
        // SAFETY: set_var mutates the process-global environment and is unsound
        // only if another thread reads or writes the environment concurrently.
        // Per this function's documented contract, all in-tree callers invoke it
        // once at the top of `main` right after CLI parsing (src/main.rs,
        // src/bin/mlx_server.rs, src/bin/bench_decode.rs,
        // src/bin/speculative_bench.rs), before any model load, MLX op, or
        // worker thread touches the environment, so no other thread is
        // accessing it here.
        unsafe { env::set_var(name, value.to_string()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_compute_capability_12_1_is_raised() {
        assert_eq!(
            cuda_graph_budget_default(Some((12, 1))),
            Some(GB10_GRAPH_BUDGET)
        );
        for cc in [
            None,
            Some((12, 0)),
            Some((10, 0)),
            Some((9, 0)),
            Some((8, 0)),
            Some((8, 6)),
            Some((7, 0)),
            Some((12, 2)),
            Some((13, 1)),
        ] {
            assert_eq!(cuda_graph_budget_default(cc), None, "{cc:?}");
        }
    }

    #[test]
    fn gb10_budget_matches_mlx_blackwell_row() {
        // MLX's `get_graph_limits` gives cc 900, 1000 and 1200 exactly this
        // pair; 1210 is the only row this default moves.
        assert_eq!(GB10_GRAPH_BUDGET.max_ops, 100);
        assert_eq!(GB10_GRAPH_BUDGET.max_mb, 1000);
    }

    #[test]
    fn operator_set_variables_win_per_variable() {
        let b = Some(GB10_GRAPH_BUDGET);
        assert_eq!(
            budget_vars_to_set(b, false, false),
            vec![(MAX_OPS_ENV, 100), (MAX_MB_ENV, 1000)]
        );
        assert_eq!(budget_vars_to_set(b, true, false), vec![(MAX_MB_ENV, 1000)]);
        assert_eq!(budget_vars_to_set(b, false, true), vec![(MAX_OPS_ENV, 100)]);
        assert!(budget_vars_to_set(b, true, true).is_empty());
    }

    #[test]
    fn no_budget_sets_nothing() {
        assert!(budget_vars_to_set(None, false, false).is_empty());
    }
}
