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
use std::path::Path;

use serde_json::Value;

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

/// MLX's byte budget for compute capability 12.1, in the units the counter
/// actually uses: `25 "MB"` is `25 << 20` input elements per captured graph
/// (`get_graph_limits` and `needs_commit`, `mlx/backend/cuda/device.cpp`).
pub const GB10_DEFAULT_MB_ELEMENTS: u64 = 25 << 20;

/// What the policy needs to know about a checkpoint, read from its
/// `config.json` before any weight is loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ModelGraphShape {
    /// Routed experts per MoE layer (`num_local_experts`, `num_experts`,
    /// `n_routed_experts`, ...), `None` for a dense model.
    pub routed_experts: Option<u64>,
    /// Elements of one stacked routed-expert projection as the loader hands it
    /// to `gather_qmm`: `experts * moe_intermediate * hidden` packed at the
    /// checkpoint's weight width (8 values per `u32` word at 4 bits, 4 at 8
    /// bits, 1 per element unquantized). `None` when any of the three
    /// dimensions is missing.
    pub expert_stack_elements: Option<u64>,
}

impl ModelGraphShape {
    /// True when a single routed-expert projection exceeds MLX's GB10 byte
    /// budget on its own, so every expert matmul commits its own CUDA graph
    /// at the default budgets.
    #[must_use]
    pub fn expert_stack_exceeds_gb10_budget(&self) -> bool {
        self.expert_stack_elements
            .is_some_and(|elements| (elements >> 20) > (GB10_DEFAULT_MB_ELEMENTS >> 20))
    }
}

fn config_sections(config: &Value) -> impl Iterator<Item = &Value> {
    ["text_config", "language_model", "llm_config"]
        .into_iter()
        .filter_map(|key| config.get(key))
        .chain(std::iter::once(config))
}

fn first_u64(config: &Value, keys: &[&str]) -> Option<u64> {
    config_sections(config).find_map(|section| {
        keys.iter().find_map(|key| match section.get(key) {
            Some(Value::Number(n)) => n.as_u64(),
            // Per-layer lists (hunyuan): the first entry is representative.
            Some(Value::Array(items)) => items.first().and_then(Value::as_u64),
            _ => None,
        })
    })
}

/// Values packed per stored element for the checkpoint's weight width:
/// `quantization.bits` (MLX affine / mxfp4 / nvfp4 checkpoints) or a
/// compressed-tensors `num_bits`, else 1 (bf16, f16, f32).
fn values_per_element(config: &Value) -> u64 {
    let bits = config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(Value::as_u64)
        .or_else(|| {
            config
                .get("quantization_config")
                .and_then(|q| q.get("config_groups"))
                .and_then(Value::as_object)
                .and_then(|groups| groups.values().next())
                .and_then(|g| g.get("weights"))
                .and_then(|w| w.get("num_bits"))
                .and_then(Value::as_u64)
        });
    match bits {
        Some(bits) if bits > 0 && bits < 32 => 32 / bits,
        _ => 1,
    }
}

/// Derive the [`ModelGraphShape`] from a parsed `config.json`. Pure, so the
/// shape rules are testable without a checkpoint on disk.
#[must_use]
pub fn model_graph_shape_from_config(config: &Value) -> ModelGraphShape {
    let routed_experts = first_u64(
        config,
        &[
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
            "num_routed_experts",
            "moe_num_experts",
        ],
    )
    .filter(|&n| n > 1);
    let expert_stack_elements = routed_experts.and_then(|experts| {
        let intermediate = first_u64(config, &["moe_intermediate_size", "intermediate_size"])?;
        let hidden = first_u64(config, &["hidden_size"])?;
        Some(experts * intermediate * hidden / values_per_element(config))
    });
    ModelGraphShape {
        routed_experts,
        expert_stack_elements,
    }
}

/// Read `<model_dir>/config.json` into a [`ModelGraphShape`]. Any failure
/// (no directory, no file, unparsable JSON) reads as a dense model, which
/// leaves MLX's defaults alone; a model that has not been downloaded yet
/// is in that set by design.
#[must_use]
pub fn model_graph_shape_from_dir(model_dir: &Path) -> ModelGraphShape {
    std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .map(|config| model_graph_shape_from_config(&config))
        .unwrap_or_default()
}

/// The graph budget to apply for a device and a checkpoint, or `None` to
/// leave MLX's table value alone.
///
/// Two gates, both required. The device must be compute capability 12.1:
/// MLX already gives 12.0, 9.0 and 10.0 the budget applied here, 8.0 has its
/// own measured row, 7.0 was measured immaterial on Volta in #1545, and
/// `None` (Metal, CPU, a CUDA build with no visible device) has nothing to
/// raise. The checkpoint must carry a routed-expert stack that on its own
/// exceeds the 12.1 byte budget ([`ModelGraphShape::expert_stack_exceeds_gb10_budget`]):
/// that is the shape on which every expert matmul commits its own graph and
/// raising both budgets measured +17% (Laguna). Dense checkpoints are left
/// alone on purpose, because the same budgets measured +4% on Qwen 3.5 4B and
/// -9% on Llama 3.1 8B, ranges disjoint, so no dense default has one sign.
/// Pure so both gates are testable without a device or a checkpoint.
#[must_use]
pub fn cuda_graph_budget_default(
    compute_capability: Option<(u32, u32)>,
    shape: ModelGraphShape,
) -> Option<CudaGraphBudget> {
    match compute_capability {
        Some((12, 1)) if shape.expert_stack_exceeds_gb10_budget() => Some(GB10_GRAPH_BUDGET),
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

/// Apply the [`cuda_graph_budget_default`] for the running device and the
/// checkpoint at `model_dir` to the process environment, leaving any variable
/// the operator already set alone.
///
/// `model_dir` is the `-m` argument when it names a local directory; `None`
/// (router mode, a repo id that is not downloaded yet, a command with no
/// model) reads as dense and applies nothing. The budget is process-global
/// and latched once, so a server hosting several checkpoints takes the
/// shape of the one it was started with.
///
/// Call this once, early in `main()`, before any MLX op runs and before
/// spawning threads: MLX latches both variables the first time a stream's
/// `CommandEncoder` is constructed (`get_graph_limits` through
/// `env::max_ops_per_buffer` / `env::max_mb_per_buffer`, whose values are
/// function-local statics), and setting an environment variable is only sound
/// while the process is effectively single-threaded. The compute-capability
/// probe this calls reads device properties only (`device_info`); it does not
/// construct a `CommandEncoder`, so the variables are still unread when they
/// are set. No-op on Metal, CPU-only and non-CUDA builds, on every CUDA
/// capability other than 12.1, and on every dense checkpoint.
pub fn apply_cuda_graph_budget_default(model_dir: Option<&Path>) {
    let ops_set = env::var_os(MAX_OPS_ENV).is_some();
    let mb_set = env::var_os(MAX_MB_ENV).is_some();
    if ops_set && mb_set {
        return;
    }
    let shape = model_dir
        .map(model_graph_shape_from_dir)
        .unwrap_or_default();
    let budget = cuda_graph_budget_default(cuda_compute_capability(), shape);
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
    use serde_json::json;

    use super::*;

    const LAGUNA: ModelGraphShape = ModelGraphShape {
        routed_experts: Some(256),
        expert_stack_elements: Some(256 * 512 * 2048 / 8),
    };
    const DENSE: ModelGraphShape = ModelGraphShape {
        routed_experts: None,
        expert_stack_elements: None,
    };

    #[test]
    fn only_compute_capability_12_1_with_an_over_budget_expert_stack_is_raised() {
        assert_eq!(
            cuda_graph_budget_default(Some((12, 1)), LAGUNA),
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
            assert_eq!(cuda_graph_budget_default(cc, LAGUNA), None, "{cc:?}");
        }
    }

    #[test]
    fn dense_and_under_budget_moe_checkpoints_keep_mlx_defaults_on_12_1() {
        assert_eq!(cuda_graph_budget_default(Some((12, 1)), DENSE), None);
        // qwen3-30b-a3b: 128 * 768 * 2048 / 8 = 25.17M packed words, under
        // the 26.2M budget, so its expert matmuls do not commit on bytes.
        let under = ModelGraphShape {
            routed_experts: Some(128),
            expert_stack_elements: Some(128 * 768 * 2048 / 8),
        };
        assert!(!under.expert_stack_exceeds_gb10_budget());
        assert_eq!(cuda_graph_budget_default(Some((12, 1)), under), None);
    }

    #[test]
    fn gb10_budget_matches_mlx_blackwell_row() {
        // MLX's `get_graph_limits` gives cc 900, 1000 and 1200 exactly this
        // pair; 1210 is the only row this default moves.
        assert_eq!(GB10_GRAPH_BUDGET.max_ops, 100);
        assert_eq!(GB10_GRAPH_BUDGET.max_mb, 1000);
        assert_eq!(GB10_DEFAULT_MB_ELEMENTS, 26_214_400);
    }

    #[test]
    fn budget_threshold_is_in_mlx_units() {
        // `needs_commit` compares `bytes >> 20` against the table value, so
        // exactly 25 << 20 elements is not over budget and one more MiB is.
        let at = ModelGraphShape {
            routed_experts: Some(2),
            expert_stack_elements: Some(25 << 20),
        };
        let over = ModelGraphShape {
            routed_experts: Some(2),
            expert_stack_elements: Some(26 << 20),
        };
        assert!(!at.expert_stack_exceeds_gb10_budget());
        assert!(over.expert_stack_exceeds_gb10_budget());
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
        assert!(budget_vars_to_set(None, false, false).is_empty());
    }

    #[test]
    fn shape_from_laguna_style_config() {
        // Laguna: compressed-tensors nvfp4, top-level MoE keys.
        let cfg = json!({
            "model_type": "laguna", "hidden_size": 2048, "num_experts": 256,
            "moe_intermediate_size": 512, "intermediate_size": 8192,
            "quantization_config": {"config_groups": {"group_0": {"weights": {"num_bits": 4}}}}
        });
        let shape = model_graph_shape_from_config(&cfg);
        assert_eq!(shape.routed_experts, Some(256));
        assert_eq!(shape.expert_stack_elements, Some(256 * 512 * 2048 / 8));
        assert!(shape.expert_stack_exceeds_gb10_budget());
    }

    #[test]
    fn shape_prefers_moe_intermediate_and_reads_nested_text_config() {
        // gemma-4-26b-a4b style: keys under text_config, MLX affine 4-bit.
        let cfg = json!({
            "model_type": "gemma4",
            "text_config": {"hidden_size": 2816, "num_experts": 128,
                            "moe_intermediate_size": 704, "intermediate_size": 11264},
            "quantization": {"group_size": 64, "bits": 4}
        });
        let shape = model_graph_shape_from_config(&cfg);
        assert_eq!(shape.expert_stack_elements, Some(128 * 704 * 2816 / 8));
        // deepseek style key, 8-bit packs 4 per word, hunyuan per-layer list.
        let cfg = json!({
            "hidden_size": 4096, "n_routed_experts": 64, "moe_intermediate_size": [3072, 3072],
            "quantization": {"bits": 8}
        });
        assert_eq!(
            model_graph_shape_from_config(&cfg).expert_stack_elements,
            Some(64 * 3072 * 4096 / 4)
        );
    }

    #[test]
    fn dense_and_malformed_configs_read_as_dense() {
        for cfg in [
            json!({"model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336}),
            json!({"num_experts": 1, "hidden_size": 8, "intermediate_size": 8}),
            json!({"num_experts": 0}),
            json!({"num_experts": 64, "hidden_size": 2048}),
            json!([]),
            json!("not an object"),
        ] {
            let shape = model_graph_shape_from_config(&cfg);
            assert!(!shape.expert_stack_exceeds_gb10_budget(), "{cfg}");
            assert_eq!(cuda_graph_budget_default(Some((12, 1)), shape), None);
        }
        // bf16 experts pack 1 per element: 8 * 64 * 64 is tiny, still dense.
        let cfg = json!({"num_local_experts": 8, "hidden_size": 64, "intermediate_size": 64});
        assert_eq!(
            model_graph_shape_from_config(&cfg).expert_stack_elements,
            Some(8 * 64 * 64)
        );
    }

    #[test]
    fn missing_model_dir_reads_as_dense() {
        let shape = model_graph_shape_from_dir(Path::new("/nonexistent/mlxcel-1798"));
        assert_eq!(shape, ModelGraphShape::default());
    }
}
