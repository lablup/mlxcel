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
//! `lm_head` or tied embedding over 26.2M packed words. A Laguna token at the
//! defaults is 200 committed graphs; each boundary costs the device about 20
//! to 30 us of launch latency and inter-graph event wait, and raising both
//! budgets to MLX's own row for consumer Blackwell, H100 and B200 (100 ops,
//! 1000 "MB") cuts it to 24 graphs and 17% of the token time. The two knobs
//! mask each other: while the byte cap forces a commit the op cap is never
//! reached, so the op budget alone reads as inert.
//!
//! The sign is a property of the model family and the workload, not of the
//! device, and not of any config-level shape rule that was tried. Raising
//! both budgets on GB10, same binary, idle host, n = 3, single-stream decode:
//! Laguna +17%, qwen3_moe (30B-A3B) +21%, qwen3_5_moe (35B-A3B) +22%, all
//! with disjoint ranges; Qwen 3.5 4B dense +4%; gemma4 26B-A4B flat; gpt_oss
//! 20B -7% with +7 GB of peak memory; Llama 3.1 8B -9%, disjoint. "Stacked
//! expert projection over the byte cap" predicts neither gpt_oss (over,
//! loses) nor qwen3_moe (under, wins). Nor does single-stream decode predict
//! the serving path: measured at n = 3 through `mlxcel-server`, qwen3_moe
//! gains at concurrency 1 and does nothing at 4 or 8, and qwen3_5_moe loses
//! 10.7% at concurrency 8 with disjoint ranges despite its +22% single-stream
//! result. So the gate is an allowlist of the families whose every measured
//! workload gains, exactly as #353's Metal default is an allowlist of
//! measured silicon generations. Today that is Laguna alone; everything else
//! keeps MLX's table value.
//!
//! The measurement behind the value here is
//! `docs/benchmark_results/cuda-graph-budget-gb10-2026-09-12.md`. This is a
//! default only: an operator-set value of either variable always wins, per
//! variable, the same contract as [`crate::hardware::apply_cuda_graph_cache_default`].

use std::env;
use std::path::Path;
use std::sync::OnceLock;

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

/// The `model_type` values (as `config.json` spells them) for which every
/// measured workload on GB10 gains from the raised budgets with disjoint
/// n = 3 ranges: single-stream decode (+17%), classic CLI decode (+15%),
/// serialized serving (+4 to +8%), the DFlash round (-8% wall) and long
/// prefill (-1.5%), at +0.4 GB of peak memory on a short prompt and +7 GB per
/// 2048-token prefill chunk. Families not listed keep MLX's defaults:
/// `gpt_oss` (-7%, +7 GB) and dense Llama (-9%) measured worse, `gemma4` MoE
/// flat, and `qwen3_moe` / `qwen3_5_moe` gained +21% / +22% single-stream but
/// failed on the serving path at n = 3: `qwen3_moe` is flat at concurrency 4
/// and 8, and `qwen3_5_moe` is -10.7% at concurrency 8 with disjoint ranges.
/// A candidate family has to clear the server at more than one concurrency
/// level, not only `mlxcel-bench-decode`.
pub const GB10_RAISED_BUDGET_MODEL_TYPES: &[&str] = &["laguna"];

/// What the policy needs to know about a checkpoint, read from its
/// `config.json` before any weight is loaded.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ModelGraphShape {
    /// The checkpoint's `model_type`, top level or under `text_config`.
    pub model_type: Option<String>,
    /// Routed experts per MoE layer (`num_local_experts`, `num_experts`,
    /// `n_routed_experts`, ...), `None` for a dense checkpoint.
    pub routed_experts: Option<u64>,
}

impl ModelGraphShape {
    /// True when this checkpoint belongs to a family in
    /// [`GB10_RAISED_BUDGET_MODEL_TYPES`] and is the MoE variant that was
    /// measured (a routed-expert count over 1 in its config).
    #[must_use]
    pub fn family_measured_to_gain(&self) -> bool {
        self.routed_experts.is_some_and(|n| n > 1)
            && self
                .model_type
                .as_deref()
                .is_some_and(|t| GB10_RAISED_BUDGET_MODEL_TYPES.contains(&t))
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
            _ => None,
        })
    })
}

/// Derive the [`ModelGraphShape`] from a parsed `config.json`. Pure, so the
/// gate is testable without a checkpoint on disk.
#[must_use]
pub fn model_graph_shape_from_config(config: &Value) -> ModelGraphShape {
    let model_type = [config, config.get("text_config").unwrap_or(&Value::Null)]
        .into_iter()
        .find_map(|section| section.get("model_type").and_then(Value::as_str))
        .map(str::to_owned);
    let routed_experts = first_u64(
        config,
        &[
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
            "num_routed_experts",
            "moe_num_experts",
        ],
    );
    ModelGraphShape {
        model_type,
        routed_experts,
    }
}

/// Read `<model_dir>/config.json` into a [`ModelGraphShape`]. Any failure
/// (no directory, no file, unparsable JSON) reads as an unlisted checkpoint,
/// which leaves MLX's defaults alone; a model that has not been downloaded
/// yet is in that set by design.
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
/// raise. The checkpoint must be a family measured to gain
/// ([`ModelGraphShape::family_measured_to_gain`]); the sign differs by family
/// and by workload (+17% on Laguna everywhere measured, -7% on gpt_oss, -9%
/// on dense Llama, and on qwen3_5_moe +22% single-stream against -10.7%
/// batched at concurrency 8), so no broader default has one sign. Pure so
/// both gates are testable without a device or a checkpoint.
#[must_use]
pub fn cuda_graph_budget_default(
    compute_capability: Option<(u32, u32)>,
    shape: &ModelGraphShape,
) -> Option<CudaGraphBudget> {
    match compute_capability {
        Some((12, 1)) if shape.family_measured_to_gain() => Some(GB10_GRAPH_BUDGET),
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

/// What [`apply_cuda_graph_budget_default`] did in this process, so the
/// startup lines of `mlxcel generate` and `mlxcel-server` can report the
/// budget the process actually runs with rather than a unit test's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedCudaGraphBudget {
    /// The budget selected by the policy.
    pub budget: CudaGraphBudget,
    /// The `model_type` that selected it.
    pub model_type: String,
    /// The variables this process set; a variable the operator had already
    /// set is absent here and keeps the operator's value.
    pub set: Vec<(&'static str, u32)>,
}

static APPLIED: OnceLock<Option<AppliedCudaGraphBudget>> = OnceLock::new();

/// The budget [`apply_cuda_graph_budget_default`] applied in this process,
/// `None` when it applied nothing or has not run.
#[must_use]
pub fn applied_cuda_graph_budget() -> Option<&'static AppliedCudaGraphBudget> {
    APPLIED.get().and_then(Option::as_ref)
}

/// One human-readable line for the startup log when a budget was applied,
/// `None` otherwise (Metal, CPU, an unlisted checkpoint, or a fully
/// operator-set pair), mirroring [`crate::cuda_arch::cuda_arch_startup_summary`].
#[must_use]
pub fn cuda_graph_budget_startup_summary() -> Option<String> {
    let applied = applied_cuda_graph_budget()?;
    let set = applied
        .set
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(" ");
    Some(format!(
        "CUDA graph budget: {set} applied for model_type {} on sm_121 (#1798); an operator-set value wins per variable",
        applied.model_type
    ))
}

/// Apply the [`cuda_graph_budget_default`] for the running device and the
/// checkpoint at `model_dir` to the process environment, leaving any variable
/// the operator already set alone.
///
/// `model_dir` is the `-m` argument when it names a local directory; `None`
/// (router mode, a repo id that is not downloaded yet, a command with no
/// model) reads as unlisted and applies nothing. The budget is process-global
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
/// capability other than 12.1, and on every checkpoint outside the allowlist.
pub fn apply_cuda_graph_budget_default(model_dir: Option<&Path>) {
    let ops_set = env::var_os(MAX_OPS_ENV).is_some();
    let mb_set = env::var_os(MAX_MB_ENV).is_some();
    if ops_set && mb_set {
        let _ = APPLIED.set(None);
        return;
    }
    let shape = model_dir
        .map(model_graph_shape_from_dir)
        .unwrap_or_default();
    let budget = cuda_graph_budget_default(cuda_compute_capability(), &shape);
    let set = budget_vars_to_set(budget, ops_set, mb_set);
    for &(name, value) in &set {
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
    let record = budget.map(|budget| AppliedCudaGraphBudget {
        budget,
        model_type: shape.model_type.clone().unwrap_or_default(),
        set,
    });
    // First caller wins; the in-tree contract is one call per process.
    let _ = APPLIED.set(record);
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn shape(model_type: &str, experts: Option<u64>) -> ModelGraphShape {
        ModelGraphShape {
            model_type: Some(model_type.to_owned()),
            routed_experts: experts,
        }
    }

    #[test]
    fn only_compute_capability_12_1_with_a_listed_family_is_raised() {
        let laguna = shape("laguna", Some(256));
        assert_eq!(
            cuda_graph_budget_default(Some((12, 1)), &laguna),
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
            assert_eq!(cuda_graph_budget_default(cc, &laguna), None, "{cc:?}");
        }
    }

    #[test]
    fn every_measured_winner_is_listed_and_every_loser_is_not() {
        assert!(shape("laguna", Some(256)).family_measured_to_gain());
        // Measured flat or worse on GB10 (gpt_oss -7%, gemma4 MoE flat,
        // Llama -9%), dense (Qwen 3.5 4B), single-stream winners that failed
        // the serving path at n = 3 (qwen3_moe flat at concurrency 4 and 8,
        // qwen3_5_moe -10.7% at 8 with disjoint ranges), and unmeasured
        // families.
        for (t, n) in [
            ("gpt_oss", Some(32)),
            ("gemma4", Some(128)),
            ("llama", None),
            ("qwen3_5", None),
            ("qwen3_moe", Some(128)),
            ("qwen3_5_moe", Some(256)),
            ("qwen3_vl_moe", Some(128)),
            ("qwen3_next", Some(512)),
            ("mixtral", Some(8)),
        ] {
            let s = shape(t, n);
            assert!(!s.family_measured_to_gain(), "{t}");
            assert_eq!(cuda_graph_budget_default(Some((12, 1)), &s), None, "{t}");
        }
        // A listed family name without a routed-expert count is not the
        // measured variant.
        assert!(!shape("laguna", None).family_measured_to_gain());
        assert!(!shape("laguna", Some(1)).family_measured_to_gain());
        assert!(!ModelGraphShape::default().family_measured_to_gain());
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
        assert!(budget_vars_to_set(None, false, false).is_empty());
    }

    #[test]
    fn shape_from_laguna_style_config() {
        let cfg = json!({
            "model_type": "laguna", "hidden_size": 2048, "num_experts": 256,
            "moe_intermediate_size": 512
        });
        let shape = model_graph_shape_from_config(&cfg);
        assert_eq!(shape.model_type.as_deref(), Some("laguna"));
        assert_eq!(shape.routed_experts, Some(256));
        assert!(shape.family_measured_to_gain());
    }

    #[test]
    fn shape_reads_nested_text_config_and_alternate_expert_keys() {
        // gemma-4-26b-a4b style: model_type at top level, MoE keys under
        // text_config; measured flat, so listed nowhere.
        let cfg = json!({
            "model_type": "gemma4",
            "text_config": {"hidden_size": 2816, "num_experts": 128}
        });
        let shape = model_graph_shape_from_config(&cfg);
        assert_eq!(shape.model_type.as_deref(), Some("gemma4"));
        assert_eq!(shape.routed_experts, Some(128));
        assert!(!shape.family_measured_to_gain());
        // A checkpoint whose text tower carries the model_type.
        let cfg = json!({"text_config": {"model_type": "laguna", "num_experts": 256}});
        assert!(model_graph_shape_from_config(&cfg).family_measured_to_gain());
        // deepseek-style expert key.
        let cfg = json!({"model_type": "deepseek_v3", "n_routed_experts": 256});
        assert_eq!(
            model_graph_shape_from_config(&cfg).routed_experts,
            Some(256)
        );
    }

    #[test]
    fn dense_and_malformed_configs_apply_nothing() {
        for cfg in [
            json!({"model_type": "llama", "hidden_size": 4096}),
            json!({"model_type": "laguna"}),
            json!({"model_type": "laguna", "num_experts": "many"}),
            json!({"num_experts": 256}),
            json!([]),
            json!("not an object"),
            json!(null),
        ] {
            let shape = model_graph_shape_from_config(&cfg);
            assert!(!shape.family_measured_to_gain(), "{cfg}");
            assert_eq!(cuda_graph_budget_default(Some((12, 1)), &shape), None);
        }
    }

    #[test]
    fn missing_model_dir_reads_as_unlisted() {
        let shape = model_graph_shape_from_dir(Path::new("/nonexistent/mlxcel-1798"));
        assert_eq!(shape, ModelGraphShape::default());
    }
}
