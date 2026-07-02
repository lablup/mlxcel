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

//! Shared helpers for single-machine in-process pipeline execution.
//!
//! Used by: CLI pipeline generate path, server pipeline runtime

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};

use crate::distributed::pipeline::{
    ChannelConfig, DeviceSpec, InProcessStageWorkerLoop, LoadedStageExecutor, ModelProfile,
    PartitionQualityReport, PipelineConfig, StageAssignment, StageExecutor,
    auto_partition_with_report, build_manual_assignments, build_model_profile,
    format_quality_report, parse_manual_partition,
};
use crate::models::sanitize_config_json;

fn equal_stage_model_profile(num_layers: usize) -> ModelProfile {
    ModelProfile::uniform(num_layers, 1024, 1024, 1024)
}

fn equal_capacity_devices(num_stages: usize, model: &ModelProfile) -> Vec<DeviceSpec> {
    // Budget bound: sum of all layer bytes plus the heaviest stage's
    // worst-case layer count (using the legacy uniform helper — the real
    // model profile carries the vector and its own headroom). This keeps
    // auto-partition happy for both the uniform test profile and the
    // config-derived profile where `layer_bytes.is_some()`.
    let per_layer_max = model
        .layer_bytes
        .as_ref()
        .and_then(|v| v.iter().copied().max())
        .unwrap_or(model.layer_param_bytes);
    let per_stage_layers = model.num_layers.div_ceil(num_stages) as u64;
    let per_stage_budget = model
        .embedding_param_bytes
        .saturating_add(model.lm_head_param_bytes)
        .saturating_add(per_layer_max.saturating_mul(per_stage_layers + 2));
    (0..num_stages)
        .map(|stage_index| DeviceSpec {
            device_id: format!("local-stage-{stage_index}"),
            available_memory_bytes: per_stage_budget,
            compute_units: 1,
        })
        .collect()
}

pub fn resolve_in_process_pipeline_num_layers(model_dir: &Path) -> Result<usize> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config_str = sanitize_config_json(&config_str);
    let config: serde_json::Value = serde_json::from_str(&config_str)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    let num_layers = config
        .get("num_hidden_layers")
        .and_then(|value| value.as_u64())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|text| text.get("num_hidden_layers"))
                .and_then(|value| value.as_u64())
        })
        .ok_or_else(|| {
            anyhow!(
                "config {} is missing an integer num_hidden_layers field (top-level or text_config.num_hidden_layers)",
                config_path.display()
            )
        })?;

    // All `num_hidden_layers` DeepSeek-V3 entries are real decoder layers.
    // Checkpoints that ship the trailing multi-token-prediction (MTP) head
    // store it at layer index `num_hidden_layers` (out of range, e.g.
    // `model.layers.61` for genuine DeepSeek-V3), and `sanitize_weights`
    // strips it, so the partitioner sees exactly `num_hidden_layers`
    // transformer blocks for every family.
    Ok(num_layers as usize)
}

pub fn resolve_in_process_stage_assignments(
    num_layers: usize,
    num_stages: Option<usize>,
    manual_spec: Option<&str>,
) -> Result<Vec<StageAssignment>> {
    let profile = equal_stage_model_profile(num_layers);
    resolve_stage_assignments_from_profile(&profile, num_stages, manual_spec).map(|(plan, _)| plan)
}

/// Resolve stage assignments for a concrete model directory. Unlike
/// [`resolve_in_process_stage_assignments`], this variant loads a
/// model-aware [`ModelProfile`] (real per-layer bytes for MoE, adjacency
/// constraints for KV-shared layers) so the auto-partitioner no longer
/// treats every layer as uniform. It also returns the partition quality
/// report, which callers surface via `--log-level debug` logging.
///
/// Used by: CLI pipeline generate, `mlxcel-server` startup
pub fn resolve_in_process_stage_assignments_for_model(
    model_dir: &Path,
    num_layers: usize,
    num_stages: Option<usize>,
    manual_spec: Option<&str>,
) -> Result<(Vec<StageAssignment>, PartitionQualityReport)> {
    let profile = build_model_profile(model_dir, num_layers).unwrap_or_else(|err| {
        // Config read or parse failed — fall back to the uniform profile
        // so we preserve the previous behaviour for callers that feed a
        // bare layer count (e.g. synthetic tests).
        tracing::debug!(
            model_dir = %model_dir.display(),
            error = %err,
            "pipeline auto-partition: falling back to uniform profile"
        );
        equal_stage_model_profile(num_layers)
    });
    // Ensure the profile's layer count matches the caller-visible layer
    // count (DeepSeek V3 stripping etc. may have reduced it).
    let profile = if profile.num_layers == num_layers {
        profile
    } else {
        let mut adjusted = profile.clone();
        adjusted.num_layers = num_layers;
        if let Some(v) = adjusted.layer_bytes.as_mut() {
            v.truncate(num_layers);
            while v.len() < num_layers {
                v.push(adjusted.layer_param_bytes);
            }
        }
        adjusted.adjacency.retain(|g| g.layers.end <= num_layers);
        adjusted
    };
    resolve_stage_assignments_from_profile(&profile, num_stages, manual_spec)
}

fn resolve_stage_assignments_from_profile(
    profile: &ModelProfile,
    num_stages: Option<usize>,
    manual_spec: Option<&str>,
) -> Result<(Vec<StageAssignment>, PartitionQualityReport)> {
    if let Some(spec) = manual_spec {
        ensure!(
            !spec.trim().is_empty(),
            "--pp-layers must not be empty when provided"
        );
        let ranges = parse_manual_partition(spec, profile.num_layers)?;
        if let Some(expected) = num_stages
            && expected > 1
        {
            ensure!(
                expected == ranges.len(),
                "--pp-size ({expected}) does not match manual partition stage count ({})",
                ranges.len()
            );
        }
        let devices = equal_capacity_devices(ranges.len(), profile);
        let plan = build_manual_assignments(&ranges, profile, &devices)?;
        let report = crate::distributed::pipeline::build_quality_report(profile, &plan);
        return Ok((plan, report));
    }

    let num_stages = num_stages.ok_or_else(|| {
        anyhow!("in-process pipeline execution requires either --pp-layers or --pp-size >= 2")
    })?;
    ensure!(
        num_stages >= 2,
        "in-process pipeline execution requires at least 2 stages"
    );
    let devices = equal_capacity_devices(num_stages, profile);
    auto_partition_with_report(profile, &devices)
}

/// Log the partition quality report at debug level. Callers on the CLI
/// generate and server startup paths should call this after resolving
/// assignments so operators can see the estimated per-stage memory and
/// any balancer warnings.
///
/// Used by: CLI pipeline generate, `mlxcel-server` startup
pub fn log_partition_quality(report: &PartitionQualityReport) {
    tracing::debug!("{}", format_quality_report(report));
}

pub fn load_in_process_stage_worker(
    model_dir: &Path,
    assignments: &[StageAssignment],
    micro_batch_size: usize,
) -> Result<InProcessStageWorkerLoop> {
    load_in_process_stage_worker_with_adapter(model_dir, assignments, micro_batch_size, None)
}

/// Worktree-aware variant that also plumbs a single-adapter LoRA path into
/// every stage's loader. Passing `None` reproduces the base-only path.
///
/// Used by: CLI pipeline generate, server pipeline runtime, tests
pub fn load_in_process_stage_worker_with_adapter(
    model_dir: &Path,
    assignments: &[StageAssignment],
    micro_batch_size: usize,
    adapter_path: Option<&Path>,
) -> Result<InProcessStageWorkerLoop> {
    ensure!(
        assignments.len() >= 2,
        "in-process pipeline execution requires at least 2 stages"
    );
    let executors: Vec<Box<dyn StageExecutor>> = assignments
        .iter()
        .map(|assignment| {
            LoadedStageExecutor::load_with_adapter(model_dir, assignment, adapter_path)
                .map(|executor| Box::new(executor) as Box<dyn StageExecutor>)
        })
        .collect::<Result<_>>()
        .with_context(|| {
            format!(
                "in-process pipeline execution is not available for model {}",
                model_dir.display()
            )
        })?;

    InProcessStageWorkerLoop::new(
        PipelineConfig::new(assignments.len() as u32, micro_batch_size)?,
        executors,
        ChannelConfig::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Minimal temp-dir helper: avoids pulling `tempfile` into the main
    /// crate's dependency tree for a two-test unit suite. The path is kept
    /// unique per-process and removed on drop.
    struct ScratchDir {
        path: std::path::PathBuf,
    }

    impl ScratchDir {
        fn new(tag: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("mlxcel-pp-{tag}-{pid}-{nanos}"));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_config(model_dir: &Path, content: &str) {
        let mut file = fs::File::create(model_dir.join("config.json")).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn num_layers_returns_raw_count_for_llama_family() {
        let dir = ScratchDir::new("llama-count");
        write_config(
            &dir.path,
            r#"{"model_type":"llama","num_hidden_layers":16,"hidden_size":128}"#,
        );
        let layers = resolve_in_process_pipeline_num_layers(&dir.path).unwrap();
        assert_eq!(layers, 16);
    }

    #[test]
    fn num_layers_counts_all_decoder_blocks_for_deepseek_v3() {
        // All `num_hidden_layers` entries are real decoder layers; the MTP
        // trailer, when present, lives at index `num_hidden_layers` (out of
        // range) and is stripped by `sanitize_weights`. The partitioner must
        // therefore see the raw config count (61 blocks for genuine
        // DeepSeek-V3, 27 for the Kimi-VL / Moonlight backbone), matching
        // `DeepSeekV3Model::num_layers()` (issue #525 round 3).
        let dir = ScratchDir::new("deepseek_v3-count");
        write_config(
            &dir.path,
            r#"{"model_type":"deepseek_v3","num_hidden_layers":61,"hidden_size":128}"#,
        );
        let layers = resolve_in_process_pipeline_num_layers(&dir.path).unwrap();
        assert_eq!(layers, 61);
    }
}
