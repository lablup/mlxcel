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

//! Partition-quality reporting for pipeline parallelism.
//!
//! Surfaces the per-stage estimated memory produced by `auto_partition()`,
//! and optionally compares it against measured per-stage memory after the
//! stage executors have finished loading. The report is intended for
//! `mlxcel-server --log-level debug` so operators can see when the
//! estimator drifts from reality.
//!
//! Used by: `auto_partition_with_report`, CLI pipeline generate (debug
//! logging), server pipeline startup (debug logging)

use super::partition::{ModelProfile, StageAssignment};

/// Quality summary for a single pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageQuality {
    pub stage_index: usize,
    pub device_id: String,
    pub layer_count: usize,
    pub estimated_bytes: u64,
    /// Measured bytes after the stage executor finishes loading. `None` at
    /// partition time; populated by
    /// [`populate_actual_memory`] when loading completes.
    pub actual_bytes: Option<u64>,
}

impl StageQuality {
    /// Drift ratio (actual / estimated) as percent. Returns `None` when
    /// either value is zero or `actual_bytes` is not yet populated.
    pub fn drift_percent(&self) -> Option<u64> {
        let actual = self.actual_bytes?;
        if self.estimated_bytes == 0 || actual == 0 {
            return None;
        }
        Some(actual.saturating_mul(100) / self.estimated_bytes)
    }
}

/// Full partition-quality report produced by [`build_quality_report`] and
/// extended by [`auto_partition_with_report`] with any warnings emitted by
/// the balancer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionQualityReport {
    pub per_stage: Vec<StageQuality>,
    pub max_stage_estimated_bytes: u64,
    pub min_stage_estimated_bytes: u64,
    /// `max / min`, expressed as a percentage where 100 means perfectly
    /// balanced. 150 means the heaviest stage carries 1.5x the lightest.
    pub imbalance_pct: u64,
    pub warnings: Vec<String>,
}

/// Build a quality report from the post-partition stage list.
///
/// At partition time `actual_bytes_per_stage` is `None` because the stages
/// have not loaded yet. The pipeline startup path calls
/// [`populate_actual_memory`] with measured values once every stage is up.
pub fn build_quality_report(
    _model: &ModelProfile,
    assignments: &[StageAssignment],
) -> PartitionQualityReport {
    build_quality_report_with_actuals(_model, assignments, None, &[])
}

/// Same as [`build_quality_report`] but allows the caller to inject the
/// measured byte counts (one per stage, same order) and balancer warnings.
pub fn build_quality_report_with_actuals(
    _model: &ModelProfile,
    assignments: &[StageAssignment],
    actual_bytes_per_stage: Option<&[u64]>,
    warnings: &[String],
) -> PartitionQualityReport {
    let mut per_stage = Vec::with_capacity(assignments.len());
    for (i, a) in assignments.iter().enumerate() {
        let actual = actual_bytes_per_stage.and_then(|v| v.get(i)).copied();
        per_stage.push(StageQuality {
            stage_index: a.stage_index,
            device_id: a.device_id.clone(),
            layer_count: a.layer_range.end.saturating_sub(a.layer_range.start),
            estimated_bytes: a.estimated_memory_bytes,
            actual_bytes: actual,
        });
    }
    let max_val = per_stage
        .iter()
        .map(|s| s.estimated_bytes)
        .max()
        .unwrap_or(0);
    let min_val = per_stage
        .iter()
        .map(|s| s.estimated_bytes)
        .min()
        .unwrap_or(0);
    let imbalance_pct = max_val
        .saturating_mul(100)
        .checked_div(min_val)
        .unwrap_or(0);
    PartitionQualityReport {
        per_stage,
        max_stage_estimated_bytes: max_val,
        min_stage_estimated_bytes: min_val,
        imbalance_pct,
        warnings: warnings.to_vec(),
    }
}

/// Overwrite the `actual_bytes` field on every stage quality entry using
/// the supplied measurement vector. Callers pass one value per stage, in
/// stage-index order.
pub fn populate_actual_memory(report: &mut PartitionQualityReport, actual_bytes_per_stage: &[u64]) {
    for (i, stage) in report.per_stage.iter_mut().enumerate() {
        stage.actual_bytes = actual_bytes_per_stage.get(i).copied();
    }
}

/// Render the report as a single multi-line string suitable for a debug
/// log line. Keeps every field on one row so the log is grep-friendly.
///
/// Used by: server startup debug logging, CLI pipeline debug logging
pub fn format_quality_report(report: &PartitionQualityReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "pipeline partition quality: {} stages, max={} bytes, min={} bytes, imbalance={}.{}x\n",
        report.per_stage.len(),
        report.max_stage_estimated_bytes,
        report.min_stage_estimated_bytes,
        report.imbalance_pct / 100,
        report.imbalance_pct % 100,
    ));
    for stage in &report.per_stage {
        let actual = match stage.actual_bytes {
            Some(v) => format!("actual={v}"),
            None => "actual=unknown".to_string(),
        };
        let drift = match stage.drift_percent() {
            Some(pct) => format!(" drift={}.{}x", pct / 100, pct % 100),
            None => String::new(),
        };
        out.push_str(&format!(
            "  stage {} device '{}' layers={} estimated={} {}{}\n",
            stage.stage_index,
            stage.device_id,
            stage.layer_count,
            stage.estimated_bytes,
            actual,
            drift,
        ));
    }
    if !report.warnings.is_empty() {
        out.push_str("  warnings:\n");
        for w in &report.warnings {
            out.push_str("    - ");
            out.push_str(w);
            out.push('\n');
        }
    }
    out
}

/// Return every warning plus a standard imbalance warning if the report
/// shows the max/min ratio exceeds the threshold. Used by
/// `auto_partition_with_report` to surface warnings through the logger.
pub fn summarize_quality_warnings(report: &PartitionQualityReport) -> Vec<String> {
    // We only re-emit the balancer-supplied warnings verbatim. The report
    // already captures imbalance in `imbalance_pct`; duplicating a
    // "significantly imbalanced" message here would double-log the same
    // scenario when `partition_balance::balance_layers` already handled it.
    report.warnings.clone()
}

#[cfg(test)]
#[path = "partition_quality_tests.rs"]
mod tests;
