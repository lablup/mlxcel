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

//! Stage-assignment assembly helpers for the pipeline partitioner.
//!
//! The core byte-balancing and adjacency-preserving algorithm lives in
//! [`super::partition_balance`]. This module glues its output to the
//! user-facing [`StageAssignment`] type, reserves embedding / lm_head
//! memory on the edge stages, runs the structural validators, and hands
//! the result to the partition-quality reporter.
//!
//! Used by: `super::partition::auto_partition`,
//! `super::partition::build_manual_assignments`

use std::ops::Range;

use anyhow::{Result, bail, ensure};

use super::partition::{
    DeviceSpec, ModelProfile, StageAssignment, validate_adjacency, validate_memory_fit,
    validate_partition,
};
use super::partition_balance::balance_layers;
use super::partition_quality::{
    PartitionQualityReport, build_quality_report_with_actuals, summarize_quality_warnings,
};

/// Run the constrained balancer against `model` and `devices`, then wrap
/// the output in [`StageAssignment`]s with embedding / lm_head reservations
/// on the edge stages.
///
/// Returns the assignments together with a [`PartitionQualityReport`] so
/// the caller can surface the estimated-vs-actual memory breakdown via
/// `mlxcel-server --log-level debug`.
pub fn assemble_auto_partition(
    model: &ModelProfile,
    devices: &[DeviceSpec],
) -> Result<(Vec<StageAssignment>, PartitionQualityReport)> {
    let n_devices = devices.len();
    let n_layers = model.num_layers;

    ensure!(n_devices > 0, "at least one device is required");
    ensure!(n_layers > 0, "model must have at least one layer");
    ensure!(
        n_devices <= n_layers,
        "more devices ({n_devices}) than layers ({n_layers}); \
         reduce the number of pipeline stages"
    );

    // Reserve embedding on stage 0 and lm_head on the last stage so the
    // per-stage budgets the balancer sees represent real layer capacity.
    let mut effective_memory: Vec<u64> = devices.iter().map(|d| d.available_memory_bytes).collect();

    if effective_memory[0] < model.embedding_param_bytes {
        bail!(
            "device '{}' has {} bytes but embedding alone requires {} bytes",
            devices[0].device_id,
            devices[0].available_memory_bytes,
            model.embedding_param_bytes
        );
    }
    effective_memory[0] -= model.embedding_param_bytes;

    let last_idx = n_devices - 1;
    if effective_memory[last_idx] < model.lm_head_param_bytes {
        bail!(
            "device '{}' has {} bytes but lm_head alone requires {} bytes",
            devices[last_idx].device_id,
            devices[last_idx].available_memory_bytes,
            model.lm_head_param_bytes
        );
    }
    effective_memory[last_idx] -= model.lm_head_param_bytes;

    let per_layer = model.effective_layer_bytes();
    let total_layer_bytes: u64 = per_layer.iter().copied().fold(0u64, u64::saturating_add);
    let total_effective: u64 = effective_memory
        .iter()
        .copied()
        .fold(0u64, u64::saturating_add);
    if total_effective < total_layer_bytes {
        bail!(
            "combined device memory ({total_effective} bytes after reservations) \
             is insufficient for {n_layers} layers ({total_layer_bytes} bytes)"
        );
    }

    let (layer_ranges, warnings) = balance_layers(
        &per_layer,
        &effective_memory,
        &model.forbidden_boundaries(),
        &model.adjacency,
        &device_ids(devices),
    )?;

    ensure!(
        layer_ranges.len() == n_devices,
        "balancer returned {} ranges for {} devices",
        layer_ranges.len(),
        n_devices
    );
    for (i, range) in layer_ranges.iter().enumerate() {
        if range.is_empty() {
            bail!(
                "balancer produced empty layer range for stage {} on device '{}'",
                i,
                devices[i].device_id
            );
        }
        let stage_layer_bytes: u64 = per_layer[range.clone()]
            .iter()
            .copied()
            .fold(0u64, u64::saturating_add);
        if stage_layer_bytes > effective_memory[i] {
            bail!(
                "stage {} on device '{}' would hold {} layer bytes but only {} \
                 remain after embedding/lm_head reservation",
                i,
                devices[i].device_id,
                stage_layer_bytes,
                effective_memory[i]
            );
        }
    }

    let assignments = build_assignments(&layer_ranges, model, devices, &per_layer);

    validate_partition(&assignments, n_layers)?;
    validate_memory_fit(&assignments, devices)?;

    let report = build_quality_report_with_actuals(model, &assignments, None, &warnings);
    for warning in summarize_quality_warnings(&report) {
        tracing::warn!("{warning}");
    }

    Ok((assignments, report))
}

/// Manual counterpart of [`assemble_auto_partition`]. Accepts a pre-made
/// range list, validates adjacency, checks memory fit, and returns the
/// stage assignments.
pub fn assemble_manual_partition(
    ranges: &[Range<usize>],
    model: &ModelProfile,
    devices: &[DeviceSpec],
) -> Result<Vec<StageAssignment>> {
    ensure!(
        ranges.len() == devices.len(),
        "manual partition has {} ranges but {} devices were provided",
        ranges.len(),
        devices.len()
    );

    // Adjacency is a hard invariant for the model, not a style preference —
    // reject manual plans that would break it early, before any weights
    // get loaded onto the wrong device.
    validate_adjacency(ranges, &model.adjacency)?;

    let per_layer = model.effective_layer_bytes();
    let assignments = build_assignments(ranges, model, devices, &per_layer);

    validate_partition(&assignments, model.num_layers)?;
    validate_memory_fit(&assignments, devices)?;

    Ok(assignments)
}

fn build_assignments(
    ranges: &[Range<usize>],
    model: &ModelProfile,
    devices: &[DeviceSpec],
    per_layer: &[u64],
) -> Vec<StageAssignment> {
    let n_stages = ranges.len();
    let mut assignments = Vec::with_capacity(n_stages);
    for (i, range) in ranges.iter().enumerate() {
        let is_first = i == 0;
        let is_last = i + 1 == n_stages;

        let mut mem: u64 = per_layer
            .get(range.clone())
            .map(|s| s.iter().copied().fold(0u64, u64::saturating_add))
            .unwrap_or(0);
        if is_first {
            mem = mem.saturating_add(model.embedding_param_bytes);
        }
        if is_last {
            mem = mem.saturating_add(model.lm_head_param_bytes);
        }

        assignments.push(StageAssignment {
            stage_index: i,
            device_id: devices[i].device_id.clone(),
            layer_range: range.clone(),
            has_embedding: is_first,
            has_lm_head: is_last,
            estimated_memory_bytes: mem,
        });
    }
    assignments
}

fn device_ids(devices: &[DeviceSpec]) -> Vec<String> {
    devices.iter().map(|d| d.device_id.clone()).collect()
}
