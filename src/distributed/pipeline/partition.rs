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

//! Core layer partitioning algorithm for pipeline parallelism.
//!
//! Given a [`ModelProfile`] (layer count, per-layer parameter cost, embedding
//! and lm_head sizes, optional per-layer byte weights, optional layer
//! adjacency constraints) and a set of [`DeviceSpec`]s (available memory,
//! compute units), this module produces a vector of [`StageAssignment`]s that
//! map contiguous layer ranges to devices.
//!
//! The auto-partitioner minimises the maximum per-stage byte load across
//! stages, subject to:
//!
//! - Every device gets at least one layer.
//! - Every device's assigned bytes fit inside its `available_memory_bytes`
//!   budget (after embedding / lm_head reservations).
//! - No constrained layer group (e.g. Gemma 4 KV-shared source/consumer
//!   pairs) is split across a stage boundary.
//!
//! Manual partitions are parsed from the `--pp-layers` CLI flag and
//! validated for correctness.
//!
//! Used by: server startup, model loading pipeline, CLI pipeline generate

use std::ops::Range;

use anyhow::{Result, bail, ensure};

use super::partition_assembly::{assemble_auto_partition, assemble_manual_partition};
use super::partition_quality::PartitionQualityReport;

/// Describes a contiguous range of layers that must stay on the same pipeline
/// stage.
///
/// The partitioner uses these to express layer-to-layer coupling that cannot
/// be broken by a stage boundary. The canonical example is Gemma 4's KV
/// sharing: the last `num_kv_shared_layers` decoder blocks read their keys
/// and values from an earlier source layer, so a stage split between source
/// and consumer would strand the cache on the wrong device.
///
/// Ranges are half-open (`start..end`, `end` exclusive) and must be
/// non-empty. Multiple overlapping or adjacent groups are allowed and get
/// merged internally.
///
/// Used by: `ModelProfile`, `auto_partition`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LayerAdjacencyGroup {
    /// Half-open range of layers that must land on the same stage.
    pub layers: Range<usize>,
    /// Human-readable reason, surfaced in warnings when the constraint
    /// forces an imbalanced plan. Example:
    /// `"gemma4 KV-shared layers 12..24 share keys with sources in 0..12"`.
    pub reason: String,
}

/// Describes the memory footprint of a model for partitioning purposes.
///
/// All sizes are in bytes. The partitioner uses these to estimate per-stage
/// memory requirements and balance the assignment across devices.
///
/// Two knobs on top of the uniform-layer baseline make the partitioner
/// accurate for MoE and KV-shared architectures:
///
/// - `layer_bytes`: per-layer parameter byte weight. When set, it overrides
///   the uniform `layer_param_bytes` assumption. Use this for MoE models
///   where expert layers are much heavier than dense layers, or Gemma 4
///   where KV-shared consumer layers carry double-wide MLPs.
/// - `adjacency`: groups of layers that must stay on the same stage. Use
///   this for KV-shared layer pairs (source layer must live with its
///   consumer), Jamba's Mamba+Transformer interleaving boundary invariants,
///   or any other cross-layer state dependency that cannot be serialised
///   over the activation channel.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelProfile {
    /// Total number of transformer/SSM layers in the model.
    pub num_layers: usize,
    /// Fallback parameter memory per layer (bytes) used when `layer_bytes`
    /// is `None`. For MoE models without an explicit per-layer vector,
    /// callers should use the average or worst-case layer size.
    pub layer_param_bytes: u64,
    /// Memory consumed by the embedding table (bytes). Assigned to the
    /// first pipeline stage.
    pub embedding_param_bytes: u64,
    /// Memory consumed by the lm_head / output projection (bytes). Assigned
    /// to the last pipeline stage.
    pub lm_head_param_bytes: u64,
    /// Optional per-layer parameter byte weights. When `Some(v)`, `v.len()`
    /// must equal `num_layers` and each entry is the layer's real byte cost
    /// (experts included for MoE, double-wide MLP included for Gemma 4
    /// KV-shared consumers). When `None`, the partitioner falls back to the
    /// uniform `layer_param_bytes` assumption.
    pub layer_bytes: Option<Vec<u64>>,
    /// Layer-adjacency constraints that the partitioner must not cut
    /// across. Empty means no constraints.
    pub adjacency: Vec<LayerAdjacencyGroup>,
}

impl ModelProfile {
    /// Construct a minimal profile that uses the uniform `layer_param_bytes`
    /// assumption and carries no adjacency constraints.
    ///
    /// This is the legacy shape — prefer it only for smoke tests and
    /// backward-compatible callers. Real model loading paths should go
    /// through [`super::partition_profile::build_model_profile`].
    pub fn uniform(
        num_layers: usize,
        layer_param_bytes: u64,
        embedding_param_bytes: u64,
        lm_head_param_bytes: u64,
    ) -> Self {
        Self {
            num_layers,
            layer_param_bytes,
            embedding_param_bytes,
            lm_head_param_bytes,
            layer_bytes: None,
            adjacency: Vec::new(),
        }
    }

    /// Byte cost of layer `idx`, honouring the per-layer override if set.
    pub fn layer_bytes_at(&self, idx: usize) -> u64 {
        debug_assert!(idx < self.num_layers);
        match self.layer_bytes.as_ref() {
            Some(vec) if idx < vec.len() => vec[idx],
            _ => self.layer_param_bytes,
        }
    }

    /// Materialise the per-layer byte cost vector. Always returns a vector
    /// of length `num_layers` — convenient for the DP partitioner without
    /// forcing every caller to memoise the fallback path.
    pub fn effective_layer_bytes(&self) -> Vec<u64> {
        (0..self.num_layers)
            .map(|i| self.layer_bytes_at(i))
            .collect()
    }

    /// Total model parameter memory (embedding + all layers + lm_head).
    pub fn total_param_bytes(&self) -> u64 {
        let layer_total = self
            .effective_layer_bytes()
            .iter()
            .copied()
            .fold(0u64, |acc, b| acc.saturating_add(b));
        self.embedding_param_bytes
            .saturating_add(layer_total)
            .saturating_add(self.lm_head_param_bytes)
    }

    /// Returns the sorted set of boundary indices `b` (meaning "split
    /// before layer `b`") that are forbidden because they would separate
    /// a [`LayerAdjacencyGroup`] across stages.
    ///
    /// A group covering `start..end` forbids boundaries at every
    /// `start+1..end`. Boundary 0 and `num_layers` are structural, never
    /// user-controlled, and are not returned here.
    pub fn forbidden_boundaries(&self) -> Vec<usize> {
        use std::collections::BTreeSet;
        let mut set = BTreeSet::new();
        for group in &self.adjacency {
            if group.layers.start >= group.layers.end {
                continue;
            }
            let interior_start = group.layers.start.saturating_add(1);
            let interior_end = group.layers.end;
            for b in interior_start..interior_end {
                if b > 0 && b < self.num_layers {
                    set.insert(b);
                }
            }
        }
        set.into_iter().collect()
    }
}

/// Describes the capabilities and constraints of a single device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceSpec {
    /// Unique device identifier (matches node ID in cluster config).
    pub device_id: String,
    /// Available memory in bytes for model parameters on this device.
    pub available_memory_bytes: u64,
    /// Number of compute units (GPU cores, neural engine cores, etc.).
    /// Used as a secondary signal for balancing; 0 means unknown.
    pub compute_units: u32,
}

/// A single pipeline stage assignment produced by the partitioner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StageAssignment {
    /// Index of this stage (0-based, ascending).
    pub stage_index: usize,
    /// Device this stage is assigned to.
    pub device_id: String,
    /// Contiguous range of layer indices assigned to this stage.
    /// Uses half-open range: `start..end` where `end` is exclusive.
    pub layer_range: Range<usize>,
    /// Whether this stage hosts the embedding table.
    pub has_embedding: bool,
    /// Whether this stage hosts the lm_head / output projection.
    pub has_lm_head: bool,
    /// Estimated total memory consumption for this stage (bytes).
    pub estimated_memory_bytes: u64,
}

/// Partition configuration: auto or manual.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PartitionConfig {
    /// Automatically distribute layers proportionally to device memory.
    #[default]
    Auto,
    /// Manually specified layer ranges (one per device, in device order).
    Manual(Vec<Range<usize>>),
}

/// Parse a manual partition specification string into layer ranges.
///
/// Format: comma-separated `start-end` pairs (inclusive on both ends),
/// e.g., `"0-15,16-31"` for a 32-layer model split across 2 devices.
///
/// Returns one `Range<usize>` per stage, using half-open ranges internally.
pub fn parse_manual_partition(spec: &str, num_layers: usize) -> Result<Vec<Range<usize>>> {
    let spec = spec.trim();
    ensure!(!spec.is_empty(), "partition spec must not be empty");

    let mut ranges = Vec::new();

    for (i, part) in spec.split(',').enumerate() {
        let part = part.trim();
        let dash_pos = part.find('-').ok_or_else(|| {
            anyhow::anyhow!("invalid range at position {i}: '{part}' (expected start-end)")
        })?;

        let start_str = &part[..dash_pos];
        let end_str = &part[dash_pos + 1..];

        let start: usize = start_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid start index in range '{part}'"))?;
        let end_inclusive: usize = end_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid end index in range '{part}'"))?;

        ensure!(
            start <= end_inclusive,
            "invalid range '{part}': start ({start}) > end ({end_inclusive})"
        );
        ensure!(
            end_inclusive < num_layers,
            "layer index {end_inclusive} exceeds model layer count ({num_layers})"
        );

        ranges.push(start..end_inclusive + 1);
    }

    Ok(ranges)
}

/// Validate a set of stage assignments for correctness.
///
/// Checks:
/// - All layers `0..num_layers` are covered exactly once (no gaps, no overlaps)
/// - Ranges are contiguous and sorted
/// - Embedding is on the first stage, lm_head on the last stage
/// - No stage exceeds its device memory (when `devices` is provided)
pub fn validate_partition(assignments: &[StageAssignment], num_layers: usize) -> Result<()> {
    ensure!(
        !assignments.is_empty(),
        "partition must contain at least one stage"
    );

    // Check stage indices are sequential 0..N.
    for (i, a) in assignments.iter().enumerate() {
        ensure!(
            a.stage_index == i,
            "stage index mismatch: expected {i}, got {}",
            a.stage_index
        );
    }

    // Check layer coverage: ranges must tile 0..num_layers exactly.
    let mut expected_start = 0;
    for a in assignments {
        ensure!(
            a.layer_range.start == expected_start,
            "gap or overlap at layer {}: stage {} starts at {} but expected {}",
            expected_start,
            a.stage_index,
            a.layer_range.start,
            expected_start
        );
        ensure!(
            a.layer_range.end > a.layer_range.start,
            "empty range on stage {}",
            a.stage_index
        );
        expected_start = a.layer_range.end;
    }
    ensure!(
        expected_start == num_layers,
        "partition covers layers 0..{expected_start} but model has {num_layers} layers"
    );

    // Check embedding placement.
    ensure!(
        assignments[0].has_embedding,
        "first stage must host the embedding table"
    );
    for a in &assignments[1..] {
        ensure!(
            !a.has_embedding,
            "only the first stage should host the embedding table (stage {} has it)",
            a.stage_index
        );
    }

    // Check lm_head placement.
    let last = assignments.last().unwrap();
    ensure!(last.has_lm_head, "last stage must host the lm_head");
    for a in &assignments[..assignments.len() - 1] {
        ensure!(
            !a.has_lm_head,
            "only the last stage should host lm_head (stage {} has it)",
            a.stage_index
        );
    }

    Ok(())
}

/// Validate that each stage fits within its device's memory.
///
/// Separated from [`validate_partition`] so callers can validate structural
/// correctness without requiring device specs.
pub fn validate_memory_fit(assignments: &[StageAssignment], devices: &[DeviceSpec]) -> Result<()> {
    ensure!(
        assignments.len() == devices.len(),
        "partition has {} stages but {} devices were provided",
        assignments.len(),
        devices.len()
    );

    for (a, d) in assignments.iter().zip(devices.iter()) {
        ensure!(
            a.estimated_memory_bytes <= d.available_memory_bytes,
            "stage {} on device '{}' requires {} bytes but only {} bytes available",
            a.stage_index,
            d.device_id,
            a.estimated_memory_bytes,
            d.available_memory_bytes
        );
    }

    Ok(())
}

/// Validate that a set of layer ranges does not cut any adjacency group.
///
/// Returns `Err` with a human-readable message naming the first violated
/// group. The message includes the group's `reason` so operators can see
/// which invariant they broke (e.g. `"gemma4 KV-shared layers 12..24 share
/// keys with sources in 0..12"`).
pub fn validate_adjacency(
    ranges: &[Range<usize>],
    adjacency: &[LayerAdjacencyGroup],
) -> Result<()> {
    for group in adjacency {
        if group.layers.start >= group.layers.end {
            continue;
        }
        for range in ranges {
            let starts_inside = range.start > group.layers.start && range.start < group.layers.end;
            if starts_inside {
                bail!(
                    "manual partition splits adjacency group {}..{} (reason: {})",
                    group.layers.start,
                    group.layers.end,
                    group.reason
                );
            }
        }
    }
    Ok(())
}

/// Automatically partition model layers across devices.
///
/// The algorithm minimises the maximum per-stage byte load across stages,
/// subject to:
///
/// 1. The embedding lives on stage 0 and the lm_head on the last stage.
/// 2. Every device receives at least one layer.
/// 3. Every device's assigned bytes fit inside its `available_memory_bytes`
///    (after embedding / lm_head reservations).
/// 4. No adjacency group is split across a stage boundary.
///
/// The per-layer byte cost is taken from `model.layer_bytes` when set, and
/// otherwise falls back to the uniform `layer_param_bytes`.
///
/// Returns an error if the model cannot fit in the combined device memory
/// or if the constraints are infeasible.
pub fn auto_partition(
    model: &ModelProfile,
    devices: &[DeviceSpec],
) -> Result<Vec<StageAssignment>> {
    auto_partition_with_report(model, devices).map(|(plan, _)| plan)
}

/// Variant of [`auto_partition`] that also returns a partition-quality
/// report. The report surfaces per-stage estimated byte sums, the overall
/// imbalance ratio, and any warnings produced while honouring the
/// constraints.
///
/// Used by: CLI pipeline generate (debug logging), server startup (debug
/// logging)
pub fn auto_partition_with_report(
    model: &ModelProfile,
    devices: &[DeviceSpec],
) -> Result<(Vec<StageAssignment>, PartitionQualityReport)> {
    assemble_auto_partition(model, devices)
}

/// Build stage assignments from manually specified layer ranges.
///
/// Pairs each range with the corresponding device (by index), sets embedding
/// on the first stage and lm_head on the last, computes memory estimates,
/// validates the result, and rejects any manual plan that splits an
/// adjacency group.
pub fn build_manual_assignments(
    ranges: &[Range<usize>],
    model: &ModelProfile,
    devices: &[DeviceSpec],
) -> Result<Vec<StageAssignment>> {
    assemble_manual_partition(ranges, model, devices)
}

#[cfg(test)]
#[path = "partition_tests.rs"]
mod tests;
