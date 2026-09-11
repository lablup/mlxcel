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

//! Per-rank sharded weight loading for tensor parallelism.
//!
//! Given a [`ModelShardPlan`] and the current rank, this module computes which
//! byte ranges of each weight tensor belong to this rank and extracts them from
//! the raw serialized data (e.g., safetensors mmap'd buffer).
//!
//! Key types:
//! - [`ShardSpec`] — describes a single rank's slice of a weight tensor
//! - [`ShardedMemoryReport`] — memory accounting across all ranks
//!
//! Key functions:
//! - [`compute_shard_spec`] — determine shard boundaries for a weight on a rank
//! - [`shard_tensor_data`] — extract the shard's bytes from contiguous raw data
//! - [`compute_sharded_shape`] — compute the shape of the shard
//! - [`validate_sharded_memory`] — verify total memory is consistent
//!
//! Used by: model loading pipeline (TP weight distribution)

use std::collections::HashMap;

use anyhow::{Result, ensure};

use super::shard_strategy::{ModelShardPlan, ShardStrategy};

/// Describes a single rank's slice of a weight tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSpec {
    /// This rank's index (0-based).
    pub rank: usize,
    /// Total number of TP ranks.
    pub tp_size: usize,
    /// Which axis to shard along (0 = rows, 1 = columns for 2D).
    pub shard_axis: usize,
    /// Start index (inclusive) along the shard axis.
    pub start_index: usize,
    /// End index (exclusive) along the shard axis.
    pub end_index: usize,
    /// Whether this shard is padded to fill a non-divisible remainder.
    pub padded: bool,
    /// Number of padding elements appended along the shard axis (0 if not padded).
    pub pad_count: usize,
    /// Sharding strategy that produced this spec.
    pub strategy: ShardStrategy,
}

impl ShardSpec {
    /// Number of elements this shard covers along the shard axis.
    pub fn shard_size(&self) -> usize {
        self.end_index - self.start_index
    }

    /// True if this weight is fully replicated (not sharded).
    pub fn is_replicated(&self) -> bool {
        self.strategy == ShardStrategy::Replicated
    }
}

/// Compute the checked product of a shape slice, returning an error on overflow.
fn checked_product(dims: &[usize]) -> Result<usize> {
    let mut product: usize = 1;
    for &d in dims {
        product = product
            .checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("overflow in shape product: {dims:?}"))?;
    }
    Ok(product)
}

/// Compute the shard specification for a weight tensor on a given rank.
///
/// For weights not found in the plan, the default is [`ShardStrategy::Replicated`]
/// (full tensor loaded on every rank).
///
/// # Arguments
/// * `weight_name` — fully qualified weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
/// * `shape` — original tensor shape
/// * `plan` — the model's shard plan
/// * `rank` — this rank's index (0..tp_size)
///
/// # Returns
/// A [`ShardSpec`] describing the byte range this rank should load.
pub fn compute_shard_spec(
    weight_name: &str,
    shape: &[usize],
    plan: &ModelShardPlan,
    rank: usize,
) -> Result<ShardSpec> {
    ensure!(!shape.is_empty(), "weight '{weight_name}' has empty shape");
    ensure!(
        rank < plan.tp_size,
        "rank {rank} out of range for tp_size {}",
        plan.tp_size
    );

    // Look up the shard plan for this weight.
    let layer_plan = plan.plan_for_weight(weight_name);

    // Check special weights: embedding and lm_head.
    let (strategy, shard_axis) = if let Some(lp) = layer_plan {
        (lp.strategy, lp.shard_axis)
    } else if weight_name == "model.embed_tokens.weight" || weight_name == "embed_tokens.weight" {
        match plan.embedding_strategy {
            ShardStrategy::VocabParallel => (ShardStrategy::VocabParallel, 0),
            _ => (ShardStrategy::Replicated, 0),
        }
    } else if weight_name == "lm_head.weight" {
        match plan.lm_head_strategy {
            ShardStrategy::VocabParallel => (ShardStrategy::VocabParallel, 0),
            _ => (ShardStrategy::Replicated, 0),
        }
    } else {
        // Default: replicate weights not in the plan (LayerNorm, biases, etc.)
        (ShardStrategy::Replicated, 0)
    };

    match strategy {
        ShardStrategy::Replicated => Ok(ShardSpec {
            rank,
            tp_size: plan.tp_size,
            shard_axis: 0,
            start_index: 0,
            end_index: shape.first().copied().unwrap_or(0),
            padded: false,
            pad_count: 0,
            strategy: ShardStrategy::Replicated,
        }),

        ShardStrategy::ColumnParallel
        | ShardStrategy::RowParallel
        | ShardStrategy::VocabParallel => {
            ensure!(
                shard_axis < shape.len(),
                "shard_axis {shard_axis} out of range for shape {shape:?} on '{weight_name}'"
            );
            let dim = shape[shard_axis];
            let (start, end, padded, pad_count) = compute_shard_range(dim, plan.tp_size, rank);
            Ok(ShardSpec {
                rank,
                tp_size: plan.tp_size,
                shard_axis,
                start_index: start,
                end_index: end,
                padded,
                pad_count,
                strategy,
            })
        }

        ShardStrategy::ExpertParallel => {
            // For expert-parallel, the shard axis is the expert dimension (axis 0
            // of the fused expert tensor). Each rank gets experts round-robin.
            // We compute a contiguous range for simplicity (expert `rank`, `rank + tp_size`, ...).
            // The actual expert dispatch handles the mapping.
            ensure!(
                !shape.is_empty(),
                "expert-parallel weight '{weight_name}' has empty shape"
            );
            let num_experts = shape[0];
            let (start, end, padded, pad_count) =
                compute_shard_range(num_experts, plan.tp_size, rank);
            Ok(ShardSpec {
                rank,
                tp_size: plan.tp_size,
                shard_axis: 0,
                start_index: start,
                end_index: end,
                padded,
                pad_count,
                strategy: ShardStrategy::ExpertParallel,
            })
        }
    }
}

/// Compute start/end indices for a dimension that may not divide evenly.
///
/// Uses "remainder distribution" strategy: the first `remainder` ranks each
/// get one extra element, so shard sizes differ by at most 1.
///
/// Returns `(start, end, padded, pad_count)`.
fn compute_shard_range(dim: usize, tp_size: usize, rank: usize) -> (usize, usize, bool, usize) {
    if tp_size == 1 {
        return (0, dim, false, 0);
    }

    let base_size = dim / tp_size;
    let remainder = dim % tp_size;

    // First `remainder` ranks get `base_size + 1` elements.
    let start = if rank < remainder {
        rank * (base_size + 1)
    } else {
        remainder * (base_size + 1) + (rank - remainder) * base_size
    };

    let shard_size = if rank < remainder {
        base_size + 1
    } else {
        base_size
    };

    let end = start + shard_size;
    let padded = shard_size == 0;
    let pad_count = 0; // No zero-padding needed with remainder distribution.

    (start, end, padded, pad_count)
}

/// Compute the shape of a shard given the original shape and shard spec.
///
/// The shard axis dimension is replaced by `spec.shard_size()`.
/// For replicated weights, the original shape is returned unchanged.
pub fn compute_sharded_shape(original_shape: &[usize], spec: &ShardSpec) -> Vec<usize> {
    if spec.is_replicated() {
        return original_shape.to_vec();
    }

    let mut sharded = original_shape.to_vec();
    if spec.shard_axis < sharded.len() {
        sharded[spec.shard_axis] = spec.shard_size();
    }
    sharded
}

/// Extract a shard's bytes from contiguous row-major tensor data.
///
/// This function computes the byte ranges for the shard and copies them into a
/// new `Vec<u8>`. For axis-0 sharding, this is a single contiguous slice. For
/// higher-axis sharding, it extracts strided slices.
///
/// # Arguments
/// * `data` — raw tensor bytes in row-major order
/// * `shape` — original tensor shape
/// * `dtype_size` — bytes per element (e.g., 2 for float16, 4 for float32)
/// * `spec` — the shard specification from [`compute_shard_spec`]
///
/// # Returns
/// The extracted shard bytes. Length equals `product(sharded_shape) * dtype_size`.
pub fn shard_tensor_data(
    data: &[u8],
    shape: &[usize],
    dtype_size: usize,
    spec: &ShardSpec,
) -> Result<Vec<u8>> {
    ensure!(dtype_size > 0, "dtype_size must be > 0");

    if spec.is_replicated() {
        return Ok(data.to_vec());
    }

    let axis = spec.shard_axis;
    ensure!(
        axis < shape.len(),
        "shard_axis {axis} out of range for shape {shape:?}"
    );

    let total_elements: usize = checked_product(shape)?;
    let expected_bytes = total_elements
        .checked_mul(dtype_size)
        .ok_or_else(|| anyhow::anyhow!("overflow computing expected bytes for shape {shape:?}"))?;
    ensure!(
        data.len() >= expected_bytes,
        "data length {} < expected {expected_bytes} for shape {shape:?} with dtype_size {dtype_size}",
        data.len()
    );

    let sharded_shape = compute_sharded_shape(shape, spec);
    let shard_elements: usize = checked_product(&sharded_shape)?;
    let shard_bytes = shard_elements
        .checked_mul(dtype_size)
        .ok_or_else(|| anyhow::anyhow!("overflow computing shard bytes"))?;
    let mut output = vec![0u8; shard_bytes];

    if axis == 0 {
        // Axis-0 sharding: contiguous byte range.
        let elements_per_row: usize = checked_product(&shape[1..])?;
        let start_byte = spec
            .start_index
            .checked_mul(elements_per_row)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing start_byte"))?;
        let end_byte = spec
            .end_index
            .checked_mul(elements_per_row)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing end_byte"))?;
        ensure!(
            end_byte <= data.len(),
            "axis-0 shard end_byte {end_byte} exceeds data length {}",
            data.len()
        );
        let slice = &data[start_byte..end_byte];
        output[..slice.len()].copy_from_slice(slice);
    } else {
        // Higher-axis sharding: need to extract strided slices.
        // Compute the stride pattern.
        //
        // For a tensor with shape [D0, D1, ..., Dn] sharded on axis `a`:
        // - outer_count = product(D0..Da)   (number of "rows" before the shard axis)
        // - inner_count = product(D(a+1)..Dn) (elements per position on shard axis)
        // - full_stride = Da * inner_count  (bytes between consecutive outer positions)

        let outer_count: usize = checked_product(&shape[..axis])?;
        let inner_count: usize = checked_product(&shape[(axis + 1)..])?;
        let full_axis_stride = shape[axis]
            .checked_mul(inner_count)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing full_axis_stride"))?;
        let shard_inner_bytes = spec
            .shard_size()
            .checked_mul(inner_count)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing shard_inner_bytes"))?;
        let shard_start_offset = spec
            .start_index
            .checked_mul(inner_count)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing shard_start_offset"))?;

        for outer_idx in 0..outer_count {
            let src_base = outer_idx
                .checked_mul(full_axis_stride)
                .and_then(|v| v.checked_add(shard_start_offset))
                .ok_or_else(|| {
                    anyhow::anyhow!("overflow computing src_base at outer_idx {outer_idx}")
                })?;
            let src_end = src_base
                .checked_add(shard_inner_bytes)
                .ok_or_else(|| anyhow::anyhow!("overflow computing src_end"))?;
            ensure!(
                src_end <= data.len(),
                "strided shard read [{src_base}..{src_end}] exceeds data length {}",
                data.len()
            );
            let dst_base = outer_idx
                .checked_mul(shard_inner_bytes)
                .ok_or_else(|| anyhow::anyhow!("overflow computing dst_base"))?;
            output[dst_base..dst_base + shard_inner_bytes]
                .copy_from_slice(&data[src_base..src_end]);
        }
    }

    Ok(output)
}

/// Byte-range specification for reading a shard directly from a file.
///
/// Used with safetensors mmap to avoid loading the full tensor into memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteRangeSpec {
    /// Byte ranges to read from the file (relative to tensor data start).
    /// For axis-0 sharding, this is a single range. For higher axes, multiple
    /// strided ranges.
    pub ranges: Vec<(usize, usize)>,
    /// Total bytes in the shard.
    pub total_bytes: usize,
}

/// Compute the byte ranges needed to read a shard from a file.
///
/// This enables efficient partial reads from safetensors files without loading
/// the entire tensor into memory first.
pub fn compute_byte_ranges(
    shape: &[usize],
    dtype_size: usize,
    spec: &ShardSpec,
) -> Result<ByteRangeSpec> {
    ensure!(dtype_size > 0, "dtype_size must be > 0");

    if spec.is_replicated() {
        let total_elements: usize = checked_product(shape)?;
        let total = total_elements
            .checked_mul(dtype_size)
            .ok_or_else(|| anyhow::anyhow!("overflow computing total bytes for shape {shape:?}"))?;
        return Ok(ByteRangeSpec {
            ranges: vec![(0, total)],
            total_bytes: total,
        });
    }

    let axis = spec.shard_axis;
    ensure!(
        axis < shape.len(),
        "shard_axis {} out of range for shape {shape:?}",
        axis
    );

    if axis == 0 {
        // Contiguous range.
        let elements_per_row: usize = checked_product(&shape[1..])?;
        let start = spec
            .start_index
            .checked_mul(elements_per_row)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing byte range start"))?;
        let end = spec
            .end_index
            .checked_mul(elements_per_row)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing byte range end"))?;
        let total = end
            .checked_sub(start)
            .ok_or_else(|| anyhow::anyhow!("end {end} < start {start} in byte range"))?;
        Ok(ByteRangeSpec {
            ranges: vec![(start, end)],
            total_bytes: total,
        })
    } else {
        // Strided ranges.
        let outer_count: usize = checked_product(&shape[..axis])?;
        let inner_count: usize = checked_product(&shape[(axis + 1)..])?;
        let full_axis_stride = shape[axis]
            .checked_mul(inner_count)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing full_axis_stride"))?;
        let shard_bytes = spec
            .shard_size()
            .checked_mul(inner_count)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing shard_bytes"))?;
        let shard_start_offset = spec
            .start_index
            .checked_mul(inner_count)
            .and_then(|v| v.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("overflow computing shard_start_offset"))?;

        let mut ranges = Vec::with_capacity(outer_count);
        let mut total_bytes = 0usize;
        for outer_idx in 0..outer_count {
            let start = outer_idx
                .checked_mul(full_axis_stride)
                .and_then(|v| v.checked_add(shard_start_offset))
                .ok_or_else(|| {
                    anyhow::anyhow!("overflow computing byte range at outer_idx {outer_idx}")
                })?;
            let end = start
                .checked_add(shard_bytes)
                .ok_or_else(|| anyhow::anyhow!("overflow computing byte range end"))?;
            ranges.push((start, end));
            total_bytes = total_bytes
                .checked_add(shard_bytes)
                .ok_or_else(|| anyhow::anyhow!("overflow accumulating total_bytes"))?;
        }

        Ok(ByteRangeSpec {
            ranges,
            total_bytes,
        })
    }
}

/// Memory accounting report for a sharded model.
#[derive(Debug, Clone)]
pub struct ShardedMemoryReport {
    /// Bytes loaded per rank (indexed by rank).
    pub per_rank_bytes: Vec<usize>,
    /// Total bytes for replicated weights (same on every rank).
    pub replicated_bytes: usize,
    /// Total bytes for sharded weights across all ranks.
    pub sharded_bytes: usize,
    /// Total bytes of the original unsharded model.
    pub total_original_bytes: usize,
    /// Number of sharded weight tensors.
    pub num_sharded_weights: usize,
    /// Number of replicated weight tensors.
    pub num_replicated_weights: usize,
}

impl ShardedMemoryReport {
    /// Maximum memory used by any single rank.
    pub fn max_rank_bytes(&self) -> usize {
        self.per_rank_bytes.iter().copied().max().unwrap_or(0)
    }

    /// Memory savings ratio compared to full replication (0.0 = no savings, 1.0 = max savings).
    pub fn savings_ratio(&self) -> f64 {
        if self.total_original_bytes == 0 {
            return 0.0;
        }
        let max_rank = self.max_rank_bytes() as f64;
        let total = self.total_original_bytes as f64;
        1.0 - (max_rank / total)
    }

    /// True if the sharded memory is consistent: sum of unique sharded bytes
    /// across ranks equals the original sharded bytes.
    pub fn is_consistent(&self) -> bool {
        // Sum of sharded portions across ranks should equal the total sharded bytes.
        let sum_sharded: usize = self
            .per_rank_bytes
            .iter()
            .map(|b| b.saturating_sub(self.replicated_bytes))
            .sum();
        sum_sharded == self.sharded_bytes
    }
}

/// Validate memory accounting for a sharded model.
///
/// Ensures that the sum of per-rank sharded memory equals the original model's
/// sharded weight memory (replicated weights are counted separately).
///
/// # Arguments
/// * `plan` — the model's shard plan
/// * `shapes` — map from weight name to tensor shape
/// * `dtype_sizes` — map from weight name to bytes-per-element
pub fn validate_sharded_memory(
    plan: &ModelShardPlan,
    shapes: &HashMap<String, Vec<usize>>,
    dtype_sizes: &HashMap<String, usize>,
) -> Result<ShardedMemoryReport> {
    let tp_size = plan.tp_size;
    let mut per_rank_bytes = vec![0usize; tp_size];
    let mut replicated_bytes = 0usize;
    let mut sharded_bytes = 0usize;
    let mut total_original_bytes = 0usize;
    let mut num_sharded = 0usize;
    let mut num_replicated = 0usize;

    for (weight_name, shape) in shapes {
        let dtype_size = dtype_sizes.get(weight_name).copied().unwrap_or(2); // Default to float16

        let original_elements: usize = checked_product(shape)?;
        let original_bytes = original_elements.checked_mul(dtype_size).ok_or_else(|| {
            anyhow::anyhow!(
                "overflow computing bytes for weight '{weight_name}' with shape {shape:?}"
            )
        })?;
        total_original_bytes += original_bytes;

        // Compute spec for rank 0 to determine strategy.
        let spec = compute_shard_spec(weight_name, shape, plan, 0)?;

        if spec.is_replicated() {
            replicated_bytes += original_bytes;
            num_replicated += 1;
            for rank_bytes in &mut per_rank_bytes {
                *rank_bytes += original_bytes;
            }
        } else {
            num_sharded += 1;
            for (rank, rank_byte_slot) in per_rank_bytes.iter_mut().enumerate() {
                let rank_spec = compute_shard_spec(weight_name, shape, plan, rank)?;
                let sharded_shape = compute_sharded_shape(shape, &rank_spec);
                let rank_elements: usize = checked_product(&sharded_shape)?;
                let rank_bytes = rank_elements
                    .checked_mul(dtype_size)
                    .ok_or_else(|| anyhow::anyhow!("overflow computing rank bytes"))?;
                *rank_byte_slot += rank_bytes;
            }
            sharded_bytes += original_bytes;
        }
    }

    let report = ShardedMemoryReport {
        per_rank_bytes,
        replicated_bytes,
        sharded_bytes,
        total_original_bytes,
        num_sharded_weights: num_sharded,
        num_replicated_weights: num_replicated,
    };

    ensure!(
        report.is_consistent(),
        "memory accounting inconsistency: sum of per-rank sharded bytes does not equal total \
         sharded bytes ({} != {})",
        report
            .per_rank_bytes
            .iter()
            .map(|b| b.saturating_sub(report.replicated_bytes))
            .sum::<usize>(),
        report.sharded_bytes,
    );

    Ok(report)
}

/// Dtype size lookup helper.
///
/// Returns bytes per element for common MLX/safetensors dtype strings.
pub fn dtype_byte_size(dtype: &str) -> usize {
    match dtype {
        "float32" | "f32" => 4,
        "float16" | "f16" | "bfloat16" | "bf16" => 2,
        "int32" | "i32" | "uint32" | "u32" => 4,
        "int16" | "i16" | "uint16" | "u16" => 2,
        "int8" | "i8" | "uint8" | "u8" => 1,
        "int64" | "i64" | "uint64" | "u64" => 8,
        "float64" | "f64" => 8,
        "bool" => 1,
        other => {
            tracing::warn!("unknown dtype '{other}', defaulting to 2 bytes (float16)");
            2
        }
    }
}

#[cfg(test)]
#[path = "sharded_loading_tests.rs"]
mod tests;
