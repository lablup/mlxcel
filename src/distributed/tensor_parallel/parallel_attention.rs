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

//! Tensor-parallel attention configuration and head assignment.
//!
//! This module computes how attention heads are distributed across TP ranks,
//! handling MHA, GQA, and MQA attention variants. It provides:
//!
//! - [`TPAttentionConfig`] — per-model attention parallelism parameters
//! - [`KVAssignment`] — how KV heads map to a rank (sharded or replicated)
//! - [`TPAttentionMetadata`] — per-rank attention shape information
//! - [`head_assignment`] — compute which Q heads belong to a rank
//! - [`kv_head_assignment`] — compute KV head mapping with GQA awareness
//! - [`compute_local_attention_shapes`] — derive all per-rank dimensions
//! - [`validate_tp_attention_config`] — check configuration consistency
//!
//! Used by: tensor_parallel forward pass (attention layer), model loading

use std::ops::Range;

use anyhow::{Result, ensure};

/// Attention type classification based on Q/KV head counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AttentionType {
    /// Multi-Head Attention: n_kv_heads == n_heads.
    MHA,
    /// Grouped-Query Attention: 1 < n_kv_heads < n_heads.
    GQA,
    /// Multi-Query Attention: n_kv_heads == 1.
    MQA,
}

impl std::fmt::Display for AttentionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MHA => write!(f, "MHA"),
            Self::GQA => write!(f, "GQA"),
            Self::MQA => write!(f, "MQA"),
        }
    }
}

/// How KV heads are assigned to a particular TP rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KVAssignment {
    /// KV heads are sharded across ranks; this rank owns the given range.
    /// Occurs when `n_kv_heads >= tp_size`.
    Sharded(Range<usize>),

    /// KV heads are replicated on this rank because `n_kv_heads < tp_size`.
    /// All KV heads are loaded on every rank. The range covers all KV heads.
    Replicated(Range<usize>),
}

impl KVAssignment {
    /// Number of KV heads this rank operates on.
    pub fn num_heads(&self) -> usize {
        match self {
            Self::Sharded(range) | Self::Replicated(range) => range.len(),
        }
    }

    /// The range of KV head indices this rank uses.
    pub fn head_range(&self) -> &Range<usize> {
        match self {
            Self::Sharded(range) | Self::Replicated(range) => range,
        }
    }

    /// Whether KV heads are replicated (not sharded).
    pub fn is_replicated(&self) -> bool {
        matches!(self, Self::Replicated(_))
    }
}

/// Configuration for tensor-parallel attention.
///
/// Captures the model's attention geometry and the TP parallelism factor.
/// From this, per-rank head assignments and shapes are derived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TPAttentionConfig {
    /// This rank's index (0-based).
    pub tp_rank: usize,
    /// Total number of TP ranks.
    pub tp_size: usize,
    /// Total number of query heads in the model.
    pub total_heads: usize,
    /// Total number of key/value heads in the model.
    /// Equal to `total_heads` for MHA, 1 for MQA, otherwise GQA.
    pub total_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Optional sliding window size (None = full causal attention).
    pub sliding_window: Option<usize>,
}

impl TPAttentionConfig {
    /// Classify the attention type based on head counts.
    pub fn attention_type(&self) -> AttentionType {
        if self.total_kv_heads == self.total_heads {
            AttentionType::MHA
        } else if self.total_kv_heads == 1 {
            AttentionType::MQA
        } else {
            AttentionType::GQA
        }
    }

    /// The GQA group size: how many Q heads share each KV head.
    ///
    /// For MHA this is 1, for MQA this equals `total_heads`.
    pub fn gqa_group_size(&self) -> usize {
        if self.total_kv_heads == 0 {
            return 0;
        }
        self.total_heads / self.total_kv_heads
    }
}

/// Per-rank attention shape metadata derived from a [`TPAttentionConfig`].
///
/// Contains all the dimensions needed to set up the local attention computation
/// on a single TP rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TPAttentionMetadata {
    /// The TP rank this metadata is for.
    pub tp_rank: usize,
    /// Number of Q heads on this rank.
    pub local_n_heads: usize,
    /// Number of KV heads on this rank.
    pub local_n_kv_heads: usize,
    /// Range of Q head indices assigned to this rank.
    pub q_head_range: Range<usize>,
    /// KV head assignment for this rank.
    pub kv_assignment: KVAssignment,
    /// Head dimension (same on all ranks).
    pub head_dim: usize,
    /// Q projection output dimension for this rank: `local_n_heads * head_dim`.
    pub local_q_dim: usize,
    /// K projection output dimension for this rank: `local_n_kv_heads * head_dim`.
    pub local_k_dim: usize,
    /// V projection output dimension for this rank: `local_n_kv_heads * head_dim`.
    pub local_v_dim: usize,
    /// O projection input dimension for this rank: `local_n_heads * head_dim`.
    pub local_o_input_dim: usize,
    /// Whether all-reduce is required after the O projection.
    pub needs_allreduce: bool,
    /// The attention type classification.
    pub attention_type: AttentionType,
    /// Optional sliding window size.
    pub sliding_window: Option<usize>,
}

/// Validate a tensor-parallel attention configuration.
///
/// Checks:
/// - `tp_rank < tp_size`
/// - `tp_size >= 1`
/// - `total_heads > 0` and `total_kv_heads > 0`
/// - `total_heads` is divisible by `tp_size`
/// - `total_heads` is divisible by `total_kv_heads` (valid GQA ratio)
/// - `head_dim > 0`
pub fn validate_tp_attention_config(config: &TPAttentionConfig) -> Result<()> {
    ensure!(config.tp_size >= 1, "tp_size must be >= 1");
    ensure!(
        config.tp_rank < config.tp_size,
        "tp_rank {} out of range for tp_size {}",
        config.tp_rank,
        config.tp_size
    );
    ensure!(config.total_heads > 0, "total_heads must be > 0");
    ensure!(config.total_kv_heads > 0, "total_kv_heads must be > 0");
    ensure!(config.head_dim > 0, "head_dim must be > 0");
    ensure!(
        config.total_heads.is_multiple_of(config.tp_size),
        "total_heads ({}) must be divisible by tp_size ({})",
        config.total_heads,
        config.tp_size
    );
    ensure!(
        config.total_heads.is_multiple_of(config.total_kv_heads),
        "total_heads ({}) must be divisible by total_kv_heads ({}) for valid GQA ratio",
        config.total_heads,
        config.total_kv_heads
    );
    // When KV heads can be sharded, they must divide evenly across ranks.
    if config.total_kv_heads >= config.tp_size {
        ensure!(
            config.total_kv_heads.is_multiple_of(config.tp_size),
            "total_kv_heads ({}) must be divisible by tp_size ({}) when sharding KV heads",
            config.total_kv_heads,
            config.tp_size
        );
    }
    Ok(())
}

/// Compute which Q heads are assigned to a given TP rank.
///
/// Q heads are evenly divided across ranks. Requires `total_heads % tp_size == 0`.
///
/// # Returns
/// Range of Q head indices for the given rank.
pub fn head_assignment(rank: usize, tp_size: usize, total_heads: usize) -> Range<usize> {
    let heads_per_rank = total_heads / tp_size;
    let start = rank * heads_per_rank;
    let end = start + heads_per_rank;
    start..end
}

/// Compute the KV head assignment for a given TP rank.
///
/// Three cases:
/// 1. **MHA / shardable GQA** (`n_kv_heads >= tp_size`): KV heads are evenly
///    sharded across ranks. Each rank gets `n_kv_heads / tp_size` KV heads.
/// 2. **MQA** (`n_kv_heads == 1`): The single KV head is replicated on all ranks.
/// 3. **Non-shardable GQA** (`1 < n_kv_heads < tp_size`): All KV heads are
///    replicated on every rank because they cannot be evenly divided.
///
/// # Returns
/// A [`KVAssignment`] describing how KV heads map to this rank.
pub fn kv_head_assignment(rank: usize, tp_size: usize, total_kv_heads: usize) -> KVAssignment {
    if total_kv_heads >= tp_size {
        // KV heads can be sharded evenly across ranks.
        let kv_per_rank = total_kv_heads / tp_size;
        let start = rank * kv_per_rank;
        let end = start + kv_per_rank;
        KVAssignment::Sharded(start..end)
    } else {
        // Not enough KV heads to shard; replicate all on every rank.
        KVAssignment::Replicated(0..total_kv_heads)
    }
}

/// Compute per-rank attention shapes from a TP attention configuration.
///
/// This is the main entry point for deriving all local dimensions needed
/// by the attention forward pass on a single rank.
pub fn compute_local_attention_shapes(config: &TPAttentionConfig) -> Result<TPAttentionMetadata> {
    validate_tp_attention_config(config)?;

    let q_range = head_assignment(config.tp_rank, config.tp_size, config.total_heads);
    let local_n_heads = q_range.len();

    let kv_assign = kv_head_assignment(config.tp_rank, config.tp_size, config.total_kv_heads);
    let local_n_kv_heads = kv_assign.num_heads();

    // Defense-in-depth: verify that local GQA group ratio is valid.
    // Each rank's Q heads must be divisible by its KV heads for correct
    // grouped-query attention. This is mathematically guaranteed by the
    // validation above, but we assert it explicitly to catch any future
    // changes that might break this invariant.
    ensure!(
        local_n_kv_heads > 0 && local_n_heads.is_multiple_of(local_n_kv_heads),
        "local GQA group invariant violated: local_n_heads ({local_n_heads}) must be divisible by local_n_kv_heads ({local_n_kv_heads}) on rank {}",
        config.tp_rank
    );

    let local_q_dim = local_n_heads * config.head_dim;
    let local_k_dim = local_n_kv_heads * config.head_dim;
    let local_v_dim = local_n_kv_heads * config.head_dim;
    let local_o_input_dim = local_n_heads * config.head_dim;

    Ok(TPAttentionMetadata {
        tp_rank: config.tp_rank,
        local_n_heads,
        local_n_kv_heads,
        q_head_range: q_range,
        kv_assignment: kv_assign,
        head_dim: config.head_dim,
        local_q_dim,
        local_k_dim,
        local_v_dim,
        local_o_input_dim,
        needs_allreduce: requires_allreduce_after_o_proj(config.tp_size),
        attention_type: config.attention_type(),
        sliding_window: config.sliding_window,
    })
}

/// Whether an all-reduce is required after the O projection.
///
/// True when `tp_size > 1` because each rank computes a partial output
/// from its subset of heads, and these must be summed.
#[inline]
pub fn requires_allreduce_after_o_proj(tp_size: usize) -> bool {
    tp_size > 1
}

/// Compute attention metadata for all ranks in a TP group.
///
/// Useful for validation: ensures all ranks together cover every Q head
/// exactly once and every KV head is accounted for.
pub fn compute_all_rank_metadata(
    tp_size: usize,
    total_heads: usize,
    total_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
) -> Result<Vec<TPAttentionMetadata>> {
    let mut all_meta = Vec::with_capacity(tp_size);
    for rank in 0..tp_size {
        let config = TPAttentionConfig {
            tp_rank: rank,
            tp_size,
            total_heads,
            total_kv_heads,
            head_dim,
            sliding_window,
        };
        all_meta.push(compute_local_attention_shapes(&config)?);
    }
    Ok(all_meta)
}

/// Verify that a set of per-rank metadata covers all Q heads exactly once
/// and that KV head assignments are consistent.
pub fn verify_head_coverage(
    all_meta: &[TPAttentionMetadata],
    total_heads: usize,
    total_kv_heads: usize,
) -> Result<()> {
    // Check Q head coverage: every Q head should appear exactly once.
    let mut q_covered = vec![false; total_heads];
    for meta in all_meta {
        for idx in meta.q_head_range.clone() {
            ensure!(!q_covered[idx], "Q head {idx} assigned to multiple ranks");
            q_covered[idx] = true;
        }
    }
    for (idx, &covered) in q_covered.iter().enumerate() {
        ensure!(covered, "Q head {idx} not assigned to any rank");
    }

    // Check KV head coverage.
    if total_kv_heads >= all_meta.len() {
        // Sharded: each KV head should appear exactly once.
        let mut kv_covered = vec![false; total_kv_heads];
        for meta in all_meta {
            for idx in meta.kv_assignment.head_range().clone() {
                ensure!(
                    !kv_covered[idx],
                    "KV head {idx} assigned to multiple ranks (expected sharded)"
                );
                kv_covered[idx] = true;
            }
        }
        for (idx, &covered) in kv_covered.iter().enumerate() {
            ensure!(covered, "KV head {idx} not assigned to any rank");
        }
    } else {
        // Replicated: every rank should have all KV heads.
        for meta in all_meta {
            ensure!(
                meta.kv_assignment.is_replicated(),
                "rank {} should have replicated KV heads but has sharded",
                meta.tp_rank
            );
            ensure!(
                meta.kv_assignment.num_heads() == total_kv_heads,
                "rank {} has {} KV heads, expected {total_kv_heads}",
                meta.tp_rank,
                meta.kv_assignment.num_heads()
            );
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "parallel_attention_tests.rs"]
mod tests;
