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

//! Tensor-parallel FFN configuration and shape computation.
//!
//! Standard transformer FFN layers (SwiGLU, GeGLU, etc.) follow a column/row
//! parallel pattern:
//!
//! ```text
//! Input -> [gate_proj (col-parallel)] -> activation -> x
//!                                                       -> [down_proj (row-parallel)] -> all-reduce -> Output
//! Input -> [up_proj   (col-parallel)] ------------------->
//! ```
//!
//! This module provides:
//!
//! - [`FFNActivationType`] — supported activation functions (SiLU, GELU, ReLU, etc.)
//! - [`TPFFNConfig`] — per-model FFN parallelism parameters
//! - [`TPFFNMetadata`] — per-rank FFN shape information
//! - [`compute_local_ffn_shapes`] — derive per-rank FFN dimensions
//! - [`validate_tp_ffn_config`] — check configuration consistency
//! - [`compute_all_rank_ffn_metadata`] — compute metadata for all ranks
//!
//! Used by: tensor_parallel forward pass (FFN layer), model loading, MoE expert sharding

use std::ops::Range;

use anyhow::{Result, ensure};

/// Activation function used in the FFN gated path.
///
/// Most modern transformer FFNs use a gated architecture where the gate path
/// applies an activation function before element-wise multiplication with the
/// up-projection path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum FFNActivationType {
    /// SiLU (Sigmoid Linear Unit), also known as Swish.
    /// Used by: Llama, Mistral, Qwen, DeepSeek, Gemma, Phi, StableLM, etc.
    #[default]
    SiLU,
    /// GELU (Gaussian Error Linear Unit).
    /// Used by: GPT-2, BERT, StarCoder, OLMo, Falcon, etc.
    GELU,
    /// GELU with fast (tanh) approximation.
    /// Used by: GPT-NeoX, Phi-2, some Falcon variants.
    GELUApprox,
    /// ReLU (Rectified Linear Unit).
    /// Used by: older models, some MoE experts.
    ReLU,
    /// Squared ReLU (ReLU then square).
    /// Used by: some research models (Primer, etc.).
    ReLUSquared,
}

impl std::fmt::Display for FFNActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SiLU => write!(f, "silu"),
            Self::GELU => write!(f, "gelu"),
            Self::GELUApprox => write!(f, "gelu_approx"),
            Self::ReLU => write!(f, "relu"),
            Self::ReLUSquared => write!(f, "relu_squared"),
        }
    }
}

impl FFNActivationType {
    /// Whether this activation is applied element-wise (all supported types are).
    ///
    /// Element-wise activations require no cross-rank communication when applied
    /// to sharded intermediate tensors, making them inherently TP-compatible.
    pub fn is_elementwise(&self) -> bool {
        true
    }

    /// Whether this FFN uses a gated architecture (gate_proj * up_proj).
    ///
    /// Gated FFNs have two column-parallel projections (gate and up) whose
    /// outputs are multiplied element-wise before the row-parallel down projection.
    /// All modern gated architectures (SwiGLU, GeGLU) use this pattern.
    pub fn is_gated(&self) -> bool {
        // All listed activations are used in gated FFN architectures.
        // Non-gated FFN (single linear + activation + linear) would return false.
        true
    }
}

/// Configuration for tensor-parallel FFN.
///
/// Captures the model's FFN geometry and TP parallelism factor.
/// From this, per-rank intermediate dimensions are derived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TPFFNConfig {
    /// This rank's index (0-based).
    pub tp_rank: usize,
    /// Total number of TP ranks.
    pub tp_size: usize,
    /// Total intermediate (hidden) size of the FFN.
    /// For gated FFNs, this is the size of gate_proj/up_proj output.
    pub intermediate_size: usize,
    /// Model hidden size (input/output dimension of the FFN block).
    pub hidden_size: usize,
    /// Activation function type.
    pub activation: FFNActivationType,
}

/// Per-rank FFN shape metadata derived from a [`TPFFNConfig`].
///
/// Contains all dimensions needed to set up the local FFN computation
/// on a single TP rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TPFFNMetadata {
    /// The TP rank this metadata is for.
    pub tp_rank: usize,
    /// Local intermediate dimension for this rank (gate_proj/up_proj output columns).
    /// Equal to `intermediate_size / tp_size`.
    pub local_intermediate_size: usize,
    /// Range of intermediate dimension indices assigned to this rank.
    pub intermediate_range: Range<usize>,
    /// Model hidden size (unchanged across ranks — full input dimension).
    pub hidden_size: usize,
    /// Gate projection shape: `[hidden_size, local_intermediate_size]` (column-parallel).
    pub gate_proj_shape: [usize; 2],
    /// Up projection shape: `[hidden_size, local_intermediate_size]` (column-parallel).
    pub up_proj_shape: [usize; 2],
    /// Down projection shape: `[local_intermediate_size, hidden_size]` (row-parallel).
    pub down_proj_shape: [usize; 2],
    /// Whether all-reduce is required after the down projection.
    pub needs_allreduce: bool,
    /// Activation function type.
    pub activation: FFNActivationType,
}

/// Validate a tensor-parallel FFN configuration.
///
/// Checks:
/// - `tp_rank < tp_size`
/// - `tp_size >= 1`
/// - `intermediate_size > 0` and `hidden_size > 0`
/// - `intermediate_size` is divisible by `tp_size`
pub fn validate_tp_ffn_config(config: &TPFFNConfig) -> Result<()> {
    ensure!(config.tp_size >= 1, "tp_size must be >= 1");
    ensure!(
        config.tp_rank < config.tp_size,
        "tp_rank {} out of range for tp_size {}",
        config.tp_rank,
        config.tp_size
    );
    ensure!(
        config.intermediate_size > 0,
        "intermediate_size must be > 0"
    );
    ensure!(config.hidden_size > 0, "hidden_size must be > 0");
    ensure!(
        config.intermediate_size.is_multiple_of(config.tp_size),
        "intermediate_size ({}) must be divisible by tp_size ({})",
        config.intermediate_size,
        config.tp_size
    );
    Ok(())
}

/// Compute per-rank FFN shapes from a TP FFN configuration.
///
/// This is the main entry point for deriving all local dimensions needed
/// by the FFN forward pass on a single rank.
///
/// # Column-parallel projections (gate_proj, up_proj)
///
/// Each rank computes a slice of the intermediate dimension:
/// - Weight shape: `[hidden_size, local_intermediate_size]`
/// - Input: full hidden state `[batch, seq, hidden_size]`
/// - Output: sharded intermediate `[batch, seq, local_intermediate_size]`
///
/// # Row-parallel projection (down_proj)
///
/// Each rank computes a partial output from its intermediate slice:
/// - Weight shape: `[local_intermediate_size, hidden_size]`
/// - Input: sharded intermediate `[batch, seq, local_intermediate_size]`
/// - Output: partial hidden state `[batch, seq, hidden_size]`
/// - Followed by all-reduce sum across ranks.
pub fn compute_local_ffn_shapes(config: &TPFFNConfig) -> Result<TPFFNMetadata> {
    validate_tp_ffn_config(config)?;

    let local_intermediate = config.intermediate_size / config.tp_size;
    let start = config.tp_rank * local_intermediate;
    let end = start + local_intermediate;

    Ok(TPFFNMetadata {
        tp_rank: config.tp_rank,
        local_intermediate_size: local_intermediate,
        intermediate_range: start..end,
        hidden_size: config.hidden_size,
        gate_proj_shape: [config.hidden_size, local_intermediate],
        up_proj_shape: [config.hidden_size, local_intermediate],
        down_proj_shape: [local_intermediate, config.hidden_size],
        needs_allreduce: config.tp_size > 1,
        activation: config.activation,
    })
}

/// Compute FFN metadata for all ranks in a TP group.
///
/// Useful for validation: ensures all ranks together cover the full
/// intermediate dimension exactly once.
pub fn compute_all_rank_ffn_metadata(
    tp_size: usize,
    intermediate_size: usize,
    hidden_size: usize,
    activation: FFNActivationType,
) -> Result<Vec<TPFFNMetadata>> {
    let mut all_meta = Vec::with_capacity(tp_size);
    for rank in 0..tp_size {
        let config = TPFFNConfig {
            tp_rank: rank,
            tp_size,
            intermediate_size,
            hidden_size,
            activation,
        };
        all_meta.push(compute_local_ffn_shapes(&config)?);
    }
    Ok(all_meta)
}

/// Verify that a set of per-rank FFN metadata covers the full intermediate
/// dimension exactly once (no overlap, no gaps).
pub fn verify_intermediate_coverage(
    all_meta: &[TPFFNMetadata],
    intermediate_size: usize,
) -> Result<()> {
    let mut covered = vec![false; intermediate_size];
    for meta in all_meta {
        for idx in meta.intermediate_range.clone() {
            ensure!(
                !covered[idx],
                "intermediate index {idx} assigned to multiple ranks"
            );
            covered[idx] = true;
        }
    }
    for (idx, &c) in covered.iter().enumerate() {
        ensure!(c, "intermediate index {idx} not assigned to any rank");
    }
    Ok(())
}

#[cfg(test)]
#[path = "parallel_ffn_tests.rs"]
mod tests;
