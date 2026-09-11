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

//! Tensor-parallel MoE (Mixture-of-Experts) configuration and shape computation.
//!
//! MoE layers support two parallelism strategies:
//!
//! **Expert Parallelism** — different experts are assigned to different ranks.
//! The router is replicated across all ranks for deterministic routing decisions.
//! Tokens are dispatched to ranks owning the target experts via all-to-all
//! communication, and results are gathered back.
//!
//! **Within-Expert Sharding** — each expert's FFN weights are sharded across
//! all ranks using the same column/row-parallel pattern as standard FFN
//! (see [`super::parallel_ffn`]). All ranks process all experts with sharded
//! weights, followed by all-reduce.
//!
//! DeepSeek-style shared experts are treated as standard FFN with TP sharding,
//! and their output is combined with the routed expert output.
//!
//! This module provides:
//!
//! - [`MoEParallelMode`] — expert-parallel vs within-expert sharding
//! - [`TPMoEConfig`] — per-model MoE parallelism parameters
//! - [`ExpertAssignment`] — which experts a rank owns (expert-parallel mode)
//! - [`TPMoEMetadata`] — per-rank MoE shape information
//! - [`compute_expert_assignment`] — round-robin expert-to-rank mapping
//! - [`compute_local_moe_shapes`] — derive per-rank MoE dimensions
//! - [`validate_tp_moe_config`] — check configuration consistency
//!
//! Used by: tensor_parallel forward pass (MoE layer), model loading

use anyhow::{Result, ensure};

use super::parallel_ffn::{
    FFNActivationType, TPFFNConfig, TPFFNMetadata, compute_local_ffn_shapes,
};

/// How MoE expert weights are parallelized across TP ranks.
///
/// This mirrors [`super::config::MoeShardMode`] but is used in the
/// per-layer computation context rather than global configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum MoEParallelMode {
    /// Whole experts assigned to ranks round-robin.
    /// Expert `i` lives entirely on rank `i % tp_size`.
    /// Best when `num_experts >= tp_size`.
    #[default]
    ExpertParallel,

    /// Each expert's FFN weights sharded across all ranks (column/row parallel).
    /// All ranks process all experts with sharded weights.
    /// Best when `num_experts < tp_size` or experts are very large.
    WithinExpertSharding,
}

impl std::fmt::Display for MoEParallelMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExpertParallel => write!(f, "expert_parallel"),
            Self::WithinExpertSharding => write!(f, "within_expert_sharding"),
        }
    }
}

/// Configuration for tensor-parallel MoE.
///
/// Captures the model's MoE geometry, TP parallelism factor, and the chosen
/// parallelism strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TPMoEConfig {
    /// This rank's index (0-based).
    pub tp_rank: usize,
    /// Total number of TP ranks.
    pub tp_size: usize,
    /// Total number of routed experts in the MoE layer.
    pub num_experts: usize,
    /// Number of experts selected per token by the router (top-k).
    pub experts_per_token: usize,
    /// Intermediate size of each expert's FFN.
    pub expert_intermediate_size: usize,
    /// Model hidden size (input/output dimension).
    pub hidden_size: usize,
    /// Whether a shared (non-routed) expert exists (DeepSeek-style).
    pub has_shared_expert: bool,
    /// Intermediate size of the shared expert (if present).
    /// Defaults to `expert_intermediate_size` when not specified.
    pub shared_expert_intermediate_size: Option<usize>,
    /// Parallelism mode for the MoE layer.
    pub mode: MoEParallelMode,
    /// Activation function used in expert FFNs.
    pub activation: FFNActivationType,
}

/// Which experts are assigned to a particular rank in expert-parallel mode.
///
/// In expert-parallel mode, experts are distributed round-robin across ranks.
/// Each rank owns a subset of experts and is responsible for processing all
/// tokens routed to those experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertAssignment {
    /// The TP rank this assignment is for.
    pub tp_rank: usize,
    /// Indices of experts owned by this rank, sorted ascending.
    pub expert_indices: Vec<usize>,
    /// Total number of experts across all ranks.
    pub total_experts: usize,
}

impl ExpertAssignment {
    /// Number of experts on this rank.
    pub fn num_local_experts(&self) -> usize {
        self.expert_indices.len()
    }

    /// Whether this rank owns the given expert index.
    pub fn owns_expert(&self, expert_idx: usize) -> bool {
        self.expert_indices.contains(&expert_idx)
    }
}

/// Per-rank MoE shape metadata derived from a [`TPMoEConfig`].
///
/// Contains all dimensions needed to set up the local MoE computation
/// on a single TP rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TPMoEMetadata {
    /// The TP rank this metadata is for.
    pub tp_rank: usize,
    /// The parallelism mode in effect.
    pub mode: MoEParallelMode,

    // -- Expert-parallel fields --
    /// Expert assignment for this rank (only meaningful in ExpertParallel mode).
    /// In WithinExpertSharding mode this field is still populated (round-robin)
    /// but should be ignored: all ranks process all experts with sharded weights.
    pub expert_assignment: ExpertAssignment,

    // -- Within-expert sharding fields --
    /// Per-expert FFN metadata when using within-expert sharding.
    /// In ExpertParallel mode, this describes the full (unsharded) expert FFN
    /// with tp_size=1 for the experts this rank owns.
    pub expert_ffn_meta: TPFFNMetadata,

    // -- Shared expert fields --
    /// Whether a shared expert is present.
    pub has_shared_expert: bool,
    /// Shared expert FFN metadata (TP-sharded like a standard FFN).
    /// `None` when `has_shared_expert` is false.
    pub shared_expert_ffn_meta: Option<TPFFNMetadata>,

    // -- Router fields --
    /// Router weight shape: `[hidden_size, num_experts]`.
    /// Replicated across all ranks for deterministic routing.
    pub router_weight_shape: [usize; 2],
    /// Whether the router is replicated (always true in current design).
    pub router_replicated: bool,
    /// Number of experts selected per token (top-k).
    pub experts_per_token: usize,

    // -- Communication --
    /// Whether all-to-all communication is needed for token dispatch
    /// (ExpertParallel mode with tp_size > 1).
    pub needs_all_to_all: bool,
    /// Whether all-reduce is needed after expert computation
    /// (WithinExpertSharding mode with tp_size > 1).
    pub needs_allreduce: bool,
}

/// Validate a tensor-parallel MoE configuration.
///
/// Checks:
/// - `tp_rank < tp_size`
/// - `tp_size >= 1`
/// - `num_experts > 0`
/// - `experts_per_token > 0` and `<= num_experts`
/// - `expert_intermediate_size > 0` and `hidden_size > 0`
/// - Mode-specific: ExpertParallel requires `num_experts >= tp_size`
/// - Mode-specific: WithinExpertSharding requires `expert_intermediate_size % tp_size == 0`
pub fn validate_tp_moe_config(config: &TPMoEConfig) -> Result<()> {
    ensure!(config.tp_size >= 1, "tp_size must be >= 1");
    ensure!(
        config.tp_rank < config.tp_size,
        "tp_rank {} out of range for tp_size {}",
        config.tp_rank,
        config.tp_size
    );
    ensure!(config.num_experts > 0, "num_experts must be > 0");
    ensure!(
        config.experts_per_token > 0,
        "experts_per_token must be > 0"
    );
    ensure!(
        config.experts_per_token <= config.num_experts,
        "experts_per_token ({}) must be <= num_experts ({})",
        config.experts_per_token,
        config.num_experts
    );
    ensure!(
        config.expert_intermediate_size > 0,
        "expert_intermediate_size must be > 0"
    );
    ensure!(config.hidden_size > 0, "hidden_size must be > 0");

    match config.mode {
        MoEParallelMode::ExpertParallel => {
            ensure!(
                config.num_experts >= config.tp_size,
                "ExpertParallel requires num_experts ({}) >= tp_size ({}); \
                 use WithinExpertSharding when num_experts < tp_size",
                config.num_experts,
                config.tp_size
            );
        }
        MoEParallelMode::WithinExpertSharding => {
            ensure!(
                config
                    .expert_intermediate_size
                    .is_multiple_of(config.tp_size),
                "WithinExpertSharding requires expert_intermediate_size ({}) \
                 divisible by tp_size ({})",
                config.expert_intermediate_size,
                config.tp_size
            );
        }
    }

    // Validate shared expert intermediate size if present.
    if config.has_shared_expert {
        let shared_size = config
            .shared_expert_intermediate_size
            .unwrap_or(config.expert_intermediate_size);
        ensure!(
            shared_size > 0,
            "shared_expert_intermediate_size must be > 0"
        );
        // Shared expert is always TP-sharded (column/row parallel).
        if config.tp_size > 1 {
            ensure!(
                shared_size.is_multiple_of(config.tp_size),
                "shared_expert_intermediate_size ({shared_size}) must be \
                 divisible by tp_size ({})",
                config.tp_size
            );
        }
    }

    Ok(())
}

/// Compute round-robin expert assignment for a given TP rank.
///
/// In expert-parallel mode, expert `i` is assigned to rank `i % tp_size`.
/// This produces balanced assignment when `num_experts` is a multiple of
/// `tp_size`, and near-balanced otherwise (some ranks get one extra expert).
pub fn compute_expert_assignment(
    rank: usize,
    tp_size: usize,
    num_experts: usize,
) -> ExpertAssignment {
    let indices: Vec<usize> = (0..num_experts).filter(|&i| i % tp_size == rank).collect();

    ExpertAssignment {
        tp_rank: rank,
        expert_indices: indices,
        total_experts: num_experts,
    }
}

/// Compute per-rank MoE shapes from a TP MoE configuration.
///
/// This is the main entry point for deriving all local dimensions and
/// assignments needed by the MoE forward pass on a single rank.
pub fn compute_local_moe_shapes(config: &TPMoEConfig) -> Result<TPMoEMetadata> {
    validate_tp_moe_config(config)?;

    let assignment = compute_expert_assignment(config.tp_rank, config.tp_size, config.num_experts);

    // Compute expert FFN metadata based on mode.
    let expert_ffn_meta = match config.mode {
        MoEParallelMode::ExpertParallel => {
            // Each rank owns whole experts; expert FFN is unsharded (tp_size=1).
            compute_local_ffn_shapes(&TPFFNConfig {
                tp_rank: 0,
                tp_size: 1,
                intermediate_size: config.expert_intermediate_size,
                hidden_size: config.hidden_size,
                activation: config.activation,
            })?
        }
        MoEParallelMode::WithinExpertSharding => {
            // Expert FFN sharded across all ranks.
            compute_local_ffn_shapes(&TPFFNConfig {
                tp_rank: config.tp_rank,
                tp_size: config.tp_size,
                intermediate_size: config.expert_intermediate_size,
                hidden_size: config.hidden_size,
                activation: config.activation,
            })?
        }
    };

    // Compute shared expert FFN metadata (always TP-sharded).
    let shared_expert_ffn_meta = if config.has_shared_expert {
        let shared_intermediate = config
            .shared_expert_intermediate_size
            .unwrap_or(config.expert_intermediate_size);
        Some(compute_local_ffn_shapes(&TPFFNConfig {
            tp_rank: config.tp_rank,
            tp_size: config.tp_size,
            intermediate_size: shared_intermediate,
            hidden_size: config.hidden_size,
            activation: config.activation,
        })?)
    } else {
        None
    };

    let needs_all_to_all = config.mode == MoEParallelMode::ExpertParallel && config.tp_size > 1;
    let needs_allreduce =
        config.mode == MoEParallelMode::WithinExpertSharding && config.tp_size > 1;

    Ok(TPMoEMetadata {
        tp_rank: config.tp_rank,
        mode: config.mode,
        expert_assignment: assignment,
        expert_ffn_meta,
        has_shared_expert: config.has_shared_expert,
        shared_expert_ffn_meta,
        router_weight_shape: [config.hidden_size, config.num_experts],
        router_replicated: true,
        experts_per_token: config.experts_per_token,
        needs_all_to_all,
        needs_allreduce,
    })
}

/// Compute MoE metadata for all ranks in a TP group.
pub fn compute_all_rank_moe_metadata(config: &TPMoEConfig) -> Result<Vec<TPMoEMetadata>> {
    let mut all_meta = Vec::with_capacity(config.tp_size);
    for rank in 0..config.tp_size {
        let rank_config = TPMoEConfig {
            tp_rank: rank,
            ..config.clone()
        };
        all_meta.push(compute_local_moe_shapes(&rank_config)?);
    }
    Ok(all_meta)
}

/// Verify that in expert-parallel mode, every expert is assigned to
/// exactly one rank.
///
/// **Note:** This function inspects the `expert_assignment` field, which
/// is only meaningful in [`MoEParallelMode::ExpertParallel`] mode.
/// In `WithinExpertSharding` mode, expert assignments are populated but
/// semantically irrelevant (all ranks process all experts); calling this
/// function in that mode will still succeed but does not validate the
/// intermediate-dimension sharding — use
/// [`super::parallel_ffn::verify_intermediate_coverage`] for that.
pub fn verify_expert_coverage(all_meta: &[TPMoEMetadata], num_experts: usize) -> Result<()> {
    let mut covered = vec![false; num_experts];
    for meta in all_meta {
        for &idx in &meta.expert_assignment.expert_indices {
            ensure!(
                idx < num_experts,
                "expert index {idx} out of range (num_experts={num_experts})"
            );
            ensure!(!covered[idx], "expert {idx} assigned to multiple ranks");
            covered[idx] = true;
        }
    }
    for (idx, &c) in covered.iter().enumerate() {
        ensure!(c, "expert {idx} not assigned to any rank");
    }
    Ok(())
}

#[cfg(test)]
#[path = "parallel_moe_tests.rs"]
mod tests;
