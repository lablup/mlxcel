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

//! Core types for tensor-parallel weight sharding.
//!
//! Defines the sharding strategies (column-parallel, row-parallel, expert-parallel,
//! vocabulary-parallel, replicated) and the per-layer shard plan that maps each
//! weight tensor to its sharding metadata.

use serde::{Deserialize, Serialize};

/// How a single weight tensor is sharded across TP ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ShardStrategy {
    /// Shard along the output dimension (columns). Each rank gets a slice of
    /// the output features. Used for attention Q/K/V projections and FFN gate/up.
    ColumnParallel,

    /// Shard along the input dimension (rows). Each rank holds a partial sum
    /// that must be all-reduced. Used for attention O and FFN down projections.
    RowParallel,

    /// Shard whole experts across ranks. Expert `i` lives on rank `i % tp_size`.
    /// Used for MoE expert FFN weights when expert count >= tp_size.
    ExpertParallel,

    /// Shard along the vocabulary dimension. Each rank holds a contiguous slice
    /// of the vocabulary. Requires all-gather for full logits.
    VocabParallel,

    /// The weight is fully replicated on every rank. No sharding or communication.
    Replicated,
}

impl std::fmt::Display for ShardStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColumnParallel => write!(f, "column_parallel"),
            Self::RowParallel => write!(f, "row_parallel"),
            Self::ExpertParallel => write!(f, "expert_parallel"),
            Self::VocabParallel => write!(f, "vocab_parallel"),
            Self::Replicated => write!(f, "replicated"),
        }
    }
}

/// Communication pattern required after the sharded operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CommPattern {
    /// No communication needed (replicated weights or column-parallel without merge).
    None,
    /// All-reduce (sum) partial results across ranks. Used after row-parallel matmul.
    AllReduce,
    /// All-gather partial outputs across ranks. Used after vocab-parallel embedding
    /// lookup or column-parallel output that needs full concatenation.
    AllGather,
    /// Reduce-scatter for combined reduce + scatter. Used in optimized pipelines.
    ReduceScatter,
}

impl std::fmt::Display for CommPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::AllReduce => write!(f, "all_reduce"),
            Self::AllGather => write!(f, "all_gather"),
            Self::ReduceScatter => write!(f, "reduce_scatter"),
        }
    }
}

/// Sharding plan for a single weight tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerShardPlan {
    /// Weight name pattern (e.g., "layers.{}.self_attn.q_proj.weight").
    /// The `{}` placeholder represents the layer index.
    pub weight_pattern: String,

    /// The sharding strategy for this weight.
    pub strategy: ShardStrategy,

    /// Axis along which to shard (0 = rows, 1 = columns). Only meaningful for
    /// ColumnParallel, RowParallel, and VocabParallel.
    pub shard_axis: usize,

    /// Communication required after the forward pass through this layer.
    pub comm_pattern: CommPattern,
}

/// Complete sharding plan for a model under tensor parallelism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelShardPlan {
    /// Number of TP ranks this plan targets.
    pub tp_size: usize,

    /// Number of transformer/SSM layers in the model.
    pub num_layers: usize,

    /// Per-layer weight shard plans. Patterns containing `{}` are expanded
    /// for each layer index `0..num_layers`.
    pub layer_plans: Vec<LayerShardPlan>,

    /// How the embedding table is sharded.
    pub embedding_strategy: ShardStrategy,

    /// How the LM head (output projection) is sharded.
    pub lm_head_strategy: ShardStrategy,

    /// Architecture name this plan was generated for.
    pub architecture: String,
}

impl ModelShardPlan {
    /// Expand layer patterns into concrete weight names.
    ///
    /// For each `LayerShardPlan` whose `weight_pattern` contains `{}`, produces
    /// `num_layers` entries with the layer index substituted. Non-templated
    /// patterns are returned as-is.
    pub fn expand_weight_names(&self) -> Vec<(String, &LayerShardPlan)> {
        let mut result = Vec::new();
        for plan in &self.layer_plans {
            if plan.weight_pattern.contains("{}") {
                for layer_idx in 0..self.num_layers {
                    let name = plan.weight_pattern.replace("{}", &layer_idx.to_string());
                    result.push((name, plan));
                }
            } else {
                result.push((plan.weight_pattern.clone(), plan));
            }
        }
        result
    }

    /// Look up the shard plan for a specific weight name.
    ///
    /// Returns `None` for weights not covered by the plan (they should be
    /// replicated by default).
    pub fn plan_for_weight(&self, weight_name: &str) -> Option<&LayerShardPlan> {
        // Try exact match first.
        for plan in &self.layer_plans {
            if !plan.weight_pattern.contains("{}") && plan.weight_pattern == weight_name {
                return Some(plan);
            }
        }
        // Try pattern match: replace `{}` with a regex-like layer index check.
        for plan in &self.layer_plans {
            if plan.weight_pattern.contains("{}") {
                // Split pattern on `{}` and check prefix/suffix.
                let parts: Vec<&str> = plan.weight_pattern.splitn(2, "{}").collect();
                if parts.len() == 2
                    && let Some(rest) = weight_name.strip_prefix(parts[0])
                {
                    // The middle should be a numeric layer index.
                    if let Some(suffix) = rest.strip_suffix(parts[1]) {
                        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                            return Some(plan);
                        }
                    } else if parts[1].is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                        return Some(plan);
                    }
                }
            }
        }
        None
    }

    /// Total number of weight patterns (before layer expansion).
    pub fn num_patterns(&self) -> usize {
        self.layer_plans.len()
    }

    /// Summary string for logging.
    pub fn summary(&self) -> String {
        let col = self
            .layer_plans
            .iter()
            .filter(|p| p.strategy == ShardStrategy::ColumnParallel)
            .count();
        let row = self
            .layer_plans
            .iter()
            .filter(|p| p.strategy == ShardStrategy::RowParallel)
            .count();
        let expert = self
            .layer_plans
            .iter()
            .filter(|p| p.strategy == ShardStrategy::ExpertParallel)
            .count();
        let repl = self
            .layer_plans
            .iter()
            .filter(|p| p.strategy == ShardStrategy::Replicated)
            .count();
        format!(
            "TP-{} plan for {} ({} layers): {} col-parallel, {} row-parallel, \
             {} expert-parallel, {} replicated, embedding={}, lm_head={}",
            self.tp_size,
            self.architecture,
            self.num_layers,
            col,
            row,
            expert,
            repl,
            self.embedding_strategy,
            self.lm_head_strategy,
        )
    }
}

#[cfg(test)]
#[path = "shard_strategy_tests.rs"]
mod tests;
