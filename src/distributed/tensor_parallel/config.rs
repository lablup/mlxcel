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

//! Tensor-parallel configuration types.
//!
//! [`ShardConfig`] captures all user-configurable sharding options that affect
//! how the shard plan is generated. It is built from CLI arguments and the
//! model configuration file.

use serde::{Deserialize, Serialize};

/// How MoE expert weights are distributed across TP ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum MoeShardMode {
    /// Whole experts are assigned to ranks round-robin.
    /// Expert `i` lives entirely on rank `i % tp_size`.
    /// Preferred when `num_experts >= tp_size`.
    #[default]
    ExpertParallel,

    /// Each expert's internal FFN weights are sharded across all ranks
    /// (column-parallel gate/up, row-parallel down), just like the non-MoE FFN.
    /// Preferred when `num_experts < tp_size`.
    WithinExpert,
}

impl std::fmt::Display for MoeShardMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExpertParallel => write!(f, "expert_parallel"),
            Self::WithinExpert => write!(f, "within_expert"),
        }
    }
}

impl std::str::FromStr for MoeShardMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "expert_parallel" | "expert" | "ep" => Ok(Self::ExpertParallel),
            "within_expert" | "within" | "we" => Ok(Self::WithinExpert),
            other => anyhow::bail!(
                "unknown MoE shard mode '{other}'; expected: expert_parallel, within_expert"
            ),
        }
    }
}

/// How embedding and LM head layers are sharded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum EmbeddingMode {
    /// Shard the vocabulary dimension across ranks. Saves memory but requires
    /// all-gather during embedding lookup and all-gather for LM head logits.
    VocabParallel,

    /// Fully replicate embedding/LM head on every rank. Simpler but uses more
    /// memory per rank. Preferred for small vocabularies.
    #[default]
    Replicated,
}

impl std::fmt::Display for EmbeddingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VocabParallel => write!(f, "vocab_parallel"),
            Self::Replicated => write!(f, "replicated"),
        }
    }
}

impl std::str::FromStr for EmbeddingMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "vocab_parallel" | "vocab" | "vp" => Ok(Self::VocabParallel),
            "replicated" | "full" | "none" => Ok(Self::Replicated),
            other => anyhow::bail!(
                "unknown embedding mode '{other}'; expected: vocab_parallel, replicated"
            ),
        }
    }
}

/// User-configurable tensor-parallel sharding options.
///
/// Built from CLI arguments (`--tp-size`, `--tp-moe-mode`, etc.) and optionally
/// refined by model config inspection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShardConfig {
    /// Number of tensor-parallel ranks. Must be >= 1.
    pub tp_size: usize,

    /// How MoE experts are distributed. Defaults to `ExpertParallel`.
    pub moe_mode: MoeShardMode,

    /// How the embedding table is sharded. Defaults to `Replicated`.
    pub embedding_mode: EmbeddingMode,

    /// How the LM head is sharded. Defaults to `Replicated`.
    pub lm_head_mode: EmbeddingMode,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            tp_size: 1,
            moe_mode: MoeShardMode::default(),
            embedding_mode: EmbeddingMode::default(),
            lm_head_mode: EmbeddingMode::default(),
        }
    }
}

impl ShardConfig {
    /// Create a config for the given TP size with default options.
    pub fn with_tp_size(tp_size: usize) -> Self {
        Self {
            tp_size,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.tp_size >= 1,
            "tp_size must be >= 1, got {}",
            self.tp_size
        );
        anyhow::ensure!(
            self.tp_size.is_power_of_two(),
            "tp_size must be a power of 2, got {}",
            self.tp_size
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
