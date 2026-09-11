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

use super::*;

#[test]
fn default_shard_config() {
    let cfg = ShardConfig::default();
    assert_eq!(cfg.tp_size, 1);
    assert_eq!(cfg.moe_mode, MoeShardMode::ExpertParallel);
    assert_eq!(cfg.embedding_mode, EmbeddingMode::Replicated);
    assert_eq!(cfg.lm_head_mode, EmbeddingMode::Replicated);
}

#[test]
fn shard_config_with_tp_size() {
    let cfg = ShardConfig::with_tp_size(4);
    assert_eq!(cfg.tp_size, 4);
    assert_eq!(cfg.moe_mode, MoeShardMode::ExpertParallel);
}

#[test]
fn shard_config_validate_valid() {
    assert!(ShardConfig::with_tp_size(1).validate().is_ok());
    assert!(ShardConfig::with_tp_size(2).validate().is_ok());
    assert!(ShardConfig::with_tp_size(4).validate().is_ok());
    assert!(ShardConfig::with_tp_size(8).validate().is_ok());
}

#[test]
fn shard_config_validate_invalid_zero() {
    let cfg = ShardConfig {
        tp_size: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn shard_config_validate_invalid_non_power_of_two() {
    let cfg = ShardConfig::with_tp_size(3);
    assert!(cfg.validate().is_err());

    let cfg = ShardConfig::with_tp_size(6);
    assert!(cfg.validate().is_err());
}

#[test]
fn moe_shard_mode_parse() {
    assert_eq!(
        "expert_parallel".parse::<MoeShardMode>().unwrap(),
        MoeShardMode::ExpertParallel
    );
    assert_eq!(
        "ep".parse::<MoeShardMode>().unwrap(),
        MoeShardMode::ExpertParallel
    );
    assert_eq!(
        "within_expert".parse::<MoeShardMode>().unwrap(),
        MoeShardMode::WithinExpert
    );
    assert_eq!(
        "within".parse::<MoeShardMode>().unwrap(),
        MoeShardMode::WithinExpert
    );
    assert!("unknown".parse::<MoeShardMode>().is_err());
}

#[test]
fn embedding_mode_parse() {
    assert_eq!(
        "vocab_parallel".parse::<EmbeddingMode>().unwrap(),
        EmbeddingMode::VocabParallel
    );
    assert_eq!(
        "vp".parse::<EmbeddingMode>().unwrap(),
        EmbeddingMode::VocabParallel
    );
    assert_eq!(
        "replicated".parse::<EmbeddingMode>().unwrap(),
        EmbeddingMode::Replicated
    );
    assert_eq!(
        "full".parse::<EmbeddingMode>().unwrap(),
        EmbeddingMode::Replicated
    );
    assert!("unknown".parse::<EmbeddingMode>().is_err());
}

#[test]
fn moe_shard_mode_display() {
    assert_eq!(MoeShardMode::ExpertParallel.to_string(), "expert_parallel");
    assert_eq!(MoeShardMode::WithinExpert.to_string(), "within_expert");
}

#[test]
fn embedding_mode_display() {
    assert_eq!(EmbeddingMode::VocabParallel.to_string(), "vocab_parallel");
    assert_eq!(EmbeddingMode::Replicated.to_string(), "replicated");
}
