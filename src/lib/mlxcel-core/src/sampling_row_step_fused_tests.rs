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

//! Fused-batch eligibility of the per-row sampling step (#2169): derived from
//! the stage list, false for every stage the fused dispatch cannot run, and
//! identical to the hand-maintained gate it replaced.

use super::{RowSampler, SamplerStage, StageFusion};
use crate::generate::SamplingConfig;
use crate::sampling::TokenBiasMap;

fn seeded(seed: u64) -> SamplingConfig {
    SamplingConfig {
        temperature: 0.8,
        seed: Some(seed),
        ..SamplingConfig::default()
    }
}

/// The pre-#2169 hand-maintained gate, kept here as the oracle.
fn pre_2169_gate(config: &SamplingConfig) -> bool {
    !config.needs_token_history()
        && config.xtc_probability <= 0.0
        && config.effective_mirostat() == 0
        && !config.needs_extended_chain()
}

/// A stochastic config that runs `stage` (and the always-on chain stages).
fn config_running(stage: SamplerStage) -> SamplingConfig {
    let base = seeded(3);
    match stage {
        SamplerStage::TokenBias => {
            let mut bias = TokenBiasMap::new();
            bias.insert(7, -1.0);
            SamplingConfig {
                token_bias: bias,
                ..base
            }
        }
        SamplerStage::RepetitionPenalty => SamplingConfig {
            repetition_penalty: 1.1,
            ..base
        },
        SamplerStage::Dry => SamplingConfig {
            dry_multiplier: 0.8,
            ..base
        },
        SamplerStage::FrequencyPresencePenalty => SamplingConfig {
            presence_penalty: 0.4,
            ..base
        },
        SamplerStage::RowFilters => SamplingConfig {
            top_n_sigma: 1.0,
            ..base
        },
        SamplerStage::FusedChain => base,
        SamplerStage::Xtc => SamplingConfig {
            xtc_probability: 0.5,
            ..base
        },
        SamplerStage::Mirostat => SamplingConfig {
            mirostat: 2,
            ..base
        },
        SamplerStage::DynamicTemperature => SamplingConfig {
            dynatemp_range: 0.5,
            ..base
        },
        SamplerStage::MinKeep => SamplingConfig {
            min_keep: 3,
            ..base
        },
        SamplerStage::AdaptiveP => SamplingConfig {
            adaptive_target: 0.4,
            ..base
        },
    }
}

#[test]
fn fused_eligible_is_false_for_every_stage_the_dispatch_cannot_run() {
    let counters_off = crate::lang_bias_counters::scoped_override(false);
    for stage in SamplerStage::ALL {
        let config = config_running(stage);
        assert!(stage.is_active(&config), "{stage:?} is not active");
        let expected = stage.fusion() != StageFusion::PerRowOnly;
        assert_eq!(RowSampler::fused_eligible(&config), expected, "{stage:?}");
    }
    let biased = config_running(SamplerStage::TokenBias);
    assert!(RowSampler::fused_eligible(&biased));
    {
        let _on = crate::lang_bias_counters::scoped_override(true);
        assert!(
            !RowSampler::fused_eligible(&biased),
            "a biased row needs the per-row pre-bias argmax read while the counters are on"
        );
        assert!(RowSampler::fused_eligible(&config_running(
            SamplerStage::FusedChain
        )));
    }
    drop(counters_off);
}

#[test]
fn fused_eligible_matches_the_pre_2169_gate() {
    let mut bias = TokenBiasMap::new();
    bias.insert(4, -2.0);
    // Every combination of: temperature, repetition penalty, penalty window,
    // frequency penalty, DRY (off, zero window, full window), XTC, mirostat,
    // dynamic temperature, min_keep, adaptive-p, token bias.
    let axes = [2usize, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2];
    let total: usize = axes.iter().product();
    for combination in 0..total {
        let mut rest = combination;
        let mut pick = |n: usize| {
            let value = rest % n;
            rest /= n;
            value
        };
        let (dry_multiplier, dry_penalty_last_n) = [(0.0, 0), (0.8, 0), (0.8, usize::MAX)][pick(3)];
        let biased = pick(2) == 1;
        let config = SamplingConfig {
            temperature: [0.0, 0.8][pick(2)],
            repetition_penalty: [1.0, 1.1][pick(2)],
            penalty_last_n: [-1, 0, 8][pick(3)],
            frequency_penalty: [0.0, 0.3][pick(2)],
            dry_multiplier,
            dry_penalty_last_n,
            xtc_probability: [0.0, 0.5][pick(2)],
            mirostat: [0, 2][pick(2)],
            dynatemp_range: [0.0, 0.5][pick(2)],
            min_keep: [0, 3][pick(2)],
            adaptive_target: [-1.0, 0.4][pick(2)],
            token_bias: if biased {
                bias.clone()
            } else {
                TokenBiasMap::new()
            },
            ..SamplingConfig::default()
        };
        let oracle = pre_2169_gate(&config);
        {
            let _off = crate::lang_bias_counters::scoped_override(false);
            assert_eq!(RowSampler::fused_eligible(&config), oracle, "{config:?}");
        }
        {
            let _on = crate::lang_bias_counters::scoped_override(true);
            assert_eq!(
                RowSampler::fused_eligible(&config),
                oracle && !biased,
                "{config:?}"
            );
        }
    }
}
