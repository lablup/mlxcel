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

//! Shared decode-loop setup policy.
//!
//! These helpers keep pre-generation setup out of the decode loops (the
//! engine's raw-completion client, the speculative generators, the scheduler)
//! so they can focus on token flow rather than seed and EOS details.

use crate::ffi;
use crate::generate::SamplingConfig;

/// Used by: the engine's raw-completion client, SpeculativeGenerator, BatchScheduler
pub fn seed_rng_if_needed(sampling: &SamplingConfig) {
    if let Some(seed) = sampling.seed {
        ffi::random_seed(seed);
    }
}

/// Used by: the engine's raw-completion client, SpeculativeGenerator, BatchScheduler
pub fn merged_eos_token_ids(model_eos: Vec<i32>, stop_token_ids: &[i32]) -> Vec<i32> {
    let mut eos_tokens = model_eos;
    for &id in stop_token_ids {
        if !eos_tokens.contains(&id) {
            eos_tokens.push(id);
        }
    }
    eos_tokens
}

/// Used by: the engine's raw-completion client, SpeculativeGenerator, BatchScheduler
pub fn initial_token_history(prompt_tokens: &[i32], needs_history: bool) -> Vec<i32> {
    if needs_history {
        prompt_tokens.to_vec()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{initial_token_history, merged_eos_token_ids, seed_rng_if_needed};
    use crate::generate::SamplingConfig;

    #[test]
    fn merged_eos_token_ids_appends_only_missing_stop_tokens() {
        let merged = merged_eos_token_ids(vec![1, 2], &[2, 3, 4]);
        assert_eq!(merged, vec![1, 2, 3, 4]);
    }

    #[test]
    fn initial_token_history_only_clones_when_needed() {
        assert_eq!(initial_token_history(&[10, 11], true), vec![10, 11]);
        assert!(initial_token_history(&[10, 11], false).is_empty());
    }

    #[test]
    fn seed_rng_if_needed_accepts_absent_seed() {
        seed_rng_if_needed(&SamplingConfig::default());
    }
}
