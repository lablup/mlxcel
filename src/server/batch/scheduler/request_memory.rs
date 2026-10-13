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
use crate::execution::request_memory::{self, RequestMemoryRejection};

impl BatchScheduler {
    /// Narrow policy scope; unsupported backends, speculative and other model
    /// families keep their current admission behavior.
    pub(super) fn request_memory_policy_enabled(&self) -> bool {
        let metal = cfg!(all(target_os = "macos", feature = "metal"))
            && mlxcel_core::default_device_is_gpu();
        #[cfg(test)]
        let metal = metal || self.request_memory_test_inputs.is_some();
        metal
            && self.active_batch.max_size() == 1
            && matches!(self.engine.model(), LoadedModel::NemotronH(_))
            && matches!(
                self.speculative_dispatch,
                crate::server::SpeculativeDispatch::Disabled
            )
    }

    /// Final admission occurs on the scheduler thread, after the preceding
    /// single-slot request is finalized. No optimistic prefix credit is taken:
    /// snapshots remain resident, but prefix restoration is skipped at enqueue.
    pub(super) fn check_request_memory(
        &self,
        seq: &SequenceInfo,
    ) -> Result<(), RequestMemoryRejection> {
        if !self.request_memory_policy_enabled()
            || !seq.images.is_empty()
            || !seq.audio.is_empty()
            || seq.vlm_embeddings.is_some()
        {
            return Ok(());
        }
        let LoadedModel::NemotronH(model) = self.engine.model() else {
            return Ok(());
        };
        let (config, element_bytes) = model.request_memory_config();
        let plan = self.prefill_plan_for(seq);
        let largest = plan
            .pieces()
            .iter()
            .map(|p| p.padded_len as u64)
            .max()
            .unwrap_or(0);
        let stop_context = if self.context_retention.context_shift {
            None
        } else {
            self.max_kv_size.map(|n| n as u64)
        };
        let retained = request_memory::retained_tokens(
            seq.prompt_tokens.len() as u64,
            seq.max_tokens as u64,
            stop_context,
        );
        let memory = mlxcel_core::memory::snapshot();
        #[cfg(test)]
        let memory = self
            .request_memory_test_inputs
            .map_or(memory, |(memory, _)| memory);
        let estimate = retained.and_then(|tokens| {
            request_memory::estimate(config, largest, tokens.max(largest), element_bytes, memory)
        });
        // Reading the recommendation does not reconstruct the configured wired
        // cap: runtime explicitly records the cap it actually applied.
        let budget = request_memory::budget_override()
            .ok()
            .and_then(|override_bytes| {
                request_memory::resolve_budget(
                    mlxcel_core::get_wired_limit() as u64,
                    crate::execution::runtime::applied_wired_limit(),
                    memory.limit_bytes,
                    override_bytes,
                )
            });
        #[cfg(test)]
        let budget = self
            .request_memory_test_inputs
            .map_or(budget, |(_, budget)| budget);
        let result = request_memory::check(estimate, budget);
        tracing::info!(
            seq_id = %seq.seq_id,
            prompt_tokens = seq.prompt_tokens.len(),
            output_tokens = seq.max_tokens,
            largest_forward_tokens = largest,
            estimated_bytes = estimate.map(|e| e.total_bytes).unwrap_or(u64::MAX),
            budget_bytes = budget.unwrap_or(0),
            resident_bytes = memory.active_bytes,
            allocator_cache_bytes = memory.cache_bytes,
            estimate = ?estimate,
            admitted = result.is_ok(),
            "request memory admission"
        );
        result.map(|_| ())
    }
}
