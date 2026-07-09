// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

use super::{DecodeStorageBackend, ServerConfig};
use crate::memory_estimate::PagedBudgetDirective;

#[test]
fn server_config_default_matches_llama_server_compatibility_defaults() {
    let config = ServerConfig::default();

    assert_eq!(config.timeout_seconds, 600);
    assert_eq!(config.context_size, 0);
    // Serving-throughput default: 4 concurrent decode slots (#628).
    assert_eq!(config.n_parallel, 4);
    assert!(config.enable_slots_endpoint);
    assert!(!config.enable_props_endpoint);
    assert!(!config.enable_metrics_endpoint);
    assert_eq!(config.default_temperature, 0.8);
    assert_eq!(config.default_top_p, 0.9);
    assert_eq!(config.default_top_k, 40);
    assert_eq!(config.default_min_p, 0.1);
    assert_eq!(config.default_repetition_penalty, 1.0);
    assert_eq!(config.default_repetition_context_size, 64);
    assert_eq!(config.default_max_tokens, 512);
    assert_eq!(config.default_dry_multiplier, 0.0);
    assert_eq!(config.default_dry_base, 1.75);
    assert_eq!(config.default_dry_allowed_length, 2);
    assert_eq!(config.default_dry_penalty_last_n, 0);
    assert_eq!(config.num_draft_tokens, 3);
    // Serving-throughput default: batched decode up to 4 sequences (#628).
    assert_eq!(config.max_batch_size, 4);
    assert_eq!(config.max_queue_depth, 1024);
    // Serving-throughput default: batched prefill up to 4 requests (#628).
    assert_eq!(config.max_batch_prefill, 4);
    assert!(!config.no_batch);
    assert_eq!(config.decode_storage_backend, DecodeStorageBackend::Auto);
    // Serving-throughput default guard: `auto` paged KV budget pairs with the
    // batched-decode default so admission sheds load instead of OOMing (#628).
    assert_eq!(config.kv_cache_budget, Some(PagedBudgetDirective::Auto));
}
