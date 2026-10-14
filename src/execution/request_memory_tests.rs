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
use mlxcel_core::prefill_plan::{PrefillCaps, PrefillPlan};

fn config() -> NemotronHConfig {
    let mut c: NemotronHConfig = serde_json::from_value(serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":128, "hidden_size":64,
        "intermediate_size":128, "num_hidden_layers":99, "num_attention_heads":4,
        "num_key_value_heads":2, "mamba_num_heads":4, "mamba_head_dim":16,
        "ssm_state_size":8, "conv_kernel":4, "n_groups":2, "chunk_size":32,
        "layers_block_type":["mamba","attention","mlp"]
    }))
    .unwrap();
    c.post_init().unwrap();
    c
}

fn estimated(t: u64, output: u64, active: u64, cache: u64) -> RequestMemoryEstimate {
    estimate(
        &config(),
        t,
        retained_tokens(t, output, None).unwrap(),
        2,
        MemorySnapshot {
            active_bytes: active,
            cache_bytes: cache,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn request_memory_budget_boundaries_and_output_growth() {
    let e = estimated(64, 1, 100, 0);
    assert!(check(Some(e), Some(e.total_bytes)).is_ok());
    assert!(check(Some(e), Some(e.total_bytes + 1)).is_ok());
    assert!(check(Some(e), Some(e.total_bytes - 1)).is_err());
    let long = estimated(64, 1024, 100, 0);
    assert!(long.kv_bytes > e.kv_bytes);
    assert!(long.total_bytes > e.total_bytes);
}

#[test]
fn request_memory_cache_accounting_counts_resident_once_not_allocator_cache() {
    let cold = estimated(64, 1, 100, 0);
    let free_cache = estimated(64, 1, 100, u64::MAX);
    assert_eq!(cold, free_cache);
    let retained_snapshot = estimated(64, 1, 500, 0);
    assert_eq!(retained_snapshot.total_bytes - cold.total_bytes, 400);
    assert_eq!(retained_snapshot.kv_bytes, cold.kv_bytes);
}

#[test]
fn request_memory_budget_only_lowers_known_limits() {
    assert_eq!(resolve_budget(1000, 700, 900, None), Some(700));
    assert_eq!(resolve_budget(1000, 700, 900, Some(5000)), Some(700));
    assert_eq!(resolve_budget(1000, 700, 900, Some(500)), Some(500));
    assert_eq!(resolve_budget(0, 0, 0, Some(500)), None);
    assert_eq!(resolve_budget(1000, 0, 0, None), Some(1000));
    assert!(check(Some(estimated(1, 1, 0, 0)), None).is_err());
    assert!(check(None, Some(u64::MAX)).is_err());
}

#[test]
fn request_memory_invalid_explicit_budgets_fail_closed() {
    for value in [
        "0",
        "0G",
        "none",
        "",
        "-1",
        "-2G",
        "NaNG",
        "infG",
        "1e300G",
        "1.5",
        "18446744073709551616",
    ] {
        assert!(parse_budget(value).is_err(), "{value}");
    }
    assert_eq!(parse_budget("1.5G").unwrap(), 1536 * 1024 * 1024);
    assert_eq!(parse_budget("1024").unwrap(), 1024);
}

#[test]
fn request_memory_overflow_and_invalid_dimensions_fail_closed() {
    assert!(retained_tokens(1, u64::MAX, None).is_none());
    assert_eq!(retained_tokens(10, u64::MAX, Some(128)), Some(128));
    assert!(estimate(&config(), u64::MAX, u64::MAX, 2, Default::default()).is_none());
    assert!(
        estimate(
            &config(),
            64,
            65,
            2,
            MemorySnapshot {
                active_bytes: u64::MAX,
                ..Default::default()
            }
        )
        .is_none()
    );
    let mut c = config();
    c.mamba_head_dim = usize::MAX;
    assert!(estimate(&c, 64, 65, 2, Default::default()).is_none());
    c = config();
    c.num_attention_heads = 0;
    assert!(estimate(&c, 64, 65, 2, Default::default()).is_none());
    assert!(estimate(&config(), 64, 65, 0, Default::default()).is_none());
}

#[test]
fn request_memory_context_stop_and_untrimmable_shift() {
    assert_eq!(retained_tokens(100, 1000, Some(128)), Some(128));
    // No trim credit for Nemotron's model-owned caches when shifting is on.
    assert_eq!(retained_tokens(100, 1000, None), Some(1100));
    assert_eq!(retained_tokens(100, 0, None), Some(101));
    assert_eq!(retained_tokens(128, 1, Some(128)), None);
}

#[test]
fn request_memory_normalized_pattern_and_loaded_dtype() {
    let c = config();
    assert_eq!(c.num_hidden_layers, 3);
    let half = estimate(&c, 64, 65, 2, Default::default()).unwrap();
    let full = estimate(&c, 64, 65, 4, Default::default()).unwrap();
    assert_eq!(full.kv_bytes, half.kv_bytes * 2);
    assert_eq!(half.ssd_bytes, full.ssd_bytes); // SSD's float32 boundary.
    assert!(full.total_bytes > half.total_bytes);
}

#[test]
fn request_memory_ssd_internal_chunk_is_not_outer_forward() {
    let mut c = config();
    let small = estimate(&c, 1024, 1025, 2, Default::default()).unwrap();
    c.chunk_size = 128;
    let large = estimate(&c, 1024, 1025, 2, Default::default()).unwrap();
    assert!(large.ssd_bytes > small.ssd_bytes);
    assert_eq!(small.activation_bytes, large.activation_bytes);
}

#[test]
fn request_memory_prefill_plan_honors_model_opt_out() {
    let caps = PrefillCaps {
        supports_chunked_prefill: false,
        ..Default::default()
    };
    let plan = PrefillPlan::new(8192, 256, caps);
    assert!(plan.is_single_pass());
    assert_eq!(plan.pieces()[0].padded_len, 8192);
    let actual = estimated(8192, 1, 0, 0);
    let misleading_global_chunk = estimated(256, 1, 0, 0);
    assert!(actual.total_bytes > misleading_global_chunk.total_bytes * 10);
}
