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

//! Raon-1's per-sequence RDA recurrent state in the KV estimate (issue
//! #3457): 18 RDA layers hold a float32 `[H, D, D]` state and a bf16 conv
//! window each, about 48.8 MB per sequence on Raon-1 mini, which context
//! sizing and prompt-cache budgets must see.

use serde_json::Value;

use super::*;

fn raon1_mini_config() -> Value {
    serde_json::from_str(include_str!("../models/raon1_mini_config_fixture.json"))
        .expect("fixture parses")
}

/// 18 x (40 x 128 x 128 x 4 + 3 x 3 x 40 x 128 x 2): the f32 delta-rule state
/// plus the `kernel - 1` raw q|k|v rows of the conv window, per sequence.
const RAON1_MINI_RDA_STATE_BYTES: u64 = 18 * (40 * 128 * 128 * 4 + 3 * 3 * 40 * 128 * 2);

/// The latent pair of the seven MLA layers, `(512 + 64)` fp16 per token.
fn mla_latent_bytes(ctx: u64, batch: u64) -> u64 {
    7 * (512 + 64) * 2 * ctx * batch
}

#[test]
fn raon1_mini_counts_its_rda_state_once_per_sequence() {
    let cfg = raon1_mini_config();
    assert_eq!(RAON1_MINI_RDA_STATE_BYTES, 47_185_920 + 1_658_880);
    for batch in [1, 4] {
        let e =
            estimate_kv_arch_from_config(&cfg, 2048, KVCacheMode::Fp16, batch).expect("estimate");
        assert_eq!(e.kind, KvArchKind::MlaCompressed);
        assert_eq!(
            e.total_bytes,
            mla_latent_bytes(2048, batch) + batch * RAON1_MINI_RDA_STATE_BYTES,
            "batch {batch}"
        );
        // The state does not grow with the context.
        assert_eq!(
            e.marginal_bytes_per_token,
            7 * (512 + 64) * 2 * batch,
            "batch {batch}"
        );
        assert!(e.detail.contains("RDA recurrent state"), "{}", e.detail);
    }
}

#[test]
fn a_config_without_rda_config_adds_no_recurrent_state() {
    let mut cfg = raon1_mini_config();
    let obj = cfg.as_object_mut().unwrap();
    obj.remove("rda_config");
    let e = estimate_kv_arch_from_config(&cfg, 2048, KVCacheMode::Fp16, 1).expect("estimate");
    assert!(!e.detail.contains("RDA"), "{}", e.detail);
    assert_eq!(e.total_bytes, 25 * (512 + 64) * 2 * 2048);
}
