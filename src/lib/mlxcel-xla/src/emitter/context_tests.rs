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

use super::{Config, emit_decode, emit_decode_ragged, emit_prefill};

fn tiny_config(context_capacity: usize) -> Config {
    Config {
        context_capacity,
        hidden: 8,
        inter: 16,
        n_layers: 2,
        n_q: 2,
        n_kv: 1,
        head_dim: 4,
        vocab: 16,
        ..Config::llama_3_2_1b()
    }
}

fn assert_capacity_shapes(context_capacity: usize) {
    let cfg = tiny_config(context_capacity);
    let prefill = emit_prefill(&cfg, false);
    let decode = emit_decode(&cfg, false);
    let ragged = emit_decode_ragged(&cfg, 4, false);
    let single_cache = format!("tensor<2x{context_capacity}x1x4xf32>");
    let batch_cache = format!("tensor<4x2x{context_capacity}x1x4xf32>");
    let prompt = format!("tensor<{context_capacity}xi32>");
    let causal_mask = format!("tensor<{context_capacity}x{context_capacity}xf32>");
    let ragged_mask = format!("tensor<4x{context_capacity}xf32>");
    let rope = format!("tensor<{context_capacity}x4xf32>");

    assert!(
        prefill.matches(&prompt).count() >= 2,
        "tokens and positions use {prompt}"
    );
    assert!(
        prefill.matches(&single_cache).count() >= 3,
        "prefill returns matching K/V"
    );
    assert!(
        prefill.contains(&causal_mask),
        "prefill mask uses {causal_mask}"
    );
    assert!(
        prefill.contains(&rope),
        "prefill position table uses {rope}"
    );

    assert!(
        decode.matches(&single_cache).count() >= 5,
        "decode input/output K/V agree"
    );
    assert!(decode.contains(&rope), "decode position table uses {rope}");

    assert!(
        ragged.matches(&batch_cache).count() >= 5,
        "ragged input/output K/V agree"
    );
    assert!(
        ragged.contains(&ragged_mask),
        "ragged mask uses {ragged_mask}"
    );
    assert!(ragged.contains(&rope), "ragged position table uses {rope}");
}

#[test]
fn graph_schema_is_consistent_at_compatibility_capacity() {
    assert_capacity_shapes(256);
}

#[test]
fn graph_schema_is_consistent_at_multimodal_capacity() {
    assert_capacity_shapes(1024);
}
