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

use super::ModelArgs;

#[test]
fn moondream2_model_args_fill_phi_style_defaults() {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(args.dim, 2048);
    assert_eq!(args.ff_dim, 8192);
    assert_eq!(args.n_layers, 24);
    assert_eq!(args.vocab_size, 51200);
    assert_eq!(args.n_heads, 32);
    assert_eq!(args.n_kv_heads, 32);
    assert_eq!(args.head_dim(), 64);
    // The moondream2 tokenizer uses `<|endoftext|>` (id 50256) as bos/eos, not
    // Moondream3's id 0. A `0` eos halts generation on the first sampled token.
    assert_eq!(args.eos_token_id, 50256);
    assert_eq!(args.bos_token_id, 50256);
}

#[test]
fn moondream2_partial_rotary_covers_half_the_head() {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({})).unwrap();
    // partial_rotary_factor = 0.5, head_dim = 64 -> 32 rotary dims.
    assert_eq!(args.partial_rotary_factor, 0.5);
    assert_eq!(args.rope_dims(), 32);
    assert_eq!(args.rope_theta, 10000.0);
}

#[test]
fn moondream2_fused_qkv_width_matches_reference() {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({})).unwrap();
    // (n_heads + 2 * n_kv_heads) * head_dim = (32 + 64) * 64 = 6144.
    assert_eq!(args.qkv_dim(), 6144);
}

#[test]
fn moondream2_model_args_accept_explicit_overrides() {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "moondream2",
        "dim": 1024,
        "n_heads": 16,
        "n_kv_heads": 16,
        "partial_rotary_factor": 0.25,
    }))
    .unwrap();
    assert_eq!(args.dim, 1024);
    assert_eq!(args.head_dim(), 64);
    assert_eq!(args.rope_dims(), 16);
    assert_eq!(args.qkv_dim(), (16 + 32) * 64);
}
