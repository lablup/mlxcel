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

use super::ModelArgs;

#[test]
fn moondream3_model_args_fill_default_moe_and_attention_dimensions() {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(args.dim, 2048);
    assert_eq!(args.n_heads, 32);
    assert_eq!(args.n_kv_heads, 32);
    assert_eq!(args.head_dim(), 64);
    assert_eq!(args.group_size, 128);
    assert_eq!(args.bits, 4);

    let moe = args
        .moe
        .expect("Moondream3 default MoE config should exist");
    assert_eq!(moe.start_layer, 4);
    assert_eq!(moe.num_experts, 64);
    assert_eq!(moe.experts_per_token, 8);
}
