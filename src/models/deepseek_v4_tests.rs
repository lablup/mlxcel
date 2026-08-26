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

//! Unit tests for the DeepSeek-V4 port. Filled in alongside the components.

use super::*;

#[test]
fn deepseek_v4_default_compress_ratios_match_reference_post_init() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type":"deepseek_v4","vocab_size":1000,"hidden_size":64,
            "num_hidden_layers":5,"num_attention_heads":8,"head_dim":16,
            "qk_rope_head_dim":4,"index_head_dim":8}"#,
    )
    .expect("parse");
    let args = args.normalized().expect("normalize");
    assert_eq!(args.compress_ratios, vec![0, 128, 4, 128, 0]);
}
