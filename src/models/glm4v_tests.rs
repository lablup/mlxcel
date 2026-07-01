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

//! Unit tests for the GLM-4V text backbone config + MRoPE bookkeeping.

use super::{Glm4vMRoPE, Glm4vTextConfig};

fn sample_text_config_json() -> &'static str {
    r#"{
        "model_type": "glm4v_text",
        "hidden_size": 4096,
        "num_hidden_layers": 40,
        "intermediate_size": 13696,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "vocab_size": 151552,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "attention_bias": true,
        "rope_scaling": {"rope_type": "default", "mrope_section": [8, 12, 12]},
        "eos_token_id": [151329, 151336, 151338, 151348]
    }"#
}

#[test]
fn parses_text_config_defaults() {
    let config: Glm4vTextConfig = serde_json::from_str(sample_text_config_json()).unwrap();
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 2);
    assert!(config.attention_bias);
    assert_eq!(config.head_dim(), 128);
    assert_eq!(config.mrope_section(), vec![8, 12, 12]);
    // partial_rotary_factor 0.5 over head_dim 128 -> 64 rotary dims.
    assert_eq!(config.rope_dims(), 64);
    assert_eq!(
        config.eos_token_id,
        Some(vec![151329, 151336, 151338, 151348])
    );
}

#[test]
fn text_config_missing_rope_scaling_falls_back_to_default_section() {
    let json = r#"{
        "hidden_size": 4096,
        "num_hidden_layers": 40,
        "intermediate_size": 13696,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "vocab_size": 151552
    }"#;
    let config: Glm4vTextConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.mrope_section(), vec![8, 12, 12]);
    assert!((config.partial_rotary_factor - 0.5).abs() < 1e-6);
    assert!(config.attention_bias);
}

#[test]
fn mrope_axis_selector_is_chunked_by_section() {
    // head_dim 128, rope_dims 64 -> 32 rotary pairs; section [8, 12, 12] must
    // assign pair->axis as [T x8, H x12, W x12].
    let mrope = Glm4vMRoPE::new(128, 10000.0, 64, &[8, 12, 12]);
    let mut expected: Vec<i32> = Vec::new();
    expected.extend(std::iter::repeat_n(0, 8));
    expected.extend(std::iter::repeat_n(1, 12));
    expected.extend(std::iter::repeat_n(2, 12));
    assert_eq!(mrope.axis_selector, expected);
    // 32 inverse frequencies, monotonically decreasing.
    assert_eq!(mrope.inv_freq.len(), 32);
    assert!((mrope.inv_freq[0] - 1.0).abs() < 1e-6);
    for w in mrope.inv_freq.windows(2) {
        assert!(w[1] < w[0]);
    }
}
