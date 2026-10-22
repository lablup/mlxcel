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

//! The Gemma 4 vision GeGLU activation follows `hidden_activation` (#2423).

use super::Gemma4VisionConfig;
use mlxcel_core::utils::GeluActivation;

fn config(hidden_activation: Option<&str>) -> Gemma4VisionConfig {
    let mut value = serde_json::json!({
        "model_type": "gemma4_vision",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "patch_size": 16,
    });
    if let Some(act) = hidden_activation {
        value["hidden_activation"] = serde_json::Value::String(act.to_string());
    }
    serde_json::from_value(value).unwrap()
}

#[test]
fn absent_or_declared_tanh_selects_tanh_evaluated_in_f32() {
    // Every local Gemma 4 ViT config (gemma4, diffusion_gemma) declares
    // `gelu_pytorch_tanh`; an absent key is the Hugging Face default, the same
    // value. (gemma-4-12b-it-4bit also omits it, but `gemma4_unified` is
    // encoder-free and never parses a `Gemma4VisionConfig`.)
    assert_eq!(
        config(None).mlp_activation().unwrap(),
        GeluActivation::TanhF32
    );
    assert_eq!(
        config(Some("gelu_pytorch_tanh")).mlp_activation().unwrap(),
        GeluActivation::TanhF32
    );
}

#[test]
fn erf_name_selects_erf_and_a_non_gelu_name_is_rejected() {
    assert_eq!(
        config(Some("gelu")).mlp_activation().unwrap(),
        GeluActivation::Erf
    );
    assert!(config(Some("silu")).mlp_activation().is_err());
}
