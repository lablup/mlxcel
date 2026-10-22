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

//! Vision model configuration types
//!
//! Deserialization for vision_config and projector config from model config.json

use serde::{Deserialize, Deserializer};

/// Deserialize null as default value (serde #[serde(default)] only handles missing fields)
fn null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Option::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}

/// Activation selected by a SigLIP/CLIP checkpoint.
///
/// Existing and unknown values retain mlxcel's exact-erf GELU behavior. The
/// explicit `gelu_pytorch_tanh` variant follows the Hugging Face polynomial.
/// A missing key also parses as `ExactGelu`; the loaders that have been
/// measured against their checkpoints resolve that case to the Hugging Face
/// class default with [`VisionConfig::default_missing_hidden_act_to_siglip`].
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
pub enum VisionHiddenActivation {
    #[serde(rename = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
    #[default]
    #[serde(other)]
    ExactGelu,
}

/// Vision encoder configuration (SigLIP)
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub model_type: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default, deserialize_with = "null_default")]
    pub hidden_act: VisionHiddenActivation,
}

impl VisionConfig {
    /// Resolve a `hidden_act` the checkpoint omitted to `gelu_pytorch_tanh`,
    /// the Hugging Face `SiglipVisionConfig` default.
    ///
    /// Hugging Face drops a config key whose value equals the class default, so
    /// a SigLIP tower without `hidden_act` was trained with the tanh GELU, and
    /// mlx-vlm's `GELUMLP` (`approx="precise"`) computes tanh as well. The
    /// serde default keeps exact erf for every other loader; this is opt-in per
    /// loader because the CLIP towers' class default is `quick_gelu`, not tanh.
    ///
    /// Used by: the Gemma 3 VLM and PaliGemma loaders (#2423). Aya Vision and
    /// Granite Vision 3.x omit the key too and are not switched yet.
    pub(crate) fn default_missing_hidden_act_to_siglip(&mut self, raw: &serde_json::Value) {
        let declared = raw.get("hidden_act").is_some_and(|value| !value.is_null());
        if !declared {
            self.hidden_act = VisionHiddenActivation::GeluPytorchTanh;
        }
    }
}

fn default_image_size() -> usize {
    224
}
fn default_num_channels() -> usize {
    3
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}

/// Full VLM model config (wraps text + vision configs)
#[derive(Debug, Clone, Deserialize)]
pub struct VLMConfig {
    pub model_type: String,
    pub text_config: serde_json::Value,
    pub vision_config: VisionConfig,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default, deserialize_with = "null_default")]
    pub image_token_index: i32,
    #[serde(default, deserialize_with = "null_default")]
    pub pad_token_id: i32,
    #[serde(default)]
    pub mm_tokens_per_image: Option<usize>,
    /// Begin-of-image token index
    #[serde(default = "default_boi_token")]
    pub boi_token_index: i32,
    /// End-of-image token index
    #[serde(default = "default_eoi_token")]
    pub eoi_token_index: i32,
    /// Which encoder layer to extract features from (LLaVA: -2 = second-to-last)
    #[serde(default = "default_vision_feature_layer")]
    pub vision_feature_layer: i32,
    /// Feature selection strategy: "default" strips CLS token, "full" keeps all
    #[serde(default = "default_vision_feature_select_strategy")]
    pub vision_feature_select_strategy: String,
}

fn default_boi_token() -> i32 {
    255999
}
fn default_eoi_token() -> i32 {
    256000
}

fn default_vocab_size() -> usize {
    257152
}
fn default_vision_feature_layer() -> i32 {
    -2
}
fn default_vision_feature_select_strategy() -> String {
    "default".to_string()
}

impl VLMConfig {
    /// Get mm_tokens_per_image, falling back to text_config value
    pub fn get_mm_tokens_per_image(&self) -> usize {
        self.mm_tokens_per_image
            .or_else(|| {
                self.text_config
                    .get("mm_tokens_per_image")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
            })
            .unwrap_or(256)
    }
}

#[cfg(test)]
mod tests {
    use super::{VisionConfig, VisionHiddenActivation};

    fn config(hidden_act: Option<&str>) -> serde_json::Value {
        let mut value = serde_json::json!({
            "model_type": "siglip_vision_model",
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "patch_size": 2
        });
        if let Some(hidden_act) = hidden_act {
            value["hidden_act"] = serde_json::Value::String(hidden_act.to_string());
        }
        value
    }

    #[test]
    fn vision_hidden_act_selects_only_the_explicit_pytorch_tanh_variant() {
        let tanh: VisionConfig = serde_json::from_value(config(Some("gelu_pytorch_tanh"))).unwrap();
        assert_eq!(tanh.hidden_act, VisionHiddenActivation::GeluPytorchTanh);

        for value in [
            config(None),
            config(Some("gelu")),
            config(Some("quick_gelu")),
        ] {
            let parsed: VisionConfig = serde_json::from_value(value).unwrap();
            assert_eq!(parsed.hidden_act, VisionHiddenActivation::ExactGelu);
        }
    }

    #[test]
    fn missing_hidden_act_resolves_to_the_siglip_default_only_when_absent() {
        let cases = [
            (config(None), VisionHiddenActivation::GeluPytorchTanh),
            (config(Some("gelu")), VisionHiddenActivation::ExactGelu),
            (
                config(Some("gelu_pytorch_tanh")),
                VisionHiddenActivation::GeluPytorchTanh,
            ),
        ];
        for (raw, expected) in cases {
            let mut parsed: VisionConfig = serde_json::from_value(raw.clone()).unwrap();
            parsed.default_missing_hidden_act_to_siglip(&raw);
            assert_eq!(parsed.hidden_act, expected, "{raw}");
        }
        let mut null_act = config(None);
        null_act["hidden_act"] = serde_json::Value::Null;
        let mut parsed: VisionConfig = serde_json::from_value(null_act.clone()).unwrap();
        parsed.default_missing_hidden_act_to_siglip(&null_act);
        assert_eq!(parsed.hidden_act, VisionHiddenActivation::GeluPytorchTanh);
    }
}
