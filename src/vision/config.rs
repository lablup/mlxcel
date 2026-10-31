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
/// A missing key also parses as `ExactGelu`; every loader of a SigLIP or CLIP
/// tower resolves that case to the Hugging Face class default with
/// [`VisionConfig::default_missing_hidden_act_by_model_type`] (or, for Gemma 3
/// and PaliGemma, [`VisionConfig::default_missing_hidden_act_to_siglip`]).
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
pub enum VisionHiddenActivation {
    #[serde(rename = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
    /// `gelu_pytorch_tanh` resolved from an omitted key by
    /// [`VisionConfig::default_missing_hidden_act_to_siglip`]; never parsed.
    /// Same formula as [`GeluPytorchTanh`](Self::GeluPytorchTanh), evaluated as
    /// one compiled kernel so the erf-to-tanh switch does not add prefill time.
    /// Kept apart because the declared variant's eager f32 path is pinned
    /// bit-for-bit to the private copy it replaced (#2423).
    #[serde(skip)]
    SiglipDefaultTanh,
    /// `quick_gelu` (`x * sigmoid(1.702 x)`), declared or resolved from an
    /// omitted key on a CLIP tower (the Hugging Face `CLIPVisionConfig`
    /// default) by [`VisionConfig::default_missing_hidden_act_by_model_type`].
    /// Also set directly by loaders whose CLIP tower config is built in code
    /// (Phi-3-V). Parsed as `ExactGelu` before #2548.
    #[serde(rename = "quick_gelu")]
    QuickGelu,
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
    /// Used by: the Gemma 3 VLM and PaliGemma loaders (#2423).
    pub(crate) fn default_missing_hidden_act_to_siglip(&mut self, raw: &serde_json::Value) {
        if !hidden_act_declared(raw) {
            self.hidden_act = VisionHiddenActivation::SiglipDefaultTanh;
        }
    }

    /// Resolve an omitted `hidden_act` to the Hugging Face class default of
    /// the tower `model_type` names: `gelu_pytorch_tanh` for
    /// `siglip_vision_model` / `siglip2_vision_model` (`SiglipVisionConfig`),
    /// `quick_gelu` for `clip_vision_model` (`CLIPVisionConfig`). Any other
    /// model type, and a declared key, are left as parsed.
    ///
    /// The serde default stays exact erf because the shared struct also
    /// parses towers whose class default is neither (#2548).
    ///
    /// Used by: the LLaVA (and its host preprocessor), Bunny, Aya Vision and
    /// Granite Vision 3.x loaders (#2548).
    pub(crate) fn default_missing_hidden_act_by_model_type(&mut self, raw: &serde_json::Value) {
        if hidden_act_declared(raw) {
            return;
        }
        match self.model_type.as_str() {
            "siglip_vision_model" | "siglip2_vision_model" => {
                self.hidden_act = VisionHiddenActivation::SiglipDefaultTanh;
            }
            "clip_vision_model" => {
                self.hidden_act = VisionHiddenActivation::QuickGelu;
            }
            _ => {}
        }
    }
}

fn hidden_act_declared(raw: &serde_json::Value) -> bool {
    raw.get("hidden_act").is_some_and(|value| !value.is_null())
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
        let quick: VisionConfig = serde_json::from_value(config(Some("quick_gelu"))).unwrap();
        assert_eq!(quick.hidden_act, VisionHiddenActivation::QuickGelu);

        for value in [config(None), config(Some("gelu")), config(Some("relu"))] {
            let parsed: VisionConfig = serde_json::from_value(value).unwrap();
            assert_eq!(parsed.hidden_act, VisionHiddenActivation::ExactGelu);
        }
    }

    #[test]
    fn missing_hidden_act_resolves_to_the_siglip_default_only_when_absent() {
        let cases = [
            (config(None), VisionHiddenActivation::SiglipDefaultTanh),
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
        assert_eq!(parsed.hidden_act, VisionHiddenActivation::SiglipDefaultTanh);
    }

    fn with_model_type(model_type: &str, hidden_act: Option<&str>) -> serde_json::Value {
        let mut value = config(hidden_act);
        value["model_type"] = serde_json::Value::String(model_type.to_string());
        value
    }

    #[test]
    fn missing_hidden_act_resolves_to_the_tower_class_default() {
        let cases = [
            // HF drops a key equal to the class default: SigLIP -> tanh.
            (
                "siglip_vision_model",
                None,
                VisionHiddenActivation::SiglipDefaultTanh,
            ),
            (
                "siglip2_vision_model",
                None,
                VisionHiddenActivation::SiglipDefaultTanh,
            ),
            // CLIPVisionConfig().hidden_act == "quick_gelu".
            ("clip_vision_model", None, VisionHiddenActivation::QuickGelu),
            // A declared key always wins.
            (
                "clip_vision_model",
                Some("gelu"),
                VisionHiddenActivation::ExactGelu,
            ),
            (
                "siglip_vision_model",
                Some("gelu_pytorch_tanh"),
                VisionHiddenActivation::GeluPytorchTanh,
            ),
            // Unknown tower types keep the serde default.
            ("vision", None, VisionHiddenActivation::ExactGelu),
        ];
        for (model_type, act, expected) in cases {
            let raw = with_model_type(model_type, act);
            let mut parsed: VisionConfig = serde_json::from_value(raw.clone()).unwrap();
            parsed.default_missing_hidden_act_by_model_type(&raw);
            assert_eq!(parsed.hidden_act, expected, "{raw}");
        }
    }
}
