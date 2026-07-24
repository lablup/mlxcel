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

//! Static compatibility contract for the first IREE LLaVA vision bundle.

use std::path::Path;

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum VisionActivation {
    ExactGelu,
    GeluPytorchTanh,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_json() -> String {
        serde_json::json!({
            "model_type": "llava",
            "vision_config": {
                "model_type": "siglip_vision_model",
                "image_size": 28,
                "patch_size": 14,
                "num_channels": 3,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "layer_norm_eps": 1e-6,
                "hidden_act": "gelu_pytorch_tanh"
            },
            "text_config": {"hidden_size": 12},
            "projector_hidden_act": "gelu",
            "vision_feature_layer": -1,
            "vision_feature_select_strategy": "full"
        })
        .to_string()
    }

    #[test]
    fn fingerprint_binds_every_graph_shape_and_semantic_field() {
        let base = LlavaVisionConfig::from_json_str(&config_json()).unwrap();
        let fingerprint = base.fingerprint();
        let variants = [
            LlavaVisionConfig {
                image_size: 42,
                ..base.clone()
            },
            LlavaVisionConfig {
                patch_size: 7,
                ..base.clone()
            },
            LlavaVisionConfig {
                channels: 1,
                ..base.clone()
            },
            LlavaVisionConfig {
                hidden: 16,
                heads: 2,
                ..base.clone()
            },
            LlavaVisionConfig {
                intermediate: 24,
                ..base.clone()
            },
            LlavaVisionConfig {
                layers: 3,
                ..base.clone()
            },
            LlavaVisionConfig {
                heads: 4,
                ..base.clone()
            },
            LlavaVisionConfig {
                layer_norm_eps: 1e-5,
                ..base.clone()
            },
            LlavaVisionConfig {
                activation: VisionActivation::ExactGelu,
                ..base.clone()
            },
            LlavaVisionConfig {
                class_token: true,
                ..base.clone()
            },
            LlavaVisionConfig {
                feature_layer: 0,
                ..base.clone()
            },
            LlavaVisionConfig {
                drop_first_token: true,
                ..base.clone()
            },
            LlavaVisionConfig {
                text_hidden: 20,
                ..base.clone()
            },
        ];
        for variant in variants {
            assert_ne!(variant.fingerprint(), fingerprint);
        }
    }

    #[test]
    fn rejects_non_static_or_unqualified_contracts() {
        let mut root: Value = serde_json::from_str(&config_json()).unwrap();
        root["vision_config"]["image_size"] = 7.into();
        assert!(
            LlavaVisionConfig::from_json_str(&root.to_string())
                .unwrap_err()
                .contains("smaller than")
        );
        root["vision_config"]["image_size"] = 28.into();
        root["vision_config"]["hidden_act"] = "quick_gelu".into();
        assert!(
            LlavaVisionConfig::from_json_str(&root.to_string())
                .unwrap_err()
                .contains("unsupported vision hidden_act")
        );
        root["vision_config"]["hidden_act"] = "gelu".into();
        root["vision_feature_select_strategy"] = "patch".into();
        assert!(
            LlavaVisionConfig::from_json_str(&root.to_string())
                .unwrap_err()
                .contains("unsupported vision_feature_select_strategy")
        );
    }

    #[test]
    fn pinned_siglip_contract_has_exact_schema() {
        let path = Path::new("/tmp/mlxcel-llava-hf-1090956d");
        if !path.join("config.json").is_file() {
            return;
        }
        let config = LlavaVisionConfig::from_model_dir(path).unwrap();
        assert_eq!(config.image_size, 384);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.hidden, 1152);
        assert_eq!(config.intermediate, 4304);
        assert_eq!(config.layers, 26);
        assert_eq!(config.feature_layer, 25);
        assert_eq!(config.text_hidden, 1024);
        assert_eq!(config.image_tokens(), 729);
        assert_eq!(config.weight_specs().len(), 423);
        assert_eq!(
            config.weight_specs().first().unwrap(),
            &VisionWeightSpec {
                name: "vision_tower.vision_model.embeddings.patch_embedding.weight".to_string(),
                shape: vec![1152, 3, 14, 14],
            }
        );
        assert_eq!(
            config.weight_specs().last().unwrap(),
            &VisionWeightSpec {
                name: "multi_modal_projector.linear_2.bias".to_string(),
                shape: vec![1024],
            }
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LlavaVisionConfig {
    pub(crate) image_size: usize,
    pub(crate) patch_size: usize,
    pub(crate) channels: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) layers: usize,
    pub(crate) heads: usize,
    pub(crate) layer_norm_eps: f32,
    pub(crate) activation: VisionActivation,
    pub(crate) class_token: bool,
    pub(crate) feature_layer: usize,
    pub(crate) drop_first_token: bool,
    pub(crate) text_hidden: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VisionWeightSpec {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
}

fn object<'a>(value: &'a Value, field: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("config.json {field} must be an object"))
}

fn usize_field(object: &serde_json::Map<String, Value>, field: &str) -> Result<usize, String> {
    let value = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("config.json vision_config.{field} must be a positive integer"))?;
    usize::try_from(value)
        .map_err(|_| format!("config.json vision_config.{field} does not fit usize"))
        .and_then(|value| {
            if value == 0 {
                Err(format!(
                    "config.json vision_config.{field} must be greater than zero"
                ))
            } else {
                Ok(value)
            }
        })
}

impl LlavaVisionConfig {
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Self::from_json_str(&text)
    }

    pub(crate) fn from_json_str(text: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(text).map_err(|error| format!("parse config.json: {error}"))?;
        if root.get("model_type").and_then(Value::as_str) != Some("llava") {
            return Err("IREE vision currently supports only model_type=llava".to_string());
        }
        let vision = object(&root, "vision_config")?;
        let vision_model_type = vision
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| "config.json vision_config.model_type is required".to_string())?;
        let class_token = match vision_model_type {
            "siglip_vision_model" => false,
            "clip_vision_model" => true,
            other => {
                return Err(format!(
                    "IREE vision supports only SigLIP/CLIP towers, got {other:?}"
                ));
            }
        };
        let image_size = usize_field(vision, "image_size")?;
        let patch_size = usize_field(vision, "patch_size")?;
        if image_size < patch_size {
            return Err(format!(
                "vision image_size={image_size} is smaller than patch_size={patch_size}"
            ));
        }
        let channels =
            vision
                .get("num_channels")
                .and_then(Value::as_u64)
                .map_or(Ok(3), |value| {
                    usize::try_from(value)
                        .map_err(|_| "vision num_channels does not fit usize".to_string())
                })?;
        if channels == 0 {
            return Err("vision num_channels must be greater than zero".to_string());
        }
        let hidden = usize_field(vision, "hidden_size")?;
        let intermediate = usize_field(vision, "intermediate_size")?;
        let layers = usize_field(vision, "num_hidden_layers")?;
        let heads = usize_field(vision, "num_attention_heads")?;
        if hidden % heads != 0 {
            return Err(format!(
                "vision hidden_size={hidden} is not divisible by num_attention_heads={heads}"
            ));
        }
        let layer_norm_eps = vision
            .get("layer_norm_eps")
            .and_then(Value::as_f64)
            .unwrap_or(1e-6) as f32;
        if !layer_norm_eps.is_finite() || layer_norm_eps <= 0.0 {
            return Err("vision layer_norm_eps must be finite and positive".to_string());
        }
        let activation = match vision.get("hidden_act").and_then(Value::as_str) {
            Some("gelu_pytorch_tanh") => VisionActivation::GeluPytorchTanh,
            None | Some("gelu") => VisionActivation::ExactGelu,
            other => {
                return Err(format!(
                    "unsupported vision hidden_act {other:?}; refusing an approximate substitution"
                ));
            }
        };
        match root.get("projector_hidden_act").and_then(Value::as_str) {
            None | Some("gelu") => {}
            other => {
                return Err(format!(
                    "unsupported projector_hidden_act {other:?}; only exact GELU is qualified"
                ));
            }
        }
        let text_hidden = object(&root, "text_config")?
            .get("hidden_size")
            .and_then(Value::as_u64)
            .ok_or_else(|| "config.json text_config.hidden_size is required".to_string())
            .and_then(|value| {
                usize::try_from(value)
                    .map_err(|_| "text_config.hidden_size does not fit usize".to_string())
            })?;
        if text_hidden == 0 {
            return Err("text_config.hidden_size must be greater than zero".to_string());
        }
        let requested_layer = root
            .get("vision_feature_layer")
            .and_then(Value::as_i64)
            .unwrap_or(-2);
        let resolved = if requested_layer < 0 {
            i64::try_from(layers).map_err(|_| "vision layer count does not fit i64".to_string())?
                + requested_layer
        } else {
            requested_layer
        };
        if resolved < 0 || resolved >= layers as i64 {
            return Err(format!(
                "vision_feature_layer={requested_layer} resolves outside {layers} encoder layers"
            ));
        }
        let strategy = root
            .get("vision_feature_select_strategy")
            .and_then(Value::as_str)
            .unwrap_or("default");
        let drop_first_token = match strategy {
            "default" => true,
            "full" => false,
            other => {
                return Err(format!(
                    "unsupported vision_feature_select_strategy={other:?}"
                ));
            }
        };
        Ok(Self {
            image_size,
            patch_size,
            channels,
            hidden,
            intermediate,
            layers,
            heads,
            layer_norm_eps,
            activation,
            class_token,
            feature_layer: resolved as usize,
            drop_first_token,
            text_hidden,
        })
    }

    #[must_use]
    pub(crate) fn patch_grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    #[must_use]
    pub(crate) fn position_count(&self) -> usize {
        self.patch_grid().pow(2) + usize::from(self.class_token)
    }

    #[must_use]
    pub(crate) fn image_tokens(&self) -> usize {
        self.position_count() - usize::from(self.drop_first_token)
    }

    #[must_use]
    pub(crate) fn fingerprint(&self) -> String {
        format!(
            "llava-vision-v1:image={}:patch={}:channels={}:hidden={}:intermediate={}:layers={}:\
             heads={}:eps={:08x}:activation={:?}:class={}:feature={}:drop_first={}:text={}",
            self.image_size,
            self.patch_size,
            self.channels,
            self.hidden,
            self.intermediate,
            self.layers,
            self.heads,
            self.layer_norm_eps.to_bits(),
            self.activation,
            self.class_token,
            self.feature_layer,
            self.drop_first_token,
            self.text_hidden
        )
    }

    pub(crate) fn weight_specs(&self) -> Vec<VisionWeightSpec> {
        let mut specs = vec![
            self.spec(
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
                [self.hidden, self.channels, self.patch_size, self.patch_size],
            ),
            self.spec(
                "vision_tower.vision_model.embeddings.patch_embedding.bias",
                [self.hidden],
            ),
        ];
        if self.class_token {
            specs.push(self.spec(
                "vision_tower.vision_model.embeddings.class_embedding",
                [self.hidden],
            ));
        }
        specs.push(self.spec(
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            [self.position_count(), self.hidden],
        ));
        if self.class_token {
            specs.extend([
                self.spec(
                    "vision_tower.vision_model.pre_layrnorm.weight",
                    [self.hidden],
                ),
                self.spec("vision_tower.vision_model.pre_layrnorm.bias", [self.hidden]),
            ]);
        }
        for layer in 0..=self.feature_layer {
            let prefix = format!("vision_tower.vision_model.encoder.layers.{layer}");
            for (suffix, shape) in [
                ("layer_norm1.weight", vec![self.hidden]),
                ("layer_norm1.bias", vec![self.hidden]),
                ("self_attn.q_proj.weight", vec![self.hidden, self.hidden]),
                ("self_attn.q_proj.bias", vec![self.hidden]),
                ("self_attn.k_proj.weight", vec![self.hidden, self.hidden]),
                ("self_attn.k_proj.bias", vec![self.hidden]),
                ("self_attn.v_proj.weight", vec![self.hidden, self.hidden]),
                ("self_attn.v_proj.bias", vec![self.hidden]),
                ("self_attn.out_proj.weight", vec![self.hidden, self.hidden]),
                ("self_attn.out_proj.bias", vec![self.hidden]),
                ("layer_norm2.weight", vec![self.hidden]),
                ("layer_norm2.bias", vec![self.hidden]),
                ("mlp.fc1.weight", vec![self.intermediate, self.hidden]),
                ("mlp.fc1.bias", vec![self.intermediate]),
                ("mlp.fc2.weight", vec![self.hidden, self.intermediate]),
                ("mlp.fc2.bias", vec![self.hidden]),
            ] {
                specs.push(VisionWeightSpec {
                    name: format!("{prefix}.{suffix}"),
                    shape,
                });
            }
        }
        specs.extend([
            self.spec(
                "multi_modal_projector.linear_1.weight",
                [self.text_hidden, self.hidden],
            ),
            self.spec("multi_modal_projector.linear_1.bias", [self.text_hidden]),
            self.spec(
                "multi_modal_projector.linear_2.weight",
                [self.text_hidden, self.text_hidden],
            ),
            self.spec("multi_modal_projector.linear_2.bias", [self.text_hidden]),
        ]);
        specs
    }

    fn spec(&self, name: &str, shape: impl IntoIterator<Item = usize>) -> VisionWeightSpec {
        VisionWeightSpec {
            name: name.to_string(),
            shape: shape.into_iter().collect(),
        }
    }
}
