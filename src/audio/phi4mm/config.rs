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

use serde_json::Value;

/// The exact published Cascades configuration. Keeping this small and strict
/// makes a future incompatible checkpoint fail at load instead of producing
/// plausible-looking but wrong audio embeddings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi4MMAudioConfig {
    pub input_size: usize,
    pub attention_dim: usize,
    pub attention_heads: usize,
    pub num_blocks: usize,
    pub linear_units: usize,
    pub time_reduction: usize,
    pub conv_channels: usize,
    pub kernel_size: usize,
    pub relative_bias_max_distance: usize,
}

impl Phi4MMAudioConfig {
    pub fn from_model_config(root: &Value) -> Result<Self, String> {
        let cfg = root
            .pointer("/audio_processor/config")
            .ok_or("Phi4MM config is missing audio_processor.config")?;
        let get = |name: &str| {
            cfg.get(name)
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .ok_or_else(|| format!("Phi4MM audio config is missing integer {name}"))
        };
        let parsed = Self {
            input_size: get("input_size")?,
            attention_dim: get("attention_dim")?,
            attention_heads: get("attention_heads")?,
            num_blocks: get("num_blocks")?,
            linear_units: get("linear_units")?,
            time_reduction: get("time_reduction")?,
            conv_channels: cfg
                .pointer("/nemo_conv_settings/conv_channels")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .ok_or("Phi4MM audio config is missing nemo_conv_settings.conv_channels")?,
            kernel_size: get("kernel_size")?,
            relative_bias_max_distance: cfg
                .pointer("/relative_attention_bias_args/t5_bias_max_distance")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .ok_or("Phi4MM audio config is missing T5 bias max distance")?,
        };
        parsed.validate_published(root)?;
        Ok(parsed)
    }

    fn validate_published(&self, root: &Value) -> Result<(), String> {
        let cfg = root.pointer("/audio_processor/config").unwrap();
        let expected = Self {
            input_size: 80,
            attention_dim: 1024,
            attention_heads: 16,
            num_blocks: 24,
            linear_units: 1536,
            time_reduction: 8,
            conv_channels: 1024,
            kernel_size: 3,
            relative_bias_max_distance: 500,
        };
        let string_is =
            |path: &str, expected: &str| cfg.get(path).and_then(Value::as_str) == Some(expected);
        let bool_is =
            |path: &str, expected: bool| cfg.get(path).and_then(Value::as_bool) == Some(expected);
        if self != &expected
            || !string_is("input_layer", "nemo_conv")
            || !string_is("activation", "swish")
            || !string_is("conv_activation", "swish")
            || !string_is("conv_glu_type", "swish")
            || !bool_is("causal", true)
            || !bool_is("batch_norm", false)
            || !bool_is("bias_in_glu", true)
            || cfg.get("depthwise_multiplier").and_then(Value::as_u64) != Some(1)
            || cfg
                .get("depthwise_seperable_out_channel")
                .and_then(Value::as_u64)
                != Some(1024)
            || cfg.get("ext_pw_kernel_size").and_then(Value::as_u64) != Some(1)
            || cfg.get("ext_pw_out_channel").and_then(Value::as_u64) != Some(1024)
            || cfg
                .pointer("/encoder_embedding_config/input_size")
                .and_then(Value::as_u64)
                != Some(80)
            || cfg.get("chunk_size").and_then(Value::as_i64) != Some(-1)
            || cfg
                .pointer("/relative_attention_bias_args/type")
                .and_then(Value::as_str)
                != Some("t5")
            || cfg
                .get("attention_group_size")
                .and_then(Value::as_u64)
                .is_some_and(|value| value != 1)
            || cfg
                .get("linear_glu_in_convm")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || cfg
                .get("use_pt_scaled_dot_product_attention")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || cfg
                .pointer("/relative_attention_bias_args/t5_bias_symmetric")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || cfg
                .pointer("/nemo_conv_settings/is_causal")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        {
            return Err(format!(
                "unsupported Phi4MM audio architecture: expected published Cascades config, got {cfg}"
            ));
        }
        let embd = root
            .pointer("/embd_layer/audio_embd_layer")
            .ok_or("Phi4MM config is missing embd_layer.audio_embd_layer")?;
        if embd.get("compression_rate").and_then(Value::as_u64) != Some(8)
            || embd.get("downsample_rate").and_then(Value::as_u64) != Some(1)
            || embd.get("projection_cls").and_then(Value::as_str) != Some("mlp")
        {
            return Err("unsupported Phi4MM audio embedding/projection configuration".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_architecture_drift() {
        let mut value: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/phi4mm_audio_config.json"
        ))
        .unwrap();
        assert!(Phi4MMAudioConfig::from_model_config(&value).is_ok());
        value["audio_processor"]["config"]["num_blocks"] = 23.into();
        assert!(
            Phi4MMAudioConfig::from_model_config(&value)
                .unwrap_err()
                .contains("unsupported Phi4MM audio architecture")
        );
        let mut value: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/phi4mm_audio_config.json"
        ))
        .unwrap();
        value["audio_processor"]["config"]["attention_group_size"] = 2.into();
        assert!(Phi4MMAudioConfig::from_model_config(&value).is_err());
    }
}
