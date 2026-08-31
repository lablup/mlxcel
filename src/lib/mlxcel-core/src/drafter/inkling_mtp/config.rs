use std::path::Path;

use serde_json::Value;

use crate::drafter::DrafterError;
use crate::inkling_layer::InklingLayerSpec;

/// Validated Inkling MTP configuration derived from the target config.
#[derive(Debug, Clone)]
pub struct InklingMtpConfig {
    raw: Value,
    num_mtp_layers: usize,
    local_layer_ids: Vec<usize>,
    block_size: usize,
    hidden_size: usize,
    vocab_size: usize,
    unpadded_vocab_size: usize,
    rms_norm_eps: f32,
    logits_mup_width_multiplier: f32,
}

impl InklingMtpConfig {
    pub fn from_dir(path: &Path) -> Result<Self, DrafterError> {
        let config_path = path.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|source| DrafterError::ConfigIo {
            path: config_path.display().to_string(),
            source,
        })?;
        let raw: Value =
            serde_json::from_slice(&bytes).map_err(|source| DrafterError::ConfigParse {
                path: config_path.display().to_string(),
                source,
            })?;
        Self::from_value(raw).map_err(DrafterError::Config)
    }

    pub fn from_value(raw: Value) -> Result<Self, String> {
        let text = raw
            .get("text_config")
            .and_then(Value::as_object)
            .ok_or_else(|| "Inkling MTP config requires a text_config object".to_string())?;
        let mtp = raw.get("mtp_config").and_then(Value::as_object);
        let num_mtp_layers = mtp
            .and_then(|config| config.get("num_nextn_predict_layers"))
            .and_then(Value::as_u64)
            .or_else(|| text.get("num_mtp_layers").and_then(Value::as_u64))
            .ok_or_else(|| {
                "Inkling MTP config requires mtp_config.num_nextn_predict_layers or text_config.num_mtp_layers"
                    .to_string()
            })? as usize;
        if !(1..=64).contains(&num_mtp_layers) {
            return Err(format!(
                "Inkling MTP layer count must be in 1..=64, got {num_mtp_layers}"
            ));
        }

        let local_value = mtp
            .and_then(|config| config.get("local_layer_ids"))
            .or_else(|| text.get("mtp_local_layer_ids"));
        let mut local_layer_ids = match local_value {
            Some(value) => value
                .as_array()
                .ok_or_else(|| "Inkling MTP local layer ids must be an array".to_string())?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            "Inkling MTP local layer ids must contain non-negative integers"
                                .to_string()
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };
        local_layer_ids.sort_unstable();
        local_layer_ids.dedup();
        if let Some(invalid) = local_layer_ids
            .iter()
            .copied()
            .find(|index| *index >= num_mtp_layers)
        {
            return Err(format!(
                "Inkling MTP local layer id {invalid} is outside 0..{num_mtp_layers}"
            ));
        }

        let hidden_size = required_usize(text, "hidden_size")?;
        let vocab_size = required_usize(text, "vocab_size")?;
        let unpadded_vocab_size = optional_usize(text, "unpadded_vocab_size").unwrap_or(vocab_size);
        if unpadded_vocab_size == 0 || unpadded_vocab_size > vocab_size {
            return Err(format!(
                "Inkling unpadded_vocab_size must be in 1..={vocab_size}, got {unpadded_vocab_size}"
            ));
        }
        let rms_norm_eps = optional_f32(text, "rms_norm_eps").unwrap_or(1e-6);
        let logits_mup_width_multiplier =
            optional_f32(text, "logits_mup_width_multiplier").unwrap_or(1.0);
        if !logits_mup_width_multiplier.is_finite() || logits_mup_width_multiplier <= 0.0 {
            return Err("Inkling logits_mup_width_multiplier must be positive and finite".into());
        }
        let block_size = raw
            .get("block_size")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(num_mtp_layers + 2);
        if block_size < 2 {
            return Err(format!(
                "Inkling MTP block_size must be at least 2, got {block_size}"
            ));
        }

        Ok(Self {
            raw,
            num_mtp_layers,
            local_layer_ids,
            block_size,
            hidden_size,
            vocab_size,
            unpadded_vocab_size,
            rms_norm_eps,
            logits_mup_width_multiplier,
        })
    }

    pub fn num_mtp_layers(&self) -> usize {
        self.num_mtp_layers
    }

    pub fn local_layer_ids(&self) -> &[usize] {
        &self.local_layer_ids
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn unpadded_vocab_size(&self) -> usize {
        self.unpadded_vocab_size
    }

    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    pub fn logits_mup_width_multiplier(&self) -> f32 {
        self.logits_mup_width_multiplier
    }

    pub fn layer_spec(&self, index: usize) -> Result<InklingLayerSpec, String> {
        if index >= self.num_mtp_layers {
            return Err(format!("Inkling MTP layer {index} is out of range"));
        }
        let text = self
            .raw
            .get("text_config")
            .and_then(Value::as_object)
            .ok_or_else(|| "Inkling MTP config lost text_config".to_string())?;
        let is_sliding = self.local_layer_ids.contains(&index);
        let prefix = if is_sliding { "swa_" } else { "" };
        let num_attention_heads = optional_usize(text, &format!("{prefix}num_attention_heads"))
            .or_else(|| optional_usize(text, "num_attention_heads"))
            .ok_or_else(|| "Inkling MTP config requires num_attention_heads".to_string())?;
        let num_key_value_heads = optional_usize(text, &format!("{prefix}num_key_value_heads"))
            .or_else(|| optional_usize(text, "num_key_value_heads"))
            .ok_or_else(|| "Inkling MTP config requires num_key_value_heads".to_string())?;
        let head_dim = optional_usize(text, &format!("{prefix}head_dim"))
            .or_else(|| optional_usize(text, "head_dim"))
            .ok_or_else(|| "Inkling MTP config requires head_dim".to_string())?;
        let dense_intermediate_size = optional_usize(text, "dense_intermediate_size")
            .or_else(|| optional_usize(text, "intermediate_size"))
            .ok_or_else(|| {
                "Inkling MTP config requires dense_intermediate_size or intermediate_size"
                    .to_string()
            })?;
        let quantization = self
            .raw
            .get("quantization")
            .or_else(|| self.raw.get("quantization_config"))
            .and_then(Value::as_object);
        let quantization_group_size = quantization
            .and_then(|value| value.get("group_size"))
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .unwrap_or(64);
        let quantization_bits = quantization
            .and_then(|value| value.get("bits"))
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .unwrap_or(4);
        Ok(InklingLayerSpec {
            hidden_size: self.hidden_size,
            rms_norm_eps: self.rms_norm_eps,
            is_sliding,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            sliding_window_size: optional_usize(text, "sliding_window_size").unwrap_or(512),
            d_rel: optional_usize(text, "d_rel").unwrap_or(16),
            rel_extent: optional_usize(text, "rel_extent").unwrap_or(1024),
            log_scaling_n_floor: optional_usize(text, "log_scaling_n_floor"),
            log_scaling_alpha: optional_f32(text, "log_scaling_alpha").unwrap_or(0.1),
            sconv_kernel_size: optional_usize(text, "sconv_kernel_size")
                .or_else(|| optional_usize(text, "conv_kernel_size"))
                .unwrap_or(4),
            dense_intermediate_size,
            quantization_group_size,
            quantization_bits,
        })
    }
}

fn required_usize(map: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    optional_usize(map, key).ok_or_else(|| format!("Inkling MTP config requires {key}"))
}

fn optional_usize(map: &serde_json::Map<String, Value>, key: &str) -> Option<usize> {
    map.get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn optional_f32(map: &serde_json::Map<String, Value>, key: &str) -> Option<f32> {
    map.get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
}
