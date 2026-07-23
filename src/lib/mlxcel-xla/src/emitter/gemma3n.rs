//! Validated Gemma3n text-backbone configuration and dense-PLE ABI metadata.
//!
//! Gemma3n is not a flag combination of the ordinary Llama emitter: AltUp keeps
//! multiple hidden planes, LAUREL adds a low-rank residual, later logical layers
//! reuse earlier physical KV slots, and multimodal callers supply one dense PLE
//! vector per logical layer.  Keep that contract explicit so an unsupported or
//! malformed checkpoint fails before graph compilation or buffer upload.

use super::config::QuantConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gemma3nLayerType {
    Full,
    Sliding,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma3nConfig {
    pub context_capacity: usize,
    pub hidden: usize,
    pub intermediate: Vec<usize>,
    pub n_layers: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub vocab: usize,
    pub per_layer_vocab: usize,
    pub hidden_per_layer_input: usize,
    pub layer_types: Vec<Gemma3nLayerType>,
    pub activation_sparsity: Vec<f32>,
    pub sliding_window: usize,
    pub rope_theta: f64,
    pub rope_local_base: f64,
    pub final_logit_softcap: f32,
    pub num_kv_shared_layers: usize,
    pub altup_num_inputs: usize,
    pub altup_active_idx: usize,
    pub altup_coef_clip: f32,
    pub altup_correct_scale: bool,
    pub laurel_rank: usize,
    pub tie_word_embeddings: bool,
    pub quantization: Option<QuantConfig>,
}

impl Gemma3nConfig {
    pub(crate) fn from_json_str(s: &str) -> Result<Self, String> {
        let root: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("parse config.json: {e}"))?;
        let root_type = root.get("model_type").and_then(serde_json::Value::as_str);
        let text = if root_type == Some("gemma3n") {
            root.get("text_config")
                .ok_or("Gemma3n config missing object `text_config`")?
        } else if root_type == Some("gemma3n_text") {
            &root
        } else {
            return Err(format!(
                "unsupported Gemma3n model_type {:?}; expected gemma3n or gemma3n_text",
                root_type
            ));
        };
        if text.get("model_type").and_then(serde_json::Value::as_str) != Some("gemma3n_text") {
            return Err("Gemma3n text_config.model_type must be `gemma3n_text`".to_string());
        }

        let u = |key: &str| -> Result<usize, String> {
            text.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| format!("Gemma3n text_config missing integer `{key}`"))
        };
        let f = |key: &str| -> Result<f64, String> {
            text.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| format!("Gemma3n text_config missing number `{key}`"))
        };
        let b = |key: &str| -> Result<bool, String> {
            text.get(key)
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| format!("Gemma3n text_config missing boolean `{key}`"))
        };
        let n_layers = u("num_hidden_layers")?;
        let vec_usize = |key: &str| -> Result<Vec<usize>, String> {
            text.get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("Gemma3n text_config missing array `{key}`"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|n| n as usize)
                        .ok_or_else(|| format!("Gemma3n `{key}` must contain integers"))
                })
                .collect()
        };
        let intermediate = match text.get("intermediate_size") {
            Some(serde_json::Value::Array(_)) => vec_usize("intermediate_size")?,
            Some(v) => vec![
                v.as_u64()
                    .ok_or("Gemma3n intermediate_size must be an integer or array")?
                    as usize;
                n_layers
            ],
            None => return Err("Gemma3n text_config missing `intermediate_size`".to_string()),
        };
        let activation_sparsity: Vec<f32> = text
            .get("activation_sparsity_pattern")
            .and_then(serde_json::Value::as_array)
            .ok_or("Gemma3n text_config missing array `activation_sparsity_pattern`")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|n| n as f32)
                    .ok_or_else(|| "Gemma3n activation sparsity must be numeric".to_string())
            })
            .collect::<Result<_, _>>()?;
        let layer_types: Vec<Gemma3nLayerType> = text
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
            .ok_or("Gemma3n text_config missing array `layer_types`")?
            .iter()
            .map(|v| match v.as_str() {
                Some("full_attention") => Ok(Gemma3nLayerType::Full),
                Some("sliding_attention") => Ok(Gemma3nLayerType::Sliding),
                other => Err(format!("unsupported Gemma3n layer type {other:?}")),
            })
            .collect::<Result<_, _>>()?;

        let quantization = parse_quantization(&root, text)?;
        let cfg = Self {
            context_capacity: crate::DEFAULT_CONTEXT_CAPACITY,
            hidden: u("hidden_size")?,
            intermediate,
            n_layers,
            n_q: u("num_attention_heads")?,
            n_kv: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            eps: f("rms_norm_eps")? as f32,
            vocab: u("vocab_size")?,
            per_layer_vocab: u("vocab_size_per_layer_input")?,
            hidden_per_layer_input: u("hidden_size_per_layer_input")?,
            layer_types,
            activation_sparsity,
            sliding_window: u("sliding_window")?,
            rope_theta: f("rope_theta")?,
            rope_local_base: f("rope_local_base_freq")?,
            final_logit_softcap: f("final_logit_softcapping")? as f32,
            num_kv_shared_layers: u("num_kv_shared_layers")?,
            altup_num_inputs: u("altup_num_inputs")?,
            altup_active_idx: u("altup_active_idx")?,
            altup_coef_clip: f("altup_coef_clip")? as f32,
            altup_correct_scale: b("altup_correct_scale")?,
            laurel_rank: u("laurel_rank")?,
            tie_word_embeddings: text
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            quantization,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub(crate) fn from_json(model_dir: &std::path::Path) -> Result<Self, String> {
        let path = model_dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_json_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub(crate) fn with_context_capacity(mut self, capacity: usize) -> Result<Self, String> {
        self.context_capacity = crate::context::validate_context_capacity_value(capacity)?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), String> {
        if self.intermediate.len() != self.n_layers
            || self.layer_types.len() != self.n_layers
            || self.activation_sparsity.len() != self.n_layers
        {
            return Err(format!(
                "Gemma3n per-layer arrays must all have num_hidden_layers={} entries \
                 (intermediate={}, layer_types={}, activation_sparsity={})",
                self.n_layers,
                self.intermediate.len(),
                self.layer_types.len(),
                self.activation_sparsity.len()
            ));
        }
        if self.n_q == 0 || self.n_kv == 0 || !self.n_q.is_multiple_of(self.n_kv) {
            return Err("Gemma3n attention heads require nonzero n_q divisible by n_kv".into());
        }
        if self.num_kv_shared_layers >= self.n_layers {
            return Err("Gemma3n num_kv_shared_layers must be smaller than layer count".into());
        }
        if self.altup_num_inputs < 2 || self.altup_active_idx >= self.altup_num_inputs {
            return Err(
                "Gemma3n AltUp requires at least two planes and a valid active index".into(),
            );
        }
        if self.altup_active_idx != 0 {
            return Err(
                "Gemma3n OpenXLA currently requires altup_active_idx=0 so the base, \
                 injection, and unembed plane contract is unambiguous"
                    .into(),
            );
        }
        if self.per_layer_vocab > self.vocab {
            return Err("Gemma3n per-layer vocab cannot exceed token vocab".into());
        }
        if self.hidden == 0 || self.hidden_per_layer_input == 0 || self.laurel_rank == 0 {
            return Err("Gemma3n hidden, per-layer hidden, and LAUREL rank must be nonzero".into());
        }
        if self
            .activation_sparsity
            .iter()
            .any(|&v| !(0.0..1.0).contains(&v))
        {
            return Err("Gemma3n activation sparsity entries must be in [0, 1)".into());
        }
        let concrete = self.kv_cache_layers();
        for kind in [Gemma3nLayerType::Full, Gemma3nLayerType::Sliding] {
            if self.layer_types[..concrete].iter().all(|&v| v != kind)
                && self.layer_types[concrete..].contains(&kind)
            {
                return Err(format!(
                    "Gemma3n shared {kind:?} layers have no concrete KV source"
                ));
            }
        }
        if self.quantization.is_some_and(|q| {
            !matches!(q.bits, 4 | 8)
                || q.group_size == 0
                || !self.hidden.is_multiple_of(q.group_size)
        }) {
            return Err(
                "Gemma3n quantization requires 4/8 bits and a hidden-aligned group_size".into(),
            );
        }
        if !self.tie_word_embeddings {
            return Err(
                "Gemma3n untied LM heads are not supported by the checkpoint contract".into(),
            );
        }
        Ok(())
    }

    pub(crate) fn kv_cache_layers(&self) -> usize {
        self.n_layers - self.num_kv_shared_layers
    }

    pub(crate) fn layer_to_cache(&self) -> Result<Vec<usize>, String> {
        let concrete = self.kv_cache_layers();
        let mut last_full = None;
        let mut last_sliding = None;
        let mut out = Vec::with_capacity(self.n_layers);
        for (index, &kind) in self.layer_types.iter().enumerate() {
            if index < concrete {
                match kind {
                    Gemma3nLayerType::Full => last_full = Some(index),
                    Gemma3nLayerType::Sliding => last_sliding = Some(index),
                }
                out.push(index);
            } else {
                out.push(
                    match kind {
                        Gemma3nLayerType::Full => last_full,
                        Gemma3nLayerType::Sliding => last_sliding,
                    }
                    .ok_or_else(|| {
                        format!("Gemma3n shared layer {index} has no matching KV source")
                    })?,
                );
            }
        }
        Ok(out)
    }

    pub(crate) fn dense_ple_shape(&self) -> [usize; 3] {
        [
            self.context_capacity,
            self.n_layers,
            self.hidden_per_layer_input,
        ]
    }

    pub(crate) fn compatibility_fingerprint(&self) -> String {
        format!(
            "gemma3n:h{}:i{:?}:l{}:q{}:kv{}:d{}:eps{:x}:v{}:pv{}:ph{}:lt{:?}:sp{:?}:\
             sw{}:rt{:x}:rl{:x}:sc{:x}:shared{}:alt{}:{}:{:x}:{}:laurel{}:tie{}:q{:?}:c{}",
            self.hidden,
            self.intermediate,
            self.n_layers,
            self.n_q,
            self.n_kv,
            self.head_dim,
            self.eps.to_bits(),
            self.vocab,
            self.per_layer_vocab,
            self.hidden_per_layer_input,
            self.layer_types,
            self.activation_sparsity
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            self.sliding_window,
            self.rope_theta.to_bits(),
            self.rope_local_base.to_bits(),
            self.final_logit_softcap.to_bits(),
            self.num_kv_shared_layers,
            self.altup_num_inputs,
            self.altup_active_idx,
            self.altup_coef_clip.to_bits(),
            self.altup_correct_scale,
            self.laurel_rank,
            self.tie_word_embeddings,
            self.quantization,
            self.context_capacity,
        )
    }
}

fn parse_quantization(
    root: &serde_json::Value,
    text: &serde_json::Value,
) -> Result<Option<QuantConfig>, String> {
    let value = root
        .get("quantization")
        .or_else(|| text.get("quantization"));
    let Some(q) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let number = |key: &str| {
        q.get(key)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| format!("Gemma3n quantization missing integer `{key}`"))
    };
    Ok(Some(QuantConfig {
        bits: number("bits")?,
        group_size: number("group_size")?,
    }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gemma3nPleDType {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Gemma3nPleMetadata {
    pub shape: [usize; 3],
    pub dtype: Gemma3nPleDType,
    pub byte_length: usize,
}

pub(crate) fn validate_gemma3n_ple(
    cfg: &Gemma3nConfig,
    metadata: Gemma3nPleMetadata,
) -> Result<(), String> {
    if metadata.shape != cfg.dense_ple_shape() {
        return Err(format!(
            "Gemma3n dense PLE shape {:?} does not match required {:?}",
            metadata.shape,
            cfg.dense_ple_shape()
        ));
    }
    if metadata.dtype != Gemma3nPleDType::F32 {
        return Err(format!(
            "Gemma3n dense PLE dtype {:?} is unsupported; expected F32",
            metadata.dtype
        ));
    }
    let elements = metadata
        .shape
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or("Gemma3n dense PLE element count overflows")?;
    let expected = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("Gemma3n dense PLE byte length overflows")?;
    if metadata.byte_length != expected {
        return Err(format!(
            "Gemma3n dense PLE byte length {} does not match required {expected}",
            metadata.byte_length
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> String {
        serde_json::json!({
            "model_type": "gemma3n",
            "quantization": {"bits": 4, "group_size": 4},
            "text_config": {
                "model_type": "gemma3n_text",
                "hidden_size": 8,
                "intermediate_size": [16, 16, 16, 16],
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "rms_norm_eps": 0.000001,
                "vocab_size": 12,
                "vocab_size_per_layer_input": 10,
                "hidden_size_per_layer_input": 2,
                "layer_types": [
                    "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention"
                ],
                "activation_sparsity_pattern": [0.5, 0.0, 0.0, 0.0],
                "sliding_window": 4,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "final_logit_softcapping": 30.0,
                "num_kv_shared_layers": 2,
                "altup_num_inputs": 4,
                "altup_active_idx": 0,
                "altup_coef_clip": 120.0,
                "altup_correct_scale": true,
                "laurel_rank": 2,
                "tie_word_embeddings": true
            }
        })
        .to_string()
    }

    #[test]
    fn parses_nested_contract_and_maps_shared_kv_by_attention_kind() {
        let cfg = Gemma3nConfig::from_json_str(&tiny_config()).unwrap();
        assert_eq!(cfg.kv_cache_layers(), 2);
        assert_eq!(cfg.layer_to_cache().unwrap(), vec![0, 1, 0, 1]);
        assert_eq!(
            cfg.quantization,
            Some(QuantConfig {
                bits: 4,
                group_size: 4
            })
        );
        assert_eq!(
            cfg.dense_ple_shape(),
            [crate::DEFAULT_CONTEXT_CAPACITY, 4, 2]
        );
    }

    #[test]
    fn structural_options_change_artifact_identity() {
        let base = Gemma3nConfig::from_json_str(&tiny_config()).unwrap();
        let mut changed = base.clone();
        changed.activation_sparsity[0] = 0.25;
        assert_ne!(
            base.compatibility_fingerprint(),
            changed.compatibility_fingerprint()
        );
        changed = base.clone();
        changed.layer_types.swap(0, 1);
        assert_ne!(
            base.compatibility_fingerprint(),
            changed.compatibility_fingerprint()
        );
    }

    #[test]
    fn dense_ple_metadata_checks_shape_dtype_and_bytes() {
        let cfg = Gemma3nConfig::from_json_str(&tiny_config())
            .unwrap()
            .with_context_capacity(2)
            .unwrap();
        validate_gemma3n_ple(
            &cfg,
            Gemma3nPleMetadata {
                shape: [2, 4, 2],
                dtype: Gemma3nPleDType::F32,
                byte_length: 64,
            },
        )
        .unwrap();
        let err = validate_gemma3n_ple(
            &cfg,
            Gemma3nPleMetadata {
                shape: [2, 4, 2],
                dtype: Gemma3nPleDType::F16,
                byte_length: 32,
            },
        )
        .unwrap_err();
        assert!(err.contains("expected F32"));
    }

    #[test]
    fn malformed_per_layer_arrays_and_shared_sources_fail_early() {
        let bad = tiny_config().replacen(
            "\"activation_sparsity_pattern\":[0.5,0.0,0.0,0.0]",
            "\"activation_sparsity_pattern\":[0.5]",
            1,
        );
        assert!(
            Gemma3nConfig::from_json_str(&bad)
                .unwrap_err()
                .contains("per-layer arrays")
        );
    }
}
