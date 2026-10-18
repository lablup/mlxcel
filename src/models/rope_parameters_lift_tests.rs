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

//! Every config struct that reads the flat `rope_theta` must also read a
//! transformers 5 `rope_parameters` block, and a flat key must still win
//! (#2443).
//!
//! The table holds one parse function per struct. Rather than a hand-written
//! minimal config per family, [`fill_required_fields`] grows a config from
//! the serde errors themselves: each `missing field` gets a placeholder, and a
//! placeholder of the wrong type is replaced by one matching the `expected`
//! clause of the next error. `rope_theta` and `rope_scaling` are never
//! fabricated, so a struct that requires either flat key and does not lift it
//! fails here instead of being papered over.

use serde_json::{Value, json};

type ParseTheta = fn(Value) -> Result<Option<f64>, String>;

trait ThetaValue {
    fn theta(&self) -> Option<f64>;
}

impl ThetaValue for f32 {
    fn theta(&self) -> Option<f64> {
        Some(f64::from(*self))
    }
}

impl ThetaValue for f64 {
    fn theta(&self) -> Option<f64> {
        Some(*self)
    }
}

impl<T: ThetaValue> ThetaValue for Option<T> {
    fn theta(&self) -> Option<f64> {
        self.as_ref().and_then(ThetaValue::theta)
    }
}

impl ThetaValue for Value {
    fn theta(&self) -> Option<f64> {
        self.as_f64()
    }
}

/// A table row: the type name and a parse function returning its RoPE base.
/// The one-argument form reads the `rope_theta` field; the two-argument form
/// takes an accessor for families that resolve the base through a method.
macro_rules! case {
    ($ty:ty) => {
        case!($ty, |c| c.rope_theta.theta())
    };
    ($ty:ty, $get:expr) => {
        (
            stringify!($ty),
            (|value: Value| {
                let get: fn(&$ty) -> Option<f64> = $get;
                serde_json::from_value::<$ty>(value)
                    .map(|config| get(&config))
                    .map_err(|e| e.to_string())
            }) as ParseTheta,
        )
    };
}

fn cases() -> Vec<(&'static str, ParseTheta)> {
    use crate::audio::qwen3_omni_moe::speech_config as omni;
    use crate::models::*;
    vec![
        // Shared Llama args: Llama, Qwen2, IQuestCoder, the Mistral3 fallback.
        case!(llama3::ModelArgs),
        // Qwen / Gemma.
        case!(qwen2_moe::ModelArgs),
        case!(qwen2_vl::Qwen2VLConfig),
        case!(qwen3_moe::ModelArgs),
        case!(qwen3_next::Qwen3NextConfig),
        case!(qwen3_vl::Qwen3VLConfig),
        case!(qwen3_vl_moe::Qwen3VLMoeConfig),
        case!(gemma::ModelArgs),
        case!(gemma2::ModelArgs),
        case!(gemma3::ModelArgs),
        case!(gemma3n::TextConfig),
        case!(recurrent_gemma::GriffinConfig),
        // Mistral / Phi / OLMo / Granite.
        case!(mixtral::ModelArgs),
        case!(phi::ModelArgs),
        case!(phi3::ModelArgs),
        case!(phimoe::ModelArgs),
        case!(phixtral::ModelArgs),
        case!(olmo::ModelArgs),
        case!(olmo2::ModelArgs),
        case!(olmo3::OLMo3Config),
        case!(olmoe::ModelArgs),
        case!(granite::ModelArgs),
        // DeepSeek / GLM / Kimi.
        case!(deepseek::ModelArgs),
        case!(deepseek_v2::ModelArgs),
        case!(deepseek_v3::DeepSeekV3Config),
        case!(deepseek_v32::ModelArgs),
        case!(deepseek_v4::ModelArgs),
        case!(glm4::ModelArgs),
        case!(glm4_moe::ModelArgs),
        case!(glm4_moe_lite::ModelArgs),
        case!(glm4v::Glm4vTextConfig),
        case!(kimi_linear::KimiLinearConfig),
        case!(dots1::ModelArgs),
        // Other text families.
        case!(afmoe::ModelArgs),
        case!(apertus::ModelArgs),
        case!(baichuan::BaichuanConfig),
        case!(bailing_moe::ModelArgs),
        case!(bailing_moe_linear::ModelArgs),
        case!(bitnet::BitNetConfig),
        case!(cohere::ModelArgs),
        case!(cohere2::Cohere2Config),
        case!(cohere2_moe::Cohere2MoeConfig),
        case!(dbrx::AttnConfig),
        case!(ernie4_5::ModelArgs),
        case!(ernie4_5_moe::ModelArgs),
        case!(exaone::ExaOneConfig),
        case!(exaone4::ModelArgs),
        case!(falcon_h1::ModelArgs),
        case!(gpt_oss::ModelArgs),
        case!(helium::ModelArgs),
        case!(hunyuan_moe::ModelArgs),
        case!(hunyuan_v1_dense::ModelArgs),
        case!(internlm2::ModelArgs),
        case!(internlm3::ModelArgs),
        case!(iquestloopcoder::ModelArgs),
        case!(klear::ModelArgs),
        case!(llada2_moe::ModelArgs),
        case!(llama4::TextArgs),
        case!(longcat_flash_ngram::LongcatFlashNgramConfig),
        case!(mimo::ModelArgs),
        case!(minicpm::ModelArgs),
        case!(minicpm3::ModelArgs),
        case!(minimax::ModelArgs),
        case!(minimax_m3::ModelArgs),
        case!(nanochat::ModelArgs),
        case!(nemotron::ModelArgs),
        case!(nemotron_nas::NemotronNASConfig),
        case!(seed_oss::ModelArgs),
        case!(smollm3::ModelArgs),
        case!(stablelm::ModelArgs),
        case!(starcoder2::StarCoder2Config),
        case!(step3p5::Step3p5Config),
        case!(telechat3::ModelArgs),
        // Families whose own resolver used to let the nested base win.
        case!(exaone_moe::ModelArgs, |c| Some(f64::from(
            c.get_rope_theta()
        ))),
        case!(mistral4::Mistral4Config, |c| Some(f64::from(c.rope_base()))),
        // VLM text configs.
        case!(ernie4_5_moe_vl::Ernie45MoeVlTextConfig),
        case!(falcon_ocr::FalconOcrConfig),
        case!(hunyuan_vl::HunyuanVlTextConfig),
        case!(mllama::config::MllamaTextConfig),
        case!(molmo::MolmoTextConfig),
        case!(molmo2::Molmo2TextConfig),
        case!(moondream2::ModelArgs),
        case!(paddleocr_vl::PaddleOcrTextConfig),
        case!(youtu_vl_lm::YoutuTextConfig),
        (
            "jina_vlm::JinaVlmTextConfig",
            (|value: Value| {
                Ok(Some(f64::from(
                    jina_vlm::JinaVlmTextConfig::from_json(&value).rope_theta,
                )))
            }) as ParseTheta,
        ),
        // Outside `src/models`.
        case!(omni::TalkerTextConfig),
        case!(omni::CodePredictorConfig),
        case!(omni::Code2WavConfig),
        (
            "mlxcel_core::drafter::dflash::DFlashConfig",
            (|value: Value| {
                mlxcel_core::drafter::dflash::DFlashConfig::from_json(&value)
                    .map(|c| Some(f64::from(c.rope_theta)))
            }) as ParseTheta,
        ),
    ]
}

/// The `` `name` `` right after `prefix` in a serde error message.
fn quoted_after<'a>(message: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = &message[message.find(prefix)? + prefix.len()..];
    rest.split('`').next()
}

/// A placeholder matching the `expected ...` clause of a type error, or the
/// first listed variant of an `unknown variant` error.
fn placeholder_for(message: &str) -> Option<Value> {
    if message.contains("unknown variant") {
        let variants = &message[message.find("expected")?..];
        return quoted_after(variants, "`").map(|v| json!(v));
    }
    let expected = &message[message.find("expected")?..];
    let value = if expected.contains("string") || expected.contains("variant identifier") {
        json!("silu")
    } else if expected.contains("bool") {
        json!(false)
    } else if expected.contains("sequence") || expected.contains("tuple") {
        json!([])
    } else if expected.contains("map") || expected.contains("struct") {
        json!({})
    } else if expected.contains("f32") || expected.contains("f64") || expected.contains("float") {
        json!(1.0)
    } else {
        return None;
    };
    Some(value)
}

fn set_path(config: &mut Value, path: &[String], value: Value) {
    let mut node = config;
    for key in &path[..path.len() - 1] {
        node = node.get_mut(key).expect("parent object exists");
    }
    node[path.last().expect("non-empty path")] = value;
}

fn get_path<'a>(config: &'a Value, path: &[String]) -> Option<&'a Value> {
    path.iter().try_fold(config, |node, key| node.get(key))
}

/// Grow `config` until `parse` accepts it, filling required non-RoPE fields.
fn fill_required_fields(parse: ParseTheta, mut config: Value) -> Result<Value, String> {
    let mut last: Option<Vec<String>> = None;
    let mut objects: Vec<Vec<String>> = Vec::new();
    for _ in 0..400 {
        let message = match parse(config.clone()) {
            Ok(_) => return Ok(config),
            Err(message) => message,
        };
        if let Some(field) = quoted_after(&message, "missing field `") {
            if matches!(field, "rope_theta" | "rope_scaling" | "rope_parameters") {
                return Err(message);
            }
            // A field already present at the top level is missing from the
            // object placeholder inserted most recently.
            let path = if config.get(field).is_none() {
                vec![field.to_string()]
            } else {
                let parent = objects
                    .iter()
                    .rev()
                    .find(|p| get_path(&config, p).is_some_and(|o| o.get(field).is_none()))
                    .ok_or_else(|| message.clone())?;
                let mut path = parent.clone();
                path.push(field.to_string());
                path
            };
            set_path(&mut config, &path, json!(1));
            last = Some(path);
            continue;
        }
        let (Some(path), Some(value)) = (last.clone(), placeholder_for(&message)) else {
            return Err(message);
        };
        if get_path(&config, &path) == Some(&value) {
            return Err(message);
        }
        if value.is_object() {
            objects.push(path.clone());
        }
        set_path(&mut config, &path, value);
    }
    Err("placeholder search did not converge".to_string())
}

#[test]
fn every_flat_rope_theta_config_lifts_rope_parameters_and_flat_wins() {
    let mut failures = Vec::new();
    for (name, parse) in cases() {
        let nested_only = json!({
            "rope_parameters": {"rope_theta": 123456.0, "rope_type": "default"}
        });
        let mut config = match fill_required_fields(parse, nested_only) {
            Ok(config) => config,
            Err(e) => {
                failures.push(format!("{name}: nested-only config does not parse: {e}"));
                continue;
            }
        };
        let nested = parse(config.clone());
        if nested != Ok(Some(123456.0)) {
            failures.push(format!("{name}: nested-only base {nested:?}, want 123456"));
        }
        config["rope_theta"] = json!(500000.0);
        let both = parse(config);
        if both != Ok(Some(500000.0)) {
            failures.push(format!("{name}: flat + nested base {both:?}, want 500000"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn llama3_nested_rope_type_llama3_resolves_llama3_scaling() {
    // The `rope_parameters` transformers 5.20 writes for Llama 3.2 1B.
    let args: crate::models::llama3::ModelArgs = serde_json::from_value(json!({
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "rms_norm_eps": 1e-5,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rope_parameters": {
            "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3",
            "rope_theta": 500000.0
        }
    }))
    .unwrap();
    assert_eq!(args.rope_theta, 500000.0);
    assert!(matches!(
        args.rope_scaling_kind(),
        crate::models::rope_utils::RopeScalingKind::Llama3 { .. }
    ));
}

#[test]
fn per_layer_type_rope_parameters_fill_gemma3_bases_and_scaling() {
    // transformers 5.20 writes Gemma 3 RoPE per layer type.
    let parse = cases()
        .into_iter()
        .find(|(name, _)| *name == "gemma3::ModelArgs")
        .unwrap()
        .1;
    let config = fill_required_fields(
        parse,
        json!({
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1000000.0}
            }
        }),
    )
    .unwrap();
    let args: crate::models::gemma3::ModelArgs = serde_json::from_value(config).unwrap();
    assert_eq!(args.rope_theta, 1_000_000.0);
    assert_eq!(args.rope_local_base_freq, 10_000.0);
    let scaling = args.rope_scaling.expect("linear scaling lifted");
    assert_eq!(scaling["type"], "linear");
    assert_eq!(scaling["factor"], 8.0);
}

#[test]
fn nested_partial_rotary_factor_reaches_phi() {
    // transformers 5 also moves `partial_rotary_factor` under
    // `rope_parameters`; Phi would otherwise rotate at its serde default
    // (0.4), so the fixture uses a different value.
    let parse = cases()
        .into_iter()
        .find(|(name, _)| *name == "phi::ModelArgs")
        .unwrap()
        .1;
    let config = fill_required_fields(
        parse,
        json!({
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "rope_type": "default"}
        }),
    )
    .unwrap();
    let args: crate::models::phi::ModelArgs = serde_json::from_value(config).unwrap();
    assert_eq!(args.partial_rotary_factor, 0.5);
}

#[test]
fn deepseek_v4_compress_yarn_reaches_rope_scaling() {
    // transformers 5.20 keeps the flat DeepSeek-V4 bases but moves the
    // compress-layer YaRN block under `rope_parameters.compress`.
    let parse = cases()
        .into_iter()
        .find(|(name, _)| *name == "deepseek_v4::ModelArgs")
        .unwrap()
        .1;
    let config = fill_required_fields(
        parse,
        json!({
            "rope_theta": 10000.0,
            "compress_rope_theta": 160000.0,
            "rope_parameters": {
                "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.125},
                "compress": {
                    "type": "yarn", "factor": 16.0, "original_max_position_embeddings": 65536,
                    "beta_fast": 32, "beta_slow": 1, "rope_theta": 160000.0, "rope_type": "yarn"
                }
            }
        }),
    )
    .unwrap();
    let args: crate::models::deepseek_v4::ModelArgs = serde_json::from_value(config).unwrap();
    assert_eq!(args.compress_rope_theta, 160000.0);
    let scaling = args.rope_scaling.expect("compress YaRN lifted");
    assert_eq!(scaling.scaling_type.as_deref(), Some("yarn"));
    assert_eq!(scaling.factor, Some(16.0));
    assert_eq!(scaling.original_max_position_embeddings, Some(65536));
}
