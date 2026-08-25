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

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlxcel_core::weights::load_weights_from_dir_with_subfolders;
use safetensors::tensor::{Dtype as SafeTensorDtype, View};
use serde_json::json;

use super::limits::{
    EMBEDDING_MAX_LENGTH_CAP, config_normalize_flag, derive_max_length, resolve_pad_token_id,
    resolve_vocab_size,
};
use super::loader::{QuantizationParams, load_embedding_model, quantization_params};
use super::tokenize_tests::bert_like_tokenizer;
use crate::models::{ModelType, get_model_type};

pub(crate) fn temp_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("mlxcel_embeddings_test_{name}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Render an `Err` as a string without requiring `Debug` on the `Ok` type.
pub(crate) fn err_string<T>(result: anyhow::Result<T>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(err) => err.to_string(),
    }
}

#[derive(Clone)]
struct OwnedTensor {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> SafeTensorDtype {
        SafeTensorDtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Write a one-tensor f32 safetensors file.
pub(crate) fn write_f32_safetensors(path: &Path, name: &str, values: &[f32]) {
    let tensor = OwnedTensor {
        shape: vec![values.len()],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    let mut views: HashMap<String, OwnedTensor> = HashMap::new();
    views.insert(name.to_string(), tensor);
    safetensors::serialize_to_file(&views, None, path).unwrap();
}

#[test]
fn subfolder_loader_prefixes_module_folders_and_skips_adapters() {
    let dir = temp_dir("subfolders");
    write_f32_safetensors(
        &dir.join("model.safetensors"),
        "encoder.weight",
        &[1.0, 2.0],
    );
    write_f32_safetensors(
        &dir.join("adapter_model.safetensors"),
        "lora.weight",
        &[9.0],
    );
    std::fs::create_dir_all(dir.join("2_Dense")).unwrap();
    write_f32_safetensors(
        &dir.join("2_Dense").join("model.safetensors"),
        "linear.weight",
        &[3.0, 4.0, 5.0],
    );
    write_f32_safetensors(
        &dir.join("2_Dense").join("consolidated.safetensors"),
        "linear.weight",
        &[0.0],
    );
    // A subfolder without safetensors contributes nothing and does not error.
    std::fs::create_dir_all(dir.join("1_Pooling")).unwrap();
    std::fs::write(dir.join("1_Pooling").join("config.json"), "{}").unwrap();
    // Hidden directories are never walked.
    std::fs::create_dir_all(dir.join(".cache")).unwrap();
    write_f32_safetensors(&dir.join(".cache").join("x.safetensors"), "hidden", &[0.0]);

    let weights = load_weights_from_dir_with_subfolders(&dir).unwrap();
    let mut names: Vec<&String> = weights.keys().collect();
    names.sort();
    assert_eq!(names, vec!["2_Dense.linear.weight", "encoder.weight"]);
    assert_eq!(
        mlxcel_core::array_shape(&weights["2_Dense.linear.weight"]),
        vec![3]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn subfolder_loader_requires_top_level_weights() {
    let dir = temp_dir("no_toplevel");
    std::fs::create_dir_all(dir.join("2_Dense")).unwrap();
    write_f32_safetensors(&dir.join("2_Dense").join("model.safetensors"), "w", &[1.0]);
    assert!(load_weights_from_dir_with_subfolders(&dir).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn quantization_params_reads_group_size_and_bits() {
    assert_eq!(
        quantization_params(&json!({"quantization": {"group_size": 64, "bits": 4}})),
        Some(QuantizationParams {
            group_size: 64,
            bits: 4
        })
    );
    assert_eq!(quantization_params(&json!({})), None);
    assert_eq!(
        quantization_params(&json!({"quantization": {"bits": 4}})),
        None
    );
}

#[test]
fn max_length_takes_the_smallest_declared_bound() {
    let dir = temp_dir("max_length");
    // Nothing declared: the hard cap.
    assert_eq!(
        derive_max_length(&dir, false, None),
        EMBEDDING_MAX_LENGTH_CAP
    );

    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"model_max_length": 512}"#,
    )
    .unwrap();
    assert_eq!(derive_max_length(&dir, false, None), 512);

    // The HuggingFace "unset" sentinel is ignored.
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"model_max_length": 1000000000000000019884624838656}"#,
    )
    .unwrap();
    assert_eq!(
        derive_max_length(&dir, false, None),
        EMBEDDING_MAX_LENGTH_CAP
    );

    std::fs::write(
        dir.join("sentence_bert_config.json"),
        r#"{"max_seq_length": 256}"#,
    )
    .unwrap();
    assert_eq!(derive_max_length(&dir, false, None), 256);

    // max_position_embeddings only counts for absolute-position encoders.
    std::fs::write(
        dir.join("config.json"),
        r#"{"max_position_embeddings": 128}"#,
    )
    .unwrap();
    assert_eq!(derive_max_length(&dir, false, None), 256);
    assert_eq!(derive_max_length(&dir, true, None), 128);

    // The operator override lowers but the smaller declared bound still wins.
    assert_eq!(derive_max_length(&dir, true, Some(64)), 64);
    assert_eq!(derive_max_length(&dir, true, Some(4096)), 128);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pad_token_and_vocab_size_resolve_from_configs() {
    let dir = temp_dir("pad_token");
    let tokenizer = bert_like_tokenizer(false);
    // No tokenizer_config: pad id falls back to 0.
    assert_eq!(resolve_pad_token_id(&dir, &tokenizer), 0);

    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"eos_token": "[SEP]"}"#,
    )
    .unwrap();
    assert_eq!(resolve_pad_token_id(&dir, &tokenizer), 102, "eos fallback");

    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"pad_token": {"content": "[CLS]"}, "eos_token": "[SEP]"}"#,
    )
    .unwrap();
    assert_eq!(
        resolve_pad_token_id(&dir, &tokenizer),
        101,
        "pad_token wins"
    );

    assert_eq!(
        resolve_vocab_size(&json!({"vocab_size": 30522}), &tokenizer),
        30522
    );
    assert_eq!(
        resolve_vocab_size(&json!({"text_config": {"vocab_size": 7}}), &tokenizer),
        7
    );
    let from_tokenizer = resolve_vocab_size(&json!({}), &tokenizer);
    assert!(from_tokenizer > 0);

    assert!(config_normalize_flag(&json!({})));
    assert!(!config_normalize_flag(&json!({"normalize": false})));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn load_embedding_model_rejects_generation_checkpoints() {
    let dir = temp_dir("qwen3_chat");
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 8}"#,
    )
    .unwrap();
    assert_eq!(get_model_type(&dir).unwrap(), ModelType::Qwen3);
    let err = err_string(load_embedding_model(&dir));
    assert!(err.contains("not an embedding checkpoint"), "{err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn load_embedding_model_reports_unported_family() {
    let dir = temp_dir("bert_unported");
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "bert", "architectures": ["BertModel"], "hidden_size": 8}"#,
    )
    .unwrap();
    assert_eq!(get_model_type(&dir).unwrap(), ModelType::Bert);
    let err = err_string(load_embedding_model(&dir));
    assert!(err.contains("not yet supported"), "{err}");
    assert!(err.contains("/v1/embeddings"), "{err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn generation_loader_bails_for_embedding_checkpoints() {
    let dir = temp_dir("bert_load_model");
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "bert", "architectures": ["BertModel"], "hidden_size": 8}"#,
    )
    .unwrap();
    let err = err_string(crate::load_model(&dir));
    assert!(err.contains("/v1/embeddings"), "{err}");
    assert!(err.contains("mlxcel embed"), "{err}");
    std::fs::remove_dir_all(dir).unwrap();
}
