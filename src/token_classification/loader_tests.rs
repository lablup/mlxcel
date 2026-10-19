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

//! Detection: Rampart is a token classifier, and the BERT rerank and
//! embedding routing next to it is unchanged.

use serde_json::json;

use super::*;
use crate::models::{ModelType, get_model_type};

/// `nativ-community/rampart-mlx-fp16`'s `config.json`, committed verbatim.
fn rampart_config() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/rampart_config.json"
    ))
    .expect("fixture parses")
}

fn bert_config(architecture: &str) -> Value {
    json!({
        "model_type": "bert",
        "architectures": [architecture],
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "id2label": {"0": "O", "1": "B-X", "2": "I-X"},
    })
}

/// Write `config` as the only file of a fresh checkpoint directory.
fn checkpoint_dir(config: &Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("config written");
    dir
}

#[test]
fn rampart_detects_as_a_bert_token_classifier() {
    let config = rampart_config();
    assert_eq!(
        detect_token_model_kind(&config).unwrap(),
        TokenModelKind::BertTokenClassifier
    );
    let labels = id2label_labels(&config).unwrap();
    assert_eq!(labels.len(), 35);
    assert_eq!(labels[0], "O");
    assert_eq!(labels[1], "B-GIVEN_NAME");
    assert_eq!(labels[34], "I-ZIP_CODE");
    assert_eq!(
        TokenModelKind::BertTokenClassifier.output_shape(),
        TokenOutputShape::TokenClassification
    );
}

#[test]
fn xlm_roberta_spellings_detect_too() {
    for model_type in ["xlm-roberta", "xlm_roberta", "XLM-RoBERTa"] {
        let mut config = bert_config("XLMRobertaForTokenClassification");
        config["model_type"] = json!(model_type);
        assert_eq!(
            detect_token_model_kind(&config).unwrap(),
            TokenModelKind::BertTokenClassifier,
            "{model_type}"
        );
    }
}

#[test]
fn sequence_classifiers_and_embedders_are_not_token_models_and_keep_their_routing() {
    for (architecture, expected) in [
        ("BertForSequenceClassification", ModelType::SequenceClassifier),
        ("BertModel", ModelType::Bert),
    ] {
        let config = bert_config(architecture);
        let error = detect_token_model_kind(&config).unwrap_err().to_string();
        assert!(error.contains("not a token-classification"), "{error}");
        assert!(error.contains(architecture), "{error}");
        let dir = checkpoint_dir(&config);
        assert_eq!(get_model_type(dir.path()).unwrap(), expected, "{architecture}");
    }
    // The generation dispatch is untouched: Rampart's encoder-only
    // `model_type` still detects as the BERT embedder there, exactly as
    // before this module existed.
    let dir = checkpoint_dir(&rampart_config());
    assert_eq!(get_model_type(dir.path()).unwrap(), ModelType::Bert);
}

#[test]
fn other_families_are_not_claimed() {
    let mut config = bert_config("DebertaV2ForTokenClassification");
    config["model_type"] = json!("deberta-v2");
    assert!(detect_token_model_kind(&config).is_err());
}

#[test]
fn id2label_must_be_present_and_complete() {
    let mut config = bert_config("BertForTokenClassification");
    config.as_object_mut().unwrap().remove("id2label");
    assert_eq!(
        detect_token_model_kind(&config).unwrap_err().to_string(),
        "Token-classification checkpoints must define id2label"
    );
    config["id2label"] = json!({});
    assert!(detect_token_model_kind(&config).is_err());

    config["id2label"] = json!({"0": "O", "2": "B-X"});
    assert!(id2label_labels(&config).is_err(), "gap at index 1");
    config["id2label"] = json!({"0": "O", "x": "B-X"});
    assert!(id2label_labels(&config).is_err());
    config["id2label"] = json!({"0": "O", "1": 7});
    assert!(id2label_labels(&config).is_err());

    // Keys are read as indices, not in JSON order.
    config["id2label"] = json!({"1": "B-X", "0": "O"});
    assert_eq!(id2label_labels(&config).unwrap(), ["O", "B-X"]);
}

#[test]
fn a_token_classifier_passed_to_extract_names_mlxcel_tag() {
    let dir = checkpoint_dir(&rampart_config());
    let error = load_span_extractor(dir.path()).err().unwrap().to_string();
    assert!(error.contains("use `mlxcel tag`"), "{error}");
    assert!(error.contains("token-classification checkpoint"), "{error}");
}

#[test]
fn wrong_subcommand_error_names_the_right_command_both_ways() {
    let dir = std::path::Path::new("/models/x");
    let tag = wrong_subcommand_error(
        dir,
        TokenModelKind::BertTokenClassifier,
        TokenOutputShape::SpanExtraction,
    );
    assert_eq!(
        tag.to_string(),
        "/models/x is a token-classification checkpoint (bert_token_classifier), not a \
         span-extraction one; use `mlxcel tag` instead of `mlxcel extract`"
    );
    assert_eq!(TokenOutputShape::SpanExtraction.subcommand(), "mlxcel extract");
    assert_eq!(TokenOutputShape::TokenClassification.subcommand(), "mlxcel tag");
}

#[test]
fn a_non_token_checkpoint_passed_to_tag_lists_the_supported_families() {
    let dir = checkpoint_dir(&bert_config("BertModel"));
    let error = format!("{:#}", load_token_classifier(dir.path()).err().unwrap());
    assert!(error.contains("cannot be loaded by `mlxcel tag`"), "{error}");
    assert!(error.contains("ForTokenClassification"), "{error}");
}
