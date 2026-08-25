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

//! Reranker-kind detection and the shared item / label helpers.
//!
//! Config-only, so nothing here builds an MLX array or needs the MLX test
//! guard.

use serde_json::json;

use super::*;
use crate::models::{ModelType, get_model_type};

/// Write a checkpoint directory carrying only `config.json`.
fn checkpoint(config: serde_json::Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string_pretty(&config).expect("config serializes"),
    )
    .expect("config written");
    dir
}

#[test]
fn detects_sequence_classifier_for_bert_modernbert_xlm_roberta() {
    for (model_type, architecture) in [
        ("bert", "BertForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
        ("xlm_roberta", "XLMRobertaForSequenceClassification"),
        ("modernbert", "ModernBertForSequenceClassification"),
    ] {
        let config = json!({"model_type": model_type, "architectures": [architecture]});
        assert_eq!(
            detect_reranker_kind(&config).expect("{model_type} detects"),
            RerankerKind::SequenceClassifier,
            "{model_type}"
        );
        // The same config must also route through `get_model_type`, which is
        // what makes `-m <cross-encoder>` work without `--reranker-model`.
        let dir = checkpoint(config);
        assert_eq!(
            get_model_type(dir.path()).expect("detection succeeds"),
            ModelType::SequenceClassifier,
            "{model_type}"
        );
    }
}

#[test]
fn detects_the_generative_kinds_by_model_type() {
    let qwen3 = json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]});
    assert_eq!(
        detect_reranker_kind(&qwen3).expect("qwen3 detects"),
        RerankerKind::GenerativeText
    );
    let qwen3_vl =
        json!({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]});
    assert_eq!(
        detect_reranker_kind(&qwen3_vl).expect("qwen3_vl detects"),
        RerankerKind::GenerativeVl
    );

    // Neither is detectable from `-m` alone: their configs are ordinary chat
    // exports, so `get_model_type` must keep returning the chat family.
    assert_eq!(
        get_model_type(checkpoint(qwen3).path()).expect("detection succeeds"),
        ModelType::Qwen3
    );
    assert_eq!(
        get_model_type(checkpoint(qwen3_vl).path()).expect("detection succeeds"),
        ModelType::Qwen3VL
    );
}

#[test]
fn rejects_multi_label_classifier() {
    let err = require_single_label(3).expect_err("a 3-label head is not a reranker");
    assert!(
        err.to_string()
            .contains("must expose exactly one output label"),
        "{err}"
    );
    require_single_label(1).expect("a one-label head is a reranker");
}

#[test]
fn num_labels_reads_num_labels_then_id2label_then_defaults_to_one() {
    assert_eq!(num_labels(&json!({"num_labels": 5})), 5);
    assert_eq!(
        num_labels(&json!({"id2label": {"0": "a", "1": "b"}})),
        2,
        "id2label length stands in for a missing num_labels"
    );
    assert_eq!(num_labels(&json!({})), 1);
    assert_eq!(
        num_labels(&json!({"num_labels": 0, "id2label": {"0": "a"}})),
        1,
        "a zero num_labels falls through rather than advertising an empty head"
    );
}

#[test]
fn rejects_unsupported_family() {
    let deberta = json!({
        "model_type": "deberta-v2",
        "architectures": ["DebertaV2ForSequenceClassification"],
    });
    let err = detect_reranker_kind(&deberta).expect_err("deberta has no reranker port");
    assert!(
        err.to_string().contains("Unsupported reranker model type"),
        "{err}"
    );

    let roberta_chat = json!({"model_type": "roberta", "architectures": ["RobertaModel"]});
    let err = detect_reranker_kind(&roberta_chat).expect_err("a plain encoder is not a reranker");
    assert!(
        err.to_string().contains("Unsupported reranker model type"),
        "{err}"
    );
}

#[test]
fn rejects_embedding_checkpoint_as_reranker() {
    // `BertModel` is the embedding export; without the classification head
    // there is nothing to read a relevance logit from.
    let config = json!({"model_type": "bert", "architectures": ["BertModel"]});
    let err = detect_reranker_kind(&config).expect_err("an embedder is not a reranker");
    assert!(
        err.to_string().contains("Unsupported reranker model type"),
        "{err}"
    );
    // And detection still routes it to the embedding family.
    assert_eq!(
        get_model_type(checkpoint(config).path()).expect("detection succeeds"),
        ModelType::Bert
    );
}

#[test]
fn logit_score_modules_json_is_generative_vl_not_embedding() {
    // `Qwen/Qwen3-VL-Reranker-2B` ships a `modules.json` whose only extra
    // module is `1_LogitScore`. That is not a Pooling module, so detection
    // must keep the checkpoint on the generative Qwen3-VL path rather than
    // claiming it as an embedding export.
    let dir = checkpoint(json!({
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "text_config": {"hidden_size": 2048},
    }));
    std::fs::write(
        dir.path().join("modules.json"),
        serde_json::to_string_pretty(&json!([
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.base.modules.transformer.Transformer",
            },
            {
                "idx": 1,
                "name": "1",
                "path": "1_LogitScore",
                "type": "sentence_transformers.cross_encoder.modules.logit_score.LogitScore",
            },
        ]))
        .expect("modules.json serializes"),
    )
    .expect("modules.json written");

    assert_eq!(
        get_model_type(dir.path()).expect("detection succeeds"),
        ModelType::Qwen3VL,
        "a LogitScore module must not be read as a Pooling module"
    );
    let config = super::loader::read_reranker_config_for_tests(dir.path());
    assert_eq!(
        detect_reranker_kind(&config).expect("kind detects"),
        RerankerKind::GenerativeVl
    );
}

#[test]
fn kinds_report_their_instruction_support_and_batch_defaults() {
    assert!(!RerankerKind::SequenceClassifier.accepts_instruction());
    assert!(RerankerKind::GenerativeText.accepts_instruction());
    assert!(RerankerKind::GenerativeVl.accepts_instruction());

    assert_eq!(
        RerankerKind::SequenceClassifier.default_batch_size(),
        DEFAULT_RERANK_BATCH_SIZE
    );
    assert_eq!(
        RerankerKind::GenerativeText.default_batch_size(),
        DEFAULT_RERANK_BATCH_SIZE
    );
    assert_eq!(
        RerankerKind::GenerativeVl.default_batch_size(),
        DEFAULT_RERANK_VL_BATCH_SIZE
    );
    assert_eq!(
        RerankerKind::SequenceClassifier.as_str(),
        "sequence_classifier"
    );
}

#[test]
fn items_report_emptiness_and_text() {
    assert!(RerankItem::default().is_empty());
    assert!(RerankItem::text("   ").is_empty());
    assert!(!RerankItem::text("hi").is_empty());
    assert_eq!(RerankItem::text("hi").text_or_empty(), "hi");
    assert_eq!(RerankItem::default().text_or_empty(), "");
    assert!(!RerankItem::text("hi").has_image());
}
