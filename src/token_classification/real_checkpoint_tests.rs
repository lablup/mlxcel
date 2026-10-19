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

//! Real-checkpoint gates for `mlxcel tag` on Rampart.
//!
//! Soft-skip when the checkpoint is absent. Fetch with
//! `mlxcel download nativ-community/rampart-mlx-fp16` (and `-4bit`), or keep
//! the MLX copies under `models/mlx/rampart-mlx-fp16` / `-4bit`.
//!
//! The parity fixture `tests/fixtures/token_classification/
//! rampart_mlx_vlm_spans.json` was produced by
//! `tests/fixtures/token_classification/generate_rampart_spans.py` with
//! mlx-vlm 0.7.8 (git b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0) on mlx CPU;
//! the fixture records the oracle revision in its `oracle` field.

use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::models::embedding_test_support::{local_checkpoint, mlx_test_guard};

const RAMPART_FP16: &str = "nativ-community/rampart-mlx-fp16";
const RAMPART_4BIT: &str = "nativ-community/rampart-mlx-4bit";

/// The upstream model card's example and its documented output.
const README_TEXT: &str =
    "I'm Sarah Connor, 1984 Cyberdyne Ave, Los Angeles CA 90012, sarah@sky.net";
const README_REDACTED: &str =
    "I'm <GIVEN_NAME> <SURNAME>, <BUILDING_NUMBER> <STREET_NAME>, Los Angeles CA 90012, <EMAIL>";

/// `local_checkpoint`, plus the `models/mlx/<lowercase name>` layout this
/// repository's validation hosts use.
fn rampart(repo_id: &str) -> Option<PathBuf> {
    let mlx = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("mlx")
        .join(crate::downloader::repo_basename(repo_id).to_ascii_lowercase());
    if mlx.join("config.json").is_file() {
        return Some(mlx);
    }
    local_checkpoint(repo_id)
}

fn keep() -> TagOptions {
    TagOptions {
        keep_labels: ["CITY", "STATE", "ZIP_CODE"].map(str::to_string).to_vec(),
        ..TagOptions::default()
    }
}

fn tag(loaded: &mut LoadedTokenClassifier, text: &str, options: &TagOptions) -> TokenClassificationResult {
    loaded.classifier.classify(text, options).expect("tagging succeeds")
}

#[test]
fn rampart_readme_example_redacts_like_upstream() {
    let Some(fp16_dir) = rampart(RAMPART_FP16) else {
        return;
    };
    let _guard = mlx_test_guard();
    let mut fp16 = load_token_classifier(&fp16_dir).expect("fp16 Rampart loads");
    assert_eq!(fp16.kind, TokenModelKind::BertTokenClassifier);
    let reference = tag(&mut fp16, README_TEXT, &keep());
    assert_eq!(reference.redacted_text, README_REDACTED);
    eprintln!("rampart fp16 ({}): {}", fp16_dir.display(), reference.redacted_text);

    let Some(q4_dir) = rampart(RAMPART_4BIT) else {
        return;
    };
    let mut q4 = load_token_classifier(&q4_dir).expect("4-bit Rampart loads");
    let quantized = tag(&mut q4, README_TEXT, &keep());
    assert_eq!(quantized.spans, reference.spans, "4-bit spans match fp16");
    assert_eq!(quantized.redacted_text, README_REDACTED);
    eprintln!("rampart 4bit ({}): {}", q4_dir.display(), quantized.redacted_text);
}

#[test]
fn rampart_spans_match_the_mlx_vlm_fixture() {
    let Some(dir) = rampart(RAMPART_FP16) else {
        return;
    };
    let _guard = mlx_test_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/rampart_mlx_vlm_spans.json"
    ))
    .expect("fixture parses");
    let keep_labels: Vec<String> = serde_json::from_value(fixture["keep_labels"].clone()).unwrap();
    let options = TagOptions {
        keep_labels,
        ..TagOptions::default()
    };
    let mut loaded = load_token_classifier(&dir).expect("fp16 Rampart loads");
    let cases = fixture["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 4, "three sentences plus the multi-window case");
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let tokens = loaded.classifier.tokenize(text).unwrap();
        assert_eq!(
            tokens.ids.len() as u64,
            case["tokens"].as_u64().unwrap(),
            "token count for {text:?}"
        );
        let result = tag(&mut loaded, text, &options);
        let actual = serde_json::to_value(&result).unwrap();
        for key in ["text", "spans", "redacted_text"] {
            assert_eq!(actual[key], case[key], "{key} for {text:?}");
        }
        eprintln!("parity ok: {} tokens, {} spans", tokens.ids.len(), result.spans.len());
    }
}
