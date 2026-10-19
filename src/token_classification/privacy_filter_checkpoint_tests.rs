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

//! Real-checkpoint gates for `mlxcel tag` on OpenAI Privacy Filter.
//!
//! Soft-skip when the checkpoint is absent. Fetch with
//! `mlxcel download mlx-community/openai-privacy-filter-bf16` (and `-8bit`),
//! or keep the MLX copies under `models/mlx/openai-privacy-filter-bf16` /
//! `-8bit`.
//!
//! The fixture `tests/fixtures/token_classification/privacy_filter_mlx_vlm.json`
//! was produced by `generate_privacy_filter_fixture.py` in the same directory
//! with mlx-vlm 0.7.8 (git 73d366ecf9233b3fc6549531a7f2386726d09ff6) on mlx
//! CPU; its `oracle` field records the revision. Run these with
//! `MLX_ENABLE_TF32=0`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::models::embedding_test_support::{local_checkpoint, mlx_test_guard};
use crate::token_classification::loader::{
    TokenClassifierLoadOptions, load_token_classifier, load_token_classifier_with,
};
use crate::token_classification::{
    LoadedTokenClassifier, TagOptions, TokenClassificationResult, TokenModelKind,
};

const PRIVACY_FILTER_BF16: &str = "mlx-community/openai-privacy-filter-bf16";
const PRIVACY_FILTER_8BIT: &str = "mlx-community/openai-privacy-filter-8bit";

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/privacy_filter_mlx_vlm.json"
    ))
    .expect("fixture parses")
}

/// `local_checkpoint`, plus the `models/mlx/<lowercase name>` layout this
/// repository's validation hosts use.
fn privacy_filter(repo_id: &str) -> Option<PathBuf> {
    let mlx = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("mlx")
        .join(crate::downloader::repo_basename(repo_id).to_ascii_lowercase());
    if mlx.join("config.json").is_file() {
        return Some(mlx);
    }
    local_checkpoint(repo_id)
}

fn tag(
    loaded: &mut LoadedTokenClassifier,
    text: &str,
    options: &TagOptions,
) -> TokenClassificationResult {
    loaded
        .classifier
        .classify(text, options)
        .expect("tagging succeeds")
}

fn case_options(case: &Value, decode: DecodeMode) -> TagOptions {
    let options = &case["options"];
    TagOptions {
        decode: Some(decode),
        replacement: options
            .get("replacement")
            .and_then(Value::as_str)
            .map(str::to_string),
        keep_labels: options
            .get("keep_labels")
            .map(|labels| serde_json::from_value(labels.clone()).unwrap())
            .unwrap_or_default(),
        ..TagOptions::default()
    }
}

/// Max |p_mlxcel - p_mlx-vlm| over the per-token label probabilities of one
/// sentence, bf16 on this device against bf16 on mlx CPU. In probability
/// space this is 2e-2 nats on a dominant label and proportionally tighter
/// below it. The log domain is not gated over all labels: the oracle's own
/// f32 forward moves tail labels (below -18 nats) by up to 4.5 nats against
/// its bf16 run while moving labels above -10 nats by at most 0.14, so a
/// 2e-2 log-domain bound over every label measures bf16 rounding of
/// near-zero probabilities, not the port.
const PROB_TOLERANCE: f32 = 2e-2;
/// Log-domain bound on labels the reference puts above -10 nats, where bf16
/// rounding is small; measured 0.14 for the oracle's own f32 vs bf16 runs.
const MASS_LOG_PROB_TOLERANCE: f32 = 0.5;

#[test]
fn privacy_filter_log_probs_match_the_mlx_vlm_fixture() {
    let Some(dir) = privacy_filter(PRIVACY_FILTER_BF16) else {
        return;
    };
    let _guard = mlx_test_guard();
    let fixture = fixture();
    let mut loaded = load_token_classifier(&dir).expect("bf16 Privacy Filter loads");
    assert_eq!(loaded.kind, TokenModelKind::PrivacyFilter);
    assert_eq!(loaded.model_type, "openai_privacy_filter");
    let labels: Vec<String> = serde_json::from_value(fixture["labels"].clone()).unwrap();
    assert_eq!(loaded.classifier.model().labels(), labels.as_slice());
    assert_eq!(loaded.classifier.window(), 128_000);

    let text = fixture["log_probs"]["text"].as_str().unwrap();
    let want_tokens: Vec<u32> =
        serde_json::from_value(fixture["log_probs"]["tokens"].clone()).unwrap();
    let tokens = loaded.classifier.tokenize(text).unwrap();
    assert_eq!(tokens.ids, want_tokens, "tokenization of {text:?}");
    assert!(tokens.word_ids.is_none(), "no word grouping");

    let scores = loaded.classifier.scores(&tokens.ids).unwrap();
    let want: Vec<Vec<f32>> =
        serde_json::from_value(fixture["log_probs"]["values"].clone()).unwrap();
    assert_eq!(scores.len(), want.len());
    // (diff, token, label, mlxcel, mlx-vlm) in log and probability space.
    let mut log_diffs: Vec<(f32, usize, usize, f32, f32)> = Vec::new();
    let mut prob_diffs: Vec<(f32, usize, usize, f32, f32)> = Vec::new();
    let mut mass_worst = 0.0f32;
    for (t, (got_row, want_row)) in scores.iter().zip(&want).enumerate() {
        assert_eq!(got_row.len(), want_row.len());
        assert_eq!(
            argmax_path(&[got_row.clone()]),
            argmax_path(&[want_row.clone()]),
            "top label of token {t}"
        );
        for (l, (g, w)) in got_row.iter().zip(want_row).enumerate() {
            let diff = (g - w).abs();
            log_diffs.push((diff, t, l, *g, *w));
            prob_diffs.push(((g.exp() - w.exp()).abs(), t, l, *g, *w));
            if *w > -10.0 {
                mass_worst = mass_worst.max(diff);
            }
        }
    }
    log_diffs.sort_by(|a, b| b.0.total_cmp(&a.0));
    prob_diffs.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!(
        "log-prob parity over {} tokens x {} labels: max |dp| {:.5} (token {} label {}), max |dlogp| {:.4} (token {} label {}: mlxcel {:.3}, mlx-vlm {:.3}), max |dlogp| on labels above -10 nats {mass_worst:.4}",
        scores.len(),
        labels.len(),
        prob_diffs[0].0,
        prob_diffs[0].1,
        labels[prob_diffs[0].2],
        log_diffs[0].0,
        log_diffs[0].1,
        labels[log_diffs[0].2],
        log_diffs[0].3,
        log_diffs[0].4,
    );
    for (diff, t, l, g, w) in log_diffs.iter().take(6) {
        eprintln!(
            "  |dlogp| {diff:.4} token {t} {}: mlxcel {g:.3} mlx-vlm {w:.3}",
            labels[*l]
        );
    }
    assert!(
        prob_diffs[0].0 <= PROB_TOLERANCE,
        "max |dp| {:.5} at token {} label {} exceeds {PROB_TOLERANCE}",
        prob_diffs[0].0,
        prob_diffs[0].1,
        labels[prob_diffs[0].2]
    );
    assert!(
        mass_worst <= MASS_LOG_PROB_TOLERANCE,
        "max |dlogp| {mass_worst:.4} on labels above -10 nats exceeds {MASS_LOG_PROB_TOLERANCE}"
    );
}

fn assert_case_matches(loaded: &mut LoadedTokenClassifier, case: &Value, label: &str) {
    let text = case["text"].as_str().unwrap();
    let tokens = loaded.classifier.tokenize(text).unwrap();
    assert_eq!(
        tokens.ids.len() as u64,
        case["tokens"].as_u64().unwrap(),
        "{label}: token count for {text:?}"
    );
    for decode in [DecodeMode::Viterbi, DecodeMode::Argmax] {
        let result = tag(loaded, text, &case_options(case, decode));
        let actual = serde_json::to_value(&result).unwrap();
        let want = &case[decode.as_str()];
        let (got, expected) = (
            actual["spans"].as_array().unwrap(),
            want["spans"].as_array().unwrap(),
        );
        if let Some(index) =
            (0..got.len().max(expected.len())).find(|&i| got.get(i) != expected.get(i))
        {
            panic!(
                "{label} ({}): span {index} of {} (mlx-vlm has {}) differs for a {}-token text:\n  mlxcel:  {:?}\n  mlx-vlm: {:?}",
                decode.as_str(),
                got.len(),
                expected.len(),
                tokens.ids.len(),
                got.get(index),
                expected.get(index)
            );
        }
        assert_eq!(
            actual["redacted_text"],
            want["redacted_text"],
            "{label} ({}): redacted_text for {text:?}",
            decode.as_str()
        );
        eprintln!(
            "{label} ({}): parity ok, {} tokens, {} spans",
            decode.as_str(),
            tokens.ids.len(),
            result.spans.len()
        );
    }
}

#[test]
fn privacy_filter_spans_match_the_mlx_vlm_fixture_under_viterbi_and_argmax() {
    let Some(dir) = privacy_filter(PRIVACY_FILTER_BF16) else {
        return;
    };
    let _guard = mlx_test_guard();
    let fixture = fixture();
    let mut loaded = load_token_classifier(&dir).expect("bf16 Privacy Filter loads");
    let cases = fixture["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 8, "six sentences, replacement and keep");
    for (index, case) in cases.iter().enumerate() {
        assert_case_matches(&mut loaded, case, &format!("bf16 case {index}"));
    }
}

/// The acceptance sentence end to end on bf16, with `--replacement`,
/// `--keep` and `--operating-point`, and the same spans from the 8-bit build.
#[test]
fn privacy_filter_acceptance_sentence_redacts_like_upstream() {
    let Some(bf16_dir) = privacy_filter(PRIVACY_FILTER_BF16) else {
        return;
    };
    let _guard = mlx_test_guard();
    const TEXT: &str = "Alice Smith can be reached at alice@example.com.";
    const REDACTED: &str = "<PRIVATE_PERSON> can be reached at <PRIVATE_EMAIL>.";

    let mut bf16 = load_token_classifier_with(
        &bf16_dir,
        &TokenClassifierLoadOptions {
            operating_point: Some("default".into()),
        },
    )
    .expect("bf16 Privacy Filter loads with the default operating point");
    let reference = tag(&mut bf16, TEXT, &TagOptions::default());
    assert_eq!(reference.redacted_text, REDACTED);
    let labels: Vec<&str> = reference.spans.iter().map(|s| s.label.as_str()).collect();
    assert_eq!(labels, ["private_person", "private_email"]);
    assert_eq!(reference.spans[0].text, "Alice Smith");
    assert_eq!(reference.spans[1].text, "alice@example.com");

    let argmax = tag(
        &mut bf16,
        TEXT,
        &TagOptions {
            decode: Some(DecodeMode::Argmax),
            ..TagOptions::default()
        },
    );
    assert_eq!(argmax.spans, reference.spans, "--argmax");
    let replaced = tag(
        &mut bf16,
        TEXT,
        &TagOptions {
            replacement: Some("[REDACTED]".into()),
            ..TagOptions::default()
        },
    );
    assert_eq!(
        replaced.redacted_text,
        "[REDACTED] can be reached at [REDACTED]."
    );
    let kept = tag(
        &mut bf16,
        TEXT,
        &TagOptions {
            keep_labels: vec!["private_email".into()],
            ..TagOptions::default()
        },
    );
    assert_eq!(
        kept.redacted_text,
        "<PRIVATE_PERSON> can be reached at alice@example.com."
    );
    assert_eq!(kept.spans, reference.spans, "--keep reports the span");
    eprintln!("privacy filter bf16 ({}): {}", bf16_dir.display(), REDACTED);

    let Some(q8_dir) = privacy_filter(PRIVACY_FILTER_8BIT) else {
        return;
    };
    let mut q8 = load_token_classifier(&q8_dir).expect("8-bit Privacy Filter loads");
    let quantized = tag(&mut q8, TEXT, &TagOptions::default());
    assert_eq!(quantized.spans, reference.spans, "8-bit spans match bf16");
    assert_eq!(quantized.redacted_text, REDACTED);
    eprintln!("privacy filter 8bit ({}): {}", q8_dir.display(), REDACTED);
}

/// The 8-bit export prints the fixture's spans too.
#[test]
fn privacy_filter_8bit_matches_the_fixture_cases() {
    let Some(dir) = privacy_filter(PRIVACY_FILTER_8BIT) else {
        return;
    };
    let _guard = mlx_test_guard();
    let fixture = fixture();
    let mut loaded = load_token_classifier(&dir).expect("8-bit Privacy Filter loads");
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        assert_case_matches(&mut loaded, case, &format!("8bit case {index}"));
    }
}

/// Text past `attention_chunk_size` (4096) and `moe_chunk_size` (2048)
/// scores the same whether attention and routing run in the default chunks
/// or in one piece, on the real bf16 weights. The oracle cannot pin this
/// case: on repeated sentences the model itself sits at 50/50 on dozens of
/// tokens, so the comparison is between two forwards of the same weights.
#[test]
fn privacy_filter_long_text_is_chunk_invariant_on_real_weights() {
    use crate::models::openai_privacy_filter::PrivacyFilterModel;
    use mlxcel_core::utils::array_to_vec_f32;

    let Some(dir) = privacy_filter(PRIVACY_FILTER_BF16) else {
        return;
    };
    let _guard = mlx_test_guard();
    let sentences = [
        "Alice Smith can be reached at alice@example.com.",
        "Call Maria Garcia on +1 617-555-0142 before 2024-05-17 to confirm.",
        "Wire the deposit to account 4111 1111 1111 1111 and quote ticket 8842.",
        "The API key is sk-live-9f8e7d6c5b4a3210 and the docs are at https://internal.example.org/wiki.",
    ];
    let text: String = (0..170)
        .map(|index| format!("Note {index}: {}", sentences[index % sentences.len()]))
        .collect::<Vec<_>>()
        .join(" ");

    let mut loaded = load_token_classifier(&dir).expect("bf16 Privacy Filter loads");
    let tokens = loaded.classifier.tokenize(&text).unwrap();
    assert!(
        tokens.ids.len() > 4096,
        "{} tokens do not reach the attention chunk",
        tokens.ids.len()
    );
    let chunked = loaded.classifier.scores(&tokens.ids).unwrap();
    drop(loaded);

    let mut args = PrivacyFilterModel::read_args(&dir).unwrap();
    args.attention_chunk_size = 1 << 20;
    args.moe_chunk_size = 1 << 20;
    let unchunked = PrivacyFilterModel::load_with(&dir, args).expect("unchunked model loads");
    let ids: Vec<i32> = tokens.ids.iter().map(|&id| id as i32).collect();
    let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);
    let logits = mlxcel_core::astype(&unchunked.logits(&input_ids), mlxcel_core::dtype::FLOAT32);
    let log_probs = mlxcel_core::log_softmax(&logits, -1);
    mlxcel_core::eval(&log_probs);
    let flat = array_to_vec_f32(&log_probs);
    let width = chunked[0].len();
    assert_eq!(flat.len(), chunked.len() * width);

    // Gated like the oracle comparison: probabilities within 2e-2, labels
    // above -10 nats within 0.5 nats. Tail labels move by nats under bf16
    // reordering alone, so the log domain is only reported there.
    let (mut worst_prob, mut worst_mass, mut worst_tail) = (0.0f32, 0.0f32, 0.0f32);
    let mut flips = 0usize;
    for (t, row) in chunked.iter().enumerate() {
        let other = &flat[t * width..(t + 1) * width];
        for (g, w) in row.iter().zip(other) {
            let diff = (g - w).abs();
            worst_prob = worst_prob.max((g.exp() - w.exp()).abs());
            if *w > -10.0 {
                worst_mass = worst_mass.max(diff);
            } else {
                worst_tail = worst_tail.max(diff);
            }
        }
        if argmax_path(&[row.clone()]) != argmax_path(&[other.to_vec()]) {
            flips += 1;
        }
    }
    eprintln!(
        "chunk invariance on {} tokens: max |dp| {worst_prob:.5}, max |dlogp| {worst_mass:.4} above -10 nats, {worst_tail:.4} below, {flips} argmax flips",
        chunked.len()
    );
    assert!(
        worst_prob <= PROB_TOLERANCE,
        "max |dp| {worst_prob} between chunked and unchunked forwards"
    );
    assert!(
        worst_mass <= MASS_LOG_PROB_TOLERANCE,
        "max |dlogp| {worst_mass} on labels above -10 nats between chunked and unchunked forwards"
    );
    assert!(
        flips * 200 <= chunked.len(),
        "{flips} of {} argmax labels flipped between chunked and unchunked forwards",
        chunked.len()
    );
}
