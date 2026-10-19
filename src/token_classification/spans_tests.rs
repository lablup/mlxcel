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

//! Span-decoding cases ported from mlx-vlm
//! (`mlx_vlm/tests/test_extraction_models.py` at git
//! b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0, mlx-vlm 0.7.8), plus the BIOES,
//! orphan-`E`, trimming, overlap and multi-byte cases the issue lists.

use super::*;

fn labels(tags: &[&str]) -> Vec<String> {
    tags.iter().map(|tag| tag.to_string()).collect()
}

/// Upstream's `BIO_LABELS`.
fn bio() -> LabelInfo {
    build_label_info(&labels(&["O", "B-NAME", "I-NAME", "B-PHONE", "I-PHONE"])).unwrap()
}

/// A BIOES set with two entities.
fn bioes() -> LabelInfo {
    build_label_info(&labels(&[
        "O", "B-X", "I-X", "E-X", "S-X", "B-Y", "I-Y", "E-Y", "S-Y",
    ]))
    .unwrap()
}

fn tag(info: &LabelInfo, name: &str) -> usize {
    info.labels.iter().position(|label| label == name).unwrap()
}

fn path(info: &LabelInfo, tags: &[&str]) -> Vec<usize> {
    tags.iter().map(|name| tag(info, name)).collect()
}

/// `(entity name, token start, token end)` runs, for readable assertions.
fn runs(info: &LabelInfo, tags: &[&str]) -> Vec<(String, usize, usize)> {
    token_spans(&path(info, tags), info)
        .into_iter()
        .map(|(span, start, end)| (info.span_names[span].clone(), start, end))
        .collect()
}

fn run(name: &str, start: usize, end: usize) -> (String, usize, usize) {
    (name.to_string(), start, end)
}

#[test]
fn bio_labels_index_like_upstream() {
    let info = bio();
    assert_eq!(info.span_names, ["O", "NAME", "PHONE"]);
    assert_eq!(info.background, 0);
    let name = &info.states_by_span["NAME"];
    assert_eq!(name.get(&Boundary::Begin), Some(&1));
    assert_eq!(name.get(&Boundary::Inside), Some(&2));
    assert_eq!(name.len(), 2);
}

#[test]
fn invalid_and_backgroundless_label_sets_fail() {
    let error = build_label_info(&labels(&["O", "NAME"])).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Invalid token label 'NAME'; expected BIO or BIOES labels"
    );
    for bad in ["B-", "X-NAME", "-NAME"] {
        let error = build_label_info(&labels(&["O", bad])).unwrap_err();
        assert!(error.to_string().contains("expected BIO or BIOES"), "{bad}");
    }
    let error = build_label_info(&labels(&["B-NAME", "I-NAME"])).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Token labels must include the background label 'O'"
    );
}

#[test]
fn label_splits_at_the_first_hyphen() {
    let info = build_label_info(&labels(&["O", "B-STREET-NAME"])).unwrap();
    assert_eq!(info.span_names, ["O", "STREET-NAME"]);
}

#[test]
fn bio_path_builds_runs() {
    let info = bio();
    assert_eq!(
        runs(&info, &["O", "B-NAME", "I-NAME", "O", "B-PHONE", "B-PHONE", "I-PHONE"]),
        [run("NAME", 1, 3), run("PHONE", 4, 5), run("PHONE", 5, 7)]
    );
    // An open span runs to the end of the path.
    assert_eq!(runs(&info, &["B-NAME", "I-NAME"]), [run("NAME", 0, 2)]);
}

#[test]
fn bioes_path_builds_runs() {
    let info = bioes();
    assert_eq!(
        runs(&info, &["B-X", "I-X", "E-X", "O", "S-X", "S-Y", "B-Y", "E-Y"]),
        [run("X", 0, 3), run("X", 4, 5), run("Y", 5, 6), run("Y", 6, 8)]
    );
    // `S` closes whatever was open before emitting itself.
    assert_eq!(
        runs(&info, &["B-X", "S-Y", "I-X"]),
        [run("X", 0, 1), run("Y", 1, 2), run("X", 2, 3)]
    );
}

#[test]
fn inside_after_a_different_entity_starts_a_new_span() {
    let info = bio();
    assert_eq!(
        runs(&info, &["B-NAME", "I-PHONE", "I-PHONE", "I-NAME"]),
        [run("NAME", 0, 1), run("PHONE", 1, 3), run("NAME", 3, 4)]
    );
    // `I` with nothing open opens a span.
    assert_eq!(runs(&info, &["O", "I-NAME", "I-NAME"]), [run("NAME", 1, 3)]);
}

#[test]
fn orphan_end_becomes_a_one_token_span() {
    let info = bioes();
    assert_eq!(runs(&info, &["O", "E-X"]), [run("X", 1, 2)]);
    // A mismatched `E` closes the open span and stands alone.
    assert_eq!(
        runs(&info, &["B-Y", "I-Y", "E-X", "I-X"]),
        [run("Y", 0, 2), run("X", 2, 3), run("X", 3, 4)]
    );
}

/// Upstream `test_token_spans_group_subwords_and_touching_words`: a word is
/// tagged if any piece is (`Mc` is `O` but `##Kay` is a name), and
/// same-entity words touching through punctuation merge.
#[test]
fn group_subwords_and_merge_touching_words() {
    let text = "call Mary Ann Smith at 617-555 now, McKay";
    let tokens: [((usize, usize), u32, &str); 13] = [
        ((0, 4), 0, "O"),
        ((5, 7), 1, "B-NAME"),
        ((7, 9), 1, "B-NAME"),
        ((10, 13), 2, "B-NAME"),
        ((14, 19), 3, "I-NAME"),
        ((20, 22), 4, "O"),
        ((23, 26), 5, "B-PHONE"),
        ((26, 27), 6, "B-PHONE"),
        ((27, 30), 7, "I-PHONE"),
        ((31, 34), 8, "O"),
        ((34, 35), 9, "O"),
        ((36, 38), 10, "O"),
        ((38, 41), 10, "B-NAME"),
    ];
    let info = bio();
    let offsets: Vec<(usize, usize)> = tokens.iter().map(|t| t.0).collect();
    let word_ids: Vec<Option<u32>> = tokens.iter().map(|t| Some(t.1)).collect();
    let tags: Vec<&str> = tokens.iter().map(|t| t.2).collect();
    let path = path(&info, &tags);

    let grouped = decode_spans(text, &path, &offsets, &info, Some(&word_ids), true).unwrap();
    let grouped: Vec<(&str, &str)> = grouped
        .iter()
        .map(|s| (s.label.as_str(), s.text.as_str()))
        .collect();
    assert_eq!(
        grouped,
        [
            ("NAME", "Mary"),
            ("NAME", "Ann Smith"),
            ("PHONE", "617-555"),
            ("NAME", "McKay"),
        ]
    );

    let token_level = decode_spans(text, &path, &offsets, &info, None, true).unwrap();
    let token_level: Vec<&str> = token_level.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(token_level, ["Ma", "ry", "Ann Smith", "617", "-555", "Kay"]);
}

#[test]
fn group_words_relabels_the_rest_of_the_word_as_inside() {
    let info = bio();
    let path = path(&info, &["O", "B-PHONE", "B-NAME", "O"]);
    let word_ids = [Some(0), Some(0), Some(0), None];
    let grouped = group_words(&path, &word_ids, &info);
    assert_eq!(grouped, [tag(&info, "B-PHONE"), tag(&info, "I-PHONE"), tag(&info, "I-PHONE"), 0]);
}

#[test]
fn whitespace_is_trimmed_and_empty_spans_are_dropped() {
    // Byte-level BPE style offsets include the leading space of a word.
    let text = "ask  Ada \t now";
    let info = bio();
    let offsets = [(0, 3), (3, 8), (8, 10), (10, 14)];
    let path = path(&info, &["O", "B-NAME", "B-PHONE", "O"]);
    let spans = decode_spans(text, &path, &offsets, &info, None, true).unwrap();
    assert_eq!(spans.len(), 1, "the whitespace-only PHONE span is dropped");
    assert_eq!((spans[0].start, spans[0].end, spans[0].text.as_str()), (5, 8, "Ada"));

    let untrimmed = decode_spans(text, &path, &offsets, &info, None, false).unwrap();
    assert_eq!(untrimmed[0].text, "  Ada");
    assert_eq!(untrimmed[1].text, " \t");
}

#[test]
fn overlapping_spans_keep_the_longest_first() {
    let text = "Ada Lovelace";
    let info = bio();
    // Tokens 0 and 1 cover the same characters (a byte-level split of one
    // character); token 2 covers the whole name under another label.
    let offsets = [(0, 3), (0, 3), (0, 12)];
    let path = path(&info, &["B-NAME", "B-NAME", "B-PHONE"]);
    let spans = decode_spans(text, &path, &offsets, &info, None, true).unwrap();
    let spans: Vec<(&str, usize, usize)> = spans
        .iter()
        .map(|s| (s.label.as_str(), s.start, s.end))
        .collect();
    assert_eq!(spans, [("PHONE", 0, 12)]);

    // Equal ranges tie-break on the label.
    let offsets = [(0, 3), (0, 3)];
    let path = path(&info, &["B-PHONE", "B-NAME"]);
    let spans = decode_spans(text, &path, &offsets, &info, None, true).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].label, "NAME");
}

#[test]
fn merging_requires_word_ids_and_the_same_label() {
    let text = "617-555";
    let info = bio();
    let offsets = [(0, 3), (3, 4), (4, 7)];
    let word_ids = [Some(0), Some(1), Some(2)];
    let phone = path(&info, &["B-PHONE", "B-PHONE", "B-PHONE"]);
    let merged = decode_spans(text, &phone, &offsets, &info, Some(&word_ids), true).unwrap();
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].text, "617-555");
    let unmerged = decode_spans(text, &phone, &offsets, &info, None, true).unwrap();
    assert_eq!(unmerged.len(), 3);
    let mixed = path(&info, &["B-PHONE", "B-NAME", "B-PHONE"]);
    let mixed = decode_spans(text, &mixed, &offsets, &info, Some(&word_ids), true).unwrap();
    assert_eq!(mixed.len(), 3);
}

#[test]
fn offsets_are_unicode_scalar_indices() {
    // "Café " is 5 chars but 6 bytes, so a byte offset would land inside "Ada".
    let text = "Café Ada, Zoë";
    let info = bio();
    let offsets = [(0, 4), (5, 8), (8, 9), (10, 13)];
    let path = path(&info, &["O", "B-NAME", "O", "B-NAME"]);
    let spans = decode_spans(text, &path, &offsets, &info, None, true).unwrap();
    let got: Vec<(usize, usize, &str)> =
        spans.iter().map(|s| (s.start, s.end, s.text.as_str())).collect();
    assert_eq!(got, [(5, 8, "Ada"), (10, 13, "Zoë")]);
    assert_eq!(redact(text, &spans, None, &[]), "Café <NAME>, <NAME>");
}

#[test]
fn placeholder_normalizes_the_label() {
    assert_eq!(placeholder("private_person"), "<PRIVATE_PERSON>");
    assert_eq!(placeholder("GIVEN_NAME"), "<GIVEN_NAME>");
    assert_eq!(placeholder("street name"), "<STREET_NAME>");
    assert_eq!(placeholder("a.b--c"), "<A_B_C>");
    assert_eq!(placeholder("__x__"), "<X>");
    assert_eq!(placeholder("--"), "<REDACTED>");
    assert_eq!(placeholder(""), "<REDACTED>");
}

#[test]
fn redaction_honors_replacement_and_keep() {
    let text = "Alice emailed bob@example.com";
    let span = |label: &str, start, end, text: &str| TokenSpan {
        label: label.to_string(),
        start,
        end,
        text: text.to_string(),
        placeholder: placeholder(label),
    };
    let spans = [
        span("private_person", 0, 5, "Alice"),
        span("private_email", 14, 29, "bob@example.com"),
    ];
    assert_eq!(
        redact(text, &spans, None, &[]),
        "<PRIVATE_PERSON> emailed <PRIVATE_EMAIL>"
    );
    assert_eq!(redact(text, &spans, Some("[x]"), &[]), "[x] emailed [x]");
    assert_eq!(
        redact(text, &spans, None, &["private_email".to_string()]),
        "<PRIVATE_PERSON> emailed bob@example.com"
    );
    // An unknown keep label is ignored.
    assert_eq!(
        redact(text, &spans, None, &["NOT_A_LABEL".to_string()]),
        "<PRIVATE_PERSON> emailed <PRIVATE_EMAIL>"
    );
    assert_eq!(redact(text, &[], None, &[]), text);
}

#[test]
fn decode_spans_rejects_mismatched_inputs() {
    let info = bio();
    assert!(decode_spans("ab", &[0, 1], &[(0, 1)], &info, None, true).is_err());
    assert!(decode_spans("ab", &[9], &[(0, 1)], &info, None, true).is_err());
    assert!(decode_spans("ab", &[1], &[(0, 1)], &info, Some(&[]), true).is_err());
}

#[test]
fn result_serializes_like_upstream_to_dict() {
    let info = bio();
    let text = "hi Ada";
    let spans = decode_spans(text, &path(&info, &["O", "B-NAME"]), &[(0, 2), (3, 6)], &info, None, true)
        .unwrap();
    let result = TokenClassificationResult {
        text: text.to_string(),
        redacted_text: redact(text, &spans, None, &[]),
        spans,
    };
    assert_eq!(
        serde_json::to_value(&result).unwrap(),
        serde_json::json!({
            "text": "hi Ada",
            "spans": [{"label": "NAME", "start": 3, "end": 6, "text": "Ada", "placeholder": "<NAME>"}],
            "redacted_text": "hi <NAME>",
        })
    );
}
