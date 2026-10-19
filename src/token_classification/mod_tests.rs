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

//! Pipeline tests over a stub [`TokenClassifierModel`] and in-memory
//! tokenizers: windowing, special-token wrapping, decode-mode resolution,
//! redaction, and char offsets. `windows_long_inputs_and_redacts` ports
//! upstream `test_token_classifier_windows_long_inputs_and_redacts`
//! (mlx-vlm 0.7.8, git b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0).

use super::*;

const CLS: u32 = 101;
const SEP: u32 = 102;

fn bio_labels() -> Vec<String> {
    ["O", "B-NAME", "I-NAME", "B-PHONE", "I-PHONE"]
        .map(str::to_string)
        .to_vec()
}

/// A word-level tokenizer: whitespace split (punctuation stays attached),
/// with a `truncation` block the pipeline must strip.
fn word_tokenizer(vocab: &[(&str, u32)]) -> MlxcelTokenizer {
    let vocab: serde_json::Map<String, serde_json::Value> = vocab
        .iter()
        .map(|(word, id)| (word.to_string(), serde_json::json!(id)))
        .chain([("[UNK]".to_string(), serde_json::json!(1))])
        .collect();
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": {"max_length": 2, "strategy": "LongestFirst", "stride": 0, "direction": "Right"},
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "unk_token": "[UNK]", "vocab": vocab},
    });
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.to_string().as_bytes())
        .expect("word-level test tokenizer parses");
    MlxcelTokenizer::HuggingFace(tokenizer)
}

/// A lower-casing WordPiece tokenizer behind the BERT pre-tokenizer, so a
/// word can span several sub-tokens and punctuation splits into its own word.
fn wordpiece_tokenizer(vocab: &[(&str, u32)]) -> MlxcelTokenizer {
    let vocab: serde_json::Map<String, serde_json::Value> = vocab
        .iter()
        .map(|(piece, id)| (piece.to_string(), serde_json::json!(id)))
        .chain([("[UNK]".to_string(), serde_json::json!(1))])
        .collect();
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                       "strip_accents": null, "lowercase": true},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 100, "vocab": vocab},
    });
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.to_string().as_bytes())
        .expect("wordpiece test tokenizer parses");
    MlxcelTokenizer::HuggingFace(tokenizer)
}

/// Upstream's `_CapitalizedNameModel`, generalized: ids in `name_ids` score
/// `B-NAME`, everything else `O`. It records every window it is handed.
struct StubModel {
    labels: Vec<String>,
    context: usize,
    special: (Vec<u32>, Vec<u32>),
    modes: Vec<DecodeMode>,
    name_ids: Vec<u32>,
    windows: Vec<Vec<u32>>,
    /// Make each row depend on the token's position inside its window, so a
    /// windowing bug changes the scores and not just the call count.
    positional: bool,
}

impl StubModel {
    fn new(context: usize, name_ids: &[u32]) -> Self {
        Self {
            labels: bio_labels(),
            context,
            special: (vec![CLS], vec![SEP]),
            modes: vec![DecodeMode::Argmax],
            name_ids: name_ids.to_vec(),
            windows: Vec::new(),
            positional: false,
        }
    }

    /// Raw scores for one wrapped window, exactly what `log_probs` returns.
    fn raw(&self, window: &[u32]) -> Vec<Vec<f32>> {
        window
            .iter()
            .enumerate()
            .map(|(position, id)| {
                let mut row = vec![-10.0f32; self.labels.len()];
                let hot = if self.name_ids.contains(id) { 1 } else { 0 };
                row[hot] = 10.0;
                if self.positional {
                    row[3] = position as f32 * 0.5 + *id as f32 * 0.01;
                    row[4] = -(position as f32);
                }
                row
            })
            .collect()
    }
}

impl TokenClassifierModel for StubModel {
    fn labels(&self) -> &[String] {
        &self.labels
    }
    fn context_size(&self) -> usize {
        self.context
    }
    fn special_tokens(&self) -> (Vec<u32>, Vec<u32>) {
        self.special.clone()
    }
    fn group_words(&self) -> bool {
        true
    }
    fn decode_modes(&self) -> &[DecodeMode] {
        &self.modes
    }
    fn log_probs(&mut self, window_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.windows.push(window_ids.to_vec());
        Ok(self.raw(window_ids))
    }
}

/// Upstream `_WordTokenizer` ids: capitalized words are names.
fn sentence_vocab() -> Vec<(&'static str, u32)> {
    vec![
        ("ask", 10),
        ("and", 11),
        ("or", 12),
        ("today", 13),
        ("Ada", 20),
        ("Grace", 21),
        ("Linus", 22),
    ]
}

const NAMES: [u32; 3] = [20, 21, 22];

#[test]
fn windows_long_inputs_and_redacts() {
    let model = StubModel::new(4, &NAMES);
    let mut classifier = TokenClassifier::new(model, word_tokenizer(&sentence_vocab())).unwrap();
    assert_eq!(classifier.window(), 2);

    let result = classifier
        .classify("ask Ada and Grace or Linus today", &TagOptions::default())
        .unwrap();
    let windows = &classifier.model().windows;
    assert_eq!(
        windows.iter().map(Vec::len).collect::<Vec<_>>(),
        [4, 4, 4, 3]
    );
    assert!(windows.iter().all(|w| w[0] == CLS && w[w.len() - 1] == SEP));
    let texts: Vec<&str> = result.spans.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["Ada", "Grace", "Linus"]);
    assert_eq!(
        result.redacted_text,
        "ask <NAME> and <NAME> or <NAME> today"
    );

    let kept = classifier
        .classify(
            "ask Ada",
            &TagOptions {
                keep_labels: vec!["NAME".to_string()],
                ..TagOptions::default()
            },
        )
        .unwrap();
    assert_eq!(kept.redacted_text, "ask Ada");

    let error = classifier
        .classify(
            "ask Ada",
            &TagOptions {
                decode: Some(DecodeMode::Viterbi),
                ..TagOptions::default()
            },
        )
        .unwrap_err();
    assert_eq!(error.to_string(), "decode must be 'argmax'");
}

/// The issue's windowing gate: a text of three windows gives the same scores
/// and path as scoring each window by hand.
#[test]
fn three_windows_match_scoring_each_window_by_hand() {
    let mut model = StubModel::new(5, &NAMES);
    model.positional = true;
    let mut classifier = TokenClassifier::new(model, word_tokenizer(&sentence_vocab())).unwrap();
    let window = classifier.window();
    assert_eq!(window, 3);

    let text = "ask Ada and Grace or Linus today Ada";
    let tokens = classifier.tokenize(text).unwrap();
    assert_eq!(
        tokens.ids.len(),
        8,
        "the tokenizer's truncation block was stripped"
    );
    let scores = classifier.scores(&tokens.ids).unwrap();
    assert_eq!(classifier.model().windows.len(), 3);

    let reference = StubModel::new(5, &NAMES);
    let reference = StubModel {
        positional: true,
        ..reference
    };
    let mut by_hand = Vec::new();
    for chunk in tokens.ids.chunks(window) {
        let mut wrapped = vec![CLS];
        wrapped.extend_from_slice(chunk);
        wrapped.push(SEP);
        for mut row in reference
            .raw(&wrapped)
            .into_iter()
            .skip(1)
            .take(chunk.len())
        {
            log_softmax_row(&mut row);
            by_hand.push(row);
        }
    }
    assert_eq!(scores, by_hand);
    assert_eq!(argmax_path(&scores), argmax_path(&by_hand));
    // Every row is a normalized distribution.
    for row in &scores {
        let total: f32 = row.iter().map(|v| v.exp()).sum();
        assert!((total - 1.0).abs() < 1e-5, "{total}");
    }
    // Positional scores make the windowing visible: the path differs from
    // scoring the whole text as one window.
    let whole: Vec<u32> = [CLS]
        .into_iter()
        .chain(tokens.ids.iter().copied())
        .chain([SEP])
        .collect();
    let mut one_window: Vec<Vec<f32>> = reference.raw(&whole)[1..=tokens.ids.len()].to_vec();
    one_window.iter_mut().for_each(|row| log_softmax_row(row));
    assert_ne!(scores, one_window);
}

#[test]
fn a_word_split_across_a_window_boundary_stays_one_span() {
    // `mckay` is `mc` + `##kay`; with one token per window the two pieces are
    // scored in different windows, and only `##kay` is tagged.
    let vocab = [
        ("call", 10),
        ("mc", 11),
        ("##kay", 20),
        ("now", 12),
        (",", 13),
    ];
    let model = StubModel::new(3, &[20]);
    let mut classifier = TokenClassifier::new(model, wordpiece_tokenizer(&vocab)).unwrap();
    assert_eq!(classifier.window(), 1);
    let result = classifier
        .classify("call McKay, now", &TagOptions::default())
        .unwrap();
    assert_eq!(classifier.model().windows.len(), 5);
    let spans: Vec<(&str, usize, usize, &str)> = result
        .spans
        .iter()
        .map(|s| (s.label.as_str(), s.start, s.end, s.text.as_str()))
        .collect();
    assert_eq!(spans, [("NAME", 5, 10, "McKay")]);
    assert_eq!(result.redacted_text, "call <NAME>, now");
}

#[test]
fn offsets_survive_multi_byte_text() {
    let vocab = [("Café", 10), ("Zoë", 20), ("Ada", 21), ("und", 11)];
    let model = StubModel::new(8, &[20, 21]);
    let mut classifier = TokenClassifier::new(model, word_tokenizer(&vocab)).unwrap();
    let result = classifier
        .classify("Café Zoë und Ada", &TagOptions::default())
        .unwrap();
    let spans: Vec<(usize, usize, &str)> = result
        .spans
        .iter()
        .map(|s| (s.start, s.end, s.text.as_str()))
        .collect();
    assert_eq!(spans, [(5, 8, "Zoë"), (13, 16, "Ada")]);
    assert_eq!(result.redacted_text, "Café <NAME> und <NAME>");
    let replaced = classifier
        .classify(
            "Café Zoë und Ada",
            &TagOptions {
                replacement: Some("█".to_string()),
                ..TagOptions::default()
            },
        )
        .unwrap();
    assert_eq!(replaced.redacted_text, "Café █ und █");
}

#[test]
fn empty_text_returns_the_text_unchanged_without_scoring() {
    let model = StubModel::new(4, &NAMES);
    let mut classifier = TokenClassifier::new(model, word_tokenizer(&sentence_vocab())).unwrap();
    for text in ["", "   "] {
        let result = classifier.classify(text, &TagOptions::default()).unwrap();
        assert!(result.spans.is_empty());
        assert_eq!(result.redacted_text, text);
    }
    assert!(classifier.model().windows.is_empty());
}

#[test]
fn first_decode_mode_is_the_default_and_argmax_can_be_forced() {
    let mut model = StubModel::new(4, &NAMES);
    model.modes = vec![DecodeMode::Viterbi, DecodeMode::Argmax];
    let classifier = TokenClassifier::new(model, word_tokenizer(&sentence_vocab())).unwrap();
    assert_eq!(
        classifier.resolve_decode(None).unwrap(),
        DecodeMode::Viterbi
    );
    assert_eq!(
        classifier.resolve_decode(Some(DecodeMode::Argmax)).unwrap(),
        DecodeMode::Argmax
    );
}

#[test]
fn construction_validates_the_model() {
    let mut no_room = StubModel::new(2, &NAMES);
    no_room.special = (vec![CLS], vec![SEP]);
    let error = TokenClassifier::new(no_room, word_tokenizer(&sentence_vocab()))
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        "context_size must leave room for at least one token"
    );

    let mut bad_labels = StubModel::new(4, &NAMES);
    bad_labels.labels = vec!["B-NAME".to_string()];
    let error = TokenClassifier::new(bad_labels, word_tokenizer(&sentence_vocab()))
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        "Token labels must include the background label 'O'"
    );

    let mut no_modes = StubModel::new(4, &NAMES);
    no_modes.modes.clear();
    assert!(TokenClassifier::new(no_modes, word_tokenizer(&sentence_vocab())).is_err());
}

#[test]
fn windows_without_special_tokens_use_the_whole_context() {
    let mut model = StubModel::new(3, &NAMES);
    model.special = (Vec::new(), Vec::new());
    let mut classifier = TokenClassifier::new(model, word_tokenizer(&sentence_vocab())).unwrap();
    assert_eq!(classifier.window(), 3);
    classifier
        .classify("ask Ada and Grace", &TagOptions::default())
        .unwrap();
    assert_eq!(classifier.model().windows, [vec![10, 20, 11], vec![21]]);
}

#[test]
fn a_model_returning_the_wrong_row_count_fails() {
    struct Short(StubModel);
    impl TokenClassifierModel for Short {
        fn labels(&self) -> &[String] {
            self.0.labels()
        }
        fn context_size(&self) -> usize {
            self.0.context_size()
        }
        fn special_tokens(&self) -> (Vec<u32>, Vec<u32>) {
            self.0.special_tokens()
        }
        fn group_words(&self) -> bool {
            true
        }
        fn decode_modes(&self) -> &[DecodeMode] {
            self.0.decode_modes()
        }
        fn log_probs(&mut self, window_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
            let mut rows = self.0.log_probs(window_ids)?;
            rows.pop();
            Ok(rows)
        }
    }
    let model = Short(StubModel::new(4, &NAMES));
    let mut classifier = TokenClassifier::new(model, word_tokenizer(&sentence_vocab())).unwrap();
    assert!(
        classifier
            .classify("ask Ada", &TagOptions::default())
            .is_err()
    );
}

#[test]
fn boxed_models_drive_the_same_pipeline() {
    let model: Box<dyn TokenClassifierModel> = Box::new(StubModel::new(4, &NAMES));
    let mut classifier = TokenClassifier::new(model, word_tokenizer(&sentence_vocab())).unwrap();
    let result = classifier
        .classify("ask Ada", &TagOptions::default())
        .unwrap();
    assert_eq!(result.redacted_text, "ask <NAME>");
}

#[test]
fn argmax_takes_the_first_maximum() {
    assert_eq!(
        argmax_path(&[vec![1.0, 3.0, 3.0], vec![2.0, 1.0], vec![]]),
        [1, 0, 0]
    );
}
