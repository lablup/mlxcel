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

use mlxcel_core::utils::array_to_vec_f32;

use super::tokenize::{
    EncodeOptions, EncodedBatch, EncodedRow, encode_batch, encode_pairs,
    strip_padding_and_truncation, truncate_keeping_trailing_special, truncate_token_ids,
};
use crate::tokenizer::MlxcelTokenizer;

pub(crate) const CLS: u32 = 101;
pub(crate) const SEP: u32 = 102;
pub(crate) const PAD: u32 = 0;

/// A BERT-shaped word-level tokenizer with the `[CLS] A [SEP]` /
/// `[CLS] A [SEP] B [SEP]` templates, so trailing-special truncation and
/// pair segment ids can be asserted without a checkpoint.
pub(crate) fn bert_like_tokenizer(with_fixed_padding: bool) -> MlxcelTokenizer {
    let padding = if with_fixed_padding {
        r#"{"strategy": {"Fixed": 8}, "direction": "Right", "pad_to_multiple_of": null,
            "pad_id": 0, "pad_type_id": 0, "pad_token": "[PAD]"}"#
    } else {
        "null"
    };
    let truncation = if with_fixed_padding {
        r#"{"max_length": 8, "strategy": "LongestFirst", "stride": 0, "direction": "Right"}"#
    } else {
        "null"
    };
    let json = format!(
        r#"{{
        "version": "1.0",
        "truncation": {truncation},
        "padding": {padding},
        "added_tokens": [
            {{"id": 0, "content": "[PAD]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}},
            {{"id": 101, "content": "[CLS]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}},
            {{"id": 102, "content": "[SEP]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}}
        ],
        "normalizer": null,
        "pre_tokenizer": {{"type": "Whitespace"}},
        "post_processor": {{
            "type": "TemplateProcessing",
            "single": [
                {{"SpecialToken": {{"id": "[CLS]", "type_id": 0}}}},
                {{"Sequence": {{"id": "A", "type_id": 0}}}},
                {{"SpecialToken": {{"id": "[SEP]", "type_id": 0}}}}
            ],
            "pair": [
                {{"SpecialToken": {{"id": "[CLS]", "type_id": 0}}}},
                {{"Sequence": {{"id": "A", "type_id": 0}}}},
                {{"SpecialToken": {{"id": "[SEP]", "type_id": 0}}}},
                {{"Sequence": {{"id": "B", "type_id": 1}}}},
                {{"SpecialToken": {{"id": "[SEP]", "type_id": 1}}}}
            ],
            "special_tokens": {{
                "[CLS]": {{"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]}},
                "[SEP]": {{"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}}
            }}
        }},
        "decoder": null,
        "model": {{
            "type": "WordLevel",
            "unk_token": "[UNK]",
            "vocab": {{
                "[PAD]": 0, "[UNK]": 1, "hello": 3, "world": 4, "a": 5, "b": 6, "c": 7,
                "d": 8, "e": 9, "f": 10, "[CLS]": 101, "[SEP]": 102
            }}
        }}
    }}"#
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .expect("bert-like test tokenizer parses");
    MlxcelTokenizer::HuggingFace(tokenizer)
}

fn opts(max_length: usize, with_token_type_ids: bool) -> EncodeOptions {
    EncodeOptions {
        add_special_tokens: true,
        max_length,
        with_token_type_ids,
    }
}

#[test]
fn encode_batch_right_pads_and_counts_real_tokens() {
    let tokenizer = bert_like_tokenizer(false);
    let batch = encode_batch(
        &tokenizer,
        &["hello world", "a"],
        opts(64, false),
        PAD,
        None,
    )
    .unwrap();
    assert_eq!(batch.batch, 2);
    assert_eq!(batch.width, 4);
    assert_eq!(batch.token_counts, vec![4, 3]);
    assert_eq!(batch.total_tokens(), 7);
    let (cls, sep, pad) = (CLS as i32, SEP as i32, PAD as i32);
    assert_eq!(batch.input_ids, vec![cls, 3, 4, sep, cls, 5, sep, pad]);
    assert_eq!(batch.attention_mask, vec![1, 1, 1, 1, 1, 1, 1, 0]);
    assert!(batch.token_type_ids.is_none());

    // The MLX arrays mirror the flat buffers.
    let ids = array_to_vec_f32(&batch.input_ids_array());
    assert_eq!(ids, vec![101.0, 3.0, 4.0, 102.0, 101.0, 5.0, 102.0, 0.0]);
    let mask = array_to_vec_f32(&batch.attention_mask_array());
    assert_eq!(mask, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
    assert!(batch.token_type_ids_array().is_none());
}

#[test]
fn encode_batch_pads_to_fixed_width_when_requested() {
    let tokenizer = bert_like_tokenizer(false);
    let batch = encode_batch(&tokenizer, &["a b"], opts(64, false), PAD, Some(6)).unwrap();
    assert_eq!(batch.width, 6);
    assert_eq!(batch.attention_mask, vec![1, 1, 1, 1, 0, 0]);
}

#[test]
fn encode_batch_truncates_keeping_trailing_special_token() {
    let tokenizer = bert_like_tokenizer(false);
    let batch = encode_batch(&tokenizer, &["a b c d e f"], opts(4, false), PAD, None).unwrap();
    assert_eq!(batch.token_counts, vec![4]);
    // [CLS] a b [SEP]: the trailing [SEP] survives, the tail of the text goes.
    assert_eq!(batch.input_ids, vec![CLS as i32, 5, 6, SEP as i32]);
}

#[test]
fn truncate_keeps_only_trailing_specials_and_handles_tiny_limits() {
    let ids = vec![CLS, 5, 6, 7, SEP];
    let special = [1, 0, 0, 0, 1];
    let row = truncate_keeping_trailing_special(ids.clone(), None, &special, 3);
    assert_eq!(row.ids, vec![CLS, 5, SEP]);

    // A limit equal to the length is a no-op.
    let row = truncate_keeping_trailing_special(ids.clone(), None, &special, 5);
    assert_eq!(row.ids, ids);

    // A limit of 1 keeps just the trailing special token.
    let row = truncate_keeping_trailing_special(ids.clone(), None, &special, 1);
    assert_eq!(row.ids, vec![SEP]);

    // Segment ids follow the same cut.
    let row = truncate_keeping_trailing_special(ids, Some(vec![0, 0, 1, 1, 1]), &special, 3);
    assert_eq!(row.type_ids, Some(vec![0, 0, 1]));

    // Verbatim token ids truncate from the right without bookkeeping.
    assert_eq!(truncate_token_ids(&[1, 2, 3, 4], 2), vec![1, 2]);
    assert_eq!(truncate_token_ids(&[1, 2], 8), vec![1, 2]);
}

#[test]
fn encode_batch_token_type_ids_for_pairs() {
    let tokenizer = bert_like_tokenizer(false);
    let batch = encode_pairs(
        &tokenizer,
        &[("hello", "world a")],
        opts(64, true),
        PAD,
        None,
    )
    .unwrap();
    let (cls, sep) = (CLS as i32, SEP as i32);
    assert_eq!(batch.input_ids, vec![cls, 3, sep, 4, 5, sep]);
    assert_eq!(batch.token_type_ids, Some(vec![0, 0, 0, 1, 1, 1]));

    // Single texts carry all-zero segment ids when asked for.
    let single = encode_batch(&tokenizer, &["hello"], opts(64, true), PAD, None).unwrap();
    assert_eq!(single.token_type_ids, Some(vec![0, 0, 0]));

    // Pair truncation keeps the trailing [SEP] of the pair template.
    let cut = encode_pairs(&tokenizer, &[("a b c", "d e f")], opts(5, true), PAD, None).unwrap();
    assert_eq!(cut.input_ids, vec![cls, 5, 6, 7, sep]);
    assert_eq!(cut.token_type_ids, Some(vec![0, 0, 0, 0, 1]));
}

#[test]
fn strip_padding_and_truncation_removes_builtin_settings() {
    let padded = bert_like_tokenizer(true);
    let hf = padded.hf_tokenizer().unwrap();
    let raw = hf.encode("a", true).unwrap();
    assert_eq!(raw.get_ids().len(), 8, "the fixture pads to a fixed 8");

    let stripped = strip_padding_and_truncation(padded);
    let hf = stripped.hf_tokenizer().unwrap();
    assert!(hf.get_padding().is_none());
    assert!(hf.get_truncation().is_none());
    let encoding = hf.encode("a", true).unwrap();
    assert_eq!(encoding.get_ids(), &[CLS, 5, SEP]);
    let long = hf.encode("a b c d e f a b c d", true).unwrap();
    assert_eq!(long.get_ids().len(), 12, "no built-in truncation either");
}

#[test]
fn from_rows_fills_missing_segment_ids_with_zeros() {
    let rows = vec![
        EncodedRow {
            ids: vec![1, 2],
            type_ids: Some(vec![0, 1]),
        },
        EncodedRow {
            ids: vec![3],
            type_ids: None,
        },
    ];
    let batch = EncodedBatch::from_rows(&rows, 9, None);
    assert_eq!(batch.input_ids, vec![1, 2, 3, 9]);
    assert_eq!(batch.token_type_ids, Some(vec![0, 1, 0, 0]));
    assert_eq!(batch.token_counts, vec![2, 1]);
}
