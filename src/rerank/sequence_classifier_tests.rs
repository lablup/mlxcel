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

//! Cross-encoder pair-encoding gates.
//!
//! The forward pass belongs to the merged `BertSequenceClassifier` /
//! `ModernBertSequenceClassifier` heads and is covered by their own tests and
//! by `real_checkpoint_tests.rs`. What `/v1/rerank` adds is the pair
//! tokenization: the checkpoint's own template, segment ids on the BERT
//! dialect, and the tokenizer's longest-first truncation.
//!
//! Tokenizer-only, so nothing here builds an MLX array.

use super::*;
use crate::embeddings::tokenize::{EncodeOptions, encode_pair_row};
use crate::embeddings::tokenize_tests::{CLS, SEP, bert_like_tokenizer};
use tokenizers::TruncationStrategy;

fn opts(max_length: usize, with_token_type_ids: bool) -> EncodeOptions {
    EncodeOptions {
        add_special_tokens: true,
        max_length,
        with_token_type_ids,
    }
}

#[test]
fn pairs_are_encoded_with_type_ids_for_bert() {
    let tokenizer = bert_like_tokenizer(false);
    let row = encode_pair_row(&tokenizer, "hello", "world", opts(32, true))
        .expect("pair encoding succeeds");
    // `[CLS] query [SEP] document [SEP]`, with segment 1 on the document half
    // and its closing separator.
    assert_eq!(row.ids, vec![CLS, 3, SEP, 4, SEP]);
    assert_eq!(
        row.type_ids.as_deref(),
        Some([0u32, 0, 0, 1, 1].as_slice()),
        "the document half must carry segment id 1"
    );

    // XLM-RoBERTa and ModernBERT do not ask for segment ids, and the batch
    // then carries none at all.
    let without = encode_pair_row(&tokenizer, "hello", "world", opts(32, false))
        .expect("pair encoding succeeds");
    assert_eq!(without.ids, row.ids);
    assert_eq!(without.type_ids, None);
}

#[test]
fn longest_first_truncation_splits_the_pair() {
    let tokenizer = with_longest_first_truncation(bert_like_tokenizer(false), 8)
        .expect("truncation configures");
    // Eight columns total, three of which the pair template spends on
    // `[CLS] ... [SEP] ... [SEP]`, so the two sides share five tokens.
    // A one-token query is shorter than its half and survives whole; the
    // document pays the whole difference.
    let row = encode_pair_row(
        &tokenizer,
        "hello",
        "a b c d e f a b c d",
        opts(usize::MAX, true),
    )
    .expect("pair encoding succeeds");
    assert_eq!(row.ids.len(), 8, "the row must fit the configured limit");
    assert_eq!(row.ids[0], CLS, "the query keeps its opening special token");
    assert_eq!(row.ids[1], 3, "the whole one-token query survives");
    assert_eq!(row.ids[2], SEP, "the query keeps its separator");
    assert_eq!(
        &row.ids[3..7],
        &[5, 6, 7, 8],
        "the document is right-truncated"
    );
    assert_eq!(*row.ids.last().expect("non-empty"), SEP);
    assert_eq!(
        row.type_ids.as_deref(),
        Some([0u32, 0, 0, 1, 1, 1, 1, 1].as_slice()),
        "the surviving document half keeps segment id 1"
    );

    // A query that is longer than its half pays too, and always from the
    // right: the tokens it keeps stay a prefix of the original query.
    let long_query = encode_pair_row(
        &tokenizer,
        "hello world a b c",
        "d e f a b c d e f a",
        opts(usize::MAX, true),
    )
    .expect("pair encoding succeeds");
    assert_eq!(long_query.ids.len(), 8);
    assert_eq!(long_query.ids[0], CLS);
    let query_half: Vec<u32> = long_query
        .ids
        .iter()
        .skip(1)
        .take_while(|&&id| id != SEP)
        .copied()
        .collect();
    assert!(
        [3u32, 4, 5, 6, 7].starts_with(&query_half),
        "the surviving query must be a prefix of the original, got {query_half:?}"
    );
    assert!(
        !query_half.is_empty(),
        "longest-first never strips the query to nothing while the document is longer"
    );
    assert_eq!(*long_query.ids.last().expect("non-empty"), SEP);

    // A pair that already fits is untouched.
    let short = encode_pair_row(&tokenizer, "hello", "world", opts(usize::MAX, true))
        .expect("pair encoding succeeds");
    assert_eq!(short.ids, vec![CLS, 3, SEP, 4, SEP]);
}

#[test]
fn truncation_is_installed_on_the_tokenizer() {
    let tokenizer = with_longest_first_truncation(bert_like_tokenizer(false), 8)
        .expect("truncation configures");
    let params = tokenizer
        .hf_tokenizer()
        .expect("huggingface tokenizer")
        .get_truncation()
        .expect("truncation is configured");
    assert_eq!(params.max_length, 8);
    assert_eq!(params.strategy, TruncationStrategy::LongestFirst);
    assert_eq!(
        params.direction,
        tokenizers::TruncationDirection::Right,
        "both sides are truncated from the right, keeping their opening tokens"
    );
}

#[test]
fn longest_first_keep_matches_the_tokenizer_fixed_point() {
    // The Qwen3-VL path reimplements the split over two token lists because
    // its texts are truncated before the template renders them. It must agree
    // with the strategy the tokenizer applies for the cross-encoder.
    use super::super::qwen3_vl_generative::longest_first_keep;
    assert_eq!(
        longest_first_keep(3, 10, 8),
        (3, 5),
        "the short side survives"
    );
    assert_eq!(longest_first_keep(10, 3, 8), (5, 3));
    assert_eq!(longest_first_keep(6, 6, 8), (4, 4));
    assert_eq!(longest_first_keep(2, 2, 8), (2, 2), "no truncation needed");
    assert_eq!(longest_first_keep(9, 9, 9), (4, 5));
}
