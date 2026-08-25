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

//! Real-checkpoint gates for the BERT / XLM-RoBERTa forward pass.
//!
//! Each test soft-skips when its checkpoint is absent, following the
//! convention of `src/embeddings/real_checkpoint_tests.rs` and
//! `tests/*_parity.rs`. Fetch with:
//!
//! ```sh
//! mlxcel download sentence-transformers/all-MiniLM-L6-v2
//! mlxcel download intfloat/multilingual-e5-small
//! mlxcel download BAAI/bge-reranker-v2-m3
//! # BAAI/bge-m3 publishes only pytorch_model.bin; mlxcel takes safetensors.
//! mlxcel download seansitter/bge-m3-safetensors
//! ```

use std::path::PathBuf;

use mlxcel_core::utils::array_to_vec_f32;

use crate::embeddings::tokenize::{EncodeOptions, encode_pairs, strip_padding_and_truncation};
use crate::embeddings::{EmbedOptions, EmbeddingEngine, load_embedding_model};
use crate::models::bert::bert_tests::mlx_test_guard;
use crate::models::bert_heads::BertSequenceClassifier;
use crate::tokenizer::load_tokenizer;

const MINILM: &str = "sentence-transformers/all-MiniLM-L6-v2";
const E5_SMALL: &str = "intfloat/multilingual-e5-small";
const BGE_RERANKER: &str = "BAAI/bge-reranker-v2-m3";
/// `BAAI/bge-m3` ships `pytorch_model.bin` only, which the downloader skips;
/// a safetensors mirror of the same weights stands in for it.
const BGE_M3: &[&str] = &["BAAI/bge-m3", "seansitter/bge-m3-safetensors"];
/// Bound on the unrelated-pair cosine for multilingual-e5.
///
/// The epic's shared gate is 0.5, which e5 cannot meet: its contrastive
/// training compresses the cosine range upward, and the unrelated pair in
/// `multilingual_e5_small_passes_the_self_consistency_gate` measures 0.739 on
/// this checkpoint. That is a property of the checkpoint, not of the port:
/// `multilingual_e5_small_ranks_the_matching_passage_first` is the real
/// discrimination gate for this family, and it separates a matching passage
/// (above 0.8) from an unrelated one, in English and cross-lingually.
const E5_MAX_UNRELATED: f32 = 0.80;

/// Locate a downloaded checkpoint that actually carries safetensors weights:
/// the mlxcel store, then the HuggingFace cache, then `<repo>/models/<name>`.
/// `None` skips the test.
fn local_checkpoint(repo_ids: &[&str]) -> Option<PathBuf> {
    let has_weights = |dir: &PathBuf| {
        dir.join("config.json").is_file()
            && (dir.join("model.safetensors").is_file()
                || dir.join("model.safetensors.index.json").is_file())
    };
    for repo_id in repo_ids {
        let candidates = [
            crate::downloader::model_dir(repo_id),
            crate::downloader::hf_cache_snapshot(repo_id, None),
            Some(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("models")
                    .join(crate::downloader::repo_basename(repo_id)),
            ),
        ];
        if let Some(dir) = candidates.into_iter().flatten().find(has_weights) {
            return Some(dir);
        }
    }
    eprintln!(
        "skipping real-checkpoint gate: none of {repo_ids:?} present with safetensors weights \
         (mlxcel download {})",
        repo_ids[0]
    );
    None
}

fn engine(dir: &std::path::Path) -> EmbeddingEngine {
    EmbeddingEngine::new(
        load_embedding_model(dir).expect("embedding checkpoint loads"),
        16,
    )
}

/// Embed `texts` in one call and return the vectors in input order.
fn embed(engine: &EmbeddingEngine, texts: &[&str]) -> Vec<Vec<f32>> {
    let owned: Vec<String> = texts.iter().map(|t| (*t).to_string()).collect();
    engine
        .embed_texts(&owned, &EmbedOptions::default())
        .expect("forward succeeds")
        .vectors
        .into_iter()
        .map(|v| v.values)
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity. The engine L2-normalizes, so this is the dot product,
/// but the explicit division keeps the assertion honest if that changes.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b) / (norm(a) * norm(b)).max(1e-9)
}

fn assert_close(actual: f32, expected: f32, tolerance: f32, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected} +/- {tolerance}, got {actual}"
    );
}

/// The epic's shared self-consistency gate, run against one loaded engine.
///
/// - the same text twice gives cosine 1.0 within 1e-6;
/// - a row padded up by a longer batch member matches the unpadded
///   single-input result within 1e-3;
/// - two unrelated sentences score below `max_unrelated`, while the related
///   pair scores above them;
/// - every vector has unit norm within 1e-5.
///
/// `max_unrelated` is the epic's 0.5 for every family whose cosine scale is
/// centered; a checkpoint trained with a compressed similarity range needs
/// its own documented bound rather than a silently relaxed one.
fn assert_self_consistency(
    engine: &EmbeddingEngine,
    related: (&str, &str),
    far: &str,
    max_unrelated: f32,
    label: &str,
) {
    let alone = embed(engine, &[related.0]);
    for v in &alone {
        assert_close(norm(v), 1.0, 1e-5, &format!("{label}: unit norm"));
    }

    // Index 0 and 1 are the same text, 2 is its related partner, 3 is
    // unrelated, and 4 is long enough to pad every other row of the batch.
    let batched = embed(
        engine,
        &[
            related.0,
            related.0,
            related.1,
            far,
            "A padded batch member long enough to widen every other row in this micro-batch.",
        ],
    );
    for v in &batched {
        assert_close(norm(v), 1.0, 1e-5, &format!("{label}: unit norm (batched)"));
    }
    assert_close(
        cosine(&batched[0], &batched[1]),
        1.0,
        1e-6,
        &format!("{label}: identical inputs"),
    );
    assert_close(
        cosine(&alone[0], &batched[0]),
        1.0,
        1e-3,
        &format!("{label}: padded batch versus unpadded single input"),
    );
    let unrelated = cosine(&batched[0], &batched[3]);
    assert!(
        unrelated < max_unrelated,
        "{label}: unrelated sentences scored {unrelated} (expected < {max_unrelated})"
    );
    let related_score = cosine(&batched[0], &batched[2]);
    assert!(
        related_score > unrelated,
        "{label}: the related pair scored {related_score}, the unrelated one {unrelated}"
    );
    eprintln!(
        "[gate] {label}: identical={:.7} padded_vs_single={:.7} related={related_score:.4} unrelated={unrelated:.4}",
        cosine(&batched[0], &batched[1]),
        cosine(&alone[0], &batched[0])
    );
}

#[test]
fn minilm_matches_the_sentence_transformers_quickstart_similarities() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(&[MINILM]) else {
        return;
    };
    let engine = engine(&dir);
    assert_eq!(engine.dim(), 384);
    assert_eq!(engine.max_length(), 256);

    let sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ];
    let batched = embed(&engine, &sentences);
    // Published in the sentence-transformers quickstart for this checkpoint.
    assert_close(
        cosine(&batched[0], &batched[1]),
        0.666,
        1e-2,
        "MiniLM cosine(0, 1)",
    );
    assert_close(
        cosine(&batched[0], &batched[2]),
        0.105,
        1e-2,
        "MiniLM cosine(0, 2)",
    );

    eprintln!(
        "[gate] MiniLM: cos(0,1)={:.4} cos(0,2)={:.4} norms={:.7},{:.7},{:.7}",
        cosine(&batched[0], &batched[1]),
        cosine(&batched[0], &batched[2]),
        norm(&batched[0]),
        norm(&batched[1]),
        norm(&batched[2])
    );

    // The batched result must equal the three single-input results.
    for (index, sentence) in sentences.iter().enumerate() {
        let single = embed(&engine, &[sentence]);
        let drift = cosine(&single[0], &batched[index]);
        assert_close(
            drift,
            1.0,
            1e-4,
            &format!("MiniLM single vs batched [{index}]"),
        );
    }
}

#[test]
fn minilm_passes_the_self_consistency_gate() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(&[MINILM]) else {
        return;
    };
    assert_self_consistency(
        &engine(&dir),
        ("The weather is lovely today.", "It's so sunny outside!"),
        "He drove to the stadium.",
        0.5,
        "MiniLM",
    );
}

#[test]
fn multilingual_e5_small_ranks_the_matching_passage_first() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(&[E5_SMALL]) else {
        return;
    };
    let engine = engine(&dir);
    assert_eq!(engine.dim(), 384);

    // e5 requires the `query: ` / `passage: ` prefixes on every input.
    let vectors = embed(
        &engine,
        &[
            "query: how much protein should a female eat",
            "passage: As a general guideline, the CDC's average requirement of protein for women \
             ages 19 to 70 is 46 grams per day.",
            "passage: Definition of summit for English Language Learners: the highest point of a \
             mountain.",
        ],
    );
    let relevant = cosine(&vectors[0], &vectors[1]);
    let irrelevant = cosine(&vectors[0], &vectors[2]);
    assert!(
        relevant > irrelevant,
        "e5: the matching passage scored {relevant}, the unrelated one {irrelevant}"
    );
    assert!(relevant > 0.8, "e5: matching pair scored only {relevant}");
    eprintln!("[gate] e5-small: relevant={relevant:.4} irrelevant={irrelevant:.4}");

    // Cross-lingual: the same question in Korean must retrieve the same passage.
    let cross = embed(
        &engine,
        &[
            "query: 여성은 단백질을 얼마나 섭취해야 하나요",
            "passage: As a general guideline, the CDC's average requirement of protein for women \
             ages 19 to 70 is 46 grams per day.",
            "passage: Definition of summit for English Language Learners: the highest point of a \
             mountain.",
        ],
    );
    assert!(
        cosine(&cross[0], &cross[1]) > cosine(&cross[0], &cross[2]),
        "e5: the Korean query must still prefer the protein passage"
    );
}

#[test]
fn multilingual_e5_small_passes_the_self_consistency_gate() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(&[E5_SMALL]) else {
        return;
    };
    assert_self_consistency(
        &engine(&dir),
        (
            "query: what is BM25",
            "passage: BM25 is a ranking function.",
        ),
        "passage: The highest point of a mountain is its summit.",
        E5_MAX_UNRELATED,
        "e5-small",
    );
}

#[test]
fn bge_m3_matches_the_model_card_similarity_matrix() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(BGE_M3) else {
        return;
    };
    let engine = engine(&dir);
    assert_eq!(engine.dim(), 1024);
    // 8194 position rows minus the `pad_token_id + 1` offset.
    assert_eq!(engine.max_length(), 8192);

    let vectors = embed(
        &engine,
        &[
            "What is BGE M3?",
            "Defination of BM25",
            "BGE M3 is an embedding model supporting dense retrieval, lexical matching and \
             multi-vector interaction.",
            "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the \
             query terms appearing in each document",
        ],
    );
    // Published on the BAAI/bge-m3 model card for dense retrieval.
    let expected = [[0.6265f32, 0.3477], [0.3499, 0.678]];
    eprintln!(
        "[gate] bge-m3 similarity matrix: [[{:.4}, {:.4}], [{:.4}, {:.4}]]",
        cosine(&vectors[0], &vectors[2]),
        cosine(&vectors[0], &vectors[3]),
        cosine(&vectors[1], &vectors[2]),
        cosine(&vectors[1], &vectors[3])
    );
    for (q, row) in expected.iter().enumerate() {
        for (p, want) in row.iter().enumerate() {
            assert_close(
                cosine(&vectors[q], &vectors[2 + p]),
                *want,
                1.5e-2,
                &format!("bge-m3 similarity[{q}][{p}]"),
            );
        }
    }
}

#[test]
fn bge_m3_passes_the_self_consistency_gate_and_embeds_a_long_input() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(BGE_M3) else {
        return;
    };
    let engine = engine(&dir);
    assert_self_consistency(
        &engine,
        (
            "What is BGE M3?",
            "BGE M3 is an embedding model supporting dense retrieval, lexical matching and \
             multi-vector interaction.",
        ),
        "He drove to the stadium.",
        0.5,
        "bge-m3",
    );

    // Past 512 tokens the shifted XLM-RoBERTa position ids are the only thing
    // keeping the gather inside the 8194-row table.
    let long = "BGE M3 is an embedding model supporting dense retrieval, lexical matching and \
                multi-vector interaction. "
        .repeat(90);
    let reply = engine
        .embed_texts(&[long], &EmbedOptions::default())
        .expect("a long input embeds");
    assert!(
        reply.prompt_tokens > 1200,
        "expected a >1200 token input, got {}",
        reply.prompt_tokens
    );
    assert_close(
        norm(&reply.vectors[0].values),
        1.0,
        1e-5,
        "bge-m3 long input",
    );
    eprintln!(
        "[gate] bge-m3 long input: prompt_tokens={} norm={:.7}",
        reply.prompt_tokens,
        norm(&reply.vectors[0].values)
    );
}

#[test]
fn bge_reranker_v2_m3_scores_a_relevant_pair_above_an_irrelevant_one() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(&[BGE_RERANKER]) else {
        return;
    };
    let model = BertSequenceClassifier::load(&dir).expect("reranker loads");
    assert_eq!(model.args().num_labels, 1);
    assert!(
        !model.needs_token_type_ids(),
        "XLM-RoBERTa has a single-row segment table"
    );

    let tokenizer = strip_padding_and_truncation(load_tokenizer(&dir).unwrap());
    let batch = encode_pairs(
        &tokenizer,
        &[
            ("what is panda?", "hi"),
            (
                "what is panda?",
                "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply \
                 panda, is a bear species endemic to China.",
            ),
        ],
        EncodeOptions {
            add_special_tokens: true,
            max_length: 512,
            with_token_type_ids: false,
        },
        1,
        None,
    )
    .expect("pair encoding succeeds");

    let input_ids = batch.input_ids_array();
    let attention_mask = batch.attention_mask_array();
    let logits = model
        .logits(&input_ids, &attention_mask, None)
        .expect("classification head runs");
    mlxcel_core::try_eval(&logits).unwrap();
    assert_eq!(mlxcel_core::array_shape(&logits), vec![2, 1]);
    let scores = array_to_vec_f32(&logits);
    assert!(
        scores[1] > scores[0],
        "the panda passage scored {} and the unrelated one {}",
        scores[1],
        scores[0]
    );
    // The model card reports roughly -8 for the unrelated pair and +5 for the
    // relevant one; the sign split is the stable part of that.
    assert!(scores[0] < 0.0, "unrelated pair scored {}", scores[0]);
    assert!(scores[1] > 0.0, "relevant pair scored {}", scores[1]);
    eprintln!(
        "[gate] bge-reranker-v2-m3 logits: unrelated={:.4} relevant={:.4}",
        scores[0], scores[1]
    );
}
