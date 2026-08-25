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

//! Real-checkpoint gates for the ModernBERT port.
//!
//! Each test soft-skips when its checkpoint is absent, the same convention
//! `src/embeddings/real_checkpoint_tests.rs` and `tests/*_parity.rs` follow.
//! Fetch with:
//!
//! ```sh
//! mlxcel download nomic-ai/modernbert-embed-base
//! mlxcel download Alibaba-NLP/gte-reranker-modernbert-base
//! ```

use std::path::{Path, PathBuf};

use mlxcel_core::utils::array_to_vec_f32;

use crate::embeddings::limits::resolve_pad_token_id;
use crate::embeddings::tokenize::{EncodeOptions, encode_pairs, strip_padding_and_truncation};
use crate::embeddings::{
    DEFAULT_EMBEDDING_BATCH_SIZE, EmbedOptions, EmbeddingEngine, load_embedding_model,
};
use crate::models::modernbert_heads::ModernBertSequenceClassifier;
use crate::models::modernbert_tests::mlx_guard;
use crate::models::{ModelType, get_model_type};
use crate::tokenizer::load_tokenizer;

const EMBED: &str = "nomic-ai/modernbert-embed-base";
const RERANKER: &str = "Alibaba-NLP/gte-reranker-modernbert-base";

/// nomic's asymmetric prompt prefixes; both must be present verbatim.
const QUERY_PREFIX: &str = "search_query: ";
const DOCUMENT_PREFIX: &str = "search_document: ";

/// Locate a downloaded checkpoint: the mlxcel store, then the HuggingFace
/// cache, then `<repo>/models/<name>`. `None` skips the test.
fn local_checkpoint(repo_id: &str) -> Option<PathBuf> {
    let candidates = [
        crate::downloader::model_dir(repo_id),
        crate::downloader::hf_cache_snapshot(repo_id, None),
        Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join(crate::downloader::repo_basename(repo_id)),
        ),
    ];
    let found = candidates
        .into_iter()
        .flatten()
        .find(|dir| dir.join("config.json").is_file());
    if found.is_none() {
        eprintln!(
            "skipping real-checkpoint gate: {repo_id} not present (mlxcel download {repo_id})"
        );
    }
    found
}

/// Cosine similarity of two vectors the engine already L2-normalized, which
/// reduces to the dot product. Every caller asserts unit norm first.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn engine(dir: &Path) -> EmbeddingEngine {
    let loaded = load_embedding_model(dir).expect("ModernBERT checkpoint loads");
    assert_eq!(loaded.model_type, ModelType::ModernBert);
    EmbeddingEngine::new(loaded, DEFAULT_EMBEDDING_BATCH_SIZE)
}

fn embed_all(engine: &EmbeddingEngine, texts: &[&str]) -> Vec<Vec<f32>> {
    let owned: Vec<String> = texts.iter().map(|t| (*t).to_string()).collect();
    engine
        .embed_texts(&owned, &EmbedOptions::default())
        .expect("forward pass succeeds")
        .vectors
        .into_iter()
        .map(|v| v.values)
        .collect()
}

#[test]
fn modernbert_embed_base_detects_and_reports_its_limits() {
    let Some(dir) = local_checkpoint(EMBED) else {
        return;
    };
    let _mlx = mlx_guard();
    assert_eq!(get_model_type(&dir).unwrap(), ModelType::ModernBert);
    let engine = engine(&dir);
    assert_eq!(engine.dim(), 768);
    // sentence_bert_config max_seq_length 8192, tokenizer model_max_length
    // 8192 and the hard cap all agree; RoPE means no positional table caps it.
    assert_eq!(engine.max_length(), 8192);
    assert_eq!(engine.vocab_size(), 50368);
    assert!(!engine.multi_vector());
    assert!(!engine.supports_images());
}

#[test]
fn modernbert_embed_base_passes_the_self_consistency_gate() {
    let Some(dir) = local_checkpoint(EMBED) else {
        return;
    };
    let _mlx = mlx_guard();
    let engine = engine(&dir);
    let query = format!("{QUERY_PREFIX}What is TSNE?");
    let tsne = format!(
        "{DOCUMENT_PREFIX}t-SNE is a dimensionality reduction technique used for visualizing \
         high-dimensional data."
    );
    let eiffel =
        format!("{DOCUMENT_PREFIX}The Eiffel Tower is a wrought-iron lattice tower in Paris.");
    let vectors = embed_all(&engine, &[&query, &tsne, &eiffel, &query]);

    for (index, v) in vectors.iter().enumerate() {
        assert_eq!(v.len(), 768, "vector {index} width");
        assert!(v.iter().all(|x| x.is_finite()), "vector {index} is finite");
        assert!(
            (l2_norm(v) - 1.0).abs() <= 1e-5,
            "vector {index} is not unit norm: {}",
            l2_norm(v)
        );
    }

    // Identical inputs must land on exactly the same vector.
    let self_cosine = cosine(&vectors[0], &vectors[3]);
    eprintln!(
        "self-consistency: identical {self_cosine:.9}, relevant {:.6}, irrelevant {:.6}",
        cosine(&vectors[0], &vectors[1]),
        cosine(&vectors[0], &vectors[2])
    );
    assert!(
        (self_cosine - 1.0).abs() <= 1e-6,
        "identical texts scored {self_cosine}, expected 1.0 within 1e-6"
    );

    // The relevant document must win, and by a clear margin.
    let relevant = cosine(&vectors[0], &vectors[1]);
    let irrelevant = cosine(&vectors[0], &vectors[2]);
    assert!(
        relevant - irrelevant >= 0.15,
        "cosine(query, t-SNE) {relevant} must beat cosine(query, Eiffel) {irrelevant} by 0.15"
    );
    assert!(
        irrelevant < 0.5,
        "unrelated sentences scored {irrelevant}, expected below 0.5"
    );
}

#[test]
fn modernbert_embed_base_is_padding_invariant_across_batches() {
    let Some(dir) = local_checkpoint(EMBED) else {
        return;
    };
    let _mlx = mlx_guard();
    let engine = engine(&dir);
    let short = format!("{QUERY_PREFIX}What is TSNE?");
    let long = format!(
        "{DOCUMENT_PREFIX}t-SNE is a dimensionality reduction technique used for visualizing \
         high-dimensional data, and it is frequently contrasted with UMAP and with classical \
         principal component analysis in the visualization literature."
    );

    let solo = embed_all(&engine, &[&short]).remove(0);
    // The same text as the shorter member of a padded batch.
    let batched = embed_all(&engine, &[&long, &short]).remove(1);
    let drift = cosine(&solo, &batched);
    let worst = solo
        .iter()
        .zip(&batched)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("padding invariance: cosine {drift:.9}, worst component drift {worst:.9}");
    assert!(
        (drift - 1.0).abs() <= 1e-3,
        "padded batch drifted from the single-input result: cosine {drift}"
    );
    // Padding cannot change the math: padding keys are blocked in both the
    // global and the sliding-window mask, RoPE positions are unchanged, and
    // LayerNorm and the MLP are per-position. All that is left is f32
    // reduction-order noise from running the same rows at a different batch
    // shape through 22 layers, measured at 1.3e-7 across repeated runs, so a
    // bound three orders above that still catches any masking or RoPE bug.
    assert!(
        worst <= 1e-4,
        "worst per-component drift {worst} exceeds the numerical-noise bound"
    );
}

#[test]
fn modernbert_embed_base_handles_a_document_beyond_4096_tokens() {
    let Some(dir) = local_checkpoint(EMBED) else {
        return;
    };
    let _mlx = mlx_guard();
    let engine = engine(&dir);
    // Well past the 128-token local window, so the alternating local layers
    // are exercised over dozens of windows rather than one.
    let sentence = "t-SNE embeds high-dimensional points into two dimensions while preserving \
                    local neighbourhood structure. ";
    let long = format!("{DOCUMENT_PREFIX}{}", sentence.repeat(220));
    let short = format!("{QUERY_PREFIX}What is TSNE?");

    // Assert the premise rather than assuming it: editing the sentence above
    // must not silently shrink this into a short-sequence test that still
    // passes while no longer covering long inputs.
    let reply = engine
        .embed_texts(std::slice::from_ref(&long), &EmbedOptions::default())
        .expect("the long document embeds");
    assert!(
        reply.prompt_tokens > 4096,
        "this gate must exceed 4096 tokens, got {}",
        reply.prompt_tokens
    );
    assert!(
        reply.prompt_tokens <= engine.max_length(),
        "the document must fit under max_length {} without truncation, got {}",
        engine.max_length(),
        reply.prompt_tokens
    );
    let solo = reply.vectors[0].values.clone();
    assert!(solo.iter().all(|x| x.is_finite()));
    assert!((l2_norm(&solo) - 1.0).abs() <= 1e-5);

    let batched = embed_all(&engine, &[&long, &short]).remove(0);
    let drift = cosine(&solo, &batched);
    eprintln!(
        "long document ({} tokens): batched-vs-solo cosine {drift:.9}",
        reply.prompt_tokens
    );
    assert!(
        (drift - 1.0).abs() <= 1e-3,
        "a {}-token document drifted inside a batch: cosine {drift}",
        reply.prompt_tokens
    );
}

#[test]
fn gte_reranker_modernbert_produces_finite_logits() {
    let Some(dir) = local_checkpoint(RERANKER) else {
        return;
    };
    let _mlx = mlx_guard();
    // Detection must keep refusing to serve a reranker as an embedder; the
    // head is reached by directory instead (#1356 wires /v1/rerank).
    let err = get_model_type(&dir).expect_err("a reranker is not an embedding checkpoint");
    assert!(err.to_string().contains("Unsupported model type"), "{err}");

    let classifier = ModernBertSequenceClassifier::load(&dir).expect("the classifier head loads");
    assert_eq!(classifier.num_labels(), 1, "id2label declares one label");
    assert_eq!(classifier.encoder().hidden_size(), 768);

    let tokenizer = strip_padding_and_truncation(load_tokenizer(&dir).unwrap());
    let pad_id = resolve_pad_token_id(&dir, &tokenizer);
    assert_eq!(pad_id, 50283, "[PAD] is id 50283 in the ModernBERT vocab");
    // The `[CLS] query [SEP] document [SEP]` template the /v1/rerank path
    // (#1356) will use.
    let pairs: &[(&str, &str)] = &[
        (
            "what is the capital of France?",
            "Paris is the capital and most populous city of France.",
        ),
        (
            "what is the capital of France?",
            "A tomato is a fruit that is commonly prepared as a vegetable.",
        ),
    ];
    let batch = encode_pairs(
        &tokenizer,
        pairs,
        EncodeOptions {
            add_special_tokens: true,
            max_length: 512,
            with_token_type_ids: false,
        },
        pad_id,
        None,
    )
    .expect("pair tokenization succeeds");
    assert_eq!(batch.batch, 2);

    let ids = batch.input_ids_array();
    let mask = batch.attention_mask_array();
    let logits = classifier.logits(&ids, &mask).expect("head runs");
    assert_eq!(mlxcel_core::array_shape(&logits), vec![2, 1]);
    let values = array_to_vec_f32(&logits);
    eprintln!(
        "gte-reranker logits: relevant {}, irrelevant {}",
        values[0], values[1]
    );
    assert!(
        values.iter().all(|v| v.is_finite()),
        "reranker logits must be finite, got {values:?}"
    );
    // Ordering is #1356's acceptance gate, but a head wired to the wrong
    // tensors would not rank the relevant document first either.
    assert!(
        values[0] > values[1],
        "the relevant document scored {} and the irrelevant one {}",
        values[0],
        values[1]
    );
}
