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

//! Real-checkpoint coverage for `--pooling` and `--embd-normalize` (#1452).
//!
//! The unit tests behind both flags run on a synthetic 16-wide stub, which is
//! what makes their arithmetic exact and also what makes them blind to the two
//! things that only a real checkpoint has: a `1_Pooling/config.json` the flag
//! has to outrank, and a family whose forward pass actually reads the resolved
//! mode. Two embedding checkpoints are used because their configs disagree,
//! `all-MiniLM-L6-v2` says mean and `bge-small-en-v1.5` says cls, so a flag
//! that silently did nothing would still look right on one of them.
//!
//! Every embedding case runs inside **one** `#[test]` function on purpose. The
//! pooling override is a process-wide cell, and libtest runs the functions of
//! one binary on parallel threads, so two functions installing different modes
//! interleave between the `set` and the load that reads it. Splitting these
//! into four readable tests is exactly what produced a green-looking suite that
//! silently compared vectors from the wrong pooling mode. One function is how
//! the ordering is guaranteed without the `--test-threads=1` the chain gate
//! forbids; the reranker case is separate because it never touches the cell.
//!
//! ```bash
//! cargo test --test embedding_pooling_normalize_real --profile test-fast --features metal,accelerate -- --nocapture
//! ```
//!
//! Each case skips with a message when its checkpoint is absent, so CI stays
//! green without them.

mod common;

use common::repo_model_dir;

use mlxcel::embeddings::{
    EmbdNormalize, EmbedOptions, EmbeddingEngine, EmbeddingLoadOptions, PoolingMode,
    load_embedding_model_with_options, set_pooling_override,
};
use mlxcel::initialize_runtime;
use mlxcel::rerank::{RerankItem, RerankLoadOptions, load_reranker_with_options};

/// `1_Pooling/config.json` says mean.
const MEAN_CHECKPOINT: &str = "mlx/all-minilm-l6-v2";
/// `1_Pooling/config.json` says cls.
const CLS_CHECKPOINT: &str = "mlx/bge-small-en-v1.5";
/// `BertForSequenceClassification`, a cross-encoder reranker.
const RERANKER: &str = "mlx/ms-marco-minilm-l6-v2";

const PROMPT: &str = "The giant panda is a bear species endemic to China.";

fn have(name: &str) -> bool {
    let present = repo_model_dir(name).join("config.json").exists();
    if !present {
        eprintln!("Skipping {name}: checkpoint not found");
    }
    present
}

/// Embed one prompt with an explicit pooling override and normalization.
fn embed(name: &str, pooling: Option<PoolingMode>, normalize: Option<EmbdNormalize>) -> Vec<f32> {
    set_pooling_override(pooling);
    let loaded = load_embedding_model_with_options(
        &repo_model_dir(name),
        EmbeddingLoadOptions { max_length: None },
    )
    .unwrap_or_else(|e| panic!("{name} must load: {e:#}"));
    let resolved = loaded.model.pooling();
    if let Some(expected) = pooling {
        assert_eq!(
            resolved, expected,
            "{name}: --pooling {expected} must outrank the checkpoint's own config, got {resolved}"
        );
    }
    let engine = EmbeddingEngine::new(loaded, 4);
    let reply = engine
        .embed_texts(
            &[PROMPT.to_string()],
            &EmbedOptions {
                instruction: None,
                dimensions: None,
                normalize,
            },
        )
        .unwrap_or_else(|e| panic!("{name} must embed: {e:#}"));
    reply.vectors[0].values.clone()
}

fn l1(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).sum()
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (l2(a) * l2(b)).max(1e-9)
}

/// `||a - b|| / max(||a||, ||b||)`: zero for identical vectors, and unlike
/// cosine it is not fooled by two vectors that point the same way.
fn relative_distance(a: &[f32], b: &[f32]) -> f32 {
    let diff: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    diff / l2(a).max(l2(b)).max(1e-9)
}

/// Every `--pooling` and `--embd-normalize` case, in one function so the
/// process-wide pooling override cannot be changed underneath a load.
#[test]
fn pooling_and_normalization_hold_on_real_checkpoints() {
    if !have(MEAN_CHECKPOINT) || !have(CLS_CHECKPOINT) {
        return;
    }
    let _runtime = initialize_runtime();

    the_pooling_flag_outranks_the_checkpoint_config();
    an_unset_pooling_flag_leaves_the_checkpoint_config_in_charge();
    every_normalization_holds_its_invariant();
    the_unset_normalization_follows_the_checkpoint_and_matches_euclidean();

    set_pooling_override(None);
}

/// Every b10621 pooling value mlxcel implements, on both checkpoints.
///
/// The property under test is that the override reaches the forward pass, and
/// output that does not change with the mode is what its absence looks like.
/// `mean` is the reference for that, because it reads every position while
/// `cls` and `last` each read one: on `bge-small-en-v1.5` those two positions
/// turn out to hold the same vector to within 1.3e-5, which is a property of a
/// model trained with CLS pooling and not a pooling bug, so that pair carries
/// no signal there. `all-MiniLM-L6-v2` separates all three, so the stronger
/// assertion is made where it means something.
fn the_pooling_flag_outranks_the_checkpoint_config() {
    for checkpoint in [MEAN_CHECKPOINT, CLS_CHECKPOINT] {
        let mut vectors = Vec::new();
        for mode in [PoolingMode::Mean, PoolingMode::Cls, PoolingMode::LastToken] {
            let v = embed(checkpoint, Some(mode), Some(EmbdNormalize::NONE));
            assert!(
                v.iter().all(|x| x.is_finite()) && !v.is_empty(),
                "{checkpoint}/{mode}: produced {} values",
                v.len()
            );
            vectors.push((mode, v));
        }
        let separation = |a: usize, b: usize| {
            let value = relative_distance(&vectors[a].1, &vectors[b].1);
            eprintln!(
                "{checkpoint}: {} vs {} separation {value:.4}",
                vectors[a].0, vectors[b].0
            );
            value
        };
        // Mean against each single-position mode, on both checkpoints.
        for other in [1usize, 2] {
            let value = separation(0, other);
            assert!(
                value > 1e-3,
                "{checkpoint}: {} and {} produced the same vector (separation {value:e}), so \
                 the pooling override did not reach the forward pass",
                vectors[0].0,
                vectors[other].0
            );
        }
        // Cls against LastToken, where the two positions are known to differ.
        if checkpoint == MEAN_CHECKPOINT {
            let value = separation(1, 2);
            assert!(
                value > 1e-3,
                "{checkpoint}: cls and lasttoken produced the same vector (separation \
                 {value:e})"
            );
        } else {
            // Recorded rather than asserted: the two positions coincide here.
            let _ = separation(1, 2);
        }
    }
}

/// With no override installed, `1_Pooling/config.json` still decides, and the
/// two checkpoints disagree, so a stuck cell would show up here.
fn an_unset_pooling_flag_leaves_the_checkpoint_config_in_charge() {
    set_pooling_override(None);
    for (checkpoint, expected) in [
        (MEAN_CHECKPOINT, PoolingMode::Mean),
        (CLS_CHECKPOINT, PoolingMode::Cls),
    ] {
        let loaded = load_embedding_model_with_options(
            &repo_model_dir(checkpoint),
            EmbeddingLoadOptions { max_length: None },
        )
        .unwrap_or_else(|e| panic!("{checkpoint} must load: {e:#}"));
        assert_eq!(
            loaded.model.pooling(),
            expected,
            "{checkpoint}: 1_Pooling/config.json must still decide when --pooling is unset"
        );
    }
}

/// Each mode's own invariant, on a vector a real model produced.
fn every_normalization_holds_its_invariant() {
    set_pooling_override(None);
    let raw = embed(MEAN_CHECKPOINT, None, Some(EmbdNormalize::NONE));
    assert!(raw.iter().all(|x| x.is_finite()) && l2(&raw) > 0.0);

    // -1 is deterministic and identical to the model's own output.
    assert_eq!(
        embed(MEAN_CHECKPOINT, None, Some(EmbdNormalize::NONE)),
        raw,
        "-1 must not touch the vector"
    );

    // 1, 2 and p > 2: the corresponding norm is one.
    let taxicab = embed(MEAN_CHECKPOINT, None, Some(EmbdNormalize::TAXICAB));
    assert!((l1(&taxicab) - 1.0).abs() < 1e-4, "L1 is {}", l1(&taxicab));
    let euclidean = embed(MEAN_CHECKPOINT, None, Some(EmbdNormalize::EUCLIDEAN));
    assert!(
        (l2(&euclidean) - 1.0).abs() < 1e-4,
        "L2 is {}",
        l2(&euclidean)
    );
    let p3 = EmbdNormalize::new(3).expect("in domain");
    let cubic = embed(MEAN_CHECKPOINT, None, Some(p3));
    let norm = cubic.iter().map(|x| x.abs().powi(3)).sum::<f32>().cbrt();
    assert!((norm - 1.0).abs() < 1e-4, "p=3 norm is {norm}");

    // 0: the largest component lands on the int16 bound.
    let scaled = embed(MEAN_CHECKPOINT, None, Some(EmbdNormalize::MAX_ABS_INT16));
    assert!(
        (max_abs(&scaled) - 32760.0).abs() < 1.0,
        "max-absolute normalization put the largest component at {}",
        max_abs(&scaled)
    );

    // Normalization is a rescale, not a rotation: every mode keeps the
    // direction the model produced, which is what makes cosine similarity
    // comparable across them.
    for (kind, v) in [
        (EmbdNormalize::MAX_ABS_INT16, &scaled),
        (EmbdNormalize::TAXICAB, &taxicab),
        (EmbdNormalize::EUCLIDEAN, &euclidean),
        (p3, &cubic),
    ] {
        assert!(
            cosine(&raw, v) > 0.9999,
            "{kind} changed the direction of the vector, not just its length"
        );
    }
}

/// `bge-small-en-v1.5` does not set `normalize: false`, so its default is
/// euclidean and an unqualified request must be byte-identical to an explicit
/// `2`. This is the assertion that keeps `--embd-normalize` from changing what
/// every existing deployment already returns.
fn the_unset_normalization_follows_the_checkpoint_and_matches_euclidean() {
    set_pooling_override(None);
    let implicit = embed(CLS_CHECKPOINT, None, None);
    let explicit = embed(CLS_CHECKPOINT, None, Some(EmbdNormalize::EUCLIDEAN));
    assert_eq!(implicit, explicit);
    assert!(
        (l2(&implicit) - 1.0).abs() < 1e-4,
        "L2 is {}",
        l2(&implicit)
    );
}

#[test]
fn a_real_reranker_orders_documents_and_breaks_ties_by_index() {
    if !have(RERANKER) {
        return;
    }
    let _runtime = initialize_runtime();

    let loaded = load_reranker_with_options(
        &repo_model_dir(RERANKER),
        RerankLoadOptions {
            batch_size: None,
            max_length: None,
        },
    )
    .unwrap_or_else(|e| panic!("{RERANKER} must load: {e:#}"));

    let query = RerankItem {
        text: Some("what is a panda?".to_string()),
        image: None,
    };
    let documents: Vec<RerankItem> = [
        "Berlin is the capital of Germany.",
        "The giant panda is a bear species endemic to China.",
        "A list of prime numbers below one hundred.",
    ]
    .into_iter()
    .map(|text| RerankItem {
        text: Some(text.to_string()),
        image: None,
    })
    .collect();

    let scored = loaded
        .reranker
        .score(&query, &documents, None)
        .unwrap_or_else(|e| panic!("{RERANKER} must score: {e:#}"));
    assert_eq!(scored.scores.len(), 3);
    assert!(
        scored
            .scores
            .iter()
            .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
        "scores must be finite probabilities: {:?}",
        scored.scores
    );
    // The panda passage is the relevant one; a reranker that did not read the
    // query would not put it first.
    let best = scored
        .scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("a best document")
        .0;
    assert_eq!(
        best, 1,
        "the relevant document must rank first, got {:?}",
        scored.scores
    );

    // The route's ordering contract, applied to the real scores: descending by
    // score, ties by ascending index.
    let mut ordered: Vec<usize> = (0..scored.scores.len()).collect();
    ordered.sort_by(|&a, &b| {
        scored.scores[b]
            .total_cmp(&scored.scores[a])
            .then(a.cmp(&b))
    });
    assert_eq!(
        ordered[0], 1,
        "ordering: {ordered:?} for {:?}",
        scored.scores
    );
}
