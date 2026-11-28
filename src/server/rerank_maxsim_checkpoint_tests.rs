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

//! Real-checkpoint gates for MaxSim reranking (#1426): the provider startup
//! installs, over the real embedding worker, on LFM2.5-ColBERT (text) and the
//! merged colSmol-256M (page images). Each soft-skips when its checkpoint is
//! absent.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use super::resolve_late_interaction_reranker;
use crate::embeddings::{EmbedOptions, EmbeddingLoadOptions, ImageInput, maxsim};
use crate::models::embedding_test_support::{local_checkpoint, mlx_test_guard};
use crate::rerank::{RerankItem, RerankerKind};
use crate::server::embedding_model::EmbeddingModelProvider;
use crate::server::embedding_worker::EmbeddingWorkerProvider;
use crate::server::rerank_model::RerankModelProvider;

const LFM2_QUERY: &str = "[Q] how do solar panels generate electricity";
const LFM2_RELEVANT: &str = "[D] Photovoltaic cells convert sunlight into electricity.";
const LFM2_UNRELATED: &str = "[D] The recipe calls for two cups of flour and one egg.";

/// PR #1414's query.
const COLSMOL_QUERY: &str = "What was the total revenue in 2023?";

/// Batch width `mlxcel embed` uses by default.
const CLI_BATCH_SIZE: usize = 16;

/// Absolute tolerance between the rerank path (the query and the documents in
/// one batch, in a different order) and the CLI's single padded batch, which
/// agree only to bf16 batched-prefill drift.
const CLI_PARITY_TOLERANCE: f32 = 5e-3;

fn load_worker(dir: &Path, id: &str) -> Arc<dyn EmbeddingModelProvider> {
    Arc::new(
        EmbeddingWorkerProvider::load(
            dir,
            id.to_string(),
            CLI_BATCH_SIZE,
            8,
            Duration::from_secs(600),
            EmbeddingLoadOptions::default(),
        )
        .unwrap_or_else(|err| panic!("{} loads: {err:#}", dir.display())),
    )
}

fn install(embedding: &Arc<dyn EmbeddingModelProvider>) -> Arc<dyn RerankModelProvider> {
    let provider = resolve_late_interaction_reranker(None, false, Some(embedding), None)
        .expect("a multi-vector embedder gets the MaxSim provider");
    assert_eq!(provider.kind(), RerankerKind::LateInteraction);
    provider
}

fn shutdown(embedding: &Arc<dyn EmbeddingModelProvider>) {
    embedding
        .shutdown_and_wait(Duration::from_secs(60))
        .expect("embedding worker exits");
}

fn lfm2_colbert_gate(repo: &str) {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(repo) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let embedding = load_worker(&dir, "lfm2.5-colbert");
    assert!(embedding.multi_vector());
    let provider = install(&embedding);

    let scored = provider
        .rerank(
            RerankItem::text(LFM2_QUERY),
            vec![
                RerankItem::text(LFM2_UNRELATED),
                RerankItem::text(LFM2_RELEVANT),
            ],
            None,
        )
        .expect("MaxSim rerank scores");
    let (unrelated, relevant) = (scored.scores[0], scored.scores[1]);
    assert!(
        relevant > unrelated,
        "{repo}: the unrelated document ranked first ({relevant} vs {unrelated})"
    );

    // `mlxcel embed -p q -p relevant -p unrelated` embeds the three texts in
    // one batch and prints MaxSim of the same rows.
    let texts = [LFM2_QUERY, LFM2_RELEVANT, LFM2_UNRELATED].map(str::to_string);
    let reply = embedding
        .embed_texts(texts.to_vec(), EmbedOptions::default())
        .expect("batched embed");
    let rows = |i: usize| -> Vec<&[f32]> { reply.vectors[i].rows().collect() };
    let cli_relevant = maxsim(&rows(0), &rows(1));
    let cli_unrelated = maxsim(&rows(0), &rows(2));
    assert!(
        (relevant - cli_relevant).abs() <= CLI_PARITY_TOLERANCE
            && (unrelated - cli_unrelated).abs() <= CLI_PARITY_TOLERANCE,
        "{repo}: rerank {relevant}/{unrelated} vs mlxcel embed {cli_relevant}/{cli_unrelated}"
    );
    assert_eq!(scored.prompt_tokens, reply.prompt_tokens);
    eprintln!(
        "GATE {repo} rerank relevant={relevant:.4} unrelated={unrelated:.4} \
         embed relevant={cli_relevant:.4} unrelated={cli_unrelated:.4} \
         prompt_tokens={}",
        scored.prompt_tokens
    );
    shutdown(&embedding);
}

#[test]
fn lfm2_colbert_bf16_maxsim_rerank_matches_embed() {
    lfm2_colbert_gate("mlx-community/LFM2.5-ColBERT-350M-bf16");
}

#[test]
fn lfm2_colbert_hf_maxsim_rerank_matches_embed() {
    lfm2_colbert_gate("LiquidAI/LFM2.5-ColBERT-350M");
}

/// The merged colSmol-256M directory
/// `tests/fixtures/generate_colsmol_rerank_fixture.py` writes, with its two
/// rendered pages.
fn colsmol_merged_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/colsmol-256m-merged");
    let ready = dir.join("config.json").is_file()
        && dir.join("rerank_pages/matching.png").is_file()
        && dir.join("rerank_pages/unrelated.png").is_file();
    if !ready {
        eprintln!(
            "skipping real-checkpoint gate: {} not present (see \
             tests/fixtures/generate_colsmol_rerank_fixture.py)",
            dir.display()
        );
    }
    ready.then_some(dir)
}

fn page(dir: &Path, name: &str) -> RerankItem {
    let path = dir.join("rerank_pages").join(name);
    let image = image::open(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    RerankItem::image(ImageInput { image })
}

#[test]
fn colsmol_maxsim_rerank_ranks_the_matching_page_first() {
    let _guard = mlx_test_guard();
    let Some(dir) = colsmol_merged_dir() else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let embedding = load_worker(&dir, "colsmol-256m-merged");
    assert!(embedding.multi_vector() && embedding.supports_images());
    let provider = install(&embedding);
    assert!(provider.supports_images());

    let scored = provider
        .rerank(
            RerankItem::text(COLSMOL_QUERY),
            vec![page(&dir, "unrelated.png"), page(&dir, "matching.png")],
            None,
        )
        .expect("MaxSim rerank scores");
    let (unrelated, matching) = (scored.scores[0], scored.scores[1]);
    // PR #1414 measured 18.7396 against 8.6879 on pages it did not commit;
    // these rendered stand-ins are held to the ordering and its 10 percent
    // relevance-margin gate.
    let margin = (matching - unrelated) / matching;
    assert!(
        matching > unrelated && margin >= 0.10,
        "colSmol ranked {matching} vs {unrelated} (margin {margin})"
    );

    // `mlxcel embed` embeds each image on its own, so this path is the CLI's.
    let query = embedding
        .embed_texts(vec![COLSMOL_QUERY.to_string()], EmbedOptions::default())
        .expect("query embeds");
    let matching_page = page(&dir, "matching.png").image.expect("image");
    let doc = embedding
        .embed_image(matching_page, EmbedOptions::default())
        .expect("page embeds");
    let q_rows: Vec<&[f32]> = query.vectors[0].rows().collect();
    let d_rows: Vec<&[f32]> = doc.vectors[0].rows().collect();
    let cli_matching = maxsim(&q_rows, &d_rows);
    assert!(
        (matching - cli_matching).abs() <= 1e-3,
        "rerank {matching} vs embed {cli_matching}"
    );
    eprintln!(
        "GATE colsmol-256m-merged rerank matching={matching:.4} unrelated={unrelated:.4} \
         margin={:.1}% embed matching={cli_matching:.4} prompt_tokens={}",
        margin * 100.0,
        scored.prompt_tokens
    );
    shutdown(&embedding);
}

/// Scores the CLI path (`mlxcel rerank -m <dir>`, #3300) and the server
/// provider give the same request; both embed through the same engine with
/// the same batch width and call split, so they agree to float noise.
const CLI_SERVER_TOLERANCE: f32 = 1e-5;

/// Score `documents` against `query` through the server provider and through
/// `load_reranker_or_late_interaction`, and assert they agree.
fn assert_cli_matches_server(dir: &Path, id: &str, query: RerankItem, documents: Vec<RerankItem>) {
    let embedding = load_worker(dir, id);
    let provider = install(&embedding);
    let server = provider
        .rerank(query.clone(), documents.clone(), None)
        .expect("server MaxSim rerank scores");
    shutdown(&embedding);

    let loaded = crate::rerank::load_reranker_or_late_interaction(
        dir,
        crate::rerank::RerankLoadOptions::default(),
    )
    .unwrap_or_else(|err| panic!("{id}: the CLI loads the embedder: {err:#}"));
    assert_eq!(loaded.kind, RerankerKind::LateInteraction);
    assert_eq!(loaded.reranker.batch_size(), CLI_BATCH_SIZE);
    let cli = loaded
        .reranker
        .score(&query, &documents, None)
        .expect("CLI MaxSim rerank scores");

    assert_eq!(cli.scores.len(), server.scores.len());
    for (index, (c, s)) in cli.scores.iter().zip(&server.scores).enumerate() {
        assert!(
            (c - s).abs() <= CLI_SERVER_TOLERANCE,
            "{id}: document {index} scores {c} on the CLI and {s} on /v1/rerank"
        );
    }
    assert_eq!(cli.prompt_tokens, server.prompt_tokens);
    eprintln!(
        "GATE {id} cli={:?} server={:?} prompt_tokens={}",
        cli.scores, server.scores, cli.prompt_tokens
    );
}

#[test]
fn lfm2_colbert_cli_rerank_matches_the_server_provider() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint("LiquidAI/LFM2.5-ColBERT-350M") else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    assert_cli_matches_server(
        &dir,
        "lfm2.5-colbert",
        RerankItem::text(LFM2_QUERY),
        vec![
            RerankItem::text(LFM2_UNRELATED),
            RerankItem::text(LFM2_RELEVANT),
        ],
    );
}

#[test]
fn colsmol_cli_rerank_matches_the_server_provider() {
    let _guard = mlx_test_guard();
    let Some(dir) = colsmol_merged_dir() else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    assert_cli_matches_server(
        &dir,
        "colsmol-256m-merged",
        RerankItem::text(COLSMOL_QUERY),
        vec![
            page(&dir, "unrelated.png"),
            RerankItem::text("Total revenue in 2023 was 4.2 billion dollars."),
            page(&dir, "matching.png"),
        ],
    );
}
