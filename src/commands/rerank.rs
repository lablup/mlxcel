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

//! `mlxcel rerank`: offline query/document scoring through the same code
//! `/v1/rerank` uses, for validating a reranker port without a server.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use serde_json::json;

use mlxcel::downloader::resolve_model_source_with_override;
use mlxcel::initialize_runtime;
use mlxcel::rerank::{ImageInput, RerankItem, RerankLoadOptions, load_reranker_with_options};

/// Arguments for `mlxcel rerank`.
#[derive(Args, Debug)]
#[command(next_help_heading = "Rerank Options")]
pub(crate) struct RerankArgs {
    /// Reranker checkpoint: a local directory or a HuggingFace `owner/name`
    /// repo-id (resolved and auto-downloaded like `mlxcel generate -m`).
    #[arg(short, long, value_name = "MODEL_OR_REPO_ID")]
    pub(crate) model: PathBuf,

    /// The query every document is scored against.
    #[arg(short = 'q', long = "query", value_name = "TEXT")]
    pub(crate) query: Option<String>,

    /// Image file to attach to the query (multimodal rerankers only).
    #[arg(long = "query-image", value_name = "FILE")]
    pub(crate) query_image: Option<PathBuf>,

    /// Text document to score. Repeat for several.
    #[arg(short = 'd', long = "document", value_name = "TEXT")]
    pub(crate) documents: Vec<String>,

    /// Image document to score (multimodal rerankers only). Repeat for
    /// several; image documents follow the text ones in the result order.
    #[arg(long = "image", value_name = "FILE")]
    pub(crate) images: Vec<PathBuf>,

    /// Task description for the generative rerankers.
    #[arg(long, value_name = "TEXT")]
    pub(crate) instruction: Option<String>,

    /// Print only the `top_n` highest-scoring documents in the ranking.
    #[arg(long = "top-n", value_name = "N")]
    pub(crate) top_n: Option<usize>,

    /// Print one JSON object instead of the text layout.
    #[arg(long)]
    pub(crate) json: bool,

    /// Token cap per scored pair (default: derived from the checkpoint).
    #[arg(long = "max-length", value_name = "N")]
    pub(crate) max_length: Option<usize>,

    /// Query/document pairs per forward pass (default: the reranker kind's).
    #[arg(long = "batch-size", value_name = "N")]
    pub(crate) batch_size: Option<usize>,

    /// Model-store root used to resolve a repo-id (defaults to the global store).
    #[arg(long = "models-dir", env = "MLXCEL_MODELS_DIR", value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,
}

/// Load and decode one image file.
fn load_image(path: &std::path::Path) -> Result<ImageInput> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let image = image::load_from_memory(&bytes)
        .with_context(|| format!("failed to decode {}", path.display()))?;
    Ok(ImageInput { image })
}

/// Short label for one document in the printed layout.
fn label(item: &RerankItem, fallback: &str) -> String {
    match item
        .text
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        Some(text) if text.chars().count() > 60 => {
            let head: String = text.chars().take(57).collect();
            format!("{head}...")
        }
        Some(text) => text.to_string(),
        None => fallback.to_string(),
    }
}

pub(crate) fn run_rerank(args: RerankArgs) -> Result<()> {
    if args.documents.is_empty() && args.images.is_empty() {
        bail!("nothing to rerank: pass at least one -d/--document or --image");
    }
    if let Some(index) = args.documents.iter().position(|d| d.trim().is_empty()) {
        bail!("document {index} is empty");
    }
    let query_text = args.query.as_deref().map(str::trim).unwrap_or("");
    if query_text.is_empty() && args.query_image.is_none() {
        bail!("nothing to score against: pass -q/--query or --query-image");
    }
    for image in args.images.iter().chain(args.query_image.iter()) {
        if !image.is_file() {
            bail!("image file does not exist: {}", image.display());
        }
    }

    let model_dir =
        resolve_model_source_with_override(&args.model, args.models_dir.as_deref(), None)?;

    // Initialize the MLX runtime (selects GPU/CPU) before any forward pass.
    let _runtime = initialize_runtime();

    let loaded = load_reranker_with_options(
        &model_dir,
        RerankLoadOptions {
            batch_size: args.batch_size,
            max_length: args.max_length,
        },
    )
    .with_context(|| format!("failed to load reranker {}", model_dir.display()))?;

    let mut query = RerankItem {
        text: (!query_text.is_empty()).then(|| query_text.to_string()),
        image: None,
    };
    if let Some(path) = args.query_image.as_deref() {
        query.image = Some(load_image(path)?);
    }
    let mut documents: Vec<RerankItem> = args
        .documents
        .iter()
        .map(|text| RerankItem::text(text.clone()))
        .collect();
    let mut image_labels: Vec<String> = Vec::with_capacity(args.images.len());
    for path in &args.images {
        documents.push(RerankItem::image(load_image(path)?));
        image_labels.push(path.display().to_string());
    }

    if !loaded.reranker.supports_images()
        && (query.image.is_some() || documents.iter().any(RerankItem::has_image))
    {
        bail!(
            "the {} reranker is text-only; drop --image / --query-image",
            loaded.kind.as_str()
        );
    }
    if args.instruction.is_some() && !loaded.kind.accepts_instruction() {
        bail!(
            "--instruction is only supported by the generative rerankers; this checkpoint is a \
             sequence classifier and scores the pair directly"
        );
    }

    let scored = loaded
        .reranker
        .score(&query, &documents, args.instruction.as_deref())?;

    let labels: Vec<String> = documents
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let fallback = image_labels
                .get(index.saturating_sub(args.documents.len()))
                .cloned()
                .unwrap_or_else(|| format!("document {index}"));
            label(item, &fallback)
        })
        .collect();

    let mut ranking: Vec<usize> = (0..scored.scores.len()).collect();
    ranking.sort_by(|&a, &b| {
        scored.scores[b]
            .partial_cmp(&scored.scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    if let Some(n) = args.top_n {
        ranking.truncate(n);
    }

    if args.json {
        let out = json!({
            "model": model_dir.display().to_string(),
            "kind": loaded.kind.as_str(),
            "model_type": loaded.model_type,
            "max_length": loaded.reranker.max_length(),
            "batch_size": loaded.reranker.batch_size(),
            "query": query_text,
            "documents": labels,
            "scores": scored.scores,
            "ranking": ranking,
            "prompt_tokens": scored.prompt_tokens,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("scores (document order):");
    for (index, score) in scored.scores.iter().enumerate() {
        println!("  [{index}] {score:.6}  {}", labels[index]);
    }
    println!();
    println!("ranking (most relevant first):");
    for (rank, &index) in ranking.iter().enumerate() {
        println!(
            "  {}. [{index}] {:.6}  {}",
            rank + 1,
            scored.scores[index],
            labels[index]
        );
    }
    eprintln!(
        "{} document(s), {} prompt tokens, kind {}, max_length {}",
        documents.len(),
        scored.prompt_tokens,
        loaded.kind.as_str(),
        loaded.reranker.max_length()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_truncates_long_text_and_falls_back_for_images() {
        let long = "x".repeat(100);
        let truncated = label(&RerankItem::text(long), "unused");
        assert_eq!(truncated.chars().count(), 60);
        assert!(truncated.ends_with("..."));
        assert_eq!(label(&RerankItem::text("short"), "unused"), "short");
        assert_eq!(
            label(
                &RerankItem {
                    text: None,
                    image: None
                },
                "chart.png"
            ),
            "chart.png"
        );
    }
}
