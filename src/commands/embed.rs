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

//! `mlxcel embed`: offline embedding through the same engine `/v1/embeddings`
//! uses, for validating a family port without a server.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use serde_json::json;

use mlxcel::downloader::resolve_model_source_with_override;
use mlxcel::embeddings::{
    EmbedOptions, EmbeddingEngine, EmbeddingLoadOptions, EmbeddingVector, ImageInput,
    load_embedding_model_with_options,
};
use mlxcel::initialize_runtime;

/// Arguments for `mlxcel embed`.
#[derive(Args, Debug)]
#[command(next_help_heading = "Embed Options")]
pub(crate) struct EmbedArgs {
    /// Embedding checkpoint: a local directory or a HuggingFace `owner/name`
    /// repo-id (resolved and auto-downloaded like `mlxcel generate -m`).
    #[arg(short, long, value_name = "MODEL_OR_REPO_ID")]
    pub(crate) model: PathBuf,

    /// Text to embed. Repeat for several inputs.
    #[arg(short = 'p', long = "prompt", value_name = "TEXT")]
    pub(crate) prompts: Vec<String>,

    /// Image file to embed (vision-language embedders only). Repeat for several.
    #[arg(long = "image", value_name = "FILE")]
    pub(crate) images: Vec<PathBuf>,

    /// Instruction forwarded to the family's text formatting (query prefix).
    #[arg(long, value_name = "TEXT")]
    pub(crate) instruction: Option<String>,

    /// Keep only the first N components of every vector (re-normalized).
    #[arg(long, value_name = "N")]
    pub(crate) dimensions: Option<usize>,

    /// Print one JSON object instead of the text layout.
    #[arg(long)]
    pub(crate) json: bool,

    /// Token cap per input (default: derived from the checkpoint).
    #[arg(long = "max-length", value_name = "N")]
    pub(crate) max_length: Option<usize>,

    /// Inputs per forward pass.
    #[arg(long = "batch-size", default_value_t = 16, value_name = "N")]
    pub(crate) batch_size: usize,

    /// Model-store root used to resolve a repo-id (defaults to the global store).
    #[arg(long = "models-dir", env = "MLXCEL_MODELS_DIR", value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity for single vectors; for multi-vector outputs the
/// ColBERT MaxSim score normalized by the query row count, which reduces to
/// cosine when both sides are single rows.
pub(crate) fn similarity(a: &EmbeddingVector, b: &EmbeddingVector) -> f32 {
    let rows_a: Vec<&[f32]> = a.rows().collect();
    let rows_b: Vec<&[f32]> = b.rows().collect();
    if rows_a.is_empty() || rows_b.is_empty() {
        return 0.0;
    }
    let cosine = |x: &[f32], y: &[f32]| {
        let denom = norm(x) * norm(y);
        if denom > 0.0 { dot(x, y) / denom } else { 0.0 }
    };
    let total: f32 = rows_a
        .iter()
        .map(|qa| {
            rows_b
                .iter()
                .map(|qb| cosine(qa, qb))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum();
    total / rows_a.len() as f32
}

fn format_vector(vector: &EmbeddingVector) -> String {
    let row = |values: &[f32]| {
        let parts: Vec<String> = values.iter().map(|v| format!("{v:.6}")).collect();
        format!("[{}]", parts.join(", "))
    };
    if vector.is_multi_vector() {
        let rows: Vec<String> = vector.rows().map(row).collect();
        format!("[{}]", rows.join(", "))
    } else {
        row(&vector.values)
    }
}

pub(crate) fn run_embed(args: EmbedArgs) -> Result<()> {
    if args.prompts.is_empty() && args.images.is_empty() {
        bail!("nothing to embed: pass at least one -p/--prompt or --image");
    }
    if let Some(empty) = args.prompts.iter().position(String::is_empty) {
        bail!("prompt {empty} is empty");
    }
    for image in &args.images {
        if !image.is_file() {
            bail!("image file does not exist: {}", image.display());
        }
    }

    let model_dir =
        resolve_model_source_with_override(&args.model, args.models_dir.as_deref(), None)?;

    // Initialize the MLX runtime (selects GPU/CPU) before any forward pass.
    let _runtime = initialize_runtime();

    let loaded = load_embedding_model_with_options(
        &model_dir,
        EmbeddingLoadOptions {
            max_length: args.max_length,
        },
    )
    .with_context(|| format!("failed to load embedding model {}", model_dir.display()))?;
    let engine = EmbeddingEngine::new(loaded, args.batch_size);
    engine
        .validate_dimensions(args.dimensions)
        .map_err(|e| anyhow!("{e}"))?;
    if !args.images.is_empty() && !engine.supports_images() {
        bail!(
            "{:?} does not accept image inputs; drop --image",
            engine.model_type()
        );
    }

    let options = EmbedOptions {
        instruction: args.instruction.clone(),
        dimensions: args.dimensions,
    };
    let mut labels: Vec<String> = Vec::new();
    let mut vectors: Vec<EmbeddingVector> = Vec::new();
    let mut prompt_tokens = 0usize;

    if !args.prompts.is_empty() {
        let reply = engine
            .embed_texts(&args.prompts, &options)
            .map_err(|e| anyhow!("{e}"))?;
        prompt_tokens += reply.prompt_tokens;
        labels.extend(args.prompts.iter().cloned());
        vectors.extend(reply.vectors);
    }
    for path in &args.images {
        let bytes =
            std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let image = image::load_from_memory(&bytes)
            .with_context(|| format!("failed to decode {}", path.display()))?;
        let reply = engine
            .embed_image(ImageInput { image }, &options)
            .map_err(|e| anyhow!("{e}"))?;
        prompt_tokens += reply.prompt_tokens;
        labels.push(path.display().to_string());
        vectors.extend(reply.vectors);
    }

    let matrix: Option<Vec<Vec<f32>>> = (vectors.len() >= 2).then(|| {
        vectors
            .iter()
            .map(|a| vectors.iter().map(|b| similarity(a, b)).collect())
            .collect()
    });

    if args.json {
        let embeddings: Vec<serde_json::Value> = vectors
            .iter()
            .map(|v| {
                if v.is_multi_vector() {
                    json!(v.rows().map(|r| r.to_vec()).collect::<Vec<_>>())
                } else {
                    json!(v.values)
                }
            })
            .collect();
        let shapes: Vec<&Vec<usize>> = vectors.iter().map(|v| &v.shape).collect();
        let out = json!({
            "model": model_dir.display().to_string(),
            "model_type": format!("{:?}", engine.model_type()),
            "dim": engine.dim(),
            "max_length": engine.max_length(),
            "multi_vector": engine.multi_vector(),
            "inputs": labels,
            "embeddings": embeddings,
            "shapes": shapes,
            "prompt_tokens": prompt_tokens,
            "similarity": matrix,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    for vector in &vectors {
        println!("{}", format_vector(vector));
    }
    if let Some(matrix) = matrix {
        println!();
        println!("cosine similarity ({} inputs):", vectors.len());
        for (i, row) in matrix.iter().enumerate() {
            let cells: Vec<String> = row.iter().map(|v| format!("{v:7.4}")).collect();
            println!("  [{i}] {}", cells.join(" "));
        }
    }
    eprintln!(
        "{} input(s), {} prompt tokens, dim {}, max_length {}",
        vectors.len(),
        prompt_tokens,
        engine.dim(),
        engine.max_length()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single(values: &[f32]) -> EmbeddingVector {
        EmbeddingVector {
            values: values.to_vec(),
            shape: vec![values.len()],
        }
    }

    #[test]
    fn similarity_is_cosine_for_single_vectors() {
        let a = single(&[1.0, 0.0]);
        let b = single(&[0.0, 2.0]);
        let c = single(&[3.0, 0.0]);
        assert!((similarity(&a, &b)).abs() < 1e-6);
        assert!((similarity(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_is_maxsim_for_multi_vectors() {
        let query = EmbeddingVector {
            values: vec![1.0, 0.0, 0.0, 1.0],
            shape: vec![2, 2],
        };
        let doc = EmbeddingVector {
            values: vec![1.0, 0.0],
            shape: vec![1, 2],
        };
        // Row 0 matches perfectly (1.0), row 1 is orthogonal (0.0): mean 0.5.
        assert!((similarity(&query, &doc) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn text_layout_prints_rows_for_multi_vectors() {
        let v = EmbeddingVector {
            values: vec![1.0, 0.5, 0.25, 0.0],
            shape: vec![2, 2],
        };
        assert_eq!(
            format_vector(&v),
            "[[1.000000, 0.500000], [0.250000, 0.000000]]"
        );
        assert_eq!(format_vector(&single(&[0.5])), "[0.500000]");
    }
}
