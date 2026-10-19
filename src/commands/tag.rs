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

//! `mlxcel tag`: tag and redact entities with a token-classification
//! checkpoint, mirroring `python -m mlx_vlm.token_classification`.

use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;

use mlxcel::downloader::resolve_model_source_with_override;
use mlxcel::initialize_runtime_checked;
use mlxcel::token_classification::{DecodeMode, TagOptions, load_token_classifier};

/// Arguments for `mlxcel tag`.
#[derive(Args, Debug)]
#[command(next_help_heading = "Tag Options")]
pub(crate) struct TagArgs {
    /// Token-classification checkpoint: a local directory or a HuggingFace
    /// `owner/name` repo-id (resolved and auto-downloaded like `mlxcel
    /// generate -m`).
    #[arg(short, long, value_name = "MODEL_OR_REPO_ID")]
    pub(crate) model: PathBuf,

    /// Text to inspect; `-` reads it from stdin.
    #[arg(value_name = "TEXT")]
    pub(crate) text: String,

    /// Independent per-token decoding, for models whose default is Viterbi.
    #[arg(long)]
    pub(crate) argmax: bool,

    /// Text every redacted span becomes (default: the label placeholder,
    /// e.g. `<GIVEN_NAME>`).
    #[arg(long, value_name = "STR")]
    pub(crate) replacement: Option<String>,

    /// Comma-separated entity labels to report but leave unredacted.
    #[arg(long, value_name = "L1,L2", value_delimiter = ',')]
    pub(crate) keep: Vec<String>,

    /// Model-store root used to resolve a repo-id (defaults to the global store).
    #[arg(long = "models-dir", env = "MLXCEL_MODELS_DIR", value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,
}

impl TagArgs {
    /// The per-request options the flags describe.
    pub(crate) fn options(&self) -> TagOptions {
        TagOptions {
            decode: self.argmax.then_some(DecodeMode::Argmax),
            replacement: self.replacement.clone(),
            keep_labels: self
                .keep
                .iter()
                .map(|label| label.trim())
                .filter(|label| !label.is_empty())
                .map(str::to_string)
                .collect(),
            ..TagOptions::default()
        }
    }
}

/// `TEXT`, or stdin for `-`.
pub(crate) fn read_text_arg(text: &str) -> Result<String> {
    if text != "-" {
        return Ok(text.to_string());
    }
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .context("failed to read the text from stdin")?;
    Ok(buffer)
}

pub(crate) fn run_tag(args: TagArgs) -> Result<()> {
    let text = read_text_arg(&args.text)?;
    let options = args.options();
    let model_dir =
        resolve_model_source_with_override(&args.model, args.models_dir.as_deref(), None)?;

    // Refuses to start when this build's CUDA architectures do not cover the
    // host GPU (#1537).
    let _runtime = initialize_runtime_checked()?;

    let mut loaded = load_token_classifier(&model_dir)
        .with_context(|| format!("failed to load token classifier {}", model_dir.display()))?;
    let result = loaded.classifier.classify(&text, &options)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    eprintln!(
        "{} span(s), kind {}, model_type {}",
        result.spans.len(),
        loaded.kind.as_str(),
        loaded.model_type
    );
    Ok(())
}
