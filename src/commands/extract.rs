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

//! `mlxcel extract`: span extraction and zero-shot classification with labels
//! chosen per request (GLiNER2.5-style `extract_entities` / `classify_text`).

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Args};
use serde_json::Value;

use mlxcel::downloader::resolve_model_source_with_override;
use mlxcel::initialize_runtime_checked;
use mlxcel::token_classification::extract::{
    parse_classify_tasks, parse_label_list, parse_labels_json,
};
use mlxcel::token_classification::gliner2_5::check_max_len;
use mlxcel::token_classification::gliner2_5_prepare::WordSplitter;
use mlxcel::token_classification::{
    ClassifyTask, ExtractOptions, SpanExtractorLoadOptions, SpanExtractorModel,
    load_span_extractor_with,
};

use super::tag::read_text_arg;

/// Arguments for `mlxcel extract`.
#[derive(Args, Debug)]
#[command(next_help_heading = "Extract Options")]
#[command(group(
    ArgGroup::new("request")
        .required(true)
        .args(["labels", "labels_json", "classify"])
))]
pub(crate) struct ExtractArgs {
    /// Span-extraction checkpoint: a local directory or a HuggingFace
    /// `owner/name` repo-id.
    #[arg(short, long, value_name = "MODEL_OR_REPO_ID")]
    pub(crate) model: PathBuf,

    /// Text to inspect; `-` reads it from stdin.
    #[arg(value_name = "TEXT")]
    pub(crate) text: String,

    /// Comma-separated entity labels to extract.
    #[arg(long, value_name = "L1,L2")]
    pub(crate) labels: Option<String>,

    /// Entity labels with descriptions: `{"label": "description", ...}`.
    #[arg(long = "labels-json", value_name = "JSON")]
    pub(crate) labels_json: Option<String>,

    /// Classify the text instead: `{"task": ["l1", "l2"] | {"labels": [...],
    /// "prompt"?, "label_descriptions"?, "multi_label"?, "threshold"?}}`.
    #[arg(long, value_name = "TASKS_JSON")]
    pub(crate) classify: Option<String>,

    /// Minimum score for an entity (or a multi-label class) to be returned.
    #[arg(long, default_value_t = 0.5, value_name = "P")]
    pub(crate) threshold: f32,

    /// Add each entity's confidence.
    #[arg(long = "include-confidence")]
    pub(crate) include_confidence: bool,

    /// Add each entity's character `start` / `end`.
    #[arg(long = "include-spans")]
    pub(crate) include_spans: bool,

    /// GLiNER2.5 only: override of the checkpoint's `max_len`, the longest
    /// sequence one request encodes (default: the checkpoint's value, 4096 on
    /// `fastino/gliner2.5-small-v1`).
    #[arg(long, value_name = "N")]
    pub(crate) max_len: Option<usize>,

    /// GLiNER2.5 only: how the text is cut into words. `char` scores every
    /// non-space character separately, for scripts without word spacing such
    /// as CJK; `whitespace` is the default.
    #[arg(long = "word-splitter", value_name = "whitespace|char")]
    pub(crate) word_splitter: Option<String>,

    /// Model-store root used to resolve a repo-id (defaults to the global store).
    #[arg(long = "models-dir", env = "MLXCEL_MODELS_DIR", value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,
}

/// What one `mlxcel extract` invocation asks the model for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExtractRequest {
    Entities {
        labels: Vec<(String, Option<String>)>,
        options: ExtractOptions,
    },
    Classify {
        tasks: Vec<(String, ClassifyTask)>,
    },
}

impl ExtractArgs {
    /// Validate the flags into a request, before any model is loaded.
    pub(crate) fn request(&self) -> Result<ExtractRequest> {
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) {
            bail!(
                "--threshold must be between 0 and 1, got {}",
                self.threshold
            );
        }
        let options = ExtractOptions {
            include_confidence: self.include_confidence,
            include_spans: self.include_spans,
        };
        if let Some(raw) = self.classify.as_deref() {
            if options != ExtractOptions::default() {
                bail!(
                    "--include-confidence and --include-spans apply to entity extraction, \
                     not --classify"
                );
            }
            return Ok(ExtractRequest::Classify {
                tasks: parse_classify_tasks(raw)?,
            });
        }
        let labels = match (self.labels.as_deref(), self.labels_json.as_deref()) {
            (Some(raw), None) => parse_label_list(raw)?,
            (None, Some(raw)) => parse_labels_json(raw)?,
            _ => bail!("pass exactly one of --labels, --labels-json or --classify"),
        };
        Ok(ExtractRequest::Entities { labels, options })
    }

    /// GLiNER2.5 load choices from `--max-len` / `--word-splitter`; `None`
    /// fields keep the checkpoint's own values. Runs before any model loads.
    pub(crate) fn gliner_load_options(&self) -> Result<SpanExtractorLoadOptions> {
        let max_len = match self.max_len {
            None => None,
            Some(max_len) => {
                check_max_len(max_len)?;
                Some(max_len)
            }
        };
        let word_splitter = self
            .word_splitter
            .as_deref()
            .map(parse_word_splitter)
            .transpose()?;
        Ok(SpanExtractorLoadOptions {
            max_len,
            word_splitter,
        })
    }
}

/// `--word-splitter whitespace|char`.
fn parse_word_splitter(raw: &str) -> Result<WordSplitter> {
    match raw {
        "whitespace" => Ok(WordSplitter::Whitespace),
        "char" => Ok(WordSplitter::Char),
        other => bail!("--word-splitter must be whitespace or char, got '{other}'"),
    }
}

/// Run one request against a loaded extractor, returning the JSON to print.
pub(crate) fn execute(
    model: &mut dyn SpanExtractorModel,
    text: &str,
    threshold: f32,
    request: &ExtractRequest,
) -> Result<Value> {
    match request {
        ExtractRequest::Entities { labels, options } => Ok(serde_json::to_value(
            model.extract_entities(text, labels, threshold, *options)?,
        )?),
        ExtractRequest::Classify { tasks } => model.classify_text(text, tasks, threshold),
    }
}

pub(crate) fn run_extract(args: ExtractArgs) -> Result<()> {
    let request = args.request()?;
    let gliner = args.gliner_load_options()?;
    let text = read_text_arg(&args.text)?;
    let model_dir =
        resolve_model_source_with_override(&args.model, args.models_dir.as_deref(), None)?;
    let _runtime = initialize_runtime_checked()?;
    let mut loaded = load_span_extractor_with(&model_dir, &gliner)
        .with_context(|| format!("failed to load span extractor {}", model_dir.display()))?;
    let output = execute(loaded.extractor.as_mut(), &text, args.threshold, &request)?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[cfg(test)]
#[path = "extract_tests.rs"]
mod extract_tests;
