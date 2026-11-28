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

//! `mlxcel decide`: typed decisions through the same validation and scorer
//! `POST /v1/decisions` uses, printed as the route's response JSON.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{ArgGroup, Args};

use mlxcel::decision::cli::{DecideArgs as DecideRequest, prepare_decide};
use mlxcel::decision::media::load_cli_media;
use mlxcel::decision::{load_decision_model, predict, validate_questions};
use mlxcel::downloader::resolve_model_source_with_override;
use mlxcel::initialize_runtime_checked;

/// Arguments for `mlxcel decide`.
#[derive(Args, Debug)]
#[command(next_help_heading = "Decide Options")]
#[command(group(ArgGroup::new("state_source").args(["state", "state_file"]).multiple(false)))]
#[command(group(
    ArgGroup::new("questions_source")
        .args(["questions", "questions_file"])
        .required(true)
        .multiple(false)
))]
pub(crate) struct DecideArgs {
    /// Decision checkpoint: a local directory or a HuggingFace `owner/name`
    /// repo-id (resolved and auto-downloaded like `mlxcel generate -m`).
    #[arg(short, long, value_name = "MODEL_OR_REPO_ID")]
    pub(crate) model: PathBuf,

    /// Text to evaluate.
    #[arg(long, value_name = "TEXT")]
    pub(crate) state: Option<String>,

    /// UTF-8 text file to evaluate.
    #[arg(long = "state-file", value_name = "PATH")]
    pub(crate) state_file: Option<PathBuf>,

    /// Image to evaluate; repeat for several (state is optional with media).
    #[arg(long = "image", value_name = "PATH_OR_URL")]
    pub(crate) images: Vec<String>,

    /// Video to evaluate; repeat for several (state is optional with media).
    #[arg(long = "video", value_name = "PATH")]
    pub(crate) videos: Vec<PathBuf>,

    /// Audio clip to evaluate (state is optional with media).
    #[arg(long, value_name = "PATH_OR_URL")]
    pub(crate) audio: Option<String>,

    /// Named questions as JSON.
    #[arg(long, value_name = "JSON")]
    pub(crate) questions: Option<String>,

    /// JSON file of named questions.
    #[arg(long = "questions-file", value_name = "PATH")]
    pub(crate) questions_file: Option<PathBuf>,

    /// Model-store root used to resolve a repo-id (defaults to the global store).
    #[arg(long = "models-dir", env = "MLXCEL_MODELS_DIR", value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,
}

impl DecideArgs {
    fn request(&self) -> DecideRequest {
        DecideRequest {
            state: self.state.clone(),
            state_file: self.state_file.clone(),
            questions: self.questions.clone(),
            questions_file: self.questions_file.clone(),
            images: self.images.clone(),
            videos: self.videos.clone(),
            audio: self.audio.clone(),
        }
    }
}

pub(crate) fn run_decide(args: DecideArgs) -> Result<()> {
    let request = args.request();
    // State, questions and the state-or-media rule are checked before the
    // model is resolved or loaded, as upstream's `decide.py` does.
    let prepared = prepare_decide(&request)?;

    let model_dir =
        resolve_model_source_with_override(&args.model, args.models_dir.as_deref(), None)?;
    let _runtime = initialize_runtime_checked()?;
    let mut loaded = load_decision_model(&model_dir)
        .with_context(|| format!("failed to load decision model {}", model_dir.display()))?;
    // Rules 1 to 8 run against the loaded model's declarations before any
    // medium is read or fetched, so a refused medium costs nothing, as on the
    // route. `predict` re-validates the same request below.
    validate_questions(
        loaded.model.decision_types(),
        loaded.model.decision_media(),
        prepared.state.as_ref(),
        &prepared.questions,
        request.media_presence(),
    )?;
    let media = load_cli_media(&request.images, &request.videos, request.audio.as_deref())?;

    let result = predict(
        loaded.model.as_mut(),
        prepared.state.as_ref(),
        &prepared.questions,
        media,
    )?;
    let response = result.to_response_json(&model_dir.display().to_string());
    // serde_json keeps non-ASCII as is, like upstream's `ensure_ascii=False`.
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}

#[cfg(test)]
#[path = "decide_tests.rs"]
mod tests;
