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

//! `mlxcel decide` input preparation.
//!
//! Everything here runs before the model loads, as upstream's `decide.py`
//! does: the state and question sources are read, the questions are parsed
//! and structurally validated (rule 4), and a call with neither state nor
//! media is refused. The model-dependent rules run later, in
//! [`super::predict`].

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::validate_question_structure;

/// The raw `mlxcel decide` arguments that carry the request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecideArgs {
    /// `--state TEXT`.
    pub state: Option<String>,
    /// `--state-file PATH`: UTF-8 text.
    pub state_file: Option<PathBuf>,
    /// `--questions JSON`.
    pub questions: Option<String>,
    /// `--questions-file PATH`: a JSON file.
    pub questions_file: Option<PathBuf>,
    /// `--image PATH_OR_URL`, repeatable.
    pub images: Vec<String>,
    /// `--video PATH`, repeatable.
    pub videos: Vec<PathBuf>,
    /// `--audio PATH_OR_URL`.
    pub audio: Option<String>,
}

impl DecideArgs {
    /// Whether any media argument was given.
    #[must_use]
    pub fn has_media(&self) -> bool {
        !self.images.is_empty() || !self.videos.is_empty() || self.audio.is_some()
    }
}

/// The state and questions of one call, ready for [`super::predict`].
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedDecide {
    /// The state as a JSON string, or `None`.
    pub state: Option<Value>,
    /// The parsed questions object.
    pub questions: Value,
}

/// Read and check the request arguments without touching the model.
pub fn prepare_decide(args: &DecideArgs) -> Result<PreparedDecide> {
    if args.state.is_some() && args.state_file.is_some() {
        bail!("--state and --state-file are mutually exclusive");
    }
    if args.state.is_none() && args.state_file.is_none() && !args.has_media() {
        bail!("one of the arguments --state --state-file --image --video --audio is required");
    }
    let questions_text = match (&args.questions, &args.questions_file) {
        (Some(_), Some(_)) => bail!("--questions and --questions-file are mutually exclusive"),
        (None, None) => bail!("one of the arguments --questions --questions-file is required"),
        (Some(text), None) => text.clone(),
        (None, Some(path)) => std::fs::read_to_string(path)
            .with_context(|| format!("failed to read --questions-file {}", path.display()))?,
    };
    let state = match (&args.state, &args.state_file) {
        (Some(text), _) => Some(Value::String(text.clone())),
        (None, Some(path)) => Some(Value::String(
            std::fs::read_to_string(path)
                .with_context(|| format!("failed to read --state-file {}", path.display()))?,
        )),
        (None, None) => None,
    };
    let questions: Value = serde_json::from_str(&questions_text)
        .with_context(|| "questions are not valid JSON".to_string())?;
    validate_question_structure(&questions)?;
    Ok(PreparedDecide { state, questions })
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
