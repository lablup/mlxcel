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

//! Refuse serve paths that drop the per-request reasoning effort (issue
//! #3457).
//!
//! A model whose `config.json` declares `reasoning_effort_steering` (Raon-1
//! mini) steers each sequence by the level its request resolved to. The
//! level travels in `ServerGenerateOptions::model_reasoning_effort` to the
//! batch scheduler's `set_sequence_reasoning_effort`. The pipeline-parallel
//! workers, the disaggregated prefill / decode frames and the XLA worker
//! never read it, so on those paths every request would run at the model
//! default while the response claimed the requested level. Until one of them
//! carries the level, startup refuses the combination by name.

use std::path::Path;

use crate::distributed::disaggregated::ServingMode;
use crate::models::reasoning_effort_steering::ReasoningEffortSteering;

/// The serve paths a launch resolved to, as far as the steering guard cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServePaths {
    /// An in-process or remote pipeline-parallel runtime, or a pipeline stage
    /// peer.
    pub pipeline_parallel: bool,
    /// The disaggregated serving role.
    pub serving_mode: ServingMode,
    /// The OpenXLA engine serves requests (`MLXCEL_BACKEND=xla` on an
    /// `xla-iree` build).
    pub xla: bool,
}

impl ServePaths {
    /// The first path in `self` that drops the reasoning effort, by name.
    fn unsteered_path(self) -> Option<&'static str> {
        if self.pipeline_parallel {
            return Some("pipeline-parallel serving (--pp-layers, --pp-auto, --pp-peer)");
        }
        match self.serving_mode {
            ServingMode::PrefillOnly => {
                return Some("disaggregated prefill serving (--node-role prefill)");
            }
            ServingMode::DecodeOnly => {
                return Some("disaggregated decode serving (--node-role decode)");
            }
            ServingMode::Hybrid | ServingMode::Router => {}
        }
        self.xla.then_some("the XLA backend (MLXCEL_BACKEND=xla)")
    }
}

/// Fail startup when `model_path` declares `reasoning_effort_steering` and
/// `paths` serves it on a path that drops the per-request level. A model
/// without the block, or a launch on the batch scheduler, passes without
/// reading the config beyond that check.
///
/// # Errors
/// Names the path and the model; also returns the config's own error for an
/// invalid steering block, since the model would not load with it either.
///
/// Used by: `start_server`
pub(crate) fn refuse_reasoning_effort_steering_on_unsteered_paths(
    model_path: &Path,
    paths: ServePaths,
) -> anyhow::Result<()> {
    // Image-only serving has no chat checkpoint, and an empty path would
    // resolve to the working directory.
    if model_path.as_os_str().is_empty() {
        return Ok(());
    }
    let Some(path) = paths.unsteered_path() else {
        return Ok(());
    };
    match ReasoningEffortSteering::from_model_dir(model_path) {
        Ok(None) => Ok(()),
        Ok(Some(_)) => anyhow::bail!(
            "{} declares reasoning_effort_steering, but {path} does not carry the per-request \
             reasoning effort: every request would silently run at the model's default level. \
             Serve this model on a single node with the batch scheduler",
            model_path.display()
        ),
        Err(err) => anyhow::bail!("{}: {err}", model_path.display()),
    }
}

#[cfg(test)]
#[path = "reasoning_steering_startup_guard_tests.rs"]
mod tests;
