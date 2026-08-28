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

//! Fill-in-the-middle flag group (llama-server b10621 parity, issue #1442).
//!
//! One flag, defined once and flattened into both server binaries so
//! `mlxcel serve` and `mlxcel-server` accept the same command line.
//!
//! b10621 declares `--spm-infill` as a value-less flag on the server example
//! with no environment binding, so this definition has none either: adding an
//! `LLAMA_ARG_*` variable mlxcel invented would be a compatibility claim
//! upstream does not make.
//!
//! Upstream reference:
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>
//!
//! Used by: mlxcel serve, mlxcel-server.

use clap::Args;

/// Shared fill-in-the-middle flag group.
#[derive(Args, Debug, Default, Clone)]
#[command(next_help_heading = "Infill (llama-server compatibility)")]
pub struct InfillArgs {
    /// Use the Suffix/Prefix/Middle ordering on `POST /infill` instead of
    /// Prefix/Suffix/Middle.
    ///
    /// Which ordering is correct is a property of the checkpoint: a model
    /// trained on one produces a fluent but wrong completion when prompted in
    /// the other, with no error to notice. CodeLlama's SentencePiece infill
    /// checkpoints are the usual reason to pass this.
    #[arg(long = "spm-infill", action = clap::ArgAction::SetTrue)]
    pub spm_infill: bool,
}
