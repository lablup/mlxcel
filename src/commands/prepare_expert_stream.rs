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

//! `mlxcel prepare-expert-stream`: write the contiguous expert file SSD expert
//! streaming reads automatically when present (issue #2882).

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use mlxcel::models::qwen4_exp::stream::source::write_contiguous_file;
use serde_json::Value;

/// Arguments for `mlxcel prepare-expert-stream`.
#[derive(Args, Debug)]
pub(crate) struct PrepareExpertStreamArgs {
    /// `qwen4_exp` checkpoint directory; the file is written inside it.
    #[arg(short, long, value_name = "PATH")]
    pub(crate) model: PathBuf,
}

pub(crate) fn run_prepare_expert_stream(args: PrepareExpertStreamArgs) -> Result<()> {
    let path = args.model.join("config.json");
    let config: Value = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))?;
    if config["model_type"].as_str() != Some("qwen4_exp") {
        bail!("{} is not a qwen4_exp checkpoint", args.model.display());
    }
    let text = config.get("text_config").unwrap_or(&config);
    let dim = |name: &str| {
        text[name]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| anyhow!("{} has no {name}", path.display()))
    };
    let report = write_contiguous_file(&args.model, dim("num_hidden_layers")?, dim("num_experts")?)
        .map_err(|error| anyhow!(error))?;
    println!(
        "Wrote {} ({:.2} GiB for {} layers), verified against the checkpoint.",
        report.path.display(),
        report.bytes as f64 / (1u64 << 30) as f64,
        report.layers,
    );
    Ok(())
}
