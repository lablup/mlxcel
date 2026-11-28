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

//! `mlxcel prepare-external-ple`: build a Qwen3.8-Flash-Next (`qwen4_exp`)
//! model view whose n-gram table is read from disk instead of living in the
//! MLX parameters (issue #2881).

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Args;
use mlxcel::models::qwen4_exp::ple_prepare::{
    PrepareLayout, PrepareOptions, prepare_external_ple_model,
};

/// Arguments for `mlxcel prepare-external-ple`.
#[derive(Args, Debug)]
pub(crate) struct PrepareExternalPleArgs {
    /// Source checkpoint with the n-gram table in its safetensors shards.
    #[arg(short, long, value_name = "PATH")]
    pub(crate) model: PathBuf,

    /// Output directory (created if missing; must be empty, on the same
    /// filesystem as the source, since shards are hard-linked).
    #[arg(short, long, value_name = "PATH")]
    pub(crate) output: PathBuf,

    /// Table layout: `interleaved` writes a one-record-per-row `ple-q4.rows`
    /// file (about 32 GB for the 320M-row table); `ranges` writes only a
    /// manifest pointing at the source shards.
    #[arg(long, default_value = "interleaved", value_name = "LAYOUT")]
    pub(crate) layout: String,

    /// Row-cache hint recorded in the manifest, in rows (0 keeps the
    /// `MLXCEL_PLE_CACHE_BYTES` budget).
    #[arg(long, default_value_t = 0, value_name = "ROWS")]
    pub(crate) cache_rows: u64,
}

pub(crate) fn run_prepare_external_ple(args: PrepareExternalPleArgs) -> Result<()> {
    let layout = match args.layout.as_str() {
        "interleaved" => PrepareLayout::Interleaved,
        "ranges" => PrepareLayout::SafetensorsRanges,
        other => bail!("--layout: unknown value {other:?}; expected interleaved or ranges"),
    };
    let report = prepare_external_ple_model(
        &args.model,
        &args.output,
        &PrepareOptions {
            layout,
            cache_rows: args.cache_rows,
        },
    )?;
    println!(
        "Wrote {} ({} rows, {:.2} GiB table, {} shards linked)",
        report.target.display(),
        report.row_count,
        report.table_bytes as f64 / (1u64 << 30) as f64,
        report.linked_files,
    );
    Ok(())
}
