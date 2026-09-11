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

//! Binary-side handler for the `mlxcel download` subcommand.
//!
//! This module is intentionally thin: it adapts the clap-parsed
//! [`mlxcel::downloader::DownloadArgs`] into [`mlxcel::downloader::DownloadOptions`]
//! and delegates to the shared [`mlxcel::downloader::download_repo`]. Both
//! `mlxcel` and `mlxcel-server` invoke the same library entry point so the
//! supported file set and flag semantics stay in sync without duplication.

use anyhow::Result;

use mlxcel::downloader::{DownloadArgs, DownloadOptions, download_repo};

pub(crate) fn run_download(args: DownloadArgs) -> Result<()> {
    let opts = DownloadOptions::from_args(&args);
    download_repo(opts)
}
