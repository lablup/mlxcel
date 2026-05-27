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

//! Binary-only command handlers.
//!
//! Keeping subcommand implementations out of `main.rs` leaves the root file as
//! argument/schema wiring while command-specific execution logic evolves in
//! isolated modules.

pub(crate) mod detect;
pub(crate) mod download;
pub(crate) mod generate;
mod generate_vlm;
pub(crate) mod inspect;
pub(crate) mod models;
mod serve;

pub(crate) use detect::run_detect;
pub(crate) use download::run_download;
pub(crate) use generate::run_generate;
pub(crate) use inspect::run_inspect;
pub(crate) use models::{run_list_local, run_remove};
pub(crate) use serve::run_serve;
