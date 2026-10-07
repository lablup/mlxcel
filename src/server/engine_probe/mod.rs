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

//! Drivers that run the CLI decode path and the server decode path on the
//! same input, for the cross-path parity harness (`mlxcel-engine-parity`) and
//! the single-stream decode benchmark (`mlxcel-bench-engine`), issue #2167.
//!
//! Epic #2166 merges the two decode paths into one batch-native engine (ADR
//! 0007). These drivers record where the paths stand before that work and are
//! rerun by every later phase: the harness to show the token streams
//! converge, the benchmark to show single-stream throughput holds.
//!
//! Both drivers call the real code: [`cli_engine`] calls `CxxGenerator` the
//! way `mlxcel generate` and `mlxcel-bench-decode` do, and [`server_engine`]
//! builds the `mlxcel-server` configuration and model worker and feeds the
//! `BatchScheduler` through its request channel. Nothing here reimplements a
//! decode loop.

pub mod cases;
pub mod cli_engine;
pub mod divergence;
pub mod prompt;
pub mod server_engine;

pub use cases::{ParityCase, describe_prefill_partition};
pub use divergence::{Divergence, first_divergence};
pub use server_engine::{
    ProbeCacheKey, ServerEngine, ServerEngineOptions, ServerEngineRequest, ServerEngineRun,
};
