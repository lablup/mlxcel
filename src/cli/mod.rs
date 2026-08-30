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

//! Shared CLI argument groups exposed across the mlxcel binaries.
//!
//! Modules under this namespace own the canonical clap definitions for flag
//! sets that more than one binary (`mlxcel`, `mlxcel-server`, future tooling)
//! must accept identically. Any binary that wants to expose a shared flag
//! group flattens the corresponding struct via `#[command(flatten)]` so the
//! help text, env-var bridges, and resolution helpers stay defined in exactly
//! one place.

pub mod batch_quant_args;
pub mod cache_args;
pub mod cache_list;
pub mod chat_compat_args;
pub mod completion;
pub mod context_args;
pub mod embedding_compat_args;
pub mod flag_surface;
pub mod ggml_compat_args;
pub mod grammar_args;
pub mod infill_args;
pub mod llama_short_flags;
pub mod logging_compat_args;
pub mod max_tokens;
pub mod multimodal_compat_args;
pub mod rope_args;
pub mod slot_args;
pub mod spec_compat_args;
pub mod speculative_args;
pub mod turbo_args;
pub mod ui_compat_args;
