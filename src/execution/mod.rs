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

//! Shared execution-plane helpers used by both the CLI and the HTTP server.
//!
//! Keeping runtime/device resolution and sampling assembly together makes new
//! entry points easier to add without re-implementing environment parsing or
//! generation defaults in multiple places.

pub mod kv_arch;
pub mod memory_estimate;
pub mod quant_advisor;
pub mod runtime;
pub mod sampling;
