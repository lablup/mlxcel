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

//! The direct `Engine` arm of the parity harness (#2172): the raw-completion
//! client over the loaded model. Since Phase 6 (#2176) that client lives in
//! `mlxcel_core::engine` and is the loop `mlxcel generate` itself runs, so
//! this arm and the `a:cli` arm differ only in the warmup pass `generate`
//! adds; a divergence between this arm and the server's dense B=1 arm is
//! scheduler policy, not engine execution.

use crate::LoadedModel;

/// The raw-completion client over a [`LoadedModel`], decoding one request at
/// a time.
pub type DirectEngine = mlxcel_core::engine::DirectEngine<LoadedModel>;
