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

//! Small helpers for exact-prefix recurrent-state snapshots.
//!
//! The snapshot container itself lives in `mlxcel-core` so it can be exposed
//! through the `LanguageModel` trait. These helpers keep the model files from
//! repeating the same optional-tensor copy/restore boilerplate.

use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) fn push_optional(
    snapshot: &mut ModelStateSnapshot,
    name: impl Into<String>,
    array: &Option<UniquePtr<MlxArray>>,
) {
    if let Some(array) = array.as_ref().and_then(|a| a.as_ref()) {
        snapshot.push_tensor(name, array);
    }
}

pub(crate) fn restore_optional(
    snapshot: &ModelStateSnapshot,
    name: impl AsRef<str>,
) -> Option<UniquePtr<MlxArray>> {
    snapshot.tensor(name.as_ref()).map(mlxcel_core::copy)
}
