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

//! The model-load boundary of the weight materialization policy (#1564).
//!
//! Every generation-model entry point in [`super`] runs its family loader
//! through [`with_weight_materialization`], so the policy in
//! [`mlxcel_core::weight_materialize`] applies to every family the same way:
//! the weights loaders register what they read, and the scope finishes once
//! the family has built its model and dropped its weight maps.

use anyhow::Result;
use mlxcel_core::weight_materialize::MaterializationScope;

/// Run `load` inside a weight materialization scope and apply the policy to
/// what it loaded.
///
/// The scope is finished only after `load` has returned, so the loader's own
/// weight maps are gone and the live references say what the model kept. A
/// failed load drops the scope without evaluating anything.
pub(crate) fn with_weight_materialization<T>(load: impl FnOnce() -> Result<T>) -> Result<T> {
    let scope = MaterializationScope::begin();
    let loaded = load()?;
    if let Some(report) = scope.finish() {
        report.emit();
    }
    Ok(loaded)
}
