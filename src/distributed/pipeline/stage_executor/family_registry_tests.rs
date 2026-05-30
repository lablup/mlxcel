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

//! Unit tests for the pipeline-parallel family registry.
//!
//! These tests run on every `cargo test`; they do not require any model
//! weights.

use super::{StageFamily, supported_families};
use std::collections::HashSet;

#[test]
fn supported_families_is_sorted_by_name() {
    let names: Vec<&'static str> = supported_families().iter().map(|f| f.name()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(
        names, sorted,
        "supported_families() must stay sorted by StageFamily::name to keep handshake payloads byte-identical"
    );
}

#[test]
fn supported_families_names_are_unique() {
    let names: Vec<&'static str> = supported_families().iter().map(|f| f.name()).collect();
    let unique: HashSet<&'static str> = names.iter().copied().collect();
    assert_eq!(
        names.len(),
        unique.len(),
        "every StageFamily must carry a distinct textual name"
    );
}

#[test]
fn new_issue_345_families_are_registered() {
    // Explicitly confirm the five families added appear in
    // the registry. This is a floor test: removing one of these without
    // bumping the pipeline capability protocol version would be a breaking
    // change for running multi-host deployments.
    let families: HashSet<StageFamily> = supported_families().iter().copied().collect();
    for family in [
        StageFamily::Mistral,
        StageFamily::Mixtral,
        StageFamily::DeepSeekV3,
        StageFamily::Llama4,
        StageFamily::Jamba,
        StageFamily::NemotronH,
    ] {
        assert!(
            families.contains(&family),
            "StageFamily::{family:?} must appear in supported_families() (regression)",
        );
    }
}

#[test]
fn all_stage_family_variants_have_stable_names() {
    // If a new variant is added, this match must be updated to give it a
    // stable on-the-wire name. The `#[non_exhaustive]` attribute would
    // otherwise hide the breakage from callers.
    for family in supported_families().iter().copied() {
        let name = family.name();
        assert!(
            !name.is_empty(),
            "StageFamily::{family:?} carries an empty name — handshake would break"
        );
        assert!(
            name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "StageFamily::{family:?} name '{name}' must be ASCII / snake_case for capability negotiation"
        );
    }
}
