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

use super::*;
use crate::imagegen::testing::{STUB_FAMILIES, write_model_index};
use tempfile::TempDir;

#[test]
fn every_shipping_row_is_a_probe_family() {
    assert!(FAMILIES.iter().any(|f| f.name == "flux2"));
    for family in FAMILIES {
        assert!(
            crate::imagegen::IMAGE_FAMILIES.contains(&family.name),
            "{} is not a probe family",
            family.name
        );
    }
}

#[test]
fn an_unrecognized_directory_is_reported_as_such() {
    let dir = TempDir::new().unwrap();
    let err = load_image_model(dir.path()).err().expect("no family");
    assert!(
        err.to_string()
            .contains("no image generation family recognized"),
        "{err}"
    );
}

#[test]
fn a_recognized_but_unported_family_is_named() {
    let dir = TempDir::new().unwrap();
    write_model_index(dir.path(), "Flux2Pipeline");
    // A table without a flux2 row, as a build that has not ported it.
    let err = load_image_model_from(dir.path(), &[])
        .err()
        .expect("unported");
    assert_eq!(
        err.to_string(),
        "image generation family flux2 is not supported by this build"
    );
}

#[test]
fn a_registered_family_is_constructed() {
    let dir = TempDir::new().unwrap();
    write_model_index(dir.path(), "Flux2Pipeline");
    let model = load_image_model_from(dir.path(), &STUB_FAMILIES).unwrap();
    assert_eq!(model.family(), "flux2");
}

#[test]
fn lookup_is_by_the_probe_name() {
    let dir = TempDir::new().unwrap();
    write_model_index(dir.path(), "QwenImagePipeline");
    let model = load_image_model_from(dir.path(), &STUB_FAMILIES).unwrap();
    assert_eq!(model.family(), "qwen_image");
}

#[test]
fn a_family_missing_from_the_given_table_is_unsupported() {
    let dir = TempDir::new().unwrap();
    write_model_index(dir.path(), "ErnieImagePipeline");
    let err = load_image_model_from(dir.path(), &STUB_FAMILIES)
        .err()
        .unwrap();
    assert!(err.to_string().contains("ernie_image"), "{err}");
}

#[test]
fn resolve_family_needs_no_runtime_and_returns_the_row() {
    let dir = TempDir::new().unwrap();
    write_model_index(dir.path(), "Flux2Pipeline");
    assert_eq!(
        resolve_family(dir.path(), &STUB_FAMILIES).unwrap().name,
        "flux2"
    );
    let err = resolve_family(dir.path(), &[]).err().unwrap();
    assert!(
        err.to_string().contains("not supported by this build"),
        "{err}"
    );
}
