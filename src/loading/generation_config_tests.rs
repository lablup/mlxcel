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

use std::fs;

use super::{GenerationConfigDefaults, read_eos_token_ids, read_generation_config_defaults};

fn write_generation_config(contents: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(dir.path().join("generation_config.json"), contents).expect("write config");
    dir
}

#[test]
fn muse_glimmer_generation_defaults_match_pinned_fixture() {
    let dir = write_generation_config(include_str!(
        "../../tests/fixtures/muse_glimmer/generation_config.json"
    ));

    let defaults = read_generation_config_defaults(dir.path());

    assert_eq!(
        defaults,
        GenerationConfigDefaults {
            eos_token_ids: vec![200001, 200008],
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(64),
        }
    );
    assert_eq!(read_eos_token_ids(dir.path()), vec![200001, 200008]);
}

#[test]
fn generation_defaults_accept_single_eos_and_partial_sampling() {
    let dir = write_generation_config(
        r#"{
            "eos_token_id": 42,
            "temperature": 0.7,
            "top_k": 32
        }"#,
    );

    let defaults = read_generation_config_defaults(dir.path());

    assert_eq!(defaults.eos_token_ids, vec![42]);
    assert_eq!(defaults.temperature, Some(0.7));
    assert_eq!(defaults.top_p, None);
    assert_eq!(defaults.top_k, Some(32));
}

#[test]
fn generation_defaults_ignore_malformed_or_out_of_range_fields() {
    let dir = write_generation_config(
        r#"{
            "eos_token_id": ["bad", 7],
            "temperature": "hot",
            "top_p": null,
            "top_k": 2147483648
        }"#,
    );

    let defaults = read_generation_config_defaults(dir.path());

    assert_eq!(defaults.eos_token_ids, vec![7]);
    assert_eq!(defaults.temperature, None);
    assert_eq!(defaults.top_p, None);
    assert_eq!(defaults.top_k, None);
}

#[test]
fn missing_generation_config_returns_empty_defaults() {
    let dir = tempfile::tempdir().expect("tempdir");

    assert_eq!(
        read_generation_config_defaults(dir.path()),
        GenerationConfigDefaults::default()
    );
}
