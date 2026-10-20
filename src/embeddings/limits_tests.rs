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

use std::fs;

use tempfile::TempDir;

use super::{EMBEDDING_MAX_LENGTH_CAP, derive_max_length};

#[test]
fn derive_max_length_reads_processor_config_max_length() {
    let tmp = TempDir::new().expect("tempdir");
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{"max_length": 4096}"#,
    )
    .expect("write processor_config.json");

    assert_eq!(derive_max_length(tmp.path(), false, None), 4096);
}

#[test]
fn derive_max_length_reads_processor_config_p_max_length() {
    let tmp = TempDir::new().expect("tempdir");
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{
            "max_input_tiles": 6,
            "p_max_length": 4096,
            "q_max_length": 512,
            "processor_class": "LlamaNemotronVLProcessor"
        }"#,
    )
    .expect("write processor_config.json");

    assert_eq!(derive_max_length(tmp.path(), false, None), 4096);
}

#[test]
fn derive_max_length_falls_through_when_processor_config_lacks_length_keys() {
    let tmp = TempDir::new().expect("tempdir");
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{"processor_class": "AutoProcessor", "max_length": 0}"#,
    )
    .expect("write processor_config.json");

    assert_eq!(
        derive_max_length(tmp.path(), false, None),
        EMBEDDING_MAX_LENGTH_CAP
    );
}

#[test]
fn derive_max_length_takes_minimum_across_processor_and_tokenizer_configs() {
    let tmp = TempDir::new().expect("tempdir");
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{"p_max_length": 4096}"#,
    )
    .expect("write processor_config.json");
    fs::write(
        tmp.path().join("tokenizer_config.json"),
        r#"{"model_max_length": 2048}"#,
    )
    .expect("write tokenizer_config.json");

    assert_eq!(derive_max_length(tmp.path(), false, None), 2048);
    assert_eq!(derive_max_length(tmp.path(), false, Some(512)), 512);
}
