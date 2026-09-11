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

use mlxcel::models::{ModelType, get_model_type};
use std::path::PathBuf;

#[test]
fn test_detect_llama_model_type() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("models/Meta-Llama-3.1-8B-Instruct-4bit");

    if !d.exists() {
        eprintln!("Skipping test: Model directory not found at {:?}", d);
        return;
    }

    let model_type = get_model_type(&d).expect("Failed to detect model type");
    assert_eq!(model_type, ModelType::Llama);
}

#[test]
fn test_detect_moondream3_model_type() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("models/moondream3-preview-4bit");

    if !d.exists() {
        eprintln!("Skipping test: Model directory not found at {:?}", d);
        return;
    }

    let model_type = get_model_type(&d).expect("Failed to detect model type");
    assert_eq!(model_type, ModelType::Moondream3VLM);
}
