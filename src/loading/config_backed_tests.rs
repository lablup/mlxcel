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

use super::is_config_backed_model_type;
use crate::model_metadata::has_config_backed_registration;
use crate::models::ModelType;

#[test]
fn config_backed_registry_covers_standard_text_models() {
    assert!(is_config_backed_model_type(ModelType::Llama));
    assert!(is_config_backed_model_type(ModelType::Gemma3));
    assert!(is_config_backed_model_type(ModelType::Gemma4));
    assert!(is_config_backed_model_type(ModelType::Step3p5));
}

#[test]
fn config_backed_registry_excludes_special_and_vlm_models() {
    assert!(!is_config_backed_model_type(ModelType::Qwen35));
    assert!(!is_config_backed_model_type(ModelType::Gemma3n));
    assert!(!is_config_backed_model_type(ModelType::Qwen3VL));
}

#[test]
fn config_backed_registry_is_driven_by_shared_registration_surface() {
    assert!(has_config_backed_registration(ModelType::Llama4));
    assert!(has_config_backed_registration(ModelType::Gemma3));
    assert!(has_config_backed_registration(ModelType::Gemma4));
    assert!(has_config_backed_registration(ModelType::Ministral3));
    assert!(!has_config_backed_registration(ModelType::Mistral3));
}
