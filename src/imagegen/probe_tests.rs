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
use serde_json::json;
use tempfile::TempDir;

fn write(dir: &std::path::Path, rel: &str, value: serde_json::Value) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, value.to_string()).unwrap();
}

fn weight_map(keys: &[&str]) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = keys
        .iter()
        .map(|k| ((*k).to_string(), json!("model-00001.safetensors")))
        .collect();
    json!({ "weight_map": map })
}

fn probe(dir: &TempDir) -> Option<&'static str> {
    detect_image_family(dir.path()).unwrap()
}

#[test]
fn flux2_from_the_pipeline_class_name() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "model_index.json",
        json!({"_class_name": "Flux2Pipeline"}),
    );
    assert_eq!(probe(&dir), Some("flux2"));
}

#[test]
fn flux2_from_the_transformer_weight_map_alone() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        weight_map(FLUX2_MARKERS),
    );
    assert_eq!(probe(&dir), Some("flux2"));
}

#[test]
fn flux2_markers_must_all_be_present() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "transformer/model.safetensors.index.json",
        weight_map(&FLUX2_MARKERS[..2]),
    );
    assert_eq!(probe(&dir), None);
}

#[test]
fn z_image_without_an_mllm_tower() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "transformer/model.safetensors.index.json",
        weight_map(Z_IMAGE_MARKERS),
    );
    assert_eq!(probe(&dir), Some("z_image"));
}

#[test]
fn ming_image_is_z_image_plus_mllm_config() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "transformer/model.safetensors.index.json",
        weight_map(Z_IMAGE_MARKERS),
    );
    write(
        dir.path(),
        "mllm/config.json",
        json!({"model_type": "bailingmm"}),
    );
    assert_eq!(probe(&dir), Some("ming_image"));
}

#[test]
fn ernie_image_markers() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "transformer/model.safetensors.index.json",
        weight_map(ERNIE_IMAGE_MARKERS),
    );
    assert_eq!(probe(&dir), Some("ernie_image"));
}

#[test]
fn qwen_image_markers() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "transformer/model.safetensors.index.json",
        weight_map(QWEN_IMAGE_MARKERS),
    );
    assert_eq!(probe(&dir), Some("qwen_image"));
}

#[test]
fn model_safetensors_index_wins_over_diffusion_pytorch_index() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "transformer/model.safetensors.index.json",
        weight_map(QWEN_IMAGE_MARKERS),
    );
    write(
        dir.path(),
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        weight_map(FLUX2_MARKERS),
    );
    assert_eq!(probe(&dir), Some("qwen_image"));
}

#[test]
fn bonsai_manifest_beats_the_flux2_class_name() {
    let dir = TempDir::new().unwrap();
    // Bonsai ships a Flux2Klein pipeline index; the class-name scan alone would
    // call it flux2.
    write(
        dir.path(),
        "model_index.json",
        json!({"_class_name": "Flux2KleinPipeline", "transformer": ["diffusers", "Flux2Transformer2DModel"]}),
    );
    write(
        dir.path(),
        "manifest.json",
        json!({"files": [
            {"remote_path": "transformer-packed-mflux/diffusion_pytorch_model.safetensors"},
            {"path": "text_encoder-mlx-4bit/model.safetensors"},
            "tokenizer/tokenizer.json",
        ]}),
    );
    assert_eq!(probe(&dir), Some("bonsai"));
}

#[test]
fn partial_bonsai_manifest_falls_back_to_the_class_name() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "model_index.json",
        json!({"_class_name": "Flux2KleinPipeline"}),
    );
    write(
        dir.path(),
        "manifest.json",
        json!({"files": ["tokenizer/tokenizer.json"]}),
    );
    assert_eq!(probe(&dir), Some("flux2"));
}

#[test]
fn component_class_names_are_scanned() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "model_index.json",
        json!({
            "_class_name": "DiffusionPipeline",
            "transformer": ["diffusers", "ErnieImageTransformer2DModel"],
        }),
    );
    assert_eq!(probe(&dir), Some("ernie_image"));
}

#[test]
fn component_dict_class_is_scanned() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "model_index.json",
        json!({"vae": {"_class_name": "Flux2VAE"}}),
    );
    assert_eq!(probe(&dir), Some("flux2"));
}

#[test]
fn nested_component_config_is_scanned() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "transformer/config.json",
        json!({"_class_name": "QwenImageTransformer2DModel"}),
    );
    assert_eq!(probe(&dir), Some("qwen_image"));
}

#[test]
fn a_text_checkpoint_is_not_an_image_family() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "config.json",
        json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
    );
    assert_eq!(probe(&dir), None);
}

#[test]
fn a_missing_path_and_an_empty_directory_are_none() {
    let dir = TempDir::new().unwrap();
    assert_eq!(probe(&dir), None);
    assert_eq!(
        detect_image_family(&dir.path().join("absent")).unwrap(),
        None
    );
}

#[test]
fn a_regular_file_is_none() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("model.safetensors");
    std::fs::write(&file, b"x").unwrap();
    assert_eq!(detect_image_family(&file).unwrap(), None);
}

#[test]
fn unparseable_json_is_skipped_not_fatal() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("model_index.json"), "{not json").unwrap();
    write(
        dir.path(),
        "config.json",
        json!({"_class_name": "Flux2Pipeline"}),
    );
    assert_eq!(probe(&dir), Some("flux2"));
}

#[test]
fn redirect_error_names_the_family_and_the_command() {
    let dir = TempDir::new().unwrap();
    write(
        dir.path(),
        "model_index.json",
        json!({"_class_name": "Flux2Pipeline"}),
    );
    let err = image_checkpoint_redirect_error(dir.path()).expect("flux2 is recognized");
    assert_eq!(
        err.to_string(),
        "image generation checkpoint (flux2): use mlxcel generate-image"
    );
    let empty = TempDir::new().unwrap();
    assert!(image_checkpoint_redirect_error(empty.path()).is_none());
}

#[test]
fn class_name_tokens_follow_the_upstream_regex() {
    assert_eq!(
        class_name_tokens("Flux2KleinPipeline"),
        ["Flux2", "Klein", "Pipeline"]
    );
    assert_eq!(
        class_name_tokens("LLaDAImagePipeline"),
        ["L", "La", "D", "A", "Image", "Pipeline"]
    );
    assert_eq!(class_name_tokens("flux2_pipeline"), Vec::<String>::new());
    assert_eq!(class_name_tokens("Flux2_pipeline"), ["Flux2"]);
    assert_eq!(class_name_tokens(""), Vec::<String>::new());
}

#[test]
fn mixed_case_acronyms_are_preserved_as_candidates() {
    let candidates = model_types_from_class_name("LLaDAImagePipeline");
    assert!(
        candidates.contains(&"llada_image".to_string()),
        "{candidates:?}"
    );
}

#[test]
fn normalize_matches_python_identifier_rules() {
    assert_eq!(normalize_model_type(" Flux-2 "), Some("flux_2".into()));
    assert_eq!(normalize_model_type("2flux"), None);
    assert_eq!(normalize_model_type("a b"), None);
    assert_eq!(normalize_model_type(""), None);
    assert_eq!(normalize_model_type("z_image"), Some("z_image".into()));
}

#[test]
fn every_family_name_is_a_normalized_identifier() {
    for family in IMAGE_FAMILIES {
        assert_eq!(normalize_model_type(family).as_deref(), Some(*family));
    }
}
