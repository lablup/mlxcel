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

use std::path::Path;

use serde_json::json;

use super::*;
use crate::imagegen::registry::resolve_family;

fn write(dir: &Path, rel: &str, value: serde_json::Value) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

/// What `prism-ml/bonsai-image-ternary-4B-mlx-2bit` ships before the weights.
fn bonsai_dir(dir: &Path) {
    write(
        dir,
        "model_index.json",
        json!({"_class_name": "Flux2KleinPipeline"}),
    );
    write(
        dir,
        "manifest.json",
        json!({"files": [
            {"remote_path": "transformer-packed-mflux/diffusion_pytorch_model.safetensors"},
            {"remote_path": "text_encoder-mlx-4bit/model.safetensors"},
            {"remote_path": "tokenizer/tokenizer.json"},
        ]}),
    );
}

#[test]
fn bonsai_is_registered_in_both_tables() {
    assert!(crate::imagegen::FAMILIES.iter().any(|f| f.name == FAMILY));
    assert!(crate::imagegen::IMAGE_FAMILIES.contains(&FAMILY));
}

#[test]
fn a_bonsai_checkpoint_resolves_to_the_bonsai_loader() {
    let tmp = tempfile::tempdir().unwrap();
    bonsai_dir(tmp.path());
    let family = resolve_family(tmp.path(), crate::imagegen::FAMILIES).unwrap();
    assert_eq!(family.name, FAMILY);
    // The row's loader is Bonsai's: it names Bonsai's own directory, where
    // FLUX.2's would report a missing `transformer/`.
    let err = (family.load)(tmp.path()).err().unwrap().to_string();
    assert!(err.contains("transformer-packed-mflux"), "{err}");
}

#[test]
fn defaults_are_upstreams_four_steps_without_guidance() {
    assert_eq!(DEFAULTS.steps, 4);
    assert!((DEFAULTS.guidance - 1.0).abs() < f32::EPSILON);
}
