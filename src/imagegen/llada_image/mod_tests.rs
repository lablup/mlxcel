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

#[test]
fn the_family_is_registered_and_probed() {
    assert!(
        crate::imagegen::registry::FAMILIES
            .iter()
            .any(|f| f.name == FAMILY)
    );
    assert!(crate::imagegen::probe::IMAGE_FAMILIES.contains(&FAMILY));
}

#[test]
fn loading_a_foreign_checkpoint_fails_by_name() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("model_index.json"),
        r#"{"_class_name": "Flux2Pipeline"}"#,
    )
    .unwrap();
    let err = load(dir.path())
        .err()
        .expect("not a LLaDA-Image checkpoint");
    assert!(
        format!("{err:#}").contains("Not a LLaDA-Image checkpoint"),
        "{err:#}"
    );
}

/// A VQ prompt past the prior's bound is the caller's fault (`400`), found
/// before any component loads: only configs and the tokenizer are read.
#[test]
#[ignore = "needs models/mlx/llada-image-turbo (configs and tokenizer only)"]
fn an_overlong_vq_prompt_is_an_invalid_request() {
    let root = std::env::var_os("MLXCEL_LLADA_IMAGE_TURBO")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models/mlx/llada-image-turbo")
        });
    let model = LladaImageModel::load(&root).expect("checkpoint configs and tokenizer");
    let mut req = ImageRequest::generate("a red panda ".repeat(4000), 0);
    req.extra
        .insert("generation_mode".into(), serde_json::Value::from("vq"));
    let err = model.resolve(&req).unwrap_err();
    assert!(
        err.downcast_ref::<InvalidImageRequest>().is_some(),
        "{err:#}"
    );
    assert!(format!("{err:#}").contains("VQ prompt"), "{err:#}");
    // The same prompt in text mode is truncated, as upstream does.
    req.extra.remove("generation_mode");
    assert!(model.resolve(&req).is_ok());
    assert!(model.text.borrow().is_none() && model.transformer.borrow().is_none());
}
