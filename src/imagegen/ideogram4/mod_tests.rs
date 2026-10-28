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

//! Snapshot validation and request checks of [`Ideogram4Model`]. The checks
//! that need the tokenizer use the real snapshot's (no weights are read) and
//! are skipped when `models/mlx/ideogram-4-fp8` is absent.

use serde_json::json;

use super::*;

/// `models/mlx/ideogram-4-fp8` under the crate root, if present.
pub(super) fn checkpoint() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/mlx/ideogram-4-fp8");
    if root.join("model_index.json").is_file() {
        Some(root)
    } else {
        eprintln!("skipping: {} is absent", root.display());
        None
    }
}

fn invalid(err: anyhow::Error) -> String {
    let message = format!("{err:#}");
    assert!(
        err.downcast_ref::<InvalidImageRequest>().is_some(),
        "not an InvalidImageRequest: {message}"
    );
    message
}

fn fake_snapshot(class_name: &str, skip: Option<&str>) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    for file in REQUIRED_FILES {
        if Some(*file) == skip {
            continue;
        }
        let path = tmp.path().join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{}").unwrap();
    }
    if skip != Some("model_index.json") {
        std::fs::write(
            tmp.path().join("model_index.json"),
            json!({ "_class_name": class_name }).to_string(),
        )
        .unwrap();
    }
    tmp
}

#[test]
fn the_layout_check_names_missing_files_and_the_wrong_pipeline() {
    let tmp = fake_snapshot("Ideogram4Pipeline", None);
    validate_layout(tmp.path()).unwrap();
    let tmp = fake_snapshot(
        "Ideogram4Pipeline",
        Some("unconditional_transformer/config.json"),
    );
    let err = validate_layout(tmp.path()).unwrap_err().to_string();
    assert!(
        err.contains("unconditional_transformer/config.json"),
        "{err}"
    );
    let tmp = fake_snapshot("Flux2Pipeline", None);
    let err = validate_layout(tmp.path()).unwrap_err().to_string();
    assert!(err.contains("not an Ideogram4Pipeline"), "{err}");
}

#[test]
fn the_registry_routes_the_snapshot_to_this_family() {
    let Some(root) = checkpoint() else { return };
    let model = crate::imagegen::load_image_model(&root).unwrap();
    assert_eq!(model.family(), FAMILY);
    assert!(model.supports(ImageTask::Generate));
    assert!(!model.supports(ImageTask::Edit));
    let defaults = model.defaults(ImageTask::Generate);
    assert_eq!((defaults.steps, defaults.guidance), (20, 7.0));
}

#[test]
fn a_plain_prompt_is_wrapped_and_its_caption_is_the_revised_prompt() {
    let Some(root) = checkpoint() else { return };
    let model = Ideogram4Model::load(&root).unwrap();
    let prepared = model
        .prepare(&ImageRequest::generate("a red fox", 0))
        .unwrap();
    assert!(prepared.prompt.was_wrapped);
    assert_eq!(
        prepared.prompt.revised_prompt().as_deref(),
        Some(prepared.prompt.text.as_str())
    );
    assert_eq!(prepared.plan.steps, 20);
    let mut req = ImageRequest::generate("a red fox", 0);
    req.extra.insert("auto_json_caption".into(), json!(false));
    let prepared = model.prepare(&req).unwrap();
    assert_eq!(prepared.prompt.text, "a red fox");
    assert_eq!(prepared.prompt.revised_prompt(), None);
}

#[test]
fn bad_requests_are_the_callers_errors() {
    let Some(root) = checkpoint() else { return };
    let model = Ideogram4Model::load(&root).unwrap();
    let check = |edit: &dyn Fn(&mut ImageRequest)| {
        let mut req = ImageRequest::generate("a red fox", 0);
        edit(&mut req);
        invalid(model.check_request(&req).unwrap_err())
    };
    assert!(check(&|r| r.prompt = "  ".into()).contains("empty"));
    assert!(check(&|r| r.width = 520).contains("divisible by 16"));
    assert!(check(&|r| r.width = 128).contains("256"));
    assert!(check(&|r| r.task = ImageTask::Edit).contains("text-to-image"));
    assert!(
        check(&|r| {
            r.extra.insert("sampler_preset".into(), json!("V4_FAST"));
        })
        .contains("V4_TURBO_12")
    );
    assert!(
        check(&|r| {
            r.extra
                .insert("prompt_expansion_model".into(), json!("gpt"));
        })
        .contains("prompt_expansion_model")
    );
    assert!(
        check(&|r| {
            r.steps = Some(3);
            r.extra
                .insert("guidance_schedule".into(), json!([1.0, 2.0]));
        })
        .contains("guidance_schedule")
    );
    assert!(check(&|r| r.negative_prompt = Some("blurry".into())).contains("negative prompt"));
    let long = "word ".repeat(4000);
    assert!(check(&|r| r.prompt = long.clone()).contains("2048"));
    // Over the byte cap, rejected before any JSON parse or tokenization.
    let huge = "{".repeat(pipeline::MAX_PROMPT_BYTES + 1);
    let message = check(&|r| r.prompt = huge.clone());
    assert!(message.contains("max_text_tokens=2048"), "{message}");
    invalid(model.generate(&ImageRequest::generate("", 0)).unwrap_err());
}
