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

//! Request validation of [`QwenImageModel`] on a config-only checkpoint (no
//! weights load): every rejection is an [`InvalidImageRequest`] raised before
//! the conditioner or the denoiser loads.

use std::path::PathBuf;

use image::DynamicImage;
use serde_json::json;

use super::*;

/// A checkpoint tree with the configs `QwenImageModel::load` reads and the
/// real checkpoint's processor pixel bounds; `vae` is `vae/config.json`.
fn fake_checkpoint(vae: &str) -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("qwen-image");
    for dir in ["transformer", "vae", "text_encoder", "processor"] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
    }
    std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
    std::fs::write(root.join("vae/config.json"), vae).unwrap();
    std::fs::write(root.join("text_encoder/config.json"), "{}").unwrap();
    std::fs::write(root.join("processor/tokenizer.json"), "{}").unwrap();
    std::fs::write(
        root.join("processor/preprocessor_config.json"),
        r#"{"size": {"longest_edge": 16777216, "shortest_edge": 65536}}"#,
    )
    .unwrap();
    (tmp, root)
}

fn model() -> (tempfile::TempDir, QwenImageModel) {
    let (tmp, root) = fake_checkpoint("{}");
    let model = QwenImageModel::load(&root).unwrap();
    (tmp, model)
}

fn edit(references: Vec<DynamicImage>, size: (u32, u32)) -> ImageRequest {
    let mut req = ImageRequest::generate("make the teapot blue", 7);
    req.task = ImageTask::Edit;
    req.reference_images = references;
    (req.width, req.height) = size;
    req
}

fn assert_invalid(model: &QwenImageModel, req: &ImageRequest, needle: &str) {
    let err = model.check_request(req).expect_err("rejected");
    assert!(
        err.downcast_ref::<InvalidImageRequest>().is_some(),
        "not an InvalidImageRequest: {err:#}"
    );
    let message = format!("{err:#}");
    assert!(message.contains(needle), "{message}");
}

#[test]
fn a_one_reference_edit_passes_validation() {
    let (_tmp, model) = model();
    let req = edit(vec![DynamicImage::new_rgba8(1024, 1024)], (1024, 1024));
    let resolved = model.resolve(&req).unwrap();
    assert_eq!((resolved.width, resolved.height), (1024, 1024));
}

#[test]
fn edit_prompts_with_vision_markers_are_rejected_before_any_load() {
    let (_tmp, model) = model();
    let mut req = edit(vec![DynamicImage::new_rgba8(1024, 1024)], (1024, 1024));
    req.prompt = "make <|image_pad|> blue".into();
    assert_invalid(&model, &req, "vision marker");
    // The CFG negative goes through the edit template too.
    let mut req = edit(vec![DynamicImage::new_rgba8(1024, 1024)], (1024, 1024));
    req.guidance = Some(4.0);
    req.negative_prompt = Some("<|vision_start|>".into());
    assert_invalid(&model, &req, "vision marker");
    // Without CFG the negative is never encoded.
    req.guidance = Some(1.0);
    model.check_request(&req).unwrap();
    assert!(model.text_encoder.borrow().is_none());
    assert!(model.denoiser.borrow().is_none());
}

#[test]
fn references_past_the_processor_aspect_limit_are_rejected_in_validation() {
    let (_tmp, model) = model();
    // 4000x10 resizes to 20480x64, past Qwen3-VL's 200:1 limit.
    let req = edit(vec![DynamicImage::new_rgba8(4000, 10)], (1024, 1024));
    assert_invalid(&model, &req, "aspect ratio");
}

#[test]
fn edit_token_limit_admits_one_full_size_reference_only() {
    let (_tmp, model) = model();
    let full = || DynamicImage::new_rgba8(2048, 2048);
    let mut one = edit(vec![full()], (2048, 2048));
    one.extra = json!({"output_resolution": 2048})
        .as_object()
        .cloned()
        .unwrap();
    model.check_request(&one).unwrap();
    let mut two = one.clone();
    two.reference_images.push(full());
    assert_invalid(&model, &two, "image tokens");
}

#[test]
fn a_vae_with_another_spatial_scale_fails_at_load() {
    let (_tmp, root) =
        fake_checkpoint(r#"{"dim_mult": [1, 2, 4], "temperal_downsample": [false, true]}"#);
    let err = QwenImageModel::load(&root).err().expect("mismatch");
    assert!(format!("{err:#}").contains("spatial scale"), "{err:#}");
}
