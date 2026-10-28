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

use image::DynamicImage;
use serde_json::json;

use super::*;

fn request(task: ImageTask, images: usize) -> ImageRequest {
    let mut req = ImageRequest::generate("a fox", 1);
    req.task = task;
    req.width = 64;
    req.height = 64;
    req.reference_images = vec![DynamicImage::new_rgb8(64, 64); images];
    req
}

fn rejection(has_enhancer: bool, req: &ImageRequest) -> String {
    let err = check_request(has_enhancer, req).unwrap_err();
    let invalid = err
        .downcast_ref::<InvalidImageRequest>()
        .expect("a request error is an InvalidImageRequest (HTTP 400)");
    invalid.0.clone()
}

#[test]
fn edits_need_exactly_one_source_image() {
    check_request(false, &request(ImageTask::Edit, 1)).unwrap();
    for n in [0, 2] {
        let msg = rejection(false, &request(ImageTask::Edit, n));
        assert!(msg.contains("exactly one source image"), "{n}: {msg}");
    }
}

#[test]
fn image_strength_outside_the_unit_interval_is_a_400() {
    for bad in [0.0, 1.5, -1.0] {
        let mut req = request(ImageTask::Edit, 1);
        req.extra.insert("image_strength".into(), json!(bad));
        assert!(rejection(false, &req).contains("image_strength"), "{bad}");
    }
}

#[test]
fn sizes_off_the_16_grid_are_a_400() {
    let mut req = request(ImageTask::Generate, 0);
    req.width = 1000;
    assert!(rejection(false, &req).contains("multiple of 16"));
}

#[test]
fn requesting_the_enhancer_without_pe_is_a_400_with_upstreams_message() {
    let mut req = request(ImageTask::Generate, 0);
    req.extra.insert("use_prompt_enhancer".into(), json!(true));
    assert_eq!(rejection(false, &req), prompt_enhancer::MISSING_ENHANCER);
    check_request(true, &req).unwrap();
}
