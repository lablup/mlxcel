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

fn rejection(req: &ImageRequest) -> String {
    let err = resolve_request(req).expect_err("request is rejected");
    err.downcast_ref::<InvalidImageRequest>()
        .expect("a content error is InvalidImageRequest (400 on the server)")
        .0
        .clone()
}

fn poster() -> ImageRequest {
    let mut req = ImageRequest::generate("a minimalist poster, bold word HELLO", 0);
    req.width = 1024;
    req.height = 1024;
    req
}

#[test]
fn defaults_are_twelve_steps_without_cfg() {
    let r = resolve_request(&poster()).unwrap();
    assert_eq!((r.width, r.height, r.steps), (1024, 1024, 12));
}

#[test]
fn any_guidance_but_one_is_a_400() {
    let mut req = poster();
    req.guidance = Some(3.0);
    let msg = rejection(&req);
    assert!(msg.contains("guidance 1.0"), "{msg}");
    req.guidance = Some(1.0);
    assert!(resolve_request(&req).is_ok());
}

#[test]
fn an_edit_is_a_400_naming_the_task() {
    let mut req = poster();
    req.task = ImageTask::Edit;
    req.reference_images = vec![image::DynamicImage::new_rgba8(16, 16)];
    let msg = rejection(&req);
    assert!(msg.contains("image edits"), "{msg}");
}

#[test]
fn sizes_outside_the_trained_range_are_a_400() {
    for (w, h) in [(128, 1024), (1024, 4096), (1000, 1024)] {
        let mut req = poster();
        req.width = w;
        req.height = h;
        rejection(&req);
    }
    let mut req = poster();
    req.steps = Some(0);
    assert!(rejection(&req).contains("steps"));
}

#[test]
fn the_registry_loads_the_family_by_its_probe_name() {
    assert!(
        crate::imagegen::registry::FAMILIES
            .iter()
            .any(|f| f.name == FAMILY)
    );
    assert!(crate::imagegen::probe::IMAGE_FAMILIES.contains(&FAMILY));
}
