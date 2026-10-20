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
use image::DynamicImage;

#[test]
fn sampling_defaults_reject_zero_steps_and_bad_guidance() {
    assert!(ImageSamplingDefaults::new(0, 1.0).is_err());
    assert!(ImageSamplingDefaults::new(4, -0.5).is_err());
    assert!(ImageSamplingDefaults::new(4, f32::NAN).is_err());
    assert!(ImageSamplingDefaults::new(4, f32::INFINITY).is_err());
    let ok = ImageSamplingDefaults::new(4, 0.0).unwrap();
    assert_eq!((ok.steps, ok.guidance), (4, 0.0));
}

#[test]
fn resolve_falls_back_to_defaults_only_when_unset() {
    let defaults = ImageSamplingDefaults::new(8, 3.5).unwrap();
    let mut req = ImageRequest::generate("a cat", 1);
    assert_eq!(req.resolve_steps(defaults), 8);
    assert_eq!(req.resolve_guidance(defaults), 3.5);
    req.steps = Some(2);
    req.guidance = Some(0.0);
    assert_eq!(req.resolve_steps(defaults), 2);
    assert_eq!(req.resolve_guidance(defaults), 0.0);
}

#[test]
fn image_i_of_n_uses_seed_plus_i() {
    let req = ImageRequest::generate("x", 40);
    assert_eq!(req.seed_for(0), 40);
    assert_eq!(req.seed_for(2), 42);
    let wrap = ImageRequest::generate("x", u64::MAX);
    assert_eq!(wrap.seed_for(1), 0);
}

#[test]
fn validate_accepts_the_default_request() {
    ImageRequest::generate("a cat", 1).validate(16).unwrap();
}

#[test]
fn validate_rejects_bad_fields() {
    let base = ImageRequest::generate("a cat", 1);

    let mut r = base.clone();
    r.prompt = "   ".into();
    assert!(r.validate(16).unwrap_err().to_string().contains("prompt"));

    for n in [0, 11] {
        let mut r = base.clone();
        r.n = n;
        assert!(
            r.validate(16)
                .unwrap_err()
                .to_string()
                .contains("between 1 and 10")
        );
    }

    let mut r = base.clone();
    r.width = 500;
    let msg = r.validate(16).unwrap_err().to_string();
    assert!(msg.contains("multiples of 16"), "{msg}");
    assert!(msg.contains("500x512"), "{msg}");

    let mut r = base.clone();
    r.height = 0;
    assert!(r.validate(16).is_err());

    let mut r = base.clone();
    r.steps = Some(0);
    assert!(r.validate(16).is_err());

    let mut r = base.clone();
    r.guidance = Some(-1.0);
    assert!(r.validate(16).is_err());
}

#[test]
fn grid_of_one_disables_the_multiple_check() {
    let mut r = ImageRequest::generate("a cat", 1);
    r.width = 513;
    r.validate(1).unwrap();
    assert!(r.validate(16).is_err());
}

#[test]
fn edit_needs_a_reference_image_and_generate_rejects_one() {
    let mut edit = ImageRequest::generate("make it blue", 1);
    edit.task = ImageTask::Edit;
    let msg = edit.validate(16).unwrap_err().to_string();
    assert!(
        msg.contains("--task edit requires at least one reference image"),
        "{msg}"
    );

    edit.reference_images.push(DynamicImage::new_rgb8(16, 16));
    edit.validate(16).unwrap();

    let mut generate = ImageRequest::generate("a cat", 1);
    generate
        .reference_images
        .push(DynamicImage::new_rgb8(16, 16));
    assert!(generate.validate(16).is_err());
}

#[test]
fn expected_len_counts_channels() {
    let img = GeneratedImage {
        pixels: vec![0; 2 * 3 * 4],
        width: 2,
        height: 3,
        channels: 4,
        seed: 0,
        revised_prompt: None,
    };
    assert_eq!(img.expected_len(), 24);
}
