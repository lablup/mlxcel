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
use crate::imagegen::pipeline::ImageTask;
use crate::imagegen::testing::StubModel;
use image::GenericImageView;
use tempfile::TempDir;

fn rgb_stub() -> StubModel {
    StubModel {
        family: "flux2",
        channels: 3,
        edit: true,
    }
}

fn rgba_stub() -> StubModel {
    StubModel {
        family: "qwen_image",
        channels: 4,
        edit: false,
    }
}

fn small(prompt: &str, seed: u64, n: u32) -> ImageRequest {
    let mut req = ImageRequest::generate(prompt, seed);
    req.width = 32;
    req.height = 16;
    req.n = n;
    req
}

#[test]
fn parse_size_accepts_both_separators_and_any_case() {
    assert_eq!(parse_size("512x768").unwrap(), (512, 768));
    assert_eq!(parse_size("512X768").unwrap(), (512, 768));
    assert_eq!(parse_size("512×768").unwrap(), (512, 768));
    assert_eq!(parse_size(" 64 x 32 ").unwrap(), (64, 32));
}

#[test]
fn parse_size_rejects_malformed_values() {
    for bad in ["", "512", "x512", "512x", "axb", "512x-1", "512x768x3"] {
        let msg = parse_size(bad).unwrap_err().to_string();
        assert!(msg.contains("WIDTHxHEIGHT"), "{bad}: {msg}");
    }
}

#[test]
fn unsupported_task_names_the_family() {
    let mut req = small("edit me", 1, 1);
    req.task = ImageTask::Edit;
    req.reference_images
        .push(image::DynamicImage::new_rgb8(32, 16));
    let err = validate_request(&rgba_stub(), &req)
        .unwrap_err()
        .to_string();
    assert!(err.contains("qwen_image") && err.contains("edit"), "{err}");
    validate_request(&rgb_stub(), &req).unwrap();
}

#[test]
fn off_grid_size_names_the_multiple() {
    let mut req = small("a cat", 1, 1);
    req.width = 30;
    let err = validate_request(&rgb_stub(), &req).unwrap_err().to_string();
    assert!(err.contains("multiples of 16"), "{err}");
}

#[test]
fn validate_returns_the_models_defaults() {
    let defaults = validate_request(&rgb_stub(), &small("a cat", 1, 1)).unwrap();
    assert_eq!((defaults.steps, defaults.guidance), (4, 1.0));
}

#[test]
fn generate_to_files_writes_one_decodable_png() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("nested").join("cat.png");
    prepare_output_dir(&out).unwrap();
    let paths = generate_to_files(&rgb_stub(), &small("a cat", 5, 1), &out).unwrap();
    assert_eq!(paths, vec![out.clone()]);
    let decoded = image::open(&out).unwrap();
    assert_eq!(decoded.dimensions(), (32, 16));
    assert_eq!(decoded.color(), image::ColorType::Rgb8);
}

#[test]
fn n_two_writes_indexed_files_with_distinct_seeds() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("cat.png");
    let paths = generate_to_files(&rgb_stub(), &small("a cat", 10, 2), &out).unwrap();
    assert_eq!(
        paths,
        vec![dir.path().join("cat-01.png"), dir.path().join("cat-02.png")]
    );
    assert!(!out.exists(), "the unnumbered name must not be written");
    let a = image::open(&paths[0]).unwrap().to_rgb8().into_raw();
    let b = image::open(&paths[1]).unwrap().to_rgb8().into_raw();
    assert_ne!(a, b, "image i uses seed + i");
}

#[test]
fn the_same_seed_reproduces_the_same_pixels() {
    let a = run_request(&rgb_stub(), &small("a cat", 3, 1)).unwrap();
    let b = run_request(&rgb_stub(), &small("a cat", 3, 1)).unwrap();
    assert_eq!(a, b);
    let c = run_request(&rgb_stub(), &small("a cat", 4, 1)).unwrap();
    assert_ne!(a[0].pixels, c[0].pixels);
}

#[test]
fn rgba_alpha_survives_to_disk() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("rgba.png");
    generate_to_files(&rgba_stub(), &small("a logo", 1, 1), &out).unwrap();
    let decoded = image::open(&out).unwrap();
    assert_eq!(decoded.color(), image::ColorType::Rgba8);
    assert!(decoded.to_rgba8().pixels().all(|p| p.0[3] == 128));
}

#[test]
fn output_extension_is_forced_to_png() {
    let dir = TempDir::new().unwrap();
    let paths = generate_to_files(
        &rgb_stub(),
        &small("a cat", 1, 1),
        &dir.path().join("x.jpg"),
    )
    .unwrap();
    assert_eq!(paths, vec![dir.path().join("x.png")]);
}

struct WrongCount;

impl ImageGenerationModel for WrongCount {
    fn family(&self) -> &'static str {
        "flux2"
    }
    fn supports(&self, _: ImageTask) -> bool {
        true
    }
    fn defaults(&self, _: ImageTask) -> ImageSamplingDefaults {
        ImageSamplingDefaults::new(1, 0.0).unwrap()
    }
    fn generate(&self, _: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        Ok(Vec::new())
    }
}

struct Truncated;

impl ImageGenerationModel for Truncated {
    fn family(&self) -> &'static str {
        "flux2"
    }
    fn supports(&self, _: ImageTask) -> bool {
        true
    }
    fn defaults(&self, _: ImageTask) -> ImageSamplingDefaults {
        ImageSamplingDefaults::new(1, 0.0).unwrap()
    }
    fn generate(&self, _: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        Ok(vec![GeneratedImage {
            pixels: vec![0; 5],
            width: 32,
            height: 16,
            channels: 3,
            seed: 0,
            revised_prompt: None,
        }])
    }
}

#[test]
fn a_model_returning_the_wrong_number_of_images_is_an_error() {
    let err = run_request(&WrongCount, &small("a cat", 1, 2))
        .unwrap_err()
        .to_string();
    assert!(err.contains("returned 0 image(s) for n=2"), "{err}");
}

#[test]
fn a_malformed_image_is_caught_before_it_reaches_a_client() {
    let err = run_request(&Truncated, &small("a cat", 1, 1))
        .unwrap_err()
        .to_string();
    assert!(err.contains("malformed image"), "{err}");
}

#[test]
fn an_unwritable_output_directory_fails_without_a_model() {
    let dir = TempDir::new().unwrap();
    // A regular file where the directory should be.
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"x").unwrap();
    let err = prepare_output_dir(&blocker.join("out.png")).unwrap_err();
    assert!(format!("{err:#}").contains("output directory"), "{err:#}");
}

#[test]
fn a_bare_filename_needs_no_directory() {
    prepare_output_dir(std::path::Path::new("out.png")).unwrap();
}

#[test]
fn random_seeds_fit_in_32_bits_and_vary() {
    let seeds: Vec<u64> = (0..64).map(|_| random_seed()).collect();
    assert!(seeds.iter().all(|s| *s <= u64::from(u32::MAX)));
    let distinct: std::collections::BTreeSet<_> = seeds.iter().collect();
    assert!(
        distinct.len() > 32,
        "seeds should rarely repeat: {distinct:?}"
    );
}
