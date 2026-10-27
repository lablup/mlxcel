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
use std::path::PathBuf;

fn args(prompt: &str) -> GenerateImageArgs {
    GenerateImageArgs {
        model: PathBuf::from("models/none"),
        prompt: prompt.to_string(),
        negative_prompt: None,
        task: TaskArg::Generate,
        image: Vec::new(),
        size: None,
        steps: None,
        guidance: None,
        seed: None,
        n: 1,
        output: None,
        gen_kwargs: None,
    }
}

#[test]
fn gen_kwargs_must_be_a_json_object() {
    assert!(parse_gen_kwargs(None).unwrap().is_empty());
    let map = parse_gen_kwargs(Some(r#"{"cfg_norm": true, "shift": 3}"#)).unwrap();
    assert_eq!(map["shift"], 3);
    assert!(
        parse_gen_kwargs(Some("[1]"))
            .unwrap_err()
            .to_string()
            .contains("an array")
    );
    assert!(
        parse_gen_kwargs(Some("7"))
            .unwrap_err()
            .to_string()
            .contains("a number")
    );
    assert!(
        parse_gen_kwargs(Some("{oops"))
            .unwrap_err()
            .to_string()
            .contains("valid JSON")
    );
}

#[test]
fn generate_defaults_to_512x512() {
    let req = build_request(&args("a cat"), 9).unwrap();
    assert_eq!((req.width, req.height), (512, 512));
    assert_eq!(req.seed, 9);
    assert_eq!(req.task, ImageTask::Generate);
    assert_eq!(req.n, 1);
}

#[test]
fn explicit_size_is_parsed() {
    let mut a = args("a cat");
    a.size = Some("768x1024".into());
    let req = build_request(&a, 1).unwrap();
    assert_eq!((req.width, req.height), (768, 1024));
    a.size = Some("big".into());
    assert!(
        build_request(&a, 1)
            .unwrap_err()
            .to_string()
            .contains("WIDTHxHEIGHT")
    );
}

#[test]
fn edit_defaults_to_the_first_reference_size() {
    let dir = tempfile::TempDir::new().unwrap();
    let first = dir.path().join("a.png");
    let second = dir.path().join("b.png");
    image::DynamicImage::new_rgb8(48, 32).save(&first).unwrap();
    image::DynamicImage::new_rgb8(16, 16).save(&second).unwrap();

    let mut a = args("make it blue");
    a.task = TaskArg::Edit;
    a.image = vec![first, second];
    let req = build_request(&a, 1).unwrap();
    assert_eq!((req.width, req.height), (48, 32));
    assert_eq!(req.reference_images.len(), 2);
    // An explicit size still wins.
    a.size = Some("64x64".into());
    let req = build_request(&a, 1).unwrap();
    assert_eq!((req.width, req.height), (64, 64));
}

#[test]
fn a_missing_reference_image_is_named() {
    let mut a = args("x");
    a.task = TaskArg::Edit;
    a.image = vec![PathBuf::from("/definitely/missing.png")];
    let err = build_request(&a, 1).unwrap_err().to_string();
    assert!(err.contains("Image file does not exist"), "{err}");
}

#[test]
fn edit_without_images_and_generate_with_images_fail_validation() {
    let mut a = args("x");
    a.task = TaskArg::Edit;
    let req = build_request(&a, 1).unwrap();
    assert!(
        req.validate(1)
            .unwrap_err()
            .to_string()
            .contains("requires at least one")
    );
}

#[test]
fn run_rejects_a_missing_model_directory_before_anything_else() {
    let err = run_generate_image(args("a cat")).unwrap_err().to_string();
    assert!(err.contains("Model directory does not exist"), "{err}");
}

#[test]
fn run_rejects_bad_n_before_loading_weights() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut a = args("a cat");
    a.model = dir.path().to_path_buf();
    a.n = 11;
    let err = run_generate_image(a).unwrap_err().to_string();
    assert!(err.contains("-n must be between 1 and 10"), "{err}");
}

#[test]
fn run_on_a_directory_with_no_family_exits_with_the_family_message() {
    let model = tempfile::TempDir::new().unwrap();
    let out = tempfile::TempDir::new().unwrap();
    let mut a = args("a cat");
    a.model = model.path().to_path_buf();
    a.output = Some(out.path().join("cat.png"));
    let err = run_generate_image(a).unwrap_err().to_string();
    assert!(
        err.contains("no image generation family recognized"),
        "{err}"
    );
}

#[test]
fn run_on_an_unported_family_names_it() {
    let model = tempfile::TempDir::new().unwrap();
    std::fs::write(
        model.path().join("model_index.json"),
        r#"{"_class_name": "Flux2Pipeline"}"#,
    )
    .unwrap();
    let out = tempfile::TempDir::new().unwrap();
    let mut a = args("a cat");
    a.model = model.path().to_path_buf();
    a.output = Some(out.path().join("cat.png"));
    // A table without a flux2 row, as a build that has not ported it.
    let err = run_generate_image_with(a, &[]).unwrap_err().to_string();
    assert!(
        err.contains("image generation family flux2 is not supported by this build"),
        "{err}"
    );
}

#[test]
fn an_unwritable_output_fails_before_the_model_is_touched() {
    let model = tempfile::TempDir::new().unwrap();
    let scratch = tempfile::TempDir::new().unwrap();
    let blocker = scratch.path().join("blocker");
    std::fs::write(&blocker, b"x").unwrap();
    let mut a = args("a cat");
    a.model = model.path().to_path_buf();
    a.output = Some(blocker.join("cat.png"));
    let err = format!("{:#}", run_generate_image(a).unwrap_err());
    assert!(err.contains("output directory"), "{err}");
    assert!(
        !err.contains("family"),
        "the model must not have been probed: {err}"
    );
}

// A stub family driven through the whole command.
use mlxcel::imagegen::{GeneratedImage, ImageFamily, ImageGenerationModel, ImageSamplingDefaults};

struct Stub {
    channels: u8,
}

impl ImageGenerationModel for Stub {
    fn family(&self) -> &'static str {
        "flux2"
    }
    fn supports(&self, _: ImageTask) -> bool {
        true
    }
    fn defaults(&self, _: ImageTask) -> ImageSamplingDefaults {
        ImageSamplingDefaults::new(1, 0.0).unwrap()
    }
    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        Ok((0..req.n)
            .map(|i| GeneratedImage {
                pixels: (0..req.width * req.height * u32::from(self.channels))
                    .map(|p| {
                        if self.channels == 4 && p % 4 == 3 {
                            100
                        } else {
                            (p as u64 + req.seed_for(i)) as u8
                        }
                    })
                    .collect(),
                width: req.width,
                height: req.height,
                channels: self.channels,
                seed: req.seed_for(i),
                revised_prompt: None,
            })
            .collect())
    }
}

fn load_rgb(_: &std::path::Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(Stub { channels: 3 }))
}
fn load_rgba(_: &std::path::Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(Stub { channels: 4 }))
}

fn stub_checkpoint() -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("model_index.json"),
        r#"{"_class_name": "Flux2Pipeline"}"#,
    )
    .unwrap();
    dir
}

#[test]
fn the_command_writes_a_decodable_png() {
    let model = stub_checkpoint();
    let out = tempfile::TempDir::new().unwrap();
    let table = [ImageFamily {
        name: "flux2",
        load: load_rgb,
    }];
    let mut a = args("a cat");
    a.model = model.path().to_path_buf();
    a.size = Some("32x16".into());
    a.seed = Some(5);
    a.output = Some(out.path().join("sub/cat.png"));
    run_generate_image_with(a, &table).unwrap();
    let png = image::open(out.path().join("sub/cat.png")).unwrap();
    assert_eq!((png.width(), png.height()), (32, 16));
}

#[test]
fn n_two_writes_numbered_files_and_rgba_keeps_alpha() {
    let model = stub_checkpoint();
    let out = tempfile::TempDir::new().unwrap();
    let table = [ImageFamily {
        name: "flux2",
        load: load_rgba,
    }];
    let mut a = args("a logo");
    a.model = model.path().to_path_buf();
    a.size = Some("16x16".into());
    a.n = 2;
    a.output = Some(out.path().join("logo.png"));
    run_generate_image_with(a, &table).unwrap();
    for name in ["logo-01.png", "logo-02.png"] {
        let png = image::open(out.path().join(name)).unwrap();
        assert_eq!(png.color(), image::ColorType::Rgba8);
        assert!(png.to_rgba8().pixels().all(|p| p.0[3] == 100), "{name}");
    }
    assert!(!out.path().join("logo.png").exists());
}

#[test]
fn the_command_rejects_an_off_grid_size_after_loading_the_stub() {
    let model = stub_checkpoint();
    let table = [ImageFamily {
        name: "flux2",
        load: load_rgb,
    }];
    let mut a = args("a cat");
    a.model = model.path().to_path_buf();
    a.size = Some("30x16".into());
    a.output = Some(model.path().join("o.png"));
    let err = run_generate_image_with(a, &table).unwrap_err().to_string();
    assert!(err.contains("multiples of 16"), "{err}");
}
