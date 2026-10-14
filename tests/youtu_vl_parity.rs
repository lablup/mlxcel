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

//! Ignored real-model Youtu-VL content tests, in the shape of
//! `tests/qwen2_5_vl_parity.rs`.
//!
//! Every other VLM test in this repository drives `tests/fixtures/test_image.png`,
//! a 224x224 solid orange square, and asserts only that the run produced tokens.
//! That cannot separate a correct description from a fluent invention, which is
//! how issue #1610 survived a full unit suite: the processor emitted patches in
//! raster order with channel-first features while both the loaded
//! `patch_embedding` weight and the encoder's own rotary/merge grouping expect
//! merge-block-major rows with channel-last features, and the model answered
//! anyway.
//!
//! `models/mlx/youtu-vl-4b-instruct` has `patch_size=16`, `spatial_merge_size=2`
//! and `window_size=256`, so `smart_resize` rounds every edge up to a multiple
//! of 32 (as `AutoProcessor` does, #1611) and the merged grid is one eighth of
//! the resized edge. 224 gives a 7x7 merged grid, a single 8x8 attention window;
//! 336 resizes to 352 for an 11x11 merged grid and 448 gives 14x14, both of
//! which are 2x2 windows.
//!
//! The solid orange fixture is also the one image that cannot catch a defect
//! in how features reach their prompt slots: every merged token of a uniform
//! image carries the same content, so a run that places only one feature
//! (#1618, where the template's single `<|image_pad|>` was never expanded)
//! still reads as orange. The two three-shape fixtures are what catch that.
//!
//! To run them:
//! ```text
//! cargo test --profile test-fast --features metal,accelerate --test youtu_vl_parity -- --ignored
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{extract_generated_body, repo_binary_path, repo_model_dir};

const MODEL: &str = "youtu-vl-4b-instruct";
const SHAPES_PROMPT: &str = "What shapes and colors are in this image? Answer briefly.";

/// The checkpoint directory, or `None` when it is not present on this machine.
///
/// `models/<name>` is the layout `tests/vlm_concurrency.rs` uses; `models/mlx/<name>`
/// is where `mlxcel download` puts a conversion.
fn checkpoint_dir() -> Option<PathBuf> {
    for candidate in [
        repo_model_dir(MODEL),
        repo_model_dir(&format!("mlx/{MODEL}")),
    ] {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    eprintln!(
        "Skipping test: no {MODEL} checkpoint under models/{MODEL} or models/mlx/{MODEL}. \
         Fetch it with `mlxcel download tencent/Youtu-VL-4B-Instruct`."
    );
    None
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Run the CLI exactly as a user would, so the loader, processor, vision tower,
/// merge and decode path under test are the shipped ones and not a test helper.
fn describe(model_dir: &Path, image: &Path, prompt: &str) -> String {
    let output = Command::new(repo_binary_path("mlxcel"))
        .args([
            "generate",
            "-m",
            model_dir.to_str().expect("model dir is utf-8"),
            "--image",
            image.to_str().expect("image path is utf-8"),
            "-p",
            prompt,
            "-n",
            "48",
            "-t",
            "0",
        ])
        .output()
        .expect("run mlxcel generate");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "mlxcel generate failed for {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        image.display()
    );
    let body = extract_generated_body(&stdout)
        .unwrap_or_else(|| panic!("no generated body in output\nstdout:\n{stdout}"))
        .to_string();
    eprintln!("{} -> {body}", image.display());
    body
}

/// Assert the description mentions every group, where a group is a set of
/// interchangeable words. "square" and "rectangle" are both correct names for
/// the red shape, and the model picks between them depending on the rendered
/// size, so the assertion must not pin one.
fn assert_mentions(body: &str, groups: &[&[&str]]) {
    let lowered = body.to_lowercase();
    let missing: Vec<String> = groups
        .iter()
        .filter(|group| !group.iter().any(|word| lowered.contains(word)))
        .map(|group| group.join("/"))
        .collect();
    assert!(
        missing.is_empty(),
        "description omitted {}\nfull description: {body}",
        missing.join(", ")
    );
}

/// The solid-color control, and the test that pins issue #1610.
///
/// At 224x224 the merged grid is 7x7, one 8x8 window. Before
/// the patch-order fix the model answered "The image is completely black and
/// contains no visible content, objects, text, or details. It is a solid black
/// square or rectangle with no variation in color or texture." for a solid
/// orange square: the language model had no grounded content and described an
/// empty frame. A test that only counts tokens passes on that output.
#[test]
#[ignore = "requires the local youtu-vl-4b-instruct checkpoint"]
fn youtu_vl_describes_the_solid_color_fixture_as_a_solid_color() {
    let Some(model_dir) = checkpoint_dir() else {
        return;
    };
    let body = describe(
        &model_dir,
        &fixture("test_image.png"),
        "Describe this image briefly.",
    );
    assert_mentions(&body, &[&["orange"]]);
    for invented in ["black", "person", "man", "woman", "shirt", "wall", "floor"] {
        assert!(
            !body.to_lowercase().contains(invented),
            "solid orange fixture described with content that is not in it ({invented})\nfull description: {body}"
        );
    }
}

/// Every object and every color at 448x448, a 14x14 merged grid.
///
/// Pins issue #1618. Before it, the chat template's single `<|image_pad|>` was
/// never expanded to one token per merged feature, so only the top-left feature
/// reached the language model and the answer was "The image contains a single,
/// solid black circle on a plain white background." The fixture is a red
/// square, a blue circle and a green triangle on light grey.
#[test]
#[ignore = "requires the local youtu-vl-4b-instruct checkpoint"]
fn youtu_vl_names_every_object_in_the_three_shape_image() {
    let Some(model_dir) = checkpoint_dir() else {
        return;
    };
    let body = describe(&model_dir, &fixture("test_image_shapes.png"), SHAPES_PROMPT);
    assert_mentions(
        &body,
        &[
            &["square", "rectangle"],
            &["circle"],
            &["triangle"],
            &["red"],
            &["blue"],
            &["green"],
        ],
    );
}

/// The same scene at 336x336, which `smart_resize` lifts to 352 for an 11x11
/// merged grid.
///
/// Pins issue #1618 for the same reason as the 448 case. Before it, the answer
/// was "The image contains a single white circle on a black background."
#[test]
#[ignore = "requires the local youtu-vl-4b-instruct checkpoint"]
fn youtu_vl_names_every_object_at_a_multi_window_grid() {
    let Some(model_dir) = checkpoint_dir() else {
        return;
    };
    let body = describe(
        &model_dir,
        &fixture("test_image_shapes_336.png"),
        SHAPES_PROMPT,
    );
    assert_mentions(
        &body,
        &[
            &["square", "rectangle"],
            &["circle"],
            &["triangle"],
            &["red"],
            &["blue"],
            &["green"],
        ],
    );
}
