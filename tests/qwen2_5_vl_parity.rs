// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Ignored real-model Qwen2.5-VL content tests.
//!
//! Every other VLM test in this repository drives `tests/fixtures/test_image.png`,
//! a 224x224 solid orange square, and asserts only that the run produced tokens.
//! That cannot separate a correct description from a fluent invention, which is
//! how issue #1596 stayed open: the vision tower's float16 RMSNorm reduction was
//! overflowing and erasing whole token rows, and the model answered anyway.
//!
//! These tests assert on content. The three-shape fixture carries three
//! unambiguous objects in three unambiguous colors, so a description that drops
//! one fails. The 336x336 rendering is the size whose window permutation is not
//! an involution, so it is the one that separates a correct un-reorder of the
//! merged tokens from one that returns the permutation it was handed.
//!
//! To run them:
//! ```text
//! cargo test --profile test-fast --features metal,accelerate --test qwen2_5_vl_parity -- --ignored
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{extract_generated_body, repo_binary_path, repo_model_dir};

const MODEL: &str = "qwen2.5-vl-3b-4bit";
const SHAPES_PROMPT: &str = "What shapes and colors are in this image? Answer briefly.";

/// The checkpoint directory, or `None` when it is not present on this machine.
///
/// `models/<name>` is the layout `tests/vlm_concurrency.rs` uses; `models/mlx/<name>`
/// is where `mlxcel download` puts a `mlx-community` conversion.
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
         Fetch it with `mlxcel download mlx-community/Qwen2.5-VL-3B-Instruct-4bit`."
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

/// Every object and every color, on the size issue #1596 reported.
///
/// Before the fix this printed "The image contains a red square and a green
/// triangle." on both the 4-bit conversion and the raw bf16 export, while
/// transformers answered "square, circle, triangle, red, blue, green".
#[test]
#[ignore = "requires the local qwen2.5-vl-3b-4bit checkpoint"]
fn qwen2_5_vl_names_every_object_in_the_three_shape_image() {
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

/// The same scene at 336x336, whose 12x12 merged grid gives a window
/// permutation that is not its own inverse.
///
/// Before the fix this printed "A red circle and a blue triangle.": the colors
/// were attached to the wrong shapes because 120 of the 144 merged tokens came
/// back from the tower in the wrong place.
#[test]
#[ignore = "requires the local qwen2.5-vl-3b-4bit checkpoint"]
fn qwen2_5_vl_names_every_object_at_a_non_involution_window_grid() {
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

/// The repository's original fixture must still read as what it is.
///
/// This is not a redundant smoke test. Before the fix, the solid orange square
/// was described as "a person wearing a white shirt and black pants, standing
/// in front of a white wall": the erased tower rows left the language model
/// with no grounded content at all, and it invented a scene. A test that only
/// counts tokens passes on that output.
#[test]
#[ignore = "requires the local qwen2.5-vl-3b-4bit checkpoint"]
fn qwen2_5_vl_describes_the_solid_color_fixture_as_a_solid_color() {
    let Some(model_dir) = checkpoint_dir() else {
        return;
    };
    let body = describe(
        &model_dir,
        &fixture("test_image.png"),
        "Describe this image briefly.",
    );
    assert_mentions(&body, &[&["orange"]]);
    for invented in ["person", "man", "woman", "shirt", "wall", "floor"] {
        assert!(
            !body.to_lowercase().contains(invented),
            "solid orange fixture described with invented content ({invented})\nfull description: {body}"
        );
    }
}
