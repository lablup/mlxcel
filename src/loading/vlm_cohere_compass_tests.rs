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

//! Gates for where the Compass loader reads its resize bounds from.
//!
//! The published checkpoint ships two sidecars that disagree, and picking the
//! wrong one changes the image token count without failing, so the precedence
//! is pinned here rather than left to whichever file is read first.

use super::compass_pixel_bounds;
use crate::vision::processors::qwen2_vl::{DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS};

fn model_dir(name: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mlxcel-compass-bounds-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    for (file, body) in files {
        std::fs::write(dir.join(file), body).expect("write sidecar");
    }
    dir
}

/// Both sidecars exactly as `mlx-community/North-Micro-Vision-Instruct-4bit`
/// ships them. HF's `AutoProcessor` builds the image processor from the nested
/// `processor_config.json` block, so 16384 / 3868706 is the reference
/// behaviour and the standalone file's 65536 / 16777216 is stale.
#[test]
fn the_nested_processor_config_block_wins_over_preprocessor_config() {
    let dir = model_dir(
        "published",
        &[
            (
                "processor_config.json",
                r#"{"image_processor": {"min_pixels": 16384, "max_pixels": 3868706,
                    "size": {"shortest_edge": 16384, "longest_edge": 3868706}}}"#,
            ),
            (
                "preprocessor_config.json",
                r#"{"size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
            ),
        ],
    );
    assert_eq!(compass_pixel_bounds(&dir), (16384, 3868706));
}

/// With no nested block the standalone sidecar is the source.
#[test]
fn preprocessor_config_is_the_fallback() {
    let dir = model_dir(
        "fallback",
        &[(
            "preprocessor_config.json",
            r#"{"size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
        )],
    );
    assert_eq!(compass_pixel_bounds(&dir), (65536, 16_777_216));
}

/// Explicit `min_pixels` / `max_pixels` beat the `size` block in the same file,
/// which is the order the HF processor reads them in.
#[test]
fn explicit_pixel_keys_win_over_the_size_block() {
    let dir = model_dir(
        "explicit",
        &[(
            "preprocessor_config.json",
            r#"{"min_pixels": 1024, "max_pixels": 4096,
                "size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
        )],
    );
    assert_eq!(compass_pixel_bounds(&dir), (1024, 4096));
}

#[test]
fn no_sidecar_falls_back_to_the_family_defaults() {
    let dir = model_dir("absent", &[]);
    assert_eq!(
        compass_pixel_bounds(&dir),
        (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
    );
}

/// Inverted bounds fall back rather than producing a degenerate grid.
#[test]
fn inverted_bounds_fall_back() {
    let dir = model_dir(
        "inverted",
        &[(
            "preprocessor_config.json",
            r#"{"size": {"shortest_edge": 16777216, "longest_edge": 65536}}"#,
        )],
    );
    assert_eq!(
        compass_pixel_bounds(&dir),
        (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
    );
}
