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

//! Gates for the Compass loader's `preprocessor_config.json` reading.

use super::compass_pixel_bounds;
use crate::vision::processors::qwen2_vl::{DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS};

fn write_preprocessor(name: &str, body: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mlxcel-compass-preproc-{name}"));
    std::fs::create_dir_all(&dir).expect("temp dir");
    std::fs::write(dir.join("preprocessor_config.json"), body).expect("write sidecar");
    dir
}

#[test]
fn published_size_block_supplies_the_pixel_bounds() {
    let dir = write_preprocessor(
        "published",
        r#"{"size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
    );
    assert_eq!(compass_pixel_bounds(&dir), (65536, 16777216));
}

#[test]
fn explicit_min_max_pixels_win_over_the_size_block() {
    let dir = write_preprocessor(
        "explicit",
        r#"{"min_pixels": 1024, "max_pixels": 4096, "size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
    );
    assert_eq!(compass_pixel_bounds(&dir), (1024, 4096));
}

#[test]
fn a_missing_sidecar_falls_back_to_the_family_defaults() {
    let dir = std::env::temp_dir().join("mlxcel-compass-preproc-absent");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let _ = std::fs::remove_file(dir.join("preprocessor_config.json"));
    assert_eq!(
        compass_pixel_bounds(&dir),
        (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
    );
}

#[test]
fn inverted_bounds_fall_back_instead_of_producing_a_degenerate_grid() {
    let dir = write_preprocessor(
        "inverted",
        r#"{"size": {"shortest_edge": 16777216, "longest_edge": 65536}}"#,
    );
    assert_eq!(
        compass_pixel_bounds(&dir),
        (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
    );
}
