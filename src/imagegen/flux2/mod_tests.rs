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
use serde_json::json;

fn extra(v: serde_json::Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap()
}

#[test]
fn gen_kwargs_parse_with_upstream_defaults() {
    let o = run_options(&Map::new()).unwrap();
    assert_eq!(o.max_sequence_length, 512);
    assert_eq!(o.tiled_vae, None);
    let o = run_options(&extra(
        json!({"max_sequence_length": 128, "tiled_vae": false}),
    ))
    .unwrap();
    assert_eq!((o.max_sequence_length, o.tiled_vae), (128, Some(false)));
    assert!(run_options(&extra(json!({"max_sequence_length": 0}))).is_err());
    assert!(run_options(&extra(json!({"max_sequence_length": 513}))).is_err());
    assert!(run_options(&extra(json!({"tiled_vae": "yes"}))).is_err());
}

#[test]
fn edit_dimensions_follow_upstream_rounding() {
    assert_eq!(round_edit_dimension(100.0), 256);
    assert_eq!(round_edit_dimension(1023.9), 1008);
    assert_eq!(round_edit_dimension(5000.0), 2048);
    // 3000x2000 scales to sqrt(1024^2 / 6e6) = 0.418: 1254 x 836 -> 1248 x 832.
    assert_eq!(resolve_edit_dimensions(3000, 2000), (1248, 832));
    // Small images are not upscaled.
    assert_eq!(resolve_edit_dimensions(512, 512), (512, 512));
}

#[test]
fn flux2_is_registered_in_both_tables() {
    assert!(crate::imagegen::FAMILIES.iter().any(|f| f.name == FAMILY));
    assert!(crate::imagegen::IMAGE_FAMILIES.contains(&FAMILY));
}

#[test]
fn load_names_the_missing_component_before_any_weight() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("FLUX.2-klein-4B");
    std::fs::create_dir_all(dir.join("transformer")).unwrap();
    let err = Flux2Model::load(&dir).err().unwrap().to_string();
    assert!(err.contains("text_encoder"), "{err}");
}
