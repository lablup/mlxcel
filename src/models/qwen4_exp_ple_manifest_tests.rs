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

//! One test per manifest validation rule, mirroring upstream's
//! `QuantizedMMapNGramEmbedding.__init__` checks. Each failure names its field.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use tempfile::TempDir;

use super::*;

/// Row width 32: an interleaved affine record is 16 + 2 + 2 = 20 bytes.
const AFFINE_STRIDE: u64 = 20;

/// A 2-row, width-32 affine shard; returns its payload offset.
fn write_shard(path: &Path) -> u64 {
    let tensor = |dtype: &str, shape: Value, start: u64, end: u64| json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]});
    let header = json!({
        "weight": tensor("U32", json!([2, 4]), 0, 32),
        "scales": tensor("BF16", json!([2, 1]), 32, 36),
        "biases": tensor("BF16", json!([2, 1]), 36, 40),
    })
    .to_string();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&[0u8; 40]);
    std::fs::write(path, file).unwrap();
    8 + header.len() as u64
}

fn ranges_manifest(dir: &Path) -> Value {
    let payload = write_shard(&dir.join("shard.safetensors"));
    let tensor = |dtype: &str, offset: u64, shape: Value| json!({"file": "shard.safetensors", "offset": payload + offset, "dtype": dtype, "shape": shape});
    json!({
        "version": 2,
        "source_root": ".",
        "layout": "safetensors_ranges",
        "row_width": 32,
        "row_count": 2,
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "shards": [{
            "row_start": 0,
            "row_count": 2,
            "weight": tensor("U32", 0, json!([2, 4])),
            "scales": tensor("BF16", 32, json!([2, 1])),
            "biases": tensor("BF16", 36, json!([2, 1])),
        }],
    })
}

fn interleaved_manifest(dir: &Path) -> Value {
    std::fs::write(dir.join("rows"), vec![0u8; (3 * AFFINE_STRIDE) as usize]).unwrap();
    json!({
        "version": 2,
        "source_root": ".",
        "layout": "interleaved_q4",
        "data_file": "rows",
        "row_width": 32,
        "row_count": 3,
        "row_stride": AFFINE_STRIDE,
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
    })
}

fn fixture(build: fn(&Path) -> Value) -> (TempDir, Value, PathBuf) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ple-store.json");
    let value = build(dir.path());
    (dir, value, path)
}

#[track_caller]
fn assert_field(value: &Value, path: &Path, field: &str) {
    let error = PleManifest::from_value(value, path, None).expect_err("must be refused");
    assert_eq!(error.field, field, "{error}");
    assert!(error.to_string().contains(field), "{error}");
}

#[test]
fn accepts_both_valid_layouts() {
    let (_dir, value, path) = fixture(ranges_manifest);
    let manifest = PleManifest::from_value(&value, &path, None).unwrap();
    assert_eq!((manifest.row_count, manifest.row_width), (2, 32));
    let (_dir, value, path) = fixture(interleaved_manifest);
    let manifest = PleManifest::from_value(&value, &path, Some(7)).unwrap();
    assert_eq!((manifest.row_count, manifest.cache_rows), (3, 7));
}

#[test]
fn nvfp4_group_16_is_accepted() {
    let (_dir, mut value, path) = fixture(interleaved_manifest);
    value["quantization"] = json!({"bits": 4, "group_size": 16, "mode": "nvfp4"});
    value["row_stride"] = json!(18);
    value["row_count"] = json!(1);
    std::fs::write(path.with_file_name("rows"), vec![0u8; 18]).unwrap();
    assert_eq!(
        PleManifest::from_value(&value, &path, None)
            .unwrap()
            .quantization,
        PleQuantization::Nvfp4x16
    );
}

#[test]
fn rejects_each_missing_required_field() {
    for field in ["version", "source_root", "row_width", "quantization"] {
        let (_dir, mut value, path) = fixture(ranges_manifest);
        value.as_object_mut().unwrap().remove(field);
        assert_field(&value, &path, field);
    }
}

#[test]
fn rejects_an_unsupported_version() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["version"] = json!(1);
    assert_field(&value, &path, "version");
}

#[test]
fn rejects_an_unsupported_quantization() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["quantization"] = json!({"bits": 8, "group_size": 32, "mode": "affine"});
    assert_field(&value, &path, "quantization");
}

#[test]
fn rejects_a_row_width_not_divisible_by_the_group() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["row_width"] = json!(33);
    assert_field(&value, &path, "row_width");
}

#[test]
fn rejects_a_zero_row_width() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["row_width"] = json!(0);
    assert_field(&value, &path, "row_width");
}

#[test]
fn rejects_an_unknown_layout() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["layout"] = json!("columnar");
    assert_field(&value, &path, "layout");
}

#[test]
fn rejects_a_negative_cache_rows() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["cache_rows"] = json!(-1);
    assert_field(&value, &path, "cache_rows");
}

#[test]
fn interleaved_requires_its_fields() {
    for field in ["data_file", "row_count", "row_stride"] {
        let (_dir, mut value, path) = fixture(interleaved_manifest);
        value.as_object_mut().unwrap().remove(field);
        assert_field(&value, &path, field);
    }
}

#[test]
fn interleaved_refuses_a_data_file_outside_the_manifest_directory() {
    for bad in ["../rows", "/etc/rows", "sub/rows"] {
        let (_dir, mut value, path) = fixture(interleaved_manifest);
        value["data_file"] = json!(bad);
        assert_field(&value, &path, "data_file");
    }
}

#[test]
fn interleaved_refuses_a_wrong_stride() {
    let (_dir, mut value, path) = fixture(interleaved_manifest);
    value["row_stride"] = json!(21);
    assert_field(&value, &path, "row_stride");
}

#[test]
fn interleaved_refuses_a_wrong_file_size() {
    let (_dir, mut value, path) = fixture(interleaved_manifest);
    value["row_count"] = json!(4);
    assert_field(&value, &path, "data_file");
}

#[test]
fn ranges_require_shards() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value.as_object_mut().unwrap().remove("shards");
    assert_field(&value, &path, "shards");
}

#[test]
fn ranges_refuse_non_contiguous_shards() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["shards"][0]["row_start"] = json!(1);
    assert_field(&value, &path, "shards[0].row_start");
}

#[test]
fn ranges_refuse_an_empty_shard() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["shards"][0]["row_count"] = json!(0);
    assert_field(&value, &path, "shards[0].row_count");
}

#[test]
fn ranges_refuse_a_wrong_dtype() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["shards"][0]["scales"]["dtype"] = json!("F32");
    assert_field(&value, &path, "shards[0].scales.dtype");
}

#[test]
fn ranges_refuse_a_file_outside_source_root() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["shards"][0]["weight"]["file"] = json!("../shard.safetensors");
    assert_field(&value, &path, "shards[0].weight.file");
}

#[test]
fn ranges_refuse_a_row_count_mismatch() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["shards"][0]["weight"]["shape"] = json!([3, 4]);
    assert_field(&value, &path, "shards[0].weight.shape");
}

#[test]
fn ranges_refuse_a_byte_range_past_the_file() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["shards"][0]["biases"]["offset"] = json!(1000);
    assert_field(&value, &path, "shards[0].biases.offset");
}

#[test]
fn ranges_refuse_a_packed_width_that_disagrees_with_row_width() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["shards"][0]["weight"]["shape"] = json!([2, 3]);
    assert_field(&value, &path, "shards[0].weight.shape");
}

#[test]
fn ranges_refuse_an_auxiliary_shape_off_the_group_layout() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    value["shards"][0]["scales"]["shape"] = json!([2, 2]);
    assert_field(&value, &path, "shards[0].scales.shape");
}

#[test]
fn ranges_accept_a_source_root_outside_the_manifest_directory() {
    let (dir, mut value, path) = fixture(ranges_manifest);
    let view = dir.path().join("view");
    std::fs::create_dir(&view).unwrap();
    value["source_root"] = json!("..");
    PleManifest::from_value(&value, &view.join("ple-store.json"), None).unwrap();
    drop(path);
}

#[test]
fn ranges_refuse_a_file_without_a_safetensors_header() {
    let (dir, mut value, path) = fixture(ranges_manifest);
    let size = std::fs::metadata(dir.path().join("shard.safetensors"))
        .unwrap()
        .len();
    std::fs::write(dir.path().join("secret"), vec![b'x'; size as usize]).unwrap();
    value["shards"][0]["weight"]["file"] = json!("secret");
    assert_field(&value, &path, "shards[0].weight.file");
}

#[test]
fn ranges_refuse_a_range_the_header_does_not_declare() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    let offset = value["shards"][0]["weight"]["offset"].as_u64().unwrap();
    value["shards"][0]["weight"]["offset"] = json!(offset - 1);
    assert_field(&value, &path, "shards[0].weight.offset");
}

#[test]
fn ranges_refuse_shards_repeated_past_the_size_of_their_files() {
    let (_dir, mut value, path) = fixture(ranges_manifest);
    let shard = value["shards"][0].clone();
    let shards: Vec<Value> = (0..16u64)
        .map(|index| {
            let mut shard = shard.clone();
            shard["row_start"] = json!(index * 2);
            shard
        })
        .collect();
    value["shards"] = json!(shards);
    assert_field(&value, &path, "shards");
}

#[cfg(unix)]
#[test]
fn interleaved_refuses_a_symlink_leaving_the_model() {
    let (_dir, value, path) = fixture(interleaved_manifest);
    let outside = TempDir::new().unwrap();
    let target = outside.path().join("secret");
    std::fs::write(&target, vec![0u8; (3 * AFFINE_STRIDE) as usize]).unwrap();
    let rows = path.with_file_name("rows");
    std::fs::remove_file(&rows).unwrap();
    std::os::unix::fs::symlink(&target, &rows).unwrap();
    assert_field(&value, &path, "data_file");
}

#[cfg(unix)]
#[test]
fn interleaved_follows_a_hugging_face_snapshot_symlink() {
    let repo = TempDir::new().unwrap();
    let snapshot = repo.path().join("snapshots/rev");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir(repo.path().join("blobs")).unwrap();
    let value = interleaved_manifest(&snapshot);
    let blob = repo.path().join("blobs/abc");
    std::fs::rename(snapshot.join("rows"), &blob).unwrap();
    std::os::unix::fs::symlink("../../blobs/abc", snapshot.join("rows")).unwrap();
    PleManifest::from_value(&value, &snapshot.join("ple-store.json"), None).unwrap();
}
