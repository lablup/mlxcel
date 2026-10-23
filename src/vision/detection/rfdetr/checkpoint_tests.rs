// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

use super::*;
use std::io::Write;

fn write_safetensors(path: &Path, entries: serde_json::Value, data_len: usize) {
    let bytes = serde_json::to_vec(&entries).unwrap();
    let mut file = File::create(path).unwrap();
    file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&bytes).unwrap();
    file.write_all(&vec![0; data_len]).unwrap();
}

#[test]
fn preflight_rejects_oversized_index_before_parsing() {
    let dir = std::env::temp_dir().join(format!("rfdetr-index-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).unwrap();
    File::create(dir.join("model.safetensors.index.json"))
        .unwrap()
        .set_len(MAX_INDEX_BYTES + 1)
        .unwrap();
    let error = inspect(&dir).err().unwrap();
    assert!(error.contains("index") && error.contains("exceeds"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn preflight_rejects_too_many_and_oversized_names() {
    let dir = std::env::temp_dir().join(format!("rfdetr-header-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let mut entries = serde_json::Map::new();
    for index in 0..=MAX_TENSORS {
        entries.insert(
            format!("tensor.{index}"),
            serde_json::json!({"dtype":"F32","shape":[1],"data_offsets":[index * 4,(index + 1) * 4]}),
        );
    }
    write_safetensors(
        &path,
        serde_json::Value::Object(entries),
        (MAX_TENSORS + 1) * 4,
    );
    assert!(inspect(&dir).err().unwrap().contains("tensor count"));

    let long = "x".repeat(MAX_KEY_BYTES + 1);
    write_safetensors(
        &path,
        serde_json::json!({(long): {"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
        4,
    );
    assert!(inspect(&dir).err().unwrap().contains("key length"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn preflight_rejects_too_many_shards_before_header_reads() {
    let dir = std::env::temp_dir().join(format!("rfdetr-shards-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).unwrap();
    for index in 0..=MAX_SHARDS {
        File::create(dir.join(format!("model-{index:04}.safetensors"))).unwrap();
    }
    assert!(inspect(&dir).err().unwrap().contains("shards"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn metadata_caps_reject_invalid_specs() {
    assert!(validate_specs([("ok", mlxcel_core::dtype::FLOAT32, vec![2, 3])]).is_ok());
    assert!(validate_specs([("bad", mlxcel_core::dtype::FLOAT16, vec![1])]).is_err());
    assert!(validate_specs([("bad", mlxcel_core::dtype::FLOAT32, vec![0])]).is_err());
    assert!(
        validate_specs([(
            "huge",
            mlxcel_core::dtype::FLOAT32,
            vec![MAX_DIMENSION as i32, MAX_DIMENSION as i32],
        )])
        .is_err()
    );
}
