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

use std::fs;
use std::path::{Path, PathBuf};

use super::Qwen3VLVisionStage;
use crate::LoadedModel;
use mlxcel_core::dtype;
use serde_json::json;

fn default_north_micro_model_dir() -> PathBuf {
    PathBuf::from("models/North-Micro-Vision-Instruct-4bit")
}

fn write_stage_dump(
    out: &Path,
    stage: Qwen3VLVisionStage,
    tensor: &mlxcel_core::MlxArray,
) -> serde_json::Value {
    let label = stage.label();
    let as_f32 = mlxcel_core::astype(tensor, dtype::FLOAT32);
    mlxcel_core::eval(&as_f32);
    let bytes = mlxcel_core::array_to_raw_bytes(&as_f32);
    let filename = format!("{label}.f32.bin");
    fs::write(out.join(&filename), &bytes).unwrap_or_else(|error| {
        panic!(
            "write Qwen3-VL stage dump {}: {error}",
            out.join(&filename).display()
        )
    });
    json!({
        "stage": label,
        "file": filename,
        "dtype": "float32",
        "shape": mlxcel_core::array_shape(tensor),
    })
}

#[test]
#[ignore = "requires mlx-community/North-Micro-Vision-Instruct-4bit and writes oracle-comparison tensors"]
fn dump_north_micro_vision_stage_tensors() {
    let model_dir = std::env::var_os("MLXCEL_QWEN3_VL_STAGE_DUMP_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(default_north_micro_model_dir);
    assert!(
        model_dir.exists(),
        "North-Micro checkpoint not found at {}; fetch with: ./target/release/mlxcel download mlx-community/North-Micro-Vision-Instruct-4bit",
        model_dir.display()
    );

    let image_path = std::env::var_os("MLXCEL_QWEN3_VL_STAGE_DUMP_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/fixtures/test_image_shapes.png"));
    let out_dir = std::env::var_os("MLXCEL_QWEN3_VL_STAGE_DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("qwen3_vl_stage_dump_{}", std::process::id()))
        });
    fs::create_dir_all(&out_dir).expect("create stage dump directory");

    let (loaded, _tokenizer) = crate::load_model(&model_dir).expect("load North-Micro checkpoint");
    let LoadedModel::CohereCompassVLM(model) = loaded else {
        panic!(
            "{} did not load as Cohere Compass / North-Micro-Vision",
            model_dir.display()
        );
    };
    let image = image::open(&image_path)
        .unwrap_or_else(|error| panic!("open {}: {error}", image_path.display()));
    let (pixel_values, grid_thw) = model.processor.preprocess_with_grid(&[image]);
    let embed_probe = model
        .text_model
        .get_embed_tokens(&mlxcel_core::from_slice_i32(&[0], &[1, 1]));
    let pixel_values = mlxcel_core::astype(&pixel_values, mlxcel_core::array_dtype(&embed_probe));

    let mut manifest = Vec::new();
    let output = model.vision_encoder.forward_with_grid_observer(
        &pixel_values,
        &grid_thw,
        |stage, tensor| {
            manifest.push(write_stage_dump(&out_dir, stage, tensor));
        },
    );
    mlxcel_core::eval(&output.hidden_states);
    for tensor in &output.deepstack_features {
        mlxcel_core::eval(tensor);
    }
    fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "model": model_dir,
            "image": image_path,
            "grid_thw": grid_thw,
            "stages": manifest,
        }))
        .expect("serialize stage dump manifest"),
    )
    .expect("write stage dump manifest");
    eprintln!("wrote Qwen3-VL stage dump to {}", out_dir.display());
}
