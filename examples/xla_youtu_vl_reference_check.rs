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

//! Production MLX/IREE capture for the independent pinned Youtu-VL HF oracle.
//!
//! This runner consumes only oracle identity and prompt metadata. It
//! independently tokenizes, preprocesses, executes the actual 27-layer vision
//! graph, prepares multimodal embeddings, captures the 40-layer MLA cache and
//! logits, then repeats the request after an engine reset to prove slot reuse.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use image::DynamicImage;
use mlxcel::tokenizer::load_tokenizer;
use mlxcel::vision::processors::youtu_vl::YoutuVLProcessor;
use mlxcel::{
    HostMultimodalPreprocessor, PreparedTensorDType, YoutuVlIreeHostPreprocessor,
    initialize_runtime,
};
use mlxcel_xla::{IreeYoutuVlDiagnosticProjector, YoutuVlReferenceDiagnosticEngine};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

const CONTRACT: &str = "youtu-vl-hf-eager-oracle-v1";
const CHECKPOINT_REPO: &str = "tencent/Youtu-VL-4B-Instruct";
const CHECKPOINT_REVISION: &str = "8d30a0e49662a1d628a472b12df264dbcd768753";
const CHECKPOINT_ARTIFACT_MANIFEST_SHA256: &str =
    "4d67dd2750d1ee8d87e68b0b52f5aa6c5b5b1dd85385df643845e393628812c9";
const FIXTURE_PATH: &str = "tests/fixtures/test_image.png";
const FIXTURE_SHA256: &str = "5e7d54e8a7d21802378c87d2d70cf551e29739fe27599ddf129ebccdad1e6261";
const PROMPT: &str = "<|image_pad|>\nDescribe the image briefly.";
const IMAGE_TOKEN_ID: i32 = 128_264;
const PATCH_SIZE: usize = 16;
const PATCHES: usize = 256;
const PATCH_WIDTH: usize = 3 * PATCH_SIZE * PATCH_SIZE;
const TEXT_HIDDEN: usize = 2_560;
const TEXT_LAYERS: usize = 40;
const KV_WIDTH: usize = 576;
const MAX_NEW_TOKENS: usize = 4;

fn argument(flag: &str) -> Option<String> {
    let args = std::env::args().collect::<Vec<_>>();
    args.iter()
        .position(|value| value == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn required_path(flag: &str) -> PathBuf {
    argument(flag)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("missing required {flag}"))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_file(path: &Path) -> String {
    sha256_bytes(&fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())))
}

fn verify_reference_manifest(root: &Path) -> Value {
    let manifest_path = root.join("manifest.json");
    let bytes = fs::read(&manifest_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", manifest_path.display()));
    let expected =
        fs::read_to_string(root.join("manifest.sha256")).expect("read reference manifest.sha256");
    assert_eq!(
        sha256_bytes(&bytes),
        expected.trim(),
        "reference manifest SHA-256 differs"
    );
    let manifest: Value = serde_json::from_slice(&bytes).expect("parse reference manifest");
    assert_eq!(manifest["schema"], 1);
    assert_eq!(manifest["contract"], CONTRACT);
    assert_eq!(manifest["producer"], "hf-transformers-eager");
    assert_eq!(manifest["fixture"]["path"], FIXTURE_PATH);
    assert_eq!(manifest["fixture"]["sha256"], FIXTURE_SHA256);
    assert_eq!(manifest["case"]["name"], "image_text");
    assert_eq!(manifest["case"]["prompt"], PROMPT);
    assert_eq!(manifest["generation"]["mode"], "greedy");
    assert_eq!(manifest["generation"]["max_new_tokens"], MAX_NEW_TOKENS);
    assert_eq!(manifest["architecture"]["vision_depth"], 27);
    assert_eq!(manifest["architecture"]["text_layers"], TEXT_LAYERS);
    assert_eq!(manifest["architecture"]["text_hidden"], TEXT_HIDDEN);
    assert_eq!(manifest["architecture"]["kv_lora_rank"], 512);
    assert_eq!(manifest["architecture"]["qk_rope_head_dim"], 64);
    assert_eq!(
        manifest["lifecycle"]["events"],
        json!([
            "reset",
            "prefill",
            "greedy_decode",
            "reset_for_reuse",
            "prefill_reuse",
            "greedy_decode_reuse"
        ])
    );
    manifest
}

fn verify_checkpoint(model: &Path, checkpoint: &Value) {
    assert_eq!(checkpoint["repo"], CHECKPOINT_REPO);
    assert_eq!(checkpoint["revision"], CHECKPOINT_REVISION);
    let artifact_manifest = &checkpoint["artifact_manifest"];
    assert_eq!(
        artifact_manifest["canonical_sha256"],
        CHECKPOINT_ARTIFACT_MANIFEST_SHA256
    );
    let files =
        serde_json::from_value::<BTreeMap<String, String>>(artifact_manifest["files"].clone())
            .expect("checkpoint artifact files must be a string map");
    let canonical = serde_json::to_vec(&files).expect("serialize canonical artifact map");
    assert_eq!(
        sha256_bytes(&canonical),
        CHECKPOINT_ARTIFACT_MANIFEST_SHA256,
        "checkpoint artifact map is not canonical"
    );
    for (filename, expected) in &files {
        let path = model.join(filename);
        assert!(path.is_file(), "checkpoint artifact is missing: {filename}");
        assert_eq!(
            sha256_file(&path),
            expected.as_str(),
            "checkpoint artifact SHA-256 differs: {filename}"
        );
    }
    let unexpected_weights = fs::read_dir(model)
        .expect("read checkpoint directory")
        .map(|entry| entry.expect("read checkpoint entry").path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("safetensors"))
        .filter_map(|path| {
            let filename = path.file_name()?.to_str()?.to_string();
            (!files.contains_key(&filename)).then_some(filename)
        })
        .collect::<Vec<_>>();
    assert!(
        unexpected_weights.is_empty(),
        "checkpoint has unpinned weight artifact(s): {unexpected_weights:?}"
    );
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_array(out: &Path, stage: &str, bytes: &[u8], dtype: &str, shape: &[usize]) -> Value {
    if dtype == "float32" {
        assert!(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
                .all(f32::is_finite),
            "{stage} contains a non-finite value"
        );
    }
    assert_eq!(
        bytes.len(),
        shape.iter().product::<usize>() * 4,
        "{stage} byte length differs"
    );
    let filename = format!("image_text.{stage}.bin");
    let path = out.join(&filename);
    fs::write(&path, bytes).unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    json!({
        "file": filename,
        "dtype": dtype,
        "shape": shape,
        "sha256": sha256_file(&path),
    })
}

fn write_f32(
    arrays: &mut Map<String, Value>,
    out: &Path,
    stage: &str,
    values: &[f32],
    shape: &[usize],
) {
    arrays.insert(
        stage.to_string(),
        write_array(out, stage, &f32_bytes(values), "float32", shape),
    );
}

fn write_i32(
    arrays: &mut Map<String, Value>,
    out: &Path,
    stage: &str,
    values: &[i32],
    shape: &[usize],
) {
    arrays.insert(
        stage.to_string(),
        write_array(out, stage, &i32_bytes(values), "int32", shape),
    );
}

fn usize_i32(values: &[usize], label: &str) -> Vec<i32> {
    values
        .iter()
        .map(|&value| i32::try_from(value).unwrap_or_else(|_| panic!("{label} value exceeds i32")))
        .collect()
}

fn reconstruct_pixels(patches: &[f32], grid_height: usize, grid_width: usize) -> Vec<f32> {
    assert_eq!(patches.len(), grid_height * grid_width * PATCH_WIDTH);
    let height = grid_height * PATCH_SIZE;
    let width = grid_width * PATCH_SIZE;
    let mut pixels = vec![0.0f32; 3 * height * width];
    for patch in 0..grid_height * grid_width {
        let patch_y = patch / grid_width;
        let patch_x = patch % grid_width;
        let row = &patches[patch * PATCH_WIDTH..(patch + 1) * PATCH_WIDTH];
        let mut cursor = 0usize;
        for channel in 0..3 {
            for dy in 0..PATCH_SIZE {
                for dx in 0..PATCH_SIZE {
                    let y = patch_y * PATCH_SIZE + dy;
                    let x = patch_x * PATCH_SIZE + dx;
                    pixels[channel * height * width + y * width + x] = row[cursor];
                    cursor += 1;
                }
            }
        }
    }
    pixels
}

fn main() {
    let model = required_path("--model");
    let reference_root = required_path("--reference");
    let image_path = required_path("--image");
    let out = required_path("--out");
    let device = argument("--device").unwrap_or_else(|| "local-task".to_string());
    let context_capacity = argument("--context-capacity")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("context capacity must be usize")
        })
        .unwrap_or(2_048);
    assert!(
        !out.exists(),
        "immutable capture output already exists: {}",
        out.display()
    );
    fs::create_dir_all(&out).unwrap_or_else(|error| panic!("create {}: {error}", out.display()));

    let reference = verify_reference_manifest(&reference_root);
    verify_checkpoint(&model, &reference["checkpoint"]);
    assert_eq!(
        sha256_file(&image_path),
        FIXTURE_SHA256,
        "fixture SHA-256 differs"
    );
    let image = image::open(&image_path)
        .unwrap_or_else(|error| panic!("open {}: {error}", image_path.display()))
        .into_rgb8();
    let image = DynamicImage::ImageRgb8(image);

    let tokenizer = load_tokenizer(&model).expect("load pinned Youtu-VL tokenizer");
    let unexpanded = tokenizer
        .encode(PROMPT, false)
        .expect("tokenize pinned Youtu-VL prompt")
        .into_iter()
        .map(|value| i32::try_from(value).expect("token id fits i32"))
        .collect::<Vec<_>>();
    let reference_unexpanded = reference["case"]["unexpanded_input_ids"]
        .as_array()
        .expect("reference unexpanded_input_ids must be an array")
        .iter()
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .expect("reference token id fits i32")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        unexpanded, reference_unexpanded,
        "independent tokenizer ids differ from HF reference"
    );

    let processor = YoutuVLProcessor::new(PATCH_SIZE, 2)
        .with_norm([0.5; 3], [0.5; 3])
        .with_max_patches_per_image(PATCHES)
        .with_resample(2);
    let (flattened_patches, spatial_shapes, patch_width) = processor
        .try_preprocess_values_with_spatial(std::slice::from_ref(&image))
        .expect("run checked Youtu-VL flattened-patch processor");
    assert_eq!(spatial_shapes, [(16, 16)]);
    assert_eq!(patch_width, PATCH_WIDTH);
    let resized_pixels = reconstruct_pixels(&flattened_patches, 16, 16);

    let _runtime = initialize_runtime();
    let mut diagnostic_projector = IreeYoutuVlDiagnosticProjector::load(&model, &device)
        .expect("load Youtu-VL diagnostic vision projector");
    let vision = diagnostic_projector
        .capture(&flattened_patches, &spatial_shapes)
        .expect("capture actual 27-layer Youtu-VL vision graph");
    assert_eq!(vision.abi_version, 2);
    assert_eq!(vision.patch_shape, [PATCHES, 1_152]);
    assert_eq!(vision.merged_shape, [64, TEXT_HIDDEN]);

    let production_preprocessor = YoutuVlIreeHostPreprocessor::load(&model, &device)
        .expect("load production Youtu-VL IREE host preprocessor");
    let prepared = production_preprocessor
        .prepare(&unexpanded, std::slice::from_ref(&image))
        .expect("prepare production Youtu-VL multimodal prefill");
    assert_eq!(prepared.embeddings.dtype, PreparedTensorDType::Float32);
    assert_eq!(
        prepared.embeddings.shape,
        [1, prepared.sequence_len, TEXT_HIDDEN]
    );
    assert_eq!(
        prepared
            .token_ids
            .iter()
            .filter(|&&token| token == IMAGE_TOKEN_ID)
            .count(),
        64
    );

    let mut engine = YoutuVlReferenceDiagnosticEngine::load(&model, &device, context_capacity)
        .expect("load Youtu-VL production diagnostic language engine");
    let fresh = engine
        .capture(&prepared, KV_WIDTH, MAX_NEW_TOKENS)
        .expect("capture fresh Youtu-VL MLA request");
    let reuse = engine
        .capture(&prepared, KV_WIDTH, MAX_NEW_TOKENS)
        .expect("reset and capture reused Youtu-VL MLA slot");
    assert_eq!(
        fresh.tokens, reuse.tokens,
        "slot reuse token stream differs"
    );
    assert_eq!(
        fresh.prefill, reuse.prefill,
        "slot reuse logits or MLA cache differs"
    );
    assert_eq!(fresh.prefill.layers, TEXT_LAYERS);
    assert_eq!(fresh.prefill.kv_width, KV_WIDTH);
    assert_eq!(fresh.prefill.logits.len(), 283_386);
    assert_eq!(fresh.prefill.kv.len(), TEXT_LAYERS * 2 * KV_WIDTH);

    let mut arrays = Map::new();
    write_f32(
        &mut arrays,
        &out,
        "resized_normalized_pixels",
        &resized_pixels,
        &[3, 256, 256],
    );
    write_f32(
        &mut arrays,
        &out,
        "flattened_patches",
        &flattened_patches,
        &[PATCHES, PATCH_WIDTH],
    );
    write_f32(
        &mut arrays,
        &out,
        "patches.window_order",
        &vision.patches_window_order,
        &[PATCHES, PATCH_WIDTH],
    );
    write_f32(
        &mut arrays,
        &out,
        "vision_rope.freqs",
        &vision.rope_freqs,
        &[PATCHES, 36],
    );
    for stage in &vision.stages {
        write_f32(&mut arrays, &out, &stage.name, &stage.values, &stage.shape);
    }
    arrays.insert(
        "prepared_embeddings".to_string(),
        write_array(
            &out,
            "prepared_embeddings",
            &prepared.embeddings.bytes,
            "float32",
            &prepared.embeddings.shape,
        ),
    );
    write_f32(
        &mut arrays,
        &out,
        "prefill_logits",
        &fresh.prefill.logits,
        &[283_386],
    );
    write_f32(
        &mut arrays,
        &out,
        "selected_kv",
        &fresh.prefill.kv,
        &[TEXT_LAYERS, 2, KV_WIDTH],
    );
    write_f32(
        &mut arrays,
        &out,
        "reuse_prefill_logits",
        &reuse.prefill.logits,
        &[283_386],
    );
    write_f32(
        &mut arrays,
        &out,
        "reuse_selected_kv",
        &reuse.prefill.kv,
        &[TEXT_LAYERS, 2, KV_WIDTH],
    );
    write_i32(
        &mut arrays,
        &out,
        "expanded_input_ids",
        &prepared.token_ids,
        &[prepared.sequence_len],
    );
    let placeholder_positions = prepared
        .token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == IMAGE_TOKEN_ID).then_some(index as i32))
        .collect::<Vec<_>>();
    write_i32(
        &mut arrays,
        &out,
        "placeholder_positions",
        &placeholder_positions,
        &[placeholder_positions.len()],
    );
    write_i32(&mut arrays, &out, "spatial_shapes", &[16, 16], &[1, 2]);
    write_i32(
        &mut arrays,
        &out,
        "window_group_index",
        &usize_i32(&vision.window_group_index, "window group index"),
        &[64],
    );
    write_i32(
        &mut arrays,
        &out,
        "reverse_group_index",
        &usize_i32(&vision.reverse_group_index, "reverse group index"),
        &[64],
    );
    write_i32(
        &mut arrays,
        &out,
        "window_cu_seqlens",
        &usize_i32(&vision.window_cu_seqlens, "window boundary"),
        &[vision.window_cu_seqlens.len()],
    );
    write_i32(
        &mut arrays,
        &out,
        "full_cu_seqlens",
        &usize_i32(&vision.full_cu_seqlens, "full boundary"),
        &[vision.full_cu_seqlens.len()],
    );
    write_i32(
        &mut arrays,
        &out,
        "greedy_tokens",
        &fresh.tokens,
        &[MAX_NEW_TOKENS],
    );
    write_i32(
        &mut arrays,
        &out,
        "reuse_greedy_tokens",
        &reuse.tokens,
        &[MAX_NEW_TOKENS],
    );

    let actual = json!({
        "schema": 1,
        "contract": CONTRACT,
        "producer": "mlxcel-xla-diagnostics",
        "checkpoint": reference["checkpoint"].clone(),
        "fixture": reference["fixture"].clone(),
        "architecture": reference["architecture"].clone(),
        "generation": reference["generation"].clone(),
        "lifecycle": reference["lifecycle"].clone(),
        "case": {
            "name": "image_text",
            "prompt": PROMPT,
            "unexpanded_input_ids": unexpanded,
            "arrays": arrays,
        },
    });
    let manifest_path = out.join("manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&actual).expect("serialize actual manifest"),
    )
    .expect("write actual manifest");
    fs::write(
        out.join("manifest.sha256"),
        format!("{}\n", sha256_file(&manifest_path)),
    )
    .expect("write actual manifest hash");
    println!("{}", manifest_path.display());
}
