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

//! Diagnostics-only LLaVA reference capture for issue #862.
//!
//! The host stages come from the same qualified preprocessor as CLI/server
//! image requests. The decoder stages come from one production IREE ragged
//! bundle, with compact selected-KV readback enabled only by
//! `xla-diagnostics`. Generated binary captures stay outside Git and are
//! compared by `spike/openxla/llava_reference_oracle.py`.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::DynamicImage;
use mlxcel::{
    HostMultimodalPreprocessor, LlavaHostPreprocessor, OwnedTensor, PreparedPositions,
    PreparedTensorDType, initialize_runtime, server::ChatTemplateProcessor,
    tokenizer::load_tokenizer,
};
use mlxcel_xla::LlavaReferenceDiagnosticEngine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
struct ReferenceManifest {
    kv_selection: KvSelection,
    generation: Generation,
    image_fixture: ImageFixture,
    converted_checkpoint: ConvertedCheckpoint,
    cases: Vec<ReferenceCase>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ArtifactManifest {
    canonical_sha256: String,
    files: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct ConvertedCheckpoint {
    artifact_manifest: ArtifactManifest,
}

#[derive(Debug, Deserialize, Serialize)]
struct ImageFixture {
    path: String,
    sha256: String,
    two_image_transform: String,
}

#[derive(Debug, Deserialize)]
struct KvSelection {
    width: usize,
}

#[derive(Debug, Deserialize)]
struct Generation {
    max_new_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct ReferenceCase {
    name: String,
    user_prompt: String,
    text: String,
    image_count: usize,
    image_transforms: Vec<String>,
    unexpanded_input_ids: Vec<i32>,
}

const FIXTURE_PATH: &str = "tests/fixtures/test_image.png";
const FIXTURE_SHA256: &str = "5e7d54e8a7d21802378c87d2d70cf551e29739fe27599ddf129ebccdad1e6261";

const VISION_BLOCK0_STAGES: [&str; 12] = [
    "vision_block0_layer_norm1",
    "vision_block0_q_proj",
    "vision_block0_k_proj",
    "vision_block0_v_proj",
    "vision_block0_attention_context",
    "vision_block0_attention_output",
    "vision_block0_attention_residual",
    "vision_block0_layer_norm2",
    "vision_block0_mlp_fc1",
    "vision_block0_mlp_activation",
    "vision_block0_mlp_fc2",
    "vision_block0_output",
];

fn argument(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
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

fn required_usize(flag: &str, default: usize) -> usize {
    argument(flag)
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{flag} must be an unsigned integer"))
        })
        .unwrap_or(default)
}

fn dtype_name(dtype: PreparedTensorDType) -> &'static str {
    match dtype {
        PreparedTensorDType::Float16 => "float16",
        PreparedTensorDType::BFloat16 => "bfloat16",
        PreparedTensorDType::Float32 => "float32",
        PreparedTensorDType::Int32 => "int32",
        _ => panic!("unsupported future prepared tensor dtype"),
    }
}

fn write_raw(
    out: &Path,
    case: &str,
    stage: &str,
    bytes: &[u8],
    dtype: &str,
    shape: &[usize],
) -> Value {
    let filename = format!("{case}.{stage}.bin");
    fs::write(out.join(&filename), bytes).unwrap_or_else(|error| {
        panic!(
            "write {} capture {}: {error}",
            stage,
            out.join(&filename).display()
        )
    });
    json!({"file": filename, "dtype": dtype, "shape": shape})
}

fn write_tensor(out: &Path, case: &str, stage: &str, tensor: &OwnedTensor) -> Value {
    write_raw(
        out,
        case,
        stage,
        &tensor.bytes,
        dtype_name(tensor.dtype),
        &tensor.shape,
    )
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn sha256_file(path: &Path) -> String {
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open pinned {}: {error}", path.display()));
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("hash pinned {}: {error}", path.display()));
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    format!("{:x}", digest.finalize())
}

fn verify_artifact_manifest(root: &Path, manifest: &ArtifactManifest) {
    let mut canonical = String::new();
    for (filename, expected) in &manifest.files {
        assert!(
            !filename.contains('/') && !filename.contains('\\'),
            "artifact manifest path must be a filename: {filename}"
        );
        let path = root.join(filename);
        let actual = sha256_file(&path);
        assert_eq!(
            actual,
            *expected,
            "converted snapshot hash differs for {}",
            path.display()
        );
        canonical.push_str(filename);
        canonical.push('=');
        canonical.push_str(expected);
        canonical.push('\n');
    }
    let actual_canonical = format!("{:x}", Sha256::digest(canonical.as_bytes()));
    assert_eq!(
        actual_canonical, manifest.canonical_sha256,
        "converted snapshot canonical manifest hash differs"
    );
}

fn transformed_image(image: &DynamicImage, transform: &str) -> DynamicImage {
    match transform {
        "identity" => image.clone(),
        "horizontal_mirror" => {
            DynamicImage::ImageRgb8(image::imageops::flip_horizontal(&image.to_rgb8()))
        }
        other => panic!("unsupported pinned image transform {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_hash_matches_the_pinned_fixture() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_PATH);
        assert_eq!(sha256_file(&fixture), FIXTURE_SHA256);
    }
}

fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

fn mem_available_kib() -> Option<u64> {
    fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

fn positions(value: &PreparedPositions, sequence_len: usize) -> Vec<i32> {
    match value {
        PreparedPositions::Sequential { start, length } => {
            assert_eq!(*start, 0);
            assert_eq!(*length, sequence_len);
            (0..sequence_len)
                .map(|position| i32::try_from(position).expect("position fits i32"))
                .collect()
        }
        other => panic!("LLaVA reference expected sequential positions, got {other:?}"),
    }
}

fn render_converted_prompt(
    processor: &ChatTemplateProcessor,
    user_prompt: &str,
    image_count: usize,
) -> String {
    let mut content: Vec<Value> = (0..image_count).map(|_| json!({"type": "image"})).collect();
    content.push(json!({"type": "text", "text": user_prompt}));
    processor
        .apply_raw(
            &json!([{
                "role": "user",
                "content": content,
            }]),
            None,
        )
        .expect("render converted checkpoint chat template")
}

fn main() {
    let model = required_path("--model");
    let reference_dir = required_path("--reference");
    let image_path = required_path("--image");
    let out = required_path("--out");
    let device = argument("--device").unwrap_or_else(|| "local-task".to_string());
    let context_capacity = required_usize("--context-capacity", 2048);
    let runtime = initialize_runtime();
    mlxcel_core::reset_peak_memory();
    let mem_available_before_kib = mem_available_kib();
    fs::create_dir_all(&out)
        .unwrap_or_else(|error| panic!("create capture directory {}: {error}", out.display()));
    let reference: ReferenceManifest = serde_json::from_str(
        &fs::read_to_string(reference_dir.join("manifest.json"))
            .unwrap_or_else(|error| panic!("read reference manifest: {error}")),
    )
    .unwrap_or_else(|error| panic!("parse reference manifest: {error}"));
    assert_eq!(reference.image_fixture.path, FIXTURE_PATH);
    assert_eq!(reference.image_fixture.sha256, FIXTURE_SHA256);
    assert_eq!(
        reference.image_fixture.two_image_transform,
        "horizontal_mirror"
    );
    assert_eq!(
        sha256_file(&image_path),
        FIXTURE_SHA256,
        "fixture image SHA-256 differs"
    );
    verify_artifact_manifest(&model, &reference.converted_checkpoint.artifact_manifest);
    let image = image::open(&image_path)
        .unwrap_or_else(|error| panic!("open fixture image {}: {error}", image_path.display()))
        .into_rgb8();
    let image = DynamicImage::ImageRgb8(image);

    let host_load_started = Instant::now();
    let preprocessor = LlavaHostPreprocessor::load(&model)
        .unwrap_or_else(|error| panic!("load LLaVA host preprocessor: {error}"));
    let chat_template = ChatTemplateProcessor::from_model_path(&model)
        .unwrap_or_else(|error| panic!("load converted checkpoint chat template: {error}"))
        .expect("converted checkpoint must include a chat template");
    let tokenizer = load_tokenizer(&model)
        .unwrap_or_else(|error| panic!("load converted checkpoint tokenizer: {error}"));
    let host_load_seconds = host_load_started.elapsed().as_secs_f64();
    let compile_started = Instant::now();
    let mut engine = LlavaReferenceDiagnosticEngine::load(&model, &device, context_capacity)
        .unwrap_or_else(|error| panic!("load LLaVA IREE diagnostic engine: {error}"));
    let compile_load_seconds = compile_started.elapsed().as_secs_f64();

    let mut captured_cases = Vec::new();
    for reference_case in &reference.cases {
        assert_eq!(
            reference_case.image_count,
            reference_case.image_transforms.len(),
            "image count and transform count diverged for {}",
            reference_case.name
        );
        let converted_prompt = render_converted_prompt(
            &chat_template,
            &reference_case.user_prompt,
            reference_case.image_count,
        );
        assert_eq!(
            converted_prompt, reference_case.text,
            "source and converted chat templates diverged for {}",
            reference_case.name
        );
        let converted_ids: Vec<i32> = tokenizer
            .encode(&converted_prompt, false)
            .unwrap_or_else(|error| {
                panic!(
                    "encode converted prompt for {}: {error}",
                    reference_case.name
                )
            })
            .into_iter()
            .map(|token| i32::try_from(token).expect("token id fits i32"))
            .collect();
        assert_eq!(
            converted_ids, reference_case.unexpanded_input_ids,
            "source and converted tokenizer ids diverged for {}",
            reference_case.name
        );
        let converted_u32: Vec<u32> = converted_ids
            .iter()
            .map(|&token| u32::try_from(token).expect("token id is non-negative"))
            .collect();
        let decoded = tokenizer
            .decode(&converted_u32, false)
            .unwrap_or_else(|error| panic!("decode converted prompt: {error}"));
        assert_eq!(
            decoded, converted_prompt,
            "converted tokenizer round-trip diverged for {}",
            reference_case.name
        );
        let images: Vec<DynamicImage> = reference_case
            .image_transforms
            .iter()
            .map(|transform| transformed_image(&image, transform))
            .collect();
        let preprocessing_started = Instant::now();
        let capture = preprocessor
            .prepare_with_reference_diagnostics(&converted_ids, &images)
            .unwrap_or_else(|error| {
                panic!(
                    "prepare LLaVA reference case {}: {error}",
                    reference_case.name
                )
            });
        let preprocessing_seconds = preprocessing_started.elapsed().as_secs_f64();
        let run = engine
            .capture(
                &capture.prepared,
                reference.kv_selection.width,
                reference.generation.max_new_tokens,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "run LLaVA reference case {} on {device}: {error}",
                    reference_case.name
                )
            });

        let mut arrays = serde_json::Map::new();
        if let Some(pixel_values) = &capture.pixel_values {
            arrays.insert(
                "processor_pixel_values".to_string(),
                write_tensor(
                    &out,
                    &reference_case.name,
                    "processor_pixel_values",
                    pixel_values,
                ),
            );
        }
        arrays.insert(
            "expanded_token_ids".to_string(),
            write_raw(
                &out,
                &reference_case.name,
                "expanded_token_ids",
                &i32_bytes(&capture.prepared.token_ids),
                "int32",
                &[1, capture.prepared.sequence_len],
            ),
        );
        let prepared_positions =
            positions(&capture.prepared.positions, capture.prepared.sequence_len);
        arrays.insert(
            "positions".to_string(),
            write_raw(
                &out,
                &reference_case.name,
                "positions",
                &i32_bytes(&prepared_positions),
                "int32",
                &[1, capture.prepared.sequence_len],
            ),
        );
        let attention_mask = vec![1i32; capture.prepared.sequence_len];
        arrays.insert(
            "attention_mask".to_string(),
            write_raw(
                &out,
                &reference_case.name,
                "attention_mask",
                &i32_bytes(&attention_mask),
                "int32",
                &[1, capture.prepared.sequence_len],
            ),
        );
        if let Some(projected) = &capture.projected_image_features {
            let selected = capture
                .selected_vision_features
                .as_ref()
                .expect("projected features require selected vision features");
            for (index, hidden_state) in capture.vision_hidden_states.iter().enumerate() {
                let stage = format!("vision_hidden_state_{index:02}");
                arrays.insert(
                    stage.clone(),
                    write_tensor(&out, &reference_case.name, &stage, hidden_state),
                );
            }
            assert_eq!(
                capture.vision_block0_states.len(),
                VISION_BLOCK0_STAGES.len(),
                "SigLIP diagnostics must capture every first-block sub-stage"
            );
            for (stage, state) in VISION_BLOCK0_STAGES
                .iter()
                .zip(&capture.vision_block0_states)
            {
                arrays.insert(
                    (*stage).to_string(),
                    write_tensor(&out, &reference_case.name, stage, state),
                );
            }
            arrays.insert(
                "selected_vision_features".to_string(),
                write_tensor(
                    &out,
                    &reference_case.name,
                    "selected_vision_features",
                    selected,
                ),
            );
            arrays.insert(
                "projected_image_features".to_string(),
                write_tensor(
                    &out,
                    &reference_case.name,
                    "projected_image_features",
                    projected,
                ),
            );
        }
        arrays.insert(
            "merged_embeddings".to_string(),
            write_tensor(
                &out,
                &reference_case.name,
                "merged_embeddings",
                &capture.prepared.embeddings,
            ),
        );
        arrays.insert(
            "first_prefill_logits".to_string(),
            write_raw(
                &out,
                &reference_case.name,
                "first_prefill_logits",
                &f32_bytes(&run.prefill.logits),
                "float32",
                &[run.prefill.logits.len()],
            ),
        );
        arrays.insert(
            "selected_kv".to_string(),
            write_raw(
                &out,
                &reference_case.name,
                "selected_kv",
                &f32_bytes(&run.prefill.kv),
                "float32",
                &[run.prefill.layers, 2, run.prefill.kv_width],
            ),
        );
        arrays.insert(
            "greedy_tokens".to_string(),
            write_raw(
                &out,
                &reference_case.name,
                "greedy_tokens",
                &i32_bytes(&run.tokens),
                "int32",
                &[run.tokens.len()],
            ),
        );
        captured_cases.push(json!({
            "name": reference_case.name,
            "image_count": reference_case.image_count,
            "image_transforms": reference_case.image_transforms,
            "arrays": arrays,
            "timings": {
                "host_preprocessing_seconds": preprocessing_seconds,
                "prefill_seconds": run.prefill_seconds,
                "decode_seconds": run.decode_seconds,
                "decode_tokens_per_second": if run.tokens.len() > 1 {
                    (run.tokens.len() - 1) as f64 / run.decode_seconds
                } else {
                    0.0
                },
            },
        }));
    }

    // Required negative cases exercise the same public boundaries, but retain
    // only their stable rejected outcome/category in the manifest.
    let _malformed = preprocessor
        .prepare(&[151646, 151646], std::slice::from_ref(&image))
        .expect_err("two placeholders for one image must be rejected");
    let overflow_tokens = vec![1i32; context_capacity + 1];
    let overflow_prepared = preprocessor
        .prepare(&overflow_tokens, &[])
        .expect("host model capacity exceeds the IREE test bucket");
    let _overflow = engine
        .capture(
            &overflow_prepared,
            reference.kv_selection.width,
            reference.generation.max_new_tokens,
        )
        .expect_err("prepared prompt beyond IREE bucket must be rejected");

    let manifest = json!({
        "schema": 1,
        "producer": "mlxcel-xla-diagnostics",
        "device": device,
        "host_preprocessor_device": runtime.device.to_string(),
        "host_compute": {
            "vision_projector": "float32",
            "prompt_embedding_lookup": "bfloat16",
            "mlx_enable_tf32": std::env::var("MLX_ENABLE_TF32")
                .unwrap_or_else(|_| "1 (MLX default)".to_string()),
        },
        "model_ownership": {
            "host": "processor, vision tower, projector, and text embedding table only",
            "iree": "single resident text decoder bundle used for prefill, KV capture, and decode",
            "duplicate_text_decoder": false,
        },
        "context_capacity": context_capacity,
        "converted_checkpoint": {
            "artifact_manifest": reference.converted_checkpoint.artifact_manifest,
        },
        "image_fixture": reference.image_fixture,
        "kv_selection": {
            "position": "last_effective_prompt",
            "kv_head": 0,
            "width": reference.kv_selection.width,
        },
        "timings": {
            "host_component_load_seconds": host_load_seconds,
            "iree_compile_and_load_seconds": compile_load_seconds,
        },
        "host_peak_rss_kib": peak_rss_kib(),
        "runtime_memory": {
            "mlx_peak_device_bytes": mlxcel_core::get_peak_memory(),
            "linux_mem_available_before_kib": mem_available_before_kib,
            "linux_mem_available_after_kib": mem_available_kib(),
            "iree_device_bytes": Value::Null,
            "iree_device_note": mlxcel_xla::llava_diagnostic_device_memory_note(&device),
        },
        "negative_cases": {
            "malformed_placeholder": {
                "passed": true,
                "outcome": "rejected",
                "category": "placeholder_count_mismatch",
            },
            "context_overflow": {
                "passed": true,
                "outcome": "rejected",
                "category": "context_capacity_exceeded",
            },
        },
        "cases": captured_cases,
    });
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).expect("serialize XLA manifest"),
    )
    .unwrap_or_else(|error| panic!("write XLA manifest: {error}"));
    println!("{}", out.join("manifest.json").display());
}
