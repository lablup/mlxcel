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

//! Actual-checkpoint eager MLX vs IREE Qwen3-VL DeepStack oracle.
//!
//! The qualified production host processor creates one patch payload that is
//! consumed by both vision implementations. The gate compares the main merger,
//! every ordered DeepStack merger, post-injection language hidden states, and
//! final-position logits. It also mutates the actual captured branches to prove
//! that dropping or zeroing one branch is detected.

use std::fs;
use std::path::{Path, PathBuf};

use image::DynamicImage;
use mlxcel::{
    HostMultimodalPreprocessor, LoadedModel, Qwen3VlIreeHostPreprocessor, initialize_runtime,
    load_model,
};
use mlxcel_xla::Qwen3VlDeepStackDiagnosticEngine;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy)]
struct Tolerance {
    atol: f32,
    rtol: f32,
}

#[derive(Debug, Clone, Copy)]
struct ComparisonStats {
    max_absolute: f32,
    max_relative: f32,
    failures: usize,
    non_finite: usize,
}

fn argument(flag: &str) -> Option<String> {
    let args = std::env::args().collect::<Vec<_>>();
    args.iter()
        .position(|argument| argument == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn required_path(flag: &str) -> PathBuf {
    argument(flag)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("missing required {flag}"))
}

fn usize_argument(flag: &str, default: usize) -> usize {
    argument(flag)
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{flag} must be an unsigned integer"))
        })
        .unwrap_or(default)
}

fn f32_argument(flag: &str, default: f32) -> f32 {
    argument(flag)
        .map(|value| {
            value
                .parse::<f32>()
                .unwrap_or_else(|_| panic!("{flag} must be a finite number"))
        })
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(default)
}

fn mlx_f32(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect()
}

fn comparison_stats(
    actual: &[f32],
    expected: &[f32],
    tolerance: Tolerance,
) -> Result<ComparisonStats, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "element count differs: actual={}, expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let mut stats = ComparisonStats {
        max_absolute: 0.0,
        max_relative: 0.0,
        failures: 0,
        non_finite: 0,
    };
    for (&observed, &reference) in actual.iter().zip(expected) {
        if !observed.is_finite() || !reference.is_finite() {
            stats.failures += 1;
            stats.non_finite += 1;
            continue;
        }
        let absolute = (observed - reference).abs();
        let relative = absolute / reference.abs().max(f32::MIN_POSITIVE);
        stats.max_absolute = stats.max_absolute.max(absolute);
        stats.max_relative = stats.max_relative.max(relative);
        if absolute > tolerance.atol + tolerance.rtol * reference.abs() {
            stats.failures += 1;
        }
    }
    Ok(stats)
}

fn compare_stage(
    reports: &mut Vec<Value>,
    stage: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: Tolerance,
) -> Result<(), String> {
    let stats = comparison_stats(actual, expected, tolerance)
        .map_err(|error| format!("{stage}: {error}"))?;
    reports.push(json!({
        "stage": stage,
        "elements": actual.len(),
        "atol": tolerance.atol,
        "rtol": tolerance.rtol,
        "max_absolute": stats.max_absolute,
        "max_relative": stats.max_relative,
        "failures": stats.failures,
        "non_finite": stats.non_finite,
        "passed": stats.failures == 0,
    }));
    if stats.failures == 0 {
        Ok(())
    } else {
        Err(format!(
            "{stage}: {} values exceeded tolerance (max_abs={}, max_rel={})",
            stats.failures, stats.max_absolute, stats.max_relative
        ))
    }
}

fn compare_vision_branches(
    reports: &mut Vec<Value>,
    actual: &[Vec<f32>],
    expected: &[Vec<f32>],
    tolerance: Tolerance,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "DeepStack branch count differs: actual={}, expected={}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        compare_stage(
            reports,
            &format!("vision.deepstack_merger_{index}"),
            actual,
            expected,
            tolerance,
        )?;
    }
    Ok(())
}

fn patch_size(model: &Path) -> usize {
    let config: Value = serde_json::from_slice(
        &fs::read(model.join("config.json")).expect("read Qwen3-VL config.json"),
    )
    .expect("parse Qwen3-VL config.json");
    config["vision_config"]["patch_size"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .expect("vision_config.patch_size must be a positive integer")
}

fn main() {
    let model_path = required_path("--model");
    let image_path = required_path("--image");
    let device = argument("--device").unwrap_or_else(|| "local-task".to_string());
    let context_capacity = usize_argument("--context-capacity", 256);
    let tolerance = Tolerance {
        atol: f32_argument("--atol", 0.08),
        rtol: f32_argument("--rtol", 0.08),
    };
    let report_path = argument("--report").map(PathBuf::from);
    let _runtime = initialize_runtime();
    let image = image::open(&image_path)
        .unwrap_or_else(|error| panic!("open {}: {error}", image_path.display()))
        .into_rgb8();
    let images = [DynamicImage::ImageRgb8(image)];

    let production_preprocessor = Qwen3VlIreeHostPreprocessor::load(&model_path, &device)
        .unwrap_or_else(|error| panic!("load Qwen3-VL IREE host preprocessor: {error}"));
    let vision_capture = production_preprocessor
        .capture_vision(&images)
        .unwrap_or_else(|error| panic!("capture Qwen3-VL IREE vision: {error}"));

    let (loaded, _) = load_model(&model_path)
        .unwrap_or_else(|error| panic!("load eager Qwen3-VL checkpoint: {error}"));
    let LoadedModel::Qwen3VL(model) = loaded else {
        panic!("checkpoint did not load as dense Qwen3-VL");
    };
    let patch_width = 3 * patch_size(&model_path).pow(2);
    assert_eq!(
        vision_capture.patch_values.len() % patch_width,
        0,
        "processor patch payload must have complete rows"
    );
    let patch_rows = vision_capture.patch_values.len() / patch_width;
    let pixels = mlxcel_core::from_slice_f32(
        &vision_capture.patch_values,
        &[patch_rows as i32, patch_width as i32],
    );
    let eager_vision = model
        .vision_encoder
        .forward_with_grid(&pixels, &vision_capture.grids);
    let eager_main = mlx_f32(&eager_vision.hidden_states);
    let eager_branches = eager_vision
        .deepstack_features
        .iter()
        .map(|feature| mlx_f32(feature))
        .collect::<Vec<_>>();

    let mut reports = Vec::new();
    compare_stage(
        &mut reports,
        "vision.main_merger",
        &vision_capture.projection.values,
        &eager_main,
        tolerance,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    compare_vision_branches(
        &mut reports,
        &vision_capture.projection.deepstack_values,
        &eager_branches,
        tolerance,
    )
    .unwrap_or_else(|error| panic!("{error}"));

    let mut dropped = vision_capture.projection.deepstack_values.clone();
    dropped
        .pop()
        .expect("actual checkpoint must expose DeepStack");
    assert!(
        compare_vision_branches(&mut Vec::new(), &dropped, &eager_branches, tolerance).is_err(),
        "negative oracle accepted a dropped actual-checkpoint DeepStack branch"
    );
    let mut zeroed = vision_capture.projection.deepstack_values.clone();
    zeroed[0].fill(0.0);
    assert!(
        compare_vision_branches(&mut Vec::new(), &zeroed, &eager_branches, tolerance).is_err(),
        "negative oracle accepted a zeroed actual-checkpoint DeepStack branch"
    );

    let unexpanded_tokens = vec![1, model.vision_start_token_id, model.image_token_id, 2];
    let prepared = production_preprocessor
        .prepare_deepstack(&unexpanded_tokens, &images)
        .unwrap_or_else(|error| panic!("prepare production DeepStack prefill: {error}"))
        .expect("Qwen3-VL preprocessor must return a DeepStack payload");
    assert!(
        prepared.prepared().sequence_len <= context_capacity,
        "prepared sequence length {} exceeds diagnostic context capacity {context_capacity}",
        prepared.prepared().sequence_len
    );
    let input_ids = mlxcel_core::from_slice_i32(
        &prepared.prepared().token_ids,
        &[1, prepared.prepared().sequence_len as i32],
    );
    let eager_embeddings = model.get_input_embeddings(&input_ids, &pixels, &vision_capture.grids);
    let mut caches = model.text_model.make_caches();
    let eager_language = model.text_model.forward_deepstack_diagnostics(
        &input_ids,
        &eager_embeddings.inputs_embeds,
        &mut caches,
        eager_embeddings.attention_mask_4d.as_deref(),
    );

    let mut iree_language =
        Qwen3VlDeepStackDiagnosticEngine::load(&model_path, &device, context_capacity)
            .unwrap_or_else(|error| panic!("load Qwen3-VL DeepStack diagnostics: {error}"));
    let iree_language = iree_language
        .capture(&prepared)
        .unwrap_or_else(|error| panic!("capture Qwen3-VL DeepStack diagnostics: {error}"));
    assert_eq!(
        eager_language.post_injection_hidden_states.len(),
        iree_language.deepstack_layers,
        "eager and IREE post-injection layer counts differ"
    );
    let sequence_len = prepared.prepared().sequence_len;
    let hidden_size = iree_language.hidden_size;
    for (layer, eager) in eager_language
        .post_injection_hidden_states
        .iter()
        .enumerate()
    {
        let target_layer = iree_language.target_layer_indices[layer];
        let eager = mlx_f32(eager);
        let layer_start = layer * iree_language.context_capacity * hidden_size;
        let actual = &iree_language.post_injection_hidden_states
            [layer_start..layer_start + sequence_len * hidden_size];
        compare_stage(
            &mut reports,
            &format!("language.post_injection_layer_{target_layer}"),
            actual,
            &eager,
            tolerance,
        )
        .unwrap_or_else(|error| panic!("{error}"));
    }

    let eager_logits = mlx_f32(&eager_language.logits);
    let vocab = iree_language.logits.len();
    let eager_last = &eager_logits[(sequence_len - 1) * vocab..sequence_len * vocab];
    compare_stage(
        &mut reports,
        "language.final_position_logits",
        &iree_language.logits,
        eager_last,
        tolerance,
    )
    .unwrap_or_else(|error| panic!("{error}"));

    let report = json!({
        "schema": 1,
        "model": model_path,
        "image": image_path,
        "device": device,
        "context_capacity": context_capacity,
        "grid_thw": vision_capture.grids,
        "vision_shape": vision_capture.projection.shape,
        "deepstack_branches": eager_branches.len(),
        "deepstack_target_language_layers": iree_language.target_layer_indices,
        "negative_dropped_branch_detected": true,
        "negative_zeroed_branch_detected": true,
        "comparisons": reports,
        "passed": true,
    });
    let rendered = serde_json::to_string_pretty(&report).expect("serialize oracle report");
    if let Some(path) = report_path {
        fs::write(&path, &rendered)
            .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    }
    println!("{rendered}");
}
