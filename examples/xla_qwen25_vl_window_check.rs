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

//! Actual-checkpoint eager MLX vs IREE Qwen2.5-VL strict oracle.
//!
//! One production processor payload drives both vision implementations. The
//! gate compares window planning, reordered patch embeddings, representative
//! window attention, every configured full-attention layer, restoration,
//! prepared M-RoPE, and final-position logits. Negative runs force Qwen2-style
//! full attention, identity permutation, and zero vision positions.

use std::fs;
use std::path::{Path, PathBuf};

use image::DynamicImage;
use mlxcel::{
    HostMultimodalPreprocessor, LoadedModel, Qwen2VlIreeHostPreprocessor, initialize_runtime,
    load_model,
};
use mlxcel_core::session::{OwnedTensor, PreparedPositions, PreparedTensorDType};
use mlxcel_xla::{
    IreeQwen25VlDiagnosticProjector, Qwen25VlDiagnosticMutation, Qwen25VlLanguageDiagnosticEngine,
};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy)]
struct Tolerance {
    atol: f32,
    rtol: f32,
}

#[derive(Debug, Clone, Copy)]
struct ComparisonStats {
    max_absolute: f32,
    max_absolute_index: Option<usize>,
    actual_at_max_absolute: Option<f32>,
    expected_at_max_absolute: Option<f32>,
    max_relative: f32,
    actual_rms: f64,
    expected_rms: f64,
    error_rms: f64,
    failures: usize,
    non_finite: usize,
}

fn argument(flag: &str) -> Option<String> {
    arguments(flag).into_iter().next()
}

fn arguments(flag: &str) -> Vec<String> {
    let args = std::env::args().collect::<Vec<_>>();
    args.iter()
        .enumerate()
        .filter(|(_, argument)| argument.as_str() == flag)
        .filter_map(|(index, _)| args.get(index + 1).cloned())
        .collect()
}

fn required_paths(flag: &str) -> Vec<PathBuf> {
    let paths = arguments(flag)
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    assert!(!paths.is_empty(), "missing required {flag}");
    paths
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
                .unwrap_or_else(|_| panic!("{flag} must be a finite non-negative number"))
        })
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(default)
}

fn mlx_f32(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect()
}

fn mlx_i32(array: &mlxcel_core::MlxArray) -> Vec<i32> {
    mlxcel_core::eval(array);
    mlxcel_core::array_to_raw_bytes(array)
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("four-byte i32 chunk")))
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct SnapshotStats {
    max_abs: f32,
    finite_nonzero: usize,
    zero: usize,
    non_finite: usize,
    f16_square_overflow_candidates: usize,
}

fn snapshot_stats(values: &[f32]) -> SnapshotStats {
    // Largest finite F16 value is 65,504, so squaring a value above its
    // square root in F16 produces infinity before the RMS mean is evaluated.
    const F16_SQRT_MAX: f32 = 255.9375;
    let mut stats = SnapshotStats {
        max_abs: 0.0,
        finite_nonzero: 0,
        zero: 0,
        non_finite: 0,
        f16_square_overflow_candidates: 0,
    };
    for &value in values {
        if !value.is_finite() {
            stats.non_finite += 1;
            continue;
        }
        let absolute = value.abs();
        stats.max_abs = stats.max_abs.max(absolute);
        if value == 0.0 {
            stats.zero += 1;
        } else {
            stats.finite_nonzero += 1;
        }
        if absolute > F16_SQRT_MAX {
            stats.f16_square_overflow_candidates += 1;
        }
    }
    stats
}

fn owned_f32(tensor: &OwnedTensor, label: &str) -> Vec<f32> {
    assert_eq!(
        tensor.dtype,
        PreparedTensorDType::Float32,
        "{label} must be float32"
    );
    tensor
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect()
}

fn owned_i32(tensor: &OwnedTensor, label: &str) -> Vec<i32> {
    assert_eq!(
        tensor.dtype,
        PreparedTensorDType::Int32,
        "{label} must be int32"
    );
    tensor
        .bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("four-byte i32 chunk")))
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
        max_absolute_index: None,
        actual_at_max_absolute: None,
        expected_at_max_absolute: None,
        max_relative: 0.0,
        actual_rms: 0.0,
        expected_rms: 0.0,
        error_rms: 0.0,
        failures: 0,
        non_finite: 0,
    };
    let mut finite = 0usize;
    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        if !observed.is_finite() || !reference.is_finite() {
            stats.failures += 1;
            stats.non_finite += 1;
            continue;
        }
        let absolute = (observed - reference).abs();
        let relative = absolute / reference.abs().max(f32::MIN_POSITIVE);
        stats.actual_rms += f64::from(observed).powi(2);
        stats.expected_rms += f64::from(reference).powi(2);
        stats.error_rms += f64::from(observed - reference).powi(2);
        finite += 1;
        if stats.max_absolute_index.is_none() || absolute > stats.max_absolute {
            stats.max_absolute = absolute;
            stats.max_absolute_index = Some(index);
            stats.actual_at_max_absolute = Some(observed);
            stats.expected_at_max_absolute = Some(reference);
        }
        stats.max_relative = stats.max_relative.max(relative);
        if absolute > tolerance.atol + tolerance.rtol * reference.abs() {
            stats.failures += 1;
        }
    }
    if finite > 0 {
        let denominator = finite as f64;
        stats.actual_rms = (stats.actual_rms / denominator).sqrt();
        stats.expected_rms = (stats.expected_rms / denominator).sqrt();
        stats.error_rms = (stats.error_rms / denominator).sqrt();
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
        "max_absolute_index": stats.max_absolute_index,
        "actual_at_max_absolute": stats.actual_at_max_absolute,
        "expected_at_max_absolute": stats.expected_at_max_absolute,
        "max_relative": stats.max_relative,
        "actual_rms": stats.actual_rms,
        "expected_rms": stats.expected_rms,
        "error_rms": stats.error_rms,
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

fn record_stage(
    reports: &mut Vec<Value>,
    failures: &mut Vec<String>,
    stage: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: Tolerance,
) {
    if let Err(error) = compare_stage(reports, stage, actual, expected, tolerance) {
        failures.push(error);
    }
}

fn emit_report(path: Option<&Path>, report: &Value) {
    let rendered = serde_json::to_string_pretty(report).expect("serialize oracle report");
    if let Some(path) = path {
        fs::write(path, &rendered)
            .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    }
    println!("{rendered}");
}

fn vision_patch_shape(model: &Path, values: &[f32], grids: &[(i32, i32, i32)]) -> [i32; 2] {
    let config: Value = serde_json::from_slice(
        &fs::read(model.join("config.json")).expect("read Qwen2.5-VL config.json"),
    )
    .expect("parse Qwen2.5-VL config.json");
    let vision = &config["vision_config"];
    let temporal = vision["temporal_patch_size"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .expect("vision temporal_patch_size");
    let rows = grids
        .iter()
        .map(|&(t, h, w)| {
            usize::try_from(t)
                .expect("positive grid temporal")
                .checked_mul(usize::try_from(h).expect("positive grid height"))
                .and_then(|count| {
                    count.checked_mul(usize::try_from(w).expect("positive grid width"))
                })
                .and_then(|count| count.checked_mul(temporal))
                .expect("vision patch row count")
        })
        .sum::<usize>();
    assert_eq!(values.len() % rows, 0, "processor patch rows must be dense");
    [
        i32::try_from(rows).expect("patch rows fit i32"),
        i32::try_from(values.len() / rows).expect("patch width fits i32"),
    ]
}

fn usize_plan(values: &[i32], label: &str) -> Vec<usize> {
    values
        .iter()
        .map(|&value| usize::try_from(value).unwrap_or_else(|_| panic!("{label} contains {value}")))
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}

fn main() {
    let model_path = required_paths("--model")
        .into_iter()
        .next()
        .expect("--model was checked");
    let image_paths = required_paths("--image");
    let device = argument("--device").unwrap_or_else(|| "local-task".to_string());
    let context_capacity = usize_argument("--context-capacity", 256);
    let tolerance = Tolerance {
        atol: f32_argument("--atol", 0.08),
        rtol: f32_argument("--rtol", 0.08),
    };
    let report_path = argument("--report").map(PathBuf::from);
    let _runtime = initialize_runtime();
    let images = image_paths
        .iter()
        .map(|path| {
            image::open(path)
                .unwrap_or_else(|error| panic!("open {}: {error}", path.display()))
                .into_rgb8()
        })
        .map(DynamicImage::ImageRgb8)
        .collect::<Vec<_>>();

    let production_preprocessor = Qwen2VlIreeHostPreprocessor::load(&model_path, &device)
        .unwrap_or_else(|error| panic!("load Qwen2.5-VL production preprocessor: {error}"));
    let processor = production_preprocessor
        .capture_processor_inputs(&images)
        .unwrap_or_else(|error| panic!("capture production processor inputs: {error}"));

    let (loaded, _) = load_model(&model_path)
        .unwrap_or_else(|error| panic!("load eager Qwen2.5-VL checkpoint: {error}"));
    let LoadedModel::Qwen25VL(model) = loaded else {
        panic!("checkpoint did not load as Qwen2.5-VL");
    };
    let mut unexpanded_tokens = vec![1];
    for _ in &images {
        unexpanded_tokens.extend([model.vision_start_token_id, model.image_token_id]);
    }
    unexpanded_tokens.push(2);
    let prepared = production_preprocessor
        .prepare(&unexpanded_tokens, &images)
        .unwrap_or_else(|error| panic!("prepare production Qwen2.5-VL prefill: {error}"));
    assert!(
        prepared.sequence_len <= context_capacity,
        "prepared sequence length {} exceeds diagnostic capacity {context_capacity}",
        prepared.sequence_len
    );
    let input_ids = mlxcel_core::from_slice_i32(
        &prepared.token_ids,
        &[
            1,
            i32::try_from(prepared.sequence_len).expect("sequence length fits i32"),
        ],
    );
    let text_embeddings = model.text_model.get_embed_tokens(&input_ids);
    let patch_shape = vision_patch_shape(&model_path, &processor.patch_values, &processor.grids);
    let pixels = mlxcel_core::from_slice_f32(&processor.patch_values, &patch_shape);
    let pixels = mlxcel_core::astype(&pixels, mlxcel_core::array_dtype(&text_embeddings));
    let eager_vision = model
        .vision_encoder
        .forward_with_grid_diagnostics(&pixels, &processor.grids);
    let eager_mlp_gate_projection_dense_f32_control = eager_vision
        .substage_probe_layer_mlp_gate_projection_dense_f32_control
        .as_slice();
    let eager_mlp_up_projection_dense_f32_control = eager_vision
        .substage_probe_layer_mlp_up_projection_dense_f32_control
        .as_slice();
    // The producer exports host-owned F32 values at each substage boundary.
    // Reject a vacuous capture immediately so the strict oracle cannot report
    // a numeric false positive.
    let eager_substage_probe = [
        ("input", eager_vision.substage_probe_layer_input.as_slice()),
        ("norm1", eager_vision.substage_probe_layer_norm1.as_slice()),
        ("query", eager_vision.substage_probe_layer_query.as_slice()),
        ("key", eager_vision.substage_probe_layer_key.as_slice()),
        ("value", eager_vision.substage_probe_layer_value.as_slice()),
        (
            "attention_context",
            eager_vision
                .substage_probe_layer_attention_context
                .as_slice(),
        ),
        (
            "attention",
            eager_vision.substage_probe_layer_attention.as_slice(),
        ),
        (
            "post_attention_residual",
            eager_vision
                .substage_probe_layer_post_attention_residual
                .as_slice(),
        ),
        ("norm2", eager_vision.substage_probe_layer_norm2.as_slice()),
        (
            "mlp_gate_projection",
            eager_vision
                .substage_probe_layer_mlp_gate_projection
                .as_slice(),
        ),
        (
            "mlp_gate_activation",
            eager_vision
                .substage_probe_layer_mlp_gate_activation
                .as_slice(),
        ),
        (
            "mlp_up_projection",
            eager_vision
                .substage_probe_layer_mlp_up_projection
                .as_slice(),
        ),
        (
            "mlp_gated_product",
            eager_vision
                .substage_probe_layer_mlp_gated_product
                .as_slice(),
        ),
        (
            "mlp_down_projection",
            eager_vision
                .substage_probe_layer_mlp_down_projection
                .as_slice(),
        ),
    ];
    for (stage, values) in eager_substage_probe
        .iter()
        .filter(|(stage, _)| matches!(*stage, "norm1" | "norm2"))
    {
        let stats = snapshot_stats(values);
        assert!(
            stats.finite_nonzero > 0,
            "eager substage probe {stage} has no finite non-zero values before IREE launch: \
             {stats:?}; layer input stats: {:?}. F16 RMSNorm squares values before its mean, \
             so any input magnitude above sqrt(65504) can zero an entire normalized row",
            snapshot_stats(eager_vision.substage_probe_layer_input.as_slice())
        );
    }

    let mut iree_projector = IreeQwen25VlDiagnosticProjector::load(&model_path, &device)
        .unwrap_or_else(|error| panic!("load Qwen2.5-VL IREE diagnostics: {error}"));
    let iree_vision = iree_projector
        .capture(
            &processor.patch_values,
            &processor.grids,
            Qwen25VlDiagnosticMutation::None,
        )
        .unwrap_or_else(|error| panic!("capture Qwen2.5-VL IREE diagnostics: {error}"));
    let iree_norm2_mlp_projection_controls = model
        .vision_encoder
        .f16_input_dense_f32_mlp_projection_controls(
            iree_vision.substage_probe_layer_index,
            &iree_vision.substage_probe_layer_norm2,
            iree_vision.patch_tokens,
        )
        .unwrap_or_else(|error| {
            panic!("project captured IREE norm2 through eager Qwen2.5-VL MLP weights: {error}")
        });

    assert_eq!(
        iree_vision.window_index,
        usize_plan(&eager_vision.window_index, "eager window_index"),
        "window_index differs"
    );
    assert_eq!(
        iree_vision.restore_indices,
        usize_plan(&eager_vision.restore_indices, "eager restore_indices"),
        "restoration permutation differs"
    );
    assert_eq!(
        iree_vision.packed_cu_seqlens,
        usize_plan(&eager_vision.packed_cu_seqlens, "eager packed cu lengths"),
        "full-attention cumulative lengths differ"
    );
    assert_eq!(
        iree_vision.window_cu_seqlens,
        usize_plan(&eager_vision.window_cu_seqlens, "eager window cu lengths"),
        "window-attention cumulative lengths differ"
    );
    assert!(
        iree_vision.attention_mask_audit.full_window_differences > 0,
        "strict oracle fixture does not structurally distinguish window and full attention"
    );
    assert_eq!(
        iree_vision.attention_mask_audit.cross_media_full_bias_leaks, 0,
        "full-attention bias leaks across media"
    );
    assert_eq!(
        iree_vision
            .attention_mask_audit
            .active_padding_full_bias_leaks,
        0,
        "full-attention bias connects active and padded bucket rows"
    );
    assert_eq!(
        iree_vision
            .attention_mask_audit
            .invalid_padded_full_bias_entries,
        0,
        "padded bucket queries must expose only one finite self key"
    );
    assert!(
        iree_vision
            .window_index
            .iter()
            .enumerate()
            .any(|(index, &position)| index != position),
        "strict oracle fixture must exercise a non-identity window permutation"
    );
    assert!(
        iree_vision.window_cu_seqlens.len() > iree_vision.packed_cu_seqlens.len(),
        "strict oracle fixture must span multiple local attention windows"
    );

    let mut reports = Vec::new();
    let mut comparison_failures = Vec::new();
    record_stage(
        &mut reports,
        &mut comparison_failures,
        "vision.reordered_patch_embedding",
        &iree_vision.reordered_patch_embedding,
        &mlx_f32(&eager_vision.reordered_patch_embedding),
        tolerance,
    );
    assert_eq!(
        iree_vision.window_layer_index, eager_vision.window_layer_index,
        "representative window layer differs"
    );
    record_stage(
        &mut reports,
        &mut comparison_failures,
        &format!(
            "vision.post_window_layer_{}",
            iree_vision.window_layer_index
        ),
        &iree_vision.post_window_layer,
        &mlx_f32(&eager_vision.post_window_layer),
        tolerance,
    );
    assert_eq!(
        iree_vision.full_layer_indices, eager_vision.full_layer_indices,
        "configured full-attention layers differ"
    );
    assert_eq!(
        iree_vision.post_full_layers.len(),
        eager_vision.post_full_layers.len(),
        "captured full-attention layer count differs"
    );
    assert_eq!(
        iree_vision.final_interval_layer_indices, eager_vision.final_interval_layer_indices,
        "final diagnostic interval differs"
    );
    assert_eq!(
        iree_vision.post_final_interval_layers.len(),
        eager_vision.post_final_interval_layers.len(),
        "captured final diagnostic interval count differs"
    );
    assert_eq!(
        iree_vision.diagnostic_layer_indices, eager_vision.diagnostic_layer_indices,
        "diagnostic layer interval differs"
    );
    assert_eq!(
        iree_vision.post_diagnostic_layers.len(),
        eager_vision.post_diagnostic_layers.len(),
        "captured diagnostic layer count differs"
    );
    let target_full_layer_index = *iree_vision
        .full_layer_indices
        .last()
        .expect("IREE diagnostics require a full-attention layer");
    assert_eq!(
        Some(&target_full_layer_index),
        eager_vision.full_layer_indices.last(),
        "target full-attention layer differs"
    );
    assert_eq!(
        iree_vision.substage_probe_layer_index, eager_vision.substage_probe_layer_index,
        "substage probe layer differs"
    );
    let substage_probe_layer_index = iree_vision.substage_probe_layer_index;
    for (capture_index, (&layer, (actual, expected))) in iree_vision
        .full_layer_indices
        .iter()
        .zip(
            iree_vision
                .post_full_layers
                .iter()
                .zip(&eager_vision.post_full_layers),
        )
        .enumerate()
        .filter(|&(_, (&layer, _))| layer != target_full_layer_index)
    {
        record_stage(
            &mut reports,
            &mut comparison_failures,
            &format!("vision.post_full_layer_{layer}_capture_{capture_index}"),
            actual,
            &mlx_f32(expected),
            tolerance,
        );
    }
    for (&layer, (actual, expected)) in iree_vision
        .final_interval_layer_indices
        .iter()
        .zip(
            iree_vision
                .post_final_interval_layers
                .iter()
                .zip(&eager_vision.post_final_interval_layers),
        )
        .filter(|&(&layer, _)| layer != target_full_layer_index)
    {
        record_stage(
            &mut reports,
            &mut comparison_failures,
            &format!("vision.post_layer_{layer}_final_interval"),
            actual,
            &mlx_f32(expected),
            tolerance,
        );
    }
    for (&layer, (actual, expected)) in iree_vision.diagnostic_layer_indices.iter().zip(
        iree_vision
            .post_diagnostic_layers
            .iter()
            .zip(&eager_vision.post_diagnostic_layers),
    ) {
        record_stage(
            &mut reports,
            &mut comparison_failures,
            &format!("vision.post_layer_{layer}_diagnostic"),
            actual,
            &mlx_f32(expected),
            tolerance,
        );
    }
    for ((stage, actual), (eager_stage, expected)) in [
        ("input", iree_vision.substage_probe_layer_input.as_slice()),
        ("norm1", iree_vision.substage_probe_layer_norm1.as_slice()),
        ("query", iree_vision.substage_probe_layer_query.as_slice()),
        ("key", iree_vision.substage_probe_layer_key.as_slice()),
        ("value", iree_vision.substage_probe_layer_value.as_slice()),
        (
            "attention_context",
            iree_vision
                .substage_probe_layer_attention_context
                .as_slice(),
        ),
        (
            "attention",
            iree_vision.substage_probe_layer_attention.as_slice(),
        ),
        (
            "post_attention_residual",
            iree_vision
                .substage_probe_layer_post_attention_residual
                .as_slice(),
        ),
        ("norm2", iree_vision.substage_probe_layer_norm2.as_slice()),
        (
            "mlp_gate_projection",
            iree_vision
                .substage_probe_layer_mlp_gate_projection
                .as_slice(),
        ),
        (
            "mlp_gate_activation",
            iree_vision
                .substage_probe_layer_mlp_gate_activation
                .as_slice(),
        ),
        (
            "mlp_up_projection",
            iree_vision
                .substage_probe_layer_mlp_up_projection
                .as_slice(),
        ),
        (
            "mlp_gated_product",
            iree_vision
                .substage_probe_layer_mlp_gated_product
                .as_slice(),
        ),
        (
            "mlp_down_projection",
            iree_vision
                .substage_probe_layer_mlp_down_projection
                .as_slice(),
        ),
    ]
    .into_iter()
    .zip(&eager_substage_probe)
    {
        assert_eq!(stage, *eager_stage, "substage probe order differs");
        record_stage(
            &mut reports,
            &mut comparison_failures,
            &format!("vision.layer_{substage_probe_layer_index}.{stage}"),
            actual,
            expected,
            tolerance,
        );
    }
    record_stage(
        &mut reports,
        &mut comparison_failures,
        &format!("vision.layer_{substage_probe_layer_index}.mlp_gate_projection_dense_f32_control"),
        &iree_vision.substage_probe_layer_mlp_gate_projection,
        eager_mlp_gate_projection_dense_f32_control,
        tolerance,
    );
    record_stage(
        &mut reports,
        &mut comparison_failures,
        &format!("vision.layer_{substage_probe_layer_index}.mlp_up_projection_dense_f32_control"),
        &iree_vision.substage_probe_layer_mlp_up_projection,
        eager_mlp_up_projection_dense_f32_control,
        tolerance,
    );
    record_stage(
        &mut reports,
        &mut comparison_failures,
        &format!(
            "vision.layer_{substage_probe_layer_index}.mlp_gate_projection_iree_norm2_dense_f32_control"
        ),
        &iree_vision.substage_probe_layer_mlp_gate_projection,
        &iree_norm2_mlp_projection_controls.gate_projection,
        tolerance,
    );
    record_stage(
        &mut reports,
        &mut comparison_failures,
        &format!(
            "vision.layer_{substage_probe_layer_index}.mlp_up_projection_iree_norm2_dense_f32_control"
        ),
        &iree_vision.substage_probe_layer_mlp_up_projection,
        &iree_norm2_mlp_projection_controls.up_projection,
        tolerance,
    );
    let target_capture_index = iree_vision
        .full_layer_indices
        .iter()
        .position(|&layer| layer == target_full_layer_index)
        .expect("target full layer is present in configured captures");
    record_stage(
        &mut reports,
        &mut comparison_failures,
        &format!("vision.post_full_layer_{target_full_layer_index}_capture_{target_capture_index}"),
        &iree_vision.post_full_layers[target_capture_index],
        &mlx_f32(&eager_vision.post_full_layers[target_capture_index]),
        tolerance,
    );
    record_stage(
        &mut reports,
        &mut comparison_failures,
        "vision.merger_window_ordered",
        &iree_vision.merger_window_ordered,
        &mlx_f32(&eager_vision.merger_window_ordered),
        tolerance,
    );
    record_stage(
        &mut reports,
        &mut comparison_failures,
        "vision.restored_projection",
        &iree_vision.restored_projection,
        &mlx_f32(&eager_vision.restored_projection),
        tolerance,
    );
    if !comparison_failures.is_empty() {
        let report = json!({
            "schema": 2,
            "model": model_path,
            "images": image_paths,
            "device": device,
            "context_capacity": context_capacity,
            "grid_thw": processor.grids,
            "patch_bucket": iree_vision.patch_bucket,
            "patch_tokens": iree_vision.patch_tokens,
            "padded_bucket_rows": iree_vision.patch_bucket - iree_vision.patch_tokens,
            "attention_mask_audit": {
                "full_window_differences": iree_vision.attention_mask_audit.full_window_differences,
                "cross_media_full_bias_leaks": iree_vision.attention_mask_audit.cross_media_full_bias_leaks,
                "active_padding_full_bias_leaks": iree_vision.attention_mask_audit.active_padding_full_bias_leaks,
                "invalid_padded_full_bias_entries": iree_vision.attention_mask_audit.invalid_padded_full_bias_entries,
                "all_diagnostic_outputs_finite_including_padding": true,
            },
            "merged_tokens": iree_vision.merged_tokens,
            "vision_hidden": iree_vision.vision_hidden,
            "text_hidden": iree_vision.text_hidden,
            "window_layer": iree_vision.window_layer_index,
            "full_attention_layers": iree_vision.full_layer_indices,
            "final_interval_layers": iree_vision.final_interval_layer_indices,
            "diagnostic_layers": iree_vision.diagnostic_layer_indices,
            "target_full_layer": target_full_layer_index,
            "substage_probe_layer": substage_probe_layer_index,
            "mlp_projection_control": {
                "runtime": "eager_mlx",
                "input": "same_eager_norm2",
                "weight": "same_checkpoint_weight_cast_to_f32",
                "operation": "dense_f32_matmul_then_f32_bias",
                "purpose": "distinguish native_f16_projection_from_f32_accumulation_contract",
            },
            "iree_norm2_mlp_projection_control": {
                "runtime": "eager_mlx",
                "input": "captured_iree_norm2_rounded_to_f16",
                "weight": "same_checkpoint_weight_cast_to_f32",
                "operation": "dense_f32_matmul_then_f32_bias",
                "purpose": "distinguish inherited_norm2_drift_from_projection_reduction_behavior",
            },
            "failed_phase": "vision",
            "failures": comparison_failures,
            "comparisons": reports,
            "passed": false,
        });
        emit_report(report_path.as_deref(), &report);
        panic!(
            "strict oracle vision comparison failed: {}",
            report["failures"]
        );
    }

    let qwen2_full = iree_vision_capture(
        &mut iree_projector,
        &processor.patch_values,
        &processor.grids,
        Qwen25VlDiagnosticMutation::Qwen2FullAttention,
    );
    let qwen2_full_attention_numerically_detected = compare_stage(
        &mut Vec::new(),
        "negative.qwen2_full_attention",
        &qwen2_full.post_window_layer,
        &mlx_f32(&eager_vision.post_window_layer),
        tolerance,
    )
    .is_err();
    let identity_mutation = iree_vision_capture(
        &mut iree_projector,
        &processor.patch_values,
        &processor.grids,
        Qwen25VlDiagnosticMutation::IdentityPermutation,
    );
    assert!(
        compare_stage(
            &mut Vec::new(),
            "negative.identity_permutation",
            &identity_mutation.reordered_patch_embedding,
            &mlx_f32(&eager_vision.reordered_patch_embedding),
            tolerance,
        )
        .is_err(),
        "negative oracle accepted an identity patch permutation"
    );
    let zero_positions = iree_vision_capture(
        &mut iree_projector,
        &processor.patch_values,
        &processor.grids,
        Qwen25VlDiagnosticMutation::ZeroVisionPositions,
    );
    assert!(
        compare_stage(
            &mut Vec::new(),
            "negative.zero_vision_positions",
            &zero_positions.post_window_layer,
            &mlx_f32(&eager_vision.post_window_layer),
            tolerance,
        )
        .is_err(),
        "negative oracle accepted zeroed vision positions"
    );

    let eager_embeddings = model.get_input_embeddings(&input_ids, &pixels, &processor.grids);
    let prepared_embedding_values = owned_f32(&prepared.embeddings, "prepared embeddings");
    let prepared_hidden = prepared.embeddings.shape[2];
    let mut production_image_rows = Vec::new();
    for (position, &token) in prepared.token_ids.iter().enumerate() {
        if token == model.image_token_id {
            let start = position * prepared_hidden;
            production_image_rows
                .extend_from_slice(&prepared_embedding_values[start..start + prepared_hidden]);
        }
    }
    compare_stage(
        &mut reports,
        "production.restored_projection",
        &iree_vision.restored_projection,
        &production_image_rows,
        tolerance,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    compare_stage(
        &mut reports,
        "prepared.merged_embeddings",
        &prepared_embedding_values,
        &mlx_f32(&eager_embeddings.inputs_embeds),
        tolerance,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let (eager_positions, eager_rope_delta) = model.mrope_diagnostics(&input_ids, &processor.grids);
    let PreparedPositions::Mrope3D { tensor, rope_delta } = &prepared.positions else {
        panic!("production Qwen2.5-VL prefill did not export M-RoPE positions");
    };
    assert_eq!(
        owned_i32(tensor, "prepared M-RoPE"),
        mlx_i32(&eager_positions),
        "prepared M-RoPE coordinates differ"
    );
    assert_eq!(
        *rope_delta, eager_rope_delta,
        "prepared M-RoPE delta differs"
    );

    let mut eager_caches = model.text_model.make_caches();
    let eager_logits = model.text_model.forward_impl(
        &input_ids,
        Some(&eager_embeddings.inputs_embeds),
        &mut eager_caches,
        eager_embeddings.attention_mask_4d.as_deref(),
    );
    let eager_logits = mlx_f32(&eager_logits);
    let mut iree_language =
        Qwen25VlLanguageDiagnosticEngine::load(&model_path, &device, context_capacity)
            .unwrap_or_else(|error| panic!("load Qwen2.5-VL language diagnostics: {error}"));
    let iree_logits = iree_language
        .capture(&prepared)
        .unwrap_or_else(|error| panic!("capture Qwen2.5-VL final logits: {error}"));
    let vocab = iree_logits.len();
    let eager_last =
        &eager_logits[(prepared.sequence_len - 1) * vocab..prepared.sequence_len * vocab];
    assert_eq!(
        argmax(&iree_logits),
        argmax(eager_last),
        "final-position greedy token differs"
    );
    compare_stage(
        &mut reports,
        "language.final_position_logits",
        &iree_logits,
        eager_last,
        tolerance,
    )
    .unwrap_or_else(|error| panic!("{error}"));

    let report = json!({
        "schema": 2,
        "model": model_path,
        "images": image_paths,
        "device": device,
        "context_capacity": context_capacity,
        "grid_thw": processor.grids,
        "patch_bucket": iree_vision.patch_bucket,
        "patch_tokens": iree_vision.patch_tokens,
        "padded_bucket_rows": iree_vision.patch_bucket - iree_vision.patch_tokens,
        "attention_mask_audit": {
            "full_window_differences": iree_vision.attention_mask_audit.full_window_differences,
            "cross_media_full_bias_leaks": iree_vision.attention_mask_audit.cross_media_full_bias_leaks,
            "active_padding_full_bias_leaks": iree_vision.attention_mask_audit.active_padding_full_bias_leaks,
            "invalid_padded_full_bias_entries": iree_vision.attention_mask_audit.invalid_padded_full_bias_entries,
            "all_diagnostic_outputs_finite_including_padding": true,
        },
        "merged_tokens": iree_vision.merged_tokens,
        "vision_hidden": iree_vision.vision_hidden,
        "text_hidden": iree_vision.text_hidden,
        "window_layer": iree_vision.window_layer_index,
        "full_attention_layers": iree_vision.full_layer_indices,
        "final_interval_layers": iree_vision.final_interval_layer_indices,
        "diagnostic_layers": iree_vision.diagnostic_layer_indices,
        "target_full_layer": iree_vision.full_layer_indices.last(),
        "substage_probe_layer": iree_vision.substage_probe_layer_index,
        "negative_qwen2_full_attention_structurally_detected": true,
        "negative_qwen2_full_attention_numerically_detected": qwen2_full_attention_numerically_detected,
        "negative_identity_permutation_detected": true,
        "negative_zero_vision_positions_detected": true,
        "final_top1": argmax(&iree_logits),
        "comparisons": reports,
        "passed": true,
    });
    emit_report(report_path.as_deref(), &report);
}

fn iree_vision_capture(
    projector: &mut IreeQwen25VlDiagnosticProjector,
    patch_values: &[f32],
    grids: &[(i32, i32, i32)],
    mutation: Qwen25VlDiagnosticMutation,
) -> mlxcel_xla::Qwen25VlVisionDiagnostics {
    projector
        .capture(patch_values, grids, mutation)
        .unwrap_or_else(|error| panic!("capture Qwen2.5-VL negative diagnostic: {error}"))
}
