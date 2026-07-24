// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Ignored real-checkpoint Molmo2 vision parity gate.
//!
//! This intentionally compares only the filtered eager MLX vision path with
//! the IREE vision projector. It never loads either text decoder.

#[cfg(feature = "xla-diagnostics")]
use std::path::PathBuf;

use anyhow::{Result, anyhow};
#[cfg(feature = "xla-diagnostics")]
use mlxcel::{initialize_runtime, load_molmo2_xla_vision_reference};
use mlxcel_xla::add_molmo2_projected_features;
#[cfg(feature = "xla-diagnostics")]
use mlxcel_xla::{IreeMolmo2VisionProjector, Molmo2VisionInput};

#[derive(Debug, Clone, Copy)]
struct Comparison {
    max_abs: f32,
    max_index: usize,
    rms: f32,
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<Comparison> {
    if actual.len() != expected.len() {
        return Err(anyhow!(
            "comparison length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        ));
    }
    if actual.is_empty() {
        return Err(anyhow!("cannot compare empty tensors"));
    }
    let mut max_abs = 0.0f32;
    let mut max_index = 0usize;
    let mut squared = 0.0f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(anyhow!(
                "non-finite comparison value at {index}: actual={actual}, expected={expected}"
            ));
        }
        let difference = (actual - expected).abs();
        if difference > max_abs {
            max_abs = difference;
            max_index = index;
        }
        squared += f64::from(difference) * f64::from(difference);
    }
    Ok(Comparison {
        max_abs,
        max_index,
        rms: (squared / actual.len() as f64).sqrt() as f32,
    })
}

#[cfg(feature = "xla-diagnostics")]
fn assert_within(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    max_abs_limit: f32,
    rms_limit: f32,
) -> Result<()> {
    let comparison = compare(actual, expected)?;
    if comparison.max_abs > max_abs_limit || comparison.rms > rms_limit {
        return Err(anyhow!(
            "{label} parity failed: max_abs={} at {}, rms={}, limits=({}, {})",
            comparison.max_abs,
            comparison.max_index,
            comparison.rms,
            max_abs_limit,
            rms_limit
        ));
    }
    eprintln!(
        "{label}: max_abs={} at {}, rms={}",
        comparison.max_abs, comparison.max_index, comparison.rms
    );
    Ok(())
}

fn active_groups(pooling: &[i32], groups: usize, group_size: usize) -> Result<Vec<usize>> {
    if group_size == 0 || pooling.len() != groups * group_size {
        return Err(anyhow!(
            "invalid pooling shape: values={}, groups={groups}, group_size={group_size}",
            pooling.len()
        ));
    }
    Ok(pooling
        .chunks_exact(group_size)
        .enumerate()
        .filter_map(|(group, values)| values.iter().any(|&value| value >= 0).then_some(group))
        .collect())
}

fn independent_scatter_add(
    token_ids: &[i32],
    image_patch_id: i32,
    base: &[f32],
    hidden_size: usize,
    projected: &[f32],
) -> Result<Vec<f32>> {
    if hidden_size == 0 || base.len() != token_ids.len() * hidden_size {
        return Err(anyhow!("invalid base embedding shape"));
    }
    let positions = token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == image_patch_id).then_some(index))
        .collect::<Vec<_>>();
    if projected.len() != positions.len() * hidden_size {
        return Err(anyhow!(
            "projected feature count does not match image positions"
        ));
    }
    let mut merged = base.to_vec();
    for (row, position) in positions.into_iter().enumerate() {
        for hidden in 0..hidden_size {
            merged[position * hidden_size + hidden] += projected[row * hidden_size + hidden];
        }
    }
    Ok(merged)
}

#[cfg(feature = "xla-diagnostics")]
fn tolerance(name: &str, default: f32) -> Result<f32> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<f32>()
            .map_err(|error| anyhow!("invalid {name}={value}: {error}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(anyhow!("cannot read {name}: {error}")),
    }
}

#[test]
fn synthetic_active_rows_skip_all_negative_groups() {
    let pooling = [0, -1, -1, -1, -1, -1, -1, -1, 4, 5, -1, -1];
    assert_eq!(active_groups(&pooling, 3, 4).unwrap(), vec![0, 2]);
}

#[test]
fn synthetic_independent_scatter_matches_production_helper() {
    let tokens = [7, 151_938, 8, 151_938, 9];
    let base = (0..15).map(|index| index as f32 * 0.25).collect::<Vec<_>>();
    let projected = [0.5, -1.0, 1.5, 2.0, 3.0, -0.25];
    let expected = independent_scatter_add(&tokens, 151_938, &base, 3, &projected).unwrap();
    let mut actual = base;
    let positions =
        add_molmo2_projected_features(&tokens, 151_938, &mut actual, 3, &projected).unwrap();
    assert_eq!(positions, vec![1, 3]);
    assert_eq!(actual, expected);
}

#[test]
fn synthetic_comparison_reports_max_and_rms() {
    let comparison = compare(&[1.0, 2.5, 3.0], &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(comparison.max_abs, 0.5);
    assert_eq!(comparison.max_index, 1);
    assert!((comparison.rms - (0.25f32 / 3.0).sqrt()).abs() < 1e-7);
}

#[test]
#[cfg(feature = "xla-diagnostics")]
#[ignore = "requires a Molmo2 checkpoint plus configured MLX and IREE runtimes"]
fn real_checkpoint_mlx_iree_vision_and_scatter_parity() -> Result<()> {
    let model = PathBuf::from(
        std::env::var("MLXCEL_MOLMO2_MODEL")
            .map_err(|_| anyhow!("MLXCEL_MOLMO2_MODEL is required"))?,
    );
    let image_path = std::env::var("MLXCEL_MOLMO2_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png")
        });
    let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".to_string());
    let max_abs_limit = tolerance("MLXCEL_MOLMO2_MAX_ABS", 0.05)?;
    let rms_limit = tolerance("MLXCEL_MOLMO2_RMS", 0.01)?;

    let _runtime = initialize_runtime();
    let reference = load_molmo2_xla_vision_reference(&model)?;
    let image = image::open(&image_path)
        .map_err(|error| anyhow!("open {}: {error}", image_path.display()))?;
    let eager = reference.project(&image)?;
    let processed = &eager.processed;
    let pooling_shape = processed
        .image_token_pooling_shape
        .map(|value| value as usize);
    let independently_active = active_groups(
        &processed.image_token_pooling,
        pooling_shape[0],
        pooling_shape[1],
    )?;
    if eager.active_groups != independently_active {
        return Err(anyhow!(
            "eager active rows {:?} disagree with processor rows {:?}",
            eager.active_groups,
            independently_active
        ));
    }

    let mut projector =
        IreeMolmo2VisionProjector::load(&model, &device).map_err(anyhow::Error::msg)?;
    if projector.image_patch_id() != reference.image_patch_id()
        || projector.text_hidden_size() != reference.text_hidden_size()
    {
        return Err(anyhow!("MLX and IREE Molmo2 metadata disagree"));
    }
    let iree = projector
        .project(Molmo2VisionInput {
            patches: &processed.pixel_values,
            patches_shape: processed.pixel_values_shape.map(|value| value as usize),
            image_token_pooling: &processed.image_token_pooling,
            pooling_shape,
            image_grid: processed.image_grid,
            image_num_crops: processed.image_num_crops as usize,
            prompt_image_patch_count: independently_active.len(),
        })
        .map_err(anyhow::Error::msg)?;
    let iree_active = iree
        .valid_pooling_counts
        .iter()
        .enumerate()
        .filter_map(|(group, &count)| (count > 0).then_some(group))
        .collect::<Vec<_>>();
    if iree_active != independently_active || iree.shape != eager.shape {
        return Err(anyhow!(
            "active-row mismatch: processor={independently_active:?}, IREE={iree_active:?}, MLX shape={:?}, IREE shape={:?}",
            eager.shape,
            iree.shape
        ));
    }
    assert_within(
        "vision projection",
        &iree.values,
        &eager.values,
        max_abs_limit,
        rms_limit,
    )?;

    let hidden = reference.text_hidden_size();
    let mut tokens = Vec::with_capacity(independently_active.len() * 2 + 1);
    tokens.push(7);
    for row in 0..independently_active.len() {
        tokens.push(reference.image_patch_id());
        tokens.push(8 + (row % 17) as i32);
    }
    let base = (0..tokens.len() * hidden)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.001)
        .collect::<Vec<_>>();
    let eager_merged = independent_scatter_add(
        &tokens,
        reference.image_patch_id(),
        &base,
        hidden,
        &eager.values,
    )?;
    let independently_iree_merged = independent_scatter_add(
        &tokens,
        reference.image_patch_id(),
        &base,
        hidden,
        &iree.values,
    )?;
    let mut production_iree_merged = base;
    add_molmo2_projected_features(
        &tokens,
        reference.image_patch_id(),
        &mut production_iree_merged,
        hidden,
        &iree.values,
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    if production_iree_merged != independently_iree_merged {
        return Err(anyhow!(
            "production scatter-add disagrees with the independent implementation"
        ));
    }
    assert_within(
        "scatter-added embeddings",
        &production_iree_merged,
        &eager_merged,
        max_abs_limit,
        rms_limit,
    )
}
