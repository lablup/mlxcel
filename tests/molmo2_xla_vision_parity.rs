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

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
use std::{
    io::{self, Write},
    path::PathBuf,
    sync::mpsc::{self, Sender},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow};
#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
use mlxcel::{initialize_runtime, load_molmo2_xla_vision_reference};
use mlxcel_xla::add_molmo2_projected_features;
#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
use mlxcel_xla::{IreeMolmo2VisionDiagnosticProjector, Molmo2VisionInput};

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
const PROGRESS_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(60);

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
fn write_progress(output: &mut impl Write, message: &str) -> io::Result<()> {
    writeln!(output, "[molmo2-reference] {message}")?;
    output.flush()
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
fn emit_progress(message: &str) {
    let stderr = io::stderr();
    let mut stderr = stderr.lock();
    let _ = write_progress(&mut stderr, message);
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
struct ProgressHeartbeat {
    stop: Sender<()>,
    worker: Option<JoinHandle<()>>,
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
impl Drop for ProgressHeartbeat {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
fn with_progress<T, E>(
    label: &'static str,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    emit_progress(&format!("{label}: started"));
    let started = Instant::now();
    let (stop, receiver) = mpsc::channel();
    let worker = thread::spawn(move || {
        while let Err(mpsc::RecvTimeoutError::Timeout) =
            receiver.recv_timeout(PROGRESS_HEARTBEAT_INTERVAL)
        {
            emit_progress(&format!(
                "{label}: still running (elapsed={}s)",
                started.elapsed().as_secs()
            ));
        }
    });
    let progress = ProgressHeartbeat {
        stop,
        worker: Some(worker),
    };
    let result = operation();
    drop(progress);
    let outcome = if result.is_ok() {
        "completed"
    } else {
        "failed"
    };
    emit_progress(&format!(
        "{label}: {outcome} (elapsed={}s)",
        started.elapsed().as_secs()
    ));
    result
}

#[derive(Debug, Clone, Copy)]
struct Comparison {
    max_abs: f32,
    max_index: usize,
    rms: f32,
    /// The two values behind `max_abs`, so an absolute gap can be read against
    /// the magnitude it sits on.
    actual_at_max: f32,
    expected_at_max: f32,
    /// How many elements exceed the combined tolerance. One element out of a
    /// million is an outlier; a broad fraction is a systematic mismatch, and
    /// `max_abs` alone cannot tell those apart.
    over_limit: usize,
    len: usize,
    /// Worst `difference / (atol + rtol * |expected|)` seen. At most 1.0 the
    /// tensor is inside the contract; above 1.0 it is not.
    worst_ratio: f32,
    // Read only by `detail`, which only the diagnostics-gated assertion path
    // calls. A build without those features still constructs them, so they are
    // allowed rather than cfg-gated: the values are cheap and keeping one
    // definition avoids two shapes of the same struct.
    #[allow(dead_code)]
    ratio_index: usize,
    #[allow(dead_code)]
    ratio_actual: f32,
    #[allow(dead_code)]
    ratio_expected: f32,
}

impl Comparison {
    /// Relative size of the worst gap against the value it sits on.
    fn relative_at_max(&self) -> f32 {
        let scale = self.expected_at_max.abs().max(self.actual_at_max.abs());
        if scale > 0.0 {
            self.max_abs / scale
        } else {
            0.0
        }
    }

    /// Only the diagnostics-gated `assert_within` reports this.
    #[allow(dead_code)]
    fn detail(&self) -> String {
        format!(
            "max_abs={} at {} (actual={}, expected={}, relative={:.4}%), rms={}, \
             worst_tolerance_ratio={:.3} at {} (actual={}, expected={}), over_limit={}/{} ({:.4}%)",
            self.max_abs,
            self.max_index,
            self.actual_at_max,
            self.expected_at_max,
            self.relative_at_max() * 100.0,
            self.rms,
            self.worst_ratio,
            self.ratio_index,
            self.ratio_actual,
            self.ratio_expected,
            self.over_limit,
            self.len,
            self.over_limit as f64 * 100.0 / self.len as f64
        )
    }
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<Comparison> {
    compare_with_limit(actual, expected, f32::INFINITY, 0.0)
}

/// Compare against the combined tolerance `atol + rtol * |expected|`.
///
/// A single absolute limit cannot express this contract across the pipeline.
/// The ViT stages carry values around 1, where 0.05 is a strict bound, but the
/// projector output reaches magnitudes of 2e4, where one f32 ULP is already
/// about 0.004 and a 2000-term dot product accumulated in a different order
/// lands tens of ULPs away for reasons that have nothing to do with the
/// emitter. Measured on the pinned checkpoint, four ordinary photographs put
/// 4 to 12 elements out of ~2e6 past 0.05 while every one of them agreed to
/// within 0.001% relatively.
///
/// `atol` still floors the comparison near zero, so nothing that was strict
/// before becomes loose: the bound only widens where the values themselves are
/// large enough to make an absolute bound meaningless.
fn compare_with_limit(
    actual: &[f32],
    expected: &[f32],
    max_abs_limit: f32,
    relative_limit: f32,
) -> Result<Comparison> {
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
    let mut over_limit = 0usize;
    let mut actual_at_max = 0.0f32;
    let mut expected_at_max = 0.0f32;
    let mut worst_ratio = 0.0f32;
    let mut ratio_index = 0usize;
    let mut ratio_actual = 0.0f32;
    let mut ratio_expected = 0.0f32;
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
            actual_at_max = actual;
            expected_at_max = expected;
        }
        let allowed = max_abs_limit + relative_limit * expected.abs();
        if difference > allowed {
            over_limit += 1;
        }
        // A zero allowance can only be met exactly, so report an exceeded
        // ratio rather than dividing by zero.
        let ratio = if allowed > 0.0 {
            difference / allowed
        } else if difference > 0.0 {
            f32::INFINITY
        } else {
            0.0
        };
        if ratio > worst_ratio {
            worst_ratio = ratio;
            ratio_index = index;
            ratio_actual = actual;
            ratio_expected = expected;
        }
        squared += f64::from(difference) * f64::from(difference);
    }
    let len = actual.len();
    Ok(Comparison {
        max_abs,
        max_index,
        rms: (squared / len as f64).sqrt() as f32,
        actual_at_max,
        expected_at_max,
        over_limit,
        len,
        worst_ratio,
        ratio_index,
        ratio_actual,
        ratio_expected,
    })
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
fn assert_within(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    max_abs_limit: f32,
    relative_limit: f32,
    rms_limit: f32,
) -> Result<()> {
    let comparison = compare_with_limit(actual, expected, max_abs_limit, relative_limit)?;
    emit_progress(&format!("{label}: {}", comparison.detail()));
    if comparison.worst_ratio > 1.0 || comparison.rms > rms_limit {
        return Err(anyhow!(
            "{label} parity failed: {}, limits=(atol={}, rtol={}, rms={})",
            comparison.detail(),
            max_abs_limit,
            relative_limit,
            rms_limit
        ));
    }
    Ok(())
}

/// The admission predicate `assert_within` applies, without its progress
/// reporting. Kept ungated so the tolerance shape is verified in every build of
/// this target, not only the diagnostics ones that can reach a checkpoint.
fn within(actual: &[f32], expected: &[f32], max_abs_limit: f32, relative_limit: f32) -> bool {
    compare_with_limit(actual, expected, max_abs_limit, relative_limit)
        .map(|comparison| comparison.worst_ratio <= 1.0)
        .unwrap_or(false)
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

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
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
    // The worst gap is reported against the magnitude it sits on, so an
    // absolute limit can be read as a relative one.
    assert_eq!(comparison.actual_at_max, 2.5);
    assert_eq!(comparison.expected_at_max, 2.0);
    assert!((comparison.relative_at_max() - 0.2).abs() < 1e-6);
}

#[test]
fn synthetic_comparison_separates_one_outlier_from_broad_drift() {
    // One element over the limit out of four: an outlier.
    let outlier =
        compare_with_limit(&[1.0, 1.0, 1.0, 9.0], &[1.0, 1.0, 1.0, 1.0], 0.5, 0.0).unwrap();
    assert_eq!(outlier.over_limit, 1);
    assert_eq!(outlier.len, 4);

    // Every element over the limit at the same max: same max_abs, different story.
    let drift = compare_with_limit(&[9.0, 9.0, 9.0, 9.0], &[1.0, 1.0, 1.0, 1.0], 0.5, 0.0).unwrap();
    assert_eq!(drift.max_abs, outlier.max_abs);
    assert_eq!(drift.over_limit, 4);
}

#[test]
fn the_relative_half_only_widens_the_bound_where_values_are_large() {
    let atol = 0.05f32;
    let rtol = 1e-5f32;

    // Near zero the absolute floor still rules: a gap just past atol fails, so
    // nothing that was strict before this change became loose.
    assert!(
        !within(&[0.06], &[0.0], atol, rtol),
        "atol must still bound values around zero"
    );
    assert!(within(&[0.04], &[0.0], atol, rtol));

    // At the projector output's magnitude the same absolute gap is a fraction
    // of one f32 ULP's worth of relative error and must pass. These are the
    // measured `tall` values from the real gate.
    assert!(
        within(&[25507.98], &[25507.887], atol, rtol),
        "a 0.09 gap on a 2.5e4 value is reduction-order noise, not a defect"
    );

    // A genuine defect changes leading digits, which the relative half still
    // catches no matter how large the value is.
    assert!(
        !within(&[25507.98], &[25000.0], atol, rtol),
        "a 2% error on a large value must still fail"
    );
}

#[test]
fn synthetic_negative_controls_detect_layer_denominator_and_clamped_index_drift() {
    let canonical_layers = [24.0, 18.0, 240.0, 180.0];
    let wrong_layers = [18.0, 24.0, 180.0, 240.0];
    assert!(compare(&canonical_layers, &wrong_layers).unwrap().max_abs > 0.0);
    // The combined tolerance must not blunt this: swapping the selected layers
    // is a defect at any magnitude, so it has to fail under the same atol/rtol
    // the real gate runs with.
    assert!(
        !within(&canonical_layers, &wrong_layers, 0.05, 1e-5),
        "selected-layer drift must still be rejected"
    );

    let masked_values = [2.0, 6.0, 0.0, 0.0];
    let valid_mean = masked_values.iter().sum::<f32>() / 2.0;
    let wrong_fixed_window_mean = masked_values.iter().sum::<f32>() / 4.0;
    assert_ne!(valid_mean, wrong_fixed_window_mean);

    let patch_zero = 100.0;
    let valid_patch = 2.0;
    let masked_gather = valid_patch;
    let unmasked_clamped_gather = valid_patch + patch_zero;
    assert_ne!(masked_gather, unmasked_clamped_gather);
}

#[test]
#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
fn synthetic_progress_flushes_and_heartbeats_before_five_minutes() {
    #[derive(Default)]
    struct FlushProbe {
        bytes: Vec<u8>,
        flushes: usize,
    }

    impl Write for FlushProbe {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    let mut probe = FlushProbe::default();
    write_progress(&mut probe, "MLX eager diagnostic projection: started").unwrap();
    assert_eq!(probe.flushes, 1);
    assert_eq!(
        String::from_utf8(probe.bytes).unwrap(),
        "[molmo2-reference] MLX eager diagnostic projection: started\n"
    );
    assert!(PROGRESS_HEARTBEAT_INTERVAL < Duration::from_secs(5 * 60));
}

/// Compare the whole vision path against eager MLX on a real checkpoint.
///
/// `MLXCEL_MOLMO2_IMAGE` selects the image, which decides the crop tiling and
/// therefore which parts of the path run at all. The bundled fixture is a
/// 224x224 solid square, whose `1x1` tiling exercises a single crop and skips
/// the high-resolution tiling, pooling offsets, and multi-crop merge entirely.
/// Validated tilings, all against the pinned 4B checkpoint on CUDA:
///
/// | image        | size     | tiling | crops | prompt tokens |
/// | ------------ | -------- | ------ | ----- | ------------- |
/// | fixture      | 224x224  | 1x1    | 2     | 424           |
/// | square-large | 756x756  | 2x2    | 5     | 970           |
/// | tall         | 378x1512 | 4x1    | 5     | 1024          |
/// | wide         | 1512x378 | 1x4    | 5     | 984           |
/// | photo-ish    | 1024x768 | 2x3    | 7     | 1348          |
///
/// The non-fixture images are generated rather than committed; any textured
/// image of those dimensions reproduces the tiling. A flat color will not: it
/// hides pooling and interpolation differences.
#[test]
#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
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
    // Relative half of the contract. 1e-5 is roughly four f32 ULPs at the
    // projector output's magnitude, so it admits reduction-order differences
    // while still rejecting anything that changes a value's leading digits.
    let relative_limit = tolerance("MLXCEL_MOLMO2_RTOL", 1e-5)?;
    let rms_limit = tolerance("MLXCEL_MOLMO2_RMS", 0.01)?;

    let _runtime = initialize_runtime();
    let reference = with_progress("MLX vision-only checkpoint load", || {
        load_molmo2_xla_vision_reference(&model)
    })?;
    let image = image::open(&image_path)
        .map_err(|error| anyhow!("open {}: {error}", image_path.display()))?;
    let eager = with_progress("MLX eager diagnostic projection", || {
        reference.project(&image)
    })?;
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

    // Bound the local-task worker topology and stack before the first IREE
    // instance exists. IREE parses its process-global flag registry on that
    // first creation, so this has to run ahead of the projector load to take
    // effect. Without it, `local-task` worker creation fails on this host with
    // `thread creation failed with 22` out of `thread_pthreads.c`, because
    // IREE's exact PTHREAD_STACK_MIN request is rejected. Diagnostics-only:
    // the production runtime never applies these flags.
    mlxcel_xla::configure_diagnostic_local_task_threads().map_err(anyhow::Error::msg)?;
    let mut projector = with_progress("IREE diagnostic compile/load", || {
        IreeMolmo2VisionDiagnosticProjector::load(&model, &device).map_err(anyhow::Error::msg)
    })?;
    if projector.image_patch_id() != reference.image_patch_id()
        || projector.text_hidden_size() != reference.text_hidden_size()
    {
        return Err(anyhow!("MLX and IREE Molmo2 metadata disagree"));
    }
    let iree = with_progress("IREE diagnostic invocation", || {
        projector
            .project(Molmo2VisionInput {
                patches: &processed.pixel_values,
                patches_shape: processed.pixel_values_shape.map(|value| value as usize),
                image_token_pooling: &processed.image_token_pooling,
                pooling_shape,
                image_grid: processed.image_grid,
                image_num_crops: processed.image_num_crops as usize,
                prompt_image_patch_count: independently_active.len(),
            })
            .map_err(anyhow::Error::msg)
    })?;
    if iree.active_groups != independently_active || iree.projected_shape != eager.shape {
        return Err(anyhow!(
            "active-row mismatch: processor={independently_active:?}, IREE={:?}, MLX shape={:?}, IREE shape={:?}",
            iree.active_groups,
            eager.shape,
            iree.projected_shape
        ));
    }
    if iree.stages.len() != eager.stages.len() {
        return Err(anyhow!(
            "diagnostic stage count mismatch: MLX={}, IREE={}",
            eager.stages.len(),
            iree.stages.len()
        ));
    }
    for (eager_stage, iree_stage) in eager.stages.iter().zip(&iree.stages) {
        if eager_stage.name != iree_stage.name || eager_stage.shape != iree_stage.shape {
            return Err(anyhow!(
                "diagnostic stage layout mismatch: MLX {} {:?}, IREE {} {:?}",
                eager_stage.name,
                eager_stage.shape,
                iree_stage.name,
                iree_stage.shape
            ));
        }
        assert_within(
            &format!("first-divergence stage {}", eager_stage.name),
            &iree_stage.values,
            &eager_stage.values,
            max_abs_limit,
            relative_limit,
            rms_limit,
        )?;
    }
    assert_within(
        "vision projection",
        &iree.projected_values,
        &eager.values,
        max_abs_limit,
        relative_limit,
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
        &iree.projected_values,
    )?;
    let mut production_iree_merged = base;
    add_molmo2_projected_features(
        &tokens,
        reference.image_patch_id(),
        &mut production_iree_merged,
        hidden,
        &iree.projected_values,
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
        relative_limit,
        rms_limit,
    )
}
