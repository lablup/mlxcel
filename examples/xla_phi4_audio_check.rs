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

//! Real-checkpoint #874 projection parity for the IREE Phi4MM audio module.
//!
//! This diagnostic intentionally loads the MLX model only to capture the
//! qualified reference, drops it, and then loads the independent IREE-only
//! audio runtime. Production XLA execution never constructs the MLX decoder.
//! Full intermediate comparisons remain diagnostic; the pinned #874 projection
//! prefix is the qualified acceptance boundary.

use std::path::PathBuf;

use mlxcel::LoadedModel;
use mlxcel_xla::{
    PHI4MM_AUDIO_CHECKPOINT_REVISION, Phi4AudioDiagnosticRuntime, phi4_audio_bucket_for_frames,
};

fn mlx_f32(array: &mlxcel_core::MlxArray) -> Result<Vec<f32>, String> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    let bytes = mlxcel_core::try_array_to_raw_bytes(&array)
        .map_err(|error| format!("export MLX reference: {error}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32")))
        .collect())
}

struct Comparison {
    max: f32,
    max_index: usize,
    actual_at_max: f32,
    expected_at_max: f32,
    rms: f32,
}

struct ReferenceCheckpoint {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn mlx_checkpoint(
    name: impl Into<String>,
    array: &mlxcel_core::MlxArray,
) -> Result<ReferenceCheckpoint, String> {
    let name = name.into();
    let shape = mlxcel_core::array_shape(array)
        .into_iter()
        .map(|dimension| {
            usize::try_from(dimension)
                .map_err(|_| format!("negative MLX diagnostic dimension {dimension}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ReferenceCheckpoint {
        name,
        shape,
        values: mlx_f32(array)?,
    })
}

fn compare(actual: &[f32], expected: &[f32], label: &str) -> Result<Comparison, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} length {}, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut max = 0.0f32;
    let mut max_index = 0usize;
    let mut squared = 0.0f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual - expected).abs();
        if difference > max {
            max = difference;
            max_index = index;
        }
        squared += f64::from(difference) * f64::from(difference);
    }
    let rms = (squared / actual.len() as f64).sqrt() as f32;
    Ok(Comparison {
        max,
        max_index,
        actual_at_max: actual[max_index],
        expected_at_max: expected[max_index],
        rms,
    })
}

fn print_comparison(label: &str, comparison: &Comparison, width: usize) {
    println!(
        "{label} max_abs={:.8} at row={} dim={} (actual={:.8}, expected={:.8}) rms={:.8}",
        comparison.max,
        comparison.max_index / width,
        comparison.max_index % width,
        comparison.actual_at_max,
        comparison.expected_at_max,
        comparison.rms,
    );
}

fn main() -> Result<(), String> {
    let model_dir = PathBuf::from(
        std::env::var_os("MLXCEL_PHI4MM_MODEL")
            .ok_or("set MLXCEL_PHI4MM_MODEL to the pinned official checkpoint")?,
    );
    let revision =
        std::env::var("MLXCEL_PHI4MM_REVISION").map_err(|_| "set MLXCEL_PHI4MM_REVISION")?;
    if revision != PHI4MM_AUDIO_CHECKPOINT_REVISION {
        return Err(format!(
            "checkpoint revision {revision} does not match pinned {PHI4MM_AUDIO_CHECKPOINT_REVISION}"
        ));
    }
    let device = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "local-task".to_string());
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/phi4mm_audio_parity.json"))
            .map_err(|error| format!("parse #874 fixture: {error}"))?;
    let audio = model_dir.join(
        fixture["audio"]
            .as_str()
            .ok_or("#874 fixture audio path is not a string")?,
    );

    let (loaded, _) = mlxcel::load_model(&model_dir)
        .map_err(|error| format!("load #874 MLX reference: {error}"))?;
    let LoadedModel::Phi4MMVLM(model) = loaded else {
        return Err("pinned checkpoint did not load as Phi4MMVLM".to_string());
    };
    let (samples, sample_rate) =
        mlxcel::audio::load_wav_file(&audio).map_err(|error| error.to_string())?;
    let batch = model.extract_audio(&[(samples, sample_rate)])?;
    let shape = mlxcel_core::array_shape(&batch.clips[0]);
    if shape.len() != 3 || shape[0] != 1 || shape[2] != 80 {
        return Err(format!("unexpected #874 feature shape {shape:?}"));
    }
    let frame_len =
        usize::try_from(shape[1]).map_err(|_| format!("negative frame length {}", shape[1]))?;
    let features = mlx_f32(&batch.clips[0])?;
    let encoder = model.audio_encoder.forward_diagnostics(&batch.clips[0])?;
    let speech = model
        .audio_projection
        .forward_diagnostics(&encoder.encoded, false);
    let vision = model
        .audio_projection
        .forward_diagnostics(&encoder.encoded, true);
    let mut references = vec![
        mlx_checkpoint("subsample.conv0", &encoder.subsample.conv0)?,
        mlx_checkpoint(
            "subsample.conv1.depthwise",
            &encoder.subsample.conv1_depthwise,
        )?,
        mlx_checkpoint(
            "subsample.conv1.pointwise",
            &encoder.subsample.conv1_pointwise,
        )?,
        mlx_checkpoint("subsample.conv1", &encoder.subsample.conv1)?,
        mlx_checkpoint(
            "subsample.conv2.depthwise",
            &encoder.subsample.conv2_depthwise,
        )?,
        mlx_checkpoint(
            "subsample.conv2.pointwise",
            &encoder.subsample.conv2_pointwise,
        )?,
        mlx_checkpoint("subsample.conv2", &encoder.subsample.conv2)?,
        mlx_checkpoint("subsample.projected", &encoder.subsample.projected)?,
        mlx_checkpoint("block0.after_ff_in", &encoder.block0.after_ff_in)?,
        mlx_checkpoint("block0.attention", &encoder.block0.attention)?,
        mlx_checkpoint("block0.after_attention", &encoder.block0.after_attention)?,
        mlx_checkpoint("block0.convolution", &encoder.block0.convolution)?,
        mlx_checkpoint(
            "block0.after_convolution",
            &encoder.block0.after_convolution,
        )?,
        mlx_checkpoint("block0.ff_out", &encoder.block0.ff_out)?,
        mlx_checkpoint("block0.output", &encoder.block0.output)?,
    ];
    for (index, values) in &encoder.selected_blocks {
        references.push(mlx_checkpoint(format!("block{index}.output"), values)?);
    }
    references.extend([
        mlx_checkpoint("encoder.output", &encoder.encoded)?,
        mlx_checkpoint("projection.speech.first", &speech.first)?,
        mlx_checkpoint("projection.speech.output", &speech.output)?,
        mlx_checkpoint("projection.vision.first", &vision.first)?,
        mlx_checkpoint("projection.vision.output", &vision.output)?,
    ]);
    drop(model);

    let frame_bucket = phi4_audio_bucket_for_frames(frame_len)
        .ok_or_else(|| format!("{frame_len} feature frames exceed the Phi4MM XLA policy"))?;
    let mut runtime = Phi4AudioDiagnosticRuntime::load(&model_dir, &device, frame_bucket)?;
    let diagnostics = runtime.capture(&features, frame_len)?;
    let expected_rows = fixture["audio_embed_size"]
        .as_u64()
        .ok_or("#874 fixture audio_embed_size is not an integer")? as usize;
    if diagnostics.valid_rows != expected_rows {
        return Err(format!(
            "projected length {}, expected {expected_rows}",
            diagnostics.valid_rows
        ));
    }
    if diagnostics.checkpoints.len() != references.len() {
        return Err(format!(
            "IREE returned {} diagnostic checkpoints, MLX captured {}",
            diagnostics.checkpoints.len(),
            references.len()
        ));
    }
    for (actual, expected) in diagnostics.checkpoints.iter().zip(&references) {
        if actual.name != expected.name {
            return Err(format!(
                "diagnostic checkpoint order drift: IREE {}, MLX {}",
                actual.name, expected.name
            ));
        }
        if actual.shape.len() != expected.shape.len()
            || actual.shape[0] != expected.shape[0]
            || actual.shape[2..] != expected.shape[2..]
            || actual.shape[1] < expected.shape[1]
        {
            return Err(format!(
                "{} shape mismatch: IREE {:?}, MLX {:?}",
                actual.name, actual.shape, expected.shape
            ));
        }
        let comparison = compare(
            &actual.values[..expected.values.len()],
            &expected.values,
            actual.name,
        )?;
        print_comparison(
            actual.name,
            &comparison,
            *expected.shape.last().unwrap_or(&1),
        );
    }
    let oracle_prefix = fixture["projection_first"]
        .as_array()
        .ok_or("#874 projection_first is not an array")?
        .iter()
        .map(|value| value.as_f64().map(|value| value as f32))
        .collect::<Option<Vec<_>>>()
        .ok_or("#874 projection_first contains a non-number")?;
    let speech_output = diagnostics
        .checkpoints
        .iter()
        .find(|checkpoint| checkpoint.name == "projection.speech.output")
        .ok_or("IREE diagnostic omitted projection.speech.output")?;
    let prefix_comparison = compare(
        &speech_output.values[..oracle_prefix.len()],
        &oracle_prefix,
        "#874 speech prefix",
    )?;
    println!(
        "device={device} bucket={frame_bucket} frames={frame_len} projected_rows={}",
        diagnostics.valid_rows,
    );
    print_comparison("#874 speech prefix", &prefix_comparison, 3072);
    if prefix_comparison.max > 0.025 {
        return Err(format!(
            "#874 speech projection prefix exceeded tolerance: max_abs={} at flat index {}",
            prefix_comparison.max, prefix_comparison.max_index
        ));
    }
    Ok(())
}
