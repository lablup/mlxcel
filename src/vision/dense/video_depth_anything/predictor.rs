// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Sliding 32-frame prediction, keyframe conditioning, and window alignment.

use anyhow::{Result, ensure};
use image::{DynamicImage, GenericImageView};
use serde_json::{Map, Value, json};
use std::path::Path;

use super::config::ResolvedVdaConfig;
use super::model::VideoDepthModel;
use super::processor::VideoDepthProcessor;
use crate::vision::dense::interpolate::resize_bilinear_nhwc;
use crate::vision::dense::{DenseInput, DenseOutput, DensePredictor};

pub const INFER_LEN: usize = 32;
pub const OVERLAP: usize = 10;
pub const INTERP_LEN: usize = 8;
pub const KEYFRAMES: [usize; 10] = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31];
const FRAME_STEP: usize = INFER_LEN - OVERLAP;
const MAX_FRAMES: usize = 4096;
const MAX_HOST_WORKING_VALUES: usize = 512 * 1024 * 1024;
const MAX_TOTAL_WORKING_VALUES: usize = 3 * 1024 * 1024 * 1024;
const BACKBONE_ACTIVE_CHUNK: usize = 16;
const OUTPUT_HEAD_CHUNK: usize = 4;
const TASKS: &[&str] = &["depth"];

pub struct VideoDepthPredictor {
    model: VideoDepthModel,
    processor: VideoDepthProcessor,
}

fn checked_values(label: &str, factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| anyhow::anyhow!("VDA {label} workset overflowed"))
    })
}

fn checked_sum(label: &str, values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |sum, &value| {
        sum.checked_add(value)
            .ok_or_else(|| anyhow::anyhow!("VDA {label} workset overflowed"))
    })
}

fn host_working_values(
    source_frames: usize,
    source_pixels: usize,
    target_h: usize,
    target_w: usize,
) -> Result<usize> {
    let output = checked_values("output", &[source_frames, source_pixels, 2])?;
    // `array_to_vec_f32` and the per-frame vectors coexist briefly for every
    // padded 32-frame window, including a one-frame still.
    let window = checked_values("window", &[INFER_LEN, source_pixels, 2])?;
    // Preprocessed frames coexist with their stacked MLX input.
    let input = checked_values("preprocessed input", &[target_h, target_w, 3, INFER_LEN, 2])?;
    checked_sum("host", &[output, window, input])
}

fn model_working_values(
    config: &ResolvedVdaConfig,
    target_h: usize,
    target_w: usize,
) -> Result<usize> {
    let patch = config.dinov2.patch_size;
    ensure!(
        target_h.is_multiple_of(patch) && target_w.is_multiple_of(patch),
        "VDA target dimensions must be patch multiples"
    );
    let patch_h = target_h / patch;
    let patch_w = target_w / patch;
    let patches = checked_values("patch grid", &[patch_h, patch_w])?;
    let frames = config.num_frames;
    let hidden = config.dinov2.hidden_size;

    // Four complete intermediate levels remain live for the temporal head.
    let retained_backbone =
        checked_values("retained backbone features", &[frames, patches, hidden, 4])?;
    // Allow several hidden-sized temporaries for the active independent-frame
    // backbone chunk in addition to the retained outputs.
    let active_backbone = checked_values(
        "active backbone chunk",
        &[BACKBONE_ACTIVE_CHUNK, patches, hidden, 6],
    )?;

    let level_areas = [
        checked_values("level 1 area", &[patches, 16])?,
        checked_values("level 2 area", &[patches, 4])?,
        patches,
        checked_values("level 4 area", &[patch_h.div_ceil(2), patch_w.div_ceil(2)])?,
    ];
    let mut projected = 0usize;
    let mut reduced = 0usize;
    for (area, channels) in level_areas.into_iter().zip(config.out_channels) {
        projected = checked_sum(
            "projected DPT levels",
            &[
                projected,
                checked_values("projected DPT level", &[frames, area, channels])?,
            ],
        )?;
        reduced = checked_sum(
            "reduced DPT levels",
            &[
                reduced,
                checked_values("reduced DPT level", &[frames, area, config.features])?,
            ],
        )?;
    }
    let temporal_paths =
        checked_values("temporal DPT paths", &[frames, patches, config.features, 5])?;

    // The high-resolution head runs four frames at a time. Count path2,
    // path1, output_conv1, the full-resolution feature plane, and both final
    // output stages while that chunk is active.
    let path2 = checked_values(
        "head path2",
        &[OUTPUT_HEAD_CHUNK, patches, 16, config.features],
    )?;
    let path1 = checked_values(
        "head path1",
        &[OUTPUT_HEAD_CHUNK, patches, 64, config.features],
    )?;
    let output1 = checked_values(
        "head output1",
        &[OUTPUT_HEAD_CHUNK, patches, 64, config.features / 2],
    )?;
    let full_half = checked_values(
        "full-resolution half head",
        &[OUTPUT_HEAD_CHUNK, target_h, target_w, config.features / 2],
    )?;
    let full_32 = checked_values(
        "full-resolution 32-channel head",
        &[OUTPUT_HEAD_CHUNK, target_h, target_w, 32],
    )?;
    let full_output = checked_values(
        "full-resolution depth head",
        &[OUTPUT_HEAD_CHUNK, target_h, target_w],
    )?;
    checked_sum(
        "model",
        &[
            retained_backbone,
            active_backbone,
            projected,
            reduced,
            temporal_paths,
            path2,
            path1,
            output1,
            full_half,
            full_32,
            full_output,
        ],
    )
}

impl VideoDepthPredictor {
    pub fn load(path: &Path) -> Result<Box<dyn DensePredictor>> {
        Ok(Box::new(Self {
            model: VideoDepthModel::load(path)?,
            processor: VideoDepthProcessor::from_path(path)?,
        }))
    }

    fn infer(&self, source: &[DynamicImage]) -> Result<DenseOutput> {
        ensure!(!source.is_empty(), "VDA requires at least one frame");
        ensure!(
            source.len() <= MAX_FRAMES,
            "VDA accepts at most {MAX_FRAMES} frames per prediction"
        );
        let (width, height) = source[0].dimensions();
        ensure!(
            source
                .iter()
                .all(|frame| frame.dimensions() == (width, height)),
            "VDA video frames must share one resolution"
        );
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| anyhow::anyhow!("VDA source size overflowed"))?;
        let output_values = pixels
            .checked_mul(source.len())
            .ok_or_else(|| anyhow::anyhow!("VDA output size overflowed"))?;
        let (target_h, target_w) = self
            .processor
            .target_size(height as usize, width as usize)?;
        let host_values = host_working_values(source.len(), pixels, target_h, target_w)?;
        ensure!(
            host_values <= MAX_HOST_WORKING_VALUES,
            "VDA predicted host workset exceeds the {}-value safety limit",
            MAX_HOST_WORKING_VALUES
        );
        let model_values = model_working_values(&self.model.config, target_h, target_w)?;
        let total_values = host_values
            .checked_add(model_values)
            .ok_or_else(|| anyhow::anyhow!("VDA total workset overflowed"))?;
        ensure!(
            total_values <= MAX_TOTAL_WORKING_VALUES,
            "VDA predicted host and model workset exceeds the {}-value safety limit",
            MAX_TOTAL_WORKING_VALUES
        );

        let mut aligned = Vec::new();
        let mut reference = Vec::new();
        for plan in conditioned_source_indices(source.len()) {
            let conditioned: Vec<_> = plan
                .iter()
                .map(|&index| self.processor.preprocess(&source[index]))
                .collect::<Result<_>>()?;
            let input = mlxcel_core::stack_owned(&conditioned, 0);
            let depth = self.model.forward_window(&input)?;
            let shape = mlxcel_core::array_shape(&depth);
            let depth = mlxcel_core::reshape(&depth, &[shape[0], shape[1], shape[2], 1]);
            let depth =
                resize_bilinear_nhwc(&depth, (height as usize, width as usize), true, false)?;
            let depth =
                mlxcel_core::reshape(&depth, &[INFER_LEN as i32, height as i32, width as i32]);
            mlxcel_core::eval(&depth);
            let flat = mlxcel_core::utils::array_to_vec_f32(&depth);
            ensure!(
                flat.len() == INFER_LEN * pixels,
                "VDA window returned {} values, expected {}",
                flat.len(),
                INFER_LEN * pixels
            );
            ensure!(
                flat.iter().all(|value| value.is_finite()),
                "VDA model produced non-finite depth values"
            );
            let window = flat
                .chunks_exact(pixels)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>();
            align_window(
                &mut aligned,
                &mut reference,
                window,
                self.model.config.metric,
                pixels,
            )?;
        }
        ensure!(
            aligned.len() >= source.len(),
            "VDA alignment returned too few frames"
        );
        let mut values = Vec::with_capacity(output_values);
        for frame in &aligned[..source.len()] {
            values.extend_from_slice(frame);
        }
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            "depth".into(),
            mlxcel_core::from_slice_f32(
                &values,
                &[source.len() as i32, height as i32, width as i32],
            ),
        );
        let mut meta = Map::new();
        meta.insert("metric".into(), Value::Bool(self.model.config.metric));
        meta.insert(
            "units".into(),
            Value::String(
                if self.model.config.metric {
                    "metres"
                } else {
                    "relative_disparity"
                }
                .into(),
            ),
        );
        meta.insert("frames".into(), json!(source.len()));
        Ok(DenseOutput {
            tensors,
            meta,
            ..Default::default()
        })
    }
}

fn window_starts(frames: usize) -> Vec<usize> {
    (0..frames).step_by(FRAME_STEP).collect()
}

fn conditioned_source_indices(frames: usize) -> Vec<[usize; INFER_LEN]> {
    let last = frames.saturating_sub(1);
    let mut plans: Vec<[usize; INFER_LEN]> = Vec::new();
    for start in window_starts(frames) {
        let mut current = std::array::from_fn(|offset| (start + offset).min(last));
        if let Some(previous) = plans.last() {
            for (destination, &keyframe) in KEYFRAMES.iter().enumerate() {
                current[destination] = previous[keyframe];
            }
        }
        plans.push(current);
    }
    plans
}

impl DensePredictor for VideoDepthPredictor {
    fn family(&self) -> &'static str {
        "video_depth_anything"
    }
    fn tasks(&self) -> &[&'static str] {
        TASKS
    }
    fn predict(&self, input: &DenseInput, _opts: &Map<String, Value>) -> Result<DenseOutput> {
        match input {
            DenseInput::Image(image) => self.infer(std::slice::from_ref(image)),
            DenseInput::Video { frames, .. } => self.infer(frames),
        }
    }
}

fn compute_scale_and_shift(
    prediction: &[Vec<f32>],
    target: &[Vec<f32>],
    pixels: usize,
) -> Result<(f32, f32)> {
    ensure!(
        prediction.len() == target.len()
            && prediction.iter().all(|row| row.len() == pixels)
            && target.iter().all(|row| row.len() == pixels),
        "VDA alignment inputs have inconsistent shapes"
    );
    let (mut a00, mut a01, mut a11, mut b0, mut b1) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for (prediction, target) in prediction.iter().zip(target) {
        for (&p, &t) in prediction.iter().zip(target) {
            a00 += p * p;
            a01 += p;
            a11 += 1.0;
            b0 += p * t;
            b1 += t;
        }
    }
    let determinant = a00 * a11 - a01 * a01;
    Ok(if determinant != 0.0 {
        (
            (a11 * b0 - a01 * b1) / determinant,
            (-a01 * b0 + a00 * b1) / determinant,
        )
    } else {
        (1.0, 0.0)
    })
}

fn transform(frame: &[f32], scale: f32, shift: f32) -> Result<Vec<f32>> {
    ensure!(
        scale.is_finite() && shift.is_finite(),
        "VDA alignment produced a non-finite scale or shift"
    );
    let transformed: Vec<f32> = frame
        .iter()
        .map(|value| (value * scale + shift).max(0.0))
        .collect();
    ensure!(
        transformed.iter().all(|value| value.is_finite()),
        "VDA alignment produced non-finite depth values"
    );
    Ok(transformed)
}

fn align_window(
    aligned: &mut Vec<Vec<f32>>,
    reference: &mut Vec<Vec<f32>>,
    window: Vec<Vec<f32>>,
    metric: bool,
    pixels: usize,
) -> Result<()> {
    ensure!(
        window.len() == INFER_LEN && window.iter().all(|frame| frame.len() == pixels),
        "VDA alignment requires one consistently shaped 32-frame window"
    );
    let align_len = OVERLAP - INTERP_LEN;
    let keyframes = &KEYFRAMES[..align_len];
    if aligned.is_empty() {
        reference.extend(keyframes.iter().map(|&index| window[index].clone()));
        aligned.extend(window);
    } else {
        let current: Vec<Vec<f32>> = (0..keyframes.len())
            .map(|index| window[index].clone())
            .collect();
        let (scale, shift) = if metric {
            (1.0, 0.0)
        } else {
            compute_scale_and_shift(&current, &reference, pixels)?
        };
        let post: Vec<Vec<f32>> = window[align_len..OVERLAP]
            .iter()
            .map(|frame| transform(frame, scale, shift))
            .collect::<Result<_>>()?;
        let begin = aligned.len() - INTERP_LEN;
        for index in 0..INTERP_LEN {
            let weight = index as f32 / (INTERP_LEN - 1) as f32;
            for pixel in 0..pixels {
                aligned[begin + index][pixel] =
                    aligned[begin + index][pixel] * (1.0 - weight) + post[index][pixel] * weight;
            }
        }
        for frame in &window[OVERLAP..] {
            aligned.push(transform(frame, scale, shift)?);
        }
        reference.truncate(1);
        for &keyframe in &keyframes[1..] {
            reference.push(transform(&window[keyframe], scale, shift)?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padding_matches_upstream() {
        let padding = |frames| (FRAME_STEP - frames % FRAME_STEP) % FRAME_STEP + OVERLAP;
        assert_eq!(padding(40), 14);
        assert_eq!(padding(100), 20);
        assert_eq!(padding(44), 10);
        assert_eq!(padding(1), 31);
    }
    #[test]
    fn window_starts_match_upstream() {
        assert_eq!(window_starts(40), [0, 22]);
        assert_eq!(window_starts(100), [0, 22, 44, 66, 88]);
    }
    #[test]
    fn keyframe_conditioning_uses_the_previous_conditioned_window() {
        let plans = conditioned_source_indices(100);
        assert_eq!(
            &plans[1][..OVERLAP],
            &[0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
        );
        assert_eq!(&plans[1][OVERLAP..], &(32..54).collect::<Vec<_>>());
        assert_eq!(plans[2][0], 0);
        // Conditioning is recursive: plan 2 takes position 12 from plan 1,
        // whose source frame is 34 after the first 22-frame advance.
        assert_eq!(plans[2][1], 34);
    }
    #[test]
    fn still_workset_counts_both_padded_window_copies() {
        let target = 14;
        let expected = 2 + INFER_LEN * 2 + target * target * 3 * INFER_LEN * 2;
        assert_eq!(host_working_values(1, 1, target, target).unwrap(), expected);
        assert_eq!(conditioned_source_indices(1).len(), 1);
        assert_eq!(conditioned_source_indices(FRAME_STEP).len(), 1);
        assert_eq!(conditioned_source_indices(FRAME_STEP + 1).len(), 2);
    }

    #[test]
    fn model_workset_bounds_custom_large_geometry_but_keeps_published_size() {
        let config = super::super::config::VdaConfig {
            encoder: "vitl".into(),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let published_host = host_working_values(64, 1920 * 1080, 518, 924).unwrap();
        let published_model = model_working_values(&config, 518, 924).unwrap();
        assert!(published_host <= MAX_HOST_WORKING_VALUES);
        assert!(published_host + published_model <= MAX_TOTAL_WORKING_VALUES);

        let custom_host = host_working_values(1, 1, 1372, 1372).unwrap();
        let custom_model = model_working_values(&config, 1372, 1372).unwrap();
        assert!(custom_host + custom_model > MAX_TOTAL_WORKING_VALUES);
    }

    #[test]
    fn singular_alignment_uses_identity() {
        assert_eq!(
            compute_scale_and_shift(&[vec![1.0; 2]], &[vec![3.0; 2]], 2).unwrap(),
            (1.0, 0.0)
        );
    }
    #[test]
    fn nonsingular_alignment_recovers_scale_and_shift() {
        let prediction = vec![vec![1.0, 2.0, 4.0]];
        let target = vec![vec![7.0, 9.0, 13.0]];
        let (scale, shift) = compute_scale_and_shift(&prediction, &target, 3).unwrap();
        assert!((scale - 2.0).abs() < 1e-5);
        assert!((shift - 5.0).abs() < 1e-5);
    }
    #[test]
    fn interpolation_includes_both_endpoints() {
        let mut windows = vec![vec![vec![1.0]; INFER_LEN], vec![vec![3.0]; INFER_LEN]];
        windows[1][0][0] = 1.0;
        windows[1][1][0] = 1.0;
        let mut output = Vec::new();
        let mut reference = Vec::new();
        align_window(&mut output, &mut reference, windows.remove(0), true, 1).unwrap();
        align_window(&mut output, &mut reference, windows.remove(0), true, 1).unwrap();
        assert_eq!(output[INFER_LEN - INTERP_LEN][0], 1.0);
        assert_eq!(output[INFER_LEN - 1][0], 3.0);
    }
}

#[cfg(test)]
#[path = "real_tests.rs"]
mod real_tests;
