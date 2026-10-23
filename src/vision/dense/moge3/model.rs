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

//! Complete MoGe3 model, sparse refinement, and camera-space inference.

use std::path::Path;

use anyhow::{Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{MAX_DECODER_PIXELS, Moge3Config};
use super::conv_stack::{ConvStack, ScaleHead};
use super::encoder::Moge3Encoder;
use super::geometry::{
    depth_map_to_point_map, intrinsics_from_focal, normalized_view_plane_uv, recover_focal_shift,
};
use super::processor::{InferOptions, validate_float_image, validate_output_budget};
use super::refiner::Sparse3dUnet;
use super::sparse::SparseShape;
use super::weights;
use crate::vision::dense::interpolate::resize_bilinear_nhwc;

pub struct Moge3Output {
    pub points: UniquePtr<MlxArray>,
    pub depth: UniquePtr<MlxArray>,
    pub mask: UniquePtr<MlxArray>,
    pub normal: UniquePtr<MlxArray>,
    pub intrinsics: UniquePtr<MlxArray>,
    pub points_per_step: Vec<UniquePtr<MlxArray>>,
    pub depth_per_step: Vec<UniquePtr<MlxArray>>,
    pub intrinsics_per_step: Vec<UniquePtr<MlxArray>>,
}
struct RawOutput {
    points: Vec<UniquePtr<MlxArray>>,
    normal: UniquePtr<MlxArray>,
    mask: UniquePtr<MlxArray>,
    scale: UniquePtr<MlxArray>,
}

pub struct Moge3 {
    config: Moge3Config,
    encoder: Moge3Encoder,
    neck: ConvStack,
    points: ConvStack,
    mask: ConvStack,
    normal: ConvStack,
    scale: ScaleHead,
    refiner: Option<Sparse3dUnet>,
}
impl Moge3 {
    pub fn config(&self) -> &Moge3Config {
        &self.config
    }

    pub fn load(path: &Path) -> Result<Self> {
        let config = Moge3Config::from_path(path)?;
        let mut weights = weights::load(path)?;
        weights::sanitize_conv_transpose_weights(&mut weights, &config)?;
        let encoder = Moge3Encoder::from_weights(&weights, &config.encoder)?;
        let neck = ConvStack::from_weights(&weights, "neck", &config.neck)?;
        let points = ConvStack::from_weights(
            &weights,
            "points_head",
            config
                .points_head
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("MoGe3 points head is required"))?,
        )?;
        let mask = ConvStack::from_weights(
            &weights,
            "mask_head",
            config
                .mask_head
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("MoGe3 mask head is required"))?,
        )?;
        let normal = ConvStack::from_weights(
            &weights,
            "normal_head",
            config
                .normal_head
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("MoGe3 normal head is required"))?,
        )?;
        let scale = ScaleHead::from_weights(
            &weights,
            config
                .scale_head
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("MoGe3 scale head is required"))?,
        )?;
        let refiner = config
            .refiner
            .as_ref()
            .map(|v| Sparse3dUnet::from_weights(&weights, v))
            .transpose()?;
        Ok(Self {
            config,
            encoder,
            neck,
            points,
            mask,
            normal,
            scale,
            refiner,
        })
    }

    fn voxelize(
        &self,
        coord: &MlxArray,
    ) -> Result<(
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        SparseShape,
        UniquePtr<MlxArray>,
    )> {
        let s = mlxcel_core::array_shape(coord);
        ensure!(
            s.len() == 4 && s[0] > 0 && s[1] > 0 && s[2] > 0 && s[3] == 3,
            "MoGe3 point coordinates must be [B,H,W,3]"
        );
        let coord = mlxcel_core::astype(coord, mlxcel_core::dtype::FLOAT32);
        let logz = mlxcel_core::reshape(
            &mlxcel_core::slice(&coord, &[0, 0, 0, 2], &[s[0], s[1], s[2], 3]),
            &[s[0], s[1], s[2]],
        );
        let zq = mlxcel_core::astype(
            &mlxcel_core::round(&mlxcel_core::multiply(
                &logz,
                &mlxcel_core::from_slice_f32(&[self.config.refiner_depth_resolution as f32], &[1]),
            )),
            mlxcel_core::dtype::INT64,
        );
        let zmin = mlxcel_core::min_axis(&mlxcel_core::min_axis(&zq, 2, true), 1, true);
        let zidx = mlxcel_core::astype(
            &mlxcel_core::subtract(&zq, &zmin),
            mlxcel_core::dtype::INT32,
        );
        let zmax = mlxcel_core::max_all(&zidx);
        mlxcel_core::eval(&zmax);
        let depth = i64::from(mlxcel_core::item_i32(&zmax)) + 1;
        let b = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(&mlxcel_core::arange_i32(0, s[0], 1), &[s[0], 1, 1]),
            &[s[0], s[1], s[2]],
        );
        let i = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(&mlxcel_core::arange_i32(0, s[1], 1), &[1, s[1], 1]),
            &[s[0], s[1], s[2]],
        );
        let j = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(&mlxcel_core::arange_i32(0, s[2], 1), &[1, 1, s[2]]),
            &[s[0], s[1], s[2]],
        );
        let coords = mlxcel_core::reshape(
            &mlxcel_core::stack_owned(&[b, i, j, zidx], -1),
            &[s[0] * s[1] * s[2], 4],
        );
        let feats = mlxcel_core::reshape(&coord, &[s[0] * s[1] * s[2], 3]);
        Ok((
            feats,
            coords,
            SparseShape::new(s[0] as i64, s[1] as i64, s[2] as i64, depth)?,
            logz,
        ))
    }
    fn refine_logz(&self, coord: &MlxArray, encoder: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let refiner = self
            .refiner
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MoGe3 refiner is disabled"))?;
        let s = mlxcel_core::array_shape(coord);
        let (feats, coords, shape, logz) = self.voxelize(coord)?;
        let update = refiner.forward(&feats, &coords, shape, encoder)?;
        Ok(mlxcel_core::add(
            &logz,
            &mlxcel_core::reshape(
                &mlxcel_core::astype(&update, mlxcel_core::dtype::FLOAT32),
                &[s[0], s[1], s[2]],
            ),
        ))
    }
    fn remap(coord: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(coord);
        let xy = mlxcel_core::slice(coord, &[0, 0, 0, 0], &[s[0], s[1], s[2], 2]);
        let logz = mlxcel_core::slice(coord, &[0, 0, 0, 2], &[s[0], s[1], s[2], 3]);
        let z = mlxcel_core::exp(&logz);
        mlxcel_core::concatenate(&mlxcel_core::multiply(&xy, &z), &z, -1)
    }

    fn forward(&self, image: &MlxArray, options: InferOptions) -> Result<RawOutput> {
        let s = mlxcel_core::array_shape(image);
        ensure!(s.len() == 4 && s[3] == 3, "MoGe3 model expects NHWC RGB");
        let aspect = s[2] as f64 / s[1] as f64;
        let (rows, cols) = token_grid(options.num_tokens, aspect)?;
        validate_decoder_budget(s[0] as usize, rows, cols, self.neck.levels())?;
        let (feature, cls) = self.encoder.forward(image, rows, cols)?;
        let batch = s[0];
        let mut inputs = Vec::with_capacity(self.neck.levels());
        for level in 0..self.neck.levels() {
            let h = rows << level;
            let w = cols << level;
            let uv = normalized_view_plane_uv(w, h, Some(aspect as f32))?;
            let uv = mlxcel_core::broadcast_to(
                &mlxcel_core::reshape(&uv, &[1, h as i32, w as i32, 2]),
                &[batch, h as i32, w as i32, 2],
            );
            inputs.push(if level == 0 {
                mlxcel_core::concatenate(&feature, &uv, -1)
            } else {
                uv
            })
        }
        let neck = self.neck.forward(&inputs)?;
        let raw = self.points.forward(&neck)?.pop().expect("points output");
        let mut normal = self.normal.forward(&neck)?.pop().expect("normal output");
        let mut mask = self.mask.forward(&neck)?.pop().expect("mask output");
        let scale = mlxcel_core::exp(&mlxcel_core::squeeze_axis(&self.scale.forward(&cls), -1));
        let mut current = mlxcel_core::astype(&raw, mlxcel_core::dtype::FLOAT32);
        let mut coords = vec![mlxcel_core::copy(&current)];
        for _ in 0..options.refine_steps {
            let logz = self.refine_logz(
                &mlxcel_core::stop_gradient(&current),
                &mlxcel_core::stop_gradient(&inputs[0]),
            )?;
            let rs = mlxcel_core::array_shape(&current);
            let xy = mlxcel_core::slice(&current, &[0, 0, 0, 0], &[rs[0], rs[1], rs[2], 2]);
            current = mlxcel_core::concatenate(&xy, &mlxcel_core::expand_dims(&logz, -1), -1);
            coords.push(mlxcel_core::copy(&current))
        }
        if !options.return_per_step {
            coords = vec![coords.pop().expect("one point step")]
        }
        let refs: Vec<&MlxArray> = coords
            .iter()
            .map(|v| v.as_ref().expect("MLX array"))
            .collect();
        let coords = mlxcel_core::concatenate_many(&refs, 0);
        let resized = resize_bilinear_nhwc(&coords, (s[1] as usize, s[2] as usize), false, false)?;
        let mut output_points = Vec::new();
        for step in 0..refs.len() {
            let begin = step as i32 * batch;
            let coord =
                mlxcel_core::slice(&resized, &[begin, 0, 0, 0], &[begin + batch, s[1], s[2], 3]);
            output_points.push(Self::remap(&coord))
        }
        normal = resize_bilinear_nhwc(&normal, (s[1] as usize, s[2] as usize), false, false)?;
        let norm = mlxcel_core::maximum(
            &mlxcel_core::linalg_norm(&normal, -1, true),
            &mlxcel_core::from_slice_f32(&[1e-12], &[1]),
        );
        normal = mlxcel_core::divide(&normal, &norm);
        let resized_mask =
            resize_bilinear_nhwc(&mask, (s[1] as usize, s[2] as usize), false, false)?;
        mask = mlxcel_core::sigmoid(&mlxcel_core::squeeze_axis(&resized_mask, -1));
        Ok(RawOutput {
            points: output_points,
            normal,
            mask,
            scale,
        })
    }

    pub fn estimate(&self, image: &MlxArray) -> Result<Moge3Output> {
        let options = InferOptions {
            num_tokens: self.config.num_tokens_range[1],
            force_projection: true,
            apply_mask: true,
            fov_x: None,
            refine_steps: if self.config.refiner.is_some() { 3 } else { 0 },
            return_per_step: false,
        };
        self.estimate_with_options(image, options)
    }
    pub fn estimate_with_options(
        &self,
        image: &MlxArray,
        options: InferOptions,
    ) -> Result<Moge3Output> {
        self.config.validate()?;
        options.validate(&self.config)?;
        validate_float_image(image)?;
        let original = mlxcel_core::array_shape(image);
        validate_output_budget(&original, options)?;
        let omit = original.len() == 3;
        let image = if omit {
            mlxcel_core::expand_dims(image, 0)
        } else {
            mlxcel_core::copy(image)
        };
        let s = mlxcel_core::array_shape(&image);
        let aspect = s[2] as f32 / s[1] as f32;
        let raw = self.forward(&image, options)?;
        let mask = mlxcel_core::greater(
            &mlxcel_core::astype(&raw.mask, mlxcel_core::dtype::FLOAT32),
            &mlxcel_core::from_slice_f32(&[0.5], &[1]),
        );
        let fixed = options.fov_x.map(|fov| {
            let value = aspect
                / (1.0 + aspect * aspect).sqrt()
                / (fov * std::f32::consts::PI / 360.0).tan();
            mlxcel_core::full_f32(&[s[0]], value, mlxcel_core::dtype::FLOAT32)
        });
        let mut points_steps = Vec::new();
        let mut depth_steps = Vec::new();
        let mut intrinsics_steps = Vec::new();
        let mut masks = Vec::new();
        for affine in raw.points {
            let affine = mlxcel_core::astype(&affine, mlxcel_core::dtype::FLOAT32);
            let (focal, shift) = recover_focal_shift(&affine, Some(&mask), fixed.as_deref())?;
            let intrinsics = intrinsics_from_focal(&focal, aspect);
            let z = mlxcel_core::reshape(
                &mlxcel_core::slice(&affine, &[0, 0, 0, 2], &[s[0], s[1], s[2], 3]),
                &[s[0], s[1], s[2]],
            );
            let mut depth = mlxcel_core::add(&z, &mlxcel_core::reshape(&shift, &[s[0], 1, 1]));
            let mut points = if options.force_projection {
                depth_map_to_point_map(&depth, &intrinsics)?
            } else {
                let xy = mlxcel_core::slice(&affine, &[0, 0, 0, 0], &[s[0], s[1], s[2], 2]);
                mlxcel_core::concatenate(&xy, &mlxcel_core::expand_dims(&depth, -1), -1)
            };
            points =
                mlxcel_core::multiply(&points, &mlxcel_core::reshape(&raw.scale, &[s[0], 1, 1, 1]));
            depth = mlxcel_core::multiply(&depth, &mlxcel_core::reshape(&raw.scale, &[s[0], 1, 1]));
            let valid = mlxcel_core::logical_and(
                &mask,
                &mlxcel_core::greater(&depth, &mlxcel_core::from_slice_f32(&[0.0], &[1])),
            );
            if options.apply_mask {
                let inf = mlxcel_core::from_slice_f32(&[f32::INFINITY], &[1]);
                points =
                    mlxcel_core::where_cond(&mlxcel_core::expand_dims(&valid, -1), &points, &inf);
                depth = mlxcel_core::where_cond(&valid, &depth, &inf)
            }
            points_steps.push(points);
            depth_steps.push(depth);
            intrinsics_steps.push(intrinsics);
            masks.push(valid)
        }
        let mut normal = mlxcel_core::astype(&raw.normal, mlxcel_core::dtype::FLOAT32);
        let final_mask = masks.pop().expect("one MoGe3 step");
        if options.apply_mask {
            normal = mlxcel_core::where_cond(
                &mlxcel_core::expand_dims(&final_mask, -1),
                &normal,
                &mlxcel_core::zeros_like(&normal),
            )
        }
        let mut points = mlxcel_core::copy(points_steps.last().expect("one point step"));
        let mut depth = mlxcel_core::copy(depth_steps.last().expect("one depth step"));
        let mut intrinsics =
            mlxcel_core::copy(intrinsics_steps.last().expect("one intrinsics step"));
        let mut mask = final_mask;
        if omit {
            points = mlxcel_core::squeeze_axis(&points, 0);
            depth = mlxcel_core::squeeze_axis(&depth, 0);
            intrinsics = mlxcel_core::squeeze_axis(&intrinsics, 0);
            mask = mlxcel_core::squeeze_axis(&mask, 0);
            normal = mlxcel_core::squeeze_axis(&normal, 0);
            for values in [&mut points_steps, &mut depth_steps, &mut intrinsics_steps] {
                for value in values.iter_mut() {
                    *value = mlxcel_core::squeeze_axis(value, 0)
                }
            }
        }
        if !options.return_per_step {
            points_steps.clear();
            depth_steps.clear();
            intrinsics_steps.clear()
        }
        Ok(Moge3Output {
            points,
            depth,
            mask,
            normal,
            intrinsics,
            points_per_step: points_steps,
            depth_per_step: depth_steps,
            intrinsics_per_step: intrinsics_steps,
        })
    }
}

pub(super) fn token_grid(num_tokens: usize, aspect: f64) -> Result<(usize, usize)> {
    ensure!(
        aspect.is_finite() && aspect > 0.0,
        "MoGe3 image aspect must be finite and positive"
    );
    let rows = (num_tokens as f64 / aspect)
        .sqrt()
        .round_ties_even()
        .max(1.0) as usize;
    let cols = (num_tokens as f64 * aspect)
        .sqrt()
        .round_ties_even()
        .max(1.0) as usize;
    Ok((rows, cols))
}

fn validate_decoder_budget(batch: usize, rows: usize, cols: usize, levels: usize) -> Result<()> {
    ensure!(levels > 0, "MoGe3 decoder must contain at least one level");
    let factor = 1usize
        .checked_shl((2 * (levels - 1)) as u32)
        .ok_or_else(|| anyhow::anyhow!("MoGe3 decoder scale overflow"))?;
    let finest = batch
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(cols))
        .and_then(|v| v.checked_mul(factor))
        .ok_or_else(|| anyhow::anyhow!("MoGe3 decoder pixel budget overflow"))?;
    ensure!(
        finest <= MAX_DECODER_PIXELS,
        "MoGe3 decoder requires {finest} active pixels, exceeding {MAX_DECODER_PIXELS}"
    );
    let max_dim = i32::MAX as usize >> (levels - 1);
    ensure!(
        rows <= max_dim && cols <= max_dim,
        "MoGe3 decoder dimensions exceed int32 indexing"
    );
    Ok(())
}
