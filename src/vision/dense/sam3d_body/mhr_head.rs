// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Full MHR pose head, including PCA hands, corrective skinning and keypoints.
use super::{
    config::Config,
    layers::{Array, Linear},
    mhr_body::{
        MhrBody,
        weights::{f32_weight, i32_weight, index_range},
    },
    mhr_utils::{
        check_last, compact_body, compact_hand, einsum, last, rot6d_to_rotmat, rotmat_to_euler_zyx,
        scalar,
    },
};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};

pub struct BodyOutput {
    pub vertices: Array,
    pub keypoints_3d: Array,
    pub joint_coords: Array,
    pub model_params: Array,
    pub shape: Array,
}
pub struct MhrHead {
    first: Linear,
    last: Linear,
    body: MhrBody,
    scale_mean: Array,
    scale_components: Array,
    hand_mean: Array,
    hand_components: Array,
    left: ScatterSet,
    right: ScatterSet,
    keypoint_mapping: Array,
    faces: Vec<[u32; 3]>,
}
struct ScatterSet {
    mask: Array,
    columns: Array,
}
impl ScatterSet {
    fn new(indices: &[i32]) -> Result<Self> {
        ensure!(indices.len() == 27, "MHR hand index count must be 27");
        index_range(indices, 136, "hand pose")?;
        let mut mask = [0_f32; 136];
        let mut columns = [0_i32; 136];
        for (k, &d) in indices.iter().enumerate() {
            mask[d as usize] = 1.;
            columns[d as usize] = k as i32;
        }
        Ok(Self {
            mask: mx::from_slice_f32(&mask, &[1, 136]),
            columns: mx::from_slice_i32(&columns, &[136]),
        })
    }
    fn forward(&self, original: &MlxArray, values: &MlxArray) -> Array {
        let replacement = mx::take(values, &self.columns, 1);
        mx::add(
            &mx::multiply(original, &mx::subtract(&scalar(1.), &self.mask)),
            &mx::multiply(&replacement, &self.mask),
        )
    }
}
impl MhrHead {
    pub fn from_weights(w: &WeightMap, config: &Config) -> Result<Self> {
        config.validate()?;
        let dim = i32::try_from(config.decoder_dim)?;
        let first = Linear::from_weights(w, "head_pose.proj.layers.0.0")?;
        let last = Linear::from_weights(w, "head_pose.proj.layers.1")?;
        for (prefix, output) in [
            ("head_pose.proj.layers.0.0", dim),
            ("head_pose.proj.layers.1", 519),
        ] {
            let _ = f32_weight(w, &format!("{prefix}.weight"), &[output, dim])?;
            let _ = f32_weight(w, &format!("{prefix}.bias"), &[output])?;
        }
        let (_, left) = i32_weight(w, "head_pose.hand_joint_idxs_left", &[27])?;
        let (_, right) = i32_weight(w, "head_pose.hand_joint_idxs_right", &[27])?;
        let (_, faces_raw) =
            i32_weight(w, "head_pose.faces", &[i32::try_from(config.num_faces)?, 3])?;
        index_range(&faces_raw, i32::try_from(config.num_vertices)?, "mesh face")?;
        let mut faces = super::mhr_body::weights::reserve(config.num_faces)?;
        faces.extend(
            faces_raw
                .chunks_exact(3)
                .map(|v| [v[0] as u32, v[1] as u32, v[2] as u32]),
        );
        Ok(Self {
            first,
            last,
            body: MhrBody::from_weights(w, config)?,
            scale_mean: f32_weight(w, "head_pose.scale_mean", &[68])?,
            scale_components: f32_weight(w, "head_pose.scale_comps", &[28, 68])?,
            hand_mean: f32_weight(w, "head_pose.hand_pose_mean", &[54])?,
            hand_components: f32_weight(w, "head_pose.hand_pose_comps", &[54, 54])?,
            left: ScatterSet::new(&left)?,
            right: ScatterSet::new(&right)?,
            keypoint_mapping: f32_weight(
                w,
                "head_pose.keypoint_mapping",
                &[308, i32::try_from(config.num_vertices + config.num_joints)?],
            )?,
            faces,
        })
    }
    pub fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }
    pub fn forward(&self, x: &MlxArray, initial: Option<&MlxArray>) -> Result<BodyOutput> {
        let dim = mx::array_shape(&self.first.weight)[1];
        check_last(x, dim)?;
        let shape = mx::array_shape(x);
        ensure!(shape.len() == 2, "MHR pose token must be F32 [B,{dim}]");
        let batch = shape[0];
        let mut pred = self
            .last
            .forward(&mx::maximum(&self.first.forward(x), &scalar(0.)));
        if let Some(initial) = initial {
            check_last(initial, 519)?;
            ensure!(
                mx::array_shape(initial) == [batch, 519],
                "MHR initial estimate batch disagrees"
            );
            pred = mx::add(&pred, initial);
        }
        let global_rotation = rotmat_to_euler_zyx(&*rot6d_to_rotmat(&last(&pred, 0, 6))?)?;
        let shape = last(&pred, 266, 311);
        let face = mx::multiply_scalar(&last(&pred, 447, 519), 0.);
        let pose = compact_body(&last(&pred, 6, 266))?;
        let mut hand_mask = [1_f32; 133];
        hand_mask[62..116].fill(0.);
        let pose = mx::multiply(&pose, &mx::from_slice_f32(&hand_mask, &[133]));
        let mut jaw_mask = [1_f32; 133];
        jaw_mask[130..].fill(0.);
        let pose = mx::multiply(&pose, &mx::from_slice_f32(&jaw_mask, &[133]));
        let scales = mx::add(
            &mx::expand_dims(&self.scale_mean, 0),
            &mx::matmul(&last(&pred, 311, 339), &self.scale_components),
        );
        let translation = mx::multiply_scalar(&mx::zeros(&[batch, 3], mx::dtype::FLOAT32), 10.);
        let pose = mx::concatenate_many(&[&translation, &global_rotation, &last(&pose, 0, 130)], 1);
        let hand = last(&pred, 339, 447);
        let left = compact_hand(&mx::add(
            &mx::expand_dims(&self.hand_mean, 0),
            &mx::matmul(&last(&hand, 0, 54), &self.hand_components),
        ))?;
        let right = compact_hand(&mx::add(
            &mx::expand_dims(&self.hand_mean, 0),
            &mx::matmul(&last(&hand, 54, 108), &self.hand_components),
        ))?;
        let pose = self.right.forward(&self.left.forward(&pose, &left), &right);
        let model_params = mx::concatenate(&pose, &scales, 1);
        let (vertices, state) = self.body.forward(&shape, &model_params, Some(&face))?;
        let vertices = mx::divide_scalar(&vertices, 100.);
        let joints = mx::divide_scalar(&last(&state, 0, 3), 100.);
        let combined = mx::concatenate(&vertices, &joints, 1);
        let keypoints = einsum("kv,bvd->bkd", &[&self.keypoint_mapping, &combined]);
        let keypoints = mx::slice(&keypoints, &[0, 0, 0], &[batch, 70, 3]);
        let flip = mx::from_slice_f32(&[1., -1., -1.], &[3]);
        Ok(BodyOutput {
            vertices: mx::multiply(&vertices, &flip),
            keypoints_3d: mx::multiply(&keypoints, &flip),
            joint_coords: mx::multiply(&joints, &flip),
            model_params,
            shape,
        })
    }
}
