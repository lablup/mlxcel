// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! MHR blend shapes, pose correctives, parent-ordered FK and linear skinning.
use super::{
    config::Config,
    layers::Array,
    mhr_utils::{check_last, component, einsum, last, quat_to_rotmat, scalar},
};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
#[path = "mhr_kinematics.rs"]
mod kinematics;
#[path = "mhr_weights.rs"]
pub(super) mod weights;
use weights::{f32_weight, i32_weight, index_range, ordered_scatter, parents_valid};

pub struct MhrBody {
    joints: i32,
    vertices: i32,
    offsets: Array,
    prerotations: Array,
    parents: Vec<i32>,
    transform: Array,
    base: Array,
    shapes: Array,
    expressions: Array,
    inverse_bind: Array,
    skin_indices: Array,
    skin_weights: Array,
    vertex_indices: Array,
    vertex_host: Vec<i32>,
    corrective_input: Array,
    corrective_output: Vec<i32>,
    corrective_weights: Array,
    corrective_linear: Array,
    minmax_min: Array,
    minmax_max: Array,
    minmax_indices: Array,
    minmax_host: Vec<i32>,
}
impl MhrBody {
    pub fn from_weights(w: &WeightMap, config: &Config) -> Result<Self> {
        config.validate()?;
        let j = i32::try_from(config.num_joints)?;
        let v = i32::try_from(config.num_vertices)?;
        let prefix = "head_pose.body_model";
        let f = |name: &str, shape: &[i32]| f32_weight(w, &format!("{prefix}.{name}"), shape);
        let i = |name: &str, shape: &[i32]| i32_weight(w, &format!("{prefix}.{name}"), shape);
        let (_, parents) = i("joint_parents", &[j])?;
        parents_valid(&parents)?;
        let (skin_indices, skin_host) = i("skin_indices", &[51337])?;
        index_range(&skin_host, j, "skin joint")?;
        let (vertex_indices, vertex_host) = i("vert_indices", &[51337])?;
        index_range(&vertex_host, v, "skin vertex")?;
        let (_, corrective_host) = i("pc_sparse_indices", &[2, 53136])?;
        index_range(&corrective_host[..53136], 3000, "corrective output")?;
        index_range(&corrective_host[53136..], (j - 2) * 6, "corrective input")?;
        let corrective_input = mx::from_slice_i32(&corrective_host[53136..], &[53136]);
        let mut corrective_output = weights::reserve(53136)?;
        corrective_output.extend_from_slice(&corrective_host[..53136]);
        let (minmax_indices, minmax_host) = i("minmax_parameter_index", &[198])?;
        index_range(&minmax_host, 204, "parameter limit")?;
        let minmax_min = f("minmax_min", &[198])?;
        let minmax_max = f("minmax_max", &[198])?;
        ensure!(
            mx::item_bool(&mx::all_all(&mx::less_equal(&minmax_min, &minmax_max))),
            "MHR parameter limits are inverted"
        );
        // This upstream buffer is retained in the checkpoint but not used by the limits helper.
        let _ = f("minmax_weight", &[198])?;
        Ok(Self {
            joints: j,
            vertices: v,
            offsets: f("joint_translation_offsets", &[j, 3])?,
            prerotations: f("joint_prerotations", &[j, 4])?,
            parents,
            transform: f("parameter_transform", &[j * 7, 249])?,
            base: f("base_shape", &[v, 3])?,
            shapes: f("shape_vectors", &[45, v, 3])?,
            expressions: f("face_shape_vectors", &[72, v, 3])?,
            inverse_bind: f("inverse_bind_pose", &[j, 8])?,
            skin_indices,
            skin_weights: f("skin_weights", &[51337])?,
            vertex_indices,
            vertex_host,
            corrective_input,
            corrective_output,
            corrective_weights: f("pc_sparse_weight", &[53136])?,
            corrective_linear: f("pc_linear_weight", &[v * 3, 3000])?,
            minmax_min,
            minmax_max,
            minmax_indices,
            minmax_host,
        })
    }
    /// Upstream exposes this helper but deliberately skips it during inference.
    #[allow(dead_code)]
    pub fn apply_parameter_limits(&self, parameters: &MlxArray) -> Result<Array> {
        check_last(parameters, 204)?;
        let s = mx::array_shape(parameters);
        ensure!(s.len() == 2, "MHR model parameters must be [B,204]");
        let clamped = mx::clip(
            &mx::take(parameters, &self.minmax_indices, 1),
            &self.minmax_min,
            &self.minmax_max,
        );
        let mut map = [None; 204];
        let mut mask = [0_f32; 204];
        for (i, &index) in self.minmax_host.iter().enumerate() {
            map[index as usize] = Some(i as i32);
            mask[index as usize] += 1.;
        }
        let columns: Vec<_> = map
            .iter()
            .map(|index| match index {
                Some(i) => last(&clamped, *i, *i + 1),
                None => mx::zeros(&[s[0], 1], mx::dtype::FLOAT32),
            })
            .collect();
        let refs = columns.iter().map(|v| &**v).collect::<Vec<_>>();
        let replacement = mx::concatenate_many(&refs, 1);
        let mask = mx::from_slice_f32(&mask, &[1, 204]);
        Ok(mx::add(
            &mx::multiply(parameters, &mx::subtract(&scalar(1.), &mask)),
            &mx::multiply(&replacement, &mask),
        ))
    }
    fn pose_features(&self, dofs: &MlxArray, batch: i32) -> Array {
        let jd = mx::reshape(dofs, &[batch, self.joints, 7]);
        let angles = mx::slice(&jd, &[0, 2, 3], &[batch, self.joints, 6]);
        let cx = mx::cos(&component(&angles, 0));
        let sx = mx::sin(&component(&angles, 0));
        let cy = mx::cos(&component(&angles, 1));
        let sy = mx::sin(&component(&angles, 1));
        let cz = mx::cos(&component(&angles, 2));
        let sz = mx::sin(&component(&angles, 2));
        let features = mx::stack_owned(
            &[
                mx::multiply(&cy, &cz),
                mx::multiply(&cy, &sz),
                mx::negative(&sy),
                mx::add(
                    &mx::multiply(&mx::negative(&cx), &sz),
                    &mx::multiply(&mx::multiply(&sx, &sy), &cz),
                ),
                mx::add(
                    &mx::multiply(&cx, &cz),
                    &mx::multiply(&mx::multiply(&sx, &sy), &sz),
                ),
                mx::multiply(&sx, &cy),
            ],
            -1,
        );
        let identity = mx::from_slice_f32(&[1., 0., 0., 0., 1., 0.], &[6]);
        mx::reshape(&mx::subtract(&features, &identity), &[batch, -1])
    }
    fn correctives(&self, dofs: &MlxArray, batch: i32) -> Result<Array> {
        let features = self.pose_features(dofs, batch);
        let weighted = mx::multiply(
            &mx::take(&features, &self.corrective_input, 1),
            &mx::expand_dims(&self.corrective_weights, 0),
        );
        let sparse = ordered_scatter(
            &mx::expand_dims(&weighted, -1),
            &self.corrective_output,
            3000,
        )?;
        let sparse = mx::maximum(&mx::squeeze_axis(&sparse, -1), &scalar(0.));
        let dense = mx::matmul(&sparse, &mx::transpose(&self.corrective_linear));
        Ok(mx::reshape(&dense, &[batch, self.vertices, 3]))
    }
    fn skin(&self, rest: &MlxArray, skeleton: &kinematics::Skeleton) -> Result<Array> {
        let ibp_rot = quat_to_rotmat(&last(&self.inverse_bind, 3, 7))?;
        let rotation = einsum("bjik,jkl->bjil", &[&skeleton.rotation, &ibp_rot]);
        let rotated = einsum(
            "bjik,jk->bji",
            &[&skeleton.rotation, &last(&self.inverse_bind, 0, 3)],
        );
        let translation = mx::add(&skeleton.position, &mx::multiply(&rotated, &skeleton.scale));
        let scale = mx::multiply(
            &skeleton.scale,
            &mx::expand_dims(&last(&self.inverse_bind, 7, 8), 0),
        );
        let vertices = mx::take(rest, &self.vertex_indices, 1);
        let cr = mx::take(&rotation, &self.skin_indices, 1);
        let ct = mx::take(&translation, &self.skin_indices, 1);
        let cs = mx::take(&scale, &self.skin_indices, 1);
        let transformed = mx::add(
            &mx::multiply(&cs, &einsum("bnij,bnj->bni", &[&cr, &vertices])),
            &ct,
        );
        let weighted = mx::multiply(&transformed, &mx::reshape(&self.skin_weights, &[1, -1, 1]));
        ordered_scatter(&weighted, &self.vertex_host, self.vertices as usize)
    }
    pub fn forward(
        &self,
        shape: &MlxArray,
        parameters: &MlxArray,
        expression: Option<&MlxArray>,
    ) -> Result<(Array, Array)> {
        check_last(shape, 45)?;
        check_last(parameters, 204)?;
        let s = mx::array_shape(shape);
        ensure!(
            s.len() == 2 && mx::array_shape(parameters) == [s[0], 204],
            "MHR shape/model batches disagree"
        );
        if let Some(expression) = expression {
            check_last(expression, 72)?;
            ensure!(
                mx::array_shape(expression) == [s[0], 72],
                "MHR expression batch disagrees"
            );
        }
        let batch = s[0];
        // The pinned inference path intentionally does not apply parameter limits.
        let padded = mx::concatenate(parameters, &mx::zeros(&[batch, 45], mx::dtype::FLOAT32), 1);
        let dofs = mx::matmul(&padded, &mx::transpose(&self.transform));
        let mut vertices = mx::add(
            &mx::expand_dims(&self.base, 0),
            &einsum("bs,svd->bvd", &[shape, &self.shapes]),
        );
        if let Some(expression) = expression {
            vertices = mx::add(
                &vertices,
                &einsum("bf,fvd->bvd", &[expression, &self.expressions]),
            );
        }
        vertices = mx::add(&vertices, &*self.correctives(&dofs, batch)?);
        let skeleton =
            kinematics::forward(&dofs, &self.offsets, &self.prerotations, &self.parents)?;
        Ok((self.skin(&vertices, &skeleton)?, skeleton.state))
    }
}

#[cfg(test)]
#[path = "mhr_tests.rs"]
mod tests;
