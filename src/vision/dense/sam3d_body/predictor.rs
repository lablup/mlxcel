// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Public dense-prediction integration; source-image boxes remain unchanged.
use super::{
    config::Config,
    model::Model,
    preprocess::{self, InputGeometry},
    sanitize,
};
use crate::vision::dense::{DenseInput, DenseOutput, DensePredictor, geometry_io::Mesh};
use anyhow::{Context, Result, bail, ensure};
use mlxcel_core as mx;
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
struct Predictor {
    path: PathBuf,
    config: Config,
}
pub fn load(path: &Path) -> Result<Box<dyn DensePredictor>> {
    let config: Config =
        super::super::registry::read_checkpoint_config(path).context("parse Body config")?;
    config.validate()?;
    Ok(Box::new(Predictor {
        path: path.into(),
        config,
    }))
}
impl DensePredictor for Predictor {
    fn family(&self) -> &'static str {
        "sam3d_body"
    }
    fn tasks(&self) -> &[&'static str] {
        &["body_mesh"]
    }
    fn predict(&self, input: &DenseInput, options: &Map<String, Value>) -> Result<DenseOutput> {
        let DenseInput::Image(image) = input else {
            bail!("SAM 3D Body supports still images only")
        };
        ensure!(
            options
                .get("task")
                .is_none_or(|v| v.as_str() == Some("body_mesh")),
            "Body task must be body_mesh"
        );
        let geometry = InputGeometry::parse(image, options)?;
        let pixels = preprocess::prepare(image, &geometry, &self.config)?;
        let weights = sanitize::load(&self.path)?;
        let model = Model::from_weights(&weights, &self.config)?;
        drop(weights);
        let (body, camera) = model.forward(&pixels, &geometry)?;
        let mut out = DenseOutput::default();
        for (name, value, shape) in [
            ("vertices", body.vertices, vec![18439, 3]),
            ("keypoints_3d", body.keypoints_3d, vec![70, 3]),
            ("joint_coords", body.joint_coords, vec![127, 3]),
            ("camera", camera, vec![3]),
        ] {
            let value = mx::reshape(&value, &shape);
            ensure!(
                mx::array_dtype(&value) == mx::dtype::FLOAT32,
                "Body output {name} must be F32"
            );
            ensure!(
                mx::item_bool(&mx::all_all(&mx::isfinite(&value))),
                "non-finite Body output {name}"
            );
            out.tensors.insert(name.into(), value);
        }
        let vertices = mx::utils::array_to_vec_f32(&out.tensors["vertices"]);
        let vertices = vertices
            .chunks_exact(3)
            .map(|v| [v[0], v[1], v[2]])
            .collect::<Vec<_>>();
        let min = std::array::from_fn::<_, 3, _>(|a| {
            vertices.iter().map(|v| v[a]).fold(f32::INFINITY, f32::min)
        });
        let max = std::array::from_fn::<_, 3, _>(|a| {
            vertices
                .iter()
                .map(|v| v[a])
                .fold(f32::NEG_INFINITY, f32::max)
        });
        out.meshes.push((
            "mesh".into(),
            Mesh {
                vertices,
                faces: model.head.faces().to_vec(),
                vertex_colors: None,
            },
        ));
        out.meta=json!({"family":"sam3d_body","task":"body_mesh","bbox":geometry.bbox,"cam_int":geometry.cam_int,"mesh_span":{"min":min,"max":max},"coordinates":"body-local meters; MHR Y/Z negated; camera is raw weak-perspective [scale,tx,ty]","image_size":self.config.image_size,"weight_precision":"BF16 backbone, F32 decoder and MHR; integer topology preserved"}).as_object().context("metadata object")?.clone();
        Ok(out)
    }
}
