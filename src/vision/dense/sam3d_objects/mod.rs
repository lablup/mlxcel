// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! SAM 3D Objects image/mask reconstruction with object-local mesh and Gaussians.
mod bundle;
mod checkpoint_schema;
pub mod config;
mod decoders;
mod depth_schema;
mod flow;
mod flow_schema;
mod layers;
pub mod mesh;
pub mod pipeline;
mod pose;
mod processing;
pub mod sparse;
mod vision;
use super::{DenseInput, DenseOutput, DensePredictor, predictor::DenseSidecars};
use anyhow::{Result, bail};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
struct Predictor {
    path: PathBuf,
    config: config::Config,
}
pub fn load(path: &Path) -> Result<Box<dyn DensePredictor>> {
    Ok(Box::new(Predictor {
        path: path.into(),
        config: bundle::config(path)?,
    }))
}
impl DensePredictor for Predictor {
    fn family(&self) -> &'static str {
        "sam3d_objects"
    }
    fn tasks(&self) -> &[&'static str] {
        &["mesh"]
    }
    fn predict(&self, input: &DenseInput, opts: &Map<String, Value>) -> Result<DenseOutput> {
        self.predict_with_sidecars(input, &DenseSidecars::default(), opts)
    }
    fn predict_with_sidecars(
        &self,
        input: &DenseInput,
        sidecars: &DenseSidecars,
        opts: &Map<String, Value>,
    ) -> Result<DenseOutput> {
        let DenseInput::Image(image) = input else {
            bail!("SAM 3D Objects supports still images only")
        };
        pipeline::run(&self.path, &self.config, image, sidecars, opts)
    }
}
