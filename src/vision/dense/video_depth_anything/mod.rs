// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Video Depth Anything temporal monocular depth prediction.

mod config;
mod dpt;
mod model;
mod motion;
mod predictor;
mod processor;
mod weights;

use anyhow::Result;
use std::path::Path;

use super::DensePredictor;

pub use predictor::{INFER_LEN, INTERP_LEN, KEYFRAMES, OVERLAP, VideoDepthPredictor};

pub fn load(path: &Path) -> Result<Box<dyn DensePredictor>> {
    VideoDepthPredictor::load(path)
}
