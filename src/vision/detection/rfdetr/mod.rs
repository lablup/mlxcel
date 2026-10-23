// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! RF-DETR object detection and instance segmentation.

mod backbone;
mod checkpoint;
mod decoder;
mod deformable;
mod layers;
mod model;
mod predictor;
mod processor;
mod projector;
mod segmentation;
mod transformer;
mod weights;

pub mod config;
pub mod postprocess;
pub mod sanitize;

pub use config::RfDetrConfig;
pub use model::{RfDetrModel, RfDetrOutput};
pub use predictor::{DEFAULT_THRESHOLD, RfDetrPredictor};
pub use processor::{ProcessedImage, RfDetrProcessor};
pub use sanitize::{needs_sanitize, sanitize};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "real_tests.rs"]
mod real_tests;
