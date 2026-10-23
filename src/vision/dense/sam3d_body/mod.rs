// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! SAM 3D Body's body-only monocular human mesh predictor.
mod backbone;
mod camera;
mod config;
mod decoder;
mod layers;
mod mhr_body;
mod mhr_head;
mod mhr_utils;
mod model;
mod ops;
mod predictor;
mod preprocess;
mod prompt_encoder;
mod rope;
mod sanitize;
pub use predictor::load;
