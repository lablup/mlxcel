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

//! Native SAM 3 image segmentation and video tracking components.

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod geometry;
mod image;
mod model;
mod options;
pub mod position;
pub mod postprocess;
mod predictor;
pub mod processing;
pub mod sam_components;
pub mod sanitize;
pub mod segmentation;
pub mod text_encoder;
pub mod tracker;
mod video;
pub mod vision;

pub use predictor::load;

#[cfg(test)]
#[path = "real_tests.rs"]
mod real_tests;
