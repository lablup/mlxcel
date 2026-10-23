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

//! Shared infrastructure for models that produce maps, keypoints, or scored instances.

pub mod dinov2;
mod dinov2_resize;
pub mod interpolate;
pub mod moge3;
pub mod output;
mod output_instances;
pub mod predictor;
pub mod preprocess;
pub mod registry;
pub mod sam3;
pub mod sam3_1;
pub mod sapiens2;
pub mod video_depth_anything;

pub use output::{DenseOutputInfo, write_dense_output};
pub use predictor::{DenseInput, DenseOutput, DensePredictor, Instance, PersonKeypoints};
pub use registry::{DenseFamily, FAMILIES, load_dense_model, resolve_family};
