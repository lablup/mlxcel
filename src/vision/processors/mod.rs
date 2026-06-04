// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Image processors for vision models
//!
//! Provides the ImageProcessor trait and processor implementations.

pub mod gemma4;
pub mod gemma4_unified;
pub mod internvl;
pub mod minicpmo;
pub mod molmo;
pub mod molmo2;
pub mod molmo_point;
pub mod moondream3;
pub mod nemotron_h_nano_omni;
pub mod phi3_v;
pub mod phi4_siglip;
pub mod phi4mm;
pub mod qwen2_vl;
pub mod siglip;
pub mod youtu_vl;

use mlxcel_core::{MlxArray, UniquePtr};

/// Trait for image preprocessors
pub trait ImageProcessor {
    /// Preprocess images to tensor format ready for vision encoder
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray>;
}
