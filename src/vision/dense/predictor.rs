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

use std::collections::BTreeMap;

use anyhow::Result;
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Image or decoded video passed to a dense-prediction family.
#[derive(Debug)]
pub enum DenseInput {
    Image(DynamicImage),
    Video {
        frames: Vec<DynamicImage>,
        /// Requested sampling rate of `frames`.
        fps: f64,
    },
}

impl DenseInput {
    /// Source width and height. Video frames are guaranteed non-empty by the loader.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        use image::GenericImageView;
        match self {
            Self::Image(image) => Some(image.dimensions()),
            Self::Video { frames, .. } => frames.first().map(GenericImageView::dimensions),
        }
    }

    pub fn is_video(&self) -> bool {
        matches!(self, Self::Video { .. })
    }
}

/// One person's keypoints in source-image pixel coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonKeypoints {
    pub bbox: [f32; 4],
    /// `(x, y, score)` in source pixels.
    pub points: Vec<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub names: Option<Vec<String>>,
}

/// Model-neutral dense-prediction result.
///
/// MLX arrays are owned through `UniquePtr` because `MlxArray` is an opaque C++
/// bridge type. Keys retain the upstream predictor's output-dictionary names.
#[derive(Default)]
pub struct DenseOutput {
    pub tensors: BTreeMap<String, UniquePtr<MlxArray>>,
    pub keypoints: Vec<PersonKeypoints>,
    pub meta: Map<String, Value>,
}

/// Common interface implemented by each dense-prediction family.
pub trait DensePredictor {
    fn family(&self) -> &'static str;
    fn tasks(&self) -> &[&'static str];
    fn predict(&self, input: &DenseInput, opts: &Map<String, Value>) -> Result<DenseOutput>;
}
