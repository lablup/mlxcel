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

//! YOLO11 icon detector used by OmniParser v2.
//!
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/yolo11/yolo11.py
//! and `inference.py` for the fixed OmniParser `icon_detect` topology: a C3k2
//! backbone, SPPF and C2PSA attention neck, and an anchor-free DFL detection
//! head. Tensors in the converted checkpoint are already in MLX NHWC layout and
//! BatchNorm uses running statistics, so inference is pure MLX.

mod config;
mod layers;
mod model;
mod predictor;
mod weights;

pub use config::Yolo11Config;
pub use predictor::{
    DEFAULT_IOU, DEFAULT_THRESHOLD, Letterbox, Yolo11Predictor, YoloDetection, non_max_suppression,
    prepare_image,
};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
