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

//! RT-DETRv2 real-time object-detection model.
//!
//! A full Rust port of the upstream mlx-vlm RT-DETRv2 model (PR #1195,
//! `references/mlx-vlm/mlx_vlm/models/rt_detr_v2/`). Architecture:
//!
//! ```text
//! Image (NHWC) -> ResNet-50/101-vd backbone (strides 8/16/32)
//!              -> per-level 1x1 conv+BN encoder input projection
//!              -> HybridEncoder: AIFI (deepest level) + FPN + PAN
//!              -> encoder query selection (top-K over flat positions x labels)
//!              -> deformable-attention decoder (iterative bbox refinement)
//!              -> {pred_logits (B, Q, num_labels), pred_boxes (B, Q, 4)}
//! ```
//!
//! Unlike text/VLM models, RT-DETRv2 produces bounding boxes rather than a
//! token stream, so it lives outside the `LanguageModel`/generate flow. It is
//! driven through [`RtDetrV2Predictor`] (see the `detect` CLI subcommand).
//!
//! The whole forward path runs in float32 for box-coordinate precision,
//! regardless of the stored checkpoint dtype (the shipped checkpoints are
//! bf16). This matches the reference, whose dtype-sensitive ops (softmax, SDPA,
//! anchor logits) are computed in f32 and whose predictor reads f32 anyway.

pub mod backbone;
pub mod common;
pub mod config;
pub mod hybrid_encoder;
pub mod layers;
pub mod model;
pub mod predictor;
pub mod processor;
pub mod sanitize;
pub mod transformer;

#[cfg(test)]
mod tests;

pub use config::{BackboneConfig, RtDetrV2Config};
pub use model::{DetectionOutput, RtDetrV2Model};
pub use predictor::{DEFAULT_THRESHOLD, Detection, DetectionResult, RtDetrV2Predictor};
pub use processor::{ProcessorConfig, RtDetrV2Processor};
