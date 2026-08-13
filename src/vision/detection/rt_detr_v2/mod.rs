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
//! https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/rt_detr_v2). Architecture:
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
//! Dtype: the forward graph inherits the checkpoint dtype. `pixel_values`
//! enters as f32, but the first conv against a bf16 weight settles the graph
//! into bf16, so `pred_logits` and `pred_boxes` come back as bf16 for the
//! shipped bf16 checkpoints. This matches the reference, which also runs in the
//! checkpoint dtype.
//!
//! Because the output dtype tracks the checkpoint rather than being fixed,
//! anything that reads these arrays back as host floats must convert by dtype
//! rather than assuming a width. [`predictor`]'s readback casts to f32 before
//! touching raw bytes for exactly this reason; see the note on its `read_output_f32`
//! for what a fixed-width parse does to a bf16 buffer.

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
