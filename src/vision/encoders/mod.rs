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

//! Vision encoder implementations
//!
//! Provides the VisionEncoder trait and encoder implementations.

pub mod deepseek_vl2;
pub mod deepseekocr_clip;
pub mod deepseekocr_qwen2;
pub mod deepseekocr_sam;
pub mod dots_ocr;
pub mod gemma3n;
pub mod gemma4;
pub mod gemma4_unified;
pub mod glm4v;
pub mod glm_ocr;
pub mod internvl;
pub mod kimi_vl;
pub mod lfm2_vl;
pub mod llama4;
pub mod minicpmo;
pub mod minicpmv4_6;
pub mod mllama;
pub mod molmo;
pub mod molmo2;
pub mod molmo_point;
pub mod moondream3;
pub mod nemotron_h_nano_omni;
pub mod paddleocr_vl;
pub mod phi4_siglip;
pub mod pixtral;
pub mod qwen2_5_vl;
pub mod qwen2_vl;
pub mod qwen3_vl;
pub mod siglip;
pub mod youtu_vl;

use mlxcel_core::{MlxArray, UniquePtr};

/// Output from a vision encoder
pub struct VisionEncoderOutput {
    pub hidden_states: UniquePtr<MlxArray>, // [batch, num_patches, hidden_dim]
}

/// Trait for vision encoders
pub trait VisionEncoder {
    fn forward(&self, pixel_values: &MlxArray) -> VisionEncoderOutput;
}
