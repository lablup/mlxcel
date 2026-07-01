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

//! Llama 3.2 Vision (`mllama`): a Llama-3 text backbone with gated
//! cross-attention layers attending to a tiled ViT image encoder.
//!
//! Faithful port of `references/mlx-vlm/mlx_vlm/models/mllama/`.
//!
//! - [`config`] — nested text + vision configuration.
//! - [`text`] — interleaved self/cross-attention text backbone.
//!
//! The vision tower lives in [`crate::vision::encoders::mllama`], the image
//! processor in [`crate::vision::processors::mllama`], and the top-level
//! runtime that fuses them in [`crate::vision::mllama_vl`].

pub mod config;
pub mod text;

pub use config::{MllamaConfig, MllamaTextConfig, MllamaVisionConfig};
pub use text::MllamaTextModel;
