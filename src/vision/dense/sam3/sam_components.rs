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

//! SAM prompt encoding and two-way mask decoding used by the video tracker.

mod attention;
mod decoder;
mod layers;
mod prompt;

pub use attention::RopeAttention;
pub(crate) use decoder::{ConvTranspose2d, OutputMlp};
pub use decoder::{MaskDecoderOutput, SamMaskDecoder};
pub use layers::LayerNorm2d;
pub use prompt::{PointPrompts, SamPromptEncoder};

pub(crate) use layers::{Conv2dLayer, TwoWayTransformer, load_array, load_layer_norm};
pub(crate) use prompt::PositionEmbedding;
