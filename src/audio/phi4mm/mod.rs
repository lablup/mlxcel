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

//! Token-exact Phi-4 Multimodal audio preprocessing and Cascades Conformer.
//!
//! Reference checkpoint: `microsoft/Phi-4-multimodal-instruct` at immutable
//! revision `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`.

mod config;
mod encoder;
mod feature_extractor;

pub use config::Phi4MMAudioConfig;
pub use encoder::{Phi4MMAudioEncoder, Phi4MMAudioProjection};
pub use feature_extractor::{
    MAX_AUDIO_DURATION_SECONDS, Phi4MMAudioBatch, Phi4MMAudioFeatureExtractor, audio_embed_size,
};
