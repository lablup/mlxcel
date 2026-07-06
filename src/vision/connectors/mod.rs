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

//! Multi-modal connectors (projectors)
//!
//! Provides the MultiModalConnector trait and connector implementations.

pub mod avg_pool;
pub mod aya_vision;
pub mod deepseek_vl2;
pub mod ernie4_5_vl;
pub mod granite4_vision;
pub mod identity;
pub mod lfm2_vl;
pub mod linear;
pub mod mistral3;
pub mod mlp;
pub mod paddleocr_vl;

use mlxcel_core::{MlxArray, UniquePtr};

/// Trait for multi-modal connectors that project vision features to text space
pub trait MultiModalConnector {
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray>;
}
