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

//! LoRA (Low-Rank Adaptation) adapter loading support
//!
//! This module provides functionality to load and apply LoRA adapters
//! to base models. LoRA enables efficient fine-tuning by adding low-rank
//! update matrices to frozen base weights.
//!
//! Uses mlxcel-core types (WeightMap) for weight storage.

mod config;
mod loader;
pub mod multi;
pub mod runtime;

pub use config::{AdapterConfig, LoRAParameters};
pub use loader::{
    apply_lora_adapters, apply_lora_adapters_scaled, apply_stage_lora_adapter, fuse_lora_weights,
    fuse_lora_weights_into,
};
pub use multi::LoraAdapterSpec;
pub use runtime::{
    RuntimeLoraAdapter, RuntimeLoraSet, finish_runtime_staging, stage_runtime_adapters,
};
