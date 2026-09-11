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

//! Qwen2 model implementation using mlxcel-core
//!
//! Qwen2 uses the same architecture as Llama with minor differences:
//! - attention_bias support (handled via config)
//! - tie_word_embeddings support (handled via config)
//!
//! For simplicity, we re-export the Llama3 implementation which handles
//! these configuration options correctly.

// Re-export Llama3 components directly
pub use super::llama3::{
    Attention, Llama3Model as Qwen2Model, MLP, ModelArgs, Quantization, RopeScaling,
    TransformerBlock,
};

// Type aliases for clarity
pub type Model = Qwen2Model;
