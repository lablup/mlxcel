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

//! Florence-2 (`florence2`) VLM loader.
//!
//! Loads the whole runtime unit through
//! [`models::Florence2VlmModel::load`]: the DaViT tower plus BART seq2seq
//! text stack from the checkpoint safetensors, and the processor (BART
//! tokenizer plus 768x768 image preprocessor) from the same directory. The
//! model rejects quantized checkpoints with a named error before any weight
//! is loaded; only bf16 / f16 exports such as
//! `mlx-community/Florence-2-base-ft-bf16` are supported.

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;

pub(crate) fn load_florence2_vlm(model_path: &Path) -> Result<LoadedModel> {
    let model = models::Florence2VlmModel::load(model_path)?;
    Ok(LoadedModel::Florence2VLM(model))
}
