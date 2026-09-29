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

//! Nemotron-Parse (`nemotron_parse`) loader.
//!
//! Loads the whole runtime unit through
//! [`models::NemotronParseVlmModel::load`]: the C-RADIO tower, neck, and
//! mBART decoder from the checkpoint safetensors (Hub or MLX-converted key
//! layout, dense or quantized decoder), plus the page processor and the
//! tokenizer from the same directory.

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;

pub(crate) fn load_nemotron_parse_vlm(model_path: &Path) -> Result<LoadedModel> {
    let model = models::NemotronParseVlmModel::load(model_path)?;
    Ok(LoadedModel::NemotronParseVLM(model))
}
