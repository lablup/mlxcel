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

//! Non-standard directory loading paths.
//!
//! These families still load directly from a model directory, but they do not
//! fit the standard config-backed text-model registry.

use anyhow::Result;
use std::fmt::Display;
use std::path::Path;

use crate::LoadedModel;
use crate::model_metadata;
use crate::models::{self, ModelType};

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_nonstandard_model_type(model_type: ModelType) -> bool {
    model_metadata::is_nonstandard_model_type(model_type)
}

pub(crate) fn try_load_nonstandard_model_from_dir(
    model_type: ModelType,
    model_path: &Path,
    path_str: &str,
) -> Result<Option<LoadedModel>> {
    Ok(match model_type {
        ModelType::Qwen35 => Some(
            super::load_pair_from_dir(path_str, models::Qwen35Model::load)
                .map(LoadedModel::Qwen35)?,
        ),
        ModelType::Qwen35Moe => Some(
            super::load_pair_from_dir(path_str, models::Qwen35Model::load)
                .map(LoadedModel::Qwen35Moe)?,
        ),
        ModelType::Gemma3n => {
            Some(load_from_dir(path_str, models::Gemma3nModel::load).map(LoadedModel::Gemma3n)?)
        }
        ModelType::DiffusionGemma => Some(
            load_from_path(model_path, |path| models::DiffusionGemmaModel::load(path))
                .map(LoadedModel::DiffusionGemma)?,
        ),
        ModelType::Mamba => Some(
            super::load_pair_from_dir(path_str, |path| models::MambaModel::load(&path))
                .map(LoadedModel::Mamba)?,
        ),
        ModelType::Mamba2 => Some(
            super::load_pair_from_dir(path_str, |path| models::Mamba2Model::load(&path))
                .map(LoadedModel::Mamba2)?,
        ),
        ModelType::Jamba => Some(
            super::load_pair_from_dir(path_str, |path| models::JambaModel::load(&path))
                .map(LoadedModel::Jamba)?,
        ),
        ModelType::FalconH1 => Some(
            super::load_pair_from_dir(path_str, |path| models::FalconH1Model::load(&path))
                .map(LoadedModel::FalconH1)?,
        ),
        ModelType::Lfm2 => Some(
            super::load_pair_from_dir(path_str, |path| models::Lfm2Model::load(&path))
                .map(LoadedModel::Lfm2)?,
        ),
        ModelType::Lfm2Moe => Some(
            super::load_pair_from_dir(path_str, |path| models::Lfm2Model::load(&path))
                .map(LoadedModel::Lfm2Moe)?,
        ),
        ModelType::Plamo2 => Some(
            super::load_pair_from_dir(path_str, |path| models::Plamo2Model::load(&path))
                .map(LoadedModel::Plamo2)?,
        ),
        ModelType::GraniteMoeHybrid => Some(
            super::load_pair_from_dir(path_str, |path| models::GraniteMoeHybridModel::load(&path))
                .map(LoadedModel::GraniteMoeHybrid)?,
        ),
        ModelType::NemotronH => Some(
            super::load_pair_from_dir(path_str, |path| models::NemotronHModel::load(&path))
                .map(LoadedModel::NemotronH)?,
        ),
        ModelType::NemotronNAS => Some(
            super::load_pair_from_dir(path_str, |path| models::NemotronNASModel::load(&path))
                .map(LoadedModel::NemotronNAS)?,
        ),
        ModelType::KimiLinear => Some(
            super::load_pair_from_dir(path_str, models::KimiLinearModel::load)
                .map(LoadedModel::KimiLinear)?,
        ),
        ModelType::LongcatFlash => Some(
            super::load_pair_from_dir(path_str, |path| models::LongcatFlashNgramModel::load(&path))
                .map(LoadedModel::LongcatFlash)?,
        ),
        ModelType::LongcatFlashNgram => Some(
            super::load_pair_from_dir(path_str, |path| models::LongcatFlashNgramModel::load(&path))
                .map(LoadedModel::LongcatFlashNgram)?,
        ),
        ModelType::Rwkv7 => {
            Some(load_from_path(model_path, models::Rwkv7::load).map(LoadedModel::Rwkv7)?)
        }
        ModelType::RecurrentGemma => Some(
            super::load_pair_from_dir(path_str, |path| models::GriffinModel::load(&path))
                .map(LoadedModel::RecurrentGemma)?,
        ),
        _ => None,
    })
}

fn load_from_dir<T, E, F>(path_str: &str, load: F) -> Result<T>
where
    F: FnOnce(String) -> std::result::Result<T, E>,
    E: Display,
{
    load(path_str.to_owned()).map_err(|err| anyhow::anyhow!("{}", err))
}

fn load_from_path<T, E, F>(path: &Path, load: F) -> Result<T>
where
    F: FnOnce(&Path) -> std::result::Result<T, E>,
    E: Display,
{
    load(path).map_err(|err| anyhow::anyhow!("{}", err))
}

#[cfg(test)]
#[path = "nonstandard_tests.rs"]
mod tests;
