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

//! Registry for standard config-backed text models.
//!
//! These model families follow the common `config.json` + standard text-weight
//! loading path, so keeping them in one module makes new architecture ports
//! easier to compare and extend.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;

use crate::LoadedModel;
use crate::model_metadata;
use crate::models::{self, ModelType};

macro_rules! match_config_backed_dir_loader {
    ($( $variant:ident => { kind: $kind:ident, directory: $directory:ident, weight: $weight:expr, adapter: $adapter:expr $(, config_backed: { dir_loader: $dir_loader:path, args: $args_ty:ty, weight_builder: $weight_builder:path, wrap: $wrap:expr })? }; )*) => {
        pub(crate) fn try_load_config_backed_model_from_dir(
            model_type: ModelType,
            path_str: &str,
        ) -> Result<Option<LoadedModel>> {
            Ok(match model_type {
                $(
                    $(
                        ModelType::$variant => {
                            Some(super::load_pair_from_dir(path_str, $dir_loader).map($wrap)?)
                        }
                    )?
                )*
                _ => None,
            })
        }
    };
}

macro_rules! match_config_backed_weight_loader {
    ($( $variant:ident => { kind: $kind:ident, directory: $directory:ident, weight: $weight:expr, adapter: $adapter:expr $(, config_backed: { dir_loader: $dir_loader:path, args: $args_ty:ty, weight_builder: $weight_builder:path, wrap: $wrap:expr })? }; )*) => {
        pub(crate) fn try_load_config_backed_model_from_weights(
            model_type: ModelType,
            config_str: &str,
            weights: &mut WeightMap,
        ) -> Result<Option<LoadedModel>> {
            Ok(match model_type {
                $(
                    $(
                        ModelType::$variant => {
                            let args: $args_ty = super::parse_model_config(config_str)?;
                            let model = $weight_builder(weights, &args)
                                .map_err(|err| anyhow::anyhow!("{}", err))?;
                            Some(($wrap)(model))
                        }
                    )?
                )*
                _ => None,
            })
        }
    };
}

macro_rules! match_config_backed_support {
    ($( $variant:ident => { kind: $kind:ident, directory: $directory:ident, weight: $weight:expr, adapter: $adapter:expr $(, config_backed: { dir_loader: $dir_loader:path, args: $args_ty:ty, weight_builder: $weight_builder:path, wrap: $wrap:expr })? }; )*) => {
        #[cfg_attr(not(test), allow(dead_code))]
        pub(crate) fn is_config_backed_model_type(model_type: ModelType) -> bool {
            model_metadata::is_config_backed_model_type(model_type)
        }
    };
}

crate::model_metadata::for_each_model_registration!(match_config_backed_dir_loader);
crate::model_metadata::for_each_model_registration!(match_config_backed_weight_loader);
crate::model_metadata::for_each_model_registration!(match_config_backed_support);

#[cfg(test)]
#[path = "config_backed_tests.rs"]
mod tests;
