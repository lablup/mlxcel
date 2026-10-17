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
            // Each arm loads into `loaded` through `load_dir_into`, so this
            // frame holds one `LoadedModel`, not one model value per arm (see
            // `load_dir_into`).
            let mut loaded: Option<LoadedModel> = None;
            match model_type {
                $(
                    $(
                        ModelType::$variant => {
                            load_dir_into(&mut loaded, path_str, $dir_loader, $wrap)?
                        }
                    )?
                )*
                _ => {}
            }
            Ok(loaded)
        }
    };
}

/// Load one config-backed family into `out`.
///
/// The dispatcher above has an arm per config-backed family. Unoptimized
/// builds give every temporary of every arm its own stack slot, so when each
/// arm built its model and `LoadedModel` by value the dispatcher's frame was
/// the sum of all of them (about 1.4 MiB on the model-worker thread, whose
/// default stack is 2 MiB), and any growth of a model struct pushed the debug
/// pipeline-parallel CI job into a stack overflow. Moving the per-family work
/// into this out-of-line function leaves one family's temporaries on the stack
/// at a time.
#[inline(never)]
fn load_dir_into<T, U, E, F, W>(
    out: &mut Option<LoadedModel>,
    path_str: &str,
    load: F,
    wrap: W,
) -> Result<()>
where
    F: FnOnce(String) -> std::result::Result<(T, U), E>,
    E: std::fmt::Display,
    W: FnOnce(T) -> LoadedModel,
{
    *out = Some(wrap(super::load_pair_from_dir(path_str, load)?));
    Ok(())
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
