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

//! Canonical LoadProfile request adapter. Never mutates a running worker or
//! the pool template: CLI > this operation > preset > defaults. Omitted values
//! inherit. The resulting startup copy follows the existing config builder.

use super::api::{NEXT_LOAD_CTX_SIZE_MAX, NEXT_LOAD_N_PARALLEL_MAX};
use crate::server::{ServerStartupConfig, router_presets::PresetCliOverrides};
use mlxcel_core::cache::KVCacheMode;

#[derive(Debug, Default, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UiLoadProfile {
    pub ctx_size: Option<u64>,
    pub n_parallel: Option<u64>,
    pub kv_cache_mode: Option<String>,
}

pub(crate) struct ProfileError {
    pub field: &'static str,
    pub code: &'static str,
    pub message: String,
}

impl ProfileError {
    pub fn unsupported(field: &'static str, message: &str) -> Self {
        Self {
            field,
            code: "unsupported",
            message: message.to_string(),
        }
    }
}

impl UiLoadProfile {
    pub fn has_overrides(&self) -> bool {
        self.ctx_size.is_some() || self.n_parallel.is_some() || self.kv_cache_mode.is_some()
    }

    pub fn validate(&self) -> Result<(), ProfileError> {
        for (value, maximum, field) in [
            (
                self.ctx_size,
                NEXT_LOAD_CTX_SIZE_MAX,
                "load_profile.ctx_size",
            ),
            (
                self.n_parallel,
                NEXT_LOAD_N_PARALLEL_MAX,
                "load_profile.n_parallel",
            ),
        ] {
            if value.is_some_and(|value| !(1..=maximum).contains(&value)) {
                return Err(ProfileError {
                    field,
                    code: "out_of_range",
                    message: format!("value must be between 1 and {maximum}"),
                });
            }
        }
        if let Some(mode) = self.kv_cache_mode.as_deref()
            && (mode.to_ascii_lowercase() != mode || mode.parse::<KVCacheMode>().is_err())
        {
            return Err(ProfileError {
                field: "load_profile.kv_cache_mode",
                code: "invalid_enum",
                message: "expected an existing KVCacheMode name from the WebUI contract"
                    .to_string(),
            });
        }
        Ok(())
    }

    pub fn apply(
        &self,
        startup: &mut ServerStartupConfig,
        cli: &PresetCliOverrides,
    ) -> Result<(), ProfileError> {
        self.validate()?;
        // Worker geometry and distributed modes remain read-only in v1.
        if startup.tp_size > 1
            || startup.pp_layers.is_some()
            || startup.pp_auto.is_some()
            || startup.pp_peer
            || startup.enable_elastic_pp
            || startup.distributed_config.is_some()
            || startup.node_role.is_some()
            || !startup.peers.is_empty()
            || !startup.prefill_peers.is_empty()
            || !startup.decode_peers.is_empty()
            || startup.serving_bind.is_some()
            || std::env::var("MLXCEL_BACKEND")
                .ok()
                .is_some_and(|backend| backend.eq_ignore_ascii_case("xla"))
        {
            return Err(ProfileError::unsupported(
                "load_profile",
                "distributed and pipeline worker profiles require a server restart",
            ));
        }
        if let Some(value) = self.ctx_size.filter(|_| !cli.ctx_size) {
            startup.ctx_size = value as usize;
        }
        if let Some(value) = self.n_parallel.filter(|_| !cli.n_parallel) {
            startup.n_parallel = value as usize;
        }
        if let Some(mode) = self.kv_cache_mode.as_deref().filter(|_| !cli.kv_cache_mode) {
            if startup.batch_kv_quant.is_enabled() {
                return Err(ProfileError::unsupported(
                    "load_profile.kv_cache_mode",
                    "startup batch KV quantization takes precedence; restart to change its mode",
                ));
            }
            let requested = mode.parse::<KVCacheMode>().map_err(|_| {
                ProfileError::unsupported("load_profile.kv_cache_mode", "invalid KV cache mode")
            })?;
            let (effective, _) = crate::cli::turbo_args::resolve_effective_kv_cache_mode(
                requested,
                &startup.model_path,
            );
            if effective != requested {
                return Err(ProfileError::unsupported(
                    "load_profile.kv_cache_mode",
                    "this model does not support the requested KV cache mode",
                ));
            }
            startup.kv_cache_mode = effective;
        }
        // Also check inherited CLI KV policy against this selected model. A
        // context-only profile must not bypass the CLI's model-family safety
        // resolver just because it omitted the KV field.
        let (effective, _) = crate::cli::turbo_args::resolve_effective_kv_cache_mode(
            startup.kv_cache_mode,
            &startup.model_path,
        );
        if effective != startup.kv_cache_mode {
            return Err(ProfileError::unsupported(
                "load_profile.kv_cache_mode",
                "this model does not support the resolved startup KV cache mode",
            ));
        }
        crate::server::startup::validate_parallel_context_startup(startup)
            .map_err(|_| ProfileError::unsupported("load_profile.ctx_size", "resolved context is below the existing minimum per-slot context; increase context or reduce parallelism"))?;
        crate::server::startup::validate_muse_glimmer_unsupported_startup(startup).map_err(
            |_| {
                ProfileError::unsupported(
                    "load_profile",
                    "this model does not support the resolved startup configuration",
                )
            },
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "load_profile_tests.rs"]
mod tests;
