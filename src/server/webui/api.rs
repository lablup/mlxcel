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

//! Shared `/ui-api/v1` bootstrap and observation DTO helpers.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;

use crate::server::router_lifecycle::{
    FieldError, MeasuredValue, RuntimeSettingValue, RuntimeSettingsReport, RuntimeSnapshot,
    SCHEMA_VERSION,
};
use crate::server::{ServerConfig, ServerStartupConfig};

#[allow(dead_code)]
pub(crate) const UI_API_PREFIX: &str = "/ui-api/v1";
pub(crate) const CATALOG_DEFAULT_PAGE_SIZE: u64 = 50;
pub(crate) const CATALOG_MAX_PAGE_SIZE: u64 = 200;
pub(crate) const WEBUI_JSON_BODY_BYTES: u64 = 2 * 1024 * 1024;
pub(crate) const METADATA_BYTES_PER_ENTRY: u64 = 16 * 1024;
pub(crate) const EVENTS_RETENTION_SECONDS: u64 = 600;
pub(crate) const TERMINAL_OPERATIONS_RETAINED: u64 = 200;
pub(crate) const TERMINAL_OPERATIONS_RETENTION_SECONDS: u64 = 3600;
pub(crate) const MAX_ACTIVE_OPERATIONS: u64 = 64;
pub(crate) const MAX_CONCURRENT_LOADS: u64 = 1;
pub(crate) const MAX_CONCURRENT_DOWNLOADS: u64 = 1;
pub(crate) const NEXT_LOAD_CTX_SIZE_MAX: u64 = 262_144;
pub(crate) const NEXT_LOAD_N_PARALLEL_MAX: u64 = 32;
pub(crate) const CURSOR_BYTES: u64 = 512;
pub(crate) const SETTINGS_FIELDS_MAX: u64 = 64;
pub(crate) const MEASUREMENTS_MAX: u64 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WebUiServerMode {
    ModelFree,
    SingleModel,
    RouterPool,
}

impl WebUiServerMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::ModelFree => "model_free",
            Self::SingleModel => "single_model",
            Self::RouterPool => "router_pool",
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WebUiActionState {
    Enabled,
    Disabled,
    ReadOnly,
}

impl WebUiActionState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
            Self::ReadOnly => "read_only",
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct BootstrapResponse {
    pub schema_version: String,
    pub server: BackendIdentity,
    pub features: Vec<&'static str>,
    pub actions: BTreeMap<&'static str, ActionAvailability>,
    pub roots: Vec<RootSummary>,
    pub limits: LimitSummary,
}

#[derive(Debug, Serialize)]
pub(crate) struct BackendIdentity {
    pub server_instance_id: String,
    pub mode: &'static str,
    pub api_base: String,
    pub auth_required: bool,
    pub build: BuildInfo,
}

#[derive(Debug, Serialize)]
pub(crate) struct BuildInfo {
    pub version: &'static str,
    pub git_commit: Option<&'static str>,
    pub target: String,
    pub features: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ActionAvailability {
    pub state: &'static str,
    pub reason: Option<&'static str>,
    pub instructions: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RootSummary {
    pub kind: &'static str,
    pub display_name: String,
    pub redacted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub writable: Option<bool>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct LimitSummary {
    pub catalog_default_page_size: u64,
    pub catalog_max_page_size: u64,
    pub json_body_bytes: u64,
    pub metadata_bytes_per_entry: u64,
    pub events_ring_size: usize,
    pub events_retention_seconds: u64,
    pub terminal_operations_retained: u64,
    pub terminal_operations_retention_seconds: u64,
    pub max_active_operations: u64,
    pub max_concurrent_loads: u64,
    pub max_concurrent_downloads: u64,
    pub next_load_ctx_size_max: u64,
    pub next_load_n_parallel_max: u64,
    pub cursor_bytes: u64,
    pub settings_fields_max: u64,
    pub measurements_max: u64,
}

pub(crate) fn bootstrap_response(
    startup: &ServerStartupConfig,
    config: &ServerConfig,
    server_instance_id: String,
    mode: WebUiServerMode,
    cache_available: bool,
) -> BootstrapResponse {
    let mut actions = BTreeMap::new();
    let lifecycle_state = if matches!(mode, WebUiServerMode::SingleModel) {
        WebUiActionState::ReadOnly
    } else {
        WebUiActionState::Enabled
    };
    let lifecycle_reason = if matches!(mode, WebUiServerMode::SingleModel) {
        Some("single-model mode does not own the router lifecycle pool")
    } else {
        None
    };
    let lifecycle_instructions = if matches!(mode, WebUiServerMode::SingleModel) {
        Some("Restart without -m/--model to use WebUI model switching")
    } else {
        None
    };
    actions.insert(
        "load",
        action(lifecycle_state, lifecycle_reason, lifecycle_instructions),
    );
    actions.insert(
        "unload",
        action(lifecycle_state, lifecycle_reason, lifecycle_instructions),
    );
    actions.insert(
        "download",
        action(
            WebUiActionState::ReadOnly,
            Some(
                if cache_available && !matches!(mode, WebUiServerMode::SingleModel) {
                    "download adapter is not mounted in this build"
                } else {
                    "no writable managed cache route is available in this mode"
                },
            ),
            Some("Downloads are handled by the WebUI download adapter in issue #1841"),
        ),
    );
    actions.insert(
        "cache_delete",
        action(
            WebUiActionState::ReadOnly,
            Some(
                if cache_available && !matches!(mode, WebUiServerMode::SingleModel) {
                    "cache removal adapter is not mounted in this build"
                } else {
                    "no writable managed cache route is available in this mode"
                },
            ),
            Some("Cache deletion is handled by the WebUI removal adapter in issue #1841"),
        ),
    );
    BootstrapResponse {
        schema_version: SCHEMA_VERSION.to_string(),
        server: BackendIdentity {
            server_instance_id,
            mode: mode.as_str(),
            api_base: config.api_prefix.clone(),
            auth_required: !config.api_keys.is_empty(),
            build: build_info(config, cache_available),
        },
        features: feature_flags(config, cache_available),
        actions,
        roots: root_summaries(startup, cache_available, mode),
        limits: limit_summary(),
    }
}

fn action(
    state: WebUiActionState,
    reason: Option<&'static str>,
    instructions: Option<&'static str>,
) -> ActionAvailability {
    ActionAvailability {
        state: state.as_str(),
        reason,
        instructions,
    }
}

fn feature_flags(config: &ServerConfig, cache_available: bool) -> Vec<&'static str> {
    let mut features = vec!["webui", "catalog", "load", "unload", "chat", "runtime"];
    let _ = cache_available;
    if config.enable_settings_endpoint {
        features.push("settings");
    }
    features
}

fn build_info(config: &ServerConfig, cache_available: bool) -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        git_commit: option_env!("MLXCEL_GIT_COMMIT"),
        target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        features: feature_flags(config, cache_available),
    }
}

fn root_summaries(
    startup: &ServerStartupConfig,
    cache_available: bool,
    mode: WebUiServerMode,
) -> Vec<RootSummary> {
    let mut roots = Vec::new();
    if cache_available {
        roots.push(RootSummary {
            kind: "cache",
            display_name: "mlxcel managed cache".to_string(),
            redacted: true,
            writable: Some(true),
            error: None,
        });
    }
    if startup.router_models_dir.is_some() {
        roots.push(RootSummary {
            kind: "models_dir",
            display_name: "configured models directory".to_string(),
            redacted: true,
            writable: Some(false),
            error: None,
        });
    }
    if startup.models_preset.is_some() {
        roots.push(RootSummary {
            kind: "preset_file",
            display_name: "configured models preset".to_string(),
            redacted: true,
            writable: Some(false),
            error: None,
        });
    }
    if matches!(mode, WebUiServerMode::SingleModel) {
        roots.push(RootSummary {
            kind: "single_model",
            display_name: "startup model".to_string(),
            redacted: true,
            writable: Some(false),
            error: None,
        });
    }
    roots
}

pub(crate) fn limit_summary() -> LimitSummary {
    LimitSummary {
        catalog_default_page_size: CATALOG_DEFAULT_PAGE_SIZE,
        catalog_max_page_size: CATALOG_MAX_PAGE_SIZE,
        json_body_bytes: WEBUI_JSON_BODY_BYTES,
        metadata_bytes_per_entry: METADATA_BYTES_PER_ENTRY,
        events_ring_size: crate::server::router_lifecycle::EVENT_RING_LIMIT,
        events_retention_seconds: EVENTS_RETENTION_SECONDS,
        terminal_operations_retained: TERMINAL_OPERATIONS_RETAINED,
        terminal_operations_retention_seconds: TERMINAL_OPERATIONS_RETENTION_SECONDS,
        max_active_operations: MAX_ACTIVE_OPERATIONS,
        max_concurrent_loads: MAX_CONCURRENT_LOADS,
        max_concurrent_downloads: MAX_CONCURRENT_DOWNLOADS,
        next_load_ctx_size_max: NEXT_LOAD_CTX_SIZE_MAX,
        next_load_n_parallel_max: NEXT_LOAD_N_PARALLEL_MAX,
        cursor_bytes: CURSOR_BYTES,
        settings_fields_max: SETTINGS_FIELDS_MAX,
        measurements_max: MEASUREMENTS_MAX,
    }
}

pub(crate) fn runtime_snapshot(
    server_instance_id: String,
    model_id: String,
    revision: u64,
    snapshot_sequence: u64,
    config: &ServerConfig,
) -> RuntimeSnapshot {
    let mut measurements = BTreeMap::new();
    measurements.insert(
        "gpu_utilization".to_string(),
        MeasuredValue {
            value: None,
            unit: "percent".to_string(),
            scope: "unknown".to_string(),
            measured_at: None,
            reason: Some("not measured by mlxcel".to_string()),
        },
    );
    measurements.insert(
        "ttft".to_string(),
        MeasuredValue {
            value: None,
            unit: "ms".to_string(),
            scope: "model".to_string(),
            measured_at: None,
            reason: Some("no request timing sample is available for this model".to_string()),
        },
    );
    RuntimeSnapshot {
        schema_version: SCHEMA_VERSION.to_string(),
        server_instance_id,
        model_id,
        revision: revision.max(1),
        snapshot_sequence,
        measurements,
        settings: settings_report(config),
    }
}

fn settings_report(config: &ServerConfig) -> RuntimeSettingsReport {
    let mut effective = BTreeMap::new();
    effective.insert(
        "ctx_size".to_string(),
        RuntimeSettingValue::Number(config.context_size as f64),
    );
    effective.insert(
        "n_parallel".to_string(),
        RuntimeSettingValue::Number(config.n_parallel as f64),
    );
    effective.insert(
        "kv_cache_mode".to_string(),
        RuntimeSettingValue::String(config.kv_cache_mode.to_string()),
    );
    effective.insert(
        "temperature".to_string(),
        RuntimeSettingValue::Number(config.default_temperature as f64),
    );
    RuntimeSettingsReport {
        scope: "server_startup".to_string(),
        effective,
        overridden_by_cli: Vec::new(),
        partial_errors: Vec::<FieldError>::new(),
    }
}

#[allow(dead_code)]
pub(crate) fn redacted_path(path: Option<&PathBuf>, fallback: &'static str) -> String {
    path.and_then(|p| {
        p.file_name()
            .map(|name| name.to_string_lossy().into_owned())
    })
    .unwrap_or_else(|| fallback.to_string())
}
