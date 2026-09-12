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

//! Strict DTOs for the WebUI lifecycle operation and event contract.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::router_lifecycle::{LifecycleSnapshot, SCHEMA_VERSION};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationState {
    Queued,
    Running,
    Cancelling,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    CatalogRefresh,
    ModelLoad,
    ModelUnload,
    Download,
    ModelRemoval,
    SettingsPatch,
}

impl OperationKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::CatalogRefresh => "catalog_refresh",
            Self::ModelLoad => "model_load",
            Self::ModelUnload => "model_unload",
            Self::Download => "download",
            Self::ModelRemoval => "model_removal",
            Self::SettingsPatch => "settings_patch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgressBytes {
    pub completed_bytes: u64,
    pub total_bytes: Option<u64>,
    pub indeterminate: bool,
}

impl Default for ProgressBytes {
    fn default() -> Self {
        Self {
            completed_bytes: 0,
            total_bytes: None,
            indeterminate: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "target_kind", rename_all = "snake_case")]
pub enum OperationTarget {
    Model {
        model_id: String,
        requested_revision: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        eviction_target_id: Option<String>,
    },
    Catalog {
        scope: String,
        model_id: Option<String>,
    },
    Download {
        repo_id: String,
        revision: Option<String>,
    },
    Settings {
        model_id: String,
        scope: String,
    },
}

impl OperationTarget {
    pub(crate) fn matches_token(&self, token: &str) -> bool {
        match self {
            Self::Model {
                model_id,
                eviction_target_id,
                ..
            } => model_id == token || eviction_target_id.as_ref().is_some_and(|id| id == token),
            Self::Settings { model_id, .. } => model_id == token,
            Self::Catalog { scope, model_id } => {
                scope == token || model_id.as_ref().is_some_and(|id| id == token)
            }
            Self::Download { repo_id, revision } => {
                repo_id == token || revision.as_deref() == Some(token)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelEvictionOutcome {
    NotNeeded,
    Displaced,
    FailedAfterDisplacement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEvictionReport {
    pub requested_target_id: Option<String>,
    pub displaced_model_id: Option<String>,
    pub outcome: ModelEvictionOutcome,
    pub rollbackable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldError {
    pub field: String,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    pub retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_errors: Option<Vec<FieldError>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
    pub request_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RuntimeSettingValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Null,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSettingsReport {
    pub scope: String,
    pub effective: BTreeMap<String, RuntimeSettingValue>,
    pub overridden_by_cli: Vec<String>,
    pub partial_errors: Vec<FieldError>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredValue {
    pub value: Option<f64>,
    pub unit: String,
    pub scope: String,
    pub measured_at: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub schema_version: String,
    pub server_instance_id: String,
    pub model_id: String,
    pub revision: u64,
    pub snapshot_sequence: u64,
    pub measurements: BTreeMap<String, MeasuredValue>,
    pub settings: RuntimeSettingsReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "result_kind", rename_all = "snake_case")]
pub enum OperationResult {
    CatalogRefresh {
        scanned_entries: u64,
        changed_entries: u64,
        snapshot_sequence: u64,
    },
    ModelLoad {
        model_id: String,
        revision: u64,
        lifecycle: LifecycleSnapshot,
        #[serde(skip_serializing_if = "Option::is_none")]
        eviction: Option<ModelEvictionReport>,
    },
    ModelUnload {
        model_id: String,
        revision: u64,
        lifecycle: LifecycleSnapshot,
        #[serde(skip_serializing_if = "Option::is_none")]
        eviction: Option<ModelEvictionReport>,
    },
    ModelRemoval {
        model_id: String,
        revision: u64,
        lifecycle: LifecycleSnapshot,
        #[serde(skip_serializing_if = "Option::is_none")]
        eviction: Option<ModelEvictionReport>,
    },
    Download {
        repo_id: String,
        revision: Option<String>,
        model_id: Option<String>,
        download: super::router_lifecycle::DownloadState,
    },
    SettingsPatch {
        model_id: String,
        settings: RuntimeSettingsReport,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    pub operation_id: String,
    pub kind: OperationKind,
    pub state: OperationState,
    pub created_at: String,
    pub updated_at: String,
    pub idempotency_scope: String,
    pub target: OperationTarget,
    pub progress: ProgressBytes,
    pub result: Option<OperationResult>,
    pub error: Option<ErrorBody>,
    pub cancellable: bool,
    pub cancel_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationAccepted {
    pub operation_id: String,
    pub state: OperationState,
    pub idempotent_replay: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pagination {
    pub limit: usize,
    pub next_cursor: Option<String>,
    pub total_known: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationsListResponse {
    pub items: Vec<Operation>,
    pub pagination: Pagination,
    pub server_instance_id: String,
    pub snapshot_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotPayload {
    pub snapshot_sequence: u64,
    pub catalog_changed: bool,
    pub operations_changed: bool,
    pub runtime_model_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRevisionPayload {
    pub model_id: String,
    pub revision: u64,
    pub lifecycle: LifecycleSnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationPayload {
    pub operation: Operation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadProgressPayload {
    pub operation_id: String,
    pub progress: ProgressBytes,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimePayload {
    pub runtime: RuntimeSnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SettingsPayload {
    pub model_id: String,
    pub settings: RuntimeSettingsReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResetPayload {
    pub reason: String,
    pub resnapshot: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatPayload {
    pub server_time: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UiEventPayload {
    Snapshot(SnapshotPayload),
    ModelRevision(ModelRevisionPayload),
    Operation(OperationPayload),
    DownloadProgress(DownloadProgressPayload),
    Runtime(RuntimePayload),
    Settings(SettingsPayload),
    Reset(ResetPayload),
    Gap(ResetPayload),
    ServerRestart(ResetPayload),
    Heartbeat(HeartbeatPayload),
}

impl UiEventPayload {
    pub(crate) fn event_type(&self) -> &'static str {
        match self {
            Self::Snapshot(_) => "snapshot",
            Self::ModelRevision(_) => "model_revision",
            Self::Operation(_) => "operation",
            Self::DownloadProgress(_) => "download_progress",
            Self::Runtime(_) => "runtime",
            Self::Settings(_) => "settings",
            Self::Reset(_) => "reset",
            Self::Gap(_) => "gap",
            Self::ServerRestart(_) => "server_restart",
            Self::Heartbeat(_) => "heartbeat",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiEvent {
    pub schema_version: String,
    pub server_instance_id: String,
    pub sequence: u64,
    #[serde(rename = "type")]
    pub event_type: String,
    pub payload: UiEventPayload,
    pub event_id: String,
    pub emitted_at: String,
}

impl UiEvent {
    pub(crate) fn new(server_instance_id: &str, sequence: u64, payload: UiEventPayload) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            server_instance_id: server_instance_id.to_string(),
            sequence,
            event_type: payload.event_type().to_string(),
            payload,
            event_id: format!("evt_{server_instance_id}_{sequence:08}"),
            emitted_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationError {
    Conflict { operation_id: Option<String> },
    TooManyActive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    UnknownEvent,
    Gap,
    ServerRestart,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancelError {
    NotFound,
    Unsupported { operation_id: String },
}
