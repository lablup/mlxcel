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

//! Bounded WebUI operation and event store shared by router lifecycle adapters.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::router_lifecycle::{LifecycleSnapshot, SCHEMA_VERSION};

pub const EVENT_RING_LIMIT: usize = 1024;
pub const EVENT_RETENTION: Duration = Duration::from_secs(600);
pub const TERMINAL_OPERATION_LIMIT: usize = 200;
pub const TERMINAL_OPERATION_RETENTION: Duration = Duration::from_secs(3600);
pub const MAX_ACTIVE_OPERATIONS: usize = 64;

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
    fn as_str(self) -> &'static str {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    pub retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    pub operation_id: String,
    pub kind: OperationKind,
    pub state: OperationState,
    pub created_at: String,
    pub updated_at: String,
    pub idempotency_scope: &'static str,
    pub target: OperationTarget,
    pub progress: ProgressBytes,
    pub result: Option<serde_json::Value>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiEvent {
    pub schema_version: &'static str,
    pub server_instance_id: String,
    pub sequence: u64,
    #[serde(rename = "type")]
    pub event_type: String,
    pub payload: serde_json::Value,
    pub event_id: String,
    pub emitted_at: String,
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

#[derive(Debug, Clone)]
struct IdempotencyRecord {
    fingerprint: String,
    operation_id: String,
}

#[derive(Debug, Default)]
struct CoordinatorInner {
    active: BTreeMap<String, Operation>,
    terminal: VecDeque<(Instant, Operation)>,
    idempotency: BTreeMap<String, IdempotencyRecord>,
    events: VecDeque<(Instant, UiEvent)>,
    next_operation: u64,
}

#[derive(Debug)]
pub struct LifecycleCoordinator {
    server_instance_id: String,
    sequence: AtomicU64,
    event_tx: tokio::sync::broadcast::Sender<UiEvent>,
    inner: Mutex<CoordinatorInner>,
}

impl LifecycleCoordinator {
    pub fn new() -> Self {
        let (event_tx, _) = tokio::sync::broadcast::channel(EVENT_RING_LIMIT);
        Self {
            server_instance_id: new_server_instance_id(),
            sequence: AtomicU64::new(0),
            event_tx,
            inner: Mutex::new(CoordinatorInner::default()),
        }
    }

    pub fn server_instance_id(&self) -> &str {
        &self.server_instance_id
    }

    pub fn snapshot_sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }

    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<UiEvent> {
        self.event_tx.subscribe()
    }

    pub fn begin_operation(
        &self,
        kind: OperationKind,
        target: OperationTarget,
        idempotency_key: Option<&str>,
        fingerprint: String,
    ) -> Result<OperationAccepted, OperationError> {
        let now = rfc3339_now();
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| OperationError::TooManyActive)?;
        prune_terminal(&mut inner);
        if let Some(key) = idempotency_key
            && let Some(record) = inner.idempotency.get(key)
        {
            if record.fingerprint == fingerprint {
                let state = inner
                    .active
                    .get(&record.operation_id)
                    .or_else(|| {
                        inner.terminal.iter().find_map(|(_, op)| {
                            (op.operation_id == record.operation_id).then_some(op)
                        })
                    })
                    .map(|op| op.state)
                    .unwrap_or(OperationState::Succeeded);
                return Ok(OperationAccepted {
                    operation_id: record.operation_id.clone(),
                    state,
                    idempotent_replay: true,
                });
            }
            return Err(OperationError::Conflict {
                operation_id: Some(record.operation_id.clone()),
            });
        }
        if inner.active.len() >= MAX_ACTIVE_OPERATIONS {
            return Err(OperationError::TooManyActive);
        }
        inner.next_operation += 1;
        let operation_id = format!("op_{}_{:06}", kind.as_str(), inner.next_operation);
        let operation = Operation {
            operation_id: operation_id.clone(),
            kind,
            state: OperationState::Queued,
            created_at: now.clone(),
            updated_at: now,
            idempotency_scope: "server_instance",
            target,
            progress: ProgressBytes::default(),
            result: None,
            error: None,
            cancellable: false,
            cancel_reason: Some("loader does not support cooperative cancellation".to_string()),
        };
        if let Some(key) = idempotency_key {
            inner.idempotency.insert(
                key.to_string(),
                IdempotencyRecord {
                    fingerprint,
                    operation_id: operation_id.clone(),
                },
            );
        }
        inner.active.insert(operation_id.clone(), operation.clone());
        let accepted = OperationAccepted {
            operation_id,
            state: OperationState::Queued,
            idempotent_replay: false,
        };
        drop(inner);
        self.publish_event("operation", serde_json::json!({ "operation": operation }));
        Ok(accepted)
    }

    pub fn update_operation(
        &self,
        operation_id: &str,
        state: OperationState,
        result: Option<serde_json::Value>,
        error: Option<ErrorBody>,
    ) -> Option<Operation> {
        let mut inner = self.inner.lock().ok()?;
        let mut operation = inner.active.remove(operation_id)?;
        operation.state = state;
        operation.updated_at = rfc3339_now();
        operation.result = result;
        operation.error = error;
        if matches!(
            state,
            OperationState::Succeeded | OperationState::Failed | OperationState::Cancelled
        ) {
            operation.cancellable = false;
            operation.cancel_reason = None;
            inner
                .terminal
                .push_back((Instant::now(), operation.clone()));
            prune_terminal(&mut inner);
        } else {
            inner
                .active
                .insert(operation.operation_id.clone(), operation.clone());
        }
        drop(inner);
        self.publish_event(
            "operation",
            serde_json::json!({ "operation": operation.clone() }),
        );
        Some(operation)
    }

    pub fn publish_model_revision(
        &self,
        model_id: &str,
        revision: u64,
        lifecycle: LifecycleSnapshot,
    ) {
        self.publish_event(
            "model_revision",
            serde_json::json!({
                "model_id": model_id,
                "revision": revision,
                "lifecycle": lifecycle,
            }),
        );
    }

    pub fn publish_event(&self, event_type: &str, payload: serde_json::Value) -> UiEvent {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let event = UiEvent {
            schema_version: SCHEMA_VERSION,
            server_instance_id: self.server_instance_id.clone(),
            sequence,
            event_type: event_type.to_string(),
            payload,
            event_id: format!("evt_{}_{sequence:08}", self.server_instance_id),
            emitted_at: rfc3339_now(),
        };
        if let Ok(mut inner) = self.inner.lock() {
            inner.events.push_back((Instant::now(), event.clone()));
            prune_events(&mut inner);
        }
        let _ = self.event_tx.send(event.clone());
        event
    }

    pub fn replay_after(&self, event_id: &str) -> Result<Vec<UiEvent>, ReplayError> {
        let inner = self.inner.lock().map_err(|_| ReplayError::UnknownEvent)?;
        let Some((_, cursor)) = inner.events.iter().find(|(_, e)| e.event_id == event_id) else {
            if event_id.contains(&self.server_instance_id) {
                return Err(ReplayError::Gap);
            }
            return Err(ReplayError::ServerRestart);
        };
        let cursor_sequence = cursor.sequence;
        Ok(inner
            .events
            .iter()
            .filter_map(|(_, event)| (event.sequence > cursor_sequence).then_some(event.clone()))
            .collect())
    }
}

impl Default for LifecycleCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

fn prune_terminal(inner: &mut CoordinatorInner) {
    while inner.terminal.len() > TERMINAL_OPERATION_LIMIT {
        inner.terminal.pop_front();
    }
    while inner
        .terminal
        .front()
        .is_some_and(|(at, _)| at.elapsed() > TERMINAL_OPERATION_RETENTION)
    {
        inner.terminal.pop_front();
    }
    prune_idempotency(inner);
}

fn prune_idempotency(inner: &mut CoordinatorInner) {
    let retained_operations: BTreeSet<String> = inner
        .active
        .keys()
        .cloned()
        .chain(inner.terminal.iter().map(|(_, op)| op.operation_id.clone()))
        .collect();
    inner
        .idempotency
        .retain(|_, record| retained_operations.contains(&record.operation_id));
}

fn prune_events(inner: &mut CoordinatorInner) {
    while inner.events.len() > EVENT_RING_LIMIT {
        inner.events.pop_front();
    }
    while inner
        .events
        .front()
        .is_some_and(|(at, _)| at.elapsed() > EVENT_RETENTION)
    {
        inner.events.pop_front();
    }
}

fn new_server_instance_id() -> String {
    format!(
        "srv_{}_{}",
        std::process::id(),
        chrono::Utc::now().timestamp_millis()
    )
}

fn rfc3339_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
