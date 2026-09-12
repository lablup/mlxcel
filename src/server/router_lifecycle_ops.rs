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
use std::time::{Duration, Instant};

use super::router_lifecycle::LifecycleSnapshot;
use super::router_lifecycle_dto::*;

pub const EVENT_RING_LIMIT: usize = 1024;
pub const EVENT_RETENTION: Duration = Duration::from_secs(600);
pub const TERMINAL_OPERATION_LIMIT: usize = 200;
pub const TERMINAL_OPERATION_RETENTION: Duration = Duration::from_secs(3600);
pub const MAX_ACTIVE_OPERATIONS: usize = 64;

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
    next_sequence: u64,
}

#[derive(Debug)]
pub struct LifecycleCoordinator {
    server_instance_id: String,
    event_tx: tokio::sync::broadcast::Sender<UiEvent>,
    inner: Mutex<CoordinatorInner>,
}

impl LifecycleCoordinator {
    pub fn new() -> Self {
        let (event_tx, _) = tokio::sync::broadcast::channel(EVENT_RING_LIMIT);
        Self {
            server_instance_id: new_server_instance_id(),
            event_tx,
            inner: Mutex::new(CoordinatorInner::default()),
        }
    }

    pub fn server_instance_id(&self) -> &str {
        &self.server_instance_id
    }

    pub fn snapshot_sequence(&self) -> u64 {
        self.inner
            .lock()
            .map(|inner| inner.next_sequence)
            .unwrap_or(0)
    }

    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<UiEvent> {
        self.event_tx.subscribe()
    }

    pub fn subscribe_for_ui(
        &self,
        last_event_id: Option<&str>,
        runtime_model_ids: Vec<String>,
    ) -> (tokio::sync::broadcast::Receiver<UiEvent>, Vec<UiEvent>) {
        let mut inner = self.inner.lock().expect("lifecycle coordinator poisoned");
        prune_events(&mut inner);
        match last_event_id.filter(|event_id| !event_id.is_empty()) {
            None => {
                let sequence = inner.next_sequence;
                let payload = UiEventPayload::Snapshot(SnapshotPayload {
                    snapshot_sequence: sequence,
                    catalog_changed: true,
                    operations_changed: true,
                    runtime_model_ids,
                });
                let event = self.ephemeral_event_locked(sequence, "snapshot", payload);
                let receiver = self.event_tx.subscribe();
                (receiver, vec![event])
            }
            Some(event_id) => {
                let replay = if let Some((_, cursor)) = inner
                    .events
                    .iter()
                    .find(|(_, event)| event.event_id == event_id)
                {
                    let cursor_sequence = cursor.sequence;
                    inner
                        .events
                        .iter()
                        .filter_map(|(_, event)| {
                            (event.sequence > cursor_sequence).then_some(event.clone())
                        })
                        .collect()
                } else {
                    let reason = if event_id.contains(&self.server_instance_id) {
                        "gap"
                    } else {
                        "server_restart"
                    };
                    let payload = ResetPayload {
                        reason: reason.to_string(),
                        resnapshot: true,
                    };
                    let event_type = if reason == "gap" {
                        "gap"
                    } else {
                        "server_restart"
                    };
                    let payload = if reason == "gap" {
                        UiEventPayload::Gap(payload)
                    } else {
                        UiEventPayload::ServerRestart(payload)
                    };
                    let event =
                        self.ephemeral_event_locked(inner.next_sequence, event_type, payload);
                    let receiver = self.event_tx.subscribe();
                    return (receiver, vec![event]);
                };
                let receiver = self.event_tx.subscribe();
                (receiver, replay)
            }
        }
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
            idempotency_scope: "server_instance".to_string(),
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
        self.append_and_broadcast_locked(
            &mut inner,
            UiEventPayload::Operation(OperationPayload {
                operation: operation.clone(),
            }),
        );
        let accepted = OperationAccepted {
            operation_id,
            state: OperationState::Queued,
            idempotent_replay: false,
        };
        Ok(accepted)
    }

    pub fn update_operation(
        &self,
        operation_id: &str,
        state: OperationState,
        result: Option<OperationResult>,
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
        self.append_and_broadcast_locked(
            &mut inner,
            UiEventPayload::Operation(OperationPayload {
                operation: operation.clone(),
            }),
        );
        Some(operation)
    }

    pub fn get_operation(&self, operation_id: &str) -> Option<Operation> {
        let inner = self.inner.lock().ok()?;
        inner.active.get(operation_id).cloned().or_else(|| {
            inner
                .terminal
                .iter()
                .find_map(|(_, op)| (op.operation_id == operation_id).then_some(op.clone()))
        })
    }

    pub fn list_operations(
        &self,
        limit: usize,
        cursor: Option<&str>,
        state: Option<OperationState>,
        kind: Option<OperationKind>,
        target: Option<&str>,
    ) -> OperationsListResponse {
        let limit = limit.clamp(1, 200);
        let offset = cursor
            .and_then(|cursor| cursor.strip_prefix("ops_"))
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(0);
        let inner = self.inner.lock().expect("lifecycle coordinator poisoned");
        let mut operations: Vec<Operation> = inner
            .active
            .values()
            .cloned()
            .chain(inner.terminal.iter().rev().map(|(_, op)| op.clone()))
            .filter(|op| state.is_none_or(|s| op.state == s))
            .filter(|op| kind.is_none_or(|k| op.kind == k))
            .filter(|op| target.is_none_or(|token| op.target.matches_token(token)))
            .collect();
        operations.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then(a.operation_id.cmp(&b.operation_id))
        });
        let total = operations.len();
        let items: Vec<_> = operations.into_iter().skip(offset).take(limit).collect();
        let next = (offset + items.len() < total).then(|| format!("ops_{}", offset + items.len()));
        OperationsListResponse {
            items,
            pagination: Pagination {
                limit,
                next_cursor: next,
                total_known: Some(total),
            },
            server_instance_id: self.server_instance_id.clone(),
            snapshot_sequence: inner.next_sequence,
        }
    }

    pub fn cancel_operation(&self, operation_id: &str) -> Result<OperationAccepted, CancelError> {
        let operation = self
            .get_operation(operation_id)
            .ok_or(CancelError::NotFound)?;
        Err(CancelError::Unsupported {
            operation_id: operation.operation_id,
        })
    }

    pub fn publish_model_revision(
        &self,
        model_id: &str,
        revision: u64,
        lifecycle: LifecycleSnapshot,
    ) -> UiEvent {
        self.publish_payload(UiEventPayload::ModelRevision(ModelRevisionPayload {
            model_id: model_id.to_string(),
            revision,
            lifecycle,
        }))
    }

    pub fn publish_reset(&self, reason: &str, event_kind: ResetEventKind) -> UiEvent {
        let payload = ResetPayload {
            reason: reason.to_string(),
            resnapshot: true,
        };
        match event_kind {
            ResetEventKind::Reset => self.publish_payload(UiEventPayload::Reset(payload)),
            ResetEventKind::Gap => self.publish_payload(UiEventPayload::Gap(payload)),
            ResetEventKind::ServerRestart => {
                self.publish_payload(UiEventPayload::ServerRestart(payload))
            }
        }
    }

    pub fn local_reset_event(&self, reason: &str, event_kind: ResetEventKind) -> UiEvent {
        let inner = self.inner.lock().expect("lifecycle coordinator poisoned");
        let payload = ResetPayload {
            reason: reason.to_string(),
            resnapshot: true,
        };
        match event_kind {
            ResetEventKind::Reset => self.ephemeral_event_locked(
                inner.next_sequence,
                "reset",
                UiEventPayload::Reset(payload),
            ),
            ResetEventKind::Gap => self.ephemeral_event_locked(
                inner.next_sequence,
                "gap",
                UiEventPayload::Gap(payload),
            ),
            ResetEventKind::ServerRestart => self.ephemeral_event_locked(
                inner.next_sequence,
                "server_restart",
                UiEventPayload::ServerRestart(payload),
            ),
        }
    }

    pub fn publish_payload(&self, payload: UiEventPayload) -> UiEvent {
        let mut inner = self.inner.lock().expect("lifecycle coordinator poisoned");
        self.append_and_broadcast_locked(&mut inner, payload)
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

    fn ephemeral_event_locked(
        &self,
        sequence: u64,
        suffix: &str,
        payload: UiEventPayload,
    ) -> UiEvent {
        let mut event = UiEvent::new(&self.server_instance_id, sequence, payload);
        event.event_id = format!("evt_{}_{suffix}_{sequence:08}", self.server_instance_id);
        event
    }

    fn append_event_locked(
        &self,
        inner: &mut CoordinatorInner,
        payload: UiEventPayload,
    ) -> UiEvent {
        inner.next_sequence += 1;
        let event = UiEvent::new(&self.server_instance_id, inner.next_sequence, payload);
        inner.events.push_back((Instant::now(), event.clone()));
        prune_events(inner);
        event
    }

    fn append_and_broadcast_locked(
        &self,
        inner: &mut CoordinatorInner,
        payload: UiEventPayload,
    ) -> UiEvent {
        let event = self.append_event_locked(inner, payload);
        let _ = self.event_tx.send(event.clone());
        event
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetEventKind {
    Reset,
    Gap,
    ServerRestart,
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
