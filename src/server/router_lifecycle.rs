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

//! WebUI-facing lifecycle, operation and event coordination for router models.
//!
//! This module is intentionally independent from the static WebUI feature flag:
//! compatibility routes and future `/ui-api/v1` adapters must share one model
//! authority so legacy load/unload, autoload and WebUI actions cannot race a
//! second registry.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: &str = "webui.ui-api.v1";

pub fn stable_model_identity(
    source: &str,
    source_rank: u8,
    redacted_source_key: &str,
    entry_key: &str,
) -> (String, String) {
    let source_key_hash = hex_sha256(redacted_source_key.as_bytes());
    let entry_key_json = serde_json::to_string(entry_key).unwrap_or_else(|_| "\"\"".to_string());
    let source_json = serde_json::to_string(source).unwrap_or_else(|_| "\"\"".to_string());
    let namespace_json =
        serde_json::to_string(&source_key_hash).unwrap_or_else(|_| "\"\"".to_string());
    let canonical = format!(
        "{{\"entry_key\":{entry_key_json},\"namespace_hash\":{namespace_json},\"source\":{source_json},\"source_rank\":{source_rank},\"version\":1}}"
    );
    let digest = Sha256::digest(canonical.as_bytes());
    use base64::Engine as _;
    let id = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    (format!("mdl_{id}"), source_key_hash)
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLifecycleState {
    Unloaded,
    Loading,
    Ready,
    Draining,
    Unloading,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    Absent,
    Downloading,
    Complete,
    Incomplete,
    Failed,
}

#[allow(unused_imports)]
pub use super::router_lifecycle_dto::{
    CancelError, ErrorBody, ErrorEnvelope, FieldError, MeasuredValue, ModelEvictionOutcome,
    ModelEvictionReport, Operation, OperationAccepted, OperationError, OperationKind,
    OperationResult, OperationState, OperationTarget, OperationsListResponse, Pagination,
    ProgressBytes, ReplayError, RuntimePayload, RuntimeSettingValue, RuntimeSettingsReport,
    RuntimeSnapshot, SettingsPayload, UiEvent, UiEventPayload,
};
pub use super::router_lifecycle_ops::{
    EVENT_RING_LIMIT, LifecycleCoordinator, MAX_ACTIVE_DOWNLOAD_OPERATIONS,
    MAX_SAFE_EVENT_SEQUENCE, ReplaySubscribeError, ResetEventKind, UiReplayCursor,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleSnapshot {
    pub state: ModelLifecycleState,
    pub download: DownloadState,
    pub busy: bool,
    pub active_requests: usize,
    pub draining_requests: usize,
    pub worker_exit_observed: bool,
    pub last_error: Option<String>,
}

#[derive(Debug)]
struct LifecycleInner {
    state: ModelLifecycleState,
    download: DownloadState,
    active_requests: usize,
    revision: u64,
    generation: u64,
    admission_stopped: bool,
    worker_exit_observed: bool,
    operation_busy: bool,
    last_error: Option<String>,
}

#[derive(Debug)]
pub struct ModelLifecycle {
    inner: Mutex<LifecycleInner>,
    notify: tokio::sync::Notify,
    operation_lock: tokio::sync::Mutex<()>,
    revision_authority: Option<Arc<AtomicU64>>,
}

impl ModelLifecycle {
    pub fn new(download: DownloadState) -> Self {
        Self::new_with_revision(download, 1)
    }

    pub fn new_with_revision(download: DownloadState, revision: u64) -> Self {
        Self::new_inner(download, revision.max(1), None)
    }

    pub fn new_with_revision_authority(
        download: DownloadState,
        revision_authority: Arc<AtomicU64>,
    ) -> Self {
        let revision = revision_authority.fetch_add(1, Ordering::SeqCst).max(1);
        Self::new_inner(download, revision, Some(revision_authority))
    }

    fn new_inner(
        download: DownloadState,
        revision: u64,
        revision_authority: Option<Arc<AtomicU64>>,
    ) -> Self {
        Self {
            inner: Mutex::new(LifecycleInner {
                state: ModelLifecycleState::Unloaded,
                download,
                active_requests: 0,
                revision,
                generation: revision,
                admission_stopped: false,
                worker_exit_observed: true,
                operation_busy: false,
                last_error: None,
            }),
            notify: tokio::sync::Notify::new(),
            operation_lock: tokio::sync::Mutex::new(()),
            revision_authority,
        }
    }

    pub async fn operation_guard(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.operation_lock.lock().await
    }

    pub fn revision(&self) -> u64 {
        self.inner.lock().map(|g| g.revision).unwrap_or(1)
    }

    pub fn generation(&self) -> u64 {
        self.inner.lock().map(|g| g.generation).unwrap_or(1)
    }

    pub fn state(&self) -> ModelLifecycleState {
        self.inner
            .lock()
            .map(|g| g.state)
            .unwrap_or(ModelLifecycleState::Failed)
    }

    pub fn snapshot(&self) -> LifecycleSnapshot {
        let Ok(guard) = self.inner.lock() else {
            return LifecycleSnapshot {
                state: ModelLifecycleState::Failed,
                download: DownloadState::Failed,
                busy: true,
                active_requests: 0,
                draining_requests: 0,
                worker_exit_observed: false,
                last_error: Some("lifecycle state is poisoned".to_string()),
            };
        };
        let busy = matches!(
            guard.state,
            ModelLifecycleState::Loading
                | ModelLifecycleState::Draining
                | ModelLifecycleState::Unloading
        ) || (guard.state == ModelLifecycleState::Ready && guard.active_requests > 0)
            || (guard.state == ModelLifecycleState::Failed && !guard.worker_exit_observed)
            || guard.download == DownloadState::Downloading
            || guard.operation_busy;
        let draining_requests = if guard.admission_stopped {
            guard.active_requests
        } else {
            0
        };
        LifecycleSnapshot {
            state: guard.state,
            download: guard.download,
            busy,
            active_requests: guard.active_requests,
            draining_requests,
            worker_exit_observed: guard.worker_exit_observed,
            last_error: guard.last_error.clone(),
        }
    }

    pub fn reserves_capacity(&self) -> bool {
        self.inner
            .lock()
            .map(|guard| {
                matches!(
                    guard.state,
                    ModelLifecycleState::Loading
                        | ModelLifecycleState::Ready
                        | ModelLifecycleState::Draining
                        | ModelLifecycleState::Unloading
                ) || (guard.state == ModelLifecycleState::Failed && !guard.worker_exit_observed)
                    || guard.operation_busy
            })
            .unwrap_or(true)
    }

    pub fn worker_exit_observed(&self) -> bool {
        self.inner
            .lock()
            .map(|guard| guard.worker_exit_observed)
            .unwrap_or(false)
    }

    pub fn mark_downloading(&self) {
        self.mutate(|g| {
            g.download = DownloadState::Downloading;
            g.revision = self.next_revision_after(g.revision);
        });
    }

    pub fn mark_download_terminal(&self, state: DownloadState) {
        self.mutate(|g| {
            g.download = state;
            g.revision = self.next_revision_after(g.revision);
        });
    }

    pub fn mark_operation_busy(&self) {
        self.mutate(|g| {
            if !g.operation_busy {
                g.operation_busy = true;
                g.revision = self.next_revision_after(g.revision);
            }
        });
    }

    pub fn mark_operation_idle(&self) {
        self.mutate(|g| {
            if g.operation_busy {
                g.operation_busy = false;
                g.revision = self.next_revision_after(g.revision);
            }
        });
    }

    pub fn mark_loading(&self) {
        self.mutate(|g| {
            g.state = ModelLifecycleState::Loading;
            g.admission_stopped = true;
            g.worker_exit_observed = false;
            g.last_error = None;
            g.revision = self.next_revision_after(g.revision);
            g.generation = g.revision;
        });
    }

    pub fn mark_ready(&self) {
        self.mutate(|g| {
            if g.state != ModelLifecycleState::Ready {
                g.revision = self.next_revision_after(g.revision);
            }
            g.state = ModelLifecycleState::Ready;
            g.admission_stopped = false;
            g.worker_exit_observed = false;
            g.last_error = None;
        });
    }

    pub fn mark_failed(&self, error: impl Into<String>, worker_exit_observed: bool) {
        self.mutate(|g| {
            g.state = ModelLifecycleState::Failed;
            g.admission_stopped = true;
            g.worker_exit_observed = worker_exit_observed;
            g.last_error = Some(error.into());
            g.revision = self.next_revision_after(g.revision);
        });
    }

    pub fn mark_loading_blocked(&self, error: impl Into<String>) {
        self.mutate(|g| {
            if g.state == ModelLifecycleState::Loading {
                g.admission_stopped = true;
                g.last_error = Some(error.into());
                g.revision = self.next_revision_after(g.revision);
            }
        });
    }

    pub fn begin_drain(&self) -> bool {
        let mut changed = false;
        self.mutate(|g| {
            if matches!(
                g.state,
                ModelLifecycleState::Loading | ModelLifecycleState::Ready
            ) {
                g.state = ModelLifecycleState::Draining;
                g.admission_stopped = true;
                g.revision = self.next_revision_after(g.revision);
                changed = true;
            }
        });
        changed
    }

    pub fn mark_unloading(&self) {
        self.mutate(|g| {
            g.state = ModelLifecycleState::Unloading;
            g.admission_stopped = true;
            g.revision = self.next_revision_after(g.revision);
        });
    }

    pub fn mark_unloaded(&self) {
        self.mutate(|g| {
            g.state = ModelLifecycleState::Unloaded;
            g.admission_stopped = false;
            g.worker_exit_observed = true;
            g.last_error = None;
            g.revision = self.next_revision_after(g.revision);
        });
    }

    pub fn mark_drain_blocked(&self, error: impl Into<String>) {
        self.mutate(|g| {
            g.state = ModelLifecycleState::Draining;
            g.admission_stopped = true;
            g.last_error = Some(error.into());
            g.revision = self.next_revision_after(g.revision);
        });
    }

    pub fn mark_worker_exit_observed(&self) {
        self.mutate(|g| {
            if !g.worker_exit_observed {
                g.worker_exit_observed = true;
                g.revision = self.next_revision_after(g.revision);
            }
        });
    }

    pub fn try_request_lease(self: &Arc<Self>) -> Result<RequestLease, &'static str> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| "lifecycle state is poisoned")?;
        if guard.admission_stopped || guard.state != ModelLifecycleState::Ready {
            return Err("model is not accepting new requests");
        }
        guard.active_requests += 1;
        Ok(RequestLease {
            lifecycle: self.clone(),
        })
    }

    pub async fn wait_for_zero_active(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self
                .inner
                .lock()
                .map(|g| g.active_requests == 0)
                .unwrap_or(true)
            {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            if tokio::time::timeout(
                deadline.saturating_duration_since(now),
                self.notify.notified(),
            )
            .await
            .is_err()
            {
                return false;
            }
        }
    }

    fn next_revision_after(&self, current: u64) -> u64 {
        let Some(authority) = &self.revision_authority else {
            return current.saturating_add(1);
        };
        loop {
            let revision = authority.fetch_add(1, Ordering::SeqCst);
            if revision > current {
                return revision;
            }
        }
    }

    fn mutate(&self, f: impl FnOnce(&mut LifecycleInner)) {
        if let Ok(mut guard) = self.inner.lock() {
            f(&mut guard);
        }
        self.notify.notify_waiters();
    }
}

pub struct RequestLease {
    lifecycle: Arc<ModelLifecycle>,
}

impl Drop for RequestLease {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.lifecycle.inner.lock()
            && guard.active_requests > 0
        {
            guard.active_requests -= 1;
        }
        self.lifecycle.notify.notify_waiters();
    }
}

#[cfg(test)]
#[path = "router_lifecycle_tests.rs"]
mod router_lifecycle_tests;
