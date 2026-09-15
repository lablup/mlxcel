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

//! Real recording-provider worker threads and lifecycle leases, without MLX.

use super::*;
use std::time::{Duration, Instant};

struct Fixture {
    _root: tempfile::TempDir,
    pool: Arc<RouterPool>,
    entry: Arc<RouterModelEntry>,
    provider: Arc<ModelProvider>,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("fake");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("config.json"), "{}").unwrap();
        let pool = Arc::new(
            RouterPool::new(
                RouterSources {
                    models_dir: Some(root.path().into()),
                    ..Default::default()
                },
                ServerStartupConfig::default(),
                Default::default(),
                PresetCliOverrides::default(),
                2,
                false,
            )
            .unwrap(),
        );
        let entry = pool.get("fake").unwrap();
        let (tx, _rx) = std::sync::mpsc::channel();
        let provider = Arc::new(ModelProvider::recording_for_route_tests(tx));
        let state = AppState::new(
            provider.clone(),
            entry.config.clone(),
            ChatTemplateProcessor::with_template("ok".into()),
            crate::tokenizer::MlxcelTokenizer::stub(),
            path,
            provider.batch_metrics().clone(),
        );
        entry.state.lock().unwrap().app = Some(LoadedApp {
            state,
            router: axum::Router::new(),
        });
        entry.lifecycle.mark_loading();
        entry.lifecycle.mark_ready();
        Self {
            _root: root,
            pool,
            entry,
            provider,
        }
    }

    fn unload(&self) -> super::super::router_lifecycle::OperationAccepted {
        self.pool
            .submit_model_action(
                &self.entry.ui_model_id,
                RouterModelAction::Unload,
                self.entry.lifecycle_revision(),
                "unload-revision-test",
                None,
            )
            .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.provider.shutdown_worker();
        assert!(
            self.provider
                .worker_exit_observer()
                .wait_timeout(Duration::from_secs(2))
        );
    }
}

async fn wait_for(mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !predicate() {
        assert!(
            Instant::now() < deadline,
            "bounded lifecycle observation timed out"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn terminal(fixture: &Fixture, id: &str) -> super::super::router_lifecycle::Operation {
    let coordinator = fixture.pool.lifecycle_coordinator();
    wait_for(|| {
        coordinator.get_operation(id).is_some_and(|op| {
            matches!(
                op.state,
                OperationState::Succeeded | OperationState::Failed | OperationState::Cancelled
            )
        })
    })
    .await;
    coordinator.get_operation(id).unwrap()
}

#[tokio::test]
async fn valid_revision_unload_drains_request_and_observes_worker_exit() {
    let fixture = Fixture::new();
    let caller_revision = fixture.entry.lifecycle_revision();
    let lease = fixture.entry.lifecycle.clone().try_request_lease().unwrap();
    let accepted = fixture.unload();
    wait_for(|| fixture.entry.lifecycle.state() == ModelLifecycleState::Draining).await;
    let drain_revision = fixture.entry.lifecycle_revision();
    assert!(drain_revision > caller_revision);
    assert!(!fixture.provider.worker_exit_observed());
    assert!(fixture.entry.state.lock().unwrap().app.is_some());
    assert_eq!(
        fixture
            .pool
            .lifecycle_coordinator()
            .get_operation(&accepted.operation_id)
            .unwrap()
            .state,
        OperationState::Running
    );
    // Legitimate discovery cannot replace a loaded/draining entry.
    fixture.pool.rescan().unwrap();
    assert!(Arc::ptr_eq(
        &fixture.pool.get("fake").unwrap(),
        &fixture.entry
    ));
    drop(lease);
    let operation = terminal(&fixture, &accepted.operation_id).await;
    assert_eq!(operation.state, OperationState::Succeeded, "{operation:?}");
    assert!(fixture.provider.worker_exit_observed());
    assert!(fixture.entry.state.lock().unwrap().app.is_none());
    assert_eq!(
        fixture.entry.lifecycle.state(),
        ModelLifecycleState::Unloaded
    );
    assert!(fixture.entry.lifecycle_revision() > drain_revision);
    assert!(
        matches!(operation.result, Some(OperationResult::ModelUnload { lifecycle, .. }) if lifecycle.worker_exit_observed)
    );
}

#[tokio::test]
async fn unload_rejects_external_revision_change_during_drain_without_stopping_worker() {
    let fixture = Fixture::new();
    let lease = fixture.entry.lifecycle.clone().try_request_lease().unwrap();
    let accepted = fixture.unload();
    wait_for(|| fixture.entry.lifecycle.state() == ModelLifecycleState::Draining).await;
    let drain_revision = fixture.entry.lifecycle_revision();
    fixture.entry.lifecycle.mark_operation_busy();
    assert!(fixture.entry.lifecycle_revision() > drain_revision);
    drop(lease);
    let operation = terminal(&fixture, &accepted.operation_id).await;
    assert_eq!(operation.state, OperationState::Failed);
    assert_eq!(operation.error.unwrap().code, "stale_revision");
    assert!(!fixture.provider.worker_exit_observed());
    assert!(fixture.entry.state.lock().unwrap().app.is_some());
}

#[tokio::test]
async fn unload_rejects_replaced_entry_during_drain_without_stopping_either_worker() {
    let fixture = Fixture::new();
    let replacement = Fixture::new();
    let lease = fixture.entry.lifecycle.clone().try_request_lease().unwrap();
    let accepted = fixture.unload();
    wait_for(|| fixture.entry.lifecycle.state() == ModelLifecycleState::Draining).await;
    // Test-only adversarial registry replacement; production rescan preserves
    // the draining entry, as the success regression above verifies.
    fixture
        .pool
        .entries
        .write()
        .unwrap()
        .insert("fake".into(), replacement.entry.clone());
    drop(lease);
    let operation = terminal(&fixture, &accepted.operation_id).await;
    assert_eq!(operation.state, OperationState::Failed);
    assert_eq!(operation.error.unwrap().code, "stale_revision");
    assert!(!fixture.provider.worker_exit_observed());
    assert!(!replacement.provider.worker_exit_observed());
}

#[test]
fn owned_drain_token_uses_shared_authority_and_survives_lease_completion() {
    let authority = Arc::new(AtomicU64::new(10));
    let lifecycle = Arc::new(ModelLifecycle::new_with_revision_authority(
        DownloadState::Complete,
        authority.clone(),
    ));
    lifecycle.mark_loading();
    lifecycle.mark_ready();
    let before = lifecycle.revision();
    let generation = lifecycle.generation();
    let lease = lifecycle.clone().try_request_lease().unwrap();
    let other = ModelLifecycle::new_with_revision_authority(DownloadState::Complete, authority);
    assert!(other.revision() > before);
    let (changed, token) = lifecycle.begin_drain_with_revision().unwrap();
    assert!(changed);
    assert!(
        token > before + 1,
        "another model consumed a shared revision"
    );
    drop(lease);
    assert_eq!(lifecycle.revision(), token);
    assert_eq!(lifecycle.generation(), generation);
    assert_eq!(lifecycle.snapshot().active_requests, 0);
    assert_eq!(lifecycle.begin_drain_with_revision(), Some((false, token)));
}

#[tokio::test]
async fn legacy_unload_cleans_failed_worker_without_new_revision_precondition() {
    let fixture = Fixture::new();
    let lease = fixture.entry.lifecycle.clone().try_request_lease().unwrap();
    let pool = fixture.pool.clone();
    let task = tokio::spawn(async move { pool.unload("fake").await });
    wait_for(|| fixture.entry.lifecycle.state() == ModelLifecycleState::Draining).await;
    fixture.provider.shutdown_worker();
    wait_for(|| fixture.provider.worker_exit_observed()).await;
    fixture
        .entry
        .lifecycle
        .mark_failed("recording worker exited during drain", true);
    drop(lease);
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(fixture.entry.state.lock().unwrap().app.is_none());
    assert_eq!(
        fixture.entry.lifecycle.state(),
        ModelLifecycleState::Unloaded
    );
    assert!(fixture.entry.lifecycle_snapshot().worker_exit_observed);
}

#[cfg(feature = "webui")]
#[test]
fn runtime_observation_requires_current_revision_without_lru_or_lease_touch() {
    let fixture = Fixture::new();
    let revision = fixture.entry.lifecycle_revision();
    let last_used = fixture.entry.last_used.load(Ordering::Relaxed);
    let before = fixture.entry.lifecycle_snapshot();
    assert!(fixture.entry.runtime_observation_state(revision).is_some());
    assert!(
        fixture
            .entry
            .runtime_observation_state(revision + 1)
            .is_none()
    );
    assert_eq!(fixture.entry.last_used.load(Ordering::Relaxed), last_used);
    assert_eq!(fixture.entry.lifecycle_snapshot(), before);
    fixture.entry.lifecycle.begin_drain();
    assert!(fixture.entry.runtime_observation_state(revision).is_none());
    assert!(
        fixture
            .entry
            .runtime_observation_state(fixture.entry.lifecycle_revision())
            .is_some()
    );
}
