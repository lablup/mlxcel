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

use std::sync::Arc;
use std::time::Duration;

use super::*;

#[derive(serde::Deserialize)]
struct IdentityVectors {
    identity_vectors: Vec<IdentityVector>,
}

#[derive(serde::Deserialize)]
struct IdentityVector {
    source: String,
    source_rank: u8,
    redacted_source_key: String,
    entry_key: String,
    source_key_hash: String,
    expected_id: String,
}

#[test]
fn stable_model_identity_matches_contract_vectors() {
    let vectors: IdentityVectors = serde_json::from_str(include_str!(
        "../../tests/fixtures/webui/identity-vectors.json"
    ))
    .expect("identity vectors parse");

    assert_eq!(vectors.identity_vectors.len(), 2);
    for vector in vectors.identity_vectors {
        let (id, source_key_hash) = stable_model_identity(
            &vector.source,
            vector.source_rank,
            &vector.redacted_source_key,
            &vector.entry_key,
        );
        assert_eq!(source_key_hash, vector.source_key_hash);
        assert_eq!(id, vector.expected_id);
    }
}

#[tokio::test]
async fn lifecycle_busy_and_capacity_axes_are_distinct() {
    let downloading = ModelLifecycle::new(DownloadState::Downloading);
    assert!(downloading.snapshot().busy);
    assert!(!downloading.reserves_capacity());

    let ready = Arc::new(ModelLifecycle::new(DownloadState::Complete));
    ready.mark_ready();
    assert!(!ready.snapshot().busy);
    assert!(ready.reserves_capacity());

    let lease = ready.clone().try_request_lease().expect("ready lease");
    let snapshot = ready.snapshot();
    assert!(snapshot.busy);
    assert_eq!(snapshot.active_requests, 1);
    assert!(ready.begin_drain());
    assert!(ready.clone().try_request_lease().is_err());
    assert!(!ready.wait_for_zero_active(Duration::from_millis(10)).await);
    drop(lease);
    assert!(ready.wait_for_zero_active(Duration::from_secs(1)).await);

    let failed = ModelLifecycle::new(DownloadState::Complete);
    failed.mark_failed("worker still owns resources", false);
    assert!(failed.snapshot().busy);
    assert!(failed.reserves_capacity());
    failed.mark_worker_exit_observed();
    assert!(!failed.snapshot().busy);
    assert!(!failed.reserves_capacity());
}

#[test]
fn lifecycle_coordinator_replays_idempotent_operations() {
    let coordinator = LifecycleCoordinator::new();
    let target = OperationTarget::Model {
        model_id: "mdl_test".to_string(),
        requested_revision: Some(7),
        eviction_target_id: None,
    };

    let accepted = coordinator
        .begin_operation(
            OperationKind::ModelLoad,
            target.clone(),
            Some("idem-1"),
            "load:mdl_test:7".to_string(),
        )
        .expect("initial operation");
    assert!(!accepted.idempotent_replay);

    let replay = coordinator
        .begin_operation(
            OperationKind::ModelLoad,
            target.clone(),
            Some("idem-1"),
            "load:mdl_test:7".to_string(),
        )
        .expect("idempotent replay");
    assert!(replay.idempotent_replay);
    assert_eq!(replay.operation_id, accepted.operation_id);
    assert_eq!(replay.state, OperationState::Queued);

    assert!(matches!(
        coordinator.begin_operation(
            OperationKind::ModelUnload,
            target,
            Some("idem-1"),
            "unload:mdl_test:7".to_string(),
        ),
        Err(OperationError::Conflict { operation_id: Some(id) }) if id == accepted.operation_id
    ));

    coordinator.update_operation(
        &accepted.operation_id,
        OperationState::Succeeded,
        None,
        None,
    );
    let replay = coordinator
        .begin_operation(
            OperationKind::ModelLoad,
            OperationTarget::Model {
                model_id: "mdl_test".to_string(),
                requested_revision: Some(7),
                eviction_target_id: None,
            },
            Some("idem-1"),
            "load:mdl_test:7".to_string(),
        )
        .expect("terminal replay");
    assert!(replay.idempotent_replay);
    assert_eq!(replay.state, OperationState::Succeeded);
}

#[test]
fn lifecycle_coordinator_prunes_idempotency_with_terminal_history() {
    let coordinator = LifecycleCoordinator::new();

    for idx in 0..250 {
        let accepted = coordinator
            .begin_operation(
                OperationKind::ModelLoad,
                OperationTarget::Model {
                    model_id: format!("mdl_{idx}"),
                    requested_revision: Some(idx),
                    eviction_target_id: None,
                },
                Some(&format!("key-{idx}")),
                format!("load:{idx}"),
            )
            .expect("operation accepted");
        coordinator.update_operation(
            &accepted.operation_id,
            OperationState::Succeeded,
            None,
            None,
        );
    }

    let accepted = coordinator
        .begin_operation(
            OperationKind::ModelLoad,
            OperationTarget::Model {
                model_id: "mdl_0".to_string(),
                requested_revision: Some(0),
                eviction_target_id: None,
            },
            Some("key-0"),
            "load:0".to_string(),
        )
        .expect("pruned idempotency key can be reused");
    assert!(!accepted.idempotent_replay);
}

#[test]
fn lifecycle_coordinator_replay_reports_gap_or_restart() {
    let coordinator = LifecycleCoordinator::new();
    let lifecycle = ModelLifecycle::new(DownloadState::Complete).snapshot();
    let first = coordinator.publish_model_revision("mdl_test", 1, lifecycle.clone());
    let second = coordinator.publish_model_revision("mdl_test", 2, lifecycle);

    let replay = coordinator
        .replay_after(&first.event_id)
        .expect("replay after first");
    assert_eq!(replay, vec![second]);

    let same_instance_missing = format!("evt_{}_99999999", coordinator.server_instance_id());
    assert_eq!(
        coordinator.replay_after(&same_instance_missing),
        Err(ReplayError::Gap)
    );
    assert_eq!(
        coordinator.replay_after("evt_other_server_00000001"),
        Err(ReplayError::ServerRestart)
    );
}

fn fixture_value(path: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|err| panic!("read {path}: {err}"));
    let mut value: serde_json::Value = serde_json::from_str(&text).expect("fixture json");
    if let Some(object) = value.as_object_mut() {
        object.remove("$schemaName");
    }
    value
}

#[test]
fn lifecycle_response_dtos_round_trip_contract_fixtures() {
    for path in [
        "tests/fixtures/webui/examples/operation.running.json",
        "tests/fixtures/webui/examples/operation.succeeded.json",
    ] {
        let value = fixture_value(path);
        let dto: Operation = serde_json::from_value(value.clone()).unwrap_or_else(|err| {
            panic!("Operation fixture {path} must deserialize through the producer DTO: {err}")
        });
        assert_eq!(
            serde_json::to_value(dto).expect("serialize"),
            value,
            "{path}"
        );
    }

    let accepted = fixture_value("tests/fixtures/webui/examples/operation.accepted.json");
    let dto: OperationAccepted = serde_json::from_value(accepted.clone()).unwrap();
    assert_eq!(serde_json::to_value(dto).unwrap(), accepted);

    let list = fixture_value("tests/fixtures/webui/examples/operations.list.json");
    let dto: OperationsListResponse = serde_json::from_value(list.clone()).unwrap();
    assert_eq!(serde_json::to_value(dto).unwrap(), list);

    for path in [
        "tests/fixtures/webui/examples/event.1.json",
        "tests/fixtures/webui/examples/event.2.json",
        "tests/fixtures/webui/examples/event.3.json",
    ] {
        let value = fixture_value(path);
        let dto: UiEvent = serde_json::from_value(value.clone()).unwrap_or_else(|err| {
            panic!("UiEvent fixture {path} must deserialize through the producer DTO: {err}")
        });
        assert_eq!(
            serde_json::to_value(dto).expect("serialize"),
            value,
            "{path}"
        );
    }

    let runtime = fixture_value("tests/fixtures/webui/examples/runtime.snapshot.json");
    let dto: RuntimeSnapshot = serde_json::from_value(runtime.clone()).unwrap();
    assert_eq!(serde_json::to_value(dto).unwrap(), runtime);

    for path in [
        "tests/fixtures/webui/examples/error.stale-revision.json",
        "tests/fixtures/webui/examples/error.unauthorized.json",
    ] {
        let value = fixture_value(path);
        let dto: ErrorEnvelope = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(dto).unwrap(), value, "{path}");
    }
}

#[test]
fn lifecycle_coordinator_lists_gets_and_reports_cancel_unsupported() {
    let coordinator = LifecycleCoordinator::new();
    let accepted = coordinator
        .begin_operation(
            OperationKind::ModelLoad,
            OperationTarget::Model {
                model_id: "mdl_test".to_string(),
                requested_revision: Some(3),
                eviction_target_id: None,
            },
            Some("ops-list-key"),
            "load:mdl_test:3".to_string(),
        )
        .expect("operation accepted");

    let listed = coordinator.list_operations(50, None, None, Some(OperationKind::ModelLoad), None);
    assert_eq!(listed.items.len(), 1);
    assert_eq!(listed.items[0].operation_id, accepted.operation_id);
    assert_eq!(listed.server_instance_id, coordinator.server_instance_id());
    assert_eq!(listed.snapshot_sequence, coordinator.snapshot_sequence());

    let got = coordinator
        .get_operation(&accepted.operation_id)
        .expect("operation retrievable");
    assert_eq!(got.state, OperationState::Queued);
    assert_eq!(
        coordinator.cancel_operation(&accepted.operation_id),
        Err(CancelError::Unsupported {
            operation_id: accepted.operation_id.clone()
        })
    );
    assert_eq!(
        coordinator.cancel_operation("missing"),
        Err(CancelError::NotFound)
    );
}

#[tokio::test]
async fn lifecycle_coordinator_broadcast_order_matches_sequence() {
    let coordinator = Arc::new(LifecycleCoordinator::new());
    let mut rx = coordinator.subscribe();
    let barrier = Arc::new(tokio::sync::Barrier::new(17));
    for idx in 0..16u64 {
        let coordinator = coordinator.clone();
        let barrier = barrier.clone();
        tokio::spawn(async move {
            barrier.wait().await;
            let lifecycle = ModelLifecycle::new(DownloadState::Complete).snapshot();
            coordinator.publish_model_revision(&format!("mdl_order_{idx}"), idx + 1, lifecycle);
        });
    }
    barrier.wait().await;

    let mut seen = Vec::new();
    for _ in 0..16 {
        let event = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event timeout")
            .expect("event channel open");
        seen.push(event.sequence);
    }
    assert_eq!(seen, (1..=16).collect::<Vec<_>>());
}

#[test]
fn lifecycle_event_ring_limit_reports_gap_for_pruned_event() {
    let coordinator = LifecycleCoordinator::new();
    let lifecycle = ModelLifecycle::new(DownloadState::Complete).snapshot();
    let first = coordinator.publish_model_revision("mdl_gap", 1, lifecycle.clone());
    for idx in 0..EVENT_RING_LIMIT {
        coordinator.publish_model_revision("mdl_gap", idx as u64 + 2, lifecycle.clone());
    }
    assert_eq!(
        coordinator.replay_after(&first.event_id),
        Err(ReplayError::Gap)
    );
}
