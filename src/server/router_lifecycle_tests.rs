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
        Some(serde_json::json!({ "ok": true })),
        None,
    );
    let replay = coordinator
        .begin_operation(
            OperationKind::ModelLoad,
            OperationTarget::Model {
                model_id: "mdl_test".to_string(),
                requested_revision: Some(7),
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
                },
                Some(&format!("key-{idx}")),
                format!("load:{idx}"),
            )
            .expect("operation accepted");
        coordinator.update_operation(
            &accepted.operation_id,
            OperationState::Succeeded,
            Some(serde_json::json!({ "idx": idx })),
            None,
        );
    }

    let accepted = coordinator
        .begin_operation(
            OperationKind::ModelLoad,
            OperationTarget::Model {
                model_id: "mdl_0".to_string(),
                requested_revision: Some(0),
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
    let first = coordinator.publish_event("catalog", serde_json::json!({ "n": 1 }));
    let second = coordinator.publish_event("catalog", serde_json::json!({ "n": 2 }));

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
