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

//! Tests for the background job API of [`super::ResponsesStore`] (#2897).

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::tests::{assert_indexes_consistent, config, make_response, make_response_with_input};
use super::*;
use crate::server::types::responses_response::{ResponseObject, ResponseStatus};

fn background_config(max_jobs: usize) -> ResponsesStoreConfig {
    ResponsesStoreConfig {
        background_max_jobs: max_jobs,
        ..ResponsesStoreConfig::default()
    }
}

fn snapshot(id: &str) -> ResponseObject {
    let mut response = make_response(id).response;
    response.status = ResponseStatus::Queued;
    response
}

fn terminal(id: &str, text: &str) -> ResponseObject {
    let mut response = make_response(id).response;
    response.status = ResponseStatus::Completed;
    response.output_text = text.to_string();
    response
}

fn admit(store: &ResponsesStore, id: &str, effective_store: bool) -> InFlightToken {
    store
        .admit_background(snapshot(id), vec![], effective_store)
        .expect("admit background job")
}

fn assert_background_accounting(store: &ResponsesStore) {
    let jobs = store.jobs.lock().expect("jobs lock");
    let summed = jobs
        .jobs
        .values()
        .fold(0usize, |bytes, job| bytes.saturating_add(job.size_bytes));
    assert_eq!(jobs.active_bytes, summed);
    let stored = store.inner.read().expect("store state lock").total_bytes;
    drop(jobs);
    assert_eq!(
        store.approximate_total_bytes(),
        stored.saturating_add(summed)
    );
}

#[test]
fn admit_background_at_limit_returns_limit_reached_and_finish_frees_slot() {
    let store = ResponsesStore::new(background_config(2));
    admit(&store, "bg_a", true);
    admit(&store, "bg_b", true);
    assert_eq!(store.active_background_jobs(), 2);

    let refused = store.admit_background(snapshot("bg_c"), vec![], true);
    assert_eq!(
        refused.unwrap_err(),
        BackgroundAdmitError::LimitReached { limit: 2 }
    );

    store.finish_background("bg_a", terminal("bg_a", "done"));
    assert_eq!(store.active_background_jobs(), 1);
    admit(&store, "bg_c", true);
    assert_eq!(store.active_background_jobs(), 2);
}

#[test]
fn admit_background_is_disabled_by_any_zero_limit() {
    for config in [
        background_config(0),
        ResponsesStoreConfig {
            max_entries: 0,
            ..ResponsesStoreConfig::default()
        },
        ResponsesStoreConfig {
            max_bytes: 0,
            ..ResponsesStoreConfig::default()
        },
    ] {
        let store = ResponsesStore::new(config);
        assert_eq!(
            store
                .admit_background(snapshot("bg"), vec![], true)
                .unwrap_err(),
            BackgroundAdmitError::Disabled
        );
        assert_eq!(store.active_background_jobs(), 0);
    }
}

#[test]
fn admit_background_refuses_when_active_bytes_exceed_budget() {
    let input = make_response_with_input("bg_a", &"x".repeat(256)).input_items;
    let probe = ResponsesStore::new(ResponsesStoreConfig::default());
    probe
        .admit_background(snapshot("bg_a"), input.clone(), true)
        .expect("probe admit");
    let one_job = probe.approximate_total_bytes();

    // Room for one job but not two. Stored entries never count against it.
    let store = ResponsesStore::new(config(16, one_job + one_job / 2));
    store
        .admit_background(snapshot("bg_a"), input.clone(), true)
        .expect("first job fits");
    let refused = store.admit_background(snapshot("bg_b"), input, true);
    assert_eq!(refused.unwrap_err(), BackgroundAdmitError::OverBudget);
    assert_background_accounting(&store);
}

#[test]
fn admit_forces_queued_and_mark_in_progress_only_moves_from_queued() {
    let store = ResponsesStore::new(ResponsesStoreConfig::default());
    let mut running = snapshot("bg");
    running.status = ResponseStatus::InProgress;
    store
        .admit_background(running, vec![], true)
        .expect("admit");
    assert_eq!(store.background_status("bg"), Some(ResponseStatus::Queued));

    assert!(store.mark_background_in_progress("bg"));
    assert_eq!(
        store.background_status("bg"),
        Some(ResponseStatus::InProgress)
    );
    assert!(!store.mark_background_in_progress("bg"));
    assert!(!store.mark_background_in_progress("unknown"));
    assert_eq!(
        store.background_snapshot("bg").map(|r| r.status),
        Some(ResponseStatus::InProgress)
    );

    store.finish_background("bg", terminal("bg", "done"));
    assert_eq!(store.background_status("bg"), None);
    assert!(!store.mark_background_in_progress("bg"));
}

#[test]
fn active_jobs_survive_ttl_expiry_and_lru_pressure() {
    let store = ResponsesStore::new(ResponsesStoreConfig {
        max_entries: 2,
        ttl: Duration::from_millis(10),
        ..ResponsesStoreConfig::default()
    });
    admit(&store, "bg_live", true);
    std::thread::sleep(Duration::from_millis(20));
    for id in ["a", "b", "c", "d"] {
        store.insert(id.to_string(), make_response(id));
    }
    assert_eq!(store.len(), 2, "stored entries obey max_entries");
    assert_eq!(store.active_background_jobs(), 1);
    assert_eq!(
        store.background_status("bg_live"),
        Some(ResponseStatus::Queued)
    );
    assert_eq!(
        store.background_snapshot("bg_live").map(|r| r.id),
        Some("bg_live".to_string())
    );
    assert_background_accounting(&store);
}

#[test]
fn concurrent_finish_records_one_object_and_every_caller_gets_it() {
    let store = Arc::new(ResponsesStore::new(ResponsesStoreConfig::default()));
    admit(&store, "bg_race", true);
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let store = store.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                store.finish_background("bg_race", terminal("bg_race", &format!("w{i}")))
            })
        })
        .collect();
    let results: Vec<ResponseObject> = handles
        .into_iter()
        .map(|h| h.join().expect("finisher thread"))
        .collect();

    let recorded = store.get("bg_race").expect("terminal stored").response;
    for result in &results {
        assert_eq!(result.output_text, recorded.output_text);
    }
    assert_eq!(store.active_background_jobs(), 0);
    assert_eq!(store.len(), 1);
    assert_background_accounting(&store);
    assert_indexes_consistent(&store);
}

#[test]
fn store_false_terminal_is_snapshot_only_and_expires() {
    let store = ResponsesStore::new(ResponsesStoreConfig {
        background_ephemeral_ttl: Duration::from_millis(30),
        ..ResponsesStoreConfig::default()
    });
    admit(&store, "bg_eph", false);
    let recorded = store.finish_background("bg_eph", terminal("bg_eph", "secret"));
    assert_eq!(recorded.status, ResponseStatus::Completed);

    assert_eq!(
        store.background_snapshot("bg_eph").map(|r| r.output_text),
        Some("secret".to_string())
    );
    assert!(store.get("bg_eph").is_none(), "hidden from get()");
    assert!(matches!(
        store.request_background_cancel("bg_eph"),
        BackgroundCancel::Terminal(ref r) if r.output_text == "secret"
    ));

    std::thread::sleep(Duration::from_millis(40));
    assert!(store.background_snapshot("bg_eph").is_none());
    assert!(store.get("bg_eph").is_none());
    assert!(matches!(
        store.request_background_cancel("bg_eph"),
        BackgroundCancel::NotFound
    ));
    assert_eq!(store.len(), 0, "lazy sweep removed the expired entry");
    assert_background_accounting(&store);
    assert_indexes_consistent(&store);
}

#[test]
fn store_true_terminal_is_readable_by_get() {
    let store = ResponsesStore::new(ResponsesStoreConfig::default());
    admit(&store, "bg_kept", true);
    store.finish_background("bg_kept", terminal("bg_kept", "kept"));
    assert_eq!(
        store.get("bg_kept").map(|s| s.response.output_text),
        Some("kept".to_string())
    );
}

#[test]
fn oversized_terminal_becomes_failed_object() {
    let probe_store = ResponsesStore::new(ResponsesStoreConfig::default());
    probe_store
        .admit_background(snapshot("bg_big"), vec![], true)
        .expect("probe admit");
    let job_bytes = probe_store.approximate_total_bytes();

    let store = ResponsesStore::new(config(16, job_bytes * 4));
    admit(&store, "bg_big", true);
    let recorded = store.finish_background("bg_big", terminal("bg_big", &"y".repeat(8192)));

    assert_eq!(recorded.status, ResponseStatus::Failed);
    assert!(recorded.output.is_empty());
    assert!(recorded.output_text.is_empty());
    let error = recorded.error.as_ref().expect("error body");
    assert_eq!(error.code, "server_error");
    assert!(error.message.contains(&(job_bytes * 4).to_string()));
    assert_eq!(
        store.background_snapshot("bg_big").map(|r| r.status),
        Some(ResponseStatus::Failed),
        "a poll gets the failed object, not a 404"
    );
    assert_background_accounting(&store);
}

#[test]
fn active_bytes_return_to_zero_after_job_lifecycle() {
    let store = ResponsesStore::new(ResponsesStoreConfig::default());
    store.insert("plain".to_string(), make_response_with_input("plain", "x"));
    let stored_before = store.approximate_total_bytes();

    let input = make_response_with_input("bg", &"z".repeat(64)).input_items;
    store
        .admit_background(snapshot("bg"), input, true)
        .expect("admit");
    assert!(store.approximate_total_bytes() > stored_before);
    assert_background_accounting(&store);

    store.mark_background_in_progress("bg");
    store.finish_background("bg", terminal("bg", "done"));
    assert_eq!(store.jobs.lock().expect("jobs lock").active_bytes, 0);
    assert_background_accounting(&store);
    assert_indexes_consistent(&store);

    store.remove("bg");
    assert_eq!(store.approximate_total_bytes(), stored_before);
}

#[test]
fn request_background_cancel_flips_token_and_keeps_job_active() {
    let store = ResponsesStore::new(ResponsesStoreConfig::default());
    let token = admit(&store, "bg_cancel", true);
    assert!(matches!(
        store.request_background_cancel("bg_cancel"),
        BackgroundCancel::Requested
    ));
    assert!(token.load(Ordering::Relaxed));
    assert!(store.background_cancel_requested("bg_cancel"));
    assert_eq!(store.active_background_jobs(), 1);

    let mut cancelled = terminal("bg_cancel", "");
    cancelled.status = ResponseStatus::Cancelled;
    store.finish_background("bg_cancel", cancelled);
    assert!(matches!(
        store.request_background_cancel("bg_cancel"),
        BackgroundCancel::Terminal(ref r) if r.status == ResponseStatus::Cancelled
    ));
    assert!(matches!(
        store.request_background_cancel("never"),
        BackgroundCancel::NotFound
    ));
}

#[test]
fn finish_for_unknown_id_returns_argument_and_stores_nothing() {
    let store = ResponsesStore::new(ResponsesStoreConfig::default());
    let returned = store.finish_background("ghost", terminal("ghost", "arg"));
    assert_eq!(returned.output_text, "arg");
    assert!(store.background_snapshot("ghost").is_none());
    assert_eq!(store.len(), 0);
}

#[tokio::test]
async fn wait_background_terminal_wakes_on_finish_and_times_out() {
    let store = Arc::new(ResponsesStore::new(ResponsesStoreConfig::default()));
    admit(&store, "bg_wait", false);

    assert!(
        store
            .wait_background_terminal("bg_wait", Duration::from_millis(10))
            .await
            .is_none(),
        "times out while the job is active"
    );

    let finisher = {
        let store = store.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            store.finish_background("bg_wait", terminal("bg_wait", "woke"))
        })
    };
    let waited = store
        .wait_background_terminal("bg_wait", Duration::from_secs(10))
        .await
        .expect("terminal object");
    assert_eq!(waited.output_text, "woke");
    finisher.await.expect("finisher task");

    let again = store
        .wait_background_terminal("bg_wait", Duration::from_millis(1))
        .await
        .expect("already terminal");
    assert_eq!(again.output_text, "woke");
    assert!(
        store
            .wait_background_terminal("unknown", Duration::from_millis(1))
            .await
            .is_none()
    );
}

#[test]
fn set_background_input_items_replaces_what_the_terminal_entry_stores() {
    let store = ResponsesStore::new(background_config(2));
    let original = make_response_with_input("job", &"x".repeat(256)).input_items;
    store
        .admit_background(snapshot("job"), original, true)
        .expect("admit background job");
    let before = store.approximate_total_bytes();

    let replacement = make_response_with_input("job", "short").input_items;
    assert!(store.set_background_input_items("job", replacement));
    assert!(store.approximate_total_bytes() < before);
    assert_background_accounting(&store);
    assert!(!store.set_background_input_items("missing", vec![]));

    store.finish_background("job", terminal("job", "done"));
    let stored = store.get("job").expect("terminal object is stored");
    assert_eq!(stored.input_items.len(), 1);
    assert_eq!(
        response_input_item_size_bytes(&stored.input_items[0]),
        response_input_item_size_bytes(&make_response_with_input("job", "short").input_items[0])
    );
    assert_background_accounting(&store);
}

#[test]
fn cancel_background_finishes_a_queued_job_at_once() {
    let store = ResponsesStore::new(ResponsesStoreConfig::default());
    let token = admit(&store, "bg_queued", true);
    store
        .in_flight_write()
        .expect("registry lock")
        .insert("bg_queued".to_string(), token.clone());

    let BackgroundCancel::Terminal(recorded) = store.cancel_background("bg_queued") else {
        panic!("a queued job is finished by the cancel");
    };
    assert_eq!(recorded.status, ResponseStatus::Cancelled);
    assert!(recorded.completed_at.is_none());
    assert!(token.load(Ordering::Relaxed));
    assert_eq!(store.active_background_jobs(), 0);
    // The cancel was acknowledged for the job's driver, and the job's late
    // terminal write is a no-op.
    assert!(store.finish_in_flight("bg_queued"));
    let late = store.finish_background("bg_queued", terminal("bg_queued", "late"));
    assert_eq!(late.status, ResponseStatus::Cancelled);
    assert_eq!(
        store.get("bg_queued").map(|s| s.response.status),
        Some(ResponseStatus::Cancelled)
    );
    // A repeat returns the recorded object.
    assert!(matches!(
        store.cancel_background("bg_queued"),
        BackgroundCancel::Terminal(ref r) if r.status == ResponseStatus::Cancelled
    ));
    assert!(matches!(
        store.cancel_background("never"),
        BackgroundCancel::NotFound
    ));
    assert_background_accounting(&store);
    assert_indexes_consistent(&store);
}

#[test]
fn cancel_background_leaves_an_in_progress_job_to_its_driver() {
    let store = ResponsesStore::new(ResponsesStoreConfig::default());
    let token = admit(&store, "bg_running", false);
    assert!(store.mark_background_in_progress("bg_running"));
    // The token alone (a refused admission sets it too) is not a cancel.
    token.store(true, Ordering::Relaxed);
    assert!(!store.background_cancel_requested("bg_running"));
    token.store(false, Ordering::Relaxed);

    assert!(matches!(
        store.cancel_background("bg_running"),
        BackgroundCancel::Requested
    ));
    assert!(token.load(Ordering::Relaxed));
    assert!(store.background_cancel_requested("bg_running"));
    assert_eq!(store.active_background_jobs(), 1);
    // No registration: nothing is left behind in the acknowledged set.
    assert!(!store.finish_in_flight("bg_running"));

    // The fallback records `cancelled`; the driver's late write loses.
    let forced = store
        .force_background_cancel("bg_running")
        .expect("the job is active");
    assert_eq!(forced.status, ResponseStatus::Cancelled);
    assert_eq!(store.active_background_jobs(), 0);
    let late = store.finish_background("bg_running", terminal("bg_running", "late"));
    assert_eq!(late.status, ResponseStatus::Cancelled);
    assert!(
        store.get("bg_running").is_none(),
        "store: false stays hidden"
    );

    // A job that finished first keeps its result.
    admit(&store, "bg_done", true);
    store.finish_background("bg_done", terminal("bg_done", "done"));
    let kept = store
        .force_background_cancel("bg_done")
        .expect("the result is retained");
    assert_eq!(kept.status, ResponseStatus::Completed);
    assert!(store.force_background_cancel("never").is_none());
    assert_background_accounting(&store);
    assert_indexes_consistent(&store);
}
