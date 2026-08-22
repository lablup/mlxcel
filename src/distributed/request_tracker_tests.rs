use super::*;

#[test]
fn request_id_uniqueness() {
    let id1 = RequestId::new();
    let id2 = RequestId::new();
    assert_ne!(id1, id2);
}

#[test]
fn request_id_display() {
    let id = RequestId::new();
    let s = format!("{id}");
    assert!(!s.is_empty());
    assert_eq!(s.len(), 36); // UUID v4 format
}

#[test]
fn request_id_from_string_valid() {
    let id = RequestId::from_string("test-request-123".to_string()).unwrap();
    assert_eq!(id.as_str(), "test-request-123");
}

#[test]
fn request_id_from_string_rejects_empty() {
    assert!(RequestId::from_string(String::new()).is_none());
}

#[test]
fn request_id_from_string_rejects_too_long() {
    let long = "x".repeat(257);
    assert!(RequestId::from_string(long).is_none());
}

#[test]
fn submit_creates_lifecycle_in_submitted_state() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();
    let state = tracker.get_state(&id).unwrap();
    assert_eq!(state, RequestState::Submitted);
}

#[test]
fn transition_through_full_lifecycle() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();

    assert!(tracker.transition(&id, RequestState::Routing));
    assert_eq!(tracker.get_state(&id).unwrap(), RequestState::Routing);

    assert!(tracker.transition(
        &id,
        RequestState::Processing {
            node_id: "node-0".to_string(),
        }
    ));

    assert!(tracker.transition(
        &id,
        RequestState::Handoff {
            from_node: "node-0".to_string(),
            to_node: "node-1".to_string(),
        }
    ));

    assert!(tracker.transition(
        &id,
        RequestState::Processing {
            node_id: "node-1".to_string(),
        }
    ));

    assert!(tracker.transition(&id, RequestState::Completed));
    assert_eq!(tracker.get_state(&id).unwrap(), RequestState::Completed);
}

#[test]
fn terminal_state_rejects_further_transitions() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();

    // Walk through valid states to reach Completed.
    assert!(tracker.transition(&id, RequestState::Routing));
    assert!(tracker.transition(
        &id,
        RequestState::Processing {
            node_id: "node-0".to_string(),
        }
    ));
    assert!(tracker.transition(&id, RequestState::Completed));
    // Cannot transition after completion.
    assert!(!tracker.transition(&id, RequestState::Routing));
    assert_eq!(tracker.get_state(&id).unwrap(), RequestState::Completed);
}

#[test]
fn failed_state_is_terminal() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();

    assert!(tracker.transition(
        &id,
        RequestState::Failed {
            reason: "timeout".to_string(),
        }
    ));
    assert!(!tracker.transition(&id, RequestState::Routing));
}

#[test]
fn lifecycle_tracks_all_transitions() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();

    assert!(tracker.transition(&id, RequestState::Routing));
    assert!(tracker.transition(
        &id,
        RequestState::Processing {
            node_id: "node-0".to_string(),
        },
    ));
    assert!(tracker.transition(&id, RequestState::Completed));

    let lifecycle = tracker.get_lifecycle(&id).unwrap();
    assert_eq!(lifecycle.transitions.len(), 4); // Submitted + 3 transitions
    assert!(lifecycle.is_terminal());
    assert!(!lifecycle.elapsed().is_zero() || lifecycle.elapsed() == std::time::Duration::ZERO);
}

#[test]
fn active_count_tracks_non_terminal() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id1 = tracker.submit();
    let _id2 = tracker.submit();
    assert_eq!(tracker.active_count(), 2);

    // Walk id1 through valid states to Completed.
    assert!(tracker.transition(&id1, RequestState::Routing));
    assert!(tracker.transition(
        &id1,
        RequestState::Processing {
            node_id: "node-0".to_string(),
        },
    ));
    assert!(tracker.transition(&id1, RequestState::Completed));
    assert_eq!(tracker.active_count(), 1);
    assert_eq!(tracker.tracked_count(), 2);
}

#[test]
fn eviction_removes_oldest_completed() {
    let config = RequestTrackerConfig { max_tracked: 3 };
    let tracker = RequestTracker::new(config);

    let id1 = tracker.submit();
    let id2 = tracker.submit();
    // Walk id1/id2 through valid transitions to Completed so eviction has
    // actual terminal entries to consider.
    for id in [&id1, &id2] {
        assert!(tracker.transition(id, RequestState::Routing));
        assert!(tracker.transition(
            id,
            RequestState::Processing {
                node_id: "node-0".to_string(),
            },
        ));
        assert!(tracker.transition(id, RequestState::Completed));
    }

    // These submissions should trigger eviction of completed requests.
    let _id3 = tracker.submit();
    let _id4 = tracker.submit();

    // At least one old completed request should have been evicted.
    assert!(tracker.tracked_count() <= 4);
}

/// Number of freshly built trackers the determinism test iterates over.
///
/// `RandomState` seeds every `HashMap` instance separately, not merely once
/// per process, so probing a single tracker twice proves nothing. Rebuilding
/// the tracker on every iteration is what exercises a different hash
/// iteration order each time.
const DETERMINISM_ITERATIONS: usize = 32;

/// Issue #1286: requests that share a `created_at` tick must be evicted in a
/// defined order rather than in `HashMap` iteration order.
#[test]
fn eviction_tie_break_is_deterministic() {
    let keys: Vec<String> = ["req-4", "req-1", "req-5", "req-0", "req-3", "req-2"]
        .iter()
        .map(|k| (*k).to_string())
        .collect();
    // Six tracked against max_tracked = 5 means to_remove = 2, and every
    // entry ties on created_at, so the two lexicographically smallest keys go.
    let expected_remaining: Vec<String> = ["req-2", "req-3", "req-4", "req-5"]
        .iter()
        .map(|k| (*k).to_string())
        .collect();

    for _ in 0..DETERMINISM_ITERATIONS {
        let tracker = RequestTracker::new(RequestTrackerConfig { max_tracked: 5 });

        // Submit all six before completing any, so the eviction that runs
        // inside submit_with_id finds no terminal entries to drop.
        for key in &keys {
            tracker.submit_with_id(RequestId::from_string(key.clone()).unwrap());
        }
        for key in &keys {
            let id = RequestId::from_string(key.clone()).unwrap();
            assert!(tracker.transition(&id, RequestState::Routing));
            assert!(tracker.transition(
                &id,
                RequestState::Processing {
                    node_id: "node-0".to_string(),
                },
            ));
            assert!(tracker.transition(&id, RequestState::Completed));
        }

        {
            let mut inner = tracker.inner.write().expect("tracker lock poisoned");
            // `Instant::now()` has nanosecond resolution on Linux and will
            // never tie naturally, so write one shared timestamp into every
            // lifecycle.
            let tied = Instant::now();
            for lifecycle in inner.requests.values_mut() {
                lifecycle.created_at = tied;
            }
            tracker.evict_if_needed(&mut inner);
        }

        let mut remaining: Vec<String> = tracker
            .inner
            .read()
            .expect("tracker lock poisoned")
            .requests
            .keys()
            .cloned()
            .collect();
        remaining.sort();
        assert_eq!(remaining, expected_remaining);
    }
}

#[test]
fn remove_request() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();
    assert_eq!(tracker.tracked_count(), 1);

    let lifecycle = tracker.remove(&id).unwrap();
    assert_eq!(lifecycle.id, id);
    assert_eq!(tracker.tracked_count(), 0);
    assert!(tracker.get_state(&id).is_none());
}

#[test]
fn submit_with_id() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = RequestId::from_string("custom-id-42".to_string()).unwrap();
    tracker.submit_with_id(id.clone());

    let state = tracker.get_state(&id).unwrap();
    assert_eq!(state, RequestState::Submitted);
}

#[test]
fn transition_unknown_request_returns_false() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = RequestId::new();
    assert!(!tracker.transition(&id, RequestState::Routing));
}

#[test]
fn request_state_display() {
    assert_eq!(format!("{}", RequestState::Submitted), "submitted");
    assert_eq!(format!("{}", RequestState::Routing), "routing");
    assert_eq!(
        format!(
            "{}",
            RequestState::Processing {
                node_id: "n0".to_string()
            }
        ),
        "processing(node=n0)"
    );
    assert_eq!(
        format!(
            "{}",
            RequestState::Handoff {
                from_node: "a".to_string(),
                to_node: "b".to_string(),
            }
        ),
        "handoff(a->b)"
    );
    assert_eq!(format!("{}", RequestState::Completed), "completed");
    assert_eq!(
        format!(
            "{}",
            RequestState::Failed {
                reason: "err".to_string()
            }
        ),
        "failed(err)"
    );
}

#[test]
fn invalid_transition_submitted_to_completed_rejected() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();
    // Submitted -> Completed is not a valid transition (must go through Routing/Processing).
    assert!(!tracker.transition(&id, RequestState::Completed));
    assert_eq!(tracker.get_state(&id).unwrap(), RequestState::Submitted);
}

#[test]
fn invalid_transition_submitted_to_processing_rejected() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();
    // Submitted -> Processing is not valid (must go through Routing first).
    assert!(!tracker.transition(
        &id,
        RequestState::Processing {
            node_id: "node-0".to_string(),
        }
    ));
    assert_eq!(tracker.get_state(&id).unwrap(), RequestState::Submitted);
}

#[test]
fn invalid_transition_routing_to_handoff_rejected() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();
    assert!(tracker.transition(&id, RequestState::Routing));
    // Routing -> Handoff is not valid (must go through Processing first).
    assert!(!tracker.transition(
        &id,
        RequestState::Handoff {
            from_node: "a".to_string(),
            to_node: "b".to_string(),
        }
    ));
    assert_eq!(tracker.get_state(&id).unwrap(), RequestState::Routing);
}

#[test]
fn valid_transition_submitted_to_failed() {
    let tracker = RequestTracker::new(RequestTrackerConfig::default());
    let id = tracker.submit();
    // Any state can transition to Failed.
    assert!(tracker.transition(
        &id,
        RequestState::Failed {
            reason: "early failure".to_string(),
        }
    ));
    assert!(matches!(
        tracker.get_state(&id).unwrap(),
        RequestState::Failed { .. }
    ));
}
