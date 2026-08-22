// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Response-shape tests for `GET /v1/internal/mtp-policy` (issue #1257).
//!
//! These drive [`build_mtp_policy_response`] rather than the axum handler, so
//! no `AppState` (and therefore no loaded model) is needed. The mapping is the
//! whole contract: the handler itself is a one-line projection.

use super::*;
use crate::server::batch::{MtpPolicyStatus, MtpPolicyUnavailableReason, MtpPolicyVerdict};

/// Build a snapshot for a settled pairing, the shape the worker publishes once
/// a verdict is in effect.
fn settled(run: bool, acceptance_rate: f64, samples: usize) -> MtpPolicySnapshot {
    MtpPolicySnapshot {
        status: MtpPolicyStatus::Settled,
        reason: None,
        decline_detail: None,
        target: Some("Gemma4-12B".to_string()),
        drafter: Some("Gemma4-12B-MTP".to_string()),
        hardware: Some("M5-16c".to_string()),
        block_size: Some(4),
        mtp_enabled: Some(run),
        verdict: Some(MtpPolicyVerdict::from_run(run)),
        acceptance_rate: Some(acceptance_rate),
        samples,
        samples_required: 4,
    }
}

#[test]
fn settled_enable_reports_the_verdict_and_the_full_key() {
    let body = build_mtp_policy_response(Some(settled(true, 0.62, 4)));

    assert_eq!(body.schema_version, MTP_POLICY_SCHEMA_VERSION);
    assert_eq!(body.state, "settled");
    assert_eq!(body.verdict.as_deref(), Some("enable"));
    assert_eq!(body.mtp_enabled, Some(true));
    assert_eq!(body.reason, None);
    // The four key fields a consumer needs to confirm the verdict describes
    // the pairing it is showing.
    assert_eq!(body.target.as_deref(), Some("Gemma4-12B"));
    assert_eq!(body.drafter.as_deref(), Some("Gemma4-12B-MTP"));
    assert_eq!(body.hardware.as_deref(), Some("M5-16c"));
    assert_eq!(body.block_size, Some(4));
    assert_eq!(body.acceptance_rate, Some(0.62));
    assert_eq!(body.samples, 4);
    assert_eq!(body.samples_required, 4);
    // Nothing is pending once settled, so no countdown is offered.
    assert_eq!(body.samples_remaining, None);
}

#[test]
fn settled_decline_reports_decline_and_a_disabled_gate() {
    let body = build_mtp_policy_response(Some(settled(false, 0.31, 4)));

    assert_eq!(body.state, "settled");
    assert_eq!(body.verdict.as_deref(), Some("decline"));
    assert_eq!(body.mtp_enabled, Some(false));
}

#[test]
fn profiling_reports_no_verdict_and_the_remaining_sample_count() {
    let snapshot = MtpPolicySnapshot {
        status: MtpPolicyStatus::Profiling,
        reason: None,
        decline_detail: None,
        target: Some("Gemma4-12B".to_string()),
        drafter: Some("Gemma4-12B-MTP".to_string()),
        hardware: Some("M5-16c".to_string()),
        block_size: Some(4),
        // Profiling forces MTP on to collect the sample, so the live gate is
        // on even though no verdict exists yet.
        mtp_enabled: Some(true),
        verdict: None,
        acceptance_rate: Some(0.55),
        samples: 2,
        samples_required: 4,
    };

    let body = build_mtp_policy_response(Some(snapshot));

    assert_eq!(body.state, "profiling");
    assert_eq!(body.verdict, None);
    assert_eq!(body.mtp_enabled, Some(true));
    assert_eq!(body.samples, 2);
    assert_eq!(body.samples_required, 4);
    assert_eq!(body.samples_remaining, Some(2));
}

#[test]
fn forced_is_reported_separately_from_settled() {
    let snapshot = MtpPolicySnapshot {
        status: MtpPolicyStatus::Forced,
        reason: None,
        decline_detail: None,
        target: Some("Gemma4-12B".to_string()),
        drafter: Some("Gemma4-12B-MTP".to_string()),
        hardware: Some("M5-16c".to_string()),
        block_size: Some(4),
        mtp_enabled: Some(true),
        verdict: None,
        acceptance_rate: None,
        samples: 0,
        samples_required: 4,
    };

    let body = build_mtp_policy_response(Some(snapshot));

    // An operator pin is not a measured verdict. A consumer that saw
    // "settled" here would tell a user this machine measured MTP as
    // worthwhile when nothing was measured at all.
    assert_eq!(body.state, "forced");
    assert_eq!(body.verdict, None);
    assert_eq!(body.acceptance_rate, None);
    assert_eq!(body.samples, 0);
    assert_eq!(body.samples_remaining, None);
    // The live gate is still reported, because that part is knowable.
    assert_eq!(body.mtp_enabled, Some(true));
}

#[test]
fn unavailable_reasons_are_distinguishable() {
    let cases = [
        (MtpPolicyUnavailableReason::NoMtpDispatch, "no_mtp_dispatch"),
        (
            MtpPolicyUnavailableReason::AdaptiveDisabled,
            "adaptive_disabled",
        ),
        (
            MtpPolicyUnavailableReason::WorkerNotReady,
            "worker_not_ready",
        ),
    ];

    for (reason, label) in cases {
        let body =
            build_mtp_policy_response(Some(MtpPolicySnapshot::unavailable(reason, Some(false))));
        assert_eq!(body.state, "unavailable");
        assert_eq!(body.reason.as_deref(), Some(label));
        assert_eq!(body.verdict, None);
        assert_eq!(body.target, None);
        assert_eq!(body.block_size, None);
        assert_eq!(body.samples, 0);
        assert_eq!(body.samples_remaining, None);
    }
}

#[test]
fn nothing_published_yet_answers_worker_not_ready_rather_than_erroring() {
    let body = build_mtp_policy_response(None);

    assert_eq!(body.state, "unavailable");
    assert_eq!(body.reason.as_deref(), Some("worker_not_ready"));
    assert_eq!(body.mtp_enabled, None);
    assert_eq!(body.samples_required, 4);
}

/// A profiling state must never look like an absent one. This is the exact
/// confusion the interim cache read could not resolve: no file on disk meant
/// both "still profiling" and "the cache root resolved somewhere else".
#[test]
fn profiling_is_distinguishable_from_unavailable() {
    let profiling = build_mtp_policy_response(Some(MtpPolicySnapshot {
        status: MtpPolicyStatus::Profiling,
        reason: None,
        decline_detail: None,
        target: Some("t".to_string()),
        drafter: Some("d".to_string()),
        hardware: Some("hw".to_string()),
        block_size: Some(8),
        mtp_enabled: Some(true),
        verdict: None,
        acceptance_rate: None,
        samples: 0,
        samples_required: 4,
    }));
    let unavailable = build_mtp_policy_response(None);

    assert_ne!(profiling.state, unavailable.state);
    assert_eq!(profiling.reason, None);
    assert!(unavailable.reason.is_some());
    // Both carry no verdict, so `verdict` alone cannot tell them apart; the
    // state label is what a consumer must branch on.
    assert_eq!(profiling.verdict, None);
    assert_eq!(unavailable.verdict, None);
}

/// The body must round-trip through JSON so a consumer in another language
/// sees stable field names, and unknown fields must not be required.
#[test]
fn response_round_trips_through_json_with_the_documented_field_names() {
    let body = build_mtp_policy_response(Some(settled(true, 0.62, 4)));
    let json = serde_json::to_value(&body).expect("serializes");

    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["state"], "settled");
    assert_eq!(json["verdict"], "enable");
    assert_eq!(json["mtp_enabled"], true);
    assert_eq!(json["target"], "Gemma4-12B");
    assert_eq!(json["drafter"], "Gemma4-12B-MTP");
    assert_eq!(json["hardware"], "M5-16c");
    assert_eq!(json["block_size"], 4);
    assert_eq!(json["acceptance_rate"], 0.62);
    assert_eq!(json["samples"], 4);
    assert_eq!(json["samples_required"], 4);
    assert!(json["samples_remaining"].is_null());
    assert!(json["reason"].is_null());

    let parsed: MtpPolicyResponse = serde_json::from_value(json).expect("round-trips");
    assert_eq!(parsed, body);
}

/// The exactness veto is a first-class state (issue #1298): not `profiling`
/// (no verdict is pending, none can arrive) and not `unavailable` (the
/// dispatch exists; the measured gate refused it). The probe's reason rides
/// along so the endpoint and the boot-time WARN tell one story.
#[test]
fn exactness_declined_reports_the_veto_and_its_reason() {
    let detail = "verify block position 0 differs from the single-token chain \
                  in 245722 of 524288 logit bytes. Disabling qmv_wide did not \
                  make it exact either.";
    let snapshot = MtpPolicySnapshot::exactness_declined(
        "Gemma4-31B".to_string(),
        "Gemma4-31B-assistant".to_string(),
        4,
        Some(detail.to_string()),
    );
    let body = build_mtp_policy_response(Some(snapshot));

    assert_eq!(body.state, "exactness_declined");
    assert_eq!(body.mtp_enabled, Some(false));
    assert_eq!(body.verdict, None, "nothing was measured by the policy");
    assert_eq!(body.reason, None, "reason is the unavailable-state field");
    assert_eq!(body.decline_detail.as_deref(), Some(detail));
    assert_eq!(body.samples, 0);
    assert_eq!(
        body.samples_remaining, None,
        "a vetoed pairing must not render a countdown"
    );

    let json = serde_json::to_value(&body).expect("serializes");
    assert_eq!(json["state"], "exactness_declined");
    assert_eq!(json["decline_detail"], detail);
    assert_eq!(json["block_size"], 4);
}
