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

use super::*;

fn pending(id: &str, response_id: &str) -> PendingApproval {
    PendingApproval {
        id: id.to_string(),
        response_id: response_id.to_string(),
        server_label: "fake".to_string(),
        tool_name: "add".to_string(),
        arguments: r#"{"a":1,"b":2}"#.to_string(),
    }
}

fn answer(id: &str, approve: bool) -> ApprovalAnswer {
    ApprovalAnswer {
        approval_request_id: id.to_string(),
        approve,
    }
}

#[test]
fn an_approval_is_consumed_once() {
    let approvals = PendingApprovals::default();
    approvals.register(vec![pending("mcpr_1", "resp_1")]);
    let answers = [answer("mcpr_1", true)];
    assert_eq!(approvals.check("resp_1", &answers).unwrap().len(), 1);
    // Checking does not consume.
    let consumed = approvals.consume("resp_1", &answers).unwrap();
    assert_eq!(consumed[0].arguments, r#"{"a":1,"b":2}"#);
    assert_eq!(
        approvals.consume("resp_1", &answers),
        Err(ApprovalError::NotPending("mcpr_1".to_string()))
    );
    assert_eq!(approvals.pending_for("resp_1"), 0);
}

#[test]
fn an_approval_only_matches_the_response_that_requested_it() {
    let approvals = PendingApprovals::default();
    approvals.register(vec![pending("mcpr_1", "resp_1")]);
    assert_eq!(
        approvals.consume("resp_other", &[answer("mcpr_1", true)]),
        Err(ApprovalError::NotPending("mcpr_1".to_string()))
    );
    assert_eq!(
        approvals.consume("resp_1", &[answer("mcpr_unknown", true)]),
        Err(ApprovalError::NotPending("mcpr_unknown".to_string()))
    );
    // Neither failed attempt consumed the real request.
    assert_eq!(approvals.pending_for("resp_1"), 1);
}

#[test]
fn every_pending_request_of_the_response_must_be_answered_together() {
    let approvals = PendingApprovals::default();
    approvals.register(vec![
        pending("mcpr_1", "resp_1"),
        pending("mcpr_2", "resp_1"),
    ]);
    assert_eq!(
        approvals.consume("resp_1", &[answer("mcpr_1", true)]),
        Err(ApprovalError::Unanswered {
            response_id: "resp_1".to_string(),
            pending: 2,
            answered: 1,
        })
    );
    assert!(matches!(
        approvals.check("resp_1", &[]),
        Err(ApprovalError::Unanswered { pending: 2, .. })
    ));
    assert_eq!(approvals.pending_for("resp_1"), 2);
    let consumed = approvals
        .consume("resp_1", &[answer("mcpr_2", false), answer("mcpr_1", true)])
        .unwrap();
    assert_eq!(consumed[0].id, "mcpr_2");
    assert_eq!(consumed[1].id, "mcpr_1");
}

#[test]
fn a_repeated_answer_is_refused() {
    let approvals = PendingApprovals::default();
    approvals.register(vec![pending("mcpr_1", "resp_1")]);
    assert_eq!(
        approvals.consume("resp_1", &[answer("mcpr_1", true), answer("mcpr_1", true)]),
        Err(ApprovalError::Duplicate("mcpr_1".to_string()))
    );
}

#[test]
fn a_response_without_pending_requests_checks_clean() {
    let approvals = PendingApprovals::default();
    assert_eq!(approvals.check("resp_none", &[]), Ok(Vec::new()));
}

#[test]
fn the_registry_is_bounded() {
    let approvals = PendingApprovals::default();
    approvals.register(
        (0..MAX_PENDING_APPROVALS + 10)
            .map(|index| pending(&format!("mcpr_{index}"), &format!("resp_{index}")))
            .collect(),
    );
    assert_eq!(approvals.pending_for("resp_0"), 0);
    assert_eq!(
        approvals.pending_for(&format!("resp_{}", MAX_PENDING_APPROVALS + 9)),
        1
    );
}
