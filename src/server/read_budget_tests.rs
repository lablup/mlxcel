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

//! Unit tests for the `--timeout` read-budget gate (#1432).

use http_body_util::BodyExt;

use super::{BudgetBody, ReadBudget};

#[test]
fn a_fresh_connection_is_armed() {
    assert!(
        ReadBudget::new().is_armed(),
        "a connection waiting for its first request is inside the read budget"
    );
}

#[tokio::test]
async fn finishing_a_request_body_stands_the_budget_down() {
    let budget = ReadBudget::new();
    let body = BudgetBody::request(axum::body::Body::from("hello"), budget.clone());
    let collected = body.collect().await.expect("body collects");
    assert_eq!(collected.to_bytes().as_ref(), b"hello");
    assert!(
        !budget.is_armed(),
        "once the request is read the server is handling, not reading"
    );
}

#[tokio::test]
async fn finishing_a_response_body_arms_the_budget_again() {
    let budget = ReadBudget::new();
    {
        let request = BudgetBody::request(axum::body::Body::empty(), budget.clone());
        let _ = request.collect().await.expect("collects");
    }
    assert!(!budget.is_armed());

    let response = BudgetBody::response(axum::body::Body::from("out"), budget.clone());
    let _ = response.collect().await.expect("collects");
    assert!(
        budget.is_armed(),
        "a keep-alive connection is waiting for the next request again"
    );
}

#[test]
fn dropping_an_unread_request_body_still_stands_the_budget_down() {
    // Every GET reaches this path: the handler never polls the body, so only
    // the drop can move the gate.
    let budget = ReadBudget::new();
    drop(BudgetBody::request(
        axum::body::Body::empty(),
        budget.clone(),
    ));
    assert!(!budget.is_armed());
}

#[test]
fn dropping_a_response_body_arms_the_budget_again() {
    // A client that disconnects mid-response drops the body without draining
    // it; the connection must not stay outside the read budget.
    let budget = ReadBudget::new();
    super::stand_down_for_tests(&budget);
    drop(BudgetBody::response(
        axum::body::Body::from("partial"),
        budget.clone(),
    ));
    assert!(budget.is_armed());
}
