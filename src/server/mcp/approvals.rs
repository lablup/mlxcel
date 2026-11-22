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

//! Tool calls waiting for the caller's approval (issue #1457).
//!
//! A response that ends with `mcp_approval_request` items registers each
//! request here together with the exact arguments the model produced. The
//! follow-up request names the stored response and answers the requests; the
//! call that then runs uses the arguments kept here, never anything the
//! client sent back. An entry is removed when it is answered, so one approval
//! runs one call.
//!
//! The registry lives in memory beside the response store: approvals do not
//! survive a restart, and an approval whose stored response was evicted can
//! no longer be answered.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

/// Entries kept before the oldest pending approval is dropped.
const MAX_PENDING_APPROVALS: usize = 4096;

/// One tool call a caller has to approve before it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingApproval {
    /// The `mcp_approval_request` item id.
    pub id: String,
    /// The stored response that ended with the request.
    pub response_id: String,
    pub server_label: String,
    pub tool_name: String,
    /// The model's arguments as a JSON object string.
    pub arguments: String,
}

/// The caller's answer to one approval request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalAnswer {
    pub approval_request_id: String,
    pub approve: bool,
}

/// Why a set of answers does not resolve a stored response's requests.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ApprovalError {
    #[error(
        "mcp_approval_response '{0}' does not match a pending approval request of the previous response; an approval can be answered once, with previous_response_id naming the response that requested it"
    )]
    NotPending(String),
    #[error("mcp_approval_response '{0}' appears more than once in the request")]
    Duplicate(String),
    #[error(
        "response '{response_id}' has {pending} pending MCP approval request(s) and this request answers {answered}; answer every one of them with an mcp_approval_response item"
    )]
    Unanswered {
        response_id: String,
        pending: usize,
        answered: usize,
    },
}

#[derive(Default)]
struct Inner {
    by_id: HashMap<String, PendingApproval>,
    order: VecDeque<String>,
}

/// In-memory registry of pending approvals.
#[derive(Default)]
pub struct PendingApprovals {
    inner: Mutex<Inner>,
}

impl PendingApprovals {
    /// Record the approval requests a just-stored response ended with.
    pub fn register(&self, approvals: Vec<PendingApproval>) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        for approval in approvals {
            inner.order.push_back(approval.id.clone());
            inner.by_id.insert(approval.id.clone(), approval);
        }
        while inner.by_id.len() > MAX_PENDING_APPROVALS {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            inner.by_id.remove(&oldest);
        }
    }

    /// Number of approval requests `response_id` is still waiting on.
    pub fn pending_for(&self, response_id: &str) -> usize {
        self.inner
            .lock()
            .map(|inner| {
                inner
                    .by_id
                    .values()
                    .filter(|approval| approval.response_id == response_id)
                    .count()
            })
            .unwrap_or(0)
    }

    /// Check that `answers` resolve exactly the pending requests of
    /// `response_id`, without consuming them. Returns the matched requests in
    /// answer order.
    pub fn check(
        &self,
        response_id: &str,
        answers: &[ApprovalAnswer],
    ) -> Result<Vec<PendingApproval>, ApprovalError> {
        let inner = self.inner.lock().map_err(|_| {
            ApprovalError::NotPending(
                answers
                    .first()
                    .map(|answer| answer.approval_request_id.clone())
                    .unwrap_or_default(),
            )
        })?;
        Self::matched(&inner, response_id, answers)
    }

    /// [`Self::check`], then remove the matched requests. A second caller
    /// presenting the same answers gets [`ApprovalError::NotPending`].
    pub fn consume(
        &self,
        response_id: &str,
        answers: &[ApprovalAnswer],
    ) -> Result<Vec<PendingApproval>, ApprovalError> {
        let mut inner = self.inner.lock().map_err(|_| {
            ApprovalError::NotPending(
                answers
                    .first()
                    .map(|answer| answer.approval_request_id.clone())
                    .unwrap_or_default(),
            )
        })?;
        let matched = Self::matched(&inner, response_id, answers)?;
        for approval in &matched {
            inner.by_id.remove(&approval.id);
        }
        let Inner { by_id, order } = &mut *inner;
        order.retain(|id| by_id.contains_key(id));
        Ok(matched)
    }

    fn matched(
        inner: &Inner,
        response_id: &str,
        answers: &[ApprovalAnswer],
    ) -> Result<Vec<PendingApproval>, ApprovalError> {
        let mut seen = HashSet::new();
        let mut matched = Vec::with_capacity(answers.len());
        for answer in answers {
            let id = answer.approval_request_id.as_str();
            if !seen.insert(id) {
                return Err(ApprovalError::Duplicate(id.to_string()));
            }
            match inner.by_id.get(id) {
                Some(approval) if approval.response_id == response_id => {
                    matched.push(approval.clone());
                }
                _ => return Err(ApprovalError::NotPending(id.to_string())),
            }
        }
        let pending = inner
            .by_id
            .values()
            .filter(|approval| approval.response_id == response_id)
            .count();
        if pending != matched.len() {
            return Err(ApprovalError::Unanswered {
                response_id: response_id.to_string(),
                pending,
                answered: matched.len(),
            });
        }
        Ok(matched)
    }
}

#[cfg(test)]
#[path = "approvals_tests.rs"]
mod tests;
