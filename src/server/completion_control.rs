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

//! Live-completion control registry for `POST /v1/chat/completions/control`
//! (b10621, #1444).
//!
//! b10621 lets a client change one thing about a completion while it is
//! generating: end the reasoning phase now (`action: "reasoning_end"`),
//! provided the request armed it with the `reasoning_control` field
//! (upstream resolves the completion by `oaicompat_cmpl_id` over the live
//! slots and calls `common_sampler_reasoning_budget_force`; see
//! <https://github.com/ggml-org/llama.cpp/blob/main/tools/server/server-context.cpp>).
//!
//! mlxcel mirrors that addressing here: every OpenAI-compatible completion
//! registers its `chatcmpl-...` / `cmpl-...` id for the duration of the
//! generation, together with the shared force flag the scheduler's
//! [`crate::server::thinking_budget::ThinkingState`] polls each step. The
//! entry is removed by a drop guard when the generation task exits, whatever
//! path it exits through, so the registry cannot leak entries.
//!
//! Ownership: entries record the API key that started the completion. A
//! control request presenting a different key gets the same answer as for an
//! unknown id, so the endpoint is not an existence oracle across keys. With
//! authentication disabled every request shares one owner, which is b10621's
//! behavior.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::stream_session::StreamOwner;

/// Outcome of a `reasoning_end` control request, mapped by the route onto
/// b10621's response bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ControlOutcome {
    /// No live completion with this id for this owner
    /// (`{"success": false, "message": "no active completion for this id"}`).
    NoActiveCompletion,
    /// The completion did not arm `reasoning_control`
    /// (`{"success": false, "message": "reasoning control not enabled for this completion"}`).
    NotArmed,
    /// The force flag was set; the scheduler closes the reasoning block at
    /// the next sampled token (`{"success": true}`).
    Forced,
}

struct ControlEntry {
    /// `Some(flag)` when the request armed `reasoning_control`; the flag is
    /// the same `Arc` the sequence's `ThinkingState` polls.
    armed: Option<Arc<AtomicBool>>,
    owner: StreamOwner,
}

/// Registry of live, controllable completions, keyed by their public
/// completion id. One instance per [`crate::server::AppState`].
#[derive(Default)]
pub(crate) struct CompletionControlRegistry {
    inner: Mutex<HashMap<String, ControlEntry>>,
}

impl CompletionControlRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Register a live completion. Returns a guard that removes the entry on
    /// drop; keep it alive for exactly the duration of the generation.
    pub(crate) fn register(
        self: &Arc<Self>,
        completion_id: String,
        armed: Option<Arc<AtomicBool>>,
        owner: StreamOwner,
    ) -> ControlRegistration {
        if let Ok(mut map) = self.inner.lock() {
            map.insert(completion_id.clone(), ControlEntry { armed, owner });
        }
        ControlRegistration {
            registry: Arc::clone(self),
            completion_id,
        }
    }

    /// Handle `action: "reasoning_end"` for `completion_id` presented by
    /// `owner`. Setting the flag is the entire action; the scheduler picks
    /// it up at the next sampling step, which is the event-order boundary:
    /// every event already committed to the stream is unaffected, and every
    /// later reasoning token is replaced by the close of the thinking block.
    pub(crate) fn reasoning_end(&self, completion_id: &str, owner: &StreamOwner) -> ControlOutcome {
        let Ok(map) = self.inner.lock() else {
            return ControlOutcome::NoActiveCompletion;
        };
        let Some(entry) = map.get(completion_id) else {
            return ControlOutcome::NoActiveCompletion;
        };
        if entry.owner != *owner {
            return ControlOutcome::NoActiveCompletion;
        }
        match &entry.armed {
            Some(flag) => {
                flag.store(true, Ordering::Release);
                ControlOutcome::Forced
            }
            None => ControlOutcome::NotArmed,
        }
    }

    fn unregister(&self, completion_id: &str) {
        if let Ok(mut map) = self.inner.lock() {
            map.remove(completion_id);
        }
    }

    #[cfg(test)]
    pub(crate) fn len_for_tests(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }
}

/// Drop guard that unregisters a completion id. Completion ids are UUIDs, so
/// a guard never removes an entry belonging to a different request.
pub(crate) struct ControlRegistration {
    registry: Arc<CompletionControlRegistry>,
    completion_id: String,
}

impl Drop for ControlRegistration {
    fn drop(&mut self) {
        self.registry.unregister(&self.completion_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn armed_flag() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    #[test]
    fn unknown_id_is_no_active_completion() {
        let reg = Arc::new(CompletionControlRegistry::new());
        assert_eq!(
            reg.reasoning_end("chatcmpl-missing", &None),
            ControlOutcome::NoActiveCompletion
        );
    }

    #[test]
    fn armed_entry_forces_and_sets_the_shared_flag() {
        let reg = Arc::new(CompletionControlRegistry::new());
        let flag = armed_flag();
        let _guard = reg.register("chatcmpl-1".into(), Some(flag.clone()), None);
        assert_eq!(
            reg.reasoning_end("chatcmpl-1", &None),
            ControlOutcome::Forced
        );
        assert!(flag.load(Ordering::Acquire));
    }

    #[test]
    fn unarmed_entry_reports_not_armed() {
        let reg = Arc::new(CompletionControlRegistry::new());
        let _guard = reg.register("chatcmpl-2".into(), None, None);
        assert_eq!(
            reg.reasoning_end("chatcmpl-2", &None),
            ControlOutcome::NotArmed
        );
    }

    #[test]
    fn guard_drop_unregisters() {
        let reg = Arc::new(CompletionControlRegistry::new());
        {
            let _guard = reg.register("chatcmpl-3".into(), Some(armed_flag()), None);
            assert_eq!(reg.len_for_tests(), 1);
        }
        assert_eq!(reg.len_for_tests(), 0);
        assert_eq!(
            reg.reasoning_end("chatcmpl-3", &None),
            ControlOutcome::NoActiveCompletion
        );
    }

    #[test]
    fn cross_owner_control_is_indistinguishable_from_unknown() {
        let reg = Arc::new(CompletionControlRegistry::new());
        let flag = armed_flag();
        let _guard = reg.register(
            "chatcmpl-4".into(),
            Some(flag.clone()),
            Some("key-a".into()),
        );
        assert_eq!(
            reg.reasoning_end("chatcmpl-4", &Some("key-b".into())),
            ControlOutcome::NoActiveCompletion
        );
        assert!(!flag.load(Ordering::Acquire));
        assert_eq!(
            reg.reasoning_end("chatcmpl-4", &Some("key-a".into())),
            ControlOutcome::Forced
        );
    }
}
