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

//! End-of-generation reconciliation for the incremental streamer (#2396).
//!
//! Used by: tool_calls::incremental

use super::IncrementalToolCallStreamer;
use crate::server::tool_calls::types::ParsedToolCall;

/// How the end-of-generation emission handles one parsed call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalCallPlan {
    /// Position in the parsed call list.
    pub parsed_index: usize,
    /// The `index` / output slot the call is emitted under.
    pub index: usize,
    /// Position in [`IncrementalToolCallStreamer::calls`] when this call was
    /// streamed and agrees with the parse by index and name.
    pub streamed: Option<usize>,
    /// For a streamed call: the rest of the final `arguments` after what was
    /// already sent, or `None` when the sent text is not a prefix of the final
    /// arguments (the call disagrees; no further delta is sent).
    pub remainder: Option<String>,
}

/// Result of [`IncrementalToolCallStreamer::plan_final`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FinalPlan {
    /// One entry per parsed call, in parsed order.
    pub calls: Vec<FinalCallPlan>,
    /// Streamed calls with no agreeing parsed call (positions in
    /// [`IncrementalToolCallStreamer::calls`]).
    pub orphans: Vec<usize>,
}

impl IncrementalToolCallStreamer {
    /// Reconcile the streamed calls with the authoritative parse.
    ///
    /// A streamed call agrees with the parsed call at the same index when the
    /// names match; it then keeps its streamed id (written into `parsed` so
    /// every later builder reuses it) and gets the unsent rest of the final
    /// arguments. Any disagreement is logged with `request_id`; the parsed
    /// call is emitted in one shot, under a fresh index when its own index was
    /// taken by a disagreeing streamed call, and the disagreeing streamed call
    /// gets no further delta.
    pub fn plan_final(&self, parsed: &mut [ParsedToolCall], request_id: &str) -> FinalPlan {
        let mut matched = vec![false; self.calls.len()];
        let mut next_free = parsed.len().max(self.ordinals);
        let mut plan = FinalPlan::default();
        for (i, call) in parsed.iter_mut().enumerate() {
            let streamed = self.call_position(i);
            let entry = match streamed {
                Some(pos) if self.calls[pos].name == call.name => {
                    matched[pos] = true;
                    let sent = &self.calls[pos].sent;
                    call.id = Some(self.calls[pos].id.clone());
                    let remainder = call
                        .arguments
                        .strip_prefix(sent.as_str())
                        .map(str::to_string);
                    if remainder.is_none() {
                        tracing::warn!(
                            target: "mlxcel::tool_calls",
                            request_id,
                            index = i,
                            name = %call.name,
                            "streamed tool-call arguments disagree with the final parse; \
                             no further argument deltas for this call"
                        );
                    }
                    FinalCallPlan {
                        parsed_index: i,
                        index: i,
                        streamed: Some(pos),
                        remainder,
                    }
                }
                Some(pos) => {
                    tracing::warn!(
                        target: "mlxcel::tool_calls",
                        request_id,
                        index = i,
                        streamed_name = %self.calls[pos].name,
                        parsed_name = %call.name,
                        "streamed tool-call name disagrees with the final parse; \
                         emitting the parsed call under a new index"
                    );
                    next_free += 1;
                    FinalCallPlan {
                        parsed_index: i,
                        index: next_free - 1,
                        streamed: None,
                        remainder: None,
                    }
                }
                None => FinalCallPlan {
                    parsed_index: i,
                    index: i,
                    streamed: None,
                    remainder: None,
                },
            };
            plan.calls.push(entry);
        }
        for (pos, matched) in matched.iter().enumerate() {
            if !matched {
                let call = &self.calls[pos];
                if call.index >= parsed.len() {
                    tracing::warn!(
                        target: "mlxcel::tool_calls",
                        request_id,
                        index = call.index,
                        name = %call.name,
                        parsed_calls = parsed.len(),
                        "streamed tool call is missing from the final parse"
                    );
                }
                plan.orphans.push(pos);
            }
        }
        plan
    }
}
