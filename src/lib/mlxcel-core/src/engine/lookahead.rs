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

//! The pieces of the decode lookahead pipeline (#632) that every driver of
//! [`Engine::submit`] shares: the kill switch, the rule for which sequences
//! can have a speculative append undone, and the device-side token feedback.
//!
//! Two drivers use them: the server scheduler's cross-tick state machine
//! (`BatchScheduler::run_decode_tick`) and the raw-completion client's B=1
//! loop ([`super::DirectEngine::generate`]), whose per-row pipeline feeds the
//! forward from a per-row draw ([`Engine::submit_forward`],
//! [`Engine::draw_row`]) the same way. Their state machines stay
//! separate: the scheduler's has to tear the pipeline down on admission,
//! preemption and membership changes and re-run a finishing tick through its
//! synchronous path so completion and prompt-cache donation see clean caches,
//! none of which a single in-call sequence has.

use super::Engine;
use crate::cache::{SequenceId, SequenceStateBackend};
use crate::generate::LanguageModel;
use crate::{MlxArray, UniquePtr};

/// The environment switch that keeps every decode step synchronous: set (to
/// any value), neither the server scheduler nor the raw-completion client
/// (`mlxcel generate`, the inference session, `mlxcel-bench-decode`, the
/// parity harness's `d:engine` arm) engages the lookahead pipeline.
pub const FORCE_SYNC_ENV: &str = "MLXCEL_FORCE_SYNC";

/// Whether [`FORCE_SYNC_ENV`] is set. Callers probe it once (at construction)
/// rather than per step.
pub fn force_sync_requested() -> bool {
    std::env::var(FORCE_SYNC_ENV).is_ok()
}

/// The `[B, 1]` int32 input of the next pipelined forward, built on the
/// device from the `[B]` tokens a [`Engine::submit`] returned, so the next
/// forward can be scheduled before those tokens reach the host. The cast
/// matches the dtype of a host-built `from_slice_i32` input.
pub fn lookahead_feedback_input(tokens: &MlxArray) -> UniquePtr<MlxArray> {
    let column = crate::ffi::reshape_token_for_forward(tokens);
    crate::ffi::astype(&column, crate::dtype::INT32)
}

impl<M: LanguageModel> Engine<M> {
    /// Whether [`Engine::unwind_appends`] undoes a pipelined step's
    /// speculative appends on `id` exactly, so the sequence may decode on the
    /// lookahead pipeline.
    ///
    /// Dense and pool-backed paged sequences have a trimmable KV tail. A
    /// family whose NATURAL backend is model-owned keeps its state in the
    /// model, which the pool trim cannot reach, so it qualifies only when the
    /// model rewinds that state itself
    /// ([`LanguageModel::supports_decode_lookahead_rewind`], #2159); it may
    /// then be allocated model-owned or paged (shadow accounting under the
    /// paged decode override, #1754). Keying on the allocated backend alone
    /// would let a model-owned family on shadow paged storage pipeline while
    /// the teardown reached none of its real state. `false` for a sequence
    /// that is not open.
    pub fn can_unwind_lookahead(&self, id: SequenceId) -> bool {
        let model_owned =
            self.model.sequence_state_layout().backend == SequenceStateBackend::ModelOwned;
        if model_owned && !self.model.supports_decode_lookahead_rewind() {
            return false;
        }
        match self.pool.get(id) {
            Some(set) => {
                matches!(
                    set.backend,
                    SequenceStateBackend::DenseKvCache | SequenceStateBackend::PagedKvCache
                ) || (model_owned && set.backend == SequenceStateBackend::ModelOwned)
            }
            None => false,
        }
    }
}
