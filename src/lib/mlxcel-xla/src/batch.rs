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

//! Continuous-batching engine for the OpenXLA / IREE backend (issue #449 M3
//! Stage 2b).
//!
//! [`XlaBatchEngine`] productizes the Stage 2a spike: `B_max` slots share one
//! rank-5 KV cache and serve a stream of requests of different lengths that join
//! and leave the batch at different times, so the device stays full. A request is
//! seeded into a free slot by the single-sequence prefill, whose KV is written
//! DEVICE-SIDE into the slot (no host round-trip; the Stage 2a scheduler used a
//! d2h+h2d host-mirror round-trip, which 2b replaces). Then every active slot
//! advances one token per [`pump`](XlaBatchEngine::pump) through the ragged decode
//! graph, each row at its own position. Greedy (argmax) sampling, fixed `B_max`,
//! contiguous per-slot KV; richer sampling and paged KV are later stages.
//!
//! The engine is backend-neutral at the request level: callers
//! [`submit`](XlaBatchEngine::submit) a prompt + token budget, [`pump`] to drive a
//! step, and read per-request [`EngineEvent`]s. It holds no server types, so the
//! Stage 2c `BatchEngine` trait + server adapter wrap it without changing it.
//!
//! Compiled only under the `iree` feature (the engine drives real IREE
//! execution). The backend-neutral [`Scheduler`] bookkeeping is split out so its
//! admit/evict/cancel logic is unit-tested without a device (the crate's own
//! tests cannot link the IREE runtime; see the `iree.rs` test note).

use std::collections::VecDeque;
use std::fmt;
#[cfg(feature = "diagnostics")]
use std::time::Instant;

#[cfg(feature = "iree")]
use mlxcel_core::session::PreparedPrefill;
#[cfg(feature = "diagnostics")]
use mlxcel_core::session::{
    OwnedTensor, PreparedAttentionBias, PreparedPositions, PreparedTensorDType,
};

#[cfg(feature = "iree")]
use std::path::Path;

use crate::Gemma3nDensePle;
#[cfg(feature = "iree")]
use crate::Gemma3nPreparedPrefill;
#[cfg(feature = "diagnostics")]
use crate::iree::PreparedPrefillDiagnostics;
#[cfg(feature = "iree")]
use crate::iree::{IreeLlama, IreeRaggedLlama};
#[cfg(any(feature = "iree", test))]
use crate::prepared::PreparedPositionMode;
use crate::prepared::{
    MropeCoordinateError, PreparedInputError, PreparedIreePrefill, mrope_decode_coordinate,
};
use crate::prepared_deepstack::PreparedDeepStack;
use crate::sampler::SampleParams;
#[cfg(feature = "iree")]
use crate::sampler::sample;
use crate::{ContextCapacityError, validate_request_capacity};
#[cfg(any(feature = "iree", test))]
use crate::{DeepStackFeatures, DeepStackPreparedPrefill};

/// Typed validation failure returned before a request enters the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XlaAdmissionError {
    EmptyPrompt,
    ZeroMaxNewTokens,
    ContextCapacity(ContextCapacityError),
    Prepared(PreparedInputError),
    Gemma3nPrepared(String),
    DeepStackPrepared(String),
}

impl fmt::Display for XlaAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt => f.write_str("XLA batched submit requires a non-empty prompt"),
            Self::ZeroMaxNewTokens => f.write_str("max_new_tokens must be >= 1"),
            Self::ContextCapacity(err) => err.fmt(f),
            Self::Prepared(err) => err.fmt(f),
            Self::Gemma3nPrepared(err) => f.write_str(err),
            Self::DeepStackPrepared(err) => f.write_str(err),
        }
    }
}

impl std::error::Error for XlaAdmissionError {}

impl From<ContextCapacityError> for XlaAdmissionError {
    fn from(value: ContextCapacityError) -> Self {
        Self::ContextCapacity(value)
    }
}

impl From<PreparedInputError> for XlaAdmissionError {
    fn from(value: PreparedInputError) -> Self {
        Self::Prepared(value)
    }
}

/// Why a request stopped generating. Cancellation is silent (the caller that
/// called [`XlaBatchEngine::cancel`] already knows), so it is not a finish reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS token was emitted.
    Stop,
    /// The token budget (`max_new_tokens`) was reached.
    Length,
}

/// A per-request event the engine emits as it pumps. `req_id` is the id
/// [`XlaBatchEngine::submit`] returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    /// A newly generated token.
    Token { req_id: u64, token: i32 },
    /// The request finished; it produces no further events.
    Finished { req_id: u64, reason: FinishReason },
}

/// An active slot's running state.
struct Slot {
    req_id: u64,
    /// Last emitted token; the next decode input for this row.
    cur: i32,
    /// Tokens currently in this slot's KV (== the next write position).
    cache_len: i32,
    /// Signed logical RoPE offset retained independently for this slot.
    rope_delta: i32,
    /// Tokens emitted so far (counts toward `cap`).
    produced: usize,
    /// Token budget (`max_new_tokens`).
    cap: usize,
    /// Sampling parameters for this request (#449 M3 Stage 2d).
    params: SampleParams,
    /// PRNG state, advanced by each sample; seeded at admit for reproducibility.
    rng: u64,
    /// Token history for history-based penalties: the prompt followed by every
    /// generated token (matching mlxcel-core's `initial_token_history` window).
    /// Empty when the request enables no penalty, so greedy requests carry no
    /// per-slot history cost.
    history: Vec<i32>,
}

/// Owned input retained while a request is queued. Logical token ids stay
/// separate from the prepared static embeddings buffers so history-based
/// sampling and request accounting never infer token history from KV length.
enum PendingInput {
    Tokens(Vec<i32>),
    Prepared(PreparedIreePrefill),
    Gemma3nPrepared {
        prepared: PreparedIreePrefill,
        dense_ple: Gemma3nDensePle,
    },
    DeepStackPrepared {
        prepared: PreparedIreePrefill,
        deepstack: PreparedDeepStack,
    },
}

impl PendingInput {
    fn logical_tokens(&self) -> &[i32] {
        match self {
            Self::Tokens(tokens) => tokens,
            Self::Prepared(prepared) => &prepared.token_ids,
            Self::Gemma3nPrepared { prepared, .. } => &prepared.token_ids,
            Self::DeepStackPrepared { prepared, .. } => &prepared.token_ids,
        }
    }

    fn effective_len(&self) -> usize {
        match self {
            Self::Tokens(tokens) => tokens.len(),
            Self::Prepared(prepared) => prepared.effective_len,
            Self::Gemma3nPrepared { prepared, .. } => prepared.effective_len,
            Self::DeepStackPrepared { prepared, .. } => prepared.effective_len,
        }
    }

    fn rope_delta(&self) -> i32 {
        match self {
            Self::Tokens(_) | Self::Gemma3nPrepared { .. } => 0,
            Self::Prepared(prepared) | Self::DeepStackPrepared { prepared, .. } => {
                prepared.positions.rope_delta()
            }
        }
    }
}

/// A queued, not-yet-admitted request.
struct Pending {
    req_id: u64,
    input: PendingInput,
    cap: usize,
    params: SampleParams,
    cancelled: bool,
}

/// Resolve a request's PRNG seed: the explicit seed if given, else a deterministic
/// per-request seed so a no-seed request is still reproducible.
fn resolve_seed(params: &SampleParams, req_id: u64) -> u64 {
    params.seed.unwrap_or(0x9E37_79B9_7F4A_7C15 ^ req_id)
}

/// Greedy stop test after a slot emits `last`: EOS wins over Length when the
/// final token is both the cap-th token and an EOS id.
fn finish_reason(produced: usize, cap: usize, last: i32, eos: &[i32]) -> Option<FinishReason> {
    if eos.contains(&last) {
        Some(FinishReason::Stop)
    } else if produced >= cap {
        Some(FinishReason::Length)
    } else {
        None
    }
}

/// Backend-neutral scheduling state: `B_max` slots plus a FIFO admission queue.
/// Holds no IREE handles, so its admit/evict/cancel bookkeeping is unit-testable
/// without a device. [`XlaBatchEngine`] owns one of these next to the IREE engine
/// and consults it to decide which slots to fill and which requests are done.
struct Scheduler {
    b_max: usize,
    eos: Vec<i32>,
    slots: Vec<Option<Slot>>,
    queue: VecDeque<Pending>,
    next_id: u64,
}

impl Scheduler {
    fn new(b_max: usize, eos: Vec<i32>) -> Self {
        let mut slots = Vec::with_capacity(b_max);
        slots.resize_with(b_max, || None);
        Self {
            b_max,
            eos,
            slots,
            queue: VecDeque::new(),
            next_id: 0,
        }
    }

    /// Queue a request and return its id (monotonically increasing).
    fn submit_input(&mut self, input: PendingInput, cap: usize, params: SampleParams) -> u64 {
        let req_id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(Pending {
            req_id,
            input,
            cap,
            params,
            cancelled: false,
        });
        req_id
    }

    fn submit(&mut self, prompt: Vec<i32>, cap: usize, params: SampleParams) -> u64 {
        self.submit_input(PendingInput::Tokens(prompt), cap, params)
    }

    /// Cancel a request by id: free its slot if active, or drop it from the queue
    /// if still pending. Returns whether the id was found (and not already
    /// cancelled). A cancelled request emits no further events.
    fn cancel(&mut self, req_id: u64) -> bool {
        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(|s| s.req_id == req_id) {
                *slot = None;
                return true;
            }
        }
        if let Some(index) = self
            .queue
            .iter()
            .position(|pending| pending.req_id == req_id && !pending.cancelled)
        {
            self.queue.remove(index);
            return true;
        }
        false
    }

    /// No active slots and nothing left to admit.
    fn is_idle(&self) -> bool {
        self.slots.iter().all(Option::is_none) && self.queue.iter().all(|p| p.cancelled)
    }

    /// Pop the next live queued request, discarding any cancelled-while-queued
    /// entries it skips past.
    fn pop_next_pending(&mut self) -> Option<Pending> {
        while let Some(p) = self.queue.pop_front() {
            if !p.cancelled {
                return Some(p);
            }
        }
        None
    }

    /// Indices of the currently free slots, in order.
    fn free_slots(&self) -> Vec<usize> {
        (0..self.b_max)
            .filter(|&s| self.slots[s].is_none())
            .collect()
    }

    /// Whether any slot is active.
    fn any_active(&self) -> bool {
        self.slots.iter().any(Option::is_some)
    }
}

/// Validate and queue atomically: every failure returns before scheduler state is
/// changed, which keeps both free and active slots untouched.
fn queue_request(
    sched: &mut Scheduler,
    prompt: &[i32],
    max_new_tokens: usize,
    params: SampleParams,
    context_capacity: usize,
) -> Result<u64, XlaAdmissionError> {
    if prompt.is_empty() {
        return Err(XlaAdmissionError::EmptyPrompt);
    }
    if max_new_tokens == 0 {
        return Err(XlaAdmissionError::ZeroMaxNewTokens);
    }
    validate_request_capacity(prompt.len(), max_new_tokens, context_capacity)?;
    Ok(sched.submit(prompt.to_vec(), max_new_tokens, params))
}

fn queue_prepared_request(
    sched: &mut Scheduler,
    prepared: PreparedIreePrefill,
    max_new_tokens: usize,
    params: SampleParams,
    context_capacity: usize,
) -> Result<u64, XlaAdmissionError> {
    if max_new_tokens == 0 {
        return Err(XlaAdmissionError::ZeroMaxNewTokens);
    }
    validate_request_capacity(prepared.effective_len, max_new_tokens, context_capacity)?;
    Ok(sched.submit_input(PendingInput::Prepared(prepared), max_new_tokens, params))
}

fn queue_gemma3n_prepared_request(
    sched: &mut Scheduler,
    prepared: PreparedIreePrefill,
    dense_ple: Gemma3nDensePle,
    max_new_tokens: usize,
    params: SampleParams,
    context_capacity: usize,
) -> Result<u64, XlaAdmissionError> {
    if max_new_tokens == 0 {
        return Err(XlaAdmissionError::ZeroMaxNewTokens);
    }
    validate_request_capacity(prepared.effective_len, max_new_tokens, context_capacity)?;
    Ok(sched.submit_input(
        PendingInput::Gemma3nPrepared {
            prepared,
            dense_ple,
        },
        max_new_tokens,
        params,
    ))
}

fn mrope_slot_coordinates(slots: &[Option<Slot>]) -> Result<Vec<[i32; 3]>, MropeCoordinateError> {
    slots
        .iter()
        .map(|slot| {
            let Some(slot) = slot else {
                return Ok([0; 3]);
            };
            let coordinate = mrope_decode_coordinate(slot.cache_len, slot.rope_delta)?;
            Ok([coordinate; 3])
        })
        .collect()
}

fn queue_deepstack_prepared_request(
    sched: &mut Scheduler,
    prepared: PreparedIreePrefill,
    deepstack: PreparedDeepStack,
    max_new_tokens: usize,
    params: SampleParams,
    context_capacity: usize,
) -> Result<u64, XlaAdmissionError> {
    if max_new_tokens == 0 {
        return Err(XlaAdmissionError::ZeroMaxNewTokens);
    }
    validate_request_capacity(prepared.effective_len, max_new_tokens, context_capacity)?;
    Ok(sched.submit_input(
        PendingInput::DeepStackPrepared {
            prepared,
            deepstack,
        },
        max_new_tokens,
        params,
    ))
}

#[cfg(any(feature = "iree", test))]
fn prepare_deepstack_input<F>(
    request: DeepStackPreparedPrefill,
    hidden_size: usize,
    context_capacity: usize,
    position_mode: PreparedPositionMode,
    prepare_features: F,
) -> Result<(PreparedIreePrefill, PreparedDeepStack), XlaAdmissionError>
where
    F: FnOnce(&DeepStackFeatures) -> Result<PreparedDeepStack, String>,
{
    let (prepared, features) = request.into_parts();
    let deepstack = prepare_features(&features).map_err(XlaAdmissionError::DeepStackPrepared)?;
    let prepared = PreparedIreePrefill::prepare_for_mode(
        &prepared,
        hidden_size,
        context_capacity,
        position_mode,
    )?;
    Ok((prepared, deepstack))
}

/// The continuous-batching engine: `B_max` slots over one ragged decode graph,
/// fed by a FIFO queue. See the module docs.
#[cfg(feature = "iree")]
pub struct XlaBatchEngine {
    engine: IreeRaggedLlama,
    sched: Scheduler,
}

#[cfg(feature = "iree")]
impl XlaBatchEngine {
    /// Load a batched engine for `model_path` with `b_max` slots on `device`
    /// (`"local-task"` for CPU, `"cuda"` in a cuda build). Compiles the bundled
    /// prefill + the ragged decode graph for `b_max`, uploads the weights
    /// resident, and reads the model's EOS ids.
    ///
    /// # Errors
    ///
    /// Propagates load/compile failures, or an unsupported `b_max` (must be one
    /// of the bundled ragged graphs).
    pub fn load(model_path: &Path, b_max: usize, device: &str) -> Result<Self, String> {
        let context_capacity = crate::context_capacity_from_env()?;
        Self::load_with_context_capacity(model_path, b_max, device, context_capacity)
    }

    /// Load with an explicitly selected static graph and KV-cache capacity.
    pub fn load_with_context_capacity(
        model_path: &Path,
        b_max: usize,
        device: &str,
        context_capacity: usize,
    ) -> Result<Self, String> {
        let context_capacity = crate::context::validate_context_capacity_value(context_capacity)?;
        let engine = IreeRaggedLlama::load(model_path, device, b_max, context_capacity)?;
        let eos = crate::read_eos(model_path);
        Ok(Self {
            engine,
            sched: Scheduler::new(b_max, eos),
        })
    }

    /// The fixed slot count this engine was compiled for.
    #[must_use]
    pub fn b_max(&self) -> usize {
        self.engine.b_max()
    }

    /// Static sequence capacity compiled into the graph and per-slot KV cache.
    #[must_use]
    pub fn context_capacity(&self) -> usize {
        self.engine.context_capacity()
    }

    /// The model's EOS token ids (from `generation_config.json`).
    #[must_use]
    pub fn eos_token_ids(&self) -> &[i32] {
        &self.sched.eos
    }

    /// No active slots and nothing queued: a driver loop can stop pumping.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.sched.is_idle()
    }

    /// Number of queued (not yet admitted) live requests.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.sched.queue.iter().filter(|p| !p.cancelled).count()
    }

    /// Number of active slots.
    #[must_use]
    pub fn active_len(&self) -> usize {
        self.sched.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Queue a request: generate up to `max_new_tokens` tokens for `prompt`,
    /// sampling per `params` (greedy when `params.is_greedy()`), stopping early on
    /// EOS. Returns the request id used in the [`EngineEvent`]s [`pump`](Self::pump)
    /// yields.
    ///
    /// # Errors
    ///
    /// Errors on an empty prompt, a prompt longer than the prefill bucket, or a
    /// zero token budget.
    pub fn submit(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, XlaAdmissionError> {
        let context_capacity = self.context_capacity();
        queue_request(
            &mut self.sched,
            prompt,
            max_new_tokens,
            params,
            context_capacity,
        )
    }

    /// Queue an owned prepared-embeddings request. Validation and static-bucket
    /// materialization complete before scheduler state changes, and the queue
    /// then owns the only runtime copy until admission or cancellation.
    pub fn submit_prepared(
        &mut self,
        prepared: PreparedPrefill,
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, XlaAdmissionError> {
        let context_capacity = self.context_capacity();
        let prepared = PreparedIreePrefill::prepare_for_mode(
            &prepared,
            self.engine.hidden_size(),
            context_capacity,
            self.engine.position_mode(),
        )?;
        queue_prepared_request(
            &mut self.sched,
            prepared,
            max_new_tokens,
            params,
            context_capacity,
        )
    }

    /// Queue a Gemma3n embeddings-plus-dense-PLE request. Both allocations move
    /// into the pending entry and are dropped immediately on cancellation or
    /// after the slot is seeded.
    pub fn submit_gemma3n_prepared(
        &mut self,
        request: Gemma3nPreparedPrefill,
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, XlaAdmissionError> {
        let context_capacity = self.context_capacity();
        let (prepared, dense_ple) = request.into_parts();
        self.engine
            .validate_gemma3n_dense_ple(&dense_ple)
            .map_err(XlaAdmissionError::Gemma3nPrepared)?;
        let prepared =
            PreparedIreePrefill::prepare(&prepared, self.engine.hidden_size(), context_capacity)?;
        queue_gemma3n_prepared_request(
            &mut self.sched,
            prepared,
            dense_ple,
            max_new_tokens,
            params,
            context_capacity,
        )
    }

    /// Queue a prepared embeddings request with compact per-layer DeepStack
    /// features. The pending entry owns both static payloads and drops them on
    /// admission, cancellation, or any error path; decode slot state retains
    /// only KV and logical token history.
    pub fn submit_deepstack_prepared(
        &mut self,
        request: DeepStackPreparedPrefill,
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, XlaAdmissionError> {
        let context_capacity = self.context_capacity();
        let hidden_size = self.engine.hidden_size();
        let position_mode = self.engine.position_mode();
        let (prepared, deepstack) = prepare_deepstack_input(
            request,
            hidden_size,
            context_capacity,
            position_mode,
            |features| self.engine.prepare_deepstack(features),
        )?;
        queue_deepstack_prepared_request(
            &mut self.sched,
            prepared,
            deepstack,
            max_new_tokens,
            params,
            context_capacity,
        )
    }

    /// Cancel a request by id (frees its slot or drops it from the queue).
    /// Returns whether it was found. A cancelled request emits no further events.
    pub fn cancel(&mut self, req_id: u64) -> bool {
        self.sched.cancel(req_id)
    }

    /// Drive one engine step and return the events it produced.
    ///
    /// Admits queued requests into free slots (each seeded by a device-side KV
    /// write that leaves live slots untouched), then advances every active slot
    /// one token through the ragged decode graph, evicting any that hit EOS or
    /// their budget. A freshly admitted request emits its prefill token here, and
    /// finishes immediately if that token is EOS or its budget is 1.
    ///
    /// Returns an empty vec only when there is nothing to do
    /// ([`is_idle`](Self::is_idle)).
    ///
    /// # Errors
    ///
    /// Propagates prefill / decode execution failures.
    pub fn pump(&mut self) -> Result<Vec<EngineEvent>, String> {
        let eos = self.sched.eos.clone();
        let mut events = Vec::new();

        // ADMIT: fill free slots from the queue. The device-side prefill writes
        // only the admitted slot's KV region, so live slots are not disturbed; its
        // first-token logits are sampled here per the request's params.
        for s in self.sched.free_slots() {
            let Some(p) = self.sched.pop_next_pending() else {
                break;
            };
            let logits = match &p.input {
                PendingInput::Tokens(prompt) => self.engine.prefill_slot_logits(s, prompt)?,
                PendingInput::Prepared(prepared) => {
                    self.engine.prefill_prepared_slot_logits(s, prepared)?
                }
                PendingInput::Gemma3nPrepared {
                    prepared,
                    dense_ple,
                } => self
                    .engine
                    .prefill_gemma3n_prepared_slot_logits(s, prepared, dense_ple)?,
                PendingInput::DeepStackPrepared {
                    prepared,
                    deepstack,
                } => self
                    .engine
                    .prefill_deepstack_prepared_slot_logits(s, prepared, deepstack)?,
            };
            let mut rng = resolve_seed(&p.params, p.req_id);
            // History-based penalties see the prompt plus generated tokens (the
            // same window mlxcel-core seeds via `initial_token_history`). Build it
            // only when a penalty is active; greedy requests keep it empty.
            let needs_history = p.params.needs_penalties();
            let mut history = if needs_history {
                p.input.logical_tokens().to_vec()
            } else {
                Vec::new()
            };
            let first = sample(&logits, &p.params, &history, &mut rng);
            if needs_history {
                history.push(first);
            }
            events.push(EngineEvent::Token {
                req_id: p.req_id,
                token: first,
            });
            let slot = Slot {
                req_id: p.req_id,
                cur: first,
                cache_len: p.input.effective_len() as i32,
                rope_delta: p.input.rope_delta(),
                produced: 1,
                cap: p.cap,
                params: p.params,
                rng,
                history,
            };
            if let Some(reason) = finish_reason(slot.produced, slot.cap, first, &eos) {
                // Finished at its first token: leave the slot free for the next admit.
                events.push(EngineEvent::Finished {
                    req_id: p.req_id,
                    reason,
                });
            } else {
                self.sched.slots[s] = Some(slot);
            }
        }

        if !self.sched.any_active() {
            return Ok(events);
        }

        // DECODE: advance all B slots in one ragged step. Inactive rows carry
        // zeros (masked no-ops) and their logits are discarded.
        let b = self.sched.b_max;
        let mut tok = vec![0i32; b];
        let mut clen = vec![0i32; b];
        let mut pos = vec![0i32; b];
        for (s, slot) in self.sched.slots.iter().enumerate() {
            if let Some(st) = slot {
                tok[s] = st.cur;
                clen[s] = st.cache_len;
                pos[s] = st.cache_len;
            }
        }
        let logits = match self.engine.position_mode() {
            PreparedPositionMode::OneD => self.engine.decode_ragged_logits(&tok, &pos, &clen)?,
            PreparedPositionMode::Mrope3D => {
                let mrope_positions =
                    mrope_slot_coordinates(&self.sched.slots).map_err(|error| error.to_string())?;
                self.engine
                    .decode_ragged_mrope_logits(&tok, &mrope_positions, &clen)?
            }
        };
        let vocab = self.engine.vocab();

        // ADVANCE + EVICT: sample each active row from its `[vocab]` logit slice.
        for (s, slot_opt) in self.sched.slots.iter_mut().enumerate() {
            if let Some(slot) = slot_opt.as_mut() {
                let row = &logits[s * vocab..(s + 1) * vocab];
                let nt = sample(row, &slot.params, &slot.history, &mut slot.rng);
                if slot.params.needs_penalties() {
                    slot.history.push(nt);
                }
                slot.cur = nt;
                slot.cache_len += 1;
                slot.produced += 1;
                let req_id = slot.req_id;
                let done = finish_reason(slot.produced, slot.cap, nt, &eos);
                events.push(EngineEvent::Token { req_id, token: nt });
                if let Some(reason) = done {
                    events.push(EngineEvent::Finished { req_id, reason });
                    *slot_opt = None;
                }
            }
        }
        Ok(events)
    }
}

#[cfg(feature = "diagnostics")]
#[derive(Debug)]
pub struct Gemma3nCanonicalDiagnosticRun {
    pub layout: crate::Gemma3nDiagnosticLayout,
    pub intermediates: Vec<f32>,
    pub token_prefill_logits: Vec<f32>,
    pub prepared_prefill_logits: Vec<f32>,
    pub prefix_decode_logits: Vec<f32>,
    pub greedy_tokens: Vec<i32>,
}

#[cfg(feature = "diagnostics")]
#[derive(Debug)]
pub struct Gemma3nAllLayerDiagnosticRun {
    pub layout: crate::Gemma3nDiagnosticLayout,
    pub intermediates: Vec<f32>,
}

#[cfg(feature = "diagnostics")]
#[derive(Debug)]
pub struct Gemma3nPrefixDecodeDiagnosticRun {
    pub active_slot: usize,
    pub carrier_tokens: Vec<i32>,
    pub carrier_positions: Vec<i32>,
    pub carrier_cache_lengths: Vec<i32>,
    pub logits: Vec<f32>,
    pub top1: i32,
}

#[cfg(feature = "diagnostics")]
fn diagnostic_argmax(values: &[f32]) -> i32 {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index as i32)
}

/// Diagnostics-only LLaVA runner that shares one production ragged prefill /
/// decode bundle for capture and greedy generation.
#[cfg(feature = "diagnostics")]
pub struct LlavaReferenceDiagnosticEngine {
    engine: IreeRaggedLlama,
}

#[cfg(feature = "diagnostics")]
impl LlavaReferenceDiagnosticEngine {
    /// Compile and load the smallest production serve bundle at an explicit
    /// static context capacity.
    pub fn load(model_path: &Path, device: &str, context_capacity: usize) -> Result<Self, String> {
        Ok(Self {
            engine: IreeRaggedLlama::load(model_path, device, 4, context_capacity)?,
        })
    }

    #[must_use]
    pub fn context_capacity(&self) -> usize {
        self.engine.context_capacity()
    }

    /// Capture production prefill logits plus selected all-layer K/V and then
    /// continue greedily on the same resident cache.
    pub fn capture(
        &mut self,
        prepared: &PreparedPrefill,
        kv_width: usize,
        max_new_tokens: usize,
    ) -> Result<LlavaReferenceDiagnosticRun, String> {
        if max_new_tokens == 0 {
            return Err("LLaVA diagnostic generation requires max_new_tokens >= 1".to_string());
        }
        let prepared_iree = PreparedIreePrefill::prepare(
            prepared,
            self.engine.hidden_size(),
            self.engine.context_capacity(),
        )
        .map_err(|error| error.to_string())?;
        let prefill_started = Instant::now();
        let prefill = self
            .engine
            .prefill_prepared_slot_diagnostics(0, &prepared_iree, kv_width)?;
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();
        let mut tokens = Vec::with_capacity(max_new_tokens);
        let mut current = diagnostic_argmax(&prefill.logits);
        tokens.push(current);
        let mut cache_len = i32::try_from(prepared.sequence_len)
            .map_err(|_| "prepared sequence length does not fit i32".to_string())?;
        let decode_started = Instant::now();
        while tokens.len() < max_new_tokens {
            if cache_len as usize >= self.engine.context_capacity() {
                return Err(format!(
                    "diagnostic decode would exceed context_capacity={}",
                    self.engine.context_capacity()
                ));
            }
            let mut carrier_tokens = vec![0; self.engine.b_max()];
            let mut carrier_positions = vec![0; self.engine.b_max()];
            let mut carrier_cache_lengths = vec![0; self.engine.b_max()];
            carrier_tokens[0] = current;
            carrier_positions[0] = cache_len;
            carrier_cache_lengths[0] = cache_len;
            let logits = self.engine.decode_ragged_logits(
                &carrier_tokens,
                &carrier_positions,
                &carrier_cache_lengths,
            )?;
            current = diagnostic_argmax(&logits[..self.engine.vocab()]);
            tokens.push(current);
            cache_len += 1;
        }
        let decode_seconds = decode_started.elapsed().as_secs_f64();
        Ok(LlavaReferenceDiagnosticRun {
            prefill,
            tokens,
            prefill_seconds,
            decode_seconds,
        })
    }
}

#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, PartialEq)]
pub struct LlavaReferenceDiagnosticRun {
    pub prefill: PreparedPrefillDiagnostics,
    pub tokens: Vec<i32>,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

#[cfg(feature = "diagnostics")]
pub fn run_gemma3n_all_layer_diagnostics(
    model_dir: &Path,
    device: &str,
    context_capacity: usize,
    prompt: &[i32],
) -> Result<Gemma3nAllLayerDiagnosticRun, String> {
    const DIAGNOSTIC_SLOTS: usize = 4;
    if prompt.is_empty() || prompt.len() > context_capacity {
        return Err(format!(
            "all-layer diagnostic prompt length {} must be in 1..={context_capacity}",
            prompt.len()
        ));
    }
    let mut engine = IreeRaggedLlama::load_with_all_layer_diagnostics(
        model_dir,
        device,
        DIAGNOSTIC_SLOTS,
        context_capacity,
    )?;
    let intermediates = engine.prefill_diagnostics_slot(0, prompt)?;
    let layout = engine.diagnostic_layout()?.clone();
    layout.validate()?;
    if intermediates.len() != layout.total_len {
        return Err(format!(
            "all-layer diagnostic output has {} values, expected {}",
            intermediates.len(),
            layout.total_len
        ));
    }
    Ok(Gemma3nAllLayerDiagnosticRun {
        layout,
        intermediates,
    })
}

/// Seed exactly one fixed batch slot with a prefix, then execute one production
/// ragged decode step with zero-valued carrier rows around it.
///
/// This bounded probe deliberately excludes the canonical diagnostic module,
/// prepared-prefill parity, and greedy continuation. It exists to validate the
/// actual production decode bundle after emitter/plumbing changes without
/// consuming a canonical-oracle approval.
#[cfg(feature = "diagnostics")]
pub fn run_gemma3n_prefix_decode_diagnostic(
    model_dir: &Path,
    device: &str,
    context_capacity: usize,
    prompt: &[i32],
) -> Result<Gemma3nPrefixDecodeDiagnosticRun, String> {
    const DIAGNOSTIC_SLOTS: usize = 4;
    const ACTIVE_SLOT: usize = DIAGNOSTIC_SLOTS - 1;
    if prompt.len() < 2 || prompt.len() > context_capacity {
        return Err(format!(
            "prefix-decode diagnostic prompt length {} must be in 2..={context_capacity}",
            prompt.len()
        ));
    }
    let mut engine = IreeRaggedLlama::load(model_dir, device, DIAGNOSTIC_SLOTS, context_capacity)?;
    let prefix = &prompt[..prompt.len() - 1];
    let prefix_logits = engine.prefill_slot_logits(ACTIVE_SLOT, prefix)?;
    if prefix_logits.len() != engine.vocab() {
        return Err("prefix prefill returned an invalid vocabulary width".to_string());
    }

    let mut carrier_tokens = vec![0; DIAGNOSTIC_SLOTS];
    let mut carrier_positions = vec![0; DIAGNOSTIC_SLOTS];
    let mut carrier_cache_lengths = vec![0; DIAGNOSTIC_SLOTS];
    carrier_tokens[ACTIVE_SLOT] = prompt[prompt.len() - 1];
    let prefix_len =
        i32::try_from(prefix.len()).map_err(|_| "prefix length does not fit i32".to_string())?;
    carrier_positions[ACTIVE_SLOT] = prefix_len;
    carrier_cache_lengths[ACTIVE_SLOT] = prefix_len;

    let all_logits =
        engine.decode_ragged_logits(&carrier_tokens, &carrier_positions, &carrier_cache_lengths)?;
    let vocab = engine.vocab();
    let start = ACTIVE_SLOT
        .checked_mul(vocab)
        .ok_or_else(|| "active decode row offset overflows".to_string())?;
    let end = start
        .checked_add(vocab)
        .ok_or_else(|| "active decode row end overflows".to_string())?;
    let logits = all_logits
        .get(start..end)
        .ok_or_else(|| "ragged decode returned a truncated active row".to_string())?
        .to_vec();
    let top1 = diagnostic_argmax(&logits);
    Ok(Gemma3nPrefixDecodeDiagnosticRun {
        active_slot: ACTIVE_SLOT,
        carrier_tokens,
        carrier_positions,
        carrier_cache_lengths,
        logits,
        top1,
    })
}

/// Execute the entire pinned Gemma3n oracle surface with one model load.
///
/// This symbol only exists behind the explicit `diagnostics` feature. Normal
/// `iree` builds expose no raw-logit or intermediate-state diagnostic API and
/// do not compile/load the optional diagnostic module.
#[cfg(feature = "diagnostics")]
pub fn run_gemma3n_canonical_diagnostics(
    model_dir: &Path,
    device: &str,
    context_capacity: usize,
    prompt: &[i32],
    greedy_steps: usize,
) -> Result<Gemma3nCanonicalDiagnosticRun, String> {
    const DIAGNOSTIC_SLOTS: usize = 4;
    if prompt.len() < 2 || prompt.len() >= context_capacity {
        return Err(format!(
            "canonical diagnostic prompt length {} must be in 2..{context_capacity}",
            prompt.len()
        ));
    }
    if greedy_steps == 0 || prompt.len().saturating_add(greedy_steps) > context_capacity {
        return Err("canonical diagnostic greedy trajectory exceeds context capacity".to_string());
    }
    let mut engine = IreeRaggedLlama::load_with_diagnostics(
        model_dir,
        device,
        DIAGNOSTIC_SLOTS,
        context_capacity,
    )?;
    let intermediates = engine.prefill_diagnostics_slot(0, prompt)?;
    let layout = engine.diagnostic_layout()?.clone();
    layout.validate()?;
    if intermediates.len() != layout.total_len {
        return Err(format!(
            "diagnostic output has {} values, expected {}",
            intermediates.len(),
            layout.total_len
        ));
    }
    let embeddings_segment = layout
        .segment("scaled_embeddings")
        .ok_or_else(|| "diagnostic layout is missing scaled_embeddings".to_string())?;
    let expected_embeddings_shape = [context_capacity, engine.hidden_size()];
    if embeddings_segment.shape != expected_embeddings_shape {
        return Err(format!(
            "diagnostic scaled_embeddings shape {:?} does not match {:?}",
            embeddings_segment.shape, expected_embeddings_shape
        ));
    }
    let embeddings_len = prompt
        .len()
        .checked_mul(engine.hidden_size())
        .ok_or_else(|| "diagnostic embedding prefix length overflows".to_string())?;
    let embeddings_end = embeddings_segment
        .offset
        .checked_add(embeddings_len)
        .ok_or_else(|| "diagnostic embedding range overflows".to_string())?;
    let embeddings = intermediates
        .get(embeddings_segment.offset..embeddings_end)
        .ok_or_else(|| "diagnostic embedding prefix is truncated".to_string())?;
    let prepared = PreparedPrefill::new(
        prompt.to_vec(),
        OwnedTensor::new(
            embeddings
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            PreparedTensorDType::Float32,
            vec![1, prompt.len(), engine.hidden_size()],
        )
        .map_err(|error| error.to_string())?,
        PreparedPositions::Sequential {
            start: 0,
            length: prompt.len(),
        },
        PreparedAttentionBias {
            tensor: OwnedTensor::new(
                vec![0; prompt.len() * std::mem::size_of::<f32>()],
                PreparedTensorDType::Float32,
                vec![1, 1, 1, prompt.len()],
            )
            .map_err(|error| error.to_string())?,
            causal: true,
        },
        Vec::new(),
    )
    .map_err(|error| error.to_string())?;

    let ple_segment = layout
        .segment("projected_ple")
        .ok_or_else(|| "diagnostic layout is missing projected_ple".to_string())?;
    if ple_segment.shape.len() != 3 || ple_segment.shape[0] != context_capacity {
        return Err(format!(
            "diagnostic projected_ple shape {:?} is not [capacity, layers, hidden]",
            ple_segment.shape
        ));
    }
    let ple_row = ple_segment.shape[1]
        .checked_mul(ple_segment.shape[2])
        .ok_or_else(|| "diagnostic PLE row width overflows".to_string())?;
    let ple_prefix_len = prompt
        .len()
        .checked_mul(ple_row)
        .ok_or_else(|| "diagnostic PLE prefix length overflows".to_string())?;
    let ple_end = ple_segment
        .offset
        .checked_add(ple_prefix_len)
        .ok_or_else(|| "diagnostic PLE range overflows".to_string())?;
    let ple_prefix = intermediates
        .get(ple_segment.offset..ple_end)
        .ok_or_else(|| "diagnostic PLE prefix is truncated".to_string())?;
    let mut ple = vec![0.0; ple_segment.len];
    ple[..ple_prefix.len()].copy_from_slice(ple_prefix);
    let request = Gemma3nPreparedPrefill::new(
        prepared,
        Gemma3nDensePle::new(
            ple,
            context_capacity,
            ple_segment.shape[1],
            ple_segment.shape[2],
        )
        .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    let token_prefill_logits = engine.prefill_slot_logits(1, prompt)?;
    let (prepared, dense_ple) = request.into_parts();
    engine.validate_gemma3n_dense_ple(&dense_ple)?;
    let prepared = PreparedIreePrefill::prepare(&prepared, engine.hidden_size(), context_capacity)
        .map_err(|error| error.to_string())?;
    let prepared_prefill_logits =
        engine.prefill_gemma3n_prepared_slot_logits(2, &prepared, &dense_ple)?;
    let prefix = &prompt[..prompt.len() - 1];
    let prefix_logits = engine.prefill_slot_logits(3, prefix)?;
    let diagnostic_logits_segment = layout
        .segment("logits")
        .ok_or_else(|| "diagnostic layout is missing logits".to_string())?;
    let diagnostic_logits = &intermediates[diagnostic_logits_segment.offset
        ..diagnostic_logits_segment.offset + diagnostic_logits_segment.len];

    let mut tokens = vec![
        diagnostic_argmax(diagnostic_logits),
        diagnostic_argmax(&token_prefill_logits),
        diagnostic_argmax(&prepared_prefill_logits),
        prompt[prompt.len() - 1],
    ];
    let prompt_position =
        i32::try_from(prompt.len()).map_err(|_| "prompt length does not fit i32".to_string())?;
    let prefix_position =
        i32::try_from(prefix.len()).map_err(|_| "prefix length does not fit i32".to_string())?;
    let mut positions = vec![
        prompt_position,
        prompt_position,
        prompt_position,
        prefix_position,
    ];
    let first_decode = engine.decode_ragged_logits(&tokens, &positions, &positions)?;
    let vocab = engine.vocab();
    let prefix_decode_logits = first_decode[3 * vocab..4 * vocab].to_vec();
    let mut greedy_tokens = vec![tokens[1]];
    if greedy_steps > 1 {
        tokens = first_decode
            .chunks_exact(vocab)
            .map(diagnostic_argmax)
            .collect();
        greedy_tokens.push(tokens[1]);
    }
    for _ in 2..greedy_steps {
        for position in &mut positions {
            *position += 1;
        }
        if positions
            .iter()
            .any(|position| *position < 0 || *position as usize >= context_capacity)
        {
            return Err("canonical diagnostic decode exceeded context capacity".to_string());
        }
        let logits = engine.decode_ragged_logits(&tokens, &positions, &positions)?;
        tokens = logits.chunks_exact(vocab).map(diagnostic_argmax).collect();
        greedy_tokens.push(tokens[1]);
    }
    if prefix_logits.len() != vocab {
        return Err("prefix prefill returned an invalid vocabulary width".to_string());
    }
    Ok(Gemma3nCanonicalDiagnosticRun {
        layout,
        intermediates,
        token_prefill_logits,
        prepared_prefill_logits,
        prefix_decode_logits,
        greedy_tokens,
    })
}

/// A single-sequence reference engine used to validate [`XlaBatchEngine`]: it
/// generates the SAME greedy stream the batched engine would for one request run
/// alone (full-prompt prefill -> first token, then scalar decode), so a per-request
/// batched stream can be checked against an independent reference (the Stage 2a
/// reference-equivalence gate). Loads the weights once and reuses them across
/// references; the KV is overwritten by each [`generate`](Self::generate).
#[cfg(feature = "iree")]
pub struct XlaReferenceEngine {
    inner: IreeLlama,
}

#[cfg(feature = "iree")]
impl XlaReferenceEngine {
    /// Load the single-sequence reference engine for `model_path` on `device`.
    ///
    /// # Errors
    ///
    /// Propagates the underlying load/compile failures.
    pub fn load(model_path: &Path, device: &str) -> Result<Self, String> {
        let context_capacity = crate::context_capacity_from_env()?;
        Self::load_with_context_capacity(model_path, device, context_capacity)
    }

    /// Load a reference engine at the same explicit capacity as a batch engine.
    pub fn load_with_context_capacity(
        model_path: &Path,
        device: &str,
        context_capacity: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: IreeLlama::load(model_path, device, context_capacity)?,
        })
    }

    /// Greedy single-sequence stream for `prompt` (up to `max_new_tokens` tokens,
    /// stopping on EOS), matching the batched engine's slot convention: the prompt
    /// is prefilled in full and its argmax is the first token, then decode advances
    /// from there.
    ///
    /// # Errors
    ///
    /// Propagates prefill / decode failures.
    pub fn generate(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        eos: &[i32],
    ) -> Result<Vec<i32>, String> {
        if prompt.is_empty() {
            return Err("XLA reference generation requires a non-empty prompt".to_string());
        }
        validate_request_capacity(prompt.len(), max_new_tokens, self.inner.context_capacity())
            .map_err(|err| err.to_string())?;
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        let first = self.inner.prefill_first(prompt)?;
        let mut out = vec![first];
        let mut cache_len = prompt.len() as i32;
        let mut cur = first;
        while out.len() < max_new_tokens && !eos.contains(&cur) {
            let nt = self.inner.decode(cur, cache_len)?;
            out.push(nt);
            cache_len += 1;
            cur = nt;
        }
        Ok(out)
    }
}

// The IREE engine cannot be exercised in the crate's own tests (the runtime link
// recipe lives in the consuming binary, not here; see iree.rs), so these cover the
// backend-neutral Scheduler bookkeeping and the stop test, which the device path
// relies on. Run under any build without the `iree` feature (e.g. `cargo test -p
// mlxcel-xla`).
#[cfg(test)]
mod tests {
    use super::*;
    use mlxcel_core::session::{
        OwnedTensor, PreparedAttentionBias, PreparedModality, PreparedPositions,
        PreparedTensorDType,
    };

    const EOS: [i32; 1] = [42];

    #[test]
    fn finish_reason_eos_wins_over_length() {
        // EOS at the budget reports Stop, not Length.
        assert_eq!(finish_reason(4, 4, 42, &EOS), Some(FinishReason::Stop));
        assert_eq!(finish_reason(4, 4, 7, &EOS), Some(FinishReason::Length));
        assert_eq!(finish_reason(2, 4, 7, &EOS), None);
        assert_eq!(finish_reason(1, 1, 7, &EOS), Some(FinishReason::Length));
    }

    fn g() -> SampleParams {
        SampleParams::greedy()
    }

    fn prepared_input(tokens: Vec<i32>, hidden: usize, capacity: usize) -> PreparedIreePrefill {
        let sequence = tokens.len();
        let embedding_bytes = vec![0; sequence * hidden * std::mem::size_of::<f32>()];
        let bias_bytes = vec![0; sequence * std::mem::size_of::<f32>()];
        let value = mlxcel_core::session::PreparedPrefill::new(
            tokens,
            OwnedTensor::new(
                embedding_bytes,
                PreparedTensorDType::Float32,
                vec![1, sequence, hidden],
            )
            .unwrap(),
            PreparedPositions::Sequential {
                start: 0,
                length: sequence,
            },
            PreparedAttentionBias {
                tensor: OwnedTensor::new(
                    bias_bytes,
                    PreparedTensorDType::Float32,
                    vec![1, 1, 1, sequence],
                )
                .unwrap(),
                causal: true,
            },
            vec![PreparedModality {
                family: "test".into(),
                item_count: 1,
                token_count: 1,
            }],
        )
        .unwrap();
        PreparedIreePrefill::prepare(&value, hidden, capacity).unwrap()
    }

    fn tensor_i32(shape: &[usize], values: &[i32]) -> OwnedTensor {
        OwnedTensor::new(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            PreparedTensorDType::Int32,
            shape.to_vec(),
        )
        .unwrap()
    }

    fn tensor_f32(shape: &[usize], values: &[f32]) -> OwnedTensor {
        OwnedTensor::new(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            PreparedTensorDType::Float32,
            shape.to_vec(),
        )
        .unwrap()
    }

    fn public_deepstack_input(rope_delta: i32) -> (PreparedIreePrefill, PreparedDeepStack) {
        let sequence = 4;
        let hidden = 2;
        let prepared = mlxcel_core::session::PreparedPrefill::new(
            vec![1, 2, 3, 4],
            tensor_f32(&[1, sequence, hidden], &[0.0; 8]),
            PreparedPositions::Mrope3D {
                tensor: tensor_i32(&[3, sequence], &[0, 1, 2, 3, 0, 1, 3, 3, 0, 2, 2, 3]),
                rope_delta,
            },
            PreparedAttentionBias {
                tensor: tensor_f32(&[1, 1, 1, sequence], &[0.0; 4]),
                causal: true,
            },
            vec![PreparedModality {
                family: "deepstack-test".into(),
                item_count: 1,
                token_count: 1,
            }],
        )
        .unwrap();
        let features = DeepStackFeatures::new(
            tensor_i32(&[1], &[1]),
            tensor_f32(&[1, 1, hidden], &[0.25, -0.5]),
            tensor_i32(&[1], &[0]),
        )
        .unwrap();
        let request = DeepStackPreparedPrefill::new(prepared, features).unwrap();
        let schema = crate::emitter::DeepStackConfig {
            target_layer_indices: vec![0],
            max_visual_positions: 2,
        };
        prepare_deepstack_input(
            request,
            hidden,
            8,
            PreparedPositionMode::Mrope3D,
            |features| {
                PreparedDeepStack::prepare(features, &schema, hidden)
                    .map_err(|error| error.to_string())
            },
        )
        .unwrap()
    }

    fn admit_next(scheduler: &mut Scheduler, slot_index: usize) -> u64 {
        let pending = scheduler.pop_next_pending().expect("pending request");
        let req_id = pending.req_id;
        scheduler.slots[slot_index] = Some(Slot {
            req_id,
            cur: 7,
            cache_len: pending.input.effective_len() as i32,
            rope_delta: pending.input.rope_delta(),
            produced: 1,
            cap: pending.cap,
            params: pending.params,
            rng: 0,
            history: Vec::new(),
        });
        req_id
    }

    fn gemma3n_input(tokens: Vec<i32>) -> PendingInput {
        PendingInput::Gemma3nPrepared {
            prepared: prepared_input(tokens, 2, 8),
            dense_ple: Gemma3nDensePle::new(vec![0.0; 8 * 2 * 2], 8, 2, 2).unwrap(),
        }
    }

    fn deepstack_input(tokens: Vec<i32>) -> PendingInput {
        PendingInput::DeepStackPrepared {
            prepared: prepared_input(tokens, 2, 8),
            deepstack: PreparedDeepStack {
                visual_positions: vec![1, -1],
                layer_features: vec![1.0, 2.0, 0.0, 0.0],
                layer_indices: vec![0],
                actual_layer_count: 1,
                actual_visual_count: 1,
                max_layer_count: 1,
                max_visual_count: 2,
                hidden_size: 2,
            },
        }
    }

    #[test]
    fn submit_assigns_increasing_ids_and_queues() {
        let mut s = Scheduler::new(2, EOS.to_vec());
        assert_eq!(s.submit(vec![1, 2], 8, g()), 0);
        assert_eq!(s.submit(vec![3], 8, g()), 1);
        assert_eq!(s.free_slots(), vec![0, 1]);
        assert!(!s.is_idle());
        assert!(!s.any_active());
    }

    #[test]
    fn request_admission_checks_prompt_plus_generation_budget() {
        let mut s = Scheduler::new(2, EOS.to_vec());
        let id = queue_request(&mut s, &[1; 768], 256, g(), 1024).expect("exact fit");
        assert_eq!(id, 0);

        let err = queue_request(&mut s, &[1; 769], 256, g(), 1024).unwrap_err();
        assert_eq!(
            err,
            XlaAdmissionError::ContextCapacity(ContextCapacityError {
                effective_prompt_len: 769,
                max_new_tokens: 256,
                context_capacity: 1024,
            })
        );
    }

    #[test]
    fn rejected_request_does_not_mutate_queue_or_live_slot() {
        let mut s = Scheduler::new(2, EOS.to_vec());
        s.slots[0] = Some(Slot {
            req_id: 41,
            cur: 7,
            cache_len: 300,
            rope_delta: 0,
            produced: 4,
            cap: 100,
            params: g(),
            rng: 9,
            history: vec![1, 2, 3],
        });
        let before_next_id = s.next_id;
        let before_queue_len = s.queue.len();

        let err = queue_request(&mut s, &[5; 900], 125, g(), 1024).unwrap_err();
        assert!(matches!(err, XlaAdmissionError::ContextCapacity(_)));
        assert_eq!(s.next_id, before_next_id, "no request id was consumed");
        assert_eq!(s.queue.len(), before_queue_len, "nothing was queued");
        let live = s.slots[0].as_ref().expect("existing slot remains active");
        assert_eq!(live.req_id, 41);
        assert_eq!(live.cur, 7);
        assert_eq!(live.cache_len, 300);
        assert!(s.slots[1].is_none(), "a free slot remains free");
    }

    #[test]
    fn pop_next_pending_skips_cancelled() {
        let mut s = Scheduler::new(2, EOS.to_vec());
        let a = s.submit(vec![1], 8, g());
        let b = s.submit(vec![2], 8, g());
        assert!(s.cancel(a)); // cancel the head while queued
        let got = s.pop_next_pending().expect("a live request remains");
        assert_eq!(got.req_id, b);
        assert!(s.pop_next_pending().is_none());
    }

    #[test]
    fn mixed_pending_inputs_keep_logical_history_and_effective_length_distinct() {
        let mut s = Scheduler::new(2, EOS.to_vec());
        let text = s.submit(vec![1, 2], 8, g());
        let prepared = prepared_input(vec![7, 8, 9], 2, 8);
        let multimodal =
            s.submit_input(PendingInput::Prepared(prepared), 8, SampleParams::greedy());
        let gemma3n = s.submit_input(gemma3n_input(vec![10, 11]), 8, SampleParams::greedy());
        let deepstack =
            s.submit_input(deepstack_input(vec![12, 13, 14]), 8, SampleParams::greedy());

        let first = s.pop_next_pending().unwrap();
        assert_eq!(first.req_id, text);
        assert!(matches!(first.input, PendingInput::Tokens(_)));
        assert_eq!(first.input.logical_tokens(), &[1, 2]);
        assert_eq!(first.input.effective_len(), 2);

        let second = s.pop_next_pending().unwrap();
        assert_eq!(second.req_id, multimodal);
        assert!(matches!(second.input, PendingInput::Prepared(_)));
        assert_eq!(second.input.logical_tokens(), &[7, 8, 9]);
        assert_eq!(second.input.effective_len(), 3);

        let third = s.pop_next_pending().unwrap();
        assert_eq!(third.req_id, gemma3n);
        assert!(matches!(third.input, PendingInput::Gemma3nPrepared { .. }));
        assert_eq!(third.input.logical_tokens(), &[10, 11]);
        assert_eq!(third.input.effective_len(), 2);

        let fourth = s.pop_next_pending().unwrap();
        assert_eq!(fourth.req_id, deepstack);
        assert!(matches!(
            fourth.input,
            PendingInput::DeepStackPrepared { .. }
        ));
        assert_eq!(fourth.input.logical_tokens(), &[12, 13, 14]);
        assert_eq!(fourth.input.effective_len(), 3);
    }

    #[test]
    fn public_deepstack_deltas_survive_pending_slots_and_zero_on_reuse() {
        let mut scheduler = Scheduler::new(3, EOS.to_vec());
        let (negative_prepared, negative_features) = public_deepstack_input(-1);
        let negative = queue_deepstack_prepared_request(
            &mut scheduler,
            negative_prepared,
            negative_features,
            4,
            g(),
            8,
        )
        .unwrap();
        let (positive_prepared, positive_features) = public_deepstack_input(2);
        let positive = queue_deepstack_prepared_request(
            &mut scheduler,
            positive_prepared,
            positive_features,
            4,
            g(),
            8,
        )
        .unwrap();
        let (zero_prepared, zero_features) = public_deepstack_input(0);
        let zero = queue_deepstack_prepared_request(
            &mut scheduler,
            zero_prepared,
            zero_features,
            4,
            g(),
            8,
        )
        .unwrap();

        assert_eq!(
            scheduler
                .queue
                .iter()
                .map(|pending| (pending.req_id, pending.input.rope_delta()))
                .collect::<Vec<_>>(),
            vec![(negative, -1), (positive, 2), (zero, 0)],
            "the scheduler must own each public request's signed delta while pending"
        );

        assert_eq!(admit_next(&mut scheduler, 0), negative);
        assert_eq!(admit_next(&mut scheduler, 1), positive);
        assert_eq!(admit_next(&mut scheduler, 2), zero);
        assert_eq!(
            mrope_slot_coordinates(&scheduler.slots).unwrap(),
            [[3, 3, 3], [6, 6, 6], [4, 4, 4]],
            "the first decode coordinate for each row is sequence_len + its own delta"
        );

        assert!(scheduler.cancel(positive));
        assert_eq!(scheduler.free_slots(), [1]);
        let replacement = scheduler.submit(vec![9, 10, 11, 12], 4, g());
        assert_eq!(admit_next(&mut scheduler, 1), replacement);
        assert_eq!(
            mrope_slot_coordinates(&scheduler.slots).unwrap(),
            [[3, 3, 3], [4, 4, 4], [4, 4, 4]],
            "reusing the cancelled positive-delta slot must restore text delta zero"
        );
    }

    #[test]
    fn cancel_each_input_form_before_and_after_admit_reuses_slots() {
        let mut s = Scheduler::new(1, EOS.to_vec());
        let queued_text = s.submit(vec![1, 2], 8, g());
        assert!(s.cancel(queued_text));

        let prepared = prepared_input(vec![4, 5, 6], 2, 8);
        let queued_prepared =
            s.submit_input(PendingInput::Prepared(prepared), 8, SampleParams::greedy());
        let pending = s.pop_next_pending().expect("prepared request remains");
        assert_eq!(pending.req_id, queued_prepared);
        let effective_len = pending.input.effective_len();
        drop(pending);

        s.slots[0] = Some(Slot {
            req_id: queued_prepared,
            cur: 7,
            cache_len: effective_len as i32,
            rope_delta: 0,
            produced: 1,
            cap: 8,
            params: g(),
            rng: 0,
            history: vec![4, 5, 6, 7],
        });
        assert!(s.cancel(queued_prepared));
        assert_eq!(s.free_slots(), vec![0]);

        let reused = s.submit(vec![9], 2, g());
        assert_eq!(s.pop_next_pending().unwrap().req_id, reused);
        assert_eq!(s.free_slots(), vec![0]);
    }

    #[test]
    fn gemma3n_ple_is_removed_immediately_on_cancel_and_slot_can_be_reused() {
        let mut s = Scheduler::new(1, EOS.to_vec());
        let gemma = s.submit_input(gemma3n_input(vec![1, 2, 3]), 8, g());
        assert_eq!(s.queue.len(), 1);
        assert!(s.cancel(gemma));
        assert!(
            s.queue.is_empty(),
            "cancellation must drop the request-owned PLE now, not at a later pump"
        );

        let replacement = s.submit(vec![9], 2, g());
        assert_eq!(s.pop_next_pending().unwrap().req_id, replacement);
        assert_eq!(s.free_slots(), vec![0]);
    }

    #[test]
    fn deepstack_side_tensors_are_dropped_on_cancel_and_slot_can_be_reused() {
        let mut s = Scheduler::new(1, EOS.to_vec());
        let deepstack = s.submit_input(deepstack_input(vec![1, 2, 3]), 8, g());
        assert_eq!(s.queue.len(), 1);
        assert!(s.cancel(deepstack));
        assert!(
            s.queue.is_empty(),
            "cancellation must drop request-owned DeepStack tensors immediately"
        );

        let replacement = s.submit(vec![9], 2, g());
        assert_eq!(s.pop_next_pending().unwrap().req_id, replacement);
        assert_eq!(s.free_slots(), vec![0]);
    }

    #[test]
    fn prepared_capacity_rejection_is_atomic() {
        let mut s = Scheduler::new(1, EOS.to_vec());
        let prepared = prepared_input(vec![1, 2, 3], 2, 4);
        let err = queue_prepared_request(&mut s, prepared, 2, g(), 4).unwrap_err();
        assert!(matches!(err, XlaAdmissionError::ContextCapacity(_)));
        assert_eq!(s.next_id, 0);
        assert!(s.queue.is_empty());
        assert_eq!(s.free_slots(), vec![0]);
    }

    #[test]
    fn cancel_active_slot_frees_it() {
        let mut s = Scheduler::new(2, EOS.to_vec());
        let id = s.submit(vec![1], 8, g());
        // simulate an admit into slot 0
        let p = s.pop_next_pending().unwrap();
        s.slots[0] = Some(Slot {
            req_id: p.req_id,
            cur: 5,
            cache_len: 1,
            rope_delta: 0,
            produced: 1,
            cap: p.cap,
            params: p.params,
            rng: 0,
            history: Vec::new(),
        });
        assert!(s.any_active());
        assert!(s.cancel(id));
        assert!(!s.any_active());
        assert!(s.is_idle());
        assert!(!s.cancel(id)); // already gone
    }

    #[test]
    fn idle_only_when_drained() {
        let mut s = Scheduler::new(1, EOS.to_vec());
        assert!(s.is_idle());
        let id = s.submit(vec![1], 8, g());
        assert!(!s.is_idle()); // queued
        assert!(s.cancel(id));
        assert!(s.is_idle()); // cancelled-while-queued counts as drained
    }
}
