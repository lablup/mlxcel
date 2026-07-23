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

#[cfg(feature = "iree")]
use mlxcel_core::session::PreparedPrefill;

#[cfg(feature = "iree")]
use std::path::Path;

#[cfg(feature = "iree")]
use crate::iree::{IreeLlama, IreeRaggedLlama};
use crate::prepared::{PreparedInputError, PreparedIreePrefill};
use crate::sampler::SampleParams;
#[cfg(feature = "iree")]
use crate::sampler::sample;
use crate::{ContextCapacityError, validate_request_capacity};

/// Typed validation failure returned before a request enters the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XlaAdmissionError {
    EmptyPrompt,
    ZeroMaxNewTokens,
    ContextCapacity(ContextCapacityError),
    Prepared(PreparedInputError),
}

impl fmt::Display for XlaAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt => f.write_str("XLA batched submit requires a non-empty prompt"),
            Self::ZeroMaxNewTokens => f.write_str("max_new_tokens must be >= 1"),
            Self::ContextCapacity(err) => err.fmt(f),
            Self::Prepared(err) => err.fmt(f),
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
}

impl PendingInput {
    fn logical_tokens(&self) -> &[i32] {
        match self {
            Self::Tokens(tokens) => tokens,
            Self::Prepared(prepared) => &prepared.token_ids,
        }
    }

    fn effective_len(&self) -> usize {
        match self {
            Self::Tokens(tokens) => tokens.len(),
            Self::Prepared(prepared) => prepared.effective_len,
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
        for p in &mut self.queue {
            if p.req_id == req_id && !p.cancelled {
                p.cancelled = true;
                return true;
            }
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
        let prepared =
            PreparedIreePrefill::prepare(&prepared, self.engine.hidden_size(), context_capacity)?;
        queue_prepared_request(
            &mut self.sched,
            prepared,
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
        let mut pos = vec![0i32; b];
        let mut clen = vec![0i32; b];
        for (s, slot) in self.sched.slots.iter().enumerate() {
            if let Some(st) = slot {
                tok[s] = st.cur;
                pos[s] = st.cache_len;
                clen[s] = st.cache_len;
            }
        }
        let logits = self.engine.decode_ragged_logits(&tok, &pos, &clen)?;
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
