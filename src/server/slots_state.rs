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

//! Server-level slot registry (llama-server b10621 compatible, issue #1440).
//!
//! b10621 serves every request through a numbered `server_slot`, and three
//! surfaces observe that number: `GET /slots` reports per-slot state, the
//! native completion responses carry `id_slot`, and `POST /slots/:id_slot`
//! saves, restores, or erases one slot's prompt cache. mlxcel's
//! continuous-batching scheduler has no per-request slot concept, so this
//! registry provides one at the HTTP boundary: a generation request acquires
//! the lowest free slot id when it starts and releases it when it finishes,
//! keeping the finished task's counters visible the way b10621 keeps
//! `task_prev` after a slot goes idle.
//!
//! The registry is observational: admission control stays with the scheduler's
//! queue, and a request that finds every slot busy still runs (it reports
//! `id_slot: -1` until a slot frees and it can bind late). `total_slots` is
//! `--parallel`, which is also what b10621 reports even when the model clamps
//! the effective decode batch below it.
//!
//! Prompt and generated text are retained only when the operator opted in:
//! `LLAMA_SERVER_SLOTS_DEBUG` (b10621's own debug switch for the redacted
//! `/slots` fields) or `--slot-save-path` (the save action needs the token
//! stream). Without either, slot state holds counters only, so `/slots` can
//! never leak prompt content that was not asked to be retained.

use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

/// What one slot's cached token stream currently is.
///
/// b10621 keeps `slot.prompt.tokens` across requests: a finished task leaves
/// its prompt+generation cached, a restore replaces it with tokens from a
/// file, and an erase clears it. mlxcel retains the same state symbolically
/// (text from the last task, or ids from a restore) and materializes token
/// ids only when a save/erase action needs the count.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum SlotCache {
    /// Nothing cached (fresh slot, or after `erase`).
    #[default]
    Empty,
    /// The last finished task's prompt and generated text, retained only when
    /// [`SlotRegistry::retains_text`] is set.
    FromTask,
    /// Token ids rehydrated by a `restore` action.
    Restored(Vec<i32>),
}

/// Per-task counters mirrored into the b10621 `/slots` slot object.
#[derive(Debug, Clone, Default)]
pub struct SlotTask {
    pub id_task: i64,
    /// Resolved request parameters, in the b10621 `params` position.
    pub params: serde_json::Value,
    pub n_prompt_tokens: usize,
    pub n_prompt_tokens_processed: usize,
    pub n_prompt_tokens_cache: usize,
    pub n_decoded: usize,
    /// `n_predict` budget for `n_remain`; `None` reports `-1` like b10621
    /// does for an unlimited budget.
    pub n_predict: Option<i64>,
    pub has_next_token: bool,
    pub has_new_line: bool,
    /// Retained only when the registry retains text (see module docs).
    pub prompt_text: String,
    /// Retained only when the registry retains text.
    pub generated_text: String,
}

#[derive(Debug, Default)]
struct Slot {
    processing: bool,
    task: Option<SlotTask>,
    cache: SlotCache,
}

/// Registry of `--parallel` slots, shared through [`super::AppState`].
#[derive(Debug)]
pub struct SlotRegistry {
    slots: Mutex<Vec<Slot>>,
    next_task_id: AtomicI64,
    retain_text: bool,
}

/// RAII handle tying one generation request to one slot id.
///
/// Acquired before generation starts; if every slot is busy the handle stays
/// unbound (`id_slot() == -1`) and re-tries binding on each progress update,
/// which is the observational analogue of b10621 assigning a slot to a queued
/// task when one frees. Dropping the handle marks the slot idle while keeping
/// the task's final counters visible, as b10621's `task_prev` does.
pub struct SlotHandle {
    registry: std::sync::Arc<SlotRegistry>,
    slot: Mutex<Option<usize>>,
    id_task: i64,
    params: serde_json::Value,
    n_predict: Option<i64>,
    prompt_text: String,
}

impl SlotRegistry {
    /// `retain_text` keeps prompt/generated text in slot state; pass true only
    /// when `LLAMA_SERVER_SLOTS_DEBUG` is set or `--slot-save-path` is
    /// configured.
    pub fn new(total_slots: usize, retain_text: bool) -> Self {
        let n = total_slots.max(1);
        Self {
            slots: Mutex::new((0..n).map(|_| Slot::default()).collect()),
            next_task_id: AtomicI64::new(0),
            retain_text,
        }
    }

    /// Whether prompt/generated text is retained in slot state.
    pub fn retains_text(&self) -> bool {
        self.retain_text
    }

    /// Number of slots (b10621 `total_slots`).
    pub fn total(&self) -> usize {
        self.slots.lock().map(|s| s.len()).unwrap_or(0)
    }

    /// Number of slots not currently processing (b10621 `n_idle_slots`,
    /// the `fail_on_no_slot` gate).
    pub fn idle_count(&self) -> usize {
        self.slots
            .lock()
            .map(|s| s.iter().filter(|slot| !slot.processing).count())
            .unwrap_or(0)
    }

    /// Begin tracking one generation request.
    pub fn begin(
        self: &std::sync::Arc<Self>,
        prompt_text: &str,
        params: serde_json::Value,
        n_predict: Option<i64>,
    ) -> SlotHandle {
        let id_task = self.next_task_id.fetch_add(1, Ordering::Relaxed);
        let handle = SlotHandle {
            registry: self.clone(),
            slot: Mutex::new(None),
            id_task,
            params,
            n_predict,
            prompt_text: if self.retain_text {
                prompt_text.to_string()
            } else {
                String::new()
            },
        };
        // Deliberately NOT bound here (#1440). Binding at route entry gave a
        // slot to a request that was still waiting in the scheduler queue,
        // which under saturation left a request that WAS decoding with no slot
        // for its whole life: measured with `--parallel 2` and five concurrent
        // streams, one request emitted all forty-one frames with
        // `id_slot: -1`, which b10621 never does because its task waits for a
        // slot before it starts. The handle binds on its first real progress
        // signal instead (`on_prefill` / `on_token` reach `try_bind` through
        // `update`), so a slot is held only by a request the worker is
        // actually serving. The slot count and the scheduler's decode width
        // are both `--parallel`, so a request that is decoding always finds
        // one free.
        handle
    }

    /// Snapshot every slot as its b10621 `/slots` JSON object.
    ///
    /// `n_ctx` is the per-slot context window and `speculative` whether the
    /// server can draft, both server-wide in mlxcel. `include_text` must only
    /// be true under `LLAMA_SERVER_SLOTS_DEBUG`, mirroring b10621's
    /// `slots_debug` gate on the `prompt` / `generated` fields.
    pub fn slots_json(
        &self,
        n_ctx: usize,
        speculative: bool,
        include_text: bool,
    ) -> Vec<serde_json::Value> {
        let slots = match self.slots.lock() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };
        slots
            .iter()
            .enumerate()
            .map(|(id, slot)| {
                let mut obj = serde_json::json!({
                    "id": id,
                    "n_ctx": n_ctx,
                    "speculative": speculative,
                    "is_processing": slot.processing,
                });
                if let Some(task) = &slot.task {
                    let n_remain = task
                        .n_predict
                        .map(|budget| budget - task.n_decoded as i64)
                        .unwrap_or(-1);
                    obj["id_task"] = task.id_task.into();
                    obj["n_prompt_tokens"] = task.n_prompt_tokens.into();
                    obj["n_prompt_tokens_processed"] = task.n_prompt_tokens_processed.into();
                    obj["n_prompt_tokens_cache"] = task.n_prompt_tokens_cache.into();
                    obj["params"] = task.params.clone();
                    // b10621 wraps the progress object in a one-element array.
                    obj["next_token"] = serde_json::json!([{
                        "has_next_token": task.has_next_token,
                        "has_new_line": task.has_new_line,
                        "n_remain": n_remain,
                        "n_decoded": task.n_decoded,
                    }]);
                    if include_text {
                        obj["prompt"] = task.prompt_text.clone().into();
                        obj["generated"] = task.generated_text.clone().into();
                    }
                }
                obj
            })
            .collect()
    }

    /// The slot's cached token stream, for the save action. `None` when the
    /// slot id is out of range, `Some(Err(()))` when the slot is processing
    /// (b10621 defers; mlxcel reports it busy).
    #[allow(clippy::type_complexity)]
    pub fn cache_for_save(
        &self,
        id_slot: usize,
    ) -> Option<Result<(SlotCache, String, String), ()>> {
        let slots = self.slots.lock().ok()?;
        let slot = slots.get(id_slot)?;
        if slot.processing {
            return Some(Err(()));
        }
        let (prompt, generated) = slot
            .task
            .as_ref()
            .map(|t| (t.prompt_text.clone(), t.generated_text.clone()))
            .unwrap_or_default();
        Some(Ok((slot.cache.clone(), prompt, generated)))
    }

    /// Install restored tokens on an idle slot (`restore` action).
    pub fn install_restored(&self, id_slot: usize, tokens: Vec<i32>) -> Option<Result<(), ()>> {
        let mut slots = self.slots.lock().ok()?;
        let slot = slots.get_mut(id_slot)?;
        if slot.processing {
            return Some(Err(()));
        }
        slot.cache = SlotCache::Restored(tokens);
        Some(Ok(()))
    }

    /// Clear an idle slot's cached tokens (`erase` action), returning what was
    /// cached so the route can report `n_erased`.
    #[allow(clippy::type_complexity)]
    pub fn erase(&self, id_slot: usize) -> Option<Result<(SlotCache, String, String), ()>> {
        let mut slots = self.slots.lock().ok()?;
        let slot = slots.get_mut(id_slot)?;
        if slot.processing {
            return Some(Err(()));
        }
        let cache = std::mem::take(&mut slot.cache);
        let (prompt, generated) = slot
            .task
            .take()
            .map(|t| (t.prompt_text, t.generated_text))
            .unwrap_or_default();
        Some(Ok((cache, prompt, generated)))
    }

    fn bind_lowest_free(&self, task: SlotTask) -> Option<usize> {
        let mut slots = self.slots.lock().ok()?;
        let id = slots.iter().position(|slot| !slot.processing)?;
        let slot = &mut slots[id];
        slot.processing = true;
        slot.task = Some(task);
        slot.cache = if self.retain_text {
            SlotCache::FromTask
        } else {
            SlotCache::Empty
        };
        Some(id)
    }

    fn with_slot_task(&self, id: usize, f: impl FnOnce(&mut SlotTask)) {
        if let Ok(mut slots) = self.slots.lock()
            && let Some(slot) = slots.get_mut(id)
            && let Some(task) = slot.task.as_mut()
        {
            f(task);
        }
    }

    fn release(&self, id: usize) {
        if let Ok(mut slots) = self.slots.lock()
            && let Some(slot) = slots.get_mut(id)
        {
            slot.processing = false;
            if let Some(task) = slot.task.as_mut() {
                task.has_next_token = false;
            }
        }
    }
}

impl SlotHandle {
    fn try_bind(&self) {
        let mut bound = match self.slot.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        if bound.is_some() {
            return;
        }
        let task = SlotTask {
            id_task: self.id_task,
            params: self.params.clone(),
            n_predict: self.n_predict,
            has_next_token: true,
            prompt_text: self.prompt_text.clone(),
            ..SlotTask::default()
        };
        *bound = self.registry.bind_lowest_free(task);
    }

    /// The bound slot id, or `-1` while unbound (every slot busy), which is
    /// also the sentinel b10621 uses on frames that carry no slot.
    pub fn id_slot(&self) -> i64 {
        self.slot
            .lock()
            .ok()
            .and_then(|s| s.map(|id| id as i64))
            .unwrap_or(-1)
    }

    fn update(&self, f: impl FnOnce(&mut SlotTask)) {
        self.try_bind();
        if let Ok(bound) = self.slot.lock()
            && let Some(id) = *bound
        {
            self.registry.with_slot_task(id, f);
        }
    }

    /// Record the prefill outcome: total prompt tokens and how many came from
    /// the KV prefix cache.
    pub fn on_prefill(&self, prompt_tokens: usize, cached_tokens: usize) {
        self.update(|task| {
            task.n_prompt_tokens = prompt_tokens;
            task.n_prompt_tokens_cache = cached_tokens;
            task.n_prompt_tokens_processed = prompt_tokens.saturating_sub(cached_tokens);
        });
    }

    /// Record one decoded token (and its text when the registry retains text).
    pub fn on_token(&self, piece: &str) {
        let retain = self.registry.retain_text;
        self.update(|task| {
            task.n_decoded += 1;
            if piece.contains('\n') {
                task.has_new_line = true;
            }
            if retain {
                task.generated_text.push_str(piece);
            }
        });
    }

    /// Record the final counters and text when generation ends.
    ///
    /// `generated_text` replaces whatever the per-token updates accumulated:
    /// the non-streaming paths never see `on_token`, and the streaming paths'
    /// final result text is authoritative anyway (it includes tail flushes
    /// the token callback never saw).
    pub fn finish(
        &self,
        prompt_tokens: usize,
        cached_tokens: usize,
        completion_tokens: usize,
        generated_text: &str,
    ) {
        let retain = self.registry.retain_text;
        self.update(|task| {
            task.n_prompt_tokens = prompt_tokens;
            task.n_prompt_tokens_cache = cached_tokens;
            task.n_prompt_tokens_processed = prompt_tokens.saturating_sub(cached_tokens);
            task.n_decoded = completion_tokens;
            task.has_next_token = false;
            if generated_text.contains('\n') {
                task.has_new_line = true;
            }
            if retain {
                task.generated_text = generated_text.to_string();
            }
        });
    }
}

impl Drop for SlotHandle {
    fn drop(&mut self) {
        if let Ok(bound) = self.slot.lock()
            && let Some(id) = *bound
        {
            self.registry.release(id);
        }
    }
}

#[cfg(test)]
#[path = "slots_state_tests.rs"]
mod slots_state_tests;
