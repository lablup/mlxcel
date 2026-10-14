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

//! Undoing the last few single-token decode writes of a [`RotatingKVCache`]
//! (issue #2159).
//!
//! [`RotatingKVCache::trim`] only rewinds `offset`/`idx`. Before the ring has
//! wrapped that is exact: a decode write lands on a slot that was never valid.
//! After the wrap each write overwrites the K/V of the position that just left
//! the window, so a trim leaves that slot holding the speculative K/V, and a
//! later step attends it. The batch scheduler's decode lookahead appends up to
//! two speculative positions it may have to unwind, so a cache that opts in
//! keeps a small log of the rows its last writes destroyed and restores them.
//!
//! Each logged write records the cursors it started from. A write that
//! overwrites a valid slot inside a [`DecodeLookaheadAppendScope`] (the
//! scheduler's speculative prime forward) also copies that slot's
//! `[B, H, 1, D]` K and V rows; outside the scope (a synchronous step, the
//! force-sync mode, a request the pipeline does not admit) no rows are copied,
//! and such a write is simply not rewindable, which no teardown asks for.
//!
//! The copies are lazy MLX nodes that read the pre-write buffer. Left
//! unevaluated they keep that whole buffer alive and stop the write's
//! `slice_update` from reusing it in place (on GB10 that cost Gemma 3 4B 12%
//! of its wrapped-window decode rate), so the model schedules them right after
//! building its forward ([`flush_decode_undo_rows`]), ahead of the forward's
//! own evaluation on the same stream.

use std::cell::Cell;
use std::collections::VecDeque;

use super::{KVCacheMode, RotatingKVCache};
use crate::{MlxArray, UniquePtr, ffi};

/// Most speculative single-token appends the batch scheduler's decode
/// lookahead ever has to unwind from one sequence: the step-n append plus the
/// step-n+1 append of a steady-tick teardown. A cache that logs this many
/// writes can serve every lookahead teardown.
pub const DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS: usize = 2;

thread_local! {
    static SPECULATIVE_APPEND: Cell<bool> = const { Cell::new(false) };
}

/// Marks the single-token writes made while it lives as speculative appends
/// that a lookahead teardown may unwind, so a logging cache copies the rows
/// they overwrite. Thread-local: the scheduler builds each forward on its own
/// thread. Nesting is not expected; the guard restores the previous state.
///
/// Used by: the batch scheduler's lookahead prime forward.
#[must_use = "the scope ends when the guard is dropped"]
pub struct DecodeLookaheadAppendScope {
    previous: bool,
}

impl DecodeLookaheadAppendScope {
    pub fn enter() -> Self {
        Self {
            previous: SPECULATIVE_APPEND.with(|flag| flag.replace(true)),
        }
    }
}

impl Drop for DecodeLookaheadAppendScope {
    fn drop(&mut self) {
        SPECULATIVE_APPEND.with(|flag| flag.set(self.previous));
    }
}

/// Rows a decode write overwrote, copied out of the pre-write buffers.
pub(crate) struct DecodeUndoRows {
    keys: UniquePtr<MlxArray>,
    values: UniquePtr<MlxArray>,
    /// Still an unevaluated copy that references the pre-write buffer.
    pending: bool,
}

/// One logged single-token write.
pub(crate) struct DecodeUndoEntry {
    /// `offset` before the write.
    offset: i32,
    /// Write cursor before the write (after any over-window pre-trim, before
    /// the wrap reset), so a rewind restores the exact cursor.
    idx: i32,
    /// Physical slot the write landed on.
    slot: i32,
    /// The slot held a valid position before the write.
    overwrote: bool,
    /// The overwritten rows, for an overwrite inside a
    /// [`DecodeLookaheadAppendScope`]. A warmup write (`!overwrote`) needs
    /// none: a rewind zeroes its slot again, as buffer growth left it.
    rows: Option<DecodeUndoRows>,
}

/// Bounded log of a rotating cache's newest single-token decode writes.
pub(crate) struct DecodeUndoLog {
    depth: usize,
    entries: VecDeque<DecodeUndoEntry>,
}

impl DecodeUndoLog {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            entries: VecDeque::with_capacity(depth),
        }
    }
}

impl RotatingKVCache {
    /// Keep an undo log of the newest `depth` single-token decode writes, so
    /// [`Self::rewind_decode_writes`] can unwind them exactly even after the
    /// ring has wrapped. `0` disables the log and drops any entries.
    ///
    /// Used by: Gemma 3 server sequence caches (decode lookahead teardown).
    pub fn set_decode_undo_depth(&mut self, depth: usize) {
        self.decode_undo = (depth > 0).then(|| DecodeUndoLog::new(depth));
    }

    /// The configured undo depth (`0` when the log is disabled).
    pub fn decode_undo_depth(&self) -> usize {
        self.decode_undo.as_ref().map_or(0, |log| log.depth)
    }

    /// Number of decode writes currently logged.
    pub fn decode_undo_len(&self) -> usize {
        self.decode_undo.as_ref().map_or(0, |log| log.entries.len())
    }

    /// Forget every logged write while keeping the configured depth. Called
    /// by every mutation that is not a logged single-token write (a
    /// multi-token append, trim, snapshot restore, buffering, detach), since
    /// the logged cursors and rows no longer describe the cache after it.
    pub(crate) fn clear_decode_undo(&mut self) {
        if let Some(log) = self.decode_undo.as_mut() {
            log.entries.clear();
        }
    }

    /// Record the single-token write about to land on slot `pos` of
    /// `keys`/`values`, which are the pre-write buffers. `offset` and `idx`
    /// are the cursors the write starts from.
    pub(crate) fn record_decode_write(
        &mut self,
        keys: &MlxArray,
        values: &MlxArray,
        pos: i32,
        offset: i32,
        idx: i32,
    ) {
        if self.buffer_size > 0 || self.mode != KVCacheMode::Fp16 {
            self.clear_decode_undo();
            return;
        }
        let max_size = self.max_size;
        let Some(log) = self.decode_undo.as_mut() else {
            return;
        };
        // A warmup write lands on a slot that was never valid, so the cursors
        // alone undo it; only a speculative overwrite needs the old rows.
        let overwrote = offset >= max_size;
        let speculative = SPECULATIVE_APPEND.with(Cell::get);
        let rows = (overwrote && speculative).then(|| {
            let k_shape = ffi::array_shape(keys);
            let v_shape = ffi::array_shape(values);
            let k_row = ffi::slice(
                keys,
                &[0, 0, pos, 0],
                &[k_shape[0], k_shape[1], pos + 1, k_shape[3]],
            );
            let v_row = ffi::slice(
                values,
                &[0, 0, pos, 0],
                &[v_shape[0], v_shape[1], pos + 1, v_shape[3]],
            );
            DecodeUndoRows {
                keys: ffi::contiguous(&k_row, false),
                values: ffi::contiguous(&v_row, false),
                pending: true,
            }
        });
        if log.entries.len() == log.depth {
            log.entries.pop_front();
        }
        log.entries.push_back(DecodeUndoEntry {
            offset,
            idx,
            slot: pos,
            overwrote,
            rows,
        });
    }

    /// Append the still-unevaluated row copies to `out` and mark them
    /// evaluated. The caller must schedule every returned array.
    pub(super) fn take_pending_decode_undo_rows(&mut self, out: &mut Vec<*const MlxArray>) {
        let Some(log) = self.decode_undo.as_mut() else {
            return;
        };
        for rows in log.entries.iter_mut().filter_map(|e| e.rows.as_mut()) {
            if rows.pending {
                rows.pending = false;
                out.push(&*rows.keys as *const MlxArray);
                out.push(&*rows.values as *const MlxArray);
            }
        }
    }

    /// Unwind the last `n` single-token writes, leaving `keys`, `values`,
    /// `offset`, and `idx` as they were before them.
    ///
    /// Logged overwrites get their old rows back, newest first; warmup writes
    /// need only the cursors. Writes older than the log are accepted only
    /// while the ring is still chronological below the window, where a cursor
    /// rewind is exact. Every refusal is an `Err` that leaves the cache
    /// untouched: speculative buffering, non-FP16 storage, a disabled log,
    /// `n > offset`, a write past the warmup that the log no longer holds, or
    /// an overwrite made outside a [`DecodeLookaheadAppendScope`].
    ///
    /// Used by: Gemma 3 `rewind_decode_appends` (decode lookahead teardown).
    pub fn rewind_decode_writes(&mut self, n: i32) -> Result<(), String> {
        let describe = |cache: &Self, why: &str| {
            format!(
                "rotating decode rewind of {n}: {why} (offset {}, idx {}, logged {}/{})",
                cache.offset,
                cache.idx,
                cache.decode_undo_len(),
                cache.decode_undo_depth()
            )
        };
        if n < 0 {
            return Err(describe(self, "negative count"));
        }
        if n == 0 {
            return Ok(());
        }
        if self.buffer_size > 0 {
            return Err(describe(self, "speculative buffering is active"));
        }
        if self.mode != KVCacheMode::Fp16 {
            return Err(describe(
                self,
                &format!("{:?} storage is not covered", self.mode),
            ));
        }
        let Some(log) = self.decode_undo.as_ref() else {
            return Err(describe(self, "the undo log is disabled"));
        };
        if n > self.offset {
            return Err(describe(self, "more writes than the cache holds"));
        }

        // The newest `logged` writes come from the log, newest last.
        let logged = (n as usize).min(log.entries.len());
        let undo = log.entries.len() - logged;
        let mut expected = self.offset;
        for entry in log.entries.iter().skip(undo).rev() {
            if entry.offset != expected - 1 {
                return Err(describe(self, "the log does not end at the current offset"));
            }
            expected = entry.offset;
        }
        let (base_offset, base_idx) = log
            .entries
            .get(undo)
            .filter(|_| logged > 0)
            .map_or((self.offset, self.idx), |e| (e.offset, e.idx));
        let unlogged = n - logged as i32;
        if unlogged > 0 && !(base_offset <= self.max_size && base_idx == base_offset) {
            return Err(describe(
                self,
                "a write past the warmup is no longer in the log",
            ));
        }
        let physical = match (self.keys.as_ref(), self.values.as_ref()) {
            (Some(k), Some(v)) => {
                let len = ffi::array_shape(k)[2];
                if ffi::array_shape(v)[2] != len {
                    return Err(describe(self, "key and value buffers disagree"));
                }
                len
            }
            _ => 0,
        };
        if log
            .entries
            .iter()
            .skip(undo)
            .any(|e| e.overwrote && e.rows.is_none())
        {
            return Err(describe(
                self,
                "an overwrite outside a lookahead append scope kept no rows",
            ));
        }
        if log
            .entries
            .iter()
            .skip(undo)
            .any(|e| e.rows.is_some() && e.slot >= physical)
        {
            return Err(describe(self, "a logged slot lies outside the buffer"));
        }

        // Validated: apply. Restore the overwritten rows newest first, so a
        // slot written twice ends with its oldest content. A warmup slot was
        // never valid; zero it again so the buffer matches one that never saw
        // the write (no step reads it either way).
        let mut log = self.decode_undo.take().expect("log checked above");
        for entry in log.entries.drain(undo..).rev() {
            match entry.rows {
                Some(rows) => self.write_slots(entry.slot, &rows.keys, &rows.values),
                None if !entry.overwrote && entry.slot < physical => self.zero_slots(entry.slot, 1),
                None => {}
            }
        }
        // Unlogged warmup writes sit at their own positions (chronological).
        let first_unlogged = base_idx - unlogged;
        if unlogged > 0 && first_unlogged < physical {
            self.zero_slots(first_unlogged, unlogged.min(physical - first_unlogged));
        }
        self.decode_undo = Some(log);

        let trimmed = self.trim_cursors(n);
        debug_assert_eq!(trimmed, n, "n <= offset was checked");
        self.idx = base_idx - unlogged;
        debug_assert_eq!(self.offset, base_offset - unlogged);
        // The restore produced new buffers, and a later warmup write must not
        // treat them as its own in-place target.
        self.inplace_marker = None;
        Ok(())
    }
}

impl RotatingKVCache {
    /// Overwrite slots `slot..slot + rows` of both buffers.
    fn write_slots(&mut self, slot: i32, keys: &MlxArray, values: &MlxArray) {
        let (Some(k), Some(v)) = (self.keys.take(), self.values.take()) else {
            return;
        };
        let len = ffi::array_shape(keys)[2];
        let k_shape = ffi::array_shape(&k);
        let v_shape = ffi::array_shape(&v);
        self.keys = Some(ffi::slice_update(
            &k,
            keys,
            &[0, 0, slot, 0],
            &[k_shape[0], k_shape[1], slot + len, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            &v,
            values,
            &[0, 0, slot, 0],
            &[v_shape[0], v_shape[1], slot + len, v_shape[3]],
        ));
    }

    /// Zero `count` slots starting at `slot`.
    fn zero_slots(&mut self, slot: i32, count: i32) {
        let (Some(k), Some(v)) = (self.keys.as_ref(), self.values.as_ref()) else {
            return;
        };
        let k_shape = ffi::array_shape(k);
        let v_shape = ffi::array_shape(v);
        let k_zeros = ffi::zeros(
            &[k_shape[0], k_shape[1], count, k_shape[3]],
            ffi::array_dtype(k),
        );
        let v_zeros = ffi::zeros(
            &[v_shape[0], v_shape[1], count, v_shape[3]],
            ffi::array_dtype(v),
        );
        self.write_slots(slot, &k_zeros, &v_zeros);
    }
}

/// Schedule the pending undo-row copies of `caches` in one evaluation.
///
/// A model whose caches log decode writes calls this right after building a
/// forward and before anything evaluates it. The copies then run (on the same
/// stream) ahead of the writes that overwrite their slots, and release the
/// pre-write buffers so those writes can reuse them in place. No-op, with no
/// MLX call, while nothing is pending (always before the ring wraps).
pub fn flush_decode_undo_rows<'a>(caches: impl IntoIterator<Item = &'a mut RotatingKVCache>) {
    let mut pending: Vec<*const MlxArray> = Vec::new();
    for cache in caches {
        cache.take_pending_decode_undo_rows(&mut pending);
    }
    if pending.is_empty() {
        return;
    }
    // SAFETY: every pointer targets an `MlxArray` owned by a `UniquePtr` in a
    // cache's undo log. The caches are mutably borrowed for the whole call, so
    // no entry is dropped or moved while `async_eval_all` reads them.
    unsafe { ffi::async_eval_all(&pending) };
}
