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

//! Per-frame host-sync counter and opt-in sub-stage timers for the
//! streaming speech pipeline (issue #2045).
//!
//! The streaming VoiceChat session calls [`begin_frame`] before each 80 ms
//! frame and [`end_frame`] after it. In between, every host round trip on
//! the streaming path (an `eval` / `try_eval` / `item` / readback that the
//! pipeline needs for its own output) reports itself with [`count_sync`],
//! which is a thread-local counter increment and costs nothing measurable.
//!
//! Sub-stage attribution is separate and off unless the frame was begun with
//! `detail = true`: [`mark`] then forces evaluation of the given arrays and
//! bills the wall time since the previous mark (or [`restart_clock`]) to the
//! named sub-stage, summing repeated names within a frame. Forcing those
//! evaluations serializes the graph, so sub-stage numbers explain where the
//! time goes but must not be read as the headline frame time. Forced
//! probe evaluations are not counted as host syncs.
//!
//! State is thread-local because MLX evaluation is thread-affine and a
//! session is driven on one thread.
//!
//! Used by: `models::nemotron_voicechat::streaming` (frame bracketing), the
//! FastConformer streaming encoder, the RNNT stream step, the VoiceChat LLM
//! step, EAR-TTS and the codec validation.

use std::cell::RefCell;
use std::time::Instant;

use mlxcel_core::MlxArray;

#[derive(Default)]
struct ProbeState {
    syncs: u32,
    detail: bool,
    clock: Option<Instant>,
    stages: Vec<(&'static str, f64)>,
}

thread_local! {
    static STATE: RefCell<ProbeState> = RefCell::new(ProbeState::default());
}

/// What one frame recorded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameProbe {
    /// Host round trips the pipeline itself made.
    pub host_syncs: u32,
    /// Sub-stage wall times in milliseconds, in first-seen order (empty
    /// unless the frame ran with `detail`).
    pub stages: Vec<(&'static str, f64)>,
}

/// Start a frame: clear the counters and choose whether [`mark`] records.
pub fn begin_frame(detail: bool) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.syncs = 0;
        s.detail = detail;
        s.stages.clear();
        s.clock = detail.then(Instant::now);
    });
}

/// Finish a frame and return what it recorded; later marks are ignored
/// until the next [`begin_frame`].
pub fn end_frame() -> FrameProbe {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.detail = false;
        s.clock = None;
        FrameProbe {
            host_syncs: std::mem::take(&mut s.syncs),
            stages: std::mem::take(&mut s.stages),
        }
    })
}

/// Record `n` host round trips.
pub fn count_sync(n: u32) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.syncs = s.syncs.saturating_add(n);
    });
}

/// Whether sub-stage marks are being recorded.
pub fn detail() -> bool {
    STATE.with(|s| s.borrow().detail)
}

/// Restart the sub-stage clock without billing anything (for code that
/// runs between stages and should not be attributed).
pub fn restart_clock() {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if s.detail {
            s.clock = Some(Instant::now());
        }
    });
}

/// Evaluate `outputs` and bill the time since the previous mark to `name`.
/// A no-op unless the current frame records detail.
pub fn mark(name: &'static str, outputs: &[&MlxArray]) {
    if !detail() {
        return;
    }
    if !outputs.is_empty() {
        let ptrs: Vec<*const MlxArray> = outputs.iter().map(|a| *a as *const MlxArray).collect();
        // SAFETY: every pointer comes from a live `&MlxArray` borrowed for
        // the duration of this call.
        unsafe { mlxcel_core::eval_all(&ptrs) };
    }
    let now = Instant::now();
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let elapsed = s
            .clock
            .map_or(0.0, |c| now.duration_since(c).as_secs_f64() * 1000.0);
        s.clock = Some(now);
        match s.stages.iter_mut().find(|(n, _)| *n == name) {
            Some((_, total)) => *total += elapsed,
            None => s.stages.push((name, elapsed)),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_syncs_and_ignores_marks_without_detail() {
        begin_frame(false);
        count_sync(2);
        count_sync(1);
        mark("ignored", &[]);
        let probe = end_frame();
        assert_eq!(probe.host_syncs, 3);
        assert!(probe.stages.is_empty());
    }

    #[test]
    fn detail_marks_sum_repeated_names_in_order() {
        begin_frame(true);
        mark("a", &[]);
        mark("b", &[]);
        mark("a", &[]);
        let probe = end_frame();
        let names: Vec<_> = probe.stages.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["a", "b"]);
        assert!(probe.stages.iter().all(|(_, ms)| *ms >= 0.0));
        // The frame is closed: marks after end_frame record nothing.
        mark("late", &[]);
        begin_frame(false);
        assert!(end_frame().stages.is_empty());
    }
}
