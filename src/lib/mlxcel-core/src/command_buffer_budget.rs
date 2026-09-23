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

//! Decode-only Metal command-buffer input budget.
//!
//! MLX commits a Metal command buffer once the element count of its distinct
//! inputs passes `MLX_MAX_MB_PER_BUFFER << 20`. During decode that cap, at
//! MLX's default, splits every token into one command buffer per one or two
//! layers and idles the GPU at each boundary; during prefill the same cap is
//! what keeps a long prompt's activations from piling up until the buffer
//! completes. [`crate::hardware::decode_mb_per_buffer`] says what the decode
//! budget should be, and [`DecodeCommandBufferBudget`] applies it for the
//! lifetime of a decode loop or decode step and restores the previous value
//! when dropped, so prefill always encodes against the device default.
//!
//! Only pipelined decode benefits, where step n+1 is encoded while the GPU
//! still runs step n. A synchronous step that encodes and then waits should
//! stay on the device default: with one large buffer the GPU cannot start
//! until the whole step is encoded, which on M1 Ultra made server decode both
//! slower and erratic.
//!
//! The override is read by MLX while it encodes, which happens on the thread
//! that calls `eval` / `async_eval`. Enter the guard on that thread, after the
//! prefill work has been encoded (an `async_eval` returns only once its graph
//! is encoded and committed), and keep it alive until the last decode eval.
//! The override is process-wide: work that another thread encodes while a
//! guard is alive also sees the decode budget. That can only move where a
//! command buffer boundary falls, never what is computed.

use crate::ffi;

/// RAII guard that raises MLX's per-command-buffer input budget to the decode
/// value while it is alive. A no-op when no decode budget applies (M5+,
/// non-Apple, `MLXCEL_DECODE_MB_PER_BUFFER=0`, or an operator-set
/// `MLX_MAX_MB_PER_BUFFER`). Guards nest: each restores the value it found.
#[must_use = "the budget is restored as soon as the guard is dropped"]
pub struct DecodeCommandBufferBudget {
    previous: Option<i32>,
}

impl DecodeCommandBufferBudget {
    /// Apply the process's decode budget
    /// ([`crate::hardware::decode_mb_per_buffer`]).
    pub fn enter() -> Self {
        Self::with_budget(crate::hardware::decode_mb_per_buffer())
    }

    /// Apply an explicit budget; `None` leaves the current value untouched.
    pub fn with_budget(budget: Option<u32>) -> Self {
        let Some(mb) = budget.and_then(|mb| i32::try_from(mb).ok()) else {
            return Self { previous: None };
        };
        let previous = ffi::metal_mb_per_buffer_override();
        ffi::set_metal_mb_per_buffer_override(mb);
        Self {
            previous: Some(previous),
        }
    }

    /// Whether this guard changed the budget (and will restore it on drop).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.previous.is_some()
    }
}

impl Drop for DecodeCommandBufferBudget {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            ffi::set_metal_mb_per_buffer_override(previous);
        }
    }
}

#[cfg(test)]
#[path = "command_buffer_budget_tests.rs"]
mod tests;
