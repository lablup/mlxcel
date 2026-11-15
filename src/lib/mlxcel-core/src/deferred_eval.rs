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

//! Evaluation errors raised by an eager `async_eval` inside a forward.
//!
//! A model forward is infallible, so when [`crate::utils::schedule_prefill_state`]
//! starts evaluating a recurrent state and MLX reports a failure (an earlier
//! command buffer ran out of memory, for example), the forward cannot return
//! it. MLX hands each error out once, so dropping it would let the caller's
//! own evaluation succeed on a failed graph. The error is kept here instead,
//! per thread, and the crate-root [`try_eval`] returns it before evaluating
//! anything else, so the request fails where it failed before eager state
//! evaluation existed.

use crate::ffi::{self, MlxArray};
use std::cell::RefCell;
use std::fmt;

thread_local! {
    static DEFERRED: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// An MLX evaluation failure, from the evaluation itself or deferred from an
/// eager state evaluation earlier on this thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalError(String);

impl EvalError {
    /// The MLX error message.
    pub fn what(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EvalError {}

/// Starts evaluating `arr`. A failure is kept for the next [`try_eval`] on
/// this thread instead of aborting the process; the first one wins.
///
/// Used by: [`crate::utils::materialized_axis_tail`] (and through it
/// [`crate::utils::materialized_state_tail`]),
/// [`crate::utils::schedule_prefill_state`]
pub(crate) fn async_eval_or_defer(arr: &MlxArray) {
    if let Err(err) = ffi::try_async_eval(arr) {
        DEFERRED.with(|slot| {
            slot.borrow_mut()
                .get_or_insert_with(|| err.what().to_string());
        });
    }
}

/// Takes the error a deferred evaluation left on this thread, if any.
pub fn take_deferred_eval_error() -> Option<String> {
    DEFERRED.with(|slot| slot.borrow_mut().take())
}

/// Evaluates `arr`, reporting a deferred evaluation error first.
///
/// Shadows the FFI `try_eval` at the crate root, so every
/// `mlxcel_core::try_eval` caller sees a failure an eager state evaluation hit
/// during the forward that built `arr`.
pub fn try_eval(arr: &MlxArray) -> Result<(), EvalError> {
    if let Some(msg) = take_deferred_eval_error() {
        return Err(EvalError(msg));
    }
    ffi::try_eval(arr).map_err(|e| EvalError(e.what().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defer(msg: &str) {
        DEFERRED.with(|slot| {
            slot.borrow_mut().get_or_insert_with(|| msg.to_string());
        });
    }

    #[test]
    fn try_eval_reports_a_deferred_error_once_and_keeps_the_first() {
        let _ = take_deferred_eval_error();
        defer("first failure");
        defer("second failure");
        let arr = ffi::full_f32(&[2], 1.0, crate::dtype::FLOAT32);
        assert_eq!(try_eval(&arr).unwrap_err().what(), "first failure");
        assert!(
            try_eval(&arr).is_ok(),
            "the deferred error is reported once"
        );
    }

    #[test]
    fn deferred_errors_stay_on_their_thread() {
        let _ = take_deferred_eval_error();
        std::thread::spawn(|| defer("other thread")).join().unwrap();
        assert_eq!(take_deferred_eval_error(), None);
    }
}
