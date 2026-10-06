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

//! The opt-in gate for the B9 pre-bias suppression counters.
//!
//! `LANG_BIAS_TOKENS_SUPPRESSED_TOTAL` and
//! `LANG_BIAS_BYTE_FRAGMENT_SUPPRESSIONS_TOTAL` need the pre-bias argmax id,
//! and reading it costs an `eval` plus a device-to-host read on every biased
//! decode step. That breaks the async lookahead pipeline, so the counters are
//! off unless the operator sets `MLXCEL_LANG_BIAS_COUNTERS=1`, and while they
//! are on a biased row leaves the batched fused path (and with it the decode
//! lookahead).
//!
//! The environment is read once per process. Tests that assert the counters
//! must not arm them by mutating the environment: the cached answer then
//! outlives the test and turns every later `ignore_eos` / `logit_bias` decode
//! in the same test binary synchronous (issue #2187, where it silently
//! disabled the Gemma 3 lookahead tests). They take a [`scoped_override`]
//! instead, which applies to the calling thread only and is undone on drop.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::OnceLock;

thread_local! {
    /// Per-thread override set by [`scoped_override`]; `None` defers to the
    /// process-wide environment setting.
    static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether the B9 pre-bias suppression counters are collected on this thread.
///
/// A live [`scoped_override`] on the calling thread wins; otherwise the
/// process-wide `MLXCEL_LANG_BIAS_COUNTERS` setting, read once, decides.
///
/// Used by: `sampling::apply_token_bias_stage`,
/// `sampling::row_supports_fused_batch_except_bias`
pub(crate) fn enabled() -> bool {
    if let Some(forced) = OVERRIDE.with(Cell::get) {
        return forced;
    }
    static FROM_ENV: OnceLock<bool> = OnceLock::new();
    *FROM_ENV.get_or_init(|| {
        std::env::var("MLXCEL_LANG_BIAS_COUNTERS")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

/// Restores the previous per-thread setting when dropped.
///
/// Not `Send`: the override lives in a thread-local, so the guard must be
/// dropped on the thread that created it.
#[must_use = "the override is undone as soon as the guard is dropped"]
pub struct ScopedOverride {
    previous: Option<bool>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for ScopedOverride {
    fn drop(&mut self) {
        OVERRIDE.with(|cell| cell.set(self.previous));
    }
}

/// Force the counters on or off for the calling thread until the returned
/// guard is dropped, without touching the process environment or the cached
/// process-wide setting. Guards nest; each drop restores what it replaced, so
/// drop them in reverse creation order (an out-of-order drop restores a stale
/// value).
///
/// Test support: production configures the counters through
/// `MLXCEL_LANG_BIAS_COUNTERS` only.
#[doc(hidden)]
pub fn scoped_override(enabled: bool) -> ScopedOverride {
    let previous = OVERRIDE.with(|cell| cell.replace(Some(enabled)));
    ScopedOverride {
        previous,
        _not_send: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_is_scoped_nested_and_thread_local() {
        let baseline = enabled();
        {
            let _on = scoped_override(true);
            assert!(enabled());
            {
                let _off = scoped_override(false);
                assert!(!enabled());
            }
            assert!(enabled(), "the inner drop restores the outer override");
            // Another thread never sees this thread's override.
            let other = std::thread::spawn(enabled).join().expect("thread runs");
            assert_eq!(other, baseline);
        }
        assert_eq!(enabled(), baseline, "the outer drop restores the baseline");
    }
}
