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

//! Where a snapshot-only family's completion snapshot ends (issue #3040).

use super::policy::parse_switch;

/// Opt-out switch for the prompt-boundary completion snapshot (#3040).
///
/// A snapshot-only family used to store its completion snapshot at the end of
/// generation, keyed by `prompt + generated`. The generated part of that state
/// was written one token at a time by decode, which does not write the bytes
/// a prefill of the same tokens writes (the kernels depend on the row count),
/// so restoring it on the next turn reproduced no cold run of that turn: on
/// Qwen3.8-Flash-Next with live replies 13 of 16 `--verify` requests differed
/// from their cold runs. The snapshot is now taken when the prefill finishes,
/// keyed by the prompt alone, and the next turn prefills the reply together
/// with its own new text.
///
/// `MLXCEL_PROMPT_CACHE_DECODED_TAIL`: `1`, `on` or `true` restores the
/// end-of-generation snapshot with its decode-written tail, which saves
/// re-prefilling the reply at the cost of that divergence; anything else,
/// including unset, keeps the prompt boundary. Read once per process.
///
/// Used by: `server::batch::scheduler::BatchScheduler::insert_prompt_end_snapshot`,
/// `server::batch::scheduler::BatchScheduler::donate_finished_sequence_cache`
pub fn decoded_tail_snapshot_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_switch(
            std::env::var("MLXCEL_PROMPT_CACHE_DECODED_TAIL")
                .ok()
                .as_deref(),
        )
    })
}

#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Pin [`decoded_tail_snapshot_enabled`] for the current test thread until the
/// guard drops, without touching the process environment.
#[cfg(test)]
pub(crate) fn override_decoded_tail_for_test(value: bool) -> impl Drop {
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let previous = TEST_OVERRIDE.with(|cell| cell.replace(Some(value)));
    Reset(previous)
}
