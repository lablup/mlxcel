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

//! The tokens of the prefill piece that follows the one being forwarded.
//!
//! A model whose forward reads something slow per token, such as Qwen4-Exp's
//! on-disk n-gram table, can start that read for the *next* piece while the
//! GPU runs the current one. The piece boundaries live in the prefill driver,
//! so the driver announces the upcoming tokens here and the model takes them.
//!
//! Like [`crate::prefill_span`] this is thread-local and scoped to the guard:
//! hold it around one piece's forward only, because the server interleaves
//! other sequences' decode ticks between two pieces of a prompt. The hint is
//! advisory. A model that ignores it, or one that sees another sequence's hint,
//! produces the same output.

use std::cell::RefCell;

thread_local! {
    static UPCOMING: RefCell<Option<Vec<i32>>> = const { RefCell::new(None) };
}

/// Live announcement of the next prefill piece's tokens.
#[derive(Debug)]
pub struct PrefillLookahead {
    previous: Option<Vec<i32>>,
}

impl Drop for PrefillLookahead {
    fn drop(&mut self) {
        let previous = self.previous.take();
        UPCOMING.with(|slot| *slot.borrow_mut() = previous);
    }
}

/// Announce `upcoming` as the piece after the one forwarded under this guard.
/// An empty slice (the terminal piece) announces nothing.
#[must_use = "the announcement ends as soon as the guard is dropped"]
pub fn announce(upcoming: &[i32]) -> PrefillLookahead {
    let value = (!upcoming.is_empty()).then(|| upcoming.to_vec());
    let previous = UPCOMING.with(|slot| slot.replace(value));
    PrefillLookahead { previous }
}

/// Take the announced tokens; a second call before the next announcement
/// returns `None`, so one forward consumes one hint.
#[must_use]
pub fn take() -> Option<Vec<i32>> {
    UPCOMING.with(|slot| slot.borrow_mut().take())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hint_is_consumed_once_and_scoped_to_its_guard() {
        assert_eq!(take(), None);
        {
            let _guard = announce(&[1, 2]);
            assert_eq!(take(), Some(vec![1, 2]));
            assert_eq!(take(), None);
        }
        assert_eq!(take(), None);
    }

    #[test]
    fn an_empty_piece_announces_nothing() {
        let _guard = announce(&[]);
        assert_eq!(take(), None);
    }

    #[test]
    fn nesting_restores_the_outer_hint() {
        let _outer = announce(&[1]);
        {
            let _inner = announce(&[2]);
            assert_eq!(take(), Some(vec![2]));
        }
        assert_eq!(take(), Some(vec![1]));
    }
}
