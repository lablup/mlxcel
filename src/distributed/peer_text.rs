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

//! Free text received from a cluster peer, made safe for this process's
//! log (#2782 item 2).
//!
//! A stage's `Error { message }`, a coordinator's `Abort { detail }` and an
//! upstream's request id are written by another process that holds the
//! cluster secret. Logged raw, a newline or an ANSI escape in them forges a
//! log line of this process. [`loggable_peer_text`] bounds the text and
//! writes control characters as escape sequences, so one frame stays one
//! log line of bounded length.
//!
//! Used by: `pipeline::runtime` (stage error replies), `pipeline::
//! remote_service` (upstream request ids), `stage_assignment` (abort
//! details, where the same bound also cuts the detail the coordinator
//! sends).

/// Longest peer text kept (bytes); a longer one is cut at a character
/// boundary before escaping.
pub(crate) const MAX_PEER_TEXT_BYTES: usize = 1024;

/// `text` cut to at most [`MAX_PEER_TEXT_BYTES`] at a character boundary.
pub(crate) fn truncate_peer_text(text: &str) -> &str {
    if text.len() <= MAX_PEER_TEXT_BYTES {
        return text;
    }
    let mut end = MAX_PEER_TEXT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// `text` made safe to log: bounded by [`truncate_peer_text`], then with
/// control characters (newlines, carriage returns, escapes) and
/// non-printable characters written as escape sequences. The result may be
/// longer than the bound (an escaped newline is two bytes), but it is one
/// line.
pub(crate) fn loggable_peer_text(text: &str) -> String {
    truncate_peer_text(text).escape_debug().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_characters_are_escaped() {
        let escaped = loggable_peer_text("boom\nINFO forged\r\u{1b}[31m\ttab");
        assert!(!escaped.contains('\n'));
        assert!(!escaped.contains('\r'));
        assert!(!escaped.contains('\u{1b}'));
        assert!(!escaped.contains('\t'));
        assert_eq!(escaped, "boom\\nINFO forged\\r\\u{1b}[31m\\ttab");
    }

    #[test]
    fn plain_text_is_unchanged() {
        assert_eq!(loggable_peer_text("pp-seq-7-00ff"), "pp-seq-7-00ff");
        assert_eq!(truncate_peer_text("short"), "short");
    }

    #[test]
    fn oversize_text_is_cut_at_a_character_boundary() {
        // Two-byte characters: the bound falls in the middle of one.
        let long = "\u{e9}".repeat(MAX_PEER_TEXT_BYTES);
        let cut = truncate_peer_text(&long);
        assert!(cut.len() <= MAX_PEER_TEXT_BYTES);
        assert!(cut.len() >= MAX_PEER_TEXT_BYTES - 1);
        assert!(cut.chars().all(|c| c == '\u{e9}'));
        assert_eq!(loggable_peer_text(&long), cut);

        // Escaping happens after the cut, so the escaped form may be longer
        // than the bound but holds exactly the bounded input.
        let newlines = "\n".repeat(MAX_PEER_TEXT_BYTES * 2);
        assert_eq!(loggable_peer_text(&newlines).len(), MAX_PEER_TEXT_BYTES * 2);
    }
}
