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

//! Pinning of Claude Code's billing stamp in Anthropic `system` text.
//!
//! Claude Code starts its system prompt with a line such as
//! `x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=sdk-cli; cch=b145e;`.
//! `cch` changes on every request and the fourth `cc_version` part changes on
//! every session, so the rendered prompt would diverge from the previous turn
//! at the stamp and cap prompt-cache reuse there (issue #2884).
//!
//! [`pin_billing_stamp`] overwrites only those two values with `f` characters
//! of the same length. The rule is deliberately narrow: it is specific to this
//! one client stamp, not a generic prompt normaliser.

use std::borrow::Cow;
use std::sync::OnceLock;

const HEADER_PREFIX: &[u8] = b"x-anthropic-billing-header:";
/// The stamp must lie within this many leading bytes of the system text.
const SEARCH_WINDOW: usize = 160;
const CCH_KEY: &[u8] = b"cch=";
const CCH_MAX_LEN: usize = 16;
const CC_VERSION_KEY: &[u8] = b"cc_version=";
const CC_VERSION_MAX_LEN: usize = 64;
/// `cc_version` dots that precede the per-session part.
const CC_VERSION_STABLE_DOTS: usize = 3;
const PLACEHOLDER: u8 = b'f';

/// Replace the volatile parts of a leading Claude Code billing stamp with `f`
/// characters of the same length, keeping every other byte.
///
/// Text that does not start with the exact header prefix, and values that are
/// unterminated, over-long, or non-ASCII, are returned unchanged. Only the
/// first line within the first 160 bytes is examined.
pub fn pin_billing_stamp(system: &str) -> Cow<'_, str> {
    let bytes = system.as_bytes();
    if !bytes.starts_with(HEADER_PREFIX) {
        return Cow::Borrowed(system);
    }

    let window = &bytes[..bytes.len().min(SEARCH_WINDOW)];
    let stamp_end = window
        .iter()
        .position(|&b| b == b'\n')
        .unwrap_or(window.len());

    let mut edits: Vec<std::ops::Range<usize>> = Vec::with_capacity(2);
    if let Some(value) = field_value(bytes, stamp_end, CC_VERSION_KEY, CC_VERSION_MAX_LEN)
        && let Some(session) = cc_version_session_part(bytes, value)
    {
        edits.push(session);
    }
    if let Some(value) = field_value(bytes, stamp_end, CCH_KEY, CCH_MAX_LEN) {
        edits.push(value);
    }

    let needs_edit = edits
        .iter()
        .any(|range| bytes[range.clone()].iter().any(|&b| b != PLACEHOLDER));
    if !needs_edit {
        return Cow::Borrowed(system);
    }

    let mut pinned = system.to_owned();
    for range in edits {
        let placeholder = "f".repeat(range.len());
        // Ranges hold only ASCII bytes, so they sit on char boundaries.
        pinned.replace_range(range, &placeholder);
    }
    Cow::Owned(pinned)
}

/// Byte range of the value of `key` within `bytes[HEADER_PREFIX.len()..end]`.
///
/// The key must start a field (right after the header colon, a space, or a
/// `;`), and its value must be ASCII, 1 to `max_len` bytes long, and closed by
/// a `;` inside the searched region.
fn field_value(
    bytes: &[u8],
    end: usize,
    key: &[u8],
    max_len: usize,
) -> Option<std::ops::Range<usize>> {
    let start = HEADER_PREFIX.len();
    let region = bytes.get(start..end)?;
    let offset = region.windows(key.len()).enumerate().find_map(|(i, w)| {
        let at_boundary = i == 0 || matches!(region[i - 1], b' ' | b';');
        (w == key && at_boundary).then_some(i)
    })?;
    let value_start = start + offset + key.len();
    let value_len = bytes[value_start..end].iter().position(|&b| b == b';')?;
    let value = value_start..value_start + value_len;
    ((1..=max_len).contains(&value_len) && bytes[value.clone()].is_ascii()).then_some(value)
}

/// Sub-range of a `cc_version` value after its third dot, if non-empty.
fn cc_version_session_part(
    bytes: &[u8],
    value: std::ops::Range<usize>,
) -> Option<std::ops::Range<usize>> {
    let (third_dot, _) = bytes[value.clone()]
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b == b'.')
        .nth(CC_VERSION_STABLE_DOTS - 1)?;
    let session_start = value.start + third_dot + 1;
    (session_start < value.end).then_some(session_start..value.end)
}

/// `MLXCEL_ANTHROPIC_PIN_BILLING_STAMP=0` passes the stamp through unchanged.
/// Exact match only; read once per process.
pub(crate) fn billing_stamp_pinning_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        pinning_enabled_from(
            std::env::var("MLXCEL_ANTHROPIC_PIN_BILLING_STAMP")
                .ok()
                .as_deref(),
        )
    })
}

fn pinning_enabled_from(value: Option<&str>) -> bool {
    value != Some("0")
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAMP: &str =
        "x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=sdk-cli; cch=b145e;";
    const PINNED: &str =
        "x-anthropic-billing-header: cc_version=2.1.170.fff; cc_entrypoint=sdk-cli; cch=fffff;";

    fn assert_unchanged(text: &str) {
        assert!(
            matches!(pin_billing_stamp(text), Cow::Borrowed(_)),
            "expected no change for {text:?}"
        );
    }

    #[test]
    fn pins_example_header_and_keeps_the_rest() {
        let text = format!("{STAMP}\nYou are a helpful agent.\n");
        assert_eq!(
            pin_billing_stamp(&text),
            format!("{PINNED}\nYou are a helpful agent.\n")
        );
    }

    #[test]
    fn inputs_differing_only_in_volatile_parts_converge() {
        let a = format!("{STAMP}\nrest");
        let other_cch = a.replace("cch=b145e", "cch=09a7c");
        let other_session = a.replace("170.bf4", "170.e21");
        assert_eq!(pin_billing_stamp(&a), pin_billing_stamp(&other_cch));
        assert_eq!(pin_billing_stamp(&a), pin_billing_stamp(&other_session));
    }

    #[test]
    fn already_pinned_text_is_borrowed() {
        assert_unchanged(PINNED);
    }

    #[test]
    fn preserves_value_lengths() {
        let text = "x-anthropic-billing-header: cc_version=2.1.170.abcdefg; cch=0123456789abcdef;";
        assert_eq!(
            pin_billing_stamp(text),
            "x-anthropic-billing-header: cc_version=2.1.170.fffffff; cch=ffffffffffffffff;"
        );
    }

    #[test]
    fn cc_version_parts_beyond_the_fourth_are_pinned_too() {
        assert_eq!(
            pin_billing_stamp("x-anthropic-billing-header: cc_version=2.1.170.ab.c1;"),
            "x-anthropic-billing-header: cc_version=2.1.170.fffff;"
        );
    }

    #[test]
    fn missing_exact_prefix_is_unchanged() {
        assert_unchanged("You are helpful. cc_version=2.1.170.bf4; cch=b145e;");
        assert_unchanged("X-Anthropic-Billing-Header: cc_version=2.1.170.bf4; cch=b145e;");
        assert_unchanged(&format!(" {STAMP}"));
    }

    #[test]
    fn fields_after_the_search_window_are_unchanged() {
        let padding = "a".repeat(160);
        assert_unchanged(&format!(
            "x-anthropic-billing-header: note={padding}; cch=b145e;"
        ));
    }

    #[test]
    fn value_closed_outside_the_window_is_unchanged() {
        let header = "x-anthropic-billing-header: ";
        let padding = "a".repeat(SEARCH_WINDOW - header.len() - "note=; cch=b14".len());
        assert_unchanged(&format!("{header}note={padding}; cch=b145e;"));
    }

    #[test]
    fn fields_on_later_lines_are_unchanged() {
        assert_unchanged("x-anthropic-billing-header: cc_entrypoint=sdk-cli;\ncch=b145e;");
    }

    #[test]
    fn over_long_cch_is_unchanged() {
        assert_unchanged("x-anthropic-billing-header: cch=0123456789abcdef0;");
    }

    #[test]
    fn empty_cch_is_unchanged() {
        assert_unchanged("x-anthropic-billing-header: cch=;");
    }

    #[test]
    fn over_long_cc_version_is_unchanged() {
        let long = "1".repeat(CC_VERSION_MAX_LEN);
        assert_unchanged(&format!(
            "x-anthropic-billing-header: cc_version=2.1.{long}.a;"
        ));
    }

    #[test]
    fn cc_version_with_three_or_fewer_parts_is_unchanged() {
        assert_unchanged("x-anthropic-billing-header: cc_version=2.1.170;");
        assert_unchanged("x-anthropic-billing-header: cc_version=2.1;");
    }

    #[test]
    fn missing_trailing_semicolon_is_unchanged() {
        assert_unchanged("x-anthropic-billing-header: cch=b145e");
        assert_unchanged("x-anthropic-billing-header: cc_version=2.1.170.bf4");
        assert_unchanged("x-anthropic-billing-header: cch=b145e\nnext; line");
    }

    #[test]
    fn keys_inside_other_field_names_are_ignored() {
        assert_unchanged("x-anthropic-billing-header: xcch=b145e; old_cc_version=2.1.170.bf4;");
    }

    #[test]
    fn non_ascii_text_after_the_header_is_untouched() {
        let text = format!("{STAMP}\n한국어 시스템 프롬프트 cch=abc;");
        assert_eq!(
            pin_billing_stamp(&text),
            format!("{PINNED}\n한국어 시스템 프롬프트 cch=abc;")
        );
    }

    #[test]
    fn non_ascii_value_and_multibyte_at_window_edge_are_safe() {
        assert_unchanged("x-anthropic-billing-header: cch=한글;");
        // A multi-byte character straddles byte 160; no field is in reach.
        let header = "x-anthropic-billing-header: ";
        let padding = "a".repeat(SEARCH_WINDOW - header.len() - 1);
        assert_unchanged(&format!("{header}{padding}한 cch=b145e;"));
    }

    #[test]
    fn kill_switch_only_recognizes_zero() {
        assert!(pinning_enabled_from(None));
        assert!(pinning_enabled_from(Some("1")));
        assert!(pinning_enabled_from(Some("false")));
        assert!(!pinning_enabled_from(Some("0")));
    }
}
