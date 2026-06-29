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

//! Shared `-n/--max-tokens` resolution for the CLI generation paths.
//!
//! llama.cpp's default `n_predict = -1` means "generate until EOS or until the
//! model context window (`n_ctx`) is full". mlxcel mirrors that: the
//! `-n/--max-tokens` flag (and the server `--n-predict`) defaults to `-1`,
//! which [`parse_max_tokens`] folds into the [`UNLIMITED_MAX_TOKENS`] sentinel.
//! The generation paths then resolve the sentinel against the loaded model's
//! context window via [`resolve_unlimited_max_tokens`]. An explicit
//! non-negative `-n N` is always kept verbatim.
//!
//! The context window itself is read from the checkpoint's `config.json` by
//! [`crate::read_model_context_window`]; when that is unavailable the caller
//! falls back to [`DEFAULT_CONTEXT_WINDOW_FALLBACK`] so the budget stays
//! bounded.

/// Sentinel stored in `max_tokens` fields meaning "no explicit cap; resolve to
/// the model context window at generation time".
///
/// Chosen as `usize::MAX` so an unresolved sentinel that ever leaked into an
/// allocation or loop bound would fail loudly rather than silently behave as a
/// plausible token budget.
pub const UNLIMITED_MAX_TOKENS: usize = usize::MAX;

/// Context window used when a checkpoint's `config.json` exposes no context
/// length. Matches the historical `mlxcel serve` default so behavior stays
/// bounded and predictable for checkpoints that omit `max_position_embeddings`.
pub const DEFAULT_CONTEXT_WINDOW_FALLBACK: usize = 4096;

/// clap `value_parser` for the `-n/--max-tokens` (and server `--n-predict`)
/// flag.
///
/// Accepts any integer. A negative value (canonically `-1`, llama.cpp's
/// "unlimited" spelling) maps to [`UNLIMITED_MAX_TOKENS`]; a non-negative value
/// is taken verbatim as the token budget.
pub fn parse_max_tokens(s: &str) -> Result<usize, String> {
    let raw: i64 = s.trim().parse().map_err(|_| {
        format!("invalid max-tokens value '{s}': expected an integer (-1 = unlimited)")
    })?;
    if raw < 0 {
        Ok(UNLIMITED_MAX_TOKENS)
    } else {
        Ok(raw as usize)
    }
}

/// Resolve a (possibly sentinel) token budget against a model context window.
///
/// * An explicit budget (anything other than [`UNLIMITED_MAX_TOKENS`]) is
///   returned unchanged, so a user who passed `-n N` always gets exactly `N`.
/// * The sentinel resolves to `context_window - prompt_len`, clamped to at
///   least 1 so a prompt that already fills the window still emits a token.
///
/// Callers must pass a non-zero `context_window` (use
/// [`DEFAULT_CONTEXT_WINDOW_FALLBACK`] when the model exposes none); a zero
/// window would otherwise collapse the unlimited budget to a single token.
pub fn resolve_unlimited_max_tokens(
    requested: usize,
    context_window: usize,
    prompt_len: usize,
) -> usize {
    if requested != UNLIMITED_MAX_TOKENS {
        return requested;
    }
    context_window.saturating_sub(prompt_len).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_negative_is_unlimited_sentinel() {
        assert_eq!(parse_max_tokens("-1"), Ok(UNLIMITED_MAX_TOKENS));
        assert_eq!(parse_max_tokens("-42"), Ok(UNLIMITED_MAX_TOKENS));
    }

    #[test]
    fn parse_non_negative_is_verbatim() {
        assert_eq!(parse_max_tokens("0"), Ok(0));
        assert_eq!(parse_max_tokens("256"), Ok(256));
        assert_eq!(parse_max_tokens(" 128 "), Ok(128));
    }

    #[test]
    fn parse_rejects_non_integer() {
        assert!(parse_max_tokens("abc").is_err());
        assert!(parse_max_tokens("1.5").is_err());
        assert!(parse_max_tokens("").is_err());
    }

    #[test]
    fn explicit_budget_ignores_context_window() {
        // A concrete `-n 50` is honored verbatim regardless of window/prompt.
        assert_eq!(resolve_unlimited_max_tokens(50, 4096, 100), 50);
        assert_eq!(resolve_unlimited_max_tokens(0, 4096, 100), 0);
    }

    #[test]
    fn unlimited_resolves_to_window_minus_prompt() {
        assert_eq!(
            resolve_unlimited_max_tokens(UNLIMITED_MAX_TOKENS, 4096, 96),
            4000
        );
    }

    #[test]
    fn unlimited_clamps_to_at_least_one() {
        // Prompt at or beyond the window still yields one token, never zero.
        assert_eq!(
            resolve_unlimited_max_tokens(UNLIMITED_MAX_TOKENS, 4096, 4096),
            1
        );
        assert_eq!(
            resolve_unlimited_max_tokens(UNLIMITED_MAX_TOKENS, 4096, 9000),
            1
        );
    }
}
