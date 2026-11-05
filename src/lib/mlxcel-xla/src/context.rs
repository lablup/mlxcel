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

//! Static context-capacity configuration and request admission for OpenXLA.

use std::fmt;

/// Smallest static shape the request-sizing ladder will compile.
///
/// Used when the operator pins nothing and the caller has no request to size
/// against (the legacy [`context_capacity_from_env`] path), and kept as the
/// floor of [`resolve_context_capacity`] so a short request still takes the
/// cheapest shape.
pub const DEFAULT_CONTEXT_CAPACITY: usize = 256;

/// The static shapes a request-sized session may compile, ascending.
///
/// The ladder bounds the padding a request pays for: the resolver picks the
/// smallest rung at or above the required positions. The top rung is far below
/// the IREE C ABI's signed 32-bit position limit, and a capacity is also the
/// sequence length every decode step attends over, so the ladder deliberately
/// stops well short of the largest checkpoint windows.
pub const CONTEXT_CAPACITY_LADDER: [usize; 8] = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

/// The larger text shape the serving bucket set derives by default.
///
/// A text-only checkpoint used to derive `[256]`, so a prompt that fit any
/// other backend was refused at admission. Serving compiles this second shape
/// next to [`DEFAULT_CONTEXT_CAPACITY`] and routes by request size, so short
/// requests keep the cheap shape and ordinary prompts still run.
pub const SERVER_TEXT_CONTEXT_CAPACITY: usize = 4096;

/// Generation budget the CLI sizes an OpenXLA static shape against when `-n` is
/// the unlimited default (issue #2855).
///
/// The unlimited sentinel resolves to the whole checkpoint window, which no
/// static graph can compile. Sizing with a bounded budget keeps the default
/// invocation cheap while still leaving room for a normal completion; an
/// explicit `-n N` sizes with `N` instead. Generation still stops at EOS, and
/// the admitted budget is clamped to the compiled shape's room.
pub const DEFAULT_XLA_GENERATION_BUDGET: usize = 512;

/// Environment variable selecting the static StableHLO context shape.
pub const CONTEXT_CAPACITY_ENV: &str = "MLXCEL_XLA_CONTEXT_CAPACITY";

/// A request whose effective prompt plus generation budget cannot fit the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCapacityError {
    pub effective_prompt_len: usize,
    pub max_new_tokens: usize,
    pub context_capacity: usize,
}

impl fmt::Display for ContextCapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "request exceeds the OpenXLA context capacity: effective_prompt_len={} + max_new_tokens={} > context_capacity={}",
            self.effective_prompt_len, self.max_new_tokens, self.context_capacity
        )
    }
}

impl std::error::Error for ContextCapacityError {}

/// Resolve the static graph capacity selected by the operator.
///
/// The value must fit the IREE C ABI's signed 32-bit position/count arguments.
/// An unset variable keeps the historical 256-token graph shape.
pub fn context_capacity_from_env() -> Result<usize, String> {
    match std::env::var(CONTEXT_CAPACITY_ENV) {
        Ok(raw) => parse_context_capacity(&raw),
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_CONTEXT_CAPACITY),
        Err(err) => Err(format!("read {CONTEXT_CAPACITY_ENV}: {err}")),
    }
}

/// Resolve the static graph capacity for a single request.
///
/// The precedence is:
///
/// 1. [`CONTEXT_CAPACITY_ENV`], which still pins exactly one shape and ignores
///    the request, as it always has.
/// 2. No request hint: [`DEFAULT_CONTEXT_CAPACITY`], the historical shape.
/// 3. Otherwise the smallest [`CONTEXT_CAPACITY_LADDER`] rung that admits
///    `required_positions` (prompt plus generation budget), bounded by the
///    checkpoint window and by the top rung. The window check runs first, so a
///    returned capacity always admits the request, and a window below the
///    ladder floor yields the window itself.
///
/// # Errors
///
/// Returns an error for a malformed [`CONTEXT_CAPACITY_ENV`] value, and for a
/// request that needs more positions than the checkpoint window or the ladder
/// can provide. Nothing is silently truncated or clamped.
pub fn resolve_context_capacity(
    required_positions: Option<usize>,
    model_window: Option<usize>,
) -> Result<usize, String> {
    let pinned = read_env_capacity(CONTEXT_CAPACITY_ENV)?;
    resolve_context_capacity_with(required_positions, model_window, pinned)
}

/// The pure form of [`resolve_context_capacity`], split out so the rule is
/// testable without touching the process environment.
pub(crate) fn resolve_context_capacity_with(
    required_positions: Option<usize>,
    model_window: Option<usize>,
    pinned: Option<usize>,
) -> Result<usize, String> {
    if let Some(pinned) = pinned {
        return Ok(pinned);
    }
    let Some(required) = required_positions else {
        return Ok(DEFAULT_CONTEXT_CAPACITY);
    };
    let ladder_top = *CONTEXT_CAPACITY_LADDER
        .last()
        .unwrap_or(&DEFAULT_CONTEXT_CAPACITY);
    let ceiling = model_window.unwrap_or(ladder_top).min(ladder_top);
    if required > ceiling {
        return Err(match model_window {
            Some(window) => format!(
                "this request needs {required} positions but the checkpoint context window is \
                 {window}; shorten the prompt or lower -n"
            ),
            None => format!(
                "this request needs {required} positions but the OpenXLA context capacity ladder \
                 tops out at {ladder_top}; shorten the prompt or lower -n"
            ),
        });
    }
    // Never compile below the floor, but never above the window either.
    let target = required.max(DEFAULT_CONTEXT_CAPACITY).min(ceiling);
    let rung = CONTEXT_CAPACITY_LADDER
        .iter()
        .copied()
        .find(|&capacity| capacity >= target)
        .unwrap_or(target);
    Ok(rung.min(ceiling))
}

/// Validate a capacity supplied through an API rather than the environment.
pub(crate) fn validate_context_capacity_value(context_capacity: usize) -> Result<usize, String> {
    if context_capacity == 0 {
        return Err("OpenXLA context capacity must be at least 1 token".to_string());
    }
    if context_capacity > i32::MAX as usize {
        return Err(format!(
            "OpenXLA context capacity {context_capacity} exceeds the IREE ABI maximum {}",
            i32::MAX
        ));
    }
    Ok(context_capacity)
}

fn parse_context_capacity(raw: &str) -> Result<usize, String> {
    let value = raw.parse::<usize>().map_err(|_| {
        format!(
            "{CONTEXT_CAPACITY_ENV} must be an integer in 1..={}, got {raw:?}",
            i32::MAX
        )
    })?;
    validate_context_capacity_value(value)
}

/// Environment variable selecting an explicit list of static context shapes.
pub const CONTEXT_BUCKETS_ENV: &str = "MLXCEL_XLA_CONTEXT_BUCKETS";

/// Resolve the set of static context shapes to compile.
///
/// Three sources, in precedence order, matching the three ways an operator can
/// reasonably want this decided:
///
/// 1. [`CONTEXT_CAPACITY_ENV`], which yields exactly one bucket. Pinning a
///    single capacity is an existing operator-facing decision and must keep
///    producing exactly the engine it produced before buckets existed, so it
///    wins over everything else rather than becoming a floor or a hint.
/// 2. [`CONTEXT_BUCKETS_ENV`], a comma-separated list, for an operator who
///    knows their traffic better than any derivation can.
/// 3. Derived from the defaults: the cheap text floor, a larger serving text
///    shape bounded by the checkpoint window, plus the worst-case image
///    expansion when the caller knows one. This is the case that makes a
///    text-only checkpoint serve ordinary prompts and an image-capable
///    checkpoint usable without the operator having to discover the numbers
///    themselves.
///
/// `image_floor` is `None` for a text-only checkpoint or a family whose worst
/// case is not derived, and a floor at or below the text floor adds no bucket,
/// since the floor already admits it.
///
/// # Errors
///
/// Returns an error for a malformed or out-of-range value in either variable.
pub fn context_capacity_buckets_from_env(
    image_floor: Option<usize>,
    model_window: Option<usize>,
) -> Result<Vec<usize>, String> {
    if let Some(pinned) = read_env_capacity(CONTEXT_CAPACITY_ENV)? {
        return Ok(vec![pinned]);
    }
    if let Some(raw) = read_env_raw(CONTEXT_BUCKETS_ENV)? {
        return parse_context_buckets(&raw);
    }
    Ok(derive_context_buckets(
        DEFAULT_CONTEXT_CAPACITY,
        model_window,
        image_floor,
    ))
}

/// The derived bucket set, split out so the rule is testable without env.
///
/// Always contains the cheap text floor and a larger serving text shape,
/// bounded by the checkpoint window, plus the image floor when it is larger
/// than the text floor. Serving then routes each request to the smallest bucket
/// that admits it, so short text requests keep the floor shape.
#[must_use]
pub fn derive_context_buckets(
    text_default: usize,
    model_window: Option<usize>,
    image_floor: Option<usize>,
) -> Vec<usize> {
    // Never compile a shape wider than the checkpoint can use; when the window
    // is below the floor, the window is the only shape left.
    let ceiling = match model_window {
        Some(window) if window < text_default => window,
        Some(window) => SERVER_TEXT_CONTEXT_CAPACITY.min(window).max(text_default),
        None => SERVER_TEXT_CONTEXT_CAPACITY.max(text_default),
    };
    let mut buckets = vec![text_default.min(ceiling), ceiling];
    if let Some(floor) = image_floor
        && floor > text_default
    {
        buckets.push(floor);
    }
    buckets.sort_unstable();
    buckets.dedup();
    buckets
}

fn read_env_raw(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(raw) => Ok(Some(raw)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("read {name}: {err}")),
    }
}

fn read_env_capacity(name: &str) -> Result<Option<usize>, String> {
    match read_env_raw(name)? {
        Some(raw) => parse_context_capacity(&raw).map(Some),
        None => Ok(None),
    }
}

/// Parse a comma-separated bucket list, sorted ascending and deduplicated.
fn parse_context_buckets(raw: &str) -> Result<Vec<usize>, String> {
    let mut buckets = Vec::new();
    for field in raw.split(',') {
        let field = field.trim();
        if field.is_empty() {
            continue;
        }
        buckets.push(parse_context_capacity(field)?);
    }
    if buckets.is_empty() {
        return Err(format!(
            "{CONTEXT_BUCKETS_ENV} must list at least one capacity, got {raw:?}"
        ));
    }
    buckets.sort_unstable();
    buckets.dedup();
    Ok(buckets)
}

/// Enforce the common text/multimodal admission invariant.
///
/// `effective_prompt_len` is the token count after any placeholder expansion.
/// The checked addition also rejects an overflowing generation budget.
pub fn validate_request_capacity(
    effective_prompt_len: usize,
    max_new_tokens: usize,
    context_capacity: usize,
) -> Result<(), ContextCapacityError> {
    if context_capacity > 0
        && effective_prompt_len
            .checked_add(max_new_tokens)
            .is_some_and(|needed| needed <= context_capacity)
    {
        Ok(())
    } else {
        Err(ContextCapacityError {
            effective_prompt_len,
            max_new_tokens,
            context_capacity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_fit_is_admitted() {
        assert_eq!(validate_request_capacity(768, 256, 1024), Ok(()));
    }

    #[test]
    fn one_token_overflow_reports_all_values() {
        let err = validate_request_capacity(769, 256, 1024).unwrap_err();
        assert_eq!(err.effective_prompt_len, 769);
        assert_eq!(err.max_new_tokens, 256);
        assert_eq!(err.context_capacity, 1024);
        assert_eq!(
            err.to_string(),
            "request exceeds the OpenXLA context capacity: effective_prompt_len=769 + max_new_tokens=256 > context_capacity=1024"
        );
    }

    #[test]
    fn expanded_multimodal_length_uses_the_same_invariant() {
        assert!(validate_request_capacity(729 + 32, 128, 1024).is_ok());
        assert!(validate_request_capacity(729 + 200, 128, 1024).is_err());
    }

    #[test]
    fn overflowing_generation_budget_is_rejected() {
        let err = validate_request_capacity(1, usize::MAX, usize::MAX).unwrap_err();
        assert_eq!(err.max_new_tokens, usize::MAX);
    }

    #[test]
    fn zero_capacity_is_never_admitted() {
        assert!(validate_request_capacity(0, 0, 0).is_err());
    }

    #[test]
    fn a_derived_set_has_a_text_ceiling_and_an_image_bucket_when_larger() {
        let wide = Some(131_072);
        assert_eq!(derive_context_buckets(256, wide, None), vec![256, 4096]);
        assert_eq!(
            derive_context_buckets(256, wide, Some(1834)),
            vec![256, 1834, 4096]
        );
        // A floor the text floor already admits adds nothing to compile.
        assert_eq!(
            derive_context_buckets(256, wide, Some(200)),
            vec![256, 4096]
        );
        assert_eq!(
            derive_context_buckets(256, wide, Some(256)),
            vec![256, 4096]
        );
        // A floor above the text ceiling is still served with its own shape.
        assert_eq!(
            derive_context_buckets(256, wide, Some(8192)),
            vec![256, 4096, 8192]
        );
    }

    #[test]
    fn a_derived_set_never_exceeds_the_checkpoint_window() {
        assert_eq!(
            derive_context_buckets(256, Some(2048), None),
            vec![256, 2048]
        );
        // A window below the floor leaves only the window.
        assert_eq!(derive_context_buckets(256, Some(100), None), vec![100]);
    }

    #[test]
    fn the_resolver_takes_the_next_rung_and_keeps_the_floor() {
        let window = Some(131_072);
        assert_eq!(
            resolve_context_capacity_with(Some(1), window, None),
            Ok(256)
        );
        assert_eq!(
            resolve_context_capacity_with(Some(256), window, None),
            Ok(256)
        );
        assert_eq!(
            resolve_context_capacity_with(Some(257), window, None),
            Ok(512)
        );
        assert_eq!(
            resolve_context_capacity_with(Some(554), window, None),
            Ok(1024)
        );
        assert_eq!(
            resolve_context_capacity_with(Some(32_768), window, None),
            Ok(32_768)
        );
    }

    #[test]
    fn the_resolver_never_exceeds_the_model_window() {
        // A window below the ladder floor yields the window itself, but only
        // when the request fits it.
        assert_eq!(
            resolve_context_capacity_with(Some(50), Some(100), None),
            Ok(100)
        );
        let err = resolve_context_capacity_with(Some(400), Some(300), None).unwrap_err();
        assert!(err.contains("needs 400 positions"), "{err}");
        assert!(err.contains("window is 300"), "{err}");
    }

    #[test]
    fn the_resolver_rejects_requests_past_the_ladder_top() {
        let err = resolve_context_capacity_with(Some(40_000), None, None).unwrap_err();
        assert!(err.contains("needs 40000 positions"), "{err}");
        assert!(err.contains("tops out at 32768"), "{err}");
    }

    #[test]
    fn a_pinned_capacity_wins_over_the_request() {
        assert_eq!(
            resolve_context_capacity_with(Some(600), Some(131_072), Some(2_048)),
            Ok(2_048)
        );
        assert_eq!(
            resolve_context_capacity_with(None, Some(131_072), Some(2_048)),
            Ok(2_048)
        );
        assert_eq!(resolve_context_capacity_with(None, None, None), Ok(256));
    }

    #[test]
    fn a_bucket_list_is_sorted_deduplicated_and_validated() {
        assert_eq!(
            parse_context_buckets("2048, 256,1024"),
            Ok(vec![256, 1024, 2048])
        );
        assert_eq!(parse_context_buckets("512,512"), Ok(vec![512]));
        assert_eq!(parse_context_buckets(" 256 , "), Ok(vec![256]));
        assert!(parse_context_buckets("").is_err());
        assert!(parse_context_buckets("0").is_err());
        assert!(parse_context_buckets("256,many").is_err());
    }

    #[test]
    fn routing_by_whole_budget_is_what_makes_migration_unnecessary() {
        // A prompt that fits the small bucket but whose generation budget does
        // not must not be admitted there, or it would outgrow its graph shape
        // mid-generation and need its KV moved to a larger one.
        let small = 256;
        let large = 2048;
        assert!(validate_request_capacity(200, 8, small).is_ok());
        assert!(validate_request_capacity(200, 512, small).is_err());
        assert!(validate_request_capacity(200, 512, large).is_ok());
    }
}
