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

//! Static context-capacity configuration and request admission for OpenXLA.

use std::fmt;

/// Compatibility capacity used when the operator does not select one.
pub const DEFAULT_CONTEXT_CAPACITY: usize = 256;

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
/// 3. Derived from the checkpoint: the text default, plus the worst-case image
///    expansion when the caller knows one. This is the case that makes an
///    image-capable checkpoint usable without the operator having to discover
///    the number themselves.
///
/// `image_floor` is `None` for a text-only checkpoint or a family whose worst
/// case is not derived, and a floor at or below the text default adds no
/// bucket, since the default already admits it.
///
/// # Errors
///
/// Returns an error for a malformed or out-of-range value in either variable.
pub fn context_capacity_buckets_from_env(image_floor: Option<usize>) -> Result<Vec<usize>, String> {
    if let Some(pinned) = read_env_capacity(CONTEXT_CAPACITY_ENV)? {
        return Ok(vec![pinned]);
    }
    if let Some(raw) = read_env_raw(CONTEXT_BUCKETS_ENV)? {
        return parse_context_buckets(&raw);
    }
    Ok(derive_context_buckets(
        DEFAULT_CONTEXT_CAPACITY,
        image_floor,
    ))
}

/// The derived bucket set, split out so the rule is testable without env.
#[must_use]
pub fn derive_context_buckets(text_default: usize, image_floor: Option<usize>) -> Vec<usize> {
    let mut buckets = vec![text_default];
    if let Some(floor) = image_floor
        && floor > text_default
    {
        buckets.push(floor);
    }
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
    fn a_derived_set_adds_an_image_bucket_only_when_it_is_larger() {
        assert_eq!(derive_context_buckets(256, None), vec![256]);
        assert_eq!(derive_context_buckets(256, Some(1834)), vec![256, 1834]);
        // A floor the text default already admits adds nothing to compile.
        assert_eq!(derive_context_buckets(256, Some(200)), vec![256]);
        assert_eq!(derive_context_buckets(256, Some(256)), vec![256]);
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
