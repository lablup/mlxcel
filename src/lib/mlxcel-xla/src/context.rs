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
}
