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

//! CLI handler for `mlxcel inspect`.
//!
//! Read-only entry point that surfaces the unified memory estimator
//! (issue #56, epic #52 capstone). Prints the byte breakdown for
//! weights / KV cache / runtime activation headroom / total vs
//! available unified memory, then exits without loading the model.
//!
//! Used by: operators sizing a model for a given host before
//! launching `mlxcel generate` or `mlxcel serve`.

use anyhow::{Result, anyhow};

use mlxcel::cli::turbo_args::resolve_kv_cache_mode;
use mlxcel::memory_estimate::{QuantHint, estimate_total_memory, format_estimate};
use mlxcel_core::cache::KVCacheMode;

use crate::InspectArgs;

/// Run the `mlxcel inspect` subcommand.
pub(crate) fn run_inspect(args: InspectArgs) -> Result<()> {
    if !args.model.exists() {
        return Err(anyhow!(
            "Model directory does not exist: {}",
            args.model.display()
        ));
    }

    // Translate the user-facing `--quant` label into the typed hint.
    let quant = parse_quant_hint(&args.quant)?;

    let kv_cache_mode = resolve_kv_cache_mode(
        args.turbo.cache_type_k.as_deref(),
        args.turbo.cache_type_v.as_deref(),
        args.turbo.kv_cache_mode.as_deref(),
    )
    .map_err(|e| anyhow!("{}", e))?;
    let kv_int8 = matches!(kv_cache_mode, KVCacheMode::Int8);

    let estimate = estimate_total_memory(&args.model, args.max_tokens, args.batch, quant, kv_int8);

    let banner = format_estimate(&args.model, &estimate);
    println!("{banner}");

    if !estimate.fits {
        // Exit successfully — `inspect` is read-only and informational.
        // The caller can pipe this to a script that checks for the
        // "DOES NOT FIT" marker. Returning Err here would conflate
        // "inspect ran successfully and reported over-capacity" with
        // "inspect itself failed".
        println!(
            "Note: this configuration is expected to fail the `--estimate-memory` \
             preflight on `mlxcel generate` / `mlxcel serve` unless `--force` is set."
        );
    }

    Ok(())
}

/// Parse the user-facing `--quant` label into a typed [`QuantHint`].
///
/// Accepts: `default`, `fp16`, `int8`, `int4`. Returns a clear
/// `anyhow::Error` for unknown labels so the CLI fails fast with a
/// usable error rather than silently coercing to the default.
fn parse_quant_hint(label: &str) -> Result<QuantHint> {
    match label.to_ascii_lowercase().as_str() {
        "default" | "" => Ok(QuantHint::Default),
        "fp16" | "float16" => Ok(QuantHint::Fp16),
        "int8" | "i8" => Ok(QuantHint::Int8),
        "int4" | "i4" => Ok(QuantHint::Int4),
        other => Err(anyhow!(
            "--quant: unknown value '{other}'; expected one of \
             default, fp16, int8, int4"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quant_hint_accepts_known_labels() {
        assert_eq!(parse_quant_hint("default").unwrap(), QuantHint::Default);
        assert_eq!(parse_quant_hint("").unwrap(), QuantHint::Default);
        assert_eq!(parse_quant_hint("fp16").unwrap(), QuantHint::Fp16);
        assert_eq!(parse_quant_hint("float16").unwrap(), QuantHint::Fp16);
        assert_eq!(parse_quant_hint("int8").unwrap(), QuantHint::Int8);
        assert_eq!(parse_quant_hint("i8").unwrap(), QuantHint::Int8);
        assert_eq!(parse_quant_hint("int4").unwrap(), QuantHint::Int4);
        assert_eq!(parse_quant_hint("i4").unwrap(), QuantHint::Int4);
        assert_eq!(parse_quant_hint("INT8").unwrap(), QuantHint::Int8);
    }

    #[test]
    fn parse_quant_hint_rejects_unknown() {
        let err = parse_quant_hint("turbo3").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown"), "expected 'unknown' in: {msg}");
        assert!(
            msg.contains("turbo3"),
            "expected label echoed back in: {msg}"
        );
    }
}
