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

//! Prompt-cache and continuous-batching flag group (llama-server b10621 parity).
//!
//! One canonical clap definition of `--cache-prompt` / `--no-cache-prompt`,
//! `--cache-reuse`, `--cache-ram` and `--cont-batching` / `--no-cont-batching`,
//! flattened into both server binaries. Upstream's definitions are in
//! [`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp).
//!
//! # Why these four and not the rest of b10621's cache surface
//!
//! Each of these names a quantity mlxcel already has. `--cache-prompt` is the
//! upstream spelling of the prompt-prefix KV cache mlxcel ships as
//! `--prompt-cache-enabled`. `--cache-ram` is that cache's byte budget, in
//! MiB instead of bytes. `--no-cont-batching` is "decode one sequence at a
//! time", which is what `--max-batch-size 1` already means here.
//! `--cache-reuse` names a quantity mlxcel does *not* have, and says so at
//! startup rather than accepting the number and ignoring it.
//!
//! b10621's `--slot-prompt-similarity`, `--kv-unified`, `--cache-idle-slots`,
//! `--ctx-checkpoints` and `--checkpoint-min-step` are deliberately absent:
//! they tune per-slot retained prompts and context checkpoints, and mlxcel has
//! neither. Its reuse is a process-wide radix trie over token prefixes rather
//! than a scan of what each slot happens to be holding, so there is no slot
//! prompt for a similarity threshold to compare against and no checkpoint
//! spacing to set. Accepting them inert would be the silent-acceptance failure
//! epic #1431 exists to remove; their manifest entries carry the divergence
//! instead. See `docs/llama-server-compat.md`.
//!
//! Used by: mlxcel serve, mlxcel-server.

use clap::Args;

/// Bytes in one MiB, the unit b10621 states `--cache-ram` in.
const MIB: usize = 1024 * 1024;

/// Shared prompt-cache and batching flag group.
#[derive(Args, Debug, Default, Clone)]
#[command(next_help_heading = "Prompt Cache (llama-server compatibility)")]
pub struct CacheCompatArgs {
    /// Enable prompt caching (llama-server spelling of `--prompt-cache-enabled`).
    ///
    /// On by default, as upstream. Pair with `--no-cache-prompt` to turn the
    /// prompt-prefix KV cache off, in which case a shared prefix such as a long
    /// system prompt is re-prefilled on every request.
    /// `Some` only when the flag or `LLAMA_ARG_CACHE_PROMPT` said something.
    /// The optional value is what makes `LLAMA_ARG_CACHE_PROMPT=0` a disable
    /// rather than an absence: a plain boolean flag collapses "the environment
    /// said false" into "nothing was set", and upstream treats the two
    /// differently.
    #[arg(
        long = "cache-prompt",
        env = "LLAMA_ARG_CACHE_PROMPT",
        overrides_with = "no_cache_prompt",
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        // clap's default `bool` parser accepts only "true" and "false".
        // llama.cpp reads its boolean environment variables with
        // `std::stoi`-style truthiness, so `LLAMA_ARG_CACHE_PROMPT=0` has to
        // mean off rather than fail the command line.
        value_parser = clap::builder::BoolishValueParser::new(),
        action = clap::ArgAction::Set
    )]
    pub cache_prompt: Option<bool>,

    /// Disable prompt caching (llama-server `--no-cache-prompt`).
    #[arg(
        long = "no-cache-prompt",
        overrides_with = "cache_prompt",
        action = clap::ArgAction::SetTrue
    )]
    pub no_cache_prompt: bool,

    /// Minimum chunk size to attempt reusing from the cache via KV shifting.
    ///
    /// `0`, upstream's default, means no KV-shift reuse and is accepted.
    /// A positive value is refused at startup: mlxcel's prompt cache reuses
    /// strict token prefixes and has no operation that removes a span from the
    /// middle of a KV cache and re-bases the rotary positions of what follows.
    /// See `docs/llama-server-compat.md`.
    #[arg(long = "cache-reuse", env = "LLAMA_ARG_CACHE_REUSE", value_name = "N")]
    pub cache_reuse: Option<i64>,

    /// Maximum prompt-cache size in MiB (`-1` = no limit, `0` = disable).
    ///
    /// The llama-server spelling of `--prompt-cache-capacity-bytes`, in MiB
    /// rather than bytes. mlxcel's own default is 2048 MiB where upstream's is
    /// 8192; passing the flag sets the budget either way.
    #[arg(long = "cache-ram", env = "LLAMA_ARG_CACHE_RAM", value_name = "N")]
    pub cache_ram: Option<i64>,

    /// Enable continuous batching (the default).
    /// `Some` only when the flag or `LLAMA_ARG_CONT_BATCHING` said something;
    /// see [`Self::cache_prompt`] for why this is not a plain `bool`.
    #[arg(
        long = "cont-batching",
        env = "LLAMA_ARG_CONT_BATCHING",
        overrides_with = "no_cont_batching",
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        // clap's default `bool` parser accepts only "true" and "false".
        // llama.cpp reads its boolean environment variables with
        // `std::stoi`-style truthiness, so `LLAMA_ARG_CONT_BATCHING=0` has to
        // mean off rather than fail the command line.
        value_parser = clap::builder::BoolishValueParser::new(),
        action = clap::ArgAction::Set
    )]
    pub cont_batching: Option<bool>,

    /// Disable continuous batching: decode one sequence at a time.
    ///
    /// Equivalent to `--max-batch-size 1`. The batch scheduler stays in place,
    /// so the prompt cache, chunked prefill and speculative decoding keep
    /// working; only the decode width is pinned to one. That is upstream's
    /// meaning too: `-nocb` stops slots interleaving, it does not remove them.
    /// mlxcel's own `--no-batch` is the stronger form, replacing the scheduler
    /// with a sequential worker.
    #[arg(
        long = "no-cont-batching",
        overrides_with = "cont_batching",
        action = clap::ArgAction::SetTrue
    )]
    pub no_cont_batching: bool,
}

/// What this group resolves to, once precedence and validation are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheCompatResolution {
    /// `Some(false)` when `--no-cache-prompt` was given, `Some(true)` when
    /// `--cache-prompt` was, `None` when neither was: the b10621 surface is
    /// silent and mlxcel's own `--prompt-cache-enabled` decides.
    pub prompt_cache_enabled: Option<bool>,
    /// Prompt-cache byte budget from `--cache-ram`, `None` when unset.
    pub capacity_bytes: Option<usize>,
    /// `true` when `--no-cont-batching` pinned the decode width to one.
    pub single_sequence_decode: bool,
}

impl CacheCompatArgs {
    /// Apply precedence and refuse what cannot be served.
    ///
    /// `--cache-reuse` is validated here rather than acted on: upstream's `0`
    /// default is accepted and anything positive is a startup error naming what
    /// is missing. A negative value is rejected as out of domain, which
    /// upstream does not do (it stores the negative and never compares against
    /// it), but silently accepting a number that can only be a typo is worse
    /// than a clear message.
    pub fn resolve(&self) -> Result<CacheCompatResolution, String> {
        self.check_cache_reuse()?;

        // `overrides_with` makes the last flag on the command line win, so
        // "both set" is not reachable from argv; the explicit `no_*` arm is
        // there so the environment-supplied `Some(true)` cannot outvote a flag.
        let prompt_cache_enabled = if self.no_cache_prompt {
            Some(false)
        } else {
            self.cache_prompt
        };
        let single_sequence_decode = self.no_cont_batching || self.cont_batching == Some(false);

        Ok(CacheCompatResolution {
            prompt_cache_enabled,
            capacity_bytes: self.resolve_cache_ram()?,
            single_sequence_decode,
        })
    }

    /// `0` is upstream's default and inert; a positive request is refused.
    fn check_cache_reuse(&self) -> Result<(), String> {
        let Some(value) = self.cache_reuse else {
            return Ok(());
        };
        if value == 0 {
            return Ok(());
        }
        if value < 0 {
            return Err(format!(
                "--cache-reuse {value} is not a chunk size: the value is a minimum number of \
                 tokens and must be zero or positive"
            ));
        }
        Err(format!(
            "--cache-reuse {value} requests KV-shift chunk reuse, which mlxcel does not \
             implement. Its prompt cache reuses a strict token prefix: it adopts a cached KV \
             set whose tokens are a prefix of the incoming request and prefills the rest. \
             Reusing a chunk that is not a prefix means deleting a span from the middle of a \
             cached KV set and re-basing the rotary positions of everything after it, and no \
             operation in this tree rewrites a cached key's rotation. Accepting the number and \
             continuing would leave the cache behaving exactly as it does at 0 while the \
             operator believed otherwise. Pass --cache-reuse 0, or drop the flag, for the \
             upstream default. Tracked by #1453."
        ))
    }

    /// b10621 states `--cache-ram` in MiB with `-1` = no limit and `0` = disable.
    fn resolve_cache_ram(&self) -> Result<Option<usize>, String> {
        let Some(mib) = self.cache_ram else {
            return Ok(None);
        };
        if mib == -1 {
            return Ok(Some(usize::MAX));
        }
        if mib < 0 {
            return Err(format!(
                "--cache-ram {mib} is out of domain: pass a size in MiB, -1 for no limit, or 0 \
                 to disable the prompt cache"
            ));
        }
        usize::try_from(mib)
            .ok()
            .and_then(|mib| mib.checked_mul(MIB))
            .map(Some)
            .ok_or_else(|| {
                format!("--cache-ram {mib} MiB does not fit in this platform's address space")
            })
    }
}

/// Resolve the group from the environment alone.
///
/// Parsing an argv of just the program name makes clap fill every field from
/// the `LLAMA_ARG_*` variable it is already bound to, so a caller that has no
/// command line still goes through one set of definitions rather than a second
/// hand-written reader that can drift from them.
pub fn from_env() -> Result<CacheCompatResolution, String> {
    use clap::Parser;

    #[derive(Parser)]
    #[command(allow_negative_numbers = true)]
    struct EnvOnly {
        #[command(flatten)]
        cache: CacheCompatArgs,
    }

    EnvOnly::try_parse_from(["mlxcel"])
        .map_err(|e| e.to_string())?
        .cache
        .resolve()
}

#[cfg(test)]
#[path = "cache_args_tests.rs"]
mod tests;
