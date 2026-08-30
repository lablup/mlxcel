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

//! Context-window retention flag group (llama-server b10621 parity, #1472).
//!
//! One canonical clap definition of `--context-shift` / `--no-context-shift`,
//! `--keep` and `--swa-full`, flattened into both server binaries. The
//! upstream definitions are in
//! [`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp)
//! and the retention behavior they configure is in
//! [`tools/server/server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server-context.cpp).
//!
//! # The b10621 contract this group configures
//!
//! With context shifting **disabled** (the default, upstream's too), a request
//! whose prompt does not fit the per-slot context is refused, and a request
//! that outgrows the context during decode stops with `truncated: true` and
//! `stop_type: "limit"`. With it **enabled**, the scheduler makes room by
//! discarding tokens after a retained prefix: `--keep` (overridable per
//! request as `n_keep`, `-1` = the whole initial prompt) is how many leading
//! tokens survive every shift, and per-request `n_discard` is how many tokens
//! past that to drop (`0` = half of what is not retained).
//!
//! Before #1472 mlxcel front-trimmed unconditionally (keeping a fixed 4-token
//! attention sink) whenever `--ctx-size` or `--max-kv-size` bounded the KV
//! window; that silent always-on shift is what this group replaces. See the
//! migration note in `docs/llama-server-compat.md`.
//!
//! `--swa-full` is refused when requested: sliding-window families build their
//! own ring caches from the checkpoint's `sliding_window` inside each model's
//! cache constructor, mlxcel has no full-size SWA cache mode to switch to, and
//! the capability the flag purchases upstream (state save/restore and context
//! shifting over SWA layers) is gated on model-owned cache adoption rather
//! than on the ring's size.
//!
//! Used by: mlxcel serve, mlxcel-server.

use clap::Args;

/// Shared context-retention flag group.
#[derive(Args, Debug, Default, Clone)]
#[command(next_help_heading = "Context Retention Options")]
pub struct ContextCompatArgs {
    /// Enable context shift on infinite text generation (default: disabled).
    ///
    /// Off, a generation that reaches the per-slot context bound stops with
    /// `truncated: true` instead of silently discarding old tokens, matching
    /// llama-server b10621's default. On, the scheduler keeps `--keep` leading
    /// tokens and discards past them to make room.
    /// `Some` only when the flag or `LLAMA_ARG_CONTEXT_SHIFT` said something;
    /// the optional value is what makes `LLAMA_ARG_CONTEXT_SHIFT=0` a disable
    /// rather than an absence.
    #[arg(
        long = "context-shift",
        env = "LLAMA_ARG_CONTEXT_SHIFT",
        overrides_with = "no_context_shift",
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        // clap's default `bool` parser accepts only "true" and "false".
        // llama.cpp reads its boolean environment variables with
        // `std::stoi`-style truthiness, so `LLAMA_ARG_CONTEXT_SHIFT=0` has to
        // mean off rather than fail the command line.
        value_parser = clap::builder::BoolishValueParser::new(),
        action = clap::ArgAction::Set
    )]
    pub context_shift: Option<bool>,

    /// Disable context shift (llama-server `--no-context-shift`, the default).
    #[arg(
        long = "no-context-shift",
        overrides_with = "context_shift",
        action = clap::ArgAction::SetTrue
    )]
    pub no_context_shift: bool,

    /// Number of tokens to keep from the initial prompt on a context shift
    /// (default: 0, -1 = all).
    ///
    /// The server-wide default for the per-request `n_keep` field. Read only
    /// when context shifting is enabled; b10621 declares no environment
    /// variable for it.
    #[arg(long = "keep", value_name = "N")]
    pub keep: Option<i64>,

    /// Use a full-size SWA cache (refused: mlxcel has none).
    ///
    /// Sliding-window families size their ring caches from the checkpoint's
    /// own `sliding_window`; there is no full-size mode for this to select, so
    /// a request for one is refused at startup rather than accepted and
    /// ignored. `LLAMA_ARG_SWA_FULL=0` and `--swa-full false` are accepted as
    /// inert spellings of the default.
    #[arg(
        long = "swa-full",
        env = "LLAMA_ARG_SWA_FULL",
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        value_parser = clap::builder::BoolishValueParser::new(),
        action = clap::ArgAction::Set
    )]
    pub swa_full: Option<bool>,
}

/// What this group resolves to, once precedence and validation are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCompatResolution {
    /// Whether the scheduler may shift (front-discard) a sequence's context
    /// to make room, rather than stopping it at the bound.
    pub context_shift: bool,
    /// Server-wide default for the retained leading tokens on a shift
    /// (`-1` = the whole initial prompt).
    pub n_keep: i64,
}

impl ContextCompatArgs {
    /// Resolve the group, refusing what cannot be served.
    pub fn resolve(&self) -> Result<ContextCompatResolution, String> {
        if self.swa_full == Some(true) {
            return Err(
                "--swa-full requests a full-size sliding-window-attention cache, which mlxcel \
                 does not have. Sliding-window families (Gemma 3/4, Exaone 4, gpt-oss, \
                 RecurrentGemma, Step 3.5, Mellum, Ministral 3, AFMoE, DeepSeek V4) build their \
                 own ring caches from the checkpoint's sliding_window inside each model's cache \
                 constructor, and the capability the flag buys in llama-server (KV state \
                 save/restore and context shifting over SWA layers) is gated on those caches \
                 being scheduler-owned, not on their size. Drop the flag (or pass --swa-full \
                 false) to serve with the checkpoint's own window."
                    .to_string(),
            );
        }

        // `overrides_with` makes the last flag on the command line win, so
        // "both set" is not reachable from argv; the explicit `no_*` arm is
        // there so an environment-supplied `Some(true)` cannot outvote a flag.
        let context_shift = if self.no_context_shift {
            false
        } else {
            self.context_shift.unwrap_or(false)
        };

        let n_keep = self.keep.unwrap_or(0);
        if n_keep < -1 {
            return Err(format!(
                "--keep {n_keep} is out of domain: pass a token count, 0 to keep nothing past \
                 the attention margin, or -1 to keep the whole initial prompt"
            ));
        }

        Ok(ContextCompatResolution {
            context_shift,
            n_keep,
        })
    }
}

#[cfg(test)]
#[path = "context_args_tests.rs"]
mod tests;
