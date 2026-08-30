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

//! Argv pre-pass translating llama.cpp's multi-letter single-dash options
//! into the long spellings clap accepts (issue #1434).
//!
//! llama.cpp's own argument parser treats `-hf`, `-ngl`, `-fa` and friends as
//! whole tokens, and its documentation uses the short forms almost
//! exclusively (`llama-server -hf ggml-org/… -ngl 99`). clap has no such
//! concept: a single dash introduces a cluster of one-letter shorts, so `-hf`
//! parses as `-h -f`. On both mlxcel server binaries `-h` is `--help`, which
//! means `mlxcel-server -hf ggml-org/foo` renders the help text and **exits
//! 0**. That is the worst possible outcome for a compatibility surface: a
//! command line that upstream honours neither runs nor reports an error, and
//! `-ngl` is the single most common llama-server flag there is.
//!
//! This pass rewrites those exact tokens before clap sees them, so they reach
//! the real option (and, for the ones mlxcel cannot support, the diagnostic
//! that explains why) instead of the help screen.
//!
//! # Why it needs the clap surface
//!
//! Rewriting every occurrence of `-hf` anywhere in argv would corrupt a
//! *value* that happens to look like one of these tokens
//! (`--chat-template -hf`, say). The pass therefore walks argv with the same
//! knowledge clap has: it asks the built [`clap::Command`] which tokens
//! consume the following argument, skips those values, and stops entirely at
//! a `--` terminator. Only a token in an option position is ever rewritten.
//!
//! Argv is handled as [`OsString`] throughout, never `String`:
//! `std::env::args()` panics on a non-UTF-8 argument, which is legal on Unix
//! and accepted today by every `PathBuf` option such as `--model`.

use std::collections::HashSet;
use std::ffi::{OsStr, OsString};

/// llama.cpp single-dash multi-letter options and the mlxcel long spelling
/// each maps onto.
///
/// Only options mlxcel actually defines belong here; a token that maps to
/// nothing must keep failing as an unknown argument rather than being
/// silently swallowed, and a token that is already an mlxcel short would be
/// shadowed by this pass rather than parsed (the tests assert neither
/// happens). Whether the rewritten option consumes the next argv entry is
/// read from the command, so value-less tokens such as `-cmoe` and
/// value-taking ones such as `-ngl` can sit in the same table. Sorted by
/// token. Extend it alongside the flag it names, and add a case to
/// `llama_short_flags_tests.rs`.
const SHORT_ALIASES: &[(&str, &str)] = &[
    ("-C", "--cpu-mask"),
    ("-Cb", "--cpu-mask-batch"),
    ("-Cbd", "--spec-draft-cpu-mask-batch"),
    ("-Cd", "--spec-draft-cpu-mask"),
    ("-Cr", "--cpu-range"),
    ("-Crb", "--cpu-range-batch"),
    ("-Crd", "--spec-draft-cpu-range"),
    ("-ag", "--agent"),
    ("-bs", "--backend-sampling"),
    ("-cb", "--cont-batching"),
    ("-cl", "--cache-list"),
    ("-cmoe", "--cpu-moe"),
    ("-cmoed", "--spec-draft-cpu-moe"),
    ("-cms", "--checkpoint-min-step"),
    ("-cram", "--cache-ram"),
    ("-ctk", "--cache-type-k"),
    ("-ctkd", "--spec-draft-type-k"),
    ("-ctv", "--cache-type-v"),
    ("-ctvd", "--spec-draft-type-v"),
    ("-ctxcp", "--ctx-checkpoints"),
    ("-dev", "--device"),
    ("-devd", "--spec-draft-device"),
    ("-dio", "--direct-io"),
    ("-dr", "--docker-repo"),
    ("-dt", "--defrag-thold"),
    ("-e", "--escape"),
    ("-fa", "--flash-attn"),
    ("-fit", "--fit"),
    ("-fitc", "--fit-ctx"),
    ("-fitt", "--fit-target"),
    ("-hf", "--hf-repo"),
    ("-hfd", "--spec-draft-hf"),
    ("-hff", "--hf-file"),
    ("-hfr", "--hf-repo"),
    ("-hfrd", "--spec-draft-hf"),
    ("-hft", "--hf-token"),
    ("-kvo", "--kv-offload"),
    ("-kvu", "--kv-unified"),
    ("-lcd", "--lookup-cache-dynamic"),
    ("-lcs", "--lookup-cache-static"),
    ("-lm", "--load-mode"),
    ("-lv", "--verbosity"),
    ("-md", "--model-draft"),
    ("-mg", "--main-gpu"),
    ("-mm", "--mmproj"),
    ("-mmdev", "--mmproj-device"),
    ("-mmu", "--mmproj-url"),
    ("-mu", "--model-url"),
    ("-ncmoe", "--n-cpu-moe"),
    ("-ncmoed", "--spec-draft-n-cpu-moe"),
    ("-ndio", "--no-direct-io"),
    ("-ngl", "--gpu-layers"),
    ("-ngld", "--spec-draft-ngl"),
    ("-nkvo", "--no-kv-offload"),
    ("-no-ag", "--no-agent"),
    ("-no-kvu", "--no-kv-unified"),
    ("-nocb", "--no-cont-batching"),
    ("-np", "--parallel"),
    ("-nr", "--no-repack"),
    ("-ot", "--override-tensor"),
    ("-otd", "--spec-draft-override-tensor"),
    ("-rea", "--reasoning"),
    ("-sm", "--split-mode"),
    ("-sp", "--special"),
    ("-sps", "--slot-prompt-similarity"),
    ("-t", "--threads"),
    ("-tb", "--threads-batch"),
    ("-tbd", "--spec-draft-threads-batch"),
    ("-td", "--spec-draft-threads"),
    ("-ts", "--tensor-split"),
    ("-ub", "--ubatch-size"),
];

/// The long spelling `token` maps onto, if any.
fn long_for(token: &str) -> Option<&'static str> {
    SHORT_ALIASES
        .iter()
        .find(|(short, _)| *short == token)
        .map(|(_, long)| *long)
}

/// Every accepted token of `cmd` that consumes the following argv entry.
///
/// Built from the command itself rather than a hand-written list so the pass
/// cannot drift from the real surface as flags are added.
///
/// The test is `min_values() > 0`, not `takes_values()`. An option declared
/// `num_args = 0..=1` with a `default_missing_value` (b10621's boolean
/// spellings: `--cont-batching`, `--kv-unified`, `--cache-idle-slots`,
/// `--context-shift`, `--cache-prompt`) MAY take a following value but does
/// not require one, and upstream's short form never passes one. Treating it
/// as value-taking here would make `-cb -m model` swallow `-m` before clap
/// ever saw it. Clap itself still accepts `--cont-batching true`, because a
/// value it does consume is examined by clap rather than by this pass.
fn value_taking_tokens(cmd: &mut clap::Command) -> HashSet<String> {
    cmd.build();
    let mut tokens = HashSet::new();
    for arg in cmd.get_arguments() {
        if !arg.get_num_args().is_some_and(|r| r.min_values() > 0) {
            continue;
        }
        if let Some(long) = arg.get_long() {
            tokens.insert(format!("--{long}"));
        }
        for alias in arg.get_all_aliases().unwrap_or_default() {
            tokens.insert(format!("--{alias}"));
        }
        if let Some(short) = arg.get_short() {
            tokens.insert(format!("-{short}"));
        }
        for short in arg.get_all_short_aliases().unwrap_or_default() {
            tokens.insert(format!("-{short}"));
        }
    }
    tokens
}

/// Rewrite llama.cpp short options in `args`, starting at index `start`.
///
/// `start` is the index of the first argument that could be an option: 1 for
/// a plain binary, 2 for `mlxcel serve`. Entries before it are copied
/// verbatim. `cmd` is the command those options are parsed against.
///
/// Returns the rewritten argv. When nothing matched, it is equal to the
/// input.
#[must_use]
pub fn expand_llama_short_options(
    cmd: &mut clap::Command,
    args: Vec<OsString>,
    start: usize,
) -> Vec<OsString> {
    let value_taking = value_taking_tokens(cmd);
    let mut out = Vec::with_capacity(args.len());
    let mut skip_value = false;
    let mut past_terminator = false;

    for (position, arg) in args.into_iter().enumerate() {
        if position < start || past_terminator {
            out.push(arg);
            continue;
        }
        if skip_value {
            // This entry is the value of the preceding option; never an
            // option position, so never rewritten.
            skip_value = false;
            out.push(arg);
            continue;
        }
        // Everything after `--` is positional by definition.
        if arg == OsStr::new("--") {
            past_terminator = true;
            out.push(arg);
            continue;
        }

        let Some(text) = arg.to_str() else {
            // A non-UTF-8 entry cannot be one of these ASCII tokens, but it
            // can still be the option whose value follows.
            out.push(arg);
            continue;
        };

        // `-hf value` and `-hf=value` are both rewritten; llama.cpp only
        // accepts the former, but clap accepts `=` on long options and a user
        // translating a command line by hand may well write it.
        let (token, inline_value) = match text.split_once('=') {
            Some((token, value)) => (token, Some(value)),
            None => (text, None),
        };

        if let Some(long) = long_for(token) {
            match inline_value {
                Some(value) => out.push(OsString::from(format!("{long}={value}"))),
                None => {
                    out.push(OsString::from(long));
                    // Whether the next entry is this option's value is a
                    // property of the LONG form, which the command knows.
                    // `-ngl 99` consumes the 99; `-cmoe` consumes nothing.
                    skip_value = value_taking.contains(long);
                }
            }
            continue;
        }

        if inline_value.is_none() && value_taking.contains(text) {
            skip_value = true;
        }
        out.push(arg);
    }

    out
}

#[cfg(test)]
#[path = "llama_short_flags_tests.rs"]
mod tests;
