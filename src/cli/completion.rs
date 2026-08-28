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

//! `--completion-bash`: a source-able bash completion script generated from
//! the live clap surface (issue #1448, epic #1431).
//!
//! b10621 prints one `_llama_completions` function listing every option it
//! knows, a `case "$prev"` block giving path completion to the file-valued
//! options, and one `complete -F` line per llama.cpp executable. This
//! reproduces that shape for the mlxcel binaries.
//!
//! # What is and is not offered
//!
//! Only **visible** arguments reach the script. mlxcel's b10621 compatibility
//! surface contains arguments that are deliberately hidden because offering
//! them would imply a backend that does not exist (`--n-gpu-layers`,
//! `--mlock`, the GGUF presets) or because they exist only to be refused
//! (`--control-vector`, `--log-prompts-dir`). Hidden long aliases are skipped
//! for the same reason: `clap::Arg::get_all_aliases` returns them, so this
//! module asks for `get_visible_aliases` instead. The
//! `--dump-flag-surface` machine interface is not a clap argument at all
//! (`crate::cli::flag_surface`), so it cannot leak here either.
//!
//! `tests/llama_logging_presets.rs` runs `bash -n` over the emitted script
//! and asserts both halves: a visible compatibility option is present and no
//! hidden one is.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Upstream reference: <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>

use std::collections::BTreeSet;

/// Value-name fragments that mark an option as taking a filesystem path.
///
/// Matched case-insensitively against the clap `value_name`, so
/// `PATH_OR_REPO_ID`, `FNAME` and `DIR` all qualify and get file plus
/// directory completion instead of the flat option list.
const PATH_VALUE_HINTS: [&str; 4] = ["PATH", "FILE", "FNAME", "DIR"];

/// Turn an executable name into a valid bash function identifier.
///
/// `mlxcel-server` becomes `_mlxcel_server_completions`. Anything that is not
/// alphanumeric collapses to `_`, so the result is always a legal name.
fn function_name(exe: &str) -> String {
    let mut name = String::from("_");
    for ch in exe.chars() {
        if ch.is_ascii_alphanumeric() {
            name.push(ch);
        } else {
            name.push('_');
        }
    }
    name.push_str("_completions");
    name
}

/// Every visible spelling of `arg`: short, long, and visible aliases.
fn visible_spellings(arg: &clap::Arg) -> Vec<String> {
    let mut spellings = Vec::new();
    if let Some(short) = arg.get_short() {
        spellings.push(format!("-{short}"));
    }
    for short in arg.get_visible_short_aliases().unwrap_or_default() {
        spellings.push(format!("-{short}"));
    }
    if let Some(long) = arg.get_long() {
        spellings.push(format!("--{long}"));
    }
    for alias in arg.get_visible_aliases().unwrap_or_default() {
        spellings.push(format!("--{alias}"));
    }
    spellings
}

/// True when `arg`'s value looks like a filesystem path.
fn takes_path(arg: &clap::Arg) -> bool {
    if !arg.get_num_args().is_some_and(|range| range.takes_values()) {
        return false;
    }
    let Some(value_name) = arg
        .get_value_names()
        .and_then(|names| names.first().map(|name| name.as_str()))
    else {
        return false;
    };
    let upper = value_name.to_ascii_uppercase();
    PATH_VALUE_HINTS.iter().any(|hint| upper.contains(hint))
}

/// Render a source-able bash completion script for `exe` from `cmd`.
///
/// `exe` is the command word `complete -F` is registered against, and
/// `subject` names the invocation whose options were harvested. The two
/// differ for `mlxcel serve`, where the completion attaches to `mlxcel` but
/// the option list comes from the `serve` subcommand; the header comment
/// says so, so a sourced script is self-describing.
///
/// `cmd` is built first so clap has propagated global settings; pass the
/// command whose options the operator actually types, which for
/// `mlxcel serve` is the `serve` subcommand rather than the `mlxcel` root.
pub fn bash_completion_script(exe: &str, subject: &str, cmd: &mut clap::Command) -> String {
    cmd.build();

    // `BTreeSet` rather than `Vec`: two arguments can legitimately share a
    // spelling shape (a short alias and another argument's short), and the
    // script must be byte-identical across runs so a test can pin it.
    let mut options: BTreeSet<String> = BTreeSet::new();
    let mut path_options: BTreeSet<String> = BTreeSet::new();
    for arg in cmd.get_arguments() {
        if arg.is_hide_set() {
            continue;
        }
        let spellings = visible_spellings(arg);
        if spellings.is_empty() {
            continue;
        }
        let is_path = takes_path(arg);
        for spelling in spellings {
            if is_path {
                path_options.insert(spelling.clone());
            }
            options.insert(spelling);
        }
    }

    let function = function_name(exe);
    let opts = options.into_iter().collect::<Vec<_>>().join(" ");

    let mut script = String::new();
    script.push_str(&format!(
        "# bash completion for `{subject}`, generated by `{subject} --completion-bash`.\n\
         # Hidden llama-server compatibility arguments are deliberately omitted.\n\n"
    ));
    script.push_str(&format!("{function}() {{\n"));
    script.push_str("    local cur prev opts\n");
    script.push_str("    COMPREPLY=()\n");
    script.push_str("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
    script.push_str("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n\n");
    // A trailing space after the last entry, exactly as b10621 prints
    // (`printf("%s ", arg)` per spelling), so a consumer that splits on
    // spaces sees the same list shape.
    script.push_str(&format!("    opts=\"{opts} \"\n\n"));
    script.push_str("    case \"$prev\" in\n");
    if !path_options.is_empty() {
        let patterns = path_options.into_iter().collect::<Vec<_>>().join("|");
        script.push_str(&format!("        {patterns})\n"));
        script.push_str(
            "            COMPREPLY=( $(compgen -f -- \"$cur\") $(compgen -d -- \"$cur\") )\n",
        );
        script.push_str("            return 0\n");
        script.push_str("            ;;\n");
    }
    script.push_str("        *)\n");
    script.push_str("            COMPREPLY=( $(compgen -W \"${opts}\" -- \"$cur\") )\n");
    script.push_str("            return 0\n");
    script.push_str("            ;;\n");
    script.push_str("    esac\n");
    script.push_str("}\n\n");
    script.push_str(&format!("complete -F {function} {exe}\n"));
    script
}

#[cfg(test)]
#[path = "completion_tests.rs"]
mod tests;
