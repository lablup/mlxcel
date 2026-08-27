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

//! Machine-readable dump of a binary's complete clap flag surface.
//!
//! `--help` renders only the visible arguments, but the llama-server b10621
//! compatibility manifest (`compat/llama-server/b10621/`, issue #1443) must
//! also see the hidden compatibility arguments (`hide = true`) that both
//! server entry points accept, for example `--n-gpu-layers` or `--no-mmap`.
//! Scanning the clap definitions in source would drift from what the built
//! binary actually parses, so instead each server binary answers a hidden
//! machine interface at runtime:
//!
//! ```text
//! mlxcel serve --dump-flag-surface
//! mlxcel-server --dump-flag-surface
//! ```
//!
//! Both print one deterministic JSON document describing every argument the
//! clap command accepts, hidden ones included, and exit. The interception is
//! positional (the token must be the first argument after the subcommand /
//! binary name), happens before `Cli::parse`, and the token is not a clap
//! argument, so the operator-facing `--help` surface is unchanged.
//!
//! Consumers:
//! - `tests/llama_compat_manifest.rs` validates the checked-in manifest's
//!   `supported` / `aliased` claims against this dump.
//! - `scripts/compat/extract_b10621_manifest.py` seeds and refreshes the
//!   manifest's mlxcel-side acceptance data from it.
//!
//! The document layout is versioned via `schema_version`; bump it on any
//! field rename or semantic change so the consumers fail loudly instead of
//! misreading the dump.

use serde_json::{Map, Value, json};

/// Sentinel token answered by [`dump_requested`]. Not a clap argument.
pub const DUMP_FLAG_SURFACE_TOKEN: &str = "--dump-flag-surface";

/// Version of the JSON document layout produced by [`flag_surface_json`].
pub const FLAG_SURFACE_SCHEMA_VERSION: u32 = 1;

/// True when `args` (the raw process argv, program name included) requests a
/// flag-surface dump: the sentinel token must be the first argument after
/// `skip` leading tokens (1 for a plain binary, 2 for `mlxcel serve`).
///
/// Positional matching keeps the sentinel out of the way of ordinary
/// argument values: only `mlxcel-server --dump-flag-surface` and
/// `mlxcel serve --dump-flag-surface` trigger the dump.
pub fn dump_requested(args: &[String], skip: usize) -> bool {
    args.get(skip).map(String::as_str) == Some(DUMP_FLAG_SURFACE_TOKEN)
}

/// Render the complete argument surface of `cmd` as deterministic JSON.
///
/// `binary` names the surface being dumped (`"mlxcel serve"` or
/// `"mlxcel-server"`). The command is built first so clap propagates global
/// settings and default values. Arguments are sorted by primary long name
/// (positional and short-only arguments sort under their clap id), so
/// repeated invocations of the same binary produce byte-identical output.
pub fn flag_surface_json(binary: &str, cmd: &mut clap::Command) -> String {
    cmd.build();

    let mut entries: Vec<(String, Value)> = cmd
        .get_arguments()
        .map(|arg| {
            let long = arg.get_long().map(str::to_owned);
            let sort_key = long
                .clone()
                .unwrap_or_else(|| format!("\u{0}{}", arg.get_id()));

            let mut obj = Map::new();
            obj.insert("id".into(), json!(arg.get_id().as_str()));
            obj.insert("long".into(), json!(long));
            let mut long_aliases: Vec<String> = arg
                .get_all_aliases()
                .unwrap_or_default()
                .into_iter()
                .map(str::to_owned)
                .collect();
            long_aliases.sort();
            obj.insert("long_aliases".into(), json!(long_aliases));
            obj.insert(
                "short".into(),
                json!(arg.get_short().map(|c| c.to_string())),
            );
            let mut short_aliases: Vec<String> = arg
                .get_all_short_aliases()
                .unwrap_or_default()
                .into_iter()
                .map(|c| c.to_string())
                .collect();
            short_aliases.sort();
            obj.insert("short_aliases".into(), json!(short_aliases));
            obj.insert(
                "env".into(),
                json!(arg.get_env().map(|e| e.to_string_lossy().into_owned())),
            );
            let defaults: Vec<String> = arg
                .get_default_values()
                .iter()
                .map(|v| v.to_string_lossy().into_owned())
                .collect();
            obj.insert("defaults".into(), json!(defaults));
            obj.insert("hidden".into(), json!(arg.is_hide_set()));
            obj.insert(
                "takes_value".into(),
                json!(arg.get_num_args().is_some_and(|r| r.takes_values())),
            );
            obj.insert("heading".into(), json!(arg.get_help_heading()));

            (sort_key, Value::Object(obj))
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let doc = json!({
        "schema_version": FLAG_SURFACE_SCHEMA_VERSION,
        "binary": binary,
        "args": entries.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
    });
    // `serde_json::to_string_pretty` on a value assembled above is infallible
    // in practice (no non-string keys, no fallible serializers); fall back to
    // an empty object rather than panicking in a release binary.
    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_owned())
}

#[cfg(test)]
#[path = "flag_surface_tests.rs"]
mod tests;
