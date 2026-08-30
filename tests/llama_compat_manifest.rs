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

//! Conformance between the checked-in llama-server b10621 compatibility
//! manifest (`compat/llama-server/b10621/`, issue #1443) and the two server
//! binaries.
//!
//! This is the binary-facing half of the manifest gate. The structural half
//! (states, issue links, counts, canonical serialization) is
//! `scripts/ci/check_llama_compat_manifest.py`; the route and native
//! request-field half is `src/server/llama_compat_tests.rs`. This file
//! covers what only the built binaries can answer:
//!
//! 1. Every mlxcel acceptance claim on an option entry (accepted spellings,
//!    clap env bindings, defaults, hidden-ness) holds against the actual
//!    clap surface of BOTH `mlxcel serve` and `mlxcel-server`, hidden
//!    compatibility arguments included. The surface comes from the hidden
//!    `--dump-flag-surface` machine interface (`src/cli/flag_surface.rs`),
//!    not from `--help`, precisely so hidden arguments are visible.
//! 2. When the official b10621 archive is available locally, the manifest's
//!    option inventory (entries, spelling grouping, environment variables)
//!    is re-derived from the real `llama-server --help` and compared
//!    exactly. CI does not download the archive; the test skips with a
//!    message unless `MLXCEL_LLAMA_B10621_DIR` points at the extracted
//!    archive directory.
//!
//! Complements `tests/cli_help_consistency.rs`, which pins the two mlxcel
//! server surfaces against EACH OTHER; this file pins the manifest's claims
//! against those surfaces. Neither replaces the other.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

mod common;
use common::resolve_repo_binary;

const MANIFEST_REL: &str = "compat/llama-server/b10621";

/// Manifest document schema, independent of the pinned llama.cpp release.
/// Kept in lockstep with `scripts/ci/check_llama_compat_manifest.py` and
/// `scripts/compat/extract_b10621_manifest.py`; bump all three together.
/// Issue #1443 follow-ups: 2 when pin.json's `shards` field changed from a
/// bare name list to a mapping of shard name to its owning-issue set, 3 when
/// every entry gained the structured `divergence` list; 4 when every entry
/// gained the `rationale` object and `by_design` joined the state vocabulary
/// (#1499). This constant is the MANIFEST schema; the `schema_version: 1`
/// asserted further down is the unrelated `--dump-flag-surface` dump format.
const MANIFEST_SCHEMA_VERSION: i64 = 4;

/// b10621's help-entry count, frozen by the pin and re-asserted by
/// `scripts/ci/check_llama_compat_manifest.py`. Every option entry is either
/// claimed (mlxcel accepts something for it) or unclaimed, so the two census
/// tests below must always partition exactly this many entries. That is the
/// invariant a floor on either half only approximated: epic #1431 converts
/// unclaimed entries into claims by design, so any fixed floor on the
/// unclaimed half goes stale, while the sum cannot.
const EXPECTED_OPTION_ENTRIES: usize = 249;

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(MANIFEST_REL)
}

/// Load every shard's entries (all kinds), keyed by entry id.
fn load_manifest_entries() -> BTreeMap<String, serde_json::Value> {
    let dir = manifest_dir();
    let mut entries = BTreeMap::new();
    for path in std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {dir:?}: {e}"))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .filter(|p| p.file_name().is_some_and(|n| n != "pin.json"))
    {
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("shard readable"))
                .unwrap_or_else(|e| panic!("{path:?} is not valid JSON: {e}"));
        assert_eq!(
            doc["schema_version"], MANIFEST_SCHEMA_VERSION,
            "{path:?}: unsupported manifest schema_version"
        );
        for entry in doc["entries"].as_array().expect("entries array") {
            let id = entry["id"].as_str().expect("entry id").to_owned();
            let prev = entries.insert(id.clone(), entry.clone());
            assert!(prev.is_none(), "entry {id} appears in two shards");
        }
    }
    assert!(
        !entries.is_empty(),
        "no manifest entries found under {dir:?}"
    );
    entries
}

/// One binary's dumped clap surface: spelling -> argument object.
struct FlagSurface {
    label: &'static str,
    by_spelling: BTreeMap<String, serde_json::Value>,
}

impl FlagSurface {
    fn arg(&self, spelling: &str) -> Option<&serde_json::Value> {
        self.by_spelling.get(spelling)
    }
}

fn dump_surface(bin_name: &str, args: &[&str], label: &'static str) -> FlagSurface {
    let (path, resolution) = resolve_repo_binary(bin_name);
    let output = Command::new(&path)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin_name} from {path:?}: {e}\n{resolution}"));
    assert!(
        output.status.success(),
        "{bin_name} {args:?} exited with {:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let doc: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("--dump-flag-surface emits JSON");
    assert_eq!(
        doc["schema_version"], 1,
        "unknown flag-surface schema version; update this test with the dump format"
    );
    let mut by_spelling = BTreeMap::new();
    for arg in doc["args"].as_array().expect("args array") {
        if let Some(long) = arg["long"].as_str() {
            by_spelling.insert(format!("--{long}"), arg.clone());
        }
        for alias in arg["long_aliases"].as_array().expect("aliases array") {
            by_spelling.insert(
                format!("--{}", alias.as_str().expect("alias str")),
                arg.clone(),
            );
        }
    }
    FlagSurface { label, by_spelling }
}

fn surfaces() -> [FlagSurface; 2] {
    [
        dump_surface("mlxcel", &["serve", "--dump-flag-surface"], "mlxcel serve"),
        dump_surface("mlxcel-server", &["--dump-flag-surface"], "mlxcel-server"),
    ]
}

/// Every option-entry acceptance claim in the manifest must hold on both
/// server binaries: claimed spellings parse, claimed env bindings are the
/// clap `env = ...` attribute, claimed defaults and hidden-ness match.
/// This is what fails when an mlxcel spelling, default, or environment
/// binding drifts away from a `supported` / `aliased` (or partially
/// claimed `deferred`) manifest entry.
#[test]
fn manifest_option_claims_hold_on_both_server_binaries() {
    let entries = load_manifest_entries();
    let surfaces = surfaces();
    let mut checked = 0usize;

    for (id, entry) in &entries {
        if entry["kind"] != "option" {
            continue;
        }
        let claim = &entry["mlxcel"];
        if claim.is_null() {
            continue;
        }

        let accepted: Vec<&str> = claim["accepted_spellings"]
            .as_array()
            .map(|a| a.iter().map(|s| s.as_str().expect("spelling")).collect())
            .unwrap_or_default();
        for spelling in &accepted {
            for surface in &surfaces {
                assert!(
                    surface.arg(spelling).is_some(),
                    "{id}: manifest claims {} accepts {spelling}, but the \
                     dumped flag surface does not contain it",
                    surface.label
                );
            }
        }
        if let Some(per_binary) = claim["accepted_on_one_binary_only"].as_object() {
            for (binary, spellings) in per_binary {
                let surface = surfaces
                    .iter()
                    .find(|s| s.label.contains(binary.as_str()))
                    .unwrap_or_else(|| panic!("{id}: unknown binary key {binary}"));
                for spelling in spellings.as_array().expect("spellings array") {
                    let spelling = spelling.as_str().expect("spelling");
                    assert!(
                        surface.arg(spelling).is_some(),
                        "{id}: manifest claims {} accepts {spelling}, but the \
                         dumped flag surface does not contain it",
                        surface.label
                    );
                }
            }
        }

        // Env, defaults, and hidden-ness are checked on the argument that
        // carries the first mutually accepted spelling.
        if let Some(reference) = accepted.first() {
            for surface in &surfaces {
                let arg = surface.arg(reference).expect("checked above");
                if claim["env_binding"] == "clap" {
                    assert_eq!(
                        arg["env"], claim["env"],
                        "{id}: {} binds env {:?} but the manifest claims {:?}",
                        surface.label, arg["env"], claim["env"]
                    );
                }
                if let Some(defaults) = claim["defaults"].as_array() {
                    assert_eq!(
                        arg["defaults"].as_array().expect("defaults array"),
                        defaults,
                        "{id}: {} default drifted from the manifest claim",
                        surface.label
                    );
                }
                if claim["hidden"] == true {
                    assert_eq!(
                        arg["hidden"], true,
                        "{id}: manifest records a hidden compatibility \
                         argument but {} no longer hides it",
                        surface.label
                    );
                }
            }
        }
        checked += 1;
    }

    // Census: every option entry is claimed or unclaimed, and the two halves
    // must add up to the frozen inventory. Counted here rather than trusted so
    // a shard that loses entries fails even when every surviving claim holds.
    let unclaimed = entries
        .values()
        .filter(|e| e["kind"] == "option" && e["mlxcel"].is_null())
        .count();
    assert_eq!(
        checked + unclaimed,
        EXPECTED_OPTION_ENTRIES,
        "{checked} claimed + {unclaimed} unclaimed option entries do not add up to b10621's \
         {EXPECTED_OPTION_ENTRIES} help entries; the manifest has lost or gained entries"
    );
}

/// The other direction: an option entry the manifest records NO mlxcel claim
/// for must genuinely not be accepted by either binary. Without this, a chain
/// could add a b10621 flag and leave its entry untouched, which is exactly
/// the silent drift the manifest exists to prevent. `src/server/llama_compat_tests.rs`
/// already asserts the same property for routes and native request fields;
/// this closes the option third of it.
#[test]
fn unclaimed_option_entries_are_accepted_by_neither_server_binary() {
    let entries = load_manifest_entries();
    let surfaces = surfaces();
    let mut checked = 0usize;

    for (id, entry) in &entries {
        if entry["kind"] != "option" || !entry["mlxcel"].is_null() {
            continue;
        }
        for spelling in entry["long_spellings"].as_array().expect("long_spellings") {
            let spelling = spelling.as_str().expect("spelling");
            for surface in &surfaces {
                assert!(
                    surface.arg(spelling).is_none(),
                    "{id}: {} accepts {spelling}, but the manifest records no \
                     mlxcel claim for this b10621 entry. Flip the entry \
                     (state, mlxcel, notes, test) in the same change that adds \
                     the argument.",
                    surface.label
                );
            }
        }
        checked += 1;
    }

    // Same census as above, from the other side.
    let claimed = entries
        .values()
        .filter(|e| e["kind"] == "option" && !e["mlxcel"].is_null())
        .count();
    assert_eq!(
        checked + claimed,
        EXPECTED_OPTION_ENTRIES,
        "{checked} unclaimed + {claimed} claimed option entries do not add up to b10621's \
         {EXPECTED_OPTION_ENTRIES} help entries; the manifest has lost or gained entries"
    );
}

/// The sentinel that produces the dumped surfaces must stay invisible to the
/// operator: it is not a clap argument, so neither binary's `--help` may
/// mention it. This is the assertion behind "the operator-facing surface is
/// unchanged" in `src/cli/flag_surface.rs`.
#[test]
fn the_flag_surface_sentinel_never_appears_in_help() {
    for (bin, args) in [
        ("mlxcel", &["serve", "--help"][..]),
        ("mlxcel-server", &["--help"][..]),
        ("mlxcel", &["--help"][..]),
    ] {
        let (path, resolution) = resolve_repo_binary(bin);
        let output = Command::new(&path)
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("failed to spawn {bin} from {path:?}: {e}\n{resolution}"));
        let help = String::from_utf8_lossy(&output.stdout);
        assert!(
            !help.contains("dump-flag-surface"),
            "{bin} {args:?} renders the hidden flag-surface sentinel; it must \
             not be a clap argument"
        );
    }
}

/// Minimal Rust reimplementation of the extractor's `--help` entry parser,
/// used by the archive-gated conformance test below. Returns canonical
/// entry id -> (sorted long spellings, env var).
fn parse_llama_help(text: &str) -> BTreeMap<String, (BTreeSet<String>, Option<String>)> {
    let mut result: BTreeMap<String, (BTreeSet<String>, Option<String>)> = BTreeMap::new();
    let mut current: Option<String> = None;
    let mut blocks: Vec<(Vec<String>, String)> = Vec::new();

    for line in text.lines() {
        if line.starts_with("----- ") {
            current = None;
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with('-') {
            // Consume the comma-joined spelling list from the head line.
            let mut spellings = Vec::new();
            let mut rest = line;
            loop {
                let token_end = rest
                    .find(|c: char| c == ',' || c.is_whitespace())
                    .unwrap_or(rest.len());
                let (token, tail) = rest.split_at(token_end);
                spellings.push(token.to_owned());
                let continues = tail.starts_with(',');
                rest = tail.trim_start_matches(',').trim_start_matches(' ');
                if !(continues && rest.starts_with('-')) {
                    break;
                }
            }
            blocks.push((spellings, rest.to_owned()));
            current = Some(String::new());
        } else if current.is_some() {
            let block = &mut blocks.last_mut().expect("head seen").1;
            block.push(' ');
            block.push_str(line.trim());
        }
    }

    for (spellings, body) in blocks {
        let longs: BTreeSet<String> = spellings
            .iter()
            .filter(|s| s.starts_with("--"))
            .cloned()
            .collect();
        let id = spellings
            .iter()
            .find(|s| s.starts_with("--"))
            .expect("entry with a long spelling")
            .clone();
        // The entry's own env binding is the LAST `(env: ...)` in its block.
        let env = body
            .rmatch_indices("(env: ")
            .next()
            .and_then(|(i, m)| body[i + m.len()..].split(')').next())
            .map(str::to_owned);
        let prev = result.insert(id.clone(), (longs, env));
        assert!(prev.is_none(), "duplicate llama help entry {id}");
    }
    result
}

/// Full-inventory conformance against the OFFICIAL b10621 archive. Skipped
/// (with a message) unless `MLXCEL_LLAMA_B10621_DIR` names the directory
/// extracted from `llama-b10621-bin-macos-arm64.tar.gz` (SHA-256
/// 429c8270608600188035e5e92f7d78dffb7900904fe7dd7e6a84f48068cd13cf);
/// CI runs offline against the checked-in manifest only.
#[test]
fn manifest_matches_official_b10621_archive_when_available() {
    let Some(dir) = std::env::var_os("MLXCEL_LLAMA_B10621_DIR") else {
        eprintln!(
            "skipping: set MLXCEL_LLAMA_B10621_DIR to the extracted \
             llama-b10621-bin-macos-arm64 directory to run the archive \
             conformance test"
        );
        return;
    };
    let dir = PathBuf::from(dir);
    let binary = dir.join("llama-server");
    let output = Command::new(&binary)
        .arg("--help")
        .env("DYLD_LIBRARY_PATH", &dir)
        .env("LD_LIBRARY_PATH", &dir)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {binary:?}: {e}"));
    assert!(
        output.status.success(),
        "llama-server --help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let help = String::from_utf8_lossy(&output.stdout);
    let reference = parse_llama_help(&help);

    let entries = load_manifest_entries();
    let mut manifest: BTreeMap<String, (BTreeSet<String>, Option<String>)> = BTreeMap::new();
    for (id, entry) in &entries {
        if entry["kind"] != "option" {
            continue;
        }
        let longs = entry["long_spellings"]
            .as_array()
            .expect("long_spellings")
            .iter()
            .map(|s| s.as_str().expect("spelling").to_owned())
            .collect();
        let env = entry["env"].as_str().map(str::to_owned);
        manifest.insert(id.clone(), (longs, env));
    }

    let reference_ids: BTreeSet<&String> = reference.keys().collect();
    let manifest_ids: BTreeSet<&String> = manifest.keys().collect();
    assert_eq!(
        reference_ids,
        manifest_ids,
        "manifest option entries do not match the official binary's help \
         entries.\nonly in binary: {:?}\nonly in manifest: {:?}",
        reference_ids.difference(&manifest_ids),
        manifest_ids.difference(&reference_ids),
    );
    for (id, (ref_longs, ref_env)) in &reference {
        let (man_longs, man_env) = &manifest[id];
        assert_eq!(
            ref_longs, man_longs,
            "{id}: long-spelling grouping drifted from the official binary"
        );
        assert_eq!(
            ref_env, man_env,
            "{id}: environment binding drifted from the official binary"
        );
    }
    assert_eq!(
        reference.len(),
        249,
        "the official b10621 binary must expose exactly 249 help entries"
    );
}
