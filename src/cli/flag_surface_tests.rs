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

//! Unit tests for the machine-readable flag-surface dump.

use clap::{Arg, ArgAction, Command};

use super::{FLAG_SURFACE_SCHEMA_VERSION, dump_requested, flag_surface_json};

fn sample_command() -> Command {
    Command::new("sample")
        .arg(
            Arg::new("visible")
                .long("visible")
                .short('v')
                .env("SAMPLE_VISIBLE")
                .default_value("7")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ghost")
                .long("ghost")
                .alias("ghost-alias")
                .hide(true)
                .action(ArgAction::SetTrue),
        )
}

#[test]
fn dump_requested_matches_only_the_exact_position() {
    let argv = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();

    assert!(dump_requested(
        &argv(&["mlxcel-server", "--dump-flag-surface"]),
        1
    ));
    assert!(dump_requested(
        &argv(&["mlxcel", "serve", "--dump-flag-surface"]),
        2
    ));
    // Not in the sentinel position: an ordinary value must not trigger it.
    assert!(!dump_requested(
        &argv(&["mlxcel-server", "-m", "--dump-flag-surface"]),
        1
    ));
    assert!(!dump_requested(&argv(&["mlxcel-server"]), 1));
}

#[test]
fn dump_includes_hidden_args_env_defaults_and_aliases() {
    let mut cmd = sample_command();
    let doc: serde_json::Value =
        serde_json::from_str(&flag_surface_json("sample", &mut cmd)).expect("dump must be JSON");

    assert_eq!(doc["schema_version"], FLAG_SURFACE_SCHEMA_VERSION);
    assert_eq!(doc["binary"], "sample");

    let args = doc["args"].as_array().expect("args array");
    let find = |long: &str| {
        args.iter()
            .find(|a| a["long"] == long)
            .unwrap_or_else(|| panic!("missing --{long} in dump: {args:?}"))
    };

    let visible = find("visible");
    assert_eq!(visible["env"], "SAMPLE_VISIBLE");
    assert_eq!(visible["defaults"][0], "7");
    assert_eq!(visible["hidden"], false);
    assert_eq!(visible["short"], "v");
    assert_eq!(visible["takes_value"], true);

    let ghost = find("ghost");
    assert_eq!(ghost["hidden"], true);
    assert_eq!(ghost["long_aliases"][0], "ghost-alias");
    assert_eq!(ghost["takes_value"], false);
}

#[test]
fn dump_is_deterministic_and_sorted_by_long_name() {
    let a = flag_surface_json("sample", &mut sample_command());
    let b = flag_surface_json("sample", &mut sample_command());
    assert_eq!(a, b, "two dumps of the same command must be byte-identical");

    let doc: serde_json::Value = serde_json::from_str(&a).expect("dump must be JSON");
    let longs: Vec<String> = doc["args"]
        .as_array()
        .expect("args array")
        .iter()
        .filter_map(|e| e["long"].as_str().map(str::to_owned))
        .collect();
    let mut sorted = longs.clone();
    sorted.sort();
    assert_eq!(longs, sorted, "entries must be sorted by primary long name");
}
