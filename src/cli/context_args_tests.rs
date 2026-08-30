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

//! Unit tests for the b10621 context-retention flag group (#1472).

use super::*;

fn args() -> ContextCompatArgs {
    ContextCompatArgs::default()
}

#[test]
fn the_defaults_match_b10621() {
    let resolved = args().resolve().expect("no flags is not an error");
    assert!(!resolved.context_shift, "b10621 defaults context shift OFF");
    assert_eq!(resolved.n_keep, 0, "b10621 defaults --keep to 0");
}

#[test]
fn context_shift_enables_and_no_context_shift_wins_over_the_environment() {
    let on = ContextCompatArgs {
        context_shift: Some(true),
        ..args()
    };
    assert!(on.resolve().expect("valid").context_shift);

    // The `no_*` flag beats an environment-supplied `Some(true)`, the same
    // precedence the cache flags use.
    let both = ContextCompatArgs {
        context_shift: Some(true),
        no_context_shift: true,
        ..args()
    };
    assert!(!both.resolve().expect("valid").context_shift);
}

#[test]
fn keep_accepts_the_minus_one_keep_all_form() {
    let group = ContextCompatArgs {
        keep: Some(-1),
        ..args()
    };
    assert_eq!(group.resolve().expect("valid").n_keep, -1);

    let group = ContextCompatArgs {
        keep: Some(-2),
        ..args()
    };
    let err = group.resolve().expect_err("below -1 is out of domain");
    assert!(err.contains("--keep -2"), "{err}");
}

#[test]
fn swa_full_is_refused_with_a_diagnostic_naming_the_missing_mode() {
    let group = ContextCompatArgs {
        swa_full: Some(true),
        ..args()
    };
    let err = group
        .resolve()
        .expect_err("mlxcel has no full-size SWA cache");
    assert!(err.contains("--swa-full"), "{err}");
    assert!(err.contains("sliding_window"), "{err}");
    // The inert spelling of the default is accepted, so a script that spells
    // out `--swa-full false` keeps working.
    let inert = ContextCompatArgs {
        swa_full: Some(false),
        ..args()
    };
    inert.resolve().expect("the default is inert");
}

#[test]
fn the_group_parses_from_a_command_line_on_a_bare_parser() {
    use clap::Parser;

    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        ctx: ContextCompatArgs,
    }

    let parsed = Probe::try_parse_from(["probe", "--context-shift", "--keep", "32"])
        .expect("the b10621 spellings parse");
    assert_eq!(parsed.ctx.context_shift, Some(true));
    assert_eq!(parsed.ctx.keep, Some(32));

    let parsed = Probe::try_parse_from(["probe", "--no-context-shift"]).expect("parses");
    assert!(parsed.ctx.no_context_shift);

    // The b10621 toggle spelling takes no value; the optional BOOL is for the
    // environment variable's truthiness.
    let parsed = Probe::try_parse_from(["probe", "--swa-full"]).expect("parses");
    assert_eq!(parsed.ctx.swa_full, Some(true));
}
