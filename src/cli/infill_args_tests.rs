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

//! Unit tests for the b10621 fill-in-the-middle flag group.

use clap::Parser;

use super::*;

#[derive(Parser)]
struct Probe {
    #[command(flatten)]
    infill: InfillArgs,
}

fn parse(argv: &[&str]) -> InfillArgs {
    let mut full = vec!["probe"];
    full.extend_from_slice(argv);
    Probe::try_parse_from(full)
        .unwrap_or_else(|e| panic!("{argv:?} must parse: {e}"))
        .infill
}

#[test]
fn an_absent_flag_leaves_prefix_suffix_middle_ordering() {
    assert!(!parse(&[]).spm_infill);
}

#[test]
fn spm_infill_selects_the_suffix_prefix_middle_ordering() {
    assert!(parse(&["--spm-infill"]).spm_infill);
}

#[test]
fn spm_infill_takes_no_value() {
    // b10621 declares the flag value-less; a trailing token must not be
    // consumed as its value.
    assert!(Probe::try_parse_from(["probe", "--spm-infill", "true"]).is_err());
}
