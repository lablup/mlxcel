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

//! `--grammar` / `--json-schema` argv-order resolution tests.

use std::ffi::OsString;
use std::path::PathBuf;

use super::{GrammarSourceArg, resolve_grammar_source};

fn cmd() -> clap::Command {
    clap::Command::new("mlxcel-server")
        .arg(clap::Arg::new("grammar").long("grammar").num_args(1))
        .arg(
            clap::Arg::new("grammar-file")
                .long("grammar-file")
                .num_args(1),
        )
        .arg(
            clap::Arg::new("json-schema")
                .long("json-schema")
                .short('j')
                .num_args(1),
        )
        .arg(
            clap::Arg::new("json-schema-file")
                .long("json-schema-file")
                .num_args(1),
        )
        .arg(clap::Arg::new("model").long("model").short('m').num_args(1))
}

fn argv(tokens: &[&str]) -> Vec<OsString> {
    tokens.iter().map(OsString::from).collect()
}

#[test]
fn the_last_grammar_flag_on_the_command_line_wins() {
    let mut c = cmd();
    let chosen = resolve_grammar_source(
        &mut c,
        &argv(&[
            "mlxcel-server",
            "--json-schema",
            "{}",
            "--grammar",
            "root ::= \"a\"",
        ]),
        1,
        Some("root ::= \"a\"".to_string()),
        None,
        Some("{}".to_string()),
        None,
    );
    assert_eq!(
        chosen,
        Some(GrammarSourceArg::Gbnf("root ::= \"a\"".into()))
    );

    let mut c = cmd();
    let chosen = resolve_grammar_source(
        &mut c,
        &argv(&[
            "mlxcel-server",
            "--grammar",
            "root ::= \"a\"",
            "--json-schema",
            "{}",
        ]),
        1,
        Some("root ::= \"a\"".to_string()),
        None,
        Some("{}".to_string()),
        None,
    );
    assert_eq!(chosen, Some(GrammarSourceArg::Schema("{}".into())));
}

#[test]
fn a_value_that_spells_a_grammar_flag_is_not_mistaken_for_one() {
    // `--model --grammar` is a (silly) model path, not a grammar flag.
    let mut c = cmd();
    let chosen = resolve_grammar_source(
        &mut c,
        &argv(&[
            "mlxcel-server",
            "--json-schema",
            "{}",
            "--model",
            "--grammar",
        ]),
        1,
        None,
        None,
        Some("{}".to_string()),
        None,
    );
    assert_eq!(chosen, Some(GrammarSourceArg::Schema("{}".into())));
}

#[test]
fn the_short_spellings_and_the_equals_form_are_recognised() {
    let mut c = cmd();
    let chosen = resolve_grammar_source(
        &mut c,
        &argv(&["mlxcel-server", "--grammar=root ::= \"a\"", "-j", "{}"]),
        1,
        Some("root ::= \"a\"".to_string()),
        None,
        Some("{}".to_string()),
        None,
    );
    assert_eq!(chosen, Some(GrammarSourceArg::Schema("{}".into())));
}

#[test]
fn everything_after_a_double_dash_is_ignored() {
    let mut c = cmd();
    let chosen = resolve_grammar_source(
        &mut c,
        &argv(&[
            "mlxcel-server",
            "--grammar",
            "root ::= \"a\"",
            "--",
            "--json-schema",
        ]),
        1,
        Some("root ::= \"a\"".to_string()),
        None,
        Some("{}".to_string()),
        None,
    );
    assert_eq!(
        chosen,
        Some(GrammarSourceArg::Gbnf("root ::= \"a\"".into()))
    );
}

#[test]
fn no_flag_resolves_to_no_grammar() {
    let mut c = cmd();
    assert_eq!(
        resolve_grammar_source(&mut c, &argv(&["mlxcel-server"]), 1, None, None, None, None),
        None
    );
}

#[test]
fn a_malformed_schema_fails_startup_but_a_malformed_grammar_defers_like_b10621() {
    // b10621 converts a schema inside the argument handler, so a bad one
    // aborts before the server listens.
    assert!(
        GrammarSourceArg::Schema("{not json".to_string())
            .load()
            .is_err()
    );
    GrammarSourceArg::Schema("{\"type\":\"object\"}".to_string())
        .load()
        .expect("a valid schema loads");

    // It never parses GBNF in the handler, so a bad grammar must NOT stop the
    // server; every constrained request fails instead.
    GrammarSourceArg::Gbnf("root ::= [unclosed".to_string())
        .load()
        .expect("GBNF parsing is deferred exactly as it is upstream");
    GrammarSourceArg::Gbnf("root ::= <think>\n".to_string())
        .load()
        .expect("token terminals are deferred to request time");

    // A missing file still aborts: upstream's read_file throws in the handler.
    assert!(
        GrammarSourceArg::GbnfFile(PathBuf::from("/nonexistent/g.gbnf"))
            .load()
            .is_err()
    );
    assert!(
        GrammarSourceArg::SchemaFile(PathBuf::from("/nonexistent/s.json"))
            .load()
            .is_err()
    );
}
