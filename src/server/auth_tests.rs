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

//! Unit tests for b10621 API-key parsing and credential extraction (#1437).

use std::path::PathBuf;

use axum::http::{HeaderMap, HeaderValue, header};

use super::{
    ApiKeys, parse_api_key_file, parse_csv_row, presented_credential, resolve_api_keys,
    unauthorized_response,
};

fn keys(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_string()).collect()
}

// ---------------------------------------------------------------------------
// --api-key CSV parsing
// ---------------------------------------------------------------------------

#[test]
fn a_single_key_parses_to_itself() {
    assert_eq!(parse_csv_row("secret"), vec!["secret".to_string()]);
}

#[test]
fn a_comma_separated_list_yields_each_field() {
    assert_eq!(
        parse_csv_row("a,b,c"),
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

#[test]
fn whitespace_is_not_trimmed() {
    // b10621's parse_csv_row does no trimming, so `--api-key "a, b"` really
    // configures a key with a leading space. Trimming here would accept a
    // credential upstream rejects.
    assert_eq!(
        parse_csv_row("a, b"),
        vec!["a".to_string(), " b".to_string()]
    );
}

#[test]
fn a_quoted_field_may_contain_a_comma() {
    assert_eq!(
        parse_csv_row("\"a,b\",c"),
        vec!["a,b".to_string(), "c".to_string()]
    );
}

#[test]
fn a_doubled_quote_inside_a_quoted_field_is_a_literal_quote() {
    assert_eq!(parse_csv_row("\"a\"\"b\""), vec!["a\"b".to_string()]);
}

#[test]
fn a_quote_in_the_middle_of_an_unquoted_field_is_literal() {
    assert_eq!(parse_csv_row("ab\"cd"), vec!["ab\"cd".to_string()]);
}

#[test]
fn empty_fields_survive_parsing_and_are_dropped_by_resolution() {
    // The parser keeps them (upstream's does too); the `!key.empty()` filter
    // in the flag handler is what drops them.
    assert_eq!(
        parse_csv_row("a,,b"),
        vec!["a".to_string(), String::new(), "b".to_string()]
    );
    let resolved = resolve_api_keys(&keys(&["a,,b"]), &[]).expect("valid");
    assert_eq!(resolved.len(), 2);
    assert!(resolved.accepts("a") && resolved.accepts("b"));
    assert!(!resolved.accepts(""), "an empty field is not a credential");
}

// ---------------------------------------------------------------------------
// --api-key-file parsing
// ---------------------------------------------------------------------------

#[test]
fn a_key_file_yields_one_key_per_line() {
    assert_eq!(
        parse_api_key_file("alpha\nbeta\ngamma\n"),
        vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
    );
}

#[test]
fn blank_lines_and_comments_do_not_become_credentials() {
    assert_eq!(
        parse_api_key_file("# a comment\n\nalpha\n\n#another\nbeta\n"),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn a_comment_marker_only_counts_in_column_one() {
    // Upstream tests `key[0] != '#'`, so an indented hash is a key, not a
    // comment. Reproduced so the same file yields the same set on both.
    assert_eq!(
        parse_api_key_file("  # indented\n"),
        vec!["  # indented".to_string()]
    );
}

#[test]
fn a_key_file_line_is_not_trimmed() {
    assert_eq!(
        parse_api_key_file("  spaced  \n"),
        vec!["  spaced  ".to_string()]
    );
}

#[test]
fn a_crlf_key_file_keeps_the_carriage_return() {
    // std::getline splits on '\n' alone, so upstream keeps the '\r'. Stripping
    // it here would make the same file authenticate on mlxcel and fail on
    // llama-server; `resolve_api_keys` warns about it instead.
    assert_eq!(
        parse_api_key_file("alpha\r\nbeta\r\n"),
        vec!["alpha\r".to_string(), "beta\r".to_string()]
    );
}

#[test]
fn a_file_without_a_trailing_newline_still_yields_its_last_key() {
    assert_eq!(
        parse_api_key_file("alpha\nbeta"),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

// ---------------------------------------------------------------------------
// Resolution across sources
// ---------------------------------------------------------------------------

#[test]
fn no_sources_means_authentication_is_disabled() {
    let resolved = resolve_api_keys(&[], &[]).expect("valid");
    assert!(resolved.is_empty());
}

#[test]
fn cli_and_file_keys_join_one_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("keys.txt");
    std::fs::write(&file, "# team keys\nfile-one\nfile-two\n").expect("write");

    let resolved =
        resolve_api_keys(&keys(&["cli-one,cli-two"]), std::slice::from_ref(&file)).expect("valid");
    assert_eq!(resolved.len(), 4);
    for key in ["cli-one", "cli-two", "file-one", "file-two"] {
        assert!(resolved.accepts(key), "{key} must authenticate");
    }
    assert!(!resolved.accepts("cli-three"));
}

#[test]
fn repeated_flag_occurrences_accumulate() {
    let resolved = resolve_api_keys(&keys(&["first", "second,third"]), &[]).expect("valid");
    assert_eq!(resolved.len(), 3);
    assert!(resolved.accepts("first") && resolved.accepts("third"));
}

#[test]
fn several_key_files_accumulate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let a = dir.path().join("a.txt");
    let b = dir.path().join("b.txt");
    std::fs::write(&a, "from-a\n").expect("write");
    std::fs::write(&b, "from-b\n").expect("write");
    let resolved = resolve_api_keys(&[], &[a, b]).expect("valid");
    assert_eq!(resolved.len(), 2);
    assert!(resolved.accepts("from-a") && resolved.accepts("from-b"));
}

#[test]
fn an_unreadable_key_file_is_a_startup_error_naming_the_path() {
    let err = resolve_api_keys(&[], &[PathBuf::from("/nonexistent/keys.txt")])
        .expect_err("a missing key file must fail startup");
    let text = format!("{err:#}");
    assert!(text.contains("--api-key-file"), "{text}");
    assert!(text.contains("/nonexistent/keys.txt"), "{text}");
}

#[test]
fn a_key_source_that_contributes_nothing_is_a_startup_error() {
    // Upstream would start with authentication silently off on a server the
    // operator meant to protect. That is the one place worth diverging.
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("empty.txt");
    std::fs::write(&file, "# only comments\n\n").expect("write");
    let err = resolve_api_keys(&[], &[file]).expect_err("must fail");
    assert!(
        format!("{err:#}").contains("authentication disabled"),
        "{err:#}"
    );

    assert!(
        resolve_api_keys(&keys(&[""]), &[]).is_err(),
        "an --api-key that parses to nothing must fail too"
    );
}

// ---------------------------------------------------------------------------
// Secret hygiene
// ---------------------------------------------------------------------------

#[test]
fn debug_never_prints_key_material() {
    let resolved =
        resolve_api_keys(&keys(&["canary-do-not-log,second-canary"]), &[]).expect("valid");
    let rendered = format!("{resolved:?}");
    assert!(!rendered.contains("canary"), "{rendered}");
    assert_eq!(rendered, "ApiKeys(2 configured)");
}

#[test]
fn the_unauthorized_body_matches_b10621_and_echoes_no_key() {
    let response = unauthorized_response();
    assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("application/json; charset=utf-8")
    );
}

// ---------------------------------------------------------------------------
// Credential extraction
// ---------------------------------------------------------------------------

fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut map = HeaderMap::new();
    for (name, value) in pairs {
        map.insert(
            axum::http::HeaderName::from_bytes(name.as_bytes()).expect("header name"),
            HeaderValue::from_str(value).expect("header value"),
        );
    }
    map
}

#[test]
fn a_bearer_authorization_header_presents_the_token() {
    let map = headers(&[("authorization", "Bearer secret")]);
    assert_eq!(presented_credential(&map), Some("secret"));
}

#[test]
fn a_bare_authorization_header_presents_itself() {
    let map = headers(&[("authorization", "secret")]);
    assert_eq!(presented_credential(&map), Some("secret"));
}

#[test]
fn x_api_key_is_used_when_authorization_is_absent() {
    let map = headers(&[("x-api-key", "secret")]);
    assert_eq!(presented_credential(&map), Some("secret"));
}

#[test]
fn x_api_key_is_used_when_authorization_is_empty() {
    // Upstream's get_header_value returns "" for an absent header, so an empty
    // Authorization takes the same fallback path as a missing one.
    let map = headers(&[("authorization", ""), ("x-api-key", "secret")]);
    assert_eq!(presented_credential(&map), Some("secret"));
}

#[test]
fn the_bearer_prefix_is_stripped_from_x_api_key_too() {
    let map = headers(&[("x-api-key", "Bearer secret")]);
    assert_eq!(presented_credential(&map), Some("secret"));
}

#[test]
fn no_credential_headers_present_nothing() {
    assert_eq!(presented_credential(&HeaderMap::new()), None);
}

#[test]
fn a_lowercase_bearer_prefix_is_not_stripped() {
    // Upstream compares the literal "Bearer " prefix, case-sensitively.
    let map = headers(&[("authorization", "bearer secret")]);
    assert_eq!(presented_credential(&map), Some("bearer secret"));
}

#[test]
fn an_empty_key_set_accepts_nothing() {
    let empty = ApiKeys::default();
    assert!(empty.is_empty());
    assert!(!empty.accepts(""));
    assert!(!empty.accepts("anything"));
}
