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

//! GBNF front-end tests, written against b10621's observable behaviour rather
//! than against this implementation.

use super::{GbnfError, GbnfVocab, compile_gbnf};

struct FakeVocab;

impl GbnfVocab for FakeVocab {
    fn tokenize_exact_one(&self, text: &str) -> Option<u32> {
        match text {
            "<think>" => Some(151667),
            "<tool_call>" => Some(151657),
            _ => None,
        }
    }
}

fn lark(src: &str) -> String {
    compile_gbnf(src, Some(&FakeVocab)).expect("grammar should compile")
}

fn err(src: &str) -> String {
    compile_gbnf(src, Some(&FakeVocab))
        .expect_err("grammar should be rejected")
        .to_string()
}

#[test]
fn a_literal_string_lowers_to_one_single_code_point_class_per_character() {
    // GBNF stores a string literal as one CHAR element per code point, and the
    // llguidance lexer munches maximally, so merging them into one lexeme is
    // the bug this pins against.
    let out = lark("root ::= \"ab\"\n");
    assert!(out.contains("start: r0"), "{out}");
    assert!(out.contains("\\x{61}"), "{out}");
    assert!(out.contains("\\x{62}"), "{out}");
    assert!(!out.contains("\"ab\""), "{out}");
}

#[test]
fn a_char_class_keeps_its_ranges_and_alternates_in_one_terminal() {
    let out = lark("root ::= [a-z0-9_]\n");
    assert!(out.contains("\\x{61}-\\x{7A}"), "{out}");
    assert!(out.contains("\\x{30}-\\x{39}"), "{out}");
    assert!(out.contains("\\x{5F}"), "{out}");
}

#[test]
fn a_negated_class_lowers_to_a_negated_regex_class() {
    let out = lark("root ::= [^\\n]\n");
    assert!(out.contains("/[^\\x{A}]/"), "{out}");
}

#[test]
fn a_trailing_dash_is_a_literal_not_a_range() {
    // b10621 treats `-` as a range separator only when the next character is
    // not `]`, so `[a-]` is two literals.
    let out = lark("root ::= [a-]\n");
    assert!(out.contains("\\x{61}"), "{out}");
    assert!(out.contains("\\x{2D}"), "{out}");
    assert!(!out.contains("\\x{61}-"), "{out}");
}

#[test]
fn dot_matches_every_code_point_including_newline() {
    let out = lark("root ::= .\n");
    assert!(out.contains("[\\s\\S]"), "{out}");
}

#[test]
fn the_full_escape_table_decodes_and_unknown_escapes_are_rejected() {
    let out = lark("root ::= \"\\t\\r\\n\\\\\\\"\\x41\\u00e9\\U0001F600\"\n");
    for expected in [
        "\\x{9}",
        "\\x{D}",
        "\\x{A}",
        "\\x{5C}",
        "\\x{22}",
        "\\x{41}",
        "\\x{E9}",
        "\\x{1F600}",
    ] {
        assert!(out.contains(expected), "missing {expected} in {out}");
    }
    assert!(err("root ::= \"\\d\"\n").starts_with("unknown escape at"));
    assert!(err("root ::= \"\\u00\"\n").starts_with("expecting 4 hex chars at"));
}

#[test]
fn identifiers_admit_dashes_and_leading_digits_but_not_underscores() {
    // `1a-b` is a legal GBNF rule name; `_` is not a word character, so a rule
    // named `a_b` is `a` followed by an unparseable `_`.
    let out = lark("root ::= 1a-b\n1a-b ::= \"x\"\n");
    assert!(out.contains("\\x{78}"), "{out}");
    assert!(err("root ::= a_b\na_b ::= \"x\"\n").starts_with("expecting newline or end at"));
}

#[test]
fn comments_are_whitespace_but_a_newline_before_the_arrow_is_not() {
    // `#` runs to end of line and counts as whitespace wherever whitespace is
    // allowed. The gap between a rule name and `::=` is parsed with
    // `newline_ok = false`, so a comment there swallows the arrow with the rest
    // of the line and the rule stops being a rule.
    let out = lark("# leading\nroot ::= \"x\" # trailing\n# footer\n");
    assert!(out.contains("\\x{78}"), "{out}");
    assert!(err("root\n ::= \"x\"\n").starts_with("expecting ::= at"));
    assert!(err("root # eats the arrow ::= \"x\"\n").starts_with("expecting ::= at"));
}

#[test]
fn an_empty_alternative_is_legal_and_survives_lowering() {
    // `space ::= | " " | "\n"` appears in every schema-generated grammar.
    let out = lark("root ::= space\nspace ::= | \" \" | \"\\n\"\n");
    let space_line = out
        .lines()
        .find(|l| l.starts_with("r1:"))
        .expect("space rule")
        .to_string();
    assert!(space_line.starts_with("r1: |"), "{space_line}");
}

#[test]
fn repetition_forms_all_parse_and_brace_zero_erases_the_item() {
    for form in ["*", "+", "?", "{2}", "{2,}", "{2,5}", "{ 2 , 5 }"] {
        let src = format!("root ::= \"a\"{form}\n");
        compile_gbnf(&src, None).unwrap_or_else(|e| panic!("{form} rejected: {e}"));
    }
    // `x{0}` drops the item entirely: the grammar matches only the empty string.
    let out = lark("root ::= \"a\"{0}\n");
    assert_eq!(out.trim_end(), "start: r0\nr0:");
}

#[test]
fn repetition_operators_chain() {
    // `a{2}{3}` applies the second repetition to what the first produced, so
    // the sequence is six copies.
    let out = lark("root ::= \"a\"{2}{3}\n");
    let root_line = out.lines().find(|l| l.starts_with("r0:")).unwrap();
    assert_eq!(root_line.matches("\\x{61}").count(), 6, "{out}");
    assert!(err("root ::= *\n").starts_with("expecting preceding item to */+/?/{ at"));
}

#[test]
fn an_oversized_maximum_is_silently_widened_but_an_oversized_minimum_throws() {
    // b10621 rewrites `x{0,5000}` to `x*` without a diagnostic, and rejects
    // `x{5000}` outright.
    compile_gbnf("root ::= \"a\"{0,5000}\n", None).expect("max above the threshold is widened");
    assert!(
        err("root ::= \"a\"{5000}\n").starts_with("number of repetitions exceeds sane defaults")
    );
}

#[test]
fn the_repetition_budget_is_multiplicative_across_nesting() {
    // The budget counts the rules a group actually synthesized, not the
    // repetition counts. `x{0,50}` synthesizes 50 helper rules, so wrapping it
    // in a group carries 51 into the next repetition: 51 * 40 crosses 2000
    // while 51 * 30 does not, and the same two repetitions written in sequence
    // never multiply at all because each item resets the budget.
    compile_gbnf("root ::= \"a\"{0,50} \"a\"{0,50}\n", None)
        .expect("sequential repetitions do not multiply");
    compile_gbnf("root ::= (\"a\"{0,50}){30}\n", None).expect("51 * 30 is under the budget");
    let msg = err("root ::= (\"a\"{0,50}){40}\n");
    assert!(msg.contains("exceeds sane defaults"), "{msg}");
    // A group whose inner repetition synthesizes nothing carries 1, so a large
    // fixed count on both levels stays legal.
    compile_gbnf("root ::= (\"a\"{40}){40}\n", None).expect("fixed counts synthesize no helpers");
}

#[test]
fn token_terminals_lower_to_llguidance_token_ranges() {
    let out = lark("root ::= <[1000]> <think> !<[1000]> !<tool_call>\n");
    assert!(out.contains("<[1000]>"), "{out}");
    assert!(out.contains("<[151667]>"), "{out}");
    assert!(out.contains("<[^1000]>"), "{out}");
    assert!(out.contains("<[^151657]>"), "{out}");
}

#[test]
fn a_token_terminal_without_a_vocab_or_without_a_single_token_is_refused() {
    let no_vocab = compile_gbnf("root ::= <think>\n", None).expect_err("needs a vocab");
    assert!(
        matches!(&no_vocab, GbnfError::Token(m) if m.starts_with("no vocab to parse token at")),
        "{no_vocab:?}"
    );
    // `<[id]>` still resolves with no vocab at all.
    compile_gbnf("root ::= <[7]>\n", None).expect("id form needs no vocab");
    let multi = compile_gbnf("root ::= <not a token>\n", Some(&FakeVocab)).unwrap_err();
    assert_eq!(multi.to_string(), "invalid token '<not a token>'");
}

#[test]
fn left_recursion_is_rejected_in_direct_indirect_and_hidden_forms() {
    for src in [
        "root ::= root \"a\" | \"a\"\n",
        "root ::= a\na ::= b \"x\" | \"a\"\nb ::= root\n",
        "root ::= nul root \"a\" | \"a\"\nnul ::= \"b\" |\n",
    ] {
        let msg = err(src);
        assert!(
            msg.starts_with("unsupported grammar, left recursion detected for nonterminal"),
            "{src} gave {msg}"
        );
    }
    // The check runs over every rule, not just the reachable ones.
    let msg = err("root ::= \"a\"\ndead ::= dead \"b\"\n");
    assert!(
        msg.starts_with("unsupported grammar, left recursion"),
        "{msg}"
    );
    // Right recursion is what repetition desugars to and must stay legal.
    compile_gbnf("root ::= \"a\" root | \"a\"\n", None).expect("right recursion is fine");
}

#[test]
fn a_missing_root_or_an_undefined_reference_is_rejected() {
    assert_eq!(
        err("start ::= \"a\"\n"),
        "grammar does not contain a 'root' symbol"
    );
    assert_eq!(
        err("root ::= missing\n"),
        "Undefined rule identifier 'missing'"
    );
}

#[test]
fn a_realistic_json_grammar_compiles() {
    let src = r#"root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9]{1,2})? ws
ws ::= | " " | "\n" [ \t]{0,20}
"#;
    let out = lark(src);
    assert!(out.starts_with("start: r0\n"), "{out}");
    assert!(out.lines().count() > 10, "{out}");
}

#[test]
fn a_grammar_larger_than_the_source_cap_is_refused_with_the_limit() {
    // The cap is mlxcel's own: b10621 bounds rule EXPANSION with its
    // multiplicative repetition budget and has no source-size limit at all, so
    // a large grammar made of many simple non-repeating rules is accepted
    // upstream and refused here. The number is pinned so it cannot drift
    // silently, and the diagnostic names it.
    let mut src = String::from("root ::= a0\n");
    let mut i = 0usize;
    while src.len() <= super::MAX_GBNF_BYTES {
        src.push_str(&format!("a{i} ::= \"x\" a{}\n", i + 1));
        i += 1;
    }
    src.push_str(&format!("a{i} ::= \"y\"\n"));
    assert!(src.len() > super::MAX_GBNF_BYTES);

    let err = compile_gbnf(&src, None).expect_err("a grammar over the cap is refused");
    assert!(
        matches!(&err, GbnfError::Unsupported(m) if m.contains(&super::MAX_GBNF_BYTES.to_string())),
        "the refusal must name the limit: {err}"
    );
    assert_eq!(super::MAX_GBNF_BYTES, 256 * 1024);
}
