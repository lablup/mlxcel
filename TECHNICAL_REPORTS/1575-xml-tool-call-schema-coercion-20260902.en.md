# Technical Report: PR #1575 - fix(server): type XML tool-call arguments by the request schema

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: pending
**Status**: Completed
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

The Qwen3-Coder and MiniMax M2 tool-call parsers guessed each argument's JSON type from the raw text because they never received the request's `tools`. A `string`-typed parameter carrying `02134` reached the client as the number `2134`, and an `integer`-typed parameter written `5.0` reached it as a float. This PR threads the request tool schema into both parsers, routes their values through the coercion helpers MiniMax M3, GLM-4.7 and LongCat already share, and extends the shared integer rule to accept a zero-fraction float literal.

---

## 1. Problem Statement

### 1.1 Background

Two of the XML tool-call grammars mlxcel supports carry no type information in their markup. Qwen3-Coder emits `<function=NAME><parameter=KEY>VALUE</parameter></function>` and MiniMax M2 emits `<invoke name="NAME"><parameter name="KEY">VALUE</parameter></invoke>`; in both, every value is a bare text run. The only place the intended type exists is the `tools` array the client sent with the request, which is exactly what the OpenAI-compatible chat route already has in hand at parse time.

Three parsers in the same file already solved this. `try_minimax_m3`, `try_glm47` and `try_longcat` take `tools`, resolve the called function's schema with `minimax_m3_function_schema`, and coerce each value by its declared type. The two XML parsers did not, because they sat in a dispatch table typed `&[fn(&str) -> Option<ToolCallParseResult>]`: the table's element type is what made passing `tools` impossible, so the parsers guessed instead.

### 1.2 Existing Issues

- **String-typed values were rewritten**: `coerce_minimax_param` tried an `i64` parse, then `f64`, then a boolean word list, before falling back to a string. A US ZIP code `02134` parses as `i64` 2134, so the leading zero was destroyed. A product code `1e5` became `100000.0`. The literal text `true` became a JSON boolean. A tool whose handler expected a string then received a number, and either failed its own validation or silently used the wrong value.
- **Integer-typed values kept a float form**: a model writing `5.0` for an `integer` parameter produced the JSON float `5.0`. Strict tool implementations reject that, and the schema said unambiguously that the value is an integer.
- **Guesses with no schema basis**: `yes`, `on`, `no`, `off` were coerced to booleans and `none`, `nil` to null. None of these have any grounding in JSON Schema; they turned legitimate string arguments into wrong-typed ones.
- **Duplicate coercion logic**: `coerce_minimax_param` was a second, weaker implementation of the same job as `minimax_m3_coerce_leaf`, so schema handling improvements landed in one and not the other.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| An agentic client receives a number where its schema declared a string, and the tool call fails or acts on a corrupted value | High | Certain for any leading-zero, exponent-like, or boolean-word string argument |
| A tool rejects an `integer` argument delivered as `5.0` | Medium | Occasional (depends on the model's spelling) |
| The two XML grammars diverge further from the three schema-aware ones as coercion evolves | Medium | Certain over time while the logic is duplicated |
| A parser change drops a tool call outright when a value fails to parse | High | Avoided by design: every path ends at the raw string |

---

## 2. Technical Review

### 2.1 Security

**Checklist:**
- [x] Input validation: all parsing operates on untrusted model output; the change adds no new indexing or slicing, only value typing
- [x] Denial of service: the existing `MINIMAX_M2_MAX_CALLS` / `MINIMAX_M2_MAX_PARAMS_PER_CALL` / `QWEN3_CODER_*` caps and the O(N^2) `break` guards on malformed tags are untouched
- [x] No panics added: no `unwrap`, `expect`, or indexing outside the test modules
- [x] No sensitive data logged

**Issues Found:**

| Issue | Severity | Status |
|-------|----------|--------|
| None identified | n/a | n/a |

One incidental improvement: `coerce_minimax_param` allocated a lowercase `String` copy of every parameter value to test three null spellings. `coerce_xml_param` uses `eq_ignore_ascii_case`, which allocates nothing.

### 2.2 Performance

**Checklist:**
- [x] Algorithm complexity: unchanged per parameter, plus one linear scan of the `tools` slice per call and one property lookup per parameter, both matching what MiniMax M3, GLM-4.7 and LongCat already do
- [x] Memory usage: one fewer `String` allocation per parameter value

Tool-call parsing runs once per completion, on a text buffer the size of one model response, so none of this is on a hot path.

### 2.3 Compatibility & Dependencies

- **Breaking changes**: none at the API level. Behavior changes for clients that relied on the old guesses, listed in section 4.1. The two public functions `try_qwen3_coder` and `try_minimax_m2` change signature, but nothing outside `tool_calls::parser` and the in-file tests calls them.
- **New dependencies**: none.
- **Coverage**: both the non-streaming route (`routes/chat.rs`, `parse_tool_calls(&result.text, tools)`) and the end-of-stream route (`parse_tool_calls(&cb.accumulated, tools_ref)`) already pass the request tools, so one change covers streaming and non-streaming alike.

### 2.4 Code Quality

- **Test coverage**: 20 new unit tests; the `server::tool_calls` suite goes from 350 to 370 tests, all passing.
- **Code complexity**: net simpler. A 37-line hand-rolled coercion ladder is replaced by a 6-line delegation to the shared helper, at the cost of a 2-variant enum in the dispatcher.
- **Technical debt**: decreased. One of the two duplicate coercion implementations is gone, and two stale comments in `parse_tool_calls` are corrected.

---

## 3. Technical Decisions

### 3.1 A `FormatParser` enum instead of widening every parser's signature

**Context:**

The dispatch table is an ordered list, and its order is load-bearing: `try_functionary_v31` must be tried before `try_qwen3_coder` (both open with `<function=`, and v3.1 declines a non-JSON body), and `try_qwen3_coder` before `try_functionary_v32`. Only 2 of the 15 entries need `tools`, but an array literal admits exactly one element type.

**Alternatives considered:**

| Option | Pros | Cons |
|--------|------|------|
| Give all 15 parsers a `tools` parameter | One uniform table type, no wrapper | 13 parsers gain an argument they ignore; every call site and test in the file changes; a reader cannot tell which parsers actually consult the schema |
| Move the two parsers out of the table, next to `try_minimax_m3` | No table change at all | Silently changes dispatch order relative to Functionary v3.1 and v3.2, which is the one property the table exists to encode |
| **Chosen: a `FormatParser` enum with `Plain` and `WithTools` variants** | Order preserved exactly; the schema-aware entries are visible at a glance; no parser gains an unused argument | One small enum and a `run` method; entries carry a wrapper name |

**Rationale:**

The table's meaning is its order, so the option that preserves the order literally, line for line, is the safe one. The enum also documents which grammars are schema-aware, which is real information a maintainer needs when adding the next format. `parse_qwen3_coder_still_runs_after_functionary_v31` pins the ordering property that the second option would have broken.

**Trade-offs:**

Each entry now carries a `Plain(...)` or `WithTools(...)` wrapper, which is slightly noisier than a bare function name. The dispatch cost is one `match` per format tried, which is irrelevant next to the string scanning each parser performs.

### 3.2 `null` overrides the declared type; nothing else does

**Context:**

`coerce_xml_param` keeps exactly one rule ahead of the schema: the bare word `null`, in any casing, becomes JSON null even for a `string`-typed parameter.

**Rationale:**

These grammars have no other spelling for a JSON null. A `<parameter=x>null</parameter>` element is the only null a model can write, and both parsers have always emitted one there, so removing the rule would have been a regression for every model that uses it. The narrower reading (a string-typed `null` is the four-character string) would also make the null unreachable whenever the client declares a type.

**Trade-offs:**

A model that genuinely wants the string `"null"` cannot express it in these grammars. That was already true before this PR, and the schema-aware grammars (MiniMax M3, GLM-4.7) do keep `"null"` as a string under a string schema, so the two families disagree on this one input. The XML behavior was kept because changing it is a regression, not a fix, and it is out of the issue's scope.

### 3.3 `integer` normalizes `5.0`, but `number` keeps the written form

**Context:**

The shared `M3Type::Integer` arm previously accepted only `i64` and `u64` literals, so an `integer`-typed `5.0` fell through to the loose fallback and stayed a float. Extending the arm raised the neighbouring question of what `number` should do with an integral literal.

**Rationale:**

Under `integer` the declared type settles it: the value is an integer, so `5.0` is `5`, and the float spelling is the model's formatting, not data. Under `number` both spellings are valid and the schema expresses no preference, so the parser preserves what the model wrote (`5` stays `5`, `5.0` stays `5.0`). Preserving it also keeps the XML parsers' pre-existing output for `number`-typed integral values unchanged, which a blanket float conversion would have altered.

**Trade-offs:**

The float acceptance is bounded by 2^53. Past that an `f64` cannot represent every integer, so converting would round silently; the rule declines instead, and the value stays a float through the fallback. That is a deliberate refusal to guess rather than a gap.

### 3.4 GLM-4.7 and LongCat take the integer rule, not the whole typed path

**Context:**

`coerce_kv_value` (GLM-4.7 and LongCat) honored only a `string` schema and sent every other type to the loose fallback. The issue asked for `integer`-typed `5.0` to become `5` in every XML grammar.

**Rationale:**

Adding an `Integer` arm to `coerce_kv_value` delivers exactly the requested fix. Routing the function through `minimax_m3_typed_coerce` instead would have changed how those two grammars treat `enum`, `anyOf`, `object`, `array` and `boolean` schemas as a side effect, which is a much larger blast radius than the issue justifies and is not covered by their existing tests. The shared helper is `parse_integer_literal`, so the rule cannot drift between the call sites.

**Trade-offs:**

GLM-4.7 and LongCat remain less schema-strict than MiniMax M3. That gap is now explicit in the `coerce_kv_value` doc comment rather than implicit, and closing it is a separate, testable change.

---

## 4. Implementation Details

### 4.1 Behavior change table

| Schema | Raw value | Before | After |
|--------|-----------|--------|-------|
| `{"type":"string"}` | `02134` | `2134` | `"02134"` |
| `{"type":"string"}` | `true` | `true` | `"true"` |
| `{"type":"string"}` | `1e5` | `100000.0` | `"1e5"` |
| `{"type":"integer"}` | `5.0` | `5.0` | `5` |
| `{"type":"integer"}` | `5.5` | `5.5` | `5.5` (loose fallback, call kept) |
| `{"type":"number"}` | `5` | `5` | `5` |
| `{"type":"object"}` | `{not json` | `"{not json"` | `"{not json"` |
| none | `5` / `true` / `null` | `5` / `true` / `null` | unchanged |
| none | `yes` / `on` / `none` | `true` / `true` / `null` | `"yes"` / `"on"` / `"none"` |
| none | `[1, 2]` | `[1, 2]` | `[1, 2]` |

One further no-schema difference: the fallback now runs a full JSON parse instead of attempting one only when the text starts with `{` or `[`, so a schema-free value written as a JSON literal is parsed as that literal. This is what MiniMax M3, GLM-4.7 and LongCat already did, so the four grammars now agree.

### 4.2 Key code changes

**File: `src/server/tool_calls/formats.rs`**

```rust
// Before
fn coerce_minimax_param(value: &str) -> serde_json::Value {
    let lower = value.to_lowercase();
    if lower == "null" || lower == "none" || lower == "nil" { return Value::Null; }
    if let Ok(i) = value.parse::<i64>() { return Value::Number(i.into()); }
    // ... f64, then "true"/"1"/"yes"/"on", then "{"/"[" JSON, then String
}

// After
fn coerce_xml_param(raw: &str, schema: Option<&serde_json::Value>) -> serde_json::Value {
    if raw.eq_ignore_ascii_case("null") {
        return serde_json::Value::Null;
    }
    minimax_m3_coerce_leaf(raw, schema)
}
```

**Reason for change:** the type belongs to the schema, not to the shape of the text. `minimax_m3_coerce_leaf` already implements the declared-type path, including `enum`, `anyOf`/`oneOf` and array-item resolution, and already ends at the raw string so an unparseable value never drops the call.

```rust
// After: the shared integer rule
fn parse_integer_literal(raw: &str) -> Option<serde_json::Value> {
    if let Some(v) = parse_exact_integer_literal(raw) {
        return Some(v);
    }
    let f = raw.parse::<f64>().ok()?;
    if f.is_finite() && f.fract() == 0.0 && f.abs() <= INTEGER_FROM_FLOAT_LIMIT {
        return Some(serde_json::Value::Number((f as i64).into()));
    }
    None
}
```

**Reason for change:** `integer` should accept the integral value a model spelled as a float, but only where the conversion is exact. `INTEGER_FROM_FLOAT_LIMIT` is 2^53.

**File: `src/server/tool_calls/parser.rs`**

```rust
// Before
let parsers: &[fn(&str) -> Option<ToolCallParseResult>] = &[ /* 15 entries */ ];
for parser in parsers {
    if let Some(mut result) = parser(text) { /* ... */ }
}

// After
enum FormatParser {
    Plain(fn(&str) -> Option<ToolCallParseResult>),
    WithTools(fn(&str, Option<&[Tool]>) -> Option<ToolCallParseResult>),
}

let parsers: &[FormatParser] = &[ /* same 15 entries, same order */ ];
for parser in parsers {
    if let Some(mut result) = parser.run(text, tools) { /* ... */ }
}
```

**Reason for change:** the table could not carry a parser that needs `tools`. The order is unchanged, which is the property the table exists to encode.

---

## 5. Learning Points

### 5.1 A collection's element type can be the reason a bug exists

**Concept:**

`try_qwen3_coder` and `try_minimax_m2` did not guess types because guessing was thought correct. They guessed because they lived in an array whose element type was `fn(&str) -> Option<...>`, and the data they needed could not travel through that type. The three parsers that escaped the array got the schema; the two that stayed did not.

**Application in this PR:**

The fix is mostly the container change. Once the table can hold a `WithTools` entry, the parser bodies need four lines: look the function up, and pass the per-parameter schema down two call levels.

**Where else this pattern shows up:**

- A registry of handlers typed to the narrowest signature that the first few members happened to need.
- A trait method that omits a context argument, so implementations reach for globals or heuristics.
- A callback list whose element type freezes at the first use case, and later members work around it.

The signal is a parser or handler that reimplements information it should have been handed.

### 5.2 Preserving the written form is part of correct coercion

**Concept:**

Coercion is usually framed as "make the value fit the type", but when a type admits several representations, the parser should not choose among them. Under `number`, both `5` and `5.0` are valid, so the parser keeps what the model wrote. Under `integer` only one is valid, so `5.0` becomes `5`.

**Application in this PR:**

`M3Type::Number` tries `parse_exact_integer_literal` before the `f64` parse; `M3Type::Integer` uses the wider `parse_integer_literal`. The `qwen3_coder_number_typed_keeps_written_form` and `minimax_m3_number_typed_keeps_written_form` tests assert both directions with `is_i64()` and `is_f64()`, because `assert_eq!(value, 5)` alone does not distinguish an integer from a float in `serde_json`.

---

## 6. Further Learning

### Key terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `minimax_m3_coerce_leaf` | Schema-directed leaf coercion: typed parse, then JSON parse, then loose literal, then raw string | The single coercion path all four XML grammars now reach |
| `minimax_m3_typed_coerce` | Strict schema-directed parse that returns `None` when the declared type does not apply | Its `None` is what lets `anyOf` try the next alternative and what keeps a call alive on a bad value |
| `kv_param_schema` | Looks a parameter up in a function schema's `properties` | The per-parameter half of the lookup, paired with `minimax_m3_function_schema` |
| 2^53 | Largest magnitude at which an `f64` represents every integer | The bound on the zero-fraction float rule |

### Related PRs and issues

- Issue #1336: the specification this PR implements, including the schema table reproduced in section 4.1.
- The GLM-4.7 and LongCat key/value parsers and the MiniMax M3 namespaced-XML parser are the prior art this PR reuses rather than reimplements.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 2 |
| Lines added | +545 |
| Lines deleted | -121 |
| Tests added | 20 |

### Changes by category

| Category | Count | Summary |
|----------|-------|---------|
| Correctness | 2 files | Schema-directed typing for the Qwen3-Coder and MiniMax M2 grammars; the shared integer rule accepts a zero-fraction float |
| Code quality | 1 | `coerce_minimax_param` removed; two stale comments in `parse_tool_calls` corrected |
| Tests | 20 | Schema, no-schema, boundary and dispatcher-order coverage in both files |

### Related commits

| Hash | Type | Message |
|------|------|---------|
| `272afe8` | fix | fix(server): type XML tool-call arguments by the request schema |

---

## 8. Follow-up Actions

### Required

- [ ] Confirm the end-to-end behavior against a real Qwen3-Coder checkpoint (the request and expected `tool_calls[].function.arguments` are in the PR body). This PR was validated by unit tests only; no checkpoint was loaded.

### Monitoring required

- Tool-call arguments reaching clients for the two grammars, specifically string-typed values that look numeric. A client that had adapted to the old coercion (for example by re-stringifying a ZIP code) will now receive the correct string and should stop compensating.

### Future improvements

- Decide whether GLM-4.7 and LongCat should run the full `minimax_m3_typed_coerce` path rather than the string and integer rules only. That is a deliberate scope boundary here, documented on `coerce_kv_value`.
- Incremental per-delta tool-call argument streaming is still out of scope: the stream path parses once at end of stream.
- Schema validation errors (missing `required` keys, `enum` rejection) are still not surfaced to the client; the parser coerces and never rejects.

---

## Appendix

### A. Test results

```
cargo test --profile test-fast --features metal,accelerate --lib server::tool_calls
test result: ok. 370 passed; 0 failed; 0 ignored; 7313 filtered out

cargo test --profile test-fast --features metal,accelerate --lib server::muse_atem
test result: ok. 11 passed; 0 failed; 0 ignored; 7672 filtered out

cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
Finished (no warnings)

cargo fmt --all -- --check
clean
```

The four cases the issue asked to keep green pass unchanged: `qwen3_coder_single_call_multiple_params_with_type_coercion`, `minimax_m2_numeric_params`, `minimax_m2_boolean_param`, `minimax_m2_null_param`.

### C. References

- `docs/code-guidelines.md`: the `// Used by:` convention applied to `coerce_xml_param` and `parse_integer_literal`.
- JSON Schema type keyword: `integer` is a distinct type from `number`, which is why the two arms differ.
