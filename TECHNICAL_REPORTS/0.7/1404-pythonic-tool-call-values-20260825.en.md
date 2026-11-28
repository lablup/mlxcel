# Technical Report: PR #1404 - Parse Python Repr Tool-Call Values

**Date**: 2026-08-25
**Author**: mlxcel contributors
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1404 makes the server's Pythonic tool-call parser accept the representation emitted by LFM2-family checkpoints: single-quoted strings, Python boolean/null literals, quoted commas, and nested list values. It also replaces an unsafe in-band comma sentinel with a direct quote- and bracket-aware scanner and adds parser and streaming regressions.

---

## 1. Problem Statement

### 1.1 Background

The Pythonic tool-call format encodes calls as `name(key=value, ...)`. Existing parsing assumed JSON-like double-quoted values even though some checkpoints emit Python `repr` syntax such as `name(query='hello', enabled=True, extra=None)`.

### 1.2 Existing Issues

- **Quote mismatch**: Single-quoted strings retained their surrounding quotes rather than becoming JSON strings.
- **Literal mismatch**: `True`, `False`, and `None` were returned as strings rather than JSON boolean or null values.
- **Incorrect comma boundaries**: Commas inside quoted strings or bracketed lists could split one argument into multiple fragments.
- **Unsafe intermediate representation**: The initial fix masked list commas with a private-use character, which would have corrupted a legitimate occurrence of that character in model output.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood before fix |
|------|--------|-----------------------|
| Tool arguments reach handlers with incorrect JSON types | High | High for affected checkpoints |
| Quoted text or list values are truncated or rejected | High | Medium |
| User-controlled private-use data is silently rewritten | Medium | Low |

---

## 2. Change Summary

### 2.1 Python Repr Value Conversion

The value parser now recognizes matching single or double delimiters, performs only the matching quote/backslash unescaping required by the protocol, and maps Python's boolean/null literals to JSON values. JSON-compatible legacy inputs keep their existing behavior.

### 2.2 Structural Argument Splitting

Arguments are scanned character by character while tracking the active quote, escape state, and bracket depth. A comma is a separator only outside quotes and at bracket depth zero, so quoted commas and nested list contents remain part of the same `key=value` segment.

### 2.3 Streaming Coverage

Regression tests exercise marker-wrapped calls through the stream filter, including input split across chunks. The Pythonic enter-only marker remains terminal tool-call framing, so ordinary text after the parsed call is not emitted as assistant content.

---

## 3. Technical Decisions

### 3.1 Use a Direct Scanner Instead of an In-Band Sentinel

**Decision:** Preserve original input bytes and detect separators structurally.

**Rationale:** No Unicode scalar value is safe as an internal sentinel when the source text is model-controlled. Direct scanning eliminates collision-driven mutation while reusing the same quote and nesting concepts needed for list parsing.

**Trade-off:** The scanner is longer than a regex substitution, but its state and separator rule are explicit and testable.

### 3.2 Limit the Python Subset

**Decision:** Support the repr constructs observed in the tool-call protocol without implementing a general Python parser.

**Rationale:** Matching quote removal, limited escapes, scalar literals, and list nesting address the checkpoint output while keeping untrusted parsing deterministic and dependency-free.

**Trade-off:** Full Python escape semantics, multiple calls in one response, call-level parentheses nesting, and quoted `)]` in the outer matcher remain unsupported.

---

## 4. Review and Quality Findings

### 4.1 Implementation Review

The correctness review found no unresolved issue after hardening. Focus covered conversion precedence, nesting and escape state, first-call-only compatibility, marker handling, and fragmented streams.

### 4.2 Security and Performance Review

No CRITICAL or HIGH finding remained. The review identified the private-use sentinel collision as a MEDIUM data-integrity issue; the final implementation removed the sentinel and added an exact preservation test. A pre-existing unmarked bare-Pythonic streaming limitation and the outer matcher's quoted `)]` limitation remain outside this issue. The scanner is linear in input length and introduces no new dependency or unbounded secondary allocation.

### 4.3 Compatibility

- **Breaking changes**: None to CLI or HTTP request schemas.
- **New dependencies**: None.
- **Behavior change**: LFM2-style Python repr values now reach tool handlers as correctly typed JSON values.

---

## 5. Validation

- `cargo test --profile test-fast pythonic_ --lib` passed 31 focused tests.
- `cargo test --workspace --profile test-fast --features metal,accelerate` passed the full workspace gate.
- `cargo clippy --workspace --all-targets -- -D warnings` passed.
- `cargo fmt --all -- --check` and `git diff --check` passed.
- CI formatting, lint, dependency policy, crate-version, cross-repository reference, OpenXLA compile, and CLA checks passed before report generation.
- Real-model validation was blocked because no local LFM2 checkpoint was available under `models/`; marker-wrapped parser and fragmented-stream regressions validate the affected protocol boundary synthetically.

---

## 6. Change Statistics

| Item | Value |
|------|-------|
| Files changed | 3 |
| Lines added | 243 |
| Lines deleted | 57 |
| Implementation commits | 2 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `fcf6dc1d5` | fix | Parse Python repr tool-call values |
| `d0ef04b7b` | fix | Harden Pythonic tool arg splitting |

---

## 7. Follow-up Considerations

- Validate a live `/v1/chat/completions` tool-call round trip when an LFM2 checkpoint becomes locally available.
- Extend the outer call recognizer if real checkpoints emit quoted `)]`, nested call parentheses, or multiple Pythonic calls in one response.
- Decide separately whether the stream filter should detect unmarked bare Pythonic calls without increasing false positives in normal assistant text.

---

## References

- Issue #1306: Python repr syntax in Pythonic tool calls
- PR #1404: parser conversion, structural splitting, and streaming regressions
