# Technical Report: PR #1175 - fix(server): fail requests a template refuses, map reasoning_effort

**Date**: 2026-08-16
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

A chat template that refuses a caller-supplied value through Jinja's `raise_exception` used to be swallowed into `render_simple_fallback` and answered with HTTP `200` from a prompt with no chat framing, no system message, and no tool declarations; the client had no way to tell that answer apart from a real one. This PR turns that refusal into a `400` carrying the template's own message, and separately maps the OpenAI-standard top-level `reasoning_effort` field onto the `reasoning_effort` chat-template kwarg, which was previously accepted and silently dropped. The discriminator between a deliberate refusal and a genuine render failure is a private `TemplateRejection` sentinel that `raise_exception` attaches as the `minijinja::Error` source, recovered by walking the error chain; `ErrorKind::InvalidOperation` cannot serve that role on its own because minijinja raises the same kind at over twenty sites in its own `value/argtypes.rs` and `format_utils.rs` for ordinary type-coercion failures, so keying on it would turn genuine engine problems into `400`s. Review and security passes on the parent PR found two HIGH items, both already fixed before this pass: a warm-up-kwargs divergence where the prompt-cache next-turn warm-up rendered a probe under a different `reasoning_effort` than the bucket's `template_sig` recorded, and a `CHANGELOG.md` gap. This finalization pass found and fixed one further defect the PR itself introduced (an availability regression where two common zero-argument tool-call spellings now hard-fail a request that previously degraded gracefully), closed a log-injection sink in the rejection-message truncation helper, corrected two stale doc claims, and added test coverage the review had flagged as missing.

---

## 1. Problem Statement

### 1.1 Background

Issue #1164, split out of the Qwen3.8-27B qualification (#1163): two defects in `reasoning_effort` handling. First, a chat template's deliberate `raise_exception` refusal degraded to a silent fallback prompt instead of failing the request. Second, the OpenAI-standard top-level `reasoning_effort` field was accepted by no code path at all, so serde dropped it before anything could act on it. Both are reachable from an ordinary OpenAI-compatible client, and together they meant the single most likely call, `reasoning_effort: "high"`, either did nothing (top-level) or silently degraded the prompt (via `chat_template_kwargs`) depending on where the caller put it.

### 1.2 Existing Issues (from the review/security pass, addressed in finalization)

- **Availability regression** (treated as top priority): `tests/fixtures/muse_glimmer/chat_template.jinja`'s `render_atem` macro raises when `tool_call.function.arguments` is not a mapping. `normalize_tool_call_arguments` only rewrote a wire-format `arguments` string into an object when it parsed as valid JSON and that JSON was an object, so `arguments: ""` (the common zero-argument spelling agentic clients echo back) and `arguments: "null"` stayed strings. Before this PR those degraded to a fallback answer; after it they are a hard `400` on input that is valid OpenAI wire format, with a message the client cannot act on. The failure is sticky: the offending `tool_calls` entry lives in conversation history and replays on every later turn.
- **CHANGELOG upgrade note named the wrong cases**: the note listed "an unsupported role, non-alternating roles, a tool declaration it refuses, an unknown kwarg value" as the traffic that turns from `200` into `400`. For the checkpoints in this tree the two highest-traffic new `400`s are neither: a `system` message at any index other than 0, and a conversation with no `user` message at all.
- **Log-injection sink in `reject_if_template_rejection`**: it logs up to 512 chars of partly caller-controlled text into a single-line plaintext `tracing_subscriber::fmt` record; a newline or an ANSI escape in that text rides through into `--log-file` output.
- **Two stale doc claims**: `MAX_MESSAGE_CHARS`'s doc comment said the server-side log line was unchanged by the cap, which stopped being true once the rejection path started logging the truncated message at `INFO` instead of the untruncated `WARN`. `docs/supported-models.md` called `TemplateRejection` a "private sentinel", but the struct is `pub` and nameable outside the crate.
- **Missing test coverage**: no test pinned the 512-char cap, including the UTF-8 boundary case a later refactor to byte slicing could reintroduce a panic on.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Zero-argument tool-call replay hard-fails every later turn in a conversation (availability regression) | Medium-High (breaks a routine agentic pattern this same PR would otherwise have shipped) | Certain before this fix, on any template requiring a mapping |
| Operators misjudge upgrade blast radius from a generic CHANGELOG note | Low (documentation only, no functional impact) | Real for any operator reading the note before upgrading |
| Caller-controlled text forges a fake log line or rewrites terminal state in `--log-file` output | Low-Medium (log integrity, not data exposure) | Real on any deployment with `--log-file` and adversarial clients |
| Stale doc claims mislead a future maintainer | Low (documentation only) | N/A, corrected |
| A later refactor of `truncate_chars` to byte slicing reintroduces a UTF-8 boundary panic | Low today (no such refactor exists), but the guard was absent | Latent until this pass added the pinning test |

---

## 2. Technical Review

### 2.1 Security

The already-fixed HIGH items (warm-up kwargs divergence, CHANGELOG gap) are described in Section 3.2 and were not touched again here beyond verification. This pass's own finding is the log-injection sink in `truncate_chars`'s callers.

**Issues Found:**

| Issue | Severity | Status |
|-------|----------|--------|
| Warm-up kwargs divergence: `render_next_turn_history` derived kwargs by hand and missed the mapped `reasoning_effort`, storing a prefill vector under a `template_sig` bucket it did not match | High | Fixed before this pass (`8dac53ea`) |
| `CHANGELOG.md` missing the `#1164` entry | High | Fixed before this pass (`2bd259b6`) |
| Availability regression: empty/`"null"` tool-call arguments hard-fail instead of degrading | Medium-High | Fixed (`3c880883`) |
| Log-injection sink: unfiltered control characters in truncated rejection text reach a plaintext log record | Medium | Fixed (`e355d4c2`) |

The control-character fix filters rather than rejects, because `truncate_chars`'s two callers (`TemplateRejection::new`, `truncate_key_for_log`) bound text already accepted into the request and headed for a log/error sink, not a request boundary that can afford to refuse. That is a deliberate difference from `src/server/florence2_worker.rs`'s `validate_task_input`, which does reject a control character outright, because it sits at a request boundary.

### 2.2 Performance

None measured or required. Every change in this pass sits on a request-preparation path (JSON normalization, string truncation) rather than a hot inference path; none of it touches tokenization, KV cache, or the generation loop.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none introduced by this pass. The availability-regression fix restores pre-existing graceful behavior for `arguments: ""` and newly extends the same treatment to `arguments: "null"`, which previously stayed a string (unchanged for any template that does not require a mapping) and now normalizes to `{}` for every template. `"[1,2]"`, bare scalars, and malformed/truncated JSON are deliberately left untouched, since there is no safe reading of them as "no arguments".
- **New Dependencies**: none.
- **Compatibility**: the empty/`"null"` normalization applies uniformly regardless of which template is loaded; a template that never inspects `arguments` (the common case) is unaffected. The control-character filter changes the exact bytes of a rejection message only when that message already contained a control character, which no template fixture in this tree's test suite produces.

### 2.4 Code Quality

- **Test Coverage**: added six new unit tests in `chat_request_tests.rs` (empty, whitespace-only, `"null"`, malformed/truncated-still-a-string, plus the existing scalar/array coverage tightened to drop `"null"` from the "stays a string" list now that it is handled separately) and one end-to-end regression test in `muse_atem_roundtrip_tests.rs` that renders a replayed empty-arguments tool call through the real Muse Glimmer fixture template. Added four new unit tests in `chat_template.rs` for the control-character filter and the 512-char cap, including a construction where the 512th character starts a run of 3-byte UTF-8 characters, so a byte-slice cap would land mid-character.
- **Code Complexity**: `normalize_tool_call_arguments` gained one early-exit branch; `truncate_chars` gained one `.map()` in its existing iterator chain. No control-flow changes elsewhere.
- **Technical Debt**: decreased. The two doc corrections remove claims that no longer matched the code; the new tests close a gap the review pass had explicitly named.

---

## 3. Technical Decisions

### 3.1 `TemplateRejection` sentinel, and why `ErrorKind::InvalidOperation` alone is not a safe discriminator

The central design question the parent PR answered: how does mlxcel tell "the template deliberately refused this input" apart from "mlxcel could not render this template", when both currently surface as the same minijinja error kind?

**Alternatives considered:**

| Option | Pros | Cons |
|--------|------|------|
| Key on `minijinja::ErrorKind::InvalidOperation` | No new type, minimal code | Not a safe discriminator: minijinja's own `raise_exception` implementation (which mlxcel registers itself, since minijinja has no built-in `raise_exception`) uses this kind, but so does the engine, at more than twenty independent call sites in `value/argtypes.rs` and `format_utils.rs` alone (a `\|items` conversion on a non-mapping, a `\|tojson` type mismatch, a string-formatting error, and more). Keying on the kind would convert real render failures into `400`s, the opposite mistake from the one being fixed. |
| Match on the error message text | No new type | The template author writes that text; matching it is fragile across templates and brittle to wording changes. |
| **Chosen: attach a private `TemplateRejection` sentinel as the error's `source()`** | Exact, type-level answer; no string matching; survives minijinja's error-wrapping paths (`{% include %}`, `super()`, loop recursion all attach via `with_source` rather than replacing) | One more type, one more `env.add_function` registration |

**Rationale:** the sentinel is attached only inside the `raise_exception` function body mlxcel itself registers on the minijinja `Environment`, so its presence is an exact signal regardless of what `ErrorKind` the wrapping `minijinja::Error` carries or what text the template author chose. `template_rejection_message` recovers it by walking the whole `anyhow::Error` chain (`err.chain().find_map(...)`) rather than checking one level, because minijinja's VM annotates a propagating error with file/line information in place rather than replacing it, and the few paths that do wrap attach the original through `with_source`.

The type's privacy is deliberately partial: `TemplateRejection` is `pub` (reachable as `mlxcel::server::chat_template::TemplateRejection` since the containing module is `pub mod chat_template`), but its constructor (`fn new`, not `pub fn new`) and its field are private. That means external code can name the type and, given a `&TemplateRejection` obtained through the crate's own error chain, read its message through the public `message()` accessor, but it cannot construct one from scratch. Nothing outside mlxcel's own render path can forge a fake rejection and cause `reject_if_template_rejection` to fire on ordinary text.

### 3.2 The warm-up kwargs divergence (found and fixed before this pass, `8dac53ea`)

This is the more consequential of the two review-flagged HIGH items, because it is a live prompt-cache correctness bug rather than a style issue. `render_next_turn_history` (the issue #1144 next-turn warm-up) derived its chat-template kwargs by hand with `extract_request_kwargs` + `merge_server_and_request`, bypassing `resolve_effective_kwargs`, the function this same PR introduced as the single source of truth for the merge. The consequence: it never saw the mapped top-level `reasoning_effort`. The bucket a warm-up probe is filed under is built by `build_prompt_cache_request_context`, whose `template_sig` *does* include the mapped value (since that function already called `resolve_effective_kwargs`). So a request setting top-level `reasoning_effort` against a template that reads it (Qwen3.8) prefilled and stored a vector rendered at the template's `xhigh` default, filed under a bucket whose signature said the effort was, say, `low`. The next turn's real render used `low`, looked up the `low` bucket, and matched nothing past the shared head, silently discarding the warm-up work rather than corrupting output. That is the same class of disagreement the function's own comment three lines below already guarded against for `preserve_thinking`; the fix makes `render_next_turn_history` call `resolve_effective_kwargs` too, so both code paths derive the merge identically.

### 3.3 Empty and `"null"` tool-call arguments: what stays ambiguous and what does not

`normalize_tool_call_arguments`'s existing rule was: rewrite the wire-format `arguments` string into an object only when it parses as valid JSON and that JSON is an object. Everything else, by design, stays a string, because a scalar or array has no safe reading as a mapping. The regression this pass found is that the rule was too narrow: an empty string is not valid JSON at all (parsing it fails outright), so it fell into "stays a string" by default rather than by intent, and the literal string `"null"` parses to JSON `null`, which is valid JSON but not an object, so it also fell into "stays a string".

**Alternatives considered:**

| Option | Pros | Cons |
|--------|------|------|
| Map only the empty string to `{}`, leave `"null"` as a string | Narrower change, matches the issue's minimum ask | `"null"` is the same "no arguments" intent from a client that `JSON.stringify()`s a `null` value rather than an empty string; leaving it unfixed would leave the same template macro raising on an equally common spelling |
| **Chosen: map both the empty/whitespace-only string and `"null"` to `{}`; leave `"[1,2]"`, scalars, and malformed JSON untouched** | Both are unambiguous "there were no arguments" signals; nothing else is | Slightly broader surface than the issue's literal wording, decided explicitly rather than left implicit |

**Rationale:** the discriminator is not "is this valid JSON" but "is this an unambiguous zero-argument spelling". An empty string and `"null"` both are; `"[1,2]"`, a bare scalar, and a truncated payload are not, since guessing at their intent would be exactly the kind of silent value translation the sibling `reasoning_effort` mapping in this same PR explicitly refuses to do (Section 3.1 of the parent PR's own design, mirrored here for arguments).

### 3.4 Control-character filtering: filter, not reject

`truncate_chars` gained a `.map(|c| if c.is_control() { ' ' } else { c })` step ahead of its existing length cap, replacing every control character (including the ANSI `ESC` that opens a terminal escape sequence) with a plain space rather than dropping it or rejecting the whole string. This mirrors neither of `truncate_chars`'s two callers being a request boundary: `TemplateRejection::new` bounds a message a template has already produced, and `truncate_key_for_log` bounds a kwargs key already accepted into the render. Compare `src/server/florence2_worker.rs`'s `validate_task_input`, which does reject a control character outright, because it runs at an actual request boundary that can afford to refuse. Filtering (not dropping) preserves the visible character count, so a value's length is not itself an information leak about where the control character was.

---

## 4. Implementation Details

### 4.1 Zero-argument normalization (`src/server/chat_request.rs`)

```rust
fn normalize_tool_call_arguments(tool_calls: &mut serde_json::Value) {
    let serde_json::Value::Array(calls) = tool_calls else {
        return;
    };
    for call in calls {
        let Some(args) = call.pointer_mut("/function/arguments") else {
            continue;
        };
        let serde_json::Value::String(s) = args else {
            continue;
        };
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed == "null" {
            *args = serde_json::json!({});
            continue;
        }
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
            && parsed.is_object()
        {
            *args = parsed;
        }
    }
}
```

### 4.2 Control-character filter (`src/server/chat_template.rs`)

```rust
fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars().map(|c| if c.is_control() { ' ' } else { c });
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}\u{2026}")
    } else {
        truncated
    }
}
```

Both of this function's callers (`TemplateRejection::new`, `truncate_key_for_log`) are fixed by this one change, which was the point of folding the filter into the shared helper rather than patching each call site.

### 4.3 End-to-end regression coverage (`src/server/muse_atem_roundtrip_tests.rs`)

`atem_replay_with_empty_string_tool_call_arguments_does_not_raise` builds a four-message conversation (user, assistant with a tool call carrying `arguments: ""`, tool response, follow-up user) and renders it through the real Muse Glimmer fixture template, asserting the render succeeds and does not contain the template's own `"Onyx ATEM chat template requires..."` rejection text. This is the test that would have failed before the fix, since `render_atem`'s `{%- if args is not mapping -%}{{- raise_exception(...) }}` fires exactly on this input.

### 4.4 512-char cap and multibyte boundary coverage (`src/server/chat_template.rs`)

`template_rejection_message_multibyte_boundary_does_not_panic` constructs a 511-ASCII-character prefix followed by 200 CJK (3-byte-in-UTF-8) characters, so the 512th character is the first of the multibyte run: a naive `&s[..512]` byte slice would cut at byte 512, one byte into that character's 3-byte encoding. `chars().take()` makes that panic structurally impossible today; the test pins the behavior so a future refactor to byte slicing would fail it immediately rather than passing until it meets adversarial input in production.

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed (parent feature commit `90f979b7`) | 14 |
| Files changed (warm-up kwargs fix, pre-existing, `8dac53ea`) | 2 |
| Files changed (CHANGELOG gap fix, pre-existing, `2bd259b6`) | 1 |
| Files changed (availability regression fix, this pass, `3c880883`) | 3 |
| Files changed (log-injection filter, this pass, `e355d4c2`) | 1 |
| Files changed (doc corrections, this pass, `18ed6fe4`) | 2 |
| Lines added / removed (this pass, combined) | +223 / -14 |
| Tests added (this pass) | 6 (`chat_request_tests`) + 1 (`muse_atem_roundtrip_tests`) + 4 (`chat_template::tests`) = 11 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Availability fix | 1 | `normalize_tool_call_arguments` maps empty/`"null"` arguments to `{}` |
| Security hardening | 1 | `truncate_chars` filters control characters, closing both of its call sites |
| Documentation accuracy | 3 | `MAX_MESSAGE_CHARS` doc comment, `TemplateRejection` privacy wording, `CHANGELOG.md` upgrade note |
| Test coverage | 3 | Zero-argument unit + end-to-end tests, 512-char cap + UTF-8 boundary + control-character tests |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `90f979b7` | fix | fail requests a template refuses, map reasoning_effort (parent PR) |
| `8dac53ea` | fix | resolve warm-up kwargs through the shared helper (pre-existing HIGH fix) |
| `2bd259b6` | docs | add the #1164 CHANGELOG entry (pre-existing HIGH fix) |
| `3c880883` | fix | treat empty and "null" tool-call arguments as no-argument calls |
| `e355d4c2` | fix | filter control characters out of truncated rejection text |
| `18ed6fe4` | docs | name the concrete #1164 upgrade-note cases, fix sentinel wording |

---

## 7. Follow-up Actions

### Required

- [ ] None; every item this finalization pass was asked to address is fixed and tested.

### Future Improvements (recorded as known limitations, not fixed here)

- `src/server/router_front.rs:657` maps a template rejection to `500` rather than `400` on the disaggregated router-front surface. Real (the OpenAI SDK retries `5xx` and not `4xx`, so one bad request becomes three plus a false server-error alert), but pre-existing and on a different surface than this PR touches; tracked as #1176, deliberately not fixed here.
- The Responses API's `reasoning.effort` remains advisory and is not mapped onto the template kwarg, so a Responses client sending `effort: "high"` at a template that would reject it still gets a silent `200`, unlike the `400` the chat-completions field now produces. Documented in `docs/responses-api.md`, deliberately deferred as a separate scope decision.
- The CLI has the same silent-degradation shape at `src/commands/generate.rs`'s `apply_user_chat_template` (`.unwrap_or_else(\|_\| user_prompt.to_string())`). Lower stakes since the operator sees the output directly, and out of this issue's server scope.

---

## Appendix

### A. Test Results

- `cargo test --release -p mlxcel --lib --features metal,accelerate server::chat_request`: 87 passed, 0 failed.
- `cargo test --release -p mlxcel --lib --features metal,accelerate server::muse`: 17 passed, 0 failed (includes the new `atem_replay_with_empty_string_tool_call_arguments_does_not_raise`).
- `cargo test --release -p mlxcel --lib --features metal,accelerate server::chat_template`: 104 passed, 1 ignored (the local-models audit, which requires `models/`), 0 failed.
- `cargo test --release -p mlxcel --lib --features metal,accelerate server::reasoning_effort_tests`: 19 passed, 0 failed.
- `cargo fmt --check`: clean on every file touched in this pass.
- `cargo clippy --release -p mlxcel --lib --features metal,accelerate --tests -- -D warnings`: clean.

### B. References

- Issue #1164 (specification), issue #1163 (the qualification run that surfaced it)
- Issue #1176 (tracks the `router_front.rs` `500`-vs-`400` gap, explicitly out of scope here)
- `src/server/chat_template.rs` (`TemplateRejection`, `template_rejection_message`, `truncate_chars`), `src/server/chat_request.rs` (`normalize_tool_call_arguments`, `reject_if_template_rejection`, `resolve_effective_kwargs`)
- `tests/fixtures/muse_glimmer/chat_template.jinja` (`render_atem`, the macro whose mapping requirement this pass's availability fix satisfies)
- PR #1175 review and security comments
