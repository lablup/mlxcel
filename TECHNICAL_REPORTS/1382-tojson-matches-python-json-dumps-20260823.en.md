# Technical Report: PR #1382 - make the chat-template `tojson` filter match Python `json.dumps`

**Date**: 2026-08-23
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust (minijinja, serde_json)
**Risk Level**: Medium (changes the rendered prompt for every template that calls `tojson`, which is most tool-capable templates)

---

## Executive Summary

mlxcel rendered chat templates with minijinja's builtin `tojson`. Chat templates published on HuggingFace are written against `transformers`, where Jinja's `tojson` is CPython `json.dumps` with `ensure_ascii=False` prefilled. The two are not the same filter, and the gap produced two distinct failures: templates calling `tojson` with `json.dumps` keyword arguments aborted the render entirely and fell back to a generic prompt with no tools, and templates calling it bare produced JSON that differed from the provider's own tokenization by whitespace.

This PR replaces the builtin with an mlxcel-owned filter that reproduces `json.dumps` byte for byte. The interesting part of the change is not the filter, it is what verifying "byte for byte" required and what that verification then found. Three points are worth carrying forward: the issue's prescribed implementation could not satisfy the issue's own acceptance criteria, the correct nesting-depth ceiling is counterintuitively *higher* than the value it appears to want to match, and reconstructing the changed test fixtures in a reference renderer proved both that the change was correct and that a separate defect had been baked into those fixtures all along.

## 1. Problem Statement

### 1.1 Background

`src/server/chat_template.rs` registered `raise_exception` and `strftime_now` on its minijinja environment and left `tojson` to the builtin. minijinja's builtin takes `indent` and calls `Kwargs::assert_all_used`, so any other keyword argument is a hard error, and it serializes through `serde_json::to_string`, which emits the compact form with no space after `,` or `:`.

Published templates do not respect either constraint:

- `poolside/Laguna-XS-2.1` calls `tojson(ensure_ascii=False)`.
- `thinkingmachines/Inkling-Small` calls `tojson(sort_keys=true, separators=(",", ":"))`.
- Many templates, Muse Glimmer among them, call a bare `tool | tojson` and get `", "` / `": "` under the reference renderer.

### 1.2 Existing Issues

The two failure modes have very different visibility, and the quieter one is the more damaging.

The loud failure is a rejected keyword argument. The render aborts, and mlxcel degrades the request to a generic `User:/Assistant:` prompt. That silently drops the tool declarations and the tool-call history, so the model is asked to use tools it was never shown. The degradation is a fallback, not an error, so nothing surfaces to the caller.

The quiet failure is separator spacing. The render succeeds and the output is valid JSON, but the token ids differ from those the model provider trained and evaluated against. Nothing is malformed, nothing logs, and the only symptom is slightly worse tool selection.

### 1.3 Risk Assessment

Medium, and mostly on the way in rather than the way out. The filter is on the path of every chat request that renders tools, so an error in it is an error in every prompt. That is why the verification standard for this change was set at differential equivalence against CPython rather than at a set of hand-written expectations.

## 2. Change Summary

Six files, plus this report.

| File | Change |
|---|---|
| `src/server/chat_template_json.rs` (new) | The filter: option resolution, the value walker, CPython string escaping, CPython float formatting, and the three bounds |
| `src/server/chat_template.rs` | Registers the filter inside `configure_environment`; module docs describe the template environment |
| `src/server/mod.rs` | Declares the new module |
| `src/server/muse_glimmer_template_tests.rs` | Two pinned render digests updated |
| `src/server/chat_request_tests.rs` | One pinned argument string updated |
| `docs/supported-models.md` | New subsection on the filter, its arguments, its limits, and its consequences |

## 3. Technical Decisions

### 3.1 The issue's prescribed implementation could not meet the issue's acceptance criteria

The issue's implementation plan said to convert through `serde_json::to_value` and walk the resulting `serde_json::Value`. The same issue required that non-finite floats render bare as `NaN` / `Infinity` / `-Infinity`, matching CPython.

Those two requirements are incompatible. `serde_json`'s `impl From<f64> for Value` is:

```rust
Number::from_f64(f).map_or(Value::Null, Value::Number)
```

`Number::from_f64` returns `None` for any non-finite input, so the conversion turns `NaN` into `null` before the walker ever sees it. No amount of care downstream recovers the distinction.

The filter therefore walks the `minijinja::Value` directly. This also solved a second problem the `serde_json` route would have introduced: keeping `2` and `2.0` apart. `minijinja::Value::is_integer()` matches only the integer representations and never `F64`, whereas an `i64::try_from` on the value truncates integral floats, which would have rendered `2.0` as `2`.

The general lesson is that an issue's implementation plan is a hypothesis about how to satisfy its acceptance criteria, and the two can contradict each other. The acceptance criteria win.

### 3.2 CPython float `repr` diverges from Rust's `{}` on three independent axes

This is the highest-risk code in the change, so it is a standalone function with its own tests rather than an inline branch. Rust's `Display` for `f64` and CPython's `repr` are both shortest-round-trip, but they disagree on presentation in three ways that compound:

1. **The `.0` suffix.** CPython writes `1.0`; Rust writes `1`.
2. **The exponent thresholds.** CPython switches to exponent form when the decimal exponent is `<= -4` or `> 16`. So `0.0001` stays fixed but `0.00001` becomes `1e-05`, and `1e15` stays fixed but `1e16` becomes `1e+16`. Rust picks different boundaries.
3. **The exponent format.** CPython always signs the exponent and pads it to at least two digits (`1e-05`, not `1e-5`).

The implementation takes the shortest round-trip digits from Rust's `LowerExp`, which matches CPython's `_Py_dg_dtoa` mode 0, and then reassembles the presentation itself.

### 3.3 Dropping HTML escaping is a real behavior change, not a spacing change

minijinja's builtin escapes `<`, `>`, `&` and `'` on the way out. `transformers` does not. Removing that escaping was necessary for equivalence, but it is a larger change than the issue described, and it deserves to be recorded as such: a JSON Schema `pattern` full of comparison operators now reaches the prompt as itself.

This was checked for a security consequence and has none in this codebase. There is no `text/html` response and no `Html(` constructor anywhere in `src/server/`, so the rendered prompt has no HTML sink. The `Value::from_safe_string` on the way out is also inert for an independent reason: the template is registered under the name `chat`, and minijinja's default auto-escape callback only enables HTML escaping for `.html`, `.htm` and `.xml` names.

### 3.4 The depth ceiling has to be higher than the limit it looks like it should match

The first implementation capped nesting at 128 "matching `serde_json`'s own default recursion limit". That reasoning is appealing and wrong.

`chat_request` re-parses `function.arguments` out of its wire string and splices the parsed result back into the message tree at roughly depth 4. That inner parse gets a *fresh* 128-level budget of its own. So an assembled `messages` value can legitimately reach about 132 levels while every individual parse stayed inside its own limit.

At a cap of exactly 128, a template that serializes a whole message rather than just the arguments object would error on a valid request. The error aborts the render, which falls back to the generic prompt, which drops the tool declarations. That is precisely the failure this filter exists to prevent, so the cap would have reintroduced the bug through the fix.

The ceiling is now 192, and the reasoning is pinned mechanically rather than left in a comment:

```rust
const _: () = assert!(MAX_DEPTH > 128);
```

A compile-time assertion is the right instrument here specifically because the mistake is attractive. Someone reading `192` next to a mention of `serde_json` will be tempted to "correct" it to `128`; this makes that a build failure rather than a silent regression that only appears on deeply nested tool schemas.

### 3.5 Bounds are set to catch amplification, not to police input

Three limits were added after review: `indent` clamped to 64 columns, output capped at 64 MiB, and the depth ceiling above. All three sit far above any legitimate render, and the reason is asymmetric cost. Crossing a limit fails the render, and a failed render costs the prompt its tool declarations. A tight bound would trade a denial-of-service risk for a silent capability regression, which is the worse of the two.

`indent` in particular is clamped rather than rejected, for the same reason: a template asking for generous indentation wants pretty output, and refusing it outright would cost more than honoring it at 64 columns.

The output cap matters more than it first appears. `tojson(indent=4)` is not a hypothetical: the Llama 3.1, Llama 4 Scout, Granite 3.3 and Granite Vision templates all call it, and two of those are on the project's own recommended-test-checkpoint list. Pretty-printing a deeply nested structure amplifies its input substantially, and `ensure_ascii` expands non-ASCII text about sixfold on top of that.

## 4. Verification

### 4.1 Differential fuzzing rather than hand-written expectations

"Byte-identical to `json.dumps`" is a claim about a large input space, and a hand-written test set only samples the cases the author already thought of. The filter was instead ported into Python and compared against the real `json.dumps`:

- About 400,000 random f64 bit patterns plus targeted sweeps across every threshold (decimal exponents from -330 to +320, integral floats, subnormals, `f64::MAX`, `0.1 + 0.2`) against CPython `repr`. Zero mismatches.
- 3,840,000 comparisons over random nested structures crossed with `ensure_ascii`, four `indent` settings, four `separators` variants and both `sort_keys` values, with payloads covering C0 controls, U+007F, non-ASCII, astral-plane characters, `<>&'`, quotes and backslashes. Zero mismatches.

### 4.2 Reconstructing the moved fixtures instead of trusting the delta

Three pinned assertions moved: two Muse Glimmer render digests and one `chat_request` argument string. A moved pin is the standard place for a real regression to hide behind an expected change, so the movements were not accepted on the grounds that they looked right.

Both renders were reconstructed in real Jinja2 3.1.6, using `transformers`' actual `tojson` for the new form and an emulation of the old minijinja builtin (compact separators plus HTML escaping) for the old form. The reconstruction reproduced **both** the old and the new digests bit for bit. The character-level diff between them is 13 inserted spaces and nothing else, which matches the 5 commas and 8 colons in `weather_tool`'s schema and the observed +13 bytes on both tests.

That is a stronger result than "the new digest is correct". It establishes that the old digest was exactly "compact plus HTML-escaped" and the new one is exactly `json.dumps`, with no third difference hiding in either.

### 4.3 Gate

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8294 passed, 0 failed, including 5804 root-lib and 1521 `mlxcel-core` tests. `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings` and `cargo fmt --all -- --check` clean.

The `--workspace` scope is worth stating explicitly rather than assuming: the workspace root is itself the `mlxcel` package, so a bare `cargo test` resolves to `-p mlxcel` and never builds the other three members. The 1521 `mlxcel-core` tests in the count are the evidence that this did not happen here.

## 5. Findings from Review

Five findings, all MEDIUM or below, all applied: the `indent` clamp and the reserved padding fill, the output ceiling, the depth headroom, an arity check on `separators` that no longer collects the whole sequence first, and a code comment recording the one known divergence from CPython (`sort_keys` sorts the stringified keys, while CPython sorts the original key objects, so numeric keys order differently).

One finding was accepted and not fixed. Sequences are materialized into a `Vec<Value>` before writing, where the builtin streamed through serde, so a template serializing an enormous lazy range allocates before emitting. It is not reachable from client JSON, only from a template that authors such a range, and the fix reshapes `write_array`'s signature in a way that interacts with both new bounds. It is recorded as a known limitation instead.

## 6. Adjacent Defects Found, Filed Separately

Verifying this change surfaced two pre-existing defects. Neither was folded into this PR, because this PR's value rests on its parity evidence and mixing in an unrelated behavior change would put that evidence in question.

**#1383 (`priority:high`)**: the Python-compatibility shim for `dict.get()` returns `Value::UNDEFINED` where Python returns `None`. minijinja's `is none` distinguishes the two, so a template's `{%- if end_turn is none -%}` fallback never fires. The concrete consequence is that every Muse Glimmer prompt ends its last assistant turn with `<|eom|>` instead of `<|eot|>`. This was found precisely because the fixture reconstruction in 4.2 produced a reference render to compare against; the divergence appears identically on both sides of this PR, which is what establishes it as pre-existing. Note that fixing it will move these same digests again.

**#1384 (`type:test`, `priority:low`)**: `server::reasoning_effort_tests` overflows the default 2 MiB thread stack under the `dev` profile while rendering the large Qwen3.8 template. It does not reproduce under `test-fast` or `release`, so the CI gate and the shipping binary are unaffected; it was bisected to before this branch by reverting only the filter registration.

## 7. What Remains Unverified

The issue's validation item (b), a real-checkpoint comparison against `poolside/Laguna-XS-2.1-NVFP4` whose rendered `<available_tools>` block should be byte-identical to `tokenizer.apply_chat_template` from the checkpoint's own config, was not run. That checkpoint requires the NVFP4 loader, which is separate work. Template rendering is covered by Laguna- and Inkling-shaped unit tests instead.

One residual assumption survives the fuzzing: the differential harness drew shortest-round-trip digits from CPython, so it validated the reassembly logic (thresholds, `.0`, exponent form) while assuming Rust's `{:e}` produces the same digit string as CPython's `_Py_dg_dtoa` mode 0. Both are shortest-round-trip with correct rounding, and the committed `every_finite_float_round_trips_through_the_repr` test guards the Rust side.

## 8. Learning Points

- An issue's implementation plan can contradict its own acceptance criteria. Check the plan against the criteria before following it, and when they conflict, the criteria are the requirement.
- "Match the limit that the layer below uses" is wrong whenever values cross that layer more than once. Here the request body is parsed twice with a fresh budget each time, so the assembled value legitimately exceeds the per-parse limit.
- When a fix's failure mode is the same as the bug's failure mode, bounds have to be set generously. A cap that fails the render reintroduces the fallback the change was written to eliminate.
- A moved test pin is where a regression hides behind an expected change. Reconstructing both the old and the new fixture in a reference implementation converts "this delta looks right" into proof, and it can surface unrelated defects that were encoded in the fixture all along.
- Pin a subtle invariant with a `const` assertion rather than a comment when the wrong value is the intuitive one.
