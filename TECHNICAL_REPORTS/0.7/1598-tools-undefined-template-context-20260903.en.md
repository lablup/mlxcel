# Technical Report: PR #1598 - fix(server): define tools only when the request sends them

**Date**: 2026-09-03
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Completed (weight-free template parity verified against both shipped checkpoints; the HTTP gate on real weights is run centrally)
**Languages**: Rust, Markdown
**Risk Level**: Medium (changes the rendered prompt, and therefore the prompt-cache prefix, for every checkpoint whose template guards on `tools is defined` or `tools is not none`)

---

## Executive Summary

`build_template_context` inserted the `tools` key into the minijinja context unconditionally, and both render paths substituted an empty `Vec<Tool>` when the request carried no tools. Any chat template that decides its tool branch with `tools is defined` or `tools is not none` therefore saw a defined, non-none, empty list and rendered its tool-calling preamble with an empty function list on every plain chat request. Two shipped families do exactly that: DeepSeek V3 derivatives (Youtu-LLM) and Llama 3.1 / 3.2 / 3.3 / 4, which includes the project's base transformer reference checkpoint.

PR #1598 changes `build_template_context` to take `tools: Option<minijinja::Value>` and to insert the key only for `Some`, and normalizes both "no tools" and an explicit `"tools": []` to `None` inside `apply_raw_inner` and `apply_inner`. Undefined rather than `none` is required, not preferred: minijinja's `iterable` test succeeds for `none` while `| length` of `none` errors, so a `none` would abort the Nemotron and MiMo family's renders and silently degrade their prompts to the plain-chat fallback. The result matches transformers (`tools=None`) and llama-server (key unset) for every guard shape in the local corpus.

---

## 1. Problem Statement

### 1.1 Background

Chat templates published on HuggingFace are written against `transformers`, where `apply_chat_template` passes `tools=None` when the caller supplies none. llama-server reaches the same state by a different route: `common_chat_tools_to_json_oaicompat` answers a null JSON value for an empty tool array, and minja's `chat-template.hpp` sets the context key only when that value is non-null. Both engines therefore render a tools-less request against a context where `tools` is absent or `None`, and template authors write their guards on that assumption.

mlxcel did not. `src/server/chat_template.rs` inserted `tools` unconditionally, and the two inner render functions turned `None` into `minijinja::Value::from_serialize(Vec::<Tool>::new())`. The comment beside both call sites explained the empty list as a workaround for `{% if tools is iterable and tools | length > 0 %}`, saying minijinja "still tries to compute `| length` of `none`" despite the short circuit. That reasoning was half right and the conclusion was wrong: minijinja 2.24 does compile `and` as a short circuit (`BinOpKind::ScAnd`), but its `iterable` test is `Value::try_iter().is_ok()`, which succeeds for `none`, so the right-hand side is reached anyway. The workaround fixed that one guard by breaking every `is defined` and `is not none` guard instead.

### 1.2 Measured impact

Measured on `POST /apply-template` with `{"messages":[{"role":"user","content":"The Fibonacci sequence begins with"}]}` and no `tools` key:

| Checkpoint | Guard shape | Rendered before | Oracle (transformers / jinja2 with `tools` undefined) |
|---|---|---|---|
| `models/mlx/youtu-llm-2b-4bit` | `tools is defined and tools is not none` | 76 tokens, with an empty `<\|begin_of_tool_description\|>` block between "available:" and "For tool call returns" | 8 tokens |
| `models/mlx/llama-3.2-1b-4bit` | `{%- if not tools is defined %}{%- set tools = none %}{%- endif %}` then `tools is not none` | 97 tokens, with `Environment: ipython` and the "Given the following functions" user preamble | 40 tokens |

Rendering each checkpoint's own template with Python jinja2 reproduced mlxcel's long output byte for byte only when `tools=[]` was passed, which is the direct confirmation that the empty list, not any other context difference, was the cause. An explicit `"tools": []` in the request body produced the same phantom block, because `effective_tools` forwards the empty slice unchanged.

### 1.3 Consequences

- **Prompt inflation.** The preamble costs 68 tokens on Youtu and 57 on Llama 3.2 for a one-line message, on every request.
- **Wrong steering.** The model is told it may call functions and given the call syntax, with no functions to call. Greedy output diverged from the mlx-lm oracle after 12 tokens on one Youtu prompt and 5 on another; feeding the 8-token oracle prompt through raw `/completion` matched the oracle 32 of 32.
- **Cache pollution.** The preamble sits in `tokens[..prefix_len]` of every prompt-cache key for these models and in the history-boundary render the cache snapshots on, so every entry for an affected model was keyed on a prompt the model should never have seen.
- **Silent.** The rendered prompt is fluent and the request succeeds, so nothing in the logs or the response shape reports the divergence. It is only visible against an external oracle or in `usage.prompt_tokens`.

---

## 2. Technical Review

### 2.1 Root cause

One line: `ctx.insert("tools", tools);` in `build_template_context`, fed by two `match tools { None => Vec::<Tool>::new() }` substitutions. The key's *definedness*, not its value, is what three of the seven guard families in the local corpus branch on, and the code had no way to express "absent".

### 2.2 The guard-shape corpus

Probing the local templates with the pinned minijinja version for each candidate value of `tools` gives the decision table this fix is built on:

| Guard shape | Representative checkpoints | `[]` (before) | `none` | undefined (after) |
|---|---|---|---|---|
| `tools is defined and tools is not none` | youtu-llm-2b-4bit | phantom block | ok | ok |
| `set tools = none` when undefined, then `tools is not none` | meta-llama-3.1-8b-instruct-4bit, llama-3.2-1b-4bit, llama-4-scout-17b-4bit | phantom block | ok | ok |
| `set tools = []` when undefined, then `tools is iterable and tools \| length > 0` | nemotron-3-nano-omni-30b-a3b-reasoning-4bit, nemotron-h-30b-4bit, mimo-v2-flash-4bit | ok | render error | ok |
| `not tools is defined or tools is none` sets `[]`, then the same iterable guard | seed-oss-36b-instruct-4bit | ok | ok | ok |
| `if tools` / `tools and tools is iterable and tools is not mapping` | Qwen 2.5 / 3 / 3.5 and later | ok | ok | ok |
| `tools is defined and tools is not none and tools\|length > 0` | ministral-3b-4bit, mistral-small-4-119b-2603-4bit | ok | ok | ok |
| `tools is defined and tools` | apertus-8b-instruct-2509-4bit, exaone4-1.2b-4bit | ok | ok | ok |

Only the undefined column is clean across every row. The `none` column is the reason this is not a one-character change from an empty vector to `Value::from(())`.

### 2.3 Compatibility and dependencies

No dependency moved. `effective_tools` in `src/server/chat_request.rs` is untouched, which also keeps this PR clear of the in-flight tool_choice work in PR #1581, which edits the same file. `tool_choice: "none"` still reaches the template as no tools, through the same path it always did.

### 2.4 Code quality

The two stale comments that misattributed the empty-list workaround to minijinja short-circuiting are replaced by the actual `iterable` and `length` semantics, the three guard shapes, and a pointer to the issue. `build_template_context` gains a comment stating the invariant that decides which keys may be inserted unconditionally, so the next key added to that block has a rule to be checked against rather than a precedent to be copied.

---

## 3. Technical Decisions

### 3.1 Undefined, not `none`, not an empty list

The decision is forced by the corpus rather than chosen. An empty list breaks the two `is defined` families. `none` breaks the Nemotron and MiMo family, and it breaks it in the worst available way: `render_simple_fallback` catches the render error and substitutes a generic `User:` / `Assistant:` prompt, so the failure surfaces as a quality regression rather than an error. Undefined is the only value that leaves all seven rows on their intended path, and it is also what both reference implementations produce.

A note on the residual risk this leaves: a template that computes `tools | length` with no preceding definedness guard would now error where it previously worked. Such a template cannot ship, because it would raise under transformers' `tools=None` for the same reason, and a static audit of the 196 chat templates under `models/` confirms that all 8 templates computing `tools | length` guard it first.

### 3.2 An explicit empty list means no tools

`"tools": []` is normalized to undefined rather than passed through as a defined empty list. This is llama-server's rule (an empty array becomes a null JSON value before the context is built) and the OpenAI reading of the field. Passing it through would preserve exactly the bug for clients that send the key unconditionally, which is a common SDK shape.

### 3.3 Normalize in the two inner functions, not at the callers

Every render in the tree funnels through `apply_raw_inner` or `apply_inner`: the chat, messages, responses, apply-template and both input_tokens routes, the Muse ATEM stream path, the prompt cache's history-boundary render, and the offline `generate` and `chat` commands. Normalizing at that choke point covers all of them in one place and leaves no caller that can forget the rule. The alternative, a shared helper each caller invokes, reintroduces exactly the per-caller drift the choke point removes.

### 3.4 `enable_thinking` stays defined

The audit the issue asked for covered every key the builder inserts. `enable_thinking` is the one other unconditional key that a template could in principle test with `is defined`, and the Youtu template does read it that way (`enable_thinking is defined and enable_thinking is false`). It stays unconditional anyway, because its definedness is the tested per-request override contract from #686 and #1114, and no shipped template was found where `enable_thinking is defined` alone flips a branch that transformers would not flip. Widening this fix to cover it would trade one measured defect for an unmeasured change to a contract with tests on it.

### 3.5 `tools` stays in `RESERVED_KEYS`

The empty tool set is the server-managed answer to "what tools does this request carry", so a `chat_template_kwargs` entry must not be able to define the key the request deliberately left undefined. Under minijinja's default lenient undefined mode, `test_kwargs_cannot_override_reserved_tools_key` keeps passing without modification: iterating an undefined `tools` yields nothing, which is the same observable result the empty list produced.

---

## 4. Implementation Details

### 4.1 The context builder

```rust
// before
    tools: minijinja::Value,
    ...
    ctx.insert("tools", tools);

// after
    tools: Option<minijinja::Value>,
    ...
    if let Some(tools) = tools {
        ctx.insert("tools", tools);
    }
```

### 4.2 Both render paths

```rust
// before, in apply_raw_inner and apply_inner alike
let tools_val = match tools {
    Some(t) => minijinja::Value::from_serialize(t),
    None => minijinja::Value::from_serialize(Vec::<Tool>::new()),
};

// after
let tools_val = tools
    .filter(|t| !t.is_empty())
    .map(minijinja::Value::from_serialize);
```

The tests-module render helper, which exists to reproduce the production context on a freshly compiled environment, carries the same expression with a comment saying it must stay identical.

### 4.3 The invariant comment

The insert block in `build_template_context` now states the rule that decides conditional against unconditional: insert a key unconditionally only when no shipped template can branch on its mere definedness. It names each key and its verdict, so a future addition is checked rather than copied.

---

## 5. Validation

| Check | Result |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib server::chat_template` | 154 passed, 3 ignored |
| `cargo test ... --lib server::chat_request` | 104 passed |
| `cargo test ... --lib server::routes` / `server::tool` / `server::prompt_cache` | 387 / 384 / 173 passed |
| `cargo test ... --lib server::reasoning_effort_tests` / `server::anthropic_translator` / `server::muse_atem_roundtrip_tests` / `server::muse_glimmer_template_tests` | 28 / 33 / 5 / 6 passed |
| `cargo test ... --lib server::chat_template::tests::local_checkpoint_templates_render -- --ignored` | passed: both checkpoints render the transformers oracle byte for byte |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `python3 scripts/ci/check_cross_repo_refs.py` | clean |
| Static audit of 196 local chat templates | all 8 templates computing `tools \| length` guard it with a definedness or none test first |
| `cargo test --test apply_template_tools_parity -- --ignored` (real weights, HTTP) | pending, run centrally |

### 5.1 The weight-free gate, and what it caught

`ChatTemplateProcessor::from_model_path` reads only the template files, so both shipped checkpoints can be rendered and compared against the oracle with no weights, no GPU and no server. That test is worth more than its size: its first version asserted a prompt the server does not produce, because the Youtu template primes an empty `<think></think>` block when `enable_thinking` is defined and false, and `server::startup` sets that default to true from the tokenizer's think markers. The test now mirrors that derivation per checkpoint. Without it, the discrepancy would have surfaced only in the real-checkpoint HTTP gate.

### 5.2 New tests

Four checkpoint-free unit tests cover the three guard shapes plus the history-boundary render, in both the typed and the raw-JSON path, for `None`, `Some(&[])` and one real tool. `tests/apply_template_tools_parity.rs` asserts the exact prompt strings and the 8 and 40 token counts over HTTP against real weights, and that a request carrying one tool still renders its tool block on both checkpoints. It starts the two servers one after the other inside a single test, because the suite shares one Metal device and two concurrently resident checkpoints is the shape that aborts runs.

---

## 6. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 3 |
| Lines added | 557 |
| Lines removed | 27 |
| Commits | 2 |

### Changes by category

| File | Change |
|---|---|
| `src/server/chat_template.rs` | `build_template_context` takes `Option<minijinja::Value>` and inserts conditionally; both inner render functions normalize empty to `None`; stale comments replaced; invariant comment added; five new tests |
| `tests/apply_template_tools_parity.rs` | New gated HTTP parity test over both shipped checkpoints |
| `docs/llama-server-compat.md` | New subsection under "Chat templates, reasoning, and output parsing" |

### Related commits

- `7d5d80d42` fix(server): leave tools undefined when a request sends none
- `4d66e8c1d` test(server): gate the tools-less prompt without loading weights

### Related issues and PRs

- Closes #1597
- Adjacent, deliberately not touched: #1581 (tool_choice enforcement, issue #1319), which edits `src/server/chat_request.rs`
- Context keys whose contracts this fix leaves alone: #686 and #1114 (`enable_thinking`), #512 (`thinking`), #775 and #819 (`thinking_mode`)

---

## 7. Follow-up Actions

### 7.1 Operational note

The rendered prefix changes for affected models, so the first request per model after upgrading misses in the prompt cache once and rebuilds it. That is a cold miss, not a compatibility break: no key digest version changes and no migration is needed.

### 7.2 Out of scope, filed separately if a checkpoint needs it

- Aligning minijinja's `iterable` test and `length` filter with Python for `none` and undefined. A template that defaults an undefined `tools` to `none` and then uses the iterable guard would still error and fall back; no template in the local corpus does this.
- The definedness of `enable_thinking`, unless a concrete template divergence is found.
- The remaining bf16-tie drift on the second Youtu prompt after the fix, which is not this defect.

### 7.3 Broader lesson

The bug was a workaround whose stated rationale was checkable and wrong. The comment asserted a minijinja behavior ("short-circuit evaluation still tries to compute `| length` of `none`") that a two-line probe against the pinned version disproves, and the fix it justified traded one broken guard family for two. When a comment explains a value choice by citing an engine behavior, the cheap move is to run the engine.

The second lesson is about where the gate lives. The byte-exact prompt for these two checkpoints is verifiable without weights, without a GPU and without a server, because the template files are the whole input. Putting that gate at the unit level made the assertion runnable in milliseconds and caught a wrong expectation before any real-checkpoint run.
