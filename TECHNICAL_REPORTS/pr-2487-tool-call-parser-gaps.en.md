# PR #2487: Repair qwen3_coder, Mistral and pythonic tool-call parser gaps

**Date**: 2026-10-20
**Status**: Completed (PR open; unverified items listed in section 5)

## Executive Summary

PR #2487 (closes #2392) fixes three tool-call parsers in `src/server/tool_calls/formats.rs` that dropped or corrupted calls upstream mlx-lm and mlx-vlm now parse: Qwen3-Coder names when the model omits `>`, parallel and hyphenated Mistral `[ARGS]` calls, and multi-call or nested pythonic lists (LFM2 / LFM2-VL). It also adds `ParsedToolCall.id`, so a valid model-emitted Mistral Nemo id reaches Chat and Responses clients instead of a freshly minted one. 12 files, +563/-62; 453 `tool_calls::` tests pass.

## 1. Problem Statement

All three parsers are pure text-to-struct functions that run once on the finished generation.

- **Qwen3-Coder.** `try_qwen3_coder` took the function name as everything up to the next `>`. For `<function=get_time\n<parameter=tz>UTC</parameter>` that `>` belongs to the parameter tag, so the name became `get_time\n<parameter=tz` and clients rejected it as an unknown tool. Parameter names had the same slice-to-`>` logic.
- **Mistral bracket.** `try_mistral_bracket` matched `^\s*(\w+)\[ARGS\]\s*(\{.*\})` once. Two calls became one call whose arguments swallowed the second, and `get-weather` never matched `\w+`.
- **Mistral Nemo id.** The JSON list form carries an `id`, but `ParsedToolCall` had no field for it and every response path minted a new id, so a client that echoes the model's id could not match results.
- **Pythonic.** `\[(\w+)\((.*?)\)\]` kept only the first capture, and the lazy `.*?` stopped at the first `)]`, which truncated nested values such as `cfg={"k": [1, 2]}` or `p=g(1)`. `split_pythonic_commas` tracked only `[`/`]` depth.

## 2. Change Summary

| Area | Change |
|---|---|
| `tool_calls/formats.rs` (+456/-56) | `match_qwen_name`; looped Mistral header scan with `MISTRAL_MAX_CALLS`; id validation in `try_mistral_nemo`; `scan_pythonic_calls` / `scan_pythonic_list` / `find_pythonic_call_end` replace the regex; `split_pythonic_commas` tracks `()[]{}`; JSON dict literals in `coerce_pythonic_value` |
| `tool_calls/types.rs` | `ParsedToolCall.id: Option<String>`, `Default` derive |
| `tool_calls/mod.rs`, `routes/chat.rs` | Chat ids prefer `call.id` (non-streaming and streaming) |
| `responses_translator.rs`, `routes/responses.rs` | Responses `call_id` prefers `call.id`; the `fc_` item id is still minted |
| `anthropic_translator.rs`, `atem.rs`, `in_process/chat_tests.rs` | `id: None` at construction sites only |
| Docs | `docs/server-features.md`, `docs/responses-api.md`, `CHANGELOG.md` |

Commits: `587a3a99` (implementation), `07175912` (EOF-truncated names), `76f314c2` (review fixes), `5656c5ea` (security-review hardening), `eefb6099` (docs).

## 3. Technical Decisions

### 3.1 Qwen3-Coder name matching follows upstream, with one exception

`match_qwen_name` ports upstream `_name_regex = r"\s*([^\s<>]+)>?"`: the name stops at the first whitespace, `<` or `>`, and one trailing `>` is consumed. A parameter whose name cannot be matched is skipped rather than failing the whole call. The quoted form is kept, but its search for the closing `>` stops at the next `<`.

The exception: a name that runs to end of input with no terminator is rejected. Upstream would accept `<function=read_file` at end of output as a call, while main rejected it, and the existing test `qwen3_coder_malformed_missing_closing_tags_keeps_prior_calls` pins that behavior. A truncated generation must not produce a call with an unknown argument set.

**Review finding (HIGH, fixed in `76f314c2`).** Changing parameter-name failure from `break` to skip-and-advance made the quoted branch rescan the whole tail for `>` on each `<parameter="a` opener. Measured: 294 ms on 0.5 MB against 0.3 ms for the unquoted case. Bounding the search at the next `<` restores linear time; a timing guard test pins it.

### 3.2 Mistral: one JSON value per header instead of one greedy regex

| Option | Pros | Cons |
|---|---|---|
| Greedy `\{.*\}` regex (old) | Simple | Swallows following calls; cannot split parallel calls |
| Brace-depth scanner | No dependency | Must re-implement JSON string escaping |
| **Header regex + `serde_json::Deserializer::into_iter::<Value>()`** | Same as Python `raw_decode`; exact JSON boundary | One header regex per call |

Each `([\w-]+)\s*\[ARGS\]\s*` header is followed by exactly one decoded JSON value, and scanning resumes where it ended. A value that fails to decode ends parsing: calls collected so far are returned, or `None` when there are none. The security review confirmed the scan is linear (100k calls in 17 ms); `MISTRAL_MAX_CALLS = 1024` was added for parity with the sibling parsers.

### 3.3 Model ids are passed through only when they are safe to echo

The security review found that empty, duplicate or arbitrary ids passed straight through. The id cannot inject anything, since it is serialized through serde JSON or SSE, and server code never keys on `call_id`. The risk is client-side: duplicate ids break result-to-call matching. `try_mistral_nemo` keeps an id only when it matches `^[A-Za-z0-9_-]{1,64}$` and is not already used in the result; otherwise the field is `None` and the response path mints an id as before.

### 3.4 Pythonic: a scanner instead of a regex

A regex cannot match balanced brackets, so the fix is a small scanner. It finds an outer `[`, reads `name(`, walks to the matching `)` while tracking quote state and `()[]{}` depth, emits one call, skips `,` and whitespace, and requires the closing `]` right after the last call. Two review-driven corrections:

- **Unterminated lists (MEDIUM, `76f314c2`).** Without the `]` check, prose such as `see [note(appendix) for details` or `[sq(x) for x in range(10)]` produced calls. Main returned `None` for the first.
- **Retry budget (MEDIUM, `5656c5ea`).** Every `[` consumed one of the 64 candidate retries, so 65 `[1] ` prefixes hid a valid call that main's regex found. Only candidates that reach `name(` now count.

The existing guard that declines when the whole body is valid JSON is unchanged, so a JSON call whose string argument contains `[calc(x=1)]` is still claimed by the JSON parser.

## 4. Verification

| Check | Result |
|---|---|
| `MLXCEL_BUILD_METAL=0 cargo test --profile test-fast -p mlxcel tool_calls::` | 453 passed, 0 failed |
| `MLXCEL_BUILD_METAL=0 cargo clippy -p mlxcel --lib --tests -- -D warnings` | Clean |
| `cargo fmt --check` | Clean |
| Fuzzing (security review, temporary module) | About 300k random inputs including multibyte UTF-8, no panics |
| Timing probes | Linear on every adversarial shape tried (pythonic worst cases at about 300 KB: 40 ms or less) |

The tests ran on a CPU-only MLX build on macOS, because this host has no Metal toolchain. The parsers do not touch MLX, so the backend does not affect these results.

## 5. Unverified and Out of Scope

- **Real-checkpoint smoke not run:** Qwen3-Coder-30B-A3B-Instruct-4bit and Devstral-Small-2507-4bit through `/v1/chat/completions` with two tools, streaming and non-streaming.
- **Anthropic surface:** `anthropic_translator.rs` and `routes/anthropic.rs` still mint `toolu_` ids and ignore `call.id`.
- **Python-only literals:** `{'k': True}` and `None` stay raw strings; only JSON-syntax dicts become objects.
- **Workspace gate:** the workspace-wide test run was not executed locally. CI runs clippy on GB10.

## 6. Learning Points

- `serde_json::Deserializer::from_str(..).into_iter::<Value>()` with `byte_offset()` is the Rust equivalent of Python's `json.JSONDecoder.raw_decode`. Use it whenever JSON is embedded in a larger text stream.
- Changing a parser from "stop on the first malformed item" to "skip and continue" can turn a bounded scan into a quadratic one. Check every inner search for an upper bound when making that change.
- A retry cap is a correctness parameter, not only a performance one: what counts as one attempt decides which valid inputs are still found.
