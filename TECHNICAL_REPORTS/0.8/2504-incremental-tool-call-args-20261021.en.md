# PR #2504: Stream tool-call arguments incrementally on Chat and Responses

**Date**: 2026-10-21
**Status**: Completed (PR open; unverified items listed in section 5)

## Executive Summary

PR #2504 (closes #2396) streams tool-call argument fragments as the model generates them. Before, mlxcel sent nothing for a call until generation ended and then sent the whole argument string at once. Chat Completions now sends `delta.tool_calls[i].function.arguments` fragments, and Responses sends `response.function_call_arguments.delta` events. This covers Hermes, Mistral `[ARGS]` and the list form, Llama 3 JSON (bare or after `<|python_tag|>`), and Qwen3-Coder XML. The end-of-generation `parse_tool_calls` stays authoritative. One behavior change applies outside streaming: Hermes, Mistral, Llama 3 and Granite now return an object `arguments` as the model's own JSON text. Review found and fixed three char-boundary panics, added panic containment, and made the streamed `response.completed` match the streamed items. 20 files, +3756/-150.

## 1. Problem Statement

Agent clients that render or pre-validate arguments as they arrive saw a long silent gap and then a burst. On Chat, `StreamFilter` suppressed tool markup during generation, and only after generation did the route emit one `tool_call_start` and one `tool_call_arguments` chunk per call. Responses had the same shape: one `FunctionCallArgumentsDelta` carried the full string. Upstream mlx-vlm (Blaizzy/mlx-vlm#2502) now streams fragments.

## 2. Change Summary

| Area | Change |
|---|---|
| `tool_calls/json_scan.rs` (new, 481) | Iterative JSON value-boundary scanner with a depth counter, shared by the final parsers and the streamer |
| `tool_calls/incremental.rs` + `incremental/` (new, about 1690) | `IncrementalToolCallStreamer`; format readers `json_call.rs`, `mistral.rs`, `qwen.rs`; `reconcile.rs` (`plan_final`) |
| `routes/tool_call_stream.rs` (new, 420) | `ChatToolCallStream`, `ResponsesToolCallStream`, fed from the per-token callbacks |
| `routes/chat.rs`, `routes/responses.rs`, `streaming_responses.rs` | Route wiring; `ResponseStreamSender` records each `output_item.done` for the final object |
| `tool_calls/formats.rs`, `parser.rs` | Raw `arguments` slices via `json_scan`; `resolve_tool_name`, `MISTRAL_HEADER_PATTERN`, `xml_param_keeps_raw_string` extracted for sharing |
| Tests | `incremental_tests.rs` (543); `tool_call_stream_route_tests.rs` (342); `json_scan` unit tests |
| Docs | `docs/responses-api.md`, `docs/server-features.md`, `CHANGELOG.md` |

Commits: `fc323904` (streamer), `b4c4f5cb` (routes), `6d1e6191` and `44959fc7` (docs), `eeb9ee0d` (test), `f29a87dc` and `b06f234e` (panic fixes and containment), `5897a60f` (Responses output parity).

## 3. Technical Decisions

### 3.1 Raw arguments text, from one shared scanner

The issue requires that a call's Chat deltas concatenate to exactly the final `arguments`. A streamer can only send model text as it arrives. So the final string must be that same text, not a `serde_json` re-serialization, which would change spacing and escapes. `json_scan` finds the bounds of the `arguments` value. The final parsers slice the raw text only after `serde_json` has validated the whole call object. `object_members` rejects trailing junk, so the slice is exactly `{`..`}` and still loads as JSON. The streamer runs the same scanner byte by byte, so the two paths cannot disagree on the argument text.

Cost: the visible change on non-streamed replies. `{"city": "Seoul"}` now keeps the model's space. Granite delegates to `try_hermes`, so it changes too. String-valued `arguments` are unchanged.

### 3.2 Qwen3-Coder: compact JSON built while streaming

Qwen3-Coder XML has no JSON to slice, so the streamer emits compact JSON with keys in generation order, matching `extract_qwen_parameters`. A parameter whose schema type is plain string streams as JSON-escaped fragments. The rule for which parameters stream is `xml_param_keeps_raw_string`, and a test locks it to `coerce_xml_param`. The final parser trims values, so the streamer drops leading whitespace and holds trailing whitespace until more text decides it. Both sides use Unicode `is_whitespace`, so the hold is exact. Other values are buffered until `</parameter>` and coerced.

### 3.3 The streamer applies the final parsers' rules

A call announced with one name and finalized with another is a protocol error, so the streamer reuses the final gates:
- thinking blocks are skipped, including primed thinking
- Mistral and bare JSON are recognized only at the start of visible text
- names go through `resolve_tool_name`
- `match_qwen_name` runs only once its answer cannot change
- the Mistral header uses the shared regex, and the same call caps apply
- once the stream commits to one format, calls in other formats are not streamed
- bare JSON waits for the `arguments` member before `Start`, so a JSON answer that has a `name` key is never announced as a call
- with a named `tool_choice`, calls for other functions keep their index slot and are never streamed

### 3.4 Tool-call id and reconciliation

Chat needs the id on a call's first chunk. `Start` carries it. A Mistral-list `id` is used only when it arrives before `name`, validated with `is_valid_model_call_id` and de-duplicated as `try_mistral_nemo` does. Otherwise the id is minted at `Start`, and a later model id is ignored for that call. `plan_final` writes the streamed id back into the parsed calls, so no second id is minted.

`plan_final` matches streamed calls to parsed calls by index and name:

| Case | Action |
|---|---|
| Match, and the streamed text is a prefix of the final arguments | Send only the unsent rest |
| Match, but not a prefix | `tracing::warn!` with the request id; no further delta |
| Name mismatch | Warn; emit the parsed call in one shot under a fresh index |
| Streamed call absent from the parse | Warn; on Responses close the item as `incomplete` |

### 3.5 Review: panics on untrusted text, and containment

**CRITICAL (`f29a87dc`).** `step_outer` advanced one byte at a time and sliced `&s[i..]`. Any non-ASCII character in visible text (Korean, accents, emoji) panicked in the per-token callback of every streamed request with tools. On Chat this happened inside the `cb_state` lock, which poisons the mutex. The fix steps by `char::len_utf8`. Every marker starts with an ASCII byte, so no marker can begin inside a character.

**HIGH (`b06f234e`).** Fuzzing found two more panics of the same kind: `step_json` on `ObjectStep::Invalid`, and the Mistral `ValueLead` with a multibyte first character. Both now consume whole characters. The same commit wraps `feed` in `catch_unwind`. A caught panic drops that fragment's events, rolls the newest call's `sent` back to what the client received, stops incremental streaming for the request, and logs at error level. The final plan sends the rest. The rollback relies on one invariant: only the newest call receives argument text. A `debug_assert` in `send` checks it.

### 3.6 Review: the Responses final object mirrors the stream

When a call starts mid-prose, `Start` closes the open message item, and text after the call opens a new one. `build_response_object` instead put calls first, merged the message text, and minted new `fc_` ids. SDKs that accumulate by `output_index` or item id would disagree with `response.completed`. After `5897a60f`, `ResponseStreamSender` records every `output_item.done` with its index. For a streamed request that parses tool calls, `response.output` and the stored copy read through `previous_response_id` are built from those records. Streamed requests without tool calls still use `build_response_object`.

## 4. Verification

| Check | Result |
|---|---|
| `server::tool_calls::` | 496 passed |
| `server::routes::` | 483 passed, 1 ignored |
| `server::responses`, `server::muse_atem`, `server::router_front` | 56, 11, 29 passed |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | Clean |
| Fuzzing (security review): 400k inputs, each fed per character, whole and in random chunks | 1.2M runs, no panics after the fixes |
| Qwen string values with NUL, control characters, U+2028, U+FEFF, `"`, `\`, `<` | 150k runs; deltas always concatenate to valid JSON equal to the final parse |
| Throughput, 4-character tokens | 28 to 130 ns per token (1 MB prose, 1 MB values, 20k short calls) |

All runs used `MLXCEL_BUILD_METAL=0 --profile test-fast --lib` on macOS. `server::routes::` and `server::router_front` ran outside the sandbox because they bind loopback ports. The unit tests feed each case per character and under other chunkings, and compare against `parse_tool_calls`. The route tests drive the real handlers through `scripted_streaming_for_route_tests`.

## 5. Unverified and Open Items

- **Not run:** `cargo test --workspace --features metal,accelerate` and the integrated `Qwen3-Coder-30B-A3B-Instruct-4bit` check with the OpenAI Python SDK. This host has no Metal or CUDA.
- **Not streamed:** bare Qwen `<function=` calls without `<tool_call>`, several calls inside one `<tool_call>`, and the Functionary v3.1 and GLM-4.7 bodies. The final parse still sends them in one shot.
- **Known behavior:** a bare Llama 3 object followed by prose can stream and then fail `try_llama3`, which leaves a Chat orphan with `finish_reason: stop`. The issue's disagreement path allows this. If an earlier streamed call fails the final parse and the next call has the same name, index matching can pair them wrongly. The prefix check then stops deltas.
- **Unchanged:** the number of Hermes blocks is uncapped, matching the final parser. Responses streaming ignores `max_tool_calls`, as before. A generation error after `Start` leaves a Responses item `in_progress`.
- **Size:** at +3756 lines the PR is above the repository's 1,000-line split guideline. `incremental.rs` is 550 lines.

## 6. Learning Points

- If the streamed fragments must equal the final value, the final value has to be the raw source text. Re-serialization and incremental output cannot both be exact.
- Sharing one scanner between the streaming and final paths prevents a whole class of disagreement, for the same reason a shared constant did in #2500.
- Code that walks untrusted text byte by byte must step by character before it slices. A panic inside a lock-holding callback is worse than a wrong answer, so contain it and fall back to the authoritative path.
