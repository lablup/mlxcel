# PR #2500: Close the remaining streamed vs non-streamed parity gaps

**Date**: 2026-10-20
**Status**: Completed (PR open; open items listed in section 5)

## Executive Summary

PR #2500 (closes #2464) closes eight gaps that PR #2458 (#2394) left between streamed and non-streamed chat output. Six came from the issue body. Two were carried over from PR #2496 and added as an issue comment: Harmony `analysis` joins, and whitespace before a tool-call marker. In each gap a client saw different `content` or `reasoning` depending on whether it streamed. The non-streamed cleaner is the reference. Items 2, 5 and 6 change both paths. Review added two fixes, for whitespace after `</think>` and for role headers with unusual whitespace. 9 files, +799/-278.

## 1. Problem Statement

| # | Gap (main) | Reference behavior |
|---|---|---|
| 1 | gpt-oss under `--reasoning-format none` streamed analysis plus answer as content | Answer only |
| 2 | `<|start|>assistant to=functions.X<|channel|>...`: non-streamed treated it as commentary, the stream suppressed it as a call | A call on both paths |
| 3 | `<tool_call>..</tool_call>\nB` streamed `\nB` | `B` |
| 4 | Several Harmony content messages streamed back to back | Joined with `\n` |
| 5 | MiniMax M3 left `]<]minimax[>[` residue without tools, on both paths | No residue |
| 6 | Kimi-VL `◁think▷` / `◁/think▷` reached content, on both paths | Text routed to reasoning |
| 7 | Several Harmony `analysis` messages streamed back to back (Chat `reasoning_content`, Responses `reasoning`) | Joined with `\n` |
| 8 | `Let me check.\n\n<tool_call>` streamed the trailing whitespace | Trimmed |

## 2. Change Summary

| Area | Change |
|---|---|
| `tool_calls/stream_filter.rs` (+312/-70) | Trailing-whitespace hold (`held_ws`) with drop rules; per-message trim and `\n` join for Harmony content and analysis; `post_close_ws` span; `pending_harmony_role_header` hold; Kimi-VL markers in `CHAT_DELIMITERS` |
| `tool_calls/formats.rs` | Shared `harmony_role_suffix`, `harmony_route(role, header)`, `HARMONY_MAX_ROLE_SUFFIX = 256` |
| `tool_calls/parser.rs` | MiniMax namespaced block stripped before the Hermes pass; Kimi-VL pair in `THINKING_MARKER_PAIRS` and `canonical_thinking_pair` |
| `routes/chat.rs`, `router_front.rs` | `ThinkingDelimiterEcho::thoughts_in_content` used in the chat stream callback, the chat flush and both router_front paths |
| Tests | 9 parity tests in `stream_parity_tests.rs` (+223); a route test in `chat.rs`; 10 filter unit tests updated |
| Docs | `CHANGELOG.md`, `docs/supported-models.md` |

Commits: `4d0a6787` (implementation), `1edd9916` (post-close whitespace), `8adab292` (role-header hold), `2dfca808` (docs).

## 3. Technical Decisions

### 3.1 Item 1: answer only under `none`

Harmony has no canonical thinking-marker pair. Under `none` the stream therefore cannot render analysis as `<think>...</think>` the way it does for marker-based families. The issue decided to match the non-streamed reply. `thoughts_in_content` lets thoughts into content only inside a block opened by an echoed marker. Muse `to=self` reasoning also has no pair, so it follows the same rule. In router_front, the non-streamed `none` reply now carries the markers and the post-close gap, matching `content_with_thinking_block` on the chat route. Before, it appended bare reasoning.

### 3.2 Items 3 and 8: hold whitespace, never retract

The constraint was that the stream stays incremental and never emits text it later takes back. Trimming whitespace before a marker therefore cannot happen after the fact, so the filter holds trailing whitespace until the next fragment decides it:

| Next event | Held whitespace |
|---|---|
| Visible text | Emitted in front of it, so code indentation is preserved |
| A tool call with nothing visible before it, end of stream, or end of a Harmony message | Dropped, where the non-streamed cleaners trim |
| Another whitespace fragment | Kept held |

Cost: a whitespace-only token reaches the client with the next visible text, which can be several tokens later during a run of indentation. Whitespace at the very start of a generation still streams. Dropping it would break the byte match under `--reasoning-format none`.

**Review finding (HIGH, `1edd9916`).** The hold swallowed whitespace that follows a thinking close: `<think>r</think>  \n` with no answer, and the space in `</think> <tool_call>..</tool_call>B`. The non-streamed path keeps both as `post_close_whitespace`. The filter now records a `post_close_ws` span from the close marker to the next delimiter. When that part of the held run is dropped, it is reported as `post_thinking_gap`. Content output does not change.

### 3.3 Items 2 and 2b: one role-header rule for both paths

`harmony_role_suffix` is shared by `try_harmony` and the filter, so a `to=functions.X` recipient in the role header routes as a call on both paths. The Muse header matcher steps aside only when `<|channel|>` appears in the `to=`…`<|message|>` region before any other `<|`. Muse headers never contain `<|channel|>`.

**Security finding (MEDIUM, `8adab292`).** Only the exact `<|start|>assistant to=` was held during streaming. With two spaces, a tab or a newline before `to=`, the role part drained as content before `<|channel|>` arrived, and the call arguments leaked. `pending_harmony_role_header` now holds `<|start|>assistant` followed by whitespace and no `<|`, for up to 256 bytes. `HARMONY_MAX_ROLE_SUFFIX` replaces two different bounds (a 274-byte stream lookback and an unbounded `try_harmony` search). A literal `<|start|>assistant ` in prose is still released at the bound.

### 3.4 Items 4 and 7: trim each Harmony body and join with `\n`

`try_harmony` trims each message body and joins them with `\n`. The filter now does the same for content and for `analysis` reasoning, which also keeps Responses `reasoning_tokens` equal between the two arms. `collect_reasoning_chunked` (128-byte chunks, from PR #2496) is unchanged, so the non-stream replay stays linear.

## 4. Verification

| Check | Result |
|---|---|
| `server::tool_calls::` | 468 passed |
| `server::routes::` | 461 passed, 1 ignored |
| `server::responses`, `server::router_front`, `server::muse_atem`, `server::reasoning_echo` | 56, 29, 11, 30 passed |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | Clean |
| Whitespace hold, 1 MiB of spaces fed per byte | 3.6 s, the same as 1 MiB of letters (linear) |
| 50k Harmony messages streamed | 0.47 s |
| Fuzzing, 20k random inputs (Kimi markers, Hangul, emoji, U+3000, U+00A0) | No panics |

All runs used `MLXCEL_BUILD_METAL=0 --profile test-fast --lib` on macOS. `router_front` and `anthropic_count_tokens_tests` bind a loopback port, so they ran outside the sandbox. All eight item tests fail on main: the source hunks were stashed and the test file kept, with a temporary main-equivalent `thoughts_in_content` for item 1. The two review tests failed on the commit before their fix.

## 5. Open Items

- **Pre-existing performance:** `try_harmony` is quadratic in message count, because each body searches to the end of the text for absent terminators. 50k messages take 18 s; about 40 ms at 32k tokens.
- **Leading whitespace:** a generation's leading whitespace streams, while the non-streamed reply trims it.
- **Not handled:** a prompt that already ends with `<|start|>assistant` leaves a bare ` to=functions.X<|channel|>` that neither path reads. Text before the first Harmony header streams, but `try_harmony` drops it. A primed Kimi-VL `◁think▷` is not recognized by `primed_open_thinking_close_marker`.
- **No cap:** `held_ws` has no cap; it is bounded by output length, which the routes already store.
- **Not run:** GB10 server scenarios.

## 6. Learning Points

- "Never retract" turns every trim rule into a hold rule. Decide what event resolves the held text, and test each resolution at every chunk boundary.
- When two paths each have their own bound for the same structure (a 274-byte lookback against an unbounded search), they disagree on long inputs. One shared constant removes that class of mismatch.
- A parity test across `chunkings()` finds bugs that whole-string tests miss, such as the role-header leak. Most of the gaps here showed up only with per-character feeding.
