# PR #2520: Responses conversation compaction, phase A

**Date**: 2026-10-21
**Status**: Completed (PR open; refs #2397, phase B pending)

## Executive Summary

PR #2520 adds conversation compaction to the Responses API. It is phase A of #2397. A client calls `POST /v1/responses/compact` (also `/responses/compact`). It gets back its instruction messages, the newest whole turns, and one `compaction` item that carries a model-written summary of the discarded history. Sending `[...output, new message]` to `/v1/responses` replays the summary in place of the discarded turns, and the summary keeps replaying across `previous_response_id` chains. Automatic compaction through `context_management`, the output cap at the remaining context, and the streamed `mlx.compaction.*` events are phase B, in a separate PR that closes the issue. A maintainer reviewed the PR and pushed two fix commits during review. Review and security found two HIGH issues, both fixed. 19 files, +2337/-16, of which about 980 lines are tests.

## 1. Problem Statement

Long agent sessions on `/v1/responses` eventually exceed the model context and fail. OpenAI's Responses protocol and Codex handle this with compaction: an explicit compact call and an opaque `compaction` item that stands in for the discarded history. Before this PR, mlxcel did not mount the endpoint, and a client that replayed a compaction item got a 400. Upstream mlx-vlm added the feature in Blaizzy/mlx-vlm#2408, with follow-ups #2418, #2458 and #2501.

## 2. Change Summary

| Area | Change |
|---|---|
| `responses_compaction.rs` (new) | Split-point selection, the `CompactionBackend` trait, `compact_history` |
| `responses_compaction/capsule.rs` (new) | Capsule encode and decode, replay, instruction dedupe |
| `routes/responses_compact.rs` (new) | Handler and backend: render, count, summary generation |
| `routes/responses_compact_media.rs` (new) | Per-compaction image cache, admitted before any fetch |
| `types/responses_request.rs`, `responses_response.rs`, `responses_translator.rs`, `responses_store.rs` | `compaction` input and output items, replay hook, chain storage |
| `app.rs`, `routes/mod.rs`, `availability_tests.rs` | Both routes mounted; 501 parity for model-less servers |
| Tests | `responses_compaction_tests.rs` (448), `responses_compact_route_tests.rs` (532) |
| Docs | `docs/responses-api.md`, `docs/server-features.md`, `CHANGELOG.md` |

## 3. Technical Decisions

### 3.1 The capsule is encoded, not encrypted

`encrypted_content` is base64url of `{"v":1,"summary":...,"retained":[...]}`. The client already holds the conversation that the capsule summarizes, so encryption would protect nothing. An HMAC key would also need key storage that survives restarts. The server treats the capsule as untrusted input:
- the encoded size is checked before decoding;
- `v` must equal 1;
- a capsule nested in `retained` is refused;
- retained items go through the normal input parser, image gate and limits.

A forged capsule can carry only roles the client could already send in `input`.

### 3.2 Selection never splits a call from its output

Instructions (system and developer messages) are kept in their roles. A split is allowed in two places:
- before a user message while no function call is open;
- right after the output that closes the last open call (mlx-vlm#2501).

A trailing unfinished tool group stays whole, including parallel calls with only some outputs. Binary search picks the earliest split point where the retained part fits `keep_tokens`. The default is 1/16 of the context, capped at 8192. Moving the cut earlier only grows the retained part, so the search is monotone. The latest user message is always kept. Selection alone does not guarantee a shorter result. A post-check does: if the compacted context is not shorter, compaction fails and leaves the context unchanged.

### 3.3 The summary pass reuses the non-streaming worker path

The summary prompt is the discarded prefix plus a fixed handoff instruction. It is generated at temperature 0 with reasoning off. It runs through the same steps as non-streaming `/v1/responses`: busy-slot check, slot accounting, metrics, and the same provider call. Token counts use the `/v1/responses/input_tokens` path. Overflow cases return HTTP 400 with type `exceed_context_size_error` and code `context_length_exceeded`. The overflow cases are a history over one window, a summary prompt that does not fit, and a result that is not shorter. The issue said "413-style". The PR keeps 400 because the server's existing overflow mapping and upstream both use 400.

### 3.4 Replay and chaining (maintainer commit d71e90ad)

The newest capsule wins and older ones are dropped. A stored response keeps the instructions, the capsule and the tail as its canonical input. Later `previous_response_id` hops therefore replay the summary again instead of losing it after one hop. A `conversation` transcript appends only the new turn. Exactly equal system or developer messages are kept once. The compact call refuses a result whose replay would exceed the body limit, so a retained image near the limit cannot produce a result that is unusable the moment it is returned.

### 3.5 Review findings

| Severity | Finding | Fix |
|---|---|---|
| HIGH | `MediaCache` fetched client-named image URLs before checking that the model accepts images, and had no image-count limit. A text-only model would fetch arbitrary URLs again (the hole #1451 closed), and one body could pin thousands of 64 MB images. | `95baf786` (maintainer): image-support, tool and image-count admission run before the first fetch |
| HIGH | Instruction dedupe used `Vec::contains`, O(k²) on every replay of a capsule history | `95baf786`: `HashSet` |
| MEDIUM | The resent system message was duplicated on chained turns | `87db02b0`, `d71e90ad` |
| MEDIUM | Each token count re-read every image URL | `87db02b0`: per-compaction cache, 1 fetch per URL (5 before, in the route test) |

## 4. Verification

| Check | Result |
|---|---|
| `responses_compact` | 28 passed |
| `server::routes::` | 493 passed, 1 ignored |
| `server::responses`, `server::types` | 74, 107 passed |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | Clean |

All runs used `MLXCEL_BUILD_METAL=0 --profile test-fast --lib` on macOS. Route tests drive the real handlers with a scripted backend. They cover the compact output layout, a replayed prompt that contains the summary and not the discarded turns, three-hop chains, conversation append counts, the byte-limit boundary, and one fetch per image URL.

## 5. Unverified and Open Items

- **Not run:** the integrated check with `mlx-community/Qwen3-30B-A3B-4bit` and the OpenAI Python SDK (`client.responses.compact`, then `create`). This host has no Metal or CUDA. Whether the SDK accepts the input-shaped message items in the compact `output` is also unverified.
- **Phase B:** `context_management` auto-compaction, the output cap at the remaining context (mlx-vlm#2458), and the streamed `mlx.compaction.started` / `mlx.compaction.completed` events (mlx-vlm#2418).
- **Known limits (documented):**
  - Retained items appear both in `output` and inside the capsule, as the issue specifies, so `[...output, new]` sends them twice. A client may send only the instruction items plus the compaction item.
  - The newest completed tool group can be summarized away.
  - The summary replays as an assistant turn right after the instructions. Templates that require strict user/assistant alternation may reject it.
  - A call that never gets an output blocks every later split.
- **Remaining MEDIUM (security review):**
  - Two dedupe passes overlap. The whole-context pass can also drop a system or developer reminder after the capsule that exactly repeats an earlier one, while the stored snapshot keeps it.
  - A URL that fails to read is retried on each count.
  - Each count of a request with images takes a worker slot.
- **Size:** at +2337 lines, phase A is above the 1,000-line guideline even after the split. About 980 of those lines are tests and 130 are docs.

## 6. Learning Points

- A cache that prefetches on behalf of a later gate moves the fetch ahead of the gate. Admission checks have to run before the first fetch, not inside the path that was cached.
- When the server stores a chained response, it must store what the next hop needs to rebuild the context, not what the client literally sent. Otherwise a summary survives one hop and is lost on the next.
