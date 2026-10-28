# PR #2700: Make the disaggregated serving and router e2e suites pass

**Date**: 2026-10-28
**Status**: Validation complete; PR open as a draft (report written on the PR branch before merge)
**Languages**: Rust, Markdown
**Risk Level**: Medium (touches the connection pool every cluster transport uses and the decode node's stop-token set; no wire or config change)

## Executive Summary

Issue #2695 found that the real-process disaggregated tests, which are `#[ignore]` and outside CI, failed on main: `disaggregated_serving_e2e` 1 of 1 and `disaggregated_router_e2e` 4 of 7. There were four independent causes, one in test code and three in product code. This PR fixes all four, each with a regression test that fails on main: the test secret helper keyed its client from different bytes than the spawned roles, the connection pool reused idle sockets whose peer had exited, the router omitted `usage.prompt_tokens_details`, and the decode node ignored the checkpoint's `config.json` EOS. It also found that the two MiniCPM "byte_fallback" e2e tests never produced a byte-fallback token, and adds a case to each that does, proven against `tokenizer.json`.

No assertion in either e2e file was weakened. The only change to an existing assertion is diagnostic: `post_completion` now returns the router's error body so a failing assertion shows it. At head `db334ac9`, all suites pass: `disaggregated_serving_e2e` 2/2, `disaggregated_router_e2e` 7/7 with no soft-skips, `disaggregated_e2e` 51/51, and the lib suite 1565/0.

## 1. Problem Statement

### 1.1 Background

The disaggregated serving path splits one request across processes: a router accepts the HTTP request, sends the prompt to a prefill node over the cluster TCP transport, the prefill node hands its KV state to a decode node, and the decode node streams the continuation back. Single-node serving is the reference contract, so any difference between a router response and a single-node response is a router bug. The real-process e2e tests check exactly that, but they need checkpoints and a GPU, so they run by hand. Recent cluster-auth work (#2650, #2674, #2684) changed the transport underneath them without a run of these suites.

### 1.2 Existing Issues

- **Handshake secret mismatch (test bug).** `TestClusterSecretFile::new` generated 32 random bytes, wrote their 64-character hex encoding to the secret file, and keyed `auth()` from the raw 32 bytes. Spawned roles resolve `--cluster-secret-file` through `ClusterSecret::from_file`, which uses the file contents verbatim (the documented `openssl rand -hex 32` contract). The in-test client and the nodes derived different HMAC keys, so `disaggregated_two_process_handoff_matches_single_node` failed with "secret mismatch". The router tests were unaffected because every party there reads the same file.
- **Router 500 after a prefill node dies.** `ConnectionPool::acquire` accepted an idle connection whenever `peer_addr()` succeeded. That stays true in CLOSE_WAIT after the peer process exits, and the first write into such a socket also succeeds. The router fails over only when `transport.send` returns `Err`, so the first post-failure request was written into a dead socket and waited for a first token until the handoff timeout (600 s by default). Rerun with `MLXCEL_DECODE_TIMEOUT=30` on unmodified code, the body was `{"error":{"message":"timed out waiting for the prefill first token"}}`, confirming the inferred cause before any fix. The test failed in 2 of 3 unmodified runs, depending on which prefill node received the first post-failure request.
- **Router omits `usage.prompt_tokens_details`.** Since #1473, single-node chat and completions report `prompt_tokens_details.cached_tokens` whenever the prompt cache is enabled, which is the default. The router's four usage builders (chat and completions, non-stream and stream usage chunk) never emitted it, so the non-stream completion bodies differed only in `usage`.
- **Decode node ignores the configured EOS.** MiniCPM-2B in llama format loads as `Llama3Model`, whose `eos_token_ids()` returns the Llama 3 ids `[128001, 128009]`, both outside this checkpoint's 122753-token vocabulary. Its real EOS (`</s>` = 2) comes only from `config.json` through the scheduler's `config_eos`. Single-node admission and the prefill handoff both call `merge_config_stop_tokens`; the decode-side `ingest_handoff_as_active` did not, and the router sends an empty `stop_token_ids`. The decode node generated past `</s>` until the budget ran out: router `completion_tokens` 16 against single node's 1.
- **Byte-fallback tests that did not reach byte fallback.** Both MiniCPM "byte_fallback" tests used a "café" prompt, but `▁café` and `▁Café` (69435) are single vocabulary pieces. The token-count path for `<0xXX>` byte pieces, which the tests are named after, was never exercised end to end.

### 1.3 Risk Assessment

| Risk if left unfixed | Impact | Likelihood |
|------|--------|------------|
| A prefill node crash turns the next request routed to it into a 600 s hang and HTTP 500, instead of a failover | High | High (any node exit while the router holds an idle pooled connection) |
| Checkpoints whose EOS only `config.json` declares run to `max_tokens` on the decode node, returning extra text and wrong `completion_tokens` / `finish_reason` | High | Medium (llama-format exports with a non-Llama-3 vocabulary) |
| Router `usage` differs from single node, breaking clients that read `prompt_tokens_details` | Low | High |
| The hand-run e2e suites stay red, so later regressions in the same area are invisible | Medium | High |

## 2. Technical Decisions

### 2.1 Fix the test helper, not `ClusterSecret::from_file`

**Context:** The client and the nodes disagreed on the key. Either side could be changed to agree.

| Option | Pros | Cons |
|--------|------|------|
| Make `from_file` hex-decode the file | Helper unchanged | Changes the operator-facing contract (`openssl rand -hex 32 > cluster.secret` would yield a different key than today) and breaks every deployed secret file |
| **Chosen: key the helper from the exact bytes it wrote** | Product unchanged; the helper is now correct by construction | None |

The helper builds `ClusterSecret::from_bytes(hex.into_bytes())` from the same string it writes, and exposes `secret()` so the router auth fixtures planned in #2692 can reuse it. The `to_hex` doc comment now states that a harness handing the hex string to a child hands over a new secret, which is the trap the original helper fell into. A follow-up commit (`86e0a899`) creates the file with `create_new` and mode 0600 on unix, so spawned roles stop logging the `chmod 600` warning.

### 2.2 Liveness check: a direct non-blocking `MSG_PEEK`

**Context:** The pool needs to know whether an idle stream's peer has closed, without consuming bytes (the stream carries AEAD-sealed frames with implicit sequence numbers since #2684) and without blocking under the pool mutex.

| Option | Pros | Cons |
|--------|------|------|
| Keep `peer_addr().is_ok()` | No cost | Cannot see CLOSE_WAIT; the bug |
| `TcpStream::peek(...).now_or_never()` through tokio (the issue's proposal) | Uses the reactor API | Returns `None` until the runtime has itself observed the FIN, so the answer depends on reactor timing |
| Health-monitor-driven eviction | No per-acquire syscall | 3 s probe interval leaves a window; the request in that window is still lost |
| **Chosen: `socket2::SockRef::from(&stream).peek(&mut [MaybeUninit<u8>; 1])`** | One `recv(MSG_PEEK)` on the already non-blocking fd; deterministic; consumes nothing and leaves codec and tokio readiness untouched | One syscall per `acquire`; cannot see a FIN behind unread bytes |

`ProtectedStream::peer_closed` maps `Ok(0)` (orderly shutdown) and any error other than `WouldBlock` / `Interrupted` (for example a reset) to "closed", and `WouldBlock` or unread bytes to "alive". `acquire` drops a closed idle stream (it was never counted in `active`) and tries the next one, then falls through to `connect_with_retry`. When the node is gone, that connect is refused, `send` returns `Err`, and the router's existing failover in `start_handoff` runs. No router code changed for this item.

The rejected `now_or_never` route matters for anyone revisiting this: it looks equivalent and passes most of the time, which is how a flaky failover test would come back.

### 2.3 `cached_tokens: 0` under the router's own prompt-cache flag

**Context:** Single node emits `prompt_tokens_details` exactly when its prompt-cache store exists, which `resolve_prompt_cache_store` builds when `config.prompt_cache.is_enabled()`.

| Option | Pros | Cons |
|--------|------|------|
| Carry a cached-token count in `ResultFrame` | Would be exact if handoff prefill ever used a cache | Wire change for a value that is always zero today |
| **Chosen: report 0 when `state.config.prompt_cache.is_enabled()`, omit otherwise** | Same predicate as single node; no wire change | The flag read is the router process's, not the worker nodes' |

Zero is exact rather than a placeholder: the handoff prefill sets `already_cached_tokens: 0` and never consults a prompt cache. The completion response and usage-chunk construction moved into `completion_response` and `completion_usage_chunk`, and the chat builders take `cache_enabled`, so all four shapes are unit-tested through one shared `router_usage_json` rule.

### 2.4 Merge `config_eos` at decode intake, leave `Llama3Model` alone

**Context:** The model reports the wrong EOS ids for this checkpoint, and the scheduler already has the correct ids.

| Option | Pros | Cons |
|--------|------|------|
| Make `Llama3Model::eos_token_ids` read `config.json` | Fixes the model's own answer | Touches every llama-format model; out of scope for an e2e repair |
| Have the router forward stop ids | Router knows nothing about the checkpoint | Requires the router to load model config |
| **Chosen: `merge_config_stop_tokens(sampling, &self.config_eos)` in `ingest_handoff_as_active`** | Same call single-node admission and the prefill handoff already make; one line | Correct only when the decode node loads the same checkpoint as the prefill node, which the topology already requires |

The merge runs before `merged_eos_token_ids` and `RowSampler::new`, and `merge_config_stop_tokens` appends only missing ids, so a request that already carries the EOS is unchanged.

### 2.5 Byte-fallback cases checked against the vocabulary

Rather than replacing the "café" prompts (the chat one is the regression for item 4), each test now loops over `MiniCpmCase` entries. A case flagged `byte_fallback` must produce, in the single-node reference, a character that no non-byte piece of `tokenizer.json` contains. `minicpm_vocab_chars` builds that set (asserting `model.byte_fallback == true`), and `assert_emits_byte_fallback` checks the premise before the unchanged parity assertions run. The new cases are " Шарик" for completions (`Ш`, 6 tokens) and "Чай" for chat (`Ч`, 4 tokens). If a future tokenizer or model change makes the case stop exercising byte fallback, the test fails with a message saying so instead of silently passing.

## 3. Implementation by Module

- `src/distributed/tcp_frame.rs`: new `ProtectedStream::peer_closed`.
- `src/distributed/connection_pool.rs`: `acquire` uses `peer_closed` instead of `peer_addr`, with a debug log when it drops a stream.
- `src/distributed/cluster_auth.rs`: `to_hex` doc only.
- `src/server/router_front.rs`: `router_usage_json`, `completion_response`, `completion_usage_chunk`; `chat_completion_json` and `chat_chunk_usage` take `cache_enabled`; the four call sites pass `state.config.prompt_cache.is_enabled()`.
- `src/server/batch/scheduler/handoff.rs`: the `merge_config_stop_tokens` call in `ingest_handoff_as_active`.
- `tests/common/mod.rs`: `TestClusterSecretFile` keys from the written bytes, creates the file 0600, adds `secret()`.
- `tests/disaggregated_router_e2e.rs`: per-case loops for the two MiniCPM tests, vocabulary premise check, error body in `post_completion` and in the two failover assertion messages. Whitespace-insensitive, the file diff is +196/-58; the rest is re-indentation inside the loops.
- `src/server/batch/serving_handoff_parity_tests.rs`: `make_request_with`, `collect_text_and_done`, the MiniCPM EOS test, and `repo_model_dir` falling back to `models/mlx/<name>` so the qwen3 siblings find checkpoints on hosts that use that layout.
- `docs/CONTINUOUS_BATCHING.md`: the failover, `prompt_tokens_details`, and configured-EOS behavior.

## 4. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 13 |
| Lines added | +1057 |
| Lines deleted | -324 |
| New tests | 7 (1 integration, 5 lib unit, 1 `#[ignore]` real-model lib) plus 2 new e2e cases |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Product fix | 3 | Pool dead-peer detection, router `prompt_tokens_details`, decode-side configured EOS |
| Test fix | 2 | Secret helper keying (and 0600 mode), byte-fallback cases that reach byte fallback |
| Test diagnostics | 1 | `post_completion` returns the error body |
| Documentation | 2 | `docs/CONTINUOUS_BATCHING.md`, `to_hex` doc |

### New tests

| Test | Guards |
|------|--------|
| `test_secret_file_auth_matches_server_resolution` | Helper key equals `ClusterSecret::from_file` key (not ignored, runs in CI) |
| `acquire_discards_idle_connection_whose_peer_closed` | Pool returns `Err`, not a stale stream, after the peer exits |
| `peer_closed_probe_is_non_consuming` | Probe leaves an unread frame intact and reports a drained, closed peer |
| `send_to_peer_that_closed_its_pooled_connection_fails` | Transport-level twin: `send` returns `Err` so callers fail over |
| `router_usage_reports_zero_cached_tokens_when_prompt_cache_enabled` | All four router usage builders |
| `router_usage_omits_prompt_tokens_details_when_prompt_cache_disabled` | Same four, cache disabled |
| `serving_handoff_parity_stops_on_config_eos_minicpm` | Decode continuation stops on `</s>`; prefill count plus decode count equals single node |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `af4723d0` | fix | fix(distributed): make disaggregated serving and router e2e pass |
| `86e0a899` | test | test: create the test cluster secret file owner-only |
| `db334ac9` | merge | Merge remote-tracking branch 'origin/main' (image-generation ports only; no overlap with this PR's files) |

## 5. Compatibility

- No wire-format, CLI, or config change.
- Router responses gain `usage.prompt_tokens_details: {"cached_tokens": 0}` when the router's prompt cache is enabled (the default). This is additive and matches single node.
- A decode node now stops on the checkpoint's configured EOS. For checkpoints where the model's built-in ids already include it, nothing changes.
- `ProtectedStream::peer_closed` uses `socket2` (already a dependency with the `all` feature); `SockRef` is cross-platform, but the behavior was exercised only on Linux.

## 6. Validation

Fail-first evidence on the fix-absent tree (GB10, `--profile test-fast`, `MLXCEL_REQUIRE_MODELS=1`, `MLXCEL_SDPA_DETERMINISTIC=1`, under `gpu-lock`):

- Secret test: the two 32-byte signatures differ.
- Pool and transport tests: "acquire handed out an idle connection whose peer had closed it" and "send to a peer that closed its pooled connection returned Ok". With `acquire` reverted to `peer_addr().is_ok()` during review, both fail again.
- MiniCPM handoff test: decode ids `[69435, 2, 1345, 72, 2, ...]` (16 tokens, text " Café</s>  .</s> ...") against reference `[69435]`.
- Router failover e2e: HTTP 500 with "timed out waiting for the prefill first token" (2 of 3 unmodified runs).

With the fix: the failover e2e passed 5 of 5 runs, the three new socket tests passed 25 of 25 repeats, and the MiniCPM handoff test reports reference " Café" (1 token, `stop`) and handoff " Café" with decode `completion_tokens` 0 (EOS was the first decode step), so 1 + 0 = 1.

Orchestrator pre-merge gate at `db334ac9`:

| Gate | Result |
|------|--------|
| fmt, version, env-var docs, kernel dtype keys, kernel port dispatch, binary assets, Windows paths checks | all exit 0 |
| Contract tests | 27 + 4 passed |
| clippy `-D warnings` | clean |
| lib (`--profile test-fast`) | 1565 passed, 0 failed |
| `remote_runtime` / `server_remote` / `multi_stage_smoke` | 6 / 5 / 1 passed |
| `disaggregated_serving_e2e --include-ignored` | 2 passed |
| `disaggregated_router_e2e --include-ignored` | 7 passed (51.7 s), no soft-skips |
| `disaggregated_e2e` | 51 passed |

## 7. Residual Risks and Follow-up

- **Failover latency is the reconnect backoff.** The router's transport uses the default `PoolConfig` (6 connect attempts, 100 ms base delay doubling), so the first request routed to a dead node waits about 3.1 s of backoff before `send` errors and failover runs. That replaces a 600 s hang, but a shorter retry budget for the router role may be worth considering.
- **A node that dies after accepting a request** still costs that request the full handoff timeout. Detection here only covers the idle-connection reuse case; `docs/CONTINUOUS_BATCHING.md` says so.
- **Unread bytes hide a FIN.** `peer_closed` reports "alive" when data is waiting, even if the peer then closed. Only `rpc_call` responses (used by `distributed::bench`) can leave such bytes on a pooled outbound stream; the disaggregated handoff path never does. Pre-existing, LOW.
- **`active` leaks on a failed send.** When `send` errors after `acquire`, the stream is dropped without `release`, so `active` stays incremented. It only feeds `stats()`. Pre-existing, LOW.
- **`prompt_tokens_details` follows the router's flag.** A router with the prompt cache enabled reports the field even if the worker nodes run with it disabled. The value is always 0, so this is a shape question only.
- **Router chat content differs by a leading space.** For the MiniCPM "café" chat case the router returns " Café" while single node returns "Café". The chat byte-fallback test compares usage, not content, so it does not catch this. Out of scope for #2695 and not yet filed as an issue.
- **`Llama3Model::eos_token_ids` is still hard-coded** to Llama 3 ids. Every serving path now merges `config_eos`, but any code that trusts the model's own EOS list for a llama-format checkpoint with another vocabulary will still be wrong.
- **Not run in CI.** The real-process suites remain `#[ignore]` and need checkpoints (`qwen3-0.6b-4bit`, `minicpm-2b-sft-4bit-llama-format-mlx`, `gemma-3-1b-it-4bit`) and a GPU. Only `test_secret_file_auth_matches_server_resolution` and the lib unit tests run in CI.

## 8. Learning Points

### 8.1 CLOSE_WAIT looks alive from the outside

When the remote process exits, its kernel sends FIN and the local socket moves to CLOSE_WAIT. `getpeername` still succeeds, and a `write` still succeeds because it only copies into the local send buffer; the RST that the peer's kernel answers with surfaces on a later call. A pool that wants to know whether reuse is safe has to read, and a non-consuming `recv(MSG_PEEK)` on a non-blocking socket answers without side effects: `0` means EOF, `EWOULDBLOCK` means nothing pending and still connected.

```rust
let mut probe = [std::mem::MaybeUninit::<u8>::uninit(); 1];
match socket2::SockRef::from(&self.stream).peek(&mut probe) {
    Ok(0) => true,  // peer closed
    Ok(_) => false, // unread bytes, alive
    Err(e) => !matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted),
}
```

### 8.2 Reactor-level peeks depend on what the reactor has seen

Tokio's `peek` waits on readiness the reactor tracks. Polled once with `now_or_never`, it returns `None` until an event loop iteration has registered the FIN, so the same socket state can read as "alive" or "closed" depending on scheduling. Going to the fd directly makes the probe a pure function of kernel state.

### 8.3 Prove a test's premise, not only its assertion

The byte-fallback tests were named, documented, and reviewed as byte-fallback coverage while their prompts never produced a byte piece. Encoding the premise ("the reference output contains a character the vocabulary cannot emit as a piece") as an assertion turns that silent drift into a failure with a clear message.

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| CLOSE_WAIT | TCP state after receiving the peer's FIN, before the local side closes | Why `peer_addr` and the first write succeed on a dead connection |
| `MSG_PEEK` | `recv` flag that returns data without removing it from the queue | Non-consuming liveness probe that keeps AEAD sequence numbers intact |
| Byte fallback | SentencePiece emitting `<0xXX>` pieces for characters absent from the vocabulary | What the MiniCPM count-parity tests are meant to exercise |
| `config_eos` | EOS ids read from `config.json` by `read_eos_token_ids` | The only correct EOS source for MiniCPM-2B in llama format |

### Related PRs and Issues

- Issue #2695: the four failures this PR fixes.
- #2650, #2674, #2684: cluster handshake, role auth, and per-frame AEAD; #2674 introduced `TestClusterSecretFile`, and #2684's AEAD stream is why the liveness probe must not consume bytes.
- PR #2671: fixed the GB10 remote-pipeline parity failures in another hand-run suite.
- Issue #2692: router auth fixtures that will reuse `TestClusterSecretFile::secret()`.
- #387, #398, #708, #1473: router token counts, chat usage, the MiniCPM fixture retarget, and single-node `prompt_tokens_details`.
