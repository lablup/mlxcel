# PR #2747: Router reasoning_only and reply parity, LLaVA and TP Llama EOS ids

**Date**: 2026-10-30 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust
**Risk Level**: Low (one additive response field on the router; the model-level EOS list grows for LLaVA Llama-backed checkpoints and the tensor-parallel Llama runtime)

## Executive Summary

PR #2719 (issue #2715) left three parity residuals, which issue #2729 collected and this PR closes. The LLaVA text model and the tensor-parallel Llama runtime now report the EOS ids the checkpoint sidecars declare, through a new `loading::merged_eos_token_ids`, so `llava-1.5-7b-4bit` reports `[2]` instead of `[]`. The disaggregated router now sets `message.reasoning_only: true` on non-stream chat replies and `delta.reasoning_only: true` on the streaming finish chunk exactly when single node does, by calling the single-node predicates (`log_if_reasoning_only`, `stream_reasoning_only`) rather than a copy of them. The real-process thinking e2e test gains a 1024-token turn that closes its thinking block, so the reasoning / answer split is finally compared between a router and a single node on a real model.

## 1. Problem Statement

### Two loaders reported a shorter EOS list than the checkpoint declares

Since #2719, `Llama3Model::eos_token_ids()` returns what its `ModelArgs` declare, plus 128001/128009 only for the 128256-token Llama 3 vocabulary. Two loaders build those args from a config that misses the sidecar ids. `build_llava_text_model` parses only `text_config`, and `llava-1.5-7b-4bit` declares no `eos_token_id` there (vocab 32064), so the model reported `[]` although `read_eos_token_ids` resolves `[2]` (`</s>`) from `tokenizer_config.json`. `TensorParallelLlamaModel::from_model_dir` reads the top-level `config.json` and never `generation_config.json` or `tokenizer_config.json`.

Generation still stopped correctly, because the server and CLI merge `read_eos_token_ids` into the sampling stop ids. The gap was for consumers that read `LanguageModel::eos_token_ids()` alone: `suppress_eos_for_cli` in `src/server/engine_probe/cli_engine.rs` and the XTC allowlist in `admission.rs`. The loaders in `src/loading/special.rs` already called `set_eos_token_ids(read_eos_token_ids(..))`, so these two paths were the outliers. #2719's report listed this as a known residual ("An empty list is possible").

### The router never emitted `reasoning_only`

Single node marks a reply `reasoning_only: true` when tokens were generated but all of them went to `reasoning_content`, typically a thinking block cut off by `max_tokens`. Without the flag a client sees `content: ""` and no signal that nothing else is coming. The router's `chat_completion_json` and `chat_chunk_finish` had no such field, so the same request through a disaggregated deployment lost the signal. Failing-first evidence on main 8385f40b: `message.reasoning_only` was `Null` for a body with `reasoning_content: "still thinking"` and `finish_reason: "length"`, and the stream finish chunk had `delta: {}`.

### No real-process test compared reasoning plus an answer

`disaggregated_router_filters_thinking_output` compared router and single-node chat replies byte for byte, but at `max_tokens: 32` qwen3-0.6b never closes `</think>`, so both sides returned `content: ""`. The code path that splits a closed thinking block from a non-empty answer was never compared between two real processes. #2719's report recorded this under "Not verified".

## 2. Technical Decisions

### Merge sidecar ids after the declared ids, in a shared helper

`merged_eos_token_ids(declared, model_dir)` returns the declared ids in their order, then each id from `read_eos_token_ids(model_dir)` that is not already present. It sits in `src/loading/mod.rs` next to `read_eos_token_ids`.

Keeping the declared ids first preserves the order the model already reported, so a checkpoint whose config declared everything sees no change at all. Replacing the list with `read_eos_token_ids` alone, as `loading::special` does, would drop the Llama 3 defaults `ModelArgs::eos_token_ids()` appends for the 128256 vocabulary whenever a sidecar lists fewer ids. A union keeps both sources.

The LLaVA `Llama` arm calls `set_eos_token_ids(merged_eos_token_ids(text_args.eos_token_ids(), model_path))` after `from_weights`. `TensorParallelLlamaModel` gains `set_eos_token_ids`, which sets the same list on every rank, and `from_model_dir` calls it once the model is built. Setting every rank rather than only `ranks[0]` (the rank `eos_token_ids()` reads) keeps the ranks consistent if a later caller reads a different one. The LLaVA `Qwen2` arm is unchanged: `Qwen2Model` has no setter, and adding one was out of scope.

### Reuse the single-node predicates instead of restating the rule

`log_if_reasoning_only` and `stream_reasoning_only` in `src/server/routes/chat.rs` go from private to `pub(crate)`, and the router calls them. Both encode non-obvious conditions: the non-stream check gates on a non-empty shaped `reasoning_content` so a Gemma 4 marker-only output that cleans to `""` does not claim reasoning, and the stream check excludes `finish_reason == "tool_calls"`. A router-local copy would drift the first time either rule changes. This follows the decision #2719 made for response shaping (`shape_plain_chat_reply`): one function called from both sides.

Non-stream: the router already had the raw handoff text and the shaped reply from #2719, so it calls `log_if_reasoning_only(&raw, &shaped.content, shaped.reasoning_content.as_deref(), completion_tokens, finish_reason)` and passes the result to `chat_completion_json`. Under `--skip-chat-parsing` the shaped reply has no reasoning, so the flag stays false.

Stream: single node decides from what the client actually received, not from the raw text alone. The router mirrors that. `emit_filtered` records `saw_reasoning_content` when it sends a non-empty reasoning delta and `saw_content` when it sends non-empty content, including the thoughts-in-content form used by `--reasoning-format none` and `deepseek-legacy`. The handoff callback accumulates the raw text. At finish the router computes `completed && stream_reasoning_only(&raw, saw_content, saw_reasoning_content, finish_reason)`; the `completed` guard matches single node, which passes `false` when generation errored.

The two flags are `AtomicBool` with `Relaxed` ordering rather than `Cell<bool>`. Both closures run inside one spawned task, so there is no cross-thread ordering to enforce, but a `&Cell` held across an `.await` makes the future `!Send` and `tokio::spawn` rejects it.

Both builders add the field only when it is true (`message["reasoning_only"] = true`, `delta["reasoning_only"] = true`) and leave it out otherwise, which is what single node's `with_reasoning_only` produces. A client that compares the two bodies sees no `"reasoning_only": false` on one side.

### Make the fake prefill peer send tokens the way a real one does

The router reads only the first token of a `first_token` frame. The `router_front_tests` fake peer from #2719 sent every token in one terminal `first_token` frame, which worked only because each existing test sent a single token. The new `reasoning_only` tests need several tokens (`"<think>"`, `"still"`, `" thinking"`), so the fake now sends `tokens[0]` in a `first_token` frame and the rest in one terminal `continuation` frame with `start_sequence: 1`, as a decode node would. Single-token tests are unaffected because `rest` is empty and the first frame stays `done: true`.

### Wait for the router to see every peer online before the first request

All seven router e2e tests now call `wait_for_router_peers_online`, which polls `GET /router/stats` until every node reports `online`. The router's health monitor probes peers once at startup and then every 3 s. A peer still loading its model at that first probe is marked down, and a request sent before the next probe gets a 503 even though `/health` and the peer's port are both up. On the loaded GB10 host this made `disaggregated_router_rejects_model_owned_family_and_keeps_serving` fail twice in a row; a manual run with logs showed the probe marking `prefill-0` down one second before it bound. Each poll request has a timeout of at most 5 s, capped by the remaining test deadline, so a hung connection cannot outlast it (commit 17a05ed3).

This is a test-side fix. The router's startup window itself is unchanged; see section 4.

### Guard the e2e comparisons against vacuous passes

The new 1024-token turn (`LONG_MAX_TOKENS`) first asserts that the single-node reference has non-empty `content` and `reasoning_content` in both the non-stream body and the concatenated stream deltas, with a message that says to raise `LONG_MAX_TOKENS` otherwise. Only then does it compare the router against the reference. The existing 32-token turn likewise asserts that single node reports `reasoning_only: true` on both shapes before comparing the router's flag. Without these preconditions, a model or template change that closes thinking earlier (or never closes it) would make both sides equal and the parity assertions would pass without testing anything.

## 3. Change Summary

10 files, +517/-23, two commits (2e2ab809 for the fix, 17a05ed3 for the 32-token reference guard and the bounded poll) plus a merge of origin/main (0a64688e, which brought in #2737 and #2740 and touches none of this PR's files).

- `src/loading/mod.rs`: `merged_eos_token_ids`.
- `src/loading/vlm_llava.rs`: the `Llama` text arm sets the merged EOS list.
- `src/distributed/tensor_parallel/llama_runtime.rs`: `TensorParallelLlamaModel::set_eos_token_ids` (every rank); `from_model_dir` calls it.
- `src/server/routes/chat.rs`: `log_if_reasoning_only` and `stream_reasoning_only` become `pub(crate)`.
- `src/server/router_front.rs`: the streaming chat path records `saw_content`, `saw_reasoning_content` and the raw text and passes `reasoning_only` to `chat_chunk_finish`; the non-stream path passes it to `chat_completion_json`; both builders take a `reasoning_only: bool`.
- Tests:
  - `generation_config_tests::merged_eos_token_ids_appends_sidecar_ids` (declared first, sidecar ids appended, empty sidecar).
  - `llama_runtime_tests::tensor_parallel_llama_set_eos_token_ids_updates_every_rank`.
  - `vlm_llava_tests::llava_text_model_reports_tokenizer_eos` (ignored, real `llava-1.5-7b-4bit`, panics on a missing model under `MLXCEL_REQUIRE_MODELS=1`).
  - `router_front_tests`: `router_non_stream_chat_marks_reasoning_only`, `router_stream_finish_chunk_marks_reasoning_only`, `router_reasoning_only_absent_when_an_answer_follows`; the fake prefill peer splits first and continuation frames; four existing `chat_completion_json` callers pass `false`.
  - `tests/disaggregated_router_e2e.rs`: `wait_for_router_peers_online` in all seven tests; the thinking test adds the 1024-token non-stream and stream comparisons, the 32-token `reasoning_only` reference guards, and the router-equals-reference `reasoning_only` checks for both shapes.

No documentation changed; no document describes router reply parity or `reasoning_only`.

## 4. Trade-offs and Residuals

- **The router now logs the reasoning-only warning.** `log_if_reasoning_only` emits a `tracing::warn!` on target `mlxcel::thinking` when it returns true. Reusing it means a truncated thinking reply through the router now logs that warning on the router, as single node already did. This is intended parity but adds router log volume for workloads with small `max_tokens`.
- **The router's startup 503 window remains.** The e2e fix waits for `/router/stats` instead of changing the router. In production, a request that arrives after the router's `/health` turns green but before its health monitor has re-probed a slow-loading peer can still get a 503 for up to about 3 s. Whether the router should re-probe on demand or delay readiness is a separate question this PR does not take on.
- **LLaVA on a Qwen2 backbone still reports only its `text_config` ids.** `Qwen2Model` has no `set_eos_token_ids`. Generation is unaffected (the stop ids are merged elsewhere), but `eos_token_ids()` alone is still short for those checkpoints.
- **The 1024-token comparison depends on cross-process determinism.** Byte equality of a 1024-token qwen3-0.6b generation between two processes was verified on GB10 with `MLXCEL_SDPA_DETERMINISTIC=1`. It was not run on Metal or ROCm. If it flakes there, the cause is likely numeric nondeterminism, not router shaping.

## 5. Validation

**Failing first on main 8385f40b** (the new tests written before the fix): `llava_text_model_reports_tokenizer_eos` failed with `left: [] right: [2]`; `router_non_stream_chat_marks_reasoning_only` failed with `message.reasoning_only` `Null`; `router_stream_finish_chunk_marks_reasoning_only` failed with finish chunk `delta: {}`; the e2e thinking test failed with `router stream finish chunk reasoning_only does not match the single-node chat route`, `left: Null right: Bool(true)`. The `merged_eos_token_ids` and `set_eos_token_ids` unit tests use new API and do not compile on main.

**Mutation check for the coverage gap.** Item 2 passes on main by design. With the router's non-stream branch returning the raw handoff text instead of calling `shape_router_chat_reply`, the pre-existing 32-token assertion fails first; with those bypassed for that run, the new assertion fails with `router 1024-token non-stream content does not match the single-node chat route`, left the raw `<think>\nOkay, so the question is asking...`, right `2 + 2 equals 4. ...`. The mutation was reverted and the suite rebuilt and rerun.

**Suites** (GB10, `--profile test-fast --features cuda`, `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, under `gpu-lock`): `server::router_front::`, the two `server::routes::chat::tests` predicate groups, `models::llama3`, `loading::generation_config_tests`, `loading::tests` and `distributed::tensor_parallel::llama_runtime` passed 236 (4 ignored are unrelated qwen3.5 large-weight debug tests). `llava_text_model_reports_tokenizer_eos --include-ignored` passed on the real checkpoint. `disaggregated_router_e2e --include-ignored` passed 7/7 with no model skips, rerun at the final head. `cargo fmt --all -- --check` and `cargo clippy --profile test-fast --features cuda,test-utils -p mlxcel --lib --tests -- -D warnings` were clean.

**Not verified.** `TensorParallelLlamaModel::from_model_dir` was not exercised on a real checkpoint; the unit test covers the setter on a model built with `from_full_weights`, so the call in `from_model_dir` is covered by review and compilation only. The router's `reasoning_only` under `--reasoning-format none` / `deepseek-legacy` (thoughts in content, so the flag should be absent) and under `--skip-chat-parsing` is reasoned from the code, not pinned by a router test. No run measured the effect on CI time of the extra 1024-token generation pair in the thinking e2e test.

## 6. Learning Points

- When a front end must emit the same extension field as the backend route, export the backend's predicate and call it with equivalent inputs. Here that meant recording "what the client received" (`saw_content`, `saw_reasoning_content`) from the router's own emit points, because that is what single node's stream predicate is defined over.
- A test fake that is simpler than the real protocol hides bugs only until a test needs more than the simple case. The fake prefill peer packed all tokens into one `first_token` frame; the router reads one token from it. Matching the real frame sequence (first token, then continuation) costs a few lines and removes a silent truncation from future tests.
- A parity test between two real processes needs a precondition that the compared fields carry signal. Asserting the reference is non-empty (or `reasoning_only: true`) before asserting equality turns a silent vacuous pass into a failure with a fix in the message.
- "Health is green" and "the router can route" are different states when a health monitor polls peers on an interval. Real-process tests should wait on the router's own view of its peers (`/router/stats`), not on the router's `/health` or a peer's open port.
