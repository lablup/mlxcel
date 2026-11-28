# PR #2719: Close the disaggregated router parity residuals from #2700

**Date**: 2026-10-28 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust, Markdown
**Risk Level**: Low to Medium (one additive wire field; the model-level EOS list changes for every checkpoint that loads as `LoadedModel::Llama`)

## Executive Summary

PR #2700 (issue #2695) made the real-process disaggregated e2e suites pass and listed three places where the router still differed from a single node. This PR (issue #2715) closes all three. The router's non-stream chat reply is now shaped by the same helper as the single-node no-tools reply (`routes::chat::shape_plain_chat_reply`), so MiniCPM's `" Café"` becomes `"Café"` on both paths. `Llama3Model::eos_token_ids` now comes from the checkpoint's `eos_token_id`, and the Llama 3 ids 128001 and 128009 are appended only for the 128256-token Llama 3 vocabulary, so MiniCPM-2B in llama format reports `[2]` instead of two out-of-vocabulary ids. `ResultFrame` gains an optional `prompt_cache_enabled` that the prefill node sets on its successful first-token frame, so the router's `usage.prompt_tokens_details` follows the worker's prompt-cache setting rather than the router's own flag.

## 1. Problem Statement

### Non-stream chat content kept the leading space

On the MiniCPM "café" case the router returned `" Café"` and single node returned `"Café"`. Detokenization was not the cause: the completions body already matched byte for byte, and the scheduler text was `" Café"` on both sides. The two paths diverged when they built the response. Single node ran the raw text through `clean_structural_tokens` (which ends in `clean_content_markers` and its `.trim()`), the `<answer>` wrapper strip, `strip_unclosed_primed_thinking` and `shape_response`. The router's non-stream branch concatenated `StreamFilter` output and never ran that step. `disaggregated_router_chat_usage_matches_single_node_byte_fallback` compared only `usage` and `finish_reason`, so it did not notice.

### `Llama3Model` hard-coded the Llama 3 EOS ids

`eos_token_ids()` returned `[128001, 128009]` for every checkpoint that loads through `llama3`. MiniCPM-2B in llama format (`vocab_size` 122753, `eos_token_id` 2) is one of them, and both ids are outside its vocabulary. Every in-tree generation path (scheduler admission, the handoff prefill and decode paths, the CLI, `bench_decode`, the pipeline runtime) merges the sidecar EOS from `read_eos_token_ids`, so no live path stopped wrongly. The model's own answer was still wrong for consumers that do not merge: the llava-family wrapper in `src/vision/mod.rs` and the tensor-parallel runtime in `llama_runtime.rs`.

### `prompt_tokens_details` followed the router's flag

Single node includes `usage.prompt_tokens_details` exactly when its prompt-cache store exists. In a disaggregated deployment that store lives on the worker, and the router never builds one, yet all four router usage builders (chat and completions, streamed and non-streamed) decided from `state.config.prompt_cache.is_enabled()` on the router. A cluster whose workers ran with `--no-prompt-cache` behind a default router reported `{"cached_tokens": 0}` where single node reports nothing.

## 2. Technical Decisions

### Share the shaping function instead of trimming in the router

The single-node no-tools shaping moved into `plain_chat_reply(raw, primed_open_thinking, answer_wrapper, reasoning_format) -> (String, ShapedResponse)`. `shape_plain_chat_reply` is the `pub(crate)` wrapper that returns only the `ShapedResponse`. The tuple form exists because the single-node route also feeds the cleaned answer, before `--reasoning-format` placement, to the next-turn prompt-cache warm-up (#1144). `extract_reasoning_content` moved with it: the no-tools path computes reasoning inside the helper, and the tool-call branch computes its own copy, so neither path computes it twice.

The issue rejected the narrower fix of trimming only the router's assembled content. That would match on the MiniCPM case and diverge again on `--reasoning-format none`, a primed open thinking block, or an `<answer>` wrapper, because each of those changes what the trim applies to. One function called from both sides makes drift a compile-level coupling rather than a convention.

In the router, the non-stream branch now accumulates the raw handoff text verbatim and calls `shape_router_chat_reply`, which returns the raw text under `--skip-chat-parsing` (single node does the same, returning `result.text` before any shaping) and otherwise calls `shape_plain_chat_reply`. The `StreamFilter` and `ThinkingDelimiterEcho` code in that branch is gone. The streaming branch is unchanged, because single-node streaming does not trim either.

### Derive EOS ids from the config, with the Llama 3 defaults gated on the vocabulary

`llama3::ModelArgs` gains `#[serde(default)] eos_token_id: Option<EosTokenId>`, a local untagged int-or-list enum modeled on `helium::TokenIdField`. Serde fails the whole config when one field has the wrong shape, so both forms must parse. `ModelArgs::eos_token_ids()` returns the declared ids in order without duplicates and, only when `vocab_size == 128256`, appends whichever of 128001 and 128009 is missing. `from_weights` stores the result in a new private `Llama3Model.eos_token_ids` field, and `set_eos_token_ids` lets a caller replace it, following the pattern in `loading::special`.

The vocabulary gate is what lets Llama 3 checkpoints keep their old behavior without inventing ids elsewhere. 128001 is an ordinary token in larger non-Llama vocabularies, so appending it unconditionally would end generation on a regular token, the hazard documented on `nemotron_nas::NemotronNASConfig`. A non-Llama-3 checkpoint that declares nothing gets an empty list; before this PR it got two ids that could never be sampled (MiniCPM) or could be sampled as regular text (a larger vocabulary), so the empty list is no worse for any of them.

The #2695 decode-intake regression (`serving_handoff_parity_stops_on_config_eos_minicpm`) depended on the old wrong answer: its premise was that the model reports `[128001, 128009]` and only the config merge supplies `2`. The test now asserts the model reports `[2]`, then calls `set_eos_token_ids(vec![128001, 128009])` so it still proves the decode path merges the config EOS.

### Report the worker's prompt-cache setting on the wire

`ResultFrame.prompt_cache_enabled: Option<bool>` is `#[serde(default)]`, following the mixed-version precedent of `generated_tokens`. The frame is JSON (`serde_json`) without `deny_unknown_fields`, so an older router ignores the new key and a newer router reads a missing key as `None`. The prefill coordinator reads `BatchScheduler::has_prompt_cache_store()` once at startup, like `handoff_supported`, and sets the field only on a successful first-token frame (`drained.error.is_none().then_some(..)`). Error frames and decode continuation frames carry `None`. `drive_handoff_result` captures it from the first frame into `HandoffOutcome`, and `HandoffOutcome::reports_prompt_cache(router_flag)` resolves `prompt_cache_enabled.unwrap_or(router_flag)`. All four usage builders use it, so an older prefill node keeps the previous behavior.

The rejected alternative was to document the router flag as the contract. Operators configure the prompt cache per worker, and the wire field costs one optional bool.

## 3. Change Summary

16 files, +821/-132, two commits (bdeb91b4 for the fix, 670034c1 to move a misplaced `helium.rs` comment) plus a merge of origin/main (164eaeed, which brought in only #2718's media-policy files).

- `src/server/routes/chat.rs`: `plain_chat_reply` and `shape_plain_chat_reply`; the no-tools branch calls them; reasoning extraction moves into the tool-call branch and the helper.
- `src/server/router_front.rs`: the non-stream chat branch accumulates raw text and calls `shape_router_chat_reply`; `HandoffOutcome.prompt_cache_enabled` and `reports_prompt_cache`; the chat and completions usage paths (both streamed usage chunks and both non-stream bodies) use it; `router_usage_json` and `completion_response` doc comments updated.
- `src/distributed/disaggregated/serving_protocol.rs`, `coordinator.rs`: the new frame field, set on the successful first-token frame and `None` on the six other frame constructions.
- `src/server/batch/scheduler/handoff.rs`: `has_prompt_cache_store()`.
- `src/models/llama3.rs`: `EosTokenId`, `LLAMA3_VOCAB_SIZE`, `LLAMA3_EOS_TOKEN_IDS`, `ModelArgs::eos_token_ids`, the stored field and its setter. `helium.rs` and three test struct literals gain `eos_token_id: None`.
- `docs/CONTINUOUS_BATCHING.md`: the `prompt_tokens_details` note now describes the worker rule and its fallback; a new note states the non-stream shaping parity. There is no Korean counterpart of this document.
- Tests: five `llama3_tests` EOS cases; `serving_protocol_tests` decodes a frame with no `prompt_cache_enabled` key as `None`; `router_front_tests` gains a fake prefill peer (`spawn_fake_prefill`, `router_with_fake_prefill`) and tests for the shaped non-stream reply, the shaping helper under each reasoning format, and the three `prompt_tokens_details` cases (worker off with router on, worker on with router off, worker `None`) across chat, completions and both usage chunks; the MiniCPM e2e test compares non-stream `content` and the streamed concatenation per case; the qwen3 thinking e2e test adds a non-stream comparison of `content` and `reasoning_content`.

## 4. Trade-offs and Residuals

- **Model-level EOS lists change.** A Llama 3.1 Instruct checkpoint declares `[128001, 128008, 128009]`, so `Llama3Model` now also reports 128008 (`<|eom_id|>`). The in-tree generation paths already merged it from the sidecar files, so only the non-merging consumers (llava-family backbones, the TP runtime) see a difference, and it brings them in line with the checkpoint. A Llama 3 derivative with a resized vocabulary (anything other than 128256) no longer gets 128001/128009 appended and reports only what its config declares.
- **An empty list is possible.** A llama-format checkpoint with no `eos_token_id` in `config.json` and a non-Llama-3 vocabulary now reports `[]`. The vision wrapper and the TP runtime still do not merge `generation_config.json` or tokenizer EOS ids, so for such a checkpoint they stop only at `max_tokens`, as they effectively did before. Making those consumers merge sidecar ids is outside this PR.
- **Prefill decides for the pair.** Only the prefill node reports its setting. If a prefill node and a decode node disagree on `--no-prompt-cache`, the prefill node's setting decides `prompt_tokens_details`. Single node has one store, so there is no single-node behavior to match for that split.
- **Router non-stream `--reasoning-format none` and `deepseek-legacy` now follow single node's non-stream form.** The old branch used `ThinkingDelimiterEcho` so the router's two response shapes agreed (#2464); the new branch rebuilds the thoughts with `content_with_thinking_block`, as single node's non-stream path does. The router's stream and non-stream forms now agree exactly as far as single node's do.
- **`reasoning_only` is not emitted.** Single node's non-stream reply carries the `reasoning_only: true` extension when the whole reply is reasoning; the router does not. The PR body records this as a follow-up.

## 5. Validation

**Failing first on 752db4f9** (the new tests written before the fix): the MiniCPM e2e chat test failed with `left: String(" Café")`, `right: String("Café")`; the fake-prefill router test failed the same way; `prompt_tokens_details_omitted_when_the_worker_cache_is_off` failed with `left: Some({"cached_tokens":0})`, `right: None`, and the "present" case with `left: None`; the MiniCPM handoff parity test failed with `left: [128001, 128009]`, `right: [2]`. The `None` fallback test and the qwen3 non-stream comparison pass on main by design. The `llama3_tests` EOS cases, the shaping helper test and the `serving_protocol` missing-key case use new API and do not compile on main.

**Suites** (GB10, `--profile test-fast --features cuda`, `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, under `gpu-lock`): `server::router_front` 40/40, `server::routes::chat` 73/73, `models::llama3` 34/34, `models::headless_llama` 3/3, `distributed::disaggregated::serving_protocol` 19/19, `serving_handoff_parity --include-ignored` 6/6, `models::llama_bidirec` 9/9, `models::helium` 28/28, `distributed::tensor_parallel::llama_runtime` 35 passed with 4 ignored. `disaggregated_router_e2e --include-ignored` 7/7 in 49 s with no soft-skips, `disaggregated_serving_e2e` 2/2. `cargo fmt --check` and `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings` were clean. The helium comment commit re-ran fmt, clippy and `models::helium`.

**Real server**: single-node `mlxcel-server` with `minicpm-2b-sft-4bit-llama-format-mlx` on 127.0.0.1:19701, greedy, `max_tokens` 16. The "café" prompt returned `"Café"` with `finish_reason: stop` and `completion_tokens: 1`; the "Чай" prompt returned `"Чай"` with `stop` and 4 tokens. Both stopped at `</s>`.

**Not verified.** The qwen3 e2e comparison passes on main too, because at 32 tokens both sides return `content: ""` and the same reasoning; no real-model run compares a reply that has both reasoning and an answer through the router. `shape_router_chat_reply_matches_the_single_node_shaping` compares the router helper against the function it delegates to, so the `--reasoning-format none` and `deepseek-legacy` forms are pinned only at the helper level, not by a real-process router run. No mixed-version cluster (old router with new prefill, or the reverse) was run; compatibility rests on the JSON encoding and the `serde(default)` unit test. The suites ran on bdeb91b4/670034c1, before the merge of origin/main (164eaeed); a pre-merge test run on the merge commit was in progress when this report was written.

## 6. Learning Points

- When two routes must return the same bytes, extract the shaping into one function both call, and test parity on the whole response body, not on the fields the bug report named. #2700's e2e test compared `usage` and `finish_reason` and left a visible `content` difference in place.
- A model-level fallback constant needs a guard that says which checkpoints it belongs to. Gating the Llama 3 ids on the 128256 vocabulary keeps old behavior for Llama 3 and stops the constant from leaking into other vocabularies, where the same id is an ordinary token.
- A regression test whose premise was the old wrong answer should assert the corrected value first and then restore the premise explicitly (`set_eos_token_ids`), so it keeps testing the path it was written for.
- For a per-worker setting that the front end reports, carry it on the wire as `Option<T>` with `serde(default)` and resolve `None` to the old behavior. Older senders keep working, and the fallback is in one method (`reports_prompt_cache`) rather than spread across four builders.
