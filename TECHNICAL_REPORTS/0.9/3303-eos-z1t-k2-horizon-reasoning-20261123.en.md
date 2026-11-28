# PR #3303: Close eos range, Z1T sequence and K2-Horizon reasoning gaps

**Date**: 2026-11-23
**Status**: PR open (closes #3299). Narrow unit and route suites pass; real checkpoints `k2-horizon-0.9b` (server, all three efforts, streaming and not), `limite-1b-violetto` and `z1t-0-mlx` (generate smoke) verified on the M3 Ultra host
**Risk Level**: Low

## Executive Summary

Three follow-ups from recent model ports land together. Limite and Xing4.0 now refuse an `eos_token_id` that is negative or does not fit `i32` instead of wrapping it to a different token. Z1T creates model-owned sequence state lazily for an unprepared id instead of panicking. IFM K2-Horizon's three `reasoning_effort` think pairs are registered in every reasoning-marker table, so its reasoning reaches `reasoning_content` instead of `content`.

## 1. Problem Statement

### 1.1 `eos_token_id` narrowing

`LimiteConfig::eos_token_ids` and Xing4.0's `parse_eos` converted `as_i64()` with `as i32`. `4294967298` became `2`, so generation stopped on the wrong token, and `-1` was accepted although no sampled id can equal it.

### 1.2 Z1T unprepared sequence id

`Z1t::forward_rows_for_sequence` used `with_existing_sequence_state` and panicked with `Z1T missing model-owned sequence state` when the scheduler had not called `prepare_sequence_state`. 32 sibling families use `with_or_create_sequence_state`.

### 1.3 K2-Horizon reasoning in `content`

The template ends the generation prompt with `<ifm|think>\n`, `<ifm|think_fast>\n` or `<ifm|think_faster>\n` (high, medium, low). The model emits only the trace and the matching close. No marker table knew these pairs, so the trace and the close tag reached `content`.

| Risk | Impact | Likelihood |
|------|--------|------------|
| Wrong stop token from a malformed config | Medium | Low |
| Server panic on an unprepared Z1T sequence id | High | Low |
| K2-Horizon clients see raw reasoning and tags in `content` | Medium | High |

## 2. Technical Review

### 2.1 Security

The new stream-filter delimiters apply to every family's output, as `<think>` already does. None of the six markers is a prefix or substring of another entry, and `max_delim_len` is derived from the table and does not grow, so the partial-match hold is unchanged. A non-K2 model that writes literal `<ifm|think>` text would have it treated as a reasoning marker, the same exposure every other marker has.

### 2.2 Performance

Primed detection builds a vector of at most three pairs once per request. The budget tracker's close check gains a two-slot `Option` comparison per token. Neither is measurable.

### 2.3 Compatibility

- No breaking change. `ThinkingTokenIds` gains a field, and its struct literals moved to `ThinkingTokenIds::new(open, close)`.
- For every family whose resolved open is not `<ifm|think>`, `primeable_pairs()` returns the one resolved pair, so primed detection is unchanged.
- Checkpoint configs with an invalid `eos_token_id` now fail to load with an error naming the value, where they previously loaded with a wrong stop id.

## 3. Technical Decisions

### 3.1 Where K2-Horizon's effort variants live

`ThinkingMarkers` holds one resolved pair. The three opens are all in the vocabulary, but the tokenizer can only report one.

| Option | Pros | Cons |
|--------|------|------|
| Add a variants field to `ThinkingMarkers` | Data travels with the markers | Every struct literal changes; the field is empty for all other families |
| A K2-specific marker function like Kimi K3's | Mirrors K3 | Still one pair; medium and low would not be detected |
| **Chosen: a `THINK_EFFORT_VARIANTS` table keyed by the resolved open** | One table, read by primed detection and the budget resolver; no change for other families | A second lookup next to the marker probe |

`prompt_primed_open_close_marker` returns the close of the pair whose open the prompt ends with, so the streaming `ThinkingDelimiterEcho` and `--reasoning-format none` echo the right pair.

### 3.2 Budget closes

The scheduler resolves one `ThinkingTokenIds` per model at startup and has no per-request view of the primed variant. `alt_closes` lets the tracker treat `</ifm|think_fast>` and `</ifm|think_faster>` as closes; without it a capped request under medium effort would keep counting the answer as reasoning and force a close into it. The forced close itself stays `</ifm|think>`; every filter accepts it, and carrying the primed variant into `ThinkingState` is left as a follow-up.

### 3.3 Xing4.0 validation point

`Xing40Config` stores `eos_token_ids: Vec<i32>`, so `validate()` cannot see the raw JSON value. `parse_eos` returns the range error from deserialization, which is where `validate()` already runs, and `validate()` refuses a negative id in configs built in code. This keeps the field type, and the edit stays inside `parse_eos` and `validate`, which a concurrent RoPE change in the same file does not touch.

## 4. Implementation Details

- `src/loading/mod.rs`: `checked_eos_token_ids(value) -> Result<Vec<i32>, String>`, using `i32::try_from` and refusing negatives.
- `src/models/limite_config.rs`, `src/models/xing4_0.rs`: callers of the helper.
- `src/models/z1t.rs`: `with_or_create_sequence_state(seq_id, || self.fresh_states(), ..)`.
- `src/tokenizer/thinking.rs`, `src/tokenizer/mod.rs`: `THINK_EFFORT_VARIANTS`, `ThinkingMarkers::primeable_pairs`, `<ifm|think>` in `SINGLE_TOKEN_PAIRS`.
- `src/reasoning_display.rs`: primed detection over `primeable_pairs`.
- `src/server/thinking_budget.rs`: `alt_closes`, `is_close`, the K2-Horizon resolver arm.
- `src/server/tool_calls/stream_filter.rs`, `parser.rs`, `src/server/routes/chat.rs`: the delimiter rows, canonical pairs, marker pairs and unclosed-primed close list.

## 5. Learning Points

A prompt-primed reasoning block is detected from the rendered prompt, not the output: the filter starts in the thinking state and the output carries only the close. A template that picks among several opens needs every open in the primed check and every close in each consumer (stream filter, parser, unclosed check, budget), or the split silently fails for the variants that are not the resolved pair.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 21 |
| Lines added | about 600 |
| Lines deleted | about 90 |
| Tests added | 9 |

| Category | Summary |
|----------|---------|
| Correctness | eos range checks; Z1T lazy state; K2-Horizon reasoning split in five tables plus the unclosed-close list |
| Tests | Limite and Xing4.0 range tests, Z1T fresh-id logits parity, six K2-Horizon tests including the real `/v1/chat/completions` handler |
| Documentation | Limitation removed from `docs/supported-models.md` |

## 8. Follow-up Actions

- Carry the primed variant's close id into `ThinkingState` so a budget under medium or low effort forces `</ifm|think_fast>` or `</ifm|think_faster>`.

## Appendix A. Test Results

- Metal, under the GPU lock: `models::limite` 15, `models::xing4_0` 17, `models::z1t` 14, `coverage_tests` 18 passed.
- `server::k2_horizon_reasoning_tests` 6, `server::tool_calls` 502, `server::thinking_budget` 46, `thinking` 188, `tokenizer::` 143, `server::routes::chat` 73, `server::chat_request` 131, `primed` 62, `loading::` 428 passed.
- `mlxcel-server -m models/k2-horizon-0.9b`: high and medium return the trace in `reasoning_content` and `42` in `content`; low closes at once with empty `reasoning_content`; no response contains `ifm|think`.
- `mlxcel generate` smoke on `limite-1b-violetto` and `z1t-0-mlx` loads and generates.
