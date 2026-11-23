# PR #3284: Add Decider-2b (decider2) decision model

**Date**: 2026-11-23
**Status**: PR open (draft, closes #2359; part of #2375). Unit, loader, real-checkpoint parity and request-limit tests pass; end-to-end checked on `models/decider-2b`
**Languages**: Rust, Python (fixture generator)
**Risk Level**: Low (new decision family; one shared change in the Qwen3.5 hidden-state path, generation paths untouched)

---

## Executive Summary

Decider-2b (`model_type: decider2`, `nativ-community/decider-2b`) is a Qwen3.5-2B finetune that answers typed questions (`choice`, `score`, `bool`/`noul`) about a state by reading label-token logits at an open answer slot. It never generates. This PR ports mlx-vlm's `decider2` in its default independent mode as a production family behind the #2343 decision API (LiquidAI d1-3B, #3282, merged while this PR was open and is the other one), so `mlxcel decide -m models/decider-2b` and `mlxcel-server --decision-model models/decider-2b` (`POST /v1/decisions`) now serve it. On the real checkpoint the port matches mlx-vlm to 1.4e-3 on a short request and to 1.48e-2 on a long JSON state, and an f32 run of both sides agrees to 0.0000 at 4 decimals, which places the remaining bf16 gap in backbone arithmetic rather than in the port. Review added a per-request prompt-token budget so that a request cannot queue an unbounded amount of prefill on the single decision worker.

## 1. Problem Statement

### 1.1 Background

#2343 added the decision layer (`src/decision/`, the `DecisionModel` trait, `mlxcel decide`, `POST /v1/decisions`, `--decision-model`) with only a `#[cfg(test)]` stub behind it. In a production build `DecisionKind` had no variants, so every load failed with "<path> is not a decision checkpoint". #2343 still had one open check: an end-to-end run on a real checkpoint. Decider-2b was the first family scheduled for it (#2359, under the decision epic #2375).

### 1.2 Existing Issues

- **No detection**: `model_type: decider2` had no arm in `src/models/detection.rs`, so `get_model_type` failed with `Unsupported model type: decider2`, and the decision loader had nothing to dispatch to.
- **No hidden-state path that later families could share**: `Qwen35Model::prefill_hidden` (added for MOSS-Image in #3107) embedded token ids and ran the layer loop in one function. #2365 (Clef) needs the same loop starting from spliced `inputs_embeds` with explicit MRoPE positions. Without a split, that would become a third copy of the loop next to `forward_prefill_with_last_hidden`.
- **f16 load cast**: `load_text_weights` casts bf16 checkpoints to f16 on Apple Silicon. For a model whose output is a calibrated probability, that cast is a numeric change, not a speed knob.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Prompt rendering differs from upstream by one token, shifting every probability | High | Medium without token-level fixtures |
| Request amplification: each question and isolated score level repeats the state, so a small body can queue billions of prefill tokens | High (worker stall, memory) | Medium on a public endpoint |
| A Qwen3.5 refactor changes generation output | High | Low (split is mechanical, `None` positions as before) |

## 2. Technical Review

### 2.1 Security

The security pass (PR comment, fixed in `e6ee24b0`) found:

| Issue | Severity | Status |
|-------|----------|--------|
| Request-controlled prefill amplification: only the state was capped (`max_state_tokens`), instruction and option text not at all. A 2 MB body of short questions over a 32k-token state could queue billions of prefill tokens and gigabytes of prompt ids | High | Fixed: 400 when one prompt exceeds `max_position_embeddings` (262144, also the fallback when `config.json` omits it) or the prompts together exceed `MAX_REQUEST_TOKENS = 1 << 20` (1048576). Encoding stops at the first row past either limit |
| State tokenized once per row | Medium | Fixed: `build_prompts` encodes the state once per request and shares the ids |
| Zero, negative or non-finite `decision_config` temperature produced non-finite answers (500) on every request | Low | Fixed: refused at load |
| Padding to 64 can push a group slightly past 8192 tokens, because groups are formed on unpadded lengths | Low | Left as is (matches upstream) |
| A 504 timeout does not cancel the in-flight prediction on the worker | Low | Open, belongs to the shared #2343 worker |

`const _: () = assert!(MAX_REQUEST_TOKENS >= 32 * 32768);` pins the budget so that 32 questions over a full 32768-token state still fit. Nesting depth is bounded by serde_json's 128-level limit; option, level and label counts are validated before any indexing.

### 2.2 Performance

Rows are batched: consecutive rows join a group while `rows_in_group * longest <= 8192` (`GROUP_TOKEN_BUDGET`), each group right-padded to a multiple of 64 (`PAD_MULTIPLE`). One forward per group returns `[rows, length, hidden]`; the scorer takes each row's hidden at its last real position and multiplies by a `[hidden, 255]` label matrix, so the LM head over the full vocabulary is never computed. No throughput numbers were measured.

### 2.3 Compatibility

- **Breaking changes**: none. `mlxcel generate -m models/decider-2b` now fails with "decider2 is a decision model; use `mlxcel decide` or `mlxcel-server --decision-model`" instead of "Unsupported model type".
- **New dependencies**: none.
- **Qwen3.5 regression**: `mlxcel generate -m models/qwen3.5-4b-4bit -n 64 --temp 0 --show-reasoning` printed identical text before and after.

## 3. Technical Decisions

### 3.1 Label logits from tied embedding rows

**Context:** The checkpoint has `tie_word_embeddings: true`. The answer is a softmax over the logits of label tokens only.

**Decision:** At load, select the names `A`..`Z` then `AA`..`ZZ` whose tokenizer encoding (no special tokens) is exactly one token, keep the first 255 (`LABEL_COUNT`), and fail with "The decider tokenizer does not provide 255 distinct label tokens" otherwise. Gather those rows of `embed_tokens` once, transpose to `[hidden, 255]`, and evaluate. Per group, logits are `matmul(slots, label_weights_t)` in bf16 then read as f32, matching upstream's order (bf16 product, cast afterwards).

**Rationale:** The tied head restricted to the labels gives the same logits as the full head at 255 columns instead of the full vocabulary. Going through `embed_tokens.forward` also dequantizes rows if the embedding is quantized.

### 3.2 Independent-mode rows and isolated score levels

**Context:** Upstream has an independent mode (one prompt per question) and a packed mode, plus an `isolated` override passed as a Python keyword argument.

**Decision:** Port independent mode only. `plan_rows` makes one row per question, except that with `isolated_levels: true` (the checkpoint's setting) a `score` question becomes one `no`/`yes` row per level, with instruction `"{instruction}\nProposed answer: {level}\nDoes the proposed answer fit?"` (a leading `"<int>:"` stripped from the level) at the score temperature. The answer normalizes the per-level `yes` probabilities by their sum (`1e-9` divisor when the sum is 0) and reports `level_fit` and `fit_mass` in metadata.

**Trade-offs:** Packed mode and the per-call override are not reachable from the API, so they stay out of scope (as stated in the issue). Isolated levels multiply row count by the level count (up to 10), which is one reason the request budget in 2.1 counts rows, not questions.

### 3.3 Split `prefill_hidden` into `hidden_from_embeddings`

**Context:** The scorer needs post-norm hidden states with no head on fresh caches, which `forward_prefill_hidden` already provides. #2365 Clef needs the same loop from `inputs_embeds` with MRoPE positions.

| Option | Pros | Cons |
|--------|------|------|
| A: Use `forward_prefill_hidden` as is | No change to Qwen3.5 | Clef must copy the layer loop a third time |
| B: Add a Decider-specific forward | Isolated | Another copy to keep in lockstep |
| **Chosen: split into embedding lookup plus `hidden_from_embeddings(h, caches, position_ids)`** | One loop for every hidden-state entry point; Clef adds only an entry point | Touches a shared model file |

`prefill_hidden` now does `embed_tokens.forward(input_ids)` and calls `hidden_from_embeddings(h, caches, None)`; the only behavioral line change is `layer.forward(&h, mask, cache, position_ids)` where `None` is passed exactly as before. `forward_internal*` and every generation path are untouched. The doc comment records the property the scorer relies on: rows of a `[B, L]` batch are independent and causal, so a right-padded row reads the same hidden at its last real position as a one-row call.

### 3.4 Keep bf16 at load

**Context:** On Apple Silicon `load_text_weights` casts bf16 weights to f16. Decider-2b's per-type temperatures (`choice 1.164`, `noul 1.624`, `score 1.124`) were fitted on the bf16 weights the reference runs.

**Decision:** Add `is_decider2` to the exclusion list next to `limite`, so the checkpoint keeps bf16. With the f16 cast an isolated score level moved by 2.1e-3 against mlx-vlm, above the 2e-3 parity bound. `bd0ac7cb` pins this with a sanitize test.

**Trade-offs:** bf16 matmul may be slower than f16 on some Apple GPUs; accuracy of the calibrated output was preferred.

### 3.5 Parity bounds and what they mean

The fixture `tests/fixtures/decision/decider2_parity.json` was generated by `tests/fixtures/generate_decider2_fixture.py` with mlx-vlm d998a620. Prompt token ids match upstream exactly for a 3-option and a 12-option choice (the latter exercises the >10-option label-id splice), the score options, an isolated score row and an annotated 8-item list state.

- Ticket request (choice, isolated 4-level score, bool): largest probability gap 1.4e-3, bound `TOLERANCE = 2e-3`.
- ~400-token JSON state (12-option choice, bool): largest gap 1.48e-2, checked at `LONG_TOLERANCE = 2e-2`, argmax labels and bool values must agree.

The 2e-2 bound looks loose, so it was checked: with the checkpoint cast to f32, mlx-vlm and this port agree to 0.0000 on every probability and metadata value of both requests, batched and one question at a time. Rendering, padding, positions, label weights, temperatures and softmax are therefore exact; the bf16 gap is backbone arithmetic. Upstream itself drifts up to 7.2e-3 from its own f32 run, and its bool moves from 0.3001 to 0.2921 between batched and alone. Usage counts are exact.

## 4. Implementation Details

| File | Role |
|------|------|
| `src/decision/decider2.rs` | `Decider2Model`: load (config, `decision_config`, prompter, pad id, backbone, label matrix), `build_prompts` with both limits, `score_rows`/`group_logits`, `DecisionModel::predict` |
| `src/decision/decider2_prompt.rs` | Label selection, `render_question`, `annotate` (`_index` on lists of 8 or more items), Python `json.dumps(ensure_ascii=False)` serialization including float formatting, prompt assembly with the >10-option splice |
| `src/decision/decider2_rows.rs` | Row plan, grouping, padding, calibrated softmax, upstream rounding and metadata (`confidence`, `certainty`, `legend`, `level_fit`, `fit_mass`) |
| `src/decision/loader.rs` | `DecisionKind::Decider2` and its constructor arm |
| `src/models/qwen3_5.rs` | `hidden_from_embeddings` split |
| `src/models/sanitize.rs` | bf16 kept for `decider2` |
| `src/models/detection.rs` | `decider2` arm that redirects `generate` to the decision entry points |
| `docs/decisions.md`, `docs/supported-models.md`, README, CHANGELOG | Family table, prompt format, limits |

Request flow: `predict` renders the state and questions, `plan_rows` expands isolated scores, `build_prompts` encodes and enforces the budgets, `score_rows` runs one backbone forward per group, and `answer` builds each typed answer from its slot of rows.

## 5. Learning Points

### 5.1 Classification by restricted head

A causal LM can serve as a calibrated classifier without generation: put the options behind single-token labels, end the prompt at an open slot (`"\nAnswer: ("`), and softmax only the label logits at that position with a fitted temperature. With tied embeddings the restricted head is a gather of embedding rows, computed once at load.

### 5.2 Tokenizer boundaries decide parity

Above 10 options, labels like `K` followed by `)` could merge with neighbouring text in BPE. Upstream encodes `"\n("`, appends the label id directly, then encodes `") {option}"`. Matching that, and matching Python's `json.dumps` byte for byte for the state, was checked at token-id level in the fixture before comparing any probability.

### 5.3 Separate port error from arithmetic noise

When a bf16 parity gap is large, run both implementations in f32. A 0.0000 gap in f32 proves the port is exact and that the bf16 gap comes from the backbone; checking upstream's own bf16 against its f32 run gives the noise floor to set the bound against.

### 5.4 Budget the expanded work, not the input

The amplification came from a cap on the wrong quantity. The state was bounded, but rows multiply it. The limit now applies to the summed prompt tokens after expansion, and encoding stops as soon as the budget is crossed, so a refused request never holds more than the budget in memory.

## 6. Related PRs and Issues

- Issue #2359: Decider-2b support (closed by this PR)
- Issue #2343: decision API; this PR closes its remaining real-checkpoint end-to-end check
- Issue #2375: decision-model epic
- Issue #2365: Clef, the planned user of `hidden_from_embeddings` with `inputs_embeds` and MRoPE positions
- PR #3107: MOSS-Image, which introduced `forward_prefill_hidden`

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 20 |
| Lines added | +3213 (1178 are the parity fixture JSON) |
| Lines deleted | -16 |
| Tests added | 19 `#[test]` functions (decider2 unit and parity, loader, sanitize, detection) |

### Changes by Category

| Category | Summary |
|----------|---------|
| Feature | Decider2 scorer, prompt rendering, row plan, loader and detection arms |
| Shared backbone | `hidden_from_embeddings` split in Qwen3.5 |
| Numerics | bf16 kept at load for `decider2` |
| Security | Per-request and per-prompt token limits, state encoded once, temperatures validated at load |
| Documentation | `docs/decisions.md`, `docs/supported-models.md`, README, CHANGELOG |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `a93e2d17` | feat | feat(decision): add Decider-2b (decider2) decision model |
| `be6f25eb` | merge | Merge origin/main into feature/issue-2359-decider2-decision-model |
| `8bc6b0b5` | docs | docs(decision): add Decider-2b changelog entry and f32 parity evidence |
| `e6ee24b0` | fix | fix(decision): bound Decider-2b prompt tokens per request |
| `bd0ac7cb` | test | test(decision): pin Decider-2b bf16 load and sync docs with the final code |

## 8. Verification and Unverified Items

### Verified

- `mlxcel decide -m models/decider-2b` on "Please refund my duplicate charge" with a 3-label department question printed `billing` with probabilities `billing 0.9198, technical 0.0506, sales 0.0296`, `confidence 0.9198`, `certainty 0.6977`, `input_tokens 37`.
- `mlxcel-server -m models/qwen3.5-4b-4bit --decision-model models/decider-2b`: the same request on `POST /v1/decisions` returned HTTP 200 with the identical answer; `/v1/models` lists `decider-2b` with capability `decision`; an 11-level score returned 400 `score needs 2 to 10 levels`.
- `mlxcel generate -m models/decider-2b` exits 1 with the decision-model message.
- Test suites: `decision::` (unit, loader, parity, request limits), `models::sanitize_tests::decider2`, `models::detection_tests`, `models::qwen3_5`, `imagegen::moss_image`, `coverage_tests`; clippy with `-D warnings`, `cargo fmt --check`, the `scripts/ci` contract checks and license headers.

### Unverified or Remaining

- The long-prompt bf16 drift (1.48e-2) is not reduced. Narrowing it would require changing Qwen3.5 kernels shared with generation, outside this PR.
- A 504 timeout does not cancel the running prediction; the shared #2343 worker keeps computing it.
- Packed mode, the per-call `isolated` override and the 31B sibling checkpoint are out of scope.
- The "before" binary in the Qwen3.5 regression check was a Nov 19 release build, not a fresh build of origin/main, so the comparison covers this PR plus whatever landed on main between Nov 19 and the merge base.
- No throughput or latency measurement for the scorer, and the f16-versus-bf16 speed cost of 3.4 was not measured.
