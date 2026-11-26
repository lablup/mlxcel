# PR #3445: Raon-1 mini reasoning-effort steering (issue #3435)

**Date**: 2026-11-26

**Status**: PR open (closes #3435). Validated on an M3 Ultra. 63 files, +2103 / -116.

**Risk Level**: Medium (adds a per-sequence channel to `LanguageModel` and the scheduler; no-op for every other family)

## Executive Summary

The final `KRAFTON/Raon-1-mini` release adds a fixed f32 vector to the MoE output of layers 12, 14, 16, 18 and 20, scaled by an alpha chosen from the request's reasoning effort. The default effort (`high`, alpha 0.05) is nonzero, so the #3225 port (alpha 0) produced logits that differed from the reference on every request that did not ask for `max`. This PR applies the steering, carries each request's effort through the server and scheduler to its own batch row, and keys the prompt cache on it.

Greedy output matches the Transformers reference at the default, `low` and `none` efforts, and teacher-forced traces on an 871-token prompt show 0 of 541 (default) and 0 of 542 (`low`) disagreements at decided positions.

## 1. Problem Statement

### 1.1 Background

The release checkpoint differs from the preview only by five `moe.reasoning_effort_steering` tensors and a `reasoning_effort_steering` config block (`alpha` per level, `default: "high"`, zero-based `layers`). The reference adds `alpha * c` in f32 to the already-rounded MoE branch and rounds once before the mHC `mix`, skipping the add when alpha is 0.

### 1.2 Existing gaps

- **No model-side effort channel.** `map_reasoning_control_kwargs` in `chat_request.rs` forwarded `reasoning_effort` only to chat templates that mention it. Raon's template does not, so the level was dropped.
- **Effort is per request, not per model.** The vendor MLX runtime keeps it per cache and the vLLM fork per request, so batched decode needs a per-row alpha.
- **Thinking and steering are separate controls.** The template reads only `enable_thinking`; steering never changes the prompt. The model card says `reasoning_effort: "none"` and `enable_thinking: false` must give the same result.

## 2. Technical Decisions

### 2.1 Per-row alpha as a `[B, 1, 1]` array

| Option | Pros | Cons |
|---|---|---|
| Model-global effort setter | Small change | Wrong for concurrent requests at different efforts |
| **Per-sequence alpha, broadcast as `[B, 1, 1]` f32** | One code path for prefill, native batched decode and per-row fallback | Needs a per-sequence channel through the trait and scheduler |

`Raon1MiniLM` keeps an `EffortState` (`raon1_mini_effort.rs`): a model default plus a per-`SequenceId` map. `steer_rows` returns `None` when no steering block exists or every alpha is 0, so the alpha-0 path allocates no MLX array and adds no host sync. A row at alpha 0 inside a steered batch stays bit-exact, because the bf16 to f32 to bf16 round trip is lossless and adding 0 changes nothing.

### 2.2 A generic trait channel, not a Raon-specific hook

`LanguageModel::set_sequence_reasoning_effort` and `sequence_reasoning_effort` default to `Ok(())` / `None`. The scheduler calls them at admission (after prefix adoption, before prefill), so an invalid level fails only that request, and copies the level onto the new sequence id on preemption and queued adoption. Preparing or releasing a sequence id clears its level, so a recycled id cannot inherit another request's effort.

### 2.3 One resolver for render, steering and cache key

`resolve_model_reasoning_effort` (`src/server/reasoning_steering.rs`) runs only when the model's `config.json` declares the steering block. Its order follows the prompt render's thinking toggle, so the prompt and the alpha cannot disagree:

1. A level the request names (`chat_template_kwargs.reasoning_effort`, top-level `reasoning_effort`, Responses `reasoning.effort`).
2. The request's own thinking-off (`enable_thinking: false`, `reasoning: false`) selects `none`.
3. The server default level (`--reasoning-effort`, `--chat-template-kwargs`).
4. A server-default thinking-off selects `none`.
5. The config default (`high`).

Two review findings shaped this order. The first version read only explicit kwargs and the server default, so a server default of thinking-off plus a request with `reasoning: true` rendered thinking on but steered at `none` (fixed in `ddf59049b`). It also let a server default level beat a request's own `enable_thinking: false` (fixed in `1ebc2ce34`). A portable `none` also renders `<think></think>`; a `none` inside `chat_template_kwargs` changes only alpha, as the model card documents.

The resolved level is folded into the prompt-cache template signature. When no level is resolved the signature is unchanged, so cache buckets for other families do not move.

## 3. Implementation Details

- `src/models/reasoning_effort_steering.rs`: parses and validates the block. `none` resolves to the config's `none` alpha when above 0, else `NO_THINKING_ALPHA = 0.1`. `default: "none"`, negative or f32-overflowing alphas (`0d425b857`: `1e39` fits in f64 but became `inf` in the forward), and out-of-range, duplicate or non-MoE layers fail at load by name. A missing vector or one not shaped `[hidden_size]` is a named load error.
- `raon1_mini*.rs`: the steer is applied in the MLP site of listed layers on prefill (`forward_rows_for_sequence`), native batched decode and the per-row fallback.
- CLI: `mlxcel generate --reasoning-effort` sets the model default and is validated against `config.json` before the 30 GB of weights load. `examples/logit_trace` reads `MLXCEL_TRACE_REASONING_EFFORT`; `scripts/raon1_mini_oracle_trace.py` takes `--effort`.
- Docs: `docs/supported-models.md`, `docs/responses-api.md`, `CHANGELOG.md`, and the `mlxcel arch` display name.

## 4. Validation

Reference: the checkpoint's `modeling_raon1_mini.py` (Transformers, bf16, CPU). Prompts: 22 tokens and 871 tokens, thinking off.

| Check | Result |
|---|---|
| `--reasoning-effort none`, "17 + 25" | `42`, then EOS |
| Greedy 64, short, default / low / none | identical to the reference |
| Greedy 64, long, default and low | identical text; low differs from max |
| Teacher-forced, long, default / low | 0 / 541 and 0 / 542 decided disagreements |
| Thinking on, default, short | diverges at generated token 4; 0 / 30 decided disagreements over the reference text, flip gap 0.75; both answer 42 |
| `max` vs pre-change binary | identical text; bit-for-bit proven on the tiny model |
| Server, low and max concurrently | each equals its solo output |
| Server, `minimal` | 400 `invalid_request_error` listing the accepted set |

Unit tests cover alpha resolution, the tiny-model steer (fails if skipped; `max` equals unsteered), per-row batched decode against solo, sequence lifecycle, scheduler admission and preemption carry, every server channel and the precedence order, and the cache key. The full local `make verify` gate passed.

## 5. Follow-up

- The server never reuses a prompt-cache prefix for Raon, even on identical repeats, before and after this PR. Raon has no model-owned snapshot/restore, which is the likely cause. The effort separation in the cache key is therefore covered only by the store test.
- Queued adoption never selects a Raon sequence (no paged blocks), so its effort carry is untested on this family.
- The disaggregated prefill handoff and the pipeline/XLA generate paths ignore the effort; Raon does not take them today.
- Thinking-on runs were compared against the reference only at the default effort. CUDA and ROCm were not run.
