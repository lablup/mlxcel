# PR #2610: Sequence-keyed state for Baichuan-M1, Mllama and Qwen3-VL DeepStack

**Date**: 2026-10-26
**Status**: Complete on Metal (M1 Ultra); Cohere Compass validated by synthetic tests only
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2610 (issue #2567) fixes per-request state isolation in three model families. Each of them advertised batching support while keeping per-request state in a single model-level slot:
- Baichuan-M1: K/V convolution state.
- Mllama: cross-attention image state and the per-layer image K/V.
- Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni and Cohere Compass: DeepStack visual features.

On `main`, concurrent requests on `mlxcel-server --parallel 4` read each other's state. Measured examples: a Llama 3.2 Vision request described the other request's image, and Baichuan-M1 output was degenerate even for a single request. Each family now uses the fallback-slot-plus-`HashMap<SequenceId, T>` shape the repository already uses for Qwen MRoPE and Gemma 4 `per_layer_inputs`.

Validation surfaced a second, independent mechanism: the decode lookahead pipeline tore down speculative steps without rewinding state that lives outside the KV cache. The PR adds a core hook for this.

## 1. Problem Statement

The scheduler prepares multimodal state at enqueue time and runs prefill later, so a burst of requests admitted in one drain tick prepares request B before request A's prefill reads the shared slot. Two families were also affected sequentially:
- Baichuan's conv state was never reset between requests.
- Mllama's image state was never cleared, so a text-only request after an image request still attended to the old image.

Opting the families out of batching (`supports_batching() == false`) was rejected in the issue for two reasons. It does not stop the sequential leaks, and the burst happens at admission, before decode width matters.

## 2. Change Summary

| Area | Change |
|---|---|
| Baichuan-M1 | Conv state moves to `ModelOwnedSequenceState`, with a CLI fallback reset by `make_caches`. Padded prefill is off. Prompt-cache adopt/donate, the KV handoff and APC (via `HYBRID_SSM_MODEL_TYPES`) decline the model. |
| Mllama | `MllamaImageState` holds the cross states and one cross K/V slot per decoder layer, in a fallback plus a map. It is bound at admission and taken and installed across preemption. A bound text-only sequence carries an explicit text-only marker. An unbound id (the CLI) reads the fallback. Text-only decode rows batch natively, while image rows run one per row. |
| DeepStack | `DeepStackState` follows the MRoPE lifecycle: it is bound by the same admission call, carried in the preemption snapshot (`QwenVlMRopeSnapshot` became a struct) and released together. Qwen3-Omni, previously never bound, is now covered. |
| Core lookahead | `LanguageModel::kv_cache_captures_sequence_state()` (default `true`). When it is `false`, the engine pipelines only if the model can rewind, and calls `rewind_decode_appends` on every lookahead teardown. |
| Baichuan rewind | Each layer keeps a bounded conv-state history (`DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS + 1` entries) and fails closed past it. Prefill tokens are copied out of the fused QKV buffer, so the history does not pin about 1.1 GB per 2048-token chunk. |

## 3. Technical Decisions

### 3.1 Reuse the existing fallback-plus-map shape

The repository had already solved this class twice: `MRopeState` and `Gemma4PerLayerInputsState`. Using the same shape here keeps the admission bind, the preemption take/install and the release in the scheduler uniform across families, and lets the CLI keep a single anonymous slot.

### 3.2 Unbound ids read the fallback; bound text-only rows never do

The first implementation treated an id with no entry as text-only. pr-reviewer found that this broke `mlxcel generate --image` on Mllama: the CLI runs under its own `SequenceId` and never calls the scheduler bind, so the model ignored the image. The final rule mirrors `MRopeState`:
- An unbound id reads the fallback.
- A sequence bound with no image is recorded as an explicit text-only entry.

With that rule, an image left in the fallback after a failed admission cannot reach a bound text-only row.

### 3.3 Fix the lookahead teardown in the engine, not in Baichuan alone

The lookahead teardown rewound model state only for families whose natural backend is model-owned. Baichuan is dense: its K/V lives in the scheduler slice, but its conv state does not. The new trait hook states exactly that property. The same predicate also drives the prompt-cache and handoff gates, so any future dense model with side state needs to answer one question to be correct on all three paths.

The rejected alternative was disabling lookahead for Baichuan. That would have been correct but slower, and it would have left the next such model exposed.

## 4. Validation and Measured Results

### 4.1 Red-then-green tests

Each new test was committed failing before its fix:
- Baichuan: consecutive generations inherit conv state; interleaved sequences differ; a lookahead unwind re-run differs from the synchronous step (1.09e-3); the prefill projection is pinned (64 MiB retained).
- Mllama: a text-only sequence after an image sequence; images bound before prefill; an unbound CLI sequence reading the fallback.
- DeepStack, for each of the three text models: bound entries; a text-only offset-0 forward.

### 4.2 Real checkpoints (M1 Ultra, release builds)

The serving check sends two different requests at once on `--parallel 4`, then a text-only request after an image request, all greedy, and compares each against the same request alone on a fresh server.

| Checkpoint | `main` 01047d75 | Branch 66888608 |
|---|---|---|
| Qwen3-VL-4B-Instruct-4bit | B diverges | identical |
| Llama-3.2-11B-Vision-Instruct-4bit | B describes A's image; text-only after image diverges | identical |
| Baichuan-M1-14B-Instruct-4bit | all diverge | identical at `max_tokens` 40 and 48 |

Single-request CLI output (`--temp 0`) is identical between `main` 1cb18ccc and the branch for Qwen3-VL and Mllama, with both text and image prompts. Baichuan-M1 single-request output changed on purpose, from `# # # ...` to coherent text that agrees in substance with mlx-lm. That comparison loaded the weights through mlx-lm and tokenized with sentencepiece directly; it is qualitative, not token parity.

## 5. Learning Points

- **A failure seen only at the final token is a symptom worth chasing.** It was deterministic, it vanished when logprobs forced synchronous decode, and it did not occur on Qwen3-VL. That combination pointed at the lookahead teardown rather than at numerics.
- **An MLX slice is a view.** Storing a one-token slice of a large projection keeps the whole buffer alive, so state that outlives a forward must be made contiguous.
- **A single-request identity check must cover every input mode the change touches.** The first comparison used a text prompt only and missed the Mllama CLI image regression.

## 6. Follow-up and Limitations

- With `--enable-vlm-prefix-cache`, an admission failure after media preparation leaves fallback slots filled. This is pre-existing for every family and outside this PR.
- Cohere Compass has no local checkpoint, so its synthetic tests are the gate.
- Epic #2566 Metal acceptance (#2587) still has to rerun the affected VLM rows on the final integrated revision.

## References

- Issue #2567, epic #2574, related #2557, #2564, #2566, #2587
- `src/vision/mllama_image_state.rs`, `src/models/deepstack_state.rs`, `src/models/baichuan.rs`, `src/lib/mlxcel-core/src/engine/lookahead.rs`
