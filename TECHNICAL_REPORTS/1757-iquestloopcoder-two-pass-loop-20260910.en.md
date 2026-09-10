# Technical Report: PR #1757 - feat(models): add the IQuest-Coder Loop two-pass decoder (iquestloopcoder)

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation and validation cycle
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium (new model family, no tensor-parallel or pipeline-parallel support, one tokenizer gap shared with the sibling family and deliberately left in place, and one acceptance criterion this checkpoint makes unmeasurable)

---

## Executive Summary

PR #1757 adds `model_type: "iquestloopcoder"`, the two-pass decoder released as `mlx-community/IQuest-Coder-V1-40B-Loop-Instruct-4bit` (21 GB, 4-bit affine, group_size 64). The shape is ordinary Llama: 80 layers, hidden 5120, 40 attention heads over 8 KV heads at head_dim 128, intermediate 27648, vocab 76800, rms_norm_eps 1e-5, rope_theta 500000, `tie_word_embeddings` false, `eos_token_id` `[2, 75864, 75869]`. What is new is that those 80 layers run twice over the same tokens with the same weights, under `loop_num` 2 and `loop_window_size` 64.

Pass 1 is plain causal attention and stores its K/V. In pass 2 each layer's attention output is a per-head sigmoid-gated mix of two branches: a global branch attending the K/V pass 1 produced for that same layer, and a local branch attending pass 2's own K/V through a 64-token sliding window. The gate is `sigmoid(q2 . gate_w[h] + gate_b[h])`, computed from the post-RoPE pass-2 query. `model.gate_projections.{i}.weight` `[40,128]` and `.bias` `[40]` are the only pass-2-specific weights and the only tensors in the checkpoint that are never quantized. Every layer therefore owns two caches: a dense `KVCache` for pass 1 and a `RotatingKVCache(64)` for pass 2.

Validation is a streaming float32 NumPy oracle written from the checkpoint's own `modeling_iquestloopcoder.py`, cross-checked against the vendor classes at 1.1e-06 on 16 of 16 shape cases. Greedy argmax matches on both test prompts and the 12-token continuation is identical. The window itself could not be validated that way, and section 6 gives the measurement that shows why: every layer's gate bias is exactly +2.0, so the local branch carries about 12 percent of the output and removing the window entirely changes no argmax and no top-10 order at 218 or 521 tokens. 14 files, +2681, 17 new unit tests plus a real-checkpoint parity harness of five checks.

---

## 1. Problem Statement

### 1.1 What the architecture is

The vendor ships `modeling_iquestloopcoder.py` alongside the weights. Its `_forward_loop` runs the whole 80-layer stack, keeps the per-layer K/V, then runs the same stack again on its own output. Pass 1 attention is causal and unremarkable. Pass 2 computes one query per token and attends it twice: once against the pass-1 K/V of the same layer, once against K/V it computes itself, masked to the last 64 positions. The two results are blended per head by a scalar gate read off the pass-2 query.

### 1.2 Why each piece is dangerous

- **Wrong cache in pass 2.** The global branch reading pass 2's K/V instead of pass 1's is a one-token edit that never errors: the model still attends over a full causal history and still produces fluent code.
- **Wrong query for the gate.** Gating on the pre-RoPE query keeps the shape, keeps the range, and shifts the mix by a few percent per head.
- **Dropped window.** The local branch without its 64-token limit is a strictly wider attention, so nothing overflows and no assertion fires.
- **Cache surface mismatch.** The generator's `Vec<KVCache>` cannot describe two caches per layer where one of them rotates, and the trait default infers cache layout from an unrelated capability flag, which is the Falcon-OCR precedent (PR #1075) where a server ran zero decoder layers.

---

## 2. Technical Decisions

### 2.1 Splitting `Attention` into three methods

The shared `llama3::Attention::forward` fuses projection, cache update and SDPA around a single cache. Pass 2 has one query and two K/V sets living in two different caches, so `src/models/iquestloopcoder.rs` exposes `get_qkv(x, offset)`, `attend(q, k, v, window)` and `project_out(attn)` separately. `llama3::MLP` is reused by direct construction, its fields being public, so the tuned SwiGLU path is shared rather than copied.

`attend` delegates to `mlxcel_core::causal_attention(q, k, v, scale, 0.0, window)` rather than building masks locally. That helper aligns causality bottom-right (`offset = k_len - q_len`), which is what both branches want, and it is already the helper that pairs correctly with `RotatingKVCache`: on a prefill longer than the window it keeps every key and enforces the window with a full-width mask instead of slicing K/V, which would leave the earliest query rows with an all `-inf` softmax row. Decode (`q_len == 1`) takes the maskless path, because the rotating cache already holds at most 64 keys. No mask is materialized on the common paths.

### 2.2 Model-owned sequence state, and what it settles

The state lives in a `ModelOwnedSequenceState<LayerCaches>` keyed by `SequenceId`, the same mechanism Gemma 3, Llama 4 and Qwen3-Next use. `make_caches()` returns an empty vector and `sequence_state_layout()` returns `SequenceStateLayout::model_owned(num_layers)`.

That declaration also settles prefix caching. `BatchScheduler`'s prompt-cache donate and adopt paths both return early on `SequenceStateBackend::ModelOwned` and record `PromptCacheRejectReason::ModelOwnedState`, so the family is excluded structurally and a request records a skip rather than a misleading cache miss. Snapshot reuse is off (the trait default) and `supports_batching` is false. `supports_padded_prefill` is false because pad tokens appended to a rotating cache that has already wrapped cannot be trimmed back out, unlike a dense cache. Tensor parallelism is refused rather than approximated: `fallback_architecture` returns `"iquestloopcoder"` and not `"llama"`, because returning `"llama"` would switch on the TP Llama runtime for a decoder it cannot serve.

This is a deviation from the issue's plan, and worth recording as such. Issue #1360 proposed `make_caches` returning `2 * num_hidden_layers` flat `KVCache` entries ordered `[pass1_0..., pass2_0...]`. That was not implemented: a flat dense list cannot represent the rotating half, and it would hand the scheduler per-sequence state it does not actually own. The model-owned route gives real per-sequence isolation instead, verified on the running server (section 8).

### 2.3 Detection guards

`"iquestloopcoder"` routes through `iquest_loop_coder_model_type`, which refuses `loop_num != 2` before any weight is read, and a weight read here is 21 GB. A `loop_num` it cannot parse as a whole number fails closed rather than falling back to the vendor default of 2.

A config relabelled `"model_type": "llama"` while still declaring `architectures: ["IQuestLoopCoderForCausalLM"]` is also routed to the loop decoder, guarded ahead of the plain Llama arm. Relabelling to `llama` is a real practice for this family, since it is how a checkpoint is made loadable by stacks that will not run its `auto_map` code, and falling through would run the stack once instead of twice and never read a single `gate_projections` tensor, producing fluent output from half the model. The architecture string is compared for equality rather than substring, because `IQuestLoopCoderForCausalLM` and the sibling family's `IQuestCoderForCausalLM` differ only by an infix.

### 2.4 Three divergences from the vendor decode path

`_forward_with_cache` is the vendor's single-token decode path and disagrees with `_forward_loop` in two ways this port does not reproduce, because reproducing them would make decode disagree with the semantics the model was trained under. First, its prefill writes the pass-2 local cache from K/V recomputed from the hidden state after the layer already ran, not from the K/V the layer's own pass-2 attention used. Second, its local cache never shrinks to `loop_window_size` after a prompt longer than the window, because `update_local` seeds it with every prompt token and then evicts one per step, so the window stays as wide as the prompt. A third quirk sits in `forward_decode_loop2`, which gates the window mask on `q2.shape[2]`, equal to 1 during decode, so the window is never applied there either. Prefill logits are unaffected by all three.

One further finding concerns `_forward_loop` itself. With `use_cache=False` the shared cache is never allocated, a fallback fires that recomputes K1/V1 from the pass-2 hidden state, and attention A and B then run on the same K/V differing only in the mask. `use_cache` defaults to `config.use_cache = true`, so every real `generate()` prefill takes the pass-1-KV path, which is the path this port implements.

---

## 3. Change Summary

| Area | Change |
| --- | --- |
| `src/models/iquestloopcoder.rs` | Config, `LoopGate`, `Attention` split into `get_qkv` / `attend` / `project_out`, `TransformerBlock`, `LayerCaches` |
| `src/models/iquestloopcoder_model.rs` | `IQuestLoopCoderModel`, `IQuestLoopCoderWrapper`, `sanitize_weights` |
| `src/models/detection.rs`, `detection_tests.rs` | `iquest_loop_coder_model_type`, the `loop_num` guard, the relabelled-config arm ahead of Llama |
| `src/models/mod.rs`, `registry.rs`, `src/loaded_model.rs`, `src/model_metadata.rs`, `src/execution/memory_estimate.rs` | Registration, metadata, memory estimation |
| `src/distributed/tensor_parallel/inference.rs` | `fallback_architecture` refuses the TP Llama runtime for this family |
| `src/tokenizer/mod.rs` | Pins the non-special added-token behavior described in section 7 |
| `src/models/iquestloopcoder_tests.rs`, `tests/iquestloopcoder_parity.rs`, `docs/supported-models.md` | 17 unit tests, the real-checkpoint harness, docs |

---

## 4. Testing: why the unit tests are differential

All three failure modes in 1.2 produce fluent output when wrong, so the tests build the wrong variant explicitly and assert the implementation does not match it.

`pass2_matches_only_the_correct_reference` compares the implementation's pass-2 attention output against a reference composed in the test, in four variants: correct, `GateFromPreRope`, `GlobalReadsPass2Kv`, and `LocalUnwindowed`. It requires a max abs difference below 1e-5 against the correct one and above 1e-3 against each wrong one, on a 12-token run against a window of 4.

`window_limits_local_branch` rewrites keys 0..7 of a 12-token run with window 4 and asserts that position 11 is unchanged (below 1e-5) while position 7 moves (above 1e-3). It separately asserts that an unwindowed branch would have moved position 11, so the test is not vacuous. 17 unit tests in total, all passing.

---

## 5. Real-checkpoint Validation

The reference is a streaming float32 NumPy implementation of the two-pass loop, written from the checkpoint's own `modeling_iquestloopcoder.py` and dequantizing the 4-bit affine weights itself one layer at a time, for a peak RSS of 1.5 GiB against a model whose fp32 form would be 160 GB. It shares no code with mlxcel. A tiny random-weight fp32 torch model built from the vendor classes agrees with it on 16 of 16 shape cases spanning `L <= window`, `window < L <= 2*window` and `L > 2*window`, at a max abs difference of 1.1e-06.

The MLX affine dequantization formula, `w[o,i] = q[o,i] * scales[o, i//64] + biases[o, i//64]` with `q` unpacked low-nibble-first, was verified four independent ways. The decisive check: MLX maps the larger-magnitude group endpoint to code 0, so about half the scales are negative and `bias` is the value at `q=0`, and for every group hitting both code 0 and code 15 the reconstructed group min and max match the true group min and max at exactly 0.0 difference, which rules out a `(q-8)*scale` reading. Nibble order was pinned by correlating the per-input-column RMS of `q_proj` against `|input_layernorm.weight|`: +0.7564 low-first against -0.0096 high-first.

Results against mlxcel on CUDA sm_121, last-position logits:

| prompt | tokens | argmax match | KL(oracle \|\| mlxcel) | logit correlation |
| --- | --- | --- | --- | --- |
| short chat | 38 | yes (66644) | 0.029 | 0.9895 |
| long chat | 218 | yes (5123) | 0.0047 | 0.9980 |

The greedy continuation from the 38-token prompt matches the oracle for all 12 tokens, decoding to `# Python Function to Reverse a String\n\nHere are several`.

---

## 6. The Negative Result: the oracle cannot validate the window

Issue #1360's acceptance criterion required an oracle comparison on a prompt longer than 128 tokens, on the stated grounds that past `loop_window_size` the local branch drops keys, so an unwindowed implementation would diverge. On this checkpoint that premise is false.

Every layer's `gate_projections.{i}.bias` is exactly +2.0 and the gate weights are small, so the gate lands at `sigmoid(2.0) = 0.8808` in all 80 layers. Measured per layer on the 218-token prompt: min 0.874602 (layer 64), max 0.884268 (layer 77), mean 0.879620, std 0.001995. The local branch therefore carries only about 12 percent of the mixed output.

Re-running the oracle with the window removed gives:

| | 218 tokens | 521 tokens |
| --- | --- | --- |
| argmax changed | no | no |
| top-10 order changed | no | no |
| mean abs logit delta | 5.01e-02 | 8.94e-03 |
| max abs logit delta | 3.37e-01 | 5.34e-02 |

The effect shrinks with length rather than growing. Meanwhile mlxcel runs f16 activations through a quantized `lm_head` against the oracle's f32, and its own deviation, after removing the constant offset the fitted relation `mlxcel = 1.09477 * oracle - 0.15529` describes, is mean 2.674e-01, which is 5.3 times the entire window effect. Measured directly on the 218-token prompt, mlxcel sits 0.2674 from the windowed oracle and 0.2678 from the unwindowed one. There is no separation.

Stated plainly: a greedy-id or logit comparison against a real-checkpoint oracle cannot distinguish a windowed implementation from an unwindowed one on this checkpoint, at any prompt length. The window is validated instead by the unit tests of section 4, which isolate the local branch where the contribution is not diluted by the gate. The acceptance criterion is recorded as met in the part that is measurable (structure, gate, pass-1 KV routing, RoPE, weight load, decode handover) and explicitly not met in the part that is not (window discrimination via the oracle).

A second consequence follows for the parity harness. The top-5 ordering assertion in `tests/iquestloopcoder_parity.rs` is applied only between entries the oracle separates by more than mlxcel can resolve. mlxcel's quantized `lm_head` resolves about 0.125 between adjacent logits at magnitude 16 to 32, and the 218-token prompt's oracle ranks 2 and 3 are 0.0395 apart, so their relative order is noise and is not asserted.

---

## 7. Tokenizer Finding

The checkpoint ships SentencePiece `tokenizer.model` plus `added_tokens.json` and `tokenizer_config.json`, with no `tokenizer.json`, `add_bos_token` false and `add_prefix_space` false. `added_tokens_decoder` holds 30 entries, of which 27 are added tokens; the other 3 are SentencePiece's own `<unk>`, `<s>` and `</s>`.

19 of the 27 are marked `"special": true` and round-trip in both directions: `token_to_id` resolves them, they encode as exactly one id, and they decode back to themselves. The other 8 are marked `"special": false` (`<think>`, `</think>`, `<tools>`, `</tools>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`). Those decode correctly, but `token_to_id` returns `None` for them and they encode into several pieces: `<think>` becomes `[66580, 30272, 66604]`, `</tool_call>` becomes `[469, 7005, 66561, 2998, 66604]`.

This is deliberate in the shared loader. `parse_special_tokens` in `src/tokenizer/mod.rs` puts non-special added tokens into a decode-only `added_token_contents` map, while HuggingFace matches every `added_tokens_decoder` entry on encode regardless of `special`, so the two differ. It is not introduced by this port and it affects the `iquestcoder` sibling identically. It is observable only on a tool-calling prompt: `chat_template.jinja` emits the six tool tags literally and emits neither `<think>` nor `</think>`. The behavior is pinned by a test rather than left undocumented, and a follow-up issue was filed. Changing it would touch every SentencePiece checkpoint in the repository, which is why it was not done here.

---

## 8. Runtime Evidence

`mlxcel inspect` estimates 28.08 GiB total for 8192 tokens of context: weights 20.85 GiB, KV 2.50 GiB, allocator overhead 4.67 GiB. Decode measured at 4.3 to 4.5 tok/s on GB10 for 48 to 64 tokens, consistent with two passes over 80 layers.

`mlxcel-server` was validated separately from the CLI, because a model can pass every CLI test while the server runs a different path. Three concurrent requests with three different prompts produced output byte-identical to their serial baselines, which is the per-sequence isolation `ModelOwnedSequenceState` exists to provide.

---

## 9. What Was Not Run

- `cargo test --workspace --profile test-fast --features metal,accelerate`, listed as an acceptance criterion in issue #1360, cannot run on this host (Linux aarch64, CUDA, no Metal, no Accelerate). It was not run. Apple Silicon behavior for this family is unverified.
- The CUDA equivalents were run scoped instead: `cargo clippy --lib --tests --features cuda -- -D warnings` and `cargo fmt --all -- --check` are both clean, and the test suites named above all pass.
- Tensor parallelism and pipeline parallelism are not enabled for this family.
- Turbo and INT8 KV cache modes are not wired for the rotating pass-2 cache, so the family stays FP16-only.
- `--max-kv-size` trimming does not reach model-owned caches, matching the other model-owned families.

---

## 10. Lessons

- **An acceptance criterion can be falsified by the checkpoint it names.** #1360's 128-token threshold rested on the window mattering at the output. A +2.0 gate bias in all 80 layers puts the local branch at 12 percent, below the noise mlxcel's own quantized `lm_head` contributes, so the criterion was unmeasurable rather than unmet.
- **The vendor's decode path is not automatically the specification.** `_forward_with_cache` disagrees with `_forward_loop` on which K/V seed the local cache and on whether the window ever narrows to 64, so copying it would have made decode disagree with the semantics the weights were trained under.
- **Tokenizer round-trip is directional.** All 8 non-special added tokens decode correctly and only fail on encode, so a decode-side check alone would have reported this family clean.
