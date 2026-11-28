# Technical Report: PR #1532 - Inkling text backbone

**Date**: 2026-08-31
**Author**: mlxcel maintainers
**Status**: Completed; synthetic and CI gates pass, while real-checkpoint validation remains deferred
**Languages**: Rust, Markdown
**Risk Level**: High

---

## Executive Summary

PR #1532 adds the Inkling text architecture as the prerequisite of epic #1313. The implementation covers the hybrid sliding/global decoder, learned banded relative-position attention without RoPE, four short-convolution states per layer, dense and sparse MoE blocks, three checkpoint weight layouts, model-owned cache snapshots, server-efficient last-logits projection, and Inkling reasoning markers. It deliberately leaves image, audio, video, MTP, fused kernels, tensor parallelism, and padded batching to the dependent issues.

The most important outcome is not simply that a new `ModelType` exists. Inkling combines stateful convolutions with a front-trimmed sliding KV window, and its router normalizes routed and shared experts together after selecting routed experts with a correction bias. Both mechanisms can return finite, correctly shaped output when implemented incorrectly. The PR therefore relies on small deterministic numerical references, cache rollback tests, malformed-weight rejection, and independent review rather than shape-only coverage.

## 1. Problem Statement

Inkling checkpoints identify themselves as `inkling_mm_model` / `InklingForConditionalGeneration`, but mlxcel previously had no detector, config, loader, decoder, cache contract, or output-marker support for the family. The architecture cannot be expressed as a small variation of an existing model: it has no RoPE, uses a learned distance band as the additive attention mask, applies short depthwise convolutions to K, V, and both residual branches, and combines routed and always-on shared experts through one logsigmoid-softmax distribution.

Checkpoint compatibility adds another independent risk surface. Original bf16/f32 weights use `model.llm.*` names and interleaved gate/up planes; the native ModelOpt NVFP4 release uses packed expert tensors and nested `hf_quant_config.json` metadata; the MLX community conversion uses affine 4-bit expert triplets. Silent acceptance of an incomplete sidecar or a packed width inconsistent with the configured input dimension can reach infallible MLX/CXX calls, so validation must fail before model construction.

## 2. Change Summary

| Area | Result |
| --- | --- |
| Architecture | Added embedding RMSNorm, hybrid sliding/global NoPE attention, banded relative logits, log-position temperature, f32 short-convolution states, dense SwiGLU, and routed/shared MoE |
| State | Added bounded sliding-KV restoration, visible-window snapshots, four convolution slots per layer, and tail rollback semantics |
| Weights | Added original bf16/f32, native NVFP4, and pre-converted affine MLX 4-bit sanitization with strict shape, packing, dtype, and sidecar validation |
| Loading | Registered `inkling_mm_model` and `inkling` across detection, metadata, directory loading, owned-weight loading, and `LoadedModel` dispatch |
| Server | Added sequence-aware last-logits hooks so prompt prefills slice the final hidden row before projecting to the 201,024-token vocabulary |
| Output | Added tokenizer, streaming, non-streaming, and thinking-budget handling for `<|content_thinking|>` / `<|end_message|>` |
| Documentation | Added the Inkling text-only support boundary to `docs/supported-models.md` |

## 3. Technical Decisions

### 3.1 Follow upstream graph semantics, not its fused kernels

The implementation was reconciled against the public mlx-vlm Inkling files: [language.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/inkling/language.py), [inkling.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/inkling/inkling.py), and [config.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/inkling/config.py). The port follows their relative-bias, routing, shared-expert, cache-ordering, and weight-mapping semantics, but keeps the issue's required graph path. Upstream Metal kernels for the mask, short convolution, router, q4 down-combine, and fused QKVR projection remain optimization follow-ups.

### 3.2 Keep convolution state and sliding-KV state distinct

Each layer stores one KV cache and four f32 convolution states. Front-trimming old attention keys must not clear convolution state, because each convolution slot already contains the latest `kernel_size - 1` activations. Conversely, tail rollback after padded or speculative positions cannot rewind those states from a KV length alone, so the rollback hook trims the KV tail and clears convolution state rather than retaining a stale future.

Snapshots serialize only the visible KV window, not reserved backing capacity. Restore also preserves the absolute offset and reconstructs the internal live-window start. This was a review-driven correction: serializing the raw allocated slab made a ten-token cache look like a 256-token live window and either rejected restoration or shifted relative distances.

### 3.3 Select with correction bias, weight without it

Routed expert selection uses `sigmoid(logit) + correction_bias`, but normalized contribution weights use the raw logits of the selected routed experts plus the raw shared-expert logits. The common distribution is logsigmoid followed by softmax, multiplied by route and learned global scales. Native NVFP4 additionally applies per-expert gate and output sidecars on opposite sides of SwiGLU. Tests compare this path against CPU scalar references with different token and top-k dimensions so broadcast-axis mistakes cannot hide behind equal dimensions.

### 3.4 Treat checkpoint metadata as untrusted input

The sanitizer and load validation reject incomplete triplets, reverse mixed dtypes, inconsistent packed widths, scale/bias leading-shape mismatches, unknown NVFP4 sidecars, non-integer group sizes, invalid schedules, and overflowing dimension conversions before invoking MLX operations. Native NVFP4 detection accepts the real nested `quantization.quant_algo` / `quantization.group_size` sidecar structure rather than assuming top-level fields.

### 3.5 Avoid full-vocabulary prompt projection

Inkling's padded vocabulary is 201,024 rows. Projecting every prompt position during server prefill can create tens of GiB of temporary logits on long requests even though sampling needs one row. A backward-compatible sequence-aware last-logits trait hook was added to the core generation interface and wired through full, chunked, and prompt-cache server prefills. Existing models retain the default behavior; Inkling slices the hidden state before the LM head.

## 4. Review and Hardening

Independent correctness and finalizer reviews found and fixed several material defects before merge:

- Visible-window snapshot serialization did not match the cache's reserved-capacity representation.
- A cache rollback hook removed the oldest prefix with `trim_front` when its contract required removing the newest tail.
- Sequence-ID server prefills bypassed the last-logits optimization and projected the full vocabulary.
- Native NVFP4 sidecar promotion looked at the wrong JSON nesting level.
- Quantized planes did not fully validate packed input widths or scale/bias leading axes.
- Mixed floating/byte expert planes could reach the wrong runtime path.
- Expert-scale broadcasting produced the wrong axis order when token count and top-k differed.
- Non-streaming parsing could interpret an ordinary Inkling answer's `<|end_message|>` as a reasoning close and delete the answer.
- Schedule typos, sidecar I/O/type errors, and hostile integer products required stricter rejection.

No unresolved CRITICAL or HIGH correctness, security, or performance findings remained at merge.

## 5. Validation

| Gate | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Pass |
| `cargo check --lib --features metal,accelerate` | Pass |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | Pass locally and in CI |
| Inkling-focused unit suite | 26/26 |
| Expert-scale CPU reference | 1/1, N=3 and K=2 |
| Registry exhaustiveness | 2/2 |
| GitHub CI | Format, clippy, deny, OpenXLA feature compile, manifests, cross-repo references, and CLA all pass |

The focused suite covers config precedence, width aliases, banded bias, log-temperature, short-convolution continuation, sliding/global causal prefill, token-by-token decode parity, router correction-bias exclusion, shared-expert CPU equivalence, native NVFP4 and affine sanitizer success paths, malformed input rejection, cache snapshot/restore, EOS, detection aliases, and all reasoning-marker paths.

## 6. Validation Limits and Follow-up

No Inkling checkpoint was available locally. The public affine MLX checkpoint is approximately 153.5 GB and the native NVFP4 release is approximately 170.7 GB, both beyond the practical memory scope of this 121 GiB host. Token-exact comparison on `Inkling-0.6B-A0.6B`, fluent generation on Inkling-Small, real throughput, and CUDA validation therefore remain unverified and must be run on suitable hardware.

The dependent epic work adds HMLP vision, dMel audio, temporal-pair video, and the native MTP drafter. Performance follow-ups may add the fused mask, short-convolution, router, q4 down-combine, and QKVR kernels after real-checkpoint parity is established.

## References

- Epic #1313 and issue #1318
- PR #1532, squash commit `5690012fee9ec9053a2cde5984c4b5ada0eb27ec`
- Public mlx-vlm Inkling implementation linked in section 3.1
- `docs/supported-models.md`
