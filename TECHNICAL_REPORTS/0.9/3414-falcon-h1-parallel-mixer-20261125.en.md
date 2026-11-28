# PR #3414: Falcon-H1 parallel mixer and exact plain CLI output

**Date**: 2026-11-25

**Status**: Integrated CPU/CUDA acceptance passed; final-head ROCm replay pending externally

## Executive Summary

Falcon-H1 now executes original Hugging Face weights through OpenXLA/IREE. Each layer evaluates attention and Mamba2 from the same normalized input, adds their outputs to the residual, then applies a separately normalized MLP. Prefill, single decode, and ragged decode share this ordering and maintain both KV and recurrent state per layer.

Acceptance also exposed a shared CLI display bug: plain completions lost their trailing whitespace through a chat-oriented filter. Full CLI rendering now preserves plain completion bytes while retaining existing reasoning, tool, Harmony, Muse, and default chat cleanup.

## Problem Statement

The existing `parallel_block` means parallel attention and MLP, which is not Falcon-H1's architecture. Original HF weights also require explicit muP factors; assuming converted, pre-scaled MLX weights would silently change the output. Independent attention/SSM head widths, explicit intermediate dimensions, optional SSM normalization, and independent input/output biases must all survive loading.

The previous server acceptance harness decoded CLI token IDs rather than comparing displayed stdout. External ROCm testing found that a Germany completion ended with two spaces and a newline: HTTP retained them, while CLI display removed them. The old result therefore proved raw-decoded text/count parity, not strict displayed-text parity.

## Change Summary and Technical Decisions

The loader uses canonical `mamba_d_head`, validates original checkpoint shapes, and applies embedding, attention, key, SSM channel, MLP, and output multipliers at their HF positions. Optional grouped SSM normalization supports both gating orders and absent normalization. Unsupported attention schedules and RoPE variants fail explicitly. The established recurrent f32 default and prefix-cache guards remain in effect.

Integration with PRs #3411 and #3413 preserves MoE routing and Mamba1 state layouts. The shared normalized-input mixer helper dispatches through `SsmMixerW`; Falcon-H1 explicitly selects `SsmKind::Mamba2`, and its normalization/bias/multiplier rules remain in that variant. Both loader and graph dispatch retain pure Mamba1 and Mamba2 handling. Registered goldens and every-family weight-layout agreement verify that integration.

`StreamFilter::flush` retains its existing chat contract. An explicit CLI-only flush mode preserves trailing whitespace only when no structural delimiter was consumed and the prompt did not prime reasoning. Raw streaming CLI already forwards text unchanged. Tests cover the exact whitespace suffix, Unicode and whitespace-only text, chunk accounting, primed reasoning, tool calls, Harmony/Muse channels, and EOF delimiter handling.

## Validation and Evidence Boundaries

The integrated source is `8c7b4412d37b91807f8d688ba2f3febbc7a7e41f`, based on `2e46fc09e47208898e708d7090b196ec47b22357`. The full XLA crate passed 560 tests with 22 ignored, including registered byte-exact goldens and every-family loader/emitter agreement. On the same integrated source, all seven focused display tests and 146 StreamFilter tests passed. Both CPU and CUDA passed the three-mode synthetic/ragged/slot-reuse gate and matched the real checkpoint's 40-token independent HF oracle. The CUDA CLI matched the same 40-token oracle. Actual displayed stdout matched all four non-streaming HTTP and all four SSE responses, including Germany's exact trailing two spaces and newline; token counts also matched. Both groups overlapped, the owned server was reaped, and the GPU lock was released. Five hybrid and six Mamba2 single-sequence regressions passed. Formatting, Python syntax, and narrow XLA clippy with `-D warnings` passed. The final CI allowlist classifies two test-only Falcon controls; the environment checker passed (398 discovered, 265 documented, 133 allowlisted), with all 15 Python regression tests passing. Only this CI metadata and evidence documentation follow the tested runtime source. The committed `integrated-validation.json` and `integrated-cuda-displayed-*` artifacts record the results, binary hashes, raw Germany stdout, and HTTP/SSE responses.

Earlier CPU/CUDA source `205d0a3dd2d92eca2dbcba3dffcefba54e825eca` passed three synthetic normalization modes, four ragged trajectories and slot reuse, the real Tiny90M 40-token HF oracle, and batch 4 / requests 8 with 170 reference-exact tokens. Its HTTP evidence used raw token decoding and is explicitly corrected above.

External ROCm gfx1151 acceptance at `852129eb1b4bf45d957f4c63b082d9d0276e4951` passed numerical, real 40-token CLI, batching/recycling, recurrent regressions, and release architecture gates. Strict displayed CLI/SSE text passed three of four prompts; all four SSE strings matched raw CLI token decoding. See the [acceptance record](https://github.com/lablup/mlxcel/pull/3414#issuecomment-7147795531) and [whitespace diagnosis](https://github.com/lablup/mlxcel/pull/3414#issuecomment-7147960509).

The committed `falcon_h1_cli_server_check.py` now compares actual CLI stdout against four concurrent 40-token non-streaming completions and four SSE completions. It preserves generated whitespace, checks token counts and stream termination, records binary hashes and raw responses, and reaps the owned server. HTTP does not expose output token IDs, so this remains exact text/count evidence.

## Learning Points and Follow-up

Output parity must compare the actual user-visible bytes; reconstructing text from token IDs can hide a display transformation. Chat cleanup and raw completion fidelity need explicit caller policies.

Request an external ROCm replay on the final integrated head for both the common text behavior and merged recurrent plumbing. Use the committed handoff with explicit `MLXCEL_XLA_DEVICE=hip`, `MLXCEL_XLA_ROCM_TARGET=gfx1151`, and `MLXCEL_XLA_PRECISION=f32` under `scripts/rocm_gpu_guard.sh --hold --`. Earlier HIP evidence, new CPU tests, CUDA replay, and canceled hosted CI are distinct observations; none substitutes for a new HIP run.
