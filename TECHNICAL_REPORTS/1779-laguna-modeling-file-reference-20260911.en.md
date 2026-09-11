# Technical Report: PR #1779 - test(laguna): add a checkpoint-code reference trace and forward tests

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle (implementation review, security and performance review)
**Status**: Completed
**Languages**: Rust, Python (out-of-band tooling only)
**Risk Level**: Low (no runtime change: the release binary's traces are byte-identical before and after; new tests, a script pair and a docs entry)

---

## Executive Summary

The Laguna port (PR #1739) was validated without an external reference, because the pinned mlx-lm has no `laguna.py`. Laguna S 2.1 was never run, and two config paths, `sqrtsoftplus` routing and sliding-layer attention sinks, had no forward-pass coverage. This PR adds a reference arm that runs the checkpoint's own `modeling_laguna.py` in float32 and writes the `examples/logit_trace` TSV. It adds two synthetic-config forward tests and records the S 2.1 run.

On XS 2.1, mlxcel shows no top-1 disagreement at a position the reference had decided, at every width where the window carries its BOS token. S 2.1 loads, generates fluent Python and keeps every traced logit finite.

---

## Problem Statement

The PR #1739 report left three gaps:

- **No external oracle.** The only comparison available was fused against unfused decode on `Laguna-XS.2-4bit`. That checkpoint declares an `attention_factor` of 1.0 where the derived value is 1.3466, so it cannot serve as a parity target anyway.
- **S 2.1 never ran.** Its 48-layer, top-10 geometry was covered by unit tests only.
- **Two config paths never ran.** No published checkpoint enables `sqrtsoftplus` or sinks. The router formula had an isolated test. The sink path had none.

---

## Change Summary

- **`scripts/laguna_oracle_trace.py`** takes `logit_trace`'s six positional arguments and reproduces its token stream: the tokenizer, the `bos_prefix` rule, chunk slicing, the prefill context pass and the emitted positions. It writes the same columns, so `compare_logit_traces.py` reads it as one arm unmodified.
  - Each chunk's logits come from one float32 forward over context plus chunk.
  - Chunks that are prefixes of one another share a forward.
  - It refuses configs that enable `sqrtsoftplus` or sinks, because the reference implements neither.
- **`scripts/compressed_tensors_nvfp4.py`** reads the checkpoint with `safe_open` and decodes NVFP4 as `code * E4M3 scale / global` in float32. It expands a MoE layer's experts only while that layer runs, which keeps XS 2.1 at a 24 GB peak on MPS.
- **Tests** (`src/models/laguna_tests.rs`):
  - `sqrtsoftplus_router_routes_and_weights_a_forward_pass` checks a three-expert block against a plain f64 reference. On one token the chosen pair differs between sigmoid and sqrt-softplus. On the other, the same pair gets 2:1 against about 3.4:1 weights.
  - `sliding_attention_sink_joins_the_softmax_on_prefill_and_decode` zeroes `q_proj` so each head's sink share is exact, and covers masked prefill, unmasked decode, the disabled flag, full layers and a missing weight.
- **Docs.** The Laguna entry in `docs/supported-models.md` now records the S 2.1 run and the reference comparison, replacing "Laguna S 2.1 was not available on the validation host".
- **Review changes.** Security review found that the harness imported from the checkpoint directory itself. There, a `modeling_laguna/` package, a compiled module or an unchecked `.pyc` would run in place of the `.py` file whose hash the trace records. The harness now copies `modeling_laguna.py` and `configuration_laguna.py` (the only relative import) into a temporary directory, hashes the copies before import and imports from there. Both digests go in the trace header. A planted package that raises ran under the old code and is ignored under the new. Smaller fixes:
  - The corpus is read as raw UTF-8, as `logit_trace` reads it.
  - The effective router function is resolved in the same order as `laguna.rs`.
  - Only the experts a layer actually routes to are decoded.
  - The usage text says the checkpoint must come from `hf download`, since `mlxcel download` does not fetch `*.py` files.
  - The prefix-sharing claim is stated as agreement to about 5e-5, not exactness.

  Rerunning all three configurations after these changes reproduced every data row byte for byte.

---

## Technical Decisions

**The checkpoint's own modeling file, not mlx-vlm.** mlx-vlm 0.6.17 folds `1 / weight_global_scale` into the E4M3 block scales and rounds them again. That is the double quantization PR #1739 rejected, so its weights would be less faithful than mlxcel's. The modeling file is what the model's publisher ships. Its only barrier on this host was memory, and decoding the planes in the harness removed it: no `compressed_tensors` install, no bf16 rounding of the dequantized weights.

**One trace format, not a second tool.** The harness writes `logit_trace`'s TSV, so the existing comparison tool and its decided-position gate apply unchanged. This forces the harness to reproduce the token stream exactly, including the quirk that chunk 0 never gets a BOS prefix. Any difference would have compared different positions. The comparison tool checks that every `target` matches, and all 1,280 rows do.

**Refuse what the reference cannot model.** The modeling file scores the router with sigmoid only, and allocates `sink` without ever reading it. A config that enables either would be traced as a different model while the header named the checkpoint. The harness refuses instead, and the two synthetic tests are those paths' only coverage.

**Teacher-forced, not free-running, for the greedy criterion.** A free-running comparison stops meaning anything at the first flipped token. The issue's fallback asks for the first divergence position and the top-two gap there, which a teacher-forced trace reports directly.

---

## Validation

**Reference comparison on XS 2.1** (reference: float32 on MPS; mlxcel: bf16, since all 10,240 traced logits lie on the bf16 grid):

| Width, context | Top-1 disagreement | At decided positions (gap ≥ 2) | Perplexity, reference / mlxcel |
|---|---|---|---|
| 1, behind 512 | 3 / 128 | 0 / 75 | 12.66 / 12.79 |
| 8, behind 512 | 34 / 640 | 0 / 296 | 17.49 / 17.23 |
| 256, BOS-anchored chunk | 18 / 256 | 0 / 107 | 17.77 / 17.86 |
| 256, chunk 0 (no BOS) | 42 / 256 | 1 / 93 | 955 / 999 |

- **The rounding class.** Every disagreement at widths 1 and 8 is at a position the reference itself had not decided. 85% to 100% of them take the reference's runner-up.
- **The window without BOS.** The one decided disagreement is at chunk 0, position 158, a window with no BOS token. There the reference picks token 90 with a gap of 2.71, and mlxcel ranks it second. The reference degrades in that window too (perplexity 955 against 17.8 once the BOS is present), so the collapse is the model's behavior, not mlxcel's.
- **First 64 positions.** At widths 8 and 1, 63 and 62 of the first 64 positions agree. Both first diverge at position 52, where the reference's top-two gap is 0.465. Width 256 first diverges at position 4 (gap 0.442), inside the window without BOS.
- **Harness self-checks.** The NVFP4 decode matches MLX's `nvfp4` dequantize bit for bit on three planes. CPU and MPS agree with 0 of 256 top-1 flips. Prefix sharing matches one forward per chunk to about 5e-5, with no top-1 flips.
- **Binary provenance.** The release binary built at the PR head produces traces byte-identical to a binary built before the PR, so the numbers describe the merged runtime.

**Laguna S 2.1** (`poolside/Laguna-S-2.1-NVFP4`, 49 shards, M1 Ultra with 128 GB, nothing else loaded):

- **Load.** It loads in 17 s and transcodes 117 planes. S 2.1 mixes formats: layers 1 to 39 carry NVFP4 routed experts (39 × 3 planes), layers 40 to 47 keep their routed experts in bf16 (its `ignore` list names `model.layers.4[0-7].mlp.experts`), and the shared experts are bf16 throughout. One checkpoint therefore exercises both the NVFP4 and the per-expert bf16 expert paths. XS 2.1 quantizes its shared experts as well, which is why it shows 234 planes.
- **Memory.** The load-time resident figure, 49.36 GiB, equals the packed NVFP4 bytes plus their E4M3 scales exactly, and XS 2.1 shows the same identity at 16.52 GiB. The 43.5 GiB of bf16 weights realize on the first forward, so the working set is about 93 GiB, under the 107.5 GB wired limit.
- **Generation.** A raw prompt and a chat-template prompt both produce fluent output at 26 to 27 tok/s: a well-formed task specification opening a Python block, and coherent reasoning about backoff and jitter.
- **Trace.** A width-8 trace behind 512 tokens has 640 rows, all finite. On the same positions, perplexity is 8.88 against XS 2.1's 17.23, and top-1 accuracy is 0.598 against 0.517.

**Tests and gate.**

- All 23 `models::laguna` tests pass.
- **Revert arms.** Each covered path, broken, fails only its new test:
  - Router hardcoded to sigmoid: `0.12928326` against a reference of `0.29387191`.
  - Sinks off at load: `1.3130562` against `0.6565281`.
  - Sinks dropped on decode only: `0.25037557` against `0.18778168`.
- **Gate.** On 9c34a829 the workspace gate passed: exit 0, 123 binaries, 11,040 passed, 0 failed. Clippy, fmt, `verify-versions`, `verify-kernel-dtype-keys`, `verify-llama-compat` and the cross-repo reference check are clean.

---

## Learning Points

- **A load-time resident figure can be half the real footprint.** mlxcel's load report counts only what load evaluates, which here is the transcoded planes. The memory-mapped bf16 weights appear at the first forward. S 2.1 reported 49 GiB and needs 93, so size a host from the checkpoint's bytes, not from that line.
- **A shipped reference covers what it implements.** The modeling file allocates `sink` and never reads it. A reference taken from the checkpoint is authoritative for the paths it runs and silent on the rest, and a harness that does not refuse those paths reports agreement for code it never exercised.
- **Split by window before reading a trace.** At width 256 with no prefill, `logit_trace` leaves chunk 0 without its BOS. On a model trained with a document separator, that one chunk runs at perplexity 950 and holds most of the disagreements. The pooled perplexity (130) looks like a defect until the chunks are separated.
- **Importing code from a checkpoint directory runs what Python finds first.** A package directory, a compiled module or an unchecked `.pyc` takes precedence over the `.py` file that was hashed. Copying the source files to a temporary directory and hashing them before import makes the recorded digest the code that ran.

---

## Follow-ups

- `logit_trace` never gives chunk 0 a BOS prefix, although its own comment says every chunk needs one (#686). Any family that collapses without BOS shows a misleading first chunk at every width.
- The `// Used by: GptOss` comment on `fast_scaled_dot_product_attention_with_sinks` (three sites in mlxcel-core) omits Laguna, DeepSeek-V4 and Falcon-OCR, and describes a sink as a first-position bias rather than an extra logit in the softmax denominator.
- Neither arm emulates the checkpoint's NVFP4 activation quantization or its FP8 KV cache.
- S 2.1 has no external reference on this host: its float32 weights exceed memory.
- The fused affine decode divergence on `Laguna-XS.2-4bit` (PR #1739 report, section 5.3) is unchanged and needs its own issue.
