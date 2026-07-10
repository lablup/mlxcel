# TurboQuant KV cache

TurboQuant modes reduce KV-cache memory by quantizing K and/or V cache tensors.
The implementation is experimental in the sense that quality and speed vary by
model family, cache mode, hardware, and server path. Use the default FP16 cache
unless you have measured the target model and workload.

The TurboQuant algorithms (PolarQuant rotation, Lloyd-Max codebooks, layer-aware V protection, sparse-V dequant) are a Rust port of [turboquant_plus](https://github.com/TheTom/turboquant_plus), Copyright 2026 Tom Turney, licensed under the Apache License 2.0. See the top-level [NOTICE](../NOTICE) file for the attribution carried forward under Apache-2.0 Section 4(d).

Implementation entry points:

- CLI flags: `src/cli/turbo_args.rs`
- cache modes: `src/lib/mlxcel-core/src/cache.rs`
- TurboQuant helpers: `src/lib/mlxcel-core/src/cache/turbo/`
- quality/behavior tests: `tests/turbo_kv_e2e.rs` and
  `src/lib/mlxcel-core/src/cache/*turbo*_tests.rs`

## Available modes

| User-facing mode | Effective cache mode | Notes |
|------------------|----------------------|-------|
| `fp16` | `KVCacheMode::Fp16` | Default and baseline. |
| `int8` | `KVCacheMode::Int8` | Per-token INT8 absmax quantization. |
| `fp16+turbo4` / `turbo4-asym` | `KVCacheMode::Turbo4Asym` | FP16 K, 4-bit TurboQuant V. Safest TurboQuant starting point. |
| `turbo4` / `turbo4-sym` | `KVCacheMode::Turbo4` | 4-bit TurboQuant K and V. K-side quantization is quality-sensitive; use only on validated/allowlisted families. |
| `turbo4-delegated` | `KVCacheMode::Turbo4Delegated` | Hot/cold split on the V side with packed cold storage. Validate speed on target hardware. |
| `fp16+turbo3` / `turbo3-asym` / `turbo3` | `KVCacheMode::Turbo3Asym` | FP16 K, 3-bit TurboQuant V. More aggressive than Turbo4 V. |

Symmetric Turbo3 is not exposed.

## Measured speed and the memory trade-off

Quantized KV cache is a memory-footprint optimization, not a decode-speed one.
On the post-#369 M1 Ultra sweep (`benchmarks/turbo_kv/`, 2026-06-19) every
quantized mode decodes slower than fp16. The dequant work (Walsh-Hadamard
inverse rotation plus a codebook lookup for the Turbo modes, a scale-and-add for
int8) runs per token and, at the context lengths tested, costs more than the
memory bandwidth it saves. On dense models the gap is wider because FFN
dominates decode and the V read that compression shrinks is a small fraction of
the work.

Decode is reported as a fraction of the same model's fp16 throughput at that
context; prefill at 8K. Two representative checkpoints below; the full
four-model sweep (adding qwen2.5-7b and the 142B dots.llm1 MoE) is in
`benchmarks/turbo_kv/`.

| Mode | dense decode 4K / 16K | dense prefill 8K | MoE decode 4K / 16K | MoE prefill 8K | KV compression |
|------|-----------------------|------------------|---------------------|----------------|----------------|
| `fp16` | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 1.00 | 1x (baseline) |
| `int8` | 0.61 / 0.41 | 1.29 | 0.48 / 0.50 | 1.14 | ~2x |
| `turbo4-delegated` | 0.63 / 0.45 | 1.27 | 0.43 / 0.34 | 1.13 | ~4x V |
| `turbo4-asym` | 0.38 / 0.37 | 0.92 | 0.24 / 0.26 | 0.80 | ~3.8x V |
| `turbo4` (sym) | 0.22 / 0.20 | 0.73 | 0.17 / 0.17 | 0.62 | ~4x K+V |
| `turbo3-asym` | 0.04 / 0.02 | 0.91 | 0.05 / 0.03 | 0.77 | ~5x V |

Dense = `qwen3-8b-4bit`, MoE = `qwen3-30b-a3b-4bit`, M1 Ultra.

What the numbers say:

- **`int8` and `turbo4-delegated` are the fast picks.** int8 (2x) is robust on
  both dense and MoE; turbo4-delegated reaches ~4x V compression at a similar
  decode speed by keeping recent V in fp16 and only the cold tail in 4-bit. Both
  also speed up prefill (less KV written), 1.1-1.3x.
- **`turbo4-asym` is the memory-and-exactness pick, not a speed pick.** K stays
  fp16 and the #369 dequant-SDPA path is parity-exact with the fp16 reference,
  but decode is ~0.2-0.4x. Reach for it when you need ~4x V compression with an
  untouched K and accept the decode cost. If you want speed at the same fp16-K +
  4-bit-V trade, use `turbo4-delegated` (~0.7x) instead. #370 tried fusing the V
  dequant into the attention kernel to close the gap and measured it a 3-7x
  regression, not a win; see the 2026-06-21 addendum in ADR 0002.
- **Symmetric `turbo4` maximizes compression and is the slowest;** use only on
  an allowlisted family (see below).
- **`turbo3-asym` is near-unusable** (0.02-0.07x, about 0.4 tok/s at 32K). It
  exists for memory-extremis only and is not a recommended mode.

The slowdown is consistent with the upstream TurboQuant+ analysis: its headline
`+22.8%` decode-at-32K figure is measured against turbo3's own full-dequant
path, not against fp16, and even with that win turbo3 stays at 0.93x of int8.
The compression is a 4.6x memory trade; the decode cost is inherent to
dequantizing a rotated, codebook-quantized cache every step. The upstream
sparse-V skip that produces the +22.8% does not carry over to mlxcel, because
mlxcel's Turbo decode is a split dequant plus native SDPA rather than a fused
flash-attention; [ADR 0002](adr/0002-turbo-kv-split-dequant-vs-fused.md) records
the measured A/B. Its 2026-06-21 addendum closes #370: routing asym through
mlxcel's fused kernel regressed decode 3-7x, because the hand-written fused
kernel is slower than native SDPA over a materialized V, so the split dequant
plus native SDPA stays the fast arrangement.

### On CUDA (GB10)

The same trade holds on CUDA, and the numbers were measured on GB10 (sm_121) with `llama-3.1-8b-4bit` (2026-07-10, `benchmarks/cuda_gb10_issue635_kvquant_2026-07-10.csv`; full matrix in `benchmark_results/cuda-kv-quant-modes-gb10-2026-07-10.md`). All six modes run without a Metal-only abort: every custom Turbo/Sparse-V kernel is gated to macOS and falls back to a plain-MLX graph path on CUDA.

Int8 decode vs fp16 is 0.74x at 2K, 0.53x at 8K, and 0.31x at 32K, so it is slower at every length and the gap widens with context, because the path dequantizes the whole INT8 window to fp16 each step and then runs the standard fp16 SDPA (no fused INT8-KV kernel cuts the read bandwidth). What Int8 buys is memory: at 32K the MLX peak drops from 10.37 GB to 8.18 GB, matching a halved fp16 KV cache. Recommendation for CUDA: keep fp16 (default) for speed, and reach for `int8` only when an fp16 KV cache will not fit a long context or you need to pack more concurrent sequences into unified memory, accepting the decode-rate cost. The Turbo modes add V-quantization quality loss with no CUDA speed benefit today, so they stay experimental on CUDA. A fused INT8-KV / paged-attention kernel (#634) is the prerequisite for turning the smaller footprint into a decode win.

## CLI and server flags

The same TurboQuant flag group is flattened into `mlxcel generate`,
`mlxcel serve`, and `mlxcel-server`:

```bash
# Legacy shorthand.
mlxcel generate -m models/<checkpoint> \
    --kv-cache-mode fp16+turbo4 \
    -p "Hello" -n 100

# llama-server-style split flags.
mlxcel-server -m models/<checkpoint> \
    --cache-type-k fp16 \
    --cache-type-v turbo4 \
    --port 8080
```

Supported split-flag combinations:

| `--cache-type-k` | `--cache-type-v` | Result |
|------------------|------------------|--------|
| `fp16` | `fp16` | `fp16` |
| `int8` | `int8` | `int8` |
| `fp16` | `turbo4` or `turbo4-asym` | `fp16+turbo4` |
| `turbo4` | `turbo4` | `turbo4` |
| `fp16` | `turbo4-delegated` | `turbo4-delegated` |
| `fp16` | `turbo3` or `turbo3-asym` | `fp16+turbo3` |

If split flags and `--kv-cache-mode` are both supplied, split flags take
precedence and a warning is logged.

Environment variables:

| Variable | Purpose |
|----------|---------|
| `LLAMA_ARG_CACHE_TYPE_K` | Env fallback for `--cache-type-k`. |
| `LLAMA_ARG_CACHE_TYPE_V` | Env fallback for `--cache-type-v`. |
| `MLXCEL_KV_BOUNDARY_V_LAYERS` | Boundary-V layer count when the CLI flag is not set. |
| `MLXCEL_SPARSE_V_THRESHOLD` | Sparse-V threshold; `0` disables sparse-V behavior. |
| `MLXCEL_SPARSE_V_KERNEL` | Set to a falsy value to disable custom sparse/dequant Metal kernels where they are used. |
| `MLXCEL_TURBO4_ASYM_DEQUANT_SDPA` | Falsy value disables the default dequant-first SDPA path for `Turbo4Asym`, falling back to the sparse-V approximation. |
| `MLXCEL_TURBO4_DELEGATED_DEQUANT_SDPA` | Falsy value disables the default dequant-first SDPA path for delegated mode. |
| `MLXCEL_TURBO4_DELEGATED_FUSED` | Truthy value opts into an older fused delegated-kernel route for comparison. |
| `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH` | Truthy value keeps an FP16 V working set for delegated-mode speed experiments. |
| `MLXCEL_TURBO4_DELEGATED_FP16_SIDECARS` | Sidecar policy for the FP16 fast path. |

Most users should not need the experimental environment variables. Prefer
published CLI flags unless you are benchmarking implementation variants. For
the broader runtime and diagnostic environment-variable reference, see
[Environment variables](environment-variables.md).

## Boundary-V layer protection

`--turbo-boundary-v N` keeps the first `N` and last `N` transformer layers' V
cache at FP16 when a Turbo mode is active. This is intended to reduce quality
loss from aggressive V quantization.

```bash
mlxcel generate -m models/<checkpoint> \
    --kv-cache-mode fp16+turbo4 \
    --turbo-boundary-v 2 \
    -p "Hello" -n 100
```

`0` disables the policy. Values larger than half the layer count are clamped by
the runtime. The flag is inert for `fp16` and `int8` cache modes.

## Symmetric Turbo4 allowlist

K-side quantization can strongly affect softmax quality. The code therefore has
an allowlist helper in `src/lib/mlxcel-core/src/cache/turbo/allowlist.rs`.
As of v0.0.27, the hard-coded allowlisted model-type prefixes are:

- `qwen3_5`
- `qwen3_5_moe`
- `qwen3_next`

Do not assume `turbo4` is safe for other dense 4-bit checkpoints. The safer
starting point is `fp16+turbo4`, which keeps K in FP16.

Note: the allowlist helper exists in `mlxcel-core`, but callers still need to
consult it before constructing a symmetric Turbo4 cache. If you are adding a new
entry or a new caller, include a quality gate in the same change.

## WHT head-dimension constraint

TurboQuant uses a Walsh-Hadamard transform. The implementation expects a
power-of-two head dimension for the production path. Models with unsupported
head dimensions must either reject TurboQuant for that cache path or use a
family-specific fallback; do not silently pad without a quality test.

## Paged cache and server batching

The paged decode layout accepts TurboQuant modes through
`PagedKvLayout::uniform_with_mode`. Server dispatch routes Turbo modes to the
paged layout when `--decode-storage-backend paged` is selected.

### Unified paged KV cache

The continuous-batching server keeps both the cross-request prompt-prefix cache
and per-sequence decode state in one refcounted, copy-on-write block pool
(epic #116). The paged backend is the default for batch-capable pool-backed
families (`--decode-storage-backend auto` resolves to paged when batching);
`dense` forces the legacy per-sequence caches.

Two requests that share a prompt prefix store that prefix's KV blocks once.
Adoption is non-consuming clone-and-pin: a borrower clones pinned references to
the matched prefix blocks, so the stored entry survives for concurrent siblings
and for deeper future matches, and any number of in-flight requests can share
one stored prefix simultaneously. Partial matches adopt too: with Automatic
Prefix Caching (APC, on by default) the match is verified per 16-token hash
block, floored to the 32-token pool block boundary, and the borrower
re-prefills only its divergent suffix on fresh blocks (full shared blocks are
never mutated; a shared partial tail forks copy-on-write). Cache entries are
accounted at their REAL pool bytes, so `--prompt-cache-capacity-bytes`
(default 2 GiB) genuinely bounds retention and the LRU eviction actually
triggers.

Paged adopt and donate are supported for the pool-backed Fp16 families
(the dense-natural backends such as qwen3 and llama3); model-owned-state families
and recurrent or hybrid SSM models keep dense or model-owned caches and stay out
of the pool.

### Exact-prefix snapshots for recurrent state

Hybrid-SSM and linear-attention families remain excluded from block sharing:
their recurrent hidden state cannot be reconstructed from a radix/APC token
prefix, and it cannot be truncated to an arbitrary earlier token. For those
families, `mlxcel-server` has an orthogonal exact-prefix snapshot bucket. Models
that implement `supports_snapshot_reuse()` can copy their full model-owned
state at turn end and restore it into a fresh sequence when the next request's
tokens begin with that exact stored prefix under the same session key. As of
v0.2.1, the supported snapshot families are Mamba, Mamba2, Jamba, Nemotron-H,
Qwen 3.5 / 3.6 text, MoE, and VLM wrappers, and Gemma 4 text, VLM, and Unified
wrappers.

The snapshot bucket has its own byte cap, entry cap, TTL, LRU counters, and
hit/miss metrics. `GET /v1/cache/stats` reports `snapshot_*` fields, while
`/metrics` exposes `mlxcel_prompt_cache_snapshot_hits_total`,
`mlxcel_prompt_cache_snapshot_misses_total`,
`mlxcel_prompt_cache_snapshot_tokens_reused_total`, and labeled snapshot
evictions. This path is deliberately whole-prefix only: it does not participate
in APC block matching, does not share SSM state across sessions, and does not
modify the block-sharing carve-out for hybrid SSM families.

The block pool is bounded with `--kv-cache-budget <bytes|auto|none>` (env
`MLXCEL_KV_CACHE_BUDGET`); the default is `auto`, which derives the cap from
the memory estimate so the batched-decode default (#628) cannot run
concurrent full-context sequences into an OOM abort. `none` (or `0`) restores
the previous unbounded pool. Under a budget the scheduler evicts cold cached
prefixes, then preempts, before rejecting a request. `GET /v1/cache/stats`
reports paged block usage (block size, allocated, live, free, bytes
reserved/in use, and the budget).

### Measured payoff

Numbers below are from an M1 Ultra with `models/qwen3-0.6b-4bit` (28 layers, 8 KV
heads, head dim 128, runtime KV dtype bf16, so about 112 KiB of KV per token).

**Memory saved per shared prefix.** A shared prefix of `P` tokens across `N`
concurrent requests occupies one pool copy instead of `N`. KV bytes per token are
`2 (K and V) * n_kv_heads * head_dim * num_layers * dtype_bytes`. A shared
1024-token system prompt is about 112 MiB; with 8 concurrent requests the pool
stores it once instead of eight times, saving roughly `7 * 112 MiB ≈ 784 MiB`.

**Prefill tokens avoided.** Only the first of `N` requests sharing a `P`-token
prefix prefills it; the other `N - 1` adopt the cached blocks and skip
`(N - 1) * P` prefill-token forward passes. `tests/paged_prefix_share_parity.rs`
confirms an adopting request decodes byte-identically to a cold run while
skipping the shared prefill.

**End-to-end server footprint.** Whole-process physical footprint
(`/usr/bin/footprint`) of `mlxcel serve` under fixed HTTP workloads
(`scripts/bench_memory_footprint.py`, `models/llama-3.2-1b-4bit`, M1 Ultra,
defaults, peak during the scenario phase):

| scenario | v0.1.4 | current default | shared tokens |
|---|---|---|---|
| idle (weights only) | 957 MiB | 983 MiB | - |
| 8 concurrent requests, shared ~3.7k-token system prompt | 1653 MiB | 1627 MiB | 29184 (all 8 adopt) |
| same prefix, 8 sequential requests | 1650 MiB | 1413 MiB | 25536 |
| one conversation, 8 turns | 2104 MiB | 1301 MiB | 10976 |
| 32 distinct prompts (churn) | 1558 MiB | 2276 MiB | 1984 |

v0.1.4 never engaged the prompt cache from the HTTP path (zero reuse), so its
footprint is the no-cache floor. The current default matches or beats it on
every sharing scenario while also skipping the shared prefill work (the
8-way burst completes its request phase 3.4x faster). The churn number is
higher because donated entries are now retained for future reuse; that
retention is real memory governed by `--prompt-cache-capacity-bytes` and is
released by LRU eviction at the cap.

**Decode throughput.** Paged decode is byte-identical to the dense backend
(`tests/paged_scheduler_parity.rs`, RMS 0). The live batched path uses the native
block-table decode kernel (`DecodeBatchContext::use_native_paged_kernel`, set by
the scheduler); a gather-then-SDPA path is kept as a correctness reference. At
batch 4 the native kernel runs at 276 tok/s for a 512-token prompt and 84 tok/s
for a 4096-token prompt, versus 146 and 7.7 tok/s for the gather reference (1.9x
and 10.9x). The gather reference degrades sharply with context because it
re-materializes the visible window every step, which is why the live path uses
the native kernel. The separate fused split-K kernel (Metal, and since #634
also CUDA via `mx.fast.cuda_kernel`)
(`MLXCEL_PAGED_ATTENTION_NATIVE`, feeding `paged_decode_attention_pooled`) is a
different code path from the block-table kernel above. Since #331 it is no
longer a plain on/off switch: an adaptive selector dispatches it on Metal only
inside the batch>=4 / ctx<=4096 / single-slab island ADR 0001 measured it
winning; on CUDA (#634) it dispatches on any single-slab layer regardless of
batch or context, since the Metal ceilings do not apply there. Gather runs
everywhere else, and the env var still force-pins either arm for A/B testing.
The chunked slab storage narrows that island further, since
the kernel declines (falling back to gather) once a layer has grown past one
slab. #710 retired this pooled entry point to a library-only API: neither this
kernel nor its selector is on the `mlxcel serve` decode path (which stays on the
block-table kernel described above), and `MLXCEL_PAGED_ATTENTION_NATIVE` is a
control for external mlxcel-core consumers and the kernel bench, not a server
knob. See ADR 0001's #710 decision record,
[ADR 0001](adr/0001-paged-attention-gather-vs-fused-kernel.md).

Pool growth appends fixed-size slabs instead of reallocating one big tensor
per layer, so extending the pool never copies existing KV and never strands a
ladder of old buffer sizes in the allocator cache. Eight concurrent requests
with distinct ~4k-token prompts (qwen3-0.6b, the nothing-shareable stress)
peak at 3739 MiB versus 5013 MiB before the change, 1.13x of the dense
backend on the same workload.

## Recommended starting points

| Workload | Recommendation |
|----------|----------------|
| General serving | Start with `fp16`. Quantized KV is for memory pressure, not speed. |
| Lower KV memory, fastest quantized option | Use `int8` (~2x) or `turbo4-delegated` (~4x V); both also speed up prefill. |
| Maximum V compression with exact K | `fp16+turbo4` (turbo4-asym), accepting ~0.2-0.4x decode; enable boundary-V for quality. |
| Considering symmetric `turbo4` | Use only on an allowlisted/validated family; it is the slowest mode. |
| Memory-extremis only | `fp16+turbo3` decodes near-unusably; confirm the footprint is worth the speed cost. |

## Advisor recommendations (`--recommend-quant`)

`mlxcel generate --recommend-quant` prints an advisory KV-cache-mode section
alongside the quantization advice. It suggests one of `fp16`, `int8`,
`turbo4-delegated`, `fp16+turbo4`, or `turbo4` per model family and context
range (it withholds `fp16+turbo3`, whose decode is near-unusable), so you have a
benchmark starting point instead of guessing. The suggestions follow the
measured trade-off above: int8 and turbo4-delegated as the fast picks,
fp16+turbo4 as the exact-K memory pick, symmetric turbo4 for allowlisted
families only.

The suggestions are advisory and opt-in only. The default inference path is
unchanged: with no flags the runtime still uses `fp16`. To use a suggested mode
you must pass `--kv-cache-mode` (or `--cache-type-k` / `--cache-type-v`)
yourself. The advisor reads only `config.json`; it never loads weights and
never changes the quantized-weight dtype. KV-cache modes quantize only the K/V
cache tensors and dequantize back to FP16 for attention, so a recommendation
cannot reintroduce the bf16-to-f16 quantized-weight promotion that the project
forbids (the `#289` regression).

How it keys the suggestion:

- **Model family** comes from the KV-architecture classifier
  (`src/execution/kv_arch.rs`): standard attention, sliding-window, MLA
  (DeepSeek), hybrid attention plus recurrent, and pure SSM. The raw
  `model_type` is also checked against the symmetric-Turbo4 allowlist
  (`src/lib/mlxcel-core/src/cache/turbo/allowlist.rs`).
- **Context range** is bucketed as short (`<=4K` tokens, interactive /
  single-request), medium (`4K-32K`), and long (`>32K`, long-context serving).
  Long context and memory-constrained serving are prioritized over raw
  short-decode tok/s, because that is where KV-cache pressure dominates.

The conservative rules the advisor applies:

| Family / context | Suggested | Also benchmark | Why |
|------------------|-----------|----------------|-----|
| Any family, short context | `fp16` | - | KV footprint is small; keep the baseline. |
| Standard / sliding-window / hybrid, medium | `int8` | `turbo4-delegated` | int8 is the fastest quantized mode (~2x); turbo4-delegated for ~4x V at a similar speed. |
| Standard / sliding-window / hybrid, long, allowlisted | `turbo4` | `turbo4-delegated` | Family passed the PPL gate; turbo4 is max compression, delegated is the faster fallback. |
| Standard / sliding-window / hybrid, long, not allowlisted | `turbo4-delegated` | `fp16+turbo4` | Fastest ~4x V compression; fp16+turbo4 is the exact-K alternative. Symmetric `turbo4` withheld off the allowlist. |
| MLA (DeepSeek), medium or long | `int8` | - | The latent dimension is not a power of two, so the Turbo Walsh-Hadamard V path does not apply; per-token INT8 has no head-dim constraint. |
| Pure SSM (Mamba/Mamba2) | `fp16` | - | No context-proportional KV cache, so Turbo modes save almost nothing. |
| Non-power-of-two head dim (e.g. Phi-2 at head_dim 80) | `int8` (medium/long), `fp16` (short) | - | The Turbo Walsh-Hadamard transform requires a power-of-two head dimension; the advisor downgrades any Turbo suggestion to `int8` or `fp16` for these families, matching the MLA treatment. |

The head-dimension check reads `head_dim` or `head_size` directly from `config.json` when present. When neither explicit field exists, it divides the hidden size by the attention head count, checking the alternate field names used by each naming convention: `hidden_size` / `d_model` / `dim` / `model_dim` for hidden size, and `num_attention_heads` / `num_heads` / `n_heads` / `n_head` for head count. This mirrors the field-name coverage in the KV-architecture classifier so the two paths agree on the derived head dimension. When none of the required fields are present, the advisor conservatively assumes Turbo is applicable and leaves the suggestion unchanged.

Symmetric `turbo4` is only ever suggested for families on the allowlist; off
the allowlist the advisor leads with `turbo4-delegated` at long context and
`int8` at medium, matching the measured trade-off above. `fp16+turbo3` is never
suggested. Treat every suggestion as a hypothesis to validate per family with
the checklist below before adopting it in production.

## Validation checklist

Before recommending a TurboQuant mode for a model family:

1. Run a short smoke generation to confirm the mode loads and decodes.
2. Compare output quality against FP16 on a representative prompt set.
3. Run perplexity/NIAH or another task-appropriate quality gate.
4. Measure prefill and decode throughput separately.
5. Test the exact server path if the mode will be used with continuous batching
   or paged decode.
6. Record hardware, MLX commit/version, model checkpoint, prompt/decode shape,
   and all cache flags.

Ignored tests in `tests/turbo_kv_e2e.rs` are the right place for hardware/model
quality gates; they are not part of the default `cargo test` run because they
require local model checkouts.

## Known limitations

- TurboQuant is not uniformly validated across all text and VLM families.
- VLM long-context and multi-image prompts need separate validation.
- Older Apple Silicon generations and non-Hopper/non-Blackwell CUDA paths may
  have different bottlenecks from the developer benchmark machines.
- Experimental environment-variable paths are useful for A/B testing but should
  not appear in user-facing recommendations without fresh benchmark data.
