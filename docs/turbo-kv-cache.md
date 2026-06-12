# TurboQuant KV cache

TurboQuant modes reduce KV-cache memory by quantizing K and/or V cache tensors.
The implementation is experimental in the sense that quality and speed vary by
model family, cache mode, hardware, and server path. Use the default FP16 cache
unless you have measured the target model and workload.

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
(gemma3, llama4, qwen3.5) and recurrent or hybrid SSM models keep dense or
model-owned caches and stay out of the pool.

The block pool can be bounded with `--kv-cache-budget <bytes|auto>` (env
`MLXCEL_KV_CACHE_BUDGET`); the default is unbounded. Under a budget the scheduler
evicts cold cached prefixes, then preempts, before rejecting a request.
`GET /v1/cache/stats` reports paged block usage (block size, allocated, live,
free, bytes reserved/in use, and the budget).

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
the native kernel. The separate fused split-K Metal kernel
(`MLXCEL_PAGED_ATTENTION_NATIVE`) is opt-in and stays off by default; see
[ADR 0001](adr/0001-paged-attention-gather-vs-fused-kernel.md).

## Recommended starting points

| Workload | Recommendation |
|----------|----------------|
| General serving | Start with `fp16`. |
| Need lower KV memory with low risk | Test `fp16+turbo4` with boundary-V enabled. |
| Need more aggressive V compression | Test `fp16+turbo3`; compare quality against FP16. |
| Considering symmetric `turbo4` | Use only on an allowlisted/validated family. |
| Long-context decode speed experiments | Benchmark `turbo4-delegated` against both `fp16` and `fp16+turbo4`. |

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
