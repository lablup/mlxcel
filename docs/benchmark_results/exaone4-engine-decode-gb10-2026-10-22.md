# ExaOne 4 CLI decode on the engine client, GB10 (2026-10-22)

Issue #2494: `mlxcel generate` on exaone-4.0-1.2b-4bit lost decode throughput when #2225 moved the CLI from `CxxGenerator` onto `DirectEngine`.

## Environment

- Host: NVIDIA GB10 (sm_121), kernel `7.0.0-1019-nvidia`, driver `580.178.04`
- Build: `MLX_CUDA_ARCHITECTURES=121 cargo build --profile test-fast --features cuda --bin mlxcel`
- Arms (binaries copied out of `target/test-fast/`):
  - `ref`: d379d2f8, the last `CxxGenerator` build (parent of 30ceea22, the bisected first bad commit)
  - `main`: 8116e266 (after #2538's custom-kernel disk cache)
  - `fix`: 8116e266 plus this change
  - `fix@null`: the `fix` binary again, as the null arm
- Each arm uses its own `MLX_PTX_CACHE_DIR`, warmed before measuring; no warm `main`/`fix` run printed `Compiling CUDA kernels`. `ref` predates #2538 and recompiles granite's fused MoE kernels in every process (visible in its granite numbers below).
- Command: `mlxcel generate -m models/mlx/<model> -p "<fixed prompt>" -n <N> --temp 0`, rate from the `[Generated N tokens in Xs = Y tok/s]` line (one-token warmup, then the whole call timed, prefill included).
- Protocol: no CI job, compiler or other GPU process running (checked before each block), under `gpu-lock`; one warm-up per arm, then rounds with the arm order rotated each round. Paired deltas are per round.

## Mechanism

The gap is a fixed per-call cost, not a per-token one. A fit of `time = fixed + n * per_token` over `-n 2, 32, 128, 384` (3 runs each, same session):

| arm | fixed | per token | median time at n=2 / 32 / 128 / 384 |
|---|---|---|---|
| ref | 70.5 ms | 4.770 ms | 85 / 226 / 668 / 1905 ms |
| main | 302.0 ms | 4.851 ms | 318 / 457 / 898 / 2175 ms |

ExaOne 4 cannot batch (`supports_batching() == false`) and keeps its KV caches in a `RefCell` inside `ExaOne4Wrapper`, reset only by `make_caches()`. Its default sequence layout is model-owned, so `CachePool::allocate_with_layout` creates a placeholder entry without calling `make_caches()`, and the wrapper has no `prepare_sequence_state`. `DirectEngine` therefore never reset the caches between sequences. `mlxcel generate` runs a one-token warmup and then the timed run on the same client, so the timed run prefilled the prompt on top of the warmup's 26 cached positions:

- Its output was wrong: the model saw the prompt twice. With `--show-reasoning -n 48`, `main`'s text differs from `ref`'s from the first token; `fix`'s text is identical to `ref`'s.
- Its prefill took a different kernel path. A gdb backtrace on every `cuLibraryLoadData` shows the warmup prefill loading the fresh-sequence kernels (cuDNN SDPA, RoPE, RMSNorm) and the timed prefill loading cuBLAS GEMM kernels (`CublasGemm::run_batched`, ten `execute_matmul` loads) and `Softmax` for the attention against a non-empty cache. nsys shows nine of those library loads at 9 to 71 ms each, about 290 ms in all (under nsys), inside the timed window. Instrumented, the timed prefill's eval took 230 to 270 ms on `main`; after the fix the whole fixed cost of a call, prefill included, is 57 ms (fit below).

The server never had this problem: `BatchScheduler` calls `make_caches()` at prefill time for every model that cannot batch. The retired CLI loop reset through `reset_with_model` between the warmup and the timed run. The fix makes `DirectEngine::open_sequence` follow the scheduler's rule. GPT-OSS, Ministral 3, Step 3.5, Mellum and Laguna hold their caches the same way, so the same reset applies to them; GPT-OSS (below) and Laguna (laguna-xs-2.1-nvfp4, greedy `-n 32` text with a warmup equals the `--profile` text without one, in the PR review) were checked on real checkpoints. `--prompt-lookup` refuses every model that cannot batch, so its output is unaffected.

## exaone-4.0-1.2b-4bit, `-n 384`, 7 rounds

| arm | median tok/s | range | paired vs ref |
|---|---|---|---|
| ref | 202.73 | 199.46..204.36 | |
| main | 179.28 | 174.94..181.42 | -12.27% (-12.91..-10.26) |
| fix | 202.04 | 197.19..203.36 | -0.72% (-1.14..+0.82) |
| fix@null | 201.94 | 199.66..204.56 | +0.10% (-2.19..+0.77) |

Null floor (`fix@null` vs `fix`, same binary): +0.11% median, -1.48..+1.46% range. The remaining `fix` vs `ref` difference is inside it.

Fit after the fix (3 runs each, same block as a fresh `ref` fit): `fix` fixed 56.8 ms, 4.970 ms per token; `ref` fixed 69.7 ms, 4.888 ms per token. The fixed cost is gone. The per-token slope is 1.7% above `ref` in this small fit; the interleaved `-n 384` comparison above does not resolve a difference.

Per-run tok/s: ref 204.23 203.77 204.36 200.39 202.73 202.32 199.46; main 177.86 181.42 179.28 179.82 177.47 180.82 174.94; fix 202.75 202.04 203.15 202.03 203.36 200.59 197.19; fix@null 199.75 202.27 204.56 201.94 201.70 203.52 199.66.

## Controls

llama-3.2-1b-4bit, `-n 384`, 5 rounds (batching model, path unchanged):

| arm | median | range |
|---|---|---|
| ref | 251.33 | 250.76..252.89 |
| main | 254.95 | 251.40..255.89 |
| fix | 254.37 | 251.58..254.76 |
| fix@null | 254.45 | 228.21..255.48 |

`main` vs `fix` paired: +0.07% (-0.10..+0.60). The 228.21 run is the bimodal GB10 low mode in the null arm.

granite-4.0-h-tiny-4bit, `-n 256` (cannot batch, so it now also gets `make_caches()` per sequence; its sequence-keyed state was already correct), 7 rounds:

| arm | median | range |
|---|---|---|
| main | 99.07 | 97.58..100.97 |
| fix | 97.38 | 95.54..100.35 |
| fix@null | 99.37 | 96.13..101.01 |

`main` vs `fix` paired +2.14% (-2.43..+3.69); null `fix@null` vs `fix` +1.55% (-0.63..+4.01) on the same binary, so the difference is within this method's floor (`main` vs `fix@null` medians differ by 0.3%). An earlier 5-round block gave the same picture (main 101.13, fix 97.96, fix@null 100.60; `ref` 74.22, paying its per-process fused-MoE compile). Generated text (`-n 48`) is identical between `main` and `fix`.

## Other families sharing the pattern

`--show-reasoning -n 48`, greedy: on gpt-oss-20b-mxfp4, `main` opened with a different reply ("The user wants..."), consistent with decoding on the warmup's state; `fix` opens like `ref` ("We need to write a detailed essay...") and differs from it only in reasoning-marker display, which has changed since d379d2f8, and in a near-tie late in the 48 tokens. ministral-3b-4bit (a `mistral3` VLM checkpoint, which does not use the `Ministral3` text wrapper) prints the same text on `main` and `fix`. No throughput was measured for these.
