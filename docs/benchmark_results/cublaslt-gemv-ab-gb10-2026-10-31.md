# cuBLASLt version and gemv-vs-cuBLASLt decode A/B (GB10): plan and status

**Status: not measured yet.** This record holds the plan, the harness and what was learned while setting it up, so the campaign can run on another host (spark-102 or any GB10 that is not a CI runner). It answers two questions:

1. Does mlxcel get faster on CUDA if it runs the cuBLASLt that the `mlx` 0.32.4 wheel ships (13.8.1.7) instead of the one it links today (the CUDA 13.0 toolkit's 13.1.1.3)?
2. Does B=1 decode of a bf16 model get faster if MLX's own gemv kernel is bypassed and those matmuls go to cuBLASLt?

Neither question has an answer in this record. The one data point below is a single sanity run and is not evidence.

## Why these two questions

Issue #2732 traced a small numerical difference between mlxcel and the `mlx` wheel oracle in the image-generation conditioners to the cuBLASLt version, not to mlxcel code (`docs/image-generation.md`, "GEMM differences against the oracle"). With the wheel's libraries preloaded into mlxcel (`LD_PRELOAD`) the two outputs were bit-identical. The difference is the kernel each version picks for small-M bf16 GEMMs: under 13.1 they run `cutlass_80_wmma` kernels, under 13.8 they run `nvjet_sm121_*` kernels, which are tuned for this GPU. That raised the performance question: if 13.8 picks sm_121-specific kernels, it may be faster where cuBLASLt is on the hot path.

Where cuBLASLt is on the hot path matters for what to expect:

| Workload | Dense matmul path on CUDA | Expected effect of a newer cuBLASLt |
|---|---|---|
| B=1 decode, bf16/f16 model | MLX gemv (`cu::can_use_gemv`: `M == 1` with transposed `B`, or `N == 1`, no bias) | little; cuBLASLt is barely called |
| Prefill (large M) | cuBLASLt | possible |
| Batched decode, M = batch (2 to 7 rows on CUDA) | cuBLASLt | likeliest place for a difference; small M is exactly where the kernel choice changed |
| Any quantized model (4bit, 8bit, nvfp4) | MLX qmv/qmm kernels | none expected (control) |

Question 2 comes from the first row. B=1 decode of `llama-3.2-1b-instruct-bf16` runs at about 88 tok/s, about 218 GB/s of weight traffic. That is below the GB10's nominal memory bandwidth, and MLX's gemv is a general kernel. If cuBLASLt's sm_121 kernels read weights closer to peak, routing M=1 to them would help every unquantized decode. The `MLXCEL_CUDA_DISABLE_GEMV=1` switch added with this record does that routing (see `docs/environment-variables.md`).

## Environment where setup ran

| Item | Value |
|------|-------|
| **Host** | spark-101, NVIDIA GB10 (sm_121), driver 580.178.04, Linux 7.0.0-1019-nvidia |
| **Linked cuBLASLt** | 13.1.1.3 from the CUDA 13.0 toolkit, `/usr/local/cuda/targets/sbsa-linux/lib` |
| **Alternate cuBLASLt** | 13.8.1.7 from the pip `nvidia-cublas` wheel the `mlx` 0.32.4 oracle venv installs, `<venv>/lib/python3.12/site-packages/nvidia/cu13/lib` (`libcublasLt.so.13`, `libcublas.so.13`, depending only on the system libc) |
| **apt candidates** (not installed) | `libcublas-13-2` 13.4.1.3, `libcublas-13-3` 13.6.0.2, `libcublas-13-4` 13.8.0.4 |

Record the same rows on the host that runs the campaign; `campaign.sh` writes most of them to `campaign.log`. spark-102 has its own toolkit install, so its linked version must be checked, not assumed.

## Harness

Everything lives in `data/cublaslt-gemv-ab-gb10-2026-10-31/harness/`:

| File | Role |
|---|---|
| `bench_wrap.sh` | `--bin` for `scripts/engine_bench_rounds.py`. Arms pass only flags, so it turns `--cublas-alt` into an `LD_PRELOAD` of `$MLXCEL_AB_CUBLAS_DIR` and `--no-gemv` into `MLXCEL_CUDA_DISABLE_GEMV=1`, then runs `mlxcel-bench-engine` from `$MLXCEL_AB_BIN_DIR` (default `target/release`) |
| `ladder_rounds.py` | Server A/B for batched decode: arms `sys`, `alt` and a null repeat of each, rotated per round; every arm is a fresh `mlxcel-server --parallel 7 --max-batch-prefill 4` followed by `scripts/bench_serving_concurrency.py --concurrency 1,4,7`. Each record names the `libcublasLt` the server actually mapped (from `/proc/<pid>/maps`), so a failed preload is visible |
| `campaign.sh` | Runs the three stages below in order and writes JSONL, text logs and `campaign.log` to `$MLXCEL_AB_OUT` (default `./ab-results`). Uses `gpu-lock` when it is installed |

Both drivers use the interleaved-rounds protocol with a null arm per arm (`--null-every-arm`) and the #1820 host gate (`--hostgate`). A delta inside the null arms' spread is unresolved.

### Stages

| Stage | Models | Arms | Shape |
|---|---|---|---|
| `version` | `llama-3.2-1b-instruct-bf16`, `qwen3.5-9b-bf16`, `qwen3-1.7b-4bit` (control) | `sys`, `alt` | CLI path, prompts 256 and 4096, 128 decode tokens, 5 rounds |
| `gemv` | `llama-3.2-1b-instruct-bf16`, `qwen3.5-9b-bf16`, `llama-3.1-8b-instruct-bf16` | `gemv` (default), `lt-sys` (no gemv, linked cuBLASLt), `lt-alt` (no gemv, alternate cuBLASLt) | CLI path, prompt 256, 256 decode tokens, 5 rounds |
| `ladder` | `llama-3.2-1b-instruct-bf16`, `llama-3.1-8b-instruct-bf16` | `sys`, `alt` | server, concurrency 1/4/7, prompt 512, 128 decode tokens, 3 rounds |

`qwen3.5` is a hybrid model, which the server clamps to `--parallel 1`, so it is left out of the ladder. CUDA batched decode covers at most 7 rows, which is why the ladder stops at 7.

## How to run it

```bash
# 1. Build (CUDA, release). The gemv switch is in the matmul overlay.
MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda --bin mlxcel-bench-engine --bin mlxcel-server

# 2. Get the alternate cuBLASLt. Either reuse an mlx 0.32.4 venv, or:
python3 -m venv /tmp/cublas138 && /tmp/cublas138/bin/pip install nvidia-cublas==13.8.1.7
# the python3.X segment follows the host's Python; `pip show -f nvidia-cublas` prints the location
export MLXCEL_AB_CUBLAS_DIR=/tmp/cublas138/lib/python3.12/site-packages/nvidia/cu13/lib

# 3. Run (hours; every arm waits for a quiet host)
MLXCEL_AB_OUT=$PWD/ab-results docs/benchmark_results/data/cublaslt-gemv-ab-gb10-2026-10-31/harness/campaign.sh
```

`MLXCEL_AB_STAGES="gemv"` runs a single stage. The models must resolve under `./models` (the checkout's `models` symlink).

**Host requirement.** The host gate waits until no CI job, compiler or foreign model process is running. On spark-101 this did not happen for over two hours (it is the `lablup-dgxspark21` Actions runner, and other sessions were building), and the campaign was stopped with zero records. Pick a host that is not a CI runner, or pause the runner for the duration.

## Status

| Item | State |
|---|---|
| `MLXCEL_CUDA_DISABLE_GEMV` switch | in this PR; exercised once (row below), not yet on a measured run |
| `version` stage | not run (stopped by the host gate on spark-101 after 2+ hours, 0 records) |
| `gemv` stage | not run |
| `ladder` stage | not run |
| Sanity run (single process each, not evidence) | `llama-3.2-1b-instruct-bf16`, prompt 256: linked 13.1 prefill 16191 tok/s, decode 88.3 tok/s; preloaded 13.8 prefill 15510 tok/s, decode 89.4 tok/s |
| gemv switch smoke (single process each through `bench_wrap.sh`, `target/test-fast`, not evidence) | `llama-3.2-1b-instruct-bf16`, prompt 256, 64 decode tokens: `gemv` decode 90.9 tok/s; `lt-sys` 70.4 tok/s; `lt-alt` 55.2 tok/s. Greedy `mlxcel generate` output with the switch on matches the default for the first ~55 of 60 tokens and then diverges, as expected from a different GEMV reduction order; both completions are coherent |

The smoke run says the switch reaches the dispatch and the cuBLASLt path is correct at M=1; it also suggests the `gemv` stage will come out in gemv's favor, which the interleaved rounds need to confirm or refute. The sanity run only shows that the preload works end to end; its deltas are well inside the single-run noise this host shows (see the GB10 profiling notes on bimodal decode).

## What to do with the result

- **If `alt` wins on prefill or the ladder** beyond the null spread: the cheapest route is to make release builds and the GB10 hosts use cuBLAS 13.8 (apt `libcublas-13-4` 13.8.0.4 next to the 13.0 toolkit, or bundling the libraries with the release). Check how the release bundle packages cuBLAS before choosing; that is not verified here. Any switch also changes small-M GEMM numerics, so fixtures and logit traces recorded against 13.1 need re-baselining, and the #2732 oracle difference disappears.
- **If `lt-*` beats `gemv`** on B=1 decode: routing M=1 bf16 matmuls to cuBLASLt becomes a real dispatch change (a per-shape or per-arch default rather than an env switch), with its own numerics check against the gemv path.
- **If neither wins:** record the numbers here, keep the switch as a diagnostic or remove it, and close the question.

Refs #2732.
