# ROCm decode baseline: Radeon 8060S (gfx1151), 2026-09-30

First throughput baseline for the experimental ROCm backend (issue #1810, epic #1801), taken with `scripts/bench_decode.sh` against the mlx-lm Python baseline from `scripts/bench_mlxlm.py`, on the same host, the same day and the same checkpoints. It is the reference later ROCm performance work (#1814) is measured against.

Raw rows:

- mlxcel: `benchmarks/rocm_strixhalo-gfx1151_2026-09-30.csv`
- mlx-lm: `benchmarks/pylm_strixhalo-gfx1151_2026-09-30.csv`

## Environment

| Item | Value |
|------|-------|
| **Hardware** | AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 32 CPU threads |
| **Memory** | 96 GiB VRAM carve-out (`mem_info_vram_total` 103079215104 bytes, also what the probe's `GPU:` line reports), 31 GiB visible to the host (`MemTotal` 32493820 kB), GTT 96 GiB (`amdgpu.gttsize=98304`) |
| **OS** | Debian GNU/Linux 13 (trixie), kernel 6.18.12+deb13-amd64 |
| **ROCm / HIP** | ROCm 10.0.0 (`/opt/rocm/core-10.0/.info/version`, packages 10.0.0-4), HIP runtime 7.15.26333 (`hipconfig --version`), AMD clang 23.0.0git |
| **mlxcel** | 0.7.0, CSV `mlxcel_commit` `2fcdbe5f`. The binaries were built with `cargo build --release --features rocm` from `784e35b7` (main); `2fcdbe5f` adds only the harness scripts, their test and `docs/benchmarks.md` on top, no Rust or C++ source, so the binary is the same |
| **MLX pin** | `81ba1c6a` (`mlx_commit`) |
| **MLX ROCm overlay** | NripeshN/mlx `rocm-support` at `75915908`, retargeted onto the pin, plus the local fixes in `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` (items 1 to 19) (`mlx_rocm_overlay_commit`) |
| **mlx-lm baseline** | mlx-lm 0.31.3 on MLX `0.32.3.dev20260912+a7d4c85a`, a source build of the spike tree described below, Python 3.12 |
| **Toolchain** | Rust 1.97.1 (`rust-toolchain.toml`) |
| **Contention** | none from other GPU tenants or builds during either sweep; see below |

### The mlx-lm build is not byte-identical to mlxcel's

MLX has no ROCm wheel, so the Python baseline runs on a source build: the spike tree from the ROCm investigation (`mlx-xcel`, HEAD `a7d4c85a`). It is the same fork commit (`75915908`) vendored onto the same MLX pin (`81ba1c6a`) as mlxcel's overlay, and carries trial commits equivalent to `LOCAL_FIXES.md` items 1 to 6 and 8 to 11, plus uncommitted edits to `device.h`, `indexing.hip` and `quantized/qmm.hip` that were in the tree when it was built. Of item 7 it has only the `Event::error` storage, not the error surfacing from #1804. It does not carry items 12 to 19: the f16 `gather_qmm` fast path, f32 activations in tiled qmv, the scatter argument width, `SearchSorted`, `Hadamard`, narrow gather indices, FFT and the `get_launch_args` removal. Those fix f16 or f32 activations (these checkpoints run bf16), operations a text decode does not obviously call, or dead code; that reading was not checked by tracing the kernels each run launched. So the two sides run the same fork kernels on the same pin as far as could be checked, which is not a proven same build. A baseline built from mlxcel's own overlay source (`target/.../_deps/mlx-src`) would remove the question; it was not built here.

## Method

Both harnesses at their defaults, which are llama-bench's pp512/tg128:

- a deterministic 512-token synthetic prompt (the shared corpus in `src/bin/bench_decode.rs` and `scripts/bench_mlxlm.py`, tokenized by each model's own tokenizer and truncated to exactly 512), except Mixtral's 513 on the mlx-lm side, a tokenizer round-trip difference within `compare_bench_csv.py`'s prompt tolerance;
- exactly 128 generated tokens, every end-of-generation token suppressed (`--ignore-eos` in mlxcel, a `-inf` logits processor in mlx-lm);
- batch 1, greedy, no speculative decoding, default KV cache, no extra flags or environment variables;
- a discarded 20-token warmup pass in the same process as the measured pass, one measured pass per model, one fresh process per model;
- 30 s cooldown after each model and 60 s after one over 10 GiB. mlxcel had no pre-warm pass (none of its candidate models is on this host); its one-token runtime probe on `Qwen3-0.6B-4bit` ran before the first model.

Commands:

```bash
MODELS_DIR=models/mlx ./scripts/bench_decode.sh all --cooldown 30 --big-cooldown 30

MLXLM_PYTHON=<mlx-xcel venv>/bin/python LD_LIBRARY_PATH=/opt/rocm/lib MODELS_DIR=models/mlx \
    ./scripts/bench_mlxlm.py all --cooldown 30 --big-cooldown 60 --big-threshold-gb 10
```

### Keeping the GPU to itself

This host also runs other GPU jobs (another development unit was testing ROCm kernels on it the same day). A benchmark sharing the GPU, or a UMA memory bus shared with a compiler, is not a measurement. Every measured sweep was therefore wrapped in a guard that:

1. waited until the host had gone 90 consecutive seconds with no process holding the GPU (`/sys/class/kfd/kfd/proc` empty, the list `rocm-smi --showpids` reads) and no compiler or `cargo` process running;
2. ran the sweep while a monitor sampled both once per second;
3. rejected the sweep if any sample in its window showed another GPU process or a compiler.

Five sweeps were rejected and rerun: one per runtime taken before the guard existed (the mlx-lm one overlapped another unit's GPU tests; the mlxcel one saw no other GPU process but compiler activity was not being logged yet), and three guarded attempts that a build or test run overlapped. Their numbers are not used. The published mlxcel sweep ran 02:31:32 to 02:40:28 KST and the mlx-lm sweep 03:21:29 to 03:29:42 KST, and in both windows every sample showed either no GPU process or only the benchmark itself, and no compiler. The sampling interval is one second, so a GPU job shorter than that could in principle have been missed; none was seen in the samples around either window.

## Results

| Model | Format | Size | mlxcel prefill tok/s | mlx-lm prefill tok/s | mlxcel decode tok/s | mlx-lm decode tok/s | Decode, mlxcel / mlx-lm |
|---|---|---|---:|---:|---:|---:|---:|
| `Qwen3-0.6B-4bit` | affine 4-bit, g64, bf16 | 0.35 GB | 4417.27 | 4316.74 | 278.48 | 220.13 | 1.27x |
| `Meta-Llama-3.1-8B-Instruct-4bit` | affine 4-bit, g64, bf16 | 4.53 GB | 1065.78 | 966.84 | 35.41 | 32.81 | 1.08x |
| `Qwen3-30B-A3B-4bit` (128 experts, 8 active) | affine 4-bit, g64, bf16 | 17.19 GB | 275.65 | 280.40 | 61.82 | 58.65 | 1.05x |
| `gpt-oss-20b-MXFP4-Q4` (32 experts, 4 active) | mxfp4 experts, affine 4-bit attention | 11.21 GB | 7.69 | 7.77 | 8.07 | 8.07 | 1.00x |
| `Mixtral-8x7B-Instruct-v0.1-4bit` (8 experts, 2 active) | affine 4-bit, g64, bf16 | 26.27 GB | 25.85 | 26.43 | 7.65 | 9.98 | not stable, see below |

`scripts/compare_bench_csv.py --before benchmarks/pylm_strixhalo-gfx1151_2026-09-30.csv --after benchmarks/rocm_strixhalo-gfx1151_2026-09-30.csv --reference` pairs all five with nothing dropped (median 1.054x over the five, Mixtral included).

Every model fit: the memory guard budgeted against 85% of the 96 GiB device memory, so nothing was skipped, and no run failed.

### What the numbers say

- On the three dense or affine-MoE models the two runtimes are close, and mlxcel is ahead on decode by 5 to 8% on the 8B and the 30B-A3B and by 27% on the 0.6B, where per-token host overhead is the largest share of the step. Prefill is within 11% either way.
- The two MoE checkpoints whose experts go through the generic `gather_qmm` path are slow in both runtimes by the same amount: Mixtral prefills at about 26 tok/s (20 s for 512 tokens) and gpt-oss at under 8 tok/s (66 s). Since mlxcel and mlx-lm agree to within 3%, this is the overlay's kernel, not mlxcel's model code. The fused MoE kernels have no ROCm port (#1814), so these are the numbers that work starts from.
- gpt-oss decode is 8.07 tok/s in both runtimes, against the 3.6 tok/s that #1808 records PR #1818 measuring through the same generic gather kernel.

### Mixtral decode does not repeat

Mixtral's decode rate did not repeat. After the two sweeps, single-model runs were alternated between the runtimes under the same guard until other units' GPU work kept the guard waiting for over half an hour; one mlx-lm repeat that a compiler overlapped is excluded:

| Runtime | Clean decode readings, tok/s |
|---|---|
| mlxcel | 7.65 (sweep), 8.22, 11.16 |
| mlx-lm | 9.98 (sweep), 8.11 |

The readings spread from about 8 to about 11 tok/s in both runtimes, while prefill for the same runs stayed at 25.8 to 26.5 tok/s every time. Five readings are too few to call the spread bimodal with confidence, but they show both runtimes see it, so the sweep's 0.77x for Mixtral is noise and is not reported as a ratio. A GPU-idle sweep earlier in the day, rejected only because compiler activity was not being logged yet, read 10.27 for mlxcel. Finding the cause (a clock or power state, placement of the 26 GB of weights, or something else) is left to #1814.

## Reproducing

1. Build: `cargo build --release --features rocm --bin mlxcel --bin mlxcel-bench-decode`. The harness refuses to run a binary older than the source.
2. Check the GPU is otherwise idle: `rocm-smi --showpids` should report no KFD process, and nothing should be compiling.
3. Run the two commands above. `bench_decode.sh` identifies the host with a one-token probe run and prints `Backend: rocm`, `ROCm: 10.0.0 (HIP 7.15.26333), gfx1151` and `Memory budget: 85% of 96.0 GiB (device memory, from the probe GPU line)` before the first model.
