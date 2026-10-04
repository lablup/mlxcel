# HIP port of the fused SSM update kernel on gfx1151 (2026-10-04)

lablup/mlxcel#2067, part of #1814. Before this change every single-token Mamba2 decode step on ROCm ran the ~55-op SSD graph (`ssm_step`), because `ssm_ports()` had no `.rocm` entry and `ssm_kernel_available()` answered with `cu::is_available()` off Apple. The decode profile ([rocm-decode-profile-gfx1151-2026-09-30.md](rocm-decode-profile-gfx1151-2026-09-30.md)) put that graph at 29.8% of granite-4.0-h-tiny's and 20.0% of Nemotron-H's decode GPU time, the largest share of any #1814 port.

The port (`SSM_HIP_SOURCE` in `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`) is the CUDA kernel with `__shfl_down(acc, o, 32)` for the lane fold. It carries the `__AMDGCN_WAVEFRONT_SIZE` `#error` guard the #1814 ports share, which is inert with ROCm 10's AMD clang 23 (it defines neither macro for gfx1151, gfx942 or gfx90a); the fold does not need it, because the shuffle width of 32 keeps each reduction inside the 32 lanes of one row on a wave64 target too. `ssm_kernel_available()` now returns `has_kernel_port(ssm_ports())` on every platform, so the model gates (`seq_len == 1 && ssm_kernel_available()` in granitemoehybrid, falcon_h1, plamo2 and nemotron_h) reach it with no model-side change. On Nemotron-H that gate (`nemotron_h.rs`, `NemotronHMamba2Mixer::forward`) selects `fused_mamba2_forward`, which runs the whole single-token mixer (input projection, convolution, this kernel, gated norm, output projection) as one C++ call, so its decode figures below measure that path against the Rust graph mixer, not the SSM step alone. The predicate also answers false when the default device is the CPU (`MLXCEL_DEVICE=cpu`), where a custom kernel cannot run. `MLXCEL_SSM_KERNEL=0` forces the graph on every backend; `MLXCEL_SSM_CUDA_KERNEL=0` is kept as an alias.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0 (HIP 7.15.26333). mlxcel `844bd94c` (the PR branch on `origin/main` `c0b71344`) for the decode and logit rows; the later commits of the PR add only the CPU-device check in `ssm_kernel_available()` and host shape checks in front of the same launch, and the greedy 128-token output of both models was re-checked identical after them, MLX pin `81ba1c6a`, ROCm overlay `75915908`, `cargo build --release --features rocm`. Checkpoints: `mlx-community/granite-4.0-h-tiny-4bit` and `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit`, the same local directories as the earlier correctness rows.

## Decode throughput

`scripts/bench_decode.sh` at its default pp512/tg128 shape, one model per run, `MLXCEL_SSM_KERNEL=0` for the graph and unset for the kernel, on the same binary. Three runs per arm, the arm order alternated between runs. Every run went through `scripts/rocm_gpu_guard.sh` (90 s with `/sys/class/kfd/kfd/proc` empty and no compiler, then 1 Hz monitoring); a parallel development unit shared the GPU, and every attempt that overlapped its work was rejected and rerun. The first run used the guard on `main`, the rest the copy fixed in lablup/mlxcel#2098 (commit `c3eab7b2`), which stops rejecting the command's own exited children; the old guard's rejections were false, never its acceptances. Raw rows: `benchmarks/rocm_strixhalo-gfx1151_2026-10-04_ssm-kernel-off.csv` and `..._ssm-kernel-on.csv`.

| Model | Graph (`MLXCEL_SSM_KERNEL=0`) tok/s | Kernel tok/s | Speedup (medians) |
|---|---|---|---|
| granite-4.0-h-tiny-4bit | 60.57 / 60.39 / 60.05, median 60.39 | 88.36 / 88.69 / 88.43, median 88.43 | 1.46x |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | 51.65 / 50.87 / 51.35, median 51.35 | 74.87 / 72.87 / 74.37, median 74.37 | 1.45x |

Run-to-run spread is under 1% for granite and under 3% for Nemotron-H in either arm, far below the change. `compare_bench_csv.py --before <off> --after <on>` (which keeps the last row per model) reports 1.47x and 1.45x. Prefill is not on this path (a prompt is more than one token) and is not compared; it varied between 379 and 608 tok/s for granite across both arms on this shared host.

## Correctness

### Kernel against the graph, one step

`ssm_update_parity_tests` (`src/lib/mlxcel-core/src/ssm_update_parity_tests.rs`) compares the kernel's output and new state with a float32 MLX-op reference of the single-token SSD step at the granite-4.0-h-tiny shape (48 heads of 64, one group, state 128) and the Nemotron-H shape (64 heads of 64, eight groups, state 128, batch 2, clipped dt), f32 and bf16. Tolerances, normalized RMS / max: f32 1e-5 / 1e-4, bf16 1.6e-2 / 7e-2; the f32 state is held to the f32 budget in both. It passes on gfx1151. It fails with normalized RMS 0.65 to 0.89 when the lane fold starts at 8 instead of 16.

A third test runs the same shape with `A_log` first in bf16 and then in f32 in one process. Nemotron-H stores `A_log` in f32 next to bf16 activations and granite in bf16, and the generated kernel signature takes each input's runtime dtype, so the launch now names the `A_log`, `B` and `C` dtypes in its template arguments (the cache key on CUDA and ROCm). Without them the second launch reuses the bf16 module and the test fails at normalized RMS 0.80.

The f32 test also runs a head size of 60, which leaves rows past `Dh` in the last threadgroup of 8. `ssm_update_kernel_refuses_unsupported_shapes` pins that shapes the kernel cannot index (a state width that is not a multiple of 32, heads that do not divide into groups; the host also checks rank, a single token, the batch and every input's element count) throw `std::invalid_argument` before any launch, and a fifth test that `MLXCEL_SSM_KERNEL=0` and `MLXCEL_SSM_CUDA_KERNEL=0` each turn the predicate off. The CPU-device rule is not covered by a test: switching the process's default device inside the shared test binary would move other tests to the CPU.

### Model logits: kernel against kernel

The `w1ctx512` traces (`1 128 8 512` with `MLXCEL_TRACE_START_TOKEN=512`) prefill 512 corpus tokens into a fresh cache and then trace one token, so every traced step meets an SSM state and takes the fused step. Until now ROCm ran the graph there and the rows compared Metal's kernel with ROCm's graph ([rocm-correctness-gfx1151-2026-09-30.md](rocm-correctness-gfx1151-2026-09-30.md#the-fused-ssm-kernel-against-the-graph-with-state)). With the port they compare kernel with kernel. New traces: `benchmarks/logit_traces/rocm_gfx1151_96cbce84/`, built at the PR's first commit (the second changes only the kernel's cache name, the same on ROCm in both). `ssmkernel0` is the same binary with `MLXCEL_SSM_KERNEL=0`. `python3 scripts/compare_logit_traces.py <reference> <candidate> --decided 2.0`, Nemotron-H files filtered with `grep -v '^\[NemotronH\] '`.

| Model | Reference | Candidate | Top-1 disagreement | Decided mismatches | Largest gap | Perplexity ref / cand |
|---|---|---|---|---|---|---|
| granite-4.0-h-tiny | Metal kernel (`metal_m5_3c9edea0`) | ROCm kernel | 3 / 128 | 0 / 60 | 0.125 | 7.090 / 7.216 |
| granite-4.0-h-tiny | Metal kernel | ROCm graph (`ssmkernel0`) | 2 / 128 | 0 / 60 | 0.125 | 7.090 / 7.290 |
| granite-4.0-h-tiny | ROCm graph (`ssmkernel0`) | ROCm kernel | 1 / 128 | 0 / 59 | 0.000 | 7.290 / 7.216 |
| granite-4.0-h-tiny | ROCm graph at `3c9edea0` | ROCm kernel | 1 / 128 | 0 / 60 | 0.125 | 7.234 / 7.216 |
| nemotron-3-nano-30b-a3b | Metal kernel (`metal_m5_3c9edea0`) | ROCm kernel | 3 / 128 | 0 / 71 | 0.250 | 7.489 / 7.442 |
| nemotron-3-nano-30b-a3b | Metal kernel | ROCm graph (`ssmkernel0`) | 4 / 128 | 0 / 71 | 0.250 | 7.489 / 7.460 |
| nemotron-3-nano-30b-a3b | ROCm graph (`ssmkernel0`) | ROCm kernel | 1 / 128 | 0 / 68 | 0.125 | 7.460 / 7.442 |
| nemotron-3-nano-30b-a3b | ROCm graph at `3c9edea0` | ROCm kernel | 4 / 128 | 0 / 68 | 0.125 | 7.489 / 7.442 |

Every row has zero disagreements on a decided position, and every disagreement puts the reference's token at rank 2 or 3 of the candidate, at a reference gap of 0.25 or less. The kernel and graph traces from one ROCm binary differ (different perplexities, one flipped token each), which shows the `default` traces took the kernel. Against Metal's kernel the ROCm kernel is as close as the ROCm graph was (3 and 3 top-1 disagreements, against 2 and 4 for the graph here and 4 and 7 in the earlier `3c9edea0` comparison).

The `w1` rows (`1 128 8 0`) have no prefill, so their single token never meets a state and runs the graph with or without the kernel. Against the graph-path traces in `benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/` both are identical at every position (granite 0 / 128, 0 of its decided positions; Nemotron-H 0 / 128, no decided position). They show the change leaves the stateless path alone, not anything about the kernel.

### Greedy output

`mlxcel generate -p "The history of the Roman Empire" -n 128 -t 0 --no-chat-template`, with and without `MLXCEL_SSM_KERNEL=0`. Both arms of both models produce coherent text, but they are not byte-identical (on CUDA they were, per [hybrid-ssm-decode-cuda-kernel-gb10-2026-07-10.md](hybrid-ssm-decode-cuda-kernel-gb10-2026-07-10.md)): granite parts at the third generated token ("a rich and complex tapestry" against "a fascinating and complex subject"), Nemotron-H after about a dozen ("a vast and complex tapestry spanning centuries," is common to both). A free-running generation conditions everything after the first flip on different text, which is why the teacher-forced traces above are the comparison; they show the flips land only where the model was undecided.

## Reproducing

```bash
cargo build --release --features rocm --bin mlxcel --bin mlxcel-bench-decode --example logit_trace
cargo test --release --features rocm -p mlxcel-core --lib ssm_update_parity_tests -- --test-threads=1
# bench_decode.sh truncates --output, so every run writes its own file.
for i in 1 2 3; do
  if [ $((i % 2)) = 1 ]; then arms="off on"; else arms="on off"; fi
  for m in granite-4.0-h-tiny-4bit NVIDIA-Nemotron-3-Nano-30B-A3B-4bit; do
    for arm in $arms; do
      if [ $arm = off ]; then e=MLXCEL_SSM_KERNEL=0; else e=MLXCEL_SSM_KERNEL=1; fi
      env $e scripts/rocm_gpu_guard.sh -- env MODELS_DIR=models/mlx \
        ./scripts/bench_decode.sh models/mlx/$m --no-cooldown --output $arm-${m%%-*}-r$i.csv
    done
  done
done
for arm in off on; do
  head -1 $arm-granite-r1.csv > $arm.csv
  for f in $arm-*-r*.csv; do tail -n +2 "$f" >> $arm.csv; done
done
python3 scripts/compare_bench_csv.py --before off.csv --after on.csv
```

The trace loop is in `benchmarks/logit_traces/rocm_gfx1151_96cbce84/README.md`.
