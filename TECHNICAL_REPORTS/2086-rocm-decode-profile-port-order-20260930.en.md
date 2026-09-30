# Technical Report: PR #2086 - Profile gfx1151 decode per kernel and rank the #1814 ports

**Date**: 2026-09-30

**Status**: Implemented and measured on the gfx1151 host; head `0d5d6db0` rebased onto origin/main `f9aefa39`, pending merge.

**Languages**: Rust (bench binary flags and phase marks, test gates), Bash (idle-GPU guard, profile driver), Python (trace cut, attribution, report; CI checker), Markdown, CSV/JSON (committed profile data)

**Risk Level**: Low (no inference path changes. The bench binary gains two flags and an opt-in env var whose defaults keep its output unchanged; three test gates widen from Metal-or-CUDA to any GPU backend; the rest is scripts, docs and data)

## Executive Summary

Issue #2061 (part of #1814, epic #1801) asked for the measurement that #1814 left as a hypothesis: where ROCm decode time goes per kernel, and in which order the five port issues split from #1814 should land. The PR adds a profiling harness, runs it on four checkpoints on a Radeon 8060S (gfx1151), publishes the result as `docs/benchmark_results/rocm-decode-profile-gfx1151-2026-09-30.md` with its raw data under `benchmarks/rocm_profiles/gfx1151_929c80ab/`, and settles a second open question from the same session: MLX's ROCm `gather_mm` works, so its numeric tests now run on ROCm.

The measured order is **#2067 > #2065 > #2064 > #2068 > #2063** (#1814 items 7, 5, 4, 8, 3). #1814's hypothesis, based on call frequency, put item 3 (fused add-RMSNorm and RoPE-append, which run per layer per token on every model) first and item 7 (SSM update, one model family) near the end. The profile reverses those two: #2067 reaches 29.8% of granite-4.0-h-tiny and 20.0% of Nemotron-3-Nano decode GPU time and more than half of their dispatches, while #2063 reaches 0% on every model because both of its fusions ship disabled on every backend.

The report's other findings: dense decode is one kernel (`qmv_wide_kernel`, 94.8% of Llama 3.1 8B decode GPU time) running at about 192 GB/s; most of #2065's 46.8% share on Qwen3-30B-A3B is expert GEMVs already near that bandwidth, so only about 5% plus about 430 dispatches per token is recoverable; the profiler's own cost scales with dispatch count (within noise on Llama, 5.3% on Qwen3, 20 to 23% on the hybrids), which is why the doc reports shares and not times.

## 1. Problem Statement

### 1.1 An order without a measurement

On 2026-09-30 #1814 was split into nine issues. Five of them (#2063, #2064, #2065, #2067, #2068) are HIP ports that fill a `.rocm = nullptr` slot in a `KernelPorts` table, and each depends on #2061. #1814 gave only a starting hypothesis for their order: `fused_add_rms_norm` and `fused_rope_qk_append` run per layer per token on every model, the two samplers once per token on every model, the MoE and SSM kernels per token only on their families, and paged attention only on the paged path. The issue said explicitly that the profile decides the order and that the frequency argument was for a reviewer to check, not to assume.

The only ROCm decode data before this PR was the #2056 baseline, which has end-to-end tok/s (Llama-3.1-8B-4bit at 35.41 tok/s) but no per-kernel breakdown, so nothing said whether the gap to the GEMV bandwidth ceiling was kernel time, graph-fallback time or host time.

### 1.2 The last `BACKEND_ENUMERATION_TODO` entry

`grouped_gemm_numeric_tests.rs` gated its three tests on `!metal_is_available() && !cuda_is_available()`, so on ROCm they returned before touching the GPU. The tests exercise MLX's own `gather_mm`, not an mlxcel port, so no `KernelPorts` table applies, and the file was the last exemption from checker rule 4 in `scripts/ci/check_kernel_port_dispatch.py`. Widening the gate without running the tests was ruled out because that is how the #1806 abort had been reached.

## 2. Change Summary

Five commits:

- **`0908c888`** `test(rocm): run MLX gather_mm numeric tests on every GPU backend`: the three gates read `crate::gpu_backend_available()`; `BACKEND_ENUMERATION_TODO` becomes an empty set.
- **`10bee856`** `feat(bench): add a ROCm per-kernel decode profile harness`: phase marks, `--temperature` / `--top-p`, the guard script, the profile driver and post-processor, unit tests, `docs/benchmarks.md`.
- **`c098e02d`** `docs(rocm): publish the gfx1151 decode profile and #1814 port order`: the results page and the 53 files under `benchmarks/rocm_profiles/gfx1151_929c80ab/`.
- **`523882d1`** `docs(rocm): correct per-step counts in the gfx1151 decode profile`: review follow-up (128 generated tokens against 127 forward passes in the window; exact per-step SSD counts of 47 and 50; #2063 opt-in ceiling widened to 0.83 to 0.89%).
- **`0d5d6db0`** `fix(rocm): harden the decode profile scripts after security review`: path scrubbing of `guard.log` in an EXIT trap, safer PID parsing and option validation in the guard, INT/TERM cleanup, a slotted `Dispatch` class. `rocm_decode_profile.py report` output on the committed data is byte-identical before and after.

### 2.1 The harness

- **Phase marks** (`src/bin/bench_decode/phase_marks.rs`). With `MLXCEL_BENCH_PHASE_MARKS=1` the bench prints four stderr lines, `warmup_start`, `measured_start`, `decode_start` and `measured_end`, each on both `CLOCK_MONOTONIC` and `CLOCK_BOOTTIME` (tracers differ in which one they stamp). `measured_end` is read before the trailing `synchronize_default()`, because the decode loop has already waited on its last token, and `decode_start` is derived as `measured_end` minus the generator's own `decode_time_ms`, since the loop's start is inside the generator. Unset, the bench's output is unchanged for `scripts/bench_decode.sh`.
- **`--temperature` and `--top-p`** on `mlxcel-bench-decode` (defaults 0.0 and 1.0, so greedy as before). They exist so a profile can see the sampler at all: greedy argmax never dispatches it.
- **`scripts/rocm_gpu_guard.sh`**: the #2056 idle-GPU check as a reusable script (section 4).
- **`scripts/rocm_decode_profile.sh`**: per model, one plain bench run and one under `rocprofv3 --kernel-trace --hip-graph-trace --stats -f csv`, both through the guard, at `bench_decode.sh`'s default pp512/tg128 shape with a 20-token same-process warmup and `--ignore-eos`. Full traces (20 to 60 MB each) go to a temporary directory unless `--trace-dir` keeps them.
- **`scripts/rocm_decode_profile.py`**: cuts the trace to the measured decode by the phase marks, assigns roles to dispatches (`assign_roles()`), decides which roles each port reaches with shipped settings (`reach()`), and writes `<run>_decode_kernels.csv` and `<run>_summary.json`; its `report` subcommand renders the tables. `tests/test_rocm_decode_profile.py` pins the role rules on synthetic decode steps.

### 2.2 Rebase onto `f9aefa39`

PR #2084 had added per-phase `[Memory]` lines to `src/bin/bench_decode.rs` (issue #2062) in the same functions this PR edits. merge-resolver resolved the conflict keeping both: the memory counters print at each phase boundary, and the phase marks print around them (`reset_peak_memory()` then `warmup_start`; `measured_start`, then the measured pass, then `decode_start` / `measured_end`, then `print_memory_phase("after measured pass")`).

## 3. Measured Results

### 3.1 Method in brief

Four checkpoints: `Meta-Llama-3.1-8B-Instruct-4bit` (dense, f16), `Qwen3-30B-A3B-4bit` (MoE, 128 experts, 8 active), `granite-4.0-h-tiny-4bit` (36 Mamba2 + 4 attention layers, MoE), `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` (23 Mamba2, 23 MoE, 6 attention). Each greedy, plus `--temperature 0.7` and `--temperature 0.7 --top-p 0.95`. ROCm 10.0.0, HIP 7.15.26333, rocprofv3 1.3.5, mlxcel `929c80ab` plus the harness edits, overlay `75915908` with `LOCAL_FIXES.md` items 1 to 26.

Per token means per generated token (the bench's tok/s denominator). The first of the 128 tokens comes from prefill, so the window holds 127 forward passes and a per-token call count reads 127/128 of the per-step count (Qwen3's 144 expert GEMVs per step show as 142.9 per token). GPU time is the union of kernel intervals in the window; host gap is window wall time minus that. The cut is checked: in every run no kernel straddles it, the device is idle 2 to 57 ms before the first decode dispatch, and the last dispatch before it is always the first token's sampling.

### 3.2 Per model (greedy)

| Model | GPU ms/token | Host gap ms/token (profiled) | Host gap ms/token (plain wall minus GPU) | Dispatches/token |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 23.34 | 1.80 | 3.63 | 492 |
| Qwen3-30B-A3B | 13.36 | 3.63 | 2.77 | 1540 |
| granite-4.0-h-tiny | 11.47 | 8.59 | 4.83 | 3197 |
| Nemotron-3-Nano | 14.28 | 8.93 | 5.05 | 2008 |

Top kernels by share of decode GPU time (calls per token):

| Model | Kernels |
|---|---|
| Llama 3.1 8B | `qmv_wide_kernel` 94.8% (160), `kernel_sdpav_1pass` 2.1% (32), `rms_norm_kernel` 1.2% (64), `binary_vv<Add>` 0.6% (64), `copy_gg_byval` 0.5% (65), `rope_single_freqs_1d` 0.4% (64) |
| Qwen3-30B-A3B | `gather_qmv_wide_kernel` 42.1% (143), `qmv_wide_kernel` 33.1% (144), `kernel_sdpav_1pass` 4.6% (48), `block_sort_kernel` 4.3% (48, router top-k), `rms_norm_kernel` 3.9% (191), compiled SwiGLU 1.6% (48) |
| granite-4.0-h-tiny | `qmv_wide_kernel` 24.3% (168), `gather_qmv_wide_kernel` 21.9% (119), `binary_vv<Add, f32>` 6.6% (143), `binary_g<Multiply, f32>` 5.6% (178), `block_sort_kernel` 3.6% (40), `rms_norm_kernel` 3.4% (116) |
| Nemotron-3-Nano | `qmv_wide_kernel` 38.6% (116), `gather_qmv_wide_kernel` 27.9% (46), `binary_vv<Add, f32>` 4.1% (92), `binary_g<Multiply, f32>` 3.4% (114), `gemv_batched_inline<f32>` 2.2% (46), `copy_v` 2.1% (229) |

Three readings follow from the tables:

- **Dense decode is `qmv`.** Llama's GEMVs move about 4.2 GB per token in 22.1 ms, 192 GB/s. Nothing in the #1814 port list touches that kernel, so no port changes dense decode materially.
- **Qwen3's expert GEMVs are near bandwidth too.** 48 layers x 8 experts x 3 matrices of 2048 x 768 at 4.5 bits is 1.02 GB per token in 5.63 ms, 181 GB/s, about what its dense GEMVs reach (0.69 GB in 4.42 ms, 157 GB/s). The fused-MoE share is therefore mostly work a fused kernel still has to do.
- **The hybrids leave the GPU idle.** 3197 and 2008 dispatches per token, a plain-run host gap of 4.8 and 5.1 ms per token (about 1.5 and 2.5 us per dispatch), and a long tail of 1 to 2 us f32 elementwise kernels, which are the Mamba2 SSD step.

### 3.3 Profiling overhead, and why the doc reports shares

Each greedy run was also taken without the profiler under the same guard:

| Model | Dispatches/token | Decode tok/s, plain | Decode tok/s, profiled | Profiled slower by |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 492 | 37.08 | 39.78 | -6.8% (faster; within run-to-run spread) |
| Qwen3-30B-A3B | 1540 | 61.99 | 58.86 | 5.3% |
| granite-4.0-h-tiny | 3197 | 61.35 | 49.86 | 23.0% |
| Nemotron-3-Nano | 2008 | 51.74 | 43.08 | 20.1% |

rocprofv3 adds host time per dispatch and barely changes kernel durations, so its cost tracks dispatch count. On the hybrids, which are the models whose ranking depends on the host gap, the profiled wall time is 20 to 23% too slow, so profiled absolute times would overstate exactly the quantity under study. Shares of decode GPU time are robust to that, and for the host gap the doc prefers plain wall time per token minus traced GPU time (the third column of 3.2). Each overhead figure is one plain and one profiled run, so the Llama reading says only that the cost there is below noise (the #2056 baseline and `LOCAL_FIXES.md` item 24 read 35.4 to 35.9 tok/s for that model).

## 4. The Idle-GPU Guard

On a UMA host another GPU tenant or a compiler competes for the same memory bus, and this host was shared with other units and orchestrator gates during the session. `scripts/rocm_gpu_guard.sh [--idle-secs N] [--max-attempts N] [--max-wait SECS] [--log FILE] -- COMMAND`:

1. waits for `--idle-secs` (default 90) consecutive seconds with `/sys/class/kfd/kfd/proc` empty and no compiler process (`/proc/<pid>/comm` matched exactly against `cargo`, `rustc`, `clang*`, `hipcc`, `nvcc`, `cc1`, `cc1plus`, `ld*`, `lld`, `collect2`);
2. runs COMMAND under a 1 Hz monitor;
3. rejects the attempt if any sample shows a GPU process that is not COMMAND or a descendant, or a compiler, and goes back to step 1.

Every sample goes to `--log`. Exit status is COMMAND's from the first clean attempt, or 75 when every attempt was contended or `--max-wait` ran out. INT and TERM stop COMMAND and the monitor. The driver scrubs local paths from the log on exit.

The committed `benchmarks/rocm_profiles/gfx1151_929c80ab/guard.log` shows the guard doing its job. The granite greedy profiled attempt at 21:03:05 was clean for three samples; sample 4 showed a second `mlxcel-bench-de` process (another unit's bench, four seconds in), sample 5 logged `CONTENDED foreign_gpu=[2682448:mlxcel-bench-de]`, and the attempt ended `REJECTED (contended), exit 0; rerunning`. The bench itself exited 0, which is the point: the output looked valid and would have been published without the guard. After further idle-streak resets, attempt 2 started at 21:15:18 and ended `CLEAN, exit 0`. A different run's attempt (Qwen3 `t0.7` at 21:48) passed the guard but was discarded by hand because a CPU-heavy trace analysis of this unit ran alongside it, which the guard does not watch; the log records the replacement. The 16 accepted runs span 19:36 to 22:14 KST, and in each one every sample showed no GPU process or only the run itself, and no compiler. When a run is rerun, its `_bench.log` keeps both attempts and the post-processor reads only the last.

## 5. Attribution and the Port Order

### 5.1 From kernel names to ports

A kernel name identifies the primitive, not the mlxcel op that asked for it: a `binary_vv<Add>` can be a residual join, part of the SSD step or part of a logit bias. Because MLX evaluates each decode step's graph in the same order every step, each op leaves the same run of dispatches, and `assign_roles()` labels those runs (`sampler_tail`, `add_rms_join_post_attn`, `add_rms_join`, `rope_append`, `moe_expert_gemv`, `moe_activation`, `moe_weighted_sum`, `moe_gather_indices`, `ssm_step`, `ssm_conv`, `ssm_silu`, `ssm_gated_norm`). The rules were checked by counts: exactly 47 SSD dispatches per Mamba2 layer per step on granite and 50 on Nemotron, whose `ssm_step` graphs are built by separate but parallel code; exactly 3 expert GEMVs, 3 aranges and 4 weighted-sum dispatches per Qwen3 layer; every SSM and MoE role count a whole multiple of its layer count. Router top-k is deliberately left out of MoE because `forward_fused_kernel` takes `topk_indices` and `scores` from the caller.

`reach()` then encodes, from the source at `929c80ab`, whether mlxcel would actually call each port for that model with shipped settings. Each table cell is the share reached; the bracket is the fallback cost the port family covers whether or not it is called.

### 5.2 Share per port under shipped settings

| Model, run | #2063 | #2064 | #2065 | #2067 | #2068 |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B, greedy | 0 (1.66) | 0 (0.08) | 0 | 0 | 0 |
| Llama 3.1 8B, t0.7 | 0 (1.65) | 0.37 | 0 | 0 | 0 |
| Llama 3.1 8B, t0.7 p0.95 | 0 (1.92) | 1.17 | 0 | 0 | 0 |
| Qwen3-30B-A3B, greedy | 0 (2.82) | 0 (0.17) | **46.79** | 0 | 0 |
| Qwen3-30B-A3B, t0.7 | 0 (2.77) | 0.69 | 46.52 | 0 | 0 |
| Qwen3-30B-A3B, t0.7 p0.95 | 0 (2.67) | 2.75 | 46.03 | 0 | 0 |
| granite-4.0-h-tiny, greedy | 0 | 0 (0.20) | 0 (25.42) | **29.82** | 0 |
| granite-4.0-h-tiny, t0.7 | 0 | 0.72 | 0 (22.51) | 30.75 | 0 |
| granite-4.0-h-tiny, t0.7 p0.95 | 0 | 2.51 | 0 (21.62) | 31.28 | 0 |
| Nemotron-3-Nano, greedy | 0 (0.26) | 0 (0.17) | 0 (29.22) | **20.01** | 0 |
| Nemotron-3-Nano, t0.7 | 0 (0.26) | 0.68 | 0 (27.48) | 20.28 | 0 |
| Nemotron-3-Nano, t0.7 p0.95 | 0 (0.25) | 3.81 | 0 (25.99) | 21.50 | 0 |

A share is an upper bound on savings; how much of it a kernel can recover decides the order:

1. **#2067 SSM update (item 7).** 29.8% (granite) and 20.0% (Nemotron) of decode GPU time, and 1679 and 1141 dispatches per token, more than half of each model's dispatches and the source of their 5 ms per token host gap. `ssm_kernel_available()` gates every single-token SSD step, so the port is reached. One kernel's traffic is the SSM state, about 113 MB per token on granite or about 0.6 ms at 180 GB/s, against the 3.42 ms the graph takes now (2.9 ms on Nemotron). Mostly recoverable.
2. **#2065 fused MoE decode (item 5).** 46.8% of Qwen3 decode GPU time is reached (`qwen3_moe.rs:223`), but 42.1 points are the expert GEMVs at 181 GB/s. What a fused kernel can recover is the activation, weighted sum and index building (4.7%, 0.62 ms per token) plus about 430 of the 524 dispatches per token. Granite and Nemotron carry 25 to 29% fallback MoE cost the port as scoped does not reach: granite's `block_sparse_moe` calls `SwitchGLU::forward`, not `forward_fused_kernel`, and Nemotron-H's default branch is `gather_qmm` (its kernel branch needs `MLXCEL_FUSED_MOE_RELU2` and #2069).
3. **#2064 samplers (item 4).** Zero in greedy decode, 0.4 to 0.7% with temperature, 1.2 to 3.8% with top-p, where a full-vocabulary `rocprim` radix sort and scans run each token. The figures include the few `--ignore-eos` logit-bias dispatches, so they read slightly high.
4. **#2068 paged attention (item 8).** No share: the bench decodes one sequence into a dense `KVCache`, and no paged kernel or paged fallback appears in any trace.
5. **#2063 fused add-RMSNorm and RoPE-append (item 3).** Zero with shipped settings on every backend: `FUSED_ADD_RMSNORM_DEFAULT` and `FUSED_ROPE_APPEND_DEFAULT` are `false` (`layers.rs:808`, `:820`) after #905 measured no decode win on Metal, and only `llama3.rs` (plus the unprofiled `gemma.rs` and `iquestloopcoder.rs`) calls them. Opted in, Llama 3.1 would reach only the post-attention join, 0.83% (0.89% in the top-p run), because its `rope_scaling` builds a frequency table that routes around the RoPE kernel (`llama3.rs:669`).

**Why #2068 ranks above #2063 with no measured share.** Both are zero here, so the tie is broken on scope, not data. #2068 has a real path this profile does not cover (batched paged serving) and 36 ROCm test skips that its port clears. #2063 has nothing on any backend's default path, and its best case if someone opts in is under 1% of one model. The doc states this explicitly so the placement is not read as a measurement.

**Against #1814's hypothesis.** The frequency argument was right that items 3 and 4 run on every model and item 7 on one family. It missed two things a profile shows immediately: a kernel that runs everywhere can still reach nothing if its caller ships it off, and per-token cost differs by more than an order of magnitude between a 47-dispatch f32 graph per layer (3.42 ms per token on granite) and a single add-plus-norm join (0.83% of Llama decode, about 0.19 ms). The ranking comment on #1814 records the new order, 7, 5, 4, 8, 3, against the old 3, 4, 5, 7, 8.

## 6. MLX's ROCm `gather_mm`

Each test was run on gfx1151 by exact name, and each also under rocprofv3 to confirm the GPU path:

| Test | Result | Kernels in its trace |
|---|---|---|
| `gather_mm_matches_dense_per_expert_reference` | pass | `gather_batched_gemm_kernel<float, false, true>` x4, `<float, false, false>` x1, a Tensile `Cijk_*` GEMM x4 (hipBLASLt, the sorted single-row case) |
| `gather_mm_selects_the_indexed_expert` | pass | `gather_batched_gemm_kernel<float, false, false>` |
| `gather_mm_half_precision_matches_reference` | pass | `gather_batched_gemm_kernel<hip_bfloat16, ...>`, `gather_batched_gemm_kernel<__half, ...>` |

The overlay implements `GatherMM::eval_gpu` itself (`matmul.cpp`) and matches the f64 host reference within the tests' tolerances. As a negative control, with the reference deliberately pointed at the wrong expert, all three failed at the value assertion (`grouped_gemm_numeric_tests.rs:111`), so a pass is evidence and not a skip. The gates now read `gpu_backend_available()` (Metal and CUDA still run them), `BACKEND_ENUMERATION_TODO` is empty, and the checker prints `0 awaiting a predicate`, which closes the last rule 4 exemption #1814 owned.

## 7. Technical Decisions

- **Cut the decode window by host clock, not by kernel name.** The warmup and measured passes run the same kernels, so kernel names alone cannot say which dispatches belong to the measured decode. Marks printed by the bench, on both clocks, plus two cleanliness checks (no straddling kernel, a device-idle gap before the first dispatch) make the cut verifiable per run.
- **Attribute by position in the step, and pin it with counts.** Role rules key off the graph order of each op and were checked against exact per-layer dispatch counts. The alternative, attributing by kernel name alone, would have charged residual adds and SSD adds to the same port.
- **Report reach under shipped settings, with the fallback cost in brackets.** A port that fills a `.rocm` slot nobody calls saves nothing. Separating "reached" from "covered" is what turns #2063 from a large bracketed number into zero and exposes granite's unwired MoE as a follow-up rather than a #2065 win.
- **Weigh shares by recoverability.** Ranking by raw share alone would put #2065 first. Bandwidth arithmetic on the GEMVs shows most of that share is not recoverable, which is why #2067 leads.
- **Shares, not times.** With profiler overhead up to 23% on the models that matter most, profiled absolute times would bias the ranking toward the dispatch-heavy hybrids.
- **Guard as a script with a log.** The #2056 guard was a procedure; making it a script with an exit code and a per-sample log makes every published run auditable and reusable by the port PRs that will need before/after numbers.
- **Run the `gather_mm` tests before widening the gate, with a negative control.** This follows the #1806 lesson directly: the gate change is backed by a trace and by a test mutation that must fail.

## 8. Validation

From the PR body, on gfx1151 before the rebase:

- `cargo test --release --features rocm -p mlxcel-core --lib grouped_gemm_numeric_tests -- --test-threads=1 --nocapture`: 3 passed; each rerun by exact name under rocprofv3; all three fail with a wrong-expert reference.
- `python3 scripts/ci/check_kernel_port_dispatch.py`: `0 awaiting a predicate`; `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt` pass.
- `cargo clippy -p mlxcel --features rocm --bin mlxcel-bench-decode -- -D warnings` and `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings`: clean.
- `cargo test --features rocm --test dead_doc_pointers`: pass. `python3 -m unittest tests/test_rocm_decode_profile.py`: 17 passed; `bash -n` on the three scripts.

Orchestrator verification after the rebase onto `f9aefa39`:

- merge-resolver resolved the `src/bin/bench_decode.rs` conflict with PR #2084's per-phase `[Memory]` lines and kept both behaviors.
- Clippy on the bench binary was clean; `tests/test_rocm_decode_profile.py` passed 17; the bench harness Python tests passed 35.
- A functional run of the rebased bench printed both the `[Memory]` lines and the four phase marks in order.
- The full `make verify-rocm` was run on the rebased head by the orchestrator.

## 9. Residual Risks and What Was Not Verified

- **One device, one session.** All numbers are from gfx1151 at `929c80ab` with overlay `75915908`. Other RDNA or CDNA parts, a different overlay, or a future `qmv` change can move the shares. The profile is reproducible with the committed scripts.
- **Overhead figures are single runs.** One plain and one profiled run per model; the Llama figure (profiled 6.8% faster) is noise, and the hybrids' 20 to 23% has no spread attached.
- **The guard's blind spots.** Sampling is 1 Hz, so a GPU job under a second can slip between samples, and CPU load from non-compiler processes is not watched (the Qwen3 `t0.7` rerun was a manual catch).
- **Shares are upper bounds, not speedups.** No A/B was run, because on ROCm none of these paths has a kernel to switch to. Each port PR has to measure its own before/after.
- **Attribution is rule-based.** The rules are pinned by per-layer counts and synthetic-step tests, but a model whose graph order differs from the four profiled here would need its own check. `reach()` encodes source at `929c80ab`; a later change to `FUSED_*_DEFAULT`, `block_sparse_moe` or `fused_moe_forward` changes the reach column.
- **#2068 is not measured.** Batched paged serving is outside this harness; its rank rests on scope, not data.
- **Not verified here:** Metal and CUDA (the widened `gather_mm` gates still run there, unchanged); `cargo test --test dead_doc_pointers` without `--features rocm` fails to link on this host (`copy_gpu_inplace` undefined from `kv_inplace_write.cpp`), unrelated to this change.

## 10. Learning Points

- **"Runs everywhere" is not "costs the most".** Call frequency ignores whether the fused path is enabled at all and how much work each call does. A profile with reach encoded answered in one pass what the frequency hypothesis had inverted.
- **A share needs a recoverability estimate before it becomes a priority.** Bandwidth arithmetic on the kernels a port replaces separates work that must still happen from overhead that can disappear.
- **Measure the profiler.** Tracing cost that scales with dispatch count skews exactly the dispatch-heavy models, so the unprofiled run is part of the method, not an extra.
- **A guard earns trust by rejecting something.** The committed log contains a real rejection of a run that exited 0, which is stronger evidence for the other 16 runs than a log of only clean samples.
- **Widen a gate with a negative control.** Passing tests plus a trace show the GPU path ran; a deliberately wrong reference that fails shows the tests can tell.

Refs: #2061, #1814, #1801, #2056, #2062, #2063, #2064, #2065, #2067, #2068, #2069, #2084, #1806, #905.
