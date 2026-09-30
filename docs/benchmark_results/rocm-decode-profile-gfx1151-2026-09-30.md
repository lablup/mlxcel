# ROCm decode profile: Radeon 8060S (gfx1151), 2026-09-30

Where ROCm decode time goes, per kernel, on four checkpoints, and which of the port issues split from #1814 would take the most of it (issue #2061). It turns #1814's call-frequency hypothesis into a measured order. The same session settles whether MLX's ROCm `gather_mm` works: it does, and its tests now run on ROCm.

Raw results, one set per run (`benchmarks/rocm_profiles/gfx1151_929c80ab/`):

- `<run>_kernel_stats.csv`: rocprofv3's own `--stats` output for the whole process (load, warmup, prefill and decode);
- `<run>_decode_kernels.csv`: the measured decode only, one row per kernel, with its class, role, port unit, calls per token and share of decode GPU time;
- `<run>_summary.json`: the numbers below, the attribution per port unit, and the window checks;
- `<run>_bench.log`, `<run>_plain_bench.log`: the bench's output under the profiler and without it;
- `guard.log`: every idle-GPU check and 1 Hz sample behind every run.

When the guard rejects an attempt and reruns it, the run's `_bench.log` keeps both attempts' output (granite greedy); the last one is the accepted run, and it is the only one `rocm_decode_profile.py` reads and the one rocprofv3's files hold.

The full kernel traces (20 to 60 MB each) are not committed; `scripts/rocm_decode_profile.sh` regenerates all of the above.

## Environment

| Item | Value |
|------|-------|
| **Hardware** | AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 32 CPU threads, 96 GiB VRAM carve-out (`mem_info_vram_total` 103079215104), 31 GiB visible to the host (`MemTotal` 32493820 kB) |
| **OS** | Debian GNU/Linux 13 (trixie), kernel 6.18.12+deb13-amd64 |
| **ROCm / HIP** | ROCm 10.0.0 (`/opt/rocm/core-10.0/.info/version`), HIP runtime 7.15.26333 (`hipconfig --version`) |
| **Profiler** | rocprofv3 1.3.5 (git `6b0e43f3`), `--kernel-trace --hip-graph-trace --stats -f csv` |
| **mlxcel** | 0.7.0, `main` at `929c80ab` plus this change's harness edits (`src/bin/bench_decode.rs` flags and phase marks, which run no kernel); built with `cargo build --release --features rocm --bin mlxcel-bench-decode` |
| **MLX pin** | `81ba1c6a` |
| **MLX ROCm overlay** | NripeshN/mlx `rocm-support` at `75915908` (`src/lib/mlx-cpp/patches-rocm/UPSTREAM`), plus `LOCAL_FIXES.md` items 1 to 26 |
| **Toolchain** | Rust 1.97.1 |

## Method

**Shape.** `scripts/bench_decode.sh`'s default: a deterministic 512-token prompt, exactly 128 generated tokens with every end-of-generation token suppressed, a discarded 20-token warmup in the same process, batch 1, default dense KV cache. Greedy for the main runs; two sampled variants (`--temperature 0.7`, and `--temperature 0.7 --top-p 0.95`) so the sampler shows up at all.

**Idle GPU.** Every run went through `scripts/rocm_gpu_guard.sh`: 90 consecutive seconds with `/sys/class/kfd/kfd/proc` empty and no compiler process (`cargo`, `rustc`, `clang*`, `hipcc`, `cc1`, `cc1plus`, `ld*`, `lld`, `collect2`), then the run under a 1 Hz monitor, rejected and rerun if any sample showed another GPU process or a compiler. Other units and orchestrator gates shared the host, so the 16 accepted runs spread from 19:36 to 22:14 KST. One attempt was rejected by the guard (granite profiled, 21:03: another unit's bench started four seconds in). One accepted attempt was discarded by hand and rerun (Qwen3 `t0.7`, 21:48: a trace analysis of this unit was using a CPU core, which the guard does not watch). In every accepted run's window every sample showed either no GPU process or only the run itself, and no compiler; `guard.log` holds every sample. A GPU job shorter than one second could in principle fall between samples.

**Decode window.** `MLXCEL_BENCH_PHASE_MARKS=1` makes the bench print, on both `CLOCK_MONOTONIC` and `CLOCK_BOOTTIME`, the start of the warmup, the start of the measured pass, the start of the measured decode and its end (`src/bin/bench_decode/phase_marks.rs`). The decode start is the end minus the generator's own `decode_time_ms`. `scripts/rocm_decode_profile.py` keeps the dispatches that start inside that window. Two checks say the cut is clean: the prefill ends in a blocking `eval` of the first token, and in every run no kernel straddles the cut and the device sits idle for 2 to 57 ms before the first decode dispatch; the last dispatch before the cut is always the first token's sampling (`arg_reduce_final` in greedy runs). rocprofv3 stamps dispatches on the same clock the marks use (both clocks agree on this host, which had not suspended).

**Numbers.** Per token means per generated token, the denominator of the bench's tok/s. The first of the 128 tokens comes out of the prefill, so the decode window holds 127 forward passes, and a per-token call count reads 127/128 of the per-step count (Qwen3's 144 expert GEMVs per step show as 142.9 per token). GPU time per token is the union of kernel intervals in the window; host gap per token is the window's wall time minus that. Shares are of the sum of kernel durations in the window (one queue, so union and sum agree to within 0.01%).

**Profiler cost.** Each greedy run was also taken without the profiler, under the same guard. The profiler adds host time per dispatch and barely changes kernel durations, so it hurts in proportion to dispatch count:

| Model | Dispatches per token | Decode tok/s, plain | Decode tok/s, profiled | Profiled slower by |
|---|---:|---:|---:|---:|
| `Meta-Llama-3.1-8B-Instruct-4bit` | 492 | 37.08 | 39.78 | -6.8% (faster; within run-to-run spread) |
| `Qwen3-30B-A3B-4bit` | 1540 | 61.99 | 58.86 | 5.3% |
| `granite-4.0-h-tiny-4bit` | 3197 | 61.35 | 49.86 | 23.0% |
| `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | 2008 | 51.74 | 43.08 | 20.1% |

Each figure is one plain and one profiled run, so the Llama reading says only that the cost there is below the noise (the #2056 baseline and `LOCAL_FIXES.md` item 24 read 35.4 to 35.9 tok/s for the same model). Read the shares, not the profiled absolute times; for the host gap, the plain run's wall time per token minus the traced GPU time is the better estimate and is given alongside.

## Results (greedy)

| Model | GPU ms/token | Host gap ms/token (profiled) | Host gap ms/token (plain wall minus GPU) | Dispatches/token |
|---|---:|---:|---:|---:|
| `Meta-Llama-3.1-8B-Instruct-4bit` (dense, f16) | 23.34 | 1.80 | 3.63 | 492 |
| `Qwen3-30B-A3B-4bit` (MoE, 128 experts, 8 active) | 13.36 | 3.63 | 2.77 | 1540 |
| `granite-4.0-h-tiny-4bit` (36 Mamba2 + 4 attention layers, MoE) | 11.47 | 8.59 | 4.83 | 3197 |
| `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` (23 Mamba2, 23 MoE, 6 attention) | 14.28 | 8.93 | 5.05 | 2008 |

Top kernels by share of decode GPU time (calls per token):

| Model | Kernels |
|---|---|
| Llama 3.1 8B | `qmv_wide_kernel` 94.8% (160), `kernel_sdpav_1pass` 2.1% (32), `rms_norm_kernel` 1.2% (64), `binary_vv<Add>` 0.6% (64), `copy_gg_byval` 0.5% (65), `rope_single_freqs_1d` 0.4% (64) |
| Qwen3-30B-A3B | `gather_qmv_wide_kernel` 42.1% (143), `qmv_wide_kernel` 33.1% (144), `kernel_sdpav_1pass` 4.6% (48), `block_sort_kernel` 4.3% (48, router top-k), `rms_norm_kernel` 3.9% (191), compiled SwiGLU 1.6% (48) |
| granite-4.0-h-tiny | `qmv_wide_kernel` 24.3% (168), `gather_qmv_wide_kernel` 21.9% (119), `binary_vv<Add, f32>` 6.6% (143), `binary_g<Multiply, f32>` 5.6% (178), `block_sort_kernel` 3.6% (40), `rms_norm_kernel` 3.4% (116) |
| Nemotron-3-Nano | `qmv_wide_kernel` 38.6% (116), `gather_qmv_wide_kernel` 27.9% (46), `binary_vv<Add, f32>` 4.1% (92), `binary_g<Multiply, f32>` 3.4% (114), `gemv_batched_inline<f32>` 2.2% (46), `copy_v` 2.1% (229) |

What that says:

- The dense model is weight-bandwidth bound and nothing in the port list touches it. Its GEMVs move about 4.2 GB per token in 22.1 ms, 192 GB/s.
- Qwen3's expert GEMVs are bandwidth bound too: 48 layers x 8 experts x 3 matrices of 2048 x 768 at 4.5 bits is 1.02 GB per token in 5.63 ms, 181 GB/s, about what the dense GEMVs reach (0.69 GB in 4.42 ms, 157 GB/s).
- The two hybrids are the ones leaving the GPU idle: 3197 and 2008 dispatches per token, a host gap of 4.8 and 5.1 ms per token without the profiler (about 1.5 and 2.5 us per dispatch), and a long tail of 1 to 2 us f32 elementwise kernels that are the Mamba2 SSD step.

## Attribution

A kernel name says which primitive ran, not which mlxcel op asked for it: a `binary_vv<Add>` can be a residual join, part of the SSD step or part of a logit bias. MLX evaluates each decode step's graph in the same order every step, so every op leaves the same run of dispatches, and `assign_roles()` in `scripts/rocm_decode_profile.py` labels those runs. Each rule was written against a printed decode step of the model it matters for and against the mlxcel code that builds the graph; `tests/test_rocm_decode_profile.py` pins them on synthetic steps.

| Role | Rule | Built by |
|---|---|---|
| `sampler_tail` | everything after the step's widest `qmv` (the lm_head, vocab rows) up to the next step's embedding gather | `sample_token_optimized` and the `--ignore-eos` logit bias |
| `add_rms_join_post_attn` | a `binary_vv<Add>` immediately followed by `rms_norm_kernel`, where the add follows o_proj's `qmv`, which follows `kernel_sdpav_1pass` | `graph_add_rms_norm` (`layers.rs:978`), the fallback of `fused_add_rms_norm` |
| `add_rms_join` | the other add + `rms_norm_kernel` pairs (next layer's input norm) | model code, never routed to the fused kernel |
| `rope_append` | `rope_single*` dispatches and the `copy_gg_byval` K/V cache writes directly after them | the reshape / `fast_rope` / cache-append graph `forward_fused_rope_append` replaces |
| `moe_expert_gemv` | `gather_qmv_wide_kernel` | `gather_qmm` in `SwitchLinear::forward` |
| `moe_activation` | compiled, binary, copy and unary dispatches between the gate/up and down expert GEMVs, except `Divide` (score normalisation, which stays outside the fused kernel) | `compiled_swiglu_activation` (or relu2) and its casts |
| `moe_weighted_sum` | binary, copy and reduce dispatches right after the down GEMV, up to the residual add | `moe_weighted_sum` |
| `moe_gather_indices` | `arange_kernel<unsigned int>` (one per `gather_qmm` call: 3 per layer in both SwitchGLU models) | the overlay's `gather_qmm` |
| `ssm_step` | inside a Mamba2 mixer (between the `qmv` before a `depthwise_conv1d_kernel`, in_proj, and the `qmv` after it, out_proj), everything that is not conv, SiLU or the gated norm | `ssm_step` (`granitemoehybrid.rs:474`, `nemotron_h.rs:672`), the graph `ssm_update_kernel` replaces |
| `ssm_conv`, `ssm_silu`, `ssm_gated_norm` | the conv1d and its state copies; the compiled SiLU kernels; the last `rms_norm_kernel` of the mixer and what follows it | the rest of the mixer, which the SSM kernel does not replace |

Checks on the rules, per decode step (127 in the window): the SSD step comes out at exactly 47 dispatches per Mamba2 layer on granite and 50 on Nemotron (46.6 and 49.6 per generated token), whose `ssm_step` graphs are built by separate but parallel code; the MoE roles give Qwen3 exactly 3 expert GEMVs, 3 aranges and 4 weighted-sum dispatches per layer; the SSM and MoE roles' dispatch counts are whole multiples of their layer counts per step. Router top-k (`block_sort_kernel`, `softmax_kernel`) is deliberately left out of MoE, because `forward_fused_kernel` takes `topk_indices` and `scores` from the caller. No A/B was run: on ROCm none of these paths has a kernel to switch to, so there is nothing to toggle.

Which of those roles each port would actually take over depends on whether mlxcel calls the ported kernel for that model with its shipped settings. `reach()` in the same script encodes that from the source at `929c80ab`:

- **#2063** (`fused_add_rms_norm`, `fused_rope_qk_append`): both paths ship off on every backend (`FUSED_ADD_RMSNORM_DEFAULT` and `FUSED_ROPE_APPEND_DEFAULT` are `false`, `layers.rs:808` and `:820`, after #905 measured no decode win on Metal), and only `llama3.rs` (with `gemma.rs` and `iquestloopcoder.rs`, not profiled) calls them. With the opt-in on, Llama 3.1 would reach only the post-attention join (`llama3.rs:1201`): its `rope_scaling` builds a frequency table, which routes around the RoPE kernel (`llama3.rs:669`).
- **#2064** (`gumbel_max_sample`, `rejection_sample`): greedy argmax calls neither. A sampled run would reach the whole sampler tail (both kernels are on by default where ported) except the few logit-bias dispatches `--ignore-eos` adds; the table counts the whole tail, so its #2064 figures are slightly high.
- **#2065** (`moe_gateup`, `moe_down`): reached by `qwen3_moe` (`qwen3_moe.rs:223`, single-token decode). Not by granite, whose `block_sparse_moe` calls `SwitchGLU::forward` and never `forward_fused_kernel`, and not by Nemotron-H, whose `fused_moe_forward` default branch is `gather_qmm`; its kernel branch needs `MLXCEL_FUSED_MOE_RELU2` and `moe_fc1_relu2` (#2069).
- **#2067** (`ssm_update_kernel`): reached by both hybrids, since `ssm_kernel_available()` gates every single-token SSD step (`granitemoehybrid.rs:440`, `nemotron_h.rs:537` and `:635`).
- **#2068** (paged attention): not on this path at all. The bench decodes one sequence into a dense `KVCache`; no paged kernel or paged graph fallback appears in any trace.

## Share per port unit

Share of decode GPU time the port would replace with shipped settings; in brackets, the cost of the fallback ops the port family covers whether or not mlxcel calls the port for that model. Dispatches per token replaced in the last column group.

| Model, run | #2063 | #2064 | #2065 | #2067 | #2068 | Dispatches/token reached |
|---|---:|---:|---:|---:|---:|---|
| Llama 3.1 8B, greedy | 0 (1.66) | 0 (0.08) | 0 | 0 | 0 | none |
| Llama 3.1 8B, t0.7 | 0 (1.65) | 0.37 | 0 | 0 | 0 | #2064: 32 |
| Llama 3.1 8B, t0.7 p0.95 | 0 (1.92) | 1.17 | 0 | 0 | 0 | #2064: 64 |
| Qwen3-30B-A3B, greedy | 0 (2.82) | 0 (0.17) | **46.79** | 0 | 0 | #2065: 524 |
| Qwen3-30B-A3B, t0.7 | 0 (2.77) | 0.69 | 46.52 | 0 | 0 | #2065: 524, #2064: 32 |
| Qwen3-30B-A3B, t0.7 p0.95 | 0 (2.67) | 2.75 | 46.03 | 0 | 0 | #2065: 524, #2064: 63 |
| granite-4.0-h-tiny, greedy | 0 | 0 (0.20) | 0 (25.42) | **29.82** | 0 | #2067: 1679 |
| granite-4.0-h-tiny, t0.7 | 0 | 0.72 | 0 (22.51) | 30.75 | 0 | #2067: 1679, #2064: 33 |
| granite-4.0-h-tiny, t0.7 p0.95 | 0 | 2.51 | 0 (21.62) | 31.28 | 0 | #2067: 1679, #2064: 65 |
| Nemotron-3-Nano, greedy | 0 (0.26) | 0 (0.17) | 0 (29.22) | **20.01** | 0 | #2067: 1141 |
| Nemotron-3-Nano, t0.7 | 0 (0.26) | 0.68 | 0 (27.48) | 20.28 | 0 | #2067: 1141, #2064: 32 |
| Nemotron-3-Nano, t0.7 p0.95 | 0 (0.25) | 3.81 | 0 (25.99) | 21.50 | 0 | #2067: 1141, #2064: 64 |

A share is an upper bound on what a port saves, since the ported kernel costs something too. How much of each share is recoverable differs sharply:

- **#2067** replaces about 47 dispatches per layer, mostly 1 to 2 us f32 elementwise kernels, with one kernel whose memory traffic is the SSM state: 48 heads x 64 x 128 x 4 bytes, read and written, is 3.1 MB per granite layer, 113 MB per token, about 0.6 ms at the GEMVs' 180 GB/s, against the 3.42 ms per token the graph takes now (2.9 ms on Nemotron). It also removes 1679 and 1141 dispatches per token, more than half of each model's dispatches, which is where the hybrids' 5 ms per token host gap comes from.
- **#2065** covers 46.8% of Qwen3's GPU time, but 42.1 points of it are the expert GEMVs, already running at 181 GB/s. A fused kernel still reads the same weights, so what it can recover is the rest (activation, weighted sum, index building: 4.7%, 0.62 ms per token) plus about 430 of the 524 dispatches per token. That is less than on Metal and CUDA, where `gather_qmm` left the GPU idle (`switch_layers.rs`, `FUSED_MOE_MAX_DFF_METAL`). Granite and Nemotron carry 25 to 29% of fallback MoE cost that this port as scoped does not reach; wiring granite's `block_sparse_moe` through `forward_fused_kernel` would.
- **#2064** is zero in greedy decode, 0.4 to 0.7% with temperature alone, and 1.2 to 3.8% with top-p, where a full-vocabulary sort (`rocprim` radix sort, `block_sort_kernel`) and scans run every token; the top-p runs also add 0.7 to 1.4 ms per token of host gap over the temperature-only runs (profiled).
- **#2063** is zero with shipped settings on every backend. Turned on, the most it could reach here is Llama's post-attention join, 0.83% of decode (0.89% in the top-p run).
- **#2068** has nothing to act on in single-stream decode. Its value is the batched paged serving path, which this profile does not measure, and the 36 paged-attention test skips on ROCm.

## Ranked port order

By share of decode GPU time reached with shipped settings, weighed by how much of it a kernel can recover:

1. **#2067 SSM update** (#1814 item 7): 29.8% of granite and 20.0% of Nemotron decode GPU time, more than half of their dispatches, and mostly recoverable.
2. **#2065 fused MoE decode** (item 5): 46.8% of Qwen3 decode GPU time reached, but bandwidth-bound GEMVs are most of it; about 5% plus 430 dispatches per token recoverable. Worth more if granite's MoE is wired to it.
3. **#2064 samplers** (item 4): 0.4 to 3.8% of decode GPU time plus 32 to 64 dispatches per token in sampled decode, zero in greedy.
4. **#2068 paged attention** (item 8): no share in single-stream decode; ahead of #2063 only because it has a path (batched serving) and 36 skipped tests that this profile does not measure, not because of a measured share.
5. **#2063 fused add-RMSNorm and RoPE-append** (item 3): zero with shipped settings on every backend; at most 0.83 to 0.89% of Llama decode if opted in.

This reverses #1814's order for items 3 and 7: implied order 7, 5, 4, 8, 3.

## MLX's ROCm `gather_mm`

`src/lib/mlxcel-core/src/grouped_gemm_numeric_tests.rs` gated its three tests on Metal or CUDA, so on ROCm they returned before touching the GPU. With the gates switched to `crate::gpu_backend_available()`, each was run by exact name on gfx1151 (`cargo test --release --features rocm -p mlxcel-core --lib grouped_gemm_numeric_tests::<name> -- --exact --test-threads=1 --nocapture`):

| Test | Result | Kernels in its rocprofv3 trace |
|---|---|---|
| `gather_mm_matches_dense_per_expert_reference` | pass | `gather_batched_gemm_kernel<float, false, true>` x4, `<float, false, false>` x1, a Tensile `Cijk_*` GEMM x4 (hipBLASLt, the sorted single-row case) |
| `gather_mm_selects_the_indexed_expert` | pass | `gather_batched_gemm_kernel<float, false, false>` |
| `gather_mm_half_precision_matches_reference` | pass | `gather_batched_gemm_kernel<hip_bfloat16, ...>`, `gather_batched_gemm_kernel<__half, ...>` |

So the overlay implements `GatherMM` itself (`GatherMM::eval_gpu`, `matmul.cpp`) and its results match the f64 host reference within the tests' tolerances. The tests do discriminate: with the reference deliberately pointed at the wrong expert, all three failed at the value assertion (`grouped_gemm_numeric_tests.rs:111`). The three gates now read `gpu_backend_available()` (Metal and CUDA still run them), the file left `BACKEND_ENUMERATION_TODO`, and `python3 scripts/ci/check_kernel_port_dispatch.py` reports `0 awaiting a predicate`.

## Reproducing

```bash
cargo build --release --features rocm --bin mlxcel-bench-decode
M="models/mlx/Meta-Llama-3.1-8B-Instruct-4bit models/mlx/Qwen3-30B-A3B-4bit models/mlx/granite-4.0-h-tiny-4bit models/mlx/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"
scripts/rocm_decode_profile.sh $M
scripts/rocm_decode_profile.sh --no-plain --temperature 0.7 $M
scripts/rocm_decode_profile.sh --no-plain --temperature 0.7 --top-p 0.95 $M
python3 scripts/rocm_decode_profile.py report benchmarks/rocm_profiles/gfx1151_<commit>
```

`docs/benchmarks.md` ("ROCm per-kernel decode profile") describes the options.
