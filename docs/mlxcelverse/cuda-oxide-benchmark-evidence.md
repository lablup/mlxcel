# cuda-oxide fused residual-add + RMSNorm evidence: benchmark and adoption decision on GB10

Evidence for [issue #2834](https://github.com/lablup/mlxcel/issues/2834), the fourth and last gated experiment in [the cuda-oxide feasibility review](cuda-oxide-feasibility.md). Host: `spark-101`, a DGX Spark that also hosts a self-hosted CI runner; every GPU command ran under `gpu-lock`.

Status: protocol frozen, no measurement collected yet. This file was committed with the Protocol section alone before any timing, logit trace or profile was taken. Results, the gate evaluation and the recommended disposition are added below it by later commits; `git log --follow` on this file shows the order.

## Protocol (frozen before measurement)

Frozen on 2026-11-27 at mlxcel `b9894a8319f70c9a92ac671f9a507d0bb09e0467` plus the harness in this change. A later change to anything in this section is listed under [Protocol adjustments](#protocol-adjustments) with its date and reason, and dated before the measurement it affects.

### Question and scope

Does the cuda-oxide implementation of the fused residual-add + RMSNorm kernel (#2832 artifact, #2833 adapter) give a benefit over the existing fused CUDA kernel (the NVRTC port in `kernels/fused_norm.cpp`) that justifies expanding cuda-oxide kernels inside the MLX path, and if a standalone benefit exists, does MLX execution keep it?

Nothing is enabled by default by this change. No other operator is ported and no independent engine is built. A result for this kernel is not evidence for attention, quantized GEMM or whole-engine performance.

### Arms

One release binary serves every model arm: built with `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121 cargo build --release --features cuda`. Arms differ only by process environment, read once per process, so one process is one arm.

| Arm | Environment | Implementation of the residual join | Role |
| --- | --- | --- | --- |
| `graph` | `MLXCEL_FUSED_ADD_RMSNORM` unset | `add` + `fast_rms_norm` (two MLX kernels), the CUDA default today | Context, and second numerical reference |
| `fused` | `MLXCEL_FUSED_ADD_RMSNORM=1` | Existing fused CUDA kernel, NVRTC-compiled, specialized per width | **Baseline for every gate, and the correctness oracle** |
| `oxide` | `MLXCEL_FUSED_ADD_RMSNORM=1 MLXCEL_CUDA_OXIDE_FUSED_NORM=1` | cuda-oxide node, precompiled PTX, one module for every width | Candidate |
| `fused-null` | same as `fused` | same as `fused` | Null arm: the method's noise floor |

Route validity. Every model run and every op-level block records the adapter's route counters (`oxide_routes`, `native_routes`, `launches`) over exactly the measured calls. The signature an arm must show: `graph` 0/0/0; `fused` and `fused-null` `native_routes > 0`, `oxide_routes = 0`; `oxide` `oxide_routes > 0`, `launches > 0`, `native_routes = 0`. A run with any other signature is invalid: it is discarded, reported as discarded, and rerun. It is never a data point. The model runs are started with `--expect-route`, which makes a mismatch a process failure.

### Artifact

`spike/cuda-oxide-fused-norm/artifacts/fused_add_rms_norm.sm_121.ptx`, SHA-256 `b54c1c93c6e8f03e3f3d2cf37a79dd412069153c1360b6161831583c01eaf86e` (cubin `677d6d6d3a210161ad649681d783c47bc41e66526d263a9d586538ab0e18825f`), compiler revision `b631530700a8a4a5f165456f2d71fe9195b323b5`, ABI version 1, as committed by #2832 and embedded by #2833. It is not rebuilt. The cuda-oxide compiler is not installed on this host, so the #2832 harness cannot run here; its committed `results/bench-sm_121.json` (host `spark-d9e2`) is a cross-host reference only, not a comparison arm.

### Checkpoint

`llama-3.2-1b-instruct-bf16` (`models/llama-3.2-1b-instruct-bf16`, the mlx-community bf16 conversion of Llama 3.2 1B Instruct): `model_type` llama, 16 layers, hidden size 2048, 32 heads, 8 KV heads, vocabulary 128256, `rms_norm_eps` 1e-5, bf16 weights. It is the smallest dense checkpoint on the host that reaches the kernel, so the residual join has the largest share of a step that any supported checkpoint gives it. Each layer has one fused join (the post-attention join in `src/models/llama3.rs`, `forward` and `forward_batched`), so a forward builds 16.

| File | SHA-256 |
| --- | --- |
| `config.json` | `8d40f9098d80a1510565233591a58ccf4dadba6a94934c1a514bf95201621efa` |
| `model.safetensors` | `f5dc593b89368a9a44a8fc3b2dacd7a3d65e43acb4ffbd98a1a40fac9c2bd6da` |
| `tokenizer.json` | `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4` |

The dtype pair that reaches the kernel is bf16 activations with a bf16 weight. A second, larger checkpoint is optional and is run only if the budget allows; if it is, it uses the same protocol and is reported separately.

### Prompts and generation settings

End-to-end prompts are synthesized by `mlxcel::server::engine_probe::prompt::synthesize_prompt_tokens`, the function `mlxcel-bench-engine` uses: the fixed text `LONG_PROMPT_CORPUS` in `src/server/engine_probe/prompt.rs` (at the commit above) repeated, tokenized once with the checkpoint's tokenizer with BOS, and truncated to the target length. Two lengths: **128 tokens (short) and 2048 tokens (long)**. No chat template. In a batch, row `i` uses the same token stream started `17 * i` tokens later with the BOS kept, so rows differ.

Sampling is `neutral_params` (temperature 0, no penalties, no filters): greedy. 128 generated tokens per measured pass, preceded in the same process by a 16-token warmup pass on the same prompt. End-of-generation tokens are not suppressed, because suppression moves a server request onto a different sampler path. Decode throughput follows `docs/benchmarks.md`: the first token comes out of prefill, so a request of N tokens contributes N-1 to the decode window. For a batch the window runs from the earliest first token to the last completion and the numerator is the sum over rows.

The logit trace reads `tests/fixtures/bench/fused-moe-dff-prompt-439-words.txt`, a text already committed.

### Measurements

Every GPU command runs as `gpu-lock run --tag issue-2834-<purpose> -- <command>`. `MLX_PTX_CACHE_DIR` points at a scratch directory for every arm, warmed once before the timed runs (except in the cold-start measurement, which controls the cache state itself). The user's cache is not touched.

**M1. End-to-end, production mode (CUDA graphs on, the default).** `examples/oxide_fused_norm_checkpoint_bench`, driven by `scripts/oxide_fused_norm_rounds.py`. It calls the same two engine probes as `mlxcel-bench-engine` (`bench_like_bench_decode` for the CLI path, the in-process `ServerEngine` with the real `BatchScheduler` for the server path) and adds what that tool lacks: per-arm environment, route counters per pass, MLX peak memory per pass, generated token ids, and concurrent requests. Cells: CLI B=1, server B=1, server B=4 (four concurrent requests through the scheduler, which decodes them with `forward_batched`), each at both prompt lengths: six cells. **7 rounds.** In a round each cell runs the four arms as four fresh processes, in an order that rotates by one position per round. Metrics per pass: time to first token (`ttft_ms`, the prefill), decode tok/s, MLX peak memory (`mlxcel_core::memory::peak_memory`, reset before the measured pass; never RSS or VmHWM, which miss CUDA unified allocations on this host), route counters, token ids. Reported: per-arm median and range, and the per-round paired relative delta of each arm against `fused` in the same round.

**M2. Correctness on the checkpoint.** (a) Teacher-forced per-position logits with `examples/logit_trace` (top-8 ids and logits at six decimals, and the target NLL, for every position), one process per arm, in three shapes: one full-width forward of 256 tokens (the prefill shape: `MODEL TEXT 256 1 8`), 112 width-one decode steps with `MLXCEL_TRACE_BATCH_ROWS=1`, and 112 width-one batched decode steps over four rows with `MLXCEL_TRACE_BATCH_ROWS=4` (`MODEL TEXT 112 1 8`; the fixture is 480 tokens, which is what four independently prefixed rows of 112 fit in). Each trace process prints its route counters. Compared with `scripts/compare_logit_traces.py` and byte for byte. (b) The greedy token ids of every M1 pass, compared between arms per cell and round. (c) Intermediate tensors: the #2832 130-case matrix through the adapter (`the_2832_matrix_passes_through_the_adapter_on_the_oxide_route`) is rerun on this commit, which compares the kernel's two outputs, the only tensors the two implementations compute differently, bit for bit at the op boundary. If (a) is not byte-identical between `oxide` and `fused`, the first diverging position is reported and the residual stream is compared layer by layer before any conclusion.

**M3. Kernel time, standalone, CUDA events.** `spike/cuda-oxide-fused-norm/scripts/standalone-bench.sh`: a driver-API program (no MLX, no cuda-oxide toolchain) that checks the PTX digest against the five manifests, loads the PTX with `cuModuleLoadData`, and launches each entry with the manifest ABI and launch rule (`grid = rows`, `block = clamp(ceil_div(ceil_div(dim, 4), 32) * 32, 32, 1024)`). Timing definitions follow #2832 so its JSON is comparable: `first_launch` = first launch at the shape, `warm_mean` = CUDA events around 200 back-to-back launches after 20 warm-ups, divided by 200. It adds: per-launch synchronized event samples (median, p90), the host wall time of the launch call alone, and of launch plus stream synchronize. Module load is host wall time around `cuModuleLoadData`. Shapes: the model's five (bf16/bf16 at 1, 4, 128, 512 and 2048 rows of width 2048: decode at B=1 and B=4, prefill of 128 tokens at B=1 and B=4, prefill of 2048 tokens at B=1) and the 18 of #2832's benchmark. **5 process repetitions**; the median over processes is reported with the range.

**M4. Kernel time through MLX and standalone, same profiler.** `nsys profile --trace=cuda --sample=none --cpuctxsw=none` on (a) the standalone program and (b) `mlxcel-core`'s `examples/oxide_fused_norm_bench --mode iso` with `MLX_USE_CUDA_GRAPHS=0`, arms `graph`, `fused`, `oxide` in one process, on the model's five shapes. Kernel durations are read from the profiler's kernel activity table per kernel name and launch geometry (`scripts/nsys_kernel_table.py`): count, median, mean, p90. **3 repetitions.** `MLX_USE_CUDA_GRAPHS=0` is needed because kernels inside CUDA graphs do not appear in that table on this setup; it is not the production mode, so M4 gives kernel durations and launch-API costs only, and no end-to-end number is taken from it.

**M5. Op level through MLX, production mode.** `oxide_fused_norm_bench --mode iso` (build, evaluate both outputs, synchronize, per iteration) and `--mode pipe` (build and `async_eval` every iteration, one synchronize at the end, which is what a decode loop does), graphs on, arms `graph`, `fused`, `oxide` interleaved in rotating order, 5 rounds of 200 iterations after 20 warm-ups, the model's five shapes plus #2832's 18. **3 process repetitions.**

MLX per-op overhead is reported only as a matched difference: for the same artifact, shape and launch geometry, (M5 `iso` wall time per op of the `oxide` arm) minus (M3 wall time of launch plus stream synchronize), both host wall clocks over one launch and one wait, with M4 confirming that the kernel duration is the same on both sides. No overhead is derived from a kernel-only number against an end-to-end number.

**M6. Cold start.** Fresh process per sample, `oxide_fused_norm_bench --mode cold`, one arm per process, shapes bf16/bf16 1 x 2048 then 1 x 4096 (a second width, because the existing kernel is compiled per width and the cuda-oxide module is not). The quantity is **first-use cost = (wall time of the first build + evaluate + synchronize of the op, including for `oxide` the artifact validation and installation that the first fused join performs) minus (steady-state median of the same call)**, taken after the device context exists and an unrelated kernel has run. States: `fused` with an empty `MLX_PTX_CACHE_DIR` (NVRTC compile), `fused` with that directory warm, `oxide` (its module bypasses the disk cache, so one state), `graph`. **5 samples per state.** The standalone module-load time from M3 is reported next to it.

**M7. Static and profile data.** Registers, stack frame (spill) size, shared memory and instruction counts with one tool on both cubins: `cuobjdump -res-usage` and `cuobjdump -sass` on the committed cuda-oxide cubin and on the existing kernel's entries in the PTX disk cache (those cache files are cubins), for the bf16 width-2048 kernel and the other widths measured. Memory traffic: bytes per launch counted analytically as in #2832 (`x` and `residual` read, `new_residual` written and re-read, `normed` written, the weight read once per row) and divided by the measured kernel duration. Launch and synchronization: CUDA API call counts and durations per op from M4's trace, and M3's launch-call and launch-plus-synchronize times. Hardware counters with `ncu` are collected if this user is allowed to; if not, the refusal is recorded and no counter value is reported.

Recorded with every run set: kernel, driver, GPU, CUDA toolkit, `MLXCEL_CUDA_ARCHITECTURES`, git commit, MLX pin, build profile and environment, PTX and checkpoint digests, CPU governor and GPU clocks if visible, and whether a CI job (`Runner.Worker`) was running on the host. A batch does not start while one is; nothing is pushed during a batch.

### Gates

Baselines: the 2 % end-to-end gate and the 10 % kernel gate compare `oxide` against `fused` (the same fusion, another implementation). `graph` is reported next to them as context and decides nothing.

**G1. Correctness.** Passes when all hold: every run shows its arm's route signature; M2(a) top-1 agreement between `oxide` and `fused` is complete on decided positions (reference top-two gap of at least 2) in all three shapes; M2(b) greedy token ids are identical between `oxide` and `fused` in every cell and round where `fused` and `fused-null` are identical to each other; M2(c) passes 130 of 130. Byte identity of the traces is the expectation and is reported; like #2832, the gate is the budget, so that a nondeterministic baseline cannot fail it by itself. `graph` against `fused` is reported as context.

**G2. No unexplained end-to-end regression above 2 % outside noise.** Per cell and per metric (`ttft_ms`, decode tok/s), from M1's per-round paired deltas, signed so that positive is better. The cell's null floor is the median absolute paired delta of `fused-null` against `fused`. A cell and metric is **flagged** when the median paired delta of `oxide` against `fused` is worse than -2 %, and it is worse in at least 6 of the 7 rounds, and its magnitude exceeds the null floor. A cell whose null floor is itself above 2 % cannot resolve the threshold and is reported as **unresolved**, neither pass nor fail. G2 passes when no cell is flagged, or every flagged cell is explained by a profile that names the cost; the unresolved cells are listed with the conclusion. Peak memory is not gated at a threshold of its own: the per-cell medians are reported for every arm, and a difference between `oxide` and `fused` above 1 % is listed as a finding to explain.

**G3. A benefit, at least one of:**

- (a) *Kernel.* The `oxide` kernel's median duration in M4 (through MLX) is at least 10 % below the `fused` kernel's on both decode shapes (1 and 4 rows) or on all three prefill shapes (128, 512, 2048 rows), and M1 shows a useful end-to-end effect: at least one cell and metric where `oxide` is better than `fused` by more than the null floor in at least 6 of 7 rounds.
- (b) *Cold start.* The `oxide` first-use cost in M6 is below the `fused` first-use cost with a warm disk cache, which is the state every process after the first is in. If it is below only the empty-cache cost, (b) is recorded as met for the first run per machine and width only, and the disposition has to weigh that explicitly instead of counting it as a general reduction.
- (c) *Maintenance.* A concrete reduction in what the tree has to carry or what a user has to run (source, runtime dependencies, per-width compilation, cache trust surface), net of the compiler toolchain, artifact and ABI burden the route adds. Judged from the tree and stated with its counterweights; not a measurement.

### Disposition rule

Exactly one, following the feasibility review's table.

| Observed | Disposition |
| --- | --- |
| G1 and G2 pass and G3 is met through MLX | Expand selected kernels within MLX |
| G1 passes, the standalone kernel is at least 10 % faster than the existing kernel on the matched shapes, and the matched M3/M4/M5 comparison shows a reproducible MLX execution cost that removes the benefit | Investigate that execution constraint |
| The compiler, the generated code or GPU support limits the result (G1 fails for a toolchain reason, or the artifact cannot be loaded or validated) | Resolve the toolchain limitation |
| G1 passes and G3 is not met | Stop: no expansion of cuda-oxide kernels on this evidence |

The disposition written here is the experiment's result and a recommendation. The maintainer makes the adoption call at merge.

### Checks made before the freeze

Functional checks only, run to learn whether the instruments work. No timing from them was recorded or used, and none appears in the results.

- The standalone program loaded the committed PTX after the digest check and wrote finite, non-zero outputs on all 23 shapes (the f32 residual sum checked exactly on the host).
- `oxide_fused_norm_bench` showed the expected counters per arm in `iso`, `pipe` and `cold` mode: `graph` 0/0/0, `fused` `native_routes` = calls, `oxide` `oxide_routes` = `launches` = calls.
- With `MLX_USE_CUDA_GRAPHS=0`, `nsys --trace=cuda` lists both kernels by name with their geometry: `custom_kernel_mlxcel_fused_add_rms_norm_bfloat16_t_bfloat16_t_2048_512` (grid `1x1xrows`, block 512) and `fused_add_rms_norm_bf16` (grid `rowsx1x1`, block 512), and the graph path's `binary_vv` and `rms_norm_small`.
- `ncu` cannot collect counters as this user: `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0`. No hardware-counter value will be reported; M7 uses the static and analytic figures.
- On the checkpoint the `oxide` arm takes the cuda-oxide route for every join at B=1 (CLI) and at B=4 (server): `native_routes = 0`, `launches = oxide_routes`, so bf16/bf16 is the pair that reaches the kernel in `forward` and in `forward_batched`.
- The `fused` arm is deterministic run to run here: two server B=4 runs gave the same token ids at both prompt lengths, and two prefill-shape logit traces were byte-identical. Byte identity between `oxide` and `fused` is therefore a meaningful expectation.
- The short prompt reaches an end-of-generation token before 128 tokens on the CLI path (98 tokens in the probe). The number of generated tokens is recorded per pass and the rates are per token, so the cell stays in; this is stated with the results.
- `cuobjdump -res-usage` reads the existing kernel's disk-cache entry (an ELF cubin despite its `.ptx` name).

### Protocol adjustments

No gate, threshold, baseline, arm, checkpoint, prompt or repetition count changed. Three measurements were added on 2026-11-27 after the first result set (commit `37ac8922`) and before any of them ran; the reasons come from that set.

1. **Driver compute cache (extends M6 and M3).** The standalone module load in M3 took 0.26 to 0.51 ms, against 255 ms in #2832's run. The CUDA driver keeps its own cache of PTX it has compiled (`~/.nv/ComputeCache`), and this host had loaded the module before, so M6's `oxide` state measured a driver-cache hit. A machine pays the PTX compilation the first time it loads the module and again after a driver update. Added: M6 with `CUDA_CACHE_DISABLE=1` for `oxide`, `fused` with an empty disk cache and `fused` with a warm one (the existing kernel's cache entries are cubins, which the driver does not compile, so those two are controls), and the standalone program with `CUDA_CACHE_DISABLE=1`, 5 samples each. For G3(b), the comparison in the normal state stays `oxide` against `fused` with a warm disk cache, both with the driver cache as the host has it; the first-run comparison is `oxide` with the driver cache off against `fused` with an empty disk cache.
2. **M4 and M5 repeated.** A CI job was running on the host when each of those two stages ended (`host-state.csv`). Both are run a second time into `*-repeat.csv`. Both sets are kept and reported; neither is dropped.
3. **CUDA API calls per arm (exploratory, new M8).** M5's `pipe` mode shows the `oxide` arm about 1.8 µs per op below `fused` at the small shapes while M4's kernel durations differ by less than 0.1 µs at one row, so the difference is on the host side. M4's API table cannot say which arm made which call because its three arms share a process. Added: `nsys` on `oxide_fused_norm_bench --mode pipe`, one arm per process, 2020 ops at bf16 1 x 2048, with CUDA graphs off and on, API calls counted per arm. It is a profile to attribute an observed difference and decides no gate.

One reporting note, also not a threshold: M5's `iso` blocks turned out bimodal on this shared host (block medians of about 30 to 50 µs or about 200 to 400 µs for the same arm and shape, depending on how long the waiting thread took to wake). The `iso` table therefore reports the minimum block median next to the median, and `pipe`, which has one wait per 200 ops, is the op-level figure the text relies on.
