# cuda-oxide fused residual-add + RMSNorm evidence: benchmark and adoption decision on GB10

Evidence for [issue #2834](https://github.com/lablup/mlxcel/issues/2834), the fourth and last gated experiment in [the cuda-oxide feasibility review](cuda-oxide-feasibility.md). Host: `spark-101`, a DGX Spark that also hosts a self-hosted CI runner; every GPU command ran under `gpu-lock`.

Status: measured. The cuda-oxide kernel is correct on the checkpoint (logit traces and greedy output identical to the existing fused kernel, every run on the intended route) and changes neither prefill nor decode throughput, and it brings no benefit that passes the gates: the kernel itself is not faster, the only cold-start gain is a one-time one per machine and width, and it adds a toolchain and an artifact contract without removing anything. **Recorded result and recommendation: stop. Do not expand cuda-oxide kernels on this evidence.** Maintainer decision (2026-11-28): stop is adopted for the default, and the #2833 adapter stays in the tree as an opt-in route, off by default at build time and at run time. It is not removed, because a later rewrite of the kernels on cuda-oxide remains possible and the adapter is the working reference for the artifact boundary. Nothing is enabled by default and no default-build behavior changes.

Order of work: this file was committed with the Protocol section alone (`93182387`) before any timing, logit trace or profile was taken; the first raw results are `37ac8922`; four additions to the protocol were each committed before they ran (`bc1e47ff`, `70fa4f9a`). `git log --follow` on this file shows it.

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

No gate, threshold, baseline, arm, checkpoint, prompt or repetition count changed. Four measurements were added on 2026-11-27 after the first result set (commit `37ac8922`), each committed before it ran; the reasons come from the results before it.

1. **Driver compute cache (extends M6 and M3).** The standalone module load in M3 took 0.26 to 0.51 ms, against 255 ms in #2832's run. The CUDA driver keeps its own cache of PTX it has compiled (`~/.nv/ComputeCache`), and this host had loaded the module before, so M6's `oxide` state measured a driver-cache hit. A machine pays the PTX compilation the first time it loads the module and again after a driver update. Added: M6 with `CUDA_CACHE_DISABLE=1` for `oxide`, `fused` with an empty disk cache and `fused` with a warm one (the existing kernel's cache entries are cubins, which the driver does not compile, so those two are controls), and the standalone program with `CUDA_CACHE_DISABLE=1`, 5 samples each. For G3(b), the comparison in the normal state stays `oxide` against `fused` with a warm disk cache, both with the driver cache as the host has it; the first-run comparison is `oxide` with the driver cache off against `fused` with an empty disk cache.
2. **M4 and M5 repeated.** A CI job was running on the host when each of those two stages ended (`host-state.csv`). Both are run a second time into `*-repeat.csv`. Both sets are kept and reported; neither is dropped.
3. **CUDA API calls per arm (exploratory, new M8).** M5's `pipe` mode shows the `oxide` arm about 1.8 µs per op below `fused` at the small shapes while M4's kernel durations differ by less than 0.1 µs at one row, so the difference is on the host side. M4's API table cannot say which arm made which call because its three arms share a process. Added: `nsys` on `oxide_fused_norm_bench --mode pipe`, one arm per process, 2020 ops at bf16 1 x 2048, with CUDA graphs off and on, API calls counted per arm. It is a profile to attribute an observed difference and decides no gate.

4. **Host cost of building the node (exploratory, new M9, added after M8).** M8 found the same CUDA API calls per op for `fused` and `oxide`, so the per-op difference in `pipe` mode is neither device time nor CUDA API time. Added: `oxide_fused_norm_bench --mode build`, which builds the lazy node and drops it without evaluating, so it times the launcher's host work alone (2000 builds per block, 5 rotated rounds, 3 processes, the model's five shapes). The example was rebuilt for this mode; the end-to-end binary is unchanged.

5. **M1 process granularity (a deviation from the frozen M1 text, found in review).** The M1 text above says each cell runs each arm as a fresh process, one per cell. The driver (`scripts/oxide_fused_norm_rounds.py`) instead starts one process per path, batch and arm and runs both prompt lengths (128 then 2048) inside it, so the 12 passes of a round come from 24 processes, not 48 (`--prompt-tokens 128 2048` is its default). The second prompt length therefore runs after the first in the same process, with a warm allocator and warm kernels. It applies to every arm alike and the paired deltas are taken within it. It was not decided after seeing data: the driver was committed in `0a5968fa` (2026-11-27 12:06 KST, 03:06 UTC) with that default, before the first stage began (03:37 UTC) and before the M1 run (03:56 UTC), and it was not changed afterwards. The frozen text was not corrected at the time, which this entry does.

One correction to the host-state record, not to a measurement: the check for a running CI job used `pgrep -f`, which also matches any shell whose command line mentions the runner's process name. The `true` recorded at the end of the repeated M4 stage (04:08 UTC) is such a false positive, caused by a monitoring loop, and so is the `true` at the end of the first M4 stage (03:50 UTC): the runner's job logs show no job between 03:48:54 and 03:53:12, and none after 03:54:40. The `true` at the end of the first M5 stage (03:54:11 UTC) is real: a job ran from 03:53:12 to 03:54:40. A `false` cannot be produced by this mistake, so every `false` stands. The scripts now use `pgrep -x`. The Host state section gives the job windows.

One reporting note, also not a threshold: M5's `iso` blocks turned out bimodal on this shared host (block medians of about 30 to 50 µs or about 200 to 400 µs for the same arm and shape, depending on how long the waiting thread took to wake). The `iso` table therefore reports the minimum block median next to the median, and `pipe`, which has one wait per 200 ops, is the op-level figure the text relies on.

## Environment

| Component | Value |
| --- | --- |
| Host | `spark-101`, Linux 7.0.0-1019-nvidia, aarch64, up 22 days, CPU governor `performance` |
| GPU | NVIDIA GB10, compute capability 12.1, driver 580.178.04, SM clock 2418 MHz at capture (max 3003) |
| CUDA Toolkit | 13.0, V13.0.88 (`/usr/local/cuda`); Nsight Systems 2025.3.2 |
| Rust | 1.97.1; no cuda-oxide compiler on this host |
| MLX | pin `fc344c986c4169d99f76b5d95ae6b4a58a359a63` |
| mlxcel | Binaries built from `b45ce7acdbc5e1c089dc66f5f3a23194bb94abd1` with a clean tree (`provenance.json`). The commits after it up to M8 changed scripts and documents only. M9 used the op-level example rebuilt at `70fa4f9a`, which added its `build` mode; the end-to-end binary was never rebuilt. |
| Build | `--release --features cuda`, `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121`; one binary for all model arms |
| `gpu-lock` | `/usr/local/bin/gpu-lock`, `run --tag issue-2834-<purpose> -- <cmd>` around every GPU command; builds outside the lock |

`docs/benchmark_results/cuda-oxide-fused-norm-gb10-2026-11-27/provenance.json` has the binary, artifact and checkpoint digests as captured. The `logit_trace` binary (M2) was not hashed at capture; its digest is recorded in the same file under `binaries_sha256_recorded_after_capture`, recomputed from the on-disk binary, whose modification time precedes every stage and which no build has replaced since. That is evidence about the file, not a capture-time record. The `oxide_fused_norm_bench` digest is that of the build before the M9 rebuild.

Host state. This host is a CI runner and was shared with other work during the day, so it was never idle in the strict sense. The first attempt at the batch did not start: main-branch CI jobs occupied the runner back to back for more than 30 minutes. Each stage records its start, end, load average and whether a CI job was running at its end in `host-state.csv`. The runner's own job logs (`~/actions-runner/_diag/Worker_*.log`) give the job windows, in UTC: jobs ran back to back, with gaps of 3 to 17 seconds, from 02:51:40 to 03:48:54, then one more from 03:53:12 to 03:54:40, and none after that until the measurements ended at 04:14. Against `host-state.csv`: M3 ran from about 03:49:00 after the last job of the long series ended; the first M4 set (03:49:42 to 03:50:17) ran with no job active; the last minute of the first M5 set (ended 03:54:11) overlapped the 03:53:12 job, an OpenXLA feature compile; M6, M7, M1, M2, both repeats, M6b, M8 and M9 ran with no job active. So one stage, the first M5 set, was partly overlapped. M4 and M5 were both repeated because the per-stage flag said `true` for both at the time; for M4 that flag was a false positive of the check (see Protocol adjustments). Both sets are in the tables and they agree. The end-to-end rounds checked before every process, and none of the 168 passes ran with a job active. Load average was between 1 and 4 during the stages; other work by the same user was running on the host throughout.

## Commands

Everything is in [`docs/benchmark_results/cuda-oxide-fused-norm-gb10-2026-11-27/run.sh`](../benchmark_results/cuda-oxide-fused-norm-gb10-2026-11-27/run.sh), one function per measurement, with the CSVs it wrote next to it.

```bash
MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121 cargo build --release --features cuda \
    --example oxide_fused_norm_checkpoint_bench --example logit_trace
MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121 cargo build --release -p mlxcel-core \
    --features cuda --example oxide_fused_norm_bench
R=docs/benchmark_results/cuda-oxide-fused-norm-gb10-2026-11-27
WORK=<private scratch dir> $R/run.sh m3 env m4 m5 m6 m7 m1 m2       # first set
WORK=<same> SET=-repeat $R/run.sh m4 m5 m6b m7 m8                    # repeat and additions 1 to 3
WORK=<same> $R/run.sh m9                                             # addition 4
MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121 cargo test --release -p mlxcel-core \
    --features cuda --lib --no-run                                   # prints the lib test binary
WORK=<same> LIB=<that binary> $R/run.sh m2c                          # M2(c), correctness rerun
```

M2(c) is the `m2c` stage: the 31 `cuda_oxide` tests of the opted-in `mlxcel-core` lib test binary (release profile), run as `gpu-lock run --tag issue-2834-finalize -- $LIB cuda_oxide --test-threads=1 --nocapture`. An earlier run of the same command, during the measurements, was quoted in this report without its output being kept; the stage was rerun at the end of the review (2026-11-27 05:22 UTC, tree at `5939da7f`, whose Rust sources are those the measurements used) and `m2c-adapter-matrix.csv` in the results directory records it, reduced from the test binary's capture by `scripts/oxide_fused_norm_m2c_summarize.py`: the command, the commit, the binary's SHA-256, each test's outcome, every field of the matrix `RESULT` line and the libtest totals (the capture itself is not committed, per the results tree's no-raw-logs rule). It is a correctness rerun and carries no timing.

## Results

### Correctness on the checkpoint (G1)

| Check | Result |
| --- | --- |
| Route signature, 168 end-to-end passes (4 arms, 7 rounds, 6 cells) | All valid. `oxide`: `oxide_routes = launches` (1600 to 2096 per pass), `native_routes = 0`. `fused`, `fused-null`: the same counts in `native_routes`, `oxide_routes = 0`. `graph`: 0/0/0. No run was discarded. |
| Route signature, op-level blocks | 4,365 of 4,365 valid (M5 two sets, M9). |
| M2(a) logit traces, `oxide` against `fused` | Byte-identical files in all three shapes: 256 positions of a 256-token forward, 112 width-one decode steps, 112 width-one batched decode steps over four rows. 0 top-1 disagreements, 0 on decided positions (47, 26 and 28 of them), largest logit delta 0. The logits are bf16 values, which six decimals print exactly. Each trace process printed its route: 16, 1808 and 1856 joins on the cuda-oxide route with as many launches. |
| M2(a) controls | `fused-null` against `fused`: byte-identical. `graph` against `fused`: byte-identical too on these three traces. |
| M2(b) greedy token ids, `oxide` against `fused` | Identical in all 6 cells and all 7 rounds; `fused` and `fused-null` identical to each other in all of them. |
| M2(c) op boundary (31 of 31 `cuda_oxide` tests passed in 74 s; recorded in `m2c-adapter-matrix.csv`) | `RESULT oxide-adapter-matrix cases=130 elements=2181390 failed=0 bitwise_identical_to_fused_cuda=130 nonfinite_cases=5 oxide_routes=130 launches=130 native_routes=0`. |

G1 passes.

One observation outside the gate: on the 128-token prompt at B=1 the `graph` arm generates other tokens than `fused` and `oxide` (128 tokens against 98, where the two fused arms reach an end-of-generation token), on the CLI and the server path alike, although the teacher-forced traces of the three arms are byte-identical on the trace corpus. Each arm is deterministic across the seven rounds. The graph path is not the oracle here and the difference predates this experiment (it is the existing fused kernel against the graph path), so it was not investigated; it is recorded because a reader comparing `graph` and `fused` throughput on that cell is comparing runs of different length.

Issue #3501 investigated it afterwards with `examples/fused_norm_divergence_probe.rs` (results in `docs/benchmark_results/issue-3501-fused-norm-divergence-gb10-2026-11-28/`): a near-tie from the fp32 summation order of the RMSNorm sum of squares, not a defect. The probe reproduces both arms' token digests (`153e97b6c47aae0d`, `ca3dde687d118faf`). Teacher-forced on the graph arm's tokens, the only argmax flip in 128 steps is generated index 11, where the graph's top two (287 and 430) are an exact bf16 tie at 18.5 (margin 0) and the fused arm has 430 at 18.5 over 287 at 18.375 (margin 0.125); the accumulated logit deviation at that step is 0.125 at most (RMS 0.028). The arms first part in the prefill, at layer 1's MLP input (`normed`): 19 of 262,144 elements, one bf16 ulp each, while `new_residual` stays byte-identical. Fed the same inputs, the two joins differ on 2 of the 2048 prefill rows (16 layers of 128), by one or two bf16 ulps, and on none of the step-11 decode rows. The two-ulp output (27 elements on layer 11, 2 of them two ulps apart) exceeds the issue's literal one-ulp budget; the rounded intermediate `bf16(x * inv)` stays within one ulp, and the two-ulp output comes from the gain multiply carrying that change across a binade. The near-tie verdict still holds because a single fp32 normalizer per side reproduces each side's row through the graph's formula, while a float64 reference agrees with the fused kernel on one row and with the graph on the other. Each differing row is reproduced exactly by the graph formula with one fp32 normalizer per side, the two on opposite sides of a rounding boundary, and a float64 reference agrees with the fused kernel on one row and with the graph on the other. The cause is launch geometry: at bf16 dim 2048 MLX runs `rms_norm_small` with a 128-thread block and two 8-wide chunks per thread (four warp partials; the `ALIGNED` argument was not observed, and it selects only the load function, not the summation order) and the fused kernel 512 threads of 4 elements with 16 warp partials, with the same xor-butterfly warp reduction; simulated in fp32 in those two orders, the sums of squares of the two differing rows are one fp32 ulp apart, in the direction the fitted normalizers predict. Making the CUDA kernel byte-identical, as the ROCm port is, would mean copying MLX's per-dim geometry and chunked loads and regenerating the cuda-oxide artifact, which is a bitwise port of the current kernel.

Issue #3510 did that on 2026-11-28, so the observation above no longer holds: on CUDA the fused kernel is byte-identical to the graph path, and the three arms generate the same tokens. The kernel now reads a row as MLX's `rms_norm` does (vectors of `16 / sizeof(compute)` elements, the block width of `dispatch_num_chunks`, thread `t` summing vectors `t`, `t + block`, ... in order), and the cuda-oxide artifact was regenerated with the same geometry (PTX SHA-256 `ed470a19c4848893a680223db9c6861363c4cb180a5c229a63307afc305357f4`, cubin `390fcc2833e532ef5bffb7d8699c2f43a0fd7dc53775079b9b2c6be938a558cd`). One round of the six cells gives one `tokens_sha256_16` per cell across `graph`, `fused` and `oxide`, with B=1 / 128 at `153e97b6c47aae0d` and 128 tokens on all three; the probe's prefill dumps of the graph and fused arms are byte-identical on all 82 tensors, and its same-input op diff reports no differing element on any of the 16 layers. Everything else in this report (the timings, the gates, the artifact digest in the Protocol) describes the kernel and artifact as they were on 2026-11-27. The identity costs kernel time: with MLX's 128-thread block at width 2048 the fused kernel and the cuda-oxide kernel take 1.3 to 2 times as long as on 2026-11-27 (3.4 against 2.2 µs at one row, 12.7 against 6.5 µs at 128 rows, 186 against 140 µs at 2048 rows, same M4 method), while the graph path's kernels are unchanged, so the graph path is now the faster one at all five shapes. That table, and the rows behind this paragraph, are in [`docs/benchmark_results/cuda-fused-norm-byte-identity-gb10-2026-11-28/`](../benchmark_results/cuda-fused-norm-byte-identity-gb10-2026-11-28/README.md).

### End to end, production mode (G2)

`m1-end-to-end.csv`, summarized by `scripts/oxide_fused_norm_rounds.py summarize` into `m1-end-to-end-summary.csv`. Medians over 7 rounds; the paired column is the median of the per-round relative deltas of `oxide` against `fused`, signed so that positive is better; the null floor is the median absolute paired delta of `fused-null` against `fused`.

| Cell | Metric | `graph` | `fused` | `oxide` | `fused-null` | `oxide` vs `fused`, paired median (min..max) | Rounds worse | Null floor | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLI B=1, 128 | TTFT ms | 14.43 | 14.63 | 14.52 | 14.41 | +1.65 % (-3.12..+3.17) | 2/7 | 1.94 % | no regression |
| CLI B=1, 128 | decode tok/s | 88.81 | 87.64 | 88.03 | 87.55 | -0.06 % (-0.89..+1.86) | 4/7 | 0.61 % | no regression |
| CLI B=1, 2048 | TTFT ms | 67.04 | 67.30 | 67.03 | 67.05 | +0.71 % (-1.04..+1.94) | 2/7 | 0.84 % | no regression |
| CLI B=1, 2048 | decode tok/s | 86.51 | 86.79 | 86.53 | 86.75 | -0.31 % (-1.15..+0.29) | 6/7 | 0.28 % | no regression |
| server B=1, 128 | TTFT ms | 15.17 | 15.18 | 15.13 | 15.26 | +0.50 % (-0.64..+2.57) | 3/7 | 1.67 % | no regression |
| server B=1, 128 | decode tok/s | 87.97 | 85.33 | 85.55 | 85.37 | +0.26 % (-0.05..+0.73) | 1/7 | 0.64 % | no regression |
| server B=1, 2048 | TTFT ms | 78.86 | 79.45 | 79.69 | 79.64 | +0.51 % (-2.54..+1.16) | 3/7 | 0.40 % | no regression |
| server B=1, 2048 | decode tok/s | 77.30 | 77.24 | 77.27 | 77.07 | -0.18 % (-1.26..+1.29) | 5/7 | 0.75 % | no regression |
| server B=4, 128 | TTFT ms | 63.95 | 65.89 | 65.59 | 65.95 | +0.45 % (-15.88..+2.45) | 3/7 | 1.89 % | no regression |
| server B=4, 128 | decode tok/s | 264.74 | 262.89 | 269.27 | 267.35 | +0.67 % (-5.92..+5.02) | 3/7 | 1.69 % | no regression |
| server B=4, 2048 | TTFT ms | 289.80 | 291.30 | 290.51 | 289.91 | +1.18 % (-0.72..+2.55) | 1/7 | 2.14 % | unresolved (null floor above 2 %) |
| server B=4, 2048 | decode tok/s | 255.15 | 254.57 | 255.46 | 255.42 | +0.21 % (-0.97..+1.24) | 1/7 | 0.40 % | no regression |

No cell is flagged. The worst median is -0.31 % (CLI B=1, long prompt, decode), worse in 6 of 7 rounds and 0.023 points beyond that cell's null floor: a difference of that size is what the null arm itself shows there (-0.18 %), and it is far inside the 2 % threshold. One cell cannot resolve 2 % (TTFT at B=4 with the long prompt, null floor 2.14 %); its `oxide` median is on the better side. No cell shows an improvement beyond its null floor in 6 of 7 rounds either, so the kernel has no measurable end-to-end effect in either direction. G2 passes, with that one cell unresolved.

Generated tokens per pass: 98 on the short prompt at B=1 for `fused`, `oxide` and `fused-null` (end of generation), 128 everywhere else, 4 x 128 at B=4. The rates are per token.

Peak memory (MLX `peak_memory`, reset before the measured pass), median per cell, `fused` and `oxide`: 2.3615 and 2.3624 GiB (CLI B=1, 128), 2.9972 and 2.9972 (CLI B=1, 2048), 3.3395 and 3.3395 (server B=1, 128), 3.4894 and 3.4894 (server B=1, 2048), 3.6663 and 3.6663 (server B=4, 128), 4.6007 and 4.6007 (server B=4, 2048). The largest difference is 0.04 %. The `graph` arm's peak is 2 to 16 MiB below both.

### Kernel time (G3a)

Kernel durations from one profiler on both sides (M4): `nsys --trace=cuda`, the standalone program, and MLX with `MLX_USE_CUDA_GRAPHS=0`. bf16 activations and weight, width 2048, block 512 for all three fused launches. Median of three capture medians, in µs; the two values are the first set and the repeat, except in the last column, which is the first set only (median of its three captures; the repeat reads 4.10 for `rms_norm_small` at 512 rows and 116.7 for `binary_vv` at 2048 rows). 821 launches per capture and shape standalone, 1100 through MLX. `MLX_USE_CUDA_GRAPHS=0` is not the production mode; these are kernel durations, not throughput.

| Rows | Model step | cuda-oxide standalone, grid `rows x 1 x 1` | cuda-oxide through MLX | Existing kernel through MLX, grid `1 x 1 x rows` | cuda-oxide vs existing | Graph path: `binary_vv` + `rms_norm_small` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | decode, B=1 | 2.14, 2.11 | 2.14, 2.14 | 2.21, 2.18 | -2.9 %, -1.5 % | 1.02 + 1.44 |
| 4 | decode, B=4 | 2.18, 2.18 | 2.21, 2.21 | 2.59, 2.59 | -14.8 %, -14.8 % | 1.47 + 1.44 |
| 128 | prefill 128, B=1 | 6.40, 6.43 | 6.69, 6.72 | 6.53, 6.56 | +2.5 %, +2.4 % | 1.66 + 1.95 |
| 512 | prefill 128, B=4 | 24.61, 24.70 | 24.26, 23.94 | 24.13, 23.97 | +0.5 %, -0.1 % | 4.48 + 4.03 |
| 2048 | prefill 2048, B=1 | 120.77, 108.45 | 139.74, 141.98 | 139.68, 142.14 | +0.0 %, -0.1 % | 114 + 30 |

Three things follow from the table.

The two kernels take the same time. The difference is under 3 % at one row and under 3 % on the three prefill shapes, with either sign. At four rows the cuda-oxide kernel is 0.38 µs shorter (14.8 %) in both sets; the existing kernel launches its rows on the grid's z axis and the cuda-oxide kernel on x, and nothing else about that shape was examined. G3(a) needs at least 10 % on both decode shapes or on all three prefill shapes. It gets 10 % on one of five. **G3(a) is not met**, before the end-to-end condition is even asked (and that one fails too: no cell improves beyond its null floor).

MLX does not change the kernel's duration at the decode shapes: 2.14 µs standalone and through MLX for the same artifact, same geometry. At 2048 rows the kernel takes 16 to 31 % longer inside MLX than standalone (140 to 142 µs against 108 to 121 µs), and the existing kernel takes exactly as long there (140 to 142 µs), so that cost belongs to how MLX runs a kernel of this size, not to either implementation. The standalone program reuses its two output buffers for every launch; MLX allocates both outputs (16 MiB together at this shape) for every evaluation. That is the likely cause and it was not isolated.

The graph path's two kernels together take about as long as one fused kernel at one row (2.5 µs against 2.2 µs) and at 2048 rows (144 µs against 140 µs), and less at 128 and 512 rows (3.6 and 8.5 µs against 6.5 and 24 µs). MLX's `rms_norm_small` runs with a block of 128 where the fused kernels use 512.

Standalone, CUDA events (M3, `m3-standalone-events.csv`, median of 5 processes). `warm` is #2832's definition (events around 200 back-to-back launches); the small shapes read 4.1 µs. The warm figures are multiples of 2.048 µs per launch (4.10 = 2 x 2.048 at 1 x 1024 and 1 x 2048, 6.14 = 3 x at 1 x 8192, 8.20 = 4 x at 128 x 2048), while the per-launch synchronized samples take values such as 4.544 and 4.672 µs, so the timer itself resolves far below 2 µs. That pattern points to a back-to-back launch rate of about one launch per 2.048 µs for kernels this short; the cause was not isolated. `launch call` is the host time of `cuLaunchKernel` alone and `launch + sync` the host time of one launch and one stream synchronize.

| act / weight | rows x dim | block | first launch µs | warm µs (min..max) | launch call µs | launch + sync µs | effective GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | 1 x 2048 | 512 | 17.0 | 4.10 (4.09..4.13) | 1.88 | 6.62 | 6.0 |
| bf16 | 4 x 2048 | 512 | 6.0 | 4.10 (4.09..4.10) | 1.86 | 6.69 | 24.0 |
| bf16 | 128 x 2048 | 512 | 12.1 | 8.20 (8.20..8.20) | 1.64 | 10.93 | 383.8 |
| bf16 | 512 x 2048 | 512 | 30.5 | 26.58 (26.51..26.86) | 1.67 | 29.25 | 473.4 |
| bf16 | 2048 x 2048 | 512 | 129.0 | 123.38 (121.68..124.54) | 3.82 | 124.95 | 407.9 |
| f32 | 1 x 4096 | 1024 | 8.0 | 5.30 (5.26..5.59) | 1.78 | 8.29 | 18.5 |
| f32 | 64 x 8192 | 1024 | 32.6 | 31.97 (28.89..34.40) | 1.66 | 34.57 | 393.6 |
| bf16 | 1 x 4096 | 1024 | 6.0 | 4.78 (4.10..5.57) | 1.62 | 7.02 | 10.3 |
| bf16 | 64 x 8192 | 1024 | 20.3 | 16.25 (15.98..16.31) | 1.58 | 20.39 | 387.1 |
| f16 | 1 x 4096 | 1024 | 6.0 | 4.10 (4.10..4.11) | 1.59 | 6.88 | 12.0 |
| f16 | 64 x 8192 | 1024 | 20.1 | 16.23 (15.91..17.79) | 1.61 | 18.40 | 387.5 |

The other 12 shapes of #2832's list are in the CSV. Against `results/bench-sm_121.json` from `spark-d9e2` (another GB10, the cuda-oxide harness): 64 x 8192 reads 36.3, 18.5 and 18.3 µs there (f32, bf16, f16) and 32.0, 16.3 and 16.2 µs here; the launch-bound shapes read 4.1 to 6.1 µs on both. The two harnesses agree to within the timer and host differences, so the driver-API program measures the same kernel the #2832 harness did.

### Op level through MLX, and where the per-op difference comes from

M5, production mode (CUDA graphs on), `pipe`: build and `async_eval` 200 ops, one synchronize at the end, per-op µs. Median of 15 blocks (3 processes x 5 rotated rounds), first set, then the repeat.

| act | rows x dim | `graph` | `fused` | `oxide` | `oxide` vs `fused` |
| --- | --- | --- | --- | --- | --- |
| bf16 | 1 x 2048 | 13.71, 14.58 | 14.50, 16.78 | 12.68, 13.80 | -1.82 µs (-12.5 %), -2.98 µs (-17.8 %) |
| bf16 | 4 x 2048 | 14.55, 14.65 | 15.18, 16.68 | 13.41, 13.72 | -1.77 µs (-11.7 %), -2.95 µs (-17.7 %) |
| bf16 | 128 x 2048 | 22.19, 15.65 | 25.17, 18.39 | 21.70, 14.78 | -3.47 µs (-13.8 %), -3.62 µs (-19.7 %) |
| bf16 | 512 x 2048 | 31.28, 27.39 | 32.05, 28.92 | 31.13, 28.99 | -0.93 µs (-2.9 %), +0.07 µs (+0.2 %) |
| bf16 | 2048 x 2048 | 168.84, 168.07 | 161.28, 161.22 | 159.08, 160.13 | -2.20 µs (-1.4 %), -1.09 µs (-0.7 %) |
| f32 | 1 x 4096 | 15.35, 15.16 | 16.51, 19.13 | 14.26, 17.16 | -2.25 µs (-13.6 %), -1.98 µs (-10.3 %) |
| f16 | 1 x 4096 | 13.59, 14.79 | 15.87, 17.68 | 13.25, 14.23 | -2.62 µs (-16.5 %), -3.44 µs (-19.5 %) |

Through MLX the `oxide` arm is 1.8 to 3.6 µs per op cheaper than `fused` on every small shape, in both sets: on the 14 shapes of at most 33 rows and width at most 4096 the paired per-block difference is negative in 97 % and 99 % of 210 blocks, median -2.4 and -3.0 µs. It is also 0.6 to 0.9 µs cheaper than the graph path there. The other 16 shapes are in `m5-op-level-mlx.csv` and `m5-op-level-mlx-repeat.csv`.

That difference is not kernel time (2.14 against 2.2 µs, above) and it is not CUDA API time. M8 profiled one arm per process, 2020 ops at 1 x 2048, and `fused` and `oxide` make the same calls per op: with graphs on, one each of `cudaGraphCreate`, `cuGraphAddKernelNode`, `cudaGraphExecUpdate`, `cudaGraphLaunch`, `cudaGraphDestroy`, `cudaLaunchHostFunc`, `cudaStreamWaitEvent` and two `cudaEventRecord`; with graphs off, one `cuLaunchKernelEx` in place of the five graph calls (`m8-cuda-api-per-arm.csv`). The graph path differs only in adding two kernel nodes and one dependency instead of one node.

M9 timed the remaining candidate, the host work of building the lazy node, with nothing evaluated (2000 builds per block, 15 blocks, µs per node or node pair):

| Rows x 2048, bf16 | `graph` (two nodes) | `fused` | `oxide` |
| --- | --- | --- | --- |
| 1 | 0.53 | 1.44 | 0.65 |
| 4 | 0.54 | 1.47 | 0.66 |
| 128 | 0.84 | 2.36 | 0.99 |
| 512 | 0.87 | 2.36 | 1.00 |
| 2048 | 0.86 | 2.35 | 1.00 |

Building the existing kernel's node costs 0.8 to 1.4 µs more than building the cuda-oxide node. The existing launcher (`fused_add_rms_norm` in `kernels/fused_norm.cpp`) goes through `fast::cuda_kernel`: it creates two one-element arrays for `eps` and `weight_bias`, a vector of four named template arguments, and five inputs; the cuda-oxide primitive takes three inputs and packs the scalars itself. So about a third to a half of the per-op gap is node construction. The rest, 1 to 2 µs, is host work at evaluation that was not attributed further: `perf` is not available to this user (`perf_event_paranoid` is 4), so no CPU profile was taken, and the candidates (resolving the per-`(T, TW, Dim, Threads)` kernel by its generated name at every evaluation, registering five inputs instead of three) are named from the code, not measured.

The gap is the dedicated primitive against the generic custom-kernel launcher. It does not depend on which compiler produced the kernel: a dedicated primitive that launched the NVRTC kernel would have the same host path. It is also too small to see end to end. A decode step builds 16 joins, so the gap is 30 to 50 µs per token against a step of 11.4 ms at B=1 (87.6 tok/s): 0.3 to 0.4 %, the size of the null floor in those cells (0.28 to 0.75 %). M1's paired deltas for the decode cells are between -0.31 % and +0.67 %.

MLX per-op cost, matched (same artifact, bf16 1 x 2048, same geometry, kernel duration 2.14 µs on both sides per M4): the standalone program needs 6.62 µs of host time for one launch and one stream synchronize. Through MLX, `iso` mode (build, evaluate both outputs, synchronize, per op) needs 28.6 µs in its best block of the repeat set and 39.4 µs in the best block of the first set for the `oxide` arm, and 30.9 and 37.7 µs for `fused`. So a synchronously evaluated op costs 22 to 33 µs more through MLX than through the bare driver, the same for either kernel. `iso` block medians on this shared host are bimodal (the same arm and shape gives about 30 to 50 µs in one block and 200 to 400 µs in another, depending on how long the waiting thread takes to wake), which is why the best block is quoted and why `iso` decides nothing here; the full distributions are in the CSVs. Without the per-op wait (`pipe`), an op costs 12.7 to 13.8 µs through MLX against 1.88 µs for the bare launch call. M8 shows what that is made of: per evaluation, MLX creates, updates, launches and destroys a CUDA graph, records two events, waits on one and enqueues one host function, about 10 µs of API time under the profiler (medians: graph launch 2.6 to 2.9 µs, host function 2.9, event record 1.4 to 1.5 each, graph create 1.1 to 1.3, graph destroy 0.5 to 0.7, exec update 0.35 to 0.4, wait event 0.3). That cost is per evaluation, not per kernel, and a model evaluation holds many ops, so it is not what a join costs inside a decode step; it is what this one-op benchmark pays.

### Cold start (G3b)

M6 and its driver-cache extension: first-use cost = first build, evaluate and synchronize of the op in a fresh process, minus the steady-state median, in ms. Median (min..max) of 5 processes. The second column is the first use of a second width in the same process.

| State | First use, 1 x 2048 | Then 1 x 4096 |
| --- | --- | --- |
| `fused`, empty disk cache (NVRTC compiles) | 697.9 (690.6..711.1) | 623.0 (616.4..636.6) |
| `fused`, warm disk cache | 5.25 (5.05..5.48) | 5.05 (4.89..5.92) |
| `oxide`, driver compute cache warm (this host's normal state) | 11.82 (11.43..12.22), of which 11.24 is the first-use `select` step (validation, device capability query and C++ install, not broken down) | -0.03 (-0.16..0.23) |
| `graph` | 7.54 (6.80..7.71) | 0.20 (-0.01..0.54) |
| `fused`, empty disk cache, `CUDA_CACHE_DISABLE=1` | 757.3 (751.1..772.4) | 680.7 (666.3..688.0) |
| `fused`, warm disk cache, `CUDA_CACHE_DISABLE=1` | 5.25 (5.03..5.55) | 5.27 (4.92..5.37) |
| `oxide`, `CUDA_CACHE_DISABLE=1` (the driver compiles the PTX) | 87.94 (84.73..116.52), of which 11.30 is the first-use `select` step | 0.02 (0.01..0.11) |

Two notes on the table. The `fused` empty-cache control moved from 697.9 ms to 757.3 ms (and from 623.0 to 680.7 ms at the second width) under `CUDA_CACHE_DISABLE=1`, while its warm control stayed at 5.25 ms; the existing kernel's disk-cache entries are cubins, which the driver does not compile, so the rise was not expected and was not investigated. The first-run comparison uses the cache-off rows on both sides. And every figure is a first-use time minus that process's steady-state `iso` median (`m6-cold-start.csv`, `steady_median_us`), which is bimodal on this host (see the reporting note under the protocol adjustments): the subtracted medians run from 0.025 to 0.87 ms across the processes, so each cold-start figure carries up to about 0.9 ms of noise from the subtraction. That is small against 11.8 and 5.25 ms and against 88 and 757 ms, and it is why the near-zero second-column values (-0.03 and 0.20 ms) mean no measurable cost, not a negative one.

Standalone module load (`cuModuleLoadData` on the committed PTX, host wall time, 5 processes): 0.26 to 0.51 ms with the driver's compute cache warm, 73.2 to 75.6 ms with `CUDA_CACHE_DISABLE=1`. #2832 reported 255 ms on `spark-d9e2`.

In the state every process after the first is in, the cuda-oxide route is slower to first use than the existing kernel: 11.8 ms against 5.25 ms at one width, about the same at two (11.8 against 10.3 ms), and it only gets ahead from the third width on. Its 11.2 ms is not the module load (0.3 ms). It is the time of the first call's `select` step, which covers the artifact validation (the PTX digest, five manifests and five schemas, and the parameter layout re-derived from the PTX), the device capability query and the C++ install, paid once per process. The three parts were not timed separately, so how much of the 11.2 ms is validation is not known. **G3(b) is not met in the normal state.**

On the first run on a machine, or after a driver update, the cuda-oxide route needs 88 ms once for every width, where the existing kernel needs 0.70 to 0.76 s for the first width and 0.62 to 0.68 s for each further one. That is a real reduction of about 0.6 s per width, once per machine. Per the frozen wording, G3(b) is **met for the first run per machine and width only**, and the disposition has to weigh it instead of counting it as a general reduction. Weighed: a model has one hidden size, so it is one compile of about 0.7 s per model family and machine, cached on disk since #2492, against 6.6 ms added to every later process. Next to a model load of seconds it is not a benefit that carries a second compiler toolchain.

### Registers, code size, traffic, launch cost (M7)

`cuobjdump -res-usage` and `-sass` on the committed cuda-oxide cubin and on the existing kernel's disk-cache entries (`m7-kernel-resources.csv`):

| Kernel | Registers | Stack bytes | Shared bytes | Constant bank bytes | SASS instructions |
| --- | --- | --- | --- | --- | --- |
| cuda-oxide `fused_add_rms_norm_bf16` (any width) | 32 | 0 | 1156 | 992 | 616 |
| cuda-oxide `_f16` | 30 | 0 | 1156 | 992 | 528 |
| cuda-oxide `_f32` | 30 | 0 | 1156 | 992 | 496 |
| cuda-oxide `_bf16_wf32`, `_f16_wf32` | 32, 30 | 0 | 1156 | 992 | 576, 512 |
| existing, bf16, width 1024 and 2048 | 18 | 0 | 1156 | 952 | 264 |
| existing, bf16, width 4096 / 8192 | 20 / 26 | 0 | 1156 | 952 | 256 / 392 |
| existing, f16, width 1024 to 4096 / 8192 | 18 / 22 | 0 | 1156 | 952 | 256 to 272 / 392 |
| existing, f32, width 1024 to 4096 / 8192 | 18 to 20 / 26 | 0 | 1156 | 952 | 160 to 168 / 208 |

Neither kernel spills (stack 0) and both report the same shared memory. The cuda-oxide kernel uses 32 registers where the existing bf16 kernel at the checkpoint's width uses 18, and 2.3 times the instructions (616 against 264). It takes the width as a launch argument where the existing kernel is specialized on it, and it carries slice lengths; which of those accounts for how much of the extra code was not broken down. The extra code and registers cost nothing measurable: a block of 512 threads at 32 registers is far from the per-SM register limit, and the durations are equal.

Why the durations are equal. The cuda-oxide kernel is a term-for-term port of the existing one with the same launch geometry (one block per row, the same block width, the same two sweeps and the same two-stage reduction), which is what #2832 required so that it could match bit for bit. Two kernels with the same structure, the same bytes read and written in the same order, and the same number of blocks and threads take the same time unless one compiler generates much worse inner loops, and neither does. At the decode shapes the kernel moves 24 KiB (one row) or 96 KiB (four rows) by the formula in the CSV (`bytes_moved`, which counts the per-row weight read) and takes 2.1 to 2.2 µs whatever the row count: it is bound by launch and block start-up, so loop code cannot show at all.

The prefill shapes are not simply memory-bound, and the table says so: at 128 and 512 rows the graph path's two kernels do the same arithmetic in 3.6 and 8.5 µs where either fused kernel takes 6.5 and 24 µs. What the fused kernels are bound by at those shapes is their own structure (512 threads per row at 4 elements each, three block-wide barriers, two passes over the row), which both implementations share. Only at 2048 rows do fused and graph path converge (140 against 144 µs). Counting traffic as #2832 did (`x` and `residual` read, `new_residual` written and re-read, `normed` written, the weight read once per row) gives 12.6 MB in 24 µs at 512 rows and 50.3 MB in 140 µs at 2048 rows through MLX, 520 and 360 GB/s effective for both kernels alike; the count includes the re-read of `new_residual` and the per-row weight read, which a cache serves, so it is not a DRAM bandwidth figure.

This also bounds what the experiment could have found. The 10 % kernel gate could only have been met by code generation, because the port was not allowed to change the algorithm. A differently structured kernel, in either language, was not built or measured.

Launch and synchronization: one bare `cuLaunchKernel` costs 1.6 to 1.9 µs of host time (3.8 µs at 2048 rows), and launch plus stream synchronize 6.6 µs at the decode shapes. The MLX-side launch costs are in the op-level section. The adapter adds no CUDA API call to what the existing kernel makes.

### What could not be measured, and why

- **Hardware counters** (occupancy, achieved bandwidth, stall reasons). `ncu` refuses as this user: `ERR_NVGPUCTRPERM`. No counter value appears in this report. Registers and spills come from `cuobjdump`, traffic from byte counts and durations.
- **Per-kernel time in production mode.** With CUDA graphs on, kernels do not appear in the profiler's kernel table on this setup. Kernel durations are from `MLX_USE_CUDA_GRAPHS=0`; end-to-end numbers are from graphs on. Nothing bridges the two except that the kernel is the same code.
- **A CPU profile of the host path.** `perf_event_paranoid` is 4. The per-op gap is attributed by elimination (kernel time, CUDA API calls) and by timing node construction; the evaluation-time remainder of 1 to 2 µs is named from the code only.
- **An idle host.** See Host state. The end-to-end rounds are interleaved with a null arm for that reason, and their null floors (0.3 to 2.1 %) are what the host allowed.
- **The cause of the 16 to 31 % longer kernel at 2048 rows inside MLX**, and of the 0.38 µs difference at four rows. Both are reported, neither was isolated.
- **Not run:** a second, larger checkpoint (optional in the protocol; the result leaves no question it would answer, since a larger model gives the join a smaller share); a quantized checkpoint; batches above 4; time to first token of a whole fresh process; Compute Sanitizer through the adapter; the #2832 harness itself (no cuda-oxide toolchain here).

## Gate evaluation

| Gate | Outcome | Basis |
| --- | --- | --- |
| G1 correctness | **Pass** | Route signatures valid in every run; logit traces byte-identical in three shapes; greedy ids identical in 6 cells x 7 rounds; 130 of 130 at the op boundary. |
| G2 no end-to-end regression above 2 % outside noise | **Pass**, one cell unresolved | No cell flagged; worst paired median -0.31 %. TTFT at B=4 with the 2048-token prompt has a null floor of 2.14 % and cannot resolve the threshold. Peak memory equal to 0.04 %. |
| G3(a) kernel at least 10 % faster with a useful end-to-end effect | **Not met** | Kernel durations within 3 % on four of five shapes; 14.8 % (0.38 µs) at four rows only. No end-to-end cell improves beyond its null floor. |
| G3(b) cold-start reduction | **Not met in the normal state; met for the first run per machine and width only** | 11.8 ms against 5.25 ms with warm caches. 88 ms against 0.70 to 0.76 s on a first run. |
| G3(c) maintenance benefit | **Not met** | See below. |

G3(c), from the tree. What the route adds: a second pinned compiler toolchain to produce the artifact (Rust 1.99.1 with compiler-development components, LLVM 21, `cargo-oxide`, an alpha project), which is absent from this host and from CI; a committed PTX and cubin with five manifests and five ABI schemas, and the validation that guards them (`cuda_oxide_artifact.rs`, `cuda_oxide_abi.rs`, the build-script digest check, about 900 lines); the adapter (`fused_norm_oxide.cpp` and its header, about 590 lines, the one launcher that includes MLX's CUDA backend headers and so sits on the pin-bump checklist); an overlay parameter; and about 2,000 lines of tests. What it removes: nothing. The existing kernel stays, because Metal, ROCm and every CUDA architecture other than sm_121 use it, and because it is the fallback for dtype pairs and shapes the artifact does not cover. The one structural advantage, a module that is not compiled per width, is the first-run saving already weighed under G3(b). No net maintenance benefit.

## Recorded result and recommendation

**Stop.** Per the frozen rule: G1 passes and G3 is not met, so the disposition is no expansion of cuda-oxide kernels on this evidence. This is the experiment's recorded result and a recommendation.

Maintainer decision (2026-11-28): stop is adopted for the default, and the #2833 adapter stays in the tree as an opt-in route, off by default at build time and at run time. It is not removed, because a later rewrite of the kernels on cuda-oxide remains possible and the adapter is the working reference for the artifact boundary.

The other three dispositions do not apply. *Expand within MLX* needs a benefit through MLX, and there is none that passes a gate. *Investigate a demonstrated execution constraint* needs a kernel that is at least 10 % faster standalone and loses that inside MLX; the kernel is not faster standalone (it runs for the same 2.1 µs as the existing kernel at the decode shapes, and for the same time at the prefill shapes), so MLX has nothing to take away. The one MLX-side cost this report did find, the longer kernel at 2048 rows, hits both implementations equally. *Resolve toolchain limitations* needs the compiler, the generated code or GPU support to be what limits the result; the toolchain did its job: the artifact loaded, validated and matched bit for bit, on GB10.

What the experiment established, for the record:

- cuda-oxide can produce a correct inference kernel for GB10 and run it inside MLX's lazy evaluation, CUDA graphs and allocator with no change to the backend, bit-identical to the existing kernel on a real checkpoint. The feasibility question the review asked is answered yes.
- For this operator the implementation language buys nothing. The Rust port has the structure of the CUDA C++ kernel, as bit identity required, and the two compilers' code runs for the same time: launch-bound at decode, and bound by that shared structure at prefill. That is what the review predicted: "Changing the implementation language does not remove an existing launch, memory read, or graph dependency. Speedups require a better algorithm, fusion boundary, tile/layout, instruction selection, or dispatch policy."
- This result is about fused residual-add + RMSNorm. It is not evidence for or against cuda-oxide on attention, quantized GEMM, or a whole engine, and it says nothing about an independent inference engine being faster or slower.

Follow-ups this report suggests, none of them done here and none of them a cuda-oxide expansion:

- The default-off adapter from #2833 now carries a cost (the pin-bump checklist entry, the artifact, the validation code) and no user benefit. The maintainer kept it as an opt-in route (decision above), so it stays on the pin-bump checklist.
- The host-side gap between the generic custom-kernel launcher and a dedicated primitive (0.8 to 1.4 µs per node at construction, 1.8 to 3.6 µs per op in all) is independent of cuda-oxide: it was measured for the one existing launcher, `fused_add_rms_norm`, built through `fast::cuda_kernel`, and other launchers built the same way may carry the same cost, which was not measured. It is below what one checkpoint could resolve end to end here, so it is a lead for a profile of a model with more custom-kernel joins per step, not a finding to act on from this data.
- On this checkpoint the existing fused kernel is not faster than the graph path it replaces: `graph` is 3.0 % ahead at server B=1 decode on the short prompt (a cell where the two generate different lengths) and within about 2 % elsewhere, it is cheaper per op in `pipe` mode up to 128 rows (13.7 against 14.5 µs at one row) but not at 2048 rows (168.8 against 161.3 µs), and its two kernels take 55 % and 35 % of the fused kernel's time at 128 and 512 rows. That matches `FUSED_ADD_RMSNORM_DEFAULT` being off on CUDA and gives no reason to change it. If the fused kernel is to earn its place on CUDA, its structure at multi-row shapes is where the time is, and that is a question for the existing kernel, not for cuda-oxide.

## Limits

- One checkpoint (Llama 3.2 1B Instruct, bf16), one GPU (GB10, sm_121), one driver, one MLX pin. The checkpoint was chosen to give the join its largest share, so a larger model would show less, not more.
- The end-to-end resolution is the null floor: 0.3 to 0.8 % for decode at B=1, 0.4 to 1.7 % at B=4, 0.4 to 2.1 % for TTFT. An effect below that is not excluded; the op-level data put the cuda-oxide route's expected decode effect at +0.3 to +0.4 %, which is below it.
- Kernel durations come from a mode (CUDA graphs off) that is not the one models run in.
- The comparison is between two implementations of one algorithm. It does not say what a restructured fused kernel could reach, in Rust or in CUDA C++; the graph path's kernel times at 128 and 512 rows show there is room.
- The host was shared and is a CI runner. Stages that a CI run overlapped were repeated; both sets are in the tables.
- `MLXCEL_CUDA_OXIDE_FUSED_NORM` and the fused kernel itself stay opt-in. Nothing here changes a default.
