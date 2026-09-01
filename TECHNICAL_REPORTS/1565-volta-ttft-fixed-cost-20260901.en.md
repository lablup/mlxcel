# Technical Report: PR #1565 - what the Volta fixed first-token cost is actually made of

**Date**: 2026-09-01

**Author**: mlxcel maintainers

**Status**: Completed as findings and instrumentation, with no performance fix. Every lever issue #1545 named was measured and none of them is where the time is. The cost was found, named, and attributed to a mechanism the issue did not suspect, a follow-up was filed outside epic #1536, and the two GB10 acceptance criteria stay unticked because this host has no sm_80 or later device.

---

## Executive Summary

Issue #1545 was filed on a CUDA API table. On a 24-token `qwen3.8-27B-4bit` run the audit saw `cudaGraphInstantiate` at 2.14 s, `cudaGraphAddKernelNode` at 1.95 s over 72,539 calls and `cuModuleLoadDataEx` at 848 ms, and concluded that graph instantiation and JIT module loading dominate the roughly 13 second fixed cost in front of the first token. The issue's own acceptance criteria required a written breakdown regardless of outcome and named "mostly inherent" as an acceptable close provided the split is measured.

**The split is measured and the premise did not survive it.** On `qwen3.8-27B-4bit` at a 2-token prompt, warm PTX cache and warm file page cache, the first token arrives 15.54 s after the model is reported loaded, over 10 repetitions with a 6.5% spread. The same prefill run a second time inside the same process takes 0.546 s. So 14.99 s of it, 96.5%, is one-time process cost, and it is not graph work.

**12.08 s of that, 77.8% of the whole first token, is materializing the language model's 15.13 GB of weights.** MLX's `load_safetensors()` returns unevaluated `Load` arrays, so nothing reaches the device until the first `eval`, which sits inside the prefill window. The CLI already says so and it is easy to read past: `Model loaded in 0.98s` is printed immediately above `resident: 0.00 GB`. The cost splits into 7.40 s of host staging (a `malloc`, a page-cache read, and a `free` per tensor) that no CUDA profiler can see, 4.19 s of pageable host-to-device copy at 3.61 GB/s, and 0.49 s of device allocation.

The 1,847 `cudaMemcpyAsync` calls the audit found are not a measurement but a structural constant. The checkpoint's index holds 2,180 tensors: 1,847 under `language_model.` and 333 under `vision_tower.`, and a text-only run touches exactly the language model. The audit measured 1,847, #1538 re-measured 1,847 on a different build, and this work measures 1,847 at `-n 1`, `-n 8` and `-n 40`. One copy per weight, paid once.

**Everything the issue title names is small.** CUDA graph instantiation is 0.10 s and saturates at 196 distinct graphs against a 2,000-entry cache, so there is no re-instantiation and no thrashing. Setting `MLX_USE_CUDA_GRAPHS=0`, which removes graph construction entirely, changes the first token by less than the default configuration's own repeat spread. Warm `cuModuleLoadDataEx` is 61 ms over 6 calls, 0.4% of the first token.

**The verdict on the branch point is general, not Volta-specific, and it was settled without GB10 hardware** by looking at what the cost is made of rather than by comparing two devices.

This PR lands `MLXCEL_PROFILE_TTFT`, the instrumentation the pre-first-token phase never had, six host-only unit tests over its arithmetic, a full measurement record, and three updates to the Volta baseline record including a new method rule.

## 1. Problem Statement

The baseline record `volta-sm70-baseline-2026-08-31.md` measured a large prompt-independent term in front of the first token and left it named but not explained. Issue #1545 inherited it with three concrete questions and one branch point.

The questions: what the fixed cost is made of, whether the JIT cache is working, and why there are 72,539 `cudaGraphAddKernelNode` calls against 4,387 `cudaGraphLaunch` calls. The branch point: whether any of this is Volta-specific, because graph construction is host-side and should look the same on GB10, and if it is general then the issue does not belong to epic #1536 at all.

Two prior findings complicated the starting point. #1538 reconciled the audit's ~13 s against its own 24.94 s arithmetically and said plainly that the confirming test had not been run. #1539 changed decode substantially (qwen 220 to 117.83 ms/token), which moves any slope fit and therefore any intercept, so nothing pre-#1539 could be quoted.

The instrumentation available was insufficient by the issue's own account: `MLXCEL_PROFILE_PIPELINE` reports per-token pipeline time only and does not cover the pre-first-token phase at all, so extending it was named as possibly part of the work.

## 2. Change Summary

Six files, 579 insertions, 2 deletions.

| File | What it does |
|---|---|
| `src/lib/mlxcel-core/src/generate.rs` | `TtftPhases`, `ttft_profile_enabled()`, and `MLXCEL_PROFILE_TTFT` instrumentation on both `--profile` paths |
| `src/lib/mlxcel-core/src/ttft_profile_tests.rs` | Six host-only unit tests over the phase arithmetic |
| `src/lib/mlxcel-core/src/lib.rs` | Wires the test module |
| `docs/benchmark_results/volta-ttft-fixed-cost-2026-09-01.md` | The measurement record |
| `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md` | #1545 post-program row, new method rule 7, confirming-test outcome row |
| `docs/environment-variables.md` | `MLXCEL_PROFILE_TTFT` |

No device code and no dispatch. No cubin anywhere in the tree changes.

## 3. Technical Decisions

### 3.1 Measure the fixed cost directly rather than extrapolate an intercept

The obvious estimator is the intercept of a prefill ladder, and it is wrong here. Prefill on this part is not affine in prompt length at the short end: the marginal rate between 2 and 46 tokens is 223 ms per token while between 331 and 601 tokens it is 114. A least-squares fit over five rungs gives 124.75 ms per token, reproducing the baseline record's 125.07 to 0.3%, but its intercept of 19.61 s carries residuals of -4.10, +0.25, +4.23, +1.29 and -1.68 s, which is a curve rather than scatter.

The estimator used instead is a process split: run the same prompt twice in one process. The first prefill pays every one-time cost and the second pays none of them, so the difference is the one-time cost with nothing extrapolated. `mlxcel-bench-decode --warmup-tokens 1 -n 1` already provides the second arm, so this needed no new code. It gives 14.99 s against the fit's 19.61 s, and it is the number the report uses.

This also retires the arithmetic in #1538's reconciliation. That subtraction assumed prefill is affine down to zero tokens, so its 0.5% agreement is tighter than the method supports. The conclusion it drew is still right and is now measured rather than inferred.

### 3.2 Attribute the remainder with a microbenchmark, because the largest term is invisible to CUDA profiling

nsys `cuda_api_sum` on the `-n 1` run totals 6.24 s of a 16.93 s profiled first token. That gap is what defeated the original audit: it read the largest row in the table it had and filed against it.

`Load::eval_gpu` in `mlx/backend/cuda/load.cpp` explains the gap. Per tensor it does one device allocation, one `malloc` of the whole tensor, one **synchronous host read** into it, one host-to-device `cudaMemcpyAsync` out of that pageable buffer, and a `cudaLaunchHostFunc` to free it. The host read is not a CUDA call and appears in no CUDA profile.

It was therefore measured on its own, with a small C program that replicates that host path over the same 1,847 tensors at their real safetensors offsets and touches no CUDA at all: `malloc` 0.002 s, read 6.763 s, `free` 0.634 s, 7.40 s total for 15.13 GB, reproducible to 0.2% across runs. `malloc` is free because glibc hands back a fresh mapping; the cost lands in the read, which is a copy out of the page cache plus a first-touch fault on every destination page, and in the `free`, which unmaps 15 GB.

That single row is 47.6% of the first token, larger than every CUDA API combined.

### 3.3 Instrument the phase, and let the instrumentation rule things out

`MLXCEL_PROFILE_TTFT` splits the pre-first-token path into `setup`, `build` (the lazy forward, host only, device idle), `sample`, `eval` and `post`, and prints the reported prefill next to the unattributed residual so a reader can check the phases account for it rather than take it on trust.

```
[TTFT] prompt=2  tok setup=0.32ms build=11.07ms sample=0.16ms eval=15693.46ms post=0.00ms | prefill=15704.69ms residual=0.00ms
[TTFT] prompt=46 tok setup=0.32ms build=11.64ms sample=0.18ms eval=25668.00ms post=0.00ms | prefill=25679.81ms residual=0.00ms
```

The residual is zero to the printed resolution in every run. Two live hypotheses die in that one line. Host-side graph construction is not the problem: `build` is every MLX op recorded for a 64-layer hybrid model with the device idle, and it costs 11.1 ms, 0.07% of the first token. It is also not a function of prompt length, because the recorded graph has the same shape at 2 tokens and at 46. Everything is inside one `eval`.

The variable is read once per process through a `OnceLock`, because a server calls this path per request. When it is unset the cost is five `Option<Instant>` constructions that are never taken.

### 3.4 Where the cost is charged is a loader property, and that is why it is not architectural

Three checkpoints, warm page cache, warm PTX cache, same command:

| Checkpoint | LM bytes | Tensors | Model load | Resident after load | First prefill | Second prefill |
|---|---|---|---|---|---|---|
| `gemma-4-12B-it-4bit` | 6.70 GB | 1,324 | 8.87 s | 6.28 GB | 2.449 s | 0.191 s |
| `gemma-4-12b-it-8bit` | 12.65 GB | 1,324 | 20.96 s | 11.85 GB | 3.134 s | 0.128 s |
| `qwen3.8-27B-4bit` | 15.13 GB | 1,847 | 0.98 s | 0.00 GB | 15.387 s | 0.562 s |

`load_gemma4_family_weights_with_backing` in `src/models/sanitize.rs` ends with `eval_all` then `detach_all`, so Gemma weights are resident when load returns. The Qwen 3.5 loader leaves everything lazy. Same device, same driver, same build, opposite reporting.

The two Gemma arms are the controlled experiment: identical tensor count at 1,324, bytes varying by 1.89x, load time moving 2.36x. The cost is driven by bytes, not by per-tensor overhead.

**This is an argument for measuring the total, not for moving the cost, and that distinction is why no fix landed.** Per byte of language model the eager Gemma path runs at 0.76 and 0.60 GB/s while the lazy Qwen path runs at 1.02 GB/s. Relabelling Qwen's 12 s from "prefill" to "model load" would make `--profile` honest and would move a server's cost from its first request to its startup, but on this evidence it would not make anything faster, and Gemma's numbers include model construction and repacking that Qwen's do not, so even the direction is unestablished. It is a follow-up with its own measurement, not a fix to land inside a diagnosis issue.

### 3.5 The graph question, answered by making instantiation saturate

| `-n` | `cudaGraphInstantiate` | `cudaGraphExecUpdate` | `cudaGraphAddKernelNode` | `cudaGraphLaunch` | `event_signal_kernel` |
|---|---|---|---|---|---|
| 1 | 103 | 469 | 8,320 | 572 | 1,847 |
| 8 | 196 | 1,379 | 25,870 | 1,575 | 1,847 |
| 40 | 196 | 5,859 | 104,558 | 6,055 | 1,847 |

Instantiation saturates at 196 and stays there. There are 196 distinct graph shapes, each instantiated once, and every later commit hits MLX's exec cache and takes `cudaGraphExecUpdate`, which grows linearly as it should. 196 is the same count #1538 measured. There is no re-instantiation to explain, and no thrashing, since mlxcel's `MLX_CUDA_GRAPH_CACHE_SIZE` default is 2,000 (#818) against a working set of 196.

The add-node count is a per-launch cost proportional to generated tokens, about 2,459 nodes per token from the `-n 8` to `-n 40` slope, so the audit's 72,539 on a 24-token run is decode, not fixed cost. At `-n 1` the whole graph-API group is 0.97 s of a 15.54 s first token.

Why so many commits for so few nodes has two causes and neither is a Volta defect. MLX's `get_graph_limits` computes `cc = major * 100 + minor * 10` and enumerates 800, 900, 1000, 1200 and 1210; a V100 is 700 and takes the fall-through of 20 nodes or 100 MB. That sounds like a Volta problem until the table is read: A100 is enumerated and gets the same 20 nodes, and GB10 gets 20 nodes with a byte budget four times tighter. The second cause is a fence per weight, below.

### 3.6 One cross-stream fence per weight, which is an mlxcel interaction rather than an MLX or a device one

MLX inserts a fence whenever an evaluated array's primitive runs on a different stream from one of its inputs (`mlx/transforms.cpp`). mlxcel loads weights on whatever stream is default at load time and then installs a dedicated per-thread generation stream inside the generator (`install_thread_local_default_stream`). Every weight is therefore a cross-stream input and gets a fence, and `Fence::update` opens with `encoder.commit()` before launching a one-thread kernel, so each fence also closes and commits a CUDA graph.

The count confirms it exactly: `event_signal_kernel` fires 1,847 times at `-n 1`, 1,847 at `-n 8` and 1,847 at `-n 40`, flat, and equal to the weight count and to what #1538 measured at `-n 24`.

Its direct cost is small, 0.16 s of GPU time plus a share of the 0.97 s of graph APIs. It is recorded because it explains the graph traffic the issue asked about and because it is the one part of the path specific to how mlxcel drives MLX. It is not fixed here: reordering stream installation against model loading is a change with its own correctness surface across the server, the batch scheduler and the speculative drafter, and 0.16 s does not justify it.

### 3.7 No fix, because the evidence does not support one

Stated as a decision rather than an omission. Both remaining knobs were swept and both are inert, against a default configuration whose own repeat spread over ten runs is 6.5%:

| Arm | First token | Against default |
|---|---|---|
| `MLX_USE_CUDA_GRAPHS=1` (default) | 15.64 s | |
| `MLX_USE_CUDA_GRAPHS=0` | 15.36 s | -1.8% |
| `MLX_MAX_OPS_PER_BUFFER=20` (MLX's default here) | 15.50 s | |
| `MLX_MAX_OPS_PER_BUFFER=100` (what H100 and B200 get) | 15.46 s | -0.2% |
| `MLX_MAX_OPS_PER_BUFFER=400` | 15.03 s | -3.0% |

Disabling CUDA graphs entirely does not measurably change the first token. That is the cleanest available refutation of the issue's own title: graph construction is 6.2% of the first token by direct attribution, and removing all of it recovers less than the noise, because the host time saved is spent launching the same kernels individually instead.

## 4. Validation

- `MLX_CUDA_ARCHITECTURES=70 make release-cuda` clean; `cuobjdump --list-elf` on the MLX archive reports 96 cubins, all sm_70 and nothing else.
- `cargo test -p mlxcel-core --features cuda ttft_profile`: 6 passed, 0 failed.
- `cargo fmt`; `cargo clippy -p mlxcel-core --features cuda --lib -- -D warnings`.
- The instrumented binary reproduces the uninstrumented one: first token 15.70, 15.64, 15.92 s against 15.54 s mean over ten uninstrumented runs, inside the 6.5% spread; second prefill 556 ms against 546 ms.
- With `MLXCEL_PROFILE_TTFT` unset, the same binary emits no `[TTFT]` line and reports unchanged prefill and decode.
- Decode re-measured as a slope over `-n 40` and `-n 120` with both runs reaching the budget: 118.15 and 118.19 ms per token, `C` -4.6 and -1.4 ms. Reproduces #1539 to 0.3% and confirms no fixed cost inside the decode loop.
- nsys reconciliation per the baseline record's rule 5: profiled 16.93 s against unprofiled 15.77 s, 107.4%.
- The cold PTX cache arm moved the cache aside and restored it; the 26-module cache is intact afterwards.

## 5. Validation Limits and Follow-up

### 5.1 A new method rule, added because breaking it produced a wrong number

The first measurement of this program returned 79.22 s for a 2-token first token against a true 15.54 s, and a later one returned 55.54 s. Both were cold page cache: this host's cgroup caps memory at 64 GB while the five checkpoints total 79 GB, so a sweep across models evicts whatever it is not reading. Mechanism 1 makes the reason direct, since the first token reads the whole language model off disk.

The baseline record gains rule 7: state the file page-cache state, and for a first-token number state it first. The effect is 5.1x against the PTX cache's 1.7x, so it is the larger of the two caches to control.

### 5.2 The confirming test #1538 left open: direction yes, magnitude no

The record predicted a short-prompt first token near 13 s rather than near 25 s. Measured on this branch: 15.54 s at 2 tokens against 25.60 s at 46 tokens in the same session. The direction is confirmed decisively and the value is 20% above 13 s, on a build carrying #1539 and two other changes the audit did not have. The 46-token rung reproduces the record's 54-token 24.94 s once prompt length is accounted for, so prefill itself has not moved.

### 5.3 The Volta-specificity verdict, and what it does not rest on

**General.** The argument does not rest on a device comparison, which is why it could be settled here:

- The dominant term, 78% of the first token, is a page-cache read plus a pageable host-to-device copy. Neither depends on compute capability.
- The cost tracks bytes of weights, 0.60 to 1.02 GB/s of language model across three checkpoints, with the Gemma pair holding tensor count fixed.
- Which phase pays is a loader property (3.4).
- The only per-architecture compilation in the path is 0.4% of a warm first token.
- The one architecture-dependent branch gives Volta the same node budget as A100 and a looser byte budget than GB10, and moving it 20x changes nothing.
- Removing CUDA graphs entirely changes nothing, so no arch-conditional graph behavior can matter either.

The one device-dependent term points the other way: the 4.19 s copy runs at 3.61 GB/s, the pageable PCIe Gen3 x16 rate, and GB10 is a coherent-memory part where those bytes never cross PCIe. The same defect should be smaller there, not absent and not different in kind, because the 7.40 s host staging half is a `read` and a `malloc` and is paid on any host.

Follow-up filed as **#1564**, outside epic #1536, carrying three candidates with measured sizes: pinned staging for the 4.19 s leg, copying straight from the mmap to remove the 7.40 s leg, and a deliberate uniform policy on which phase is charged.

### 5.4 The SASS-diff technique does not apply here

#1539 proved a guard could not reach Ampere by compiling at sm_80 and sm_121 and diffing `cuobjdump --dump-sass`; #1541 found the technique does not transfer for a host-dispatch-only translation unit; #1544 found it does for an AOT CUTLASS one. Neither case applies. This PR touches Rust host code and Markdown, adds no `__device__` or `__global__` code and no dispatch a compute capability could select between, so no cubin changes and the comparison would be of identical inputs.

### 5.5 Deferred to GB10

Two acceptance criteria stay unticked, and the deferral is narrower than the issue assumed because the Volta-specificity question was answered locally. What still needs sm_80 or later, each with the command and the predicted result written into the record: the size of the host-to-device leg on a coherent-memory part (`cudaMemcpyAsync` should keep its 1,847 calls, since that count is a property of the checkpoint, and lose most of its 4.19 s); whether the host staging leg is still 7.40 s on Grace's memory subsystem, which the GPU-free microbenchmark can answer alone; and whether `event_signal_kernel` is still one per weight, which tests whether the fence pattern of 3.6 is present there.

### 5.6 Pre-existing test failures, isolated rather than assumed

Two failures on this host are unrelated to this change and match already-filed issues: `tests/cuda_qmm_determinism.rs` at prefill (#1558) and `sampling::tests::temperature_one_support_unchanged` on 1 ULP (#1563).

## References

- Issue #1545, and epic #1536 whose Phase 4 this is.
- `docs/benchmark_results/volta-ttft-fixed-cost-2026-09-01.md`, the full record.
- `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md` (#1538), the baseline this measures against.
- #1539, whose `qmv` float accumulators set the decode rate reproduced here.
- #1564, the follow-up this work's verdict required.
- #818, the CUDA graph cache capacity default.
- `mlx/backend/cuda/load.cpp`, `mlx/backend/cuda/device.cpp`, `mlx/backend/cuda/event.cu`, `mlx/backend/cuda/jit_module.cpp`, `mlx/transforms.cpp`.
- `src/lib/mlxcel-core/src/weights.rs`, `src/models/sanitize.rs`, `src/lib/mlxcel-core/src/lib.rs` (`ensure_persistent_ptx_cache`).
