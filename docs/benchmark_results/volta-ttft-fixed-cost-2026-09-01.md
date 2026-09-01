# Volta (sm_70): what the fixed first-token cost is actually made of

Issue #1545, phase 4 of epic #1536. The baseline record `volta-sm70-baseline-2026-08-31.md` measured a large prompt-independent term in front of the first token and left it named but not explained. This document explains it, and the explanation is not the one the issue was filed on.

**The short version.** On `qwen3.8-27B-4bit` at a 2-token prompt with everything warm, the first token arrives 15.54 s after the model is reported loaded. A second prefill of the same prompt in the same process takes 0.55 s. So 14.99 s of it, 96.5%, is one-time process cost, and 12.08 s of that one-time cost, 78% of the whole first-token time, is moving the language model's 15.13 GB of weights from the page cache to the device. The model was never loaded when `Model loaded in 0.98s` printed: the line right after it says `resident: 0.00 GB`. CUDA graph instantiation, which the issue title names, is 0.10 s. Turning CUDA graphs off entirely changes the first-token time by less than the run-to-run spread.

## Environment

Everything from the baseline record's environment table, plus:

| Item | Value |
|------|-------|
| **mlxcel** | 0.6.0 on `perf/issue-1545-ttft-fixed-cost`, branched from `e5cae856`, which carries #1537, #1538, #1539, #1541 and #1544 |
| **Build** | `MLX_CUDA_ARCHITECTURES=70 make release-cuda`. `cuobjdump --list-elf` on the MLX archive: 96 cubins, all `sm_70`, nothing else |
| **PTX cache** | warm, `~/.cache/mlxcel/cuda-ptx/9a795735ad9a`, 26 modules, unless a row says cold |
| **File page cache** | warm unless a row says cold. This is a new control and it matters more than the PTX cache; see "The page cache is the third cache" |
| **Contention** | none, `nvidia-smi --query-compute-apps` empty before every run |

The instrumentation added for this issue is `MLXCEL_PROFILE_TTFT`. It prints one `[TTFT]` line per `--profile` generation splitting the pre-first-token phase into `setup`, `build` (MLX is lazy, so this is host-side graph construction with the device idle), `sample`, `eval` and `post`, plus the reported prefill and the unattributed residual. `MLXCEL_PROFILE_PIPELINE` covers only the decode loop, which is why the pre-first-token phase had no instrumentation before.

## Method

The baseline record's six rules apply unchanged and two of them do real work here. Rule 5, that nsys absolute times are only trustworthy when they reconcile against an unprofiled wall clock: the profiled first token is 16.93 s against 15.77 s unprofiled in the same configuration, a ratio of 107.4%, which is small enough to carry the attribution below. Rule 6, that percentage shares do not compare across runs: every comparison here is of absolute time and of instance counts, and the instance counts are stated because several of them are exact structural constants rather than measurements.

Two additions:

**The fixed cost is measured directly, not extrapolated.** A prefill ladder gives an intercept, but prefill on this part is not affine in prompt length at the short end: the marginal rate between 2 and 46 tokens is 223 ms per token while between 331 and 601 tokens it is 114, so a least-squares intercept over the whole ladder overshoots. The fit below gives 19.61 s; the direct measurement gives 14.99 s. The direct measurement is the number this document uses, and it is obtained by running the same prompt twice in one process: the first prefill pays every one-time cost, the second pays none of them, and the difference is the one-time cost with nothing extrapolated.

**Page-cache state is stated on every row.** See its own section for why.

## The confirming test #1538 left open

The baseline record reconciled the audit's ~13 s against its own 24.94 s arithmetically and said plainly that the confirming test had not been run: measure the first token at a 3-token prompt, warm cache, model already loaded, and check that it lands near 13 s rather than near 25 s. Run, on this branch:

| Prompt tokens | First token after load | Reps |
|---|---|---|
| 2 | **15.54 s** | 10, spread 6.5% |
| 46 | 25.60 s | 2 |
| 54 (the record's rung, chat template applied) | 25.58 s | 2 |

**The direction is confirmed and the magnitude is not.** A short prompt lands nowhere near 25 s, which is what the record predicted and what the reconciliation depended on. It lands at 15.5 s rather than 13 s, 20% above the audit's figure, on a build that carries #1539 and two other changes the audit did not have. The 46-token rung reproduces the record's 54-token 24.94 s to within its own prompt-length difference, so prefill itself has not moved.

What this changes about the record's arithmetic: the reconciliation subtracted a marginal prefill rate from each figure to get 12.60 s against 12.66 s. That subtraction assumes prefill is affine down to zero tokens, and the ladder below shows it is not, so the agreement to 0.5% is tighter than the method can support. The conclusion the record drew from it is still right, and is now measured rather than inferred: neither figure was wrong, and quoting either without its prompt length was.

## The prefill ladder, re-measured on this branch

`-n 1 --profile`, no chat template, two repetitions per rung, warm PTX cache, warm page cache.

| Prompt tokens | Prefill (s) | Reps | Segment rate |
|---|---|---|---|
| 2 | 15.766 | 15.826 / 15.706 | |
| 46 | 25.595 | 25.746 / 25.444 | 4.48 tok/s |
| 136 | 40.810 | 41.265 / 40.355 | 5.91 tok/s |
| 331 | 62.195 | 62.247 / 62.143 | 9.12 tok/s |
| 601 | 92.909 | 93.019 / 92.799 | 8.79 tok/s |

Least squares over the five rungs: **124.75 ms per prompt token, 8.02 tok/s marginal**, intercept 19.61 s. The marginal rate reproduces the baseline record's 125.07 ms per token to 0.3%, so #1539, #1541 and #1544 left prefill alone, which is what they each predicted. The intercept does not reproduce as well (19.61 s against 18.41 s) and should not be trusted to: the residuals are -4.10, +0.25, +4.23, +1.29 and -1.68 s, a clear curve rather than scatter, and the 2-token rung is the one furthest below the line.

## The breakdown

`qwen3.8-27B-4bit`, 2-token prompt, `-n 1`, warm PTX cache, warm page cache. The two columns are two independent measurements of the same run, and they are kept apart on purpose: the process split says how much of the first token is one-time, and the attribution says what that one-time work is.

**The process split.**

| Quantity | Time |
|---|---|
| First prefill in the process (`mlxcel generate --profile`) | 15.536 s (10 reps, 14.957 to 15.965) |
| Second prefill of the same prompt in the same process (`mlxcel-bench-decode --warmup-tokens 1 -n 1`) | 0.546 s (4 reps) |
| **One-time cost** | **14.990 s, 96.5% of the first token** |

**The attribution.** nsys `cuda_api_sum` and `cuda_gpu_kern_sum` with `--cuda-graph-trace=node` on the same command, profiled total 16.93 s against 15.77 s unprofiled, 107.4%. The host staging row is not from nsys at all and is explained under it.

| Phase | Time | Share of the first token | Source |
|---|---|---|---|
| Host staging of 15.13 GB of weights: `malloc`, read, `free`, 1,847 tensors | 7.40 s | 47.6% | standalone microbenchmark, no GPU, 2.04 GB/s |
| Host-to-device copy of the same 15.13 GB: `cudaMemcpyAsync`, 1,847 calls | 4.19 s | 27.0% | nsys, 3.61 GB/s, pageable |
| Device allocation for those weights: `cudaMallocAsync`, 2,255 calls | 0.49 s | 3.2% | nsys |
| **Weight materialization, total** | **12.08 s** | **77.8%** | |
| CUDA graph construction, launch and teardown: add-node, instantiate, launch, update, destroy, add-child, add-deps | 0.97 s | 6.2% | nsys |
| GPU kernel execution, every kernel including 1,847 `event_signal_kernel` | 0.74 s | 4.8% | nsys `cuda_gpu_kern_sum` |
| `cudaDeviceSynchronize`, one call | 0.20 s | 1.3% | nsys |
| `cudaLaunchKernel`, 5,287 calls | 0.13 s | 0.8% | nsys |
| JIT module load: `cuModuleLoadDataEx`, 6 calls, warm cubin cache | 0.06 s | 0.4% | nsys |
| Everything else: MLX host op recording, fence waits, allocator bookkeeping | 1.36 s | 8.7% | residual |
| **Total** | **15.54 s** | | |

The single largest term is host work that no CUDA profiler can see, which is why the original audit's `cuda_api_sum` table could not close the gap and why the issue was filed against the largest row that table did show.

## What the new instrumentation says, and what it rules out

`MLXCEL_PROFILE_TTFT=1`, same command, three repetitions at a 2-token prompt and one at 46:

```
[TTFT] prompt=2  tok setup=0.32ms build=11.07ms sample=0.16ms eval=15693.46ms post=0.00ms | prefill=15704.69ms residual=0.00ms
[TTFT] prompt=2  tok setup=0.33ms build=11.07ms sample=0.16ms eval=15631.19ms post=0.00ms | prefill=15642.43ms residual=0.00ms
[TTFT] prompt=2  tok setup=0.32ms build=11.40ms sample=0.16ms eval=15905.82ms post=0.00ms | prefill=15917.39ms residual=0.00ms
[TTFT] prompt=46 tok setup=0.32ms build=11.64ms sample=0.18ms eval=25668.00ms post=0.00ms | prefill=25679.81ms residual=0.00ms
```

The residual is zero to the printed resolution in every run, so the named phases account for the reported prefill exactly rather than approximately.

Two things are ruled out by this and both were live hypotheses. **Host-side graph construction is not the problem**: `build` is the whole lazy forward, every MLX op recorded for a 64-layer hybrid model with the device idle, and it costs 11.1 ms, 0.07% of the first token. **It is also not a function of prompt length**: 11.07, 11.07 and 11.40 ms at 2 tokens against 11.64 ms at 46, because the recorded graph has the same shape either way. Generator setup, sampler construction and the post-eval work are microseconds. Everything this issue is about is inside one `eval` call, which is why the rest of this document is about what happens inside it.

## Mechanism 1: the weights are loaded inside the first token, not at model load

`load_weights_from_dir` (`src/lib/mlxcel-core/src/weights.rs`) calls MLX's `load_safetensors()`, which returns unevaluated arrays whose primitive is `Load`; the function's own doc comment says MLX "materializes tensors on demand". The demand arrives at the first `eval`, which is inside the prefill window. The CLI says so already and it is easy to read past:

```
Model loaded in 0.975s (resident: 0.00 GB, peak: 0.00 GB).
```

Zero bytes resident after a 15 GB model "loaded". By the end of the run the same process reports an MLX peak of 15.61 GB.

What each materialization costs is `Load::eval_gpu` in `mlx/backend/cuda/load.cpp`: one device allocation, one `malloc` of the whole tensor, one **synchronous host read** into it, one host-to-device `cudaMemcpyAsync` out of that pageable buffer, and a `cudaLaunchHostFunc` to free it. The host read is not a CUDA call and does not appear in `cuda_api_sum`.

The count is exact rather than approximate, and that is what makes the attribution safe. `qwen3.8-27B-4bit`'s index holds 2,180 tensors: 1,847 under `language_model.` and 333 under `vision_tower.`. A text-only run touches the language model and nothing else. The audit measured 1,847 `cudaMemcpyAsync` calls, #1538 re-measured 1,847 on a different build, this document measures 1,847 at `-n 1`, `-n 8` and `-n 40`, and the checkpoint has exactly 1,847 language-model tensors. One `cudaMemcpyAsync` per weight, flat in the number of generated tokens because it happens once.

The host half was measured on its own, with a small C program that replicates `Load::eval_gpu`'s host path over the same 1,847 tensors at their real offsets and touches no CUDA at all: `malloc` 0.002 s, read 6.763 s, `free` 0.634 s, 7.40 s total for 15.13 GB, 2.04 GB/s, reproducible to 0.2% across runs. `malloc` is free because glibc hands back a fresh mapping; the cost lands in the read, which is a copy out of the page cache plus a first-touch page fault on every destination page, and in the `free`, which unmaps 15 GB.

## Mechanism 2: which phase pays is a loader property, not an architecture property

Three checkpoints, warm page cache, warm PTX cache, same command:

| Checkpoint | Language-model bytes | Tensors | Model load | Resident after load | First prefill | Second prefill |
|---|---|---|---|---|---|---|
| `gemma-4-12B-it-4bit` | 6.70 GB | 1,324 | 8.87 s | **6.28 GB** | 2.449 s | 0.191 s |
| `gemma-4-12b-it-8bit` | 12.65 GB | 1,324 | 20.96 s | **11.85 GB** | 3.134 s | 0.128 s |
| `qwen3.8-27B-4bit` | 15.13 GB | 1,847 | 0.98 s | **0.00 GB** | 15.387 s | 0.562 s |

The Gemma 4 family loader ends with `eval_all` followed by `detach_all` over every loaded array (`load_gemma4_family_weights_with_backing`, `src/models/sanitize.rs`), so its weights are resident when load returns and its first token costs seconds rather than tens of seconds. The Qwen 3.5 loader does not, so the same work lands in prefill. The two Gemma arms hold the tensor count fixed at 1,324 and vary the bytes by 1.89x, and the load time moves 2.36x with the bytes, which is the controlled version of the claim that this cost is driven by bytes and not by per-tensor overhead.

**This is an argument for measuring the total, not for moving the cost.** Per byte of language model, the eager Gemma path runs at 0.76 and 0.60 GB/s while the lazy Qwen path runs at 1.02 GB/s. Relabelling Qwen's 12 s from "prefill" to "model load" would make the `--profile` output honest and would move a server's cost from its first request to its startup, but on this evidence it would not make anything faster, and the Gemma path's numbers include model construction and repacking that Qwen's do not, so even the direction is not established. That is a follow-up with its own measurement, not a fix to land here.

## Mechanism 3: one cross-stream fence per weight

MLX inserts a fence whenever an evaluated array's primitive runs on a different stream from one of its inputs (`mlx/transforms.cpp`). mlxcel loads weights on whatever stream is default at load time and then, inside the generator, installs a dedicated per-thread generation stream (`install_thread_local_default_stream`). The forward therefore runs on a different stream from every `Load`, so every weight is a cross-stream input and gets a fence.

The count confirms it: `event_signal_kernel` fires **1,847 times at `-n 1`, 1,847 at `-n 8` and 1,847 at `-n 40`**, exactly flat, exactly the weight count, and exactly what #1538 measured at `-n 24`. `Fence::update` calls `AtomicEvent::signal(Stream, value)`, which opens with `encoder.commit()` before launching a one-thread kernel, so each of those fences also closes and commits a CUDA graph.

Its direct cost is small: 0.16 s of GPU time and a share of the 0.97 s spent in graph APIs. It is recorded because it explains the graph traffic, and because it is the part of this path that is specific to how mlxcel drives MLX rather than to MLX or to the device.

## Graph re-instantiation: how many distinct graphs, and why

`cudaGraphInstantiate` against generated tokens, same prompt, same model:

| `-n` | `cudaGraphInstantiate` | `cudaGraphExecUpdate` | `cudaGraphAddKernelNode` | `cudaGraphLaunch` | `event_signal_kernel` |
|---|---|---|---|---|---|
| 1 | 103 | 469 | 8,320 | 572 | 1,847 |
| 8 | **196** | 1,379 | 25,870 | 1,575 | 1,847 |
| 40 | **196** | 5,859 | 104,558 | 6,055 | 1,847 |

**Instantiation saturates at 196 and stays there.** There are 196 distinct graph shapes in this model's decode and prefill, each instantiated exactly once; every later commit hits MLX's graph-exec cache and takes the `cudaGraphExecUpdate` path, which grows linearly with generated tokens as it should. 196 is the same count #1538 measured. There is no re-instantiation to explain, and no thrashing: the cache holds 2,000 entries by mlxcel's default (`hardware::cuda_graph_cache_default`, #818) against a working set of 196, a factor of ten of headroom.

The 72,539 add-node calls against 4,387 launches that the issue asked about are not a fixed cost and not a pathology. Add-node runs on every commit whether or not the exec is cached, so it tracks kernels launched: 2,459 nodes per generated token from the `-n 8` to `-n 40` slope, and roughly 5,800 for prefill plus the first token. At `-n 1` the whole graph-API group costs 0.97 s of a 15.54 s first token.

Why so many commits for so few nodes: `get_graph_limits` in `mlx/backend/cuda/device.cpp` computes `cc = major * 100 + minor * 10` and switches on it, enumerating 800, 900, 1000, 1200 and 1210. A V100 is 700 and matches none, so it takes the fall-through of 20 nodes or 100 MB per graph, and mlxcel does not override it because `apply_metal_ops_per_buffer_default` only sets `MLX_MAX_OPS_PER_BUFFER` on Apple Silicon. On top of that, every fence commits.

Being on an unenumerated path sounds like a Volta problem and is not one. A100 is enumerated and gets the same 20 nodes; GB10 is enumerated and gets 20 nodes with a 25 MB byte budget, four times tighter than Volta's. And the knob does nothing here anyway, which was measured rather than assumed.

## Two knobs that do nothing, measured

Same command, 2-token prompt, `-n 1`, two repetitions per arm. The default configuration's own repeat spread over ten runs is 6.5%, which is the band every one of these differences sits inside.

| Arm | First token | Against default |
|---|---|---|
| `MLX_USE_CUDA_GRAPHS=1` (default) | 15.64 s | |
| `MLX_USE_CUDA_GRAPHS=0` | 15.36 s | -1.8% |
| `MLX_MAX_OPS_PER_BUFFER=20` (MLX's default here) | 15.50 s | |
| `MLX_MAX_OPS_PER_BUFFER=100` (what H100 and B200 get) | 15.46 s | -0.2% |
| `MLX_MAX_OPS_PER_BUFFER=400` | 15.03 s | -3.0% |

**Disabling CUDA graphs entirely does not measurably change the first-token time.** That is the cleanest available refutation of this issue's own title. Graph construction is 6.2% of the first token by direct attribution, and removing all of it recovers less than the noise, because the host time it saves is spent launching the same kernels one at a time instead.

## The JIT cache: confirmed, and it caches SASS rather than PTX

The cache directory is `~/.cache/mlxcel/cuda-ptx/<mlx-commit>`, here `~/.cache/mlxcel/cuda-ptx/9a795735ad9a`, pointed at `MLX_PTX_CACHE_DIR` by `ensure_persistent_ptx_cache` (`src/lib/mlxcel-core/src/lib.rs`) and keyed on the pinned MLX commit so it survives rebuilds.

It is working, and the files are not what the name says. `compiler_supports_device_sass` in `mlx/backend/cuda/jit_module.cpp` returns true for any NVRTC 12 or later, so on this host NVRTC emits a cubin, not PTX, and MLX writes it out under a `.ptx` extension. All 26 files in the cache are `ELF 64-bit LSB executable, NVIDIA CUDA architecture`, and `cuobjdump --list-elf` reports `sm_70` for every one of them and nothing else. `cuModuleLoadDataEx` is therefore loading finished SASS on a warm cache, with no driver JIT to pay.

Cold against warm, `qwen3.8-27B-4bit`, 2-token prompt, `-n 1`, cache moved aside and restored:

| Arm | First token | `cuModuleLoadDataEx` | Modules written |
|---|---|---|---|
| Cold cache (profiled) | 28.94 s | 48.8 ms over 6 calls | 6 |
| Warm cache (profiled) | 16.93 s | 61.3 ms over 6 calls | 0 |
| Warm cache (unprofiled) | 14.96 s | | 0 |

**The module load costs the same either way.** The same 6 calls take 49 ms cold and 61 ms warm, a difference inside the run-to-run spread, because both arms are loading a cubin. What the cache saves is the NVRTC compilation that produces it, which is not a CUDA runtime API and appears in the wall clock only: **12.0 s for 6 modules**, profiled against profiled.

This settles the audit's 848 ms over 13 calls against #1538's 172 ms over the same 13. Neither is a warm-cache figure for this model: a warm 6-module text-only run costs 61 ms, and a partially warm cache is the only way to reach hundreds of milliseconds. It also answers the audit's puzzle that two consecutive identical runs took the same 33.00 s: they should, because on a warm cache module loading is 0.4% of the first token and invisible at that resolution.

The cold run wrote exactly 6 modules. The 26 in the cache accumulate across the five checkpoints in this program, so a per-model cold cost of 12 s is the right figure and the full 26-module cost is not reachable in one run.

## The page cache is the third cache

The baseline record's rule 3 says to state the PTX cache state. This program needs a rule 7, and the effect is larger.

| Arm | First token, `qwen3.8-27B-4bit`, 2-token prompt |
|---|---|
| Warm page cache, warm PTX cache | 15.54 s |
| Cold PTX cache, warm page cache | 28.94 s (profiled), a 1.7x penalty |
| Cold page cache, warm PTX cache | **79.22 s**, a **5.1x** penalty |
| Checkpoint evicted mid-suite | 55.54 s |

Both of those slow rows were produced accidentally, by running the suite in an order that evicted the checkpoint. This host's cgroup caps memory at 64 GB and the five checkpoints in `models/mlx-community` total 79 GB, so a sweep across models evicts whatever it is not currently reading, and a run that follows one lands on a cold file. The mechanism is direct: mechanism 1 says the first token reads the whole language model off disk, so the first token is a disk benchmark whenever the page cache is cold.

Practically: warm the checkpoint (`cat model*.safetensors > /dev/null`) immediately before any first-token measurement, and say in the record that you did. A TTFT number quoted without page-cache state can be wrong by a factor of five, which is worse than quoting one without PTX cache state.

## Is this Volta-specific? No.

The issue named this as the branch point for its own scope, so it is answered explicitly.

**General.** The dominant term, 78% of the first token, is a host read of the checkpoint plus a pageable host-to-device copy. Neither depends on compute capability. Both depend on the loader that owns the checkpoint (mechanism 2), the host's page cache, and the link to the device.

The evidence, in the order it was collected:

- The cost tracks bytes of weights, not architecture. Across three checkpoints on one device it runs at 0.60 to 1.02 GB/s of language model, and the two Gemma arms hold tensor count fixed while varying bytes by 1.89x and move 2.36x.
- The one part of the path that is compiled per architecture, the JIT modules, is 0.4% of a warm-cache first token.
- The one part where Volta takes an architecture-dependent branch, `get_graph_limits`, gives it the same node budget as A100 and a looser byte budget than GB10, and moving that knob across a 20x range changes nothing measurable.
- Removing CUDA graphs entirely changes nothing measurable, so no arch-conditional graph behavior can matter either.
- The fence-per-weight pattern comes from mlxcel installing a generation stream after loading weights on the default stream. That is host-side and architecture-independent.

**What is device-dependent is the size of the H2D term, and it points the other way.** The 4.19 s copy runs at 3.61 GB/s, which is pageable host-to-device over PCIe Gen3 x16. GB10 is a coherent-memory part, so the same 15.13 GB would not cross a PCIe link at all and that term would shrink or vanish. The defect would be smaller there, not different in kind, and certainly not absent: the 7.40 s host staging half is a `read` and a `malloc` and would be paid on any host.

**Consequence for epic #1536.** This is not a Volta decode problem and does not belong to the epic's kernel-math track. It surfaced on Volta because a V100 makes every second of it visible next to a decode loop that is itself slow. The follow-up is filed as **#1564**, outside #1536, as a general first-token-latency issue in the loader and weight-materialization path, and it carries the three candidates named below with their measured sizes.

## Decode is unaffected, and carries no fixed cost of its own

Re-measured on this branch so the record has a same-build number, `qwen3.8-27B-4bit`, 85-token prompt that generates past 120 tokens, slope over `-n 40` and `-n 120`, both runs reaching the budget:

| Rep | ms/token | Fixed decode cost `C` |
|---|---|---|
| 1 | 118.15 | -4.6 ms |
| 2 | 118.19 | -1.4 ms |

118.17 ms per token reproduces #1539's 117.83 to 0.3%, and `C` is zero within measurement error. **All of the fixed cost this issue is about sits in front of the first token, and none of it is inside the decode loop.** That is worth stating because the audit derived its ~13 s from a fit over total time, which cannot tell the two apart.

## What was not changed, and why

No fix landed. Every candidate the issue named was measured and none of them is where the time is:

- **CUDA graph instantiation**, the issue's title: 0.10 s of a 15.54 s first token, 196 instantiations for 196 distinct graphs with a cache ten times larger than it needs. Disabling graphs entirely recovers nothing.
- **The 72,539 add-node calls**: a per-launch cost proportional to generated tokens, 0.66 s at `-n 1`, not a fixed cost.
- **`cuModuleLoadDataEx` and the JIT cache**: 0.06 s warm. The cache is working, is arch-correct, and stores SASS.
- **`MLX_CUDA_GRAPH_CACHE_SIZE` thrashing**: 196 shapes against a 2,000 capacity.
- **`cudaMemcpyAsync`**: the largest CUDA row, and it turned out to be the weight upload rather than a per-step readback. It is real, it is 27% of the first token, and it is not reducible without changing how weights are staged.

What would reduce it is out of this issue's scope and needs its own measurement, and is filed as #1564: staging weights through pinned memory instead of `malloc` (the H2D leg runs at 3.61 GB/s, the pageable rate), or copying straight from the mmap so the 7.40 s host staging leg disappears, both of which are MLX changes in `Load::eval_gpu`; and deciding deliberately, per loader, whether weight materialization is charged to load or to the first token, which is an mlxcel change with a memory-peak consequence. Each is a real candidate with a measured size attached, which is more than the issue started with.

## Deferred to GB10

No sm_80 or later device exists on this host. Epic #1536's `## GB10 (sm_121) continuation` section lists this issue's Volta-specificity question, and it is answered here on local evidence, so what is deferred is narrower than the issue assumed:

- **Answered locally, no GB10 run needed**: whether the cost is Volta-specific. It is not, and the argument does not rest on a comparison. It rests on the composition of the cost, which is 78% host staging and PCIe transfer, on the JIT cache being 0.4% of it, and on both arch-dependent knobs being measurably inert.
- **Still needs GB10**: the size of the host-to-device leg on a coherent-memory part. The prediction is that `cudaMemcpyAsync` keeps its 1,847 calls, since that count is a property of the checkpoint, and that its total time falls well below 4.19 s. Measure it with `nsys profile -t cuda --cuda-graph-trace=node` on `qwen3.8-27B-4bit` at a short prompt with `-n 1`, and report `cudaMemcpyAsync` total time and call count alongside the reported prefill.
- **Still needs GB10**: whether the host staging leg is the same 7.40 s there. It is a `malloc` plus a page-cache read, so it should scale with the host's memory bandwidth and not with the GPU, but Grace's memory subsystem is different enough from this Xeon that it is worth a number rather than an assumption. The microbenchmark used here is self-contained and needs no GPU.
- **Still needs GB10**: whether `event_signal_kernel` is still 1,847 instances, which tests whether the fence-per-weight pattern is present there too.

## Reproduce

Assumes a Volta-class part, this branch, and `./models/mlx-community/` holding the checkpoints.

```bash
# 0. Build, single architecture, explicit.
MLX_CUDA_ARCHITECTURES=70 make release-cuda

# 1. Warm the page cache for the checkpoint under test. Do this before every
#    first-token measurement; a cold file costs 5x.
cat ./models/mlx-community/qwen3.8-27B-4bit/*.safetensors > /dev/null

# 2. The first token, and the phase breakdown the instrumentation prints.
MLXCEL_PROFILE_TTFT=1 ./target/release/mlxcel generate \
  -m ./models/mlx-community/qwen3.8-27B-4bit --no-chat-template -p "Hi." -n 1 --profile

# 3. The same prefill a second time in one process. The difference against
#    step 2 is the one-time cost, with nothing extrapolated.
./target/release/mlxcel-bench-decode \
  -m ./models/mlx-community/qwen3.8-27B-4bit --no-chat-template -p "Hi." \
  --warmup-tokens 1 -n 1

# 4. Attribution. --cuda-graph-trace=node is mandatory. Reconcile the profiled
#    prefill against step 2 before reading any absolute number.
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o ttft_warm_n1 \
  ./target/release/mlxcel generate -m ./models/mlx-community/qwen3.8-27B-4bit \
  --no-chat-template -p "Hi." -n 1 --profile
nsys stats --report cuda_api_sum --report cuda_gpu_kern_sum --format csv ttft_warm_n1.nsys-rep

# 5. Distinct graph count. Instantiate saturates; update grows with -n.
for n in 1 8 40; do
  nsys profile -t cuda --cuda-graph-trace=node -o g_$n \
    ./target/release/mlxcel generate -m ./models/mlx-community/qwen3.8-27B-4bit \
    --no-chat-template -p "Hi." -n $n --profile
  nsys stats --report cuda_api_sum --format csv g_$n.nsys-rep \
    | grep -E 'cudaGraphInstantiate|cudaGraphExecUpdate|cudaGraphAddKernelNode'
done

# 6. The two inert knobs.
MLX_USE_CUDA_GRAPHS=0 ./target/release/mlxcel generate -m ... -n 1 --profile
MLX_MAX_OPS_PER_BUFFER=400 ./target/release/mlxcel generate -m ... -n 1 --profile

# 7. Cold JIT cache. Move it aside, do not delete it.
mv ~/.cache/mlxcel/cuda-ptx/9a795735ad9a{,.saved}
# ... one run ...
rm -rf ~/.cache/mlxcel/cuda-ptx/9a795735ad9a && mv ~/.cache/mlxcel/cuda-ptx/9a795735ad9a{.saved,}

# 8. The host staging leg on its own, no GPU. Build a manifest of the
#    language_model.* tensors from the safetensors headers, then in C, for
#    each one, malloc(nbytes), pread at its absolute offset, free. That is
#    exactly what Load::eval_gpu does before it calls cudaMemcpyAsync, and it
#    is the largest single row in the breakdown above.
python3 - <<'EOF'
import json, struct, glob, os
d = './models/mlx-community/qwen3.8-27B-4bit'
for shard in sorted(glob.glob(d + '/*.safetensors')):
    with open(shard, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    for k, v in hdr.items():
        if k == '__metadata__' or not k.startswith('language_model'):
            continue
        a, b = v['data_offsets']
        print(os.path.abspath(shard), 8 + n + a, b - a)
EOF
```

## References

- Epic #1536, and the baseline record `volta-sm70-baseline-2026-08-31.md` this measures against. Its `## GB10 (sm_121) continuation` section holds what could not be settled on this machine.
- #1538, which produced that record and left the confirming test above open.
- #1539, whose `qmv` float accumulators set the decode rate reproduced here.
- Graph cache capacity default and its rationale: `src/lib/mlxcel-core/src/hardware.rs`, issue #818.
- Weight loading: `src/lib/mlxcel-core/src/weights.rs`, `mlx/backend/cuda/load.cpp`.
- Eager materialization precedent in-tree: `load_gemma4_family_weights_with_backing`, `src/models/sanitize.rs`.
- JIT cache: `ensure_persistent_ptx_cache` in `src/lib/mlxcel-core/src/lib.rs`, `mlx/backend/cuda/jit_module.cpp`.
- Graph commit limits and fences: `mlx/backend/cuda/device.cpp`, `mlx/backend/cuda/event.cu`, `mlx/transforms.cpp`.
- #1564, the follow-up this document's verdict required.
- Related host-overhead work from the GB10 program: #632, #633.
