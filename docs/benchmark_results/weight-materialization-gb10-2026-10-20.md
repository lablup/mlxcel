# Weight materialization: pinned staging and eager load (GB10)

Issue #1564, the follow-up that `volta-ttft-fixed-cost-2026-09-01.md` filed. That record found that 78% of a short-prompt first token on a V100 was the language model's weights moving from the page cache to the device inside the first `eval`, and named three separable candidates: the host-to-device leg of `Load::eval_gpu`, its host staging leg, and whether a loader charges materialization to model load or to the first token. This record measures all three on GB10, before and after.

**The short version.** On `qwen3.8-27B-4bit` at a 2-token prompt, warm page cache and warm JIT cache, the first token falls from **4.02 s to 0.67 s** and load plus first token from **4.37 s to 2.24 s** (-49%). The pinned staging slots in `Load::eval_gpu` remove 2.15 s of the 4.37 s; making materialization eager moves the remaining 1.25 s of weight upload from the first token to model load, which changes where the time is reported and not how much there is. Decode is unchanged (75.6 against 76.1 ms per token, inside the repeat spread) and peak memory moves by under 0.1 GB on this checkpoint and by 0.2 to 0.5 GB on a checkpoint whose loader repacks every expert.

## Environment

| Item | Value |
|------|-------|
| **Host** | NVIDIA GB10 (sm_121), 20-core Grace (Cortex-X925 + Cortex-A725), 121.7 GB unified LPDDR5x, driver 615.71.09, CUDA 13.0, Linux 7.0.0-1019-nvidia |
| **Not measured here** | A Tesla V100 or any PCIe-attached device. The V100 figures below are quoted from the 2026-09-01 record, not re-measured |
| **Before** | `main` at `89ced0a7`, `MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda` |
| **After** | `perf/issue-1564-eager-weight-load`, same build command |
| **Checkpoint** | `mlx-community/Qwen3.8-27B-4bit`, downloaded for this record: 2,180 tensors, 1,847 under `language_model.` (15.13 GB) and 333 under `vision_tower.` |
| **Repack checkpoint** | `allenai/OLMoE-1B-7B-0924`, raw bf16, 3,219 tensors, 13.84 GB, every expert stored separately and stacked by the loader. Its `config.json` omits `rms_norm_eps`, which mlxcel requires on both branches, so the local copy was given the HF default `1e-5` |
| **Page cache** | warm on every row: `cat model*.safetensors > /dev/null` immediately before each run (0.6 s for 15 GB, so already resident) |
| **JIT cache** | warm on every row: `~/.cache/mlxcel/cuda-ptx/<pin>` populated by an earlier run of each binary. The first run on this host compiled the kernels and is excluded |
| **Contention** | none; no build or other GPU job during any measured row |

The largest checkpoints on this host, `DeepSeek-V2-Lite` (raw bf16, 30 GB), its FP8 variant and `openai/gpt-oss-20b` (original layout), do not load on `main` either (`Weight not found: model.layers.1.mlp.switch_mlp.gate_proj.weight` and equivalent), so the 15 GB `qwen3.8-27B-4bit` is the largest checkpoint measured here, and OLMoE stands in for the repacking loaders.

## What changed

1. **`Load::eval_gpu` (`src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/load.cpp`)** uploads through four reused 64 MiB pinned host slots per device instead of a pageable `malloc` per tensor. The read into one slot overlaps the copy out of the previous one, and the per-tensor `malloc`, the first-touch page faults on a fresh mapping and the `free` that unmapped the checkpoint are gone. A failed `cudaHostAlloc` prints one warning and leaves that process on the upstream pageable path; `MLXCEL_CUDA_PINNED_LOAD=0` selects that path explicitly.
2. **One materialization policy for every loader** (`src/lib/mlxcel-core/src/weight_materialize.rs`). The weights loaders register every tensor they return with a scope that `loading::load_model` and its adapter variants open around family construction. When the model is built, the scope evaluates and detaches, through `eval_all` and `detach_all`, every registered tensor except modality-tower tensors (`vision_tower.`, `visual.`, `audio_tower.` and similar names), which stay lazy until the first image or clip, and tensors nothing kept after construction. `MLXCEL_WEIGHT_MATERIALIZATION=lazy` restores first-use materialization.
3. **`MLXCEL_PROFILE_TTFT`** prints a `[TTFT] load` line for the scope and appends the `Load` upload counters to the existing `[TTFT]` line, so the host staging leg, which no CUDA profiler sees, is reported in-process next to the copy back-pressure.

## Method

`mlxcel generate -m <ckpt> --no-chat-template -p "Hi." -n 1 --profile` with `MLXCEL_PROFILE_TTFT=1`. Five arms, three repetitions each, run as two interleaved rounds (A B C D E, A B C D E), so six repetitions per arm. `Model loaded in` gives load, `Prefill:` gives the first token, and `/usr/bin/time` gives the process wall clock from exec to exit.

| Arm | Binary | Environment |
|---|---|---|
| A | before | none |
| B | after | `MLXCEL_WEIGHT_MATERIALIZATION=lazy MLXCEL_CUDA_PINNED_LOAD=0` (should reproduce A) |
| C | after | `MLXCEL_WEIGHT_MATERIALIZATION=lazy` (pinned staging only) |
| D | after | `MLXCEL_CUDA_PINNED_LOAD=0` (eager only) |
| E | after | none (the shipped default: eager and pinned) |

## First token and load, `qwen3.8-27B-4bit`

Medians of six, range in brackets.

| Arm | Model load (s) | First token (s) | Load + first token (s) | Process wall (s) | Resident after load |
|---|---|---|---|---|---|
| A before | 0.349 [0.328-0.353] | **4.018** [3.841-4.340] | **4.369** | 5.44 [5.16-5.84] | 0.00 GB |
| B after, lazy + pageable | 0.331 [0.321-0.351] | 4.053 [3.841-4.482] | 4.382 | 5.43 [5.24-5.79] | 0.00 GB |
| C after, lazy + pinned | 0.330 [0.326-0.353] | 1.900 [1.867-1.968] | 2.229 | 3.31 [3.23-3.47] | 0.00 GB |
| D after, eager + pageable | 1.950 [1.930-2.022] | 0.679 [0.672-0.697] | 2.635 | 3.70 [3.61-3.82] | 14.10 GiB |
| E after, eager + pinned | 1.577 [1.548-1.627] | **0.667** [0.660-0.685] | **2.242** | 3.30 [3.27-3.43] | 14.10 GiB |

B reproduces A to within 0.4%, so the new binary with both changes switched off is the baseline, and every difference below is attributable to one of the two switches.

**The same prefill a second time in one process** (`mlxcel-bench-decode --warmup-tokens 1 -n 1`, three repetitions per binary): 0.141 s before, 0.142 s after. The one-time cost in front of the first token is therefore 3.88 s before and 0.53 s after. What is left of the 0.53 s is not weight I/O (`eval weights=0` on every E row): it is JIT module loads, CUDA graph instantiation and the 1,847 cross-stream fences described below.

## The two legs, separately

In-process counters from `Load::eval_gpu` (`staging` = the host read into the staging buffer plus `malloc` on the pageable path; `h2d_wait` = host time blocked on a slot whose copy had not finished; `upload` = wall clock inside `Load::eval_gpu`), medians of six:

| Arm | Staging leg (s) | h2d_wait (s) | Upload total (s) | `cudaMemcpyAsync` calls |
|---|---|---|---|---|
| B lazy + pageable | 2.197 | 0 | 3.306 | 1,847 |
| C lazy + pinned | 0.885 | 0.047 | 1.195 | 1,865 |
| D eager + pageable | 0.948 | 0 | 1.618 | 1,847 |
| E eager + pinned | 0.901 | 0.044 | 1.238 | 1,865 |

The copy leg from nsys (`-t cuda,nvtx --cuda-graph-trace=node`, one run per arm; the profiled first token was 4.16 s for A, 2.01 s for C and 0.76 s for E, 3.6 to 13% above the unprofiled medians):

| Arm | `cudaMemcpyAsync` host API time | `[CUDA memcpy Host-to-Device]` device time | Calls |
|---|---|---|---|
| A before | 0.888 s | 0.261 s | 1,847 |
| C lazy + pinned | 0.018 s | 0.282 s | 1,865 |
| E eager + pinned | 0.016 s | 0.263 s | 1,865 |

On GB10 the DMA itself costs the same 0.26 s either way, about 58 GB/s. What the pageable path pays is host time: a pageable `cudaMemcpyAsync` is synchronous while the driver stages the buffer, 0.89 s of API time that the pinned path reduces to 0.02 s. This is the coherent-memory prediction of the V100 record (the copy leg shrinks well below its 4.19 s) with the refinement that on GB10 the surviving cost was the host-side staging copy, not the link.

The call count is 1,865 rather than 1,847 because a pinned upload issues one copy per 64 MiB chunk: the two tensors larger than a slot (the embedding and the untied `lm_head`, about 636 MB each) take ten copies each, 18 extra. The count is still a property of the checkpoint and of the slot size, and it is flat in `-n`.

**The host staging leg without a GPU**, the V100 record's microbenchmark rebuilt for this host: the 1,847 `language_model.*` tensors at their real offsets, warm page cache, three runs.

| Path | malloc | read | free | Total | Rate |
|---|---|---|---|---|---|
| Upstream: `malloc`, `pread`, `free` per tensor | 0.002 s | 2.40-2.66 s | 0.45-0.53 s | 2.85-3.20 s | 4.7-5.3 GB/s |
| mlxcel: `pread` into a reused, pre-faulted 64 MiB buffer | 0 | 0.80-0.89 s | 0 | 0.80-0.89 s | 17-19 GB/s |

The V100 host measured 7.40 s for the upstream path, so Grace runs it 2.3 to 2.6 times faster, and the reused buffer is a further 3.6 times faster on top. The in-process staging leg of arms C and E (0.89 to 0.90 s) matches the reused-buffer microbenchmark, as it should. Arm B's in-process 2.20 s against the microbenchmark's 2.4 to 2.7 s read excludes `free`, which runs on CUDA's host-callback thread. Arm D's pageable staging leg measured 0.95 s, less than half of arm B's, with the same code; why the pageable read is cheaper at load than inside the first `eval` was not investigated, and it does not affect the shipped default.

## Decode is unchanged

`mlxcel-bench-decode --ignore-eos --warmup-tokens 5`, a 45-token prompt, `-n 40` and `-n 120`, every run reaching its budget, two repetitions per binary. Slope = (decode at 120 - decode at 40) / 80.

| Binary | Rep 1 | Rep 2 |
|---|---|---|
| Before | 75.64 ms/token | 75.59 ms/token |
| After | 76.74 ms/token | 75.37 ms/token |

The difference of the means, 0.4 ms per token (0.6%), is smaller than the after binary's own repeat spread of 1.4 ms. Greedy output (`-n 24 --temp 0`) is byte-identical across A, E, D and the lazy arm, and an image request on the same checkpoint produces identical output on both binaries.

## Peak memory

| Checkpoint | Measure | Lazy | Eager |
|---|---|---|---|
| `qwen3.8-27B-4bit` | MLX peak after the first prefill (`mlxcel-bench-decode`, 3 runs) | 15.54 GB | 15.57-15.60 GB |
| `qwen3.8-27B-4bit` | Resident after load | 0.00 GB | 15.13 GB (14.10 GiB), 333 vision tensors left lazy |
| `OLMoE-1B-7B-0924` (repacks every expert) | MLX peak after the first prefill (2 runs) | 13.93 GB (before and after binaries) | 14.40 GB, 14.14 GB |

Eager materialization reads the sources, and a loader that repacks keeps its repacks lazy until the first token, so for one forward both a source and its repacked result can be live. MLX frees each source as soon as its repacked result is produced, so this costs a fraction of the repacked bytes rather than a second copy of them: 0.2 to 0.5 GB (1.5 to 3.4%) on a checkpoint where all 3,072 expert tensors are restacked, not the 2x a naive reading predicts. The 256 MiB of pinned staging slots is host memory outside MLX's allocator and is not in either column.

## Decision: weight materialization is charged to model load

Every family loaded through `load_model` and its adapter variants now materializes its text weights before `Model loaded` prints. The reasons, all measured above:

- `--profile` was reporting 3.35 s of weight upload as prefill on this checkpoint (12 s on the V100). The first token is now model compute plus JIT and graph setup only, and the new `eval weights=` field says so on every run.
- A server pays the upload at startup rather than on its first request.
- With pinned staging it costs nothing in total: E's load plus first token, 2.242 s, matches C's 2.229 s, the same work with the lazy policy.
- Text-only runs keep vision and audio towers lazy, so the eager policy holds no memory a lazy run would not.
- Peak memory is unchanged on a checkpoint without repacks and 1.5 to 3.4% higher on one that repacks every expert.

The V100 record's objection, that the eager Gemma 4 path loaded at 0.60 to 0.76 GB/s against the lazy path's 1.02 GB/s, does not carry over: here eager and lazy upload the same bytes at the same rate (E against C), and that Gemma 4 path builds arrays from its own mmaps instead of MLX `Load` primitives. It is unchanged by this issue and is eager either way.

Not covered by the scope, so still first-use: tensor-parallel loads (`--tp-size` > 1, not measurable on this one-GPU host), pipeline-parallel stages, and embedding and reranker loads.

## Still in the first token: one fence per weight

`event_signal_kernel` still fires 1,847 times on the first token in every arm, eager included, all of them after the last weight copy. The model holds many weights as lazy arrays derived from the `Load` arrays (copies and repacks recorded on the load-time stream), and those run on the generation stream at the first token, which is mechanism 3 of the V100 record. Eager materialization of the sources does not remove it. It is the next candidate inside the remaining 0.53 s and is not addressed here.

## Reproduce

```bash
# Build (cuDNN, OpenBLAS and LAPACK headers must be visible to CMake).
MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda --bin mlxcel --bin mlxcel-bench-decode

M=./models/mlx-community/qwen3.8-27B-4bit
cat $M/model*.safetensors > /dev/null      # warm the page cache before every run

# First token, with the load line and the upload fields.
MLXCEL_PROFILE_TTFT=1 ./target/release/mlxcel generate -m $M --no-chat-template -p "Hi." -n 1 --profile

# The other arms of the table.
MLXCEL_WEIGHT_MATERIALIZATION=lazy MLXCEL_CUDA_PINNED_LOAD=0 ./target/release/mlxcel generate ...
MLXCEL_WEIGHT_MATERIALIZATION=lazy ./target/release/mlxcel generate ...
MLXCEL_CUDA_PINNED_LOAD=0 ./target/release/mlxcel generate ...

# Second prefill in the same process, and the MLX peak after it.
./target/release/mlxcel-bench-decode -m $M --no-chat-template -p "Hi." --warmup-tokens 1 -n 1

# Copy leg on the device.
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o ttft ./target/release/mlxcel generate -m $M \
  --no-chat-template -p "Hi." -n 1 --profile
nsys stats --report cuda_api_sum --report cuda_gpu_mem_time_sum --format csv ttft.nsys-rep

# Decode slope.
for n in 40 120; do ./target/release/mlxcel-bench-decode -m $M --no-chat-template \
  -p "<45-token prompt>" --ignore-eos --warmup-tokens 5 -n $n; done
```

The host staging microbenchmark is the V100 record's step 8 (the same manifest script), with a second mode that reads each tensor in 64 MiB pieces into one buffer allocated and touched once before the loop.

## References

- #1564, and `volta-ttft-fixed-cost-2026-09-01.md` (#1545), which measured the V100 baseline and named the three candidates.
- `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/load.cpp`, `load_staging.h`.
- `src/lib/mlxcel-core/src/weight_materialize.rs`, `src/loading/materialize.rs`.
- `MLXCEL_WEIGHT_MATERIALIZATION`, `MLXCEL_CUDA_PINNED_LOAD` and `MLXCEL_PROFILE_TTFT` in `docs/environment-variables.md`.
