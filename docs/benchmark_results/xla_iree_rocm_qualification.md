# OpenXLA / IREE ROCm (`hip`) Backend Qualification Report (`gfx1151`)

- **Issue**: `#851` (`test(xla): qualify the OpenXLA/IREE engine on ROCm/HIP`)
- **Date**: 2026-10-14
- **Status**: Qualified (`local-task`, `local-sync`, and `hip` HAL drivers verified end-to-end)

---

## 1. Phase 0: Machine & Software Baseline

| Component | Recorded Value |
|---|---|
| Hostname / OS | `strix-halo` — Debian GNU/Linux 13.6 (`trixie`), kernel `6.18.12+deb13-amd64` (`x86_64`) |
| APU / CPU | `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S` (16 cores / 32 threads) |
| GPU ASIC | `AMD Radeon 8060S Graphics` — `gfx1151` (`amdgcn-amd-amdhsa--gfx1151`, RDNA 3.5, compute capability `11.5`, 20 WGP / 40 CU, wave32, max clock 2900 MHz) |
| GPU Memory | 98,304 MB (`96 GiB`) UMA carveout (`memory-total-global-memory-size: 98304 MB`, `memory-supports-managed-memory: 1`, 2 MiB L2) + 30.8 GiB GTT |
| NPU | `AMD IPU Device` (`0000:c7:00.1`, `aie2p` / `RyzenAI-npu5`, FW `1.1.0.320`, `/dev/accel/accel0`) |
| ROCm / HIP | `/opt/rocm` -> `/opt/rocm/core-10.0`, HIP version `7.15.26333-0000000`, `libamdhip64.so.7.15.26333-0000000` |
| Compilers | AMD clang `23.0.0git` (`8f497e0992fb7513f7f78a6f6b6f1056c375e961`), system GCC `14.2.0` (`Debian 14.2.0-19`), CMake `4.0.2`, Ninja `1.12.1` |
| Device Nodes | `/dev/kfd` (`crw-rw---- root:render`), `/dev/dri/card0` (`root:video`), `/dev/dri/renderD128` (`root:render`); user in `video` and `render` groups |
| Pinned IREE Version | `3.12.0rc20260721` (upstream commit `dc9601f88654749456c7cee4ae87e13de2654e1e`, LLVM `24.0.0git`) |
| IREE Wheel (`iree-compile`) | `iree_base_compiler-3.12.0rc20260721-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` (`sha256: 04b97c78f82200a28f3d9f16ec81fa8ac72e8431e410d7de06c3cb005e23ab86`) |

---

## 2. Phase 1: Toolchain Provisioning & Minimal StableHLO -> ROCm Smoke

### 2.1 Automated Setup (`scripts/iree/setup-rocm.sh` & `make iree-rocm`)

`scripts/iree/setup-rocm.sh` (invoked via `make iree-rocm` or auto-selected on ROCm hosts by `make iree`) provisions `/home/devops/.cache/mlxcel/iree-rocm-3.12.0rc20260721`:

1. Installs the pinned `iree-base-compiler==3.12.0rc20260721` wheel into `<IREE_ROCM_HOME>/venv` (supporting `python3 -m venv`, `uv venv`, or `python3 -m venv --without-pip` + `zipfile` extraction on Debian/Ubuntu hosts without `python3-venv` `ensurepip`) and verifies `iree-compile --iree-hal-list-target-backends` advertises `rocm`.
2. Clones `iree-org/iree` at `dc9601f88654749456c7cee4ae87e13de2654e1e`, initializes the required runtime submodules (`flatcc`, `cpuinfo`, `hip-build-deps`), and builds `iree_runtime_unified` and `iree-run-module` with:
   - `-DIREE_HAL_DRIVER_DEFAULTS=OFF`
   - `-DIREE_HAL_DRIVER_LOCAL_SYNC=ON`
   - `-DIREE_HAL_DRIVER_LOCAL_TASK=ON`
   - `-DIREE_HAL_DRIVER_HIP=ON`
   - `-DIREE_ROCM_TEST_TARGET_CHIP=""` (required by IREE's `CMakeLists.txt` when a ROCm GPU is present on the build host, even when `-DIREE_BUILD_TESTS=OFF`).
3. Emits build and runtime environment variables via `eval "$(scripts/iree/setup-rocm.sh --env)"`:
   - `IREE_ROCM_HOME`
   - `IREE_ROCM_COMPILE`
   - `MLXCEL_XLA_IREE_COMPILE`
   - `MLXCEL_XLA_ROCM_TARGET` (auto-detected from single-GPU `rocminfo` / `amdgpu-arch`, e.g. `gfx1151`)
   - `LD_LIBRARY_PATH` (includes `/opt/rocm/lib` when `libamdhip64.so` is present there; binaries also embed `DT_RPATH` `/opt/rocm/lib` via `rocm_rpath.rs`).

### 2.2 HAL Driver & Device Enumeration

```text
$ "$IREE_ROCM_HOME/build/tools/iree-run-module" --list_drivers
             hip: HIP HAL driver (via dylib)
      local-sync: Local execution using a lightweight inline synchronous queue
      local-task: Local execution using the IREE multithreading task system

$ "$IREE_ROCM_HOME/build/tools/iree-run-module" --dump_devices=hip
# --device=hip://GPU-58580000-0000-0000-0000-000000000000
#   AMD Radeon 8060S Graphics
- amdhip64_dylib_path: /opt/rocm/lib/libamdhip64.so
- gpu-compute-capability: 11.5
- gpu-arch-name: gfx1151
- launch-max-block-dims: (1024, 1024, 1024)
- block-max-thread-count: 1024
- block-max-32-bit-register-count: 196608
- block-max-shared-memory: 64 KB
- memory-is-integrated-memory: 0
- memory-supports-managed-memory: 1
- memory-total-const-memory-size: 2047 MB
- memory-total-global-memory-size: 98304 MB
- memory-l2-cache-size: 2097152 bytes
- gpu-compute-unit-count: 20
- gpu-compute-max-clock-rate: 2900 mHz
- gpu-memory-max-clock-rate: 1000 mHz
- gpu-warp-size: 32
```

### 2.3 Minimal StableHLO Modules (`5/5` Runs Each)

| Module | Operation | Target Flags | Expected Output | `hip` (`gfx1151`) Result |
|---|---|---|---|---|
| `add_f32.mlir` | `stablehlo.add` (`tensor<4xf32>`) | `--iree-hal-target-device=hip --iree-rocm-target=gfx1151` | `4xf32=11 22 33 44` | `4xf32=11 22 33 44` (5/5 PASS) |
| `matmul_f32.mlir` | `stablehlo.dot_general` (`2x3xf32 @ 3x2xf32 -> 2x2xf32`) | `--iree-hal-target-device=hip --iree-rocm-target=gfx1151` | `2x2xf32=[22 28][49 64]` | `2x2xf32=[22 28][49 64]` (5/5 PASS) |
| `matmul_f16.mlir` | `stablehlo.dot_general` (`2x3xf16 @ 3x2xf16 -> 2x2xf16`) | `--iree-hal-target-device=hip --iree-rocm-target=gfx1151` | `2x2xf16=[22 28][49 64]` | `2x2xf16=[22 28][49 64]` (5/5 PASS) |

---

## 3. Phase 2: Failure-Mode Diagnostics & Single-Sequence `mlxcel generate`

### 3.1 Deliberate Failure-Mode Diagnostics

The C shim (`src/lib/mlxcel-xla/csrc/xla_iree.c`) formats full multi-line IREE statuses via `iree_status_fprint` and annotates errors by phase (`[compile]`, `[runtime-init]`, `[module-load]`, `[weight-upload]`).

| Failure Condition | Stage | Observed Diagnostic |
|---|---|---|
| Unset `MLXCEL_XLA_ROCM_TARGET` on `--device hip` | `[compile]` | `Error: [compile] MLXCEL_XLA_DEVICE=hip requires MLXCEL_XLA_ROCM_TARGET (for example export MLXCEL_XLA_ROCM_TARGET=gfx1151). Detect your GPU ASIC with rocminfo \| grep -E 'Name:\s+gfx' or eval "$(scripts/iree/setup-rocm.sh --env)".` |
| Invalid `MLXCEL_XLA_ROCM_TARGET=sm_80` | `[compile]` | `Error: [compile] invalid MLXCEL_XLA_ROCM_TARGET="sm_80": expected an AMD GPU architecture identifier like gfx1151, gfx1100, or gfx942. Check rocminfo \| grep -E 'Name:\s+gfx'.` |
| Ambiguous `MLXCEL_XLA_ROCM_TARGET="gfx1100,gfx1151"` | `[compile]` | `Error: [compile] ambiguous MLXCEL_XLA_ROCM_TARGET="gfx1100,gfx1151": specify a single AMD GPU architecture (for example gfx1151, gfx1100, or gfx942).` |
| Unsupported `MLXCEL_XLA_PRECISION=fp8` | `[compile]` | `Error: [compile] unsupported MLXCEL_XLA_PRECISION="fp8" on device="hip": expected f16, bf16, or f32.` |
| Mismatched `gfx90a` VMFB loaded on `gfx1151` GPU | `[module-load]` | `INTERNAL; HIP driver error 'hipErrorNoBinaryForGpu' (209): no kernel image is available for execution on the device; mismatched target chip? missing/wrong bitcode directory?; while invoking native function hal.executable.create` |
| Missing/invalid GPU (`HIP_VISIBLE_DEVICES=99`) | `[runtime-init]` | `FAILED_PRECONDITION; HIP driver error 'hipErrorNoDevice' (100): no ROCm-capable device is detected; [runtime-init] failed to open IREE HAL device 'hip'` |
| Missing `libamdhip64.so` search path | `[runtime-init]` | `UNAVAILABLE; HIP runtime library 'amdhip64.dll'/'libamdhip64.so' not available: please ensure installed and in dynamic library search path ... [runtime-init] failed to open IREE HAL device 'hip'` |

### 3.2 Single-Sequence `mlxcel generate` (`local-task` vs `hip`)

Prompt: `"The capital of France is"` (`--temp 0 --no-chat-template -n 32`):

| Model | Device (`Precision`) | 32-Token Greedy Output | Throughput |
|---|---|---|---|
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | `local-task` (`f32`) | `" Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital"` | `0.42 tok/s` (`76.35 s`) |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | `hip` (`f16` default) | Byte-identical (`32/32` tokens) | `14.22 tok/s` (`2.25 s`, **33.9x** CPU) |
| `Qwen3-0.6B-4bit` | `local-task` (`f32`) | `" Paris. True or False?\nA. True\nB. False\nC. Not sure\nD. Not Given\nAnswer:\nThe capital of France is Paris"` | `4.78 tok/s` (`6.69 s`) |
| `Qwen3-0.6B-4bit` | `hip` (`f16` default) | Matches structure (`31/32` tokens; near-tie capitalization at token 23) | `21.05 tok/s` (`1.52 s`, **4.4x** CPU) |

---

## 4. Phase 3: Architecture Parity, Continuous Batching, HTTP Server, and Stability

### 4.1 64-Token Greedy Trajectory Parity (`xla_oracle_check` vs CPU `f32` Oracle)

| Model | `MLXCEL_XLA_PRECISION` / `MLXCEL_XLA_QUANT` | Device | 64-Step Greedy Match vs `local-task` `f32` Oracle | Result |
|---|---|---|---|---|
| `Llama-3.2-1B-Instruct-4bit` | `f32` (load-time dequant) | `hip` | `64 / 64` token-exact | `TOKEN-EXACT PASS` |
| `Llama-3.2-1B-Instruct-4bit` | `f16` (default on `hip`) | `hip` | `64 / 64` token-exact | `TOKEN-EXACT PASS` |
| `Llama-3.2-1B-Instruct-4bit` | `bf16` (load-time dequant) | `hip` | `64 / 64` token-exact | `TOKEN-EXACT PASS` |
| `Llama-3.2-1B-Instruct-4bit` | `f16` + `MLXCEL_XLA_QUANT=packed` | `hip` | `64 / 64` token-exact | `TOKEN-EXACT PASS` |
| `Qwen3-0.6B-4bit` | `f32` (load-time dequant) | `hip` | `64 / 64` token-exact | `TOKEN-EXACT PASS` |
| `Qwen3-0.6B-4bit` | `f16` (default on `hip`) | `hip` | `64 / 64` token-exact | `TOKEN-EXACT PASS` |
| `Qwen3-0.6B-4bit` | `bf16` (load-time dequant) | `hip` | `63 / 64` (1 near-tie token at step 22: `16246` vs `2661`, re-converges immediately at steps 23..64) | Near-tie drift at 1 step |

### 4.2 Continuous Batching (`XlaBatchEngine` / `xla_batch_bench` on `hip`)

Tested on `mlx-community/Llama-3.2-1B-Instruct-4bit` (`f16` default, staggered request lengths 20..62 and token budgets 12..24, exercising mid-stream slot eviction and device-side slot admission):

| Batch Slots (`B_max`) | Requests (`N`) | Total Generated Tokens | Pump Steps | Sequential Reference Baseline | Batched Throughput (Warmup, Run 1, Run 2, Run 3) | Measured Mean (3 Runs) | Reference Equivalence |
|---:|---:|---:|---:|---:|---|---:|---|
| `4` | `8` | `170` | `46` | `8.42 tok/s` | `67.00`, `64.89`, `65.78`, `66.73 tok/s` | **`65.80 tok/s`** (`7.8x`) | `REFERENCE-EXACT PASS` (`8/8` requests exact) |
| `8` | `16` | `342` | `46` | `9.64 tok/s` | `78.71`, `83.34`, `82.61`, `80.22 tok/s` | **`82.06 tok/s`** (`8.5x`) | `REFERENCE-EXACT PASS` (`16/16` requests exact) |

### 4.3 `mlxcel-server` OpenAI `/v1/completions` on `hip`

- Startup command: `MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=hip ./target/release/mlxcel-server -m Llama-3.2-1B-Instruct-4bit --port 18451`
- Startup log: `OpenXLA model Llama-3.2-1B-Instruct-4bit loaded in 4.884s (B_max=4, capacities=[256], device=hip)`
- Non-streaming `POST /v1/completions` (`max_tokens: 16`, `temperature: 0.0`): HTTP 200, returned `" Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\n"`, `"finish_reason": "length"`, `"usage": {"prompt_tokens": 6, "completion_tokens": 16, "total_tokens": 22}`.
- Streaming `POST /v1/completions` (`stream: true`, `max_tokens: 16`, `temperature: 0.0`): HTTP 200 SSE stream, emitted 16 token delta events + final `finish_reason: "length"` event + `data: [DONE]`.
- Peak GPU memory while serving: `vram_used = 4,364.0 MiB`, `gtt_used = 20.2 MiB`.

### 4.4 20-Iteration Stability Smoke Loop on `hip`

20 back-to-back invocations of `MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=hip ./target/release/mlxcel generate -m Llama-3.2-1B-Instruct-4bit -p "The capital of France is" -n 16 --temp 0 --no-chat-template`:

- **Completions**: `20 / 20` succeeded with 0 hangs, 0 crashes, and byte-identical output across all 20 iterations (`11.39` to `11.89 tok/s`).
- **GPU Memory After Teardown**:
  - Iteration 1 post-exit: `vram_used = 147.8 MiB`, `gtt_used = 14.1 MiB`
  - Iteration 10 post-exit: `vram_used = 147.8 MiB`, `gtt_used = 14.1 MiB`
  - Iteration 20 post-exit: `vram_used = 147.8 MiB`, `gtt_used = 14.1 MiB` (matches pre-run baseline `147.7 MiB` VRAM / `14.1 MiB` GTT; zero leaked device allocations).

---

## 5. Phase 4: Precision, Packed Quantization, Data-Tiling Spike (`#574`), and Benchmarks

### 5.1 Performance & Memory Summary Table (`Llama-3.2-1B-Instruct-4bit`, Context Capacity 256)

All measurements use 1 warmup run followed by 3 measured runs on `gfx1151` (`hip`) and `AMD Ryzen AI Max+ 395` (`local-task`):

| Mode / Workload | Device | Precision / Quant | Cold Compile + Load (s) | Warm Load (s) | Prefill Latency (`n=1`, s) | Single-Seq Decode (`n=64`, Run 1 / 2 / 3 `tok/s`) | Mean Throughput (`tok/s`) | Peak VRAM / GTT (`MiB`) |
|---|---|---|---:|---:|---:|---|---:|---:|
| Single-seq (`n=32`) | `local-task` (CPU) | `f32` (load-dequant) | `8.4` | `4.1` | — | `0.42` | `0.42` | Host RAM only |
| Single-seq (`n=32`) | `hip` (`gfx1151`) | `f16` (default) | `11.65` (compile `~5.85s`) | `5.80` | `0.177` (`0.19 / 0.17 / 0.17`) | `14.15 / 14.12 / 14.22` | **`14.16`** | `~2,150 / 14.1` |
| Single-seq (`n=64`) | `hip` (`gfx1151`) | `f32` (load-dequant) | `18.21` (incl. 64-tok run) | `3.15` | — | `5.82 / 5.33 / 5.97` | **`5.71`** | `~4,200 / 14.1` |
| Single-seq (`n=64`) | `hip` (`gfx1151`) | `f16` (default) | `9.89` (incl. 64-tok run) | `4.50` | `0.177` | `11.40 / 10.94 / 11.55` | **`11.30`** | `~2,150 / 14.1` |
| Single-seq (`n=64`) | `hip` (`gfx1151`) | `bf16` (load-dequant) | `26.60` (incl. 64-tok run) | `4.08` | — | `10.01 / 10.07 / 9.99` | **`10.02`** | `~2,150 / 14.1` |
| Single-seq (`n=64`) | `hip` (`gfx1151`) | `f16` + `packed` (`ui32`) | `762.76` (incl. 64-tok run) | `2.40` | — | `0.08 / 0.07 / 0.06` | **`0.07`** | `~1,100 / 14.1` |
| Batched decode (`B_max=4`, `N=8`) | `hip` (`gfx1151`) | `f16` (default) | — | `4.88` | — | `64.89 / 65.78 / 66.73` | **`65.80`** | `4,296.6 / 14.1` |
| Batched decode (`B_max=8`, `N=16`) | `hip` (`gfx1151`) | `f16` (default) | — | `5.12` | — | `83.34 / 82.61 / 80.22` | **`82.06`** | `4,364.0 / 20.2` |

### 5.2 Packed In-Graph Dequant (`MLXCEL_XLA_QUANT=packed`) & `--iree-global-opt-data-tiling` (`#574`)

Running `scripts/xla/fusion_spike.sh` on the emitted `Llama-3.2-1B-Instruct-4bit` packed decode graph (`--iree-hal-target-device=hip --iree-rocm-target=gfx1151`) produces the same structural behavior observed on CUDA (`#573` / `#574`):

```text
target device: hip (rocm target: gfx1151)
packed decode graph: /tmp/mlxcel-xla-vmfb/decode-c256-f764045494f0a33e.mlir

== packed decode: flow.dispatch under an iree-compile flag sweep ==
  baseline                 1209
  aggressive-fuse          1208
  generalize-mm            1209
  early-trunc              1209
  horiz-contract           1209
  fuse-multi-use           1209
  data-tiling              FAIL('tensor.expand_shape' op expected collapsed rank (2) to equal the number of reassociation maps (3))
  data-tiling+rocm-pad     1209
  data-tiling+rocm-dt      FAIL('tensor.expand_shape' op expected collapsed rank (2) to equal the number of reassociation maps (3))
  all-fusion               1208

== full VMFB compilation check under data-tiling (#574) ==
  data-tiling full compile: FAIL ('tensor.expand_shape' op expected collapsed rank (2) to equal the number of reassociation maps (3))

== microtests: one matmul, dispatch count (2 = producer + matmul, not fused) ==
  dequant -> matmul      2 dispatch(es)
  bare f16 matmul        2 dispatch(es)
  bare int8 matmul       2 dispatch(es)

int8 kernel arithmetic (upcast vs tensor/matrix cores):
  sext i8->i32 (upcast): 8   dp4a/mma/wmma/mfma/sdot: 0
```

**Key Takeaways**:
1. **Default precision on `hip`**: `f16` (`default_precision("hip") == Precision::F16`) is `2.0x` faster than `f32` (`11.30 tok/s` vs `5.71 tok/s` on 64-token runs, `14.16 tok/s` on 32-token runs) and `64/64` token-exact with `f32`.
2. **`bf16` on `hip`**: Lowers and executes cleanly on `gfx1151` (`10.02 tok/s`, `64/64` token-exact on `Llama-3.2-1B-Instruct-4bit`), unlike `metal-spirv` where `bf16` cannot legalize (`#612`).
3. **`MLXCEL_XLA_QUANT=packed` on `hip`**: Compiles and executes `64/64` token-exact on `Llama-3.2-1B-Instruct-4bit` (unlike Metal where the command buffer faults, `#613`), but regresses decode throughput to `0.07 tok/s` because IREE 3.12.0rc20260721 emits `1,209` flow dispatches and materializes the unpacked `f16` weights to global memory every decode step (`dequant -> matmul` = 2 dispatches). Enabling `--iree-global-opt-data-tiling` fails during `flow` lowering with `'tensor.expand_shape' op expected collapsed rank (2) to equal the number of reassociation maps (3)` (identical to CUDA in `#574`), while `--iree-rocm-encoding-layout-resolver=pad` compiles with `1,209` unfused dispatches. Consequently, `MLXCEL_XLA_QUANT=packed` remains opt-in (off by default) on `hip`.

---

## 6. Support Decision & Support Boundaries

- **Qualification Verdict**: **Conditionally Supported**
  - **Supported (Production-Ready)**:
    - Target devices: AMD ROCm/HIP (`MLXCEL_XLA_DEVICE=hip`) with a single unambiguous ASIC specified via `MLXCEL_XLA_ROCM_TARGET` (qualified on `gfx1151` / AMD Radeon 8060S Graphics; architecture validation permits `gfx1100`, `gfx942`, `gfx90a`).
    - Precision modes: `f16` (default on `hip`, recommended for performance, 2.0x faster than f32), `f32` (load-time dequantization, reference oracle), and `bf16` (load-time dequantization).
    - Execution paths: Single-sequence generation (`mlxcel generate`), continuous batching (`XlaBatchEngine` / `xla_batch_bench` for `B_max=4` and `B_max=8`), and HTTP serving (`mlxcel-server` non-streaming and streaming SSE).
  - **Conditionally Supported (Opt-In / Experimental)**:
    - Packed int4/int8 in-graph dequantization (`MLXCEL_XLA_QUANT=packed`): Correctness verified (64/64 token-exact), but throughput is compute-bound / memory-bandwidth regressed (`0.07 tok/s`) because IREE 3.12.0rc20260721 lacks native quantized-matmul fusion for ROCm (emits 1,209 unfused dispatches and materializes f16 weights to DRAM each step). Kept opt-in behind `MLXCEL_XLA_QUANT=packed`.
  - **Blocked / Not Supported**:
    - Data tiling optimization (`--iree-global-opt-data-tiling`) on ROCm in IREE 3.12.0rc20260721: Fails during `flow` compilation with `'tensor.expand_shape' op expected collapsed rank (2) to equal the number of reassociation maps (3)` (reproducible with `scripts/xla/fusion_spike.sh`, tracked as an upstream IREE issue in #574).

