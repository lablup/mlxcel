# cuda-oxide adoption feasibility for mlxcelverse

Review date: 2026-11-02. Status: kernel-first investigation sequence agreed; implementation and performance validation pending.

Implementation planning update: 2026-11-03. The initial issues below cover the bounded experiment, not production enablement or an independent engine.

## Recommendation

**Proceed with a small, optional Linux CUDA kernel experiment. Do not replace the MLX CUDA backend or make cuda-oxide a mandatory mlxcel build dependency.** The strongest integration path is to compile selected Rust kernels separately, then load their PTX/cubin through mlxcelverse while retaining MLX's allocation, lazy evaluation, streams, and CUDA Graph scheduling.

한국어 요약: 우선 MLX와 분리된 Rust 커널 패키지 및 독립 실행 벤치마크를 만들고, 얇은 어댑터를 통해 기존 MLX 경로에 연결한다. 첫 대상은 fused residual-add + RMSNorm이며, 커널 자체 성능과 MLX 통합 비용을 구분해서 측정한다. 양쪽 모두 효과가 있으면 기존 경로에 확대하고, 커널은 유효하지만 MLX 실행 구조가 이득을 제한한다는 근거가 확보되면 독립 추론 엔진을 검토한다. 컴파일러·GPU 지원 문제가 원인이면 먼저 해결한다. 독립 엔진은 기존 MLX 백엔드의 전면 대체와 구분되는 후속 선택지이며, 이번 커널 실험의 일정 추정을 적용하지 않는다. GB10의 일반 SIMT 실행 가능성과 고급 Tensor Core 경로 지원은 별도로 확인해야 한다.

| Scope | Feasibility | Decision |
| --- | --- | --- |
| New elementwise, reduction, or fused kernels written in Rust | High in principle; MLX adapter still required | Best initial scope |
| Replace selected CUDA implementations of existing fused kernels | Medium to high | Opt-in, with the existing implementation as fallback |
| KV-cache writes, RoPE fusion, paged-attention variants | Medium | Follow only after ownership and graph ordering are proven |
| Replace tuned QMV/QMM, grouped GEMM, or SDPA | Technically possible; performance benefit uncertain | Require profiler evidence and architecture-specific experiments |
| Replace MLX's CUDA runtime/backend | Low practical value for this proposal | Out of scope |
| Add an independent native CUDA inference path alongside MLX | Plausible, but substantially broader than kernel integration | Conditional follow-up after the kernel experiment |
| One Rust implementation for CUDA, Metal, and ROCm | Not supplied by cuda-oxide | Retain existing backend ports |

These ratings are engineering judgments from source inspection, not implementation or benchmark results.

## Priority decision: kernel integration before an independent engine

The agreed sequence is **independent kernel package → MLX integration and comparison → independent inference engine only if the evidence justifies it**. An independent engine remains a viable architectural option; it is not the first implementation milestone.

The immediate uncertainty is whether cuda-oxide produces correct, useful kernels on the actual target GPU. Building an engine first would add weight loading, device memory management, KV storage, and execution scheduling before resolving that uncertainty.

| Criterion | Kernel integration into MLX | Independent inference path |
| --- | --- | --- |
| Work before the first useful comparison | One kernel, artifacts, and adapter | Model execution plus runtime infrastructure |
| Attribution of performance differences | Same model/executor; selected kernel changes | Kernel and executor effects must be separated |
| Coverage expansion | Models using the supported operation can benefit | Model families need native execution implementations |
| Control over scheduling and memory | Constrained by MLX | Owned by the new engine |
| Cost of an unsuccessful experiment | Bounded | Substantial |

Design the first experiment so its work survives either outcome:

- Keep kernel source, specialization policy, artifact production, and ABI manifests independent of MLX headers and array types.
- Confine MLX allocation and scheduling integration to a thin adapter.
- Run the same kernel artifact in a standalone harness and through MLX. Match dtype, shape, launch geometry, numerical policy, and timing boundaries; report cold setup separately from steady-state device execution.
- Compare the integrated path against the existing fused CUDA implementation and an end-to-end model workload. A standalone launch cannot represent the cost or benefits of MLX graph scheduling by itself.
- Use profiling and controlled comparisons to identify adapter, launch, synchronization, and allocation costs. Do not label a difference between unrelated benchmark setups as MLX overhead.

| Observed result | Next action |
| --- | --- |
| Kernel is correct and useful both standalone and through MLX | Expand selected kernels within the existing path |
| Kernel is useful standalone, but a reproducible MLX execution constraint prevents the benefit | First test whether a bounded adapter improvement resolves it; if not, scope an independent engine prototype |
| Compiler, generated code, or GPU support is the limiting factor | Resolve or defer that toolchain/target issue before investing in an engine |
| Correct execution but no meaningful speed, startup, or maintenance benefit | Stop expansion or select another profiler-justified candidate; do not infer that an independent engine would be faster |

Fused normalization validates the integration method. Its result alone cannot establish the feasibility or performance of attention, quantized matmul, or a complete inference engine.

### Conditional follow-up: a native CUDA inference path

This option would coexist with MLX and own model execution, weights, KV storage, device memory, streams, and CUDA Graph execution. cuda-oxide would supply selected kernels; established CUDA libraries could still supply GEMM and other operations. Writing every operation in Rust is not a prerequisite.

The existing [ComputeBackend seam](../../src/backend/mod.rs) and [Session implementation](../../src/backend/session.rs) provide a starting point for single-sequence integration. They are not a fully backend-neutral serving interface: the load boundary returns `LoadedModel`, the batched server path remains coupled to MLX, and some session methods accept `MlxArray`. These boundaries need explicit design work before native CUDA batching or multimodal serving can reuse them.

Place a future independent engine in a separate crate rather than expanding the MLX-specific responsibilities of mlxcelverse. Reuse tokenizer/configuration logic, checkpoint conventions, request policies, and correctness fixtures where possible. Model forward code, device storage, sampling execution, and server batching require an explicit dependency audit rather than an assumption of drop-in reuse.

The first engine milestone, if justified, should be one text-only dense model on one GPU in BF16/FP16, with weight loading, prefill, persistent KV state, decode, and token output running without MLX execution. Select a Llama- or Qwen-family checkpoint with a strong existing oracle and benchmark set. Compare intermediate tensors and logits as well as generated output. Quantization, paged attention, continuous batching, MoE, multimodal input, and distributed execution follow as separately scoped milestones.

The kernel experiment's effort estimate below does not cover this engine. Scope and estimate it separately after the kernel and integration measurements; a faster standalone kernel is necessary evidence for that direction, not sufficient proof of a faster engine.

## Evidence and reproducibility

The local checkout was clean before this documentation change. Inspected revisions:

| Component | Revision |
| --- | --- |
| mlxcel | `d771735d3dc9baf7a777dab08e9bd98e83083323` |
| MLX pin in mlxcelverse CMake | `7ca40033132dae444d45f7fc3daae57e24c13f6c` |
| NVIDIA/cuda-rust source checkout | `b631530700a8a4a5f165456f2d71fe9195b323b5` |
| mlxcel Rust toolchain | `1.97.1` |
| cuda-oxide Rust toolchain at the inspected revision | `1.99.1`, with compiler-development components |

The intended project here is NVIDIA's SIMT compiler, formerly under `NVlabs/cuda-oxide` and now under `NVIDIA/cuda-rust/cuda-oxide`, rather than the unrelated older project with the same name. Current source and installation instructions supersede older launch articles that mention a nightly compiler and CUDA 12. See the [current repository](https://github.com/NVIDIA/cuda-rust) and [pinned cuda-oxide README](https://github.com/NVIDIA/cuda-rust/blob/b631530700a8a4a5f165456f2d71fe9195b323b5/cuda-oxide/README.md).

This review inspected source and documentation only. It did not install the compiler, compile an oxide kernel, launch GPU work, measure speedups, or validate a specific driver/GPU combination. Source support and example availability are not hardware acceptance evidence.

## What cuda-oxide provides

cuda-oxide is a custom Rust compiler backend: Rust MIR passes through Pliron and LLVM to device code. It supports kernel specialization through generics/policies and exposes GPU mechanisms such as shared memory, warp operations, atomics, barriers, and architecture-dependent matrix instructions. The project explicitly describes itself as alpha, with incomplete features and potential API breakage. It is a kernel programming toolchain, not an inference engine or an automatic optimizer for MLX graphs. [Source: pinned README](https://github.com/NVIDIA/cuda-rust/blob/b631530700a8a4a5f165456f2d71fe9195b323b5/cuda-oxide/README.md).

The inspected installation guide specifies Linux, Ampere-or-newer GPUs, CUDA Toolkit 13.0+, driver 580+, LLVM/Clang 21+, and pinned Rust 1.99.1. Windows is unsupported by that toolchain. The artifact container documents Linux on AMD64 and ARM64 as supported host platforms, but that does not by itself establish a working GB10 build. PTX version and driver compatibility must be checked independently of the toolkit version. [Installation](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/installation.html), [pinned toolchain](https://github.com/NVIDIA/cuda-rust/blob/b631530700a8a4a5f165456f2d71fe9195b323b5/cuda-oxide/rust-toolchain.toml), [artifact host platforms](https://github.com/NVIDIA/cuda-rust/blob/b631530700a8a4a5f165456f2d71fe9195b323b5/cuda-oxide/crates/oxide-artifacts/README.md).

Rust safety is useful inside kernels, but the C++ boundary remains a proof obligation. Generated typed launches check argument types; raw launches require the caller to establish geometry and resource validity. Slices and `DisjointSlice<T>` lower to pointer-plus-length arguments. Passing an MLX allocation to C++ launch code does not automatically inherit Rust's aliasing or lifetime guarantees. [Launch API and argument mapping](https://github.com/NVIDIA/cuda-rust/blob/b631530700a8a4a5f165456f2d71fe9195b323b5/cuda-oxide/cuda-oxide-book/gpu-programming/launching-kernels.md).

### GB10 and Tensor Core limits

Keep ordinary SIMT compilation separate from the CuTe preview and specialized matrix instructions. The inspected CuTe preview matrix supports `sm_100a` and `sm_120a`; it explicitly rejects `sm_121a`. Its published validation is therefore not evidence for GB10 acceleration. Likewise, the documented `tcgen05` path is for datacenter Blackwell `sm_100a`, not a general promise for every Blackwell GPU. [CuTe support matrix](https://github.com/NVIDIA/cuda-rust/blob/b631530700a8a4a5f165456f2d71fe9195b323b5/cuda-oxide/cuda-oxide-book/cute/support-matrix.md), [matrix accelerator guide](https://github.com/NVIDIA/cuda-rust/blob/b631530700a8a4a5f165456f2d71fe9195b323b5/cuda-oxide/cuda-oxide-book/advanced/matrix-multiply-accelerators.md).

For GB10, first demonstrate a basic SIMT kernel with a supported emitted target and inspect its PTX, then test the actual driver loader. Do not relabel an `sm_120a` or `sm_100a` artifact as a GB10 artifact. Any architecture-specific optimization needs its own compiler, instruction, and hardware validation.

## Existing mlxcelverse architecture

mlxcelverse already supports CUDA acceleration. It is a C++/CMake layer rather than a Rust crate; see its [layout and maintenance rules](../../src/lib/mlxcelverse/README.md).

```text
Rust models / mlxcel-core
  -> cxx bridge
  -> mlxcelverse launchers and MLX operations
  -> MLX lazy evaluation / CUDA command encoder
  -> CUDA kernels, library calls, and graph execution
```

Relevant integration surfaces:

| Local source | Existing responsibility | Implication |
| --- | --- | --- |
| [CMakeLists.txt](../../src/lib/mlxcelverse/CMakeLists.txt) | Pins MLX and applies backend overlays | An oxide experiment should not require replacing all overlays |
| [mlxcel-core/build.rs](../../src/lib/mlxcel-core/build.rs) | Builds C++ launchers and the cxx bridge | Natural artifact packaging hook, with the compiler kept outside ordinary builds |
| [kernel_port.h](../../src/lib/mlxcelverse/kernels/kernel_port.h) | Selects per-backend `CustomKernelFunction` ports | Oxide is a CUDA implementation choice, not a new hardware backend |
| [fused_norm.cpp](../../src/lib/mlxcelverse/kernels/fused_norm.cpp) | Existing CUDA/Metal/HIP fused normalization and dtype specialization | Concrete correctness reference and experiment candidate |
| [jit_module.cpp](../../src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp) | Builds or loads PTX/cubin and caches modules | `KernelBuilder` already has a `precompiled` branch; promising lower-level reuse point |
| [device.cpp](../../src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/device.cpp) | Commands, CUDA Graph dependencies, captured-child-graph barriers | New launches must participate in this scheduling model |

The precompiled loader is **not** a complete public MLX custom-kernel adapter. Existing `fast::cuda_kernel` launchers consume CUDA source and construct MLX graph operations. A precompiled kernel needs an MLX primitive/adapter that allocates outputs, registers inputs and outputs with the encoder, packs arguments, and schedules execution during evaluation. Loading a `CUfunction` alone does not accomplish this.

The CUDA overlays also preserve precision, quantized matmul, graph ordering, upload, JIT caching, and Windows loading behavior. Replacing them wholesale would reopen unrelated correctness and maintenance work; see the [CUDA overlay inventory](../../src/lib/mlxcelverse/overlays/cuda/README.md).

## Integration alternatives

| Alternative | Benefit | Main cost | Assessment |
| --- | --- | --- | --- |
| Compile Rust kernels separately; consume PTX/cubin from C++ MLX adapter | Keeps existing allocator, scheduler, application toolchain | Artifact extraction, launch ABI, MLX primitive integration | Recommended experiment |
| Call oxide host runtime through a Rust/C ABI bridge | Reuses generated launch machinery | Foreign buffer lifetimes, context/stream ownership, graph integration, second toolchain | Possible second option |
| Add a separate native CUDA inference engine alongside MLX | Owns execution and memory policy; can reuse independent kernel artifacts | Model execution, weight/KV storage, sampling, and serving integration | Conditional follow-up, not the initial experiment |
| Replace MLX CUDA backend with oxide runtime | Maximum Rust ownership of the GPU layer | Reimplement lazy execution, operators, memory policy, graph handling, and library integration | Disproportionate scope |
| Continue CUDA C++ and existing JIT kernels | Lowest integration risk; mature path | Keeps current kernel authoring model | Baseline every experiment must beat or otherwise justify |

The host-runtime alternative is not ruled out by missing handle interop: the inspected `cuda-core` provides unsafe `borrow_raw` and owner-retaining wrappers for external CUDA resources. Those APIs do not establish the lifetime and dependency contract of MLX arrays automatically. [Runtime source](https://github.com/NVIDIA/cuda-rust/blob/b631530700a8a4a5f165456f2d71fe9195b323b5/cuda-core/src/runtime.rs).

### Proposed artifact boundary

Use a separate kernel package/toolchain and publish immutable artifacts with the consuming build. Keep ordinary mlxcel Rust builds on their existing pin. Avoid adding the full compiler backend to the application dependency graph.

cuda-oxide's `.oxart` bundles can describe PTX, cubin, and IR payloads plus entry metadata; the container is not itself executable PTX. The packaging step must select/extract the intended payload and record its symbol and ABI. Its current entry metadata is not a complete C++ argument schema, so emit or maintain a checked manifest rather than infer argument layout from a symbol name. [Artifact format](https://github.com/NVIDIA/cuda-rust/blob/b631530700a8a4a5f165456f2d71fe9195b323b5/cuda-oxide/crates/oxide-artifacts/README.md).

The proposed adapter must enforce:

1. **ABI:** explicit scalar widths, pointer/length order, alignment, dtype, layout, and allowed aliasing; compare against emitted PTX parameters. Do not pass Rust structs or slice objects as though they were C++ structs.
2. **Execution:** enqueue through the MLX command encoder on its context/device/stream; retain arrays and modules through asynchronous completion and graph replay. A direct launch at graph construction time is incorrect.
3. **Graph behavior:** register read/write dependencies and preserve captured-child-graph barriers. Shared CUDA context or stream identity alone is insufficient.
4. **Specialization:** include artifact digest, compiler revision, target, dtype combinations, shape/tile policy, ABI revision, and relevant numerical flags in module identity. Preserve current cache trust checks.
5. **Fallback:** choose the existing CUDA implementation before enqueue for unsupported shapes, architectures, or unavailable artifacts. An execution-time illegal memory access is an error, not a safe opportunity to retry on the same context.
6. **Portability:** keep Metal/ROCm/CPU and Windows on current paths. Linux-produced PTX might be portable to a compatible Windows driver, but packaging and execution there remain a separate unverified project.

All integration names and switches in a future implementation should be documented as new features; this review introduces none.

## Where optimization may pay off

| Candidate | Potential gain | Existing constraint / risk | Priority |
| --- | --- | --- | --- |
| Fused residual-add + RMSNorm | Tune reductions, vectorization, register use; avoid NVRTC compilation for this kernel | Already fused; preserve intermediate dtype rounding, mixed weight dtype, and accumulation policy | First proof of integration |
| RoPE + KV append / KV writes | Reduce memory traffic or launches if a missing fusion is identified | In-place cache aliases, strides, offsets, page boundaries, and concurrent sequences | Second-stage candidate |
| Sampling / small reductions | Explore specialized sizes and fewer intermediates | RNG and acceptance semantics; reduction determinism | After basic adapter acceptance |
| Paged attention / quantized KV | Potentially meaningful bandwidth/layout optimization | Masks, page addressing, GQA, quantization, long-context accuracy, shared-memory pressure | Later, profiler-driven |
| QMV/QMM / MoE GEMM / SDPA | Architecture-specific scheduling and tensor-core opportunities | Competes with existing tuned kernels and vendor libraries; GB10 target limits | Separate research effort |

Changing the implementation language does not remove an existing launch, memory read, or graph dependency. Speedups require a better algorithm, fusion boundary, tile/layout, instruction selection, or dispatch policy. Safe indexing may introduce checks; removing them requires measured need and a reviewed safety proof. Ahead-of-time device artifacts can reduce the selected kernel's NVRTC cold-start work, but PTX driver JIT, module loading, and other MLX JIT kernels remain.

For the first experiment, use the current fused implementation as the performance baseline and the graph implementation as an additional numerical oracle. Comparing only against an unfused graph would incorrectly attribute existing fusion benefits to oxide.

## Bounded proof of concept and acceptance gates

### Initial implementation issues

Execute these issues in order. Only the first is initially ready; later issues depend on the preceding implementation and its required GPU evidence. The issue bodies define the concrete acceptance checklists.

| Issue | Scope | Depends on | Required result |
| --- | --- | --- | --- |
| [#2831](https://github.com/lablup/mlxcel/issues/2831) | Isolated toolchain/package, artifact manifest, standalone vector smoke | None | Reproducible build and actual GB10 execution with verified outputs |
| [#2832](https://github.com/lablup/mlxcel/issues/2832) | Standalone fused residual-add + RMSNorm, f32/f16/bf16 validation | #2831 | Reusable artifacts, numerical fixtures, and standalone benchmark |
| [#2833](https://github.com/lablup/mlxcel/issues/2833) | Default-off MLX adapter for the same kernel artifacts | #2831, #2832 | Lazy execution, graph ordering, lifetimes, route verification, and safe fallback |
| [#2834](https://github.com/lablup/mlxcel/issues/2834) | Matched standalone/integrated comparison and real-checkpoint report | #2833; reuses #2832 artifacts and fixtures | Raw measurements and an evidence-based expand, investigate, resolve, or stop decision |

The first practical stopping point is #2831: establish that this toolchain and artifact contract work on the intended GPU before writing the inference kernel. #2832 then tests kernel semantics without MLX runtime integration. Hardware-unavailable or compile-only results do not complete the GPU acceptance gates. A measured lack of benefit is a valid final outcome for #2834.

Production packaging/default enablement, other operator families, and an independent inference engine are deferred. They are not silently included in this issue batch. The documentation PR only references these issues and must not close them on merge.

### Planning estimates

Effort estimates below are planning ranges for one engineer familiar with MLX/CUDA. They are not delivery commitments and exclude upstream compiler fixes and hardware waiting time.

| Phase | Deliverable | Estimated effort | Stop / continue gate |
| --- | --- | --- | --- |
| 0. Toolchain and target probe (#2831) | Independent kernel package, reproducible pinned build, simple standalone launch, emitted ABI/artifact inventory on the actual target | 2–4 working days | Stop if target/toolchain cannot execute correctly without substantial upstream work |
| 1. Standalone fused normalization (#2832) | f32 then f16/bf16 fused residual-add + RMSNorm, odd/tail widths, deterministic fixtures, standalone benchmark | 3–5 working days | Stop or defer if the artifact cannot express the required semantics or parity budget |
| 2. MLX adapter and matched comparison (#2833, #2834) | One precompiled kernel in MLX lazy execution; graph on/off; zero host round trips; same artifacts standalone and integrated; differential tests, profiling, and end-to-end comparison | 7–15 working days | Apply the priority decision table; stop if correct scheduling requires replacing the backend, and distinguish kernel limitations from integration costs |
| 3. Production hardening | Packaging, fallback, non-CUDA builds, diagnostics, target CI | 5–10 working days | Separate approval after experiment results |

Thus a decision-quality kernel proof is roughly **2–5 engineer-weeks**, with production hardening additional. This estimate is provisional and does not cover an independent inference engine or full backend replacement.

Validation must include:

- **Numerics:** supported dtypes and mixed weight dtypes; odd and small dimensions, tails, large rows, empty/invalid inputs, noncontiguous inputs or explicit refusal, NaN/Inf policy, and residual rounding. Set per-dtype tolerances before measuring. Follow existing [fused norm parity tests](../../src/lib/mlxcel-core/src/fused_norm_parity_tests.rs).
- **Scheduling/lifetime:** lazy chains, asynchronous completion, repeated graph replay, graph on/off, stream/context changes, allocator reuse, concurrent requests, and module teardown. Run memory/race sanitizers where supported.
- **Performance:** kernel latency and end-to-end prefill/decode separately; B=1 and representative batches; short/long contexts; cold/warm process and cache states. Warm up, synchronize measurement boundaries, repeat runs, report variance, and record GPU clocks/power policy, driver, compiler, targets, and source revisions.
- **Profiling:** register pressure/spills, occupancy, bandwidth, launch count, and synchronization. Check generated PTX/SASS rather than infer instruction quality from Rust source.
- **Packaging:** clean build without the oxide compiler for ordinary users, reproducible artifact checksums, missing/unsupported artifact fallback, and unchanged Metal/ROCm/CPU behavior. Preserve upstream licenses/notices and review generated-artifact redistribution terms before shipping.
- **Operational isolation:** on hosts where `nvidia-smi` reports `NVIDIA GB10`, run every GPU-allocating probe/test/benchmark through `gpu-lock run --tag <purpose> -- <command>`; if unavailable or busy, stop or wait rather than bypass it.

Suggested experiment thresholds, to agree before execution: all correctness gates pass; no unexplained representative end-to-end regression above 2% outside measurement noise; and either a repeatable kernel benefit of at least 10% with useful end-to-end impact, a documented cold-start reduction, or a concrete maintenance improvement worth the added compiler/ABI burden. These are proposed criteria, not measured results.

## Decision after this review

There is enough source evidence to justify the bounded experiment, especially because mlxcelverse already has a precompiled CUDA module-loading path. There is not enough evidence to claim faster inference, production readiness, GB10 Tensor Core acceleration, or a safe full-backend migration.

The next implementation should establish both standalone kernel behavior and correct participation in MLX's execution model, then compare against today's fused CUDA implementation. Keep the current backend as the reference and fallback. Expand within MLX when the integrated result is useful; consider an independent inference path only when measured execution constraints justify its additional scope. The independent kernel package and benchmarks should remain reusable in either direction.
