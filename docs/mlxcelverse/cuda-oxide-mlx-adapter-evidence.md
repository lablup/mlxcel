# cuda-oxide fused residual-add + RMSNorm evidence: MLX adapter on GB10

Evidence for [issue #2833](https://github.com/lablup/mlxcel/issues/2833), the third gated experiment in [the cuda-oxide feasibility review](cuda-oxide-feasibility.md). Run date: 2026-11-26 (UTC). Host: `spark-101`, a DGX Spark that also hosts a self-hosted CI runner; every GPU command below ran under `gpu-lock`.

Status: the precompiled cuda-oxide module from [#2832](cuda-oxide-fused-norm-evidence.md) runs inside MLX lazy evaluation as an alternative implementation behind `fused_add_rms_norm`, default off at build time and at run time. The 130-case matrix ran through the adapter with every case on the cuda-oxide route and every output bit-identical to the existing fused CUDA kernel. Scheduling, lifetime, fallback and fault tests ran with CUDA graph capture on and off. One overlay file changed, only to let this module skip the PTX disk cache (`jit_module.cpp`/`.h`: a trailing `cu::DiskCache` parameter that defaults to `DiskCache::Use`, so every other caller is unchanged). This report carries no throughput number and enables nothing by default; that is [#2834](https://github.com/lablup/mlxcel/issues/2834).

Update 2026-11-28 ([issue #3510](https://github.com/lablup/mlxcel/issues/3510)): the existing CUDA kernel now follows MLX's `rms_norm` summation order and is byte-identical to the graph path, and the artifact was regenerated to stay bit-identical to it: PTX SHA-256 `ed470a19c4848893a680223db9c6861363c4cb180a5c229a63307afc305357f4`, cubin `390fcc2833e532ef5bffb7d8699c2f43a0fd7dc53775079b9b2c6be938a558cd`, same compiler revision and ABI version. The 130-case matrix through the adapter is again bit-identical to the kernel on every case, and now also to the graph on every row without a non-finite input; the test requires both. The tables below are the 2026-11-26 run with the earlier artifact and are left as measured. Results of the rerun are in [`docs/benchmark_results/cuda-fused-norm-byte-identity-gb10-2026-11-28/`](../benchmark_results/cuda-fused-norm-byte-identity-gb10-2026-11-28/README.md).

## Environment

| Component | Value |
| --- | --- |
| Host | `spark-101`, Linux 7.0.0-1019-nvidia, aarch64 |
| GPU | NVIDIA GB10, compute capability 12.1, driver 580.178.04 |
| CUDA Toolkit | 13.0, V13.0.88 (`/usr/local/cuda`) |
| Rust | 1.97.1, the application toolchain; no cuda-oxide compiler, LLVM 21 or Rust 1.99 on this host |
| MLX | pin `fc344c986c4169d99f76b5d95ae6b4a58a359a63`, version 0.32.5 |
| mlxcel | `c5afefbd54e2d52fe56c99b15ad4e971318a4cbe` plus this change |
| Build | `--profile test-fast --features cuda`, `MLXCEL_CUDA_ARCHITECTURES=121` |
| Artifact | `spike/cuda-oxide-fused-norm/artifacts/fused_add_rms_norm.sm_121.ptx`, SHA-256 as of the 2026-11-26 run `b54c1c93c6e8f03e3f3d2cf37a79dd412069153c1360b6161831583c01eaf86e` (regenerated on 2026-11-28, current digest `ed470a19c4848893a680223db9c6861363c4cb180a5c229a63307afc305357f4`), compiler revision `b631530700a8a4a5f165456f2d71fe9195b323b5`, ABI version 1, committed by #2832 and not rebuilt here |
| `gpu-lock` | `/usr/local/bin/gpu-lock`, `run --tag issue-2833 -- <cmd>` around every test binary. Builds ran outside the lock. |

## What was built

The kernel is an MLX primitive, `OxideFusedAddRmsNorm` in [`kernels/fused_norm_oxide.cpp`](../../src/lib/mlxcelverse/kernels/fused_norm_oxide.cpp), with inputs `(x, residual, weight)` and outputs `(normed, new_residual)`. `fused_add_rms_norm` in `kernels/fused_norm.cpp` calls the adapter after its own shape validation and before `select_kernel_port`. When the adapter returns no node, the existing port builds it, unchanged.

| Concern | Who owns it | How |
| --- | --- | --- |
| When the kernel runs | MLX | The node is built lazily. Everything that touches the device is in `eval_gpu`, on the stream MLX evaluates the node on. |
| Output memory | MLX allocator | `cu::malloc_async(out.nbytes(), encoder)`, as `CustomKernel::eval_gpu` does. No host copy of any tensor exists on this path. |
| Ordering against other nodes | MLX command encoder | `set_input_array` for the three inputs, `set_output_array` for both outputs, then `add_kernel_node_raw`. The encoder derives the graph dependencies from the registered buffers, including against captured child graphs. |
| Buffer lifetime | MLX | `gpu::eval` adds the node's inputs and sibling outputs to the encoder's temporaries, which `commit_impl` moves into the completion handler. Copies of non-contiguous inputs are added with `add_temporary`. |
| Module lifetime | MLX JIT module cache | `cu::get_jit_module(device, name, builder, false, cu::DiskCache::Bypass)` with the precompiled flag (the `false` is upstream's `use_disk_cache`, which the overlay ignores). The in-memory cache is never unloaded, so a replayed graph cannot outlive its module. The last argument keeps the module out of the PTX disk cache (see Cache identity). |
| Kernel arguments | this adapter | 13 parameters packed in the order the manifest states: pointers as device addresses, the five lengths as 64-bit element counts, `eps` and `weight_bias` as f32, `dim` as u32. `cuGraphAddKernelNode` copies them, so they only live until the node exists. |
| Launch geometry | manifest rule | `grid = (rows, 1, 1)`, `block = (fused_norm_cuda_block(dim, n_reads), 1, 1)`, no dynamic shared memory. The block width is the value `fused_norm.cpp` already computed for the existing port: MLX's own `rms_norm` block for the row width and the pair's compute dtype since #3510 (`fused_norm_threads(dim)` when this evidence was taken). The validator checks each manifest's `n_reads` against its dtype pair and refuses an artifact built for another rule. |

`fast::precompiled_cuda_kernel` could not be used as it is: `CustomKernel` packs inputs, outputs, then bool/int/float scalars, while the emitted entry interleaves `(pointer, u64 length)` pairs with the scalars. The dedicated primitive is the only new piece; the loader, the encoder and the allocator are used unmodified, and the PTX disk cache and its trust rules are untouched for every other module. The cost is that this one file includes MLX's CUDA backend headers (`backend/cuda/{device,jit_module,allocator}.h`, `backend/gpu/copy.h`), which the other launchers do not, so it is listed in the pin-bump checklist in `CONTRIBUTING.md`.

## Switches and fallback

| Switch | Kind | Effect |
| --- | --- | --- |
| `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1` | build environment | Compiles the adapter and embeds the PTX, after the build script checks its SHA-256 against all five manifests. Linux with `--features cuda` only; anywhere else, or with an invalid value, the build fails. Unset, nothing of the adapter is compiled. |
| `MLXCEL_CUDA_OXIDE_FUSED_NORM=1` | runtime | Validates and installs the artifact at the first fused residual join. Requires `MLXCEL_FUSED_ADD_RMSNORM=1`, which is what makes model code call the fused kernel at all on CUDA. |
| `MLXCEL_CUDA_OXIDE_FUSED_NORM_DIR=<dir>` | runtime, optional | Loads the artifact from a directory laid out like the spike package instead of the embedded copy, through the same validation. |

The build switch is an environment variable and not a Cargo feature because a feature, like a build dependency, changes the build-script unit's metadata hash and with it `OUT_DIR`, which would rebuild MLX from source for every build that toggles it. For the same reason the build-time digest check uses a SHA-256 written out in `build_support/cuda_oxide_fused_norm.rs` instead of a new build dependency. Measured here: turning the variable on after a default build re-ran the build script in the same `OUT_DIR`, recompiled the bridge, and finished in 57 s; the cold MLX build before it took 12 min 22 s.

Where each refusal happens:

| Condition | When | Result |
| --- | --- | --- |
| PTX digest, size, ABI version, target, payload kind, launch rule, compiler revision, or parameter layout does not match the manifests and schemas; a file is missing | install, before any node exists | Nothing is installed. One line on stderr. The existing kernel serves every call. |
| Device 0 is not sm_121 | install | Same. |
| Dtype pair without an entry (for example f32 activations with an f16 weight); more than 2^32-1 elements; more than 2^31-1 rows (the grid's x extent, one block per row); default device is the CPU | per call, when the node is built | The existing port builds the node. The reason is recorded. |
| Module fails to load; node evaluated on a device of another architecture | `eval_gpu`, before the encoder or allocator is touched | The evaluation throws to the caller. Nothing retries on the existing kernel. |

Input layout is the one configuration that cannot be decided before enqueue: whether an array is row-contiguous is a property of its evaluated buffer, and a lazy input has none yet. The adapter therefore applies the policy `fast::cuda_kernel` applies to the existing port: a non-contiguous input is copied to a packed buffer inside `eval_gpu`, on the same stream, and the copy is kept alive as a temporary.

The route marker is three process-wide counters read through `mlxcel_core::cuda_oxide_fused_norm::route_counters()`: calls routed to the cuda-oxide node, calls that took the existing kernel, and cuda-oxide kernels enqueued by `eval_gpu`. Every parity assertion below is paired with an assertion on them.

## Cache identity

MLX's JIT module cache, in process and on disk, is keyed on the module name and is consulted before the payload. This module takes the in-process cache only: it is created with the disk cache off, because a disk hit would load bytes that nobody compared to the validated payload. The name is `mlxcel_oxide_fused_norm_abi1_sm_121_b631530700a8_ed470a19c4848893a680223db9c68613` for the current artifact (`..._b54c1c93c6e8f03e3f3d2cf37a79dd41` for the 2026-11-26 one): ABI version, target, the first 12 hex digits of the compiler revision, and the first 32 of the payload SHA-256. The dtype pair is the entry symbol inside the module (five entries, one module). Shape is not part of the identity because the artifact is not specialized on it: `dim` and the row count are launch arguments, unlike the NVRTC port, which compiles one kernel per `(T, TW, Dim, Threads)`.

Validation never depends on the cache: it runs on the bytes about to be installed, before the module name is used. The test `a_warm_cache_cannot_mask_a_wrong_manifest_or_a_missing_artifact` first runs the embedded artifact, so the module is loaded in the process, then shows in the same process that a manifest pinning another digest and a directory without the PTX are both refused and leave the existing kernel serving the call, and that a rebuilt payload with consistently regenerated manifests gets a different module name and launches. After the run the per-user PTX disk cache (`/tmp/mlx-1000/0.32.5/ptx/sm121-nvrtc13.0/`) held no `mlxcel_oxide_fused_norm*` file, which an earlier revision of the adapter wrote one of per artifact.

`a_planted_cache_entry_is_never_loaded_for_the_module` shows the bypass directly. It writes a `.ptx` and a `.txt` kernel map with different contents under a fresh module name in every `sm*` directory of that cache, installs the real payload under that name, and runs a case: the output is bit-identical to the existing kernel, one launch, and the planted files are byte for byte unchanged afterwards. A module that consulted the cache would have loaded them, or failed on them and rewritten them. The test removes its files when it ends.

## Commands and results

Builds, outside the lock:

```bash
cargo test -p mlxcel-mlx-pin cuda_oxide
MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 cargo test --profile test-fast -p mlxcel-core --lib --features cuda --no-run
MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 cargo test --profile test-fast -p mlxcel-core --features cuda --test cuda_oxide_fused_norm_layer_route --no-run
```

GPU runs. `$LIB` is the `mlxcel_core` unit-test binary and `$LAYER` the `cuda_oxide_fused_norm_layer_route` binary that the builds above print.

| # | Command | Result |
| --- | --- | --- |
| 1 | `gpu-lock run --tag issue-2833 -- $LIB cuda_oxide --test-threads=1 --nocapture` | 31 passed, 0 failed |
| 2 | `MLX_USE_CUDA_GRAPHS=0 gpu-lock run --tag issue-2833 -- $LIB cuda_oxide --test-threads=1 --nocapture` | 31 passed, 0 failed |
| 3 | `MLXCEL_CUDA_GRAPH_CAPTURE_BARRIER=0 gpu-lock run --tag issue-2833 -- $LIB cuda_oxide_fused_norm_exec --test-threads=1` | 8 passed, 0 failed |
| 4 | `gpu-lock run --tag issue-2833 -- $LIB fused_norm_parity_tests --test-threads=1` | 10 passed, 0 failed (existing suite, opted-in build, nothing installed) |
| 5 | `gpu-lock run --tag issue-2833 -- $LAYER --test-threads=1 --nocapture` | 1 passed: graph path, adapter untouched (0 routes) |
| 6 | `MLXCEL_FUSED_ADD_RMSNORM=1 gpu-lock run --tag issue-2833 -- $LAYER --test-threads=1 --nocapture` | 1 passed: 2 existing-kernel routes, 0 cuda-oxide |
| 7 | `MLXCEL_FUSED_ADD_RMSNORM=1 MLXCEL_CUDA_OXIDE_FUSED_NORM=1 gpu-lock run --tag issue-2833 -- $LAYER --test-threads=1 --nocapture` | 1 passed: 2 cuda-oxide routes, 2 launches, 0 existing-kernel |
| 8 | row 7 with `MLX_USE_CUDA_GRAPHS=0` | 1 passed: 2 cuda-oxide routes, 2 launches |
| 9 | row 7 with `MLXCEL_CUDA_OXIDE_FUSED_NORM_DIR=$PWD/spike/cuda-oxide-fused-norm` | 1 passed: 2 cuda-oxide routes, 2 launches |
| 10 | `gpu-lock run --tag issue-2833 -- $LIB --test-threads=1` (the whole opted-in lib binary, one process) | 2517 passed, 0 failed, 6 ignored, 396.72 s. Run on the commit before the `cuda_oxide_abi.rs` split, which moved code without changing it; rows 1, 2 and 7 were repeated after it. Not repeated after the review follow-ups below (3 tests added, the disk-cache bypass, the grid-limit refusal); rows 1 to 9 were, with `--tag pr-3451-finalize`. |

The 31 tests of rows 1 and 2 are 12 artifact-validation tests that need no GPU, 8 execution tests, 10 fallback and fault tests, and the matrix. `cargo test -p mlxcel-mlx-pin cuda_oxide` (the build gate: SHA-256 vectors, the switch, staging, tampered PTX, wrong manifest, missing files) passed 8 of 8 without building MLX.

### Numerical matrix

`the_2832_matrix_passes_through_the_adapter_on_the_oxide_route` rebuilds the #2832 fixtures from a port of the standalone harness's case table and seeded generator (`src/cuda_oxide_fused_norm_fixtures.rs`), since the harness itself needs the cuda-oxide toolchain. It runs each case through the existing fused kernel and the graph path with nothing installed, then through the cuda-oxide node, and scores the outputs with the criteria #2832 froze: the `fused_norm_parity_tests::tolerance_for` budgets for `normed` against both references, `new_residual` equal to the fused kernel bit for bit and within 0 (f32) or 1 % (16-bit) differing elements of the graph `add`, and non-finite rows class-exact in all three.

```
RESULT oxide-adapter-matrix cases=130 elements=2181390 failed=0 bitwise_identical_to_fused_cuda=130 nonfinite_cases=5 oxide_routes=130 launches=130 native_routes=0
```

The same line was printed with graph capture on (row 1) and off (row 2). The test asserts the case count, the element count, one route and one launch per case, and zero existing-kernel routes during the cuda-oxide pass; the reference pass asserts the opposite.

### Scheduling and lifetime

Each test computes its reference with the existing fused kernel on the same inputs and requires identical bytes, plus the launch count.

| Test | What it exercises |
| --- | --- |
| `a_lazy_chain_launches_nothing_until_it_is_evaluated` | Two chained joins, the second consuming both outputs of the first. After construction: 2 routes, 0 launches. After evaluation: 2 launches. f32 and bf16 with an f32 weight. |
| `a_replayed_graph_gives_the_same_bytes_every_time` | The same two-join graph 60 times over three input sets of one shape, so the cached CUDA graph is rebound and relaunched. 120 launches. |
| `joins_order_correctly_between_captured_child_graphs` | `matmul -> join -> matmul -> join`, with a third matmul on the first join's residual output, evaluated as one graph, 12 times. Each matmul is a captured cuBLAS child graph. Also run with the capture barrier off (row 3). |
| `outputs_survive_allocator_reuse` | 40 rounds over five shapes and dtype pairs with outputs dropped each round and the buffer cache cleared every ninth, while one early pair of outputs is held and re-read at the end. |
| `inputs_and_sibling_outputs_dropped_by_the_caller_stay_alive` | The caller drops the inputs and one of the two outputs before evaluating the other, with `eval` and with `async_eval`. 16 x 8192 bf16. |
| `non_contiguous_inputs_are_copied_like_the_existing_port_copies_them` | A transposed `x`, a `residual` that is an offset slice of a taller array, a `weight` sliced out of a longer vector. |
| `several_threads_can_drive_the_route_at_once` | Four threads, each building and evaluating its own arrays, 25 evaluations each, four dtype pairs. 100 launches. Each thread runs on its thread's default stream; which one was not inspected. |
| `distinct_gpu_streams_can_drive_the_route_at_once` | Three threads, each on its own stream made with `new_stream_on_gpu(0)` and installed as that thread's default stream, 20 evaluations each over three dtype pairs, a barrier lining the iterations up so the launches overlap. The three `stream_index` values are asserted distinct, every output equals the existing kernel's bytes, and the process-wide counters read 60 routes, 60 launches, 0 existing-kernel routes, so no stream fell back or skipped a launch. |

The captured-child-graph case was also inspected, not only compared. `MLX_SAVE_CUDA_GRAPHS_DOT_FILE=<prefix> MLXCEL_CUDA_GRAPHS_DOT_VERBOSE=1 gpu-lock run --tag issue-2833 -- $LIB joins_order_correctly_between_captured_child_graphs --test-threads=1` wrote one dump per committed graph. Each of the 12 cuda-oxide rounds is a single CUDA graph of seven nodes: three child graphs (the matmuls) and four kernel nodes, two of them `fused_add_rms_norm_f32`. Its edges are `0->1, 1->2, 1->3, 1->4, 2->3, 3->4, 3->5, 4->5, 4->6, 5->6` with node 0 the first matmul, node 1 the first join, nodes 2 and 4 the other matmuls and node 3 the second join. So the first join waits for the matmul that produces its input, both later matmuls wait for the join whose outputs they read, and the second join waits for both of its producers.

### Fallback and faults

| Test | What it shows |
| --- | --- |
| `the_route_is_off_until_an_artifact_is_installed`, `the_runtime_switch_decides_whether_anything_is_installed` | Nothing installed means the existing kernel; only `1`/`true`/`on`/`yes` installs; a directory without the artifact disables the route instead of falling back to the embedded copy. |
| `a_dtype_pair_without_an_entry_takes_the_existing_kernel` | f32/f16, f32/bf16, f16/bf16 and bf16/f16 take the existing kernel with identical output, 0 launches, reason `DtypePair`. |
| `more_elements_than_the_validated_limit_take_the_existing_kernel` | A lazy 70000 x 65536 input (never evaluated) is routed to the existing kernel; 65535 x 65536 takes the route. |
| `more_rows_than_a_grid_holds_take_the_existing_kernel` | A lazy 65536 x 40000 x 1 input (2.6e9 elements, inside the element limit, 2.6e9 rows) is routed to the existing kernel with reason `GridLimit`, decided before enqueue; 65536 x 32767 x 1 (2,147,352,576 rows) takes the route. |
| `a_cpu_default_device_never_takes_the_route` | With the CPU as default device the adapter returns no node (reason `DeviceNotGpu`), 0 cuda-oxide routes. |
| `a_device_of_another_architecture_is_refused_before_anything_is_built` | The target check refuses sm_90, sm_86, sm_120 and a non-CUDA device. |
| `a_warm_cache_cannot_mask_a_wrong_manifest_or_a_missing_artifact` | See Cache identity. |
| `an_evaluation_time_failure_is_an_error_not_a_retry` | A payload that cannot load, installed around the validator on purpose, makes `try_eval` return an error twice, with 0 launches and 0 existing-kernel routes, and leaves no entry in the PTX disk cache. The same for the real payload declared for sm_90. Afterwards both implementations still evaluate on the same stream. |

## Ordinary builds

| Check | Result |
| --- | --- |
| `cargo test --profile test-fast -p mlxcel-core --lib --features cuda --no-run` without the variable, then `nm` on the test binary for `mlxcel_oxide_fused_norm` and `OxideFusedAddRmsNorm` | 0 matching symbols. The opted-in build of the same sources has them. The default binary lists 2495 tests, the opted-in one 2526 (the 31 above). |
| Same binary, `gpu-lock run --tag issue-2833 -- $LIB fused_norm --test-threads=1` | 10 passed, 0 failed: the existing `fused_norm_parity_tests`, and no `cuda_oxide` test exists in the binary |
| `spike/cuda-oxide-fused-norm/scripts/check-clean-build.sh` | `PASS clean-build toolchain='rustc 1.97.1 (8bab26f4f 2026-07-14)' oxide_compiler=absent` |
| `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `mlxcelverse/CMakeLists.txt` | Unchanged by this change. No dependency, build dependency or feature was added. |
| `overlays/cuda/mlx/backend/cuda/jit_module.{cpp,h}` | One trailing parameter, `DiskCache disk_cache = DiskCache::Use` (a new `enum class DiskCache { Use, Bypass }`), on `JitModule` and `get_jit_module`; the constructor takes an empty cache directory for `Bypass`. A typed enum rather than a second `bool`, so it cannot be confused with upstream's `use_disk_cache` flag, which the overlay ignores since #2492. No existing caller passes it, so their behavior is unchanged. |

Without the build variable the bridge compiles the same sources with the same include paths as before (MLX itself is built with the changed `jit_module` overlay in every profile); the only edit to a file the bridge compiles is an `#ifdef MLXCEL_CUDA_OXIDE_FUSED_NORM` block in `fused_norm.cpp` and its include. The Rust modules are behind `cfg(mlxcel_cuda_oxide_fused_norm)`, which only the build script sets. The adapter is not reachable on Metal, ROCm, CPU-only or Windows builds: the build script refuses the variable there.

## Not covered

- Throughput, and any comparison of launch or graph overhead against the existing kernel. That is #2834.
- A second architecture. The committed PTX is `.target sm_121`; on any other device the install is refused, which is tested, but the route itself has only ever run on GB10.
- More than one GPU. The target check reads device 0. A node evaluated on a device of another architecture throws (tested by declaring the artifact for sm_90), but no multi-GPU host was used.
- Compute Sanitizer on the adapter path. #2832 ran it on the standalone kernel; it was not repeated through MLX.
- Three review items are documented follow-ups, not implemented: every route decision takes a shared lock on the installed-artifact mutex; `last_fallback()` is last-writer-wins across threads, so under concurrency it reports some recent decision, not necessarily the caller's; and the override directory's artifact files are read without a size bound or a regular-file check, so a FIFO there would block the install.
- The existing port the adapter falls back to computes its grid from `static_cast<int>(x.size() / dim)` in `fused_norm.cpp`, which wraps above 2^31-1 rows. That predates this change; the adapter now refuses such shapes before enqueue, but the existing port still receives them.
- The element limit is tested at the routing decision on lazy shapes only. Nothing near 2^32 elements was allocated or launched.
- A checkpoint. No model was loaded. The route as model code reaches it is covered at `layers::fused_add_rms_norm` with the real environment switches (rows 5 to 9), not by a generation run with a logit comparison.
- `MLXCEL_CUDA_OXIDE_FUSED_NORM_DIR` accepts any payload whose manifests pin it. The manifests give integrity, not authenticity, so the directory has to be as trusted as the binary. The embedded artifact has no such exposure.
