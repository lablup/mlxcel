# Technical Report: PR #1562 - tagging the pre-Ampere grouped GEMM arm Sm70 instead of Sm75

**Date**: 2026-09-01

**Author**: mlxcel maintainers

**Status**: Completed. The branch the issue suspected of being dead turned out to be live on the epic's own MoE checkpoint, the tag on it was wrong, and the retag moves no device code. One acceptance criterion epic #1536 deferred to GB10 is closed here instead, two are left unticked because they need hardware this host does not have, and one pre-existing test failure was surfaced and shown not to be caused by this change.

---

## Executive Summary

`dispatch_cutlass_arch` in `patches/mlx/backend/cuda/gemms/grouped_gemm_unaligned.cu` mapped every device below compute capability 8.0 to `cutlass::arch::Sm75`. That tag names the `m16n8k8` MMA that Turing introduced. A Tesla V100 is compute capability 7.0, one generation earlier, and has only the `8x8x4` HMMA shape. The dispatch therefore described hardware the device does not have on every Volta part that reached it, and `get_grouped_mm_funcion` opened with a matching `Sm75` placeholder.

Issue #1544 was filed verification-first: it asked whether this is a live defect, a dead branch, or benign, and said that a negative result would be a complete answer. The answer is none of those three.

**The branch is live, on the checkpoint epic #1536 benchmarks.** Issue #629's sorted-MoE prefill fast path routes a quantized `GatherQMM` into `cutlass_grouped_gemm_unaligned` once the batch clears `B >= min_rows * num_experts`; for `gemma-4-26b-a4b-it-4bit` that is a 128-token prompt. An nsys profile at 573 prompt tokens shows 180 `cutlass::Kernel<GemmGrouped>` launches taking 3.8% of GPU time. The #1538 baseline profiled MoE at a 46-token prompt, below the gate, which is why its kernel table shows none and why the issue expected the branch to be dead.

**Nothing was computing wrong, and that was measured rather than inferred.** The pre-Ampere arm resolves to `GemmConfiguration`'s primary template, which is `OpClassSimt` with `InstructionShape<1, 1, 1>`. Both tensor-core specializations are constrained on `Arch::kMinComputeCapability >= 80`, so no pre-Ampere tag can select an MMA atom of any shape. CUTLASS consequently erases the tag: the kernel that runs is an `MmaSimt` / `OpMultiplyAdd` / `MmaPipelined` instantiation whose mangled name carries no architecture token at all.

**The retag moves no device code on any architecture.** Compiling the translation unit before and after at `compute_70`, `compute_80` and `compute_121` yields the same 51 device symbols with byte-identical SASS bodies at all three, 144 MB of dump compared per symbol. That is also what decided the design against giving Turing its own arm.

What the change delivers is a correct description, a decision that can be tested without a GPU, two `static_assert`s on the preconditions that make the single pre-Ampere arm legitimate, and the numeric gate the grouped GEMM never had.

## 1. Problem Statement

The dispatch, before:

```cpp
template <typename F>
void dispatch_cutlass_arch(cu::Device& device, F&& f) {
  if (device.compute_capability_major() < 8) {
    f(type_identity<cutlass::arch::Sm75>{});
  } else if (device.compute_capability_major() == 8) {
    f(type_identity<cutlass::arch::Sm80>{});
  } else {
    f(type_identity<cutlass::arch::Sm90>{});
  }
}
```

The `< 8` arm is written as though pre-Ampere means Turing. It does not: Volta is a generation below Turing. CUTLASS guards its Turing MMA behind `CUTLASS_ARCH_MMA_SM75_SUPPORTED`, so an `Sm75` configuration compiled for an sm_70 target does not necessarily fail the build. It can compile to a path that traps or degenerates at runtime, which is a failure mode that greedy text generation does not reveal.

Two things had to be established before the tag could responsibly be touched, and in that order. Whether the branch executes at all on this part, and if it does, whether its output is right. Precedent from #1541 in this same epic: `gemms/gather_gemm.cu` is compiled into `libmlx.a` and has no undefined reference anywhere in the archive, because mlxcel's own `matmul.cpp` overlay routes `GatherMM::eval_gpu` elsewhere. A translation unit being compiled proves nothing about it being called.

## 2. Change Summary

Ten files. The functional change is about forty lines across two of them; the rest is evidence and tests.

`gemms/grouped_gemm_arch.h` (new) holds the architecture decision as a `constexpr` function of the compute capability major version, with the measurements that justify one arm covering Volta and Turing and the condition under which that stops being safe. `grouped_gemm_unaligned.cu` switches on it, replaces the `Sm75` placeholder in `get_grouped_mm_funcion` with a named `GroupedGemmFn` initialised to `nullptr` behind a loud guard, and gains two `static_assert`s.

`cpp/grouped_gemm_arch_probe.cpp` (new) plus a line in `build.rs` expose the shipped function to Rust through a C shim, compiled unconditionally rather than behind the `cuda` feature, following the pattern #1541 established. `grouped_gemm_arch_tests.rs` (new) enumerates the mapping over every architecture; `grouped_gemm_numeric_tests.rs` (new) compares `gather_mm` against an `f64` dense per-expert reference on the device.

`docs/benchmark_results/grouped-gemm-arch-v100-2026-08-31.md` (new) is the full record. The #1538 baseline gains its reserved post-program row and an annotation on its MoE section explaining why no grouped GEMM appears in its kernel table.

## 3. Technical Decisions

### 3.1 Reachability was settled before the dispatch was touched

Three nsys runs with `--cuda-graph-trace=node`, same binary and checkpoint, varying only the prompt length and the kill switch:

| Run | Prompt | `B` | `cutlass::Kernel<GemmGrouped>` | `prepare_grouped_mm_data` |
|---|---|---|---|---|
| default | 573 tokens | 4584 | 180 inst, 853.8 ms, 3.8% | 180 inst |
| `MLXCEL_GATHER_QMM_GROUPED=0` | 573 tokens | 4584 | absent | absent |
| default | 46 tokens | 368 | absent | absent |

The third run reproduces the #1538 baseline's MoE profile exactly at 326 `qmm_naive` instances, which is the cross-check that it is the same measurement at a different prompt length rather than a different measurement.

The static half reproduces #1541's technique and inverts its result: `nm -C` on the built `libmlx.a` finds undefined references to all three of `cutlass_gather_mm`, `cutlass_grouped_gemm_unaligned` and `cutlass_segmented_mm` from `matmul.cpp.o`, while `mlx::core::gather_mm(bool, bool, ...)` in `gather_gemm.cu` still has none.

### 3.2 A single pre-Ampere arm, decided by measurement rather than by taste

The issue offered a choice between retagging the `< 8` arm and splitting Turing out. The control settles it: compiling the file with an `Sm70` arm added alongside the `Sm75` one emits the identical 51 device symbols and a byte-identical 493,538-line SASS dump, against 26 extra host-side text symbols and 194,704 more bytes of object. A separate Turing arm buys a second host-side copy of instantiations that emit the same device code.

The arm therefore stays one arm and its tag names the floor of the range it covers. That is only legitimate while the configuration under it is SIMT, which is a property of the configuration and not a promise from CUTLASS, so the file now asserts it:

```cpp
static_assert(
    std::is_same_v<
        GemmConfiguration<float, cutlass::arch::Sm70, 8, true>::OpClass,
        cutlass::arch::OpClassSimt>, ...);
```

The `kEnableTF32 = true` instantiation is the load-bearing one: it is the arm `MLX_ENABLE_TF32` reaches, and asserting it rules out a pre-Ampere tag ever selecting a tensor-core specialization. The day the pre-Ampere arm gains a tensor-core operator, which is what #1543 is doing next door for `qmm_naive`, the build fails and the tag has to be revisited instead of silently downgrading Turing.

### 3.3 The decision is a pure function, so its non-regression claim is a unit test rather than a GPU

The tag mapping depends on one integer. Moving it into a header and exposing it through a C shim makes "the `== 8` and `> 8` arms are untouched" answerable by enumeration on any host, including one with no NVIDIA hardware. `only_the_pre_ampere_arm_changed` walks compute capability 0 through 32, compares the shipped function against a restatement of the pre-#1544 mapping, and asserts that nothing at or above 8 moves. Epic #1536 would otherwise have deferred that to a GB10 host.

`Sm75` is deliberately absent from the enum. An enumerator there is a template argument the GEMM is instantiated over, so listing a tag the function never returns would emit a dead copy of every instantiation in the arm.

### 3.4 The default initializer was dead, and it named an architecture

`get_grouped_mm_funcion` opened with `grouped_gemm_v2<GemmConfiguration<float, cutlass::arch::Sm75>>`. It was never the value returned, because `dispatch_float_types` throws on every dtype it does not dispatch and every dtype it does assigns `fun`. It did force one template instantiation into the binary purely to serve as an initializer, and it read like a pre-Ampere default. It is now a named function-pointer type initialised to `nullptr`, with a throw if selection ever falls through, so no architecture is named there at all.

### 3.5 `kStages` is structurally unreachable from a pre-Ampere tag

The issue asked whether `static const int kStages = 3; // use SM80_CP_ASYNC` can be applied where `cp.async` does not exist. It cannot: that member belongs to `GemmConfiguration<float, cutlass::arch::Sm80, kAlignmentC, true>`, an explicit full specialization on `Sm80` that no other tag can name. The pre-Ampere arm gets the primary template's `kStages = 2`, and the profiled kernel is an `MmaPipelined` instantiation, which is the 2-stage mainloop. Asserted rather than left as reasoning, because a 3-stage pipeline without `cp.async` is either a build failure or a silent serialization and neither announces itself.

## 4. Validation

**Reachability**: three nsys profiles, table in 3.1.

**Correctness, end to end**: greedy at `-t 0.0`, 64 tokens, a 285-token prompt above the gate, comparing the grouped path against the legacy `qmm_naive` path. All four combinations of {before, after} x {grouped, legacy} produce a byte-identical 1,793-character continuation. The same pair also measures what #629 bought on this part: prefill 12,642.88 ms grouped against 28,016.23 ms legacy, a 2.22x speedup at identical output.

**Correctness, unit level**: `grouped_gemm_numeric_tests.rs` compares `gather_mm` against a dense per-expert reference accumulated in `f64` from the same host bytes that were uploaded, on `gemma-4-26b-a4b-it`'s real expert dims (`k = 2816, n = 704` and the transposed orientation), across both entry points, both `kAlignmentC` arms, both operand layouts, and f32, bf16 and f16. A constant-per-expert case makes a mis-gathered index move a whole slab rather than a rounding error. The pre-existing `test_gather_mm` asserted the output shape and never looked at a value.

**Device-code delta**: per-symbol `cuobjdump --dump-sass` comparison at three targets.

| Target | Symbols before | Symbols after | Only in one | Bodies differing | SASS compared |
|---|---|---|---|---|---|
| `compute_70` | 51 | 51 | 0 | 0 | 58,211,476 bytes |
| `compute_80` | 51 | 51 | 0 | 0 | 35,077,812 bytes |
| `compute_121` | 51 | 51 | 0 | 0 | 50,952,400 bytes |

**Throughput**: five repetitions per cell, warm PTX cache, `nvidia-smi --query-compute-apps` asserted empty before every run, decode as a slope over `-n 40` to `-n 120`. 4-bit decode 29.66 to 29.73 ms/token, 8-bit 31.68 to 32.71, 4-bit prefill 12,720 to 12,644 ms, 8-bit prefill 15,212 to 15,606 ms. Every delta is smaller than the before arm's own repetition spread, which is the only outcome byte-identical device code permits.

**Suites**: `cargo test -p mlxcel-core --release --features cuda --lib -- --test-threads=1` reports 1672 passed, 1 failed, 1 ignored. `cargo clippy -p mlxcel-core --release --features cuda --lib --tests -- -D warnings` and `cargo fmt -p mlxcel-core -- --check` are clean. `cuobjdump --list-elf libmlx.a` reports 96 cubins, every one sm_70.

## 5. Validation Limits and Follow-up

### 5.1 One pre-existing test failure, isolated rather than assumed

`sampling::tests::temperature_one_support_unchanged` fails, asserting bit-exactness of `fused_sample_probs` at `T = 1.0` and missing by 1 ULP on 5 of 64 entries. This branch touches no sampling code, but that is an argument rather than evidence, so the change was isolated: reverting `grouped_gemm_unaligned.cu` to its `c2e54939` version, rebuilding, and re-running the test reproduces a byte-identical failure. It is pre-existing on `main` and not caused by this change. It is the same class of sm_70 float-reduction non-determinism #1557 recorded for `tests/cuda_qmm_determinism.rs`. This is also the first run of the full `mlxcel-core` library suite in this epic; #1559 ran only a targeted filter, which is why it had not surfaced. It deserves its own issue.

### 5.2 The baseline row is re-measured rather than quoted

The #1538 row this issue reserved reads 38.61 and 34.29 ms/token. That record predates #1539, which moved MoE decode, so the before column here was re-measured on this host from this worktree. At 29.66 against 31.68 ms/token the MoE pair also inverts the baseline's 4-bit against 8-bit finding, in the same direction #1539's own record reports for the dense pair.

### 5.3 The `cuFuncSetAttribute` gap inherited from #1541: recorded, not fixed

#1541 handed this issue the missing dynamic shared-memory opt-in in `gather_gemm.cu`. Two findings decided against fixing it here. The gap is in the shared encoder rather than one launch site: `CommandEncoder::add_kernel_node_raw` sets `sharedMemBytes` and calls `cudaGraphAddKernelNode` without ever calling `cudaFuncSetAttribute`, so every launch site must opt in itself. But the reachable Volta configuration is nowhere near the ceiling, measured from the profile rather than computed: the grouped GEMM asks for 10,320 bytes against the 49,152-byte non-opt-in limit, 21% of it. `gather_gemm.cu` also remains unreachable in an mlxcel build, which reproduces here.

What is genuinely open, and new: the sm_80 tensor-core configurations in this same file use much larger tiles and, in the tf32 arm, three stages. That is the plausible place for this path to trip the ceiling, and it needs an Ampere-or-later part to answer. Left as a follow-up rather than fixed blind.

### 5.4 The SASS-diff technique transfers here, unlike #1541

#1539 recovered its sm_80+ criterion by compiling at sm_80 and sm_121 and diffing `cuobjdump --dump-sass`. #1541 found the technique does not transfer when the translation unit is host dispatch only and its object holds no device code, and said so instead of claiming a meaningless identical diff. Which case a file is in has to be checked before the diff means anything. This one holds real device code: `grouped_gemm_unaligned.cu.o` carries one sm_70 cubin with 51 functions, so the identical dumps in section 4 are a result. One methodological note: whole-file dumps differ between the two builds because the cubin emits the same functions in a different order, so the comparison has to be per symbol.

### 5.5 Deferred to GB10, and one criterion recovered

No sm_80-or-later part exists on this host. Per epic #1536's `## GB10 (sm_121) continuation`:

- **GB10 MoE output byte-identical** and **GB10 MoE throughput unmoved**: not run, left unticked. What is recovered locally is the mechanism rather than the measurement. The tag mapping is provably unchanged for every compute capability at or above 8, and the device code that mapping selects is byte-identical before and after at `compute_80` and `compute_121`, per symbol. That leaves no route by which GB10 output or throughput could move, but both are measurements and this is an argument.
- **The `== 8` and `> 8` branches untouched**: closed here rather than deferred, by enumeration on the shipped function plus the cross-architecture SASS comparison.
- **`cargo test --features cuda` green on sm_121**: needs a GB10 host.

The `CUDA sm_70 compile` CI check is not evidence for any of this. CUDA 13 removed Volta support and cannot compile `compute_70`, so that job passes in about 11 seconds by skipping; the local build is the only real coverage.

### 5.6 Re-check after #1543

#1543 gives Volta a tensor-core MMA atom for quantized GEMM. If that work ever reaches the grouped GEMM's pre-Ampere `GemmConfiguration`, the `static_assert` in section 3.2 fires and the single pre-Ampere arm has to be split so Turing keeps `m16n8k8`. That is the intended trigger, and it is why the assert names `grouped_gemm_arch.h`.

## References

- Issue #1544, and epic #1536, the Volta decode program.
- Full measurement record: `docs/benchmark_results/grouped-gemm-arch-v100-2026-08-31.md`.
- Baseline this is a delta against: `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md` (#1538).
- The sorted-MoE prefill fast path that makes this branch reachable on a quantized checkpoint: #629, and `docs/benchmark_results/moe-prefill-grouped-gemm-gb10-2026-07-10.md`.
- The `cuFuncSetAttribute` gap inherited from #1541, and its `gather_gemm.cu` reachability finding: `docs/benchmark_results/qmm-naive-tile-v100-2026-08-31.md`.
- The SASS-diff technique: #1539 for the case where it transfers, #1541 for the case where it does not.
- The pre-existing sm_70 bit-exactness class: #1557, on `tests/cuda_qmm_determinism.rs`.
