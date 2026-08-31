# Grouped GEMM on Volta: the `Sm75` tag is reachable, wrong, and inert

Issue #1544 of epic #1536. It asked whether `grouped_gemm_unaligned.cu` mapping every pre-Ampere part to `cutlass::arch::Sm75` is a live defect, a dead branch, or benign, and said in as many words that a negative result would be a complete answer. The answer is none of the three, and it is more interesting than any of them: **the branch is live on the epic's own MoE checkpoint, the tag on it is wrong, and the tag reaches no device code at all.** All three halves are measured below.

This record follows the six methodology rules of `volta-sm70-baseline-2026-08-31.md` without exception. Read that document first; everything here is a delta against it.

## Environment

Identical to the baseline record except for the commit.

| Item | Value |
|------|-------|
| **GPU** | Tesla V100-PCIE-32GB, sm_70, compute capability 7.0, 32 GB HBM2, 80 SMs |
| **Driver / toolkit** | 575.51.03 / CUDA 12.9.41 |
| **mlxcel** | 0.6.0 on `fix/issue-1544-grouped-gemm-arch-tag`, branched from `c2e54939` (main, after #1537, #1538, #1539 and #1541) |
| **MLX** | pinned commit `9a795735` |
| **nsys** | Nsight Systems 2025.3.1.90 |
| **Build** | `MLX_CUDA_ARCHITECTURES=70 cargo build --release --features cuda`; `cuobjdump --list-elf libmlx.a` reports 96 cubins, every one sm_70 |
| **PTX cache** | warm. The first run on this build reported `Compiling CUDA kernels (first run on this host)` and took 54.0 s of prefill against 12.6 s warm; that run was discarded and every number below is warm-cache |
| **Contention** | none. `nvidia-smi --query-compute-apps` was asserted empty before every timed run by the harness |

## Is it reached? Yes, on `gemma-4-26b-a4b-it-4bit`, in prefill

The issue expected the answer to be no, on the reasoning that mlxcel's fused MoE path bypasses the grouped GEMM. That reasoning is right about decode and wrong about prefill, and the difference is a prompt-length gate nobody had crossed.

`quantized.cpp:359` carries the #629 sorted-MoE prefill fast path: when `right_sorted_ && transpose_ && M == 1` and the batch clears `B >= min_rows * E` (`min_rows` default 8, `MLXCEL_GATHER_QMM_GROUPED_MIN_ROWS`), a quantized `GatherQMM` dequantizes the whole expert stack once and hands the result to `cutlass_grouped_gemm_unaligned`. For this checkpoint `E = 128` and `top_k = 8`, so the gate is `prompt_tokens * 8 >= 1024`, that is **128 prompt tokens**. The baseline record profiled MoE at a 46-token prompt, which is `B = 368` and below the gate, which is exactly why no CUTLASS kernel appears in it.

Three runs, `nsys profile -t cuda,nvtx --cuda-graph-trace=node`, same binary, same checkpoint:

| Run | Prompt | `B` | `cutlass::Kernel<GemmGrouped>` | `prepare_grouped_mm_data` | `affine_dequantize` |
|---|---|---|---|---|---|
| **A** default | 573 tokens | 4584 | **180 inst, 853.8 ms, 3.8% of GPU time** | 180 inst | 185 inst, 6934 ms |
| **C** `MLXCEL_GATHER_QMM_GROUPED=0` | 573 tokens | 4584 | absent | absent | 5 inst, 0.0 ms |
| **D** default | 46 tokens | 368 | absent | absent | 4 inst, 0.0 ms |

Run D also reproduces the baseline record's MoE profile exactly, at 326 `qmm_naive` instances, which is the cross-check that D is the same measurement the baseline made and not a different one.

So the answer to the issue's first question is: **reached, 180 launches per prefill on a 573-token prompt, on the checkpoint the epic names.** It is not a dead branch, and it is not reached only by some hypothetical non-quantized checkpoint. It is also reached by every non-quantized MoE checkpoint through `SwitchLinear::Regular::forward`, which is a second live caller, but that one needed no gate to find.

## Why the wrong tag never broke anything

The issue's premise is that `cutlass::arch::Sm75` selects Turing's `m16n8k8` MMA, which does not exist on sm_70. The premise is right about what the tag names and wrong about what this configuration does with it.

The pre-Ampere arm resolves to `GemmConfiguration`'s primary template, which is `cutlass::arch::OpClassSimt` with `InstructionShape<1, 1, 1>`: plain FFMA, no MMA atom of any shape. The two tensor-core specializations are both constrained on `Arch::kMinComputeCapability >= 80`, so no pre-Ampere tag can select either one, with or without `MLX_ENABLE_TF32`. The kernel that actually ran in run A says the same thing in its own name:

```
void cutlass::Kernel<cutlass::gemm::kernel::GemmGrouped<
  cutlass::gemm::threadblock::MmaPipelined<
    cutlass::gemm::GemmShape<(int)128, (int)128, (int)8>, ...
    cutlass::gemm::warp::MmaSimt<cutlass::gemm::GemmShape<(int)32, (int)64, (int)8>, ...
      cutlass::arch::OpMultiplyAdd ...
```

`MmaSimt`, `OpMultiplyAdd`, `MmaPipelined`. No `MmaTensorOp`, and no `Sm70`, `Sm75` or `Sm80` token anywhere in the mangled name: CUTLASS consumes the arch tag during host-side type selection and, for this configuration, erases it. That is the whole reason a V100 has been running a Turing-tagged grouped GEMM and producing correct text.

## Correctness, measured rather than inferred

A CUTLASS path compiled for an architecture it does not match can return plausible numbers that are wrong, and greedy decoding will not show it. Two independent checks, because "the text looks fine" is not one.

**End to end, against the reference implementation.** The same binary, the same 285-token prompt (`B = 2280`, above the gate), greedy at `-t 0.0`, 64 tokens, with and without `MLXCEL_GATHER_QMM_GROUPED`. The grouped GEMM path and the legacy `qmm_naive` path produce a **byte-identical 1,793-character continuation**. This is the comparison the issue asked for, on the shapes the model actually uses, against a reference that shares none of the grouped GEMM's code.

The same pair also measures what #629 bought on this part: prefill **12,642.88 ms grouped against 28,016.23 ms legacy, a 2.22x speedup**, at identical output. The grouped GEMM is not incidental to MoE prefill on Volta; it is most of it.

**Unit level, against an `f64` host reference.** `grouped_gemm_arch_tests.rs` compares `gather_mm` against a dense per-expert reference computed in `f64` from the same host bytes that were uploaded, on `gemma-4-26b-a4b-it`'s real expert dims (`k = 2816, n = 704` and the transposed `704 x 2816`), across both entry points (`cutlass_gather_mm` for `m > 1`, `cutlass_grouped_gemm_unaligned` for the sorted `m == 1` case), both alignment arms (`n = 704` and `n = 703`), both operand layouts, and `float32`, `bfloat16` and `float16`. A separate case gives each expert a distinct constant matrix so a mis-gathered index moves every element of a slab by a whole multiple instead of a rounding error.

The pre-existing `test_gather_mm` asserted the output shape and never looked at a value, which is how a wrongly tagged grouped GEMM went four architectures without anyone checking its arithmetic.

## The retag, and what it moves: nothing

`dispatch_cutlass_arch`'s pre-Ampere arm now selects `cutlass::arch::Sm70`, and the decision moved into `gemms/grouped_gemm_arch.h` as a pure function of the compute capability major version so it can be enumerated without a GPU. The `Sm75` placeholder that initialised `fun` in `get_grouped_mm_funcion` is gone; `dispatch_float_types` throws on every dtype that would have left it in place, so it was never returned, and it existed only to force one more template instantiation into the binary.

How much device code that moves, by compiling the translation unit before and after with the production flag set and comparing `cuobjdump --dump-sass` per symbol:

| Target | Symbols before | Symbols after | Only in one | Bodies differing | SASS text compared |
|---|---|---|---|---|---|
| `compute_70` | 51 | 51 | 0 | **0** | 58,211,476 bytes |
| `compute_80` | 51 | 51 | 0 | **0** | 35,077,812 bytes |
| `compute_121` | 51 | 51 | 0 | **0** | 50,952,400 bytes |

Whole-file dumps differ only in the order the cubin emits functions in, which is why the comparison is per symbol rather than by `diff`. The technique is #1539's, and unlike #1541's case it transfers cleanly here: this translation unit holds real device code (`grouped_gemm_unaligned.cu.o` carries one `sm_70` cubin with 51 functions), so an identical SASS dump is a result rather than a tautology.

A **control** run says the same thing from the other side. Adding an `Sm70` arm *alongside* the `Sm75` one, so the object holds both, emits the identical 51 device symbols and a byte-identical 493,538-line dump, against 26 extra host-side text symbols and 194,704 more bytes of object. That is what settled the design: a separate Turing arm would buy a second host-side copy of instantiations that emit the same device code, so the pre-Ampere arm stays one arm and its tag names the floor of the range it covers.

The erasure is a property of the configuration, not a promise from CUTLASS. `grouped_gemm_unaligned.cu` now carries a `static_assert` pair on the pre-Ampere configuration's `OpClass`, including under `kEnableTF32`, so that the day someone gives this arm a tensor-core operator (the obvious candidate is the Volta MMA work in #1543) the build fails and the tag has to be revisited instead of silently downgrading Turing.

## `kStages` and `cp.async`

The issue asked whether the `static const int kStages = 3; // use SM80_CP_ASYNC` at `:245` can be applied to a pre-Ampere arch tag, where `cp.async` does not exist. **It cannot**, and the reason is structural rather than circumstantial: that member belongs to `GemmConfiguration<float, cutlass::arch::Sm80, kAlignmentC, true>`, an explicit full specialization on `cutlass::arch::Sm80`. No other tag can name it. The pre-Ampere arm gets the primary template's `kStages = 2`, which `MmaPipelined` implements with ordinary global-to-shared copies, and the kernel that ran in run A is a `MmaPipelined` instantiation.

Asserted rather than left as reasoning, because a 3-stage pipeline without `cp.async` is either a build failure or a silent serialization and neither announces itself:

```cpp
static_assert(GemmConfiguration<float, cutlass::arch::Sm70, 8, true>::kStages == 2, ...);
```

## Shared memory, and the `cuFuncSetAttribute` gap handed over from #1541

#1541 found that `gemms/gather_gemm.cu` computes `smem_bytes` and never opts into a dynamic shared-memory maximum above 48 KB, the same class of defect #1559 fixed in `qmm_naive`, and handed it to this issue on the grounds that #1544 owns the grouped-GEMM path.

Two findings, and the conclusion is to record rather than fix.

**The gap is in the shared encoder, not in one launch site.** `CommandEncoder::add_kernel_node_ex` forwards `smem_bytes` to `add_kernel_node_raw`, which sets `kernel_params.sharedMemBytes` and calls `cudaGraphAddKernelNode` (or `cudaLaunchKernelExC` off the graph path). Neither overload calls `cudaFuncSetAttribute` anywhere, so every launch site that wants more than 48 KB has to opt in itself. `GemmGroupedEncoder::encode` does not.

**The reachable Volta configuration does not come close to the ceiling.** Measured from the run A profile rather than computed:

| Kernel | Static smem | Dynamic smem | Registers | Threads | Instances |
|---|---|---|---|---|---|
| `cutlass::Kernel<GemmGrouped>` | 0 | **10,320 B** | 144 | 256 | 180 |
| `prepare_grouped_mm_data` | 0 | 512 B | 32 | 1024 | 180 |
| `qmm_naive_kernel` | 0 | 16,384 B | 224 | 128 | 232 |

10,320 bytes is 21% of the 49,152-byte non-opt-in ceiling, so the grouped GEMM path has no hole to fix on sm_70. `gather_gemm.cu` remains unreachable in an mlxcel build, which reproduces here: `nm -C` on the built `libmlx.a` finds zero undefined references to `mlx::core::gather_mm(bool, bool, ...)`, while all three of `cutlass_gather_mm`, `cutlass_grouped_gemm_unaligned` and `cutlass_segmented_mm` do carry one, from `matmul.cpp.o`.

What is **not** settled: the sm_80 tensor-core configurations in this same file are much larger tiles (`256 x 128 x 32`, and 3 stages in the tf32 arm) and are the plausible place for this path to trip the ceiling. No Ampere-or-later part exists on this host, so that is an argument and not a measurement, and it is left as a follow-up rather than fixed blind. It belongs to epic #1536's `## GB10 (sm_121) continuation`.

## MoE decode and prefill, before and after

Measured on this host from this worktree, five repetitions each, warm PTX cache, `nvidia-smi --query-compute-apps` asserted empty before every run. Decode is a slope over `-n 40` to `-n 120` at a fixed 46-token prompt, per rule 1; prefill is the reported `Prefill:` line at `-n 1` on the 285-token prompt, which is the arm that crosses the grouped-GEMM gate.

The before column is re-measured here rather than quoted from the baseline, because #1539 moved MoE decode after that record was written.

Decode, `-n 40` to `-n 120` at a 46-token prompt, five repetitions, mean of the per-repetition slopes:

| Checkpoint | Arm | Per-rep ms/token | Mean | tok/s | Spread |
|---|---|---|---|---|---|
| `gemma-4-26b-a4b-it-4bit` | before | 30.48 / 31.24 / 29.67 / 28.15 / 28.73 | **29.66** | **33.72** | 10.42% |
|  | after | 29.28 / 30.34 / 30.02 / 28.87 / 30.16 | **29.73** | **33.63** | 4.96% |
| `gemma-4-26b-a4b-it-8bit` | before | 29.92 / 32.04 / 30.79 / 32.24 / 33.44 | **31.68** | **31.56** | 11.11% |
|  | after | 32.10 / 33.67 / 32.46 / 31.80 / 33.53 | **32.71** | **30.57** | 5.71% |

Prefill at `-n 1` on the 285-token prompt, which is the arm above the `B >= 8 * E` gate and therefore the one that runs the grouped GEMM, five repetitions:

| Checkpoint | Arm | Per-rep ms | Mean | tok/s | Spread |
|---|---|---|---|---|---|
| `gemma-4-26b-a4b-it-4bit` | before | 12736 / 12769 / 12715 / 12703 / 12674 | **12720** | **22.41** | 0.75% |
|  | after | 12684 / 12701 / 12494 / 12703 / 12640 | **12644** | **22.54** | 1.66% |
| `gemma-4-26b-a4b-it-8bit` | before | 13527 / 15680 / 15670 / 15603 / 15580 | **15212** | **18.74** | 14.15% |
|  | after | 15679 / 15497 / 15660 / 15571 / 15624 | **15606** | **18.26** | 1.17% |

**Every delta is smaller than the before arm's own repetition spread**, which is the only outcome byte-identical device code permits: 4-bit decode +0.26% against a 10.42% spread, 4-bit prefill -0.59% against 0.75%, 8-bit decode +3.24% against 11.11%. The MoE pair's decode noise is the same noise the baseline record measured at 7.64% and 13.48% and gave five repetitions for; it has not improved, and no conclusion here rests on a difference smaller than it.

One disclosure on the 8-bit prefill row, per rule 3. Its before arm's first repetition reads 13527 ms against 15633 ms for repetitions 2 to 5, which is the 27 GB checkpoint's first prefill in that process after its own JIT modules compiled; the same run reported 35946 ms of prefill inside its first decode measurement against 12181 ms afterwards. Repetition 1 is left in the table rather than dropped, which is why that one before cell carries a 14.15% spread against 1.17% after. Comparing repetitions 2 to 5 on both sides instead gives 15633 ms before against 15588 ms after, a -0.29% delta.

The before column also does not match the row this issue reserved in the baseline record, which reads 38.61 and 34.29 ms/token. That record predates #1539, which moved MoE decode; the before column here is re-measured on this host from this worktree rather than quoted, and 29.66 against 31.68 ms/token is where #1539 left the pair. It also inverts the baseline's 4-bit against 8-bit finding for the MoE pair, in the same direction #1539's own record reports for the dense pair.

Every run reached its token budget; the harness asserts `generated_tokens == n` before a slope is computed, per rule 2, and `nvidia-smi --query-compute-apps` was asserted empty before each run.

## What shipped, and what did not

**Shipped: the correct tag.** The pre-Ampere arm selects `cutlass::arch::Sm70`. On the evidence above this changes no device code on any architecture, which is the honest description of it: a correctness fix to a description, not a performance change. It matters because the description is load-bearing the moment the configuration under it gains a tensor-core operator, and #1543 is doing exactly that work next door.

**Shipped: the decision as a testable pure function.** `gemms/grouped_gemm_arch.h` plus the C shim in `cpp/grouped_gemm_arch_probe.cpp`, enumerated by `grouped_gemm_arch_tests.rs` over every compute capability from 0 to 32 with no GPU involved. This is what closes the "zero change on sm_80+" criterion locally instead of deferring it.

**Shipped: the numeric gate that did not exist.** The grouped GEMM now has tests that look at its values, on the model's real expert dims, across both entry points, both alignment arms and three dtypes.

**Shipped: two `static_assert`s.** One pins the pre-Ampere configuration as SIMT, which is the precondition for one arm covering Volta and Turing. One pins its stage count at 2, which is the `cp.async` question the issue raised.

**Shipped: the placeholder removed.** `get_grouped_mm_funcion` starts from `nullptr` behind a named function-pointer type and throws if selection somehow falls through, instead of starting from a `Sm75` instantiation that read like a pre-Ampere default.

**Did not ship: a Turing arm.** Measured and refused; see the control above.

**Did not ship: a shared-memory opt-in on this path.** Measured at 10,320 bytes against a 49,152-byte ceiling, so there is nothing to opt into on the reachable configuration.

**Did not ship: anything about the grouped GEMM's speed.** Out of scope by the issue's own wording, and the retag cannot move it.

## Deferred to GB10 (sm_121)

No Ampere-or-later device exists on this host. Per epic #1536's `## GB10 (sm_121) continuation`:

- **GB10 MoE output byte-identical**: not run here. What is recovered locally is stronger than a compile check and weaker than a run: the tag mapping is provably unchanged for every compute capability major version at or above 8 (`only_the_pre_ampere_arm_changed` enumerates it), and the device code that mapping selects is byte-identical before and after at `compute_80` and `compute_121`, per symbol, over 35.1 MB and 51.0 MB of SASS. That leaves no mechanism by which GB10 output could move, but output identity is a measurement and this is an argument.
- **GB10 MoE throughput unmoved**: same, and for the same reason. Zero device-code delta at `compute_121` is the whole of the case.
- **The sm_80 tensor-core configurations against the 48 KB dynamic shared-memory ceiling**: genuinely open, and new. See the shared-memory section; it needs an Ampere-or-later part to answer and should become its own issue rather than riding on this one.

The `cuobjdump --dump-sass` technique #1539 used **does** transfer to this file, unlike #1541's case, and the reason is worth recording because the two look alike from the outside: `grouped_gemm_unaligned.cu` compiles its CUTLASS kernels ahead of time, so its object holds 51 device functions and its SASS dump is a real artifact at every target, whereas `qmm_naive.cu` is host dispatch only and dumps nothing. Which case a file is in has to be checked before the diff means anything.

## Reproduce

```bash
# 1. Build. Single architecture, explicit.
MLX_CUDA_ARCHITECTURES=70 cargo build --release --features cuda
cuobjdump --list-elf target/release/build/mlxcel-core-*/out/build/lib/libmlx.a \
  | grep -oE 'sm_[0-9]+a?' | sort | uniq -c

# 2. Reachability. A is the grouped path, C and D are the controls. The gate is
#    prompt_tokens * top_k >= 8 * num_experts, which is 128 prompt tokens for
#    this checkpoint. --cuda-graph-trace=node is mandatory.
M=./models/mlx-community/gemma-4-26b-a4b-it-4bit
LONG=$(python3 -c "print(('Virtual memory lets an operating system give each process its own address space. '*40).strip())")
SHORT="Write a detailed technical explanation of how virtual memory works in a modern operating system. Cover page tables, the TLB, page faults, and swapping. Be thorough."
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o A ./target/release/mlxcel generate -m $M -p "$LONG"  -n 4
MLXCEL_GATHER_QMM_GROUPED=0 \
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o C ./target/release/mlxcel generate -m $M -p "$LONG"  -n 4
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o D ./target/release/mlxcel generate -m $M -p "$SHORT" -n 4
for r in A C D; do nsys stats --report cuda_gpu_kern_sum --format csv $r.nsys-rep | grep -c GemmGrouped; done

# 3. Per-kernel shared memory, registers and block size, from the same profile.
python3 - <<'PY'
import sqlite3
c = sqlite3.connect("A.sqlite").cursor()
for row in c.execute("""SELECT s.value, COUNT(*), MAX(k.staticSharedMemory),
                               MAX(k.dynamicSharedMemory), MAX(k.registersPerThread)
                        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
                        WHERE s.value LIKE '%GemmGrouped%' GROUP BY s.value"""):
    print(row)
PY

# 4. Correctness against the reference implementation. Greedy, same prompt,
#    grouped path against the legacy qmm path. Expect byte-identical output.
#    Any substantive prompt of at least 128 tokens will do; this one comes out
#    at 285 with the chat template applied.
AB=$(python3 -c "print(('Explain in detail how a modern operating system implements virtual memory. Describe the structure of multi-level page tables, how the translation lookaside buffer caches translations and what happens on a TLB miss, how the kernel handles a page fault for a page that has been swapped to disk, how copy-on-write is used when a process forks, how the page replacement policy decides which frame to evict, and how huge pages change the tradeoffs. Then compare the design choices Linux and Windows make in each of these areas, and explain why a database engine might bypass the page cache entirely with direct I/O. Be precise and technical throughout, and give concrete numbers where they matter. '*2).strip())")
./target/release/mlxcel generate -m $M -p "$AB" -n 64 -t 0.0 --profile > grouped.txt
MLXCEL_GATHER_QMM_GROUPED=0 ./target/release/mlxcel generate -m $M -p "$AB" -n 64 -t 0.0 --profile > legacy.txt
diff grouped.txt legacy.txt

# 5. Device-code delta of the retag, per symbol, at three targets. Take the exact
#    nvcc line out of the build's compile_commands.json so the flags are the
#    shipped ones, swap --generate-code for the target under test, compile the
#    file before and after the change, and compare cuobjdump --dump-sass split
#    by "Function :". Whole-file diff is not the right comparison: the cubin
#    emits the same functions in a different order.

# 6. The host-side enumeration and the numeric tests. No GPU needed for the
#    first, a GPU needed for the second.
cargo test --lib grouped_gemm_arch_tests::
cargo test --release --features cuda --lib grouped_gemm_arch_tests::gather_mm

# 7. Decode slope and prefill, five repetitions, warm cache, per rules 1 and 2.
P="Write a detailed technical explanation of how virtual memory works in a modern operating system. Cover page tables, the TLB, page faults, and swapping. Be thorough."
for rep in 1 2 3 4 5; do for n in 40 120; do
  ./target/release/mlxcel generate -m $M -p "$P" -n $n --profile
done; done
```

## References

- Epic #1536, the Volta decode program. Its `## GB10 (sm_121) continuation` holds what could not be verified here.
- Baseline this is a delta against: `volta-sm70-baseline-2026-08-31.md`, and its six methodology rules.
- The sorted-MoE prefill fast path that makes this branch reachable on a quantized checkpoint: #629, `moe-prefill-grouped-gemm-gb10-2026-07-10.md`, and `patches/mlx/backend/cuda/quantized/quantized.cpp:359`.
- The `cuFuncSetAttribute` gap this issue inherited: #1541, `qmm-naive-tile-v100-2026-08-31.md`, section "`gather_gemm.cu`: checked, and nothing to fix", and the fix #1559 made in `qmm_naive`.
- The SASS-diff technique: #1539, and #1541 for the case where it does not transfer.
- Dispatch and the arch decision: `patches/mlx/backend/cuda/gemms/grouped_gemm_unaligned.cu` and `gemms/grouped_gemm_arch.h`.
