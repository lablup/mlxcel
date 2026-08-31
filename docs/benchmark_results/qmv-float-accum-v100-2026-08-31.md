# `qmv` float accumulators below Ampere: V100 before and after

This is the measurement record for issue #1539, Phase 1 of epic #1536. It re-runs the reproduce commands of `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md` on the same host, at the same warm-cache state, against a build with and without the change, and it follows that document's six methodology rules without exception. Read that record first: everything here is a delta against it, and every rule it states applies here unchanged.

The change itself is one accumulator type. `qmv` selected float at `bits >= 8` and the element type below that, so a bf16 checkpoint at 4 bits accumulated in bf16, and no pre-Ampere part has a bf16 ALU to run that on. Below Ampere the kernel now accumulates bf16 in float, behind a `__CUDA_ARCH__ < 800` guard in device code. Ampere and later are untouched, and the SASS section below shows that literally rather than by argument.

## Environment

Identical to the baseline record: Tesla V100-PCIE-32GB (sm_70, 32 GB HBM2, 900 GB/s), driver 575.51.03, CUDA 12.9.41, gcc 13.3.0, Nsight Systems 2025.3.1.90, single GPU with nothing else resident. `MLX_CUDA_ARCHITECTURES=70 make release-cuda` for both arms; `cuobjdump --list-elf` reports 96 cubins and every one of them sm_70 in both. Warm PTX cache throughout (`~/.cache/mlxcel/cuda-ptx/9a795735ad9a`), which matters because the cache is keyed on the MLX pin and not on this patch, so neither arm paid a cold-cache cost.

CUDA 12.x is a hard requirement rather than a preference: CUDA 13 removed Volta support and its nvcc rejects `compute_70` outright.

The "before" arm is `origin/main` at `07ea6b1f`. Its decode slopes reproduce the baseline record to within 1.2% on every checkpoint (`qwen3.8-27B-4bit` 219.96 against 220.33, dense 4-bit 124.26 against 124.41, dense 8-bit 66.75 against 65.96), and its `qmv` kernel times reproduce it to four significant figures at identical instance counts, so the two records are measuring the same machine in the same state.

## Decode throughput

Slope over `-n 40` to `-n 120` at a fixed 46-token prompt, three repetitions each (five for the MoE arm), warm cache. Every run is asserted to have generated exactly `-n` tokens before it enters a slope.

| Checkpoint | Before | After | Speedup |
|---|---|---|---|
| `qwen3.8-27B-4bit` | 219.96 ms/tok (4.55 tok/s) | **117.83 ms/tok (8.49 tok/s)** | **1.87x** |
| `gemma-4-12B-it-4bit` | 124.26 ms/tok (8.05 tok/s) | **71.66 ms/tok (13.95 tok/s)** | **1.73x** |
| `gemma-4-12b-it-8bit` (control) | 66.75 ms/tok (14.98 tok/s) | 66.73 ms/tok (14.99 tok/s) | 1.00x |
| `gemma-4-26b-a4b-it-4bit` (MoE) | 35.45 ms/tok (28.21 tok/s) | 29.56 ms/tok (33.83 tok/s) | 1.20x |

Per-repetition spreads: 0.14% and 0.23% on `qwen3.8-27B-4bit`, 2.03% and 1.83% on the dense 4-bit arm, 0.44% and 0.75% on the 8-bit arm. The MoE arm spans 24.31% before and 8.73% after, so its 1.20x is ordinal and not a precise ratio, exactly as the baseline record warns.

**The 8-bit arm is the control, and it does not move.** 66.75 to 66.73 ms/token is 0.03%, well inside its own 0.44% repeat spread. That is the expected result and it is the reason to trust the rest: at `bits >= 8` the kernel already accumulated in float, so the guard changes nothing there, and any drift in that row would have meant something else changed with it.

## Kernel attribution

`nsys profile -t cuda,nvtx --cuda-graph-trace=node`, with each profiled run reconciled against the same run unprofiled. Instance counts are identical between the arms of each pair, which is what makes the absolute times comparable.

| Run | Instances | `qmv` before | `qmv` after | Ratio |
|---|---|---|---|---|
| `qwen3.8-27B-4bit`, `-n 24`, 61-token prompt | 11,431 | 5.2165 s | **2.5984 s** | **2.01x** |
| `gemma-4-12B-it-4bit`, `-n 120`, 46-token prompt | 39,151 | 12.8369 s | **6.5654 s** | **1.96x** |

Reconciliation: 101.3% before and 101.9% after on the qwen run, 100.7% and 103.3% on the dense run. All four are close enough to unprofiled wall clock for the absolute times to carry the conclusion, per rule 5.

The attribution closes:

- Dense 4-bit: the `qmv` delta is (12.8369 - 6.5654) / 120 = **52.26 ms/token** against a measured slope delta of 124.26 - 71.66 = **52.60 ms/token**. `qmv` accounts for **99.4%** of the improvement.
- `qwen3.8-27B-4bit`: (5.2165 - 2.5984) / 24 = **109.09 ms/token** against a slope delta of **102.13 ms/token**, that is 106.8%. The overshoot is the profiling overhead difference between the two arms (101.3% against 101.9%), and it says the same thing: nothing outside `qmv` is needed to explain the result.

`qmm_naive` is the second control. It runs prefill, it accumulates in float at every bit width, and the guard cannot reach it: 14.5756 to 14.5580 s on the qwen run (-0.12%) and 10.8808 to 10.9412 s on the dense run (+0.55%), both inside run-to-run noise at identical 497 and 329 instance counts.

## Roofline attainment

Achieved bandwidth is the checkpoint's per-step weight traffic divided by `qmv`'s own time per generated token, which is the quantity the issue's acceptance criterion names.

| Checkpoint | Bytes/step | `qmv` s/token before | after | Achieved before | after |
|---|---|---|---|---|---|
| `qwen3.8-27B-4bit` | 14.42 GB | 0.2174 | **0.1083** | 66.3 GB/s (7.37%) | **133.2 GB/s (14.80%)** |
| `gemma-4-12B-it-4bit` | 6.17 GB | 0.1070 | **0.0547** | 57.7 GB/s (6.41%) | **112.8 GB/s (12.53%)** |

**The issue's acceptance criterion of 25% of the 900 GB/s roofline is not met.** `qmv` on `qwen3.8-27B-4bit` reaches 14.80%, slightly more than double the 7.37% it started from. Reaching 25% would require 64.1 ms/token of `qmv` time against the 108.3 ms/token it now takes, a further 1.69x that this change does not deliver and was never going to.

That criterion was written before the controlled measurement existed. The measurement that quantified this issue put the ceiling at 2.14x, taken from an 8-bit sibling checkpoint whose `qmv` accumulates in float, and it said in terms that the figure was an upper bound because the two arms also differ in dequantization work. The delivered 2.01x is 94% of that ceiling. The 25% roofline figure and the 2.14x ceiling are not consistent with each other, and the ceiling is the one that was measured: the 8-bit arm itself only reaches 26.7% of roofline while reading 1.9x the bytes, so no accumulator change alone could put a 4-bit arm reading half the bytes at the same fraction.

What is left is stated rather than hidden. `qmv` at 14.8% of roofline is still not bandwidth bound, and the remaining headroom belongs to the later items in #1536.

## The 4-bit against 8-bit inversion, after

The baseline record's central finding was that an 8-bit checkpoint decoded 1.886x faster than its 4-bit sibling on this part while reading 1.9x the bytes. That inversion is now almost closed:

| Pair | Before | After |
|---|---|---|
| Dense `gemma-4-12B-it-{4bit,8bit}` | 8-bit 1.886x faster | 8-bit 1.074x faster |

The 4-bit arm went from 88.6% behind to 7.4% behind while continuing to read 6.17 GB against 11.65 GB per step. The residual 7.4% is what is left once the accumulator is equalized, and by elimination it is the dequantization work that the 4-bit arm does and the 8-bit arm does not. That is consistent with the baseline record's `qmm_naive` control, which measured the dequantization component alone at 5.9% pointing the other way, and it is small enough that the practical guidance changes: on 32 GB of HBM2 a 4-bit checkpoint now costs 7% of decode rate and saves 47% of the weight footprint, where before it cost 89%.

## Register pressure and occupancy

Float accumulators are twice the accumulator register footprint of bf16 ones, so this was checked rather than assumed. Resource usage comes from `cuobjdump -res-usage` on the `qmv.cu` object compiled at `sm_70` with the exact flags cmake uses, before and after.

| Kernel, `<8, 16, 64>` bf16 / uint4b | REG before | REG after | Blocks/SM | Warps/SM |
|---|---|---|---|---|
| `qmv_kernel`, `has_residue_k = false` | 55 | 61 | 4 -> 4 | 32 -> 32 |
| `qmv_kernel`, `has_residue_k = true` | 59 | 66 | 4 -> 3 | 32 -> 24 |
| `gather_qmv_kernel`, `has_residue_k = false` | 54 | 62 | 4 -> 4 | 32 -> 32 |
| `gather_qmv_kernel`, `has_residue_k = true` | 59 | 66 | 4 -> 3 | 32 -> 24 |
| `qmv_multirow_kernel`, `max_x_rows = 8` | 110 to 112 | 168 | 2 -> 1 | 16 -> 8 |

Blocks per SM are computed for the launch geometry the dispatcher uses, 8 warps of 32 threads per block, against sm_70's 65,536-register file at its 8-register-per-thread allocation granularity.

**Nothing spills.** `STACK` and `LOCAL` are zero for every one of the 378 `qmv`-family instantiations in both arms, which is the outcome that mattered: 168 registers is well inside the 255 limit, so the multirow kernel pays occupancy and not local-memory traffic.

The residue-k variant of the single-row kernel does lose a block per SM, and that variant is the one this workload runs: `gemma-4-12B-it-4bit` has k = 3840, and 3840 % 512 is 256, so `has_residue_k` is true. The measured 1.96x on that model is therefore a number that already includes the occupancy loss. `elems_per_thread` was not re-tuned, because the issue conditions that on occupancy dropping "enough to eat the win" and 24 warps per SM did not come close to eating it.

The multirow kernel halves its occupancy, from 16 warps per SM to 8. It is not exercised by any measurement here: it fires only for batched decode with 2 to 8 input rows (`MLXCEL_QMV_MULTIROW`, `MLXCEL_QMV_MULTIROW_MAX_ROWS`, #906), and single-stream `mlxcel generate` never reaches it. It is also the kernel that amortizes one weight-tile load across every row, so it is the least occupancy-sensitive member of the family. Batched-decode throughput on sm_70 is unmeasured here and is the one place this change carries an unquantified risk; the row window is already autotuner-tunable if it turns out to matter.

Leaving the multirow accumulator at bf16 while promoting the single-row one was not an option: #725 guarantees that a multirow launch is bit-identical to the per-row launches it replaces, and `qmv_multirow_matches_per_row_qmv_bitwise` enforces it.

## Zero diff on sm_80 and later

The guard is meant to leave Ampere and later untouched. That is checkable on a host with no such device, because nvcc does not need the target part to emit code for it. Compiling `qmv.cu` from the unpatched and patched sources at one architecture with the production flag set, then diffing the disassembly:

| Architecture | `cuobjdump --dump-sass` | `cuobjdump -res-usage` |
|---|---|---|
| sm_80 | **identical**, 389,998 lines | identical |
| sm_121 | **identical**, 437,519 lines | identical |
| sm_70 | differs, as intended | differs, as tabulated above |

Byte-for-byte identical, not merely equivalent. The sm_80 and sm_121 device passes take the `#else` arm, which is upstream's rule spelled the same way, and the accumulator appears in neither the kernel template signature nor its mangled name, so the host pass has nothing to disagree with either.

What this does not cover is runtime output on a real sm_80+ part. Byte-identical greedy generation on GB10 and unmoved GB10 throughput remain unverified here and are tracked in the `## GB10 (sm_121) continuation` section of #1536. Identical SASS is strong evidence for both, since a device executing the same instructions on the same inputs produces the same outputs, but it is evidence and not the measurement.

## Numerics

Float accumulation is strictly more accurate than the emulated bf16 accumulation it replaces, so output quality improves on pre-Ampere parts. It also changes them: **greedy decode on a Volta host is not token-identical to a build from before this change.** No published Volta baseline exists for it to regress against, which is why this is acceptable, but it is a visible behavior change and it is recorded in the changelog as one.

Determinism run to run is unaffected. Three repetitions of a greedy 32-token generation (`--temp 0 --seed 0`) on `gemma-4-12B-it-4bit` produce one distinct output on the before build and one on the after build, and the two builds produce the same tokens as each other on that prompt. `qmv_multirow_matches_per_row_qmv_bitwise` (#725) passes, so multirow and per-row launches still agree bitwise across bits 4 and 8, group sizes 32, 64 and 128, and bf16, f16 and f32. A new `qmv_matches_qmm_across_bits_and_group_sizes` compares `qmv` at `M = 1` against `qmm` at `M = 8` on identical operands across bits 4 and 8 and group sizes 32, 64 and 128, which is the parity check the issue asked for.

### A pre-existing prefill nondeterminism on sm_70, found while checking this

`tests/cuda_qmm_determinism.rs` **fails on this host, before and after the change alike**, and the failure is not this issue's. It repeats a 64-token prefill plus eight decode steps ten times with fresh caches and requires byte-identical logits at every step. Pointed at `gemma-4-12B-it-4bit` it diverges at step 0, the prefill, on both builds. Prefill is `M = 64`, so it dispatches to `qmm_naive` and never reaches `qmv`; the guard cannot touch it. Two further facts pin that down: the divergence is between iterations of one process, not between the two builds, and iteration 0's prefill logits hash to the same value (5078372879123904733) on the patched and unpatched builds, so prefill output is bit-identical across the change.

The test has almost certainly never run against this checkpoint before. Its default model directory is `models/llama-3.2-1b-4bit`, which is not present on this host or in CI, and the test returns early rather than failing when the directory is missing, so every CI run of it so far has been a skip.

This is worth a follow-up in its own right: it means quantized prefill on sm_70 is not bitwise reproducible for this checkpoint, which is the same class of defect #910 fixed for `qmm_sm80` on sm_121. It is out of scope for #1539, which changes only the decode-side accumulator, and it is recorded here so it is not lost.

f16 is deliberately outside this change. sm_70 and sm_75 have a native fp16 FMA at twice the fp32 rate, so promoting f16 accumulators there would spend throughput to buy precision, the opposite of the trade this makes for bf16, and f16 policy below Ampere belongs to #1542.

## Reproduce

```bash
MLX_CUDA_ARCHITECTURES=70 make release-cuda
cuobjdump --list-elf target/release/build/mlxcel-core-*/out/build/lib/libmlx.a \
  | grep -oE 'sm_[0-9]+a?' | sort | uniq -c    # expect: 96 sm_70

# Decode slope. Same prompt at two -n values, three repetitions, and confirm
# "Generated tokens" equals -n in every run before computing a slope.
P="Write a detailed technical explanation of how virtual memory works in a modern operating system. Cover page tables, the TLB, page faults, and swapping. Be thorough."
for rep in 1 2 3; do for n in 40 120; do
  ./target/release/mlxcel generate -m ./models/mlx-community/gemma-4-12B-it-4bit -p "$P" -n $n --profile
done; done
# slope = (decode_ms(120) - decode_ms(40)) / 80

# Kernel attribution. --cuda-graph-trace=node is mandatory, and the profiled
# decode time must be reconciled against the same run unprofiled.
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o qmv_after \
  ./target/release/mlxcel generate -m ./models/mlx-community/qwen3.8-27B-4bit \
  -p "Explain what a GPU tensor core does." -n 24
nsys stats --report cuda_gpu_kern_sum --format csv qmv_after.nsys-rep
# Sum every qmv_kernel row: the kernel is instantiated separately for
# has_residue_k true and false, and both appear.

# Register pressure and the sm_80+ SASS diff, from the cmake compile line for
# qmv.cu with only --generate-code changed.
cuobjdump -res-usage qmv.cu.o
cuobjdump --dump-sass qmv.cu.o

MLXCEL_TEST_QMM_MODEL_DIR=models/mlx-community/gemma-4-12B-it-4bit \
  cargo test --release --features cuda --test cuda_qmm_determinism
cargo test --release --features cuda -p mlxcel-core --lib qmv_
```

## References

- Baseline this is measured against: `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md`, and issue #1538.
- The change: `src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/qmm/qmv.cu`, `qmv_accumulator`.
- Epic #1536, whose `## GB10 (sm_121) continuation` section holds what could not be verified on this machine.
- Compute-capability probe and the recorded architecture list: #1537.
- Multirow qmv and its bit-identity contract: #725, and the row window knob, #906.
