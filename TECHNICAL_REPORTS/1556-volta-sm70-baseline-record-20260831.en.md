# Technical Report: PR #1556 - Volta sm_70 baseline record and CI gate

**Date**: 2026-08-31

**Author**: mlxcel maintainers

**Status**: Completed. The sm_70 compile gate has passed on this PR but has never run on a GB10 host; GB10 verification is deferred.

---

## Executive Summary

PR #1556 (issue #1538) turns a one-off Volta audit into a reproducible committed record, and closes the hole that made Volta support accidental in the first place: before this change nothing in the repository compiled MLX for anything below sm_80, so an architecture-conditional compile break could sit on `main` indefinitely.

The measurement work matters more than it looks. Six methodology rules are stated in the document, and three of them were derived from wrong results produced during this work rather than from principle. A record that omits any one of them is not reproducible, and in two cases the error changes the answer by more than the effects epic #1536 exists to measure.

The 4-bit against 8-bit inversion is confirmed as a deliverable of this issue: on Volta an 8-bit checkpoint decodes **1.886x faster** than the config-identical 4-bit one while reading nearly twice the bytes, and the mechanism is isolated to `qmv` at 97.6% attribution.

## 1. Problem Statement

Epic #1536's diagnosis rested on a single ad-hoc session on one V100. Every later sub-issue states its acceptance criteria as a delta against that baseline, so an unreproducible baseline makes every subsequent "we improved X by N%" unfalsifiable.

Separately, there was no build coverage below sm_80 anywhere. `ci.yml` pinned `MLX_CUDA_ARCHITECTURES: "121"` in both CUDA jobs; `release.yml` ships `90a;100;121` on aarch64 and `80;86;89;90a;100;120` on x86_64; `build.rs` auto-detects the host with a `90a` last resort. Volta built only because someone happened to build on one.

## 2. Change Summary

4 files, +547 / -6.

- `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md` (new, ~400 lines): environment, build provenance, the six methodology rules, decode and prefill throughput, the cold PTX cache measurement, kernel profiles, the 4-bit/8-bit analysis, a harness comparison, a full reproduce-vs-audit table, the CI decision, known traps, and an empty post-program comparison table for #1539 through #1545 to fill.
- `benchmarks/cuda_v100_2026-08-31.csv` (new): the same CSV shape the GB10 sweeps in `benchmarks/` use, so the two are directly comparable.
- `.github/workflows/ci.yml`: adds the `cuda-sm70-compile` job and a `cuda_arch` path filter.
- `scripts/bench_decode.sh`: slope-aware changes.

## 3. Technical Decisions

### 3.1 Six methodology rules, three of them earned the hard way

Rules 1, 4 and 6 were already known going in. Rules 2, 3 and 5 came out of actual wrong results:

- **Rule 2, both runs must reach the token budget.** A `"Hi."` prompt makes every instruct checkpoint here emit EOS after about ten tokens, so `-n 40` and `-n 120` return the same generation and the slope becomes a difference of two nearly equal numbers over nearly zero. The first attempt produced slopes ranging from a division-by-zero to 193 ms/token on a model whose real figure is 220. The harness now asserts `generated_tokens == n` before computing a slope.
- **Rule 3, state the PTX cache state.** Discovered en route: mlxcel keeps a persistent PTX cache keyed on the pinned MLX commit, and a first run after a fresh machine or a pin bump costs **4.8x** the wall clock of the second, with prefill at 6.8x. A first-token cost quoted without its cache state can be wrong by a factor of six.
- **Rule 5, nsys absolutes are trustworthy only when they reconcile against an unprofiled wall clock.** Graph-node instrumentation is not applied evenly across kernel types. The dense pair reconciles at 102.4% and 101.6% and carries the kernel-level conclusion; the MoE 4-bit arm inflates to 144.0% and is reported as a wall-clock observation only.

Encoding these as numbered rules with the failure that produced each one is the durable part of this PR. The numbers will be superseded by #1539 onward; the rules will not.

### 3.2 A compile-only sm_70 gate, and an explicit refusal of the alternatives

`cuda-sm70-compile` runs `cargo check --features cuda --all-targets` at `MLX_CUDA_ARCHITECTURES: "70"` on the existing self-hosted CUDA runner, then asserts with `cuobjdump --list-elf` that the resulting `libmlx.a` holds sm_70 and nothing else.

Two details carry the design. `cargo check` still runs the build script, so nvcc compiles MLX even though nothing links, which buys the arch-conditional compile coverage without a link step. And the cubin assertion is what turns "the environment variable was set" into "nvcc actually emitted pre-Ampere code". Without it the job would pass on a misconfiguration that silently built for the wrong target, which is precisely the failure class the job exists to catch.

The job takes its own persistent target directory, because the architecture list is part of the build-script fingerprint and sharing with an sm_121 job would make the two invalidate each other on every run.

**A GPU-backed Volta job was declined, and so was adding sm_70 to the release matrix.** Both refusals are documented in the issue, the PR, the document and the job comment rather than left as silent omissions. The reasoning is that the realistic failure mode is an arch-conditional compile break, which a compile-only gate catches, while a GPU job would cost a runner and a full link for a failure mode nobody has observed.

### 3.3 Three harnesses, and naming which number each produces

The document compares `make bench-model`, the slope method, and nsys, and states what each is good for. This matters because the repository's own harness reports single-run `tokens / wall_time`, which on this host understates decode by up to 2.5x and, on the MoE pair, **reports the opposite ordering to reality**. Recording that as an argument against the harness rather than as evidence about the models is what keeps the CSV artifact useful without letting it mislead.

## 4. Validation

Host: Tesla V100-PCIE-32GB (sm_70), CUDA 12.9.41, driver 575.51.03, 16 cores. Build provenance re-verified: 96 cubins, all sm_70.

Against the audit, with a 10% reproduction band:

- **Exact matches** on every instance count that should be deterministic: `qmv` at 11,431 (qwen), 39,151 (dense, both arms), 28,084 (MoE, both arms); `cudaMemcpyAsync` at 1,847 calls; dense `qmv` times 12.8366 s / 5.9911 s against 12.84 / 5.99; MoE `qmv` 2.6224 / 1.4921 against 2.62 / 1.49.
- **Reproduced within band**: qwen decode 220.33 ms/tok against 239; qwen prefill 8.00 tok/s against ~7.7; roofline attainment 7.27% against 7.7%; dense 4-bit 124.41 against 122.04; dense 8-bit 65.96 against 65.46; dense advantage 1.886x against 1.86x; dense attribution 97.6% against ~101%.
- **Did not reproduce**: the MoE 8-bit advantage at 12.6% against 19.9% (direction solid over five repetitions per arm, magnitude not; that arm's spread is 13.5%, so MoE numbers are ordinal only), and `qmm_naive` instance counts at exactly half the audit's on both models, which is a counting difference rather than a performance one.

### 4.1 TTFT, reconciled after the fact

The record initially logged TTFT as unreproducible: ~13 s in the audit against 24.94 s here. The two figures turned out to be measuring different things. Both are intercepts of a two-point slope fit, but each intercept carries the prefill cost of its own prompt, and the audit fitted at a 3-token prompt while this record fits at 46 tokens. Subtracting each one's prefill at the measured 8.00 tok/s marginal rate leaves 12.60 s against 12.66 s, agreeing to 0.5%.

This is an arithmetic reconciliation, not a controlled experiment, and the document says so and names the cheap confirming test that has not been run. What it settles is that neither figure was wrong; quoting either without its prompt length was.

## 5. Validation Limits and Follow-up

- **The `cuda-sm70-compile` job has passed on this PR but has never run on a GB10 host.** GitHub Actions cannot be exercised from this machine beyond the PR's own run.
- The methodology section has not been checked against a GB10 sweep, and the `scripts/bench_decode.sh` change is unverified on the GB10 sweep path, though it is unaffected by construction.
- Whether the graph-construction cost in TTFT is Volta-specific or general remains open; it needs a GB10 comparison and is #1545's question.
- MoE kernel-level attribution is unavailable on this host: both this record's and the audit's MoE profiles fail the reconciliation check, so only the wall-clock ordering is reportable there.

These belong to the `## GB10 (sm_121) continuation` section of epic #1536.

## References

- Issue #1538, epic #1536 (including its GB10 continuation section)
- The record: `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md`
- CSV artifact: `benchmarks/cuda_v100_2026-08-31.csv`
- Accumulator selection the 4-bit/8-bit analysis isolates: `src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/qmm/qmv.cu`
- The scalar-FMA control that moves the other way: `mlx/backend/cuda/device/gemm_sm70.cuh`
