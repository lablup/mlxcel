# Technical Report: PR #2029 - Finish the Kernel Port Standardization

**Date**: 2026-09-29

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: C++ (kernel dispatch), Rust (gates and bridge), Python (CI checker)

**Risk Level**: Low

## Executive Summary

PR #2026 introduced `select_kernel_port` and a per-kernel `KernelPorts` table, and converted the nine two-way launchers. This PR closes the three gaps that pass left open and corrects one claim its report recorded. The shared background, the helper's design and the `static_assert` rationale are in the `2026-*` pair, which this PR also updates; this report covers only what is new here.

Two new checker rules, one of which found five launchers that every earlier survey had missed, and one of which governs a defect on the Rust side that the C++ rules could not see. Plus a correction: the two SDPA launchers were never at risk, and saying they were came from grepping the wrong predicate.

## 1. Problem Statement

Three problems, related by being the same defect at different layers.

**A single-port launcher was invisible to the checker.** Rules 1 and 2 describe how a *multi-port* launcher goes wrong: it compares the backend kind, or it hand-rolls the refusal. A Metal-only launcher does neither. It goes wrong by calling `get_x_kernel().get()` with no resolution at all, and the check passed on a tree where one had been reverted to exactly that shape by hand.

**A Rust gate named its backends instead of its question.** Three test helpers and one example read `metal_is_available() || cuda_is_available()`. The condition was not wrong, and that is precisely the difficulty: it names the two backends that happen to have ports, so on a third backend it reads as a missing term rather than as what it means. The obvious repair, adding `rocm_is_available()`, is wrong twice over. That function does not exist, and if it did, `fused_norm_ports().rocm` and `fused_rope_ports().rocm` are still null, so widening the gate would walk the caller past the port table into the launcher's refusal, which is the abort #2026 had just converted into a catchable error.

**A `Result<()>` conversion is not caught by the compiler.** Converting a launcher to `Result` splits in two. A launcher returning a value becomes `Result<T>` and every call site stops compiling. A void launcher becomes `Result<()>`, and an ignored `Result<()>` is only the `unused_must_use` lint, so `cargo check` stays green and the refusal is swallowed at runtime.

## 2. Change Summary

- `scripts/ci/check_kernel_port_dispatch.py`: rule 3 (no launcher reaches a kernel holder directly; port-table bodies are blanked with byte offsets preserved so line numbers hold) and rule 4 (no Rust gate enumerates Metal and CUDA). The C++ exemption list is now empty.
- Five launchers brought into the port table, found by rule 3: the four SDPA and turbo4 delegated entry points plus `sparse_v`. Their bridge declarations in `lib.rs` became `Result`, and their call sites carry an `expect` naming the gate that checked the port.
- Gates renamed to the question they ask: `fused_add_rms_norm_available()` and `fused_rope_qk_append_available()` in the two parity test modules, `gpu_backend_available()` in `rms_norm_small_axis_tests.rs`, and `gpu_backend_kind()` plus the two per-kernel predicates in `examples/fused_norm_rope_microbench.rs`.
- Three swallowed `Result<()>` values fixed, in `layers.rs:1141` and the two parity test helpers, plus two in the example.
- `kernel_port.h`: a note that a void launcher's conversion surfaces only under `-D warnings` clippy, and that PR-time CI's `-p mlxcel` invocation does not cover it.
- The `2026-*` report pair corrected and extended.

## 3. Technical Decisions

### The gate asks the kernel, not the backend list

"Does this backend have this kernel's port" is now the kernel's own `*_available()`, which returns `has_kernel_port` on the same table the dispatch reads, so the predicate and the dispatch cannot answer differently and a backend that gains a port needs no edit in the test. "Is there a GPU at all" is `gpu_backend_available()`.

Separating them was not cosmetic. `rms_norm_small_axis_tests.rs` exercises MLX's own `fast::rms_norm` dispatch configuration, where the broken precision band is a property of the reduction's `N_READS` tiling rather than of any mlxcel port. The narrow gate had been skipping it on ROCm for no reason. Under the correct predicate both sweeps run on gfx1151 and pass, which is coverage the rename produced rather than restored.

### Rule 4's exemptions name a missing predicate, not a deferred convention

Unlike the C++ exemption list, which was a convention not yet applied and therefore had to shrink to nothing, rule 4's two entries are blocked on something that does not exist yet.

`ffi_tests.rs` gates the fused paged-decode kernel. `paged_attention.cpp` has a `KernelPorts` table but exports no `*_available()` for the bridge to call, so there is nothing narrower to ask. `grouped_gemm_numeric_tests.rs` gates MLX's own `gather_mm`, where `gpu_backend_available()` would be the honest predicate, but whether MLX's ROCm backend implements the grouped-GEMM path is unverified here. Widening a gate on an assumption about a backend is how #1806's abort was reached, so this one waits for an answer rather than taking one. Both are tracked to #1814.

### One real bug class, and two things deliberately not called bugs

The three dropped `Result<()>` values are real swallowed refusals. Two other findings from the same sweep are not, and recording the difference matters more than the count.

`fused_xielu` in `apertus.rs` has no backend term in its gate, which reads like a reachable refusal. It is not: `mlx_cxx_kernels.cpp:463` early-returns an elementwise fallback when no Metal device is present, so the launcher falls back rather than refusing. A gate change made here was reverted. This was the third time in this line of work that a missing Rust-side term was read as "reachable" without first checking for a C++-side fallback.

`nemotron_h.rs` gained a `custom_kernels_available()` term, and that is defensive rather than a fixed bug. The path was traced, not executed, because this host has no Nemotron-H checkpoint. The comment at the call site says so, so a later reader does not inherit the change as evidence.

## 4. Validation

Run on the gfx1151 host (AMD Radeon 8060S, RDNA 3.5, ROCm 10.0.0).

- ROCm gate unchanged from baseline: one failing target, `-p mlxcel-core --lib`, which is the nvfp4 abort of #1806, and that abort is the run's only `terminate called`.
- `clippy --workspace --all-targets --features rocm -- -D warnings`, `cargo fmt --check`, and the crate-version, dtype-key, llama-compat and port-dispatch checks: clean.
- All four checker defenses confirmed by negative control rather than assumed. Reintroducing a two-way dispatch, hand-rolling a refusal, and reaching a holder directly each fail with the file and line. Rule 4 was controlled in three spellings, `a || b`, `!a && !b`, and the reverse order without a path prefix, and caught all three.

### A method note

The newly enabled small-axis sweeps were run directly rather than read off the suite result. The nvfp4 abort terminates the test binary, so nothing alphabetically after `ffi_tests::compiled_qgelu_mlp_global_scale_native_nvfp4_prefill_matches_reference` reaches the runner at all. A suite that ends in an abort reports nothing about the tests it never started, and treating "not in the failure list" as "passed" would have been wrong here.

### Which clippy invocation is the gate

This cost four rounds. `cargo clippy -p mlxcel --lib --tests` is what PR-time CI runs and what the earlier passes used, and it builds neither mlxcel-core nor examples, so it reported clean while five `Result<()>` sites were unlinted. `make verify` and `make verify-rocm` lint `--workspace --all-targets` and found every one. The note now lives in `kernel_port.h` beside the declaration requirement, since that is where someone converting the next launcher will be reading.

## 5. Learning Points

- A condition that is correct today can still be a defect if it encodes *which* answers exist rather than *what* is being asked. `metal_is_available() || cuda_is_available()` was never wrong; it was unreadable in a way that invites a specific wrong repair, and the repair it invites would have reintroduced the abort.
- A checker earns trust only from a negative control. Rule 3 exists because a hand-made regression passed the check, and rule 4 was controlled in three spellings before being believed.
- Two of the five findings in this sweep were not bugs. Counting them as such would have made the record half false, and the useful artifact is the reason each one is not: a C++-side fallback in one case, an untraced path in the other.

## 6. What Is Not Verified

- **`verify-rocm-smoke` did not run on the final tree.** Its fixture lived under `/tmp`, which this host clears on reboot, and no local checkpoint remains. Every other step of `verify-rocm` ran on the final tree. Set `MLXCEL_ROCM_SMOKE_MODEL` to a local checkpoint to close this.
- **Metal and CUDA were not run.** Neither is available on this host.
- **Whether MLX's ROCm backend implements `gather_mm`.** This is what keeps `grouped_gemm_numeric_tests.rs` in rule 4's exemption list.

## 7. Remaining Work

- #1814: fill the `.rocm` slots, and export the support predicates that rule 4's two exemptions are waiting on.
- #1806: the nvfp4 abort, still the ROCm gate's only failure.

Refs #1801, #1803, #1814, #1885, #2018, #2026
