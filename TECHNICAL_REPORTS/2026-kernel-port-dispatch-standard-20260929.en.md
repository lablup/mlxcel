# Technical Report: PR #2026, choose custom-kernel ports through one helper

**Date**: 2026-09-29

**Status**: Implemented and validated on the gfx1151 host; pending merge. Metal and CUDA were not exercised, since neither is available here; the argument that their behavior is unchanged is by construction and is stated below.

**Languages**: C++ (kernel dispatch), Python (CI checker), Make, GitHub Actions

**Risk level**: Medium. Nine launchers change how they obtain a kernel, though the resolved kernel is identical on every backend that has a port.

## Executive summary

Every fused kernel chose its per-backend port by hand, as `use_cuda ? cuda_port : metal_port`. That expression reads "not CUDA" as "Metal". It was correct while Metal and CUDA were the only backends and became a defect the moment a third one existed. This PR replaces all nine hand-written choices with one helper, adds a table per kernel that both the dispatch and the support predicate read, and adds two mechanical defenses so the pattern cannot regrow.

The change fixes no user-visible bug on its own. What it removes is the need to remember something at every future launcher and every future backend.

## Problem statement

The defect this pattern produced is documented across three issues, and the sequence is the argument for standardizing rather than continuing to patch.

Issue #1803 found the original idiom, `!metal::is_available()` read as "CUDA", and replaced it with a named backend kind. It fixed the CUDA arm and left the false arm still meaning Metal, at nine sites. On ROCm those sites took the Metal arm, `fast::metal_kernel` threw, and because the bridge declarations were not `Result` the throw crossed a `noexcept` cxx extern and ended the process rather than failing the call.

Issue #1885 then fixed two of the nine, the samplers, after they aborted a gate run. PR #2018 fixed the other four two-way launchers plus the paged ones, also after a gate run surfaced them, months later. Each fix was correct in isolation and the set of them drifted: two guard comments cited support predicates that do not exist in the tree, one hardcoded an entry-point name that is wrong for one of its two callers, and the refusal wording differed per site.

A separate inconsistency had the same root. `ssm_kernel_available()` answered the "is there a port" question with `cu::is_available()` while the launcher it gates guarded on `custom_kernels_available()`. Two predicates for one question, with nothing keeping them in agreement.

None of this is a story about careless edits. It is what happens to a convention that has to be re-applied at every new launcher: it gets missed at some of them, and only the backend that lacks the port ever finds out, at runtime, as an abort.

## Change summary

- `src/lib/mlx-cpp/turbo/kernel_port.h` and `.cpp` (new): a `KernelPorts` table of per-backend getters, `select_kernel_port` which resolves one or throws a message naming the entry point and the fallback, and `has_kernel_port` which answers the support predicate from the same table.
- `src/lib/mlx-cpp/turbo/gpu_backend.h`: `kGpuKernelBackendCount`, pinned by a `static_assert` in `kernel_port.cpp`.
- Nine launchers converted: `gumbel_max_sample`, `rejection_sample`, `fused_add_rms_norm`, `fused_rope_qk_append`, `ssm_update_kernel`, `run_fused_moe_two_kernel`, `paged_attention_decode`, `paged_attention_decode_v2_partial`, `paged_attention_merge_states`. No backend comparison remains outside the helper.
- `scripts/ci/check_kernel_port_dispatch.py` (new), a `verify-kernel-port-dispatch` make target added to both `verify` and `verify-rocm`, and an unconditional hosted CI job.
- `run_fused_moe_two_kernel` takes its caller's entry-point name, fixing a mislabel introduced by PR #2018.
- Checker rule 3, added after reverting a Metal-only launcher by hand and watching the check still pass. Rules 1 and 2 describe how a multi-port launcher goes wrong; a single-port one goes wrong by reaching `get_x_kernel().get()` with no resolution at all, which neither sees. Rule 3 found five launchers that every earlier survey of this branch had missed, so the exemption list is now empty.
- Checker rule 4 and the Rust-side gates it governs: the parity-test helpers, one benchmark example, and the launcher call sites they guard.

## Technical decisions

### The enforcement is a compile error, not a warning

`-Wswitch` already flags a new enumerator in an exhaustive switch, and relying on that was considered and rejected. This repository compiles C++ with warnings enabled but not as errors, and the MLX build emits dozens of them on every run, so a new warning would not be seen. The `static_assert` on `kGpuKernelBackendCount` is a hard stop whose message names the three places to extend. Adding a backend now cannot silently read as "no port" at every kernel.

### All nine, not a subset

Converting some launchers and leaving others would mean the checker has to exempt the remainder, and an exemption list is the mechanism by which the old pattern would survive. Exemptions that do exist are named with a reason and a tracking issue rather than left as bare entries.

### A null table entry rather than an absent case

"No port for this backend" is written into the table as `nullptr` instead of being what a dispatch falls through to. That is the whole difference between this shape and the old one. It also means a Metal-only kernel and a fully ported one have the same shape, and it gives issue #1814 a named slot to fill rather than a dispatch to restructure.

### The support predicate reads the dispatch's table

This is the part that removes an existing bug class rather than a hypothetical one. With `has_kernel_port`, a predicate and its launcher cannot disagree, which makes `expect` at a gated call site correct by construction rather than by inspection.

### Two launchers left out deliberately

`turbo4_delegated_sdpa.cpp` and `sparse_v_sdpa.cpp` are Metal-only and have no refusal of their own. They do not need one to be safe: both are reached only through `cache::turbo::sparse_v::kernel_enabled()` (defined at `sparse_v.rs:151`, guarding every call at `:751`, `:929`, `:1243` and `:1318`, and the module's own parity test at `:1568`), which returns false unconditionally off macOS because dispatching the JIT kernel without a Metal device hangs.

This report originally said their reachability off Apple was unresolved and treated a latent abort as likely. That was wrong, and the way it was wrong is worth recording: the claim came from `sparse_v_available()`, which does gate only on an environment threshold and a KV cache mode with no backend term, without finding the kernel-level gate one layer below it. Two predicates answering different questions, and only the higher one was grepped for.

Verified afterwards by execution on the gfx1151 host rather than by reading again: `--kv-cache-mode turbo4-asym` applies to 24 of 28 layers, and forcing the sparse-V branch with `MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=0` still generates normally instead of aborting.

Bringing them into the table is therefore a consistency improvement, not a bug fix. What remains true is that their safety rests on a Rust-side gate rather than on the launcher, so a caller that skips that gate would get an abort and not an error.

### A Rust gate names its question, not the backends that answer it today

Three test helpers and one example gated on `metal_is_available() || cuda_is_available()`. Nothing about that condition is wrong, which is exactly the problem: it names the two backends that happen to have ports, so on a third backend it reads as a missing term. The obvious repair is to add `rocm_is_available()`, and it is the wrong one twice over. That function does not exist, and if it did, `fused_norm_ports().rocm` and `fused_rope_ports().rocm` are still null, so widening the gate would walk the tests past the port table into the launcher's refusal, which is what this branch had just finished converting into a catchable error.

Two different questions were wearing one name. "Does this backend have this kernel's port" is now `fused_add_rms_norm_available()` / `fused_rope_qk_append_available()`, which read `has_kernel_port` on the same table the dispatch reads. "Is there a GPU at all" is now `gpu_backend_available()`. The distinction is not cosmetic: `rms_norm_small_axis_tests.rs` tests MLX's own `fast::rms_norm` dispatch config rather than an mlxcel port, so the narrow gate had been skipping it on ROCm for no reason. Under the correct predicate both of its sweeps run on gfx1151 and pass, which is coverage the rename produced rather than restored.

Rule 4 keeps the spelling out. Its two exemptions are not deferred convention but missing predicates, each named: `ffi_tests.rs` gates the fused paged-decode kernel, which has a `KernelPorts` table but exports no `*_available()` for the bridge to call, and `grouped_gemm_numeric_tests.rs` gates MLX's own `gather_mm`, where `gpu_backend_available()` would be the honest predicate but whether MLX's ROCm backend implements the grouped-GEMM path is unverified here. Widening a gate on an assumption is how #1806's abort was reached, so both wait on #1814.

## Validation

- Full ROCm gate on gfx1151: one failing target, `-p mlxcel-core --lib`, unchanged from before this branch. That failure is the nvfp4 abort of issue #1806 and the only `terminate called` in the run.
- `cargo fmt --all -- --check`, `actionlint` with shellcheck at error severity, and `clippy --workspace --all-targets --features rocm -- -D warnings`: all clean.
- Each of the four defenses was confirmed by negative control rather than assumed to work. Reintroducing a two-way dispatch, hand-rolling a refusal, and reaching a holder directly each produce a checker failure naming the file and line. Rule 4 was controlled in three spellings, `a || b`, `!a && !b` and the reverse order without a path prefix, and caught all three. Adding a `Vulkan` enumerator fails the build on the `static_assert` with the intended message.
- `rms_norm_small_axis_tests` was run directly rather than inferred from the suite, because the nvfp4 abort terminates the test binary before any alphabetically later test reaches the runner. Both sweeps pass on gfx1151.

### Which clippy invocation is the gate

Converting a launcher to `Result` splits into two cases that are not caught alike. A launcher returning a value becomes `Result<T>`, so every call site stops compiling and `cargo check` finds them. A void launcher becomes `Result<()>`, and an ignored `Result<()>` is only the `unused_must_use` lint: `cargo check` stays green and the refusal is swallowed at runtime.

This cost four rounds on this branch. `cargo clippy -p mlxcel --lib --tests` is what PR-time CI runs and is what the earlier passes used, and it does not build mlxcel-core or examples, so it reported clean while five sites were unlinted: two in `mlxcel-core`'s parity tests, one in `layers.rs`, and two in the benchmark example, all of them `Result<()>`. `make verify` and `make verify-rocm` lint `--workspace --all-targets` and found every one. The note now lives in `kernel_port.h` beside the declaration requirement, since that is where someone converting the next launcher will be reading.

### Method note worth keeping

The bulk conversion was scripted, and the script placed a table inside a function body in three files because its insertion anchor was "the preceding blank line", which is not a structural boundary in C++. Two of those were caught immediately; the third survived a verification step that counted brace depth, which cannot distinguish a namespace from a function. The compiler caught all three. When editing C++ structure textually, the anchor has to be structural (here, the closing brace of a named holder accessor) and the verification has to be a compile, not a heuristic.

## What is not verified

- **Metal and CUDA were not run.** Neither is available on this host. The argument that their behavior is unchanged is that for any backend with a port the resolved kernel is the same object the old expression resolved, and the new throw path requires the table entry to be null, which it is not on those backends.
- **The refusal path was not re-probed after the refactor.** It was verified by execution before this change, in PR #2018, and the message text moved into the helper unchanged in shape. The gate exercises the success path on ROCm only through the launchers that fall back earlier.
- **`verify-rocm-smoke` did not run on the final tree.** The 0.6B fixture it loads lived under `/tmp`, which this host clears on reboot, and no local checkpoint remains. Every other step of `verify-rocm` was run on the final tree; the smoke last passed earlier in this branch's work, before the rule 4 changes, none of which touch a code path a generate run reaches. Set `MLXCEL_ROCM_SMOKE_MODEL` to a local checkpoint to close this.
- **Whether MLX's ROCm backend implements `gather_mm`.** This is what keeps `grouped_gemm_numeric_tests.rs` on the narrow gate and in rule 4's exemption list rather than on `gpu_backend_available()`. It is a question about MLX, answerable by running those three tests with the gate widened, and it is deliberately not answered by widening the gate and seeing what happens in a merge gate.

## Remaining work

Issue #1814 fills the `rocm` entries these tables now carry. Two follow-ups fall out of this work: bringing the two Metal-only SDPA launchers into the table once their reachability off Apple is settled, and resolving whether `sparse_v_available()` needs a backend term, which is a correctness question rather than a performance one.
