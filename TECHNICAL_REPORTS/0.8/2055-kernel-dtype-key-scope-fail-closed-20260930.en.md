# Technical Report: PR #2055 - Make the Kernel Dtype-Key Checker's Scope Fail-Closed

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: Python (CI checker), Bash (companion test), Makefile, GitHub Actions YAML, Markdown

**Risk Level**: Low (CI scripts and docs only; no source, kernel or build path changes. The new failure modes are all in the checker, and the current tree passes)

## Executive Summary

`scripts/ci/check_kernel_dtype_keys.py` guards the #1053 and #1054 bug class: on CUDA, a JIT kernel's cache name is `name + template_arguments_hash(template_args)` and nothing else, so a launch whose `template_args` name no input dtype silently reuses the first dtype's compiled module for every later dtype. The checker decided what to check with two rules nobody could see from the checked files: it globbed only `*.cpp` in two directories, and it skipped any file without the string `cuda_kernel(`. A refactor that moved a launch into a header, a shared helper or a HIP-only file shrank the scope while the check still printed OK. Issue #1875 (part of epic #1801) recorded this; it had already happened once, since the #1804 ROCm fault probe in `mlx_cxx_bridge.cpp` calls `fast::hip_kernel` and was never in scope.

The PR makes the scope fail-closed. The checker now scans every C, C++, CUDA, HIP and Objective-C++ source and header that git tracks or would track, treats a file as in scope when it calls `cuda_kernel(` or `hip_kernel(` outside a comment, reports the in-scope count next to the scanned count, and pins the in-scope set in `EXPECTED_IN_SCOPE` so that any drift fails. A new companion test, `check_kernel_dtype_keys_test.sh`, mutates throwaway copies of the tree across 16 cases and runs from both `make verify-kernel-dtype-keys` and the `kernel dtype keys` CI job. The current tree passes with 9 files in scope of 188 scanned.

## 1. Problem Statement

### The defect

Before this PR, `main()` iterated `SEARCH_DIRS = ("src/lib/mlx-cpp/turbo", "src/lib/mlxcel-core/cpp")` with `glob("*.cpp")`, and `check_file` returned early when `"cuda_kernel(" not in src`. Three consequences followed:

- A launch that moved into a header (`.h`, `.hpp`, `.cuh`) was never read, with or without the token.
- A launch that moved out of a file into a shared helper took the whole file out of scope, including the `template_args` initialiser it left behind.
- A file launching only through `fast::hip_kernel` was out of scope, even though ROCm has the same exposure (below).

The success line compounded this: it reported `scanned`, which counted every `*.cpp` before the token filter. The issue's refresh recorded `18 source files scanned` while 8 were actually checked. Adding an unrelated `.cpp` raised the number; dropping every launcher would not have lowered it.

### Why HIP has the same exposure

The ROCm overlay builds the kernel name exactly as CUDA does: `"custom_kernel_" + name + template_arguments_hash(template_args)` at `src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/custom_kernel.cpp:230-231`. `rocm::get_jit_module` (`patches-rocm/mlx/backend/rocm/jit_module.cpp:502`) memoises the compiled module in a process-global map keyed by `std::to_string(mlx_device.index) + ":" + name` and invokes the builder only on a miss. The device index is the only addition, and it does not carry dtypes, so the ROCm cache key is as dtype-blind as CUDA's. Metal is the only backend whose key appends the input dtypes.

### It had already happened

The #1804 ROCm fault probe (`rocm_fault_probe_array` in `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`) is a `fast::hip_kernel` launch in a file with no CUDA launch, so it sat outside the old scope from the day it landed. It passes `template_args` inline as `{}` at a fixed float32, so it has no dtype-key bug today, but the checker could not have caught one. The one other HIP launch, BitNet in `mlx_cxx_kernels.cpp`, was covered only because the same file also contains `cuda_kernel(`. Epic #1814 will add more `.rocm` ports, some of which could land in their own files.

## 2. Change Summary

- **`scripts/ci/check_kernel_dtype_keys.py`** (297 lines changed).
  - *Scan set.* `source_files()` takes `git ls-files -z --cached --others --exclude-standard` when `--root` is a work tree's top level, so tracked and untracked non-ignored files are seen and ignored build output and `references/` checkouts (which hold MLX's own launchers) stay out. Outside a work tree it walks the tree, skipping only top-level hidden directories and `PRUNED_DIRS` (`target`, `node_modules`, `build`, `site`, `__pycache__`). Files are filtered by `SOURCE_SUFFIXES`, which covers `.c .cc .cpp .cxx .cu .h .hh .hpp .hxx .cuh .inc .ipp .hip .m .mm`.
  - *Scope token.* `launches_jit_kernel()` matches `\b(cuda|hip)_kernel\s*\(` on the source after `without_comments()` has blanked comments. The lexer `LEXEME_RE` recognises raw strings (kernel sources are raw strings that contain `//`), ordinary strings, character literals (with a lookbehind so digit separators such as `1'000` do not open one) and comments; only comments are blanked, and newlines are kept so line numbers hold. A match preceded on its line by `CustomKernelFunction` is a definition or declaration in the vendored MLX tree and does not count. `\b` keeps `precompiled_cuda_kernel(` out, since it loads a prebuilt binary rather than JIT-compiling under a hashed name.
  - *Pin.* `EXPECTED_IN_SCOPE` lists the 9 launcher files: the seven turbo files, `mlx_cxx_kernels.cpp`, and now `mlx_cxx_bridge.cpp` (with a comment naming the #1804 probe). `scope_failures()` fails on an empty scope, on a pinned file that no longer exists, on a pinned file that exists but stopped launching (with a message explaining that its `template_args` are now unchecked), and on an unpinned new launcher.
  - *Output.* Success reads `kernel-dtype-keys: OK, 9 in scope (launching cuda_kernel or hip_kernel) of 188 source files scanned.` Failure prints the same counts, then rule failures and scope failures in separate blocks.
  - *CLI.* A `--root DIR` argument points the check at another tree; the companion test uses it. The rule itself (`TEMPLATE_ARGS_RE`, `DTYPE_BINDING_RE`, the per-initialiser check) is unchanged, as the issue's scope required.
- **`scripts/ci/check_kernel_dtype_keys_test.sh`** (new, 271 lines). Copies every directory holding a pinned launcher (including the vendored MLX trees, so the `CustomKernelFunction hip_kernel(` definitions are present) into a throwaway tree, mutates it, runs the checker with `--root`, and asserts the exit code and a message substring. Mutations go through a Python helper that fails when its target string matches nothing, so a case cannot pass because its mutation stopped applying. Cases include an untouched control, a dropped dtype key, a launch moved into a header in the same and in a new directory, a launch moved into a helper, a commented-out launch, a deleted launcher, a HIP-only launcher, the real bridge probe with int-only and dtype-keyed initialisers, an unpinned but correct new launcher, a comment-only header with a raw string containing `"` and `/*`, a `.hip` unit in a nested `build/` directory, a git work tree with an ignored `references/mlx` and an untracked launcher, and an empty scope.
- **`Makefile`.** `verify-kernel-dtype-keys` also runs the companion test, so `make verify` and `make verify-rocm` both run it. Its help text now says CUDA and HIP and cites #1875.
- **`.github/workflows/ci.yml`.** The `kernel dtype keys` job's first step is renamed for CUDA and HIP, a second step runs the companion test, and the job comment explains both.
- **`docs/code-guidelines.md`.** The enforcement paragraph describes the HIP rule, the scan set, the pin, the companion test, and two remaining limits (inline `template_args`, launches without a direct call).
- **`CONTRIBUTING.md`.** The `make verify` prerequisite list said six targets and omitted `verify-kernel-port-dispatch`; it now lists seven and describes the dtype-key gate as covering CUDA and HIP with a scope check.

Commit history: `58559aa3` is the fail-closed scope, the pin, the companion test and the wiring; `e3d01509`, from review, lists files through git, scans `.hip`, prunes directories only at the top level, and replaces GNU-only `sed -i` in the test with a Python helper.

## 3. Technical Decisions

### Pin the scope rather than only widen it

The issue weighed three options: widen the glob, print both counts, and pin the expected set. Widening catches a launch moved into a header, and printing both counts makes a drop visible to someone reading the log, but only the pin catches the case the issue was about: a launch moving into a shared helper, which leaves the original file's `template_args` behind and unchecked, while the helper file itself may carry none. The PR does all three. The cost is one more line to edit when a launcher is legitimately added or retired, and the failure message says exactly which line.

### Distinguish "deleted" from "stopped launching"

A deleted launcher and a launcher whose call moved elsewhere need different responses: the first only needs the pin updated, the second means some `template_args` are no longer checked. `scope_failures()` checks whether the pinned path still exists and prints a different message for each, so a reviewer does not have to work out which of the two happened.

### Fail on an unpinned new launcher

Without this, the pin would slowly go stale: new launchers would be checked by the rule but not guarded against later moving out. Requiring every in-scope file to be pinned keeps the pin equal to the scope at all times, which is what makes a later drop detectable.

### Let git decide what is scanned

The first commit pruned directories by name at any depth. Review found this both over-prunes (a nested directory named `build` holding real sources) and under-prunes (local `references/` checkouts contain MLX's own `cuda_kernel(` launchers and failed the local run as unpinned). Using `git ls-files --cached --others --exclude-standard` delegates the decision to `.gitignore`, which the repository already maintains, and still includes untracked files so a new launcher is seen before it is committed. The walk fallback exists for the companion test's throwaway copies, which are not work trees; it prunes only top-level directories.

### A lexer, not a substring test

The old filter was `"cuda_kernel(" in src`, which treats a comment mentioning the token as a launch. Blanking comments requires knowing where strings are, and kernel sources in this tree are raw strings that contain `//` and `/*`. `LEXEME_RE` matches raw strings, strings and character literals so they are skipped as units, and blanks only comments. The digit-separator lookbehind was needed because `0xFF'FF` would otherwise open a character literal and hide the code after it.

### Why HIP joins the rule unchanged

The rule's premise is that the cache key omits input dtypes. The ROCm overlay's `custom_kernel.cpp:230-231` and `jit_module.cpp:502` show the same name construction and a process-global memo keyed by device index and name, so the premise holds for HIP exactly. No HIP-specific rule is needed; the token just has to include `hip_kernel(`.

## 4. Validation

Author's runs (gfx1151): `make verify-kernel-dtype-keys verify-kernel-port-dispatch verify-versions verify-llama-compat verify-fmt` passes; `cargo test --features rocm --test dead_doc_pointers` passes (2 tests); the checker passes on the primary checkout (188 scanned, 9 in scope). All 11 negative cases of the companion test fail when run against the previous checker, which shows each case detects something the old checker missed.

Orchestrator verification (gfx1151 host, branch on origin/main `2c8045fe`). The change touches only CI scripts and docs, so the gates it affects were run rather than the full test suite:

- `make verify-kernel-dtype-keys`: exit 0. The checker reports 9 in scope of 188 source files scanned, and all negative-test cases in the companion test passed.
- `make verify-versions verify-kernel-port-dispatch verify-llama-compat verify-fmt`: exit 0.
- `.github/workflows/ci.yml` parses as YAML, with the new companion-test step at line 371.

## 5. Learning Points

- **A scope rule that is only run against a tree satisfying it cannot fail.** The old checker was never wrong about the files it read; it simply had no way to notice it was reading fewer. The companion test's approach, mutating a copy the way a refactor would and asserting failure, is the general fix for any CI check whose scope is derived rather than listed.
- **Report what was checked, not what was looked at.** `18 source files scanned` read as coverage while 8 files were checked. Printing the in-scope count separately turns a silent shrink into a visible number.
- **Guard the test's own mutations.** A negative case that edits a file with `sed` passes vacuously if the pattern stops matching after an unrelated edit. The test's Python replace helper fails on zero matches, so each case proves its mutation applied.
- **Check vendored backends for shared assumptions.** The HIP gap was found by reading the ROCm overlay's `custom_kernel.cpp` and `jit_module.cpp` and seeing the CUDA key shape repeated. When a rule is justified by one backend's internals, the other backends' equivalents should be read before assuming they differ.
- **Portable shell matters for local gates.** `sed -i` without a suffix works on GNU and fails on BSD sed, and this test runs from `make verify`, which is the macOS developer gate. Review caught it before merge.

## 6. What Is Not Verified

- **Metal and CUDA hosts.** Neither is available here. The change touches no Metal or CUDA code path, but the `kernel dtype keys` CI job on `ubuntu-latest` is the first run outside gfx1151.
- **macOS shell.** The companion test was made BSD-sed-safe but was not run on macOS, where `make verify` will execute it.
- **Launch forms the checker still cannot see.** A launch that passes `template_args` inline (as the #1804 probe does with `{}`) is in scope with nothing to check, and a launch reached through a function pointer or a macro defined elsewhere does not bring its file into scope. The pin catches a pinned file switching to such a form, not a new file that starts with one. Both limits are documented in `docs/code-guidelines.md` and the script docstring.
- **Full test suite.** Not run by the orchestrator, since no Rust or C++ source changed.

## 7. Remaining Work

- When #1814 adds `.rocm` ports, each new launcher file must be added to `EXPECTED_IN_SCOPE` in the same PR; the checker will fail until it is, which is the intended behavior.
- If a future launcher needs inline `template_args`, extending the rule to read them would close the first documented limit.

Refs: #1875 (closed by this PR), #1801, #1053, #1054, #1803, #1804, #1814, PR #1877, PR #2026, PR #2029.
