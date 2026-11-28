# Technical Report: PR #2073 - Validate MLX_ROCM_FFT_CACHE_SIZE on ROCm

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: C++ (ROCm overlay header), Rust (integration test), Markdown

**Risk Level**: Low (one header-only parsing function and two constructor guards in the ROCm overlay, a comment in `fft.hip`, a new ROCm-gated test and docs; Metal and CUDA builds never copy `patches-rocm/`)

## Executive Summary

Issue #2051 (part of epic #1801) came out of the security review of PR #2049 (#1876) and was already present on main. The ROCm overlay read `MLX_ROCM_FFT_CACHE_SIZE`, the capacity of the hipFFT plan cache, with an unchecked `std::stoul`. A mistyped value did not get rejected with a message: `0` led to undefined behavior at the first plan insert, junk threw a bare `stoul` out of every FFT, and `-1` and `8abc` were accepted silently as `SIZE_MAX` and 8. The users that notice are the audio paths (Kokoro TTS, Phi-4-multimodal audio), which run FFTs on the GPU.

The PR adds `capacity_from_env()` to the overlay's `lru_cache.h`. It parses the variable with the same `strtol` rules `gpu_watchdog_seconds()` already applies to `MLX_ROCM_GPU_WATCHDOG_SECS`: unset gives the default silently, 1 to `INT_MAX` is used, and anything else prints one stderr line naming the variable and the default, then uses the default. Both cache constructors now throw `LRUCache requires capacity > 0.` on 0, as upstream CUDA does. `tests/rocm_fft_cache_env.rs` runs one child process per value and checks the warning count and a GPU rfft against the CPU stream.

The PR also surfaced a build-system defect, now tracked in #2075: the overlay's HIP objects depend only on their own `.hip` source, so a header-only fix is not compiled on a warm build.

## 1. Problem Statement

### An unchecked parse behind a function-local static

`LRUBytesKeyCache`'s constructor did `capacity_ = std::stoul(env)`. Its only env-driven user is `fft_plan_cache()` in `fft.hip`, a function-local static initialized at the first device FFT (default 128 since #2049, 8 before). Per value, confirmed with a host g++ 14 probe of the header and on gfx1151:

| Value | Old behavior |
|---|---|
| `0`, `0x10` (read as 0) | Accepted. The first `put()` runs `while (size() >= capacity_)`, true at `0 >= 0`, and calls `back()` and `pop_back()` on an empty `std::list`: undefined behavior. The probe threw `std::bad_alloc`; with `-D_GLIBCXX_ASSERTIONS` it aborts on `!this->empty()`. On gfx1151 the first GPU rfft failed with `std::bad_alloc`. |
| `abc`, empty | `std::invalid_argument` with `what()` equal to `stoul`, naming no variable. |
| `99999999999999999999999` | `std::out_of_range`, also `stoul`. |
| `-1` | Accepted silently as `SIZE_MAX`: the cache never evicts. |
| `8abc` | Accepted silently as 8. |

A throw from a static's initializer leaves the static uninitialized, so every later FFT re-ran the constructor and threw again. The failure was persistent for the life of the process, and its message did not point at the environment variable.

### Upstream CUDA is not a template to copy

Upstream CUDA at the pinned MLX 81ba1c6a reads `MLX_CUDA_FFT_CACHE_SIZE` with `atoi` through `env::get_var`. `0` or junk becomes 0 and the constructor throws `LRUCache requires capacity > 0.`, which is at least not undefined behavior, but it still fails the first FFT, and `-1` still becomes `SIZE_MAX`. Matching CUDA exactly would have kept two of the defects.

## 2. Change Summary

- **`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/lru_cache.h`** (+47/-4): new `inline size_t capacity_from_env(const char* env_var, size_t default_capacity)` in `mlx::core::rocm`. `std::strtol` base 10 with `errno` reset; rejected if nothing was parsed, `*end != '\0'`, `ERANGE`, `v <= 0` or `v > INT_MAX`. A rejection prints `[ROCm] ignoring invalid MLX_ROCM_FFT_CACHE_SIZE="<value>" (expected a positive integer); using the default 128` and returns the default. `LRUBytesKeyCache` takes its capacity from it; `LRUBytesKeyCache` and `LRUCache(size_t)` both throw `std::runtime_error("LRUCache requires capacity > 0.")` on 0.
- **`.../rocm/fft.hip`** (+3/-1): the comment on `fft_plan_cache()` states the rule and points at `capacity_from_env`. The edit also forces a warm build to recompile `fft.hip` (section 3).
- **`tests/rocm_fft_cache_env.rs`** (new, 304 lines, `#![cfg(feature = "rocm")]`, skips unless `gpu_backend_kind()` is ROCm): a parent test re-invokes its own binary on an ignored child test, once per value, one child at a time, with `--test-threads=1`. Values: `0`, `-1`, `abc`, empty, `8abc`, `0x10`, `99999999999999999999999`, `2147483648` (rejected), and `16`, `1`, `2147483647`, unset (accepted). The child runs rfft on the GPU twice over lengths 400 and 512 and compares against the CPU stream (tolerance 1e-5), then prints a done marker. The parent drains stdout and stderr on their own threads, kills a child past a 120 s budget, and requires exit status 0, the marker, and exactly one stderr line mentioning the variable for a rejected value and none for an accepted one.
- **`src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`**: item 25 (24 is reserved for #2052), with the old behavior per value, the new rule, the CUDA comparison and the build-dependency note, marked as an upstreaming candidate for #1813.
- **`docs/environment-variables.md`**, **`docs/installation.md`**: the valid range (a positive integer up to 2147483647) and that any other value warns and uses the default.

Commits: `512d05b3` is the fix, test and docs; `0c287245` rewraps the environment-variables paragraph.

## 3. Technical Decisions

### Reuse the watchdog variable's parser rules instead of a new convention

The overlay already had one validated integer variable, `MLX_ROCM_GPU_WATCHDOG_SECS`, parsed in `device.cpp` with `strtol`, full-string match, and warn-plus-default on anything invalid. Copying those rules means the two ROCm variables behave the same way and one sentence in the docs describes both. Clamping (for example turning `0` into 1) was rejected for the same reason: it would be a second convention, and it would silently run with a value the user did not ask for.

`strtol` accepts leading whitespace and a `+` sign (` 16` and `+16` both read as 16, checked in the host probe). That was judged acceptable. `0x10` is rejected as trailing junk because the base is fixed at 10.

### Bound the value at INT_MAX

The upper bound is `std::numeric_limits<int>::max()`, not `SIZE_MAX` or `LONG_MAX`. CUDA reads its equivalent variable into an `int`, so no value above `INT_MAX` is meaningful on either backend, and a bound that fits `int` keeps `2147483648` (which fits `long` on Linux) from being taken as a near-unbounded cache. The test pins both edges: `2147483647` accepted, `2147483648` rejected.

### Warn once, and let the static initialize

Returning the default on a bad value, rather than throwing, is what makes the warning print once: the static's initializer completes, so it never runs again. Throwing would have repeated the persistent-failure pattern the issue describes. Rejecting silently would have hidden a configuration mistake, so the line names the variable, the value in quotes (which makes an empty or whitespace value visible) and the default it fell back to.

### Keep a zero-capacity guard in both constructors

After `capacity_from_env()` a zero capacity cannot come from the environment, so the throw in `LRUBytesKeyCache` is defense in depth against a future caller passing a default of 0. `LRUCache(size_t)` gets the same guard even though its only user, `graph_cache_{400}` in `device.h`, is a constant: `put()` has the same `while (size() >= capacity_)` loop, and a zero capacity there would be the same undefined behavior. The message matches upstream CUDA's word for word.

### A process per value

The plan cache is a function-local static that reads the variable once. Testing twelve values in one process would test only the first. The test follows the pattern of `tests/rocm_gpu_faults.rs`: re-invoke the test binary on an ignored child test with the variable set. The child is gated on a private environment variable, so an `--include-ignored` sweep does not run it with the host's environment. Children run one at a time because they share the GPU.

### Edit fft.hip so the fix is actually compiled

In the overlay's `CMakeLists.txt`, each HIP object is built by an `add_custom_command` whose `DEPENDS` lists only `${hip_src}`. The compiler's header dependencies are not tracked, so changing `lru_cache.h` leaves `hip_objs/fft.o` in place on a warm build. The PR author saw exactly this: the first test run with the header fix still showed the old behavior, and it passed only after `fft.hip` changed. The comment edit in `fft.hip` is therefore load-bearing for anyone building this branch on top of an existing build directory. `Device`, which constructs `graph_cache_`, lives in `device.cpp`, a regular C++ source whose header dependencies CMake does track, so the `LRUCache(size_t)` guard does not depend on the same trick.

## 4. Validation

PR author (gfx1151, ROCm 7.15):

- `cargo test --features rocm --test rocm_fft_cache_env`: all 12 cases pass.
- Revert check: with `lru_cache.h` reverted to main, all 8 invalid cases fail. `0` and `0x10` fail the first GPU rfft with `std::bad_alloc`; `abc`, empty and the overflow fail with `stoul`; `-1`, `8abc` and `2147483648` print no warning.
- `cargo test --features rocm --test rocm_fft_plan_cache` and `--test rocm_gpu_faults`: pass.
- `cargo clippy --features rocm --test rocm_fft_cache_env -- -D warnings`, `cargo fmt --check`, `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`, `cargo test --test dead_doc_pointers`: pass.
- A host g++ 14 probe of the header with `-D_GLIBCXX_ASSERTIONS` covered the same values plus ` 16` and `+16` (both accepted as 16).

Orchestrator verification (gfx1151, branch on origin/main `2cee9cf4`):

- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke run (32 tokens) passed. Because this PR edits `fft.hip`, the gate's warm build compiled the fix.
- `verify-test-rocm` failed in four targets, none caused by this PR. Three are the known baseline failures: `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, and #2037's `gelu_approx_matches_mlx_nn_bit_for_bit` and `tests::family_order_is_exhaustive`. The fourth is the pre-existing intermittent `rocm_mxfp4_quant::quantized_matmul_matches_dequantized_reference`, tracked in #2072.
- The new `tests/rocm_fft_cache_env.rs` passed inside the full suite.

## 5. Learning Points

- **A custom build command without header dependencies makes header fixes invisible.** The overlay compiles each `.hip` file through `add_custom_command(... DEPENDS ${hip_src})`, so CMake knows nothing about the headers a HIP object includes. A header-only fix builds cleanly, the old object is linked, and the first test run reports the old behavior, which looks like the fix is wrong. Here the first test run with the `lru_cache.h` fix still failed, and it passed only once `fft.hip` changed and was recompiled. Until #2075 lands (for example with `DEPFILE` and `-MD`, or `IMPLICIT_DEPENDS`), any change to an overlay header needs a clean build of the HIP objects or an edit to every `.hip` file that includes it, and a "fixed" result on a warm build should be confirmed to come from a recompiled object. `lru_cache.h` alone reaches 37 HIP sources through `device.h`.
- **An exception out of a static's initializer repeats forever.** C++ retries a function-local static's initialization after a throw, so a bad environment value turned into a failure on every FFT for the life of the process. Returning a default from a validated parse is what makes the warning print once.
- **Say which variable is wrong.** `std::stoul`'s `what()` is the function name. The new line names the variable, quotes the raw value and states the fallback, which is what a user needs to fix the setting.
- **Copy a sibling's convention, not the upstream's.** Upstream CUDA's `atoi` handling fails on `0` and accepts `-1`. The overlay's own watchdog variable already had a better rule, and reusing it keeps the ROCm variables consistent.
- **Revert the fix and watch the test fail.** The revert run is what proves each of the eight invalid values is actually caught by the test.

## 6. Caveats and What Is Not Verified

- **Metal and CUDA** were not run. The change is limited to `patches-rocm/`, which only ROCm builds copy, plus docs and a `#![cfg(feature = "rocm")]` test.
- **Other HIP objects that include `lru_cache.h`** (through `device.h`) are not rebuilt on a warm build. They neither read the variable nor construct `graph_cache_`, so nothing they do changes, but a partially stale build is still a state nobody tested directly. #2075 removes the gap.
- **The revert evidence is not in the repository.** The eight-case revert run and the host probe were done by hand.
- **Only gfx1151** was run.
- **Leading whitespace and `+` are accepted.** This follows `strtol` and the watchdog variable, and was a deliberate choice, but the docs describe the value only as a positive integer.

## 7. Remaining Work

- #2075: make HIP objects depend on the headers they include, so a header-only overlay change recompiles on a warm build.
- #1813: propose item 25 to the fork alongside the other upstreaming candidates.
- #2072: the intermittent `rocm_mxfp4_quant` failure seen in the gate, unrelated to this PR.

Refs: #2051 (closed by this PR), #1801, #1876, #1813, #2037, #2052, #2072, #2075, PR #2049.
