# Technical Report: PR #1772 - fix(mlx): bump MLX pin to 81ba1c6a so mxfp8 block scales round up

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle (implementation review, security and performance review, finalization)
**Status**: Completed on Metal (M1 Ultra). CUDA compiled and linked in CI (GB10, sm_121); no CUDA test or inference ran. M5 Max not measured.
**Languages**: C++, Rust, CMake, YAML
**Risk Level**: Medium-High (every MLX-owned kernel and host path moves by 99 upstream commits; measured on one GPU generation)

---

## Executive Summary

Issue #1769 reported that `fp8_block_requantize_matches_direct_path` failed deterministically on an M5 Max after its macOS 27 upgrade, and suspected the toolchain or the GPU generation. Running the issue's own decision criterion on an M1 Ultra, also on macOS 27, produced the byte-identical failure message. The test had simply never run on Metal: #1742 was validated on Linux/CUDA only.

The cause is in MLX, not in mlxcel's requantize path and not in the test. At the pinned commit `9a795735`, the Metal and CPU backends choose the mxfp8 E8M0 block exponent as `round(log2(amax / 448))`. Rounded to nearest in log2 space, the scale lands below `amax / 448` for about half the blocks, those blocks' largest values scale past E4M3's 448 ceiling, and they saturate, losing up to `1 - 2^-1/2` (29.3%) of the block maximum. CUDA rounds the exponent up, which is why the test passed where it was written. Upstream fixed Metal and CPU in ml-explore/mlx#4353, six days after the pin.

So the test's bound was correct and the quantizer was wrong. The same quantizer runs in production: `requantize_block_fp8_weights` is the only E8M0 quantize caller in the tree, and every vendor FP8 checkpoint loaded on Metal went through it.

The maintainer chose to move the pin to upstream main `81ba1c6a` rather than patch around it. That brought 98 further commits, and three of them change behaviour underneath the bridge in ways that compile or pass tests while being wrong: a parameter inserted mid-signature, a new link dependency, and an accessor that started throwing. All three are handled here. The first surfaced at compile time; review found the other two.

---

## 1. The failure, and why the issue's hypothesis did not hold

The issue asked for one experiment before any change: run the same seeded test on generation-13 hardware. The result on M1 Ultra:

```
element 4: mxfp8 error 0.50390625 exceeded 0.3002931 (group max 4.8046875)
```

That is the M5 Max message character for character. Both hosts ran macOS 27 and Xcode 27, so the experiment ruled out a generation difference but could not by itself rule out a toolchain one. What ruled that out was the history: #1742's validation section lists only CUDA runs, so there was never a macOS 26 pass to regress from.

The mechanism reads directly off the pinned kernel (`mlx/backend/metal/kernels/fp8.h`, `struct fp8_e8m0`):

```cpp
float le = metal::log2(x);
int n = int(metal::round(le));
```

For block 0 of the fixture, `log2(4.8046875 / 448) = -6.54` rounds to `-7`. The block maximum then scales to `4.8046875 * 128 = 615`, and E4M3's conversion saturates anything at or above 448. Element 4, `4.00390625`, scales to 512.5, saturates, and decodes as `448 / 128 = 3.5`, an error of 0.50390625. A host emulation of the pinned kernel over the whole seeded fixture reproduces that first failure exactly and gives the full picture: 301 of 650 blocks saturate, with a worst loss of 0.2928 of the group maximum. The theoretical ceiling for round-to-nearest in log2 space is `1 - 2^-1/2 = 0.2929`.

The issue proposed, if both hosts failed, widening the bound from half an E4M3 step (`group_max / 16`) to a full step (`group_max / 8`), on the theory that the quantizer rounded toward zero. It does not round toward zero; it saturates. A full-step bound would still fail on the saturated maxima, and any bound loose enough to pass would have certified the defect. That option was rejected.

---

## 2. Why a pin bump

Three options were on the table once the cause was known.

| Option | What it costs | Why it was or was not taken |
|---|---|---|
| Widen the test bound | nothing | Certifies a production accuracy defect. Rejected. |
| Overlay a backport of ml-explore/mlx#4353 | Whole-file overlays of `fp8.h`, the 2,000-line `fp_quantized.h`, CPU `quantized.cpp`, and `ops.cpp` (already a CUDA-only overlay) | Small in lines, but four new overlays to carry and later retire, one of them colliding with an existing overlay's target. |
| Move the pin to upstream main | 99 commits of upstream change to absorb and validate | Chosen by the maintainer. The tree tracks upstream main already, and the range carries other fixes this tree would want. |

The cost of the chosen option is that it moves the numbers everywhere MLX owns a kernel, which is why the validation in section 6 is broad rather than confined to the fp8 path.

---

## 3. Overlay reconciliation

This tree carries 28 whole-file overlays under `src/lib/mlx-cpp/patches/` (applied to every build) and `src/lib/mlx-cpp/patches-cuda/` (CUDA builds only). Upstream touched seven of their targets between the pins. Each of those seven was three-way merged with `git merge-file overlay old-upstream new-upstream`. The check that the merge preserved both sides is that each overlay's delta against the new base has the same `+/-` count it had against the old base: upstream's changes came in, and ours stayed. The other 21 targets are byte-identical across the range and were left alone. Two of those are mlxcel-only files with no upstream counterpart.

| Overlay target | Upstream commits | Delta, old base / new base |
|---|---:|---|
| `metal/quantized.cpp` (`MLXCEL_QMV_WIDE`) | 5 | +59/-1, +59/-1 |
| `metal/kernels/utils.h` | 2 | +5/-4, +5/-4 |
| `metal/compiled.cpp` | 1 | reduced to the mixed-dtype cast (see below) |
| `cuda/device/binary_ops.cuh` | 1 | +53/-0, +53/-0 |
| `cuda/quantized/quantized.cpp` | 1 | +358/-82, +361/-82 (sync comment only) |
| `patches-cuda/fast.cpp` | 2 | +33/-6, +33/-6 |
| `patches-cuda/ops.cpp` | 8 | +46/-11, +46/-11 |

The only conflict was cosmetic: upstream's ml-explore/mlx#4353 bumped the copyright year on the same header line where the `ops.cpp` overlay starts its own comment block.

Two of the merges matter beyond their line counts. `metal/compiled.cpp` now emits `cast_to<>` for fused `AsType` (ml-explore/mlx#4351). That function is defined in the new `kernels/utils.h`. Had the `utils.h` overlay been left at its old base while `compiled.cpp` took the new one, every JIT-fused kernel containing a cast would have failed to compile at runtime rather than at build time. `metal/quantized.cpp` carries the #1187 `qmv_wide` off-switch. The predicate it gates, `mode != "affine" || gen >= 15`, is unchanged upstream, and so is its single caller, so the switch still gates exactly what it did.

Review found one thing the reconciliation method cannot see: an overlay that was already wrong at the old base. `metal/compiled.cpp` had been emitting `elem_to_loc_1<uint>` for 1-D inputs since an earlier sync applied part of ml-explore/mlx#3720 by hand and missed this line. In the large-index kernel that negative strides select, `uint(stride)` wraps a stride of -1 to about 2^32, which is an out-of-bounds read. Nothing in the tree reaches it today. The line now matches upstream, the overlay's delta is down to the mixed-dtype cast it exists for, and its header says so and warns to compare against upstream on every bump. The CUDA mixed-type `FloorDivide` overload in `binary_ops.cuh` was aligned the same way: it floors, like upstream's float branch after ml-explore/mlx#4108.

---

## 4. What the new pin changes underneath the bridge

### 4.1 A parameter inserted mid-signature

ml-explore/mlx#4458 added `const std::optional<array>& global_scale` to `mlx::core::gather_qmm` between `mode` and `sorted_indices`. All 13 bridge call sites pass `sorted_indices` positionally. The risk in this class is a silent rebinding, where the bool binds to the new parameter and compiles. It does not happen here because `array`'s scalar constructor is `explicit`, so `bool` has no implicit route to `std::optional<array>` and the old calls fail to compile. A small parser inserted `/* global_scale = */ std::nullopt` into each call, and refused to touch any call that did not have exactly the expected 11 arguments. `fast::scaled_dot_product_attention` also gained a parameter (`force_fused`, ml-explore/mlx#4185), ahead of the stream, but no call in the tree passes a stream, so nothing rebinds.

### 4.2 A new link dependency

ml-explore/mlx#4208 moved Cholesky onto cuSOLVER, and the new pin's `gpu::init()` creates a cuSOLVER handle cache on every CUDA start. MLX's CMake links `CUDA::cusolver` PRIVATE, which cargo never sees, and `link_cuda()` in `mlxcel-core/build.rs` did not name it, so every `--features cuda` link would have failed on `cusolverDnCreate`. `link_cuda()` now names it. `docs/installation.md` now lists the shared libraries a prebuilt CUDA binary links, because a runtime-only CUDA install without cuSOLVER will now fail at launch.

That this reached review rather than CI is a gap in CI's path filters. The only PR job that links a CUDA binary, `xla-link`, triggered on the IREE half of the link recipe but not on `mlxcel-core/build.rs` or the MLX pin. The one CUDA job that did run for this PR was `cargo check`, which never links. `ci.yml` now triggers `xla-link` on both paths, and that is also what verified this change (section 7).

### 4.3 An accessor that started throwing

ml-explore/mlx#3742 made `array::detach_event()` call `Event::check_error()`, so `array::is_available()` now throws when a launch's command buffer failed, and the throw clears the error. Every event a failed command buffer signals points at the same encoder `Error`. At the old pin, `is_available()` on a signalled event returned true and dropped the error silently.

The rejection sampler relied on the old behaviour. `drain_pending_verification()` inspects converged flags from earlier launches, including launches other requests stashed, and it used `is_available()` as a non-blocking status query. It runs inside `fused_sample`, which is not a `Result` bridge function. After a GPU fault, the next sampling call on any request would have thrown through cxx and terminated the server. It would also have consumed the error that the owning request's own evaluation exists to report.

The drain now asks `stashed_launch_state()`, which reads the array's status, whether its event is signalled, and whether the event's error pointer holds a live message, all without calling anything that checks and clears. A failed launch is dropped unread. Metal's completion handler and the CPU scheduler both store an event's error before they signal it, and CUDA attaches none, so an error is visible by the time the signal is.

A regression test (`a_failed_stashed_launch_is_dropped_without_throwing_or_consuming_its_error`) builds that state through MLX's public `Event`/`Error` API. It signals a real event, attaches a synthetic error afterwards, stashes a launch carrying it, drains, and asserts that nothing threw, the error is still live, and the slot is empty. With the drain reverted to `is_available()`, the test binary aborts with `terminating due to uncaught exception of type std::runtime_error`, which is the failure mode itself.

### 4.4 Checked and not reaching the tree

`StreamContext` now throws when destroyed on another thread (ml-explore/mlx#4462); nothing in the tree uses it. The GGUF bounds checks and the lazy-source `save` fix (ml-explore/mlx#4212, ml-explore/mlx#4378, ml-explore/mlx#4434) cover loaders mlxcel does not call. `get_array_buffer_size` and `fast::cross_entropy` are additions.

---

## 5. The test

The byte-identity test, `fp8_block_requantize_matches_direct_path`, now asserts only the byte identity its name and doc comment claim. The accuracy check moved to `fp8_block_requantize_round_trip_stays_within_half_an_e4m3_step`, on the same seed (`0x51D3_9E11`) and the same 130x160 shape, because the padded trailing blocks are part of what is bounded. Its doc comment carries the derivation. With the scale rounded up, the block maximum scales into `(224, 448]`, and every element rounds to nearest. An element in the top binade moves by at most 16 of 256 or more units, `2^-4` of the maximum; lower binades move less.

The test also gained a direct check of the property the bound depends on: for each block, decode the E8M0 byte and assert `group_max <= 448 * scale`. On the old pin this is what fails first, and it names the cause:

```
block 0: E8M0 scale 2^-7 is below amax / 448 = 0.0107247485, so its maximum 4.8046875 scales to 615 and saturates at 448 (the scale was rounded down, see ml-explore/mlx#4353)
```

The comparison is exact for this fixture, not approximately so. Every fixture value is an E4M3 decode times a bf16 scale, at most 12 significant bits. `amax / 448` therefore cannot fall within f32 rounding distance of a power of two without equalling it.

---

## 6. Validation

All runs on M1 Ultra (generation 13), macOS 27.0, Xcode 27.0. The old-pin arm was built from `ca00c467` and isolated with its own `mlx.metallib`. The binaries resolve the metallib through an absolute path into the build directory, so without that copy the old binary would have loaded the new kernels once the rebuild overwrote them.

### 6.1 The fp8 round trip

| | Old pin | New pin |
|---|---|---|
| New test | fails at block 0 (above) | passes |
| Blocks saturated (of 650) | 301 (host emulation) | 0 (device) |
| Worst error / group max | 0.2928 (host emulation) | 0.0489, i.e. 0.96875 at 19.796875 (device) |

The device's new-pin figure equals the host emulation's round-up figure to every printed digit.

### 6.2 Turbo launchers

`sparse_v_kernel_threshold_zero_matches_graph` passes; `delegated_fused_kernel_matches_reference_over_200_steps` at 1.7263e-4 and `delegated_steel_envelope_matches_cold_only_fused_over_200_steps` at 1.5259e-4 max RMS, against the 5e-3 contract.

### 6.3 Teacher-forced logit traces

`examples/logit_trace` over wikitext-2, old pin against new, compared with `scripts/compare_logit_traces.py`, at width 1 (128 positions), width 8 behind 512 tokens of context (640), and width 256 (512):

| Checkpoint | Top-1 disagreement (w1 / w8 / w256) | Disagreement on decided positions |
|---|---|---|
| qwen2.5-7b-instruct-4bit | 0 / 3 / 1 | 0 at every width |
| gemma-3-4b-it-4bit | 0 / 0 / 0, byte-identical in effect | 0 |
| qwen3-30b-a3b-4bit | 4 / 14 / 15 | 0 at every width |
| nvidia-nemotron-3-nano-30b-a3b-4bit | 4 / 21 / 8 | 0 at every width |
| gemma-4-26b-a4b-it-4bit | 0 / 0 / 0, byte-identical in effect | 0 |

The Gemmas use GELU and do not touch upstream's switch to `precise::exp` in `Sigmoid` (ml-explore/mlx#4461). The SiLU families do, and move within the rounding class. qwen3-30b-a3b moved most because MoE routing turns a last-ulp change in a router logit into a different expert. Over 4,096 positions at width 256, 1 of 1,720 decided positions disagrees, at the reference's rank 2, and perplexity moves by -0.20%.

### 6.4 Branches the short runs could not reach

A 575-token prompt never reaches three dispatch changes, so each was driven directly. Runs are greedy with arms alternated, and figures are the median of 3 on a quiet machine.

| Branch | Checkpoint, context | Result |
|---|---|---|
| head-dim-512 vector decode past 1,024 keys | gemma-4-26b-a4b, 2,396 tokens | identical text on old pin, new pin, and new pin with `MLX_SDPA_D512_MIN_KL` disabling the kernel; 71.7 / 71.6 / 71.4 tok/s |
| GQA-8 two-pass decode past 8,192 keys (ml-explore/mlx#4077) | qwen3-30b-a3b, 8,819 tokens | identical text; decode 55.6 to 60.8 tok/s (+9.4%), prefill unchanged |
| head-dim-72 fused attention in vision towers (ml-explore/mlx#4330) | gemma-3-4b and gemma-4-26b with an image | image prefill 239 to 279 tok/s on gemma-3-4b; descriptions diverge after 20 to 50 tokens into equally faithful text; decided answers (the fixture square's colour, gemma-4-26b's count of six bars) unchanged |

### 6.5 Throughput at short context

`mlxcel generate --profile`, 575-token prompt, 128 tokens, median of 3: qwen2.5-7b prefill 681 to 677 and decode 100.3 to 100.4 tok/s; qwen3-30b-a3b 582 to 586 and 76.1 to 75.8; gemma-4-26b-a4b 713 to 709 and 70.9 to 70.8. All within 0.6%.

### 6.6 Gates

At the final head, the workspace gate (`cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1`) reports 122 binaries, 10,986 passed, 0 failed, 359 ignored. That is the issue's 10,981 passed plus #1770's two repairs, this issue's one, and the two new tests. Workspace clippy with `-D warnings`, `cargo fmt --check`, both pin parsers, and the cross-repo reference check are clean.

---

## 7. What is not verified

- **CUDA execution.** CI's `OpenXLA feature link` job, which this PR's filter change now triggers, built `mlxcel-core` at the new pin and linked a `--features cuda,xla-iree` release test binary on GB10 (sm_121) in 12m19s, so the merged CUDA overlays compile and the cuSOLVER link resolves. No CUDA test or inference ran. The green `CUDA sm_70 compile` check is a skip: its runner's CUDA 13.0 cannot target sm_70, so it compiles nothing.
- **M5 Max.** The quantize kernel has no generation-specific branch, and the issue's M5 failure message matches M1 Ultra's, but the NAX paths that changed in this range run only on generation 17 and were not measured: ml-explore/mlx#4171, ml-explore/mlx#4352 and ml-explore/mlx#4392.
- **A real FP8 checkpoint on Metal.** None is on this host. The fixture is the measurement vehicle, and on it the quantizer's behaviour is now exactly the round-up rule.
- **VLM coverage.** Two checkpoints, one synthetic image and one solid-colour fixture.

---

## 8. Follow-ups

- **`array_evaluated_bytes` is another non-`Result` bridge function that waits.** It goes through `array::eval()`, and at the new pin it throws for a failed launch even after the event has landed; the old pin threw only before. The server's lookahead token read uses it, so a GPU fault there still terminates the process. The fix is to declare it `Result` and route the error through the scheduler's step-failure handling, which is a change of its own.
- **The mixed-dtype cast in `metal/compiled.cpp` also casts comparison inputs.** For a comparison, the output dtype is `bool`, so the overlay would cast float inputs to `bool` before comparing. No compiled function in the tree contains a comparison today, so this is latent. It should be guarded before one does.
- **The NemotronH loader prints progress to stdout.** That contaminates `examples/logit_trace`'s TSV, which is written to stdout, and `compare_logit_traces.py` rejects the file until the `[NemotronH]` lines are filtered.

---

## 9. Learning points

- **Run an issue's decision criterion before accepting its framing.** The issue expected M1 Ultra either to pass (a backend divergence) or to fail with a bound that never held. It failed identically, and the reason was neither: a quantizer fixed upstream after the pin, behind a test that had only ever run on the one backend that rounds up.
- **A green check proves only what it ran.** Three checks on this PR looked like CUDA coverage and were not. `cargo check` never links, the sm_70 job skipped, and the link job was not triggered. The missing library was found by reading the upstream diff.
- **Pin bumps break semantics, not only signatures.** The signature change failed to compile. The link change and the accessor change would have compiled, passed the Metal gate, and failed in production. Read upstream's accessor and error-path changes, not only its headers.
- **Isolate every arm of a pin A/B, runtime artifacts included.** A binary that resolves its metallib by absolute path is only half of an arm.
