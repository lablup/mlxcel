# Technical Report: PR #2867 - fix(metal): bound quantized loader reads to the tile extent

**Date**: 2026-11-06
**Status**: In review (open PR; closes #2852)
**Languages**: Metal/C++, Rust, CMake
**Risk Level**: Medium

---

## Executive Summary

PR #2867 fixes the intermittent NaN that kept reopening issue #2852. It is not a leaking process global or stale allocator contents: MLX's Metal `QuantizedBlockLoader::load_safe` honored only the row extent (`.y`) of its tile and ignored the non-row extent (`.x`), so the partial-K tile of a transposed quantized product dequantized the next output row's packed bytes and group scale. In the Nemotron-H nvfp4 fixture, where `FF = 48` makes the last K tile 16 of 32 wide, that read left the last expert's packed weight plane by 8 bytes and its E4M3 scale plane by 1 byte. A stale `0x7F` scale byte decodes to NaN, the matching x tile is zero-filled, and `0 * NaN` turned the last output column of that expert's 20 pairs into NaN, exactly the reported `non-finite 20 of 2560` at 40 tokens. The fix bounds each thread's reads by `src_tile_dim.x` and zero-fills the remainder in all three loader copies (steel fp, affine, NAX). A new failing-first regression test reproduces exactly 20 non-finite elements on the unfixed tree and passes with the fix; the full serial lib suite is 11368 passed, 0 failed, 239 ignored. The NAX copy is compile-validated only on this M3 Ultra, which has no NAX device.

---

## 1. Problem Statement

### 1.1 Background

Issue #2852 tracked a single intermittent failure in `models::nemotron_h::tests::nemotron_h_runs_nvfp4_experts_sorted_and_unsorted` at the post-merge workspace gate: "nvfp4 routed experts differ from the dequantized reference at 40 tokens". The same test passed 10 of 10 times in isolation with max absolute error 7.6e-6 against a 1e-3 bound, and 40 or more full-suite runs did not reproduce it. The failure was order-dependent, which is worse than a stable red test because it lands on unrelated PRs.

Earlier work ruled out the obvious process globals. PR #2866 falsified a leaked MLX default device by forcing the CPU and measuring 5.7e-6 at 40 tokens, still inside the bound, and committed the measurement as a guard test. The `fused_moe_enabled` `OnceLock`, `SSD_PREFILL_DISPATCH_LOGGED`, the rope override singleton, a leaked per-thread default stream, and a compiled-function cache were each excluded with a probe. That left MLX allocator and Metal command-buffer state, which pointed at stale memory being read by a kernel rather than a value being written wrong.

The failing test builds a one-layer nvfp4 Nemotron-H MoE from deterministic synthetic weights, dequantizes the same packed planes as an f32 reference, and compares `SwitchMLP::forward` at 5 tokens (unsorted gather) and 40 tokens (sorted gather). Only the 40-token arm failed, so the defect is on the sorted gather path, not in the shared dequantization or the unsorted kernel.

### 1.2 Existing Issues

- **`load_safe` checked only half its contract.** `load_safe(short2 src_tile_dim)` documents `src_tile_dim` as the tile's valid extent, but the loader tested only `.y` (BROWS). The `.x` extent, which in the `reduction_dim == 1` layout is the reduction (K) axis, was ignored.
- **A partial K tile dequantized the next output row.** The fixture's `FF = 48` gives fc2 a K of 48 against a 32-wide K tile, so the final tile is 16 wide. The loader still dequantized all `n_reads` packs, taking the upper half of the 16-wide tile from the next output row's packed bytes and its group scale.
- **The last output row of the last expert read past the planes.** Those reads leave the packed weight plane by 8 bytes (past offset 3072) and the E4M3 scale plane by 1 byte (past offset 384).
- **A stale scale byte became a NaN output.** E4M3 `0x7F` decodes to NaN. The matching x tile is zero-filled, so the product is `0 * NaN`, which is NaN and reaches a valid output column instead of being harmlessly discarded.
- **The symptom accounted for every observation.** Exactly 20 pairs route to the last expert, producing `non-finite 20 of 2560` at 40 tokens, with every offending index congruent to 31 modulo 32: the last output column of each of those 20 rows. The 5-token arm takes another kernel and never failed.
- **The ordering decided only the poison.** Which earlier test last left a `0x7F` in the size class the 384-byte scale plane reuses determined whether the read landed on a NaN, which is why the failure looked order-dependent.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| The out-of-bounds read corrupts a served quantized matmul | High (wrong or NaN logits) | Low on this fixture, but the same loader serves every transposed quantized product, so any model with a non-multiple K can hit it |
| A stale NaN byte is misread as a numerical tolerance problem | Medium (the 1e-3 bound gets widened and hides the defect) | High if the derivation is not recorded, which is why this report records it |
| The NAX copy is wrong in a way that only an M5-class machine sees | Medium (runtime corruption on NAX hardware) | Unknown: only compile-validated here |
| Zero-filling the remainder changes in-bounds numerics | Medium (silent accuracy change) | Low: the x loader already zero-fills there, so the product it feeds is the same |
| The regression test is flaky rather than deterministic | Medium (a canary that cannot be trusted) | Low: the pool is cleared and seeded explicitly before the arms run |

---

## 2. Technical Review

### 2.1 Security

No network, authentication, input parsing, or serialization surface changes. The fix is a bounds check inside a device-side loader. It removes an out-of-bounds read, so it strictly reduces the kernel's memory-touching surface. No user data is logged.

### 2.2 Performance

The change adds one integer division, one comparison, and a zero-fill loop per `load_safe` call. `load_safe` runs once per K tile per thread group, not per output element, so the cost is in the noise of the cooperative load. In-bounds tiles now dequantize the same packs with the same scale, and the zero-filled remainder is what the x loader already placed there, so the arithmetic and memory traffic of answer-bearing tiles are unchanged. No benchmark was run because the change is a bounds guard, not a throughput path.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. Kernel behavior for full tiles is unchanged.
- **New Dependencies**: none. The three overlay headers are verbatim upstream copies plus the marked `load_safe` bodies, pinned to upstream MLX `7ca40033132dae444d45f7fc3daae57e24c13f6c`.
- **Compatibility**: the fix covers the steel fp loader used by nvfp4, the affine loader used by float and half scales, and the NAX loader. The NAX path is not exercised on this host.
- **Overlay maintenance**: `src/lib/mlxcelverse/CMakeLists.txt` registers the three files in `_metal_patch_files` and records the refresh rule: take the new upstream file and re-apply the three marked `load_safe` bodies.

### 2.4 Code Quality

The new regression test `nemotron_h_nvfp4_partial_k_tile_does_not_read_past_the_expert_planes` is deterministic: it calls `clear_cache()`, seeds every 4-byte size in `388..=768` with `0x7F`, then runs both arms and asserts that every output is finite. The 388-to-768 window is the allocator's `reuse_from_cache(384)` acceptance range `[384, 768)`, so the 384-byte scale plane necessarily reuses a seeded buffer. The parity diagnostic now reports non-finite counts per side (`got`, `reference`), which distinguishes a kernel-produced NaN from a reference-produced one. `cargo fmt --all -- --check` and `cargo clippy --profile test-fast --features metal,accelerate -p mlxcel --lib --tests -- -D warnings` are clean.

---

## 3. Technical Decisions

### 3.1 Bound reads by `src_tile_dim.x` and zero-fill the remainder

**Context:**
`load_safe` is called with the tile's valid extent when a K or N extent does not fill the kernel tile. The loader checked `.y` (rows) but not `.x`, so it dequantized `n_reads` packs regardless of the K extent. `Fp4HalfLoader::load_safe` in the same header already bounded its columns, which is the pattern the fix follows.

**Alternatives Considered:**

| Option | Pros | Cons |
|---|---|---|
| Option A: bound each thread's reads by `src_tile_dim.x`, zero-fill the rest | Mirrors the existing `Fp4HalfLoader` pattern, keeps in-bounds numerics identical, no host-side change | Adds a branch and a fill loop per loader call |
| Option B: mask the output after the kernel | No loader change | The NaN is already in the accumulator; `0 * NaN` cannot be masked after the fact, and the scale read itself is still out of bounds |
| Option C: pad the weight and scale planes on the host | Keeps the kernel untouched | Allocates and copies per expert, changes plane strides, and hides the contract violation instead of fixing it |
| **Chosen: Option A** | Fixes the contract at its source and matches an existing, reviewed pattern in the same file | Small per-call branch |

**Rationale:**
The read has to be prevented, not repaired: once the NaN scale multiplies a zero-filled x tile, the result is NaN and no downstream mask can recover the intended zero. Bounding the read also skips the scale load entirely when a thread's first pack is already past the tile, which is what stops the last output row from touching the byte past the 384-byte scale plane.

**Trade-offs:**
The loader now carries a fill loop for partial tiles. It is the same fill the x loader already performs, so no scratch layout changes and no extra allocation.

### 3.2 Apply the fix to all three loader copies

**Context:**
The overlay carries three copies of the loader: the steel fp loader (`fp_quantized.h`), the affine loader (`quantized.h`), and the NAX copy (`fp_quantized_nax.h`). Upstream duplicates the code per kernel family.

**Rationale:**
All three share the same `.y`-only check. Fixing one leaves the same defect reachable through the affine path (float and half scales) and through `fp_gather_qmm_rhs_nax`, which reaches `loader_w.load_safe(tile_w)` with `tile_w.x = k_remain` when K is not a multiple of the 64-wide NAX K tile. The CMake overlay list registers all three so a future upstream refresh cannot silently drop one.

**Trade-offs:**
Three copies must be kept in sync on every upstream refresh. The refresh rule is recorded next to the file list for that reason.

### 3.3 A seeded-pool regression test instead of a probabilistic reproduction

**Context:**
The original failure appeared once per contaminated buffer pool, so a plain rerun cannot prove the fix.

**Alternatives Considered:**

| Option | Pros | Cons |
|---|---|---|
| Option A: seed the reuse window explicitly, assert finiteness | Deterministic on every run, encodes the exact mechanism | Depends on the allocator's reuse policy, which the test documents |
| Option B: retry the fixture until it fails | No allocator knowledge needed | Non-deterministic, slow, and unusable as a gate |
| Option C: assert the read count from the kernel | Direct | Needs kernel instrumentation and a separate build path |
| **Chosen: Option A** | Fails on the unfixed tree every time, passes with the fix | Tied to the current reuse window |

**Rationale:**
`clear_cache()` empties the pool, then every 4-byte size from 388 to 768 is created and dropped with `0x7F7F7F7F` contents. Any buffer the 384-byte scale plane can reuse is therefore poisoned, so the out-of-bounds byte is a NaN on every run. The test then asserts finiteness at both arms and keeps the existing allclose check, so it fails for the right reason and not on a shifted tolerance.

**Trade-offs:**
If the allocator's reuse rule changes, the seeded range must follow. The comment records why the range is `[384, 768)`.

### 3.4 The falsified alternatives

**Context:**
The symptom looked like an unwritten output row or a leaking global, and three earlier investigations pursued those.

**Rationale:**
The stale-pool under-write hypothesis was tested directly: two runs over the same graph with an all-NaN pool and an all-7.0 pool produced bit-identical results, so no valid output element is left unwritten. That rules out an output row the kernel never stores. The earlier probes also missed the real read for a mechanical reason: an all-NaN `f32` poison writes `0x7F` only at offsets congruent to 3 modulo 4, while the out-of-bounds byte is 4-aligned, so the poison never landed on it. The ordered section-byte seeding removes both confounders.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
quantized matmul (steel / affine / NAX)
  └── QuantizedBlockLoader::load_safe(src_tile_dim)
        checks src_tile_dim.y only
        dequantizes n_reads packs per thread
        └── partial K tile reads next output row's bytes and group scale

[After]
quantized matmul (steel / affine / NAX)
  └── QuantizedBlockLoader::load_safe(src_tile_dim)
        checks src_tile_dim.y and src_tile_dim.x
        reads = min(n_reads, valid_packs - bj) when bi < src_tile_dim.y
        zero-fills the remaining packs
        └── partial K tile stays inside its own row and plane
```

### 4.2 Key Code Changes

**File: `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/fp_quantized.h`**

Before, the extent check used only `.y` and every thread read all of its packs:

```cpp
void load_safe(short2 src_tile_dim) const thread {
  if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
    return;
  }

  float scale =
      float(dequantize_scale<T, group_size>(*scales)) * inv_scale_enc;
  for (int i = 0; i < n_reads; i++) {
    dequantize<T, bits>(
        src[i * bytes_per_pack], scale, dst + i * pack_factor);
  }
}
```

After, each thread reads only the packs inside the tile and zero-fills the rest:

```cpp
const int valid_packs =
    (int(src_tile_dim.x) + pack_factor - 1) / pack_factor;
const int reads = int(bi) < int(src_tile_dim.y)
    ? min(int(n_reads), valid_packs - int(bj))
    : 0;

if (reads <= 0) {
  for (int i = 0; i < n_reads * pack_factor; i++) {
    dst[i] = T(0);
  }
  return;
}

float scale =
    float(dequantize_scale<T, group_size>(*scales)) * inv_scale_enc;
for (int i = 0; i < reads; i++) {
  dequantize<T, bits>(src[i * bytes_per_pack], scale, dst + i * pack_factor);
}
for (int i = reads; i < n_reads; i++) {
  for (int j = 0; j < pack_factor; j++) {
    dst[i * pack_factor + j] = T(0);
  }
}
```

**Reason for change:** `valid_packs - bj` is the number of packs this thread may own inside the tile. The early return when that is non-positive also prevents the `*scales` load, which is the read that goes one byte past the 384-byte plane for the last output row of the last expert.

**File: `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/quantized.h`**

The affine loader gets the same bound, with the scale and bias loads moved below the early return:

```cpp
const int valid_packs =
    (int(src_tile_dim.x) + pack_factor - 1) / pack_factor;
const int reads = int(bi) < int(src_tile_dim.y)
    ? min(int(n_reads), valid_packs - int(bj))
    : 0;

if (reads <= 0) {
  for (int i = 0; i < n_reads * pack_factor; i++) {
    dst[i] = T(0);
  }
  return;
}

T scale = *scales;
T bias = *biases;
```

**Reason for change:** this is the twin used by float and half scales. It had the same `.y`-only check and the same out-of-plane scale and bias reads.

**File: `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/fp_quantized_nax.h`**

The NAX copy computes the valid read count the same way, then limits the group loop and zero-fills the tail:

```cpp
const int valid_groups =
    (valid_reads + n_reads_per_scale - 1) / n_reads_per_scale;
int k = 0;
for (int i = 0; i < valid_groups; i++) {
  float scale =
      float(dequantize_scale<T, group_size>(scales[i])) * inv_scale_enc;
  for (int j = 0; j < n_reads_per_scale; j++) {
    if (k < valid_reads) {
      dequantize<T, bits>(src[k * bytes_per_pack], scale, dst + k * pack_factor);
    } else {
      for (int p = 0; p < pack_factor; p++) {
        dst[k * pack_factor + p] = T(0);
      }
    }
    k++;
  }
}
for (; k < n_reads; k++) {
  for (int p = 0; p < pack_factor; p++) {
    dst[k * pack_factor + p] = T(0);
  }
}
```

**Reason for change:** NAX keeps its own copy and reaches the same partial-K finalize through `fp_gather_qmm_rhs_nax`, but with a 64-wide K tile. It is fixed for symmetry; it cannot be exercised on this host.

**File: `src/models/nemotron_h_tests.rs`**

The new test clears the pool, poisons the reuse window, and asserts finiteness on both arms:

```rust
mlxcel_core::memory::clear_cache();
let seeds: Vec<_> = (97..=192u32)
    .map(|words| {
        mlxcel_core::from_slice_u32(&vec![0x7F7F_7F7Fu32; words as usize], &[words as i32])
    })
    .collect();
drop(seeds);

run_nvfp4_expert_arms(|tokens, got, want| {
    let got_values = mlxcel_core::utils::array_to_vec_f32(got);
    let non_finite: Vec<usize> = /* indices of non-finite values */;
    assert!(non_finite.is_empty(), ...);
    /* keep the allclose and parity checks */
});
```

**Reason for change:** the failure is a read past memory the kernel owns, so finiteness is the correct assertion. A tolerance check would pass on a shifted-but-finite result.

**File: `src/lib/mlxcelverse/CMakeLists.txt`**

Registers the three overlay files in `_metal_patch_files` and records the refresh rule: take the new upstream file and re-apply the three marked `load_safe` bodies.

### 4.3 Data Model Changes

None.

---

## 5. Learning Points

### 5.1 A tile extent is a two-dimensional contract

**Concept:**
Cooperative loaders that fill a register tile from a strided plane take a `short2` extent. When only one component is checked, the other is silently assumed to equal the compile-time tile size. That assumption holds for every full tile and fails only on the boundary tile of a non-multiple extent, which is exactly the case the `_safe` variant exists to handle.

**Application in this PR:**
The loader checked `.y` (BROWS) and ignored `.x` (the K extent in the transposed layout). The boundary tile then read a full 32 packs where only 16 were valid, taking the upper half from the next output row.

**Common Use Cases:**
- Any `load_safe`-style cooperative tile loader with a runtime extent.
- Quantized matmul kernels where K, N, or group count is not a multiple of the tile.
- Code review of `_safe` variants: verify that every component of the extent argument is used.

### 5.2 `0 * NaN` turns a discarded lane into a wrong answer

**Concept:**
A partial tile is usually harmless because the padded lanes multiply zero activations. That reasoning fails when the padded weight lane is NaN: IEEE 754 defines `0 * NaN` as NaN, so the "discarded" lane contaminates the accumulator.

**Application in this PR:**
The x tile is correctly zero-filled at the partial K position, but the weight side read a stale `0x7F` E4M3 scale byte and dequantized to NaN. The product was NaN, and it landed in a real output column.

**Common Use Cases:**
- Quantized kernels with block scales, where a stale scale byte can be any bit pattern.
- Any masked or padded reduction: masking an operand does not make a non-finite other operand safe.

### 5.3 Poison patterns must match the read width and alignment

**Concept:**
An allocator poison that writes a specific byte pattern is only effective at the offsets it actually writes. An `f32` all-NaN poison writes `0x7F` in the high byte of each word, leaving the other three bytes at their previous values.

**Application in this PR:**
The out-of-bounds read is 4-aligned and reads an E4M3 scale byte, so an all-NaN `f32` pool wrote `0x7F` at offsets congruent to 3 modulo 4 and never poisoned the read. Seeding at word granularity with `0x7F7F7F7F` removes the alignment gap.

**Common Use Cases:**
- Deterministic reproduction of out-of-bounds reads.
- Allocator or cache poisoning in tests: seed the exact size class and byte offset the bug reads.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---|---|---|
| `QuantizedBlockLoader` | MLX Metal loader that dequantizes a packed weight tile into a register tile | The defective component |
| `src_tile_dim` | `short2` carrying the tile's valid row and column extent | `.x` was ignored; that is the bug |
| `reduction_dim == 1` | Layout where the reduction axis is BCOLS, used by every transposed quantized product | Makes `.x` the K extent |
| Partial-K tile | Final K tile when K is not a multiple of the tile width (48 against 32 here) | Where the out-of-bounds read happens |
| E4M3 | 8-bit float format used for nvfp4 block scales | `0x7F` decodes to NaN |
| nvfp4 | 4-bit weight format with block scales at group size 16 | The fixture's quantization |
| `load_safe` | Bounds-checked variant of the loader, called on boundary tiles | The function that was fixed |
| NAX | Apple GPU generation with its own kernel copy | The copy that is compile-validated only here |

### Related Technologies/Frameworks

- **MLX**: Apple's array framework. The Metal backend and its quantized matmul kernels live under `mlx/backend/metal/kernels/`.
- **Metal**: Apple's GPU API and shading language. The loaders are C++ templates compiled to Metal.
- **mlxcel Metal overlays**: the project's mechanism for patching upstream MLX sources at build time, registered in `src/lib/mlxcelverse/CMakeLists.txt`.

### Related PRs/Issues

- Issue #2852: the intermittent failure this PR closes.
- PR #2866: falsified the default-device hypothesis and added the permanent guard test, narrowing the search to allocator and command-buffer state.
- PR #2853 and PR #2865: the neighboring speculative-decoding work from the same batch, included here only because they share the `TECHNICAL_REPORTS` record.
- Issue #2515 and issue #2409: earlier changes to the same test binary and test order that determined when the failure surfaced.

---

## 7. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 5 |
| Lines added | +6988 |
| Lines deleted | -4 |
| Tests added | 1 regression test, plus per-side non-finite counters in the parity diagnostic |
| Semantic kernel change | About 20 lines across three `load_safe` bodies |

The large line count is the three overlay headers: each is a verbatim upstream copy plus the marked fix, so most lines are the upstream file, not new logic.

### Changes by Category

| Category | Count | Summary |
|---|---|---|
| Correctness | 3 headers | Bound `load_safe` reads by `src_tile_dim.x` and zero-fill the remainder in the steel fp, affine, and NAX loaders |
| Tests | 1 file | Deterministic seeded-pool regression test; per-side non-finite reporting |
| Build | 1 file | Register the three overlays and record the refresh rule |
| Documentation | 0 files | The derivation lives in the header comments and this report |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `e2c2546` | fix | fix(metal): bound quantized loader reads to the tile extent |

---

## 8. Follow-up Actions

### Required

- [ ] Exercise the NAX loader on an M5-class machine. The NAX copy is compile-validated only because this M3 Ultra has no NAX device, so its runtime behavior is unverified.
- [ ] Re-apply the three `load_safe` bodies after any upstream MLX refresh. The CMake file list records the rule, but nothing enforces it mechanically.

### Unvalidated Surface

- The NAX copy (`fp_quantized_nax.h`) has not run. It shares the derivation and the pattern of the steel fix, but its 64-wide K tile path and its group loop were never executed here.
- The affine loader is covered by the full serial lib suite, but no test targets a partial-K float or half-scale case specifically; its coverage is incidental.
- The regression test depends on the Metal allocator's `reuse_from_cache(384)` acceptance window staying `[384, 768)`. A change to that policy would require re-deriving the seeded range.
- No served-model smoke test was run. The fix is validated by the nvfp4 fixture and the lib suite, not by a real nvfp4 checkpoint.

### Monitoring Required

- Watch `nemotron_h_nvfp4_experts_sorted_and_unsorted` and the new partial-K test on the CI hardware lane, especially after MLX or quantization changes.
- The per-side non-finite counts in the parity diagnostic are the canary: `got` only means a kernel-produced value, both sides means something shared.

### Future Improvements

- Add a central helper for the `load_safe` extent bound so the three copies cannot drift apart on refresh.
- Consider a host-side assertion or debug kernel mode that detects a read past the plane in development builds, so a future contract violation fails loudly instead of depending on a poisoned allocator.

---

## Appendix

### A. Test Results

| Check | Result |
|---|---|
| `nemotron_h_nvfp4_partial_k_tile_does_not_read_past_the_expert_planes` on the unfixed tree | Failed with exactly 20 non-finite elements at 40 tokens |
| Same test with the fix | Passed |
| `cargo test --profile test-fast --features metal,accelerate -p mlxcel --lib models::nemotron_h::tests:: -- --test-threads=1` | Passed |
| Full serial lib suite with the steel and affine loader fix | 11368 passed, 0 failed, 239 ignored |
| `cargo fmt --all -- --check` | Pass |
| `cargo clippy --profile test-fast --features metal,accelerate -p mlxcel --lib --tests -- -D warnings` | Pass |

### B. Measurement Environment

| Field | Value |
|---|---|
| Host | M3 Ultra, 512 GB unified memory, macOS, Metal |
| NAX device | Absent (the NAX loader is compile-validated only) |
| Fixture | One-layer nvfp4 Nemotron-H MoE, deterministic synthetic weights, `FF = 48` |
| Arms | 5 tokens (unsorted gather), 40 tokens (sorted gather) |
| Reference | f32 dequantization of the same packed planes |
| Bound | `1e-3 + 1e-3 * abs(ref)` (unchanged) |

### C. Evidence Record

| Observation | Value |
|---|---|
| Non-finite location | `got` only (`got 20`, `reference 0`) |
| Non-finite count and total | 20 of 2560 at 40 tokens |
| Non-finite indices | `[63, 95, 319, 351, ... 2399]`, all congruent to 31 modulo 32 |
| Experts and pairs | The last expert, whose 20 pairs each contribute their last output column |
| Weight plane overshoot | 8 bytes past 3072 |
| Scale plane overshoot | 1 byte past 384 |
| Stale scale byte | E4M3 `0x7F`, decodes to NaN |
| Matching x tile | Zero-filled, so the product is `0 * NaN` |
| Falsified alternative | Stale-pool under-write: all-NaN and all-7.0 pools gave bit-identical results |
| Why earlier probes missed it | All-NaN `f32` writes `0x7F` only at offsets congruent to 3 modulo 4; the read is 4-aligned |
| 5-token arm | Uses another kernel, never failed |
| Seeded range | Every 4-byte size in `388..=768`, matching `reuse_from_cache(384)`'s `[384, 768)` window |

### D. References

- `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/fp_quantized.h`
- `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/quantized.h`
- `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/fp_quantized_nax.h`
- `src/lib/mlxcelverse/CMakeLists.txt`
- `src/models/nemotron_h_tests.rs`
- Issue #2852, PR #2866, PR #2867
- Upstream MLX `7ca40033132dae444d45f7fc3daae57e24c13f6c`
