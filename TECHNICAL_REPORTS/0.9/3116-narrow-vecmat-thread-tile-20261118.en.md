# Technical Report: PR #3116 - give the narrow decode vecmats one output per thread

**Date**: 2026-11-18
**Status**: Completed
**Languages**: Rust (MLIR emitter and tests), MLIR (tuning spec), Bash (measurement)
**Risk Level**: Medium

---

## Executive Summary

The default f16-resident decode graph on `gfx1151` ran its per-layer projections at about 42 GB/s while the 128256-wide LM head reached 153 GB/s in the same trace. The compiler lowered every rank-1 vecmat with `thread = [8, 0]` and `workgroup = [256, 0]`, so the projections launched N/256 one-warp workgroups: 8 for the 2048-wide q, o and down and 32 for gate/up. That is under one wave on a 20-WGP device, and the serial K reduction cannot hide memory latency. The ROCm tuning spec now gives every single-row f16-by-f16-to-f32 contraction with `128 <= N <= 8192` one output per thread and `reduction = [0, 32]`, whatever K is. A paired six-round A/B measures the decode step at 41.05 to 73.98 tok/s (+80.25%, won in 6 of 6 rounds, `text EQUAL`). The wide tile stays for `N >= 8193` with `K >= 4096`, so the LM head keeps the config it had.

---

## 1. Problem Statement

### 1.1 Background

Issue #2873 asked for the remaining decode-kernel efficiency gap on the OpenXLA/IREE ROCm backend. The argmax half of that issue was already fixed by #2875, a two-stage reduce that measures 0.030 ms per step. What remained on the default f16 path, the one used when `MLXCEL_XLA_QUANT` is unset, was the small-M vecmat lowering: every per-layer projection sat far below the bandwidth the LM head reached with the same compiler and the same trace.

The protocol is the one the other breakdowns use: `rocprofv3 1.3.5 --kernel-trace -f csv`, llama-3.2-1b-instruct-4bit, greedy, capacity 256, two runs (`n=1` and `n=128`) differenced per kernel and divided by 127 so the shared prefill cancels. Before the change (main at `b21825cf`) the untraced step was 24.06 ms (41.6 tok/s), 279 dispatches per step, and 24.44 ms of trace kernel time.

### 1.2 Existing Issues

- **Issue 1**: the compiler's default rank-1 vecmat tile emits one warp per workgroup, so a 2048-wide projection launches 8 workgroups and gate/up launches 32, less than one wave on 20 WGP.
- **Issue 2**: the serial K reduction inside each thread cannot cover memory latency when the grid is that small, so q/o measure 42 GB/s where the same-run head measures 153 GB/s.
- **Issue 3**: the previous tuning spec matched long-K single-row contractions (`K >= 4096`) and raised only the reduction chunk, so it could not reach q, o or gate/up on a model whose hidden size is 2048.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A matcher still parses but stops matching | High (compiles on the default tile, only throughput changes) | Handled by the compiler-backed test added in `838c19cf` |
| The tile reaches the LM head and grows its activation re-read | Medium (regression on the largest kernel) | Low: the N bound is `umax = 8192` and disjoint from the wide matcher |
| The tile reaches the ragged multi-row graph | Medium (a config measured for one row applied to N rows) | Low: the batch predicate requires empty batch dims |
| The reduction chunk change reorders the sum | High (greedy output changes) | Low: the generated assembly accumulates serially over ascending K for both tiles; `text EQUAL` in every paired run |

---

## 2. Technical Review

### 2.1 Security

No new input, network or file surface. The spec is a static asset applied only when `device == "hip"` and the runtime config is `RuntimeConfig::Dense(_)`. A malformed spec fails the compile loudly; there is no silent fallback. The security review found no security issue.

**Checklist:**
- [x] Input validation (unchanged; no model, request or environment input reaches the matcher)
- [x] Authentication/Authorization (not applicable)
- [x] Data encryption (not applicable)
- [x] Logging (no model data)

**Issues Found:** none.

### 2.2 Performance

`gfx1151`, llama-3.2-1b-instruct-4bit, capacity 256, `MLXCEL_XLA_QUANT` unset, `rocprofv3` subtraction:

| Measurement | Before | After |
|------|--------|-------|
| Decode step (untraced) | 24.06 ms (41.6 tok/s) | 13.54 ms (73.5 to 74.5 tok/s) |
| Trace kernel time | 24.44 ms | 12.48 ms |
| Dispatches per step | 279 | 279 |
| `vecmat_2048x2048` (q, o) | 42 GB/s | 219 GB/s |
| `vecmat_512x2048` (k, v) | 44 GB/s | 81 GB/s useful (162 counting the re-read) |
| `vecmat_8192x2048` (gate, up) | 140 GB/s | 196 GB/s |
| `vecmat_2048x8192` (down) | 104 GB/s | 230 GB/s |
| `vecmat_128256x2048` (LM head, unchanged config) | 153 GB/s | 228 GB/s |
| argmax (two-stage reduce) | 0.031 ms | 0.030 ms |

The head's 153 to 228 GB/s difference between the two runs is the clock state of the traces, not the change: its config is excluded by the N bound. Read through the head as the same-run control, q/o move from 0.27 to 0.96 of the head's bandwidth and down from 0.68 to 1.01. Gate/up move from 0.92 to 0.86, a change smaller than the 1.49x the unchanged head records between the traces, so their absolute 140 to 196 GB/s is not separable from the clock shift and is reported as such.

The reportable effect size is the paired interleaved A/B (`scripts/xla/ab_decode_pair.sh`, same model, capacity and token budget, 6 rounds): 41.05 to 73.98 tok/s, +80.25%, won in 6 of 6 rounds with `text EQUAL`.

On the untied qwen2.5-coder-7b-instruct-4bit (hidden 3584, intermediate 18944, 28 layers) the untraced decode runs at 15.44 to 15.46 tok/s against 12.28 before, with greedy output unchanged over 48 tokens. A first version with a separate short-K matcher capped at K = 2048 reached only the down projection there and measured 12.94 tok/s; q and o have K = 3584 and stayed on the compiler default.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. The tuning spec is not part of the model ABI, no dispatch is added or removed, the weight layout is untouched, and the reduction order is unchanged.
- **New Dependencies**: none.
- **Compatibility**: the spec applies to the `hip` dense graphs only, as before. The ragged and uniform-B graphs keep the compiler default through the empty-batch predicate, so their committed goldens are unchanged. Prefill contractions are not rank-1 and cannot match.

### 2.4 Code Quality

- **Test Coverage**: a textual test pins the entrypoint wiring, the narrow tile, the output-width bounds and the absence of a reduction bound; a second pins the wide fallback's two predicates and their disjointness. An `#[ignore]`d compiler-backed test emits the single-sequence decode graph, compiles it with the pinned spec, and reads the selected configs out of the executable dump. It is ignored because it needs the version-matched ROCm `iree-compile`; the security review confirmed it passes normally and fails when `umax` is narrowed to 64.
- **Code Complexity**: one matcher replaces the old one, and a second keeps the long-K behavior. The two predicates are disjoint, so their order in `transform.foreach_match` cannot change the result.
- **Technical Debt**: decreased. The spec's comments now state the measured mechanism and the exact predicates instead of a partial description of the reduction-only fix.

---

## 3. Technical Decisions

### 3.1 One output per thread, not a larger reduction chunk

**Context:** the projections were limited by grid size (8 to 32 one-warp workgroups), while the reduction chunk only changes the work inside a thread.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Raise the reduction chunk again | No change to the output mapping | Does not add workgroups; the grid stays under one wave |
| **Chosen: `thread = [1, 0]`, `workgroup = [32, 0]`, `reduction = [0, 32]`** | N/32 warps instead of N/256 fills the device | One output per thread re-reads the activation row per output |
| Split K across workgroups | More parallelism over K | Not expressible: `SetSplitReductionSizes` needs a matmul-like op with two parallel loops, and a rank-1 vecmat has one |

**Rationale:** occupancy, not the reduction tile, was the binding constraint. q/o and down reach the head's same-run bandwidth once the launch fills the device.

**Trade-offs:** each thread re-reads the activation row. That cost grows with N, which is why the match stops at 8192 and leaves the LM head on the previous config.

### 3.2 Bound N, not K

**Context:** the tile's occupancy is set by the output count. Every output still walks K serially, so the tile does not depend on the reduction length.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep a `K >= 4096` gate like the previous matcher | Conservative | Leaves q and o on a 3584-hidden model on the default tile, which cost 0.66 tok/s in the first version |
| **Chosen: any K with `128 <= N <= 8192`** | Covers hidden 2048 and 3584 with one matcher | Requires the N bound to keep the head out |

**Rationale:** the K gate did no work once both long-K and short-K matchers carried the same tile. The N floor of 128 keeps very narrow projections on the default; the N ceiling of 8192 excludes the LM head, where the activation re-read grows with N.

### 3.3 Keep the wide long-K config behind a disjoint predicate

**Context:** the previous `thread = [8, 0]`, `workgroup = [256, 0]` config is still the right one for a wide long-K head.

**Rationale:** `N >= 8193` starts exactly where the narrow matcher stops (IREE's `umax` is closed), and `K >= 4096` keeps the wide tile off llama's K = 2048 head, which is deliberately left on the compiler default. Disjoint bands make the entrypoint order irrelevant.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/lib/mlxcel-xla/assets/rocm/bigk_vecmat_tuning.mlir`**
```mlir
transform.named_sequence @match_single_row_vecmat(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %arg0,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>]
    : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %batch_dims, [] : !transform.param<i64>
  %rhs = transform.get_operand %arg0[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.dim_bounds %rhs[1], umin = 128, umax = 8192 : !transform.any_value
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{reduction = [0, 32], thread = [1, 0], workgroup = [32, 0]}>, translation_info = <pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [32, 1, 1] subgroup_size = 32>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
```
**Reason for change:** the tile raises the launch from N/256 to N/32 warps while keeping the contraction's output mapping one-to-one.

**File: `src/lib/mlxcel-xla/assets/rocm/bigk_vecmat_tuning.mlir`, wide fallback**
```mlir
transform.iree.match.dim_bounds %rhs[0], umin = 4096, none : !transform.any_value
transform.iree.match.dim_bounds %rhs[1], umin = 8193, none : !transform.any_value
```
**Reason for change:** the added N lower bound makes the two vecmat bands disjoint, so the wide tile cannot reach a narrow output and the narrow tile cannot reach the head.

**Files: `src/lib/mlxcel-xla/src/tuning.rs`, `src/lib/mlxcel-xla/src/iree.rs`**
The asset's doc comment, the `ROCM_BIGK_VECMAT_TUNING` doc comment and the compile-site comment now describe the measured mechanism and the new bands, and the tests pin both matchers.

### 4.2 Data Model Changes

None. The spec changes compiler inputs, not tensors or the runtime ABI.

---

## 5. Learning Points

### 5.1 Occupancy is the first tile decision for a narrow contraction

**Concept:** for a rank-1 contraction, the output count sets the grid. A tile that gives one warp per workgroup leaves the device partly idle until N is large enough to fill it, and no amount of reduction tiling inside a thread changes the grid.

**Application in this PR:** q/o and down moved from 42 and 104 GB/s to 219 and 230 once the launch used every output as a workgroup's lane. The 128256-wide head was already at that bandwidth with the old tile because its grid was already large.

### 5.2 Separate traces need a same-run control

**Concept:** two traces taken at different times also differ in clock state. A kernel whose config did not change is the control that separates the change from the clock.

**Application in this PR:** the unchanged head measures 153 GB/s in one trace and 228 in the other. Reporting the per-kernel columns through that control is why gate/up is described as not separable from the clock shift rather than as a 40 percent gain, and why the paired A/B is the effect size the report asserts.

### 5.3 A textual test cannot see a matcher that stops matching

**Concept:** a tuning spec can parse, apply, and match nothing. The compile succeeds and only throughput changes, so no unit test that reads the asset as text can detect it.

**Application in this PR:** the compiler-backed test emits the real decode graph, compiles it with the spec, and asserts the dumped configurations. It costs a compiler invocation, so it is `#[ignore]`d and documented for the edit that needs it.

---

## 6. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 6 code and docs (the two report files not counted) |
| Lines added | +308 |
| Lines deleted | -29 |
| Tests added | 2 (one compiler-backed and `#[ignore]`d) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Performance | 2 | Narrow single-row tile, wide long-K fallback with an explicit N bound |
| Tests | 2 | Narrow and wide predicate test, executable-configuration observation |
| Documentation | 3 | Crate README breakdown, tuning-spec and compile-site comments, environment-variable baseline |
| Compatibility | 1 | Ragged graph and prefill excluded by predicates; no ABI change |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `051ffa9c` | perf | give the narrow decode vecmats one output per thread |
| `8beb4208` | test | pin the wide vecmat predicates and correct the tuning notes |
| `1a2814fa` | docs | name the authoritative figure for the #2873 breakdown |
| `838c19cf` | test | observe the vecmat tuning spec on the emitted decode graph |

---

## 7. Follow-up Actions

### Required
- [ ] None.

### Monitoring Required
- Long-context decode on `hip` with the default f16 path: q/o and down should stay near the LM head's bandwidth, and gate/up should not fall below its prior share.

### Future Improvements
- k and v remain at 81 GB/s useful because N = 512 caps one output per thread at 16 warps. Raising that needs a concatenated k/v upload-time weight layout; the 0.8 ms per step bounds the headroom at roughly 0.5 ms.
- A workgroup-level K split is not expressible in the tuning spec for a rank-1 contraction; only a graph or operator change makes it possible.

---

## Appendix

### A. Test Results

- `cargo test -p mlxcel-xla --lib tuning:: -- --include-ignored`: 5 passed (4 textual, 1 compiler-backed).
- `cargo test -p mlxcel-xla --lib emitter::`: 216 passed.
- `cargo fmt -p mlxcel-xla -- --check`: clean.
- `cargo clippy -p mlxcel-xla --all-targets -- -D warnings`: clean.
- `scripts/xla/validate_arch.sh` on `hip`: PASS (structural, token-exact, serve `REFERENCE-EXACT` 16/16).

### B. Performance Benchmarks

Paired interleaved A/B through `scripts/xla/ab_decode_pair.sh`, `gfx1151`, llama-3.2-1b-instruct-4bit, capacity 256, 6 rounds.

| Arm | Median (tok/s) | Delta | Rounds won | Output |
|-----|----------------|-------|------------|--------|
| before | 41.05 | | | `text EQUAL` |
| after | 73.98 | +80.25% | 6/6 | `text EQUAL` |

Untied checkpoint, untraced single runs: qwen2.5-coder-7b-instruct-4bit 12.28 to 15.44/15.46 tok/s (+26%), greedy output unchanged over 48 tokens.

### C. References
- Issue #2873 (decode-kernel efficiency), #2875 (two-stage argmax), #2921 and #2956 (grouped-query decode matchers).
- PR #3106 (native-path decode breakdown, and the measurement protocol this report reuses).
- `src/lib/mlxcel-xla/README.md`, section "Decode step breakdown (the default f16 path, issue #2873)".
