# PR #3360: OpenXLA SSM scan primitive with an HF fp32 oracle and a measured prefill lowering

**Date**: 2026-11-24
**Status**: PR open (closes #3337, epic #564 Phase 1 unit U1). Validated on ROCm gfx1151 with IREE `local-task` and `hip`. Not reachable from `generate`: wiring into a model graph is #3338, shim state buffers and the generate path are #3339 and #3340.

## Executive Summary

The OpenXLA/IREE emitter had no recurrent mixing primitive, so no SSM or hybrid family in epic #564 could be expressed. This PR adds a causal depthwise conv1d (prefill, tail, decode step), a selective scan (one-step decode and whole-prompt prefill) and a full Mamba2 mixer layer in `emitter/ssm.rs`, plus three prefill lowerings of the scan in `emitter/ssm_scan.rs`. One mixer layer is emitted as a standalone probe graph, executed through the IREE auxiliary ABI, and compared against HF `Mamba2Mixer.torch_forward` in fp32: all 11 gate cases pass on `local-task` and `hip` with a worst `y` error of 1.4e-6 against a 5e-5 tolerance.

The prefill decision is chunked SSD with `chunk_size = 128`. On the `state-spaces/mamba2-130m` mixer dimensions it emits 157 StableHLO ops and 31 to 33 dispatches at both 256 and 4096 rows, compiles in under a second on `llvm-cpu` and `rocm`, and needs 242 MiB of transient at 4096. The associative scan is correct but needs 9.2 GiB of transient at 4096; the sequential unroll has 783 dispatches at 256 and does not compile within 1200 s at 4096 on either target.

Review added two gate cases that the original gate could not see (a carried state across four chunks, and a dt clamp), and the security review bounded every `SsmConfig` product with checked arithmetic before the dims start coming from untrusted `config.json` in #3338.

## 1. Problem Statement

Mamba2 and the SSM layers of every hybrid in epic #564 need two operators the emitter did not have: a causal depthwise conv1d over the `xBC` projection, and the selective scan `h_t = exp(dt_t * A) * h_{t-1} + dt_t * (x_t outer B_t)`, `y_t = C_t . h_t + D * x_t`. Each is needed in a one-step decode form and a whole-prompt prefill form.

The decode form is a direct translation. The prefill form is the actual problem:

- The prefill graph is padded to `context_capacity` (default 256, set by `MLXCEL_XLA_CONTEXT_BUCKETS`; the shim's `xla_llama_prefill` requires `lp == context_capacity`), so the scan has to run over padded rows without letting them advance the state.
- The builder has no `while` op, and IREE-CUDA rejected scatter in Stage 2a. The scan must be built from ops that already lower: `dot_general`, `pad`, slices, `dynamic_slice`, `dynamic_update_slice`, `select`, `reduce_window`.
- A recurrence over `T` rows written naively is `T` unrolled steps, which grows the graph linearly in the capacity. The issue listed three candidate forms (chunked SSD, a log-depth associative scan, a full unroll) and required measuring them at 256 and 4096 before committing to one.

The recurrence also has to stay f32 when the graph precision is f16, because the state accumulates over the whole prompt. The existing `dot_general` demotes its inputs to the graph precision, so the scan needed a contraction that does not.

The issue made this a spike plus primitive: prove one layer against HF before anything is wired into a model, and record the lowering decision with numbers.

## 2. Change Summary

| File | Change |
|---|---|
| `emitter/ssm.rs` (new) | `Mamba2MixerW`, `SsmLayerState`; `conv1d_causal_prefill`, `conv1d_tail`, `conv1d_causal_step`; `selective_scan_step`, `selective_scan_prefill`; `mamba2_mixer_prefill`, `mamba2_mixer_prefill_form`, `mamba2_mixer_step`; `ssm_state_types`; stable `softplus`, gated RMSNorm, group-to-head broadcast |
| `emitter/ssm_scan.rs` (new) | `SsmPrefillForm { ChunkedSsd, Associative, Sequential }`, `selective_scan_prefill_form`, `exp_segsum`, and the three scan bodies |
| `emitter/config.rs` | `SsmConfig` next to `MoeConfig` with the issue's fields; `d_inner`, `in_proj_dim`, `validate`; `SSM_DEFAULT_CHUNK_SIZE = 128`; `tiny_mamba2` test constructor |
| `emitter/builder.rs` | `abs`, `log_plus_one`, `dot_general_keep_f32` |
| `emitter/model.rs` | `ProbeMode`, `emit_ssm_probe(c, hidden, t, mode)`, `emit_ssm_probe_with(.., form, precision)` |
| `emitter/mod.rs`, `emitter/ssm_tests.rs` (new) | module wiring, 9 unit tests, the ignored `dump_ssm_probe_for_execution_check` |
| `ssm_probe.rs` (new), `lib.rs`, `cache_dir.rs` | `SsmProbeSpec` (weight layout, I/O shapes, emit), `run_ssm_probe` over the auxiliary IREE ABI, bounds-checked `parse_npy_f32` / `npy_f32_bytes`, 4 tests; `XlaCache::SsmProbe` (`ssm-probe`) |
| `examples/xla_ssm_probe.rs`, `Cargo.toml` | probe runner for `local-task` or `hip`, `required-features = ["xla-iree"]` |
| `spike/openxla/ssm_oracle.py`, `ssm_gate.sh`, `ssm_lowering_measure.py` | HF fp32 oracle (emit inputs, check results), the 11-case gate, the lowering measurements |
| `spike/openxla/FINDINGS_ssm.md`, `README.md` | measurements and decision; x86_64 oracle venv and probe commands |
| `CHANGELOG.md`, `scripts/ci/check_env_var_docs.py` | Added entry; `MLXCEL_SSM_MODE` and `MLXCEL_SSM_T` allowlisted as test-only dump variables |

19 files, +2841 / -5.

### Mixer layer

`mamba2_mixer_prefill_form` follows HF `torch_forward` step by step:

1. `in_projection` produces `zxbcdt` and slices it as `[z: d_inner, xBC: conv_dim, dt: H]`.
2. The carried conv tail `conv0 [K-1, C]` is concatenated in front of `xBC`, the depthwise conv runs over that left context (`stablehlo.convolution` with `feature_group_count = conv_dim`, torch `[C,1,K]` kernel transposed to `[K,1,C]`), and the new tail is `dynamic_slice(xp, real_len)` of `K-1` rows. Using `conv0` as left context is what makes a continued prefill exact and makes `real_len = 0` return `conv0` unchanged.
3. `silu`, then `split_xbc` into `x [T,H,P]`, `B`, `C [T,G,N]`.
4. `discretize_dt`: `softplus(dt + dt_bias)` computed as `max(x,0) + log1p(exp(-|x|))` so it is finite for every input, then the `time_step_limit` clamp (each side emitted only when it is active).
5. Padding mask: `dt = select(iota < real_len, dt, 0)` on `[T, H]`. With `dt = 0` the decay is `exp(0) = 1` and the input term is `0 * x`, so padded rows pass the state through in every scan form. The mask is applied after the clamp, so a nonzero dt floor cannot leak into padded rows.
6. `A = -exp(A_log)`, the scan, the `D` skip, then gated RMSNorm `rmsnorm(y * silu(z)) * w` over groups of `norm_group_size`, and `out_proj`.

`mamba2_mixer_step` is the same pipeline on `[B, ..]` with `conv1d_causal_step` (an f32 multiply-reduce over the window of carried tail plus new row) and `selective_scan_step`. Every step form carries an explicit batch axis so the ragged decode can reuse it.

### Chunked SSD prefill

`chunked_ssd` pads `T` to `nc * Q` rows and computes, with five `dot_general_keep_f32` contractions and no unroll:

1. Intra-chunk: `y_diag = (C B^T o L)(x dt)` per chunk, where `L = exp(segsum(dt A))` is `[nc, H, Q, Q]`.
2. Each chunk's own end state from a zero start: `x dt` decayed to the chunk end, contracted with `B`.
3. Inter-chunk carry: the chunk states with `h0` prepended, `[nc+1, H, P*N]`, contracted against the chunk-level `exp(segsum)` of the per-chunk total decays, `[H, nc+1, nc+1]`. Row `z` is the state entering chunk `z`; the last row is `h_T`.
4. Off-diagonal output: `C` against the state entering each chunk, decayed from the chunk start to each row.

`exp_segsum` follows HF's `segment_sum`: it masks the per-step log decays below the diagonal, runs `reduce_window_prefix_add` along the last axis, and puts `-inf` above the diagonal before the `exp`. It never forms a difference of two prefix sums, so near-diagonal entries carry no cancellation error, and it never exponentiates a positive value.

Group sharing (`G < H`) is a broadcast plus reshape (`heads_from_groups`), matching HF's `repeat_interleave`: head `h` reads group `h / (H / G)`.

### f32 on the f16 path

`selective_scan_prefill_form` casts every scan input to f32 on entry, and every contraction inside the scan goes through `Builder::dot_general_keep_f32`, which asserts f32 operands and emits an f32 result regardless of the builder precision. Only in_proj, out_proj and the prefill conv follow the graph precision.

### Probe and oracle

`emit_ssm_probe` emits `main(mixer weights..., x, [real_len,] h0, conv0) -> (y, h_T, conv_tail)` with no norm or residual around the mixer, in `Mamba2MixerW` field order, optional biases only when the config enables them. `run_ssm_probe` compiles it into the `ssm-probe` cache and runs it through the auxiliary ABI. `ssm_oracle.py` builds one HF `Mamba2Mixer` (transformers 5.12.1, slow path, no `mamba_ssm`) with seeded weights, writes inputs and expected `y`, `h`, conv tail as `.npy`, and checks a result directory by max-abs error. For carried-state prefill cases, which HF prefill cannot start from, the reference is HF's own decode branch run token by token from the same `h0` and tail.

## 3. Technical Decisions

**Chunked SSD over the associative scan and the unroll.** All three forms share the probe interface and the mixer around the scan, so the comparison isolates the scan. Measured on the mamba2-130m mixer dimensions (`hidden 768`, `d_inner 1536`, `H 24`, `P 64`, `N 128`, `G 1`, `K 4`), `real_len = capacity - capacity/8`, with random (not zero) padded rows:

| Option | At 4096 rows | Verdict |
|---|---|---|
| Chunked SSD | 157 ops, 33 dispatches, compile 0.6 to 0.8 s, 242 MiB transient (Q=128) | **Chosen** |
| Associative (Hillis-Steele over `(exp(dt A), dt x outer B)`) | 213 ops, 37 dispatches, 9385.9 MiB transient, 3.3 s / 1.1 s run | Correct, but `[T, H, P, N]` per level does not fit a real model in 31 GiB |
| Sequential unroll | 102476 ops; no compile within 1200 s on either target | Rejected; already 6476 ops and 783 dispatches at 256 |

**The cross-chunk carry is one `dot_general`, not the unrolled sequence the issue suggested.** The issue proposed a short unrolled sequence of `T / Q` chunk-state updates. HF's form instead builds a chunk-level decay matrix and contracts the stacked chunk states against it once. It is the same math, and it keeps the op count independent of `T` (157 ops at both 256 and 4096). The cost is `O((T/Q)^2)` storage and the `reduce_window_prefix_add` work inside `exp_segsum`, which section 6 bounds.

**`chunk_size = 128`.** Q=64, 128 and 256 give the same error and op count. Q=128 has the smallest transient at 4096 (242 MiB, against 315 for Q=64, where the `[H, nc+1, nc+1]` carry and per-chunk states grow, and 362 for Q=256, where the `[nc, H, Q, Q]` decay matrices grow), and the best or near-best run time.

**Keep the recurrence f32 on the f16 path.** With `MLXCEL_XLA_PRECISION=f16`, chunked Q=128 against HF fp32 gives `y` 2.0e-3 / 1.9e-3 at 256 and 2.2e-3 / 2.1e-3 at 4096 (local-task / hip), `h` at most 6.0e-5, conv tail at most 7.5e-4. The `y` error does not grow from 256 to 4096 rows, so the f32 recurrence adds no accumulated drift, and the conv-tail error is the f16 in_proj rounding alone. A demoted-scan variant was not measured: the design calls for an f32 recurrence and nothing in the measurements argues for dropping it. A dedicated `dot_general_keep_f32` was added rather than changing `dot_general`, so the dense graphs and their byte-exact goldens are untouched.

**`conv0` as the left context of the prefill conv.** The issue specified `conv1d_causal_prefill` and `conv1d_tail` with a zero-padded fresh-sequence contract, and those keep it. The mixer prefill instead concatenates the carried tail, so a prefill that continues from a state is exact. This is what the "real_len 77 / 200 from a carried state" gate cases verify.

**Additive contract changes only.** The `SsmConfig` fields and the `ssm.rs` signatures are as written in #3337. The additions are `SsmConfig::{d_inner, in_proj_dim, validate}` (which also requires `conv_kernel >= 2`, since the tail is `K-1` rows), `SSM_DEFAULT_CHUNK_SIZE`, `ssm_state_types`, `mamba2_mixer_prefill_form`, and an extra `hidden` argument on `emit_ssm_probe`, because `SsmConfig` carries no model width.

## 4. Review Outcomes

**Implementation review** (2283ee0b). The chunked SSD was checked against HF `torch_forward` term by term, including `G > 1`: discretization, the segsum built from non-positive sums only, the group mapping, the `D` skip, gated norm order, and the chunk-level carry. No defect in the math, but mutation testing found two gate gaps:

- No case carried a nonzero state across more than two chunks. With a zero `h0` and at most two chunks, the state entering every chunk needs no decay, so zeroing the chunk-level decay matrix left `y` within 1e-6 on every prefill case and was caught only by `h_T`. The new 4-chunk carried case (chunk 64, real_len 200) fails that mutation with `y` off by 1.05.
- HF's default `time_step_limit` is `(0, inf)`, so the clamp was never executed. `--time-step-limit` was added to the oracle with a prefill and a decode clamp case; removing the clamp fails them by 1.7e-2 / 1.9e-2 in `y`.

Wrong group mapping and an off-by-one segsum mask already failed the original gate.

**Security and performance review** (cf1bfc0b). No CRITICAL or HIGH. The IREE shim boundary already rechecked every buffer length against its shape with checked arithmetic.

| Finding | Severity | Status |
|---|---|---|
| `SsmConfig::validate` used unchecked products; in release a wrapped `H*P` could satisfy the `conv_dim` equation | MEDIUM | Fixed: `d_inner`, `conv_dim`, `in_proj_dim`, `H*P*N`, `(K-1)*conv_dim` checked; `+inf` dt floor, `-inf` ceiling, NaN limits, negative or non-finite `norm_eps` rejected |
| Zero rows reached `t.div_ceil(0)` in the chunked scan | LOW | Fixed: the probe spec rejects zero rows, zero hidden and rows past i32; the emitter asserts |
| The example's `.npy` reader could panic on a truncated header and ignored the declared shape | LOW | Fixed: `parse_npy_f32` in the library with bounds-checked parsing and exact data length; each input's shape must match the probe's |
| Fixed `/tmp/ssm*` paths in the README and measurement script | LOW | Fixed: `mktemp -d` |
| Long-context cost of `exp_segsum` and the carry | MEDIUM (performance) | Deferred to #3338, see section 6 |

**Remaining LOW, recorded in FINDINGS.** On the f16 path the prefill conv uses `Builder::convolution`, so its result is f16 before the f32 cast, while the decode conv is an f32 multiply-reduce. `convolution_f32_accumulate` (f16 operands, f32 result) does not compile: iree-compile rejects the depthwise mixed-type convolution with a `tensor.collapse_shape` type mismatch. If #3338's f16 graphs need the gap closed, the fix is a K-tap shifted multiply-add in f32.

## 5. Tests and Verification

New unit tests: 9 in `emitter/ssm_tests.rs` (config validation, overflow and non-finite rejection, zero-row refusal, prefill and decode probe signatures, every prefill form emitting the same interface, ragged and short capacities, scan contractions staying f32 on the f16 path, conv tail and step shapes) and 4 in `ssm_probe.rs`. `cargo test --profile test-fast -p mlxcel-xla --lib`: 479 passed on the final head, with the dense goldens unchanged.

Gate (`spike/openxla/ssm_gate.sh`), capacity 256, a tiny mixer with `G = 2 < H = 4` so the group broadcast is exercised. Max-abs error against HF fp32, tolerance 5e-5:

| case | y local-task | y hip | h (worst) | conv tail (worst) |
|------|-------------:|------:|----------:|------------------:|
| decode, B=1 | 3.0e-7 | 4.8e-7 | 3.0e-8 | 3.0e-7 |
| decode, B=3, linear biases | 4.8e-7 | 4.8e-7 | 1.2e-7 | 2.4e-7 |
| prefill, real_len 100 < 256 | 7.2e-7 | 7.2e-7 | 3.3e-9 | 1.8e-7 |
| prefill, real_len = 256 | 7.2e-7 | 8.3e-7 | 4.2e-9 | 1.8e-7 |
| prefill, real_len 2 < K-1 | 3.0e-7 | 6.0e-7 | 4.1e-10 | 3.0e-7 |
| prefill, real_len 0 | 0 | 0 | 0 | 0 |
| prefill, real_len 0 from a carried state | 0 | 0 | 0 | 0 |
| prefill, real_len 77 from a carried state, biases | 9.5e-7 | 8.3e-7 | 1.2e-6 | 3.6e-7 |
| prefill, real_len 200 from a carried state, chunk 64 (4 chunks) | 1.3e-6 | 1.1e-6 | 6.9e-7 | 3.6e-7 |
| prefill, real_len 100, dt clamped to [0.01, 0.05] | 6.3e-7 | 7.7e-7 | 2.9e-8 | 1.8e-7 |
| decode, B=2, dt clamped to [0.01, 0.05] | 4.8e-7 | 6.0e-7 | 6.0e-8 | 2.4e-7 |

The table is from `FINDINGS_ssm.md`; the PR's test plan reports a worst `y` of 1.4e-6 on the final head. Each case added in review fails under its targeted mutation.

Lowering measurements (`ssm_lowering_measure.py`), mamba2-130m mixer dimensions, gfx1151, toolchain `iree-rocm-3.12.0rc20260721`. Compile columns are `seconds / peak RSS MiB`; error columns are max-abs `y`; `h` and conv-tail errors are below 3e-7 and 2.2e-6 in every measured row. Run time is one invocation, first call included, no warmup or repeats.

context_capacity 256:

| form | ops | dispatches | llvm-cpu compile | rocm compile | transient MiB | err local-task | err hip | run local-task / hip (s) |
|------|----:|-----------:|-----------------:|-------------:|--------------:|---------------:|--------:|-------------------------:|
| chunked Q=64 | 157 | 33 | 0.6 / 174 | 0.8 / 210 | 21.8 | 5.2e-6 | 6.0e-6 | 0.012 / 0.008 |
| chunked Q=128 | 157 | 33 | 0.6 / 174 | 0.8 / 211 | 15.9 | 5.2e-6 | 6.0e-6 | 0.012 / 0.009 |
| chunked Q=256 | 157 | 31 | 0.6 / 175 | 0.8 / 214 | 26.4 | 5.2e-6 | 6.0e-6 | 0.015 / 0.009 |
| associative | 173 | 29 | 0.6 / 167 | 0.6 / 201 | 587.4 | 5.0e-6 | 6.0e-6 | 0.126 / 0.053 |
| sequential | 6476 | 783 | 11.8 / 248 | 12.4 / 307 | 11.6 | 5.2e-6 | 6.0e-6 | 0.026 / 0.010 |

context_capacity 4096:

| form | ops | dispatches | llvm-cpu compile | rocm compile | transient MiB | err local-task | err hip | run local-task / hip (s) |
|------|----:|-----------:|-----------------:|-------------:|--------------:|---------------:|--------:|-------------------------:|
| chunked Q=64 | 157 | 33 | 0.8 / 176 | 0.8 / 208 | 315.1 | 6.7e-6 | 6.0e-6 | 0.129 / 0.085 |
| chunked Q=128 | 157 | 33 | 0.6 / 174 | 0.8 / 209 | 241.9 | 6.7e-6 | 6.0e-6 | 0.125 / 0.088 |
| chunked Q=256 | 157 | 33 | 0.6 / 173 | 0.8 / 212 | 362.3 | 6.7e-6 | 6.0e-6 | 0.151 / 0.097 |
| associative | 213 | 37 | 0.6 / 167 | 0.6 / 201 | 9385.9 | 6.7e-6 | 6.0e-6 | 3.292 / 1.146 |
| sequential | 102476 | n/a | timeout > 1200 s / ~0.8 GiB at kill | timeout > 1200 s / ~0.8 GiB at kill | n/a | not run | not run | n/a |

The HF reference for these runs uses `chunk_size=32` so the 4096-row reference fits in memory; the HF chunk size does not change the math.

Gates: `cargo fmt --all -- --check`; clippy `-D warnings` on mlxcel-xla (`--all-targets`, and lib plus tests with `--features iree`) and on the `xla_ssm_probe` example.

## 6. Pending Work and Limitations

- **Not integrated.** No model graph calls the mixer yet. `config.json` parsing, weight specs and model graphs are #3338; shim state buffers and the generate path are #3339 and #3340. The CHANGELOG entry says so.
- **Long-context cost of chunked SSD.** `reduce_window_prefix_add` lowers as a full-width window, so `exp_segsum` costs `O(L^2)` per row. Intra-chunk that is `T*H*Q^2` adds, about 2x the `y_diag` contraction at Q=128, P=64. The inter-chunk carry is `H*(T/Q)^3` adds on an `[H, nc+1, nc+1]` tensor: negligible at 4096 (nc = 32), but it overtakes the intra-chunk work near T = 185k and needs about 6 GiB per intermediate at T = 1M. If #3338 targets long contexts, the carry needs a bounded form (a prefix sum over chunk totals, or a scan over chunks), measured against HF on hip.
- **Prefill and decode conv rounding differ on the f16 path** (section 4). Fix if needed: K-tap f32 multiply-add instead of the convolution op.
- **Input bounds for #3338.** Cross-check every config dim against the checkpoint tensor shapes, the real upper bound on them, and clamp `real_len` to `[0, capacity]` on the host.
- **Only `local-task` and `hip` (gfx1151) were run.** CUDA was not measured.
- **The oracle is HF's slow path,** not `mamba_ssm`. Real CUDA kernels may differ from it by more than 5e-5; the gate is defined against the slow path.
- **Run times are indicative only**: one invocation, first call included.

## 7. Learning Points

- A scan that cannot use `while` can still avoid an unroll: Mamba2's state-space duality rewrites the recurrence as masked matrix products inside chunks plus one matrix product across chunks. The graph size then depends on shapes only, not on the capacity, which is what kept 4096 rows at 157 ops.
- Build decay matrices as masked prefix sums with `-inf` above the diagonal (HF's `segment_sum`), not as `exp(cumsum_i - cumsum_j)`. The difference form cancels catastrophically near the diagonal and can exponentiate large positive values.
- Padding through a recurrence is cheapest at the discretization step: `dt = 0` makes the decay 1 and the input 0, so every scan form passes the state through without its own mask. Apply the mask after any clamp with a nonzero floor.
- A gate that only checks outputs can miss a whole term. A zero initial state and two chunks made the chunk-level decay invisible in `y`, and HF's default `(0, inf)` limit made the clamp dead code. Mutation testing each term against the gate found both; each review case now fails under its mutation.
- When a reference cannot express a case (HF prefill always starts from zero), its own one-step path run token by token is a valid reference for the continued case.
- Configuration dims that will come from untrusted files should be validated with checked arithmetic before they are first used, because a wrapped product can satisfy a consistency equation that was meant to reject them.
