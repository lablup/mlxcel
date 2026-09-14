# Metal kernel-selection gate for issue #1803: M1 Ultra, 2026-09-13

Regression gate for issue #1803, which replaced the `!metal::is_available()` idiom with an explicit `GpuKernelBackend` at eleven sites. It answers one question: on Metal, does mlxcel still select exactly the kernels it selected before the change?

The answer is yes. Every arm that reaches a changed decision point is byte-identical or counter-identical, the workspace gate is green, and the single predicate the change rests on resolves the way the change assumes.

This closes the Metal half of #1803's last acceptance criterion. The CUDA half is open as #1873, which should be run with the same arms and the same traps; `scripts/paged_decode_counter_ab.sh` in this tree is the harness for its arm B.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra (Mac13,2), 128 GB unified memory |
| OS | macOS 27.0 (26A428) |
| Backend | Metal, `--features metal,accelerate` |
| Baseline | `8bf7ffb0` (#1861, FFT through hipFFT) |
| Arm | `dda6a7e8` (#1803, route custom kernels by GPU backend kind) |
| MLX pin | `81ba1c6a` |
| Checkpoints | `mlx-community` 4-bit, read from a shared local model directory |
| Machine state | uptime 3 days, background load ~3.8 throughout |

Every verdict below is an equality or a counter comparison, both insensitive to machine load. Nothing here is a performance claim, and none was attempted, because the box was not quiet.

## Why the pair is `8bf7ffb0` → `dda6a7e8`

`dda6a7e8`'s parent is exactly `8bf7ffb0`, so the pair isolates #1803. The nearer-looking `9ead2b7d` would have mixed #1861 (FFT) into the difference, because the #1803 branch was stacked on the FFT branch rather than on `main`.

The branch was later rebased and merged as `d492c81b` (PR #1877, after #1869 auto-closed on base-branch deletion). That does not invalidate these numbers:

```
git diff 8bf7ffb0 dda6a7e8   963 lines
git diff 4641dc7a d492c81b   963 lines   ->  byte-identical
```

The base also drifted, because the WebUI epic #1834 landed in between (120 files, 19267 insertions). The intersection of those 120 files with the 27 files #1803 touches is empty, so measuring on the older base gives the same answer as measuring on the newer one.

## The change, restated

#1803 is an idiom swap at eleven sites. `!mlx::core::metal::is_available()`, read as "CUDA", becomes `gpu_kernel_backend() == Cuda`; `metal::is_available() || cu::is_available()` becomes `custom_kernels_available()`. It also adds that predicate as a *new* gate on five production decisions: `fused_moe_enabled()`, `split_kv_enabled()`, `paged_decode_backend()`, the head of `paged_decode_batched()`, and BitNet checkpoint load.

On Metal the two idioms are equal by construction. `resolve_gpu_kernel_backend()` asks `metal::is_available()` first, so a Metal host resolves to `Metal`, which makes `custom_kernels_available()` true and `use_cuda` false at every swapped site, which is what the boolean did. The only way to break that is `metal::is_available()` being false on a Metal host, and there the *old* code set `use_cuda = true` and threw "No Metal back-end". The arm is therefore never different and strictly safer on Metal.

Everything below tests that construction rather than trusting it.

## Arm 0: the predicate

#1803 ships its own probe. `MLXCEL_DEBUG_KERNEL_BACKEND=1` makes `gpu_kernel_backend()` print the resolved backend once.

```
arm:      [mlxcel] custom kernel backend: metal      (no "no custom kernel ports" suffix)
baseline: nothing, because the probe does not exist there
```

All five new gates therefore answer the way the pre-#1803 code behaved.

The asymmetry does a second job: this line can appear only in the arm, so an A/B that shows it cannot secretly be the same binary twice. Arm A carries the probe for exactly that reason.

## Arm A: greedy byte-identity through `generate`

`scripts/ab_output_equality.sh` at `56d82fbc`, which forces `--temp 0` and `--show-reasoning` on both sides and runs the baseline twice as a self-reproduction control. 64 tokens, prompt *"Explain mixture-of-experts routing in two sentences."*

| configuration | changed site it reaches | result |
|---|---|---|
| `Qwen3-30B-A3B-4bit` | `fused_moe_enabled()` gate, `run_fused_moe_two_kernel` | EQUAL |
| `bitnet-b1.58-2b-4t` | `bitlinear_matmul`, load refusal | EQUAL |
| `bitnet-b1.58-2b-4t-4bit` | same | EQUAL |
| `Meta-Llama-3.1-8B-Instruct-4bit`, `MLXCEL_FUSED_ADD_RMSNORM=1 MLXCEL_FUSED_ROPE_APPEND=1` | `fused_add_rms_norm`, `fused_rope_qk_append` | EQUAL |
| `falcon-h1-tiny-90m-instruct-4bit` | `ssm_update_kernel` | EQUAL |
| `deepseek-v2-lite-chat-4bit-mlx`, `MLXCEL_MLA_ABSORBED=1 MLXCEL_MLA_SPLIT_KV=1` | `split_kv_enabled()` gate | EQUAL |

The backend probe fired on the arm run of every row and on no baseline run.

### The first pass of this table was wrong and looked right

It ran five checkpoints on default settings and reported 5/5 EQUAL. Three of those rows never touched #1803 at all, and the probe was silent for them. The causes are worth naming because each is easy to walk into again:

- `FUSED_ADD_RMSNORM_DEFAULT` and `FUSED_ROPE_APPEND_DEFAULT` are both `false`. Llama short-circuited in `fused_add_rms_norm` before reaching the availability check, so no fused-norm site was exercised.
- `falcon-mamba-7b-4bit-instruct` is `model_type: falcon_mamba` and does not use `ssm_update_kernel`. The families that do are `falcon_h1`, `granitemoehybrid`, `nemotron_h` and `plamo2`.
- `split_kv_enabled()` short-circuits on `absorbed_enabled()`, so `MLXCEL_MLA_SPLIT_KV=1` on its own never reached the gate; `MLXCEL_MLA_ABSORBED=1` is also required.

An EQUAL row whose probe stayed silent proves that two binaries agree on a path neither of them changed. Carrying the probe inside the A/B is what separated the six real rows from the three empty ones.

## Arm B: the batched paged decode path

`mlxcel generate` never reaches the paged bridges. The production route is

```
models/llama3.rs -> cache::paged_batch_decode_attention -> PagedBlockPool::paged_decode_batched
  -> launch_v2 / launch_cascade -> paged_attention_decode_v2_partial, paged_attention_merge_states
```

and #1803 adds a `custom_kernels_available()` early return at the head of `paged_decode_batched`. Reaching it needs a paged-backed cache, which `--decode-storage-backend paged` selects. It is **not** `--kv-unified`: that flag is read by the KV-budget code and never by the batch scheduler, whose choice is made in `effective_decode_storage_backend` (`src/server/batch/scheduler/mod.rs`).

Byte-identity cannot judge this arm. If the new early return fired, the server would answer from the gather fallback: same text, dead fused path. The verdict is the counters the server already exports, for precisely this reason. From the `paged_decode_batched` doc comment, on why the outcome is a value and not a log line:

> the first cut of this used `tracing::debug!` and a whole production benchmark sweep then compared gather against gather without anything saying so

`Meta-Llama-3.1-8B-Instruct-4bit`, `--decode-storage-backend paged --max-batch-size 4 --metrics`, `MLXCEL_PAGED_ATTENTION_NATIVE=1`, four identical requests at temperature 0:

```
                                              baseline    arm
mlxcel_paged_decode_launches_total{fused_v2}      8352   8352
mlxcel_paged_decode_launches_total{gather}           0      0
mlxcel_paged_decode_launches_total{cascade}          0      0
mlxcel_cascade_failures_total                        0      0
generated text                                        EQUAL
```

Both arms took the fused path the same number of times and neither fell back.

### The discriminator is `gather_fallbacks`

Not `declines`, and not the log. #1803's early return sits inside `paged_decode_batched` and returns `Ok((None, NotServable))`, which the caller's `None` arm folds into `gather_fallbacks` (`cache/paged_batch_decode.rs`). `declines` counts only rejections made before the pool is touched, and it is not published to the gauges at all: `observability.rs` exports `v2_launches`, `gather_fallbacks` and the four cascade counters. Had the early return fired on the arm, the table would read `fused_v2` 0 and `gather` 8352.

The `report_once` line is corroboration, not the verdict. It prints once per outcome *kind*, and `NotServable` is one kind covering every reason, so a batched prefill rejected as "not a single-token decode step" can take that slot and a later kernel-port rejection would never print.

### The first pass of this arm was also wrong and also looked right

Without `MLXCEL_PAGED_ATTENTION_NATIVE=1`, both sides declined with

```
paged decode v2: gather: 1 visible KV tokens across 1 request(s) is below the 4096-token dispatch floor
```

and the run compared gather against gather, which is the exact failure the doc comment above describes. The EQUAL text did not catch it; the outcome line did. `resolve_paged_v2_dispatch` treats the env override as a force, bypassing the measured floor, and a forced dispatch still passes the kernel's own structural declines, so it cannot manufacture a launch the kernel would reject.

A second trap surfaced in the same run: `/metrics` is off unless `--metrics` is passed, and `curl -f` had been failing silently, so the first pass collected no counter file at all and the report template printed empty rows.

`mlxcel_paged_decode_launches_total` counts *layer decode steps*, not tokens. It read 8352 for both a 10-token and a 6300-token prompt, because the request count, the generated tokens and the layer count are what drive it. A reader expecting it to scale with context will misread a correct run as a broken one.

## Arm C: the workspace gate

```
cargo test --workspace --profile test-fast --features metal,accelerate
11119 passed, 0 failed, 361 ignored, 123 suites, 9m42s, exit 0
```

`--workspace` matters: without it three member crates including `mlxcel-core` drop out (#1007).

Tests that reach the bridges #1803 converted to `Result`, all passing:

- `mla::split_kv::split_kv_tests::merge_rejects_natural_log_lse_units`, the call site #1803 rewrote to `.expect("merge_states")`
- `mla::split_kv::split_kv_tests::split_kv_matches_the_decompressed_reference_at_every_chunk_length`
- `paged_v2::cascade_launch::cascade_launch_tests::merge_rejects_natural_log_lse_units_on_the_cascade_path`
- `ffi_tests::test_fused_paged_decode_matches_gather_over_200_steps`, the v1 `paged_attention_decode`
- `ffi_tests::test_fused_paged_decode_native_vs_fallback_matrix`
- `ffi_tests::test_fused_paged_decode_gqa_and_batched`
- `models::bitnet::tests::bitlinear_matmul_known_ternary_case`

`paged_attention_merge_states` is covered here rather than in arm B. Arm B's fused launches reported `merge skipped`, and `report_once` prints one line per outcome kind, so a later multi-chunk launch would not re-report; raising the prompt to 6300 tokens did not make the merge visible in the log.

The sampler is covered here too. Fourteen `sampling_gumbel_tests` pass, including `fused_sample_routes_the_no_filter_path_to_the_gumbel_kernel`, which proves the routing actually enters the kernel, and several that pin its output against a softmax reference.

### The BitNet test needed separate handling

#1803 gives `bitlinear_matmul_known_ternary_case` a skip branch that prints a reason and returns when the backend has no BitLinear port, and `cargo test` swallows `eprintln!` from a passing test. The absence of a skip notice in the gate log therefore proves nothing, which is the `0.00s` trap from #1722. Three independent lines settle it:

- the skip branch is `if !custom_kernels_available()`, and arm 0 showed that predicate true here, so the branch is unreachable
- re-running the test alone with `--nocapture` printed no skip notice
- it elapsed 0.07s rather than 0.00s

## Did the arm actually contain the change

PR #1867 made the A/B gate refuse two byte-identical binaries, which catches a `cargo build` that no-ops entirely. It cannot catch the partial no-op: Rust recompiles, stale C++ objects are reused, the binaries differ in bytes, and the table reports EQUAL for a change that is not in the arm. Reading the build log is the documented cover, and #1803 makes it mechanical because it adds a new translation unit.

- `gpu_backend.o` does not exist after the baseline build and does exist after the arm build. That is an existence proof, not a timestamp inference.
- All nine edited `.cpp` files (`fused_norm`, `fused_rope_append`, `paged_attention`, `paged_attention_v2`, `paged_attention_v2_merge`, `sampling`, `sampling_rejection`, `mlx_cxx_kernels`, `mlx_cxx_bridge`) carry mtimes later than the arm build's start.
- The binaries differ, and only the arm contains the probe string.

Build times, for anyone reproducing: baseline 11m16s cold in a fresh clone, arm 9m12s incremental.

## What was not measured

- **The CUDA half.** No GB10 access from this host; `bastion.lablup` does not resolve here. Tracked as #1873.
- **A baseline-versus-arm comparison of the sampler.** Greedy decode never enters `gumbel_max_sample` and `mlxcel generate` has no seed flag, so no byte-identity arm can cover it. The arm's own gate covers the kernel against a reference; what is absent is a two-arm diff.
- **The workspace gate on the baseline commit.** The criterion asks for green gates on the change, so only the arm was run.
- **Absolute timing.** See the machine-state row above.

## For whoever runs the CUDA half

Four traps cost a re-run each here, and all four apply unchanged on CUDA:

1. An EQUAL row whose backend probe stayed silent proves nothing. Run with `MLXCEL_DEBUG_KERNEL_BACKEND=1` and treat a silent arm as a failed arm, not a passing one.
2. Without `MLXCEL_PAGED_ATTENTION_NATIVE=1` the paged arm declines at the 4096-token dispatch floor and compares gather against gather.
3. `/metrics` needs `--metrics`, and a failing `curl -f` is silent.
4. The discriminator is `gather_fallbacks`, not `declines`, which is not exported.

`scripts/paged_decode_counter_ab.sh` encodes all four.
