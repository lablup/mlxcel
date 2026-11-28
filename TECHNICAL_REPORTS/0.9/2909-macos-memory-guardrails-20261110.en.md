# macOS memory guardrails for large resident models

- **PR**: #2909 (open, not merged)
- **Issue**: #2883 (closes)
- **Date**: 2026-11-10 (generation date; the PR is open)
- **Branch**: `feature/issue-2883-macos-memory-guardrails`
- **Author**: Jeongkyu Shin
- **Reviewer**: not yet assigned
- **Measurement status**: no soak and no timed measurement has been run for this change.

## Executive Summary

The fit check compared a model load against total RAM and Metal's recommended working set. Neither says what macOS can give right now without compressing or swapping. This PR adds an "available now" figure, a reserve of `max(7 GiB, 15% of RAM)`, and guardrails built on them: a warning in `inspect` and `--estimate-memory`, process-memory metrics on the server, an opt-in `mlock` of weights (`MLXCEL_WIRE_WEIGHTS=1`), and a growth gate on the `qwen4_exp` PLE row cache. All new behavior is advisory or opt-in. The `FITS` verdict is unchanged, no load is refused, and wiring stays off by default.

## Problem Statement

On a large-model macOS host, an idle server can have its weights compressed or swapped, because GPU wiring of a buffer ends a few seconds after its last command buffer. Separately, a store that grows in steps (row cache, and later expert streaming #2882 and prompt-cache snapshots #2885) has no way to ask "can I take one more step without pushing macOS under its reserve". Total RAM overstates what is available, and Metal's working set is a GPU cap, not a statement about free memory.

## Change Summary

- `src/lib/mlxcel-core/src/host_memory.rs` (new, 409 lines): the reusable figure, the reserve, process footprint, and `AvailabilityGate`.
- `src/lib/mlxcel-core/src/weight_wiring.rs` (new, 250 lines): opt-in chunked `mlock` of evaluated weights, hooked into `weight_materialize.rs`.
- `cpp/mlx_cxx_bridge.{cpp,h}`: new `array_data_address` bridge call.
- `src/execution/memory_availability.rs`, `memory_estimate.rs`: the macOS section and warning in `inspect` and `--estimate-memory`.
- `src/server/model_worker.rs`, `routes/metrics.rs`, `routes/metrics_process_memory.rs`: footprint log after load and `/metrics` gauges.
- `src/models/qwen4_exp_ple_cache.rs`, `qwen4_exp_ple_storage.rs`: row cache stops growing before the reserve.
- `scripts/macos_memory_soak.sh` (committed, not run), `docs/macos-memory.md`, env var docs, CHANGELOG.

### host_memory.rs (read from the diff)

- **Available now** is `free + purgeable + min(inactive + speculative, file_backed)` pages. The `min` cap excludes inactive anonymous memory of other programs, which macOS frees only by compressing or swapping. `file_backed` is `external_page_count`, which also contains active file pages, so the cap is a loose upper bound on reclaimable cache, not an exact one.
- **Speculative double count fix**: the kernel's `free_count` already includes speculative pages, so the platform code passes `free_count - speculative_count` (saturating) as `free`, then adds speculative once inside the `min`. The second commit exists for this. It is covered by a fixed-input test and was checked against `vm_stat`.
- **Reserve**: `max(7 GiB, 15% of RAM)`, computed in `u128` to avoid overflow. `usable_bytes` is available-now minus reserve, saturating at zero. `shortfall_bytes(demand)` is the excess of demand over usable. `wired_ceiling_bytes` is total RAM minus reserve.
- **Unknown is not zero**: off macOS, or if the kernel query fails, both queries return `None`. Callers treat that as "unknown" and keep previous behavior.
- **Platform code**: `hw.memsize` via `sysctlbyname`, cached in a `OnceLock`; `host_statistics64` with a host port taken once (each `mach_host_self` call adds a send right); `task_info(TASK_VM_INFO)` into a hand-declared `TaskVmInfoRev1` struct through `phys_footprint`. The footprint read returns `None` if the kernel returned fewer units than the struct holds, so a partly filled struct is never trusted. The struct layout is declared by hand to match `<mach/task_info.h>`; nothing in the PR checks it against the header automatically.
- **AvailabilityGate**: caches one status for 250 ms, and tracks `granted_since_probe`, the bytes it approved since the last read, since the cached status cannot reflect them. A step is approved only if `step + granted_since_probe` fits within usable bytes. A refused step does not add to the grant counter. With an unknown status every step is approved. The mutex recovers from poisoning rather than panicking. The probe is injectable, so tests need no sleeping.

### weight_wiring.rs (read from the diff)

- Off unless `MLXCEL_WIRE_WEIGHTS` is `1`, `true`, `on` or `yes`. Under the `lazy` materialization policy it logs a warning and does nothing, since lazy weights are not evaluated. Off macOS it warns and is ignored.
- Ranges (address from the new bridge call, length from `nbytes`) are rounded out to whole pages, sorted and merged, so a page shared by two small tensors is locked once.
- `wire_ranges` takes the kernel calls as closures (`wired_now`, `lock`, `unlock`) so the policy is unit-tested with fixed inputs. It checks the system wired total before starting, locks 1 GiB chunks, re-reads the wired total after each chunk, and unlocks and stops the chunk that pushes the total above `RAM - reserve`. A failed `mlock` counts the range as failed, records the first `errno`, abandons the rest of that range, and moves on to the next. The pre-check stops at `>=` the ceiling while the per-chunk check stops at `>` it, a small asymmetry that has no practical effect at GiB granularity.
- The ceiling is compared with the real system wired count, not a prediction, because pages Metal already wired for the GPU are in that count and `mlock` adds nothing for them.
- If the wired total cannot be read, the pass stops (treated as ceiling reached), which is the conservative direction.
- `array_data_address` returns 0 for an unevaluated array and for a strided view (neither row nor column contiguous), because address plus `nbytes` would not cover a strided view's bytes. Such tensors are skipped and counted under requested but not wired. The third commit fixes this.
- Scope limits stated in the code: multimodal towers stay lazy and are not wired, and repacks at the first token (expert stacking, dtype conversion) replace wired sources with unwired results.

### Other pieces

- `qwen4_exp` PLE row cache charges each step with its bookkeeping and reuses one availability query for 250 ms. The prompt-cache stores are deliberately not gated: entries exist before insert and a byte budget bounds them, so there is no step to refuse.
- `/metrics` exports `phys_footprint`, compressed bytes, MLX active and cache memory, and a wired-weight gauge. Off macOS the process-memory families are absent rather than zero.

## Technical Decisions

- **Advisory, not refusing.** `inspect` prints numbers and a warning, and `FITS` is unchanged. A strict mode was considered and left out because the figure has not been validated against real pressure. Ruled out: refusing loads on an unmeasured heuristic.
- **Cap reclaimable cache at file-backed pages** instead of counting all inactive memory. This avoids reporting other programs' compressible anonymous memory as free. Cost: the cap is approximate.
- **Read the real wired count** instead of predicting what `mlock` will add, since Metal's own wired pages overlap.
- **Closures for kernel calls** so the wiring and gate policies are tested without a macOS host or privileges.
- **Measure, then keep.** `MLXCEL_WIRE_WEIGHTS` stays off, `MLXCEL_WIRED_LIMIT` keeps its default, and nothing is refused until a soak shows a benefit.
- **A shared gate** lives in `mlxcel_core::host_memory` so #2882 and #2885 reuse it instead of re-deriving the figure.

## Verification and What Is Unverified

Done, as stated in the PR body: unit tests for the arithmetic, gate, wiring policy, report text and metrics; a real `mlock` of an evaluated MLX buffer; `mlxcel inspect` of the OptiQ checkpoint (before the double-count fix, and with no warning on a 512 GB host, so the warning path is tested only by unit tests); a short `qwen3.5-4b-4bit` server run with wiring on (2.20 GiB wired); clippy on both library crates, a metal and accelerate `cargo check`, fmt and the env-var doc check. Linux was not compiled and shellcheck is not installed.

**No soak and no timed measurement has been run.** `scripts/macos_memory_soak.sh` is committed but not executed. Not verified: whether wiring prevents idle compression on 128 GB and 64 GB machines, whether the available-now figure predicts real swapping, any decode or load time effect of wiring, the accuracy of the hand-declared `task_vm_info` layout on other macOS releases, and how `munlock` of a rejected chunk interacts with pages another component has also locked. `docs/macos-memory.md` states that nothing has been measured.

## Learning Points

- On macOS, `host_statistics64` `free_count` includes speculative pages. Adding speculative separately double counts them.
- Test kernel-facing policy by injecting the syscalls as closures.
- A "stop growing" gate needs a running tally of what it already approved, because a cached reading cannot show it.

## Follow-ups

- Run the soak on a quiet 128 GB and a 64 GB machine, then decide on a wiring default.
- Adopt `AvailabilityGate` in #2882 and #2885.
- Compile on Linux in CI to confirm the non-macOS stubs.
