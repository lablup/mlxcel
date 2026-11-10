# PR #2936: Hardening the ragged batch-engine residuals

**Date**: 2026-11-10
**Status**: Completed (closes #2923)

## Executive Summary

This PR closes the six ragged continuous-batching residuals grouped in #2923 from the #469 audit. The lazy rank-5 batch KV is now zeroed with a device fill instead of a `calloc`ed host mirror plus upload; M-RoPE decode coordinates are bounded by the compiled context capacity before any IREE call, in both the Rust wrapper and the C entry points; a ragged reset refuses a batch size the loaded graph was not compiled for; a prefill KV slot is marked valid only after both the K and V transfers land, and decode rejects an active row whose slot is invalid; `examples/xla_batch_bench.rs` grew bucket, seeded-sampling and cancellation modes that print their own PASS lines; and startup now warns when the bucket set that survived loading lost every image-capable shape. The real-device `#[ignore]` suite passes 4/4, the bench prints REFERENCE-EXACT with all three new modes, and the FFI additions are additive. 15 files, +1421/-88.

## 1. Problem Statement

The #469 hardening run left six items that were real but too small to hold up that PR. Each one is a defect class rather than a tuning opportunity.

- **Host KV mirror**: `xla_ensure_batch_kv_inner` built the zeroed cache with `calloc` and uploaded it, spending `4*B*L*S*nkv*d` host bytes and a full H2D transfer per bucket on first admit, for a buffer that is never read on the host.
- **M-RoPE coordinate bound**: both decode paths computed `cache_len + rope_delta` and indexed a `[context_capacity, rot]` table without an upper bound. A coordinate at the capacity is an out-of-bounds device read, not an error the graph can report. The `Negative` half was checked; the upper half was not.
- **Reset versus compiled B**: `xla_llama_ragged_reset` cleared state for any `bsz`, so a mismatch surfaced later as an opaque shape failure inside IREE with no indication which value was wrong.
- **Non-atomic slot store**: the prefill slot store transferred K, then V. A failed V transfer left resident K behind and decode read a half-written slot as if it were valid.
- **Bench coverage**: `xla_batch_bench` exercised only `XlaBatchEngine` with greedy sampling, so the bucket-set path, seeded sampling and cancellation were never validated on device.
- **Startup image guard scope**: the pre-load image guard judged the requested capacities. If the only image-capable bucket then failed to load, nothing warned and image requests were refused later at admission with no startup signal.

## 2. What Changed

### 2.1 Batch KV zeroing on the device (`d9bbabdb`)

`xla_alloc_zeroed_bv` allocates the device-local buffer, zeroes it with `iree_hal_device_queue_fill` using a 4-byte zero pattern, waits on the fill's semaphore, and only then publishes the buffer view. `xla_ensure_batch_kv_inner` uses it for both the K and V caches and releases both views if either allocation fails. Test-only counters record host bytes and H2D bytes spent by the lazy path.

The semaphore wait is the important part: the fill is enqueued, not synchronous, and the fill's result must be visible before the first decode that binds the cache. The helper awaits before returning, so no un-awaited fill can escape.

### 2.2 M-RoPE coordinate bound (`4b7cd7ca`, device gate `cf6d3a0d`, C guard `7ba1d6a2`)

`mrope_decode_coordinate` and `mrope_slot_coordinates` now return a typed `MropeCoordinateError::OutOfRange { coordinate, context_capacity }` (and `Negative`) before any IREE call. The context capacity is threaded through `prepared.rs` and `iree.rs` so the check uses the compiled capacity, not a constant.

The C shim repeats the upper-bound check at the ragged decode validator and at the single-sequence `xla_llama_decode` and `xla_llama_decode_mrope` entry points, and NULL-checks their pointer arguments. The Rust wrapper cannot be the only guard because C callers reach those entry points directly; the single-sequence addition closes the bypass that a ragged-only guard would have left open.

### 2.3 Ragged reset bound to the compiled B (`7cb73243`)

Load now calls the new additive `xla_llama_ragged_bind(ctx, compiled_b)` before the reset. `xla_llama_ragged_reset` rejects an unbound context and `bsz != rg_compiled_b`, naming both values. No existing FFI signature changed, so spike and `iree-ffi` callers are unaffected.

### 2.4 Slot validity as a state (`42d807b5`, assertion `9d907220`)

The slot store clears `rg_valid[slot]` before the K transfer and sets it only after both transfers succeed. `xla_llama_decode_ragged_impl` rejects an active row whose slot is invalid with `row=%d slot=%d token=%d cache_len=%d is active but its KV slot is not valid`. K bytes are not rolled back on failure because the engine re-prefills a slot before admitting it again; the flag is what makes the half-written state unobservable.

### 2.5 Bench modes (`0e0dadc8`)

- `--buckets c1,c2,...` drives an `XlaBucketSet` with one routing-coverage request per capacity and prints `BUCKETS REFERENCE-EXACT PASS`.
- `--seed N` with `--temperature` and `--top-p` runs two independent seeded engines, checks determinism, and checks a `top_k=1` pass against the greedy references, printing `SEEDED SAMPLING PASS`.
- `--cancel N` cancels N admitted requests after their first token and requires no later event for them, printing `CANCEL PASS`.

Request ids are mapped through a `HashMap<u64, usize>` because a bucket set namespaces ids per bucket, so the submission index is not the id. The bench exits nonzero on any mismatch, so it is its own gate.

### 2.6 Startup warning for a lost image bucket (`26fb5055`, docs `31787aee`)

`xla_image_surviving_bucket_warning(&engine.capacities(), &capacities, image_floor)` is a pure function over the requested and surviving capacities. After `XlaBucketSet::load` it warns when the request had an image-capable bucket, none survived at or above the floor, and something was actually dropped. The message names the dropped capacities, the largest survivor, the floor and `MLXCEL_XLA_CONTEXT_CAPACITY`. `docs/environment-variables.md` now documents the drop behavior.

### 2.7 Review commits

- `a93ca23a` gates `tests/xla_ragged_hardening.rs` on `xla-iree,test-utils` so the plain `--all-targets` compile CI runs does not resolve test-support-only imports.
- `a4b112f1` allowlists the build-internal `MLXCEL_XLA_TEST_SUPPORT` define in `scripts/ci/check_env_var_docs.py`, because it is a `build.define` for the C shim, not an operator environment variable.
- `7ba1d6a2` is the single-sequence C guard described in 2.2.
- `31787aee` documents the bucket-drop warning.

## 3. Technical Decisions

### 3.1 Device fill instead of a host mirror

The alternatives were to keep `calloc` plus upload, to map the device buffer on the host and `memset` it, or to zero the cache inside the graph. The fill keeps host memory flat and costs one queued fill per bucket instead of a full upload; mapping touches the host path the change exists to remove; a graph-side zero would add a dispatch to every graph that binds the cache. The cost is a semaphore and an explicit wait, which is the correct ordering primitive here because the runtime does not order work across separate entry-point calls.

### 3.2 Typed Rust error plus a repeated C check

Repeating the bound in C is deliberate duplication at a safety boundary. Relying on the Rust wrapper alone leaves the direct C API unguarded, and clamping the coordinate would decode with silently wrong positions instead of failing. The typed error carries both the coordinate and the capacity, so the caller can report exactly what overflowed.

### 3.3 Additive bind rather than a reset signature change

Changing `xla_llama_ragged_reset(bsz)` to take the compiled B as well would break every existing caller for no safety gain: the reset still needs the requested `bsz` to compare against. A separate bind records context state once at load, keeps the ABI stable, and makes the unbound case detectable.

### 3.4 Invalidate first, publish last

Clearing the validity flag before the K transfer makes every failure window observable as invalid. The alternative, setting validity between K and V, leaves the K-only state readable; clearing after a failure cannot work because the failure is exactly what must be detected. The cost is that a failed store leaves K bytes resident until the slot is re-prefilled, which the engine already guarantees.

### 3.5 The bench as its own gate

The three modes need a real engine, tiny weights and a reference comparison. Adding three more `#[ignore]` integration tests would duplicate the fixture that the example already owns. Printing explicit PASS lines and exiting nonzero on mismatch gives CI a usable signal without adding an ignored test that CI will not run anyway.

## 4. Validation

Environment: AMD Strix Halo (`gfx1151`) through ROCm, pinned IREE ROCm distribution `iree-rocm-3.12.0rc20260721`, device `hip`. Every cargo and GPU command ran under `scripts/rocm_gpu_guard.sh --hold`, with `OPENSSL_INCLUDE_DIR`, `OPENSSL_LIB_DIR`, `IREE_ROCM_HOME` set and `eval "$(scripts/iree/setup-rocm.sh --env)"`, `CARGO_BUILD_JOBS=8`.

Real-device hardening suite (`#[ignore]` gates):

```
cargo test --release --features xla-iree,test-utils --test xla_ragged_hardening -- --ignored --test-threads=1
```

4 passed, 0 failed. The stderr shows the slot validator rejecting `row=0 slot=0 token=7 cache_len=3` and the reset rejecting `bsz=4` against `compiled_b=8`. The gates are `#[ignore]`d and live in the root crate because the `mlxcel-xla` unit-test binary cannot link the IREE runtime, following `tests/xla_prepared_prefill.rs`.

Compile, lint and format:

```
cargo fmt --all -- --check
cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings
cargo check --features xla-iree --all-targets
```

All exit 0; clippy prints only vendored IREE C header warnings.

Unit and tooling gates: `decode_coordinate` 3 passed, `mrope_slot_coordinates` 1 passed, `host_preprocessor` 29 passed (1 ignored), `check_env_var_docs.py` exit 0, python tooling 214 passed.

Bench modes, on a generated tiny llama checkpoint (no real checkpoint on this host; this proves reference-exactness, not throughput):

```
cargo run --release --features xla-iree,test-utils --example xla_batch_bench -- \
  --model /tmp/xla2923-fixture/model --prompts /tmp/xla2923-fixture/prompt_ids.json \
  --device hip --batch 8 --requests 16 --maxcap 48 --buckets 256 --seed 7 --cancel 2
```

`RESULT: REFERENCE-EXACT PASS`, `BUCKETS REFERENCE-EXACT PASS`, `SEEDED SAMPLING PASS (deterministic=true, top_k=1==greedy=true)`, `CANCEL PASS (2/2 cancelled, non-cancelled requests reference-exact=true)`. The same flags at `--batch 4` also passed, and `--buckets 256,2048` routed both capacities.

## 5. Learning Points

- **HAL fills are asynchronous.** `iree_hal_device_queue_fill` returns as soon as the work is enqueued; correctness depends on waiting on the signal semaphore before the buffer view is published. A fill pattern must be 1, 2, 4 or 8 bytes, so zeroing f32 KV with a 4-byte zero is both the natural and the legal choice.
- **A validity flag is a small state machine.** Clearing before the first transfer and setting after the last is what turns a partially written buffer into a detectable state; ordering it any other way leaves a readable half-state.
- **Testing a removed code path needs instrumentation.** The host-mirror test can only assert zero host and H2D bytes because the counters were added with the removal; the path it replaced always spent those bytes, so a zero reading is meaningful.
- **A pure predicate is testable without the engine.** The bucket-drop warning is a function over two capacity lists, so all four cases are unit tests rather than device tests.

## 6. Follow-ups

- Pre-existing, out of scope: `xla_llama_prefill` and `xla_llama_prefill_slot_logits` bound token counts but not position values, so a direct C caller can still hand an out-of-range prefill position to IREE. Changing four prefill entry points would broaden this PR; it belongs in a follow-up issue.
- From #2922: B=4 ragged decode with the native multi-row GEMV is about 5 percent below the f16 path because the kernel uses eight fixed accumulators regardless of B.
- From #2921: the KV read convert and context matmul are each about 103 us per step, and the tuning matcher is pinned to the measured batch of 8, so other capacities are not yet covered.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 15 |
| Lines added | +1421 |
| Lines deleted | -88 |
| Commits | 13 |
| Real-device gates | 4 passed |

| Area | Files | Summary |
|------|-------|---------|
| C shim | `csrc/xla_iree.c` | Device fill, coordinate bounds, compiled-B bind/reset, slot validity |
| Rust wrapper | `prepared.rs`, `iree.rs`, `batch.rs` | Typed M-RoPE errors, capacity threading, compiled-B load path |
| Bench | `examples/xla_batch_bench.rs` | Buckets, seeded sampling, cancel modes |
| Guard | `src/server/model_worker.rs`, `src/multimodal/host_preprocessor.rs` | Surviving-bucket warning |
| Tests | `tests/xla_ragged_hardening.rs`, `host_preprocessor_tests.rs` | Four device gates, four warning unit tests |
| Tooling/docs | `check_env_var_docs.py`, `build.rs`, `docs/environment-variables.md` | Allowlist, test-support define, bucket docs |
