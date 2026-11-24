# PR #3372: OpenXLA per-slot recurrent state in continuous batching; Mamba2 serves via mlxcel-server

**Date**: 2026-11-24
**Status**: PR open (closes #3340, epic #564 Phase 1 unit U4). Validated on ROCm gfx1151 with IREE `hip`. Attention-free recurrent models only: hybrids (attention plus recurrent layers) are still refused by the batch engine until #3344, and the prompt-cache guard is #3341.

## Executive Summary

After #3371 (#3339), Mamba2 generated single-sequence on the xla backend, but the continuous-batching engine knew only per-slot KV, so `IreeRaggedLlama::load` refused recurrent configs and `mlxcel-server` could not serve them. This PR makes the ragged decode graph take and return one `[B, ...dims]` tensor per recurrent-state spec, gives the C shim a per-slot state set `sscache_b[]` stored under the same `rg_valid[slot]` flag as the KV, and lifts the load refusal for attention-free models. `MLXCEL_BACKEND=xla mlxcel-server -m <mamba2 checkpoint>` now serves with continuous batching, and every batched request is token-identical to its single-sequence output.

The second commit reverses a decision from #3371: recurrent configs are now pinned to f32 on every device, single-sequence and batched, the way MLA configs are. The serve gate exposed that at the old f16 hip default the single-sequence Mamba2 decode departs from the HF fp32 oracle after about 16 tokens on some prompts, so the batched stream was being compared against a drifting reference. The pin costs throughput: the serve gate on `mamba2-130m` ran at 285.6 tok/s batched at f16 and 123.7 tok/s at f32.

As in #3371, most of the design and review effort went into cross-request isolation: a recycled slot, a torn store, a request refused in Rust, a cancelled or disconnected request, and a failed ragged decode must each leave no row able to continue a state it did not finish.

## 1. Problem Statement

#3371 delivered the single-sequence half of the state ABI and left the batched half refused at every layer:

- **Emitter.** `emit_decode_ragged_with_native` called `c.reject_recurrent("ragged decode")`, and the ragged argument schema always declared `kcache` / `vcache`. The Mamba2 block in the ragged path was `unreachable!`.
- **Shim.** The context held per-slot KV only (`kcache_b`, `vcache_b`, allocated by `xla_ensure_batch_kv`, written by `xla_store_prefill_kv_slot` under `rg_valid`, swapped after each ragged decode). `xla_require_kv` refused `xla_llama_ragged_reset`, `prefill_slot_logits`, the diagnostics and embeddings slot prefills, and the ragged decode for any context with `n_state > 0`.
- **Runtime.** `IreeRaggedLlama::load` refused any config with recurrent state, and its compile step always emitted an embeddings prefill, which the emitter returns `Err` for on a recurrent config.
- **Required module entries.** The shim required `decode_step.state_import` whenever `n_state > 0`. A ragged decode module threads state as arguments and has no such entry, so a ragged recurrent bundle could not have loaded even with the other refusals lifted.
- **Server startup.** The serve worker asks every OpenXLA checkpoint for a Gemma3n audio tower, and `Gemma3nXlaAudioConfig` parsed `config.json` strictly. Mamba2 configs carry a bare `Infinity` in `time_step_limit`, so server startup failed with `parse config JSON: expected value at line 25` before any of the batching code ran.
- **Isolation rules carried over from #3339** (issue comment on #3340): a refused or failed prefill drops that slot's state; a request refused in Rust before reaching the shim must not leave a slot's previous state live; a recycled slot never sees the previous occupant's state; a failed ragged decode should invalidate every live slot, mirroring `rg_valid`.

Issue #3340 named two risks: the single validity flag must cover KV and state atomically, because a partial state store observed as valid would corrupt a request silently, and the ragged graph grows by the state inputs and outputs at `B = 8`.

## 2. Change Summary

| File | Change |
|---|---|
| `csrc/xla_iree.c` | `sscache_b[XLA_MAX_STATE]`; `xla_ensure_batch_state` (zeroed `[rg_bsz, ...dims]`, all or nothing, overflow-checked) and `xla_release_batch_state`; `xla_validate_batch_state`; `xla_store_prefill_kv_slot` becomes `xla_store_prefill_slot(c, slot, kc, vc, states)`; new export `xla_llama_slot_invalidate`; `xla_require_kv` takes a slot and invalidates it on refusal; `ragged_reset` releases the state set; slot prefill pushes `state_zero[]` and pops the state results; ragged decode pushes and swaps `sscache_b[]`; `xla_llama_decode_ragged_impl` wraps a new `_inner` and clears every `rg_valid` byte on failure for stateful bundles; `decode_step.state_import` no longer a required entry; test counters `xla_test_batch_state_{allocs,bytes,reset_counters}` |
| `src/emitter/model.rs` | `RaggedArgs { kv: Option<(Val, Val)>, state: Vec<Val> }`; KV args only when `has_kv_cache()`; state args after the KV; no RoPE vectors and no key mask without KV; results built as a list (token or logits, then K/V when present, then one `[B, ...]` result per state); `reject_recurrent` only for hybrids |
| `src/emitter/model_recurrent.rs` | `RecurrentLayers::split_rows` / `join_rows` (slice per layer, write back with `dynamic_update_slice`); the ragged Mamba2 block runs `ssm::mamba2_mixer_step` on `[B, hidden]`; `take_batched_state_args` |
| `src/emitter/recurrent_config.rs`, `builder.rs`, `mod.rs` | `Config::pins_f32_precision` (MLA or recurrent state); `resolve_pinned_precision_with_override` |
| `src/emitter/recurrent_ragged_tests.rs` (new) | 5 emitter tests: per-slot state without KV at B=4 and B=8, f32 state at every precision with sampling, one write-back per state per layer, hybrid refused, dense ragged signature unchanged |
| `src/emitter/recurrent_tests.rs` | Drops the ragged case from the "graphs without a state contract reject Mamba2" test; adds `recurrent_config_pins_f32_unless_precision_is_explicit` |
| `src/iree.rs` | `effective_precision_with_override` uses the pin for every dense config; `IreeRaggedLlama::load` refuses only `has_recurrent_state() && has_kv()`; ragged compile skips the embeddings prefill for recurrent configs; `begin_slot_prefill` (validate slot, then `xla_llama_slot_invalidate`) at all 8 slot entries; test-support counter accessors; `recurrent_runtime_pins_f32_unless_precision_is_explicit` |
| `src/gemma3n_audio_config.rs` | Tolerant HF JSON parser; a quoted non-finite number in a config that has `audio_config` is refused |
| `src/lib.rs` | Re-exports the batch state counters under `test-support` |
| `tests/xla_ssm_state_carry.rs` (new) | Synthetic two-layer Mamba2; 5 device tests (ignored without IREE) |
| `tests/xla_ssm_single_seq.rs` | Doc comment update for the narrowed `xla_require_kv` scope |
| `scripts/xla/validate_arch.sh`, `README.md`, `FINDINGS_ssm.md`, `docs/supported-models.md`, `docs/environment-variables.md`, `CHANGELOG.md` | Serve gate applies to Mamba2; per-slot ABI section; measurements and the precision decision; Mamba2 marked as served; the f32 pin in the `MLXCEL_XLA_PRECISION` row; changelog entry (and the #3339 entry amended to say f32 is now the default) |

19 files, +1430 / -224, in two commits.

### Ragged graph contract

The ragged `decode_step.main` for an attention-free recurrent config takes the existing inputs up to `cache_len`, then one `[B, ...dims]` input per `Config::state_tensors` entry (`ssm_h [B, L, H, P, N]`, `ssm_conv [B, L, K-1, conv_dim]`, f32), and returns the token or logits followed by the advanced state tensors. There is no `kcache` / `vcache` input or result, no RoPE gather and no key mask; the `Ragged` layout receives unread scalar placeholders for those. Rows are independent: an inactive row is advanced like any other and the shim discards it. State stays f32 at every contraction precision (`mamba2_ragged_decode_samples_and_emits_at_every_precision`). The graph contains no `scatter`, `stablehlo.while`, `stablehlo.custom_call` or `util.global`, and exactly one gather (the token embedding).

For `mamba2-130m` the state layout from #3371 is `[24, 24, 64, 128]` and `[24, 3, 1792]`, so one slot's row is 18,874,368 + 516,096 bytes (about 18.5 MiB) and a `B = 8` set is about 148 MiB per buffer generation. These figures are computed from the layout, not measured.

Dense ragged MLIR is byte-identical to main: the validation goldens pass and `attention_ragged_decode_keeps_its_kv_results` checks the dense signature.

### Shim slot lifecycle

| Event | Shim behavior |
|---|---|
| `xla_llama_ragged_reset` | Releases `sscache_b[]` along with the existing KV reset. The next slot store allocates a fresh zeroed set, so no row survives a reset. |
| Rust slot entry (`begin_slot_prefill`) | Range-checks the slot, then calls `xla_llama_slot_invalidate` before any other check. A request refused in Rust (empty or oversized prompt, bad prepared payload) leaves the slot invalid. |
| `xla_llama_prefill_slot_logits` | Clears `rg_valid[slot]`, re-zeroes `state_zero[]` and pushes it after `real_len`, pops K/V only when `has_kv`, pops and shape-validates each state result, then calls `xla_store_prefill_slot`. |
| `xla_store_prefill_slot` | Clears the flag, validates KV and every state result, ensures the batch KV and state sets exist, copies K, V (when `has_kv`) and each state row at `slot * row_bytes` device to device, and sets the flag only after the last copy. Nothing is rolled back on failure; the slot stays invalid until a later prefill. |
| Ragged decode | Refuses unless every `sscache_b[i]` exists; pushes `sscache_b[]` after the KV; validates each `[rg_bsz, ...dims]` result; swaps KV and state together after every result validated. |
| Failed ragged decode (pre-invoke, invoke, validation or readback) | Swaps nothing. On a bundle with `n_state > 0`, clears every `rg_valid` byte. Attention-only bundles keep their flags as before. |
| Embeddings and diagnostics slot prefills | Still refused by `xla_require_kv` for any stateful context; the refusal now also invalidates the slot. |
| `xla_llama_free` | `xla_release_state` also releases the per-slot set. |

Cancel and disconnect only free the scheduler slot. The stale row is not cleared, because the next admit into that slot overwrites the whole row from a zero-state prefill and the flag is clear until it does.

### Rust and engine

`XlaBatchEngine`'s `Slot`, `XlaBucketSet` and `src/server/batch/xla_worker.rs` gained no fields: the state lives on the device keyed by slot, like the KV. Each bucket context owns its own `sscache_b` set. `IreeRaggedLlama::load` now refuses only hybrids, and its compile step emits no embeddings prefill for a recurrent config, matching the single-sequence bundle.

## 3. Technical Decisions

**Thread state through the ragged decode as arguments, not globals.** The single-sequence decode keeps state in module globals seeded by `@state_import`. The ragged path already threads its KV as inputs and swaps the outputs, so state follows that pattern: one input and one result per spec, swapped in the shim. This keeps a single slot-addressed storage model for the batch (device buffers indexed by row), lets a slot prefill write one row with a device copy, and needs no per-slot import entry. A test asserts the ragged graph has no `util.global`. The cost is that the arguments are not donated, so each step writes a new state buffer of `B x row` bytes on the device, the same pattern the KV already has.

**Per-layer slices and `dynamic_update_slice`, not scatter.** `split_rows` slices `[B, 1, ...]` per recurrent layer and reshapes to `[B, ...]`; `join_rows` writes each advanced layer back at its layer index. Every row of the batch is written each step, so no per-row index is needed and the graph stays free of `scatter`, which the ragged tests ban. `mamba2_ragged_decode_writes_every_layer_row_back` pins the count at one write per state tensor per layer.

**No RoPE and no mask without KV.** An attention-free model reads neither. Emitting them would add gathers and compares that IREE may or may not fold away; skipping them makes the absence checkable (one gather in the whole graph) and leaves the dense emission path untouched, which the byte-identical goldens confirm.

**One validity flag for KV and state.** This answers the issue's main risk. A separate state flag would allow the ragged decode to see a slot whose KV is valid and whose state is not, or the reverse. With one flag, cleared before the first transfer and set after the last, a store that fails after `ssm_h` but before `ssm_conv` is invalid as a whole. The test hook was moved so that on an attention-free model it fails after the first state transfer, which is the case the issue asked the torn-store test to catch.

**Every slot prefill starts from the zero state.** The slot prefill pushes `state_zero[]`, never the row's current contents, and overwrites the entire row. A recycled slot therefore cannot read its previous occupant's state, and eviction (cancel, disconnect, finish) needs no device work. Zeroing rows on evict was the alternative; it adds a device fill on a path that today does nothing, and still needs the overwrite-on-admit guarantee to be safe after a failure.

**Invalidate in Rust before any check.** This is the #3371 lesson applied to slots. A request the Rust engine refuses never reaches the shim's own invalidation, so `begin_slot_prefill` calls `xla_llama_slot_invalidate` right after the range check at all eight slot entry points. The test `refused_requests_and_failed_decodes_invalidate_slot_state` fails with this reverted.

**A failed decode invalidates every slot, only for stateful bundles.** A failed step swaps nothing, so the resident buffers keep their pre-step contents in both cases. The shim still treats a partly run recurrent step as leaving no trustworthy per-row state and refuses every row until its slot is prefilled again, the batched form of the LOW item from the #3339 security review (a retried decode continuing from partly advanced state). Attention-only bundles keep their previous behavior, so the dense path does not change. In production the rule rarely matters on its own, because any pump error already ends the serve worker after `fail_all`; it matters for direct `IreeRaggedLlama` callers and for any future recovery path.

**Lazy allocation, released on reset.** `xla_ensure_batch_state` allocates on the first slot store after a reset, mirroring `xla_ensure_batch_kv`, and either the whole set exists afterwards or none of it does. Production resets only at load, so no client can trigger reallocation. `state_allocation_counters_show_one_allocation_per_reset` fails if the reset stops releasing the set.

**Keep `xla_require_kv` for the paths that still have no state contract.** The embeddings and diagnostics slot prefills have no state inputs or outputs. They keep refusing any stateful context, now also invalidating the slot. The emitter and `IreeRaggedLlama::load` keep refusing hybrids, so a KV plus state model cannot run a path that was validated only for one of them.

**`decode_step.state_import` becomes optional.** Only the single-sequence decode module has it. Making it a required entry would have prevented the ragged bundle from loading. The single-sequence path now refuses at the first import instead of at load (LOW, accepted in review; the import stays pending, so a decode never runs on stale globals).

**Pin recurrent configs to f32 (reversing #3371).** #3371 kept the f16 hip default because one 40-token prompt was token-exact. The serve gate in this PR found two f16 differences (`xla_batch_bench` B=4 with a 64-token prompt at 7/8, request 6 from token 17; the server check at 3/4). Both workloads were exact at f32. On the differing server prompt ("In computer science, a linked list is") the batched f16 stream matched the HF fp32 oracle and it was the single-sequence f16 decode that departed after 16 tokens. The difference is rounding between the `[B, hidden]` and single-row projections, not state handling. Options were to keep f16 and accept a reference that drifts, to pin only the ragged bundle (which would leave the CLI drifting from the oracle and from the server), or to pin both. Pinning both is the only choice under which the CLI, the server and the oracle agree. `Config::pins_f32_precision` folds this into the existing MLA rule, and an explicit `MLXCEL_XLA_PRECISION` still overrides it for measurement.

**Tolerant parser in the Gemma3n audio probe, strict where it matters.** The probe runs on every OpenXLA checkpoint at server startup, so it must parse Mamba2's bare `Infinity`. A config that actually has an `audio_config` is still refused if quoting was needed, because the audio readers would treat a quoted number as a missing field and take its default.

## 4. Review Outcomes

**Implementation review** (orchestrator-required fix in 0807595c):

| Finding | Severity | Status |
|---|---|---|
| At the f16 hip default the single-sequence reference drifts from the HF fp32 oracle, so the serve gate (7/8) and the server check (3/4) compared batched output against a wrong reference | HIGH | Fixed: recurrent configs pinned to f32 like MLA, overridable by an explicit `MLXCEL_XLA_PRECISION`; `recurrent_config_pins_f32_unless_precision_is_explicit` fails without the pin; every gate re-run at the new default |
| A single-sequence bundle without `decode_step.state_import` fails at the first decode rather than at load | LOW | Open, accepted: the refusal is clear and the import stays pending |
| #3344 plans an f16 Nemotron-H run; hybrids inherit the f32 pin unless `MLXCEL_XLA_PRECISION` is set | Note | For #3344 |

**Security and performance review** (head 0807595c): no CRITICAL, HIGH or MEDIUM finding, no commit.

Checked: cross-request isolation for refused admits, torn stores, recycled slots, cancel, disconnect, failed ragged decode (pre-invoke, invoke, validation, readback), bucket switch and shutdown. Bounds: the slot is range-checked before any offset; `rg_bsz * row` is checked in `xla_ensure_batch_state`, and the reset that changes `rg_bsz` also releases the set, so the bound stays valid for later stores. `state_desc` is written only at create (and copied into buckets), and prefill and decode results are shape-validated against it before a copy or swap. Resource use: the state set is fixed at `B x row` per bucket and allocated once per reset; per step the state is threaded device side with no host copy.

| Finding | Severity | Status |
|---|---|---|
| On a pinned config (MLA, recurrent) an unrecognised `MLXCEL_XLA_PRECISION` (for example `fp8` or `F16`) is ignored and f32 is used, while dense configs reject it | LOW | Open: the safe direction, and MLA behaved this way before this PR, but an operator typo goes unreported |
| Any prefill or decode error stops the whole serve worker, and the request popped in a failed pump is dropped | Pre-existing | Out of scope |

**Finalize**: no change; docs and CHANGELOG already stated the f32 default.

## 5. Tests and Verification

All runs on ROCm gfx1151, IREE `hip`, with `AntonV/mamba2-130m-hf` @ `05e8773f`, at the default precision after the pin (f32) unless stated.

Before the pin (f16 hip default at the time):

| check | precision | result |
|-------|-----------|--------|
| `validate_arch.sh --batch 4 --requests 8 --maxcap 24` | f16 | PASS |
| `xla_batch_bench`, oracle prompt, `--cancel 2` | f16 | PASS |
| `xla_batch_bench`, oracle prompt, `--buckets 256,512` | f16 | PASS |
| `xla_batch_bench`, B=8, 16 requests, 64-token prompt | f16 | PASS |
| `xla_batch_bench`, B=4, 8 requests, 64-token prompt | f16 | 7/8 (request 6 differs from token 17) |
| same | f32 | PASS |
| server, 4 concurrent temperature-0 `/v1/completions` vs `mlxcel generate` | f16 | 3/4 |
| same | f32 | 4/4 |

After the pin:

| check | result |
|-------|--------|
| `mlxcel generate`, 40 greedy tokens vs HF fp32 oracle, 5 prompts ("The capital of France is", "In computer science, a linked list is", "Once upon a time, in a small village by the sea,", "The three primary colors are", "Water boils at a temperature of") | 5/5 token-exact; `xla_oracle_check` 5/5 |
| `validate_arch.sh --batch 4 --requests 8 --maxcap 24` (structural, token-exact, serve reference-exact) | PASS; 123.7 tok/s batched vs 75.0 sequential |
| `xla_batch_bench`, B=4, 8 requests, 64-token prompt | PASS (8/8); 122.8 tok/s |
| `xla_batch_bench`, oracle prompt, `--cancel 2` | PASS (2/2 cancelled); 122.5 tok/s |
| `xla_batch_bench`, oracle prompt, `--buckets 256,512` | PASS; 98.3 tok/s |
| `xla_batch_bench`, B=8, 16 requests, 64-token prompt | PASS; 108.7 tok/s |
| `mlxcel-server`, 4 concurrent temperature-0 `/v1/completions` | 4/4 equal the CLI |

The throughput figures are single runs as reported in `FINDINGS_ssm.md`; no variance was measured.

`tests/xla_ssm_state_carry.rs` (synthetic two-layer HF-layout Mamba2, capacity 32, B=4; every batched stream compared against the `IreeLlama` single-sequence reference):

- `mid_stream_admit_into_a_recycled_slot_matches_the_reference`
- `half_full_and_full_batches_give_identical_per_request_ids`
- `torn_state_store_leaves_the_slot_invalid_and_the_next_admit_recovers` (uses `IreeRaggedLlama::test_set_slot_store_failure`)
- `refused_requests_and_failed_decodes_invalidate_slot_state`
- `state_allocation_counters_show_one_allocation_per_reset`

Run with `cargo test --release --features xla-iree,test-utils --test xla_ssm_state_carry -- --ignored`: 5/5 on hip. `xla_ssm_single_seq` 6/6 and `xla_ragged_hardening` 6/6. Each carry fix was reverted and its test failed: the Rust slot invalidation and the decode-failure invalidate-all fail `refused_requests_and_failed_decodes_invalidate_slot_state`; the state row copy fails 4 tests; the release in reset fails `state_allocation_counters_show_one_allocation_per_reset`.

Unit tests added: 5 in `recurrent_ragged_tests.rs`, 1 in `recurrent_tests.rs` and 1 in `iree.rs` for the pin (both fail without it), 1 in `gemma3n_audio_config.rs`. `cargo test --profile test-fast -p mlxcel-xla --lib`: 520 passed on the PR head as reported in the PR body. The branch was already up to date with origin/main when this report was written, so the suite was not re-run for the report.

Dense regression against main (e062a10e), on hip: `xla_ragged_hardening` 6/6, `xla_greedy_timing` 3/3, `xla_decode_scalar_reuse` 4/4, `xla_prepared_prefill` 4/4 on both. `llama-3.2-1b` `xla_batch_bench` B=4, 8 requests: REFERENCE-EXACT PASS on both, 64.56 tok/s before and 63.53 tok/s after (single runs).

Static gates: `cargo fmt --check`; `cargo clippy --features xla-iree --all-targets -- -D warnings`; mlxcel-xla clippy with and without `iree`.

## 6. Pending Work and Limitations

- **Hybrids are refused.** The shim's store and decode already handle `has_kv` together with state; the emitter's `reject_recurrent` for KV plus state and the `IreeRaggedLlama::load` hybrid refusal are the points #3344 has to lift. Hybrids inherit the f32 pin.
- **Prompt cache.** A recurrent state cannot be rebuilt from a token-prefix hash. The guard that keeps recurrent models off the prompt cache and APC is #3341, not this PR.
- **f16 is not token-exact for recurrent models.** It remains available through `MLXCEL_XLA_PRECISION` for measurement only. The pin cut the serve gate's batched throughput on `mamba2-130m` from 285.6 to 123.7 tok/s; single-sequence throughput at f32 versus f16 was not measured.
- **One real checkpoint.** Token exactness rests on `mamba2-130m` (`n_groups = 1`), 5 prompts of 40 tokens, and the synthetic fixture. Larger Mamba2 checkpoints were not run.
- **State memory and graph size.** The per-slot set is about 18.5 MiB per slot for `mamba2-130m` (computed), and each step writes a new set because the arguments are not donated. Compile time and graph size at `B = 8`, named as a risk in the issue, were not measured separately.
- **Single-sequence bundle without `state_import`** fails at the first decode rather than at load (LOW, accepted).
- **Unrecognised precision override on pinned configs** is silently ignored (LOW, pre-existing for MLA).
- **Serve worker failure mode** (any prefill or decode error stops the worker; the popped request is dropped) is pre-existing and unchanged.
- **CUDA and Metal not run.** Those are #3342 and #3343.

## 7. Learning Points

- When per-slot state joins per-slot KV, put it under the same validity flag. Two flags create combinations (KV valid, state not) that no test enumerates; one flag cleared before the first copy and set after the last makes every partial store invalid by construction.
- Starting every admit from a known zero state makes eviction free. If the admit path never reads a row's previous contents, cancel, disconnect and finish need no device work, and the isolation argument depends on one code path instead of every exit path.
- The #3371 rule generalizes: whatever the native layer does on failure, a request refused above it never triggers that cleanup. Invalidating at the very top of each entry, before validation, is the robust form.
- Recovery rules can differ by state kind on the same path. A KV row is append-only, so a retried step rewrites the same position; a recurrent row is a running summary, so continuing from a step of unknown progress cannot be checked by shape. Refusing every row until it is re-admitted is the rule that needs no reasoning about how far a failed step got.
- A serve gate is also a precision gate. Comparing batched output against single-sequence output exercised more prompts and more tokens than the one-prompt oracle check in #3371, and showed the single-sequence reference itself was wrong at f16. The batched stream was right and the reference was not; checking both against the fp32 oracle is what separated the two.
- A startup probe that runs on every checkpoint needs the most tolerant parser of any reader, and a strict check only where its result is consumed. The Gemma3n audio probe blocked Mamba2 serving without being related to it.
