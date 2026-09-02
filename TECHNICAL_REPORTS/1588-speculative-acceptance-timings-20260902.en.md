# Technical Report: PR #1588 - feat(server): report per-request speculative acceptance

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Completed (unit and route coverage green; the real target/drafter end-to-end check is run by the merge orchestrator, see Appendix C)
**Languages**: Rust
**Risk Level**: Low (additive optional response fields; the non-speculative decode path is unchanged in both cost and wire shape)

---

## Executive Summary

When a drafter serves a request, the scheduler already computes that request's verify rounds and its proposed and accepted draft-token totals, and then dropped all three into a `tracing` line. A client had no way to learn whether speculation helped its own request, which is what an operator tuning `--draft-max` / `--draft-block-size` and a client A/B-ing the speculative and classic paths both need. PR #1588 carries the counters on `GenerationResult` and surfaces them on the native `/completion` `timings` object under llama-server's own `draft_n` / `draft_n_accepted` names, plus two mlxcel keys (`draft_rounds`, `draft_kind`), and as a top-level `timings` object on `/v1/chat/completions`. The block is present only when a verify round actually ran, so "no drafter" and "the drafter accepted nothing" stay distinguishable rather than collapsing into a row of zeros.

---

## 1. Problem Statement

### 1.1 Background

Three code paths in `mlxcel-server` own a speculative request's whole round loop, and each of them ends up holding an exact acceptance summary for that one request:

- the DFlash B=1 burst, where `DFlashGenerator::run` returns a `DFlashDiagnostics` carrying `rounds`, `proposed_tokens` and `accepted_tokens` (`src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs`);
- the MTP B=1 burst, where the generator returns an `MtpAcceptanceSummary` with `rounds`, `proposed_tokens` and `accepted_draft_tokens`;
- the tick-cooperative MTP slice (#734), whose session spans every scheduler tick the request occupies and whose `finish_session` therefore returns the run's totals already accumulated.

All three consumed the summary for two internal purposes and then discarded it. The DFlash arm logged it at `info` and fed the aggregate Prometheus `spec_decode_*` counters; the MTP arms converted it into an `MtpBurstProfile` for the adaptive policy (#333). `GenerationResult`, the value every response shape is built from, carried no speculative field at all, so `GenerateEvent::Done` reached the HTTP layer with the acceptance already thrown away.

### 1.2 What that cost a client

`/metrics` exposes only the aggregate MTP policy state, which answers "is speculation paying for itself across this deployment" and cannot answer "did it pay for itself on my request". The two questions come apart exactly where it matters: acceptance is a property of the prompt's entropy far more than of the deployment, so a per-deployment mean is the wrong number for a client deciding whether to keep the drafter on for its own workload. A client A/B-ing the two paths was left comparing wall-clock times with no way to attribute a difference to acceptance rather than to load.

### 1.3 The issue's file map had drifted

The issue body pointed at `TimingInfo` in `src/server/types/response.rs:552` as the timings type to extend, and at three `finish_with_cache` call sites to thread the counters through. Neither matched the tree at implementation time: there is no `TimingInfo` in `response.rs` (the native timings type is `NativeTimings` in `src/server/types/native_completion.rs`), and the three finalize sites the issue named have since been funnelled through one method, `SequenceInfo::take_generation_result` (`src/server/batch/sequence.rs`). The second drift is the more useful one, because it turns a three-site thread into a one-parameter change.

### 1.4 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A per-token allocation or clone added to the batch decode hot path | High if it happened | Eliminated by construction: the counters are `Copy`, computed once per request at finalize, and the classic path passes `None` |
| A non-speculative response body changes shape and breaks an existing client | High if it happened | Eliminated: `None` serializes to no key at all, pinned by tests on both the native and the chat shapes |
| A mid-stream frame reports a running `draft_n` that reads as a total | Medium | Avoided: only the frame that already carries `finish_reason` (chat) or `stop: true` (native) carries the block |
| Reporting zeros for a request no drafter served, making "off" look like "on but useless" | Medium | Avoided: `SpeculativeStats::from_counts` returns `None` below one round |

---

## 2. Technical Review

**Security.** No new input is parsed and no request field is added; the change is write-only on the response side. The counters are integers derived from the server's own round loop, carry nothing request-identifying, and cannot be influenced by a client beyond the sampling parameters it already controls. The one information-disclosure question worth asking is whether `draft_kind` leaks deployment configuration: it names one of three fixed drafter families, which `GET /props` already reports under `speculative` (draft model basename, kind override, `n_max`), so the field discloses nothing that endpoint did not.

**Performance.** The non-speculative path is the one to protect, since it is every request on a deployment without `--draft-model`. `SpeculativeStats` is four machine words and `Copy`; `take_generation_result` gained one `Option` parameter that the classic call sites pass as `None`, which is a move of a `None` into a struct field. No allocation, no clone, no branch on the per-token path: the value is produced once, at finalize, by code that had the summary in hand anyway. The speculative paths read a `Copy` summary they were already reading for the policy profile, so they add no work either.

**Correctness.** The single funnel is what makes the coverage claim checkable: every MLX-scheduler response is built by `take_generation_result`, and the parameter is therefore either supplied or explicitly `None` at each of its five call sites, with no third possibility for a path to silently fall through.

---

## 3. Technical Decisions

### 3.1 A parameter on `take_generation_result`, not a field on `SequenceInfo`

The issue's plan threaded a new `finish_with_cache_and_speculative` through three finalize sites. The obvious alternative once `take_generation_result` was found to be the single funnel was a field on `SequenceInfo`, set by whichever round loop ran and read at finalize.

The parameter won on two grounds. First, `SequenceInfo` is constructed as a struct literal in fifteen places across production and test code, so a field means fifteen edits and a sixteenth every time a test file is added, for a value only three of those sites could ever set. Second, and more important, a field would be *mutable state a request carries*, which invites a future path to set it early and a later path to read it after the request's shape has changed; a parameter can only be supplied at the moment the result is built, by the code that just finished the round loop. The five call sites are `speculative_burst.rs` (both bursts and the slice, through `finalize_burst_stream`), `scheduler/decode_tick.rs`, `scheduler/prefill.rs`, and two tests.

### 3.2 `Option<SpeculativeStats>` gated on rounds, not a zeroed struct

`SpeculativeStats::from_counts` returns `None` when `draft_rounds == 0`. A request that finished inside prefill (immediate EOS, or `n_predict: 1`) never gave the drafter a round to run, and reporting it as `{"draft_n": 0, "draft_n_accepted": 0}` would tell a client a drafter ran and proposed nothing. That is precisely the distinction the block exists to make, so the gate lives in the constructor rather than at each of the three call sites, where it could drift.

The MTP arms already had a similar filter for the adaptive policy, `rounds > 0 || probe_rounds > 0`. The client-facing filter is deliberately narrower: a run consisting only of classic-step probe rounds (#736) drafted nothing, and describing it as a speculative request would misdescribe it, even though it is a legitimate timing sample for the policy.

### 3.3 `DrafterKind` in the model, `&'static str` on the wire

The issue specified `draft_kind: String`. Storing `DrafterKind` instead keeps the value `Copy` and allocation-free through the scheduler, and defers the string to the wire type, where `DrafterKind::as_str` returns a `&'static str` and the whole path stays allocation-free. It also makes the value type-checked rather than a string that could be spelled three ways, and keeps `DrafterKind::as_str` the one place the canonical names live, matching `--draft-kind`.

### 3.4 Upstream's field names, upstream's gate, and two additions declared as a divergence

`draft_n` and `draft_n_accepted` are llama-server's own optional pair: upstream's `result_timings` appends them to `timings` only when `draft_n > 0`. Reproducing both the names and the gate means a client that already reads llama-server timings reads mlxcel's unchanged, which is the point of the compatibility surface.

The two mlxcel keys needed a justification rather than a preference. `draft_rounds` is not recoverable from upstream's pair, and without it the mean accepted length per round, `(draft_n_accepted + draft_rounds) / draft_rounds`, cannot be computed, which is the figure an operator tuning block size actually wants. `draft_kind` has no upstream analogue because upstream has one draft-model concept where mlxcel has three drafters. Both are additive keys on an object upstream already leaves optional, so nothing a b10621 client reads changes.

Per epic #1431's rule, that difference is recorded as a checked `divergence` entry on both native route entries in `compat/llama-server/b10621/routes.json`, with its rationale and revisit condition, rather than as free text in `notes`. The entries' prose count of permanent differences was updated from two to three in the same edit, so the machine-checked array and the prose beside it cannot disagree.

### 3.5 The chat `timings` object is the draft block alone

llama-server puts a `timings` object on its own OpenAI chat responses, which is why the field is spelled `timings` rather than something mlxcel-specific. mlxcel's carries the four draft keys and only those. Emitting the full nine-key block would mean the prompt and predicted rates appear and vanish with the drafter, since the whole object is gated on speculation being active, and a wire shape that flickers with an unrelated deployment setting is worse than one that is either absent or complete.

### 3.6 Flatten rather than four `Option` fields on `NativeTimings`

`NativeTimings` carries `#[serde(flatten)] Option<SpeculativeTimings>`, attached by a `with_speculative` builder. Flattening puts the draft keys directly on the `timings` object, which is where upstream puts its pair, while keeping one type that the chat routes reuse whole. The builder keeps `NativeTimings::new`'s signature untouched: its arithmetic was measured against the pinned b10621 binary and its five parameters are pinned by existing tests, so growing it would have rippled through `native_completion_tests.rs` for no gain.

---

## 4. Implementation Details

### 4.1 The counters and their single funnel

`SpeculativeStats` (`src/server/model_provider.rs`) holds `draft_kind: DrafterKind`, `draft_rounds`, `draft_n` and `draft_n_accepted`, and `GenerationResult` gains `speculative: Option<SpeculativeStats>`. `SequenceInfo::take_generation_result` takes the block as its third parameter and assigns it to the result.

### 4.2 The three producing paths

| Path | Source of the counters | Where the block is built |
|---|---|---|
| DFlash B=1 burst | `DFlashDiagnostics { rounds, proposed_tokens, accepted_tokens }` | `run_dflash_on_qwen35`, beside the existing `DFlash diagnostics` log line |
| MTP B=1 burst | `MtpAcceptanceSummary { rounds, proposed_tokens, accepted_draft_tokens }` | `run_mtp_burst`, beside the `MtpBurstProfile` the adaptive policy gets |
| Tick-cooperative MTP slice | the same summary, from `generator.finish_session` | `BatchScheduler::finalize_speculative_slice` |

Both summary types are `Copy`, so each site reads the same value the policy path reads rather than re-deriving it. The slice path needs no accumulation across slices: the session outlives every tick the request occupies, so `finish_session` already returns the run's totals. It resolves the kind through `SpeculativeDispatch::drafter_kind()` rather than assuming one, and `Option::zip` makes a dispatch that runs no drafter produce no block instead of a default.

The DFlash driver's return type changed from a three-element tuple to a named `DFlashTargetRun` struct, so the diagnostics can escape the function without a fourth tuple member that a later reader would have to count positions to identify.

### 4.3 Paths that deliberately report nothing

The default-off B>1 batched burst passes `None`: its round loops return per-row tokens and no per-row acceptance counters, and a window-wide figure attributed to one row would be wrong. The disaggregated router front reports nothing either, because it assembles its stream from the prefill/decode handoff protocol rather than from a `GenerationResult`, and that protocol carries a generated-token count and no acceptance counters; surfacing the block there is a handoff-protocol change, not a response-shaping one. Both are recorded where a reader would look: in the code, and in `docs/speculative-acceptance.md`.

### 4.4 The wire

`SpeculativeTimings` (`src/server/types/native_completion.rs`) serializes `draft_n`, `draft_n_accepted`, `draft_rounds`, `draft_kind` in that order, upstream's pair first. It flattens into `NativeTimings` and is the whole `timings` object on `ChatCompletionResponse` and `ChatCompletionChunk`, both of which gained `#[serde(skip_serializing_if = "Option::is_none")] timings` plus a `with_speculative_timings` builder in the style of the existing `with_cached_tokens`.

Route wiring: `NativeOutcome` gained a `speculative` member, so the native non-streaming body and the final streaming frame are built through the one `build_native_response` function they already shared and cannot drift. The per-token `timings_per_token` frames are untouched and carry the nine base keys only. On the chat route the block is attached at the four non-streaming return sites and on the finish chunk of a stream.

---

## 5. Learning Points

### 5.1 An absent key and a zero are different answers, and only one of them is honest

The temptation in a change like this is to make the fields non-optional and report zeros when no drafter ran, because it simplifies every consumer. It also destroys the only distinction the numbers are for: a client seeing `draft_n_accepted: 0` cannot tell whether speculation is off or whether it is on and useless, and those call for opposite actions. Gating in the constructor rather than at the call sites is what keeps the rule from drifting as new producing paths are added.

### 5.2 Compatibility is names *and* gates

Matching llama-server's `draft_n` spelling while emitting it unconditionally would still have broken a strict client, because upstream's key is absent below one round and a client may treat presence as the signal. Reproducing the gate was as load-bearing as reproducing the name.

### 5.3 A funnel found late is worth more than the plan it replaces

The issue's plan predated the refactor that made `take_generation_result` the single finalize funnel, and following it would have produced a second `finish_with_cache` variant plus three threaded call sites. Checking the tree first turned that into one parameter with five call sites, of which three are one-word edits. The general form: when an issue's implementation plan names several parallel call sites, check whether they have since been unified before threading anything through them.

---

## 6. Further Learning

### Key Terms

- **Draft / verify round**: one drafter forward proposing `k` tokens, followed by one target forward verifying them. The unit `draft_rounds` counts.
- **Acceptance**: how many of a round's proposals the target kept. `draft_n_accepted / draft_n` is the fraction of drafted work that survived; `(draft_n_accepted + draft_rounds) / draft_rounds` is the tokens emitted per target forward, which is what drives throughput.
- **Bonus token**: the token a round emits from the target's own verify logits regardless of how many proposals were accepted. It is why the mean accepted length per round adds `draft_rounds` before dividing.
- **`result_timings`**: llama-server's timings struct, whose optional `draft_n` / `draft_n_accepted` pair this change reproduces.

### Related PRs/Issues

- #1314: this issue.
- #1431: the llama-server b10621 compatibility epic, whose divergence-as-a-checked-field rule this change follows.
- #1477 / #1441 / #1466 / #1525: the native `/completion` route's prior compatibility work, whose route entries this change amends.
- #734: the tick-cooperative MTP slice, whose session accumulation this change relies on.
- #333 / #736: the adaptive MTP policy and its classic-step probe rounds, whose filter this change deliberately narrows for the client-facing block.

---

## 7. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 25 |
| Lines added | ~769 |
| Lines removed | ~34 |
| New public types | 2 (`SpeculativeStats`, `SpeculativeTimings`) |
| New tests | 10 |

### Changes by Category

- **Model / scheduler**: `src/server/model_provider.rs`, `src/server/model_worker.rs`, `src/server/batch/sequence.rs`, `src/server/batch/speculative_burst.rs`, `src/server/batch/scheduler.rs`, `src/server/batch/scheduler/decode_tick.rs`, `src/server/batch/scheduler/prefill.rs`, `src/server/florence2_worker.rs`.
- **Wire types**: `src/server/types/native_completion.rs`, `src/server/types/response.rs`, `src/server/types/stream.rs`.
- **Routes**: `src/server/routes/native_completion.rs`, `src/server/routes/chat.rs`, `src/server/router_front.rs` (documentation of the deliberate gap only).
- **Tests**: `src/server/types/native_completion_tests.rs`, `src/server/routes/native_route_tests.rs`, `src/server/batch/speculative_slice_tests.rs`, `src/server/model_provider_test_support.rs`, plus mechanical `None` at existing call sites.
- **Docs and manifest**: `docs/speculative-acceptance.md`, `docs/llama-server-compat.md`, `compat/llama-server/b10621/routes.json`.

---

## 8. Follow-up Actions

### Monitoring Required

- The end-to-end identity `predicted_n == draft_n_accepted + draft_rounds` (within one for the EOS round) is asserted only by the orchestrator's real-checkpoint run, not by a unit test, because it needs a real drafter. A future change to how a round's bonus token is emitted would break it silently in the unit lane.

### Future Improvements

- The disaggregated router front reports nothing (4.3). Carrying the counters over the handoff protocol would close the last serving path, at the cost of a protocol field.
- The B>1 batched burst would need per-row acceptance counters out of its round loops before it could report anything. It is off by default today, so this is not urgent.
- Per-token draft and verify timing on the wire (`draft_time_ms`, `verify_time_ms`) was explicitly out of scope for #1314. The diagnostics already carry both, so the plumbing exists if a case for exposing them appears.

---

## Appendix

### A. Test Results

| Suite | Result |
|---|---|
| `--lib server::batch::speculative` | 87 passed |
| `--lib server::types` | 98 passed |
| `--lib native_route_tests` | 32 passed |
| `--lib stream_route_tests` | 9 passed |
| `--lib server::routes::chat` | 48 passed |
| `--lib server::batch::scheduler` | 137 passed |
| `--lib server::model_provider` | 57 passed |
| `--lib server::llama_compat_tests` | 3 passed |
| `--lib server::batch::stop_sequence_tests` | 8 passed |
| `--lib server::max_tokens_route_tests` | 7 passed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `cargo check --profile test-fast --features metal,accelerate --bins` | clean |
| `make verify-llama-compat` | passed, including its own negative cases |

### B. What the new tests pin

- `timings_carries_no_draft_keys_without_a_drafter`: nine keys, unchanged, no `draft_*` key present.
- `timings_carries_the_b10621_draft_pair_for_a_speculative_request`: thirteen keys, upstream's pair plus the two extensions, base keys still reporting what they did.
- `a_zero_round_speculative_run_reports_no_draft_block` and `every_drafter_kind_renders_its_canonical_name`: the gate and the three canonical names. Only two of the three are reachable today: `internal-mtp` resolves to the classic dispatch, which the burst gate declines, so no request is served speculatively under it and none reports a block.
- `the_done_result_carries_the_acceptance_counters_of_a_drafted_request`: drives the real two-round generator script and asserts the `Done` event carries the counters the round loop produced, with `draft_rounds > 0` and `0 < draft_n_accepted <= draft_n`.
- `the_done_result_reports_no_acceptance_for_an_undrafted_finish`: the classic-path contract.
- Route level: the four keys on `/completion`, `/completions` and `/v1/chat/completions`; their absence without a drafter on both shapes; and that only the `finish_reason` chunk of a chat stream carries `timings`.

### C. Post-merge validation (orchestrator)

```bash
./target/release/mlxcel-server \
  -m models/mlx/qwen3.5-4b-4bit \
  --draft-model models/mlx/qwen3.5-4b-dflash \
  --draft-kind dflash \
  --draft-block-size 16 \
  --port 8080

curl -s localhost:8080/completion -H 'content-type: application/json' \
  -d '{"prompt":"Write the numbers one to thirty in words.","n_predict":96,"temperature":0}' \
  | jq '.timings | {draft_kind, draft_rounds, draft_n, draft_n_accepted, predicted_n}'

curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.5-4b-4bit","messages":[{"role":"user","content":"Write the numbers one to thirty in words."}],"max_tokens":96,"temperature":0}' \
  | jq '.timings'
```

Expected: `draft_kind == "dflash"`, `draft_rounds > 0`, `0 < draft_n_accepted <= draft_n`, `predicted_n == draft_n_accepted + draft_rounds` within one for the EOS round, and the same figures in the server's `DFlash diagnostics` log line for that request. The chat body's `timings` carries those four keys and only those. Restarting without `--draft-model` must answer a `/completion` `timings` with none of the four keys and a chat body with no `timings` key.
