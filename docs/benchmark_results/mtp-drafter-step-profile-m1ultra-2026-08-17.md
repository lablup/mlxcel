# MTP drafter step profile, Qwen 3.8 27B on M1 Ultra

Issue #1185, Phase 0: attribute the drafter step cost before optimizing it.

## Setup

| Field | Value |
|---|---|
| Host | Apple M1 Ultra, 128 GB, macOS 26.6.1, Xcode 26.6.0 |
| Load average during runs | 6.5 to 6.9 (1 min), recorded before and after each arm |
| Build | `cargo build --release --features metal,accelerate`, `main` at `ffde948e` plus this branch |
| Target | `models/qwen3.8-27b-4bit` |
| Drafter | `models/qwen3.8-27b-mtp-bf16` |
| Command | `mlxcel generate --draft-kind mtp --draft-block-size 3 -n 120 --temp 0` |
| Prompt | The 120-token transformer-explanation prompt from #1182's reproduction comment |
| Exactness gate | Passed without an override. M1 Ultra is generation 13, so `MLXCEL_MTP_ALLOW_INEXACT` was **not** set and these numbers come from a shipping configuration. |

Both arms produced identical acceptance (0.7320) and identical `emitted_per_verify` (2.4286) over 49 rounds, which is the control that they ran the same work.

## Round split, unprofiled (the honest cost)

49 rounds, 120 generated tokens, `decode_ms` 6326.6, so **129.1 ms per round**.

| component | total ms | ms/round | share |
|---|---:|---:|---:|
| verify forward (T=3) | 4922.9 | 100.5 | 77.8% |
| **accept hook** | **691.7** | **14.1** | **10.9%** |
| `draft_block` | 502.4 | 10.3 | 7.9% |
| verify finalize (rollback) | 53.0 | 1.1 | 0.8% |
| speculative walk, shared-kv re-arm | 0.1 | <0.01 | ~0% |
| residual | | 3.2 | 2.5% |

**The accept hook is larger than the entire `draft_block`.** It had no counter before this change, so it sat in the residual. It is not bookkeeping: for the Qwen 3.5 MTP drafter it is a second drafter forward per round, appending the accepted tokens against the target's true hidden and precomputing the next round's seed.

## Drafter step split

`draft_block` ran 48 steps across the 49 rounds.

| component | profiled ms | share | unprofiled ms |
|---|---:|---:|---:|
| layers (pre-FC norms, decoder layers, final norm) | 387.09 | **68.2%** | 7.0 |
| LM head (borrowed 248,320-wide target head) | 111.76 | **19.7%** | 0.2 |
| id upload + target embedding lookup | 34.26 | 6.0% | 2.6 |
| `eval` + `item_i32` readback | 32.76 | 5.8% | 491.3 |
| last-position slice + `fused_sample` | 0.65 | 0.1% | 0.3 |
| **components total** | **566.53** | | 501.35 |
| **wall-clock total** | **567.78** | | 501.57 |
| **per step** | **11.83** | | **10.45** |

Profiled components close against the wall-clock total to within 0.2%.

### Read the two columns differently

MLX is lazily evaluated and a drafter step synchronizes exactly once, at the `eval` before the sampled id is read back. So in the unprofiled column **98% of the measured component time lands in `readback_ms`** and the other four buckets measure graph construction, not the work they name. That column's only load-bearing numbers are the totals. The log line says so and reports `dominant="unattributed"` rather than naming a winner.

`MLXCEL_MTP_DRAFT_PROFILE=1` evaluates each component before the next begins, which makes the split real and costs **1.38 ms per step, 13.2%**, because the syncs break pipelining. Quote 10.45 ms as the step cost and the profiled column only for attribution.

### Non-step drafter forwards

The accept hook and the prefill seed also reach `forward_hidden_stack`: 98 such forwards over the run, profiled at 731.2 ms of layers and 36.7 ms of ids, against the round loop's independently measured `accept_hook_ms` of 762.8 ms in the same arm. Those two agreeing from opposite directions is the cross-check that the bucket separation is correct.

**Not attributed:** the non-step bucket's LM head, sample and readback read 0, because `set_seed_from_hidden` uses the unprofiled sampling helper. The hook's projection and readback therefore sit inside `accept_hook_ms` without a split. Instrumenting them is the obvious next increment and was left out here rather than reported as a zero.

## What this says about #1185's phase ranking

The issue ranked the phases from a cost model. On this hardware the measurement disagrees with parts of it:

- **Phase 3 (4-bit drafter weights) targets the largest component.** Layers are 68.2% of the step.
- **Phase 2b (reduced draft vocabulary) targets the second largest.** The borrowed 248,320-wide LM head is 19.7% of the step, in line with the issue's observation that it is 46% of the drafter's memory traffic.
- **Phase 1 (remove the host round trip) targets a small one.** Readback is 5.8% profiled. The related pipelining effect is worth 13.2%, but that is what pipelining already buys, not what removing the round trip would add.
- **The accept hook is not in any phase.** At 14.1 ms per round it is larger than `draft_block` and currently unaddressed by Phases 0 through 4.

## Limits of this record

- **One host, one pairing, one prompt, one repetition per arm.** The two arms are internally consistent (identical acceptance, components closing against wall-clock) but this is not a medians-of-five protocol, because Phase 0's question is attribution rather than a speedup claim.
- **This is an M1 Ultra.** #1185's profile was taken on an M5 Max, where the round is 2.41x faster overall while the drafter step did not get faster at all. Whether the *split* transfers is unmeasured, and the phase ranking above should be re-taken there before it is acted on.
- The machine carried background load throughout. Within-run shares survive that; absolute milliseconds do not.
