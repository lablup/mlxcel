# MTP drafter step profile, Qwen 3.8 27B on M5 Max

Issue #1185, Phase 0, second host. The M1 Ultra decomposition is in
[`mtp-drafter-step-profile-m1ultra-2026-08-17.md`](mtp-drafter-step-profile-m1ultra-2026-08-17.md);
this file measures the same arms on generation 17 hardware and reports where
the two disagree.

## Setup

| Field | Value |
|---|---|
| Host | Apple M5 Max, 40 GPU cores, 128 GB, macOS 26.6.1, Xcode 26.5 (17F42) |
| Load average during runs | 6.5 before the first arm, 5.9 after the second, recorded around each arm |
| Build | `cargo build --release --features metal,accelerate`, `main` at `33951285` |
| Target | `models/qwen3.8-27b-4bit` |
| Drafter | `models/qwen3.8-27b-mtp-bf16` |
| Command | `mlxcel generate --draft-kind mtp --draft-block-size 3 -n 120 --temp 0` |
| Prompt | The 120-token transformer-explanation prompt from #1182's reproduction comment |
| Exactness gate | **Failed, and overridden.** This host is generation 17, so the gate shipped in #1189 is fail-closed here and `MLXCEL_MTP_ALLOW_INEXACT=1` was required to engage MTP at all. These numbers therefore do **not** come from a shipping configuration. #1185's Blocker section asks for exactly this to be stated. |

The probe's own reading, logged at startup in both arms:

```
MTP exactness probe FAILED but MLXCEL_MTP_ALLOW_INEXACT is set: verify block
position 0 differs from the single-token chain in 165506 of 496640 logit bytes.
```

That is 33.3% of the logit bytes, against the 141k of 496k the M1 Ultra probe
reports at block 12.

Both arms produced identical acceptance (0.6667), identical
`emitted_per_verify` (2.3333) and the same 51 rounds, 102 proposed and 68
accepted tokens, which is the control that they ran the same work.

Because MTP is inexact on this host, the generated token stream differs from
the M1 Ultra run (51 rounds and acceptance 0.6667 here, 49 rounds and 0.7320
there). Per-round and per-step figures compare cleanly; run totals do not.

## Round split, unprofiled (the honest cost)

51 rounds, 120 generated tokens, `decode_ms` 2985.9, so **58.5 ms per round**.

| component | total ms | ms/round | share | M1 Ultra share |
|---|---:|---:|---:|---:|
| verify forward (T=3) | 1797.1 | 35.24 | 60.2% | 77.8% |
| `draft_block` | 544.2 | 10.67 | 18.2% | 7.9% |
| **accept hook** | 540.7 | 10.60 | 18.1% | 10.9% |
| verify finalize (rollback) | 8.2 | 0.16 | 0.3% | 0.8% |
| speculative walk, shared-kv re-arm | 0.1 | <0.01 | ~0% | ~0% |
| residual | 95.5 | 1.87 | 3.2% | 2.5% |

`prefill_seed_ms` was 164.8 and sits outside `decode_ms`, as on the other host.

### The share shift is not the finding. The absolute values are.

Reading the two share columns alone suggests the drafter merely grew in
relative weight because verify shrank. The per-round wall clock says something
stronger:

| component | M1 Ultra ms/round | M5 Max ms/round | speedup |
|---|---:|---:|---:|
| round total | 129.1 | 58.5 | 2.21x |
| verify forward (T=3) | 100.5 | 35.24 | **2.85x** |
| accept hook | 14.1 | 10.60 | 1.33x |
| `draft_block` | 10.3 | 10.67 | **0.97x** |

The 27B verify forward is 2.85 times faster on this host. The drafter block is
not faster at all. It is 3.6% slower, which is inside the run-to-run spread and
should be read as flat rather than as a regression.

So the drafter side goes from 18.8% of the round on M1 Ultra to 36.3% here, and
it does so by standing still while everything around it accelerates. That is
#1185's premise ("now the binding constraint on M5-class hardware") measured
rather than modeled.

## Drafter step split

`draft_block` ran 51 steps across the 51 rounds.

| component | profiled ms | share | M1 Ultra share | unprofiled ms |
|---|---:|---:|---:|---:|
| layers (pre-FC norms, decoder layers, final norm) | 422.99 | **73.5%** | 68.2% | 0.98 |
| LM head (borrowed 248,320-wide target head) | 124.90 | **21.7%** | 19.7% | 0.03 |
| `eval` + `item_i32` readback | 17.32 | 3.0% | 5.8% | 542.80 |
| id upload + target embedding lookup | 10.10 | 1.8% | 6.0% | 0.28 |
| last-position slice + `fused_sample` | 0.10 | 0.02% | 0.1% | 0.05 |
| **components total** | **575.41** | | | 544.14 |
| **wall-clock total** | **575.68** | | | 544.16 |
| **per step** | **11.29** | | | **10.67** |

Profiled components close against the wall-clock total to within 0.05%, and the
unprofiled column to within 0.004%.

### The two columns read the same way they do on M1 Ultra

The unprofiled column puts 99.75% of its measured component time in
`readback_ms`, because MLX is lazily evaluated and a drafter step synchronizes
exactly once. Its only load-bearing numbers are the totals, and the log line
reports `dominant="unattributed"` accordingly.

`MLXCEL_MTP_DRAFT_PROFILE=1` costs **0.62 ms per step, 5.8%** here, against
1.38 ms and 13.2% on M1 Ultra. Quote 10.67 ms as the step cost and the profiled
column only for attribution.

### The attribution transfers, and sharpens

Every component keeps its rank across the two hosts, and the two largest ones
gain share rather than losing it: layers 68.2% to 73.5%, LM head 19.7% to
21.7%. Together they are 95.2% of the profiled step here, against 87.9% there.

Per step, in absolute terms, no drafter component is faster on this host:

| component | M1 Ultra ms/step | M5 Max ms/step |
|---|---:|---:|
| layers | 8.06 | 8.29 |
| LM head | 2.33 | 2.45 |

A single-token step through an 829 MB bf16 drafter does not get faster on
hardware that runs the 27B verify forward 2.85 times faster. This file does not
establish why. What it does establish is that the cost is not a property of the
target model's speed, so it will not be absorbed by faster hardware.

### Non-step drafter forwards

102 forwards over the run, profiled at 499.78 ms of layers and 8.65 ms of ids,
against the same arm's independently measured `accept_hook_ms` of 559.58 ms.

**The cross-check is looser here and should not be quoted as agreement.** The
two sides differ by 51.1 ms, 9.1% of the hook, where the M1 Ultra run closed to
0.7%. The direction is consistent with the known gap (`set_seed_from_hidden`
uses the unprofiled sampling helper, so the hook's projection and readback sit
inside `accept_hook_ms` without a split), but a gap that grows by an order of
magnitude across hosts is not explained by that alone. Instrumenting the hook's
projection and readback, already named as the obvious next increment, would
settle it.

## What this says about #1185's phase ranking

The M1 Ultra measurement disagreed with parts of the issue's cost model. This
one disagrees the same way, more strongly:

- **Phase 3 (4-bit drafter weights) targets the largest component, and it is
  larger here.** Layers are 73.5% of the step.
- **Phase 2b (reduced draft vocabulary) targets the second largest, also
  larger here.** The borrowed 248,320-wide LM head is 21.7% of the step.
- **Phase 1 (remove the host round trip) is worth less here, not more.**
  Readback is 3.0% profiled against 5.8% on M1 Ultra, and the pipelining effect
  it relates to is 5.8% against 13.2%. The faster host has less host round trip
  to remove, not more.

The ranking is therefore the same on both hosts, and the gap between the top
two phases and Phase 1 widens on the hardware #1185 is aimed at.

## Reproducing

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cargo build --release --features metal,accelerate --bin mlxcel

PROMPT="Explain in detail how a transformer neural network works, covering \
attention, feed-forward layers, normalization, and training."
for mode in 0 1; do
  MLXCEL_MTP_ALLOW_INEXACT=1 MLXCEL_MTP_DRAFT_PROFILE=$mode RUST_LOG=info \
  ./target/release/mlxcel generate -m models/qwen3.8-27b-4bit \
    --draft-model models/qwen3.8-27b-mtp-bf16 --draft-kind mtp \
    --draft-block-size 3 -p "$PROMPT" -n 120 --temp 0 2>&1 \
  | grep -E "round-loop diagnostics|drafter step profile"
done
```

`MLXCEL_MTP_ALLOW_INEXACT=1` is required on generation 15 and later. Without it
the gate refuses to engage MTP and the run falls back to classic decode, which
produces no round-loop diagnostics at all.

Update 2026-08-22: the paragraph above describes the pre-#1199 gate this
profile was measured under, and the recipe no longer reproduces these
kernels. Since #1199 the gate retries a failing probe with `qmv_wide`
disabled before the override is consulted, so on generation 15+ the same
command now engages MTP on the narrow kernel with byte-identity kept, and
the override is inert. Reproducing this profile's fast-kernel arm needs
`MLXCEL_QMV_WIDE=1` alongside `MLXCEL_MTP_ALLOW_INEXACT=1`; see
`qmv-wide-pin-tax-m3ultra-2026-08-22.md` for the live verification of all
four recipes.
