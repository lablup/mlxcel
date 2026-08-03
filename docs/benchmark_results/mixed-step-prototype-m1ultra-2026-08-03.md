# Mixed prefill/decode step: chunked-prefill starvation, measured

Issue #908 / ADR 0005. M1 Ultra (64 GB), macOS 26.6, Metal.
`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`, `--parallel 8 --prefill-chunk-size 512`.

## What this measures, and why it is a counter comparison

#908 assumed tick alternation stalls decode while a chunked prefill runs, and
proposed a ragged fused step to remove the stall. The implementation found the
premise inverted: `decide_action` returns `Decode` whenever any sequence is
active, so a chunked prefill never interleaves with decode at all. It waits.

That makes the question "does the prefill advance while decode streams run?",
which is a progress question, not a latency one. So this measures counters
(`mlxcel_batch_prefill_chunks_total`, `..._mixed_steps_total`,
`..._decode_steps_total`) sampled every 5 s, rather than ITL percentiles.
Counters are the right instrument here for a second reason: the machine was
carrying load 12.9 (baseline) and 19.8 (mixed) from unrelated work, which would
make any absolute latency number meaningless. Whether a counter advances or is
pinned is not sensitive to that.

Workload: four 400-token decode streams start, then at t≈6 s one ~10k-token
prompt is admitted. 19 chunks at 512 tokens. Probe:
`scripts/bench/starvation_probe.sh` (see PR #1010).

## Result

| t | baseline chunks | baseline decode | mixed chunks | mixed decode |
|---|---|---|---|---|
| 5 s | 1 | 138 | 7 | 123 |
| 10 s | 1 | 207 | 12 | 128 |
| 15 s | 1 | 254 | 17 | 133 |
| 20 s | 1 | 322 | **19** | 194 |
| 25 s | 1 | 394 | 19 | 290 |
| 30 s | 7 | 399 | 19 | 381 |
| 40 s | **19** | 399 | 19 | 399 |

`mixed_steps_total` is 0 across the baseline arm and 18 in the mixed arm, which
confirms the arms are what they claim to be.

**Baseline: the prefill is pinned at chunk 1 for 20 s** while decode advances
138 → 394. It resumes only at t=30 s, once decode has drained to its 399-step
ceiling, and finishes at t=40 s. One chunk of 19 ran, then nothing, until the
decode batch emptied.

**Mixed: the prefill advances concurrently**, roughly one chunk per decode
step, and all 19 chunks are done by t=20 s. Decode still reaches 399. Nothing
was traded away for it.

Time to finish the admitted request's prefill: **40 s → 20 s**. Decode step
count at completion: unchanged.

## What this does and does not license

It does **not** rescue the ragged fused step. ADR 0005 rejects that on its own
terms: the ceiling is `D_lin / (C/P + D)`, 3-20% on these shapes, and it costs
a second forward signature in every family. Nothing here moves that number,
because the win above is not the fused-kernel term. Decomposed:

```
benefit(ragged) = benefit(co-schedule) + D_lin per chunk
```

The 40 s → 20 s is entirely `benefit(co-schedule)`. `MLXCEL_MIXED_STEP` happens
to deliver it because co-scheduling is a precondition for mixing, not because
mixing is doing the work. The same benefit is available from a fairness policy
in `decide_tick` with no forward-path change at all, which is what ADR 0005
recommends and what option (a) describes.

So the correct reading is: #908's remedy is rejected, and the symptom that
motivated it is real but has a different cause and a cheaper fix. The prototype
stays behind `MLXCEL_MIXED_STEP`, defaulted off, as the executable evidence for
that claim.

The starvation itself is a live serving defect independent of this epic: a
long prompt admitted into a busy batch makes no progress until the batch
drains, which is unbounded when streams keep arriving. It is pinned by
`chunked_prefill_starves_until_active_batch_drains` in
`src/server/batch/tick_policy.rs` and tracked separately for the fairness fix.

## Reproducing

```bash
export DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer
cargo build --release --features metal,accelerate
scripts/bench/starvation_probe.sh baseline
PORT=18994 scripts/bench/starvation_probe.sh mixed
```

Both arms were run on the same binary, back to back, single server process at a
time. The probe kills its server on every exit path.
