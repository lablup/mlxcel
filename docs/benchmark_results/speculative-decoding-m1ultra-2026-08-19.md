# Speculative decoding (MTP) on M1 Ultra, 2026-08-19

The M5 Max rows in `README.md` and `docs/benchmarks.md` were remeasured on
2026-08-18 with the acceptance diagnostics and the contention guard, and M3
Ultra rows were added on 2026-08-19. This is the same protocol run on M1
Ultra, making it the third host and the first one on Apple GPU generation 13.
It also answers a question neither of the other two sweeps could: whether the
unprobed Gemma 4 exactness gate (#1188) lets a real divergence through.

Headline: every ratio falls again on M1 Ultra, the prose row becomes an
outright regression, and the Qwen pairing is a wash. Acceptance is not the
reason. A block-4 verify round costs 2.70 classic decode steps here against
1.50 on M3 Ultra and 1.27 on M5 Max, and that single quantity orders all three
hosts.

It also settles how to read the M3 Ultra result. That sweep observed its
ratios falling while both arms got faster, and read the fall as the classic
baseline gaining more than the MTP arm did. The arithmetic is right, but M1
Ultra has both arms *slower* than M5 Max and the ratios fall further still, so
the direction the arms moved is not the mechanism. The round cost is, and it
reproduces the M3 Ultra ordering without reference to either arm's absolute
speed.

## Environment

| Field | Value |
|---|---|
| Host | Mac Studio, Apple M1 Ultra, 128 GB unified memory, macOS 26.6.1 (25G76) |
| Build | `cargo build --release --features metal,accelerate` |
| Branch | `bench/m1-ultra-speculative` at `a1d459e4` (`feat/mtp-acceptance-diagnostics`), plus the `INDEXER_EXTRA_NAMES` addition committed with this record |
| Harness | `scripts/bench_speculative.sh --reps 4`, eight samples per arm, ABBA blocks, two warm-ups discarded |
| Target | `models/gemma-4-12b-it-4bit`, `models/qwen3.8-27b-4bit` |
| Drafter | `models/gemma-4-12b-it-assistant-4bit`, `models/qwen3.8-27b-mtp-4bit` |
| Sampling | `--temp 0` |

The host is somebody's desktop, so the sweep ran through
`scripts/with_indexers_paused.sh` with the then-five default indexers plus
Outlook, Teams, Telegram and KakaoTalk suspended for its duration. `netbird`
runs as root and could not be signalled; it sat at 1.9% and never approached
the gate. `mdworker` was **not** in the default list while these rows were
measured, which is what cost the enumeration rerun below; it was added to the
wrapper afterwards.

## Results

Gemma 4 Unified 12B + 4-bit assistant. The M5 Max column is the 2026-08-18
measurement carried over for comparison, not remeasured here.

| Output | Tokens | Block | acceptance | emitted/verify | classic | MTP | M1 Ultra | M5 Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| enumeration | 400 | 4 | 0.997 | 3.990 | 34.2 | 50.4 | **1.48x** | 3.14x |
| source code | 300 | 5 | 0.815 | 3.934 | 34.9 | 43.5 | **1.25x** | 2.79x |
| prose | 400 | 5 requested, 4 effective | 0.525 | 2.574 | 34.5 | 32.7 | **0.95x** | 1.90x |

Qwen 3.8 27B with its own 4-bit MTP head, the source-code prompt, 300 tokens:

| Path | acceptance | emitted/verify | decode tok/s | speedup |
|---|---:|---:|---:|---:|
| classic decode (no drafter) | n/a | n/a | 23.8 | 1.00x |
| MTP, block 3 (the drafter's declared width) | 0.855 | 2.694 | 23.4 | **0.98x** |

## Why the hosts differ, in units that survive their different clocks

Dividing a round's wall time by the host's own classic decode step removes the
clock difference and leaves what the verify actually costs:

| Host | Block | round cost in classic steps | break-even emitted/verify |
|---|---:|---:|---:|
| M5 Max | 4 | 1.27 (enumeration), 1.29 (prose) | ~1.28 |
| M3 Ultra | 4 | 1.50 (enumeration), 1.51 (prose) | ~1.51 |
| M1 Ultra | 4 | 2.70 (enumeration), 2.72 (prose) | ~2.71 |
| M1 Ultra | 5 | 3.15 (source code) | ~3.15 |

Two prompts with nothing in common agree on the round cost to within 1% on
each host. That is the control that this is a property of the host and the
block width rather than of the prompt, and it is what licenses treating the
speedup as emitted-per-verify divided by round cost. The M5 Max and M3 Ultra
rows are derived from their published classic, MTP and emitted-per-verify
figures rather than measured here; only the M1 Ultra rows come from this
sweep.

Acceptance is the other factor, and it is nearly host-independent: the
enumeration rows read 0.997 with 3.990 emitted per verify on both machines, to
three digits. The source-code and prose rows differ slightly between hosts
(0.815 against 0.784, 0.525 against 0.489) because the two hosts' generations
diverge at some position and the continuations then differ; the enumeration
prompt is predictable enough that they do not.

So M1 Ultra needs 2.71 tokens per verify at block 4 to break even. Enumeration
delivers 3.990 and gains 48%. Prose delivers 2.574, lands just under the line,
and loses 5%. M5 Max needs 1.28 and M3 Ultra 1.51, which every prompt on those
hosts clears comfortably.

The mechanism is the `use_qmv_wide` split documented in
`src/models/speculative_exactness.rs`: from Apple GPU generation 15 a
quantized projection at `M >= 2` runs one wide pass, while generation 13 has
no such path and runs the block as narrow passes whose cost grows with the
width. M1 Ultra pays nearly per-position for the verify that the generation
15+ hosts amortise. That the two generation 15+ hosts differ from each other
as well, 1.28 against 1.51, is not explained by this split and was not
investigated.

The Qwen pairing is the same story at a worse starting point. Its acceptance
is the highest measured here and it still does not clear the round cost. The
2026-08-16 record for this pairing on this host measured 0.59x to 0.70x with
the bf16 drafter; quantizing the drafter to 4-bit (#1185 Phase 3) moved it to
break-even and no further. Nothing here argues for enabling it on generation
13. Two caveats the other hosts carry do not apply: generation 13 never takes
`qmv_wide`, so the probe passes without a fallback and no 17 to 20% is being
paid, and the 8.2% MTP spread M5 Max showed on this pairing did not appear
here either (0.2% classic, 1.3% MTP).

## Block width, and what the round cost does across it

Both pairings swept with `scripts/bench_block_width.sh`, eight rounds per
width, widths interleaved and the starting width rotated per round so any
drift spreads across the table instead of pooling at one end. Same code
prompt, 300 tokens, same wrapper.

Gemma 4 12B + 4-bit assistant, classic arm 34.86:

| width | decode tok/s | spread | acceptance | emitted/verify | round cost | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 43.3 | 1.9% | 0.876 | 2.743 | 2.21 | 1.242x |
| 4 | 42.9 | 3.9% | 0.799 | 3.398 | 2.76 | 1.231x |
| 5 | 43.0 | 3.0% | 0.815 | 3.934 | 3.19 | 1.234x |
| 6 | 41.4 | 1.0% | 0.760 | 4.153 | 3.50 | 1.188x |
| 8 | 38.7 | 1.2% | 0.669 | 4.530 | 4.08 | 1.110x |
| 10 | 35.9 | 1.3% | 0.614 | 5.155 | 5.01 | 1.030x |
| 12 | 33.8 | 3.2% | 0.519 | 5.155 | 5.32 | 0.970x |

Widths 3, 4 and 5 are tied inside their spreads, so this host has no peak to
name, only a band and then a fall from 6 onwards. Width 5 reads 43.0 against
the 43.5 the published block-5 row measured, 1.1% apart and inside the spread,
which is the cross-check that the sweep and the row are the same measurement.
Width 12 is a **regression**, 0.97x, on the one prompt that gains at every
other width; no width in either generation 15+ host's range does that. The 10
and 12 rows share an emitted-per-verify of 5.155 because at 300 tokens both
land on 58 rounds, so read that tail as coarse.

Qwen 3.8 27B + its 4-bit MTP head, classic arm 23.79:

| width | decode tok/s | spread | acceptance | emitted/verify | round cost | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 23.6 | 1.2% | 0.935 | 1.929 | 1.94 | 0.992x |
| 3 | 23.1 | 1.3% | 0.855 | 2.694 | 2.77 | 0.971x |
| 4 | 21.9 | 1.1% | 0.781 | 3.322 | 3.61 | 0.921x |
| 5 | 20.1 | 1.7% | 0.673 | 3.691 | 4.37 | 0.845x |
| 6 | 17.7 | 0.7% | 0.567 | 3.785 | 5.09 | 0.744x |
| 8 | 14.2 | 2.1% | 0.424 | 3.934 | 6.59 | 0.597x |

No width pays. The peak is at the narrowest width measured and every step up
is a clean loss, each gap outside the spread it was measured with, so
"optimal at 3 to 4" is not a statement about this host.

### The slope is where the two effects separate

The round-cost column is derived from the measured ones, so `emitted / round
cost` reproducing the speedup column is an identity and not a prediction. What
is not an identity is its shape. Fitting `cost = a + b K` over each sweep:

| pairing, host | fit | largest residual |
|---|---|---:|
| Gemma 4 12B, M3 Ultra (gen 15) | `1.14 + 0.090 K` | 0.08 |
| Gemma 4 12B, M1 Ultra (gen 13) | `1.35 + 0.346 K` | 0.20 |
| Qwen 3.8 27B, M1 Ultra (gen 13) | `0.46 + 0.771 K` | 0.06 |

The cost is affine in the block width on all three, and the two effects this
record has been describing qualitatively land in the slope rather than the
intercept:

- **Host.** Same pairing, 0.090 against 0.346, a factor of 3.8, with the fixed
  parts close at 1.14 and 1.35. That is the `use_qmv_wide` split priced per
  block position: generation 15+ absorbs another position into one wide pass,
  generation 13 pays for another narrow one.
- **Target.** Same host, 0.346 against 0.771. That is the GatedDeltaNet
  recurrence: 48 of the target's 64 layers process tokens in sequence, so an
  extra position is charged nearly in full rather than amortised.

They compose, and the Qwen pairing on M1 Ultra carries both, which is why it
is the only configuration measured here that loses at every width. The M3
Ultra fit uses that host's published width table. M5 Max is not fitted: its
sweep covers only widths 3 to 6, too short a lever arm to separate slope from
intercept.

### The shipped default

Passing no `--draft-block-size` at all, four runs per pairing after two
discarded warm-ups:

| pairing | reported block | tok/s | acceptance | emitted/verify | vs the published row |
|---|---:|---:|---:|---:|---|
| Gemma 4 12B | 4 | 42.85 | 0.799 | 3.398 | 1.5% under the block-5 row, inside spread |
| Qwen 3.8 27B | 3 | 23.3 | 0.855 | 2.694 | the same row |

Both reproduce their sweep row's acceptance and emitted-per-verify to three
digits, which is the check that the default really lands where the sweep says.
This host is the third distinct answer to that question: on M5 Max the default
sits 5.8% below the peak, on M3 Ultra it lands exactly on the peak, and here
it is inside the spread of everything else in the band. The published M1 Ultra
rows therefore neither understate nor overstate what an untuned user gets.

## Byte identity: the probed pairing holds, the unprobed one does not

Classic and MTP output compared directly at `temperature 0`, generated text
only (loader lines and the stats line stripped), on this host:

| Pairing | Probe | code | enumeration | prose |
|---|---|---|---|---|
| Qwen 3.8 27B + 4-bit MTP head | passes | identical | not run | not run |
| Gemma 4 12B + 4-bit assistant | none (#1188) | identical | identical | **diverges** |

The prose divergence is at byte 892 of a 1755-byte generation:

```
classic: ...tokens in parallel (as long as they are provided as inp
MTP:     ...tokens in parallel (the attention mechanism allows this
```

Three runs per arm: each arm is byte-identical to itself every time and the
two arms disagree with each other every time. This is the block-versus-chain
path, not run-to-run noise.

This is direct evidence for #1188. The doc's rule of thumb, that an affine
4-bit target on generation 13 is byte-identical below the batch limit of 12,
holds at the op level but does not carry to the model level for this pairing:
two of three prompts agree and one does not. The Qwen pairing, which measures
the property at startup instead of predicting it, comes out identical on the
same test. One 400-token generation per arm reproduces the whole thing.

The Qwen check needed `--show-reasoning`, because this checkpoint spends the
whole 300-token budget in its reasoning channel and the CLI suppresses that by
default; without the flag both arms capture as empty and the comparison is
vacuous rather than passing.

## What was thrown away, and what is not measured

Two rows were rejected by the harness and remeasured rather than published:

- The prose row of the first sweep. A Time Machine first backup (3.1 TB, 20%
  complete) started mid-run and reached 595% CPU. The guard reported 38% of
  the run contended, peak 627%, and spreads of 3.1% and 6.4%. Remeasured after
  `tmutil stopbackup` at spreads of 0.5% and 1.7%. Both runs read 0.95x, which
  is what the guard's own note predicts: a steady load depresses both arms
  together and the ratio survives even when the absolute numbers do not.
- The enumeration row of the second sweep, at 37% contended and peak 1612%
  from `mdworker`. The published enumeration row is the first sweep's clean
  measurement; the contended rerun read the same 1.48x.

The source-code row was measured cleanly in both sweeps, at 1.25x each time,
which is the reproducibility check across the pair.

Not measured here:

- **Block widths outside the swept range.** Gemma was swept at 3 to 12 and
  Qwen at 2 to 8. Gemma below 3 and Qwen at 1 were not run, and the Qwen table
  peaks at its narrowest measured width, so its optimum may lie below the range
  rather than at its edge.
- **Any prompt but the code one, at any width.** Both sweeps use the
  source-code prompt only, so the widths where prose or enumeration peak are
  not established.
- **The `get_qmv_batch_limit` crossover on this host.** M3 Ultra's tail was
  traced to a kernel boundary at width 12; M1 Ultra is also a `d` part and
  reads the same generation 13 table, but no `MLXCEL_QMV_WIDE` on/off verify
  timing was run here to locate it, and `use_qmv_wide` is false on this
  generation anyway, so the two gates do not interact the way they do there.
- **The batch-capable Gemma 4 31B + bf16 assistant pairing.** The checkpoints
  are not on this host.
- **Anything under concurrency.** Every number here is single-request offline
  CLI. The acceptance figures in particular are known to be structurally worse
  under the server's grant rotation for the Qwen drafter; see
  `qwen38-mtp-m1ultra-2026-08-16.md`.
- **Whether the Gemma prose divergence changes output quality.** It is one
  reproducible token flip and its continuation. `examples/logit_trace` plus
  `scripts/compare_logit_traces.py` are the tools for that question and were
  not run.
