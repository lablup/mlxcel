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

- **Block-width sweeps on M1 Ultra.** Widths other than the adaptive
  controller's choice were not run. The M5 Max width numbers do not transfer,
  because the round cost grows differently on this generation.
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
