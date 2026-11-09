# Temperature 0: plain decode versus block-built KV history (#2778)

A temperature 0 run that goes through a block verification path (`--prompt-lookup`,
or a draft model) can leave the plain-decode stream. This document records what was
measured, the cause the evidence supports, why the code is not changed, and the
commands that let a maintainer redo every number. It follows the method in
`docs/benchmarks.md` ("Judging a change that moves the numbers"): byte-identity says
only yes or no, so the comparisons below are teacher-forced and gated on how decided
the reference was.

**Status: known limitation, root cause open, no code change here.** Issue #2778
recorded the decision that this does not block 0.8.0 and left the root cause open.
Release notes must not claim that temperature 0 speculative output matches plain
decode. Plain decode is unaffected. The analysis is scoped: it describes the flips
reproduced below and does not explain the issue's original Qwen3 position 197 swing
(plain prefers its token by 2.0, most block histories prefer the other by about 35),
which may still be a real KV history difference (see "Limits").

## Cause of the reproduced flips

The flips reproduced here are **argmax ties**. At equal context, a width-8 verify
block moved the reference's own top-1 logit by at most 0.75 on Qwen3-0.6B-4bit with
fp16 KV, and the single top-1 disagreement was at a reference top-two gap of 0.0. In
a free-running stream the rest of the output follows from the flipped token.

The proposed mechanism is that a block computes the rows after the first through a
different arithmetic path (the quantized projections run at `M = block width`, and
MLX picks a kernel per `M`). For these runs that is a hypothesis: the matched
reference history is one prefill forward rather than eight decode steps, and the
commands here compare logits, not K/V tensors ("Limits" records the separate KV dump
reported on #2778). When the KV cache is stored quantized (`--kv-cache-mode int8`,
`fp16+turbo3`, `fp16+turbo4`) end-to-end streams leave the plain stream in 4 of 6
checkpoint/mode cells measured, against 0 of 2 with fp16 KV; those divergences were
not measured at the logit level.

Nothing below shows corrupted KV contents, and nothing below rules them out for the
position 197 report. Where two paths compute different logits on a tie, each token is
the argmax of the logits its own path computed.

What each statement rests on:

| Claim | Status | Evidence |
|---|---|---|
| At equal context, a verify block moves logits at rounding scale and flips only an exact tie | Measured (Qwen3-0.6B-4bit, fp16 KV) | 1 |
| Row 0 of a block, which sees only the sequentially built context, matches a decode step on the top-1 logit | Measured (Qwen3-0.6B-4bit, 48 positions) | 1 |
| Comparing a block against a width-1 run at a different context length overstates the disagreement | Measured | 1 |
| The block's own K/V arithmetic causes the rows 1 to 7 logit deltas | **Hypothesis**; no K/V comparison here, and the reference history is one prefill forward | 1, Limits |
| The one-token decode split itself is harmless | Measured | 2 |
| Repeating a width-8 trace is bit-equal | Measured | 1 |
| With fp16 KV, plain and `--prompt-lookup` emit the same stream | Measured: 2 checkpoints, 1 prompt, 160 tokens | 3 |
| With quantized KV, plain and `--prompt-lookup` leave each other | Measured: 4 of 6 cells | 3 |
| A rounding-level K/V difference is amplified by quantization into a visible one | **Hypothesis**; the Gemma 3 logit-level probe does not show larger deltas under int8 | 4, Limits |

## Evidence

Hardware: Apple M1 Pro, Metal. Build with
`cargo build --release --features metal,accelerate --example logit_trace` and
`cargo build --release --features metal,accelerate`. Traces are teacher-forced on one
520-token corpus (Qwen3 tokenizer; the Gemma 3 tokenizer gives 536 plus a BOS anchor)
with `PREFILL=128`, top 8. The source revision of these runs is not recorded. They
predate the MLX pin advance to `9f806b36` (#2876), which adds a Metal 3-16-row affine
`qmv_mma` in the width range traced here, so rerun them before treating the numbers
as current.

### Fixtures

```bash
# Run from the repository root; later blocks use $REPO and this scratch directory.
REPO=$PWD
mkdir -p /tmp/mlxcel_2778 && cd /tmp/mlxcel_2778
cat > passage.txt <<'EOF'
The transformer architecture processes sequences through stacked attention layers. Each layer attends over every preceding token, building contextual representations that the feed-forward network refines. Positional encodings let the model distinguish order, and residual connections carry information forward across depth. Training adjusts billions of parameters so that the final projection predicts the next token from context.
EOF
for i in 1 2 3 4 5 6 7 8; do cat passage.txt; done > corpus_long.txt
{ printf 'Copy the following passage exactly, word for word. Do not summarize, do not paraphrase, do not add commentary. Output only the passage as given:\n\n'; cat passage.txt; } > copy_prompt.txt
```

### Evidence 1: block width at equal context

Width 1 is the decode step; width 8 is a verify block of the size `--prompt-lookup`
drafts. The width is the variable under test.

Comparing a width-8 trace with a width-1 trace row by row is confounded. The
`logit_trace` harness gives every chunk a `PREFILL`-token window, so row `j` of a
width-8 chunk follows `PREFILL + j` tokens while the width-1 chunk follows
`PREFILL`. Context length alone moves the logits. The first pass below shows the
size of that confound; the matched pass removes it.

First pass, context lengths unmatched:

```bash
M=~/.cache/mlxcel/models/mlx-community/Qwen3-0.6B-4bit
T=$REPO/target/release/examples/logit_trace
$T $M corpus_long.txt 1 384 8 128 > w1.tsv
$T $M corpus_long.txt 8  48 8 128 > w8.tsv
python3 "$REPO/scripts/compare_logit_traces.py" w1.tsv w8.tsv --decided 2
```

| Measure | Result |
|---|---|
| positions compared | 384 |
| top-1 disagreement, all positions | 5 / 384 (1.30%) |
| top-1 disagreement, decided positions (gap >= 2.0) | 0 / 354 (0.00%) |
| largest reference top-two gap at a disagreement | 0.000 |
| where the reference's choice landed in the candidate | rank 2, 5 of 5 |
| logit delta on the reference's own choice | p50 0.375, p90 1.0, p99 2.0, max 2.375 |
| perplexity, width 1 versus width 8 | 1.0853 versus 1.0773 (-0.74%) |

Matched pass: row `j` of every width-8 block is compared with a width-1 run traced
at `PREFILL = 128 + j`, so both see the same token history and the same input token.
`scripts/matched_width_trace.sh` writes the traces and
`scripts/compare_matched_width.py` compares them (it takes a few minutes; it runs one
width-8 trace and eight width-1 traces):

```bash
LOGIT_TRACE=$T "$REPO/scripts/matched_width_trace.sh" $M corpus_long.txt out_q3     # WIDTH=8 PREFILL=128 BLOCKS=48
python3 "$REPO/scripts/compare_matched_width.py" out_q3
```

| Row | positions | top-1 disagreements | decided (gap >= 2) | disagreements on decided | max logit delta |
|---|---|---|---|---|---|
| 0 | 48 | 0 | 45 | 0 | 0.000 |
| 1 | 48 | 1 | 44 | 0 | 0.250 |
| 2 | 48 | 0 | 44 | 0 | 0.375 |
| 3 | 48 | 0 | 45 | 0 | 0.250 |
| 4 | 48 | 0 | 44 | 0 | 0.375 |
| 5 | 48 | 0 | 43 | 0 | 0.375 |
| 6 | 48 | 0 | 46 | 0 | 0.250 |
| 7 | 48 | 0 | 47 | 0 | 0.750 |
| all | 384 | 1 | 358 | 0 | |

- The single disagreement is at corpus position 193, row 1, where the reference's top
  two logits were equal (gap 0.0).
- Row 0: all 48 positions have the same reference top-1 logit in both runs (no
  nonzero delta). Row 0 sees only the sequentially built history.
- Rows 1 to 7: 336 positions, 195 with a nonzero logit delta on the reference's own
  choice; p50 0.125, p90 0.25, p99 0.375, max 0.75.
- Perplexity: 1.0771 (width 1) versus 1.0773 (width 8), +0.023%.
- Repeating the width-8 trace is bit-equal: `cmp` of the `w8.tsv` files from two
  independent runs reports no difference.

The unmatched first pass sets no `MLXCEL_TRACE_START_TOKEN`, so it traces positions 0
to 383, while the matched pass traces 128 to 511. The two cover the same number of
positions but not the same positions, and the first 128 positions of the unmatched
pass have shorter context windows. Unmatched rows 1 to 7 (336 positions) have 309
positions with a nonzero logit delta on the reference's own choice (p50 0.375, p90
1.0, max 2.375) and 5 top-1 disagreements, all at gap 0.0; the matched pass has 195 of
336 (p50 0.125, p90 0.25, max 0.75) and 1 disagreement. Row 0 has zero deltas in both
passes (48 of 48 on the top-1 logit), since its context and input token are the same
in both arms of each pass. The smaller matched deltas are consistent with context
length accounting for most of the unmatched movement. What remains is in rows 1 to 7,
and in this run it flips only an exact tie. The width-1 history here is one prefill
forward, not eight sequential decode steps.

### Evidence 2: the decode-step split

`MLXCEL_TRACE_DECODE_STEP=1` feeds the last context token through its own
single-token forward before the traced block (the state a speculative verify block
runs in, #1793). Compare it against the same context pass without the split, at block
width 64:

```bash
C=corpus_long.txt
MLXCEL_TRACE_DECODE_STEP=1 $T $M $C 64 8 8 256 > armB.tsv   # split
$T $M $C 64 8 8 256 > armC.tsv                                            # no split
python3 "$REPO/scripts/compare_logit_traces.py" armB.tsv armC.tsv --decided 2
```

Result: 0 / 512 top-1 disagreements, 0 / 444 on decided positions. The split
introduces no change of its own.

The control confirms the comparison is sensitive: with no context pass (`PREFILL=0`,
a fresh cache for each 64-token chunk, a different context from arms B and C)

```bash
$T $M $C 64 8 8 0 > armA.tsv
python3 "$REPO/scripts/compare_logit_traces.py" armA.tsv armB.tsv --decided 2
```

disagrees on 324 / 512 positions (63.28%; 38 / 114 decided, largest gap at a
disagreement 3.875), and the same against arm C. That is context length sensitivity,
not block arithmetic.

### Evidence 3: end to end, plain versus `--prompt-lookup`

```bash
PROMPT=$(cat copy_prompt.txt)
for kv in fp16 int8 fp16+turbo3 fp16+turbo4; do
  for arm in plain pl; do
    extra=""; [ "$arm" = pl ] && extra="--prompt-lookup"
    "$REPO/target/release/mlxcel" generate -m "$MODEL" -p "$PROMPT" -n 160 --temp 0 --seed 42 \
      --kv-cache-mode $kv --show-reasoning $extra > ${kv}_${arm}.raw 2>&1
    grep -v -e '^Runtime device' -e '^Wired memory' -e '^KV cache mode' -e '^Loading model' \
      -e '^Detected' -e '^GPU:' -e '^Converting' -e '^Note:' -e '^Model loaded' \
      -e '^Generating' -e '^\[Prompt lookup\]' -e '^\[Generated' ${kv}_${arm}.raw > ${kv}_${arm}.txt
  done
done
python3 - <<'EOF'
P = len(open('copy_prompt.txt').read())          # 577: the prompt echo
def gen(p):
    g = open(p).read()[P:]
    return g[1:] if g.startswith('\n') else g    # the prompt-lookup path prints one extra \n
def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
for kv in ('fp16', 'int8', 'fp16+turbo3', 'fp16+turbo4'):
    print(kv, first_diff(gen(f'{kv}_plain.txt'), gen(f'{kv}_pl.txt')))
EOF
```

`MODEL` is `~/.cache/mlxcel/models/mlx-community/Qwen3-0.6B-4bit`, and then the bf16
`Qwen/Qwen3-0.6B` checkpoint directory. Only the generated text is compared: the
banner and statistics lines are dropped, as are the first `len(copy_prompt.txt)`
characters (the prompt echo) and one leading newline; the token streams do not
differ there. `int8` is per-token absmax quantization applied to all 28 layers;
`fp16+turboN` keeps K in fp16 and stores V in N bits, applied to 24 of 28 layers.

| Checkpoint | KV mode | plain versus `--prompt-lookup` |
|---|---|---|
| Qwen3-0.6B-4bit | fp16 | identical (748 characters) |
| Qwen3-0.6B-4bit | int8 | identical (894 characters) |
| Qwen3-0.6B-4bit | fp16+turbo3 | identical (839 characters) |
| Qwen3-0.6B-4bit | fp16+turbo4 | diverges at generated character 432 |
| Qwen3-0.6B bf16 | fp16 | identical (865 characters) |
| Qwen3-0.6B bf16 | int8 | diverges at generated character 82 |
| Qwen3-0.6B bf16 | fp16+turbo3 | diverges at generated character 276 |
| Qwen3-0.6B bf16 | fp16+turbo4 | diverges at generated character 673 |

The first divergences are free-form reasoning text where several continuations are
plausible: for bf16 int8, `... without any summary or paraphrase.` (plain) against
`... without any summary or commentary.` (prompt lookup); for bf16 turbo3,
`The passage talks about` against `The passage is about`. The KV mode also changes
the plain stream by itself (fp16 versus int8, both plain, diverge at generated
character 192 on 4-bit and 175 on bf16), so the quantized stream is a different
stream, not a corrupted copy of the fp16 one. Rerunning the bf16 `fp16+turbo3` pair
reproduced both streams exactly. Whether each of those positions was a near-tie at
the logit level was not measured: a Qwen3 trace cannot install a KV storage mode
(Evidence 4).

### Evidence 4: the KV-storage arm of the trace, on a model that supports it

`MLXCEL_TRACE_KV_CACHE_MODE` stores KV the way `--kv-cache-mode` does. It reaches
only model-owned caches (Gemma 3 and Gemma 4 among the shipped families) and exits
with "this model does not own its KV caches" on others; Qwen3 and Llama are such
models. As a probe of the arm, the matched-width comparison of Evidence 1 was run on a
4-bit Gemma 3 270m checkpoint, once per storage mode. `$GEMMA` is a local checkpoint
(`model_type` `gemma3_text`; hidden size 640, 18 layers; quantization group size 64,
bits 4); its hub id was not recorded, so the run is not tied to a public snapshot id.

```bash
for kv in fp16 int8; do
  LOGIT_TRACE=$T MLXCEL_TRACE_KV_CACHE_MODE=$kv "$REPO/scripts/matched_width_trace.sh" $GEMMA corpus_long.txt out_gm_$kv
  python3 "$REPO/scripts/compare_matched_width.py" out_gm_$kv
done
```

| KV mode | top-1 disagreements | decided disagreements | row 0 nonzero delta | rows 1-7 delta p50 / p99 / max | perplexity width 1 versus 8 |
|---|---|---|---|---|---|
| fp16 | 0 / 384 | 0 / 364 | 6 of 48 | 0.125 / 1.0 / 2.0 | 1.0433 versus 1.0439 (+0.056%) |
| int8 | 0 / 384 | 0 / 365 | 44 of 48 | 0.048 / 0.303 / 2.20 | 1.0503 versus 1.0500 (-0.034%) |

No flip appears at either mode on this checkpoint. Under int8 the p50 and p99 deltas
are smaller than under fp16, and the max is slightly larger (2.20 against 2.0). This
probe therefore does **not** support the amplification
hypothesis at the logit level; it shows only that the arm runs and what its output
looks like. Unlike Qwen3, Gemma 3's row 0 is not bit-identical under either mode, so
the "row 0 matches" statement above is a Qwen3-0.6B-4bit observation, not a general
one.

## Why there is no code change here

Making a verify row bit-equal to the decode step needs an arithmetic path that is
bit-equal at that width. Whether one exists depends on the kernel MLX dispatches for
`M` and on the hardware: `docs/benchmarks.md` records an affine 4-bit target as
byte-identical for block widths below 12 on Apple GPU generation 13/14 and divergent
from width 2 on generation 15 and newer until `qmv_wide` is disabled, and the MTP and
DFlash paths gate on a runtime block-versus-chain exactness probe for that reason. The
token-only loop behind `--prompt-lookup` and a full-model drafter has no such gate;
`docs/speculative-acceptance.md` ("Offline routing") documents that its stream is the
target's own argmax at every position and is not guaranteed byte-identical to plain
decode. Adding a gate there is a throughput trade this document does not decide. In
the matched-width trace the two arms are not bit-equal (195 of 336 rows 1 to 7 differ
on the top-1 logit).

In the trace evidence the divergence is confined to positions where the reference's
top two logits were equal. That does not establish that no regression exists: #2778
records that whether the behaviour predates #2768 was not measured, and the same is
true here. The quantized end-to-end divergences of Evidence 3 were not measured at the
logit level. Quantized KV storage is a user-selected trade of exactness for memory,
and Evidence 3 shows the plain stream itself changes with the KV mode.

## Limits of this evidence

- The original report, Qwen3 `copy` at position 197 (plain prefers its token by 2.0,
  most block histories prefer the other by about 35), is **not reproduced**. The
  prompt text and the KV mode of the original runs are not in the issue, and the runs
  here are on M1 Pro, not M1 Ultra. A 35-logit swing is far larger than anything
  measured here (largest disagreement gap 0.0 in the matched pass, largest logit
  delta 2.375 unmatched, 0.75 matched). If it was measured under the default fp16 KV,
  this document does not explain it. The issue's own trigger for reopening stays: a
  reproducer that shows corrupted KV contents.
- The commands in this document compare no K/V tensors, so the issue's first
  acceptance criterion is still open. A separate harness posted on #2778
  (`examples/kv_history_divergence.rs`, not in this tree) reports, on
  Qwen3-0.6B-4bit, M1 Pro and a 64-token context, zero differing KV bytes between
  one-token appends and block appends at widths 1, 2, 4 and 8, differing bytes from
  width 16 up, and no argmax flip in those spans. If that holds, the rows 1 to 7
  deltas in Evidence 1 are not explained by block width 8 alone and include the
  difference between the reference's one-prefill history and the block's history.
  The statement that the histories differ at rounding scale rests on logits here.
- The quantization-amplification step (a rounding-level K difference crossing a
  quantization step becomes a full step) was not measured at the logit level. The
  Qwen3 end-to-end evidence is consistent with it (4 of 6 quantized cells flip, 0 of 2
  fp16), but the Gemma 3 probe does not show it, and eight cells on one prompt do not
  separate it from an unrelated difference between the arms.
- Qwen3 streams were checked at 160 generated tokens on one prompt, two checkpoints,
  one machine. The trace evidence is one corpus.
- The Llama `prose` case in the issue (position 137, gap 0.016) is a true near-tie
  and needs no further explanation.
