# Temperature 0: plain decode versus block-built KV history (#2778)

A temperature 0 run that goes through a block verification path (`--prompt-lookup`,
or a draft model) can leave the plain-decode stream. This document records what was
measured, the cause the evidence supports, why the code is not changed, and the
commands that let a maintainer redo every number. It follows the method in
`docs/benchmarks.md` ("Judging a change that moves the numbers"): byte-identity says
only yes or no, so the comparisons below are teacher-forced and gated on how decided
the reference was.

**Status: known limitation, not a bug, no code change.** Issue #2778 recorded the
decision that this does not block 0.8.0. Release notes must not claim that
temperature 0 speculative output matches plain decode. Plain decode is unaffected.
The conclusion is scoped: it holds for the flips reproduced below, and it does not
explain the issue's original 35-logit swing (see "Limits").

## Cause

The flip is an **argmax decidedness flip** driven by a small numerical difference in
the KV history. A block verification path builds the K/V of every row after the
first with a different arithmetic path than a one-token decode step builds it (the
quantized projections run at `M = block width`, and MLX picks a kernel per `M`).
That difference is rounding-class. It changes the emitted token only where the
reference's top two logits are tied or nearly tied, and the rest of the stream
follows from that one token. When the KV cache is stored quantized
(`--kv-cache-mode int8`, `fp16+turbo3`, `fp16+turbo4`) end-to-end streams leave the
plain stream in 4 of 6 checkpoint/mode cells measured, against 0 of 2 with fp16 KV.
Nothing below shows corrupted KV contents: the temperature 0 contract is "the argmax
of the logits the path computed", and two paths that compute slightly different
logits legitimately disagree on a tie.

What each statement rests on:

| Claim | Status | Evidence |
|---|---|---|
| At equal context, a verify block moves logits at rounding scale and flips only an exact tie | Measured (Qwen3-0.6B-4bit, fp16 KV) | 1 |
| Row 0 of a block, which sees only the sequentially built context, matches a decode step on the top-1 logit | Measured (Qwen3-0.6B-4bit, 48 positions) | 1 |
| Comparing a block against a width-1 run at a different context length overstates the disagreement | Measured | 1 |
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
with `PREFILL=128`, top 8.

### Fixtures

```bash
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
T=./target/release/examples/logit_trace
$T $M corpus_long.txt 1 384 8 128 > w1.tsv
$T $M corpus_long.txt 8  48 8 128 > w8.tsv
python3 scripts/compare_logit_traces.py w1.tsv w8.tsv --decided 2
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
scripts/matched_width_trace.sh $M corpus_long.txt out_q3     # WIDTH=8 PREFILL=128 BLOCKS=48
python3 scripts/compare_matched_width.py out_q3
```
Control arm, fresh cache versus real history: a separate trace with `PREFILL=0`
and 64-token chunks on the same corpus —
`$T $M corpus_long.txt 64 8 8 0 > armA.tsv` — disagrees with the matched cache on
324 of 512 decided-eligible positions (63.28%, largest gap 3.875). That is context
length sensitivity, not the block arithmetic: B and C (identical histories, with and
without the single-token decode split) agree on 0 of 512.

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
- Repeating the width-8 trace is bit-equal: `cmp` of two independent `w8.tsv`
  runs reports no difference (`cmp w/o8_0.tsv w/o8_0_rep.tsv` agrees).

The unmatched first pass and the matched pass are computed over the same 384 positions,
so their difference isolates context length. Unmatched rows 1 to 7 (336 positions) have
309 positions with a nonzero logit delta on the reference's own choice (p50 0.375,
p90 1.0, max 2.375) and 5 top-1 disagreements, all at gap 0.0; the matched pass has
195 of 336 (p50 0.125, p90 0.25, max 0.75) and 1 disagreement. Row 0 is identical in
both passes (48 of 48 zero deltas on the top-1 logit), as it should be: its context and
input token are the same in every arm. Most of the unmatched movement was context
length, not block width. What remains is concentrated in the rows the block itself
built, and it flips only an exact tie. The width-1 history here is one prefill forward,
not eight sequential decode steps; the j=0 control (every row compared at its identical
context and input) shows zero disagreement and zero nonzero delta across all 384 rows.

### Evidence 2: the decode-step split

`MLXCEL_TRACE_DECODE_STEP=1` feeds the last context token through its own
single-token forward before the traced block (the state a speculative verify block
runs in, #1793). Compare it against the same context pass without the split:

```bash
C=corpus_long.txt
MLXCEL_TRACE_DECODE_STEP=1 $T $M $C 64 8 8 256 > armB.tsv   # split
$T $M $C 64 8 8 256 > armC.tsv                                            # no split
python3 scripts/compare_logit_traces.py armB.tsv armC.tsv --decided 2
```

Result: 0 / 512 top-1 disagreements, 0 / 444 on decided positions. The split
introduces no change of its own.

The control confirms the comparison is sensitive: with no context pass (`PREFILL=0`,
a fresh cache for each 64-token chunk, a different context from arms B and C)

```bash
$T $M $C 64 8 8 0 > armA.tsv
python3 scripts/compare_logit_traces.py armA.tsv armB.tsv --decided 2
```

disagrees on 324 / 512 positions (38 / 114 decided), and the same against arm C.

### Evidence 3: end to end, plain versus `--prompt-lookup`

```bash
PROMPT=$(cat copy_prompt.txt)
for kv in fp16 int8 fp16+turbo3 fp16+turbo4; do
  for arm in plain pl; do
    extra=""; [ "$arm" = pl ] && extra="--prompt-lookup"
    ./target/release/mlxcel generate -m "$MODEL" -p "$PROMPT" -n 160 --temp 0 --seed 42 \
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
  MLXCEL_TRACE_KV_CACHE_MODE=$kv scripts/matched_width_trace.sh $GEMMA corpus_long.txt out_gm_$kv
  python3 scripts/compare_matched_width.py out_gm_$kv
done
```

| KV mode | top-1 disagreements | decided disagreements | row 0 nonzero delta | rows 1-7 delta p50 / p99 / max | perplexity width 1 versus 8 |
|---|---|---|---|---|---|
| fp16 | 0 / 384 | 0 / 364 | 6 of 48 | 0.125 / 1.0 / 2.0 | 1.0433 versus 1.0439 (+0.056%) |
| int8 | 0 / 384 | 0 / 365 | 44 of 48 | 0.048 / 0.303 / 2.20 | 1.0503 versus 1.0500 (-0.034%) |

No flip appears at either mode on this checkpoint, and the int8 deltas are not larger
than the fp16 ones. This probe therefore does **not** support the amplification
hypothesis at the logit level; it shows only that the arm runs and what its output
looks like. Unlike Qwen3, Gemma 3's row 0 is not bit-identical under either mode, so
the "row 0 matches" statement above is a Qwen3-0.6B-4bit observation, not a general
one.

## Why there is no code fix

A fix would have to make a block verify row bit-equal to the decode step that would
have produced the same position. The matched-width trace shows the arithmetic is not
bit-equal by construction (different `M`, different kernel; 195 of 336 rows 1 to 7
differ on the top-1 logit), and bit-equality across `M` is not a property any
quantized kernel set provides. Chasing it would give up verify throughput for a
guarantee that `docs/benchmarks.md` already says is spent once the arithmetic path
differs. The behaviour that is guaranteed, and kept, is that every emitted token is
the argmax of logits the target computed (`docs/speculative-acceptance.md`, "Greedy
argmax"). In the trace evidence the divergence is confined to positions where the
reference's top two logits were equal, so there is no regression to revert and no
reverted arm that could fail. The quantized end-to-end divergences of Evidence 3 were
not measured at the logit level. Quantized KV storage is a user-selected trade of
exactness for memory; a quantized stream differing from the fp16 stream is the same
kind of difference as the one measured here.

## Limits of this evidence

- The original report, Qwen3 `copy` at position 197 (plain prefers its token by 2.0,
  most block histories prefer the other by about 35), is **not reproduced**. The
  prompt text and the KV mode of the original runs are not in the issue, and the runs
  here are on M1 Pro, not M1 Ultra. A 35-logit swing is far larger than anything
  measured here (largest disagreement gap 0.0 in the matched pass, largest logit
  delta 2.375 unmatched, 0.75 matched). If it was measured under the default fp16 KV,
  this document does not explain it. The issue's own trigger for reopening stays: a
  reproducer that shows corrupted KV contents.
- No KV contents were compared tensor by tensor, so the target-only reproducer in the
  issue's first acceptance criterion is still open. The statement that the histories
  differ at rounding scale rests on logits, not on a K/V dump.
- The quantization-amplification step (a rounding-level K difference crossing a
  quantization step becomes a full step) was not measured at the logit level. The
  Qwen3 end-to-end evidence is consistent with it (4 of 6 quantized cells flip, 0 of 2
  fp16), but the Gemma 3 probe does not show it, and eight cells on one prompt do not
  separate it from an unrelated difference between the arms.
- Qwen3 streams were checked at 160 generated tokens on one prompt, two checkpoints,
  one machine. The trace evidence is one corpus.
- The Llama `prose` case in the issue (position 137, gap 0.016) is a true near-tie
  and needs no further explanation.
