# The qmv_wide narrow pin priced on batched serving, M3 Ultra, 2026-08-22

Issue #1261 asked for a number: when the MTP exactness gate buys back
temperature-0 byte-identity by disabling `qmv_wide` for the whole process
(#1199, #1258), what does everything else in that process pay? The issue's
framing assumed the main victim is batched decode at `B >= 2`, "whose
per-step projections are exactly the `M >= 2` shape `qmv_wide` exists for",
and gated any scoping work (its Step 2) on that tax being material.

Headline: **the batched-decode tax is at most 1% and indistinguishable from
zero at B = 8, and not because the two kernels are close: the `M = B`
projection shape the issue assumed does not exist in the shipped decode path
of any family the pin can fire on.** Both MTP families run batched decode as
per-sequence `M = 1` forwards, which dispatch `qmv` under either pin. The
collateral tax that does exist today is on **prompt-cache-hit TTFT**: an
adopted-prefix request prefills only its short suffix, whose `M` lands in
the qmv window, and that one forward costs 15.4 ms more under the narrow
pin (12.6 ms on the Qwen target), which reads as +33% on this harness's
cache-hit TTFT. The mixed-workload arm (one MTP stream plus four classic
streams, the shape the pin actually arises in) agrees end to end: the
classic streams lose 0.1 to 2.6% beside a narrow-pinned MTP stream, and
the loss tracks the MTP stream's own verify slowdown occupying the shared
worker, not the classic streams' kernels.

Per the issue's own exit condition ("if the B-sweep delta is small on
production shapes, the right fix may be documenting the tax and stopping
there"), the scoping work is not built. The numbers, the mechanism, and the
boundary where this answer would change are recorded below.

## Environment

| Field | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB unified memory, macOS 26.6.1 (25G76) |
| Apple GPU generation | 15 (`applegpu_g15d`; `use_qmv_wide` holds for affine, the exactness probe fails wide, and where the narrow retry passes the gate pins the process narrow by default) |
| Build | `cargo build --release --features metal,accelerate` |
| Branch | `update/issue-1261-qmv-wide-pin-tax` from `main` at `dd21ada4` (harness scripts added, no runtime change) |
| Harness | `scripts/bench_qmv_wide_pin.sh sweep` / `mixed` (added by this work), driving `scripts/bench_serving_concurrency.py` and `scripts/bench_qmv_pin_mixed.py` |
| Wrapper | `scripts/with_indexers_paused.sh` (17 indexers suspended; `INDEXER_RESUME_DEADLINE` raised to cover the runs) |
| Time Machine | not running (`tmutil status` Running = 0) |
| Sweep target | `models/gemma-4-31b-it-4bit` (batch-capable, no drafter loaded), server `--parallel 8 --metrics` |
| Confirmation target | `models/qwen3.8-27b-4bit`, same protocol at half the boot count |
| Mixed pairing | `models/qwen3.8-27b-4bit` + `models/qwen3.8-27b-mtp-4bit`, `--draft-block-size 3` (the 31B + bf16 pairing was attempted first and cannot be pinned narrow on this host; see below) |
| Sampling | temperature 0 everywhere |

Both arms of the B-sweep pin the kernel explicitly: `MLXCEL_QMV_WIDE=1`
(wide) versus `MLXCEL_QMV_WIDE=0` (narrow). Any value of `MLXCEL_QMV_WIDE`
counts as an operator pin (`qmv_wide_pinned_by_operator` in
`src/models/speculative_exactness.rs`), so the gate's retry can never flip
an arm mid-run, and with no drafter loaded the gate has no probe to run in
the first place. Boots alternate in two balanced ABBA blocks (wide, narrow,
narrow, wide, then narrow, wide, wide, narrow); each boot runs one
discarded warm-up pass and two measured passes, giving 8 samples per arm
per cell. A pass is the B = 1, 2, 4, 8 ladder (512-token prompt, 256-token
budget) plus a long-context B = 4 cell (4096-token prompt).

## Where `M >= 2` quantized matmuls actually occur in serving

The issue's premise was that batched decode at `B >= 2` runs its per-step
projections at `M = B`. The scheduler does run batched decode as one
`forward_batched()` call per step (`execute_batched_decode` in
`src/server/batch/scheduler.rs`, input shape `[B, 1]`), but what the model
does with that input decides the dispatch, and neither MTP family stacks it:

- **Gemma 4** decodes per row on both of its routes. The measured
  checkpoint carries `embed_vision.*` weights, so detection routes it to
  `LoadedModel::Gemma4VLM` (`gemma4_has_vision_weights` in
  `src/models/detection.rs`), and `execute_batched_decode`
  (`src/server/batch/scheduler.rs:7300`) calls
  `forward_batched_with_context_and_ids` with the batch's sequence ids.
  `Gemma4VLModel` overrides that entry point
  (`src/vision/gemma4_vl.rs:687`) and delegates to
  `forward_batched_with_seq_ids_dispatch`
  (`src/multimodal/batched_dispatch.rs:60`), which slices `input_ids` row
  by row and calls `forward_with_sequence_id` once per sequence. The
  text-only route (`LoadedModel::Gemma4`, `models::Gemma4Wrapper`)
  overrides nothing and inherits the trait default
  (`src/lib/mlxcel-core/src/generate.rs:736`), which is the same per-row
  loop over `forward()`. Either way `M = 1` per forward, results
  concatenated.
- **Qwen 3.5** (`src/models/qwen3_5.rs:3388`) overrides it but branches to
  the same per-row loop whenever the input is single-token, which is every
  decode step. Its joint path only serves multi-token batched prefill.

`M = 1` always dispatches `qmv`; the pin selects between `qmv` and
`qmv_wide` only at `M >= 2` (`dispatch_qmv` in the
`src/lib/mlx-cpp/patches/mlx/backend/metal/quantized.cpp` overlay). So
batched decode in a pinned process never reaches the kernel the pin turns
off. The families that do stack decode rows into one `M = B` forward,
Gemma 3 (`src/models/gemma3.rs:1302`) and Llama 4
(`src/models/llama4.rs:1801`), are not MTP families (`mtp_capable_target`
covers Gemma 4 and Qwen 3.5 only), so no process they run in is ever pinned
by the gate.

Two server-side surfaces do put an `M` inside the qmv window
(`2 <= M < get_qmv_batch_limit`, and the limit is 12 for every projection
of this target on this host: hidden 5376 and intermediate 21504 put q, k,
v, o, gate, up, down and the 262k LM head all above the table's 4096
branch):

1. **Prompt-cache-adopted prefill.** A request whose prompt is cached up to
   the last few tokens prefills only the suffix. The harness's repeated
   identical prompts produce exactly this, and the server log shows it:
   `cached=441/445 prompt tokens`, a 4-token suffix, one `M = 4` forward
   per request. This is the one real tax the sweep caught, and it shows in
   TTFT, not in decode throughput.
2. **The speculative verify forward itself** (`M = K`), which is the work
   the pin exists for and was already priced: 17 to 20% on the Qwen verify
   forward, ~23% on the Gemma 4 verify forward (`docs/benchmarks.md`).
   Neither is the "23% of throughput" the same file quotes for the M5 Max
   code row, which is end-to-end decode; the two 23s are different
   quantities.

Fresh full prefill and chunked-prefill chunks run at `M >= 473` in these
configurations, far above every batch limit, and take the matrix-matrix
kernel regardless of the pin. The long-context cell below confirms that
control directly.

The per-row structure also shows in the absolute numbers, which is worth
recording so nobody reads this B-sweep as a healthy-scaling baseline:
aggregate decode throughput rises from 29.5 tok/s at B = 1 to only 43.3 at
B = 4 and falls back to 33.7 at B = 8, because each batched step pays B
row-forwards (partially overlapped by MLX's async pipeline) rather than one
amortized pass. `/metrics` confirms the steps are batched at the scheduler
level (`batch_decode_tokens_total / batch_decode_steps_total` tracked the
batch size at 3.83 tokens per step with 4 streams active) while the model
executes rows serially. Diagnosing that scaling is outside this issue's
scope; what matters here is that it is identical in both arms.

## B-sweep: Gemma 4 31B, wide pin vs narrow pin

Per-request decode rate (tokens after the first divided by the span from
first to last token), medians over 8 measured passes per arm. Every
generation in the ladder ran 59 tokens in both arms (the model answers the
synthetic prompt briefly and stops; verified with per-stream `usage`
counts), so the two arms decode the same number of tokens per stream and
the rate comparison is length-matched. The generated bytes differ between
arms mid-stream, which is the kernel non-identity the exactness gate
exists for; both arms end on the same 59th token.

| B | wide, tok/s | spread | narrow, tok/s | spread | narrow / wide |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.95 | 0.3% | 29.90 | 0.7% | 0.998 |
| 2 | 20.95 | 0.5% | 20.90 | 0.0% | 0.998 |
| 4 | 10.90 | 0.0% | 10.80 | 0.9% | 0.991 |
| 8 | 4.20 | 0.0% | 4.20 | 0.0% | 1.000 |

The per-stream decode delta is 0.0 to 0.9% with every spread at or under
0.9%, against the harness's 4% trust limit. Aggregate throughput (all
completion tokens over the level's wall span, which includes prefill) reads
0.6 to 1.2% lower on the narrow arm; the next section shows that deficit is
the TTFT tax, not decode.

**This is the number issue #1261 Step 1 asked for: the B = 2/4/8
batched-decode tax of the narrow pin on this pairing is at most 1%, inside
or touching the run spread.** It is zero for the structural reason above:
these decode steps never dispatch the kernel the pin disables.

## The tax that is real: cache-hit TTFT

Mean time to first token per level, same passes. After the discarded
warm-up, every ladder request adopts its prompt from the cache and
prefills a 4-token suffix, one `M = 4` quantized-matmul forward, which is
inside the qmv window and therefore does change kernels under the pin:

| B | wide TTFT, ms | spread | narrow TTFT, ms | spread | delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 64.3 | 9.0% | 79.8 | 5.1% | +15.5 ms |
| 2 | 70.0 | 1.0% | 93.1 | 0.4% | +23.0 ms (+33%) |
| 4 | 116.2 | 0.4% | 154.6 | 0.5% | +38.4 ms (+33%) |
| 8 | 208.8 | 0.7% | 278.3 | 0.7% | +69.5 ms (+33%) |

The B = 1 row's spreads (9.0% and 5.1%) are above the 4% trust limit, one
request per pass being too few to average TTFT jitter out, so that row is
indicative only; the B = 2, 4, 8 rows are all at or under 1.0%.

The four deltas are one number in disguise. Concurrent identical requests
prefill their suffixes serially, so the i-th request's TTFT carries i
suffix forwards and the level mean carries `(B + 1) / 2` of them. One
suffix forward costing `d` more narrow predicts mean deltas of `1d, 1.5d,
2.5d, 4.5d`; the measured `15.5, 23.0, 38.4, 69.5` fit `d = 15.4 ms` to
within 0.2 ms on every row, the B = 1 row included. So the entire TTFT
effect is a single `M = 4` suffix forward costing **+15.4 ms** under the
narrow pin, repeated once per queued cache-hit request.

Two controls pin the mechanism:

- **Uncached prefill is untaxed.** The discarded warm-up passes, whose
  first requests prefill the full 445 tokens through the matrix-matrix
  kernel, show no arm difference beyond cold-boot noise (B = 1 warm-up
  TTFT 1479.8 ms wide vs 1369.3 ms narrow, the narrow boot the faster
  one).
- **Chunked long prefill is untaxed.** The long-context cell (4096-token
  prompt, chunked at 512, every chunk `M >= 473`) measures TTFT 23848.5 ms
  wide vs 23885.3 ms narrow (+0.2%, spreads 0.1%).

## The long-context cell is a text-divergence casualty, not a kernel cost

The long-context B = 4 cell's throughput columns read *higher* for the
narrow arm (6.4 vs 5.8 tok/s per stream), stable across all 16 samples.
That is not a kernel effect: at temperature 0 the two arms legitimately
generate different text (the same last-ulp kernel difference the exactness
gate polices), and on this prompt the divergence changes the generation
length itself: 130 tokens under the wide pin, 182 under the narrow, every
stream, verified with per-stream `usage` counts. Different token counts
mean different effective batch occupancy over the window, so the
throughput columns compare different workloads and are reported as
contaminated rather than averaged into the tax. `docs/benchmarks.md`
carries the same warning for the offline arms: turning the flag changes
the text, so cross-flag throughput is only comparable when the lengths
happen to match, as they do (at 59 tokens) in the ladder above. The cell's
TTFT columns are unaffected (prefill precedes generation) and serve as the
chunked-prefill control above.

## Confirmation on Qwen 3.8 27B

Same protocol at half the boot count (one ABBA block, 4 samples per arm per
cell), `models/qwen3.8-27b-4bit`, no drafter. Qwen 3.8 streams its
reasoning channel, which `bench_serving_concurrency.py` did not count as
decoded tokens; the script now counts `reasoning_content` deltas alongside
`content` (without that fix a reasoning model reports no TTFT and no
decode rate at all). Every ladder stream decodes its full 256-token budget
in both arms, so this table is length-matched by construction.

| B | wide, tok/s | spread | narrow, tok/s | spread | narrow / wide |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.35 | 0.3% | 36.30 | 0.3% | 0.999 |
| 2 | 25.80 | 0.0% | 25.75 | 0.4% | 0.998 |
| 4 | 10.70 | 0.0% | 10.70 | 0.0% | 1.000 |
| 8 | 4.90 | 0.0% | 4.90 | 0.0% | 1.000 |

Decode tax 0.0 to 0.2%, spreads at or under 0.4%: the second pinned family
confirms the first. The single-stream 36.3 tok/s agrees with the 35.7 the
offline harness records for this checkpoint on this host in
`docs/benchmarks.md`.

The TTFT columns repeat the suffix mechanism with this family's own
numbers. Qwen's cache-hit suffix is 5 tokens (`cached=479/484`), and the
cache-hit levels read +18.8 ms at B = 2, +31.6 ms at B = 4, +56.8 ms at
B = 8 (all +27 to +28%, spreads at or under 2.3%), fitting one `M = 5`
suffix forward at **d = 12.6 ms** across all three levels. Qwen's B = 1
level happened to run uncached every pass (its 1210 ms TTFT is a full
485-token prefill, not a suffix), which turns that row into another
control: matrix-kernel prefill, TTFT 1210.2 vs 1223.8 ms, +1.1%. The
long-context cell agrees: chunked prefill TTFT +0.4%, decode delta 0.0%.

## Found on the way: the 31B pairing now declines MTP outright on this host

The mixed arm was first attempted on the 31B + bf16 assistant pairing,
`--draft-block-size 4`, and could not be: **on current `main` the exactness
probe for that pairing fails under `qmv_wide` and fails again without it**
("verify block position 0 differs from the single-token chain in 231782 of
524288 logit bytes ... Disabling qmv_wide did not make it exact either"),
so the default-env gate declines MTP, restores the wide kernel, and no
narrow pin ever arises for it. Both server boots reproduced it, block 4,
same byte counts.

This is new information and it collides with a fresh default: #1217
measured this pairing at 1.95x to 2.65x on this host and turned the
batch-capable B = 1 burst on for generation 15+, but that measurement ran
at `9e2c6675`, which predates #1258's Gemma probe. With the probe in
place, the burst #1217 enabled is vetoed at serve time by the exactness
gate on this same host, and the speedup is reachable only by forfeiting
byte-identity (`MLXCEL_MTP_ALLOW_INEXACT=1`, with or without the pin: a
retry that fails restores the wide kernel, so for this pairing the
override alone does reach the fast kernel). The plausible mechanism is the
one #1258 names for the M1 Ultra prose divergence: a narrow-kernel
divergence in the 262144-wide LM head, which the 12B pairing (whose retry
passes, see the recipes below) shares in shape but not in hidden width.
Deciding what to do about that default belongs to a follow-up, not to this
measurement; it is recorded here because anyone rerunning #1217's rows on
current `main` will hit it.

For this document's purpose the consequence is narrower: the pairing the
mixed arm can price is the one whose retry actually pins the process
narrow on this host, and that is the Qwen pairing
(`qwen3.8-27b-4bit` + `qwen3.8-27b-mtp-4bit`, block 3), the same
combination `docs/benchmarks.md` records as "passes after dropping
`qmv_wide`" here.

## Mixed workload: one MTP stream plus four classic streams

The shape the pin actually arises in: `qwen3.8-27b-4bit` +
`qwen3.8-27b-mtp-4bit`, block 3, `--parallel 8`, with
`MLXCEL_ENABLE_MTP_B1=1 MLXCEL_MTP_ADAPTIVE=0` pinning the burst decision
and `MLXCEL_MTP_SLICE_GRANT_ROUNDS=0` disabling slice-slot rotation, so
the first stream holds the tick-cooperative MTP slot for its whole
generation and every concurrent eligible request falls back to classic
decode (the pre-#746 behaviour; the scheduler's "speculative slice slot
busy; seq ... falls back to classic decode" line is in both arms' logs).
Each boot runs three windows (`scripts/bench_qmv_pin_mixed.py`): one long
MTP stream is started, and once it is decoding, four identical classic
streams (512-token prompts, 256-token budgets) run beside it; only the
classic streams' decode rates are read, and a window is valid only if the
MTP stream decoded through at least 95% of it (all windows here: 99.7 to
100%). Window 0 of each boot is the discarded warm-up. Boots alternate
A B B A.

Arm identities, evidenced from each boot's gate line:

- **Arm A**, default env: INFO "probe failed under qmv_wide ... and passed
  without it. Disabling qmv_wide for this process". The MTP stream runs
  the exact narrow verify and the whole process is pinned narrow.
- **Arm B**, `MLXCEL_QMV_WIDE=1 MLXCEL_MTP_ALLOW_INEXACT=1`: WARN "probe
  FAILED but MLXCEL_MTP_ALLOW_INEXACT is set". The process stays wide and
  the MTP stream forfeits byte-identity. (The issue text's recipe for this
  arm, the override alone, does not produce it; see the recipes section.)

The MTP stream's own numbers are not compared across arms (its text
differs by construction). Per-stream classic decode rate, mean of the four
streams, both boots of each arm:

| window | arm A (narrow pin) | arm B (wide) | A / B |
|---|---:|---:|---:|
| 1 | 7.89, 7.94 | 7.92, 7.92 | 0.999 |
| 2 | 7.11, 7.11 | 7.30, 7.29 | 0.974 |

The two windows are genuinely different workloads, deterministically so:
at temperature 0 the MTP stream replays the same text in every boot, so
window 1 (where it emitted 325 tokens in both arms) and window 2 (593
narrow, 612 wide) repeat their own numbers to 0.6% across boots but do not
match each other. Pooling them into one median would manufacture an 8 to
11% spread out of window heterogeneity, so the comparison is per
like-window: **the classic streams lose 0.1% (window 1) to 2.6% (window
2) beside a narrow-pinned MTP stream.**

The mechanism is not the classic streams' own kernels, which are per-row
`M = 1` in both arms; it is worker-occupancy: the narrow verify forward
runs 17 to 20% longer, each MTP slice holds the worker that much longer
per round, and the classic batch gets correspondingly fewer ticks. The
MTP stream's window 2 emission (593 vs 612 tokens, 3% fewer narrow) is
the same effect seen from the other side. So even in the mixed shape, the
pin's cost to bystander streams on current code is bounded by the MTP
stream's own slowdown diluted across the batch, single figures of a
percent, not the kernel-sized 17 to 20% this pairing's verify forward
pays.

## Gate recipes: which env reaches which kernel

Issue #1261's description of its mixed-workload arm reads
"`MLXCEL_MTP_ALLOW_INEXACT=1` with the switch left wide", and
`docs/benchmarks.md` said reproducing the pre-gate fast rows "needs
`MLXCEL_MTP_ALLOW_INEXACT=1`". Both descriptions predate what #1199's
merged ordering actually does: `mtp_exactness_gate` runs
`retry_without_qmv_wide` **before** `allow_inexact()` is ever consulted,
and the override feeds only the engage/decline decision, never the kernel
switch. So on a host where the narrow retry passes, which is every
generation 15+ host measured so far, the override alone is inert: the
retry pins the process narrow first and the flag is never load-bearing.

Verified live with all four recipes on the pairing #1258 measured
(`gemma-4-12b-it-4bit` + 4-bit assistant, block 5, the
`bench_speculative.sh` code prompt, 300 tokens, offline CLI, two
interleaved samples per recipe):

| recipe | gate log line | kernel served | tok/s |
|---|---|---|---:|
| default env | INFO "probe failed under qmv_wide ... Disabling qmv_wide for this process" | narrow, byte-identity kept | 117.29, 116.98 |
| `MLXCEL_MTP_ALLOW_INEXACT=1` | the same INFO retry line; the ALLOW_INEXACT warning never fires | narrow, byte-identity kept | 117.14, 116.85 |
| `MLXCEL_QMV_WIDE=1 MLXCEL_MTP_ALLOW_INEXACT=1` | WARN "probe FAILED but MLXCEL_MTP_ALLOW_INEXACT is set" | wide, inexact | 139.18, 139.12 |
| `MLXCEL_QMV_WIDE=1` | WARN "retry was skipped because MLXCEL_QMV_WIDE is pinned"; the CLI declines MTP | none (classic decode only) | declined |

Three independent lines of evidence agree:

- **The log lines.** The override-alone run logs the retry's INFO line and
  never the loud warning, which is the gate saying the flag did nothing.
- **The bytes.** The default run and the override-alone run produce
  **byte-identical generated text**, both samples; the pinned-wide run
  diverges from them six words in ("write this function" versus
  "implement this"), which is the same divergence
  `docs/benchmarks.md` records for this host's fast kernel.
- **The throughput.** 117 versus 139 tok/s reproduces the byte-identical
  and fast-kernel figures (117.5 / 138.5) that `docs/benchmarks.md`
  carries for this pairing on this host.

So the working recipes on generation 15+ are: default env for exact MTP on
the narrow kernel; `MLXCEL_QMV_WIDE=1` alone to keep the wide kernel and
give up MTP; both together to research the fast kernel with MTP and give
up byte-identity. `MLXCEL_MTP_ALLOW_INEXACT` alone remains meaningful only
where both kernels diverge, so no exact configuration exists (the
pre-#1199 state of generation 15+, and any future hardware whose narrow
arm also fails the probe). `docs/environment-variables.md` and
`docs/benchmarks.md` now say this; the throughput tables in #1199 and
#1258 whose `MLXCEL_MTP_ALLOW_INEXACT=1` rows say "fast kernel" describe
the pre-merge gate and carry a correction note. The measured mixed arm B
above uses the corrected recipe.

## What this settles, and what would reopen it

Issue #1261's acceptance criteria, against this record:

1. **The B = 2/4/8 batched-decode tax is measured on a generation 15+
   host**: at most 1%, spreads at or under 1%, two families, structural
   mechanism identified. The real collateral tax is +15.4 ms per cache-hit
   suffix prefill forward, +33% on this harness's cache-hit TTFT.
2. **The tax is documented as accepted, with the number**: this document.
   Scoping the exact kernel to the verify forward (the issue's Step 2) is
   not built, per the issue's own exit condition. The per-request cost is
   tens of milliseconds of TTFT on cache-hit requests and nothing
   measurable anywhere else in serving today.
3. **The probe measures the same kernel configuration the verify forward
   serves with**: unchanged and true by construction. The pin is
   process-wide and never restored, so probe-time selection and serve-time
   selection cannot differ; under `MLXCEL_QMV_WIDE=1
   MLXCEL_MTP_ALLOW_INEXACT=1` the probe measures wide and the process
   serves wide. Step 2 designs were what could have broken this, and none
   shipped.

What would reopen Step 2: **a real joint batched decode for an MTP
family.** The moment Gemma 4 or Qwen 3.5 stacks decode rows into one
`M = B` forward the way Gemma 3 and Llama 4 already do, batched decode
lands in the qmv window and the pin starts taxing it at whatever the
kernel gap is at that `M` (the op-level data in `docs/benchmarks.md`
measures 1.7 to 1.9x on verify forwards at `M` = 10 to 13). Whoever builds
that should rerun this sweep, expect a material number, and only then
weigh the issue's Step 2 candidates. Until then the narrow pin's collateral
is priced: 15.4 ms per cache-hit prefill, zero on decode.
