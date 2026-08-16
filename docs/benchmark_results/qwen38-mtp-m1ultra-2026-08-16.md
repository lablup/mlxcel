# Qwen3.8-27B MTP speculative decoding — M1 Ultra, 2026-08-16

Measurement record for issue #1165 (qwen3_5_mtp drafter support). All numbers
measured through the production call paths on the branch that adds the
feature, at commit range `5bb9bd1d..` (drafter + adapter + chain-parity
kernel).

## Environment

- Mac Studio, Apple M1 Ultra, 128 GB unified memory, macOS 26.6.1.
- Target: `models/qwen3.8-27b-4bit` (mlx-community conversion, 4-bit affine,
  untied lm_head, vocab 248320). Drafter: `models/qwen3.8-27b-mtp-bf16`
  (`mlx-community/Qwen3.8-27B-MTP-bf16`, 15 tensors, bf16, loaded as f16).
- Machine load (1-min avg) recorded per section below. The box carries
  background load; every comparison below is a within-run ratio between arms
  measured back to back, which survives it. Within-arm dispersion is shown by
  min-max over the repetitions.

## End-to-end: MTP vs classic decode (offline CLI, production path)

`mlxcel generate -m <target> -p <prompt> -n 300 -t 0.0 [--draft-model <drafter>]`,
long-form prompt ("Explain in detail how a transformer neural network
works..."), 300 generated tokens, 5 repetitions per arm, decode tok/s as
printed by the CLI. Load 1.9-5.0 across the section.

| arm | decode tok/s (median) | min-max | acceptance rate | emitted/verify | vs classic |
|---|---|---|---|---|---|
| classic | 23.45 | 23.44-23.46 | n/a | n/a | 1.00x |
| MTP, block 4 (CLI default) | 13.81 | 13.79-13.85 | 0.465 | 2.39 | 0.59x |
| MTP, block 3 (drafter-configured) | 16.48 | 16.39-16.56 | 0.591 | 2.18 | 0.70x |

Acceptance is deterministic across repetitions (temperature 0). Round mix at
block 3: 55 full / 52 partial / 30 zero accepts over 137 rounds, i.e.
**first-draft acceptance 78%** — the jointly-trained head drafts well.
Exactness: temperature-0 output is byte-identical to classic decode on a
5-prompt gate (offline CLI) and token-identical on the server path (see the
detokenizer note below).

## Where the time goes (per-round split, from the round-loop diagnostics)

Block 4 run: 125 rounds, decode 21.67 s -> 173 ms/round emitting 2.39 tokens.

| phase | ms/round | share |
|---|---|---|
| verify forward (K=4) | 137.6 | 68% |
| draft (3 drafter steps) | 19.9 | 10% |
| accept-hook extension (est.) | ~13 | 6% |
| verify finalize (rollback) | 2.7 | 1% |
| walk + shared-kv re-arm | <0.3 | ~0% |

A classic decode step is 42.6 ms. The verify forward at K=4 costs 3.2 classic
steps, which is the entire story: at 137 ms the round emits 2.39 tokens
(72 ms/token) against classic's 42.6 ms/token.

## Root cause of the verify cost: the target's own multi-token forward

Isolation (`tests/qwen38_mtp_chain_parity.rs::verify_forward_cost_scaling`,
medians over 20 timed calls per shape):

| T | `forward_speculative` (per-position verify attention + GDN snapshot capture) | plain batched prefill path (same block, no verify machinery) |
|---|---|---|
| 1 | 43.6 ms | 43.6 ms |
| 2 | 75.7 ms | 75.0 ms |
| 3 | 106.2 ms | 104.4 ms |
| 4 | 135.5 ms | 133.2 ms |
| 8 | 244.0 ms | 238.7 ms |

The verify machinery itself (per-position attention dispatch loop, GDN
rollback snapshots) costs under 2% at every T. The cost lives in the target's
**plain multi-token forward**: every extra token in a block adds ~29 ms, about
two thirds of a full single-token step, on both paths equally. The
K-token-block-for-the-price-of-one premise of speculative decoding does not
hold on this GPU generation for this model. This is precisely the pre-M5
regression already encoded for Gemma 4 in `mtp_b1_default`
(`src/server/batch/speculative_burst.rs`): M1 Ultra measured 0.75-0.96x for
Gemma 4 MTP while M5 Max measured 1.2-1.4x, and the static per-hardware
default plus the adaptive MTP policy (issue #333) exist exactly to gate this.
Qwen fares somewhat worse than Gemma here because each drafter step and
accept-hook extension pays a full-vocab (248320) LM-head projection.

**Not measured: M5-class hardware.** The Gemma precedent (verify amortization
holds, 1.2-1.4x) projects a positive sign for this pairing there, but no M5
machine was available to this run; the cell is open.

## Chain-parity kernel cost (correctness requirement, not an optimization)

The temperature-0 exactness gate initially failed: one near-tie argmax flip
per ~100-250 generated tokens. Cause: the gated-delta Metal kernel emits its
recurrent state in the storage dtype (f16), so a T=1 decode chain rounds the
state after every token while a T=K verify block carried float32 state across
the block and rounded once. The chain-parity kernel variant
(`gated_delta_step_seqpar`) rounds at every in-block step, making T=K
bit-identical to K single-token steps (pinned bitwise in
`parity_kernel_block_is_bitwise_equal_to_single_token_chain`). It is used
ONLY by the speculative verify capture path and the rollback replay; classic
prefill and decode are untouched.

- Isolated kernel cost (27B GDN geometry, medians over 100 dispatches):
  ratios parity/standard 1.089 (T=1), 1.117 (T=3), 0.834 (T=4), 1.008
  (T=64) — inside the dispatch-noise band; the kernel's time loop was already
  sequential in T, the variant only adds a per-step register round.
- End-to-end attribution through the production MTP path
  (`MLXCEL_GDN_CHAIN_PARITY=0` diagnostic toggle, 3 reps): 14.29 tok/s
  without parity vs 13.81 with, ~3.4% — the price of byte-exactness, and not
  the reason MTP loses on this box (see above).

## DFlash blast radius (shared verify machinery)

`forward_speculative` is also DFlash's verify path, so the parity kernel
changes a shipped feature. Same-binary A/B on identical prompts
(`qwen3.5-27b-4bit` + `qwen3.5-27b-dflash`, server path, temp 0):

| request | arm | wall | rounds | acceptance | emitted/verify |
|---|---|---|---|---|---|
| 120 tok | parity OFF (pre-change numerics) | 17.09 s | 37 | 0.152 | 3.22 |
| 120 tok | parity ON | 16.79 s | 38 | 0.146 | 3.13 |
| 80 tok | parity OFF | 16.69 s | 41 | 0.0705 | 1.93 |
| 80 tok | parity ON | 16.90 s | 41 | 0.068 | 1.93 |

No material change: walls within run noise, acceptance within near-tie
reshuffling. (A separately quoted main-branch baseline with acceptance 0.274
at 120 tokens used different prompts and is not attributable; the controlled
same-prompt A/B above is the before/after.)

## Server E2E and the boundary detokenizer note

`mlxcel-server` with the MTP pair engages the tick-slice path (55-57 slices
for a 150-token completion) and produces **token-identical** output to the
classic server (150/150 token ids equal, `logprobs` comparison); a 170-token
run is byte-identical in text as well. A 150-token run's *text* differed in
the final characters only because the token budget cut mid-BPE-merge and the
classic streaming detokenizer flushes a trailing partial merge differently
from the burst finalizer — a pre-existing boundary artifact independent of
MTP (token ids agree).

The adaptive MTP policy (issue #333) governs engagement: on this box its
profiling phase attempts MTP and its estimator settles per pairing; the
static pre-M5 default declines. `MLXCEL_ENABLE_MTP_B1` pins either way.
