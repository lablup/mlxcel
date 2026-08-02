# Speculative-decoding acceptance rules

Speculative decoding proposes tokens with a cheap draft model and verifies them
with the expensive target model. The *acceptance rule* decides which proposals
survive. It determines two things at once: how many tokens a round emits per
target forward (throughput) and what distribution the emitted stream follows
(correctness). This document states which rule each code path runs, what each
one guarantees, and how to tell from a log which one a given run used.

## The rules

Write `p` for the target model's effective next-token distribution at a verify
position (after temperature, token bias, penalties, XTC, and top-k / top-p /
min-p filtering) and `q` for the distribution the drafter actually drew its
proposal from.

### Sampler-match (the default at `temperature > 0`)

Accept the drafted token iff it equals a fresh draw from `p`; emit that draw on
both branches. **Lossless**: every emitted token is a fresh conditional draw
from the target, so the stream is exactly a target-only sample. Its acceptance
probability is `sum_x p(x) q(x)`.

This is what `SpeculativeGenerator` has always done. Issue #902 describes the
verify loop as comparing against the target's *argmax*; for this generator that
is inaccurate, and the difference is the whole point: the code calls
`sample_token_optimized(pos_logits, target_sampling, history)`, which is `argmax`
only when `temperature == 0`.

### Greedy argmax (`temperature == 0`)

Sampler-match where the sampler happens to be `argmax`. Lossless by the same
argument; it is exactly greedy target decoding.

### Argmax-against-argmax (MTP and DFlash only)

Accept iff the drafted token equals `argmax(target_logits)`, regardless of
temperature; emit the argmax on mismatch. **Not lossless** at `temperature > 0`:
the served stream is the target's greedy stream, not a sample from `p`. Its
acceptance probability is `q(argmax p)`.

### Modified rejection sampling (opt-in)

From Leviathan et al. 2023, "Fast Inference from Transformers via Speculative
Decoding" (Algorithm 1) and Chen et al. 2023, "Accelerating Large Language
Model Decoding with Speculative Sampling".

1. Draw `u ~ U[0, 1)`. Accept the drafted token `t` iff `u * q(t) <= p(t)`.
2. On the first rejection, emit one token drawn from `normalize(relu(p - q))`
   and end the chain.
3. If every drafted token was accepted, emit a bonus token drawn from `p`.

Lossless, because `min(q, p) + relu(p - q) == p` pointwise (the residual's
normalizer cancels the rejection probability exactly). Acceptance probability
`sum_x min(p(x), q(x))`.

## The acceptance ceiling

For a `q`-distributed proposal and a `p`-distributed emission, the largest
probability the two can coincide is `sum_x min(p(x), q(x))`, the
maximal-coupling bound, equivalently `1 - TV(p, q)`. **Every**
distribution-preserving rule is capped there:

| Rule | Lossless | Acceptance probability |
|------|----------|------------------------|
| Sampler-match | yes | `sum_x p(x) q(x)` |
| Modified rejection sampling | yes | `sum_x min(p(x), q(x))` (the ceiling) |
| Argmax-against-argmax | **no** | `q(argmax p)` (not bounded by the ceiling) |

Since `min(a, b) >= a * b` on `[0, 1]`, rejection sampling always accepts at
least as often as sampler-match. But `q(argmax p)` is under no such constraint:
with `p(a) = 0.3` and `q(a) = 0.95` the argmax rule accepts at 0.95 where the
ceiling for any correct rule is about 0.30.

**Consequence, stated plainly:** no implementation can make a
distribution-preserving rule strictly improve mean accepted length against
argmax acceptance. That is not a defect to fix; it is the price of
unbiasedness. Issue #902's required outcome ("mean accepted draft length at
temp 0.7 strictly improves vs argmax acceptance") is unachievable in general
and should be read as applying against sampler-match, where the improvement is
real and provable.

## Which rule each path runs today

| Path | `temperature == 0` | `temperature > 0` default | `temperature > 0` opt-in |
|------|--------------------|---------------------------|--------------------------|
| `SpeculativeGenerator` (classic; `mlxcel generate --draft-model`) | greedy argmax (lossless) | **sampler-match** (lossless) | modified rejection sampling (lossless, acceptance-optimal) |
| Gemma 4 MTP round loop | argmax (lossless here) | argmax-against-argmax (**biased**) | not wired |
| DFlash round loop | argmax (lossless here) | argmax-against-argmax (**biased**) | not wired |

The MTP and DFlash verify paths select the target token with
`argmax_per_position` / `argmax_logits_to_array` regardless of the sampler, and
their drafters return token ids without the distribution they were drawn from
(`Drafter::draft_block` returns `Vec<i32>`). Making them lossless requires
widening both the drafter interface and `MtpTarget::verify_forward`'s output.
Those are the paths where this feature would buy correctness rather than only
acceptance rate, and where the acceptance trade against argmax is a real
decision rather than a free win.

## RNG dependency

Modified rejection sampling draws randomness the previous rules did not:

* one `U[0, 1)` per verified draft position, for the accept test;
* one categorical draw per rejection, for the residual.

Both go through MLX's default key sequence, the same stream the fused sampler
consumes, so `mlxcel`'s `--seed` (MLX `random::seed`) still reproduces a run
exactly. But the stream is consumed differently than before: **at an equal seed
the emitted tokens differ from a pre-change run**, in the same sense that
`MLXCEL_SAMPLING_GUMBEL` changes the stream without changing the distribution.
To reproduce a token stream recorded before this landed, set
`MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT=0`.

Greedy (`temperature == 0`, or `top_k == 1`) draws nothing new: the rule
selector returns the argmax rule before any distribution tensor is built, so
temperature-0 output is byte-identical.

## Opt-in, and why it is not the default

`MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT=1` (also `true`, `yes`, `on`) enables
modified rejection sampling. It is **off by default**, and
`SpeculativeGenerator::with_stochastic_acceptance(bool)` overrides the
environment programmatically.

The default is off because on the only path it currently reaches, the rule it
replaces was already lossless. It therefore buys acceptance rate and nothing
else, and the available gain is exactly
`sum_x min(p, q) / sum_x p(x) q(x)`, which collapses toward 1 whenever the
drafter is confident: `q(t*) ~ 1` makes `min(p, q)` and `p * q` coincide.
Measured at about 1.02 on a Llama-3.1-8B / Llama-3.2-1B pair at temperature
0.7. Two extra full-vocabulary passes and an extra host sync per verified
position is not worth two percent of theoretical acceptance.

This follows the same pattern as `MLXCEL_FUSED_QK_NORM` (#326) and the #905
fusions: the machinery, the proof and the tests land and are available, but the
default does not move until a measurement justifies it.

## Reading the rule off a run

**The `mlxcel` CLI installs no `tracing` subscriber by default**, so the
info-level instrumentation below is invisible on the only binary that runs
`SpeculativeGenerator`. The acceptance summary is therefore printed on
**stdout**, unconditionally, after every speculative run:

```
[Speculative acceptance] rule=stochastic rounds=153 proposed=612 positions_tested=195 accepted=46 per_position_acceptance=0.2359 acceptance_rate=0.0752 mean_accepted_len=0.3007 (stochastic (modified rejection sampling))
```

`rule=` is the stable identifier, one of `stochastic`, `greedy-argmax`,
`sampler-match`, `sampler-match-no-proposal-distribution`. It is what proves an
A/B arm is not silently the fallback.

### The two acceptance rates are not interchangeable

| Field | Definition | Use |
|-------|------------|-----|
| `per_position_acceptance` | `accepted / positions_tested` | **The one theory predicts.** Compare across arms and against `sum_x min(p, q)`. |
| `acceptance_rate` | `accepted / proposed` | How much of each drafted block survived. Moves with block length. |
| `mean_accepted_len` | `accepted / rounds` | Tokens gained per target forward. Drives throughput. |

`proposed` counts every position of every drafted block, but a chain stops at
its first rejection and never tests the positions behind it. So
`acceptance_rate` is depressed by exactly the amount the chain truncated, and
comparing it across two arms with different chain-length profiles is
meaningless. Confusing these two is enough to make a correct implementation
look like a 2x regression.

### Optional tracing

Setting `RUST_LOG` now installs a subscriber for the offline CLI commands
(writing to stderr, so piped stdout stays clean), which surfaces the one-shot
`speculative acceptance rule active` line, the per-call `speculative decode
finished` line, and the drafter auto-detection line. Default output is
unchanged when `RUST_LOG` is unset. `mlxcel serve` is unaffected: it installs
its own subscriber.

## Verifying the gain before trusting a throughput number

`MLXCEL_SPECULATIVE_ACCEPT_DIAG=1` adds a second line:

```
[Speculative acceptance diagnostic] closed_form_sum_min=0.2213 closed_form_sum_prod=0.2087 (measured per-position acceptance must sit at sum_min, and sum_min >= sum_prod always)
```

`closed_form_sum_min` is `sum_x min(p(x), q(x))` averaged over the verify
positions the run actually tested, which is the probability modified rejection
sampling accepts. `closed_form_sum_prod` is `sum_x p(x) q(x)`, the probability
the pre-#902 rule accepts. Because `min(a, b) >= a * b` for `a, b` in `[0, 1]`,
`sum_min >= sum_prod` holds for every pair of distributions, always. The
diagnostic therefore turns "the acceptance rate looks wrong" into an arithmetic
statement:

* measured `per_position_acceptance` should sit at `closed_form_sum_min`. If it
  does not, the accept test is wrong.
* `closed_form_sum_min` should be at or above `closed_form_sum_prod`. If it is
  not, `p` or `q` is wrong.

It costs two full-vocabulary reductions and a host readback per tested
position, so it is off by default.

### The size of the gain depends on the drafter's entropy, not on the code

`sum_x min(p, q)` collapses onto `sum_x p(x) q(x)` whenever the drafter is
nearly deterministic: if `q(t*) ~ 1` then `min(p(t*), q(t*)) = p(t*)` and
`p(t*) q(t*) ~ p(t*)`, so the two rules accept at the same rate and this change
buys nothing. The gain appears when `q` is diffuse. Measured on
Llama-3.1-8B-Instruct-4bit with a Llama-3.2-1B-Instruct drafter, 200 tokens,
`--num-draft-tokens 4`, one trajectory each:

| Temperature | `closed_form_sum_min` | `closed_form_sum_prod` | Ratio | measured `per_position_acceptance` |
|---|---|---|---|---|
| 0.7 | 0.2808 | 0.2740 | 1.02x | 0.2902 |
| 1.0 | 0.1738 | 0.1613 | 1.08x | 0.1735 |
| 1.5 | 0.6809 | 0.0494 | 13.8x | 0.6966 |

The measured value sits on `sum_min` at every temperature, which is the
correctness result. The *ratio* column is the throughput headroom, and on this
pair at 0.7 there is essentially none: the 1B drafter is close to deterministic
there. Between-trajectory variance in the agreement profile is large (two
200-token runs at the same temperature differed by nearly 2x in `sum_min`), so
a single short throughput run at 0.7 on this pair measures noise, not the
change. Read `closed_form_sum_min / closed_form_sum_prod` first: it is the
upper bound on what any throughput measurement can show.

The temperature 1.5 row is a real measurement of the mechanism but comes from a
degenerate high-temperature trajectory where both models agree on repetitive
text; do not read it as a typical production number.

## Measuring the change

The classic `SpeculativeGenerator` is the path this rule changed, and it is
reachable only from the offline CLI (the server routes speculative work through
the MTP and DFlash round loops). Run the A/B against `mlxcel generate` with a
draft model, not against `speculative_bench`, whose pairings are all MTP or
DFlash and would compare the unchanged fallback against itself:

```bash
# Baseline arm: the default sampler-match rule.
MLXCEL_SPECULATIVE_ACCEPT_DIAG=1 \
  ./target/release/mlxcel generate \
    -m models/<target> --draft-model models/<draft> --num-draft-tokens 4 \
    --temp 0.7 --seed 1234 -n 256 -p "<prompt>"

# Opt-in arm: modified rejection sampling.
MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT=1 MLXCEL_SPECULATIVE_ACCEPT_DIAG=1 \
  ./target/release/mlxcel generate \
    -m models/<target> --draft-model models/<draft> --num-draft-tokens 4 \
    --temp 0.7 --seed 1234 -n 256 -p "<prompt>"
```

Confirm the `rule=` field differs between arms (`sampler-match` vs `stochastic`), then compare `per_position_acceptance` and `mean_accepted_len`.
Repeat at `--temp 0.0`, where both arms must produce identical output and
identical `rule=greedy-argmax`.

**A note on the drafter-kind line.** With no explicit `--draft-kind`, the CLI
prints the drafter kind it *auto-detected* from the drafter's `config.json`,
which is frequently `dflash`, and then ignores it and runs the classic
`SpeculativeGenerator` anyway. A second line now states which path actually
runs:

```
Resolved drafter kind: dflash (block_size = 16, default)
Drafter kind was auto-detected only; running the classic SpeculativeGenerator path (pass --draft-kind explicitly to select a kind-specific round loop).
```

Passing `--draft-kind` explicitly does **not** get you a DFlash round loop in
`mlxcel generate`; it returns an error saying the offline path does not
construct one.

## Regression guards

In `src/lib/mlxcel-core/src/speculative/`:

* `distribution_tests.rs::speculative_stream_matches_the_target_distribution` —
  chi-square goodness of fit of the emitted stream against the target's exact
  categorical, over 12000 tokens (`MLXCEL_SPEC_DIST_SAMPLES` overrides), dof 5,
  rejected above 25.7448 (alpha = 1e-4).
* `distribution_tests.rs::chi_square_rejects_known_wrong_acceptance_rules` —
  power calibration. The same statistic must reject a rule that resamples from
  `p` instead of the residual, and an argmax-against-argmax rule.
* `distribution_tests.rs::filtered_target_support_is_never_violated` — zero
  tolerance: one emitted token outside a `top_k = 2` target's support fails.
* `distribution_tests.rs::temperature_zero_stream_is_byte_identical_to_greedy_target_only`
  — zero tolerance on greedy output.
* `distribution_tests.rs::kv_cache_length_is_exact_in_both_termination_regimes`
  — cache rewind arithmetic, pinned in the all-reject and all-accept regimes.
* `stochastic_accept_tests.rs` — the accept rate against `min(1, p/q)`, the
  residual against `normalize(relu(p - q))`, and the `sum min(p, q)` optimum.
