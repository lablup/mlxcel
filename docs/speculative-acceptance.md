# Speculative-decoding acceptance rules

Speculative decoding proposes tokens with a cheap draft model and verifies them
with the expensive target model. The *acceptance rule* decides which proposals
survive. It determines two things at once: how many tokens a round emits per
target forward (throughput) and what distribution the emitted stream follows
(correctness). This document states which rule each code path runs, what each
one guarantees, and how a request reports what speculation did.

## The rules

Write `p` for the target model's effective next-token distribution at a verify
position (after token bias, penalties, top-k / top-p / min-p filtering on the
untempered distribution, XTC on the renormalised filtered row, and the final
temperature scaling) and `q` for the distribution the drafter actually drew
its proposal from.

### Sampler-match (the default at `temperature > 0`)

Accept the drafted token iff it equals a fresh draw from `p`; emit that draw on
both branches. **Lossless**: every emitted token is a fresh conditional draw
from the target, so the stream is exactly a target-only sample. Its acceptance
probability is `sum_x p(x) q(x)`.

### Greedy argmax (`temperature == 0`)

Sampler-match where the sampler happens to be `argmax`. Lossless by the same
argument; it is exactly greedy target decoding.

### Argmax-against-argmax (MTP, DFlash, DSpark and the Muse Glimmer assistant)

Accept iff the drafted token equals `argmax(target_logits)`, regardless of
temperature; emit the argmax on mismatch. **Not lossless** at `temperature > 0`:
the served stream is the target's greedy stream, not a sample from `p`. Its
acceptance probability is `q(argmax p)`.

The Muse Glimmer assistant drafter (#1343) runs on the same DFlash round loop
under this rule, with no dispatch difference from the Qwen 3.5 DFlash drafter:
its proposals are sampled per masked slot under the request's sampler and
accepted against the target's argmax. At `temperature == 0` it is exactly
greedy target decoding, subject to the block-versus-chain exactness probe
below, including past the 2048-token sliding window of the target's rotating
caches.

Its verify width is the one adaptive width in the DFlash loop, so the probe
runs more than once: the round loop warms up at the drafter's declared depth
of four rows and widens to the requested ceiling only when a measurement
window pays for it, and the gate clears both widths before the burst engages.
Probing the ceiling alone would admit the warm-up width unmeasured, which is
not a formality: the forward width selects which quantized-matmul kernel MLX
dispatches, and `docs/benchmarks.md` records the same comparison reading
20.6 percent disagreement at width 8 and 0.0 percent at width 32.

The LFM2 DSpark drafter runs on the DFlash round loop and therefore this rule,
with one difference in dispatch: it is greedy-only. The drafter's Markov head
proposes a chain trained against greedy targets, and the server declines any
request whose sampler is not argmax (`temperature > 0` with `top_k != 1`) to
classic decode with one warning, rather than serving the biased greedy stream
under a sampling request. At `temperature == 0` it is exactly greedy target
decoding, subject to the block-versus-chain exactness probe below.

## The acceptance ceiling

For a `q`-distributed proposal and a `p`-distributed emission, the largest
probability the two can coincide is `sum_x min(p(x), q(x))`, the
maximal-coupling bound, equivalently `1 - TV(p, q)`. **Every**
distribution-preserving rule is capped there:

| Rule | Lossless | Acceptance probability |
|------|----------|------------------------|
| Sampler-match | yes | `sum_x p(x) q(x)` |
| Argmax-against-argmax | **no** | `q(argmax p)` (not bounded by the ceiling) |

`q(argmax p)` is under no such constraint: with `p(a) = 0.3` and `q(a) = 0.95`
the argmax rule accepts at 0.95 where the ceiling for any correct rule is about
0.30. So no distribution-preserving rule can strictly improve mean accepted
length against argmax acceptance in general.

Modified rejection sampling (Leviathan et al. 2023, Chen et al. 2023) attains
the ceiling. An opt-in implementation shipped on the classic draft-model loop
and was removed with it in v0.8.0 (#2754), because no remaining path could use
it: see the next section.

## Which rule each path runs today

| Path | `temperature == 0` | `temperature > 0` |
|------|--------------------|-------------------|
| `PromptLookupDrafter` on the engine's token-only loop (`mlxcel generate --prompt-lookup`) | greedy argmax (lossless) | **sampler-match** (lossless, and acceptance-optimal here) |
| `PromptLookupDrafter` in the server scheduler (`--draft-kind prompt-lookup`, the same `Engine::verify_round`) | greedy argmax (lossless) | **sampler-match** (lossless, and acceptance-optimal here) |
| Gemma 4 MTP round loop | argmax (lossless here) | argmax-against-argmax (**biased**) |
| DFlash round loop (Qwen 3.5 DFlash drafter) | argmax (lossless here) | argmax-against-argmax (**biased**) |
| DFlash round loop (LFM2 DSpark drafter) | argmax (lossless here, probe-gated) | declines to classic decode |
| DFlash round loop (Muse Glimmer assistant drafter) | argmax (lossless here, probe-gated) | argmax-against-argmax (**biased**) |

"Classic decode" on this page means plain non-speculative decoding. The MTP and DFlash verify paths select the target token with
`argmax_per_position` / `argmax_logits_to_array` regardless of the sampler, and
their drafters return token ids without the distribution they were drawn from
(`Drafter::draft_block` returns `Vec<i32>`). Making them lossless requires
widening both the drafter interface and `MtpTarget::verify_forward`'s output.

Prompt lookup proposes deterministically, so its `q` is one-hot on the drafted
token `d`. The two acceptance probabilities then coincide at `p(d)`
(`sum_x p(x) q(x) = sum_x min(p(x), q(x)) = p(d)`), and on a rejection both
rules emit a draw from `p` conditioned on not being `d`. Sampler-match is
therefore already the acceptance-optimal rule for that path.

The CUDA server's prompt-lookup governor additionally pauses low-yield rows
when the EMA of accepted proposals plus one falls below four tokens after at
least two drafted rounds and the latest round is also below the floor. A
latest round meeting the floor can repay startup EMA debt without pausing.
An isolated zero-accept round after an accepted-token EMA at least equal to
the floor receives one round of grace; partial low-yield acceptance does not.
Grace never extends beyond the first consecutive low-yield round; pausing
still requires the emitted-token EMA to be below the floor. This measured
field-boundary heuristic leaves the acceptance rule itself unchanged.
This is a proposal-cost decision, not a change to
the sampler-match acceptance rule. Paused rows still observe emitted tokens
and run shadow lookups; seven matching shadow tokens restore verification.
Plain asks, retracted proposals, and gated-out rows add no verify evidence.
The CLI has no break-even floor and retains its two-token shadow confirmation.
`adaptive = false` ignores the floor, and the graded policy is unchanged.

## Offline routing

`mlxcel generate --draft-model <dir>` runs one of two round loops: MTP (explicit
`--draft-kind mtp`, or an MTP `model_type`) or DFlash (a DFlash drafter directory,
explicit or auto-detected, on a target with an offline DFlash arm, today Laguna).
Anything else fails before any drafter load, with an error naming
`--draft-kind mtp`, `--draft-kind dflash` and `--prompt-lookup`. That covers an
ordinary full model, which the classic draft-model loop accepted until v0.8.0
(#2754).

A DFlash or DSpark drafter on a target without an offline arm gets the same
refusal plus a pointer to `mlxcel-server`, which drives the DFlash round loop for
more targets (#1168). These checkpoints ship no `embed_tokens` and no `lm_head`,
because they borrow both from the target when they bind. The LFM2 DSpark drafters
(#1339) and the Muse Glimmer assistant (#1343) have the same shape.

The discriminator is structural (a `dflash_config` block and/or
`architectures: ["DFlashDraftModel"]` or `["Lfm2DSparkDraftModel"]` in the
drafter's `config.json`, or the Muse assistant's own
`model_type: muse_glimmer_assistant` / `["MuseGlimmerAssistantModel"]`), not
the resolved `DrafterKind`. An ordinary full model auto-resolves to `dflash` too, so the resolved kind cannot tell
the two apart. The same structural check runs inside `get_model_type`, so
`mlxcel generate -m <dflash-dir>` is rejected as "not a standalone model" too.

## Reading a run

Setting `RUST_LOG` installs a subscriber for the offline CLI commands (writing to
stderr, so piped stdout stays clean), which surfaces the drafter auto-detection
line. Default output is unchanged when `RUST_LOG` is unset. `mlxcel serve` is
unaffected: it installs its own subscriber. `mlxcel generate --prompt-lookup`
prints its drafter statistics on stderr after the run.

### Per-request acceptance on the response itself

`mlxcel-server` reports what speculation did for **one request** on that
request's own response, so a client can answer "did the drafter help me" without
access to the server's logs (#1314). The block appears only when a drafter
executed at least one verify round; a classic-decode deployment, and a
speculative request that finished inside prefill, answer the body they always
did.

On the native `/completion` and `/completions` routes the counters ride the
existing `timings` object:

```json
"timings": {"cache_n": 0, "prompt_n": 12, "predicted_n": 64, "...": "...",
            "draft_n": 72, "draft_n_accepted": 55, "draft_rounds": 9,
            "draft_kind": "dflash"}
```

`draft_n` and `draft_n_accepted` are llama-server's own optional pair, spelled
as upstream spells them and appended the way upstream appends them, so a client
that already reads llama-server timings reads mlxcel's unchanged.
`draft_rounds` and `draft_kind` are mlxcel additions: upstream has neither a
round count nor a drafter-kind concept, and without the round count the mean
accepted length per round is not recoverable from the pair.

`/v1/chat/completions` carries the whole block as a top-level `timings` object,
on the non-streaming response and on the streaming chunk that carries
`finish_reason`. b10621 puts a `timings` object on its own OpenAI chat
responses, built from the same struct its native route uses, so mlxcel's is that
same object rather than the `draft_*` half alone: a client that probes for the
key and then reads `predicted_per_second` off it finds the key it expects. The
one difference that remains is presence, upstream emits the block on every chat
completion while mlxcel emits it only for a drafted request, and it is recorded
against both chat routes in `compat/llama-server/b10621/routes.json`.

Only the `finish_reason` chunk carries it: the counters are the run's totals,
and a mid-stream frame reporting a running `draft_n` would be reporting a total
that is not one yet. For the same reason the native route's per-token
`timings_per_token` frames carry the nine base keys and none of the four.

The base keys mean the same thing on a drafted request as on a classic one
(#1592). `prompt_ms` covers queue wait plus the target prefill, and
`predicted_ms` covers every verify round from the moment the prefill finished,
so `predicted_per_second` is the drafted decode rate and is comparable to the
classic figure for the same prompt. The burst runs to completion before it
streams, so it stamps the first-token instant at its prefill end rather than
when the finished tokens are replayed; before the fix that replay was the
stamp, `prompt_ms` absorbed the whole burst, and `predicted_ms` reported the
sub-millisecond replay (95000 tokens per second on a 96-token request). The
first drafted request after startup also pays the drafter's disk load, which
is a once-per-process cost and is excluded from every `timings` field; it
stays on the `Drafter loaded` log line. See the `timings` note in
[`docs/llama-server-compat.md`](llama-server-compat.md#completions-and-embeddings-are-native-routes-not-openai-aliases-1441).

The mean accepted length per round is `(draft_n_accepted + draft_rounds) /
draft_rounds`: every round emits its accepted drafts plus one bonus token. It
is left to the client rather than reported, so a client that wants a different
aggregate is not stuck with this one. `draft_kind` carries a `--draft-kind`
name: `dflash`, `mtp` or `prompt-lookup`.

The block covers the three paths that own a request's whole round loop: the
DFlash B=1 burst, the MTP B=1 burst, and the tick-cooperative MTP slice (whose
counters are the session totals across every slice, not one slice's), plus
prompt lookup's per-tick verify rounds, whose counters are the drafter's
totals for the request (a proposal retracted because no forward followed it is
not counted). The
default-off B>1 batched burst reports nothing, because its round loops return
per-row tokens without per-row acceptance counters.

The disaggregated router front reports nothing either. It assembles its stream
from the prefill/decode handoff protocol rather than from a finished
`GenerationResult`, and that protocol carries a generated-token count and no
acceptance counters, so surfacing the block there is a handoff-protocol change
and not a response-shaping one.

## Regression guards

In `src/lib/mlxcel-core/src/engine/speculative_distribution_tests.rs`, all on the
engine's token-only verify loop:

* `speculative_stream_matches_the_target_distribution` (and `_at_block_size_one`):
  chi-square goodness of fit of the emitted stream against the target's exact
  categorical, over 12000 tokens (`MLXCEL_SPEC_DIST_SAMPLES` overrides), dof 5,
  rejected above 25.7448 (alpha = 1e-4).
* `chi_square_rejects_known_wrong_acceptance_rules`: power calibration. The same
  statistic must reject emitting the proposal untested and an
  argmax-against-argmax rule.
* `filtered_target_support_is_never_violated`: zero tolerance. One emitted token
  outside a `top_k = 2` target's support fails.
* `temperature_zero_stream_is_byte_identical_to_greedy_target_only`: zero
  tolerance on greedy output.
* `acceptance_accounting_is_exact_in_both_deterministic_regimes`: the
  `SpeculativeRounds` counters in the all-reject and all-accept regimes.

`engine/verify_round_rollback_tests.rs` (`kv_cache_length_is_exact_in_both_acceptance_regimes`,
`kv_cache_stays_consistent_across_mixed_accept_reject_rounds`) pins that after
every round the state holds the prompt plus every emitted token except the
pending one, on every layer. `src/lib/mlxcel-core/src/cache/kv_trim_tests.rs` pins `KVCache::trim`, and
`src/commands/generate_tests.rs` pins the offline routing matrix.
