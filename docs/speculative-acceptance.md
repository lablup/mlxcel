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
and was removed with it in v0.8.0 (#2754); the full-model drafter that
replaced that loop (#2757) runs the sampler-match rule of the engine's
token-only loop, which is lossless, and does not restore the
acceptance-optimal rule: see the next section.

## Which rule each path runs today

| Path | `temperature == 0` | `temperature > 0` |
|------|--------------------|-------------------|
| `PromptLookupDrafter` on the engine's token-only loop (`mlxcel generate --prompt-lookup`) | greedy argmax (lossless) | **sampler-match** (lossless, and acceptance-optimal here) |
| `PromptLookupDrafter` in the server scheduler (`--draft-kind prompt-lookup`, the same `Engine::verify_round`) | greedy argmax (lossless) | **sampler-match** (lossless, and acceptance-optimal here) |
| `FullModelDrafter` on the engine's token-only loop (`mlxcel generate --draft-model <ordinary model>`, `mlxcel-server --draft-model <ordinary model>`, #2757) | greedy argmax, no probe gate, not byte-identical to plain decode (the same class and the same streams as prompt lookup: see [Offline routing](#offline-routing); the MLA families have an opt-in exact block, see [Token-only verify blocks, per family](#token-only-verify-blocks-per-family)) | **sampler-match** (lossless): the drafter proposes its own argmax, the target's draw decides |
| Gemma 4 MTP round loop | argmax (lossless here) | argmax-against-argmax (**biased**) |
| DFlash round loop (Qwen 3.5 DFlash drafter) | argmax (lossless here) | argmax-against-argmax (**biased**) |
| DFlash round loop (LFM2 DSpark drafter) | argmax (lossless here, probe-gated) | declines to classic decode |
| DFlash round loop (Speculators or OpenBMB DSpark drafter on a dense Qwen3 or Llama target, #3261) | argmax (lossless here, probe-gated) | refused offline, declines to classic decode in the server |
| DFlash round loop (Inkling 2 bundled DSpark drafter, #3374) | argmax (lossless here, probe-gated) | refused offline, declines to classic decode in the server |
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

`mlxcel generate --draft-model <dir>` runs one of three round loops: MTP (explicit
`--draft-kind mtp`, or an MTP `model_type`), DFlash (a DFlash drafter directory,
explicit or auto-detected, on a target with an offline DFlash arm, today Laguna),
or the full-model drafter (#2757): an ordinary smaller checkpoint, auto-detected
when the directory carries no DFlash-family marker and no MTP `model_type`, or
selected with `--draft-kind full-model`. The full-model drafter is the engine's
token-only loop, the one `--prompt-lookup` runs, with the draft model's own KV
caches rolled back next to the target's; it needs a drafter that shares the
target's tokenizer, and a draft vocabulary larger than the target's is refused
before the load. Anything else fails before any drafter load, with an error
naming `--draft-kind mtp`, `--draft-kind dflash`, `--draft-kind full-model` and
`--prompt-lookup`. A target or a drafter whose state a cache trim cannot roll
back (recurrent, hybrid and model-owned families) is refused by the one rule
`Engine::verify_rounds_unsupported` applies to prompt lookup (#2495), on
`mlxcel-server` too. The CLI prints the drafter's acceptance and draft-forward
counts on stderr after the run.

Greedy exactness is the token-only loop's, which does not reach the bar the MTP
and DFlash paths reach through their block-versus-chain probe: at temperature 0
the stream is the target's own argmax at every position, but it is not guaranteed
byte-identical to plain decode, and nothing gates on the difference. On the
M1 Ultra record for #2757, 32 of 36 real-pair runs matched plain decode. The other
four come from the target alone: `--prompt-lookup`, which has no draft model and
no draft caches, emits the same divergent streams on the same prompts, and a
teacher-forced replay of the target over its own plain stream reproduces the flip
once its KV history is built by multi-token forwards. The pull request for #2757 holds that evidence.
A drafter's proposals cannot move the stream on their own, because the
target decides every position from its own caches; a misaligned draft cache costs
acceptance only. The sampled stream passes the chi-square and Markov guards.

### Token-only verify blocks, per family

Issue #3418 traced where a token-only verify block parts from single-token decode and decided each family by cost. Nothing gates on the result: prompt lookup and the full-model drafter run on every model they ran on before. Numbers and commands are in [speculative-exactness-followup-m3ultra-2026-11-25.md](benchmark_results/speculative-exactness-followup-m3ultra-2026-11-25.md).

| Family | First difference | Decision |
|--------|------------------|----------|
| DeepSeek V3 (`DeepSeekV3Model`: deepseek_v3, kimi_k2, joyai_llm_flash, kimi_vl, kimi_k25, youtu) | Layer 0: a single query takes the fused absorbed MLA kernel, a block takes rope-folded SDPA chosen by the `use_absorbed_mla` cost model; on bf16 also gemv against gemm and `gather_mm` | `MLXCEL_MLA_ROW_INVARIANT=1` makes the block byte-identical (per-row MLA, row-invariant dense products, per-position float experts). Off by default: it costs prompt lookup 24% (8-bit) to 43% (bf16) of its throughput on a copy-heavy reply. A quantized checkpoint also needs `MLXCEL_QMV_WIDE=0` |
| Xing4.0 | The same layers, and the hyper connection's dense mixer and batched `hc_expand` | The same switch; equal on the 8-bit and optiq checkpoints with `MLXCEL_QMV_WIDE=0` |
| GLM-4.7-Flash, engine path | The block attention, where the MTP path already ran row-wise | The same switch routes the block to the MTP gate's row-wise verify forward; equal with `MLXCEL_QMV_WIDE=0` |
| DeepSeek V3.2 | The same split in its DSA attention, plus the indexer's top-k scored from a block-wide product | The same switch (#3421): every block row runs the decode step's single query over its prefix of the stored window with its own top-k selection, and the indexer, dense and gate products are row-invariant. Covered by a tiny-model test only; the 352 GB checkpoint is not loaded |
| DeepSeek V2 (`deepseek-v2-lite-chat-4bit-mlx`) | Layer 0 output on both kernels: the latent cache's absorbed attention takes the fused single-query kernel for one query and SDPA for a block (or per-head SDPA on the decompressed layout) | The same switch (#3421): the block appends and attends one row at a time on either layout. Equal on the narrow kernel in all 9 cells |
| Kimi Linear, Kimi K3, LongCat (both variants) | Not reached: the token-only round refuses these models (model-owned recurrent or n-gram state that a cache trim cannot roll back), so no verify block forms | Out of scope (#3421). Kimi Linear's KDA layers never see a token-only block, so their block-versus-chain behavior is not measured |
| Arcee, K2 Horizon (Llama-class, bf16) | Layer 0 output (layer 1 in one Arcee cell): gemv against gemm in the shared dense projections, the compiled MLP and the block SDPA | The same switch (#3421): `Llama3Model` and `ArceeModel` enter the scope, the projections take the row-invariant kernel, the fused rope and MLP launchers give way to the composed graph inside it, and a verify block attends one row at a time. Equal on the wide kernel in all 9 cells |
| OLMoE (4-bit) | Layer 0 output, on both quantized kernels: the fused single-token MoE kernel against the block's `gather_qmm`, and the block SDPA | The same switch (#3421): a verify block routes and runs the experts one position at a time and attends one row at a time. Equal with `MLXCEL_QMV_WIDE=0` in all 9 cells |

With the switch on and the narrow kernel, `--prompt-lookup` text equalled plain decode on six of six checkpoint and prompt pairs on an M3 Ultra (gigachat3 8-bit, Kimi-VL, moonlight; a copy prompt and a 2.5K-token prompt each); with it off, two of the six diverged. On the bf16 checkpoint both settings matched plain decode on both prompts.

On the #3421 families (issue #3425) exact lookup stays faster than plain decode on a copy-heavy reply (OLMoE 1.15x, DeepSeek V2 1.46x, K2 Horizon 1.09x, Arcee 3.23x), though the switch takes 34% of OLMoE's lookup throughput there. On prose, where acceptance is under 0.1, prompt lookup is slower than plain decode on **DeepSeek V2** (`deepseek-v2-lite-chat-4bit-mlx`, 146.3 against 173.2 tok/s with the switch on) and **OLMoE** (`olmoe-1b-7b-0125-4bit`, 353.8 against 414.4), with the switch on or off; on Arcee and K2 Horizon it is within 3% of plain decode. Section 4 of the follow-up page has the table.

The Gemma 4 26B A4B MoE checkpoint is a different case, because its MTP path is gated: its verify block now runs the routed experts per position and its attention row-wise, so the startup probe passes on the narrow kernel and the drafter engages as it does on the 12B and 31B checkpoints (`MLXCEL_GEMMA4_MOE_VERIFY_PER_POSITION`).

First-generation Inkling (`inkling`, `inkling_mm_model`, issue #3422) now takes token-only verify rounds (`--prompt-lookup`, the server's `--draft-kind prompt-lookup` and the full-model drafter). Its state is model-owned, so a round is admitted on `supports_speculative_verify_rewind`, which follows `MLXCEL_INKLING_ROW_INVARIANT` (on by default): a verify block of 2 to 16 rows runs in the MTP verify scope, is byte-identical to the single-token chain, and records each convolution's input window, and `rewind_decode_appends` drops the rejected tail (the whole block on a failed round or a width warm-up) by trimming the KV tail and taking each convolution state from that window. A block wider than 16 rows also records its windows and rewinds, but its rows are not byte-identical to the chain. With the variable off the gate refuses the family as before. Measurements are in [inkling-prompt-lookup-m3ultra-2026-11-25.md](benchmark_results/inkling-prompt-lookup-m3ultra-2026-11-25.md).

A DFlash or DSpark drafter on a target without an offline arm gets the same
refusal plus a pointer to `mlxcel-server`, which drives the DFlash round loop for
more targets (#1168). These checkpoints ship no `embed_tokens` and no `lm_head`,
because they borrow both from the target when they bind. The LFM2 DSpark drafters
(#1339) and the Muse Glimmer assistant (#1343) have the same shape. Laguna and
Inkling 2 have offline arms; Inkling 2's drafter is the `mtp.safetensors` shipped
in the target checkpoint, so `--draft-model` names the target directory itself
and the pairing is detected from its `mtp_config` (#3374). Dense Qwen3 and Llama
targets have an offline arm too, for the Red Hat Speculators and OpenBMB DSpark
drafters (#3261); the Speculators checkpoint ships its own `embed_tokens` and
`lm_head`, the OpenBMB one borrows both.

The discriminator is structural (a `dflash_config` block and/or
`architectures: ["DFlashDraftModel"]`, `["Lfm2DSparkDraftModel"]` or
`["Qwen3DSparkModel"]` in the drafter's `config.json`, a root
`speculators_model_type: "dspark"`, or the Muse assistant's own
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
  The target is context-free, so these two guard the sampler path into the
  stream and cannot see a wrong accept/reject decision.
* `markov_stream_keeps_the_target_conditional`: the same statistic over
  `(previous, next)` on a target whose distribution depends on the previous token,
  dof 30. This is the stochastic guard on the accept/reject decision.
* `chi_square_rejects_known_wrong_acceptance_rules`: power calibration on host draws. The same
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
