# Qwen 3.5 DFlash verify divergence from classic decode (GB10, 2026-09-20)

Issue #1935, from the sweep recorded in `draft-block-width-default-gb10-2026-09-20.md` (issue #1797, PR #1937). That sweep found greedy speculative output on `models/mlx/qwen3.5-4b-4bit` with `models/mlx/qwen3.5-4b-dflash` not byte-identical to classic decode at any verify width, identically at 2, 3, 4, 5, 6, 7, 8 and 16, and could not separate the candidates. This record separates them.

Nothing here is a throughput measurement. Every arm compares greedy token streams, which do not depend on host load, so the arms were deliberately run while the host was busy with other sessions' builds and with mlxcel CI. Timing numbers appear only where they are evidence that a kill switch was live, and are labelled as such.

## Host

```
host: spark-101
kernel: 7.0.0-1019-nvidia
arch: aarch64
nvidia_driver: 580.178.04
gpu: NVIDIA GB10, compute capability 12.1
cuda_toolkit: 13.0
mlx_pin: 81ba1c6a0e50a9268b931579c2d4f1158b9aab5a
rustc: 1.97.1 (8bab26f4f 2026-07-14), the version rust-toolchain.toml pins
mem_total_gib: 121.7
```

Driver budget: the boot's cumulative kernel `NV_ERR_NO_MEMORY` count was 0 before this session and 0 after it, across roughly thirty server starts and a dozen in-process model loads. That matches the zero delta the #1797 sweep measured on the same driver and contradicts the 38-per-run figure from `580.173.02`, which is why the next long sweep should still calibrate from its own first run rather than inherit either number.

## Method

Served arms: one `mlxcel-server` per arm, `--ignore-eos --max-batch-size 1`, a fixed 158-token Python prompt (the #1797 harness's `prompt_retry.txt`), one non-streaming `POST /v1/completions` at `temperature 0` with `max_tokens 200`, compared by the sha256 of the completion text and by the per-token list the `logprobs` field returns. In-process arms: `#[ignore]`-gated tests in `src/models/qwen3_5_dflash_probe_tests.rs`, which load the real checkpoint and drive the model directly.

The classic reference is anchored twice. Two classic served arms at the start and end of the session are byte-identical to each other (`2c76b0a181`), and `MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate --no-chat-template --temp 0` on the same prompt produces the same 200 tokens byte for byte. It also supplies the prompt and reference token ids the replay arms take.

Every in-process arm below uses `Qwen35Model::forward_internal` as the stand-in for served classic decode, and that stands on the call graph rather than on the equality above. The scheduler's B = 1 decode calls `LanguageModel::forward_with_sequence_id` (`src/server/batch/scheduler/decode_tick.rs`), whose Qwen 3.5 impl is `forward_with_sequence_caches(input_ids, None, seq_id)`, which resolves the MRoPE entry and calls `forward_internal`. The offline CLI reaches the same function through `LanguageModel::forward`, which is the same call with `seq_id = None`. The two endpoints agreeing on 200 argmaxes is then a consistency check on that reading, not the reading itself.

## The root cause: the burst prefilled the prompt as if it were a verify block

`Qwen35Model::forward_speculative` sends every full-attention layer through `Qwen3NextAttention::attend_per_position`, which computes attention one query position at a time so each row sees exactly the prefix a single-token decode step would see. That is correct for a verify block and is not the computation classic decode runs over a prompt, which is one batched causal attention.

The DFlash burst's first call is the prompt prefill, through `SpeculativeTarget::prefill_forward_with_capture_layers`. That hook defaults to the verify forward, and Qwen 3.5 did not override it, so the burst prefilled the prompt the per-position way. It therefore entered its first round holding different KV and gated-delta state than classic decode, and its greedy output drifted away within a few tokens at every width alike.

Measured, before the fix: the two prefills' last-row logits differ in 173206 of 496640 bytes, and the first cache state to differ is layer 4's, the layer after the first full-attention layer (`full_attention_interval` is 4, so layers 0 to 2 are gated-delta and layer 3 is the first attention layer). Layers 0 to 3 agree because their inputs are identical; layer 3's attention output differs without its own K and V differing, which is what puts the difference in the attention computation rather than in a projection.

The MTP path already carries this fix. `Qwen35Model::forward_prefill_with_last_hidden` exists for exactly this reason, and its doc comment has said so since #1182: `forward_speculative` "does not match the classic *prefill* numerics", and "the classic path's first token is sampled from the batched causal prefill logits, so MTP's first bonus must come from the same computation to stay byte-identical". The DFlash path never got it.

### How long this has been true

`attend_per_position` landed on 2026-05-24 (`4038da96`, issue #78), eleven days after the B = 1 speculative burst was wired (`ca938b67`, 2026-05-13). At PR #1795's merge commit `e391ae9c` (2026-09-11), `Qwen35DecoderLayer::forward_with_capture` already called `forward_with_position_ids_verify(..., true)` and `Qwen35Model` still did not override `prefill_forward_with_capture_layers`, so the burst prefilled the same way then.

This answers issue #1935's first question without a rebuild at that commit, and more conclusively than a rebuild would: the property did not break after PR #1795, it never held as PR #1795's record stated. `dflash-verify-fixed-cost-gb10-2026-09-11.md` reports byte-identity at widths 2, 4 and 8 on this pairing, and that claim should be read as not established.

## What the fix changes

`Qwen35Model::forward_prefill_with_capture_layers` runs `forward_internal`'s loop, batched causal mask and all, while capturing the per-layer hidden states the drafter is seeded from, and returns no gated-delta rollback snapshots (a prefill is never rolled back, and they are prompt-sized rather than block-sized). Both `SpeculativeTarget` impls for the family override the hook to call it, so the B = 1 and B > 1 burst arms are both covered.

In process, against the real 200-token classic transcript:

| arm | prefill last-row logit bytes differing | greedy argmax positions disagreeing, of 201 |
|---|---:|---:|
| before the fix | 173206 of 496640 | not measured; the served arms parted at token 9 |
| after, block 4 | 0 | 0 |
| after, block 2 | 0 | 0 |
| after, block 8 | 0 | 2, first at 93 |

After the fix every layer's cache state is byte-identical between the two prefills, not merely the logits. The width-8 disagreement is the block-versus-chain divergence described below and is not something a prefill fix could reach.

One thing about that cache comparison is worth stating because the first version of the test got it wrong: it has to run immediately after both prefills, before the classic arm's single-token loop advances its caches. Run afterwards it reports the offset gap the chain loop itself opened (358 against 158) and reads as a layer-0 divergence that is not there.

Served, same session, same binary per row:

| arm | completion sha256 | first token differing from classic |
|---|---|---:|
| classic (twice, opening and closing) | `2c76b0a181` | |
| before the fix, widths 2 and 4 | `57d291aca2` | 9 |
| before the fix, widths 8 and 16 | `57d291aca2` | 9 |
| after the fix, widths 2 and 4 | `3e60b1574c` | 105 |
| after the fix, widths 8 and 16 | `57d291aca2` | 9 |

Widths 8 and 16 landing back on the pre-fix hash is not the fix failing to apply there. Their verify block is independently divergent, the flip at token 9 is a near tie (the classic choice carries logprob -0.875 against -0.750 for the speculative one), and once a greedy trajectory flips at a near tie the rest of it follows deterministically, so two different causes reach the same completion.

## The second, separate defect: widths 8 and 16 are genuinely not exact

`Qwen35Model::probe_block_chain_exactness`, run on the real checkpoint, reports:

| width | verdict |
|---|---|
| 2 | byte-identical to the single-token chain |
| 4 | byte-identical to the single-token chain |
| 8 | differs at block position 0 in 229971 of 496640 logit bytes |
| 16 | differs at block position 0 in 229971 of 496640 logit bytes |

The boundary is `if (can_use_qmv && (M * B < 8))` in `mlx/backend/cuda/quantized/quantized.cpp`: a verify block of 8 or more rows leaves the `qmv` family for `qmm_sm80`, and the two are not bit-equal. That is a real numerical difference between two kernels rather than a defect, and it is the failure class the exactness probe exists for. It is also what the #1797 record's forced-`MLX_ENABLE_TF32=0` pass was seeing when its speculative arms split into 2 through 7 against 8 and 16; that grouping was a correct observation of a second effect, described there as if it were the only one.

`Qwen35Model::dflash_exactness_allows` now runs that probe through the shared `mtp_exactness_gate`, and both Qwen 3.5 `DFlashTargetModel` impls call it. Deliberately not `mtp_exactness_allows`, whose first precondition is `metal_is_available()` and which would therefore decline every CUDA burst without measuring one.

Served, with the gate in place:

| arm | completion sha256 | bursts completed | what happened |
|---|---|---:|---|
| classic | `2c76b0a181` | 0 | |
| width 2 | `3e60b1574c` | 2 | probe passed, burst engaged |
| width 4 | `3e60b1574c` | 2 | probe passed, burst engaged |
| width 8 | `2c76b0a181` | 0 | probe declined, served by classic decode, byte-identical to classic |
| width 16 | `2c76b0a181` | 0 | probe declined, served by classic decode, byte-identical to classic |
| width 16, `MLXCEL_MTP_ALLOW_INEXACT=1` | `57d291aca2` | 2 | override engaged the burst and logged the forfeit |

## What remains, and what it is not

Widths 2 and 4 still part from classic decode at generated token 105. This is its own finding and it does not share a cause with the prefill: the prefill is now byte-exact, and the prefill bug was masking this one rather than producing it.

It is reproducible and deterministic, and it is invariant to everything cheap. Widths 2 and 4 produce identical completions despite drafting 1 and 3 proposals per round over 120 and 79 rounds, so it does not track the drafter's proposal pattern, its round boundaries, or its allocation pattern. It is unchanged by `MLXCEL_QMV_MULTIROW=0`, by `MLX_USE_CUDA_GRAPHS=0`, by `--no-warmup`, and by omitting `logprobs` from the request.

Against that, the in-process target path is exhaustively exact over the same transcript at widths 2 and 4:

- prefill: every layer's cache state byte-identical to `forward_internal`'s;
- verify block: byte-identical to the single-token chain at every one of the 32 layers at block position 0, and 0 of 201 greedy argmax positions disagreeing over the full transcript;
- rollback: 0 disagreements across accept patterns `1,2,3`, `1,2,3,4`, `4,1`, `4,4,1`, `1,4,2,4,3` and `2,4,1,3,4,4,1,1`, which mix full-accept rounds that never rewind with partial ones that do, and with the rejected rows carrying a wrong token id (9999) the way a real round's rejected rows carry the drafter's wrong proposals;
- rollback again, against the served run's **own** accept sequence rather than a synthetic cycle: the width-4 server logs its per-round accept lengths (a `debug` line this change adds), and replaying those 79 rounds in process reproduces the served burst's exact cache history, 56 rewinds at the same depths in the same order. It disagrees with the chain at 0 of 201 positions, with the rejected rows wrong and with them correct alike;
- with the served capture list `[1, 8, 15, 22, 29]` rather than no capture, and with the round loop's own whole-block `argmax_last_axis` rather than one call per sliced row: 0 disagreements either way.

The last of those is the strongest exclusion available short of running the real drafter: the served burst's round structure is not an approximation of it any more, it is the same 79 rounds with the same rewind depths, and the target reproduces classic decode through all of them. Rejected-row content is excluded as well, since feeding the correct tokens and feeding 9999 give the same answer.

So the residual is in the served burst wrapper rather than in the target's forward, verify or rewind, and one hypothesis is left standing: the drafter executing MLX work between the target's forwards, which every arm here omits. The next step is to drive `DFlashGenerator::run` in process with the real drafter bound and bisect from there, which is the one configuration these arms do not reproduce. That is a test worth building and is not in this change.

## Controls, including one that was void

**Gated-delta chunked scan: excluded.** `MLXCEL_GDN_CHAIN_PARITY=0` swaps the verify block's bit-exact sequential loop for the 64-row chunked scan. Before the prefix fix the switch was live (burst 5511 ms against 7826 ms, acceptance 0.5325 against 0.5281) and changed the completion not at all, which is what excluded the scan as the cause of the token-9 divergence. After the fix the same switch does change the completion (`3e60b1574c` against `08f8cfb1fe`), which is the expected direction: it forfeits an exactness the fix had just made load-bearing. Both readings belong in the record, because the first one on its own would suggest the scan can never matter.

**Multirow quantized matmul: excluded, twice.** `MLXCEL_QMV_MULTIROW=0` routes a 4-row verify through the same `qmv_kernel` classic decode uses. The completion is unchanged before the fix and unchanged after it.

**TF32: not the discriminator.** `MLX_ENABLE_TF32=0` moves both the classic arm (`92e90219fd`) and the width-4 arm (`f89987e15e`), and does not bring them together.

**`MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0`: void, not negative.** This switch diverts a 2-to-32-query masked SDPA call from cuDNN to MLX's ops fallback. On this checkpoint it can do nothing: `head_dim` is 256 and `cuda_sdpa_materializes_scores` records that cuDNN's flash SDPA requires `head_dim <= 128`, so these calls never reach cuDNN and there is nothing to divert. The arm changed neither the completion nor the burst time (5511 ms against 5480 ms), and the unchanged time is how it was identified as inert. Recording it as "no effect" would have read as evidence that the verify's attention dispatch does not matter, which it is not.

**`MLXCEL_SDPA_VECTOR_LARGE_D=0`: live, and informative about the shape of the problem rather than the cause.** It moves classic decode's single-row attention off the fused `sdpa_vector` kernel, and classic's completion changes (`4c37650547`). So this pairing's greedy output is not robust to the attention kernel it decodes with, which is worth knowing before reading any byte-identity claim about it.

## Files

Diagnostics: `src/models/qwen3_5_dflash_probe_tests.rs`, six `#[ignore]`-gated tests. The three that take a recorded transcript read it from `MLXCEL_Q35_PROBE_PROMPT` and `MLXCEL_Q35_PROBE_REFERENCE`; `MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate` prints both lines. To replay a served burst's own round structure, read its accept lengths from the server at `RUST_LOG=mlxcel::server::batch::dflash_target=debug`, add one to each (the log reports accepted draft tokens, the test takes kept rows) and pass them as `MLXCEL_Q35_PROBE_ACCEPTS`.

```
cargo test --release --features cuda -p mlxcel --lib -- --ignored --test-threads=1 \
  --nocapture qwen3_5_dflash_probe_tests
```

`--test-threads=1` is required for every CUDA test run on this host.
