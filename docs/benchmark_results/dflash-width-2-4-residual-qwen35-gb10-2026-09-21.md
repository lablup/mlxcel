# Qwen 3.5 DFlash: the residual at verify widths 2 and 4 (GB10, 2026-09-21)

Issue #1935, continuing from `dflash-verify-divergence-qwen35-gb10-2026-09-20.md` (PR #1939). That record fixed the burst's prompt prefill, which had been computed as if it were a verify block, and left one symptom open: at widths 2 and 4 the served greedy completion still parts from classic decode, at generated token 105 rather than token 9.

This record answers what that residual is.

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
MLX_CUDA_ARCHITECTURES: 121
mem_total_gib: 121.7
```

Driver budget: the boot's cumulative kernel `NV_ERR_NO_MEMORY` count was 0 before this session. Nothing here is a timing measurement, so every arm ran while the host was also compiling; greedy token streams do not depend on host load, and the two classic arms below are the control that says so.

## Method

Served arms: one `mlxcel-server` per arm, `--ignore-eos --max-batch-size 1`, the #1797 harness's 158-token Python prompt, two non-streaming `POST /v1/completions` per arm at `temperature 0` with `max_tokens 200` and `logprobs`, both against the same server process so a cross-request difference would show as two different token lists. In-process arms: `#[ignore]`-gated tests in `src/models/qwen3_5_dflash_probe_tests.rs`.

The classic reference is anchored on this binary: `MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate --no-chat-template --temp 0 --max-tokens 200` reproduces the served classic completion exactly and supplies the 158 prompt ids and 200 generated ids the in-process arms take. One trap worth recording: the prompt file ends in a newline and shell command substitution strips it, which tokenizes to 157 ids instead of 158 and shifts the whole reference by one token.

Data and harness: `data/dflash-width-2-4-residual-gb10-2026-09-21/`.

## What the served arms establish

| arm | completion sha256 | tokens | rounds | rewinds | first token differing from classic |
|---|---|---:|---:|---:|---:|
| classic (twice) | `2c76b0a181` | 200 | | | |
| width 4 (twice) | `3e60b1574c` | 200 | 79 | 56 | 105 |
| width 2 (twice) | `3e60b1574c` | 200 | 114 | 28 | 105 |

**The round algebra holds, in all four speculative bursts.** Checked offline from a per-round `debug` transcript this change adds, with no GPU and nothing replayed: every round's bonus is the previous round's last emitted token, `accepted` is the longest common prefix of the drafter's proposals and the target's block argmax, and the emitted tokens are exactly `draft[:accepted] + [target[accepted]]`. So every token the burst emits is the target's own argmax under a cache history consistent with the tokens already emitted, and neither the round loop nor the burst wrapper is mis-bookkeeping anything. This is what makes the rest of the record a question about the target's forward rather than about the loop around it.

**The residual does not track round structure.** Width 4 runs 79 rounds with 56 rewinds, width 2 runs 114 with 28, and the two emit the same 200 ids. At the divergence the two arms even reach it at different block positions: width 4 at row 2 of the block `[11, 198, 262, 1866]`, width 2 at row 1 of `[198, 262]`. Both take 5741 there; classic takes 11439.

**The logprobs place it just below the reporting floor.** Reported logprobs on this checkpoint are quantized to 0.125. Of the 105 positions before the divergence, 100 are bit-identical between classic and both speculative arms, and the five that differ do so by exactly one step, at the same five indices (62, 68, 86, 88, 96) with the same values at both widths. At 105 itself classic's 11439 and the burst's 5741 both report -2.125.

That pattern is a difference in the logits that is below the reporting floor almost everywhere, surfacing as a one-step logprob difference from index 62 onward and finally crossing an argmax at 105, where the top two candidates sat within a reporting step of each other. It is not a rollback, not a round-boundary effect, and not nondeterminism: both requests in each arm are byte-identical to each other.

## The cause: a verify row and the decode step it stands for are not the same call

`Qwen3NextAttention::attend_per_position` is what makes a verify block reproduce single-token decode. For each query position `i` of a `[B, H, T, D]` block it attends `queries[:, :, i:i+1, :]` to the causal prefix `keys[:, :, ..prefix + i + 1, :]` with no mask, so each row sees exactly the prefix a decode step would. The arithmetic is right. The layout is not.

A one-row slice of a `[B, H, T, D]` tensor has its heads `T * D` apart. Single-token decode hands the same attention call a freshly built `[B, H, 1, D]`, whose heads are `D` apart. Same values, same shape, different strides. On CUDA this checkpoint's `head_dim` is 256, which `MLXCEL_SDPA_VECTOR_LARGE_D` (issue #675) routes to the fused `sdpa_vector` kernels, and that kernel is not bit-equal across the two layouts. The key and value slices are unaffected: both paths slice the same cache buffer to `prefix + i + 1`, so their strides already match, which is why the query row is the whole of it.

Three arms establish this, and each rules out what the others cannot.

**At `T = 1` the two forwards are byte-identical.** Prefilling classic with `forward_internal` and the burst with `forward_prefill_with_capture_layers`, then walking the real 200-token transcript one token at a time, every step's logits agree in all 496640 bytes and the prefill's do too. At `T = 1` the one-row slice IS the whole tensor, so the layouts coincide and the difference has nowhere to appear. This is what rules out everything that is not a block: the projections, the gated-delta layers, the rotary, the prefill, the caches.

**At `T = 4` it reproduces with no drafter at all.** Replaying the served width-4 run's own 79 rounds and 56 rewinds in process, with the rejected rows carrying a wrong token id, the target disagrees with the classic chain at exactly one of 201 greedy positions, at 105, taking 5741 where classic takes 11439. The served position, from a replay that contains no drafter and no server. That kills the standing hypothesis that the drafter's interleaved MLX work was the cause: there is no drafter in this arm. Driving `run_dflash_on_target` with the real drafter bound reproduces the same position, which is consistent rather than additional.

**Turning the fused kernel off makes the two byte-identical.** `MLXCEL_SDPA_VECTOR_LARGE_D=0` routes single-query attention off `sdpa_vector` for `head_dim` 256 on both paths. Served, same binary:

| arm | completion sha256 | first token differing from classic |
|---|---|---:|
| classic, stock | `2c76b0a181` | |
| width 4, stock | `3e60b1574c` | 105 |
| classic, `MLXCEL_SDPA_VECTOR_LARGE_D=0` | `4c37650547` | |
| width 4, `MLXCEL_SDPA_VECTOR_LARGE_D=0` | `4c37650547` | none, 200 of 200 |

Both arms move, which is expected and is why the row is not evidence about classic decode; what matters is that they move onto each other. A single kill switch collapsing a 95-token divergence to nothing is stronger than any of the exclusions that preceded it, and it takes the remaining candidates with it: rollback, the gated-delta scan, the quantized matmul kernel, the drafter, the round loop, the server process. None of those is disabled by this switch, and all of them stop mattering when it is off.

## Why the gate reported the property intact

`Qwen35Model::probe_block_chain_exactness` compares a verify block against a single-token chain in logit bytes, which is the right comparison. It prefills 8 synthetic tokens first: `PROBE_PROMPT_LEN` is 8, chosen so the attention layers "hold a real KV prefix rather than the empty-cache special case". An 8-token prefix is not a served KV layout. The cache has not grown past its first step-aligned allocation, and the difference this record is about does not appear there.

Measured on the real checkpoint on this host, the probe reports byte-identity at widths 2 and 4 and divergence at 8 and 16, which is what PR #1939 recorded and wired the gate to. The served arms disagree with it at widths 2 and 4. So the gate's verdict was a false pass, not a correct pass that something downstream then violated, and the probe needs a prefix long enough to reproduce the layout a served request has before its verdict means anything.
