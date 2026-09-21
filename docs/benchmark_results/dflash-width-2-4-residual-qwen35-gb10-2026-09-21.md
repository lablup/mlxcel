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

## The cause: the verify block's attention does not reproduce decode's, on one kernel

`Qwen3NextAttention::attend_per_position` is what makes a verify block reproduce single-token decode. For each query position `i` of a `[B, H, T, D]` block it attends `queries[:, :, i:i+1, :]` to the causal prefix `keys[:, :, ..prefix + i + 1, :]` with no mask, so each row sees exactly the prefix a decode step would, and `target_verify && l > 1` is the only condition that selects it. Classic decode takes the `l == 1` arm of the same match and reaches the same `layers::attention` entry point with the same absence of a mask.

One switch decides whether those two calls agree.

**`MLXCEL_SDPA_VECTOR_LARGE_D=0` collapses the divergence, in two independent arms.** That switch does one thing: it decides whether `head_dim` 256 and 288 are accepted by CUDA's `supports_sdpa_vector` gate, and so whether a single-query attention call takes the fused `sdpa_vector` kernels or the materializing fallback (issue #675). Both classic decode and every row of `attend_per_position` are single-query calls, so both move together.

Served, one binary, temperature 0:

| arm | completion sha256 | first token differing from classic |
|---|---|---:|
| classic, stock | `2c76b0a181` | |
| width 4, stock | `3e60b1574c` | 105 |
| classic, `MLXCEL_SDPA_VECTOR_LARGE_D=0` | `4c37650547` | |
| width 4, `MLXCEL_SDPA_VECTOR_LARGE_D=0` | `4c37650547` | none, 200 of 200 |

Both arms move, which is expected and is why this table is not evidence about classic decode; what matters is that they move onto each other. In process, replaying the served width-4 run's own 79 rounds and 56 rewinds against a classic chain computed in the same run, the same switch takes the target from 1 disagreement of 201 greedy positions to 0.

A single kill switch collapsing a 95-token divergence to nothing is stronger than any of the exclusions that preceded it, and it takes the remaining candidates with it. Rollback, the gated-delta scan, the quantized matmul kernel, the drafter, the round loop and the server process are none of them disabled by this switch, and all of them stop mattering when it is off.

**Two more arms place it at `T > 1` and nowhere else.** At `T = 1`, prefilling classic with `forward_internal` and the burst with `forward_prefill_with_capture_layers` and then walking the real 200-token transcript one token at a time, every step's logits agree in all 496640 bytes and the prefill's do too. `attend_per_position` at `T = 1` slices a one-row tensor, so its call is classic's call, and nothing differs. At `T = 4` the divergence reproduces in process with no drafter and no server, at exactly the served position, taking 5741 where classic takes 11439. That kills the hypothesis PR #1939 left standing, that the drafter's interleaved MLX work was the cause: there is no drafter in that arm.

### What it is not: the per-row tensor layouts

The obvious reading of "same math, one kernel, two answers" is that the two calls hand the kernel differently-laid-out tensors. A one-row slice of a `[B, H, T, D]` block does carry the block's strides, and single-token decode does hand the same call a `[B, H, 1, D]` built from a one-row forward. Measured, that is not the difference. Copying the per-row query slice to a fresh contiguous array leaves the served width 2 and 4 completions exactly where they were (`3e60b1574c`, first differing token 105), and copying the key and value slices along with it changes nothing either. The change that produced those arms is reverted; the negative result is recorded here instead.

Reading the CUDA gate confirms why. `supports_sdpa_vector` admits a call when `q.shape(2) < 4`, which both satisfy, and `sdpa_vector`'s own `q_copy_unless` and `kv_copy_unless` predicates accept both layouts without copying: a `[1, H, 1, D]` query passes on `strides[3] == 1 && strides[2] == D * H && strides[1] == D`, which the block's slice satisfies as well as decode's tensor does, and a key or value whose batch dimension is 1 is accepted whatever its strides. The 1-pass and 2-pass split is on `k.shape(2) > 1024`, which neither reaches. So the arguments the kernel sees are the same shape, the same strides and the same values, and the difference is in what MLX does around the call rather than in what is handed to it.

## Why the gate reported the property intact

`Qwen35Model::probe_block_chain_exactness` compares a verify block against a single-token chain in logit bytes, which is the right comparison. It prefills 8 synthetic tokens first: `PROBE_PROMPT_LEN` is 8, chosen so the attention layers "hold a real KV prefix rather than the empty-cache special case". An 8-token prefix is not a served KV layout. The cache has not grown past its first step-aligned allocation, and the difference this record is about does not appear there.

Measured on the real checkpoint on this host, the probe reports byte-identity at widths 2 and 4 and divergence at 8 and 16, which is what PR #1939 recorded and wired the gate to. The served arms disagree with it at widths 2 and 4. So the gate's verdict was a false pass, not a correct pass that something downstream then violated, and the probe needs a prefix long enough to reproduce the layout a served request has before its verdict means anything.
