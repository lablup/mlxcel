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
