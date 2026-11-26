# Raon-1 mini DFlash2 drafting (M3 Ultra, 2026-11-26)

Issue #3436. `KRAFTON/Raon-1-mini` (bf16 + MXFP8, `models/raon-1-mini`) drafted by `KRAFTON/Raon-1-mini-dflash2` (bf16, `models/raon-1-mini-dflash2`) through `mlxcel generate --draft-model ... --draft-kind dflash`, against plain decode of the same binary. Temperature 0, presence penalty 0, 256 generated tokens, default reasoning effort (`high`), chat template on.

Host: Apple M3 Ultra, 512 GB, Metal. Every run of a table is serialized under the shared GPU lock, arms interleaved plain, width 3, width 5, width 7, twice (ABAB). Load average 3.11 / 4.68 / 6.20 at the start (20:52) and 1.73 / 3.20 / 5.18 at the end (20:56) of the main tables; 7.67 / 4.96 / 5.60 to 5.98 / 4.80 / 5.52 for the row-invariant kernel table (20:58 to 20:59).

Width is the verify block: one bonus row and `width - 1` drafts (`--draft-block-size`).

## Greedy equality

Every drafted run printed exactly the text of plain decode, on all three prompts and all three widths, in both rounds (21 of 21 comparisons identical, `diff` of the generated text). The long prompt is 3137 tokens (the first 14000 bytes of `README.md` plus a summary instruction), past the 2048-token prefill chunk. With `MLXCEL_RAON_ROW_INVARIANT=1` drafted output is again identical to plain decode under the same setting (3 of 3).

The startup probe (`Raon1MiniModel::dflash_exactness_allows`, three synthetic draws through `mtp_exactness_gate`) passed at every width, so the drafter engaged without `MLXCEL_MTP_ALLOW_INEXACT`.

Sampled drafting (each verify row sampled with plain decode's sampler and accepted while it equals the draft) on the code prompt at `--temp 0.7 --top-p 0.95 --seed 7`, 200 tokens, one run each: plain 32.64 tok/s, width 5 49.32 tok/s (1.51x, acceptance 0.500, 2.97 emitted per verify). The distribution itself is pinned by the `drafter::dflash::sampled_verify` tests on the round loop (a Markov target's joint distribution at widths 2 and 4, and a presence penalty that sees the accepted drafts), not by these runs.

## Throughput, default kernels

tok/s, round 1 / round 2. Speedup is the mean of the two rounds over plain decode.

| Prompt | Plain | Width 3 | Width 5 | Width 7 |
|---|---|---|---|---|
| Short prose (19 tokens, hash-table collisions) | 33.55 / 33.54 | 46.26 / 47.00 (1.39x) | 43.69 / 43.50 (1.30x) | 36.86 / 36.70 (1.10x) |
| Code (binary search tree in Python) | 33.48 / 33.49 | 53.66 / 53.25 (1.60x) | 58.80 / 59.11 (1.76x) | 54.31 / 54.34 (1.62x) |
| Long prompt (3137 tokens, summary) | 27.87 / 27.86 | 49.20 / 48.68 (1.76x) | 46.99 / 46.97 (1.69x) | 39.71 / 39.96 (1.43x) |

Acceptance (identical in both rounds, the runs are deterministic):

| Prompt | Width 3: rate, mean accepted, emitted per verify | Width 5 | Width 7 |
|---|---|---|---|
| Short prose | 0.583, 1.16, 2.16 | 0.371, 1.49, 2.48 | 0.257, 1.53, 2.52 |
| Code | 0.738, 1.48, 2.48 | 0.589, 2.36, 3.36 | 0.459, 2.75, 3.75 |
| Long prompt | 0.659, 1.32, 2.32 | 0.447, 1.78, 2.77 | 0.309, 1.84, 2.83 |

A verify round costs about 1.8 decode steps at width 5 (short prose: 5644 ms of verify over 103 rounds, 54.8 ms each, against 29.8 ms per plain token), so the wider blocks pay off only where acceptance is high. Width 5 is the default when `--draft-block-size` is not given (`raon_dflash2::DEFAULT_VERIFY_WIDTH`): it ties width 3 on the mean of the three prompts (1.58x each) and leads on code, and it is also the vendor runtime's default draft length. Widths above 8 are refused (the drafter's trained block).

The vendor runtime (`raon_mlx_model.py generate --draft-tokens 4 --draft models/raon-1-mini-dflash2`, the same bf16 drafter quantized to NVFP4 at load, mlx 0.32.4) accepted 13 of 55 proposed tokens on the short prose prompt (0.236; its confidence cut shortens drafts and it falls back to plain decode while acceptance stays low), so the low acceptance on prose is the drafter's, not this port's.

## Throughput, row-invariant kernel (`MLXCEL_RAON_ROW_INVARIANT=1`)

Off (the default), a verify block forms every dense product one row at a time, which reads each weight once per row; the output head alone is 1.3 GB of bf16. On, every Raon forward (decode included) forms them with the row-invariant matvec kernel, which reads the weights once for up to 16 rows. Code prompt, one round:

| Arm | tok/s | Acceptance rate, emitted per verify |
|---|---|---|
| Plain | 33.63 | |
| Width 3 | 59.91 (1.79x) | 0.746, 2.48 |
| Width 5 | 70.13 (2.09x) | 0.589, 3.31 |
| Width 7 | 72.00 (2.14x) | 0.471, 3.75 |

Plain decode runs at the same speed with the kernel on (33.63 against 33.48 tok/s), but its greedy text differs from the default kernels' (a different reduction order in every dense product), so the kernel stays opt-in: turning it on by default would change Raon-1 mini's classic output for every user, which needs its own reference comparison.

## Final state notes

Prefill captures the four mHC streams only for the newest 2048 prompt rows, the drafter's context window; the drafter is told how many earlier rows were left out so positions stay absolute. Older rows never reach a window of that size, so greedy equality and acceptance do not depend on it, and a long prompt no longer holds its full `[n, 51200]` feature history. Sampled drafting serves temperature, top-k / top-p / min-p, penalties and token bias; Mirostat and adaptive-p keep state from one token to the next that independently sampled verify rows cannot carry, so `mlxcel generate` rejects them with a named error and the server decodes those requests classically.

## Reproduce

```bash
mlxcel generate -m models/raon-1-mini -p "<prompt>" -n 256 --temp 0
mlxcel generate -m models/raon-1-mini -p "<prompt>" -n 256 --temp 0 \
  --draft-model models/raon-1-mini-dflash2 --draft-kind dflash --draft-block-size 5
```
