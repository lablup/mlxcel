# Inkling 2 Mini bundled DSpark drafter (M3 Ultra, 2026-11-24)

Greedy decode of `models/inkling-2-mini` (native) with and without the DSpark drafter bundled in its `mtp.safetensors` (issue #3374), next to mlx-vlm 466069f5 running the same pairing (`--draft-model models/inkling-2-mini`). Host: Apple M3 Ultra, 512 GB. All runs are serialized under the shared GPU lock in one ABAB block (A = classic decode, B = drafter on, block size 8), 256 tokens at temperature 0, `--show-reasoning`. Load averages were 2.16 at the start and 8.39 at the end of the block (other worktrees were compiling), so absolute numbers carry a few percent of noise; the A and B runs of each pair are adjacent.

## Exactness gate

The block-versus-chain probe declines on this host at every width tried (8 and 2): verify block position 0 differs from the single-token chain in 464983 of 800232 logit bytes at width 8 (559076 at width 2), also with `qmv_wide` disabled. A multi-row Inkling 2 forward takes different kernels than a one-row step (the grouped expert matvec of #3359 and the matmul tiling), so the drafter-on run is not byte-identical to classic decode. Without `MLXCEL_MTP_ALLOW_INEXACT=1`, `mlxcel generate --draft-model` refuses with a named error and the server falls back to classic decode and reports `exactness_declined` on `/v1/internal/mtp-policy`. Every drafter-on number below was taken with `MLXCEL_MTP_ALLOW_INEXACT=1`.

mlx-vlm has no such gate and shows the same effect: its drafter-on text leaves its own classic text after 213 characters (prompt 1) and 214 characters (prompt 2).

## Throughput and acceptance

Prompt 1: `Explain why the sky is blue.` Prompt 2: `Write a Python function that returns the n-th Fibonacci number, with a short docstring.`

| Prompt | Arm | Run 1 tok/s | Run 2 tok/s | Mean tok/s | Rounds | Acceptance rate | Mean accepted per round | Emitted per verify |
|---|---|---|---|---|---|---|---|---|
| 1 | A classic | 36.61 | 34.47 | 35.54 | | | | |
| 1 | B drafter | 38.45 | 38.65 | 38.55 | 117 | 0.170 | 1.18 | 2.18 |
| 2 | A classic | 36.40 | 35.46 | 35.93 | | | | |
| 2 | B drafter | 55.91 | 53.99 | 54.95 | 82 | 0.307 | 2.12 | 3.11 |

The drafter-on runs are deterministic (both B runs of a prompt emit the same text and the same round counts). Speedup over classic decode: 1.08x on prompt 1 and 1.53x on prompt 2.

mlx-vlm 466069f5, one run each, same block:

| Prompt | Classic tok/s | Drafter tok/s | Accepted drafts per round | Acceptance rate |
|---|---|---|---|---|
| 1 | 33.91 | 42.23 | 1.27 | 18.3% |
| 2 | 33.10 | 65.93 | 2.71 | 39.2% |

## Agreement with mlx-vlm

The drafter itself matches upstream: with `RUST_LOG=mlxcel_core::drafter::dflash=debug`, the proposals of prompt 2's rounds 0, 1 and 2 (34, 4 and 4 context rows) are the same 7 token ids as mlx-vlm's, and so are rounds 4 and 5, which follow the same bonus tokens. The two runs separate at the target: in round 2 both accept all 7 proposals, but the bonus row picks token 149899 here and 11 (a comma) in mlx-vlm, a near tie in the 8-row verify block. From there the texts differ, which is why the acceptance figures above are not directly comparable.

Greedy text agreement (characters of the reasoning text before the first difference):

| Pair | Prompt 1 | Prompt 2 |
|---|---|---|
| mlxcel classic vs mlx-vlm classic | 756 | 214 |
| mlxcel drafter vs mlx-vlm drafter | 213 | 71 |
| mlxcel drafter vs mlxcel classic | 591 | 71 |
| mlx-vlm drafter vs mlx-vlm classic | 213 | 214 |

## Other checks

- Image prompt (`tests/fixtures/test_image.png`, `What is in this image?`, 48 tokens): the drafter run, which prefills from the merged image embeddings, emits the same 48 tokens as classic decode (47.5 vs 29.0 tok/s, one run each).
- `models/inkling-2-mini-4bit` (no `mtp.safetensors`) runs classic decode as before (43.05 tok/s, 64 tokens); `--draft-model` on that directory keeps routing to the full-model drafter.
- Server: with `MLXCEL_MTP_ALLOW_INEXACT=1`, one chat request ran as a DSpark burst (34 rounds, acceptance 0.272, 2.79 emitted per verify) and `/v1/internal/mtp-policy` reported `state: unavailable`, `reason: dflash_dispatch`, `target`/`drafter`: `inkling-2-mini`, `block_size: 8`. Without the override it reported `state: exactness_declined` with the probe sentence, and requests fell back to classic decode.
- The one-time f64 fold of `hidden_out` into the 201024-row LM head takes about 8.8 s on the first bind (the server's warm-up request absorbs it).
