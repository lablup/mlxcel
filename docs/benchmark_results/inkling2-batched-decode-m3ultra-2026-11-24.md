# Inkling 2 Mini native batched decode on M3 Ultra (#3377)

Server throughput, batch parity and B = 1 identity of Inkling 2 Mini with the native batched forward against `451ec68f` (origin/main), where every row ran its own single-sequence forward and padded prefill was declined. Arm A is `451ec68f`; arm B is `451ec68f` plus this PR.

## Environment

| Item | Value |
|---|---|
| Host | Apple M3 Ultra, 512 GB, macOS 27.2, shared with other worktrees |
| Build | `cargo build --release --features metal,accelerate --bin mlxcel --bin mlxcel-server --example logit_trace`, one build per arm |
| Checkpoints | `models/inkling-2-mini` (`thinkingmachines/Inkling-2-Mini` b85c3cfa) and `models/inkling-2-mini-4bit` (`mlx-community/Inkling-2-Mini-4bit` 9c0e2960) |
| Lock | every GPU run under `/usr/bin/lockf -k /Users/devops/Development/.mlxcel-gpu.lock` |

## Server throughput

`mlxcel-server -m models/<name> --parallel 8` (defaults otherwise, so up to 4 cold prompts prefill as one cohort in arm B), then `python3 scripts/bench_serving_concurrency.py --concurrency 1,4,8 --prompt-tokens 512 --max-tokens 128` (501 prompt tokens). Order per checkpoint: A, B, A, B, with a 20 s pause before each server start. Aggregate tok/s, rounds 1 / 2:

| Checkpoint | Concurrency | A aggregate | B aggregate | Change | A TTFT ms | B TTFT ms |
|---|---|---|---|---|---|---|
| inkling-2-mini | 1 | 26.3 / 26.6 | 26.4 / 26.5 | +0.0% | 543 / 538 | 551 / 537 |
| inkling-2-mini | 4 | 26.5 / 26.6 | 48.6 / 48.5 | +83% | 2080 / 2071 | 1833 / 1827 |
| inkling-2-mini | 8 | 25.9 / 26.1 | 62.7 / 62.4 | +141% | 4188 / 4180 | 3556 / 3592 |
| inkling-2-mini-4bit | 1 | 30.6 / 30.7 | 30.7 / 30.6 | +0.0% | 498 / 497 | 497 / 504 |
| inkling-2-mini-4bit | 4 | 30.9 / 30.8 | 50.5 / 50.6 | +64% | 1938 / 1916 | 1773 / 1774 |
| inkling-2-mini-4bit | 8 | 30.3 / 30.3 | 62.7 / 63.0 | +107% | 3880 / 3869 | 3529 / 3503 |

Every request succeeded. Arm A's aggregate stays flat with concurrency because each row ran its own forward; arm B shares one forward per tick. The lower time to first token at concurrency 4 and 8 comes from the batched prefill cohort.

## Batch parity (teacher-forced)

`MLXCEL_TRACE_BATCH_ROWS=N ./logit_trace models/<name> README.md 64 4 8 512` (4 chunks of 64 positions, each a width-one `forward_batched_with_ids` step over N rows with distinct contexts; only row 0 is emitted), compared with `scripts/compare_logit_traces.py`. "Decided" means a reference top-two gap of at least 2.

| Checkpoint | Pair | Top-1 disagreement | Decided disagreement | Largest gap at a disagreement | Perplexity |
|---|---|---|---|---|---|
| inkling-2-mini | A B=1 vs B B=1 | 0/256 (byte-identical trace) | 0/98 | none | +0.000% |
| inkling-2-mini | B B=1 vs B B=4 | 2/256 | 0/98 | 0.051 | +0.159% |
| inkling-2-mini | B B=1 vs B B=8 | 3/256 | 0/98 | 0.110 | +0.024% |
| inkling-2-mini-4bit | A B=1 vs B B=1 | 0/256 (byte-identical trace) | 0/98 | none | +0.000% |
| inkling-2-mini-4bit | B B=1 vs B B=4 | 7/256 | 0/98 | 0.189 | -0.519% |
| inkling-2-mini-4bit | B B=1 vs B B=8 | 7/256 | 0/98 | 0.189 | -0.519% |

Every disagreement lands at rank 2 of the other arm, at a position the reference was undecided about.

## B = 1 greedy output

`mlxcel generate -m models/<name> -p "Explain why the sky is blue." -n 128 --temp 0 --show-reasoning`, both arms: the text between `Generating...` and the stats line has the same md5 (`fd9dbb8d` native, `3b8906ff` 4-bit). #3369 recorded `32e7b82d` for this command over an extraction it does not specify, which this record could not reproduce; the comparison here uses one extraction for both arms, and the B = 1 traces above are byte-identical. Decode: native 36.41 (A) / 35.25 (B) tok/s, 4-bit 35.59 / 35.70 tok/s, single runs on a shared host.

## Concurrent requests against solo requests (free-running)

Four chat prompts (temperature 0, 64 tokens), first solo, then 4 and 8 at once (8 repeats the four), against each arm's server. Arm A matches solo 4/4 and 8/8 on both checkpoints, since every row ran alone. Arm B matches 3/4 and 3/8 (native) and 2/4 and 5/8 (4-bit); the differing rows agree for the first 47 to 238 characters and then take another near-tied word. With `--max-batch-prefill 1` (batched decode only, no prefill cohort) arm B on the native checkpoint matches 3/4 and 4/8 with the same split points (127 and 219 characters), so the split comes from batched decode arithmetic (matmuls over B rows), the same class the traces show, not from the padded prefill.

## Raw logs

Server logs, bench output, traces and the greedy outputs were kept in the session scratchpad of the run; the numbers above are copied from them.
