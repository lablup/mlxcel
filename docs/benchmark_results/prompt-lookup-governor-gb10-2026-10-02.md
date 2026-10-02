# Prompt-lookup drafting policy on GB10, 2026-10-02

Issue #2091, PR #2092. Prompt lookup (`mlxcel generate --prompt-lookup`, PR #2074) slowed prose 3 to 21% on GB10 while speeding up replies that copy their prompt. This record measures what a verify block costs per width on GB10, compares PR #2074's governor (`--prompt-lookup-policy graded`) with the CUDA policy this change adds and makes the CUDA default (`gated`), and checks greedy parity.

Raw data under [`data/prompt-lookup-governor-gb10-2026-10-02/`](data/prompt-lookup-governor-gb10-2026-10-02/):

- `harness/ab.py` and `harness/prompts/`: the interleaved A/B driver and the five prompts;
- `results.json`, `run2-final.jsonl`, `run2-load.log`: the final matrix below, one line per case with every repetition's rate and the arms' `[Prompt lookup]` counters, and the load average sampled every 30 s during it;
- `run1-probation-on-any-token.jsonl`, `run3-full-at-1.75.jsonl`: the two other full matrices, on variants that were not kept (see "How the policy was settled");
- `verify_width_cost.txt`: `examples/verify_width_cost` on all four models.

## Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DGX Spark, sm_121), 20 CPU cores, 121 GiB unified memory |
| **OS / driver** | Linux 7.0.0-1019-nvidia, driver 580.178.04, CUDA 13.0 (V13.0.88) |
| **mlxcel** | `34c627d7` on `update/issue-2091-prompt-lookup-cuda-governor`, `cargo build --release --features cuda --bin mlxcel` |
| **MLX pin** | `81ba1c6a` |
| **Toolchain** | Rust 1.97.1 |
| **Checkpoints** | `models/mlx/qwen3-1.7b-4bit`, `qwen3-4b-4bit`, `qwen3-8b-4bit`, `meta-llama-3.1-8b-instruct-4bit` (affine 4-bit) |

The host is shared: other sessions' builds ran intermittently. The GPU was held under `gpu-lock` for every run and no CI job was running. Load average over the final matrix: median 3.1, range 1.5 to 4.5 (38 samples). The interleaving below is what makes that tolerable: every repetition runs all three arms back to back, so drift lands on every arm alike.

## Verify width cost

`examples/verify_width_cost` prefills a fixed 300-token prompt, then times a pipelined one-token step (submitted from the still-lazy argmax before the host reads it, as `CxxGenerator` does) and a synchronous forward of `w` tokens plus the argmax and its host read, trimming `w - 1` positions per round so the cache grows like a rejected verify. Median of 40 rounds per width, in pipelined steps:

| Width | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 1.23 | 1.37 | 1.76 | 2.43 | 3.04 | 3.47 | 3.84 | 3.91 | 4.02 |
| Qwen3-4B | 1.11 | 1.25 | 1.61 | 2.18 | 2.96 | 3.35 | 3.81 | 3.95 | 4.00 |
| Qwen3-8B | 1.07 | 1.15 | 1.59 | 2.10 | 2.86 | 3.37 | 3.85 | 3.30 | 3.28 |
| Llama-3.1-8B | 1.08 | 1.16 | 1.61 | 2.08 | 2.80 | 3.24 | 3.69 | 2.95 | 2.97 |

Width 1 is the synchronous step itself: 7 to 23% over a pipelined one, more on the smaller models, where the host's share of a step is larger. Below 8 rows the affine path runs `qmv`'s multirow kernel, instantiated at 2, 4 and 8 accumulator rows (`dispatch_multirow_width`), so 5 to 7 rows pay for the 8-row instantiation; from 8 rows `quantized.cpp` switches to `qmm_sm80`, which on the 8B models is cheaper than the 7-row multirow launch. PR #2074's default block is 7 proposals, verify width 8.

A drafted round in the PR #2074 loop also drains the pipeline: a proposal found while a plain step is in flight waits for that step to be read, the verify runs synchronously, and two synchronous plain rounds follow before pipelining resumes. On prose, where proposals rarely land, every drafted round is close to a pure loss of its verify cost plus those syncs.

## The two policies

`graded` (PR #2074, unchanged, still the default on Metal and ROCm): the block is twice the recent accepted average plus two, capped at 7; after three drafted rounds in a row land nothing it pauses 4 rounds, doubling to 32, and then probes again with a drafted round.

`gated` (new, the default on CUDA builds):

- the block is narrow (2 proposals, verify width 3) or full (7); it switches to full once a narrow block lands whole, and back when the accepted average falls;
- a reply starts narrow and on probation, and probation ends only when a round lands a whole narrow block; a round that lands nothing while on probation pauses;
- a pause has no timer: while paused, every round still looks the proposal up and checks it against the tokens decoding emits next, and drafting resumes only after two proposed tokens in a row came true (a "shadow confirmation"), on probation again; paused rounds pipeline at once.

## Final matrix

Greedy (`--temp 0`), `-n 400` (story `-n 800`), whole-call tok/s from the CLI's `[Generated ...]` line (prefill included, after `mlxcel generate`'s own warmup). Three repetitions per arm, interleaved; the cell is the median arm rate over the median plain rate of the same case.

Plain decoding, tok/s:

| Model | edit_fn | edit_json | summary | write | story |
|---|---|---|---|---|---|
| Qwen3-1.7B | 179.6 | 178.3 | 167.0 | 181.9 | 187.2 |
| Qwen3-4B | 83.2 | 82.8 | 82.2 | 85.9 | 85.4 |
| Qwen3-8B | 50.7 | 50.0 | 49.6 | 50.7 | 52.2 |
| Llama-3.1-8B | 51.1 | 51.3 | 49.8 | 50.9 | 51.6 |

Prompt lookup over plain:

| Model | policy | edit_fn | edit_json | summary | write | story |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | graded | 1.16x | 1.39x | 0.82x | 1.07x | 0.83x |
| Qwen3-1.7B | gated | 1.22x | 1.41x | 0.99x | 1.02x | 0.98x |
| Qwen3-4B | graded | 1.27x | 1.71x | 1.38x | 0.94x | 0.87x |
| Qwen3-4B | gated | 1.39x | 1.64x | 1.42x | 0.98x | 0.98x |
| Qwen3-8B | graded | 1.41x | 1.90x | 1.59x | 0.98x | 0.91x |
| Qwen3-8B | gated | 1.51x | 1.91x | 1.62x | 0.97x | 0.98x |
| Llama-3.1-8B | graded | 1.54x | 2.02x | 0.94x | 0.96x | 0.93x |
| Llama-3.1-8B | gated | 1.66x | 1.97x | 1.02x | 0.99x | 1.00x |

The `graded` rows reproduce the issue's baseline, measured on PR #2074's merged head, to within 1 to 3 points (1.7B story 0.83x against 0.84x, 8B edit_json 1.90x against 1.81x). Per-repetition spread within an arm was under 2% for most cases; the largest was 6.3%, one repetition of one case.

Against the issue's criteria: every story row reaches 0.98x or more; three of four write rows do; Qwen3-8B's write row stays at 0.97x (0.98x under `graded` in the same run, 0.96x to 0.97x for `gated` across the three matrices). The edit and summary rows keep their gains: the largest drop against `graded` is Qwen3-4B edit_json, 1.71x to 1.64x (4%); most rise, because narrow blocks are cheaper per landed token on GB10 than full ones.

Counters and greedy parity (reply text against plain decoding, first repetition):

| Model | prompt | graded drafted/rounds | gated drafted/rounds | gated shadow confirmations | graded vs plain | gated vs plain |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | edit_fn | 41/63 | 41/63 | 0 | identical | identical |
| Qwen3-1.7B | edit_json | 51/66 | 51/66 | 0 | identical | identical |
| Qwen3-1.7B | summary | 24/99 | 4/115 | 2 | identical | identical |
| Qwen3-1.7B | write | 2/62 | 4/64 | 0 | identical | identical |
| Qwen3-1.7B | story | 30/323 | 18/492 | 10 | diverges at char 536 | diverges at char 1149 |
| Qwen3-4B | edit_fn | 36/57 | 34/55 | 0 | identical | identical |
| Qwen3-4B | edit_json | 45/52 | 46/55 | 0 | diverges at char 1346 | identical |
| Qwen3-4B | summary | 31/44 | 30/43 | 0 | identical | identical |
| Qwen3-4B | write | 3/138 | 3/138 | 0 | identical | identical |
| Qwen3-4B | story | 70/715 | 12/794 | 7 | diverges at char 659 | diverges at char 1474 |
| Qwen3-8B | edit_fn | 36/57 | 34/55 | 0 | identical | identical |
| Qwen3-8B | edit_json | 45/52 | 46/53 | 0 | identical | identical |
| Qwen3-8B | summary | 31/44 | 30/43 | 0 | identical | identical |
| Qwen3-8B | write | 4/123 | 6/127 | 0 | diverges at char 525 | diverges at char 543 |
| Qwen3-8B | story | 54/679 | 13/684 | 7 | diverges at char 475 | diverges at char 1479 |
| Llama-3.1-8B | edit_fn | 35/51 | 34/50 | 0 | identical | identical |
| Llama-3.1-8B | edit_json | 43/45 | 45/47 | 0 | identical | identical |
| Llama-3.1-8B | summary | 23/83 | 29/89 | 0 | identical | identical |
| Llama-3.1-8B | write | 4/129 | 4/115 | 0 | diverges at char 606 | diverges at char 560 |
| Llama-3.1-8B | story | 42/694 | 13/761 | 7 | diverges at char 668 | identical |

Every reply that `graded` keeps byte-identical to plain decoding, `gated` keeps identical too. The divergences are the documented near-tie class (a multi-token verify forward rounds differently from the one-token path); with fewer verify rounds, `gated` reaches them later or not at all. Story rounds outnumber `graded`'s because a paused `gated` loop emits one token per round where a `graded` drafted round sometimes emitted two.

## The remaining row

A round-by-round trace of Qwen3-8B's email (`gated`, debug build) shows where its 3% goes. The reply restates a short fragment of the prompt around tokens 45 to 55; lookup finds it, a narrow block lands whole, the block widens, and two full blocks (width 8, 3.3 pipelined steps on this model) land nothing as the fragment ends. `graded` spends its loss on the same fragment. Requiring two clean narrow blocks before a full one (`GATED_FULL_AT` 1.75, the third matrix) left this row at 0.97x and made the Qwen3-1.7B and Qwen3-4B emails diverge from plain decoding, so it was reverted. What would remove the remaining cost is the round-loop change the issue lists first: verify from the in-flight token instead of draining it, and drop the two synchronous rounds after a drafted round. It changes where verify blocks start, and with it which near-ties flip, so it needs its own parity pass.

## How the policy was settled

Three full matrices, same binary build procedure, interleaved arms:

1. `43dc0755`, probation ended by any landed token (load median 1.2): story 0.98x to 1.00x, Qwen3-8B write 0.96x with 6 drafted rounds, 3 landed tokens and no pause: single landed tokens kept lifting probation. Fixed in `34c627d7`.
2. `34c627d7`, probation ends on a whole narrow block (load median 3.1): the final matrix above.
3. `7224525e`, full block after two whole narrow blocks (load median 4.9): Qwen3-8B write unchanged at 0.97x, two write rows lost parity. Reverted.

## Metal

No Apple Silicon machine was reachable from the measuring host. `graded` stays the default everywhere except CUDA builds and makes exactly PR #2074's decisions (same governor arithmetic, no shadow lookups), so the Apple Silicon numbers in PR #2074 still describe Metal. `--prompt-lookup-policy gated` is available there for anyone who wants to measure it.
