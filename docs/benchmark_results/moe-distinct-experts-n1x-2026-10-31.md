# MoE expert overlap in batched decode on RTX Spark N1X

Issue #2784. Each (token, expert) pair in Gemma 4's MoE reads one expert matrix, so a batch of B tokens reads `top_k * B` matrices per gather. A kernel that groups rows by expert could read each selected matrix once. This records how much that could save, and the conclusion is not to build it: about 13% aggregate at 8 clients at best, and about 5% at the default 4 slots.

## Method

RTX Spark N1X (sm_121), Windows 11 arm64, CUDA 13.4, gemma-4-26b-a4b-it-4bit (30 layers, E=128, top_k=8). `mlxcel-server --max-batch-size 8 --ignore-eos`, chat requests started together at temperature 0 with 120 output tokens. Prompts were 8 different ones (code, history, a recipe, engineering, physics, a story, networking, policy). `MLXCEL_PROFILE_MOE_INNER=1` prints `[MOE] name=distinct_experts tokens=T pairs=P distinct=D` per gather stage and forces an eval per operation, so absolute times are inflated and only ratios are used.

## Distinct experts per layer

| Clients | Prompts | Pairs | Distinct, median (min-max) | Mean read reduction |
|---|---|---:|---:|---:|
| 8 | different | 64 | 45 (15-60) | 30.4% |
| 4 | different | 32 | 28 (12-32) | 12.5% |
| 8 | identical | 64 | 8 (8-8) | 87.5% |

Identical prompts on greedy decode make every row pick the same experts, so that row is a harness artifact. `scripts/bench_serving_concurrency.py` sends one prompt to every client, which is why this used different ones.

## Time saved by reading each expert once

A temporary patch timed `gate_proj` over only the first row of each distinct expert, before the real call, then removed. The medians over 3 runs at 8 clients, excluding the first 150 gathers:

| Operation | ms |
|---|---:|
| `up_proj`, 64 pairs | 2.51-2.53 |
| `gate_proj`, distinct experts only | 1.93-1.97 |
| `down_proj`, 64 pairs | 2.08-2.21 |
| Everything before the gather, attention included | 4.69-4.85 |
| Whole layer | about 13 |

Reading 30% fewer matrices saves about 22% of a gather. The gathers are about 54% of a layer, so a perfect grouped kernel saves about 12% of a step, or about 13% aggregate at 8 clients. A real kernel still multiplies all 64 rows, which this upper bound omits. At 4 clients the read reduction is 12.5%, which by the same ratio gives about 5%, and 4 clients do not take this path today (fewer than 64 pairs use `compiled_switch_qgeglu`).

## Decision

Not built. The kernel would need a variable row count per expert in `GatherQMM`, for the MoE decode path alone, and the bound is at the size of this model's run-to-run spread (about 6% with identical kernels at 4 clients). Everything before the gather, attention included, is about 36% of a layer, and attention is split by row, so that is where more of the 8-client step goes without a new kernel.

Not measured: other MoE models. gpt-oss 20B has E=32, so it can overlap more, but its mxfp4 gather takes a different kernel. GB10 was not measured.
