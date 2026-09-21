# What byte-identical Qwen 3.5 DFlash costs, priced (GB10, 2026-09-21)

Issue #1935, deciding what PR #1944 should be. The residual there is genuine numerics in CUDA's fused `sdpa_vector` path, and #1944 answers it by declining the burst. This record answers the question that has to come first, which side of the comparison that kernel actually serves, and then prices every option the dispatch allows against one set of classic brackets.

## Which side does the fused kernel serve

Both. It is not an asymmetry between a vector path and a block path, and that matters, because a proposal to disable the fused path "only in the verify block" rests on there being one.

`supports_sdpa_vector` gates on the query row count **of the individual call**, not of the block: `query_sequence_length = q.shape(2)` and `supported_vector_config = sdpa_supported_head_dim && query_sequence_length < 4` (`src/lib/mlx-cpp/patches/mlx/backend/cuda/scaled_dot_product_attention.cu`). Classic decode reaches it with `l == 1`, through `Qwen3NextAttention::forward`'s third arm. The verify block reaches it through `attend_per_position`, which slices `queries[:, :, i:i+1, :]` per row precisely so each row computes what a decode step computes, so every one of its calls also has `q.shape(2) == 1`. Neither is chunked first: `materializing_sdpa_query_chunk` returns `None` below `q_len == 2`, and mlxcel's own note on `cuda_sdpa_materializes_scores` records that `supports_sdpa_vector` "only covers `q_len < 4`".

Measured rather than inferred, from the served arms below: `MLXCEL_SDPA_VECTOR_LARGE_D=0` moves **both** sides. Classic's completion changes (`2c76b0a181` to `4c37650547`), which a flag that did not serve classic's decode call could not do. The width-3 and width-4 bursts change too (`3e60b1574c` to `4c37650547`), and the burst's only `q_len < 4` attention calls are those per-row verify ones: its prefill is one batched causal call at `q_len` 158, its first bonus is sampled from prefill logits, and its rollback does no attention. The drafter's own attention moves as well, but a drafter cannot change greedy text, because every emitted token is a target argmax.

So `LARGE_D=0` does not move classic onto the block's kernel. It moves both off the fused kernel and onto the materializing fallback, and byte-identity is restored because the fallback is consistent across the two call sites while the fused kernel is not.

**The consequence for "disable it only in verify": it is not an option.** It would put the verify rows on the fallback and classic decode on the fused kernel, which is the one configuration guaranteed not to be byte-identical, since byte-identity here is exactly the two sides agreeing on a kernel. It was not measured because the dispatch predicate answers it and building the knob would need an MLX rebuild to test something already decided.

What does survive, and is not the same idea, is that the flag is a process-global `static` read once. A server process can run entirely on the fallback. That confines the cost to processes that opted into speculation instead of forfeiting the kernel for everyone, and it is priced below as option B.

## Host and method

One session, one binary (`origin/main`, sha `f2ba1688`, verified to carry none of #1944's gate before it ran), seven arms, n = 3 per arm after a discarded warm-up. One `mlxcel-server` per arm, `--ignore-eos --max-batch-size 1`, the #1797 harness's fixed 158-token prompt, 200 tokens per request, `MLX_ENABLE_TF32=1`. Arms differ in exactly the variable in their tag.

Driver: `data/dflash-sdpav-options-gb10-2026-09-21/harness/price_options.py`, which reuses the #1797 harness's request path and the #1820 host gate (sustained-quiet CPU predicate matching on `/proc/<pid>/comm` with stopped processes dropped, foreign-model check, memory floor, cumulative `NV_ERR_NO_MEMORY` trip wire). The driver `NV_ERR_NO_MEMORY` delta was 0 on every arm.

The brackets are the shipped configuration: no drafter, the fused path left on.

## Result

Brackets: opening 56.55 to 57.24, closing 57.19 to 57.84 tok/s. **They overlap**, so this session did not drift. The union spread, 1.29 tok/s or 2.3%, is the resolution floor, and any difference smaller than it is reported below as unresolved rather than as a result.

| arm | mean tok/s | min to max | vs classic | completion sha256 | acceptance |
|---|---:|---:|---:|---|---:|
| classic-open (fused) | 56.84 | 56.55 to 57.24 | 0.99x | `2c76b0a181` | |
| classic-novec (`LARGE_D=0`) | 57.57 | 57.37 to 57.94 | 1.01x | `4c37650547` | |
| w3, `LARGE_D=0` | 64.39 | 64.09 to 64.79 | 1.13x | `4c37650547` | 0.526 |
| w4, `LARGE_D=0` | 62.86 | 62.68 to 63.08 | 1.10x | `4c37650547` | 0.464 |
| w3, fused | 69.59 | 69.19 to 69.81 | 1.22x | `3e60b1574c` | 0.599 |
| w4, fused | 67.55 | 67.07 to 67.97 | 1.18x | `3e60b1574c` | 0.511 |
| classic-close (fused) | 57.55 | 57.19 to 57.84 | 1.01x | `2c76b0a181` | |

Resolved against the 1.29 floor: the burst beats classic in every configuration (w3 on the fallback clears the closing bracket by 6.25 tok/s), and the fused kernel is worth 4.39 tok/s to the width-3 burst. Unresolved: width 3 against width 4 in either configuration (gaps 1.21 and 1.02), and the fused kernel's worth to classic decode (0.13). The #1945 session resolved width 3 above width 4 against a tighter 0.40 floor; this session does not contradict it, it simply cannot see it.

Two facts the table carries that are easy to miss. **The fallback costs the burst twice**: slower per-row attention and lower acceptance (0.599 to 0.526 at width 3, 0.511 to 0.464 at width 4), because different numerics change how often the drafter's proposal survives the target's argmax. And **`LARGE_D=0` changes what classic decode produces** (`4c37650547`, not `2c76b0a181`), so the fallback is not a free correctness switch: it changes every request's output on that process, speculative or not.

## The options, priced

**A. Decline the burst, which is PR #1944 as written.** 1.00x, output identical to today's shipped classic (`2c76b0a181`), and nothing else in the process changes. Blast radius is the narrowest available: the guard lives in `Qwen35Model::probe_block_chain_exactness`, so it touches the Qwen 3.5 family's DFlash gating and no other family's probe, and no non-speculative request anywhere. The cost is the burst's 1.18x to 1.22x, forfeited unless an operator acts.

**B. Run the process on the fallback (`MLXCEL_SDPA_VECTOR_LARGE_D=0`).** 1.13x at width 3, 1.10x at width 4, and byte-identical to the drafter-less server **in the same configuration**. It satisfies the contract that speculative text equals what the same server would produce without the drafter. What it does not do is preserve today's answers: the process serves `4c37650547` where it serves `2c76b0a181` today, for every request. The flag's cost to classic decode is unresolved at this prompt length, and that is not a general result: issue #675 records that the fallback's cost grows with key length while the fused kernel's does not, so a long-context server would pay more than this 158-token prompt shows. Blast radius: the flag is process-global and covers `head_dim` 256 and 288, so it reaches the gemma family, qwen3.5, qwen3.6, baichuan-m1 and paligemma2; scoping it to speculation-configured processes limits who pays but still changes every model in such a process.

**C. Disable the fused path only for the verify block.** Not an option, for the dispatch reason above.

**D. `MLXCEL_MTP_ALLOW_INEXACT=1`.** 1.22x at width 3, 1.18x at width 4, and not byte-identical (`3e60b1574c`). Per-process opt-in; nothing changes for anyone who does not set it. This is the status quo escape and it is what an operator who wants throughput and does not need byte-identity should use.

## What the numbers support

Byte-identical speculation is achievable and is worth 1.13x; dropping the byte-identity requirement is worth 1.22x; and the fused kernel is worth about 8% to the burst while being worth nothing measurable to classic decode at this context length.

A is the better default anyway, because it is the only option that leaves today's output and today's non-speculative performance exactly as they are, and because B's price is paid by every request on the process rather than by the ones that asked for speculation. B is worth documenting as the operator recipe for byte-identical speculation rather than shipping as a default, and it needs no new code: #1944's guard stands down when the flag is off, so an operator who sets `MLXCEL_SDPA_VECTOR_LARGE_D=0` gets the burst engaged and byte-identical on the same binary.
