# The f32 promotion audit

`lablup/mlxcel#1711` found that `mistral4.rs` built its Llama-4 attention scale from a host `&[f32]`, so the multiply promoted `q` to f32, and that dtype rode the residual stream through every later layer, forcing each matmul to promote its own weight. The checkpoint measured 37% of mlx-vlm before the cast and 100% after, which is 5.19x on M5 Max.

That issue's description asked for a static audit of every other site where an f32 constant is constructed and combined with an activation. This file records the result, because the useful part is the negative one: of 50 candidate sites, one was a defect. `scripts/audit_f32_promotion.py` runs the same two searches, and reports 49 in broad mode now that the one defect is fixed.

## What the search looks for

An f32 array is built by `from_slice_f32`, `full_f32`, `zeros_f32`, `ones_f32` or `arange_f32` without being told a dtype, and then reaches `multiply`, `add`, `subtract`, `divide`, `maximum` or `minimum` with another operand. MLX promotes on the wider operand, so if that other operand is an activation, the result leaves the line as f32.

Three refinements matter, and each was learned by getting it wrong first. A constructor that takes a dtype captured earlier (`full_f32(&[1], alpha, input_dtype)`) is not a source, and missing that put 80 sites on the first list instead of 50. A block that computes in f32 on purpose and casts at the end is not a defect either, and the cast is usually applied to a derived variable several lines later, so a search that only looks for a cast of the constructed name reports the whole block. And a constructor immediately re-bound through a cast is fixed, not broken: without that check the narrow search reported the very site it had just caused to be repaired.

## Why 49 of the 50 are not defects

**The promotion is cast away before it reaches the residual stream.** The five MoE routers that multiply `topk_scores` by an f32 `routed_scaling_factor` (`glm4_moe`, `solar_open`, `deepseek`, `glm4_moe_lite`, `dots1`) all pass their scores to `switch_layers::moe_weighted_sum`, which casts to `array_dtype(expert_out)` before the multiply and to the requested output dtype after the sum. The f32 arithmetic is real but confined to a `[n_tokens, k]` tensor. `ernie4_5_moe_vl` reaches the same function.

**The block computes in f32 deliberately and casts at the end.** `llama4.rs` captures `q_dtype` before building the attention scale and casts before multiplying into `q`, which is what `#1711` meant when it said llama4 was the original of the pattern and needed no change. `gemma3n.rs`'s sparse activation ends in `astype(..., array_dtype(&gate))`. `deepseek_v4_hyper`'s Sinkhorn gate ends in `astype(&collapsed, x_dtype)`. `bailing_moe_linear`'s chunk decay casts inside the same expression.

**The quantity is f32 by design.** `deepseek_v4_indexer` documents its inputs as f32 and returns int32 indices. Rope angle tables in `glm4v` and `hunyuan_vl` cast the positions to f32 first, on purpose. `kimi_linear` passes `FLOAT16` explicitly, and the search flagged it only because the literal is not `array_dtype`.

**The site runs once per image or per request rather than per token.** Fourteen of the 50 are in `src/vision/encoders/`, eight elsewhere under `src/vision/`, six in `src/audio/`, and one in `lora/loader.rs` at load time. A promotion there costs prefill, not decode, and none of them was found to escape either.

## The one that was real

`solar_open.rs`, in `gptq_to_mlx_tensors`. Step 4 casts the scales to f16 with a comment saying it does so because `quantized_matmul` and `gather_qmm` expect that dtype. Step 5 then computes the biases, and the two branches disagreed: the `qzeros` branch passed `scales_dtype` down to `compute_mlx_biases_from_qzeros`, while the symmetric branch built its zero point from a host `f32` and let MLX promote, returning f32 biases beside f16 scales. A GPTQ checkpoint without `qzeros` loaded a mismatched pair into every quantized matmul.

The function stated its own intent two lines above the line that broke it, which is the shape worth remembering. The search that finds this one is narrower than the general audit: a site inside a function that names an intended dtype and does not use it at that call.

`gptq_biases_match_the_scales_dtype_in_both_branches` pins it, and asserts the two dtypes are equal rather than naming f16 twice, because the pairing is the invariant. Reverting the cast fails the test with `left: 10` against `right: 9`.

## Re-running it

```bash
python3 scripts/audit_f32_promotion.py --mode narrow   # 19 sites
python3 scripts/audit_f32_promotion.py --mode broad    # 49 sites
```

The audit is a search, not a test, and it is not wired into CI. Nothing here forbids a new f32 constructor; the point is that the broad form has a high false-positive rate and the narrow form is where the one defect came from. The script was checked the same way the fix was: it reports nothing for the repaired code and names `solar_open.rs:196` again the moment the cast is removed.
