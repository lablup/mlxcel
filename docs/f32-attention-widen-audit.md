# The f32 attention widen audit

`lablup/mlxcel#1710` ported upstream's f32 attention guard into `src/models/phi.rs` and `src/models/stablelm.rs` after `phi-2-4bit` went NaN at layer 29 of 32 in f16 and every later token decoded as `!`. The defect had survived because `lablup/mlxcel#1709`'s `gelu_approx` was widening the residual stream by accident and holding the scores in range. That issue asked whether the other families upstream guards this way carry the guard in their mlxcel ports, and closed without an answer. This file is that answer.

The useful half is the negative one. Of the eight upstream files the search returns, three are already matched, two are a different guard with a different failure mode, one is not this guard at all, one has no mlxcel port, and one needed a change. A family that legitimately needs no widen is a result.

## What the search looks for

GitHub code search, or `git grep` in a local clone, for the literals `queries.astype(mx.float32)` and `keys.astype(mx.float32)`, scoped to `ml-explore/mlx-lm` and `Blaizzy/mlx-vlm`.

```bash
git -C mlx-lm  grep -nF -e "queries.astype(mx.float32)" -e "keys.astype(mx.float32)" origin/main -- '*.py'
git -C mlx-vlm grep -nF -e "queries.astype(mx.float32)" -e "keys.astype(mx.float32)" origin/main -- '*.py'
```

**Re-run 2026-09-14**, against `ml-explore/mlx-lm` at `dcbcf78` and `Blaizzy/mlx-vlm` at `45d6e125`, both dated 2026-09-12. Upstream moves, so a later reader should re-run rather than trust the table.

The literals are a starting point and not the verdict. Three distinct things match them.

**The overflow guard**, which is what #1710 was about: f16's largest finite value is 65504, the `q @ k^T` products can pass it, and the softmax turns the resulting inf into NaN. Widening the queries alone suffices, because MLX promotes the score matmul to match its wider operand.

**A block-selection guard**, which shares no failure mode with it: the scores drive a discrete top-k choice over KV blocks, and rounding flips the ones near the cut-off. Getting this wrong produces no NaN and no visible breakage, just different blocks selected. `minimax_m3_vl` says so by construction and `qwen4_exp` says so in a comment.

**Neither**, where the literal appears in a scale multiply or a mean reduction that casts straight back.

## The verdicts

| Upstream site | Upstream widens | Class | mlxcel | Verdict |
|---|---|---|---|---|
| [mlx-lm `phi.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/phi.py), [mlx-vlm `phi/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi/language.py) | queries | overflow | `src/models/phi.rs:199` | matches, no change |
| [mlx-lm `stablelm.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/stablelm.py), [mlx-vlm `stablelm/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/stablelm/language.py) | queries and keys | overflow | `src/models/stablelm.rs:197-198` | matches, no change |
| [mlx-lm `phixtral.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/phixtral.py), [mlx-vlm `phixtral/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phixtral/language.py) | queries | overflow | `src/models/phixtral.rs:507` | over-widened keys and values; **narrowed to match** |
| [mlx-vlm `paligemma/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/paligemma/language.py) | queries and keys, only when `model_type != "gemma"` | overflow, but see below | none; text attention is `src/models/gemma.rs:181` and `src/models/gemma2.rs:194` | no widen needed, measured |
| [mlx-vlm `minimax_m3_vl/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minimax_m3_vl/language.py) (four copies) | indexer queries and keys | block selection | `src/models/minimax_m3_indexer.rs:219`, unwidened | unverified, checkpoint absent |
| [mlx-vlm `qwen4_exp/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen4_exp/language.py) | pooled keys and query | block selection | no port | nothing to do |
| [mlx-vlm `nemotron_labs_diffusion/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_labs_diffusion/language.py) | queries and keys | overflow | no port | nothing to do |
| [mlx-vlm `muse_glimmer/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/muse_glimmer/language.py) | not this guard | neither | not applicable | nothing to do |

Every checkpoint named below ships f16, so the 65504 ceiling is the live question for each. None of them is bf16, which carries float32's exponent range and would make a widen a precision argument rather than a range one.

## The one that changed: phixtral

`src/models/phixtral.rs` widened the queries, the keys **and** the values. Both upstreams widen the queries alone and pass keys and values at their own dtype. That is the same over-widen `3576d734` removed from `phi.rs` and `stablelm.rs`; this file was left behind.

The narrowing keeps the queries cast and drops the other two. Measured on M1 Ultra with `phixtral-4x2_8-4bit`:

- **Generation is byte-identical.** `MLXCEL_PRINT_TOKEN_IDS=1 ... -p "Paris is" -n 32 --temp 0` produces the same 32 ids on both binaries.
- **Decode throughput does not move.** `-n 128` from an 11-token prompt: before 95.06 tok/s median, after 94.94, a ratio of 0.9987 across a 1.6% spread. At a 1061-token prompt with `-n 256`, where the doubled V read should cost most: before 57.22, after 57.19, a ratio of 0.9995 across a 1.1% spread.

The second result contradicts the expectation #1829 was written with, which was that removing two f32 copies of the KV cache per decode step would recover throughput. It does not, at either context length, on this checkpoint. The narrowing lands on upstream parity and on not paying for work whose output is provably identical, not on a speed number. For scale, the guard that #1710 *added* cost `phi-2-4bit` 164.10 tok/s down to 128.76; this one is two orders of magnitude smaller than that.

## Why paligemma and Gemma 2 need nothing

Upstream's paligemma attention has two branches. When `text_config.model_type == "gemma"` it calls `scaled_dot_product_attention` with no widen, which is PaliGemma 1 and matches `src/models/gemma.rs`. The `else` branch, taken by PaliGemma 2 (`model_type == "gemma2"`), is a full manual eager attention: scale the queries, compute f32 scores, apply the `attn_logit_softcapping` tanh, softmax in f32, cast back. Its comment states the motivation, and it is not overflow:

> Match HF eager semantics: compute logits in fp32, softmax in fp32, then cast attention weights back to query dtype before value matmul.

So the question for mlxcel is only whether the scores can leave f16 range on the affected checkpoints. Measured directly, by computing `q @ k^T` in f32 at every layer of a real decode and taking the maximum absolute value. f32 on purpose: an f16 measurement would saturate and hide the answer it was asked for.

| Checkpoint | Layer samples | max &#124;q @ k^T&#124; | Headroom under 65504 |
|---|---|---|---|
| `paligemma2-3b-ft-docci-448-6bit`, with an image | 624 | 659.9 | 99x |
| `gemma-2-2b-it-4bit` | 650 | 270.1 | 242x |
| `gemma-2-9b-8bit` | 1050 | 900.8 | 73x |

Each run generated 24 tokens. That matters: the first attempt ran PaliGemma 2 without `--image`, which makes it emit EOS at once, and collected 78 samples from three prefill forwards of a generation that produced nothing. A measurement taken from a degenerate run is not a measurement.

For comparison, #1710 measured stablelm's worst product at 5534.5 on a 16-token prompt, an 11.8x headroom, and judged that safe. Gemma 2's 73x is the tightest here and is six times safer than that. A widen would also land on every plain Gemma 2 text checkpoint sharing `src/models/gemma2.rs`, so its cost would be paid across the family for a range that is not in reach.

One property of Gemma 2 is worth writing down because it changes the failure mode rather than the verdict. `attn_logit_softcapping` is 50.0, and `tanh(inf)` is 1.0, so a score that did overflow would saturate at the cap rather than become NaN. The output would lose the ordering among the overflowed entries, which is a quieter failure than phi's.

## What was not verified

**MiniMax-M3.** `src/models/minimax_m3_indexer.rs:219` computes its token scores with `matmul(q, k_t)` at the input dtype, where all four upstream copies widen both operands. This is the block-selection class, so a wrong verdict is silent: no NaN, just a different set of KV blocks entering the sparse attention. The path is reached only when `should_apply_sparse(kv_len)` holds, that is `kv_len > 2 * topk_blocks * block_size`, so it needs a long context to exercise at all. No M3 checkpoint is present on this machine, and `docs/supported-models.md` records the family as exceeding it. Recorded as unverified rather than as a pass, because a skip reported as `ok` is how the NaN guard from `lablup/mlxcel#1718` came to cover nothing on either machine (`tests/common/mod.rs`).

**Adjacent and out of scope.** `src/models/deepseek_v4_indexer.rs` scores pooled keys for the same kind of discrete block selection. It was not part of #1829's site list and was not audited here.

## Two entries that will match the search again

Recorded so a later re-run does not re-open them.

`muse_glimmer/language.py` matches `queries.astype(mx.float32)` and is not this guard: it multiplies by `qk_scale_factor` in f32 and casts straight back to `queries.dtype` before attention.

`qwen4_exp/language.py` (Qwen3.8-Flash-Next, later GLM-5 Next) has two matches with different meanings. At line 931 a mean over pooled keys is taken in f32 and cast straight back, which is reduction precision. At line 975 the block-selection scores are computed in f32, which is the same class as `minimax_m3_vl`. The file was added upstream on 2026-08-26 and is absent from #1829's table; mlxcel does not port the family, so there is nothing to do, but the omission is why the table there reads seven files and this one reads eight.
