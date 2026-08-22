# Technical Report: PR #1385 - apply llama3 rope_scaling on the shared Llama path

**Date**: 2026-08-23
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust (MLX FFI), Metal (fused launcher gating)
**Risk Level**: High (changes decode numerics for Llama 3.x, Qwen2 and eight VLM families that share one attention implementation)

---

## Executive Summary

Every Llama 3.1, 3.2 and 3.3 checkpoint declares a `rope_scaling` block with `rope_type: "llama3"`. The shared Llama attention parsed the block into a struct and never read a field of it, so these models rotated with the plain `base^(-2i/d)` frequency table. Short prompts hid it, because the scaled and unscaled tables nearly agree at low positions. Past a few thousand tokens the unscaled low-frequency bands rotate up to 8x too fast (32x on Llama 3.2 1B/3B), which is precisely the regime the `llama3` scheme exists to fix.

Three things about this change are worth carrying forward, and none of them are in the issue that requested it.

First, the issue's stated scope was wrong by an order of magnitude. `llama3::ModelArgs` is not Llama's alone; Qwen2 and eight VLM loaders build their text decoder from it. Second, two of the issue's own prescriptions would each have shipped a regression: its suggested `#[serde(alias)]` fix hard-errors on the five local checkpoints that spell both key names, and its acceptance criterion "unsupported types are named load errors" stops InternVL3 from loading at all. Third, the fix as first written introduced a new silent-NaN path of exactly the class the issue exists to eliminate, which two independent reviews found separately.

Along the way the change fixed an unrelated, unreported production defect: `deepseek-coder-1.3b-4bit` was emitting degenerate `):):):):` output on a six-token prompt because its `linear` scaling was being ignored too.

## 1. Problem Statement

### 1.1 Background

`src/models/llama3.rs` declared `pub rope_scaling: Option<RopeScaling>` and referenced it on exactly that one line. Nothing consumed it. Worse, `RopeScaling` was declared with `#[serde(rename = "type")]` on its `rope_type` field, so a config that spells only `rope_type`, which every Llama 3.x config does, left the field `None` even in the parsed struct. The block was doubly dead.

`src/models/mllama/config.rs`'s `to_llama3_args` rebuilt the text args from a fixed key list with no `rope_scaling` entry, so the Llama 3.2 Vision decoder could not have received the block even if the shared path had read it.

### 1.2 Existing Issues

The defect class is the dangerous one: parsed but never applied. There is no error, no warning, and no malformed output. The model produces fluent text at every length; it simply degrades on long contexts in a way that looks like ordinary model weakness rather than a bug. `docs/supported-models.md` had even recorded the gap from the other direction, noting in the TeleChat3 entry that `llama3::ModelArgs` "declares a `rope_scaling` field that nothing reads", and that note had been sitting there.

### 1.3 Risk Assessment

High, and the risk is on the way in. One attention implementation backs Llama 3.x, Qwen2, Pixtral, LLaVA, SmolVLM, Idefics2, Idefics3, InternVL, FastVLM and LocateAnything. Any error in the new table is an error in all of them, and any newly enforced strictness is a load failure in all of them.

## 2. Change Summary

26 files, roughly 2060 insertions. The substantive pieces:

| Area | Change |
|---|---|
| `src/models/rope_utils.rs` (new) | Shared `rope_scaling` reader and `llama3_rope_freqs`, ported from mlx-lm's `initialize_rope` / `Llama3RoPE`. Reads through a `serde_json::Map`, screens every scalar, warns once per checkpoint on an unimplemented scheme |
| `src/models/llama3.rs` | `Attention` carries `rope_scale` and `rope_freqs`; graph and batched paths route through one helper each; three fused launchers gated; `#[serde(skip)] checkpoint_label` for diagnostics |
| `src/lib/mlxcel-core/src/lib.rs` | `fast_rope_batched_with_freqs`, mirroring `fast_rope_batched` |
| `src/models/mllama/config.rs`, `text.rs` | Forwards `rope_scaling`; hoists the resolve above the layer loop |
| `src/models/apertus.rs` | Now calls the shared function instead of its private copy |
| 11 loaders, pipeline executors, TP runtime | Fill `checkpoint_label` |

## 3. Technical Decisions

### 3.1 The issue's suggested serde fix would have broken InternVL3

The issue proposed adding `#[serde(alias = "rope_type")]` so both spellings are read. Serde generates a per-field `Option` and rejects a second write to it regardless of which spelling produced the value, so a config carrying **both** keys becomes a hard `duplicate field` parse error.

Five local checkpoints carry both: `internvl3-1b`, `apertus-8b-instruct-2509-4bit`, `afm-4.5b`, `paddleocr-vl-bfloat16` and `telechat3-36b-thinking-4bit`. The "minimal fix" would therefore have converted a silent no-op into a load failure for all of them.

The reader instead walks a `serde_json::Map`, looking up `type` and then `rope_type`, which is also the order upstream `initialize_rope` uses. The resulting reader is strictly more permissive than the derived struct it replaced: a list-valued `factor` (longrope), a float `original_max_position_embeddings`, a non-string `type` and a doubled key all used to be hard load errors and now parse.

This is the second time a serde alias has produced a duplicate-field trap in this codebase. It is worth treating `#[serde(alias)]` on a field whose two spellings may co-occur as a defect pattern rather than a convenience.

### 3.2 Unimplemented schemes warn and continue rather than failing the load

The issue's acceptance criteria required that `yarn`, `dynamic` and `longrope` become named load errors on this path. Implemented literally, that stops `models/internvl3-1b` from loading: its `text_config` declares `rope_type: "dynamic"` with text architecture `Qwen2ForCausalLM`, and `vlm_internvl.rs` passes the whole `text_config` into these args.

The tradeoff is asymmetric. A model that loads today and is subtly wrong on long contexts is worse than one that is right, but far better than one that does not load. So an unimplemented scheme emits a single warning naming the checkpoint and the scheme, and decodes on the plain table, which is exactly today's behavior.

This is a deliberate, documented departure from a written acceptance criterion rather than an oversight, and the PR body says so. `dynamic` becomes properly implementable once #1324's shared DynamicNTK helper lands.

### 3.3 The first implementation introduced a new silent-NaN path

Two independent reviews converged on this, which is the strongest signal a finding can carry.

The `linear` arm screened `factor` for positivity and finiteness and fell back with a warning. The `llama3` arm, five lines below, did `spec.factor.unwrap_or(1.0)` and fed the result straight into the band arithmetic. Consequences, simulated at Llama 3.1 geometry:

- `"factor": 0` produces 35 of 64 entries exactly `0.0`. MLX's fast rope divides the position by the table entry, `reciprocal(0)` is `inf`, and every logit is NaN for every token. Nothing throws, and `sampling.rs` compares with `partial_cmp(..).unwrap_or(Equal)`, so it does not even panic. It emits garbage.
- An absent factor, or one given as a JSON string, defaults to `1.0` and builds a table bit-identical to the plain one, which is the exact silent no-op this PR exists to remove. The old derived struct at least turned a string factor into a load error.
- `1e39` saturates to `inf` through the f64 to f32 cast; `-8` reverses the rotation direction on the low band. Both finite-looking, fluent, and wrong.

Every scalar is now screened, and a screened-out block warns once and decodes on the plain table.

**One guard was deliberately not added.** `low_freq_factor == high_freq_factor` looks like a division by zero in the smooth-band denominator `(hf - lf)`, and screening it out is the obvious defensive move. It would have broken Llama 4 Scout, which ships exactly that. The interpolation is unreachable when the two are equal: the medium band is `wavelen > L/hf && wavelen < L/lf`, and equal factors make those bounds identical, so the conjunction is unsatisfiable. Upstream reaches the same table through an `mx.where` that discards the unselected branch. The check was resolved by scanning every local `config.json` before deciding, not by reasoning about it in the abstract.

### 3.4 A `pub` function whose safety invariant lived nowhere

`Attention::from_weights_with_rope` is public and takes `args` plus a separately resolved table, with nothing tying them to the same model. MLX requires the table to be `[dims / 2]` and throws from C++ otherwise, and `fast_rope_with_freqs` is bridged as a bare `UniquePtr`, not a `Result`. A mismatch therefore arrives as an uncatchable `std::terminate` at the first generated token rather than as a load error.

All four in-tree callers pass a table resolved from the same args, so this was never live. It is now a load error anyway, because "no current caller gets this wrong" is not a property a public signature should rely on, and this codebase has already been bitten by the MLX-precondition-terminates-at-first-inference class.

### 3.5 A third fused launcher the issue did not name

The issue named two opt-in fused launchers to gate. There is a third, `forward_fused_rope_append` from #905, which derives frequencies from `rope_base` in Metal so a table routes around it, but which takes the position scale as a real kernel parameter (`theta = rope_params[1] * pos * inv_freq`) and was being handed a hardcoded `1.0`. It now receives the real `rope_scale`, so `linear` works through the fused path instead of being silently ignored there.

Its environment variable also had to join the bypass notice, and the detection had to go through `fused_rope_append_enabled()` rather than a presence check, because that variable is tri-state and `=0` must not report a lost kernel.

## 4. Verification

### 4.1 The table math, checked against upstream rather than against itself

The frequency table was recomputed against the real `mlx_lm.models.rope_utils.Llama3RoPE._freqs`: max relative deviation 3.14e-7 at Llama 3.1 geometry (128 dims, base 5e5, factor 8) and 1.11e-7 at Llama 3.2 (64 dims, factor 32), the residual being scalar `powf` versus vectorized `pow`.

The direction of the table was confirmed empirically rather than by reading, because inverting it is the most plausible way to be wrong while looking right: `mx.fast.rope(x, d, base=b)` is bit-comparable to `mx.fast.rope(x, d, base=None, freqs=b**(arange(0,d,2)/d))`, while the inverted sense differs by 3.3.

### 4.2 Token-exactness, and a near-miss worth recording

The first long-prompt run matched 55 of 64 tokens. The useful move was not to accept that or to start debugging, but to ask whether the mismatching step was decidable at all. Measuring the oracle's own top-2 logprob margins showed step 55 was an **exact 0.00000 tie** against a median margin of 3.375, so the argmax there is decided by floating-point noise, not by the model.

The probe was rebuilt as a needle-retrieval task where the discriminating step has a margin of 0.21875 against a median of 7.32812. Results against the mlx-lm oracle on identical weights:

| Checkpoint | Prompt | Result |
|---|---|---|
| `llama-3.1-8b-4bit` (factor 8) | 4484 tokens, 64 generated | token-exact; pre-change binary diverges at token 5 |
| `llama-3.2-1b-4bit` (factor 32) | 4485 tokens, 48 generated | token-exact; pre-change answers `Q17`, post-change answers the correct `QX-7734` |

The factor-32 case was independently confirmed by forcing `rope_scaling: None` on the oracle and reproducing the pre-change wrong answer, which identifies the missing table as the cause rather than merely correlating with it.

Short prompts on both checkpoints are byte-identical to the pre-change binary, which is the other half of the claim: the fix must change long-context behavior and nothing else.

### 4.3 Cross-family sweep

All 38 local checkpoints declaring `rope_scaling` were mapped against whether they reach `llama3::ModelArgs`. Eight do. Five are the Llama 3.x family. The other three are the interesting ones:

- **`deepseek-coder-1.3b-4bit`** declares `linear` factor 4 and was emitting degenerate `):):):):` on a six-token prompt. `linear` scales every band rather than only the low ones, which is why this one changed on a short prompt while the Llama 3.x rows stayed byte-identical. Now matches the reference. This was an unreported, user-visible defect.
- **`idefics3-8b-llama3-4bit`** declares `llama3` factor 8 in its `text_config` and now applies it, a VLM the issue never mentioned.
- **`internvl3-1b`** declares `dynamic`, loads, warns once, and decodes byte-identically.

Everything else has its own module and never reaches these args. Non-regression confirmed on Qwen2 checkpoints with and without a block.

Server batched decode was exercised end to end: a 4484-token request and a 6-token request issued concurrently at `--max-batch-size 4` produce output identical to the same two issued alone, which is what validates `fast_rope_batched_with_freqs` on real weights rather than in a unit test.

### 4.4 Gate

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8330 passed, 0 failed. `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings` and `cargo fmt --all -- --check` clean.

One gate run failed on `text_only_forward_produces_finite_logits` in `tests/hunyuan_vl_parity.rs`. That is the known flake tracked as #997, which names this exact test: it passes in isolation, it passed on this branch before the review-fix commit, the fix commit touches no file any hunyuan path reaches, and it only fails under full-workspace load, matching the signature #997 and the closed #1023 describe.

## 5. Deliberate Divergence from mlx-vlm

`CLAUDE.md` names mlx-vlm parity as a core project principle, so this is recorded as a decision rather than left as a side effect.

mlx-vlm's `language.py` for `idefics2`, `idefics3` (and SmolVLM) and `internvl_chat` builds a plain `nn.RoPE(dims, traditional, base)` and drops the block entirely; `llava`, `llava_next` and `pixtral` read it only far enough to honor `linear`. Only `mistral3` and `mllama` call `initialize_rope`. After this change, mlxcel applies the table for all of them.

The divergence is deliberate and mlxcel is the correct side. HuggingFace's `Idefics3Model` wraps `LlamaModel`, which does apply `rope_scaling`, so mlxcel now matches the behavior these checkpoints were trained under and mlx-vlm is the implementation dropping it. Locally observable on `models/idefics3-8b-llama3-4bit`.

## 6. What Remains Unverified

`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit` is not present locally, so the issue's vision token-exactness criterion is **unverified**. The `mllama` config forwarding is covered by a unit test, and the VLM route into these args is exercised on real weights by `idefics3-8b-llama3-4bit`, but that is coverage by proxy, not the criterion the issue asked for.

The pipeline stage executors and the tensor-parallel runtime changes are unit-tested only; no multi-node hardware was available.

## 7. Learning Points

- An issue's stated scope is a claim about the code, not a fact about it. Grepping for who *builds* the shared type turned a "Llama 3.x and Vision" change into a ten-family one before any code was written.
- Two of this issue's own prescriptions would each have shipped a regression. Acceptance criteria are requirements; implementation plans and suggested one-line fixes are hypotheses, and they can conflict with the criteria they are meant to satisfy.
- A defensive guard can be a regression. Screening `low_freq_factor == high_freq_factor` is the obvious safe move and would have broken Llama 4 Scout. Scanning the actual checkpoints beat reasoning about the arithmetic.
- When a parity run lands at 55 of 64, measure whether the mismatching step is decidable before treating it as a defect or as noise. An exact logprob tie means the test, not the code, is what needs rebuilding.
- A `pub` function whose safety rests on "every current caller happens to do the right thing" should encode the invariant, especially when violating it crosses an FFI boundary that turns an error into a process abort.
