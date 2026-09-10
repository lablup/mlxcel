# Technical Report: PR #1739 - feat(models): add the Laguna family with compressed-tensors NVFP4 experts

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Completed (21 family tests, the `models::` suite, clippy, fmt and CI green; two published checkpoints run on an M5 Max; no external token-exact reference exists for this family, and Laguna S 2.1 was not downloaded)
**Languages**: Rust, Markdown
**Risk Level**: Medium (a new family, plus two edits inside code every other family shares: a per-expert output multiplier in `SwitchLinear`, and a BOS rule that eight tokenize sites now call instead of spelling out)

---

## Executive Summary

Poolside's Laguna (`model_type: "laguna"`, XS 2.1 / XS.2 / S 2.1) is a hybrid sliding/full attention MoE code model, and its published NVFP4 checkpoints keep every routed expert and the shared expert in the compressed-tensors `nvfp4-pack-quantized` layout. Nothing in `src/` could read that layout. `sanitize.rs` has an NVFP4 repack, but it keys on `.weight_scale_2` and covers only the ModelOpt spelling; the compressed-tensors triplet (`weight_packed` / `weight_scale` / `weight_global_scale`) had no reader anywhere in the tree. This PR adds the family across `laguna.rs`, `laguna_layers.rs` and `laguna_sanitize.rs`, transcodes that layout into MLX native NVFP4 planes at load, and carries the one scalar MLX native NVFP4 has no slot for as a per-expert sidecar applied after the routed matmul.

Two of the changes reach past Laguna. `SwitchLinear::Quantized` gained an optional `[num_experts]` output multiplier, which is the routed analogue of the `QuantizedWeight::apply_global_scale` that dense linears already had. And `MlxcelTokenizer::prompt_carries_bos` replaced the literal `<bos>` / `<s>` prefix test at every site that tokenizes a rendered prompt, which removes an off-by-one the whole Llama 3 lineage has been carrying: its chat template emits `<|begin_of_text|>` and the checkpoint's `TemplateProcessing` post-processor then prepended the same id again.

---

## 1. The quantization layout, and the scalar with nowhere to go

### 1.1 Three tensors in, three tensors out

compressed-tensors NVFP4 stores one quantized linear as `weight_packed` (E2M1 codes, two per byte), `weight_scale` (E4M3 block scales, one per 16 input features), and `weight_global_scale` (one f32 for the whole tensor). The value a row reconstructs to is `code * E4M3(scale) / global`. MLX native NVFP4 accepts the first two and has no third slot.

`laguna_sanitize::transcode_expert_planes` handles the routed case. For each MoE layer and each of `gate_proj` / `up_proj` / `down_proj` it takes the `num_experts` triplets, stacks each of the three tensors on a new leading axis, and then does the minimum work on each:

- The packed bytes are reinterpreted, not converted. `view_packed_as_u32` makes the buffer contiguous and views it as `uint32`, because two E2M1 codes per byte with the low nibble first is exactly MLX native NVFP4's eight-codes-per-word order. The one copy in that function exists because the loader hands back a `U8` array whose buffer may be shared; an `INT8` plane is viewed in place.
- The E4M3 block scales are kept byte for byte when the loader delivered them as `U8`. When it promoted `F8_E4M3` to a float, `encode_block_scales` decodes to f32 and re-encodes, which is lossless for values that came from E4M3 in the first place. That re-encode is threaded over up to 16 chunks, which matters at 234 planes.
- `weight_global_scale` becomes `1 / global` as an f32 `[num_experts]` vector stored at `{prefix}.global_scale`.

The dense case (`transcode_dense_planes`) needed no new consumer: `UnifiedLinear` already reads a `{prefix}.global_scale` key and already has `apply_global_scale`, both from the Inkling ModelOpt work. So the shared expert, which is a plain SwiGLU MLP under unquantized key names, loads through the existing dense path with a `[1]` sidecar.

### 1.2 Why `1 / global` is not folded into the block scales

The obvious simplification is to multiply `1 / global` into every E4M3 block scale during the transcode and emit a plane MLX can consume with no sidecar at all. It was rejected because E4M3 has 3 mantissa bits, so the quotient is generally not representable and the fold would quantize a second time on top of the checkpoint's own quantization. The block scales are the one part of this layout that survives the transcode untouched, and keeping them exact is worth a multiply per routed matmul.

The cost of the sidecar is stated in the code rather than hidden: `SwitchLinear::quantized_ref` returns `None` for a plane that carries a `global_scale`, so those planes never enter the fused affine decode kernel and always take the `gather_qmm` path. That is why the fused and unfused runs on the NVFP4 checkpoint are byte-identical (section 5.2). It is a correctness guarantee, not a coincidence, and it costs the NVFP4 checkpoint whatever the fused kernel would have bought at width 1.

### 1.3 Where the multiply happens

`apply_expert_global_scale` runs after `gather_qmm`, not before. It takes the selected experts' entries with the same index tensor the matmul used, expands the trailing axes to match the output rank, multiplies, and casts back to the output dtype. Doing it on the output rather than on the weights means the scale is applied `k` times per token instead of `num_experts` times per forward, and it keeps the quantized plane itself untouched so a second load of the same weights is idempotent.

---

## 2. Config geometry is untrusted input

`config.json` reaches the loader from `mlxcel generate -m <org>/<repo>`, and several Laguna values travel from there into MLX as raw arguments. The security pass (commit `d00262ea`) closed four ways that ended badly, all of them silent or late.

| Hazard | What reached MLX | Failure shape | Guard |
|---|---|---|---|
| `num_experts_per_tok` missing | `argpartition(kth = num_experts)`, one past the end of the score row | MLX throws; the throw crosses the cxx bridge as an uncatchable abort on the first routed token, after the whole checkpoint is resident | `ModelArgs::validate`, `validate_router_geometry`, and a clamp in `router_select` |
| `num_experts` under-declared | router indices the stacked planes do not cover | `gather_qmm` and the `take` behind the sidecar do not range-check a positive index, so the out-of-bounds read reaches the logits | `validate_router_geometry` compares the router width against each stacked plane; `check_no_expert_beyond` refuses a checkpoint carrying an expert past the declared count |
| Expert planes disagreeing on shape or dtype | `mlx::core::stack` | Same uncatchable abort, mid-load. A promoted mixed dtype would be worse: the packed view reinterprets it at the wrong width and loads without complaint | `PlaneLayout` equality across the whole run, checked before any plane is removed from the map |
| `head_dim < 2` | `Ord::clamp(2, head_dim)` while resolving the partial-rotary width | Rust panic with `min > max` | `ModelArgs::validate` rejects it at load; the helper's own lower bound tracks `head_dim` so a directly constructed `ModelArgs` cannot panic either |

Three of the four are the same class of defect. A value that is wrong by one is not caught by any type, is not range-checked by the kernel it reaches, and produces either an abort with the model already in memory or a wrong number that nothing distinguishes from a right one. `num_experts_per_tok` reaches the first state from the serde default alone: a config that declares sparse layers through `mlp_layer_types` but never spells the key out yields 0, and `kth` is then the full expert count. A value in the other direction was the quieter one: `kth` went negative, `slice` took a negative start, and the block routed fewer than `k` experts with nothing reported anywhere.

The router is guarded in two places on purpose. `ModelArgs::validate` runs before the weights are read, so a bad config fails in milliseconds instead of after a 16 GB load, and it is the only guard that can see a config with no checkpoint behind it. `validate_router_geometry` runs against the loaded tensors, where the router width is a shape rather than a config key, and it is the only one that can catch a `config.json` that disagrees with its own weights. The clamp inside `router_select` is the third layer and exists for callers that build a block directly, including the unit tests.

`check_no_expert_beyond` also has a performance reason that is easy to miss. Without it, a 256-expert 40-layer export that declares 8 experts would stack the first 8 and then send every leftover plane through `transcode_dense_planes` as a dense linear nothing ever reads, which is tens of thousands of f32 materializations and E4M3 re-encodes before the load fails for an unrelated reason.

---

## 3. The BOS rule, and the two count routes

### 3.1 What the literal prefix test could not see

Every tokenize site in the tree tested `!prompt.starts_with("<bos>") && !prompt.starts_with("<s>")` to decide `add_special_tokens`. Laguna's chat template opens with `〈|EOS|〉`, id 2, which the checkpoint uses as both BOS and EOS, and its `tokenizer.json` post-processor is a `TemplateProcessing` with `single: [〈|EOS|〉, A]`. Neither literal matches, so every chat request would have started with a doubled id 2.

`MlxcelTokenizer::prompt_carries_bos` keeps the two literals and adds the tokenizer's own BOS spelling, resolved through `bos_token_id()` and `id_to_token`. Eight call sites now delegate to it: the server scheduler's dispatch-thread tokenizer, the router front end, the diffusion worker's two request paths, the XLA admission worker, offline `generate`, `chat`, and the two benchmark binaries, plus the two ordered-media paths in `model_worker.rs` which take the answer as a parameter so they compute it once per request rather than once per segment. The #633 invariant that pre-tokenizing on the dispatch thread must be byte-identical to tokenizing on the scheduler thread is what forces them all onto one definition.

The rule can only subtract a BOS that the rendered text already carries, and the unit tests pin both directions: a rendered prompt gets exactly one BOS id, and a raw prompt with no BOS text still gets exactly one.

### 3.2 The count routes were a separate defect

`/tokenize` and `/v1/messages/count_tokens` both passed `add_special = true` unconditionally, with a comment claiming that this is what makes the number comparable to `tokens_evaluated`. It was the opposite: for any template that emits its own BOS, the count was one higher than what the request would actually evaluate. Both routes now follow `prompt_carries_bos`, and on the Laguna checkpoint the difference is visible directly, 51 ids against 52.

This is the part of the PR with the widest blast radius, because the Llama 3 lineage has the same template shape and has been over-counting by one on those routes for as long as they have existed. Upstream `apply_chat_template` passes `add_special_tokens=False` for exactly this reason.

---

## 4. Two places this port does not follow mlx-lm

### 4.1 `attention_factor` is honored

mlx-lm's `laguna.py` drops the YaRN block's `attention_factor` key and always derives the factor from `factor`. `transformers.modeling_rope_utils._compute_yarn_parameters` uses the declared value when the config carries one, and the `modeling_laguna.py` that ships inside each published checkpoint inherits that behavior. This port follows the checkpoint's own modeling file, so `layer_rope` overwrites the derived `YarnRope.mscale` when the entry declares `attention_factor`.

The choice is observable on exactly one published checkpoint. `mlx-community/Laguna-XS.2-4bit` declares `attention_factor: 1.0` where the derived value would be 1.3466, so the two runtimes rotate that checkpoint differently. XS 2.1 declares the derived value, so both agree there. Anyone diffing this port against mlx-lm on XS.2 will find the gap here and nowhere else, which is why the module header names it before anything else.

### 4.2 Each attention layer builds its own mask

The shell does not build one prefill mask and hand it down. `Attention::forward` derives the key axis from `mlxcel_core::array_shape(&cache_k)[2]`, the tensor the cache actually returned, rather than from the cache offset, and builds a plain causal band on full layers or a window-clipped one on sliding layers.

Deriving from the returned tensor is what keeps the sliding layers aligned against two different cache implementations. The plain `RotatingKVCache` exposes `min(offset, window - 1)` prior keys; the buffered variant that speculative rollback (#1351) will arm exposes up to `window + buffer`. A mask sized from the offset would be right for one and wrong for the other, and wrong here means attending across the window boundary, which produces plausible text rather than an error.

The same forward-looking restraint shows up in `forward_with_capture` and `LagunaCache::trim`, which are staged for the DFlash drafter and unreachable today. Both are documented as such rather than left to look like live code.

---

## 5. Validation

Host: M5 Max, 128 GB, Metal. Everything below was re-run against the final commit.

### 5.1 Gates

| Gate | Result |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib models::` | 1647 passed, 0 failed, 65 ignored; `models::laguna_tests` is 21 of them |
| `cargo clippy` (scoped), `cargo fmt --all -- --check` | clean |
| CI on the PR | green |
| Same suites under `--features cuda` on the GB10 (sm_121) host | passed on an earlier commit |
| `cargo test --workspace --profile test-fast --features metal,accelerate` | not run whole on this shared host; CI runs the workspace gate |

The 21 family tests are checkpoint-free. They cover the config parse and both schedules, per-layer RoPE resolution on both layer types, the three router score functions, selection on `scores + bias` with weights taken from bias-free `scores`, the per-head gate and its width rejection, all four sanitize paths, prefill causality on both layer types, decode logits matching prefill, and the five load-path rejections from section 2.

### 5.2 `models/laguna-xs-2.1-nvfp4`, the native compressed-tensors path

`mlxcel generate -m models/laguna-xs-2.1-nvfp4 -p "Write a Python retry wrapper with exponential backoff." -n 64 --no-chat-template --temp 0` transcodes 234 planes, loads in 3.2 s at 16.52 GB resident, and writes fluent Python at 53 to 79 tok/s. The 234 decomposes into 39 MoE layers times 3 routed projections (117 stacked planes) plus the same layers' shared-expert triplets (117 dense planes), which is the whole compressed part of that export.

`MLXCEL_FUSED_MOE=0` produces a byte-identical continuation. That is the expected result rather than a surprising one, and it is what section 1.2 predicts: planes carrying a `global_scale` are excluded from the fused decode kernel, so both arms run the same `gather_qmm`.

### 5.3 `models/laguna-xs.2-4bit`, and a divergence that is not this port

The pre-stacked affine checkpoint runs the same command at 121 to 128 tok/s unfused and 108 tok/s fused. Its two greedy continuations diverge from about token 10. The fused and unfused arms of the pre-existing affine decode kernel are what differ, not anything in this family: `models/qwen3-30b-a3b-4bit` diverges at the same point under the same A/B on the same host, and that checkpoint predates this branch entirely. Running the control is what turns "the new family is unstable under `MLXCEL_FUSED_MOE`" into "this host's fused affine kernel diverges on every MoE checkpoint tried", which are different bugs with different owners.

### 5.4 Server

`mlxcel-server` on port 19347 with the NVFP4 checkpoint:

- `/apply-template` renders a prompt opening with `〈|EOS|〉`.
- `/tokenize` returns 51 ids with `add_special=false` (`[2, 97, ...]`) against 52 with `add_special=true` (`[2, 2, ...]`), which is the doubled BOS made visible.
- A `/v1/chat/completions` request with `enable_thinking=false` reports `prompt_tokens = 51`, matching the tokenize route, and ends `finish_reason: "stop"` after 672 of 700 allowed tokens with no `</assistant>` in the text, so id 24 fired the stop rather than the length cap.
- Both count routes answer `input_tokens: 51`.

---

## 6. What is not established

**Token-exactness against an external reference.** The pinned mlx-lm checkout (0.31.3) carries no `laguna.py`, so there is no oracle on either host to diff greedy ids against. Every family in this tree that claims token-exactness has one; Laguna does not, and the standing claim is that both checkpoints load, run, and produce fluent on-topic code at the expected speeds. The fused/unfused A/B in section 5.2 is a self-comparison, not a parity result.

**Laguna S 2.1.** About 100 GB, not downloaded. Its 48-layer config path, its `factor: 128` / `beta_fast: 32` YaRN block, and its 72-head sliding layers are covered by unit tests only.

**The `sqrtsoftplus` router and the attention sink.** Both are implemented from the config semantics and tested against the formula, and no published checkpoint sets either. `swa_attention_sink_enabled` is absent from all three configs, and `moe_router_score_func` is absent from all three, so both arms run only in tests.

**The adapter route.** `loading/config_backed.rs` reaches `LagunaModel::from_weights` without running the sanitize, so an NVFP4 checkpoint fails there with a missing-weight error. Laguna registers `adapter: None` and every other `ConfigBacked` family is wired the same way, so this is a property of that route rather than of this family.

**Tensor parallel and batching.** The family is not wired for TP, and `supports_batching` returns false because the mixed full/sliding caches live in a `RefCell` that per-sequence KV isolation cannot share.

One performance item was reviewed and left alone: `prompt_carries_bos` recomputes `bos_token_id()` per call, which is two tokenizer encodes, at eight sites. Microseconds against a request, and a cache would add state to a type that currently has none.

---

## 7. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 31 |
| Lines added | 3271 |
| Lines removed | 44 |
| New modules | 3 (`laguna.rs` 672, `laguna_layers.rs` 758, `laguna_sanitize.rs` 489) |
| Tests added | 21 family tests, 2 tokenizer tests, 1 detection test |

### Changes by area

- `src/models/laguna.rs`: config, per-layer RoPE resolution (the per-layer-type entry overlaid on the loose top-level scalars, explicit `attention_factor` preferred, `partial_rotary_factor` fallback chain), `validate`, the model shell, the mixed-cache `LanguageModel` wrapper.
- `src/models/laguna_layers.rs`: the gated QK-norm attention with per-layer head counts and partial YaRN rotation, the per-layer mask, the router and its selection rule, `validate_router_geometry`, the MoE block with its shared expert, the dense MLP, the decoder layer.
- `src/models/laguna_sanitize.rs`: the compressed-tensors transcode, the pre-stacked `gate_up_proj` split, per-expert bf16 stacking, router key renames, tied-head removal. Idempotent by construction.
- `src/models/switch_layers.rs`: the `global_scale` field on `SwitchLinear::Quantized`, `apply_expert_global_scale`, the fused-path exclusion, and shape validation of the sidecar against the stacked plane count.
- `src/tokenizer/mod.rs`: `prompt_carries_bos` and `bos_token_string`, with tests for the Laguna shape and the Llama 3 shape.
- Eight tokenize sites plus the two count routes, converted to the shared rule.
- Detection, registry, metadata, `LoadedModel`, memory estimate, the TP fallback table, the `--swa-full` message, and `docs/supported-models.md`.

### Commits

| Hash | Type | Subject |
|---|---|---|
| `e9bc766b` | feat | add the Laguna family with compressed-tensors NVFP4 experts |
| `e78d14eb` | docs | record what Laguna was validated on, and the YaRN gap |
| `e9863f07` | fix | count prompt tokens with the generation add_special rule |
| `d00262ea` | fix | bound the router and expert-plane geometry at load |

### Related issues

Closes #1347. Touches the #633 tokenize invariant. Stages `LagunaCache::trim` and the capture arm for #1351.

---

## 8. Follow-up

**No parity oracle.** Building one means porting `laguna.py` into the pinned mlx-lm checkout or vendoring the checkpoint's `modeling_laguna.py` under `transformers`, then diffing greedy ids on identical prompts. Until then `docs/supported-models.md` states what the runs do establish, which is the mitigation rather than the fix.

**XS.2 will disagree with mlx-lm.** The `attention_factor` decision in section 4.1 is deliberate and documented in three places, but the first person to build a parity oracle will hit it and should read section 4.1 before treating it as a defect.

**The fused affine decode divergence.** Section 5.3 rules it out as this family's problem and leaves it where it was. It reproduces on Qwen3 MoE on the same host and deserves its own issue.

### Transferable lesson

The four hazards in section 2 have one shape: a config scalar that no type checks, that the MLX kernel it eventually reaches does not range-check either, and whose failure arrives either as an abort with 16 GB already resident or as a number that looks like every other number. What made them findable was not reading the new code more carefully. It was asking, for each config field the new module declares, which MLX argument it becomes and what that argument does with a value outside its range. `afmoe` and `bailing_moe` had already answered that question for the router, and their guards were the template; the two NVFP4-specific ones (the expert-plane layout check and the `head_dim` clamp) had no precedent and came from tracing the same path on a layout no other family uses. A new model family is mostly a translation exercise, and the parts that are not translation are exactly the parts where nobody has walked the argument path before.
