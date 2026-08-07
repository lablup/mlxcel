# Technical Report: PR #1082 - feat(models): support quantized Florence-2 checkpoints (4-bit / 8-bit / 6-bit / 3-bit)

**Date**: 2026-08-08
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Completed (with one out-of-scope defect found and recorded rather than fixed)
**Languages**: Rust, Markdown
**Risk Level**: Medium (new load path for a family that previously refused these checkpoints outright; the dense path is unchanged and pinned)

---

## Executive Summary

Florence-2 rejected every checkpoint that carried quantization metadata, so eight published `mlx-community` conversions could not be loaded at all. This lands the quantized path: projections, embedding tables, and the LM head route through the tree's `UnifiedLinear` / `UnifiedEmbedding` layers, which choose dense or packed per weight prefix, and the #854 refusal is narrowed to the handful of tensors this family still consumes dense rather than removed.

The validation is pinned against upstream mlx-vlm running *the same 4-bit checkpoint*, not against the bf16 one. That choice is the substance of the report: comparing a 4-bit run to a bf16 run can only measure how lossy quantization is, never whether the graph is correct, because the two are computing on different weights by construction. Comparing against a second implementation on identical packed bytes isolates the port. Both claims are then made: mlxcel reproduces upstream's 4-bit activations at the dense parity tests' own tolerances, and the distance it travels from bf16 is the same distance upstream travels between the same two checkpoints, to within 5.8e-4 relative RMS.

A second finding came out of the required `large-ft` smoke test and is deliberately not fixed here: the `large-ft` release generates degenerate output in this tree *and* in upstream mlx-vlm, at bf16 as well as 4-bit, and it did so before this branch.

---

## 1. Problem Statement

### 1.1 Background

Epic #850 landed Florence-2 across #852 (BART seq2seq stack), #853 (DaViT tower), #854 (fusion), #855, and #856, for bf16 and f16 exports only. `Florence2Model::load` refused anything else:

```
Florence-2 quantized checkpoints are not supported yet: the BART text stack and
the DaViT tower are built from dense layers. Use a bf16 or f16 export, ...
```

That refusal was correct when it was written. Both halves were built from dense `Linear` and `Embedding`, and a packed `uint32` tensor reaching MLX does not raise a catchable error; MLX throws in C++, the throw crosses the cxx bridge, and the process aborts. Refusing early with a named error beat half-loading and dying inside a kernel. The gap was that the refusal had become permanent, and it was gated on config metadata rather than on anything the loader could not actually handle.

### 1.2 What was blocked

`mlx-community/Florence-2-{base,large}-ft-{3,4,6,8}bit`, all eight. The `base-ft` 4-bit conversion is 163 MB against 542 MB for bf16, so the blocked path was also the one that matters most on a memory-constrained Mac.

### 1.3 What the issue text got wrong

The issue was a sketch, and the checkpoints disagreed with it in five places. These are recorded in full in section 5 because they are the part of this work most likely to mislead the next reader. The most consequential: the issue's proposed solution lists only linear projections, and the embedding tables are the larger half of the change.

### 1.4 Risk assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A packed tensor reaching a dense op and aborting the process | High (uncatchable, no diagnostics) | Certain if any prefix is missed |
| Dequantizing on a wrong group size, producing plausible garbage | High (silent) | Low, but undetectable without a cross-implementation reference |
| Promoting bf16 scales to f16 and perturbing every reconstructed weight | Medium (silent, small) | Certain if the existing unconditional conversion is left in place |
| Regressing the dense path while rewiring it | Medium | Guarded by the five existing parity tests |

---

## 2. Technical Review

### 2.1 What the conversion actually packs

Read off `Florence-2-base-ft-4bit`, 1062 tensors, 198 quantized stems:

| Packed | Dense |
|---|---|
| BART q/k/v/out_proj, fc1, fc2 (encoder and decoder) | every `.bias` on those projections |
| `language_model.lm_head` | every LayerNorm weight and bias |
| `language_model.model.shared` | `image_projection` |
| encoder and decoder `embed_positions` | `visual_temporal_embed.pos_idx_to_embed` |
| `image_pos_embed.{row,column}_embeddings` | `vision_tower.convs.*.proj` |
| DaViT window / channel attention `qkv`, `proj` | the depthwise `*.dw` convolutions |
| DaViT `ffn.fn.net.fc1`, `fc2` | |

The split is not arbitrary. Upstream quantizes by walking `nn.Module`s, so every `nn.Linear` and `nn.Embedding` is packed and everything registered as a raw `nn.Parameter` or as an `nn.Conv2d` is not. That rule is what makes the narrowed refusal in section 3.3 well-founded rather than a guess.

### 2.2 Three defects the issue did not name

**The activation dtype came off a packed plane.** `Florence2TextModel::from_weights` derived its dtype from `shared.weight`. On a quantized checkpoint that tensor is `uint32`. The decoder builds its additive causal mask in that dtype and the fusion path casts image features to it before concatenation, so an integer dtype there is not a cosmetic problem. It now reads `scales` on the quantized arm:

```rust
let dtype = match shared.quantized() {
    Some(quantized) => mlxcel_core::array_dtype(&quantized.scales),
    None => mlxcel_core::array_dtype(shared.weight()),
};
```

**`convert_bf16_weights` ran unconditionally in three places, not one.** The issue points at `model.rs`. `Florence2TextModel::load` and `Florence2DaViT::load` called it too. Scales and biases are dequantization operands rather than activations, so rounding them perturbs every weight the model reconstructs. All three now gate on the checkpoint declaring no quantization, matching what `finish_vlm_weights_common` already does for the shared VLM loader.

**`sanitize` filled `model.shared` with one plane.** BART ties the encoder and decoder token tables to `model.shared`, and exports vary in which of the three they materialize, so the sanitizer copies from `embed_tokens` when `shared` is absent. Copying only `.weight` would present a packed table as a dense one, which is exactly the abort case. It now carries `.scales` and `.biases` across as well.

### 2.3 The position tables changed shape of access

`Florence2Encoder` and `Florence2Decoder` held `embed_positions` as a raw tensor and took `slice(table, [offset + 2, 0], [offset + 2 + seq, d_model])`. A packed table cannot be sliced that way: its stored width is a function of bit depth, not model width. Both now hold a `UnifiedEmbedding` and gather:

```rust
let positions = mlxcel_core::arange_i32(POSITION_OFFSET, POSITION_OFFSET + seq, 1);
let pos = self.embed_positions.forward(&positions);
```

On the dense arm the gather returns exactly the rows the slice returned, which the unchanged dense parity tests confirm. The load-time bound check moved with it, from a hand-written shape comparison to the shared `validate_embedding_table` guard, whose quantized arm reconstructs the logical width from `scales` and the reconciled group size. Without that reconstruction the old `cols == d_model` comparison rejects every quantized checkpoint, since `cols` there is 96, not 768.

### 2.4 Plumbing

`Florence2Quantization { group_size, bits }` is parsed from the top-level `quantization` object and carried on both `Florence2TextConfig` and `Florence2VisionConfig`. The block sits *beside* `text_config` and `vision_config`, not inside either, so only the whole-document parsers can fill it; `from_text_config` and `from_vision_config` leave it at the dense default and `from_model_config` overwrites. The DaViT side threads it through `BlockParams`, which is `Copy` and already carries the per-stage geometry, so no new parameter runs through three struct layers.

---

## 3. Technical Decisions

### 3.1 Reference on the same weights, not on the dense weights

The issue asks for outputs "consistent with the bf16 path within a stated tolerance". Taken literally that is the wrong experiment. 4-bit affine reconstruction error is roughly 3.3% of a group's dynamic range per weight, and the DaViT tower is pre-norm, so twelve blocks of it accumulate along an unnormalized residual stream. The measured divergence at the tower output is 25% relative RMS. Any tolerance loose enough to accept that is far too loose to catch a wrong-rows position gather, and any tolerance tight enough to catch one rejects a correct implementation.

Upstream mlx-vlm on the same 4-bit checkpoint has no such problem: both runtimes dequantize identical bytes with identical parameters, so anything beyond f16 op-ordering noise is a defect in this port. That is the primary pin, and its tolerances are the dense parity tests' own (1e-2 on image features, 8e-2 on encoder hidden states, 5e-2 on step-0 logits).

### 3.2 The dense comparison became a cross-implementation check

Having measured the divergence, throwing it away would waste it. `florence2_quantization_cost_matches_mlx_vlm` asserts mlxcel's 4-bit-versus-bf16 distance against upstream's 4-bit-versus-bf16 distance on the same two checkpoints:

| stage | mlxcel rel. RMS | mlx-vlm rel. RMS | mlxcel cosine | mlx-vlm cosine |
|---|---|---|---|---|
| image features | 0.25486 | 0.25540 | 0.967595 | 0.967458 |
| encoder hidden | 0.41641 | 0.41583 | 0.913923 | 0.914181 |
| step-0 logits | 0.11858 | 0.11861 | 0.992970 | 0.992966 |

Agreement is 5.8e-4 on relative RMS and 2.6e-4 on cosine. The test's bar is 5e-3 and 2e-3, about an order of magnitude of headroom, and still far too tight for a wiring defect: every failure mode this guards against moves cosine by 1e-1 or more. A hand-picked threshold would only have said "not absurd"; this says the port pays the reference implementation's cost and no more.

Note that the two metrics are not independent after a LayerNorm. `sqrt(2 * (1 - 0.967458))` is 0.2546, which is the measured relative RMS at that stage. They are reported together anyway because they stop being redundant at any stage where a gain shift, rather than a direction change, dominates.

### 3.3 The refusal was narrowed, not removed

`reject_unsupported_quantized_tensors` scans the sanitized weight map for `.scales` on a stem this implementation consumes dense: `image_projection`, `visual_temporal_embed.*`, `convs.*.proj`, and any `*.dw`. Those are exactly the tensors upstream's module walk cannot reach, so a checkpoint carrying them packed is non-standard rather than merely newer, and refusing it with the tensor named is more useful than aborting inside `conv2d`. It runs after `sanitize` in both `Florence2Model::load` and `Florence2DaViT::load`.

### 3.4 The declared bit range is permissive

`quantization.bits` is bounded at 1..=32 rather than to the four widths published today, matching `mlxcel_core::layers::validate_quantization_params` and for the same reason: the unified layers re-derive an effective bit width from the packed shapes, so an allowlist would reject a legitimate future export. `group_size` is bounded at 1..=4096. Neither bound is a correctness check; both keep a value that can match no real tensor out of the reconciler.

---

## 4. Validation

### 4.1 Numeric

`tests/florence2_quantized_parity.rs`, checkpoint-gated, four tests. Observed on this box:

```
image_features:  mean -0.004166 (ref -0.004168), std 0.815221 (ref 0.815222)
encoder_hidden:  mean  0.067354 (ref  0.067450), std 3.563928 (ref 3.566599)
step0 logits:    max abs deviation 0.0061 at index 0 (tol 0.05)
greedy ids:      4-bit [0, 879, 27740, 868], bf16 [0, 879, 27740, 868]
```

The greedy sequence is identical between 4-bit and bf16 on this input, and matches upstream's on the 4-bit weights.

### 4.2 CLI

COCO `000000039769` for caption and detection, and a generated 640x200 PNG reading `HELLO MLXCEL` for OCR, because the cats image has no text and `<OCR>` correctly returns `-` on it.

| task | `base-ft-bf16` | `base-ft-4bit` | `base-ft-8bit` |
|---|---|---|---|
| `<CAPTION>` | Two cats are sleeping on a pink blanket. | Two cats laying on a pink blanket next to remotes. | Two cats are sleeping on a pink blanket. |
| `<OD>` | cat, cat, couch, remote, remote | cat, cat, couch, remote, remote | cat, cat, couch, remote, remote |
| `<OCR>` | HELLO MLXCEL | HELLO MLXCEL | HELLO MLXCEL |
| `<OCR_WITH_REGION>` | [43.2, 71.1, 599.4, 71.1, 599.4, 127.7, 43.2, 127.7] | [41.9, 70.9, 599.4, 70.9, 599.4, 127.7, 41.9, 127.7] | not run |

8-bit reproduces the dense caption word for word and the dense boxes to within 0.7 px. 4-bit keeps every detection label and every box to within about 1 px, reads the text exactly, and rephrases the caption. Rephrasing a free-form generation is what 4-bit weights do; it is not what a wiring defect does, which is why the numeric pin rather than this table is the correctness argument.

### 4.3 Checkpoint-free

`src/models/florence2/florence2_quantized_tests.rs`, ten tests that need no weights, so CI exercises them: config parsing and its range guards, the distinction between "declares quantization" and "parses to the dense defaults" (a real 4-bit group-64 export declares exactly the dense fallback values, so only the block's presence can separate them), the narrowed refusal in both directions, and the three-plane `model.shared` fill.

---

## 5. Corrections to the Issue Text

1. **Group size does not vary.** The issue warns it might. All eight conversions declare `group_size: 64`; only `bits` changes. The code still reads it, because `UnifiedLinear` trusts the declared group size and re-derives bits from shapes, so a wrong one mis-strides everything silently.
2. **Embedding tables are the larger half and go unmentioned.** `model.shared`, `lm_head`, both `embed_positions`, and both `image_pos_embed` tables are packed. Implementing only the linear projections would have aborted at the first lookup.
3. **`image_projection` is not quantized.** The issue's phrasing suggests it might be. It is a raw `nn.Parameter` applied as a right-hand matmul, so `nn.quantize` never sees it. Same for the temporal buffer and the conv stack. Those three are what the narrowed refusal still rejects.
4. **The activation dtype and the sanitizer fill are not mentioned at all.** Both are in section 2.2.
5. **`convert_bf16_weights` is wrong in three files, not one.**
6. **The 4-bit `config.json` carries `vision_config.hidden_size: 768`, which the bf16 one does not.** Inert in both runtimes, since the tower width comes from `dim_embed`. Recorded because an earlier issue in this chain flagged a nonexistent `hidden_size`.

---

## 6. `large-ft` Is Broken Upstream and Was Already Broken Here

The issue asks for a `large-ft` smoke test. That test found something, and it is not this change.

`Florence-2-large-ft-4bit` loads cleanly and its packing is internally consistent: 294 quantized tensors, every one a 4-bit / group-64 fit against its own `scales` width, with exactly the expected dense set. It then generates a run of BOS for every task. Four measurements place the cause outside this branch:

- `Florence-2-large-ft-bf16` on the **pre-change `main` binary** produces the same BOS run for `<CAPTION>`, while `<OD>` and `<OCR>` work.
- Upstream mlx-vlm on `Florence-2-large-ft-4bit` produces `greedy_ids = [0, 0, 0, ...]`.
- Upstream mlx-vlm on `Florence-2-large-ft-bf16` produces the same.
- `base-ft` is correct at bf16, 4-bit, and 8-bit in both runtimes, through the same code.

So the reference implementation behaves this way too, and the quantized `large-ft` path here is not doing anything the dense one is not. Diagnosing it belongs in its own issue against the `large-ft` family. `docs/supported-models.md` now states the behavior instead of claiming the family works, which corrects a claim that predates this PR. The smoke test keeps `large-ft-4bit` in its list, because loading it still proves the packing path is not `base-ft`-shaped (`d_model` 1024, twelve BART layers, `dim_embed` up to 2048), and it asserts only load plus finite logits.

---

## 7. Learning Points

**Pick the reference that can answer the question you are asking.** The instinct on a quantization change is to diff against the dense checkpoint, and that comparison cannot distinguish a correct lossy implementation from an incorrect one, because both are far from dense. A second implementation on identical packed bytes can. The dense comparison still earned its place, but as a cross-implementation equality check rather than as a threshold.

**A packed tensor's shape is not its shape.** Three separate load-time guards in this family compared a stored width against a model width, and all three are wrong for a quantized checkpoint (`cols` is 96 where `d_model` is 768). The tree already had the right primitive in `validate_quantized_packing`; the work was noticing that the comparison, not the value, was the bug.

**Config-derived predicates need to say what they mean.** `Florence2Quantization::DENSE` is `{64, 4}`, which is exactly what a genuine 4-bit group-64 export declares. Testing `parsed != DENSE` to detect quantization would therefore have silently treated every 4-bit checkpoint as dense and promoted its bf16 scales. Only the block's presence carries that information, which is why `config_is_quantized` exists as a separate predicate and has its own test.

**A smoke test that only asserts "loads" will find things you then have to decide about.** The `large-ft` result is not a failure of this issue and fixing it here would have been scope creep, but silently dropping the checkpoint from the test list would have buried a real defect. Recording it, correcting the documentation, and leaving the load coverage in place is the honest middle.

---

## 8. Not Covered

- **`mlxcel-server` still refuses Florence-2 at startup.** Out of scope by construction; that is issue #1073, implemented directly after this one. `src/server/` is untouched here.
- **The `large-ft` degeneracy is diagnosed only far enough to place it outside this change.** No hypothesis about its cause is offered, and none should be read into section 6.
- **3-bit and 6-bit are not exercised end to end.** They declare the same `group_size: 64` packing and load through the same path, and the bit-width smoke test picks them up automatically if the directories are present, but neither was downloaded for this run. 4-bit and 8-bit were, at `base-ft`, plus 4-bit at `large-ft`.
- **No performance measurement.** This is a capability change; whether 4-bit Florence-2 decodes faster than bf16 on this hardware was not measured and no claim is made.

---

## References

- Issue #1072, and the epic chain #850, #852, #853, #854, #855, #856
- `tests/florence2_quantized_parity.rs`, which documents the reference-regeneration recipe in its module comment
- mlx-vlm florence2: [florence2.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/florence2.py), [language.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/language.py), [vision.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/vision.py)
- PR #1078, whose `dequantize` biases guard covers the quantized-embedding path this change relies on
