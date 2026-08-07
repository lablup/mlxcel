# Technical Report: PR #1075 - feat(models): add Falcon-OCR (falcon_ocr) early-fusion VLM support

**Date**: 2026-08-07
**Author**: Jeongkyu Shin
**Reviewer**: implementation and security review cycle
**Status**: Completed (layout detection itself stays out of scope; see Section 5)
**Languages**: Rust, Markdown
**Risk Level**: Medium (new model family, isolated to `falcon_ocr`; review caught a defect that silently skipped the entire decoder on the server path, fixed before merge)

---

## Executive Summary

PR #1075 adds Falcon-OCR (`model_type: falcon_ocr`, roughly 300M parameters), TII's document-OCR VLM and the only model in the tree with no vision tower at all. Each 16x16 RGB patch is flattened and pushed through a single linear `img_projector` straight into the token stream, and one 22-layer decoder reads image and text together under a hybrid mask that is bidirectional inside every image block and causal everywhere else. Issue #848 suggested reusing the Falcon-H1 backbone; the checkpoint's own tensors said otherwise, and the port follows the tensors.

Three defects surfaced during bring-up and review, and all three are fixed in the merged state: an inclusive-cumsum boundary error at `<|end_of_image|>` that a flawed cross-check helper initially agreed with, a `reset_runtime_state` ordering hazard that produced fluent garbage instead of a crash, and a scheduler-level bug where the model's `sequence_state_layout()` fell through to the trait default, handed `forward_impl` an empty cache slice, and skipped all 22 decoder layers with no error on the server path while the CLI stayed correct. Review also found and closed two HIGH-severity issues in the layout-aware CLI driver: unbounded memory amplification from materializing every cropped region up front, and a quadratic decode cost from duplicating the whole prefill state on every generated token.

Validated end to end on the real `tiiuae/Falcon-OCR` checkpoint against a PyTorch reference built from the checkpoint's own vendor code: two rendered documents transcribe exactly and token-for-token match the reference, and a five-region layout run returns each region's exact text with category-specific formatting (Markdown headings for titles, HTML for tables). The full CUDA merge gate on the merge candidate passed 7590 tests with 5 failures that are identical on `main` (4 TF32 artifacts, 1 pre-existing integer underflow); the PR added 225 net passing tests.

---

## 1. Problem Statement

### 1.1 Background

Issue #848 asked for Falcon-OCR support and described it, from the upstream mlx-vlm directory and an earlier triage note, as a compact early-fusion OCR model: no separate vision encoder, 16x16 image patches linearly projected into the token stream, and a hybrid attention mask that is bidirectional within each image block and causal over text. The issue's implementation plan suggested a specific starting point: reuse the Falcon-H1 backbone (`src/models/falcon_h1.rs`), since the name suggests a shared lineage, while flagging that upstream's `language.py` looked Llama-derived and that the shipped `config.json` should be checked first. It also expected per-head Q/K RMSNorm weights among the decoder's custom pieces, and listed layout-aware post-processing from `layout.py` as an acceptance criterion alongside the raw OCR decode.

None of those three particulars survived contact with the actual checkpoint. Falcon-H1 is a Mamba2 and attention hybrid; Falcon-OCR's weights describe a small Llama-derived decoder with five bespoke pieces and no SSM state at all. The per-head norms are not weight tensors the checkpoint happens to omit; they are non-parametric operations that the vendor code runs unconditionally. And the layout module the issue asked for is easy to write and easy to leave unreachable, which is exactly what happened on the first pass.

### 1.2 Existing Issues

- **The Falcon-H1 reuse suggestion was unverified against the checkpoint's own tensors.** Following it would have meant retrofitting a Mamba2 hybrid onto weights that describe a plain attention decoder.
- **The checkpoint ships exactly five tensors per layer** (`attention.wqkv.weight`, `attention.wo.weight`, `attention.sinks`, `feed_forward.w13.weight`, `feed_forward.w2.weight`) and none of them are named `attention_norm`, `ffn_norm`, `q_norm`, or `k_norm`. Reading that as "the norms are missing" would have skipped every pre-norm in the stack, including the ones the issue explicitly expected.
- **`src/vision/falcon_ocr_layout.rs` started out reachable only from its own `pub mod` line.** An orphaned module satisfies the letter of "port `layout.py`'s post-processing" while leaving the acceptance criterion, working layout-aware OCR, unmet.
- **The layout driver's first draft trusted its inputs.** A `--layout-detections` JSON file is attacker-shaped input on any server-adjacent path, and the first cut read it unbounded and cropped every surviving region before the first OCR token.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Building on the Falcon-H1 backbone before checking the checkpoint | High (wrong architecture family entirely) | Avoided by inspecting `config.json` and the tensor list first |
| Treating the five-tensor-per-layer checkpoint as missing Q/K norm weights | High (every layer's activations wrong) | Avoided by reading the checkpoint's own `modeling_falcon_ocr.py` |
| `sequence_state_layout()` falling through to the `supports_batching()`-derived default | Critical (all 22 layers skipped, no error, server-only) | Certain until instrumented; found by running the real checkpoint against `mlxcel-server` |
| Unbounded per-region crop materialization in the layout driver | High (a few KB of JSON can commit roughly 10 GB before decode starts) | Certain on any detections file listing several page-sized boxes |
| Duplicating the whole prefill state on every decode step | Medium (correct output, decode cost quadratic in prompt length) | Certain on every image prompt, CLI and server alike |

---

## 2. Technical Review

### 2.1 Checkpoint inspection over issue speculation

The shipped `config.json` settled the backbone question and the key-scheme question in the same read. It uses the raw key scheme rather than the HF one: `dim` 768, `n_layers` 22, `n_heads` 16, `head_dim` 64, `n_kv_heads` 8, `ffn_dim` 2304, `norm_eps` 1e-5, `spatial_patch_size` 16, `img_id` 227, `img_end_id` 230, `image_cls_token_id` 244, and `image_reg_1` through `image_reg_4` at 245 through 248. `torch_dtype` is float32.

The tensor list settled the backbone question directly: 115 tensors total, 22 layers times five plus five globals. Per layer: `attention.wqkv.weight` at `[2048, 768]` (a fused 16-head times 64-dim query, plus 8-head times 64-dim key and value), `attention.wo.weight` at `[768, 1024]`, `attention.sinks` at `[16]`, `feed_forward.w13.weight` at `[4608, 768]` (a row-interleaved gate and up projection, `2 * 2304`), and `feed_forward.w2.weight` at `[768, 2304]`. The five globals are `tok_embeddings.weight`, `img_projector.weight` at `[768, 768]`, which equals `16 * 16 * 3`, the flattened patch dimension, `freqs_cis_golden` at `[16, 16, 2]`, `norm.weight`, and an untied `output.weight`. Nothing in that list is a Mamba2 state-space parameter, and nothing in it is a per-head norm weight either, which is the subject of the next section.

### 2.2 The per-layer-norm discrepancy

The issue's architecture notes, carried over from the upstream directory listing, expected per-head Q and K RMSNorm weights among the decoder's custom pieces. The checkpoint's own `modeling_falcon_ocr.py` resolves the discrepancy: every pre-norm in the stack, including the per-head Q and K norms, is `F.rms_norm(x, (dim,))` with no learnable weight (lines 140, 145, and 146 for the attention-block norms, line 230 for the feed-forward norm). Line 339 is the one exception, the single parametric `norm` that runs once before the LM head.

The distinction matters for more than bookkeeping. A weightless RMSNorm still divides by the root-mean-square and still needs an epsilon, and PyTorch resolves an unset `eps=None` to `torch.finfo(float32).eps`, not to the config's `norm_eps`. mlxcel's port keeps the two constants separate: `fast_rms_norm_no_weight` runs with `f32::EPSILON` for every non-parametric norm, and the config's `norm_eps` (1e-5) applies only to the one parametric `norm` before the LM head. This deliberately departs from mlx-vlm's cross-check implementation, which applies `rms_norm_eps` uniformly to all of them; the checkpoint's own vendor code is the higher-fidelity source here; see Section 3.2.

### 2.3 Two bring-up bugs

**The `<|end_of_image|>` boundary was inside the bidirectional block.** The reference builds the hybrid mask with inclusive cumsums over the image-start and image-end markers, so both counters have to absorb the boundary token itself before the membership test runs. Getting this backwards puts the closing tag inside the region that attends bidirectionally rather than causally, which shifts every position after it. The methodological point is worth recording on its own: the first cross-check helper written to validate the mask had copied the same off-by-one, so a whole-matrix comparison against it agreed and validated the bug against itself. What caught it was a narrow unit-test assertion on a specific boundary position rather than the aggregate comparison, and an independent re-derivation from `attention.py::get_image_prefix_mask_mod` later confirmed the corrected rule: inclusive cumsums place `<|image_cls|>` inside the bidirectional block and `<|end_of_image|>` outside it.

**`reset_with_model` wiped the prefill state between the image stage and prefill itself.** Falcon-OCR's per-request positional state, temporal positions, spatial coordinates, and the decode-time rope delta, is written when the image stage processes the prompt and consumed by prefill and every decode step after it. The generate loop calls `reset_runtime_state()` after the image stage has written that state and before prefill reads it, so overriding the reset to clear it there discarded the positions for the very request about to run. The symptom was not a crash; it was fluent LaTeX-flavored garbage, which is a harder signal to trace back to a state-lifecycle bug than an error would have been. The fix leaves `reset_runtime_state` deliberately unoverridden and instead handles staleness at the prefill boundary: a prompt length that does not match the stashed positions evicts the entry, which is also what keeps an image turn from shifting the positions of a following text-only turn.

### 2.4 The critical defect: an empty cache slice with no error

The most serious defect surfaced after the model otherwise worked on the CLI. `src/models/falcon_ocr.rs` did not override `sequence_state_layout()`. The trait default, at `src/lib/mlxcel-core/src/generate.rs:547`, infers the allocation layout from `supports_batching()`: true maps to a dense per-layer KV cache, false maps to `SequenceStateLayout::model_owned`, the placeholder used for SSM and recurrent runtimes that keep their own state and expect no external cache at all. Falcon-OCR returns `supports_batching() == false`, not because it owns its state, but because its per-request positional bookkeeping is single-row and does not support the batched decode path. The default had no way to distinguish those two reasons, and it picked the wrong one.

`SequenceStateLayout::model_owned`'s external cache vector is empty by construction. `forward_impl` walks `self.layers.iter().zip(caches.iter_mut())`, so an empty slice zips to zero iterations: all 22 decoder layers were skipped, and the LM head read the raw token embedding straight through. Nothing panicked and nothing logged. The CLI path was unaffected because it builds its own caches directly through `make_caches()` and never asks the scheduler for a layout, so only `mlxcel-server` was broken, and it was broken for every request against this model, image or text-only alike.

It surfaced by running `mlxcel-server` against the real checkpoint rather than by code inspection: a rendered document transcribed as `anner'\td colspan others than any of these two disl_i because if you may not quite`, and two differently sized pages returned byte-identical output regardless of their content. A temporary instrumentation probe confirmed the shape of the bug directly, `caches.len()=0 layers=22` on the server against `caches.len()=22` on the CLI for the same checkpoint. The fix declares `SequenceStateLayout::dense_kv_cache(self.layers.len())` on `FalconOcrTextModel` and delegates it from `FalconOcrVlModel`, mirroring the identical override already present on Phi4MM (`src/vision/phi4mm_vl.rs:629`), which declines batching for an unrelated reason and hits the same default. After the fix the server reproduces the CLI output exactly, verified against a four-request sequence mixing image and text-only turns.

### 2.5 Two HIGH-severity findings in the layout driver

**Unbounded memory amplification.** `plan_layout_regions` materialized one full `DynamicImage` crop per surviving detection and held all of them before the first OCR token ran. `crop_imm` copies rather than views, and the driver's own read cap admits a 32 MiB detections file describing, in a few kilobytes of JSON, twenty whole-page boxes on a large document image; at that point the up-front crop list costs roughly 10 GB resident before any generation starts. The fix splits `crop_region` into a geometry predicate, `plan_layout_region_boxes`, that computes clamped crop rectangles without touching pixel data, and a separate cropping step that the driver now calls one region at a time inside its own loop. Peak image memory is the page plus one crop, regardless of how many entries the detections file lists. Three supporting gaps closed alongside it: the file read is capped at 32 MiB and the cap is enforced against bytes actually read rather than `metadata().len()`, which reports zero for a fifo and would let a fifo bypass the limit entirely; the detection count is capped at 4096 entries, checked before the vector is built, since nested-box suppression is quadratic in the entry count; and the class name is bounded to 64 characters with a control-character check, because it is the one attacker-controlled string this command echoes back to the operator's terminal, in the summary line's `no text category:` list, and was an ANSI escape-injection channel before the check.

**Quadratic decode.** `src/models/falcon_ocr.rs`, in the neighborhood of line 547, resolved the stashed prefill state on every decode step by duplicating it: a heap clone of the whole prompt's position vector plus a copy of the `[1, L, 2]` spatial-coordinate array across the FFI boundary. Both results were discarded immediately, because the decode arm (`l == 1`) reads only the scalar `rope_delta` and never touches the positions vector or the spatial coordinates. An image prompt is thousands of tokens, so decode did O(prompt_len) work per generated token on both the CLI and the server, making total decode cost quadratic in prompt length. The fix adds a borrow-based `with_entry` accessor on `FalconOcrRuntimeState`, the same pattern `qwen_mrope_state.rs` already uses for its own per-sequence state, and a `decode_rope_delta` built on it that reads the scalar without duplicating the entry. `take_for_prefill`'s length check now goes through the same borrowing accessor, so an eviction no longer pays for a duplicate it is about to discard.

### 2.6 JSON validation completeness

The `--layout-detections` parser's per-entry validation was audited as a complete case list rather than a representative sample: top-level shape (object with a `detections` array, or a bare array), entry type, class-name presence and type across all four accepted alias keys, bounding-box presence and type across all three accepted alias keys and both coordinate spellings, exact field arity, per-axis numeric type, finiteness checked after the value is downcast to `f32` (so a value like `1e400`, which overflows to infinity only in the narrower type, is still caught), and inverted or empty geometry. Every gap the audit found was at the aggregate level, not the per-field level, and all four (the byte cap, the entry cap, the class-name bound, and the finiteness check ordering) are closed in the merged state.

### 2.7 Validation evidence

Plain OCR on a rendered document with known ground truth (`The quick brown fox jumps over the lazy dog. Invoice 2026-0848. Total due: $1,234.56`) returns that text exactly and matches the PyTorch reference token for token. A separate five-region page, rendered with PIL and given hand-written detections that include a nested `formula` box and a textless `picture` box so the suppression and skip paths both run rather than being merely asserted, produces:

```
Falcon-OCR layout: 7 detection(s); 1 nested box(es) dropped; no text category: picture; 5 region(s) to OCR
[1] title       -> # Quarterly Revenue Report
[2] section_header -> ## Executive Summary
[3] text        -> Revenue grew steadily across all three regions during the quarter. ...
[4] table       -> <table><thead><tr><th>Region</th>...
[5] page_footer -> Page 1 of 4
```

All five regions match the drawn text in file order, the nested formula is dropped, and the textless picture is skipped rather than sent to OCR. The category instruction is load-bearing rather than cosmetic: the title region returns `#` and the section-header region returns `##` from otherwise identical single-line crops, and the table region returns HTML rather than flat cell text, which confirms the per-category prompts actually reach the task head instead of being ignored.

### 2.8 Full CUDA merge gate

The full local gate on the merge candidate passed 7590 tests with 5 failures, and those same 5 fail identically on `main` (4 are TF32 reassociation artifacts, 1 is a pre-existing integer underflow unrelated to this model). The PR's own module-level counts, after the two post-review fixup commits added regression coverage for the layout defect and the decode duplication, land at 61 passed for `--lib falcon_ocr`, 6 passed for `--test falcon_ocr_parity`, and 20 passed for the CLI driver's own test module, on top of the unchanged 26 for `models::detection` and 12 for the two `model_metadata` suites. Net across the full gate, the PR added 225 passing tests.

---

## 3. Technical Decisions

### 3.1 A dedicated decoder module instead of the Falcon-H1 backbone

**Context:** Issue #848's implementation plan suggested reusing `src/models/falcon_h1.rs` and its `"falcon_h1"` detection arm as the starting point, while flagging upstream's `language.py` as looking Llama-derived and asking to confirm against the shipped `config.json` first.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Reuse `falcon_h1.rs`, extend its detection arm | Less new code; fits an existing module and an existing arch string | Architecturally wrong: Falcon-H1 is a Mamba2 and attention hybrid, and the checkpoint's 115 tensors describe a plain Llama-derived decoder with no SSM state at all |
| **Chosen: a dedicated `falcon_ocr.rs` module** | Matches the checkpoint's actual tensor layout and the vendor code exactly | New model: fused-QKV split, per-head sinks through the sink-aware SDPA entry point, squared-ReLU MLP, and the 3-D rotary are all bespoke, not inherited |

**Rationale:** the tensor list is not ambiguous evidence. A Mamba2 hybrid would ship selective-scan state parameters (`A_log`, `D`, `dt_bias`, convolution weights); this checkpoint ships none of them and ships exactly the five per-layer tensors a fused-QKV attention decoder needs. Following the issue's suggested starting point past that evidence would have meant building the wrong architecture correctly.

### 3.2 Follow the checkpoint's own eps handling over the cross-check implementation's

**Context:** mlx-vlm's `falcon_ocr` module, used during bring-up as a cross-check, applies `rms_norm_eps` uniformly to every RMSNorm call, parametric and non-parametric alike. The checkpoint's own `modeling_falcon_ocr.py` does not: it passes `eps=None` to `F.rms_norm` for every non-parametric call, which PyTorch resolves to `torch.finfo(float32).eps`, a different, larger constant than the config's `norm_eps` (1e-5).

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Skip the non-parametric pre-norms entirely, reading five tensors per layer as "the norms are missing" | Simplest code path | Wrong: `F.rms_norm(x, (dim,))` still runs on every pre-norm in the reference; skipping it changes every layer's activations |
| Follow mlx-vlm's uniform `rms_norm_eps` for every norm call | One constant, less code, agrees with an existing MLX implementation | Diverges from the checkpoint's own vendor code on the five non-parametric norms per layer, which is the majority of the norm calls in the stack |
| **Chosen: `fast_rms_norm_no_weight` with `f32::EPSILON` for non-parametric norms, `norm_eps` only for the final parametric `norm`** | Matches `modeling_falcon_ocr.py` exactly, including PyTorch's `eps=None` resolution | Two active epsilon constants in one decoder layer, which is easy to transpose if the code is touched without the distinction in mind |

**Rationale:** the checkpoint's own vendor code is the ground truth for what a specific checkpoint's weights expect, and mlx-vlm agreeing with it everywhere it was checked does not make mlx-vlm the tiebreaker on the one place they diverge. `torch.finfo(float32).eps` and 1e-5 differ by roughly two orders of magnitude, so this is not a cosmetic choice.

### 3.3 A CLI driver over the primitives, not a detector port or dead code

**Context:** `src/vision/falcon_ocr_layout.rs` shipped with its layout primitives implemented but reachable only from its own `pub mod` declaration. Issue #848 lists layout-aware post-processing from `layout.py` as an acceptance criterion, and the reference's first stage, PP-DocLayoutV3 through `transformers.AutoModelForObjectDetection`, is a separate object-detection architecture that mlxcel does not ship.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Leave the module unreachable and consider the criterion satisfied because the code exists | Zero additional surface area | Does not meet the criterion in substance; nothing exercises category routing, nested-box suppression, or cropping against a real page |
| Port PP-DocLayoutV3 as part of this issue | Closes the loop end to end with no external input required | A separate object-detection architecture; explicitly out of scope for #848, and large enough to be its own port |
| **Chosen: `mlxcel generate --layout-detections <FILE>`, accepting the `mlxcel detect --format json` shape** | Wires every primitive to a real, tested CLI flow today; composes with a future in-tree layout detector without a format change, since the accepted schema is the existing detector's own output shape | Detection stays external: a user has to supply boxes from somewhere, and the report documents (Section 5) that the one detector combination tried does not currently produce usable boxes for this task |

**Rationale:** the acceptance criterion is about post-processing being reachable and correct, not about detection being solved in the same issue. Building the driver around the existing `Detection` shape and the existing `mlxcel detect --format json` output means the primitives are exercised by real tests today, through 20 CLI-driver tests plus the plan-to-regions and reading-order coverage, and inherit a working detector automatically whenever one lands, rather than waiting on a second, larger port.

---

## 4. What Was Not Verified

Four things are explicitly outside what this PR exercised. Multi-image prompts were not tested; the port handles one image block per prompt. Quantized conversions do not exist for this checkpoint yet, so the `w13` de-interleaving logic that would need to carry `scales` and `biases` alongside the weight is implemented but untested against a real quantized checkpoint. Concurrent server requests were not load-tested; the per-sequence state binding, `bind_falcon_ocr_state_to_sequence`, follows the same pattern already validated under load for Qwen-MRoPE and Gemma-4 and is covered by unit tests, but this PR did not run a live multi-request server workload against it. And passing the literal string `<|OCR_PLAIN|>` as the `-p` prompt produces garbage, while ordinary prompts work correctly, because the task token is appended automatically and a prompt that already contains it collides with that logic; this is a documented usage gotcha rather than an open defect.

---

## 5. The Cost, Stated Plainly

Layout detection itself is not part of this port, and that is a real gap, not a rounding error. `--layout-detections` cannot close the loop on its own: the boxes have to come from somewhere else. `mlxcel detect` does load RT-DETR-family checkpoints and produces exactly the JSON shape this flag consumes, but pointing it at the in-tree `docling-layout-heron` document-layout checkpoint on a rendered page returned boxes that do not correspond to the page content, a `page_footer` box at `[0.00, 103.37, 758.77, 124.19]` on a 1000x1300 page whose actual footer sits around y=1220. That mismatch is a separate, pre-existing defect in that detector pairing, and it is documented here rather than fixed here, because whether it is a predictor bug, a preprocessing mismatch for this checkpoint family, or the wrong model for the input is a question this PR did not investigate.

The layout driver also does not batch. The reference OCRs crops in chunks of 32; this driver runs one region per generation. Region counts per page are small, so this is a throughput note rather than a correctness one, but it means a dense page with dozens of regions pays per-region model-load-adjacent overhead that a batched path would not.

Anyone who needs quantized Falcon-OCR or multi-image Falcon-OCR prompts today does not have them; both are plausible follow-ups on top of the loader and mask machinery this PR ships, not blocked by anything architectural.

---

## 6. Lessons

- **A checkpoint's own tensors overrule an issue's implementation plan, every time they disagree.** The Falcon-H1 suggestion and the per-head-norm-weight expectation were both reasonable readings of upstream's directory structure, and both were wrong once the actual `config.json` and tensor list were read.
- **A cross-check implementation can validate a bug against itself.** The first `<|end_of_image|>` boundary error passed a whole-matrix comparison because the comparison helper carried the same off-by-one. Aggregate agreement between two independently written pieces is not independent evidence when they share a source of truth; a narrow, specific assertion is what caught it.
- **A model can pass every CLI test and still be completely broken on the serving path.** The `sequence_state_layout()` defect produced correct output everywhere the CLI's own cache construction was exercised and silently wrong output everywhere the scheduler's inference was exercised instead. The two paths need to be tested as separate claims, not treated as equivalent because one of them works.
- **A few kilobytes of attacker-shaped input can commit gigabytes before the first unit of real work runs.** The layout driver's memory bound was never about the detections file's own size; it was about what that file's numbers instructed the process to allocate on its behalf.
- **Category prompts were measured, not assumed.** The same crop returning `#`, `##`, or an HTML table depending only on which instruction was prepended is direct evidence the routing reaches the model, rather than a plausible claim about it.
