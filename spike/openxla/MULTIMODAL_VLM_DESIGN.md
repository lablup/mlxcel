# Design: the multimodal / VLM track for the OpenXLA/IREE backend (#503)

Status: Proposed (2026-07-01). Scoping design for Window E of epic #493 (OpenXLA
architecture-coverage parity). Follows ADR 0004 (the compute-backend session seam
and the StableHLO/MLIR family) and the continuous-batching design in
`STAGE2_DESIGN.md`.

Decision: DEFER the implementation to a dedicated follow-up epic, #566. This
document records the design so the follow-up work is cheap to pick up; it does not
land a reference architecture (none is required for this issue's acceptance). See
"Decision and rationale" for why.

## Goal

Accept multimodal (vision-language and audio-language) requests on the
OpenXLA/IREE backend and produce correct output. The target set (from issue #503):
Qwen2-VL / Qwen2.5-VL / Qwen3-VL, Gemma3n, Phi4MM, Molmo / Molmo2, Youtu-VL.

The OpenXLA path is text-only today at two independent points, and both must
change:

1. The serve worker rejects any request carrying images, audio, or video at admit
   time.
2. The emitted graph only accepts token ids: it gathers input embeddings from
   `params['embed']` inside the graph, so there is no way to feed vision/audio
   features into the token stream even if an encoder produced them.

A VLM's job is to run a vision (or audio) encoder, project its features into the
text hidden size, and splice those features into the LM input-embedding sequence
at placeholder-token positions. Three of those four steps (encoder, projector,
merge) have no analog in the OpenXLA path today; the fourth (the LM forward) is
the text path we already have, but only once it can start from embeddings instead
of token ids.

## What already exists (the base we build on)

- The OpenXLA text LM path: config-driven emit (`Config::from_json`,
  `emitter/config.rs`), the four graph flavors (`emit_prefill` / `emit_decode` /
  `emit_decode_batched` / `emit_decode_ragged`, `emitter/model.rs`), the shared
  attention core `AttnLayout` with its RoPE and mask hooks (#494, #495), the
  continuous-batching engine `XlaBatchEngine` (`batch.rs`), the IREE runtime
  `IreeRaggedLlama` (`iree.rs`), and the serve worker `XlaServeWorker`
  (`src/server/batch/xla_worker.rs`).
- The graph's embedding step is INTERNAL: the arg schema takes `embed`
  (`emitter/model.rs`, `Args` at `model.rs:108`; the prefill schema
  `build_prefill_arg_schema` at `model.rs:2262`) plus a `token: i32` input, and
  the graph gathers the row `embed[token]` and applies any embedding scale
  (`scale_embedding`, `model.rs:793`). There is no input-embeddings argument: the
  caller passes token ids, never a `[seq, hidden]` embedding tensor.
- The serve seam already carries multimodal bytes end to end. `ModelRequest::Generate`
  (`src/server/model_provider.rs:36`) has `images: Vec<Vec<u8>>` (`:39`),
  `audio: Vec<Vec<u8>>` (`:42`), and `videos: Vec<ResolvedVideo>` (`:53`). The
  routes decode and attach them; only the XLA worker drops them.
- The MLX engine has a large, mature VLM stack that is the reference for the
  encoder / projector / merge work and, in the near term, a candidate preprocessor
  the OpenXLA path can call host-side:
  - The seam `src/vision/mod.rs`: `VisionModule` (`mod.rs:118`, an
    `encoder: Box<dyn VisionEncoder>` + `connector: Box<dyn MultiModalConnector>` +
    `processor: Box<dyn ImageProcessor>` + `image_token_id`), its
    `get_input_embeddings` (`mod.rs:167`, embed text, encode pixels, project,
    merge at image-token positions), and `VisionLanguageModel` (`mod.rs:229`,
    wraps a text `LoadedModel` + a `VisionModule` and implements `LanguageModel`).
  - `MergeStrategy` (`mod.rs:103`): `Gemma3` (an additive 4D mask, bidirectional
    over image tokens) vs `LLaVA` (token replacement, standard causal mask). The
    merged result is an `InputEmbeddings` (`src/vision/merge.rs:38`).
  - The traits `VisionEncoder` (`encoders/mod.rs:47`), `MultiModalConnector`
    (`connectors/mod.rs:29`), `ImageProcessor` (`processors/mod.rs:38`), with
    per-family encoders for every target: `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`,
    `gemma3n`, `phi4_siglip` (Phi4MM), `molmo` / `molmo2`, `youtu_vl` (all under
    `src/vision/encoders/`), plus the shared `siglip` / `pixtral` towers.
  - The server-side wiring: `SequenceInfo.vlm_embeddings: Option<InputEmbeddings>`
    (`src/server/batch/sequence.rs:225`), `is_vlm_request` (`sequence.rs:322`),
    and the host-side computation `prepare_and_compute_vlm_embeddings`
    (`src/multimodal/vlm_runtime.rs:195`). The MLX scheduler already computes the
    merged embeddings host-side and feeds them to the text model as input
    embeddings, which is exactly the shape the OpenXLA path needs.

## The gap: four sub-problems

### 1. Embedding injection (the load-bearing emitter change)

Everything else depends on this. The text graph starts from token ids and gathers
`embed[token]` internally, so a VLM cannot substitute vision features for the
placeholder tokens. The design adds a prefill-from-embeddings graph entry:

- A new emit flavor (e.g. `emit_prefill_embeds`) whose arg schema takes an input
  `hidden: [Lp, hidden] f32` (or the device precision) instead of `token: [Lp]
  i32`, and SKIPS the `embed` gather and the embedding scale (the caller applies
  any scale when it builds the merged tensor, matching how the MLX `merge` keeps
  outputs in the text-model dtype). Everything after the embedding step (norms,
  attention, FFN, final projection) is byte-identical to the existing prefill.
- The host builds the merged embedding sequence: it embeds the text tokens (a
  gather it can do itself from the `embed` weight, which the loader already holds
  as an f32 buffer, `iree.rs` `load_weights`), runs the encoder + projector to get
  the vision features, and writes those features over the placeholder-token
  positions (LLaVA token-replacement) or prepares the additive mask (Gemma3).
- Correctness anchor: a text-only prompt run through the embeddings path (host
  gather then prefill-from-embeddings) must be token-exact with the same prompt
  run through the token-id path. That equivalence is the first test and it needs no
  vision at all.

This is analogous to how the SSM track adds a new state resource: a new graph
entry plus a new host-provided input, reusing the rest of the pipeline. It is
smaller than the SSM scan problem (no new cross-token op), but it is a genuine
emitter + engine + shim change, not a per-family config tweak.

### 2. Encoder execution

Two options, sequenced:

- Host-encode first (unblock). Reuse the MLX vision stack (`VisionModule` /
  `prepare_and_compute_vlm_embeddings`) as a host-side preprocessor that produces
  the merged `InputEmbeddings`, and feed those to the prefill-from-embeddings path.
  This gets a working VLM on the OpenXLA text path quickly and keeps the encoder
  out of the graph. It couples the OpenXLA VLM to the MLX vision code and does not
  run the encoder on XLA/IREE, so it is a stepping stone, not the parity goal.
- On-XLA encoder (parity goal). Emit the vision tower as its own StableHLO graph.
  A ViT / SigLIP encoder is largely the shared attention core already in the
  emitter (patch embedding is a conv or a linear over patches, then transformer
  blocks) plus 2D or rotary position and, for Qwen2-VL, windowed attention. This
  reuses `AttnLayout` and the RoPE/mask hooks; the new pieces are patch embedding,
  2D position, and the vision-specific windowing. Audio towers (Conformer-style for
  Phi4MM / Gemma3n) are a separate encoder graph on the same seam.

The follow-up epic defines an encoder-execution seam so the two options share the
same downstream (projector + merge + prefill-from-embeddings), and families can
move from host-encode to on-XLA without touching the LM path.

### 3. Projector handling

The connector maps encoder features to the text hidden size: a linear, an MLP, an
average-pool or pixel-shuffle downsample (the MLX `connectors/`: `linear`, `mlp`,
`avg_pool`, `aya_vision`, `mistral3`, `identity`). Small either way: a host-side
matmul when host-encoding, or a few ops appended to the encoder graph when
on-XLA. Not a bottleneck; it rides whichever choice the encoder makes.

### 4. Serve-path changes

`XlaServeWorker::admit` (`src/server/batch/xla_worker.rs:128`) rejects multimodal
at `xla_worker.rs:139` ("the OpenXLA backend is text-only"). The design replaces
that rejection with a multimodal admit path:

- Decode the images (reuse `decode_request_images`), run the processor + encoder +
  projector (host or graph), and build the merged embedding sequence, expanding the
  prompt's placeholder tokens to the per-image token count first (the same
  accounting the MLX `vlm_prompt` path does).
- Submit a prefill-from-embeddings request to the engine. `XlaBatchEngine::submit`
  (`batch.rs:272`) and `IreeRaggedLlama::prefill_slot_logits` (`iree.rs:711`) take
  token ids today; they gain an embeddings-seeded variant, and the C shim's
  prefill ABI (`xla_llama_prefill_slot_logits` in `csrc/xla_iree.c`) gains an
  embeddings input (or the host uploads the embeddings as a device buffer the
  prefill graph reads). Decode is unchanged: once a slot is seeded, generation
  continues token-by-token exactly as text, so the whole continuous-batching engine
  and the KV cache carry over untouched.
- Audio and video ride the same path: the audio encoder produces features that
  merge into the same embedding sequence.

## Positioning and attention subtleties (per family)

- M-RoPE (Qwen2-VL / 2.5-VL / 3-VL): a multimodal 3D rotary position where image
  tokens carry 2D grid positions. This is a RoPE-table + position-input variant
  that reuses the `pick_rope` / `apply_rope` hooks (`emitter/model.rs`) rather than
  a new primitive, but it needs a per-row position input richer than the current
  scalar `pos`.
- Gemma3 bidirectional image attention (`MergeStrategy::Gemma3`): image tokens
  attend to each other bidirectionally, which is an additive attention-mask
  variant. The emitter already threads a per-layer mask (`add_mask`, the
  sliding-window mask from #495); the bidirectional image mask is another mask
  variant on that hook.
- Placeholder-token accounting: the prompt string carries N placeholder tokens per
  image that must match the encoder's output token count; a mismatch is a silent
  corruption, so it is validated at admit.

## Validation plan

Reuse the epic's gate, adapted for multimodal:

1. HF fp32 VLM oracle: run the target model in transformers on an image + prompt,
   capture the temp-0 continuation.
2. XLA on the bf16 checkpoint (dequant offline if only a quantized checkpoint
   exists, per the epic's validation pattern).
3. Single-sequence token-exact vs the oracle on the same image + prompt
   (`MLXCEL_BACKEND=xla`).
4. Serve reference-exact via the continuous-batching harness.
5. The text-only equivalence anchor from sub-problem 1 (embeddings path ==
   token-id path for a text prompt) runs first and independently, so the
   prefill-from-embeddings change is validated before any encoder is involved.

## Reference architecture

Not required for this issue's acceptance (design-doc-merged OR recorded deferral).
If one is attempted in the follow-up epic, a LLaVA-style token-replacement VLM with
a SigLIP tower and an MLP connector is the simplest injection path (standard causal
mask, no M-RoPE), and Qwen2-VL is the most representative target (M-RoPE +
windowing exercise the hardest emitter hooks). Even the simplest choice pulls in
the whole foundation (prefill-from-embeddings, encoder execution, projector,
serve-path admit), which is why a reference is not in scope here.

## Risks and open questions

- Encoder-on-XLA scope: a ViT is close to the existing attention core, but patch
  embedding, 2D / M-RoPE position, and Qwen2-VL windowing are real work and their
  IREE-CUDA / Metal lowering must be validated (spike-first, as Stage 2a did for
  the ragged KV write). Host-encode-first avoids this risk for the first cut.
- Dynamic image token counts vs the static prefill bucket: the number of vision
  tokens varies with image resolution, but the prefill graph is bucketed
  (`PREFILL_LP = 256`, `iree.rs:75`). Large or multiple images may exceed the
  bucket; a larger-bucket or chunked-prefill graph may be needed (the same bucket
  limit text prompts already have).
- Coupling to the MLX vision stack in the host-encode path: acceptable as a
  stepping stone, but the on-XLA encoder is what makes the OpenXLA VLM independent.
- Precision: the encoder and merge run in reduced precision on GPU
  (`resolve_precision`); expect the epic's sub-0.01-logit near-tie tolerance, more
  so through a deep vision tower.
- Audio modality (Phi4MM, Gemma3n) adds an audio encoder and its own preprocessing;
  it reuses the embedding-injection path but is a separate encoder effort.

## Staged plan (for the follow-up epic #566)

- Foundation: the prefill-from-embeddings graph entry, the engine + shim
  embeddings-seeded prefill, and the text-only equivalence test.
- Serve-path admit: replace the multimodal rejection with decode + host-encode +
  merge + embeddings submit; placeholder-token accounting.
- Encoder execution: the host-encode seam (reusing the MLX vision stack), then the
  on-XLA ViT / SigLIP encoder graph + projector.
- Reference: one VLM end to end (CLI token-exact + serve reference-exact).
- Breadth: Qwen2-VL / 2.5-VL / 3-VL (M-RoPE, windowing), Gemma3-style
  bidirectional image mask, Molmo / Molmo2, Youtu-VL.
- Modality: Phi4MM and Gemma3n audio (audio encoder + injection).

## Decision and rationale

Deferred to the follow-up epic. Accepting a single multimodal request requires,
before any token can be produced: a new prefill-from-embeddings graph entry, an
engine + C-ABI change to seed a slot from embeddings, a serve-path multimodal
admit that decodes and merges, and an encoder execution path (host-reuse or an
on-XLA vision graph). That is a foundation-sized effort spanning the emitter, the
runtime shim, the engine, and the serve worker, not the cheap per-family dense
work. Acceptance for issue #503 explicitly allows a design doc merged or "an
explicit, recorded decision to defer with the follow-up epic created", and that is
the sanctioned path here. This document is that record; the follow-up epic #566
carries the staged plan above.
