# Model Addition Guide

This guide is the practical checklist for adding new text models and VLMs to `mlxcel`.
It points to the concrete control surfaces that must stay consistent. If this repository later adds a maintainer workflow document, keep this checklist aligned with it.

## Goals

- Keep new model additions predictable.
- Reuse existing control-plane helpers instead of adding new one-off branches.
- Add tests alongside the integration points that are easiest to regress.
- Treat `mlx-lm` / `mlx-vlm` as useful references, not as the only acceptable
  source for a port.

## Before You Start

1. Identify the implementation source you will use for the port. This can be
   an MLX reference implementation, but it does not have to be one:
   - `mlx-lm` text model implementations when the family already exists there.
   - `mlx-vlm` implementations when the family already exists there.
   - Hugging Face Transformers or an official PyTorch implementation when it is
     the clearest source of truth for config fields, tensor names, module
     layout, forward semantics, and processor behavior.
   - vLLM or SGLang implementations when production inference behavior is the
     useful reference, especially for KV-cache layout, paged-attention
     assumptions, MoE routing, rope scaling, speculative paths, or multimodal
     request preparation.
   - Vendor model repositories, model cards, conversion scripts, or released
     inference examples when they are the only public source for architecture
     quirks.
   - No complete reference implementation, when you only have a checkpoint,
     `config.json`, tokenizer/processor files, a paper, or partial vendor
     notes. This is acceptable, but it should be treated as a reconstruction
     task with tighter validation checkpoints.
   - If local `references/` checkouts are not present, clone or inspect the
     relevant upstream repositories separately. Do not vendor those repositories
     into this tree.
2. Decide whether the architecture is:
   - A brand new model family
   - A format alias of an existing family
   - A VLM wrapper around an existing text model
3. Check whether an existing loader helper already matches the new model:
   - `src/model_metadata.rs`
   - `src/loading/mod.rs`
   - `src/loading/vlm.rs`
   - `src/models/mod.rs`
   - Start with the converged registration surface in `src/model_metadata.rs`.
     Standard text models should extend that registration table first, because
     `src/loading/config_backed.rs` now consumes the same source of truth.
4. Check whether the change also touches shared execution policy:
   - `src/execution/runtime.rs` for device/environment behavior
   - `src/execution/sampling.rs` for user-facing sampling defaults and greedy-vs-sampled assembly

## Choosing and Using Reference Implementations

The goal is not to mechanically translate one Python file into Rust. The goal
is to identify the model contract that `mlxcel` must implement: config
normalization, tensor naming, graph topology, cache semantics, prompt or image
preparation, and generation behavior.

Prefer the reference that is closest to the question you are answering:

- **MLX references (`mlx-lm`, `mlx-vlm`)** are usually the fastest path when
  they already support the model, because checkpoint loading, quantization
  conventions, and MLX tensor behavior tend to match `mlxcel` closely.
- **Hugging Face Transformers or official PyTorch code** is often the
  architecture source of truth. Use it to confirm module shapes, config field
  names, activation order, normalization placement, rotary embedding behavior,
  tied embeddings, and processor/tokenizer conventions.
- **vLLM and SGLang** are useful production inference references. Use them to
  understand serving-time details such as KV-cache shape and update policy,
  paged attention assumptions, MoE expert routing, prefix caching, speculative
  decode behavior, and multimodal batching constraints. Map those ideas onto
  existing `mlxcel` cache/runtime abstractions instead of importing their
  scheduler or CUDA-specific structure directly.
- **Other PyTorch inference engines or vendor examples** can be the best source
  for model-specific quirks, especially when the model has not landed in MLX
  or Transformers yet. Capture the exact commit or release used for validation
  in the PR description or benchmark notes.

When references disagree, record which behavior is authoritative for the
checkpoint you are adding. For example, a model card may document the chat
template, Transformers may define the tensor/module contract, and vLLM may show
the serving-time cache layout. Keep those responsibilities separate while
porting.

Do not copy reference-code boundaries blindly:

- Keep route selection in `src/model_metadata.rs` and `src/loading/`.
- Keep prompt, media, and processor policy in the existing multimodal helpers.
- Keep serving behavior in the shared execution/server layers.
- Preserve `mlxcel` naming and test conventions even when the reference uses a
  different file layout.

## When There Is No Complete Reference

Some model additions start from a checkpoint and metadata rather than a working
inference implementation. In that case, make the first PR a conservative
loader/runtime reconstruction rather than a broad family port.

Use all available artifacts as partial references:

- `config.json`, `generation_config.json`, tokenizer files, processor files,
  and chat templates.
- SafeTensors key names, tensor shapes, quantization metadata, and tied-weight
  relationships.
- Model card notes, architecture diagrams, paper equations, release examples,
  and conversion scripts.
- The nearest existing family in `mlxcel`, `mlx-lm`, Transformers, vLLM,
  SGLang, or another PyTorch inference engine.

Recommended workflow:

1. Inspect the checkpoint first. Build a tensor-name and shape inventory before
   writing model code, and compare it with the nearest existing family.
2. Identify the minimum viable path: text-only before VLM, single-device before
   tensor/pipeline parallelism, greedy decode before advanced sampling behavior.
3. Add explicit config normalization for every inferred default. Do not hide
   guessed defaults inside the model constructor.
4. Keep unsupported variants out of detection until they are validated with a
   real checkpoint.
5. Add shape/config tests even before numerical parity is available.
6. Run a real smoke test and record the prompt, generated token count, and any
   known limitations in the PR or benchmark notes.

Validation expectations are different without a reference. Exact parity may not
be possible at first, but the implementation should still prove that:

- all required tensors are consumed or intentionally ignored
- tensor shapes match the reconstructed graph
- cache updates advance correctly across prefill and decode
- generation is stable for at least one real checkpoint
- failures are explicit for unsupported configs instead of silently falling
  through to a wrong family

If later a reference implementation appears, add a follow-up comparison against
that implementation and tighten the tests or benchmark notes accordingly.

## Text Model Checklist

1. Add the implementation file under `src/models/`.
2. Register the module and re-export in `src/models/mod.rs`.
3. Add a `ModelType` variant in `src/models/mod.rs`.
4. Extend `get_model_type()` in `src/models/detection.rs`.
   - Prefer shared helpers such as `detect_text_or_vlm()` and `detect_hunyuan_model_type()` when the new model fits an existing pattern.
5. Add the corresponding `LoadedModel` variant in `src/loaded_model.rs`.
   - Prefer extending the existing dispatch helpers instead of adding new repeated match tables:
     `delegate_language_model!` in `src/loaded_model.rs` and `VlmRuntimeRef`
     in `src/loaded_model_capabilities.rs`
6. Wire loading in `src/loading/mod.rs`.
   - Prefer existing helpers like `load_pair_from_dir()` and `load_owned_model_from_config!`.
   - Update `src/model_metadata.rs` so kind, adapter support, route selection,
     and standard config-backed registration stay centralized before touching
     the router.
   - If the model follows the standard text-model path, extend the shared
     registration surface in `src/model_metadata.rs` instead of adding a
     parallel entry list in `src/loading/config_backed.rs`.
7. If LoRA/adapters are supported, verify `load_model_from_weights()` in `src/loading/mod.rs`.
   - Non-standard adapter paths should extend `src/loading/special.rs` instead of growing `load_model_from_weights()` directly.
8. If the family carries a **quantized MoE expert type of its own** rather than
   using `switch_layers::SwitchLinear`, bound the declared quantization pair at
   the point the loader stores it:
   `switch_layers::validate_expert_quantization_params(prefix, group_size, bits)?`.
   The shared loader does this for you; a family-local `SwitchLinear` /
   `SwitchGLU` / `ExpertLinear` does not inherit it. See
   [Quantization Parameter Bounds](#quantization-parameter-bounds).

## VLM Checklist

1. Implement or reuse the vision encoder under `src/vision/encoders/`.
2. Implement or reuse the connector under `src/vision/connectors/`.
3. Implement or reuse the processor under `src/vision/processors/`.
4. Add the VLM `ModelType` detection in `src/models/detection.rs`.
   - If the base text family has both text-only and VLM variants, prefer `detect_text_or_vlm()`.
5. Add the loader entry in `src/loading/vlm.rs` or the matching family module under `src/loading/`.
   - Prefer shared helpers for config parsing, token defaults, and weight remapping.
   - Keep family-specific assembly grouped with its nearest peers:
     `src/loading/vlm_qwen.rs`, `src/loading/vlm_llava.rs`, `src/loading/vlm_gemma.rs`,
     `src/loading/vlm_pixtral.rs`, `src/loading/vlm_siglip.rs`, `src/loading/vlm_special.rs`.
   - Update `src/model_metadata.rs` so the router knows the family is
     multimodal and adapter loading policy remains explicit.
   - Register the directory entry point in `try_load_vlm_model_from_dir()` in `src/loading/mod.rs` so `load_model()` stays as a thin dispatcher.
6. Add or reuse the `LoadedModel` capability helpers in
   `src/loaded_model_capabilities.rs`.
   - Prefer extending `VlmRuntimeRef` or an existing multimodal helper over
     adding family-specific getters that only CLI/server use.
7. Reuse prompt helpers where possible:
   - Qwen-VL token insertion: `src/multimodal/qwen_vl.rs`
   - Generic image-token block expansion: `src/multimodal/vlm_prompt.rs`
   - Phi3V prompt tag handling: `src/multimodal/phi3v_prompt.rs`

## When to Create a New Family Module

Do not create a new file by default. Create one when the family has a distinct
control-plane identity.

Create a new loader family module when:

- config normalization is not a small variant of an existing family
- token defaults or weight-key remapping need dedicated tests
- the VLM wrapper uses a different prompt/runtime assembly path
- adding the logic inline would obscure an existing family boundary

Keep the model in an existing module when:

- it is primarily an alias or small config delta
- the same loader tests already express the policy
- the family is still recognizable after the change

If you are unsure, extend the existing family module first and split only when
the test file or router starts to lose a clear boundary.

## Quantization Parameter Bounds

`config.json` is untrusted input: it arrives with a downloaded HuggingFace
repository. A `group_size` or `bits` that no tensor layout can describe passes
every Rust-side check and then violates an undocumented MLX precondition. MLX
reconstructs a quantized matrix's unpacked width as `w.shape(-1) * 32 / bits`,
so a declared `"bits": 0` is a division by zero and anything above 32 collapses
the quotient. Because `gather_qmm`, `quantized_matmul`, `quantized_embedding`
and `dequantize` all cross the cxx bridge as `UniquePtr<MlxArray>` rather than
`Result`, the resulting C++ throw is an uncatchable `std::terminate` at the
**first forward pass**, not a load error, and `catch_unwind` does not contain it
(see [`docs/adr/0003-release-panic-unwind-with-core-thread-abort.md`](adr/0003-release-panic-unwind-with-core-thread-abort.md)).
In `mlxcel-server` that is a remote denial of service triggered by loading a
model.

The rule is therefore: **bound the pair wherever it is stored, before anything
derived from it is kept.** Not at the model's load boundary. The boundary does
not dominate, because pipeline stage executors, `*StageModel::from_filtered_weights`
entry points, VLM text wrappers and config-bridging helpers all build layers
without ever calling the family's own model constructor, and several families
build the quantized variant as a bare struct literal from another module.

What already carries the bound, so you inherit it for free:

| Loader | Covers |
|--------|--------|
| `reconcile_quantization_layout` | every `UnifiedLinear` / `UnifiedEmbedding` |
| `SwitchLinear::from_stacked_parts` | MoE experts via the shared `switch_layers` loader |
| `QuantizedMultiLinear::{new, from_weights}` | MLA `embed_q` / `unembed_out` |
| `QuantizedEmbedding::new` | hand-built embeddings that skip the map loader |
| `FusedQKVLinear::from_weights_separate_with_mode` | fused QKV projections |
| `infer_mla_quantization_params` | the MLA `kv_b_proj` decomposition in `sanitize_weights` |

What you must do yourself:

- A **family-local quantized expert type**: call
  `switch_layers::validate_expert_quantization_params(prefix, group_size, bits)?`
  in the branch that builds the quantized variant. Gate only that branch: a bf16
  expert plane carries no packing and must stay loadable at any declared pair.
- A **per-prefix quantization block** (gpt-oss style, a map of overrides rather
  than one triple): walk every entry during config validation as well, because
  a bound on the top-level defaults says nothing about an individual override,
  and vice versa.
- Any **new by-hand constructor** that stores a pair and hands it to an MLX
  quantized op: make it fallible and bound it, rather than trusting the caller.

Add a regression test that drives the bad values through your real loader rather
than through `validate_quantization_params` directly, and pair every hostile case
with a positive control so a guard cannot pass by rejecting everything quantized.
`switch_layers::insert_stacked_quantized_expert_plane` and
`switch_layers::HOSTILE_QUANT_PARAMS` are the shared test fixtures; see any
`src/models/*_tests.rs` guard test for the shape.

## Where Regressions Usually Happen

- `src/models/mod.rs`
  - Missing module export or `ModelType` variant
- `src/models/detection.rs`
  - Missing aliases in `get_model_type()`
  - Text/VLM misclassification when `vision_config` is present
- `src/loading/mod.rs`
  - Divergence between `load_model()` and `load_model_from_weights()`
  - Adding a standard config-backed model as a one-off special case instead of the shared loader helpers
  - Adding a VLM directly into `load_model()` instead of `try_load_vlm_model_from_dir()`
- `src/model_metadata.rs`
  - Forgetting to update text/VLM kind, adapter support, route policy, or the
    shared standard-text registration entry before wiring loaders
- `src/loading/config_backed.rs`
  - Bypassing the shared registration surface and adding new one-off loader logic
  - Forgetting wrapper constructors for models such as `Llama4`, `Gemma3`, or `Ministral3`
- `src/loading/nonstandard.rs`
  - Leaving directory-only loader families in `src/loading/mod.rs` instead of the non-standard registry
- `src/loading/special.rs`
  - Adding adapter/owned-weight special handling inline instead of the special-weight registry
  - Forgetting Qwen3.5 text-config normalization or owned-weight sanitization before construction

Keep `src/loading/mod.rs` focused on route selection. If a new model family adds
substantial construction logic, prefer a dedicated sibling module and call it
from the router instead of growing `load_model()` or `load_model_from_weights()`
inline.

For very large model families, extract internal helper hotspots into a focused
sibling helper module when the code changes for different reasons than the main
decoder stack. Current examples:

- `src/models/gemma3n_helpers.rs`
- `src/models/llama4_helpers.rs`
- `src/loading/vlm.rs` and sibling family modules under `src/loading/`
  - Wrong default token IDs
  - Missing top-level quantization inheritance
  - Incorrect weight-key remapping between text and vision towers
- `src/loaded_model.rs` / `src/loaded_model_capabilities.rs`
  - Missing dispatch arm for a new variant
  - Missing capability wiring in `VlmRuntimeRef`
  - Updating the all-model dispatch macro but forgetting the multimodal
    capability switchboard

## Test Expectations

Add tests in the same slice as the model/control-plane change.

- For model detection helpers:
  - Add unit tests near `src/models/detection.rs`
- For sanitization helpers:
  - Add unit tests near `src/models/sanitize.rs`
- For loader normalization or token-default logic:
  - Add tests near `src/loading/tests.rs` or the relevant `src/loading/vlm*_tests.rs`
- For shared vision merge contracts:
  - Add tests near `src/vision/merge_tests.rs`
- For prompt/token expansion logic:
  - Add tests in dedicated helper test files such as `src/multimodal/qwen_vl_tests.rs`, `src/multimodal/vlm_prompt_tests.rs`, `src/multimodal/phi3v_prompt_tests.rs`
- For runtime validation:
  - Run `scripts/run_quality_gate.sh`
  - Add at least one local smoke test when a matching model exists
  - If the slice touched MLX-heavy ignored helper tests, run them explicitly
    with `--ignored --test-threads=1`

## Shared Function Policy

When touching shared functions used by multiple model families, update the local usage comments in the shared helper files, especially under:

- `src/lib/mlxcel-core/src/layers.rs`
- `src/lib/mlxcel-core/src/utils.rs`

Those comments act as the retest list for future changes.

## Cross-Cutting Execution Helpers

Keep entry-point policy in the shared execution layer when the behavior must be
identical across CLI, server, and future frontends.

- `src/execution/runtime.rs`
  - Environment-driven device selection (`MLXCEL_DEVICE`)
  - GPU wired-memory limit setup
- `src/execution/sampling.rs`
  - Centralized `SamplingConfig` assembly from resolved request defaults
  - Shared greedy vs non-greedy branching

If a new frontend or request type needs different defaults, resolve those
defaults at the edge and keep the final conversion in `src/execution/`.

Keep CLI-only prompt formatting and terminal output behavior in
`src/commands/generate.rs` instead of moving it into shared loading or server
modules.

For server-only boot behavior, keep startup policy in `src/server/startup.rs`
instead of growing `src/server/mod.rs`:

- API key / chat-template resolution precedence
- startup-time normalization of CLI-compatible flags
- warmup behavior
- Unix-socket vs TCP binding

Keep shared server types in the focused modules as well:

- `src/server/config.rs` for request/default configuration structs
- `src/server/state.rs` for `AppState` and metrics containers
- `src/server/model_provider.rs` for the public request/response channel API
- `src/server/model_worker.rs` for the long-lived worker thread, VLM request prep, and decode state

Keep server edge adapters out of the route files once more than one endpoint
needs the same behavior:

- `src/server/chat_request.rs` for OpenAI chat message flattening and prompt fallback assembly
- `src/server/request_options.rs` for request-default merging into `ServerGenerateOptions`
- `src/server/media.rs` for `data:` / `file://` image-source parsing
- `src/server/streaming.rs` for shared SSE channel and `[DONE]` emission helpers

## Dry-Run Workflow Validation

This section validates that the current architecture actually reduces ambiguity
when adding new model support.

### Dry Run A: Standard Text Model

Assume a new text model that follows the existing config-backed loading path.

Required surfaces today:

1. `src/models/<family>.rs`
2. `src/models/mod.rs`
3. `src/models/detection.rs`
4. `src/model_metadata.rs` through the converged registration surface plus
   `static_model_descriptor()` / `model_load_policy()`
5. `src/loading/config_backed.rs` only if shared config-backed loading behavior
   itself must change
6. `src/loaded_model.rs`
7. `src/loaded_model_capabilities.rs` only if the family changes multimodal
   capability exposure
8. tests near `src/models/detection_tests.rs`, `src/models/sanitize_tests.rs`, and `src/loading/tests.rs`

What should *not* happen:

- no new one-off construction branch inside `load_model()`
- no direct CLI or server changes unless user-visible behavior changes
- no family-specific getter added to `LoadedModel` if an existing capability is enough

Why this is better than the old path:

- route selection is centralized instead of duplicated across multiple loading matches
- adapter support is declared in one policy surface
- standard text constructor registration no longer lives in a separate parallel table
- the expected edit list is short enough to review before coding starts

### Dry Run B: New VLM Family

Assume a new VLM family needs its own token defaults and weight-key remapping.

Required surfaces today:

1. text model and/or VLM wrapper under `src/models/` or `src/vision/`
2. `src/models/mod.rs`
3. `src/models/detection.rs`
4. `src/loading/vlm_<family>.rs`
5. `src/loading/vlm.rs`
6. `src/model_metadata.rs` through `static_model_descriptor()` / `model_load_policy()`
7. `src/loaded_model.rs` through enum wiring
8. `src/loaded_model_capabilities.rs` through `VlmRuntimeRef`
9. `src/multimodal/` only if prompt/runtime preparation is truly new
10. tests near `src/loading/vlm_<family>_tests.rs` and any new multimodal helper test file

What should *not* happen:

- no concrete model-type checks added to CLI or server request paths
- no family-specific loading logic added directly to `src/loading/mod.rs`
- no duplicated prompt-rewrite logic across CLI and server

Why this is better than the old path:

- the family router lives in `src/loading/vlm.rs`
- family assembly stays beside peer VLM loaders
- multimodal frontends depend on capabilities, not family names

### Review Checklist for a Real Addition

Before opening a PR for a new model or VLM family, confirm:

1. loading policy was updated through `src/model_metadata.rs`
2. `LoadedModel` wiring stayed inside `src/loaded_model.rs` and
   `src/loaded_model_capabilities.rs` rather than creating a new one-off
   family getter
3. CLI and server still depend on shared helpers rather than the concrete model type
4. unit tests cover the new policy or normalization logic
5. at least one smoke test exists when a local model is available

## Commit Strategy

Prefer small checkpoints that isolate one control-plane surface:

- model detection
- loader normalization
- prompt preparation
- runtime initialization

This keeps regressions searchable and makes future model additions easier to compare against previous slices.
