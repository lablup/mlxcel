# Technical Report: PR #1768 - feat(vision): add the Kimi K3 MoonViT3D tower and navit image path

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: implementation, security and performance review cycle
**Status**: Pre-merge (the seven unit tests the issue names plus the family, detection, renderer, request and loader suites, clippy and fmt clean; the real tower and projector shards run against an out-of-band numpy oracle, and the preprocessor is checked against the checkpoint's own Python; the end-to-end path is a load and placeholder gate on a 4-layer truncated copy, because the full 93-layer model fits no single host, which is #1734)
**Languages**: Rust, Python (fixtures only), Markdown
**Risk Level**: Medium (a new vision family plus a new request path into an existing text backbone; the shared edits are `kimi_vl`'s rope and position-embedding helpers becoming `pub(crate)`, and `KimiK3Model` gaining an embeddings-injection forward)

---

## Executive Summary

Kimi K3's vision side is MoonViT3D: a native-resolution ViT that reads 14x14 patches, adds a 64x64 learnable position grid resampled to each image's patch grid, runs 27 pre-norm blocks with RMSNorm at `2^-7` and a fused `wqkv` whose 1536-wide qkv space is decoupled from the 1024-wide hidden state, then merges each 2x2 patch block over the temporal mean (`sd2_tpool`) and projects the result to the 7168-wide text width (`patchmergerv2`). This PR adds the tower, the merger, the projector, the navit image preprocessor, VLM detection, the loader, and the prompt placement on both the HTTP and the CLI front, on top of the #1741 text backbone and the #1743 XTML renderer.

Three things here are worth more than the port itself. The first is the oracle: `tests/fixtures/kimi_k3_vision/generate_reference.py` parses the safetensors shards itself and recomputes the tower in numpy without importing torch, transformers, mlx or the Rust code, so the `#[ignore]`d harness compares two independent transcriptions rather than a port against itself. The second is that the checkpoint ships its own preprocessing Python, and running it directly reproduces the Rust processor's geometry on every case the unit tests pin and its pixel tensor value for value, which is a stronger statement than a transcription agreeing with itself. The third is what the review changed: the tower attended through the graph SDPA that materializes the full score matrix, nothing bounded what one request's images add up to, the whole request was staged as one pixel tensor the tower then sliced apart, and the rotary tables silently promoted the entire stack to f32.

---

## 1. MoonViT3D, and where it is not MoonViT

`src/vision/encoders/kimi_vl.rs` already implements MoonViT for Kimi-VL and Kimi-VL 2.5. Reusing it wholesale would have been wrong in six places, and the module doc of `moonvit3d.rs` lists them because each is a silent-wrong-answer risk rather than a load failure:

- RMSNorm at `2^-7` (`0.0078125`) instead of LayerNorm. The epsilon is a power of two because the reference writes it as `2 ** -7`, and it is large enough that rounding it to `1e-2` would show.
- `qkv_hidden_size` (1536) decoupled from `vt_hidden_size` (1024), so `head_dim = 1536 / 12 = 128` rather than `1024 / 12`. The fused `wqkv` is `[4608, 1024]` and `wo` is `[1024, 1536]`; a port that derives the head dimension from the hidden size loads and then attends over the wrong slices.
- Bilinear position resampling with half-pixel sampling, where MoonViT is bicubic. `pos_emb_interpolation_mode` is validated against `"bilinear"`, and any other value is refused at load rather than quietly taking the bicubic path.
- A fixed sincos time table added only when `t > 1`. `time[0] = concat(0, 1)` is not zero, so adding it to a still image would shift every patch by a constant the reference never applies.
- `sd2_tpool`: the temporal mean comes before the 2x2 spatial grouping, so the merged token count does not depend on the frame count.
- The weight keys are flat under the block (`encoder.blocks.N.wqkv`, `.wo`, `.norm0`, `.norm1`, `.mlp.fc0`, `.mlp.fc1`, `encoder.final_layernorm`).

What is genuinely shared is shared. The 2D rotary table is the same 32-frequency, x/y-alternating construction applied as interleaved pairs, so `kimi_vl_rope.rs` became `pub(crate)`, gained an `angles()` split out of `cos_sin()` so the layout can be asserted directly in a test, and carries a `Used by:` line naming both encoders. `gelu_tanh` and `temporal_sinusoid` are reused the same way, and their rosters were updated too. That is the convention in `docs/code-guidelines.md`: the comment above a shared function is the discovery mechanism for what breaks when it changes.

Attention runs per image rather than through a block-diagonal mask. The issue allows either. Per image is the simpler correctness story, images cannot attend to each other by construction, and it is what makes the streaming memory shape in section 5 possible.

## 2. The navit rule, and the ceiling nobody had written down

Preprocessing is a direct port of `media_utils.py`:

```text
s      = min(1, sqrt(65536 / (max(1, w // 14) * max(1, h // 14))), 7168 / w, 7168 / h)
new_w  = min(max(1, floor(w * s)), 7168);  new_h likewise
resize bicubic; flatten alpha onto an 8 px chessboard (white 255, gray 180, white top-left)
pad right and bottom with black to multiples of 28;  normalize (x / 255 - 0.5) / 0.5
patches [gh * gw, 3, 14, 14] row-major over (gh, gw);  num_tokens = gh * gw / 4
```

The arithmetic runs in f64 because the reference runs in Python floats and `int()` truncates toward zero. Using f32 or rounding would move a boundary case by one patch, and one patch is one wrong placeholder count.

Two properties of this rule were written down nowhere and are now in the module doc. First, `patch_limit_on_one_side` never binds on the published config: it is reached only through the side scales, which cap a side at `512 * 14 = 7168` pixels, and 7168 is already a multiple of the 28-pixel padding unit, so the post-padding assertion the reference makes cannot fire. The check is kept anyway, because a checkpoint may configure the two limits differently. Second, the real ceiling is the patch budget plus padding: just under 17,000 merged tokens, about 67,000 patches, for one image shaped near 1821x7069. That number is what makes the budget in section 5 concrete, and it is also the width the attention kernel has to survive.

One deviation is documented rather than fixed: Pillow resizes palette (`P`) images with nearest-neighbour sampling before the alpha fill, while the `image` crate decodes palettes to RGB or RGBA at load, so such a file is resized bicubically here.

## 3. One prompt, two fronts

An image contributes `<|media_begin|>image {w}x{h}<|media_content|>` + `<|media_pad|>` x `gh * gw / 4` + `<|media_end|>`, with `w x h` the **original** size, before any resize. The four bracketing tokens are control ids, so the block is assembled from ids and never re-encoded from its spelling; nothing a user writes into a message can forge one.

The two fronts differ in who renders it:

- `/v1/chat/completions` renders the block inside the XTML prompt, sized from the image header before the pixels are decoded, using the navit parameters read from the checkpoint's `preprocessor_config.json`. The worker then verifies the runs it finds against the grids its processor actually produced, and refuses a disagreement rather than scattering features into the wrong positions.
- `mlxcel generate --image` hands over a prompt with no placeholders, so the blocks are spliced in front of it in image order.

`place_kimi_k3_image_tokens` is the one function that decides which happened: a prompt that already carries `<|media_pad|>` runs is verified, a prompt with none is spliced. Two images rendered as one merged run of the right total is refused, because the runs have to match per image.

The merge itself is `merge_llava` at id 163605, guarded by an explicit count check. The scatter pairs the k-th placeholder with the k-th row, so fewer placeholders than rows drops image content and more leaves placeholder embeddings in the stream, both without any shape error. `merge_media_features` counts first and returns an error naming both numbers.

## 4. Detection, the loader, and the dtype policy

`model_type: "kimi_k3"` is now the VLM when `config.json` carries `vision_config` **and** the checkpoint ships `vision_tower.*` tensors, and the text backbone otherwise. Both halves matter: a config stripped of `vision_config` cannot build a tower whatever tensors are present, and a text-only export with a leftover `vision_config` must not be routed to a loader that will fail on the missing tower. Three detection tests pin the three combinations. The prefix probe Inkling already had (index first, shard headers otherwise) was generalized into `checkpoint_has_weight_prefix` rather than copied.

The loader is `src/loading/vlm_kimi_k3.rs`. It reads through the index-filtered loader, so a layer-truncated local copy never mmap-opens the shards of the layers it drops; it splits the vision keys off **before** the text sanitize, which drops everything outside `model.` and `lm_head.`; it runs the Axis A weight-load surgery hook the way the text and common VLM routes run it; and it takes the Apple Silicon dtype decision from `config_has_quantization_metadata`, the text path's own predicate. That last one is the subtle part: the published checkpoint declares its mxfp4 experts as a compressed-tensors `quantization_config` under `text_config` and has no MLX `quantization` block, so it counts as quantized and nothing is promoted from bf16 to f16. The tower and projector therefore run in bf16 beside the bf16 attention and dense planes of the text side, exactly as the text-only route leaves them.

The review added one load-time cross-check here. `vision_config.patch_size` and `merge_kernel_size` must agree with `preprocessor_config.json`: the processor decides the placeholder count and the patch size it cuts, the tower decides the merge that turns encoder tokens into projected rows, and when the two disagree the failure used to surface much later as a count mismatch at merge time or a shape error inside the tower.

## 5. What the review changed

**The graph attention path (HIGH).** The tower called `scaled_dot_product_attention`, which materializes the `[1, heads, L, L]` score matrix. `L` here is one image's whole patch count, not a decode step: at the navit ceiling of about 67,000 patches that is hundreds of gigabytes for a single image, and a 4000x3000 photo already asks for 61,776. The fast kernel takes its place. This is the finding that matters most, because it fails on a legitimate photograph rather than on an adversarial one.

**A per-request media budget (HIGH).** The image count, the pixel dimensions and the decode allocation were each capped, and nothing bounded what those images add up to. Sixteen 4000x3000 photos ask for 247,104 media tokens and 988,416 tower tokens from one anonymous POST, and the tower's attention is quadratic in each image's patch count. `check_media_token_budget` now bounds the sum, default 32,768 merged tokens with `MLXCEL_KIMI_K3_MAX_MEDIA_TOKENS` to override, checked at the render boundary where the header dimensions are first known and again on the worker before any pixel is normalized. The default admits one worst-case image with room to spare and about 30 images of ordinary size; a 1024x768 photo costs 1,036 tokens.

**Streaming the tower (HIGH).** The request path concatenated every image into one host `Vec<f32>`, copied it into one MLX array, cast and transposed the whole batch, and then let the tower slice it back apart per image, because attention is per image anyway. At the navit ceiling that is gigabytes of avoidable peak, and an allocation failure inside an MLX call aborts the process rather than returning. `project_image_stream` preprocesses, runs and projects one image at a time, writing the channels-last `[N, 14, 14, 3]` layout the conv wants directly rather than transposing after the fact, and evaluating each image's projected rows before the next starts so its activations are released instead of accumulating in one lazy graph. Only the projected rows, which are the actual payload, survive the loop.

**f32 promotion at block 0 (MEDIUM, with a wide blast radius).** The rotary tables are built in f32, as the reference builds them, and nothing cast q and k back afterwards. The attention output was therefore f32, `wo` returned f32, the residual add promoted the stream, and every one of the 27 blocks ran its bf16 weights against f32 activations for the rest of the tower. The cast back is two lines; the harness now asserts the tower returns the dtype it was handed, so the promotion cannot come back unnoticed.

**Images on reordered tool results (HIGH).** A run of tool messages is sorted into the assistant's tool-call order, while the caller's image prompts and the worker's pixels are both in wire order. Two same-grid images on out-of-order tool results would therefore swap features while passing every count check the path has. The XTML grammar renders a tool result as text and has nowhere to put an image, so an `image_url` part on a `tool` message is refused.

**`--surgery` as a silent no-op (HIGH).** This route reads the shards directly and never reached the hook that `load_text_weights` and `finish_vlm_weights_common` run, so a `--surgery` pipeline installed on the command line did nothing at all on a Kimi K3 VLM and said nothing about it.

**A degenerate oracle fixture (MEDIUM).** The real-weights harness ran on a flat single-colour 224x224 square. All 256 patches were identical, so the patchify order, the intra-patch layout, the resize, the padding and the alpha path were all invisible to the comparison, and "the pixels match exactly" was very nearly a tautology. It now runs on `navit_probe.png`, a 303x181 crop with 230 distinct colours: neither side is a multiple of 28, so both axes pad, and no two patches are alike.

**Order of work.** Geometry is planned before any pixel work, so the placeholder placement, its verification against a pre-rendered prompt, and the budget check all decide before the tower runs. An oversized or inconsistent request now fails in microseconds instead of after 27 blocks.

**Smaller ones.** An `image_url` content part rendered with no image prompt supplied put the placeholder text into the rendered string and nothing into the ids, dropping the image from what the model reads. The router front-end attached the chat renderer without the model directory, so it sized pad runs with the published navit defaults rather than the checkpoint's own. The projector indexed `shape[0]` before checking the rank. The navit grid arithmetic ran in u32 over values a checkpoint config supplies, and now runs in u64 with an explicit error when the patch tensor would not fit an i32 shape. `media_proc_cfg` geometry is bounded the way `processors::locateanything` bounds its own, `fixed_output_tokens` is refused rather than ignored, and a zero chessboard square no longer divides by zero.

## 6. Validation

**The oracle.** `generate_reference.py` reads the bf16 shards by parsing the safetensors header itself and recomputes the patch embedding, the position term, all 27 blocks, the merge and the projector in numpy float32. It imports numpy and pillow and nothing else: no torch, no transformers, no mlx, no safetensors, and none of the Rust code. `reference.json` holds the geometry, the pixel statistics, per-stage statistics after the patch embedding and blocks 0, 13 and 26, and a strided sample of the projected rows.

**The tower on real weights.** `kimi_k3_tower_real_weights` (`#[ignore]`d) loads `vision_tower.*` and `mm_projector.*` from shards 00095 and 00096 of `models/kimi-k3-8l-mxfp4` (168 tensors, about 1.5 GB) and runs the same image twice, once with the weights cast to f32 and once in the checkpoint's own bf16. In f32 the relative mean-absolute error over the 8,624 sampled projected values is 7.3e-6, the per-token statistic 9.3e-7 and the final norm 2.0e-7, so the two transcriptions agree to float32 rounding. In bf16, which is what the loader runs, the sample sits at 3.0e-2 relative while the per-token statistic is within 2.9e-4 and the final norm within 2.0e-3. The bf16 arm is also compared against the f32 arm of the same run, at 3.0e-2: same code, same MLX, so that number isolates the execution precision and does not drift when a kernel changes, which the oracle gate does. The preprocessed pixels match the dump over all 181,104 values.

**The checkpoint's own Python.** The preprocessing half can be checked against vendor code, because the checkpoint ships `media_utils.py` and `kimi_k3_vision_processing.py` and they need only numpy and pillow. Running `navit_resize_image` directly reproduces the Rust plan on every case the unit test pins (4000x3000 to 15,444 tokens, 10000x100 capped by the side limit to a 6x512 grid, 100x100 to 16, 8000x8000 to 129x129, and the 303x181 fixture to 77), and the vendor pixel tensor matches `reference.json` value for value: same shape, same grid, identical mean absolute value, identical first sixteen values. The tower half has no vendor equivalent to run, since the checkpoint ships no modeling file and torch is not installed, which is why the numpy transcription exists.

**End to end.** `mlxcel generate -m models/kimi-k3-8l-mxfp4 --image tests/fixtures/kimi_k3_vision/navit_probe.png -p "What is in this picture?" -n 64` detects `KimiK3VLM`, loads in 5.1 s at 45.4 GB resident, splices one image block of 77 `<|media_pad|>` tokens for the (1, 14, 22) grid, and generates 64 finite tokens at 34.4 tok/s. On `mlxcel-server` at port 20342 a `/v1/chat/completions` request with a base64 `image_url` part is 179 prompt tokens against 94 for the same text alone, which is the 77 pads plus the block framing and the `image 303x181` label, and the same image attached to a `tool` message is refused at the HTTP boundary with a named error. All of this is a load, placeholder-count and finite-logits gate and nothing more: the copy runs 4 of the published 93 layers, so the text it produces is meaningless and is not a correctness result. The full model is about 1.4 TB at 4 bits and needs a pipeline over at least three 512 GB hosts, which has no stage executor yet; #1734 tracks it.

## 7. What is not verified, and what was left

The workspace gate was not run as one command on this shared machine; the affected suites were run per module.

Reported and not fixed, listed in the PR body as well:

- The tower re-runs in full on every turn of a multi-turn conversation: the Kimi K3 branch discards the `active_caches` and `image_cache_keys` the other families use for an opportunistic vision cache.
- Preprocessing is per pixel: `get_pixel` plus `push` for the patchify, and a per-pixel loop for the chessboard composite. `flatten()` and `resized_rgb()` clone whole images, and the padding path fills a whole new image to add a border at most 27 pixels wide.
- `memory_estimate.rs` has no term for the vision working set. That gap is shared across every VLM family in the tree, not something this family should fix alone.
- `decode_request_images_with_limits` warns and skips an image it cannot decode, so a request with one bad image reaches the K3 path with fewer images than the prompt was rendered for. The failure message now names that as a cause, but refusing at the decode boundary is shared cross-family behavior and belongs in its own change.
- `checkpoint_has_weight_prefix` trusts a parsed index and does not fall back to the shard headers when it finds no match, so a truncated index would downgrade a real VLM to text-only. Adding the fallback makes every text checkpoint scan every shard header at detection time, which wants a measurement first.
- Per-image work that could be memoized within one forward: the rope and position tables are rebuilt for each image, and the patch-embedding conv over `[N, 14, 14, 3]` with stride 14 is a matmul in disguise. Both want a measurement before a change, per `docs/benchmarks.md`.
- The published layout has no flag to force the text-only route, the VLM route skips LoRA and `sanitize_tied_embeddings`, and `init_pos_emb_time` is unused so `t > 4` extrapolates the time table rather than being refused. Video is out of scope for this issue: the published processor rejects it.

## 8. Learning points

**An oracle that shares code with the port is not an oracle, and an oracle over a degenerate input is barely one.** The value of `generate_reference.py` is that it re-derives the tower from the specification and the raw bytes; had it called mlx, agreement would have meant only that one implementation is self-consistent. The same reasoning applies one level down to the input. On a flat square every patch is identical, so a comparison that passes says nothing about the patch order, the intra-patch layout or the padding. The fixture has to be able to fail.

**Check the vendor code where the vendor code exists.** Half of this port has a runnable reference and half does not. Running the half that exists costs one script and turns "our transcription agrees with our transcription" into "our transcription agrees with theirs" for every geometry decision that sizes a prompt.

**A per-item cost with no per-request bound is a per-request unbounded cost.** Image count, pixel dimensions and decode allocation were each capped, and the product of the three was still six figures of tower tokens. The bound belongs where the costs are first summed, not where each one is computed.

**Two attention entry points are not interchangeable, and the difference only shows at width.** The graph SDPA and the fast kernel agree numerically and differ by whether the score matrix exists. A decode step never notices. A native-resolution vision tower over a whole image notices at the first realistic photograph.

**A helper that returns f32 promotes everything downstream of it.** The rotary tables are f32 on purpose. What was missing is the cast back, and its absence is invisible in every correctness check, because f32 activations are more accurate, not less. Asserting the dtype at the boundary is what turns it into a test failure rather than a memory bill.

**Reordering and positional payloads do not mix.** The tool-result sort exists for a good reason and predates images by two issues. Adding a second stream that indexes positionally into the same message list is what made it a correctness bug, and the safest fix was to refuse the combination rather than to keep two orders in sync.

---

## References

- Issue #1342 (this PR), epic #1331, siblings #1741 (text backbone) and #1743 (tokenizer and XTML renderer)
- #1734: the full Kimi K3 across a pipeline of hosts
- `https://huggingface.co/moonshotai/Kimi-K3/blob/main/kimi_k3_vision_processing.py`, `media_utils.py`
- `docs/supported-models.md` (Kimi K3 vision entry), `docs/code-guidelines.md` (shared-function convention), `docs/benchmarks.md` (judging a change that moves the numbers)
