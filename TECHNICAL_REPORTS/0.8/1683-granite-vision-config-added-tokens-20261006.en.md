# Technical Report: Issue #1683 - Granite Vision descriptive prompts drew a refusal

**Date**: 2026-10-06

**Status**: Implemented and validated on GB10 (CUDA), awaiting merge. Apple Silicon run not done on this host.

**Language**: Rust (tokenizer loader, Granite Vision prompt expansion, CLI summary, tests)

**Risk**: Low to medium (tokenization changes for any `tokenizer.json` checkpoint whose `tokenizer_config.json` declares extra added tokens; 3 of 172 local checkpoints carry such entries and only 2 change)

## Summary

`granite-vision-3.2-2b-4bit` answered colour questions correctly but refused descriptive prompts ("I can't see the image"). The diverging stage is tokenization of the rendered prompt. The chat template rendered the same text as mlx-vlm 0.6.17, but `<image>` (id 49155) is declared only in the checkpoint's `tokenizer_config.json` `added_tokens_decoder`, not in `tokenizer.json`. The `tokenizers` crate therefore split it into `<`, `image`, `>` (`[46, 893, 48]`). `insert_granite_vision_image_tokens` found no placeholder and fell back to splicing the 1485 image tokens after the first token, which is the `<` of `<|system|>`. The image embeddings were present (so the colour question worked), but they sat inside the broken system marker, outside the user turn, and the user turn still carried the literal `<image>` text.

## Fix

- `src/tokenizer/added_tokens.rs` (new): after `tokenizer.json` loads, register the `added_tokens_decoder` entries it lacks, as `transformers`' `PreTrainedTokenizerFast.__init__` does. An entry is skipped when its content or id is already known; it is added only when its declared id is exactly the id the crate assigns next (mirroring `AddedVocabulary::add_tokens`), registration is batched by runs of the `special` flag, and each id is verified. A gap stops reconciliation with a warning instead of failing the load.
- `InsertedGraniteVisionTokens` and the Granite Vision / Granite 4 Vision preparation summaries gain `spliced`. The CLI now says "no <image> placeholder in the prompt; spliced ..." instead of reporting an in-place expansion, which is what made this failure look healthy.

## Stage-by-stage comparison against mlx-vlm 0.6.17 (CUDA, same checkpoint and fixture)

- Rendered text: identical.
- Prompt ids: differed at the placeholder (`[46, 893, 48]` vs `49155`); after the fix all 1545 ids match for the descriptive and colour questions (also 1538 / 1539 for the other two prompts).
- Pixel values: identical layout `(2 tiles, 384, 384)`, same normalized values.
- SigLIP taps and projector: match within bf16 noise once mlx-vlm's `nn.GELU(approx="fast")` is swapped for exact GELU (projector tile mean -0.00171 vs -0.00172, std 0.1617 vs 0.1611). With its default fast GELU mlx-vlm drifts further from both.
- Feature packing and merge: mlx-vlm 0.6.17's `granite_vision` model does not implement LLaVA-Next packing. It concatenates `image_newline` broadcast to `[2, 729, D]` on axis 0 and zips 4 blocks of 729 rows against the 1485 placeholders, so its language model sees about 2976 positions. mlxcel follows HF `LlavaNext.pack_image_features` (base tile, permute, unpad, newline column, flatten), which agrees with the 1485-token count both processors produce. Generations past this stage are therefore not comparable, and the first-token logprob gap is not a parity metric.

## Validation

- `tokenizer::added_tokens::tests::config_only_added_token_encodes_to_its_declared_id` and `multimodal::granite_vision_prompt_parity_tests` were run against the loader with the reconciliation call disabled and both failed (`[0, 2, 3, 4, 5, 6, 1]` vs `[8]`; the parity test hit the spliced-fallback assertion); with it they pass. `tokenizer::` and `granite_vision` selectors: 123 passed.
- CLI and `mlxcel-server` on GB10, `--temp 0`: "What is in this image? Describe it briefly." gives "In this image we can see a red color."; "Describe this image." gives "The image provided is a solid, uniform orange color. ..."; "What do you see?" gives "orange"; the colour question gives "Orange". No refusals. Server `prompt_tokens` 1545 / 1538 / 1539 / 1545 match mlx-vlm.

## Findings not addressed here

- Text-only requests on this checkpoint lose the user message (CLI and server render `<|user|>\n<|assistant|>`): the typed `chat_template.jinja` selects only `type == 'text'` items, and a plain-string (or flattened) content renders nothing. mlx-vlm wraps text into a typed list. Separate defect.
- Granite Vision's `vision_config` omits `hidden_act`; HF's SigLIP default is `gelu_pytorch_tanh`, while mlxcel's missing-key default is exact GELU. The numeric difference is small (see above) and the default is a repo-wide policy.
