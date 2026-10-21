# PR #2458: Align streamed tool-call and marker cleanup with non-streamed output

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with gpt-oss-20b, Aya Vision, GLM-4.1V-Thinking and Kimi-VL on `mlxcel-server`; PR open

## Summary

Closes #2394. The server's streaming `StreamFilter` and the non-streamed cleanup (`clean_structural_tokens`, `clean_content_markers`, `parse_tool_calls`) disagreed on the same model output in five places. This PR makes both paths agree, with a parity test harness that feeds every case whole, one character at a time, at several chunk widths and split at every boundary.

## Problem

1. Whitespace after a tool call was streamed as content even when nothing visible followed.
2. Newlines after a thinking close were trimmed non-streamed but streamed when they arrived in a later chunk.
3. Unclosed tool calls and tool markup with no tools in the request were handled differently; non-streamed `clean_content_markers` also leaked Hermes `<tool_call>` blocks.
4. Output markers (Aya Vision `<|START_RESPONSE|>` / `<|END_RESPONSE|>`, GLM `<|begin_of_box|>` / `<|end_of_box|>`) reached the client, and Kimi-VL did not stop at `<|im_assistant|>`.
5. gpt-oss Harmony output streamed the `analysis` text and channel markup into `delta.content`, and the non-streamed no-tools reply leaked them too.

## Technical decisions

- **Held whitespace.** Whitespace after a tool call is held until visible text arrives, and dropped if another call or a thinking block opens or the stream ends.
- **Post-thinking gap as a field.** Newlines after a thinking close are trimmed across `feed()` calls and reported in a new `FilterOutput::post_thinking_gap`. `routes/chat.rs` and `router_front.rs` send it after the close marker under `--reasoning-format none` / `deepseek-legacy`, so the #1470 byte-exact test still passes.
- **Hermes blocks.** `strip_hermes_tool_blocks` drops Hermes tool blocks on the non-streamed no-tools path; the module doc states the #1743 rule.
- **Marker stripping.** The four `<|...|>` markers are stripped in both `CHAT_DELIMITERS` and `clean_content_markers`. Kimi-VL appends the `<|im_assistant|>` id to `eos_token_ids`, reading `tokenizer_config.json` `added_tokens_decoder` first and `tokenizer.json` `added_tokens` second, skipping duplicates, and doing nothing when the token is absent.
- **GLM answer wrapper gated on the tokenizer.** `<answer>` / `</answer>` are ordinary text for most models, so they are stripped only when the tokenizer defines both as added tokens (`answer_wrapper_enabled`), which is true for GLM-4.1V-Thinking. The streamed path sets `StreamFilter::with_answer_wrapper`, and the non-streamed routes call `strip_answer_wrapper` on content; reasoning keeps the tags.
- **Harmony through a shared classifier.** The filter matches `<|start|>assistant ... <|channel|>...<|message|>` headers through `formats::HarmonyRoute`, now shared with `try_harmony`. `<|end|>`, `<|return|>` and `<|call|>` act only after Harmony output has been seen, so Phi-3's `<|end|>` is unaffected. `clean_structural_tokens` runs `try_harmony`, which fixes the non-streamed no-tools path.
- **Review hardening.** The wait for a Muse or Harmony header to reach `<|message|>` is capped at 256 bytes (`MAX_HEADER_HOLD`), so a literal `<|channel|>` cannot stall the stream or make every `feed()` rescan a growing buffer. Once Harmony output is seen, other families' markers no longer change state, so a `</tool_call>` inside call arguments or a `</think>` inside `analysis` cannot push the rest into content. A literal `<|channel|>` in a non-Harmony reply is kept as text at end of stream, as `try_harmony` keeps it.

## Changes

- `src/server/tool_calls/stream_filter.rs`, `parser.rs`, `formats.rs`, `mod.rs`: filter state, shared Harmony classifier, marker lists, answer-wrapper gate.
- `src/server/tool_calls/stream_parity_tests.rs` (new): parity harness.
- `src/server/routes/chat.rs`, `src/server/router_front.rs`, `src/server/in_process/chat.rs`, responses and anthropic routes: `post_thinking_gap`, answer-wrapper gate.
- `src/loading/vlm_kimi_vl.rs`: `<|im_assistant|>` eos.
- `src/tokenizer/`: answer-wrapper detection test.
- `docs/supported-models.md`.

## Verification

GB10, `--profile test-fast --features cuda`, test binaries under `gpu-lock` with `--test-threads=1`, on the final head: `server::tool_calls` 433 passed; `server::`, `tokenizer::` and `reasoning_display` together 3674 passed, 0 failed, 21 ignored. Clippy (`--lib --tests -D warnings`) and fmt are clean.

Of the 11 original parity tests, 10 fail on main; the Phi-3 `<|end|>` guard passes on both by design. The review tests fail on their parent commits, and `answer_tags_are_ordinary_text_for_other_models` fails at runtime on the previous head that stripped `<answer>` for every model.

`mlxcel-server`, temperature 0:

| Checkpoint | main | this PR |
|---|---|---|
| gpt-oss-20b-mxfp4, 4 arms (stream / non-stream, tools / no tools) | streamed analysis text and channel markup in `delta.content`; non-streamed no-tools reply leaked them too | all four agree: no tools gives content `51` with the analysis as `reasoning_content`; tools gives empty content, the same reasoning, `get_weather {"city":"Paris"}`, `finish_reason: tool_calls` |
| aya-vision-8b | `<|START_RESPONSE|>...<|END_RESPONSE|>` | plain sentence, streamed and non-streamed |
| glm-4.1v-9b-thinking-4bit | `<answer>There are <|begin_of_box|>3<|end_of_box|> shapes...` | `There are 3 shapes in the image.` on both paths |
| kimi-vl-a3b-thinking-4bit | stops (its `config.json` already lists 163586) | stops; the new eos path is covered by fixture tests |

## Follow-ups

- Under `--reasoning-format none`, gpt-oss streams analysis plus answer as content while the non-streamed reply returns only the answer. Harmony has no canonical thinking-marker pair, so its rendering under `none` needs a decision.
- `try_harmony` ignores a `to=` recipient placed in the role header (`<|start|>assistant to=functions.X<|channel|>...`); pre-existing.
- `<tool_call>..</tool_call>\nB` with no text before the call streams `\nB` but non-streamed trims to `B`; item 1's wording requires keeping the separator.
- MiniMax M3 namespace residue on the no-tools path; Harmony content split across messages is joined with `\n` non-streamed but streamed back to back; Kimi-VL `◁think▷` markers reach content on both paths.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 13 |
| Lines added | 1497 |
| Lines deleted | 116 |

| Commit | Message |
|--------|---------|
| `e517fdfb` | fix: align streamed tool-call and marker cleanup with non-streamed output |
| `3efec1fa` | fix: strip GLM-4.1V-Thinking answer wrapper on both paths |
| `f8d3a32c` | fix: keep a literal <\|channel\|> outside Harmony output when streaming |
| `e013673e` | fix(security): bound the stream header hold and isolate Harmony mode |
| `4f8eb72a` | test: cover Harmony route classifier and stray Hermes close tags |
| `176626dc` | fix: strip the <answer> wrapper only for GLM-4.1V-Thinking tokenizers |

## Related

- #1470 (byte-exact reasoning passthrough), #1743 (tool markup without tools).
