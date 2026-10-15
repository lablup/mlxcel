# PR #2319: Encode non-special added tokens on the SentencePiece path

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) with real IQuest checkpoints before merge; ERNIE-4.5-VL is tokenizer-only and the Baichuan-M1 greedy diff is not usable (see "Not verified")
**Risk**: Medium (changes token ids for any SentencePiece-path prompt that contains a non-special added token; prompts without one are unchanged)

## Summary

Issue #1758: the SentencePiece fallback (`tokenizer.model` with no `tokenizer.json`) registered only `"special": true` added tokens for encoding. Entries marked `"special": false` decoded to their own spelling but, on encode, were split into 3 to 5 ordinary pieces. IQuest-Coder's `<think>`, `<tool_call>`, `<tools>`, `<tool_response>` and their closers were affected, so a rendered tool prompt reached the model as ordinary text instead of the single reserved ids it was trained on.

1. New `src/tokenizer/sp_added_vocab.rs` holds every `added_tokens_decoder` entry and splits the input on all of them with leftmost-longest matching, then applies each matched token's `rstrip`, `lstrip` and `single_word` flags to its neighbours. This is a port of the loop in the transformers slow tokenizer's `tokenize`.
2. `special` now only governs `skip_special_tokens` on decode and what `parse_special: false` stops matching. Non-special tokens are still matched under `parse_special: false`, as llama.cpp and the tokenizers backend do.
3. With no decoder table, `added_tokens.json` supplies plain non-special tokens, the same precedence as `read_added_tokens_sorted` on the tokenizers path.

## Problem

`parse_special_tokens` in `src/tokenizer/mod.rs` split the decoder by `special`. Special entries went into the encode table; non-special ones went into `added_token_contents`, which decode consulted and encode did not. HuggingFace makes no such distinction on encode: every entry is an `AddedToken` in the added vocabulary and is matched before SentencePiece sees any text.

The impact shows only on tool-calling prompts. The IQuest chat template emits the six tool tags literally, and the plain chat path uses only special tokens. After the fix the 7B tool prompt shrinks from 193 to 181 tokens and the Loop one from 205 to 185.

### Survey

Local `~/models/mlx` (184 checkpoints) has two SentencePiece-path checkpoints, both IQuest. A remote survey of the 153 repos in `docs/model-catalog.tsv` found seven SentencePiece-path snapshots.

| Checkpoint | Non-special entries |
|------------|---------------------|
| IQuest-Coder-V1-7B-Instruct-8bit | 8 (`<think>`, `</think>`, six tool tags) |
| IQuest-Coder-V1-40B-Loop-Instruct-4bit (not catalogued) | 8, same layout |
| Baichuan-M1-14B-Instruct-4bit | 20 (`<tool_call>`, FIM and repo markers, `<C_A>`, `<B_FUNC>`, ...) |
| ERNIE-4.5-VL-28B-A3B-Thinking-4bit | 1015 (image and video markers, `<|LOC_0|>` to `<|LOC_1000|>`) |
| ERNIE-4.5-0.3B-PT-4bit, internlm3-8b-instruct-4bit, TeleChat3-36B-Thinking-4bit | 0 |

No surveyed checkpoint has an `added_tokens.json` entry missing from its decoder.

## Technical decisions

- Match the slow tokenizer, not the fast one. These checkpoints declare slow-only tokenizer classes, so the transformers slow path is the reference; the flags (`single_word`, `lstrip`, `rstrip`) are ported rather than ignored, because IQuest declares `</s>` and `<unk>` as `single_word`.
- Ignore `normalized`. The slow tokenizer reads it only under `do_lower_case`, which no SentencePiece checkpoint sets.
- Keep non-special tokens matchable under `parse_special: false`. Dropping them would make `/tokenize` and logit-bias text keys disagree with llama.cpp for user-defined tokens.
- Resolve a content declared twice to the higher id, and route `token_to_id` through the same table as encode, so the two cannot disagree.
- Read added tokens leniently (`read_sp_added_tokens` skips a malformed entry) because the SentencePiece loader never failed on one and a checkpoint that loads today must keep loading. The per-entry parser is shared with the strict tokenizers-path reader in `added_tokens.rs`.
- An `added_tokens.json` entry is always non-special. transformers' legacy loader would mark one special when it matches a declared special token; no surveyed checkpoint relies on that.

## Changes

- `src/tokenizer/sp_added_vocab.rs` (new): `SpAddedVocab` with `split`, `id_of` and `read_sp_added_tokens`.
- `src/tokenizer/mod.rs`: the `special_token_to_id` / `special_tokens_sorted` tables and `split_with_special_tokens` are replaced by `added_vocab` and one `encode_split(text, add_special_tokens, parse_special)` used by both encode entry points.
- `src/tokenizer/added_tokens.rs`: `parse_added_token_entry` extracted for sharing.
- `src/tokenizer/sp_added_vocab_tests.rs` (new), `CHANGELOG.md`, `docs/supported-models.md` (IQuest entries no longer describe the gap).

## Behavior change

IQuest declares `</s>` `single_word: true`, so `d</s>e` now encodes as text, as in transformers. A `single_word` special token glued to a word no longer fires.

## Verification

- Tokenization diff, old versus new binary, on all seven tokenizers (base prompts, each non-special token embedded mid-word and space-delimited, plain and tool-calling prompts through each chat template) against a transformers 4.57.1 reference: the new binary matches on every prompt; the old one mismatched on every prompt containing a non-special token. Prompts without one are unchanged.
- Greedy ids (`/completion`, temperature 0, `MLXCEL_SDPA_DETERMINISTIC=1`, old binary self-parity checked) on `iquest-coder-v1-7b-instruct-8bit` and `iquest-coder-v1-40b-loop-instruct-4bit`: ids identical for prompts without non-special tokens; the tool-calling prompts shrink as above and both still emit the `get_weather` call.
- Revert check: with non-special entries dropped from the encode table, 5 of the new or inverted tests fail.
- `tokenizer::` (137 pass), fmt, `cargo clippy --profile test-fast --features cuda --lib --tests -D warnings` and the contract checks pass.

## Not verified and follow-up

- ERNIE-4.5-VL was checked tokenizer-only because its weights are not on the box. The image placeholder and its markers are now single ids, which should let image expansion find the template's placeholder, but no end-to-end run exists.
- Baichuan-M1: old-binary self-parity already fails on CUDA even for unchanged prompt ids (pre-existing nondeterminism), so its greedy diff is not usable. The old path echoed a literal `<C_A>`; the new one answers.
- Checkpoints that previously relied on a non-special token splitting into pieces would change ids. The survey found none such, but a new SentencePiece checkpoint with a common-substring added token should be diffed the same way.
