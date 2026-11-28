# Technical Report: Issue #2122 - Accept untyped items and output_text parts in /v1/responses input

**Date**: 2026-10-06

**Status**: Implemented and validated on GB10, awaiting merge.

**Language**: Rust (Responses request types, input translator, tests), Markdown

**Risk**: Low (input is only accepted where it was previously a 422 or a 400; typed inputs parse and lower exactly as before)

## Summary

A client replaying `response.output` on turn 2 sends `{"role":"assistant","content":[{"type":"output_text","text":"Paris."}]}` with no `type` on the item. `/v1/responses` returned 422 `data did not match any variant of untagged enum ResponseInput`. Two independent defects caused it, and fixing either one alone still failed the request.

## 1. Items without `type`

`ResponseInputItem` was `#[serde(tag = "type")]`, so an EasyInputMessage matched no variant. It now deserializes through `serde_json::Value`: an object with no `type` and a `role` is read as `message`; anything else untyped keeps the missing-`type` error. The tagged parsing moved to a private mirror enum (`TaggedInputItem`) because serde cannot call the derived impl from a hand-written one on the same type.

## 2. output_text and refusal parts

These deserialized as `ResponseInputPart::Unknown` and were then rejected by `ContentPart::try_from`. `ResponseInputPart` gains `OutputText { text }` and `Refusal { refusal }`, lowered to `ContentPart::Text`. The translator rejects them on user, system and developer messages with the existing "input part type '...' is not supported by this server" error, so only an assistant turn may carry them. `ResponseInputContent::as_text`, the store size estimate and function-call-output lowering handle the new variants.

## Validation

- `responses_input_parts_tests`: untyped message parses as `Message` and `{"content":"x"}` still fails; typed assistant `output_text` and `refusal` lower to assistant text; `output_text` on user, system and developer errors.
- `reasoning_echo::tests::endpoint::responses_accepts_untyped_items_and_output_text_parts`: the exact 3-message request returns 200 and the rendered prompt contains `Paris.` as the assistant turn.
- GB10, `jamba-v0.1-4bit` (AI21-Jamba-Reasoning-3B): the same request returns 200 with a Germany/Berlin answer.
