# Local chat

The Chat screen uses the existing authenticated `/v1/chat/completions?autoload=false` transport. Choose a model in the shared picker, load it explicitly in Models, then send a message. The picker is not a load action. Each turn records the opaque model ID, catalog revision, actual inference ID, and numerical request parameters before dispatch. A model change never reroutes the request already running.

## Interaction and cancellation

Enter sends outside IME composition; Shift+Enter inserts a newline. New, rename, delete, system prompt, copy, and explicit edit-and-regenerate are local conversation operations. Editing asks before discarding that response and subsequent turns. Reasoning is collapsed by default; tool calls are inspectable structured text and are never executed. A response must include a finish reason, nonempty content/reasoning/tool data, and a complete SSE stream before it becomes complete. Network errors, authentication loss, navigation and Stop do not retry generation or mark a partial answer successful.

Stop aborts the transport and retains cancelled partial output. A follow-up runtime observation is useful diagnostic context, not a per-request cancellation receipt. The isolated real-browser acceptance harness separately observes the server's request lease count rise and return to zero after Stop. Closing a page never unloads a shared model.

First-delta latency is client-observed send-to-first-content/reasoning/tool-delta time, including transport. Usage is reported by the server, never counted from words. The optional decode rate is explicitly a client estimate: `(reported completion tokens - 1) / (stream duration - first-delta latency)`. Missing usage/timing remains unknown, not zero.

## Rendering and local images

Output is untrusted. The deliberately small Markdown subset supports paragraphs, headings, lists, fenced code, inline code/emphasis and safe HTTP(S) links. It does not interpret raw HTML; Markdown images are blocked. The code highlighter is a local lazy-loaded module, not a network dependency. Incomplete streaming code remains plain text. A turn is bounded to 256,000 stored output characters, Markdown display to 65,536 characters and 512 blocks, input to 32,000 characters, and stream updates are batched at 50 ms. Historical immutable turns are memoized; copying/selecting or scrolling away from the end disables automatic scrolling.

Images require a ready provider's `vision_input` capability. Only local PNG, JPEG and non-animated WebP are accepted; no remote URL, SVG, HTML, camera, audio or filesystem path input is exposed. The browser applies additional safety ceilings of four new images per turn, 8 MiB per image, 4096 pixels per dimension and 16 million pixels. These are **browser ceilings**, not claims about server configuration. Signature/header checks precede optional browser decoding; platforms without the decoder API leave full decode validation to the server.

Bootstrap `media_limits` projects the actual resolved image-count, payload, width, height and decoder-allocation authority. Its body budget is the actual inference JSON cap, additionally limited by the router's dispatch buffer in router/model-free mode; the separate 2 MiB administrative control limit does not cap chat. All historical and new image parts are checked together against current server limits. The complete UTF-8 serialized JSON request, including base64, system prompt, transcript and envelope, must fit the lower of the server budget and the explicit 16 MiB browser request ceiling.

## Privacy and storage

History is memory-only unless the user explicitly enables origin-scoped IndexedDB. Signing out clears in-memory chat. No bearer credential is written by chat; browser history does not reuse server Responses stores. Opt-in history must be enabled again after reload before previously saved data is read. Disabling saving stops writes but does not erase old data: Clear All explicitly deletes both saved and in-memory conversations. Imports replace current history only after validation and confirmation.

History uses a strict versioned schema with 50 conversations, 1000 total turns and a 16 MiB aggregate serialized budget. Save/import/export reject unknown fields and unsupported versions, and report quota/unavailable-storage errors. Restored streaming turns become interrupted. Image bytes are excluded unless separately consented, bounded to 8 MiB total when saved, and revalidated against current image constraints before restoration. Import/load/delete operations fence late callbacks and temporarily prevent conflicting chat actions.

## Component reuse

Buttons, fields and model/conversation selects compose the merged `@lablup/ui-common@0.1.0-alpha.19` adapters. Native textareas, file inputs, details/summary, transcript sections and download anchors are product-specific semantic compositions, not independently reimplemented library controls. Styling uses the approved shared shell and tokens; no floating-panel theme, external font, CDN or package source copy is added.

## Validation boundary

Unit and contract tests are not real-model or native accessibility acceptance. `webui/tests/chat.spec.ts` exercises hosted browser fixtures at 1440 and 390 pixels, IME, untrusted output, 10,000-token-sized rendering, accessibility and storage absence. Its reported render time is fixture-to-DOM time, not inference throughput. Existing visual baselines are not silently regenerated.

Root-owned real acceptance requires an isolated secured production server, an already loaded model and no competing request producers:

```sh
MLXCEL_CHAT_REAL_URL=http://127.0.0.1:8080/webui/ \
MLXCEL_CHAT_REAL_KEY_FILE=/private/path/session-key \
MLXCEL_CHAT_REAL_MODEL_ID=mdl_opaque_id \
MLXCEL_CHAT_REAL_ISOLATED=1 \
pnpm --dir webui exec playwright test --config playwright.chat-real.config.ts
```

Repeat with a dense and hybrid/MoE checkpoint. For an actual provider-supported VLM, also set `MLXCEL_CHAT_REAL_IMAGE_FILE` to a known local PNG/JPEG/WebP fixture and inspect the captured response for sensible content. Missing configuration fails rather than skips. Manual Safari/VoiceOver and actual native 200% page zoom are deferred to the final integrated session at the maintainer's request, not recorded as passed.
