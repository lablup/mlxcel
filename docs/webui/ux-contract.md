# WebUI UX, localization and layout contract

## Navigation and screen states

The WebUI uses a desktop layout with a 224–280 px sidebar, restrained top toolbar, content pane, and optional 280–360 px inspector. Navigation items are Models, Chat, Activity, and Settings. Every async surface has exactly these states: empty, loading, ready, partial, stale, error, unauthorized, and offline. Each non-ready state includes one recovery action and a stable `data-testid`.

Models: search/filter the local catalog, inspect source/support/completeness/capabilities, Load, Use in Chat, Unload, Add Model by public HuggingFace repo ID after explicit consent, and cache Delete when server-allowed. Chat: model picker, stream, Stop, copy, edit/regenerate, system prompt, request-only sampling, distinct content/reasoning/tool views, finish reason and usage. Activity: active loads/downloads/drains, bounded operation history, TTFT/decode/context/slot metrics with units and provenance. Settings: browser preferences, request parameters, loaded-model live settings, next-load profile, and restart-only server flags separated by scope.

Changing selected models never reroutes an in-flight conversation. Closing a browser tab never unloads a shared model. Other tasks such as embeddings, rerank, audio and image generation appear as capabilities and copyable API examples, not misleading chat buttons.

## String, key and confirmation contract

Every user-facing string has a stable key, English value, Korean value and primary test ID in `tests/fixtures/webui/strings.json`. Korean copy is not optional because CJK long-name and layout tests are release gates. Destructive confirmation copy must include the model display name, source kind, consequence, and the typed confirmation token when required.

Minimum destructive keys: `models.delete.confirm.title`, `models.delete.confirm.body`, `models.delete.confirm.token_label`, `models.unload.confirm.body`, `downloads.cancel.confirm.body`, `settings.clear_history.confirm.body` (rendered by Chat's Clear All dialog), plus Chat's `chat.transcript.edit.confirm.body`, `chat.list.delete.confirm.body`, `chat.privacy.replace_saved.confirm.body` and `chat.privacy.replace_import.confirm.body`. Confirmations render through `ConfirmDialog` in `webui/src/design-system/primitives.tsx`; `webui/src/i18n/drift.test.ts` rejects `window.confirm`, `window.alert`, `window.prompt` and inline `locale === 'ko'` copy branches in `webui/src` outside test files and the `i18n/` catalog directory. Error copy must distinguish stale revision, unsupported action, authorization failure, offline/reset, partial settings success, deletion refusal, and indeterminate download progress.

## Keyboard and accessibility

Global shortcuts, in the order the keyboard help dialog lists them from `globalShortcuts` in `webui/src/design-system/shell.tsx`: `⌘/Ctrl+K` opens the command palette, `⌘/Ctrl+N` starts a new conversation and opens Chat, `⌘/Ctrl+Enter` sends the message from the chat composer, `Esc` closes the open dialog or navigation drawer, `[`/`]` move primary navigation while the sidebar has focus, `?` opens keyboard help. None fires while typing in an edit field, inside an open dialog or the navigation drawer, during IME composition or with Alt; the chat composer handles `⌘/Ctrl+Enter` and `⌘/Ctrl+N` itself, outside composition.

Browsers reserve `⌘/Ctrl+N` for a new window. In a stock Chrome, Safari or Firefox window the key opens a browser window and never reaches the page; the handler still calls `preventDefault` wherever the event does arrive, and headless Chromium delivers it, which `webui/tests/browser.spec.ts` checks. "New conversation" therefore stays reachable from the command palette and the Chat page header.

Text inputs must be IME-safe and never send while composition is active. Focus is visible, restored after dialogs, and kept inside modals. Dialog, listbox, menu, table and toast roles follow WCAG 2.2 AA expectations with screen-reader summaries for streaming updates.

## Metrics, units and context labels

Use IEC bytes for file/model sizes, decimal tokens/s for throughput, milliseconds for TTFT, seconds for operation age, and tokens for context. Every metric shows `unknown`, `unsupported`, or `not yet measured` when the value is null; never show zero as a placeholder. One documented exception: an idle request slot (not processing) whose `prompt_tokens` is null reads `0 / D tokens` with an empty bar when the request context `D` is known, because the server reports null for a slot that holds no task, not for a failed measurement (`src/server/slots_state.rs`), and an empty slot holds no context. A processing slot with a null count, and any slot without a known context, still reads `unknown`. Context labels must say actual resolved context, not the theoretical family maximum. #1815 unified KV arithmetic is not a prerequisite.

## Visual and viewport contract

Use system fonts, 4 px spacing, semantic color/material/elevation/radius/motion tokens, default light/dark following the OS, explicit color-scheme override, a standard and a glass theme family, glass intensity 0–100 default 35, reduce-transparency, reduce-motion and high-contrast overrides. Glass belongs on sidebar/toolbar/popovers; tables, transcripts and forms use stable readable surfaces. Without `backdrop-filter`, use an opaque fallback. Motion is 120–180 ms and disabled under reduced motion.

Layout floor: every route holds without page-level horizontal scroll down to a 195 CSS px viewport, which is a 390 px window at 200 percent browser zoom and below the 320 CSS px that WCAG 2.2 SC 1.4.10 requires. Below 320 px the toolbar stops being sticky, keeps its title and subtitle on one truncated line each (full text in `title`), and puts its actions and the loaded-model chips on rows of their own. Single-column grid tracks use `minmax(0, 1fr)` so intrinsic widths cannot push the page wider; nothing clips overflow on `html`, `body` or the shell. `webui/tests/shell.spec.ts` pins the floor at 195×422 and, as a regression guard, 320×422, in English and Korean, over the gallery Controls, States and Data tabs, the open navigation drawer, the signed-out login and the signed-in Models, Chat, Activity and Settings routes with a loaded-model chip in the toolbar. It asserts `scrollWidth - clientWidth <= 1` on each and that the toolbar title does not overlap any toolbar control.

Minimum shared states the browser suite must exercise: Models empty, Models populated with long CJK name, model details with unsupported action, download indeterminate, load failure, Chat streaming with reasoning and tool-call panes, Activity operation history with SSE reset, Settings partial success, unauthorized, offline, and mobile off-canvas navigation. Reference viewports are 390×844, 1024×768, and 1440×900 at device scale factor 1, plus light/dark/high-contrast variants for core states. These cases assert accessibility and measured geometry rather than comparing pixels, and they use the same fixtures as route/client tests; passing against mock responses does not prove runtime acceptance.

## Shared design-system artifact (#1843)

The canonical shell, semantic tokens and shared controls are documented in [`docs/webui/design-system.md`](design-system.md). Page-specific work must compose `webui/src/design-system/` primitives, use string keys from `webui/src/i18n/catalog.ts`, and keep `tests/fixtures/webui/strings.json` synchronized. Glass intensity uses `data-glass-intensity=0..100` CSS rules so CSP can continue to forbid inline style attributes.
