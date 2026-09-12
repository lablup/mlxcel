# WebUI UX, localization and screenshot contract

## Navigation and screen states

The WebUI uses a desktop layout with a 224–280 px sidebar, restrained top toolbar, content pane, and optional 280–360 px inspector. Navigation items are Models, Chat, Activity, and Settings. Every async surface has exactly these states: empty, loading, ready, partial, stale, error, unauthorized, and offline. Each non-ready state includes one recovery action and a stable `data-testid`.

Models: search/filter the local catalog, inspect source/support/completeness/capabilities, Load, Use in Chat, Unload, Add Model by public HuggingFace repo ID after explicit consent, and cache Delete when server-allowed. Chat: model picker, stream, Stop, copy, edit/regenerate, system prompt, request-only sampling, distinct content/reasoning/tool views, finish reason and usage. Activity: active loads/downloads/drains, bounded operation history, TTFT/decode/context/slot metrics with units and provenance. Settings: browser preferences, request parameters, loaded-model live settings, next-load profile, and restart-only server flags separated by scope.

Changing selected models never reroutes an in-flight conversation. Closing a browser tab never unloads a shared model. Other tasks such as embeddings, rerank, audio and image generation appear as capabilities and copyable API examples, not misleading chat buttons.

## String, key and confirmation contract

Every user-facing string has a stable key, English value, Korean value and primary test ID in `tests/fixtures/webui/strings.json`. Korean copy is not optional because CJK long-name and layout tests are release gates. Destructive confirmation copy must include the model display name, source kind, consequence, and the typed confirmation token when required.

Minimum destructive keys: `models.delete.confirm.title`, `models.delete.confirm.body`, `models.delete.confirm.token_label`, `models.unload.confirm.body`, `downloads.cancel.confirm.body`, `settings.clear_history.confirm.body`. Error copy must distinguish stale revision, unsupported action, authorization failure, offline/reset, partial settings success, deletion refusal, and indeterminate download progress.

## Keyboard and accessibility

Global shortcuts: `⌘/Ctrl+K` focus command/search, `⌘/Ctrl+N` new chat, `⌘/Ctrl+Enter` send, `Esc` stop current transient interaction or close dialog, `[`/`]` move primary navigation when the sidebar has focus, `?` open keyboard help. Text inputs must be IME-safe and never send while composition is active. Focus is visible, restored after dialogs, and kept inside modals. Dialog, listbox, menu, table and toast roles follow WCAG 2.2 AA expectations with screen-reader summaries for streaming updates.

## Metrics, units and context labels

Use IEC bytes for file/model sizes, decimal tokens/s for throughput, milliseconds for TTFT, seconds for operation age, and tokens for context. Every metric shows `unknown`, `unsupported`, or `not yet measured` when the value is null; never show zero as a placeholder. Context labels must say actual resolved context, not the theoretical family maximum. #1815 unified KV arithmetic is not a prerequisite.

## Visual and viewport contract

Use system fonts, 4 px spacing, semantic color/material/elevation/radius/motion tokens, default light/dark following the OS, explicit theme override, glass intensity 0–100 default 35, reduce-transparency, reduce-motion and high-contrast overrides. Glass belongs on sidebar/toolbar/popovers; tables, transcripts and forms use stable readable surfaces. Without `backdrop-filter`, use an opaque fallback. Motion is 120–180 ms and disabled under reduced motion.

Minimum shared screenshots: Models empty, Models populated with long CJK name, model details with unsupported action, download indeterminate, load failure, Chat streaming with reasoning and tool-call panes, Activity operation history with SSE reset, Settings partial success, unauthorized, offline, and mobile off-canvas navigation. Reference viewports are 390×844, 1024×768, and 1440×900 at device scale factor 1, plus light/dark/high-contrast variants for core states. Screenshot tests use the same fixtures as route/client tests; mock screenshots do not prove runtime acceptance.

## Shared design-system artifact (#1843)

The canonical shell, semantic tokens and shared controls are documented in [`docs/webui/design-system.md`](design-system.md). Page-specific work must compose `webui/src/design-system/` primitives, use string keys from `webui/src/i18n/catalog.ts`, and keep `tests/fixtures/webui/strings.json` synchronized. Glass intensity uses `data-glass-intensity=0..100` CSS rules so CSP can continue to forbid inline style attributes.
