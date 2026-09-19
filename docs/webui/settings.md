# Scoped WebUI settings

Settings uses the npmjs-published `@lablup/ui-common` adapters and the shared authenticated client. It is one route with four tabs, one per scope, each addressed by a route hash:

| Tab | Hash | Scope | Timing and storage |
|---|---|---|---|
| Appearance | `#settings/appearance` | This browser | Browser preferences; no server mutation. |
| Requests | `#settings/requests` | This browser session | Memory-only defaults for future requests; missing fields are omitted. Chat freezes each turn and owns its system prompt. |
| Model | `#settings/model` | The selected model | Next-load browser profile, then the existing opt-in `/settings` live values of a Ready model; partial PATCH results are not all-fields success. |
| Server | `#settings/server` | The selected model's server | Read-only startup values (restart required), the observed context from `/props`, and the raw tokenizer probe. |

A bare `#settings`, or a section the running build does not know, opens Appearance. Switching tabs replaces the hash rather than pushing a history entry, so Back leaves Settings instead of stepping back through its tabs. Links elsewhere in the UI open the tab they mean: the Models pending-profile note opens `#settings/model`, and the Chat sampling note opens `#settings/requests`.

The route renders one `PageHeader`, whose description is the active tab's one-line summary. The contract statements below used to be inline paragraphs; in the UI each is now a help tooltip next to its section heading, and the storage statement sits on the page header.

## Storage and privacy

Appearance and explicitly saved load profiles use this browser's preferences. Request defaults stay in memory; API credentials and system prompts are never stored by Settings. Conversation history is controlled separately in Chat.

## Requests

These defaults stay in browser memory only. Blank fields are omitted and inherit server defaults. Existing turns are frozen; changing defaults never alters an in-flight request. System prompts belong to individual conversations in Chat and are not stored here.

Each of the seven request fields (`max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`) shows the value the next request would use and where it comes from, resolved in this order:

1. A next-turn override typed in Chat's "Parameters for next turn".
2. The browser-session default saved on this tab.
3. The server default: `current["default_" + field]` from the selected Ready model's `/settings`, when its schema has that entry. A server `null` (such as an unset seed) is shown as "not set".
4. Otherwise "server default, not readable": no Ready model is selected, `--settings` is off, the schema has no such entry, or the read failed.

The Chat hint uses the same resolution and wording (`resolveEffectiveParameter` and `describeEffectiveParameter` in `webui/src/features/settings/generation-defaults.ts`), so the same state reads the same in both places. Chat never reads `/settings` itself: it resolves against the server defaults Settings last read for the selected model revision, a memory-only record that every Settings read refreshes.

## Model: live changes

WebUI does not enable `--settings`, `--props`, `--metrics`, or slots implicitly. A selected model must be ready before observation; all model-scoped requests include `autoload=false`. Changes affect newly admitted requests. Before PATCH, a fresh fingerprint detects external changes already visible to the client and requires reconfirmation. This is not atomic compare-and-swap: another writer can change values between GET and PATCH. Applied fields clear their draft; rejected fields retain their input and error, then effective values are read again. Unknown transport outcomes are not automatically retried. Reset only stages mutable startup defaults until explicitly applied. An unapplied draft survives a tab switch, held in memory for up to 32 workers together with the fingerprint it was written against; when the fingerprint has moved by the time the tab returns, the conflict notice appears and Apply must be pressed again, as for a stale Apply.

The current server schema supplies types, allowed enum choices, help, mutability, and read-only reasons, but no numeric min/max metadata. Server validation remains authoritative for numeric ranges.

Live values are grouped by name, never by a fixed list (the schema may hold up to 256 entries): Sampling (`default_*` except `default_dry_*`), DRY (`default_dry_*`), Diffusion (`diffusion_*`), Template (`chat_template_kwargs`, `lang_bias_config`, and any other name containing `template` or `bias`), and Other. Each control is typed from its schema kind:

| Kind | Control |
|---|---|
| `bool` | Toggle |
| `str` with `allowed` | Select with exactly the allowed values; a current value outside them stays visible as a disabled option |
| `str` | Text field |
| `int` | Number input, step 1 |
| `float` | Number input, any step |
| `array`, `object`, `object_or_null` | JSON textarea, checked on blur ("Enter valid JSON") |
| any `_or_null` kind | The control above plus a "Leave unset" toggle that disables it and stages `null` |

Labels and help come from the UI catalog (`settings.live.<name>` and `settings.live.<name>.help`), with the raw key shown as secondary text; a name without a catalog entry shows its raw key and the server's English help. Displayed floats are the shortest decimal, at most 7 significant digits, that reads back as the same f32 (the server stores f32 and sends its f64 widening, so `0.8` arrives as `0.800000011920929`). What is sent is exactly what was typed: the draft stays text and `parseSettingInput` remains the only validator.

## Model: next-load profiles

Only `ctx_size` (1–262144), `n_parallel` (1–32), and existing `KVCacheMode` names are accepted. Null/unset fields are omitted. Unsafe keys, arbitrary argv, paths, tokens, network configuration, and distributed topology cannot be imported or submitted. Reusable and model-specific profiles are distinct; model-specific values replace the reusable profile for that model. Version 1 documents are bounded to 64 KiB and 200 opaque model IDs; version 0 single-profile documents migrate explicitly. A saved profile is pending, not active, and survives failed loads.

Explicit CLI settings take precedence over this profile; the profile overrides a preset only where CLI has not pinned a value. Effective resolved values must be read after loading. Saving is not loading: a loaded model requires an explicit unload and load from Models, and failed loads retain the pending profile.

Precedence is explicit startup CLI/environment pins, then the operation profile, then the preset/default resolution. Existing startup validators and model-family KV restrictions run before admission or eviction; accepting a browser profile does not promise a model supports every mode. Unsupported substitutions are rejected instead of presented as applied. Profiles affect only that operation; an unprofiled later load uses the original template. Draft models, adapters, distributed topology and other worker geometry are read-only in v1. Single-model mode offers restart instructions rather than a fake Apply action. The saved profile JSON and its display-only CLI flags sit behind "Show as CLI flags", with a copy button; they contain only validated numbers/enums and omit model paths and secrets. The Clipboard API exists only in a secure context, so on a plain-HTTP LAN origin the copy button reports that the flags have to be selected by hand.

## Server: observed context

Read active `/props` after a successful reload. `n_ctx` is an effective per-slot context; zero/missing means unknown. `total_slots`, actual KV mode and geometry are separate facts, not a calculated shared-pool capacity: do not multiply the per-slot value into a shared-pool capacity. Unified KV work remains tracked separately in #1815. When `/props` does not answer, the tab says so and asks for a restart with `--props`.

Optional raw tokenization checks are sent only when Check is pressed and never persisted. Raw tokens exclude chat templates, conversation history and media, so the count is a lower bound; it never certifies that a complete chat request fits, and server validation is final.

Startup values are the schema's read-only entries, listed as text rather than disabled inputs. The server's reason for each one (restart required, worker-owned, and so on) is its own wording and sits behind the "Why these values are read-only" disclosure. With `--settings` off, the Server tab shows only the `/props` facts and the disabled-endpoint banner.

## Validation boundaries

Component and fake-worker tests verify the request, partial-update, scope, precedence and lifecycle boundaries. They do not replace a real checkpoint reload showing changed `/props`. Native Safari/VoiceOver and native 200% zoom are deferred to the final integrated manual session. Native `Field`, number and JSON inputs, textarea and confirmation Dialog are documented shared-component exceptions where the published package has no equivalent or cannot participate in the native modal top layer; no page-specific styling or CSP relaxation is introduced.
