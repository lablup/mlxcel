# Scoped WebUI settings

Settings uses the npmjs-published `@lablup/ui-common` adapters and the shared authenticated client. Available scopes are deliberately separate:

| Scope | Timing and storage |
|---|---|
| Appearance/privacy | Browser preferences; no server mutation. |
| Generation | Memory-only defaults for future requests; missing fields are omitted. Chat freezes each turn and owns its system prompt. |
| Loaded model | Existing opt-in `/settings` schema/current/fingerprint; partial PATCH results are not all-fields success. |
| Next load | Explicitly saved browser profile, attached to a future model-load operation. No hot mutation or automatic unload/reload. |
| Server startup | Read-only schema fields; restart required. |

## Live changes

WebUI does not enable `--settings`, `--props`, `--metrics`, or slots implicitly. A selected model must be ready before observation; all model-scoped requests include `autoload=false`. Before PATCH, a fresh fingerprint detects external changes already visible to the client and requires reconfirmation. This is not atomic compare-and-swap: another writer can change values between GET and PATCH. Applied fields clear their draft; rejected fields retain their input and error, then effective values are read again. Unknown transport outcomes are not automatically retried. Reset only stages mutable startup defaults until explicitly applied.

The current server schema supplies types, allowed enum choices, help, mutability, and read-only reasons, but no numeric min/max metadata. Server validation remains authoritative for numeric ranges. Optional raw tokenization checks exclude chat templates/history/media and never certify that a complete chat request fits.

## Next-load profiles

Only `ctx_size` (1–262144), `n_parallel` (1–32), and existing `KVCacheMode` names are accepted. Null/unset fields are omitted. Unsafe keys, arbitrary argv, paths, tokens, network configuration, and distributed topology cannot be imported or submitted. Reusable and model-specific profiles are distinct; model-specific values replace the reusable profile for that model. Version 1 documents are bounded to 64 KiB and 200 opaque model IDs; version 0 single-profile documents migrate explicitly. A saved profile is pending, not active, and survives failed loads.

Precedence is explicit startup CLI/environment pins, then the operation profile, then the preset/default resolution. Existing startup validators and model-family KV restrictions run before admission or eviction. Unsupported substitutions are rejected instead of presented as applied. Profiles affect only that operation; an unprofiled later load uses the original template. Single-model mode offers restart instructions rather than a fake Apply action. Display-only CLI flags contain only validated numbers/enums and omit model paths and secrets.

Read active `/props` after a successful reload. `n_ctx` is an effective per-slot context; zero/missing means unknown. `total_slots`, actual KV mode and geometry are separate facts, not a calculated shared-pool capacity. Unified KV work remains tracked separately in #1815.

## Validation boundaries

Component and fake-worker tests verify the request, partial-update, scope, precedence and lifecycle boundaries. They do not replace a real checkpoint reload showing changed `/props`. Native Safari/VoiceOver and native 200% zoom are deferred to the final integrated manual session. Native `Field`, textarea and confirmation Dialog are documented shared-component exceptions where the published package has no equivalent or cannot participate in the native modal top layer; no page-specific styling or CSP relaxation is introduced.
