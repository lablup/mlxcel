# PR #1863 WebUI design system and provider integration report

**Date**: 2026-09-13
**Status**: Provider-backed shell implemented; manual Safari/VoiceOver/style acceptance and unavailable GB10 CI remain outside local validation
**Risk Level**: Medium

## Executive summary

PR #1863 now connects the WebUI design shell to the shared #1842 provider instead of presenting static placeholders or a second authentication cache. The production routes use the provider snapshot and actions for login, logout, connection state, selected catalog identity, lifecycle labels, catalog/operation freshness and safe schema-mismatch recovery. The direct `#gallery` artifact route stays isolated for deterministic visual baselines and does not contact the local API before an explicit login.

## Change summary

The implementation wraps the app in `WebUiProvider`, routes LoginView submissions through `actions.login`, routes logout through `actions.logout`, and keeps the session key in provider/client memory only. Auth failures are reduced to localized presentation codes so raw tokens or server messages are not reflected in the DOM. Models, Chat and Activity remain honest staged routes: when signed out they show the provider-backed login surface, and when authenticated they report backend mode, build version, provider state, catalog count, operation count and snapshot sequence without loading a model or starting inference. The toolbar selected-model pill is derived only from a selected catalog entry and lifecycle state; otherwise it stays “No model selected.”

## Validation status

Final local validation at this stage:

- `pnpm --dir webui run typecheck` passed.
- `pnpm --dir webui run lint` passed.
- `pnpm --dir webui run unit` passed: 8 files, 60 Vitest tests.
- `pnpm --dir webui run browser` passed: 12 Playwright tests, including the strict Darwin gallery baselines and behavior/a11y checks.
- `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` passed: 41 WebUI contract fixtures plus DTO drift and schema strictness checks.
- `make verify-webui-bundle` passed and verified deterministic checked-in assets with bundle digest `cd7c54baf8b191aee79e80beb5ff4711ff91596adbb80fa52ea76edccf74f7dc`.

## Acceptance boundaries

The new provider/mock HTTP tests cover no initial requests before submit, Bearer-prefixed bootstrap/catalog/operations/events calls, 401 session purge, malformed bootstrap schema fail-closed recovery, wrong-key/offline localized errors, no token DOM/storage/URL reflection and no autoload or inference endpoint calls while browsing authenticated routes. Actual Safari on macOS 27, VoiceOver, native browser 200% zoom and final user style approval were not executed in this environment and remain manual follow-up gates. CUDA/GB10 validation is not claimed because the required runner is down and the agreed path is local CI plus skipping unavailable required GB10 jobs.
