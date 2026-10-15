# PR #2324: Ship the WebUI release docs and list --webui in --help

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) with a real checkpoint before merge; reverse proxy and part of the integration matrix are not verified (see "Not verified and follow-up")
**Risk**: Low (help output, docs, NOTICE and one test tightening; no runtime serving path changes)

## Summary

Issue #1900, the last documentation child of epic #1910, makes the bundled WebUI discoverable and documented.

1. `--webui` and `--no-webui` now appear in `mlxcel-server --help` and `mlxcel serve --help` under a "WebUI Options" heading. `--ui` and `--no-ui` stay as hidden aliases. Every other b10621 UI, tools, MCP, proxy and agent argument stays hidden and still fails startup with its diagnostic.
2. New docs: `docs/webui/README.md`, `user-guide.md` and `.ko.md`, `contributing.md` and `.ko.md`, linked from `README.md`, `docs/README.md`, `docs/server-features.md` and `docs/llama-server-compat.md`.
3. `NOTICE` gains an `@lablup/ui-common` section, because its compiled code is in the bundle while its license text lived only in `third-party-licenses.txt`.

## Problem

The WebUI shipped behind a flag that `--help` did not list. A user had to know the b10621 spelling `--ui` to find it, and the llama-compat manifest recorded `--ui` as hidden. The docs described the UI only in scattered notes, and one of them still claimed mlxcel ships no UI. Nothing guarded the `hidden` claim in the manifest in the direction that matters here: an entry saying "visible" could be silently re-hidden by a later clap change.

## Technical decisions

- Make `--webui` the primary long name and `--ui` the alias, not the reverse. clap lists the primary name, and `--webui` is mlxcel's own product spelling.
- List only this pair. The rest of the b10621 group is a compatibility surface that fails startup, and advertising it in `--help` would invite users to pass arguments that cannot work.
- Set the heading per argument (`WEBUI_HELP_HEADING`) instead of `next_help_heading`, so it cannot bleed into flag groups flattened after `UiCompatArgs`.
- Check `hidden` claims in both directions in `tests/llama_compat_manifest.rs`. The old test only caught an entry claiming hidden that was visible. The new branch catches an entry claiming visible that is hidden, which is the regression this PR could otherwise suffer.
- Pin the help text with a binary-level test (`tests/webui_cli_help.rs`) as well as a unit test, so the section, the hidden remainder and both entry points are checked against built binaries.
- Capture every documented error message verbatim from the binaries rather than paraphrasing, so the troubleshooting section can be diffed against real output.
- Record the integration-matrix "Bundle budgets and deterministic assets" row as the CI `WebUI bundle` job result with digests, so the guide links to evidence instead of a claim.

## Changes

- `src/cli/ui_compat_args.rs`: `--webui`/`--no-webui` visible under `WEBUI_HELP_HEADING`, `--ui`/`--no-ui` hidden aliases, doc comments updated.
- `src/main.rs`, `src/bin/mlx_server.rs`: a short "WebUI" pointer in each `after_help`.
- `compat/llama-server/b10621/ui-tools-mcp-gcp.json`: `--ui` recorded as `hidden: false`; `--ui-config` notes no longer say mlxcel ships no UI.
- Tests: `src/cli/ui_compat_args_tests.rs`, `tests/webui_cli_help.rs` (new), `tests/llama_compat_manifest.rs`.
- Docs: `docs/webui/` (five new files plus the README), edits to `README.md`, `docs/README.md`, `docs/server-features.md`, `docs/llama-server-compat.md`, `docs/webui-integration-matrix.md`, and `NOTICE`.

## Behavior change

Help output gains a section and an `after_help` paragraph. No flag is added or removed, and parsing of `--ui`, `--no-ui` and the other b10621 arguments is unchanged. Scripts that grep `--help` for the hidden group still find nothing.

## Verification

Real checkpoint `~/models/mlx/qwen3-0.6b-4bit`, against both `mlxcel-server --webui` and `mlxcel serve --webui`, built with the `test-fast` profile and `--features cuda`.

- Both `--help` outputs list `--webui` and `--no-webui`; grepping for `--ui-config`, `--tools`, `--mcp-servers`, `--ui-mcp-proxy` and `--agent` prints nothing.
- 192 catalog entries and none loaded at start. A chat to the unloaded model returns 400. Load via `POST /ui-api/v1/model-actions` returns 202 and the model is ready in 2 s. A streaming chat finishes with `stop` and `[DONE]`. Unload returns 202.
- Every documented error message was captured verbatim from the binaries. The not-compiled message was reproduced on a `--no-default-features --features cuda,surgery` build.
- Revert check: re-hiding the pair fails all three new checks (`help_lists_only_the_webui_pair_under_its_heading`, `webui_pair_is_listed_and_the_rest_of_the_group_stays_hidden`, `manifest_option_claims_hold_on_both_server_binaries`).
- `tests/cli_help_consistency.rs` (27), `make verify-llama-compat`, clippy with `-D warnings`, fmt and the contract checks pass.
- Review found one HIGH docs error: the deletion confirmation asks for the model's `mdl_` ID, not the word `DELETE`. Fixed in both languages, along with smaller misattributions.

## Not verified and follow-up

- The reverse-proxy section was checked against `scripts/webui/verify_reverse_proxy.py`, not a real nginx.
- Some integration-matrix rows remain not-run; the list is on #1834.
- The workspace-wide test suite was left to CI.
- Out of scope, observed: a single-model server started with `&` from a non-interactive shell ignored SIGINT and needed SIGTERM, while router mode exited on SIGINT. This needs its own issue.
