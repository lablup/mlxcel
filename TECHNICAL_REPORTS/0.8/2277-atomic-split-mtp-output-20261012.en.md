# PR #2277: Atomic and exclusive `split-mtp` output

**Date**: 2026-10-12
**Status**: Implemented, reviewed, and verified with the original GLM-4.7-Flash checkpoint
**Risk**: Medium (checkpoint output commit semantics and a byte-compatible safetensors writer)

## Summary

PR #2277 prevents `mlxcel split-mtp` from exposing partial checkpoints or following output-file symlinks. The command now builds the complete drafter in a private sibling staging directory, creates every staged file exclusively, and publishes a new or empty output with one directory rename. A non-empty output is refused unless `--force` is present.

Forced replacement remains deliberately file-by-file because a populated directory cannot be atomically replaced portably. Before replacement, the command checks the destination again, refuses foreign safetensors files and symlinked drafter paths, and removes only a stale safetensors index plus the known drafter files it replaces.

## Problem

The previous directory writer created the destination first and wrote `model.safetensors`, `config.json`, and companion files directly into it. Serialization, JSON writing, and copying used APIs that truncate existing files and follow symlinks. An interrupted conversion could therefore leave a directory that looked like a checkpoint but mixed incomplete and old artifacts. A competing writer could also interleave files after the CLI preflight check.

The output guard recognized checkpoint-shaped directories but allowed unrelated non-empty directories without `--force`. Path spellings such as `.` and `..`, Unix non-UTF-8 names, and an ancestor symlink retarget during conversion also required explicit handling once staging was introduced.

## Design

- A uniquely named staging directory is created beside the resolved output with `create_dir`; on Unix its mode is fixed to `0700`. An RAII guard removes it after every error.
- Safetensors, config, and companion files use `create_new`. The safetensors writer streams its header and tensor payloads to the exclusive file, avoiding a second multi-gigabyte serialization buffer.
- The writer reproduces safetensors 0.7 canonical dtype/name ordering, offsets, metadata, and eight-byte header padding. A mixed-dtype, multi-name regression compares the result byte for byte with `safetensors::serialize_to_file`.
- An absent or empty output is committed with one directory rename. If another writer creates a non-empty output first, the loser fails and its staging directory is removed without changing the winner.
- With `--force`, the commit path rechecks foreign weights, all seven drafter destinations, destination types, and the stale index before renaming staged files individually.
- Path resolution preserves Unix non-UTF-8 names and normalizes unresolved components without creating them. The first resolved target is stored in the staging guard; commit rejects a changed resolution and writes only to that stored target. The source-checkpoint alias guard runs before conversion and again against the stored target.

## User-visible behavior

Any non-empty `--output` now requires `--force`, including a directory that contains only an unrelated file. The CLI help explains that foreign safetensors files and symlinked drafter files are refused even with force. Fresh and empty output directories keep the same successful result, while `.` and `..` output spellings and Unix non-UTF-8 names remain supported.

## Verification

- Focused surgery unit suite: 26 passed. Coverage includes private permissions, cleanup, a concurrent winner, forced replacement, dangling and present symlinks, ancestor-symlink retargeting, source aliases through unresolved `..`, special path spellings, non-UTF-8 names, exclusive file creation, and serializer equivalence.
- Focused CUDA-feature CLI suite: 10 passed. It covers the default non-empty refusal, force rules, foreign weights, symlink handling, source alias preflight, and help text.
- Formatting and patch whitespace checks passed. The focused Clippy result is recorded in the PR test plan.
- Issue contract checks passed for versions, kernel dtype, port dispatch, Llama compatibility, ROCm overlay, Python 3.12/3.13, and WebUI behavior.
The real-checkpoint acceptance used `zai-org/GLM-4.7-Flash` revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67`. Only the three shards containing the 212 MTP source tensors were needed. Both the baseline at `190b83c3` and the final candidate ran `split-mtp --q-bits 4`, producing 54 output tensors, 15 quantized tensors, and a 0.72 GB payload. `model.safetensors` and `config.json` matched byte for byte:

- weights SHA-256: `84018ae89ff88bf727f2552523748a21acb4e824d7f22bcf77d8a439d5f4dd61`
- config SHA-256: `7218072d09e5bb06e3eb005b2481ea0b1da08e7e7cd9ad59c69dbd132a754c24`

## Review findings

Review added three path regressions before submission. Staging names no longer require UTF-8 and correctly resolve `.` and `..`. The staging guard binds commit to its first resolved target and rejects an ancestor symlink that is retargeted during conversion. Unresolved paths such as `model/new/..` are recognized as source aliases without creating `new` or modifying any source file. The follow-up security review found no remaining issue in scope.
