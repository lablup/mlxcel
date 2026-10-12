# Technical Report: PR #2276 - Portable A/B Output Directories

**Date**: 2026-10-12
**Status**: Ready for review
**Languages**: Bash
**Risk Level**: Low

## Executive Summary

The A/B output-equality and paged-decode gates now create their default artifact directories with templates accepted by both GNU and BSD `mktemp`. If a default or requested output directory cannot be created, each gate prints a clear error and exits 2 before running a model or writing artifacts.

## 1. Problem Statement

GNU `mktemp -d -t <prefix>` rejects a prefix without `X` placeholders. The previous scripts left `OUT_DIR` empty after that failure and continued, which could misreport the setup failure as a model error and form output paths at the filesystem root.

## 3. Technical Decisions

### Use path templates and fail before execution

Both scripts use `${TMPDIR:-/tmp}/<gate>.XXXXXX`, which works on GNU/Linux and macOS. The guards then reject an empty directory value or a failed `mkdir -p` with the scripts' inconclusive exit status, so all later artifact paths remain rooted under a validated directory.

The regression fixture uses stub binaries and HTTP tools to exercise the full shell flows without a checkpoint or server. Its fake curl consumes piped request data, preserving the production pipeline's `pipefail` behavior without an intermittent SIGPIPE from the fake jq producer.

## 7. Change Summary

| Category | Count | Summary |
|---|---:|---|
| Production scripts | 2 | Portable default directories and explicit exit-2 setup failures |
| Regression tests | 1 | Default path, invalid explicit path, direct `mktemp` failure, and no-model-invocation coverage |

## Appendix: Test Results

- `bash scripts/ci/ab_scripts_outdir_test.sh` passed, including the direct `TMPDIR` failure paths for both gates.
- Twelve concurrent copies of the focused test passed after the fixture consumed the piped request body.
- The focused test failed against the pre-fix `origin/main` scripts with eight expected assertions.
- `bash -n scripts/ab_output_equality.sh scripts/paged_decode_counter_ab.sh scripts/ci/ab_scripts_outdir_test.sh` and `git diff --check` passed.
