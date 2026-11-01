# PR #2818: Prune raw logs from benchmark data

**Date**: 2026-11-01 (generated before merge)
**Status**: In review
**Languages**: Python, Bash, Markdown
**Risk Level**: Low (no runtime code changes; removed files remain in git history)

## Executive Summary

`docs/benchmark_results` shrinks from 2,661 files and 18.5 MB to 1,974 files and 14.0 MB. Uncited raw logs are removed, three failed break-even attempts become one `.7z` each, and a new CI guard stops raw logs from accumulating again.

## 1. Problem Statement

The directory held about 30% of the repository's tracked files. `check_binary_assets.py` deliberately ignores text, so no gate limited per-run server logs, build logs and empty stderr captures. The cost was noise in checkouts and searches, not clone size: the path accounts for about 3 MiB of a 97 MiB pack.

## 3. Technical Decisions

- **No history rewrite.** About 3 MiB does not justify a force-push on a shared repository.
- **Archive server logs out, not in.** With server logs, `attempt-3.7z` is 40 KB, over the 32 KB per-file limit, and the three archives would take nearly all of the 94 KB binary headroom. Without them the total is 47 KB.
- **Keep plain-text notes for failed attempts.** Existing report citations to `attempt-2/` and `attempt-3/summary-corrected.txt` keep resolving.
- **Allowlist, not size ceilings.** Text ceilings fire on unrelated files. An allowlist makes each new log a reviewable choice, and stale entries fail.

## 5. Learning Points

Searching for file names is not enough when pruning evidence. `model-owned-storage/analyze.py` reads `server-*.log` by glob, so its 60 logs looked uncited. They were restored after the scripts were checked for readers as well as writers.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 754 before this report |
| Tracked files under the path | 2,661 to 1,974 |
| Size under the path | 18.5 MB to 14.0 MB |
| Allowlisted files | 145 |
| Binary budget headroom | 48 KB |

## 8. Follow-up Actions

- The guard does not catch test logs such as `full-cuda.log`, because their names are not distinctive.
- Five break-even diagnostic server logs (2.0 MB) remain because their READMEs cite them.
- Six data directories still have no link from any document.

## Appendix: Validation

Both guards and self-tests, `tests/test_batched_decode_benchmark_evidence.py`, all checksum lists, the archives' internal lists and the native manifest chain pass. The four `binary-after.sha256` style files already failed because they hash an uncommitted binary. `model-owned-storage/analyze.py` and break-even `test_analyzer.py` fail identically before and after in this checkout because the recorded commit and `mlx-cpp` source are missing.
