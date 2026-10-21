# PR #2471: Refresh the HashMap-order section's citations to the current tree

**Date**: 2026-10-20
**Status**: Implemented; PR open
**Risk**: Low (documentation only; one section of `docs/code-guidelines.md`)

## Summary

Closes #1706. The "HashMap Iteration Order" section of `docs/code-guidelines.md` records why this defect class has no CI gate (#1287), and the argument rests on specific evidence: four LRU eviction sites, their key types, a file count, and a handful of file:line references into tests and the Makefile. Several of those had drifted. Two of the four sites had been converted to an explicit total order (#1299) and the other two had been replaced by a `BTreeSet<LruKey>` index (#1519), the file count was presented as current when it was the size of the tree at measurement time, and three line references pointed at the wrong lines. The section now states what is true of the tree today and says how the conversions bear on the decision.

## Problem

A reader checking the section against the code found `src/server/prompt_cache/store.rs:315` and `:336` pointing at nothing relevant, `responses_store.rs:243` and `conversation_store.rs:151` pointing at files with no `min_by_key` call at all, and a claim that "all four keys are `std::time::Instant`" that stopped being true when #1299 appended the entry digest to the `prompt_cache` keys. A section whose job is to keep a decision honest loses that standing the first time a citation is wrong, which is the failure mode the section itself warns about ("the rule would lose its credibility the first time somebody checked").

## Technical decisions

- Describe the four sites by function (`evict_oldest`, `evict_oldest_snapshot`) and by what happened to them, and give the two surviving `min_by_key` calls their current line numbers, rather than only re-pointing four line numbers that would drift again.
- Keep the 1,248-file count as the size of the tree #1287 measured and say so, with a note that the count has since passed 2,000. Replacing it with today's count would claim the prototypes were re-run over today's tree, which they were not.
- Say what a re-run would report today: the two `store.rs` calls would still be flagged, because a regex cannot see that the tuple key is now total (the same limit the second candidate check hits on the #1286 sites), and the two stores would produce no flag. That turns the conversions into support for the decision: zero live bugs became zero flagged bugs plus two permanent false positives, which is the trajectory the decline predicted.
- Add #1519 to "Choosing BTreeMap instead" as the worked example of switching the container, next to #1277 as the example of keeping the `HashMap` and sorting instead.
- Compute every line number from the working tree by script (searching for the cited constant, call, or comment text) and assert the structural claims about the two stores in the same script, so no number in the diff was typed by hand.

## Changes

- `docs/code-guidelines.md`, "HashMap Iteration Order" section only: the enforcement paragraph and its first bullet reworded as above; `#1519` sentence added to "Choosing BTreeMap instead"; line references for `src/lang_bias.rs` (755 to 774), `src/distributed/registry_tests.rs` (257 to 271) and the `Makefile` range recording the withdrawn #1265 conclusions (604-616 to 713-725) updated. `sanitize.rs:388` and `request_router_tests.rs:865` were still correct.

## Verification

- A script located each cited constant, the two `min_by_key` calls keyed on `(slot.entry.last_used(), *digest.as_bytes())`, the `LruKey` definition range in `store_budget.rs`, and the Makefile block, and substituted those numbers into the text.
- The same script asserted that `responses_store.rs` and `conversation_store.rs` contain no `min_by_key` / `max_by_key` call, hold a `BTreeSet<LruKey>`, and evict through `state.lru.iter().next()`.
- `grep` for each old citation over the section returns nothing.
- No Rust source changed.

## Follow-ups

- Line references into source files drift with every edit above them. The section could cite symbols only (constant names, function names) and leave line numbers out; this PR kept the section's existing style and did not make that change.

## Statistics

| Item | Value |
|---|---|
| Files changed | 1 |
| Lines added | 5 |
| Lines deleted | 5 |
| Tests added | 0 |

## Related

- Issue #1706: docs: the code-guidelines HashMap-order section cites line numbers and counts that no longer exist.
- Issue #1287: the decision this section records.
- PR #1299: total order for the four LRU eviction `min_by_key` calls.
- PR #1519: byte-bounded response stores with the `BTreeSet<LruKey>` index.
