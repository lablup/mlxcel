# Technical Report: PR #1290 - Recording the HashMap iteration order class, and declining to gate it

## Executive Summary

Seven instances of one defect class were found and fixed in a three-day sweep. This PR writes the class down in `docs/code-guidelines.md` and records, with numbers, the decision not to enforce it with a static check. No Rust changed.

The class: a `HashMap` iteration result becomes ordered state or feeds an order-sensitive consumer. Nothing in the toolchain flags it. The types are correct, `cargo check` and clippy are clean, and the tests pass most of the time.

## 1. Problem Statement

Seven instances across five modules, none caught by review or CI: #1265 (four test fixtures), #1267 (`lang_bias.rs`), #1276 (RT-DETRv2 layout sniffing), #1277 (four registry accessors) and #1286 (three eviction paths). Two of them reached primary request paths, and one silently double-transposed conv weights into a shape-valid tensor nothing downstream could flag.

Recurrence at that rate is the argument for writing the rule down rather than relying on each reviewer rediscovering it.

## 2. Technical Decisions

### 2.1 What the guideline emphasizes

Two points get the most space, because they are the ones that cost the most to rediscover.

**`RandomState` seeds per map instance, not per process.** The common mental model, "iteration order is randomized per run", implies a fixed binary on a fixed machine sees a fixed order. It does not. Measured here: build ten maps from the same five keys inside one process, over 200 processes; 127 runs produced 10 distinct orders out of 10, 62 produced 9, 11 produced 8. Not one of the 200 had the ten maps agree. The split moves between measurement runs because the experiment is itself random, and the guideline says so rather than presenting one split as a constant.

**A sort is not automatically a fix.** `sort_by_key` and `sort_by` are stable, so ties keep input order, and when the input came from a `HashMap` that retained order is hash order. Three of the seven sorted their candidate list and were still nondeterministic. `sort_unstable_by` is not the remedy either: it substitutes a different arbitrary tie order rather than defining one. The remedy is a key that is a total order.

The guideline also records the testing rule, because every fix in the family needed it and each of its three points is a way to write a test that passes against unfixed code: rebuild the map inside each iteration and loop 32 or more times; construct an actual tie, since distinct keys already give a total order; and write equal `Instant` values in deliberately, since `Instant::now()` never collides at nanosecond resolution. PR #1281's test failed on 27 of 64 freshly built maps, which is why a single-shot version would have passed and shipped nothing.

### 2.2 Declining the static check, and the number that decided it

Two candidate checks were prototyped in the shape of `check_kernel_dtype_keys.py` and run over all 1,248 `.rs` files, against the current tree and against the reconstructed pre-fix tree of each fix.

The `min_by_key` / `max_by_key` on an unordered-map receiver rule looked strongest on its face: 4 flags, 0 false positives. Its catch rate against the seven is **0 of 7**, a number the issue never computed. None of the seven used either method. Gating on it would mean suppressing four of four findings on the day it landed, and a suppression comment does not re-arm on the change that would make the pattern real, which is the key becoming coarser than `Instant`. Those four hits are true positives for the lint and zero live defects: every key is an `Instant` stamped once per call, so the tie is unreachable today. Reporting them as four bugs would have been false and would have cost the rule its credibility on first inspection.

The `Vec` from a map view rule fails for a different and more decisive reason: it flags all three #1286 sites **after** PR #1288 fixed them. The fix keeps the shape and changes only whether the sort key is total, a distinction no regex can draw. Its 3-of-7 held only against pre-fix trees, so on any fixed tree those three become permanent false positives and it can never be a gate.

The general form is undecidable from source alone, since whether an order matters depends on a consumer that is frequently in another module. The guideline says the rule is enforced by review and by the testing rule, and to re-open the question if a future instance takes a shape either candidate would have caught.

One improvement over the issue's analysis survives: a tree-wide scan for `type X = HashMap<...>` removes the type-alias blind spot that `WeightMap` creates, and it resolves `Memo` and `DetachedMap` as well. That limit is fixable. The other three misses are not.

## 3. Change Summary

| File | Change |
| --- | --- |
| `docs/code-guidelines.md` | New `## HashMap Iteration Order` section: the rule with wrong and right examples, the per-instance measurement, order-sensitive consumers, the stable-sort subsection, the total-order remedies, the testing rule, the `BTreeMap` alternative, the seven instances, and the enforcement decision with its numbers |
| `docs/README.md` | Index entry |

## 4. Review Findings

The implementation corrected the issue that specified it, in ways worth keeping.

The issue described #1286 as open with its three instances unfixed on `main`. That was stale by roughly an hour: PR #1288 merged before the work started. It is not cosmetic, because it inverts the `Vec` rule's evaluation, turning three catches into three false positives. The stale text is a process error on the filing side: #1287 was filed to record a class while instances of that class were still being fixed, and it was not refreshed before implementation.

The `RandomState` split did not reproduce exactly (127/62/11 measured against 133/52/15 in the issue). Correctly treated as the experiment being random rather than as an error, with the invariant that the ten maps never agree holding in both runs, and the guideline states the caveat.

The reconstruction recipe in the issue, `git show <fix>^:<one file>`, understates any tree-wide rule, since both file-local type detection and the alias table need the whole tree. `git archive <fix>^ src` is the working form.

## 5. Validation

No cargo gate was run and none applies: the diff is two Markdown files with zero Rust changed, so `make verify-test-cuda` would exercise nothing this PR touches. CI's `changes` path filter reached the same conclusion and skipped every Rust job while `crate versions`, `kernel dtype keys` and `cross-repo refs` passed.

Checks that do apply, all exit 0: `make verify-fmt`; the new section's relative link to `scripts/ci/check_kernel_dtype_keys.py` resolves; `docs/README.md` indexes it; and the added lines carry no em dash and no AI attribution.

Every in-tree citation the section makes was verified rather than carried over: the four test-loop constants and their line numbers, the 27-of-64 figure in PR #1281, the withdrawal at `Makefile:604-616`, and the seven issue and PR pairs.

## 6. Related Work

- #1287: the issue this closes.
- #1265 and PR #1266, #1267 and PR #1269, #1276 and PR #1281, #1277 and PR #1284, #1286 and PR #1288: the seven instances the guideline documents.
- #1283 and PR #1285: found in the same window. A different class, an `err_expect` that reddened the lint gate for every contributor, but the same underlying gap, which is that nothing ran the check at PR time.

Left for a separate decision: the four `min_by_key` LRU eviction sites are genuinely fragile and now have no mechanical guard. They are safe only because every key is an `Instant` stamped once per call. Pinning them with a total-order tie-break on the map key is small, and `src/server/responses_store.rs` and `src/server/conversation_store.rs` are already being touched by #1248.
