# Technical Report: PR #1171 - docs: correct apc_consistent_prefix_len doc claim that matched_len is preserved

**Date**: 2026-08-16
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

The doc comment on `apc_consistent_prefix_len` (`src/server/prompt_cache/apc_lookup.rs`) claimed that "if every covered block agrees, the input `matched_len` is preserved". That was false: the function always returns `consistent_blocks * block_size`, so a non-block-aligned `matched_len` loses its tail past the last full block boundary. The unit test that appeared to pin the documented behavior could not detect the difference, because its `matched_len` of 64 was already block-aligned to a block size of 16. This PR corrects the doc, renames the misleading test, and adds a test with a non-aligned `matched_len` that genuinely distinguishes flooring from preservation.

No production behavior changed. Every changed line in `apc_lookup.rs` begins with `///`, verified mechanically, which matters because `tests/prompt_cache_e2e.rs` now derives its cache-slack assertion from exactly this flooring.

---

## 1. Problem Statement

### 1.1 Background

The defect was recorded as a follow-up in the technical report for PR #1158 (implementation of #1156). That PR fixed a deterministically failing e2e assertion by deriving its slack from `DEFAULT_APC_BLOCK_SIZE`, and in doing so had to reason carefully about the flooring. The false doc line was noticed then, flagged on the PR as a pre-existing production doc bug, and left out of scope.

### 1.2 Existing Issues

- **Issue 1**: The doc claimed `matched_len` is preserved on full agreement. The function has exactly two kinds of return, three early exits returning literal `0` and the final `consistent_blocks * block_size`. No path returns `matched_len`.
- **Issue 2**: `matching_chains_preserve_matched_len` asserted the false claim in its very name, and its input (`matched_len = 64`, `BLOCK = 16`) was already block-aligned, so it passed identically under either contract and could never catch the discrepancy.
- **Issue 3** (found during review): the first replacement wording, "never the raw input `matched_len`", was itself imprecise. With an already block-aligned `matched_len` and full agreement, the return does equal `matched_len` numerically. It is computed rather than passed through, and for a PR whose entire purpose is to kill that confusion, the distinction had to be explicit.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A future implementer "fixes" the flooring to match the doc, breaking the APC safety property and the e2e slack derivation | Medium | Low |
| A reader trusts the doc when reasoning about cache credit and miscomputes expected `cached_tokens` | Low | Medium |

---

## 2. Technical Review

### 2.1 Security

No security surface is added: the diff is doc comments plus tests, with the function body untouched. The security pass returned zero findings at every severity and confirmed the invariant that actually matters here, that a candidate is never adopted past a verified divergence. The described behavior only ever removes tokens, since the result is bounded above by `matched_len`, and the corrected text explicitly warns against turning it into a pass-through. The dense branch of `try_adopt_cached_prefix` truncates to exactly the APC-floored `matched_len` and the paged branch floors further to the pool block size, so no path adopts KV past a verified boundary.

Unbounded-work was examined as the one place a genuine finding could plausibly live and dismissed on evidence: `matched_len` is attacker-influenced only through the prompt, and it is bounded by `tokens.len()` at both selection tiers (`match_depth.min(dl.token_len)` in `lookup.rs`, and `common_prefix_len` in `select_best_by_scan`), while `cap_tokens` is additionally min'd with `request_tokens.len()` so the slice cannot panic. The work is one linear hash pass over at most `matched_len` tokens. This is pre-existing and untouched.

**Issues Found:**
| Issue | Severity | Status |
|-------|----------|--------|
| Replacement wording "never the raw input `matched_len`" imprecise for the block-aligned case | Low | Fixed (`daf4a69a`) |
| New doc paragraph ran into the pre-existing short-circuit sentence, so rustdoc merged them into one block | Low | Fixed (`daf4a69a`) |
| Two em dashes in newly added comments, against project style | Low | Fixed (`599b0931`) |

### 2.2 Performance

None. No executable production line changed.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none
- **New Dependencies**: none
- **Compatibility**: unchanged; the return value is bit-for-bit identical

### 2.4 Code Quality

- **Test Coverage**: increased by one test that pins the real contract, where previously no test could distinguish the two candidate contracts
- **Code Complexity**: unchanged
- **Technical Debt**: decreased; a false claim in a safety-relevant helper is removed, and the follow-up item recorded in PR #1158's report is closed

---

## 3. Technical Decisions

### 3.1 Correct the doc rather than change the behavior

**Rationale:** The flooring is deliberate. APC verifies the Merkle-DAG chain at block granularity, so crediting a partial block would credit tokens that were never verified. `tests/prompt_cache_e2e.rs` also derives its slack (`block size - 1`) from this exact behavior, so changing the return would break a test that was only just repaired. The doc was the thing that was wrong.

### 3.2 Choose 73 tokens with a block size of 16 for the new test

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep only the existing aligned test and fix the doc | Smallest diff | Leaves the contract unpinned, so the doc could drift back |
| Add an aligned test with a different length | Consistent with the existing style | Still cannot distinguish floored from preserved |
| **Chosen: non-aligned `matched_len = 73`, `BLOCK = 16`, expect 64** | Fails under a pass-through implementation and under a ceil-based cap | Requires reasoning about the partial trailing block |

**Rationale:** 73 is deliberately not a multiple of 16. `BlockHashChain::compute` uses `div_ceil`, so the candidate chain holds 5 hashes, the fifth covering only 9 tokens, while `coverable_blocks = 73 / 16 = 4` caps the comparison at 4 blocks so that partial block is never compared. The expected result is `4 * 16 = 64`, and a preserving implementation would return 73, so both assertions fail under the false contract.

The security pass identified a second mutation the test also catches, which is the more dangerous one: if `coverable_blocks` were computed with `div_ceil` instead of floor, the fifth candidate hash would be compared as though it were a full block and the function would return 80, crediting 16 tokens for only 9 verified ones. The current floor makes that index structurally unreachable, and this test is what pins it. The test is therefore not a tautology.

### 3.3 Rename the existing test rather than delete it

**Rationale:** `matching_chains_preserve_matched_len` asserted the false claim in its name. Its body is still a valid check of the aligned case, so it was renamed to `aligned_matching_chains_return_the_block_floored_len` with the body unchanged, and its comment now says explicitly that this case cannot distinguish the two contracts and points at the new test. No other file referenced the old name.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/server/prompt_cache/apc_lookup.rs`**
```rust
// Before
/// If every covered block agrees, the input `matched_len` is preserved.

// After
/// The return value is always a multiple of `block_size`, never a
/// pass-through of the raw input `matched_len`, even when every covered
/// block agrees: the result is `consistent_blocks * block_size`, where
/// `consistent_blocks` is capped by the blocks covered by `matched_len`
/// (`floor(matched_len / block_size)`), by the candidate chain's length,
/// and by the request's own recomputed chain length. An already
/// block-aligned `matched_len` comes back numerically equal only because
/// the arithmetic lines up, not because it was preserved. Any tokens past
/// the last agreeing block boundary are dropped, including a
/// non-block-aligned tail of `matched_len`. This flooring is deliberate and
/// should not be "fixed" to pass `matched_len` through unchanged.
```

**File: `src/server/prompt_cache/apc_lookup_tests.rs`**
```rust
#[test]
fn non_aligned_matched_len_floors_to_covered_blocks() {
    let tokens: Vec<i32> = (0..73).collect();
    let extra = empty_extra();
    let candidate_chain = BlockHashChain::compute(&tokens, BLOCK, ApcHashAlgo::Sha256, &extra);
    assert_eq!(candidate_chain.hashes.len(), 5);
    let consistent = apc_consistent_prefix_len(
        &tokens, &candidate_chain.hashes, BLOCK, ApcHashAlgo::Sha256, &extra, tokens.len(),
    );
    assert_eq!(consistent, 64);
    assert_ne!(consistent, tokens.len());
}
```

**Reason for change:** The `hashes.len() == 5` assertion documents that the partial trailing block exists, and the `assert_ne!` states the point of the test directly, that the result is not the input.

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 2 |
| Lines added | +44 |
| Lines deleted | -4 |
| Tests added | 1 (plus 1 renamed) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Documentation | 1 | False preservation claim replaced with the accurate flooring description |
| Test Correctness | 2 | Misleading test renamed, non-aligned test added |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `58517b0a` | docs | correct apc_consistent_prefix_len doc claim that matched_len is preserved |
| `599b0931` | docs | replace em dashes in the new apc_consistent_prefix_len comments |
| `daf4a69a` | docs | tighten the apc_consistent_prefix_len flooring wording |

---

## 8. Follow-up Actions

### Required
- [ ] None; all three acceptance criteria are met

### Future Improvements
- The pre-existing em dashes elsewhere in `apc_lookup.rs` and across `src/` are untouched. They are a repo-wide condition and cleaning them belongs to a dedicated pass, not to this issue.

---

## Appendix

### A. Test Results

| Command | Result |
|---------|--------|
| `cargo test --lib server::prompt_cache::apc_lookup` | 10 passed, 0 failed |
| `cargo test --lib server::prompt_cache` | 168 passed, 0 failed |
| `cargo check --lib --tests` | clean |
| `cargo clippy --lib --tests -- -D warnings` | clean |
| `cargo fmt --check` | clean |

`tests/prompt_cache_e2e.rs` was deliberately not run. It is `#[ignore]`d and needs a real model plus a live server, and it cannot be affected here: the diff was mechanically confirmed to contain no changed line in `apc_lookup.rs` that does not begin with `///`, so the function it depends on is byte-identical.

### B. Verification of the no-behavior-change claim

```
git diff origin/main...HEAD -- src/server/prompt_cache/apc_lookup.rs \
  | grep -E "^[+-]" | grep -vE "^[+-]{3}" | grep -vE "^[+-]///"
```

Empty output, before and after the two follow-up commits.

### C. References
- Issue #1160 (specification), PR #1158 and its technical report (which recorded this as a follow-up), issue #1156
- `src/server/prompt_cache/apc_lookup.rs` (the corrected doc), `src/server/prompt_cache/block_hash.rs` (`BlockHashChain::compute`, `div_ceil`), `tests/prompt_cache_e2e.rs` (the slack derivation that depends on the flooring)
- PR #1171 review and security comments
