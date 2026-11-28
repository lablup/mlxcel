# Technical Report: PR #1121 - docs(core): annotate the three tracked shared functions in utils.rs

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust (comment only), Markdown
**Risk Level**: Low

---

## Executive Summary

PR #1121 closes issue #1110. `docs/code-guidelines.md` opens with the shared-function `Used by:` rule and names `create_causal_mask`, `softcap` and `repeat_kv` in `utils.rs` as the components to track. The PR annotates all three and adds a policy section for helpers with too many callers to enumerate.

The defect turned out to be worse than the issue described. The issue recorded `create_causal_mask` as carrying no annotation. It carried one, and that annotation was stale in a way that inverted its meaning: it named families that no longer call the function at all. Replacing a wrong roster is a stronger fix than adding a missing one, because a missing annotation makes a contributor go look, while a wrong one tells them not to.

---

## 1. Problem Statement

### 1.1 Background

The `Used by:` convention exists so a contributor can see the blast radius of a shared helper before changing it. `docs/code-guidelines.md` states the purpose in those terms: prevent a fix for one model from breaking others, and make it clear which models need retesting. The convention is otherwise well followed across `src/`, and a neighbour in the same file, `create_causal_mask_with_left_padding`, carries `/// Used by: BatchQuantizedKVCache, BatchTurboQuantKVCache`. So this was an omission, not a file-level exemption.

### 1.2 Existing Issues

- **`create_causal_mask` carried a stale annotation, not a missing one.** The pre-PR line read: `Used by: Llama, Qwen, Mixtral, Gemma, Cohere, Phi, OLMo, Exaone, GLM4, MiniCPM, DeepSeek, Hunyuan, StarCoder2 and other causal attention callers`. Grep at this commit shows that `mixtral.rs`, `phi.rs`, `phi3small.rs`, `starcoder2.rs`, `llama3.rs`, `gemma.rs`, `gemma2.rs`, `cohere.rs`, `glm4.rs`, `olmoe.rs` and `qwen3_moe.rs` call it **zero** times each. Those families moved to the implicit-causal fused-SDPA path, passing `mask: None` when `seq_len > 1`. The annotation was therefore not merely incomplete: it named as users the exact families that are unaffected by a change to the function, and omitted the hybrid, sliding-window, VLM and MLA decoders that actually depend on it. A contributor trusting it would have retested the wrong set.
- **`repeat_kv` and `softcap` carried nothing.** Both are short enough to enumerate literally, so no judgement call was involved and the gap was pure omission.
- **The guideline had no answer for a 40-caller helper.** The issue asked for one explicitly, so the next person with a large shared function would not re-litigate the choice between enumerating and summarizing.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Contributor changes `create_causal_mask` and retests the families the stale list named, none of which call it, while the hybrid and VLM decoders that do call it go untested | High | Medium |
| Contributor reads the guideline, opens its own named example, finds no annotation, and concludes the convention is unenforced | Medium | Medium |
| A future large helper gets an enumerated 40-name roster that is stale within a release | Medium | Medium |

---

## 2. Technical Review

### 2.1 Every List Derived by Grep at This Commit

No list was copied from the issue body or from the previous annotation. Measured at commit `f0bf3a2c`:

| Function | Issue's figure | Measured | Command |
|---|---|---|---|
| `create_causal_mask` | 46 | 44 non-test under `src/models` (46 including `diffusion_gemma/tests.rs` and `phi3small_tests.rs`), 55 across all of `src` | `grep -rln '\bcreate_causal_mask(' src --include='*.rs'` |
| `repeat_kv` | 12 | 12 under `src/models`, 14 including the DeepSeek-OCR Qwen2 vision encoder and the Qwen3-Omni MoE speech layers | `grep -rln '\brepeat_kv(' src --include='*.rs'` |
| `softcap` | 1 | 1 production caller (RecurrentGemma), plus the core unit tests | `grep -rn '\bsoftcap(' src --include='*.rs'` |

The issue's 46 for `create_causal_mask` counted two test files. The nine callers outside `src/models` are `lib.rs`, `layers.rs`, `cache.rs`, the tensor-parallel Llama runtime, the GLM4 pipeline stage executor, disaggregated handoff, and three test files.

`softcap` needs one caveat that a naive grep does not surface: `src/lib/mlxcel-xla/src/emitter/model.rs` defines its own private `fn softcap` for the XLA emitter, which is a different function in a different crate. It appears in the grep output and is not a caller of the `utils.rs` helper. The annotation counts only genuine callers.

### 2.2 Verification of the "Not Used By" Half

Each family named in a group was verified by per-file grep to have at least one call, and each family named in the "not used by" list was verified to have zero. That second check is the one that matters most here, because the "not used by" clause is the part that would have prevented the original staleness from misleading anyone.

### 2.3 Scope Containment

The `.rs` diff is provably comment-only: every added and removed line in `src/lib/mlxcel-core/src/utils.rs` matches `^\s*(///|//)`. No function body, signature, or value changed. `cargo fmt --check` is clean, which follows from `rustfmt` not reflowing comments at the project's settings.

---

## 3. Technical Decisions

### 3.1 Rule Plus Representatives Instead of a 44-Name Roster

| Option | Pros | Cons |
|---|---|---|
| List all 44 non-test callers by name | Exact today | Wrong by the next release, and long enough that nobody repairs it, which is how the original annotation got into this state |
| Name only the exceptions ("all decoders except X, Y") | Short | The exception set is itself large and shifting; it inverts the same maintenance problem |
| **Chosen: a rule for why a caller is on the list, representatives per group, an explicit "not used by", and the regenerating grep** | Survives churn because the rule holds even as membership changes; a reader can rebuild the exact set in one command | Not literally exhaustive, so a reader who needs the precise set must run the grep |

The rule is the load-bearing part: callers are the decoders that materialize an explicit prefill mask rather than leaving `mask: None` for fused SDPA to apply causality itself. That sentence stays true when a model is added or migrated. The four groups (hybrid and mixed-layer stacks, sliding-window and chunked families, VLM decoders, MLA and custom-attention decoders) each name their representatives, and the annotation closes with the grep that regenerates the exact list.

### 3.2 Keep the "Not Used By" Half

It would have been shorter to list only the users. The negative half is what makes the annotation robust against the exact failure it replaced: a contributor assuming that a shared helper still covers a family that moved off it several releases ago. Naming Llama3, Mixtral, Gemma, Gemma2, Cohere, Phi, GLM4, StarCoder2, Qwen3Moe and OLMoE as explicitly not-callers, with the reason (they pass `mask: None` for `seq_len > 1`), converts the old wrong roster into a documented correction.

### 3.3 Record the Policy in the Guideline, Not Just in the Comment

The issue asked for this directly. `docs/code-guidelines.md` gains a "When the caller list is too long to enumerate" section holding the four-part policy, with `create_causal_mask` as the worked example. The section also notes that public items take `///` rather than the `//` shown in the older format example, so the annotation survives into rustdoc. Recording it once in the guideline means the next 40-caller helper does not reopen the same choice.

### 3.4 Place the Annotations at the End of the Doc Block

All three annotations sit at the end of the existing doc block, after the `# Returns` section, matching the neighbouring `create_causal_mask_with_left_padding`. This keeps the function's own description first and the caller roster last, so a long roster does not push the actual documentation below the fold.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 2 |
| Lines added | +62 |
| Lines deleted | -2 |
| Executable lines changed | 0 |
| Functions annotated | 3 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| Correctness of an existing annotation | `src/lib/mlxcel-core/src/utils.rs` | `create_causal_mask`: stale roster naming non-callers replaced with a rule-level annotation, four caller groups with representatives, the non-`src/models` callers, an explicit "not used by", and the regenerating grep |
| New annotation | `src/lib/mlxcel-core/src/utils.rs` | `repeat_kv`: literal list of all 14 callers plus the reason most decoders never call it |
| New annotation | `src/lib/mlxcel-core/src/utils.rs` | `softcap`: names RecurrentGemma as the single production caller and records that Gemma 2 and Gemma 3 route through the fused `compiled_softcap` / `compiled_softcap_sdpa` kernels instead |
| Convention | `docs/code-guidelines.md` | New "When the caller list is too long to enumerate" section holding the policy, with `create_causal_mask` as the worked example, and a note that public items use `///` |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `f0bf3a2c` | docs | docs(core): annotate the three tracked shared functions in utils.rs |

---

## 5. Validation and Follow-up

### Passed

- `.rs` diff provably comment-only: every added and removed line matches `^\s*(///|//)`.
- `cargo fmt --check` clean.
- `python3 scripts/ci/check_cross_repo_refs.py` clean.
- Each family named in an annotation verified by per-file grep to have at least one call; each family in the "not used by" list verified to have zero.

### Correction Against the Issue

The issue's table recorded `create_causal_mask` as unannotated, with 46 model-file callers. Both figures needed correcting at implementation time: the function did carry an annotation, and 46 counted two test files against a real non-test count of 44. Neither correction changes what the PR had to do, but the first changes what the defect was. The remedy for a missing annotation is to add one; the remedy for an annotation that names non-callers is to replace it and to say in the file why those families are no longer users, which is what the "not used by" clause does.

### Follow-up Candidates

- No CI check enforces this convention, unlike the JIT kernel dtype-key rule enforced by `make verify-kernel-dtype-keys` and `scripts/ci/check_kernel_dtype_keys.py`. A checker that flags a `pub fn` in `utils.rs` or `layers.rs` with more than N cross-module callers and no `Used by:` line would keep this from recurring. The issue notes it is larger than the issue itself.
- The staleness this PR found is not specific to `create_causal_mask`. Any long-lived `Used by:` roster elsewhere in `src/` can rot the same way, and nothing has audited the rest of them.
- `docs/code-guidelines.md` also lists KVCache, Attention and Normalization in `layers.rs` as tracked shared components. This PR covered only the `utils.rs` half of that list.
