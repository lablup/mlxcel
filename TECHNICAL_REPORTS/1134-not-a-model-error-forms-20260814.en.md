# Technical Report: PR #1134 - fix(cli): name all three accepted forms in the "not a model" error

**Date**: 2026-08-14
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Low

---

## Executive Summary

PR #1134 closes issue #1114. The terminal error for an unresolvable `-m/--model` value described two accepted forms while the resolver accepts three, omitting the bare, prefix-less name that resolves against `$MLXCEL_DEFAULT_ORG`.

The omission mattered more than a missing sentence usually does, because of where this error arm sits. A bare name reaches it only by failing `is_repo_segment`, which is to say only when it contains a character outside `[A-Za-z0-9._-]`. The typical arrival is therefore a user who typed a bare name with a stray character, and the message answered them by naming the two forms they were not using and suggesting the more laborious of the two.

Message text only. No resolution logic changed.

---

## 1. Problem Statement

### 1.1 Background

The resolver's precedence has three accepted forms and one error arm:

1. an existing local path, used verbatim;
2. an `owner/name` repo-id;
3. a bare single segment, expanded to `<$MLXCEL_DEFAULT_ORG>/<segment>` (default `mlx-community`);
4. otherwise `not_a_model_error`.

Form 3 is first-class: the module docs describe it, and `README.md` puts it in the quick start as `mlxcel run Qwen3.5-0.8B-4bit` with the note "Bare name resolves to mlx-community/<name>". The error named only forms 1 and 2.

### 1.2 Existing Issues

- **The message contradicted the resolver.** It asserted the value was "neither an existing path nor a valid HuggingFace repo-id", an exhaustive-sounding claim over a set that was missing a third of the actual answer.
- **It was wrong precisely where it fires.** `is_repo_segment` accepts `[A-Za-z0-9._-]`; anything with a space or other stray character falls through to the error arm. So the population reaching this message skews heavily toward bare-name users, and bare name is the form it did not mention.
- **The suggested repair was the expensive one.** "Pass a local model directory or a repo-id" asks for a full `owner/name` when fixing one character would have worked.
- **It was internally inconsistent.** `bad_default_org_error`, on the same code path, already spelled out "the org must be a single path segment (`[A-Za-z0-9._-]`)".

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| User abandons the bare-name form the README advertises | Low | High |
| User believes a typo'd bare name is an unsupported syntax | Medium | Medium |
| Future edit re-drops a form, since nothing pinned the list | Low | Medium |

---

## 2. Technical Review

### 2.1 The Message

Before and after, on the issue's own example:

```
$ mlxcel generate -m "Qwen3 4B" -p x -n 1

# before
Error: model 'Qwen3 4B' is neither an existing path nor a valid HuggingFace repo-id (expected
`owner/name`, e.g. `mlx-community/Qwen3-4B-4bit`). Pass a local model directory or a repo-id to
auto-download.

# after
Error: model 'Qwen3 4B' is not a model mlxcel can resolve. Accepted forms: a local model directory;
a HuggingFace repo-id `owner/name` (e.g. `mlx-community/Qwen3-4B-4bit`); or a bare model name made
only of [A-Za-z0-9._-], which resolves against $MLXCEL_DEFAULT_ORG (default `mlx-community`). A
repo-id or bare name is auto-downloaded.
```

### 2.2 Why the Opening Clause Had To Change

The issue asked to extend the message. Extension alone would not have worked: "is neither an existing path nor a valid HuggingFace repo-id" is itself an enumeration of two forms, so appending a third would leave the sentence contradicting its own list. The opening is now a non-enumerating statement ("is not a model mlxcel can resolve") and the enumeration happens once, in the list.

This is the one change with a knock-on cost, since two tests asserted on the old substring. Both are updated, which the issue anticipated by asking for exactly that check.

### 2.3 Scope

`is_repo_segment`, `expand_bare_name`, `resolve_model_source_with_override` and every resolution branch are untouched. The only behavioural surface is the text of one `anyhow!`.

---

## 3. Technical Decisions

### 3.1 State the Character Class Rather Than Describe It

"Made only of `[A-Za-z0-9._-]`" is the same class `is_repo_segment` enforces, written the same way `bad_default_org_error` writes it. A prose paraphrase ("letters, digits, dots, underscores and hyphens") would read more smoothly and drift more easily; the literal class is greppable against the predicate it documents.

### 3.2 Pin the List in the Test

The old tests asserted one substring each, which is why the list could go stale without failing anything. The updated test pins all three form names, the character class, the `$MLXCEL_DEFAULT_ORG` mention and the preserved example. This is the cheap insurance the defect itself argues for: nothing had been holding the message accountable to the resolver.

### 3.3 Preserve the Example

`mlx-community/Qwen3-4B-4bit` is retained verbatim, per the issue's third acceptance criterion, and now doubles as the illustration for form 2 specifically rather than for the message as a whole.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 3 |
| Resolution-logic changes | 0 |
| Accepted forms named, before / after | 2 / 3 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| Resolver | `src/downloader/resolver.rs` | `not_a_model_error` names all three forms and states the character class and default org; doc comment explains why the bare-name form dominates this arm |
| Tests | `src/downloader/resolver_tests.rs` | Both assertions on the old substring updated; the first also pins the three forms, the class, the org variable and the example |
| Documentation | `CHANGELOG.md` | Entry under `## [Unreleased]` / `### Fixed` |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `7c1744ad` | fix | fix(cli): name all three accepted forms in the "not a model" error |

---

## 5. Validation and Follow-up

### Passed

- `cargo test --profile test-fast --features cuda --lib downloader::resolver`: 33 passed, 0 failed.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`: clean (exit code captured directly, not through a pipeline).
- `cargo fmt --all -- --check`: clean.
- Against the built binary: `mlxcel generate -m "Qwen3 4B"` prints the three-form message, which is the issue's exact reproduction; `no/such/nested/path` reaches the same message, confirming the multi-segment arm is unchanged.
- `grep -rn "neither an existing path" src/ tests/ docs/ README.md`: no stale reference to the old text outside one scenario-describing test comment.

### Follow-up Candidates

- **No test ties an error message's enumerations to the code they describe.** This message drifted from the resolver for as long as form 3 has existed, and the fix is a hand-written assertion list. The same class of drift is what `tests/cli_help_consistency.rs` handles for the flag surface, and error-message invariants have no equivalent home.
