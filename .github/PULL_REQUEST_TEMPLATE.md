<!--
Thanks for opening a pull request! Please fill out the sections below.
For the full contributor contract see AGENTS.md and CONTRIBUTING.md.
-->

## Summary

<!-- One or two sentences: what does this PR do, and why? -->

## Related issues

<!-- Use closing keywords ("Closes #123", "Fixes #456") so the issue auto-closes on merge.
     For tracking-only references use "Refs #..." or "Part of #..." instead. -->

Closes #

## Type of change

<!-- Tick all that apply. -->

- [ ] `feat` — new user-visible feature
- [ ] `fix` — bug fix
- [ ] `perf` — performance improvement (include before/after numbers in the PR body)
- [ ] `refactor` — internal restructuring without behavior change
- [ ] `chore` — build, CI, dependencies, release infrastructure
- [ ] `docs` — documentation only
- [ ] `test` — tests only

## Test plan

<!--
What did you run to convince yourself this works? Be specific.
For inference changes, real-checkpoint validation is required — synthetic-only is not enough.
-->

- [ ] `cargo fmt --all -- --check` (enforced by CI — violations block merge)
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test --release`
- [ ] `cargo deny check`
- [ ] Validated with a real checkpoint (specify which, e.g. `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`): ...

## Notes for reviewers

<!-- Anything reviewers should know: assumed invariants, follow-up work, risks, alternatives considered. -->

## Checklist

- [ ] PR title uses a Conventional Commits prefix (`feat:`, `fix:`, etc.)
- [ ] One logical change per PR (split unrelated changes)
- [ ] Updated `docs/` if user-facing behavior or supported models changed
- [ ] Updated `// Used by: ...` comments on any shared function I modified (see `docs/code-guidelines.md`)
- [ ] No secrets, credentials, or `.env` files committed
