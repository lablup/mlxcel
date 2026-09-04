# Technical Report: PR #1620 - fix(server): resolve absolute file:// URLs under --media-path

**Date**: 2026-09-04
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (A/B validated against a live server on both builds, at six URL spellings)
**Languages**: Rust
**Risk Level**: Low-Medium (touches a confinement boundary; the change only converts refusals into acceptances or into more accurate refusals, and adds no second containment path)

---

## Executive Summary

`--media-path` permitted `file://` reads only for references interpreted as relative to the root, so the absolute form every other tool accepts could never resolve. `file:///srv/media/cat.png` under `--media-path /srv/media` was probed at `<root>/srv/media/cat.png` and refused with `file does not exist or cannot be opened`, naming a file that is present and readable. The path actually probed was never disclosed, which made the failure undiagnosable from outside the server.

PR #1620 keeps b10621's concatenation as the primary resolution and adds an absolute-path fallback that goes through the same containment check. An absolute path inside the root now resolves to exactly what the relative spelling resolves to; one outside the root is refused as an escape rather than as a missing file.

---

## 1. Problem Statement

### 1.1 Background

The media root was introduced in #1451 to confine local reads. It reproduces llama-server b10621's `handle_media`, which evaluates `media_path + file_path`: a pure string concatenation, not a join. That distinction is load-bearing. A Rust `Path::join` with an absolute component would discard the root and turn a compatibility feature into an arbitrary-file read, so `relative_component` strips leading separators before joining. `an_absolute_looking_path_is_concatenated_not_joined` pins the behaviour.

### 1.2 Existing Issues

The concatenation is correct and the confinement is correct, but nothing stated the rule. `--help` said only "Directory that local `file://` media URLs are resolved against", and the compatibility document said "the request's path is resolved against the configured directory". Neither says that an absolute path is appended rather than substituted. An operator reading either would write the absolute form, get a "does not exist" error naming a file that does exist, and have no way to infer why.

The repository's own benchmark harness had already fallen into it. `scripts/bench_embeddings.py` built `f"file://{IMAGE}"` from an absolute `IMAGE`, and every image cell of the 2026-09-04 embedding pass returned HTTP 400 as a result.

### 1.3 Risk Assessment

Any change here is a change to a confinement boundary, so the design constraint was that the fallback must not introduce a second containment path. It reuses the existing `canonical.starts_with(root)` test rather than adding one, and it is gated strictly on `Path::is_absolute` so a relative reference can never canonicalize against the server's working directory.

---

## 2. Technical Review

### 2.1 Root cause

`resolve_media_file_in` had exactly one resolution strategy:

```rust
let relative = relative_component(raw);       // strips leading '/' and '\'
let canonical = tokio::fs::canonicalize(root.join(relative)).await?;
if !canonical.starts_with(root) { return Err(Escape); }
```

For an absolute reference, `relative_component` strips the leading separator and the remainder is appended to the root, so the probe lands somewhere that does not exist. The failure is structural rather than incidental: there was no code path under which an absolute reference could succeed.

### 2.2 Where the fix has to land

`media_root::resolve_media_file` has a single caller, `read_confined_bytes_with_limit` in `src/server/media.rs`, reached from `try_read_image_url_with_limits`. That one function serves the embeddings route, the rerank route, and chat, responses and Anthropic image parts. Putting the fallback inside the shared resolver therefore reaches every live handler at once, which is what the issue's last acceptance criterion asks for and what a standalone helper would not have achieved.

---

## 3. Technical Decisions

### 3.1 Fallback, not replacement

The concatenation stays primary. The fallback runs only when the concatenated candidate fails to canonicalize and the reference is absolute. That ordering means nothing which resolved before resolves differently now, so every existing containment test holds unchanged, and a file that happens to exist at both `<root>/<abs>` and `<abs>` keeps b10621's answer.

### 3.2 Escape, not Unresolvable, for an absolute path outside the root

An absolute path that canonicalizes outside the root is now `MediaPathError::Escape`. This is a deliberate, observable change: upstream answers every absolute spelling with `file does not exist or cannot be opened`, inside the root or outside it. The file is still never opened, so the security property is unchanged, but a client can now distinguish the two cases.

`an_absolute_looking_path_is_concatenated_not_joined` changes its expectation for `file:///etc/passwd` accordingly. The half of that test which proves the concatenation is still primary, that `file:///ok.png` resolves to `<root>/ok.png`, is untouched, because that is the assertion keeping the arbitrary-file read out.

### 3.3 The divergences are recorded as checked fields, not prose

Both observable changes went into the `divergence` array of the `--media-path` entry in `compat/llama-server/b10621/multimodal-and-audio.json`, which `scripts/ci/check_llama_compat_manifest.py` validates, rather than into the entry's free-text `notes`. The entry's `state` stays `aliased`, satisfying the checker's rule that a non-empty `divergence` forbids `supported`.

This matters because the alternative has already failed here once. In the b10621 compatibility work, divergences written into free-text notes while the state field still claimed `supported` produced 37 false claims out of the first 59 reviewed. A divergence that only a human reader can see is not recorded.

### 3.4 A self-explaining refusal that does not leak the root

`MediaPathError::Unresolvable` now carries a trailing clause naming `--media-path` as the directory paths are resolved against. The candidate actually probed goes to the server log at `debug` and never into the response, because it spells out the configured root to an unauthenticated caller.

### 3.5 One boundary pinned rather than changed

`validate_media_filename` enforces b10621's 255-byte cap on the whole path and runs before either resolution strategy, so an absolute path longer than 255 bytes is refused as `NotAllowed` and never reaches the fallback. That is upstream-faithful and is left as it is, documented so the next reader does not treat it as a bug.

---

## 4. Validation

`models/mlx/qwen2.5-vl-3b-instruct` on `/v1/chat/completions`, `--media-path <repo>/tests/fixtures`, with one binary built from `main` and one from the branch. The chat route answers 400 when an image part is dropped, so status alone separates accepted from refused.

| URL | Before | After |
|---|---|---|
| `file://<repo>/tests/fixtures/test_image.png` (absolute, inside root) | 400, `file does not exist or cannot be opened` | **200**, image consumed |
| `file://test_image.png` | 200 | 200 |
| `test_image.png` | 200 | 200 |
| `file:///test_image.png` | 200 | 200 |
| `file://<scratch>/outside/outside_image.png` (absolute, outside root) | 400, `file does not exist or cannot be opened` | 400, `file path escapes the --media-path root` |
| `file:///etc/passwd` | 400, `file does not exist or cannot be opened` | 400, `file path escapes the --media-path root` |
| `file://absent.png` | 400, `...: absent.png` | 400, `...: absent.png (paths are resolved relative to the --media-path root)` |

The last row also emitted, at `debug` and only to the log, `local media reference did not resolve under the --media-path root reference="absent.png" probed=<repo>/tests/fixtures/absent.png`.

Gates: `cargo test --workspace --profile test-fast --features metal,accelerate` green; `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --all -- --check` and `scripts/ci/check_llama_compat_manifest.py` all clean. CI on the PR passed every job including `llama-compat manifest`.

---

## 5. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 5 |
| Lines added | 235 |
| Lines removed | 21 |

### Changes by category

- `src/server/media_root.rs`: the absolute fallback, a private `unresolvable()` helper that logs the probed candidate and keeps it out of the error, the extended message, module docs.
- `src/server/media_root_tests.rs`: five new cases, the changed expectation, two assertions on the message.
- `src/cli/multimodal_compat_args.rs` and `docs/llama-server-compat.md`: the resolution rule with one accepted absolute and one accepted relative form.
- `compat/llama-server/b10621/multimodal-and-audio.json`: two checked divergences.

### Landed separately

`docs/benchmark_results/embeddings-rerank-m5max-2026-09-04.md` is updated on `bench/0.7.0-refresh` (PR #1617) as commit `c3a999b1e`, because benchmark results belong to that branch rather than to a code PR. Its finding is rewritten to past tense for what the pass measured, with what changed since stated after it.

### Related issues

Closes #1612. Follows #1451, which introduced the confinement.

---

## 6. Follow-up Actions

### Residue

`_image_data_uri()` in `scripts/bench_embeddings.py` still describes the pre-#1612 behaviour in its docstring, as does the `--media-path` comment on the server command line in that file. Both remain correct about why the harness sends a data URI, which needs no server flag and keeps the ladder reproducible where `--media-path` was never set, and both are now stale about what a `file://` URL can be. Neither affects a measurement.

### Transferable lesson

A compatibility behaviour that is deliberate, pinned by a test, and undocumented is indistinguishable from a bug at the boundary. The concatenation here was all three, and the cost was not the behaviour itself but that nothing stated it: the project's own benchmark harness sent the one form that could not work, and the error message named a readable file as missing. When a rule is load-bearing enough to pin with a test, it is load-bearing enough to state in `--help`.
