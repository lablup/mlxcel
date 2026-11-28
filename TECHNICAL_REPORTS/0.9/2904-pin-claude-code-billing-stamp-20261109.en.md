# Technical Report: PR #2904 - fix(server): pin Claude Code billing stamp in Anthropic system text

**Date**: 2026-11-09
**Reviewer**: -
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

Claude Code starts its system prompt with a billing header whose `cch` value changes on every request and whose fourth `cc_version` part changes on every session. `mlxcel-server` now overwrites those two values with `f` characters of equal length before the prompt is rendered, so consecutive Anthropic Messages requests share their prefix. On two real checkpoints, turn 2 went from 0 reused tokens to 3789/3821 (`qwen3.8-27b-4bit`, snapshot reuse) and 1984/2007 (`gemma-3n-e2b-it-4bit`, KV reuse).

---

## 1. Problem Statement

### 1.1 Background

The prompt cache keys on rendered tokens, so a single changed byte caps reuse at that position. The stamp looks like this:

```
x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=sdk-cli; cch=b145e;
```

### 1.2 Existing Issues

- `AnthropicSystem::to_text` output reached the leading system message unchanged, and nothing in `src/` handled the stamp.
- With templates that render the system text first, the second turn missed the cache entirely. With templates that render tools first, the divergence sits after the tool list (Strata measured about 22K tokens in).

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Every Claude Code turn re-reads the system prompt and history | High | High for Claude Code clients |
| The model sees altered stamp text | Low | Certain, by design |

---

## 3. Technical Decisions

### 3.1 Same-length `f` placeholders on the two volatile values only

**Context:** llama.cpp (ggml-org/llama.cpp#21793) pins `cch` only. Strata pins both values.

| Option | Pros | Cons |
|--------|------|------|
| Remove the header | Shortest prompt | Changes the structure the model sees |
| Pin `cch` only | Smallest edit | Reuse still breaks across sessions |
| **Chosen: pin `cch` and the `cc_version` session part** | Stable within and across sessions, same token shape | Specific to one client's format |

**Rationale:** The rule fires only on the exact case-sensitive prefix, inside the first line of the first 160 bytes, for bounded `;`-terminated ASCII values. Everything else passes through, so ordinary system prompts are unaffected.

**Trade-offs:** It is deliberately not a generic normaliser. Another client that stamps prompts needs its own documented rule. The window counts bytes where the issue says characters; bytes is the stricter limit.

### 3.2 Apply in the translator, default on, with a kill switch

Pinning runs in `anthropic_request_to_chat` on the flattened text, which covers the string and block-array forms of `system`. A `role: "system"` turn inside `messages` is not pinned. `MLXCEL_ANTHROPIC_PIN_BILLING_STAMP=0` restores pass-through; it is read once per process.

---

## 4. Implementation Details

| File | Change |
|------|--------|
| `src/server/anthropic_billing_stamp.rs` | New: `pin_billing_stamp` returning `Cow<str>`, field parsing, kill switch, unit tests |
| `src/server/anthropic_translator.rs` | Pins the top-level `system` text, translator tests |
| `docs/server-features.md`, `docs/environment-variables.md` | Behaviour and switch documented |

Matching works on bytes of ASCII values, so non-ASCII text after the header is untouched and replacement ranges stay on character boundaries. A field key must follow the header colon, a space, or `;`.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 5 |
| Lines added | +383 |
| Lines deleted | -2 |
| Tests added | 18 in the new module, 5 in the translator |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `2e36a6df` | fix | fix(server): pin Claude Code billing stamp in Anthropic system text |

---

## 8. Follow-up Actions

### Monitoring Required

- Run the real `claude` CLI against `mlxcel-server` with a tools-first template and confirm reuse past the tool list. The recorded run used requests shaped like Claude Code's, not the CLI.

### Future Improvements

- A hostile value containing a space or `=` can make the two edit ranges overlap, so pinning is not idempotent on such input (about 1.8% of an adversarial corpus). It cannot panic and affects only the sender's own prompt. Rejecting those characters in values would remove it.
- A stamped request copies the system text one extra time (about 280 µs for 16 MiB).

---

## Appendix

### A. Test Results

- `cargo test --lib --profile test-fast --features metal,accelerate anthropic`: 111 passed.
- `cargo test --lib server::anthropic` on the final commit: 57 passed.
- `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`: clean.
- `cargo fmt --check` and `make verify-env-var-docs`: pass.
- Not run per PR: the workspace-wide test and clippy commands from the issue. The orchestrator's final gate covers them after merge.

### B. Measurements

Turn 2 of two Claude Code-shaped `/v1/messages` requests with different stamps, block-array `system`, 12 tools, release binary. The kill-switch arm follows the pre-change request path and stands in for "before".

| Model | Before (`=0`) | After (default) |
|-------|---------------|-----------------|
| `qwen3.8-27b-4bit` | 0/3825 | 3789/3821 |
| `gemma-3n-e2b-it-4bit` | 0/2012 | 1984/2007 |

Reuse on `gemma-3n-e2b-it-4bit` covers its whole 1984-token first prompt. On `qwen3.8-27b-4bit` it stops 5 tokens short of the 3794-token first prompt; the cause of that gap was not independently confirmed. Because these templates render the stamp near the start, "before" is a full miss rather than partial reuse.

### C. References

- Issue #2884
- ggml-org/llama.cpp#21793
- Strata `serve/frontend.py`, `pin_billing_stamp`, at `146df3c`
