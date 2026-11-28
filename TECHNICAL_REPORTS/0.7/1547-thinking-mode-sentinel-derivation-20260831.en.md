# Technical Report: PR #1547 - Derive the thinking_mode sentinel from the chat template

**Date**: 2026-08-31
**Author**: mlxcel maintainers
**Status**: Completed; reproduced against the real checkpoint template before and after, with both reproduction tests confirmed to fail against the fix before their assertions were flipped
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

A DeepSeek-V4 request with thinking enabled rendered the non-thinking branch and returned an ordinary completion. Nothing errored. The caller asked for reasoning and got a normal answer, with no signal that anything had been dropped.

The cause was a hardcoded sentinel value. PR #811 injects `thinking_mode: "enabled"` into the chat-template context when thinking is enabled and the template mentions the `thinking_mode` identifier. DeepSeek-V4's template mentions it, but gates on `'thinking'`, so the injected value fell straight through to the else branch.

The interesting part is not the mismatch. It is that #811 shipped with tests covering this exact mechanism, and those tests could not see the bug, because the test fixture and the implementation encoded the same assumption about how templates spell the value.

## 1. Problem Statement

### 1.1 The two conventions

#811 (issue #775) was written for templates shaped like this:

```jinja
{% if thinking_mode is defined and thinking_mode == "enabled" %}<think>{% else %}<no_think>{% endif %}
```

The variable is compared directly against a literal. Inject `"enabled"` and the thinking branch opens.

DeepSeek-V4-flash uses a different idiom:

```jinja
{%- set mode = thinking_mode|default('chat') -%}
{%- if loop.last and mode == 'thinking' -%}<think>{%- else -%}</think>{%- endif -%}
```

The variable is aliased through `default()`, and the alias is compared against `'thinking'`. `"enabled"` is not `'thinking'`, so the injection is a no-op that still looks like it worked: the kwarg is present in the context, the template reads it, and the gate says no.

### 1.2 Why detection did not help

`wants_thinking_mode_alias()` is a substring scan for the identifier:

```rust
fn wants_thinking_mode_alias(&self) -> bool {
    self.template.contains("thinking_mode")
}
```

That returns `true` for both conventions. So the alias fired correctly on DeepSeek-V4 and delivered a value the template had no use for. Detection and injection disagreed about what the variable means, and nothing in the pipeline could notice.

### 1.3 Latent when filed, live when fixed

Issue #819 was filed on 2026-07-19 and explicitly scoped its own impact:

> None at runtime yet: `model_type: deepseek_v4` is not loadable (`Error: Unsupported model type: deepseek_v4`; the backbone port is tracked in #523). The gap becomes user-visible the moment #523 lands.

#523 landed in PR #1455 on 2026-08-27. The issue's own precondition had been met and its body still said the impact was none, which is the failure mode a stale issue body produces: the text reads as reassuring precisely when it has stopped being true.

## 2. Technical Decisions

### 2.1 Derive the sentinel rather than table-drive it

The issue offered two options: scan the template for its gating literal, or keep a per-family table (`deepseek_v4` maps to `"thinking"`) in front of the `"enabled"` default.

Derivation was chosen because a table encodes the same brittleness one level up. It answers for the families someone thought to enumerate and silently reverts to the wrong default for the next template that invents a third spelling, which is exactly how this bug arrived. Reading the gate out of the template answers for any convention that expresses itself as a comparison.

### 2.2 Fall back rather than guess

The derivation returns a literal only when it finds exactly one distinct candidate. Zero candidates, or several, fall back to `"enabled"`.

This is the load-bearing decision, and it follows from the failure mode. A wrong sentinel is silent: the template takes its else branch and produces plausible output. A derivation that guesses when the gate is ambiguous would therefore trade a known-wrong default for an unpredictably-wrong one, with the same invisibility. Falling back preserves the behaviour of every template that works today and confines the change to templates whose gate can actually be read.

### 2.3 Exclude the `default()` argument

`thinking_mode|default('chat')` contributes no candidate. It cannot: `'chat'` is this template's OFF value, and injecting it when the caller asked for reasoning would be worse than injecting nothing at all.

The exclusion is structural rather than special-cased. Only literals appearing in `==` comparisons are collected, and a `default()` argument is not a comparison, so it is never a candidate on any template.

### 2.4 Keep #811's precedence untouched

An explicit client-supplied `thinking_mode` kwarg still wins over the derived value, and nothing is injected when thinking is disabled. Both were already correct, and both are now pinned by tests, since the derivation created a new opportunity to break them.

## 3. Implementation

`src/server/chat_template.rs`, +350/-10.

`thinking_mode_sentinel()` replaces the boolean at both production call sites, and `build_template_context` takes `Option<&str>` in place of `thinking_mode_alias: bool`. The derivation is three steps:

1. **Collect the identifiers carrying the value.** `thinking_mode` itself, plus any alias introduced by `{% set <name> = thinking_mode ... %}`. The scan splits on `set ` and accepts a left-hand side that is a bare identifier whose right-hand side starts with `thinking_mode`.
2. **Collect the gating literals.** Walk every `==` in the template. If an identifier from step 1 ends the left side as a whole word, read a leading string literal from the right side; and symmetrically for the reversed comparison order. Whole-word matching matters so `my_thinking_mode` is not mistaken for the variable.
3. **Decide.** Exactly one distinct literal, return it. Otherwise return `None` and let the caller fall back.

Both directions of `==` are handled because Jinja templates in the wild write both, and the cost of covering the second is a few lines against a silent failure if a template uses it.

## 4. Test Strategy

The bug survived a test suite, so the tests here are built to discriminate rather than merely pass.

**Both reproduction tests were written to assert the broken behaviour first**, run against the unfixed code to confirm they passed, then run against the fix to confirm they failed, and only then flipped to assert the correct behaviour. A test that has never failed is not known to be capable of failing.

- `issue_819_deepseek_v4_gate_opens_from_a_derived_sentinel`: the aliased shape opens `<think>`, and the derived sentinel is asserted to equal `"thinking"` rather than only checking the rendered output, so a future change that produces the right render for the wrong reason still fails.
- `issue_819_real_checkpoint_template_opens_the_thinking_branch`: the same check driven by `models/deepseek-v4-flash-4bit/chat_template.jinja` itself, so the inline fixture cannot drift from the file it stands in for. Skips when the checkpoint is absent. Asserts the derived render is byte-equal to the explicit-workaround render.
- `thinking_mode_sentinel_falls_back_when_the_gate_is_not_readable`: eight cases covering no mention, mentioned without comparison, two competing gate literals, the #811 convention unchanged, the same literal compared twice (the real V4 shape), reversed comparison order, a longer identifier ending in the variable's name, and the `default()` OFF value never being chosen.
- `derived_sentinel_keeps_811_precedence`: an explicit kwarg overrides the derived value even when the template does not recognise it; nothing is injected when thinking is disabled.

## 5. Validation

| Gate | Result |
| --- | --- |
| `cargo test --lib server::` | 2752 passed, 0 failed |
| `cargo test --lib server::chat_template` | 146 passed, 0 failed |
| `cargo test --lib reasoning_stream` | 23 passed |
| `cargo clippy --lib --tests -- -D warnings` | exit 0 |
| `cargo fmt --check` | clean |

Reproduced against the real checkpoint template through `ChatTemplateProcessor::apply_with_kwargs` before the fix:

```
[819] enable_thinking=true   tail: ">hi<｜Assistant｜></think>"   non-thinking branch
[819] thinking_mode=thinking tail: "｜>hi<｜Assistant｜><think>"   the documented workaround
```

After the fix, `enable_thinking=true` produces a render byte-equal to the explicit workaround.

## 6. Learning Points

**A test and the implementation it covers can share an assumption, and then the test does not test.** `THINKING_MODE_LIKE_TEMPLATE` compares `thinking_mode` directly against `"enabled"`; the implementation injected `"enabled"`. Both were written from the same mental model of how a template spells the value, so the suite confirmed the model rather than the behaviour. The tests were not missing; they were parallel to the code instead of perpendicular to it. When a fixture is authored alongside the feature it validates, it is worth asking what a fixture written by someone who had never seen the implementation would look like.

**A silent wrong answer deserves a conservative fix.** The whole reason this bug was expensive is that nothing errored. That property should propagate into the fix: a derivation that guesses under ambiguity would preserve the invisibility while making the outcome less predictable. Falling back to the prior behaviour keeps the blast radius to templates whose intent is unambiguous.

**A stale issue body is worse than no issue body.** #819 accurately described its impact as none, and that sentence stayed on the page after the precondition it depended on had been met. A reader checking whether this mattered would have been told, in writing, that it did not. Issue bodies that assert current state have a shelf life, and the ones that name a specific unblocking event are worth revisiting when that event occurs.

## 7. What Remains Unverified

- **Only two template conventions are covered by real fixtures.** The `== "enabled"` shape and the aliased `== 'thinking'` shape both have tests against real or realistic templates. Any third convention falls back to `"enabled"`, which is safe but not correct for that template. No survey of published chat templates was done to find a third.
- **Multi-step aliasing is not handled.** `{% set a = thinking_mode %}{% set b = a %}{% if b == 'x' %}` derives nothing and falls back. One level of aliasing covers the known cases; deeper chains were not implemented because no template is known to use them.
- **Comparison operators other than `==`** (`in`, `!=`, `is`) are not scanned. A template gating with `thinking_mode in ['thinking', 'deep']` falls back.
- **No end-to-end server test.** The fix is verified at the `ChatTemplateProcessor` level, not through a live `/v1/chat/completions` request against a loaded DeepSeek-V4. The existing server integration tests in `tests/chat_template_kwargs.rs` spawn a server with a small model and would need a V4-shaped template fixture to cover this path.

## 8. Follow-up Actions

None filed. The unverified items above are bounded and each falls back safely rather than failing; filing issues for conventions nobody has been observed to use would repeat the mistake #1531 was closed for, where an issue was opened on a hypothesised rather than measured problem.

## References

- Issue #819, and its predecessors #775 (the original thinking gate) and #811 (the mechanism this corrects)
- Issue #523 / PR #1455, the DeepSeek-V4 port whose landing made this live
- `models/deepseek-v4-flash-4bit/chat_template.jinja`, the template that motivated the change
- `src/server/chat_template.rs`, `thinking_mode_sentinel` and `derive_thinking_mode_sentinel`
