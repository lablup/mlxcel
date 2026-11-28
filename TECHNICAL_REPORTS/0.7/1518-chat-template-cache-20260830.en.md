# Technical Report: Chat Template Compile Cache

## Summary

PR #1518 resolves issue #1235 by caching the compiled MiniJinja chat template environment on each `ChatTemplateProcessor`. Before this change, every typed render, raw JSON render, history render and prompt-cache probe render rebuilt the environment, re-registered filters/functions/method callbacks and recompiled the immutable template source.

The implementation moves compilation behind a processor-owned `OnceLock`, stores an owned `Environment<'static>` through MiniJinja's owned-template API, and keeps all request-specific values outside the cache. Messages, tools, kwargs, generation-prompt mode and thinking aliases are still rebuilt per render.

## Problem

The loaded template string is immutable for the processor lifetime, but `apply_inner` and `apply_raw_inner` both created a new MiniJinja environment and added the same `"chat"` template on every call. A single chat-completions request can call the renderer multiple times: once for the primary prompt, once for a prompt-cache history boundary, and again for next-turn warm-up probes.

That made modern 5-20 KB chat templates pay parse/compile and filter registration cost repeatedly even though only the render context changes between calls.

## Implementation

- Added a `compiled_template` cache to `ChatTemplateProcessor`, shared across clones through `Arc<OnceLock<_>>`.
- Added `compile_chat_template_environment`, which configures the MiniJinja environment once and inserts the template through `add_template_owned`.
- Replaced duplicated environment construction in both raw and typed render paths with `render_template`, so `apply_inner`, `apply_raw_inner`, history renders and probe renders share the same compiled template.
- Cached immutable parse-failure results as well, so a malformed immutable template is not reparsed on every fallback attempt.
- Preserved `TemplateRejection` behavior by leaving `raise_exception` as a render-time callable in the cached environment and rebuilding request contexts per render.

## Correctness

The cache owns only the operator/model-supplied template and MiniJinja environment configuration. It does not retain request data, tool declarations, kwargs or generation-prompt mode. This keeps request isolation intact while allowing the bytecode and callable registration to be reused.

The test-only compile counter is stored per processor rather than globally, avoiding parallel test-order coupling. It verifies that typed and raw paths share one compile, clones reuse the same cache, malformed templates cache their parse failure, and concurrent render races converge to one successful compile.

## Compatibility

The focused byte-identity test compares cached renders against a fresh-environment oracle that mirrors the pre-change behavior. It covers typed render, typed history render, raw JSON render and raw JSON history render with tools and kwargs. The existing `server::chat_template::tests` suite also passed, preserving the broad established fixture coverage.

The #1176 rejection/fallback split remains intact: template-side `raise_exception` still exposes the `TemplateRejection` sentinel, while engine failures and parse failures remain non-rejections and continue to be eligible for fallback behavior at the request layer.

## Performance Evidence

A bounded debug-profile smoke run over 400 renders reported:

- Fresh environment/render loop: 27.248578 ms
- Cached environment/render loop: 7.99117 ms

This is a local unit-level measurement, not a production throughput benchmark. It demonstrates the expected direction and approximate magnitude for the isolated compile/render boundary without claiming end-to-end serving performance.

## Validation

- `cargo fmt` passed.
- `cargo test --lib cached_template` passed: 6 passed, 1 ignored.
- `cargo test --lib server::chat_template::tests` passed: 90 passed, 2 ignored.
- `cargo test --lib server::reasoning_effort_tests` passed: 28 passed.
- `cargo test --lib history_render` passed: 5 passed.
- `cargo test --lib cached_template_performance_smoke_reports_delta -- --ignored --nocapture` passed and produced the timing evidence above.
- `cargo clippy --lib --tests -- -D warnings` passed.
- `git diff --check` passed.

## Skipped Validation

Broad cargo workspace tests, serial all-tests, workspace clippy and cold release builds were intentionally skipped under the wave-runner watchdog guard. Local model-template audit tests requiring a populated `models/` directory remained ignored.

## Risk Notes

The cached environment is shared across cloned processors. This is intentional because clones represent the same immutable template lifetime, but future code that introduces mutable template source on a processor must allocate a new cache with the new source.

Malformed-template errors now surface through a cached internal error wrapper instead of a newly created MiniJinja error for each call. The user-visible parse context is preserved and the rejection discriminator remains false; no current tests or request behavior depend on downcasting parse errors to `minijinja::Error`.
