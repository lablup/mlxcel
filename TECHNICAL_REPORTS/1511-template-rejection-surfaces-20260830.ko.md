# 기술 보고서: PR #1511 - Template rejection surfaces

**작성일**: 2026-08-30
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Medium

## 요약

PR #1511은 issue #1176을 해결한다. Chat template의 `raise_exception(...)` rejection이 Responses API, offline CLI, disaggregated router front end에서 사라지지 않도록 보존했다. 의도적인 template refusal은 이제 named client error로 요청을 중단하고, malformed template이나 unsupported template-engine failure는 기존 fallback path를 유지한다.

## 1. 문제 정의

Single-node `/v1/chat/completions` route는 이미 template-raised rejection을 request error로 처리했지만, 인접 surface에서는 이를 삼키거나 잘못 분류할 수 있었다. `/v1/responses`에는 현재 `reasoning.effort` forwarding 동작을 고정하는 regression coverage가 필요했고, offline CLI는 모든 template error에서 raw user prompt로 fallback했으며, disaggregated router는 모든 preparation error를 HTTP 500으로 변환했다.

이 failure mode는 reasoning-effort control에서 특히 위험했다. 값은 OpenAI vocabulary와 model-specific vocabulary 사이에서 의도적으로 translation하지 않고 그대로 전달된다. 따라서 model template이 `high`를 reject하면 unframed fallback prompt로 조용히 generation하지 말고 visible client error를 반환해야 한다.

## 2. 기술적 선택과 그 이유

### 2.1 Preparation 과정에서 rejection sentinel 보존

`prepare_chat_request_with_cache`는 이제 template rejection에 user-facing context를 추가하면서 original render error를 source로 유지한다. 따라서 disaggregated router 같은 outer route boundary에서도 `template_rejection_message`를 계속 사용할 수 있고, single-node chat 및 Responses route가 이미 노출하던 error text는 바뀌지 않는다.

### 2.2 CLI fallback은 engine failure에만 유지

Offline CLI prompt helper는 이제 `Result<String>`을 반환한다. `template_rejection_message`가 sentinel을 찾으면 named error를 반환하고, malformed syntax처럼 minijinja가 engine-side 이유로 render하지 못한 경우에는 기존처럼 raw prompt fallback을 반환한다.

### 2.3 Reasoning value translation 금지 유지

Responses translator 동작은 변경하지 않았다. `reasoning.effort`는 그대로 chat request에 복사된다. 새 test는 `high` 값이 변경 없이 template까지 도달하고, model-specific template이 이를 reject할 수 있음을 증명한다.

### 2.4 Router template rejection을 400으로 매핑

Disaggregated router는 chat request preparation 이후 preserved sentinel을 확인한다. Template rejection에는 single-node route와 같은 `400 invalid_request_error` response shape를 반환하고, 관련 없는 preparation failure는 기존 generic 500 path에 남긴다.

## 3. 변경 요약

| 카테고리 | 변경 수 | 주요 내용 |
|---|---:|---|
| Error propagation | 1 | Preparation error wrapping 후에도 template rejection source를 보존. |
| CLI behavior | 1 | CLI prompt resolution을 template rejection에 대해서만 fallible하게 변경. |
| Router behavior | 1 | Disaggregated router template rejection을 500에서 400으로 변경. |
| Responses coverage | 2 | Rejection 및 engine-fallback regression test 추가. |
| CLI coverage | 3 | Prompt-helper test 업데이트 및 rejection-path coverage 추가. |
| Router coverage | 1 | Template rejection HTTP 400 regression coverage 추가. |

## 4. 검증

- `cargo test --bin mlxcel resolve_cli_prompt`: 5 passed.
- `cargo test --bin mlxcel apply_user_chat_template_wraps_prompt_as_user_message`: 1 passed.
- `cargo test --bin mlxcel vlm_chat_template`: 4 passed.
- `cargo test --lib responses_reasoning_effort_template_rejection_survives_translation`: 1 passed.
- `cargo test --lib responses_template_engine_failure_still_uses_prompt_fallback`: 1 passed.
- `cargo test --lib router_chat_maps_template_rejection_to_bad_request`: 1 passed.
- `cargo test --lib template_rejection`: 11 passed.
- `cargo clippy --lib --tests -- -D warnings`: passed.
- `cargo clippy --bin mlxcel -- -D warnings`: passed.
- Hosted PR checks passed: cargo-clippy, cargo-deny, cargo-fmt, OpenXLA feature compile, crate versions, cross-repo refs, kernel dtype keys, llama-compat manifest, Detect changes, license/cla. MLX pin extraction과 OpenXLA feature link는 change detection에 의해 skipped였다.

Broad workspace tests, serial all-tests, cold release build는 이 issue batch의 workflow guard가 금지하므로 실행하지 않았다.

## 5. 리뷰 메모

- **Correctness**: 영향을 받은 모든 surface에서 template rejection과 engine failure path가 분리된다. Responses forwarding은 그대로 유지되므로 unsupported value 판단은 계속 loaded template이 담당한다.
- **Security**: Rejection message는 HTTP 또는 CLI output에 도달하기 전에 기존 `TemplateRejection`의 512자 cap과 control-character filtering을 거친다. Prompt content를 새로 log하지 않는다.
- **Performance**: 새 검사는 render failure path에서 기존 error chain만 확인한다. 성공적인 render 및 generation path는 변경하지 않았다.
- **Compatibility**: Malformed 또는 unsupported template에 대한 기존 fallback behavior는 유지되어, mlxcel이 render하지 못하는 template을 가진 model serving compatibility를 보존한다.

## 6. 후속 조치

- 해당 checkpoint를 사용할 수 있는 deployment에서 Qwen3.8 Responses 및 disaggregated-router live smoke test를 실행한다.
- Responses 동작은 계속 verbatim forwarding으로 문서화한다. 향후 product decision으로 value translation을 도입한다면 별도 issue에서 compatibility requirement를 명시해야 한다.
